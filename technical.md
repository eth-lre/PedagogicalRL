# ⚙️ Technical Overview

This document outlines the implementation details behind our reinforcement learning pipeline for pedagogical tutor alignment. It covers memory management, model offloading, vLLM usage, and multi-node scalability.


## Conversation State Machine

Each tutor-student interaction is modeled as a **multi-turn state machine**. To manage memory efficiently, only the models required for a given phase are loaded into memory.

```python
class ConversationState(Enum):
    START = 0
    TEACHER_TURN = 1
    STUDENT_TURN = 2
    JUDGE_TURN = 4
    GENERATE_SOLUTION = 5
    REWARD_TURN = 6
    END = 7
```

### Model Loading per State

| Conversation State  | Models in Memory                     |
| ------------------- | ------------------------------------ |
| `START`             | None (setup only)                    |
| `TEACHER_TURN`      | ✅ Tutor model (e.g., Qwen2.5-7B)     |
| `STUDENT_TURN`      | ✅ Student model (e.g., Llama-3.1-8B) |
| `JUDGE_TURN`        | ✅ Judge model (e.g., Qwen2.5-14B)    |
| `GENERATE_SOLUTION` | ✅ Student model                      |
| `REWARD_TURN`       | ✅ Judge model                        |
| `END`               | None (cleanup)                       |

At each transition between states, any unused model is offloaded to reduce VRAM usage. All conversations are processed in big batches.

## Multi-Node Training

To fully utilize distributed hardware:

* A dedicated `vLLM` server is launched **per node**
* Each node handles its own share of conversations
* Models are **offloaded** during reward computation and backpropagation

This ensures **no GPUs remain idle**, and all nodes actively participate in training.



## Updating vLLM weights

To prevent out-of-VRAM problems, we use a shared-memory-based model loading mechanism implemented in:

* `src/utils/shared_memory.py`
* `src/utils/utils.py`

This mechanism:

* Gathers tensors incrementally to **CPU memory**
* Stores them in **shared memory**
* Passes **only references** to each node’s local `vLLM` server for loading

### ⚠️ Note on Tensor Parallelism

During training, we encountered problems dynamically loading model weights with `tensor_parallel_size=4` when using shared memory and vLLM.
To avoid issues, we recommend:

* Using `tensor_parallel_size=2` **or**
* Setting `use_experimental_shared_memory=false` to disable shared memory and instead:

  * Save weights to disk
  * Reload vLLM from the folder each time

This avoids memory loading conflicts, though it may make the pipeline slower.


## Data-Parallel vLLM

We allow multiple instances of a model per process with `src/vllm/data_parallel_vllm.py`. This lets you:

* Launch more than one vLLM instance per node
* Configure GPU assignment and model instances in your YAML configs

This improves throughput slightly when the model already fits comfortably on less GPUs. 




## Masked User Turns

* Only **tutor model outputs** are used during weight updates:
  * **User turns are masked** during loss and advantage calculation
  * This makes training more stable
