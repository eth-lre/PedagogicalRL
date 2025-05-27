# 👨‍🏫 From Problem-Solving to Teaching Problem-Solving

### *Aligning LLMs with Pedagogy using Reinforcement Learning*

**What if your LLM could teach—rather than just answer?**
This project provides a recipe to transform a standard instruction-tuned language model into a math tutor that actually teaches, using multi-turn **GRPO** in a classroom environemnt with a **synthetic student**.

LLMs today excel at solving problems, but good teaching is about **strategically withholding answers** and guiding learners through their own reasoning. This repo implements a scalable, annotation-free framework that aligns LLM behavior with **pedagogical principles** like Socratic questioning, helpful scaffolding, and solution withholding.

We train 7B-sized open models to come close to closed-source models like LearnLM—**without any human labels**—and maintain performance on reasoning tasks like GSM8K and MATH500.

> For details on implementation, memory management, vLLM usage, multi-node setup, and model loading strategies, see [technical.md](technical.md).

🔗 **Paper**: [arxiv.org/abs/2505.15607](https://arxiv.org/abs/2505.15607)
🔗 **Code License**: CC-BY-4.0

---

## 🚀 Quick start

```bash
# 1) create a clean environment
conda create -n pedagogy python=3.11
conda activate pedagogy

# 2) install core deps
pip install -r requirements.txt          # torch, trl, vllm, ...

# 3) add your secrets to a .env file
nano .env
```

---

## 🧪 Environment variables

| Variable             | Purpose                                  |
| -------------------- | ---------------------------------------- |
| `HF_TOKEN`           | For pulling/pushing models on 🤗 Hub     |
| `WANDB_API_KEY`      | Weights & Biases tracking (optional)     |
| `OPENROUTER_API_KEY` | Optional, for routing through OpenRouter |

---

## 🎓 Training

### 1 · Supervised Fine-Tuning (baseline)

```bash
accelerate launch \
  --config_file config/deepspeed/zero2_4GPU.yaml \
  train_sft.py --config-name 7b.yaml
```

### 2 · Pedagogical Reinforcement Learning (ours)

```bash
./start_rl_training.sh \
  --config_file config/deepspeed/zero3_4GPU.yaml \
  --config-name 7b.yaml
```

🎛 Common CLI knobs:

| Flag                                                   | Description                              |
| ------------------------------------------------------ | ---------------------------------------- |
| `generation.extra_penalty_for_rejected_judges`         | λ — penalty for leaking solutions        |
| `generation.number_judge_attempts=0`                   | λ = 0 (no pedagogy constraint)           |
| `generation.use_thinking=true` / `force_thinking=true` | Enable `<think>` tags for tutor planning |

---

## 📈 Evaluation

### Pedagogical Benchmarks

```bash
python eval.py --config-name Qwen2.5-7B-Instruct.yaml
```

### Reasoning Benchmarks (MMLU, GSM8K, MATH500)

Using [LightEval](https://github.com/huggingface/lighteval):

```bash
lighteval vllm \
  "model_name=Qwen/Qwen2.5-7B-Instruct,gpu_memory_utilization=0.85,max_model_length=8192,dtype=bfloat16,\
   generation_parameters={max_new_tokens:8192,temperature:0.6,top_p:0.95}" \
  "lighteval|math_500|0|0,helm|mmlu|5|0,lighteval|gsm8k|4|0" \
  --use-chat-template
```

---

## 🧱 Project layout

```
.
├── config/
│   ├── deepspeed/             # ZeRO configs
│   ├── eval/                  # eval configs
│   └── train_rl/7b.yaml       # RL training config
├── prompt_templates/          # system + judge prompts
├── src/
│   ├── classroom.py           # multi-agent training loop
│   ├── grpo/                  # GRPO trainer logic
│   ├── inference_providers/   # OpenRouter / Gemini adapters
│   └── vllm/                  # vLLM helpers
├── utils/                     # reward functions + logging
├── train_rl.py                # main RL entry point
├── train_sft.py               # SFT script
└── start_rl_training.sh       # launcher for RL
```

---

## 📄 Citation

If you use this work, please cite:

```
@misc{dinucujianu2025problemsolvingteachingproblemsolvingaligning,
      title={From Problem-Solving to Teaching Problem-Solving: Aligning LLMs with Pedagogy using Reinforcement Learning}, 
      author={David Dinucu-Jianu and Jakub Macina and Nico Daheim and Ido Hakimi and Iryna Gurevych and Mrinmaya Sachan},
      year={2025},
      eprint={2505.15607},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.15607}, 
}
```