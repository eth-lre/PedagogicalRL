from dataclasses import dataclass, field
from typing import Optional, Union

import transformers
from packaging import version
from transformers import TrainingArguments


@dataclass
class ClassroomGRPOConfig(TrainingArguments):
    if version.parse(transformers.__version__) <= version.parse("4.50.3"):
        from transformers.training_args import _VALID_DICT_FIELDS

        _VALID_DICT_FIELDS.append("model_init_kwargs")
    else:
        _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]

    # Parameters that control the model and reference model
    model_init_kwargs: Optional[Union[dict, str]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `GRPOTrainer` is provided as a string."
        },
    )
    disable_dropout: bool = field(
        default=False,
        metadata={
            "help": "Whether to disable dropout in the model. This is useful for training with a reference model, as "
            "it prevents the model from generating different logprobs for the same input."
        },
    )

    # Parameters that control the data preprocessing
    # The default value remove_unused_columns is overwritten from the parent class, because in GRPO we usually rely on
    # additional columns to compute the reward
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    num_generations: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of generations to sample."
        },
    )
    max_completion_length: Optional[int] = field(
        default=8192,
        metadata={"help": "Maximum length of the generated completion."},
    )
    shuffle_dataset: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )

    # Parameters that control generation
    temperature: float = field(
        default=0.9,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )

    # Parameters that control generation acceleration powered by vLLM
    vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host of the Classroom vLLM server to connect to."},
    )
    vllm_server_port: int = field(
        default=8005,
        metadata={"help": "Port of the Classroom vLLM server to connect to."},
    )

    # Parameters that control the training
    learning_rate: float = field(
        default=5e-7,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )
    beta: float = field(
        default=0.04,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs."
        },
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations per batch (denoted as μ in the algorithm)."},
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )
    epsilon_high: Optional[float] = field(
        default=None,
        metadata={
            "help": "Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the "
            "lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`."
        },
    )
    scale_rewards: bool = field(
        default=True,
        metadata={
            "help": "Whether to scale the rewards by dividing them by their standard deviation. If `True` (default), "
            "the rewards are normalized by the standard deviation, ensuring they have unit variance. If `False`, no "
            "scaling is applied. The Dr. GRPO paper recommends not scaling the rewards, as scaling by the standard "
            "deviation introduces a question-level difficulty bias."
        },
    )
    loss_type: str = field(
        default="grpo",
        metadata={
            "help": "Specifies the loss formulation to use. Supported values are `grpo`, `bnpo`, and `dr_grpo`. "
            "`'grpo'`: Aggregates token-level losses by normalizing over sequence length. Not recommended due to "
            "length bias—this approach tends to prefer shorter completions with positive advantages and longer ones "
            "with negative advantages. "
            "`'bnpo'`: Aggregates token-level losses by normalizing number of active token in the local batch. "
            "Note that normalization is performed over the local batch only, so results may slightly vary depending "
            "on the local batch size, despite a constant effective batch size. When using "
            "`per_device_train_batch_size==1`, the loss is equivalent to the GRPO loss. "
            "`'dr_grpo'`: Aggregates token-level losses by normalizing with a global constant. This method was "
            "introduced in the Dr. GRPO paper to eliminate length bias. The value of the constant corresponds to "
            "`max_completion_length`."
        },
    )

    use_liger_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the Liger loss."
        },
    )


    # Special parameters
    use_experimental_shared_memory: bool = field(
        default=False,
        metadata={
            "help": "Whether to use shared memory to pass the weights to the vLLM server."
            "This is useful for large models, as it avoids the need to gather the weights to GPU memory."
        },
    )
    
    ######## Not used
    batch_size_reference_model: int = field(
        default=1,
        metadata={
            "help": "Batch size for computing logits for the reference model."
        },
    )

    ######## Not used
    offload_reference_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to offload the reference model to CPU when not in use."
        },
    )
    ######## Not used
    offload_optimizer_and_weights: bool = field(
        default=False,
        metadata={
            "help": "Whether to offload the optimizer and weights to CPU when not in use."
        },
    )
    
    save_policy_to_disk_every_n_steps: int = field(
        default=1000,
        metadata={
            "help": "Number of steps to save the policy to disk. vLLM seems to go haywire if we always do online updates, so "
            "we save the policy to disk every n steps and load it back into vLLM. This is a workaround for now."
        },
    )