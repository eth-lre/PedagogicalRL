from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LoraConfig:
    enable: bool = False
    rank: int = 16
    alpha: float = 32
    target_modules: Any = "all-linear"
    dropout: float = 0.01
    bias: str = "none"


@dataclass
class ModelvLLMConfig:

    # Sampling parameters
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0

    # vLLM settings
    max_length: int = 8192
    max_num_seqs: int = 256

    gpu_memory_utilization: float = 0.5
    number_of_gpus_per_instance: int = 4
    max_number_of_instances: int = -1
    from_0: bool = True

    load_and_unload: bool = True

    bits_and_bytes: bool = False
    enable_sleep_mode: bool = True
    use_v0: bool = True
    enforce_eager: bool = False


@dataclass
class TeacherModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    use_openrouter: bool = False
    use_gemini: bool = False
    vllm: ModelvLLMConfig = field(default_factory=ModelvLLMConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)


@dataclass
class StudentModelConfig:
    model_name_or_path: str = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
    use_openrouter: bool = False
    use_gemini: bool = False
    vllm: ModelvLLMConfig = field(default_factory=ModelvLLMConfig)


@dataclass
class JudgeModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-14B-Instruct-AWQ"
    use_openrouter: bool = False
    use_gemini: bool = False
    vllm: ModelvLLMConfig = field(default_factory=ModelvLLMConfig)


@dataclass
class RewardModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-Math-RM-72B"
    vllm: ModelvLLMConfig = field(default_factory=ModelvLLMConfig)


@dataclass
class GenerationConfig:

    # Prompts
    student_personas_prompts_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "simple_student": "prompt_templates/personas/simple_student.txt",
        }
    )

    judges_rules_prompts_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "does_not_leak_answer": "prompt_templates/judges/does_not_leak_answer.txt",
            "follows_pedagogical_values": "prompt_templates/judges/follows_pedagogical_values.txt",
        }
    )

    student_initial_attempt_prompt_path: str = (
        "prompt_templates/student_initial_attempt_prompt.txt"
    )
    student_final_prompt_path: str = "prompt_templates/student_final_prompt.txt"
    teacher_prompt_path: str = "prompt_templates/teacher_prompt.txt"
    initial_attempt_wrapper_prompt_path: str = (
        "prompt_templates/initial_attempt_wrapper_prompt.txt"
    )
    student_attempt_prompt_path: str = "prompt_templates/student_attempt_prompt.txt"

    max_turns: int = 15  # Will actually be 16 turns.
    max_tokens_in_conversation: int = 8192
    max_tokens_per_turn: int = 1024
    max_tokens_per_student_attempt: int = 3900  # A small buffer for the reward model.
    max_tokens_per_judge_attempt: int = 2048
    tokenizer_to_use: str = "Qwen/Qwen2.5-7B-Instruct"

    # Number of attempts we will average over the final student generation.
    number_student_attempts: int = 8
    # If any judge attempt is rejected we give a negative reward and no reward computation is performed.
    number_judge_attempts: int = 2
    ignore_rejected_judge: bool = False  # We do not ignore rejected judge attempts.
    forced_conversation_type: Optional[str] = None  # 'guided' or 'attempt
    use_thinking: bool = False
    force_thinking: bool = False

    # Reward config
    extra_penalty_for_rejected_judges: float = 0.25

    # Server settings
    server_port: int = 8005
    use_experimental_shared_memory: bool = False

    student_names: list[str | None] = field(
        default_factory=lambda: [
            "Alex",
            "Jamie",
            "Taylor",
            "Jordan",
            "Sam",
            "Casey",
            "Morgan",
            "Riley",
            None,
        ]
    )


@dataclass
class Dataset:
    name_or_path: str = "rd211/Big-Math-RL-Verified-Filtered"
    split: str = "train"
    ratio: float = 1.0


@dataclass
class DatasetConfig:
    train_datasets: list[Dataset] = field(default_factory=lambda: [Dataset()])
    max_train_examples: int = -1


@dataclass
class TrainConfig:
    gradient_checkpointing: bool = True

    num_samples_per_problem: int = 8
    number_of_problems_per_batch: int = 16
    per_device_train_batch_size: int = 2

    lr_scheduler_type: str = "constant"
    optimizer: str = "paged_adamw_8bit"

    epochs: int = 1
    max_steps: int = -1
    deepspeed_config_path: Optional[str] = None

    beta: float = 0.001
    learning_rate: float = 5e-7

    mu: int = 2
    epsilon: float = 0.2

    batch_size_ref_model: int = 4
    save_policy_to_disk_every_n: int = 1


@dataclass
class HuggingFaceConfig:
    name: str = "<model_name>"
    push_to_hub: bool = False


@dataclass
class LoggingConfig:
    wandb: bool = False
    wandb_project: str = "train_rl"
    wandb_run_name: str = "Qwen2.5-7B-Instruct"
    wandb_entity: Optional[str] = None
    run_group: str = "7b"
    wandb_tags: list[str] = field(default_factory=list)
    save_dir: str = "checkpoints"
    save_steps: int = 10


@dataclass
class RLModelTrainingConfig:
    train: TrainConfig = field(default_factory=TrainConfig)

    teacher_model: TeacherModelConfig = field(default_factory=TeacherModelConfig)
    student_model: StudentModelConfig = field(default_factory=StudentModelConfig)
    judge_model: JudgeModelConfig = field(default_factory=JudgeModelConfig)
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    skip_first_samples: int = 0

    seed: int = 42
