from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LoraConfig:
    enable: bool = False
    rank: int = 256
    alpha: float = 512
    target_modules: Any = "all-linear"
    dropout: float = 0.01
    bias: str = "none"


@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    max_length: int = 16384
    lora: LoraConfig = field(default_factory=LoraConfig)


@dataclass
class Dataset:
    name_or_path: str = "rd211/Pedagogical-SFT"
    split: str = "train"
    ratio: float = 1.0


@dataclass
class DatasetConfig:
    train_datasets: list[Dataset] = field(default_factory=list)
    val_datasets: list[Dataset] = field(default_factory=list)
    max_train_examples: Optional[int] = None
    max_val_examples: Optional[int] = None


@dataclass
class HuggingFaceConfig:
    name: str = "<model_name>"
    push_to_hub: bool = False


@dataclass
class LoggingConfig:
    wandb: bool = False
    wandb_project: str = "train_sft"
    wandb_run_name: str = "Qwen2.5-7B-Instruct"
    wandb_entity: Optional[str] = None
    run_group: str = "7b"
    wandb_tags: list[str] = field(default_factory=list)
    save_dir: str = "output"


@dataclass
class TrainConfig:
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    per_device_train_batch_size: int = 1
    lr_scheduler_type: str = "cosine"
    optimizer: str = "adamw_hf"
    epochs: int = 1
    max_steps: int = -1
    deepspeed_config_path: Optional[str] = None


@dataclass
class SFTModelTrainingConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42
