from dataclasses import dataclass, field
from .train_rl_model import (
    TeacherModelConfig,
    StudentModelConfig,
    JudgeModelConfig,
    RewardModelConfig,
    LoggingConfig,
    GenerationConfig,
)


@dataclass
class Dataset:
    name_or_path: str = "rd211/Big-Math-RL-Verified-Filtered"
    split: str = "test"
    ratio: float = 1.0


@dataclass
class DatasetConfig:
    eval_datasets: list[Dataset] = field(default_factory=lambda: [Dataset()])
    max_val_examples: int = 500


@dataclass
class EvalConfig:

    teacher_model: TeacherModelConfig = field(default_factory=TeacherModelConfig)
    student_model: StudentModelConfig = field(default_factory=StudentModelConfig)
    judge_model: JudgeModelConfig = field(default_factory=JudgeModelConfig)
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    generation: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(
            ignore_rejected_judge=True,
            use_thinking=False,
            force_thinking=False,
            number_judge_attempts=2,
            number_student_attempts=8,
        )
    )

    num_samples_per_problem: int = 1

    score_using_pedagogical_reward: bool = True
    pedagogical_reward_model: str = "eth-nlped/Qwen2.5-1.5B-pedagogical-rewardmodel"

    recompute_initial_attempts: bool = True

    seed: int = 42
