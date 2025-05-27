from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers.trainer_utils import get_last_checkpoint
import os
import hydra
import wandb
import torch
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from config.train_rl_model import RLModelTrainingConfig
from transformers import set_seed
from dotenv import load_dotenv
from transformers import AutoTokenizer
from datasets import Dataset
from src.grpo.config import ClassroomGRPOConfig
from src.grpo.trainer import ClassroomGRPOTrainer
from src.utils.utils import (
    construct_end_of_conversation_reward_func,
    construct_end_rm_reward_func,
    construct_length_reward_func,
    construct_thinking_reward_func,
    init_logger,
)
import warnings

from utils.data import load_datasets

warnings.filterwarnings("ignore")
load_dotenv()

logger = init_logger()

cs = ConfigStore.instance()
cs.store(name="config", node=RLModelTrainingConfig)


@hydra.main(config_path="config/train_rl", version_base=None)
def main(cfg: RLModelTrainingConfig):

    #############################################################################
    # Setup
    #############################################################################

    # We merge the config with the default config
    default_config = OmegaConf.structured(RLModelTrainingConfig)
    cfg = OmegaConf.merge(default_config, cfg)

    model_config = cfg.teacher_model
    train_config = cfg.train
    logging_config = cfg.logging
    lora_config = model_config.lora
    data_config = cfg.dataset

    set_seed(cfg.seed)

    kwargs = [InitProcessGroupKwargs(timeout=timedelta(hours=10))]
    accelerator = Accelerator(kwargs_handlers=kwargs)

    if logging_config.wandb and accelerator.is_main_process:
        wandb.init(
            project=logging_config.wandb_project,
            name=logging_config.wandb_run_name,
            entity=logging_config.wandb_entity,
            group=logging_config.run_group,
            tags=logging_config.wandb_tags,
            config=OmegaConf.to_object(cfg),
        )
    accelerator.wait_for_everyone()

    torch_dtype = torch.bfloat16
    model_kwargs = dict(
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch_dtype,
        use_cache=False if train_config.gradient_checkpointing else True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=True
    )

    #############################################################################
    # Load the datasets
    #############################################################################

    logger.info(f"Loading datasets from {data_config.train_datasets}")
    train_dataset, _ = load_datasets(data_config, cfg.seed)
    logger.info(f"Loaded {len(train_dataset)} training examples")

    def apply_template(example):
        problem = example["problem"]
        answer = example["answer"]

        return {"prompt": problem, "answer": answer}

    train_dataset: Dataset = train_dataset.map(
        apply_template, num_proc=4, desc="Applying template"
    )

    #############################################################################
    # Rewards
    #############################################################################

    end_rm_reward = construct_end_rm_reward_func(cfg.generation.server_port)
    thinking_reward = construct_thinking_reward_func(cfg.generation.server_port)
    end_of_conversation_reward = construct_end_of_conversation_reward_func(
        cfg.generation.server_port
    )
    length_reward = construct_length_reward_func(cfg.generation.server_port)

    #############################################################################
    # Training
    #############################################################################

    trainer = ClassroomGRPOTrainer(
        model=model_config.model_name_or_path,
        reward_funcs=[
            end_rm_reward,
            thinking_reward,
            end_of_conversation_reward,
            length_reward,
        ],
        args=ClassroomGRPOConfig(
            gradient_accumulation_steps=cfg.train.num_samples_per_problem
            * cfg.train.number_of_problems_per_batch
            // cfg.train.per_device_train_batch_size
            // accelerator.num_processes,
            gradient_checkpointing=train_config.gradient_checkpointing,
            num_generations=cfg.train.num_samples_per_problem,
            per_device_train_batch_size=cfg.train.per_device_train_batch_size,
            num_iterations=cfg.train.mu,
            epsilon=cfg.train.epsilon,
            beta=cfg.train.beta,
            learning_rate=cfg.train.learning_rate,
            optim=cfg.train.optimizer,
            bf16=True,
            run_name=cfg.logging.wandb_run_name,
            model_init_kwargs=model_kwargs,
            hub_model_id=cfg.huggingface.name,
            hub_private_repo=False,
            report_to=["wandb"] if logging_config.wandb else [],
            save_strategy="steps",
            lr_scheduler_type=train_config.lr_scheduler_type,
            num_train_epochs=train_config.epochs,
            max_steps=train_config.max_steps,
            max_completion_length=model_config.vllm.max_length,
            logging_steps=1,
            save_steps=cfg.logging.save_steps,
            save_on_each_node=False,
            save_only_model=False,
            save_total_limit=3,
            output_dir=cfg.logging.save_dir,
            max_grad_norm=1.0,
            temperature=cfg.teacher_model.vllm.temperature,
            vllm_server_port=cfg.generation.server_port,
            use_experimental_shared_memory=cfg.generation.use_experimental_shared_memory,
            batch_size_reference_model=cfg.train.batch_size_ref_model,
            save_policy_to_disk_every_n_steps=cfg.train.save_policy_to_disk_every_n,
        ),
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    last_ckpt = None
    if os.path.isdir(cfg.logging.save_dir):
        last_ckpt = get_last_checkpoint(cfg.logging.save_dir)
        logger.info(f"Last checkpoint: {last_ckpt}")

    logger.info("Training...")
    train_results = trainer.train(resume_from_checkpoint=last_ckpt)
    logger.info("Training complete!")
    logger.info(train_results)

    trainer.model.config.use_cache = True
    trainer.save_model(logging_config.save_dir + "/model")

    if cfg.huggingface.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
