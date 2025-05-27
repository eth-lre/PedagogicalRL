import torch
import hydra
import wandb
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from config.train_sft_model import SFTModelTrainingConfig
from transformers import set_seed
from dotenv import load_dotenv
from accelerate import Accelerator
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from utils.data import load_datasets
from src.utils.utils import (
    init_logger,
)
import warnings
warnings.filterwarnings("ignore")
logger = init_logger()
load_dotenv()

cs = ConfigStore.instance()
cs.store(name="config", node=SFTModelTrainingConfig)


@hydra.main(config_path="config/train_sft", version_base=None)
def main(cfg: SFTModelTrainingConfig):

    accelerator = Accelerator()

    #############################################################################
    # Setup
    #############################################################################

    model_config = cfg.model
    train_config = cfg.train
    logging_config = cfg.logging
    lora_config = model_config.lora
    data_config = cfg.dataset

    set_seed(cfg.seed)

    if logging_config.wandb and accelerator.is_main_process:
        wandb.init(
            project=logging_config.wandb_project,
            name=logging_config.wandb_run_name,
            entity=logging_config.wandb_entity,
            group=logging_config.run_group,
            tags=logging_config.wandb_tags,
            config=OmegaConf.to_object(cfg),
        )
        include_fn = lambda x: "output" not in x and (
            x.endswith(".py")
            or x.endswith(".yaml")
            or x.endswith(".txt")
            and "myenv" not in x
        )
        wandb.run.log_code(".", include_fn=include_fn)

    logger.info("Loading model")
    if accelerator.is_main_process:
        snapshot_download(model_config.model_name_or_path)

    accelerator.wait_for_everyone()

    torch_dtype = torch.bfloat16
    model_kwargs = dict(
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch_dtype,
        use_cache=False if train_config.gradient_checkpointing else True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_name_or_path, trust_remote_code=True
    )

    
    ### LoRA
    
    peft_config = None
    if lora_config.enable:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.rank,
            lora_alpha=lora_config.alpha,
            target_modules=lora_config.target_modules,
            lora_dropout=lora_config.dropout,
            bias=lora_config.bias,
        )

    logger.info("Getting datasets")
    train_dataset, val_dataset = load_datasets(data_config, cfg.seed)
    logger.info(train_dataset)
    logger.info(val_dataset)

    # Preprocess the dataset.
    def format_data(example):
        text = tokenizer.apply_chat_template(example["conversation"], tokenize=False)
        return {
            "text": text,
        }

    train_dataset = train_dataset.map(format_data, num_proc=4)
    val_dataset = val_dataset.map(format_data, num_proc=4)

    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=SFTConfig(
            bf16=True,
            run_name=cfg.logging.wandb_run_name,
            model_init_kwargs=model_kwargs,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps,
            gradient_checkpointing=train_config.gradient_checkpointing,
            per_device_train_batch_size=train_config.per_device_train_batch_size,
            per_device_eval_batch_size=train_config.per_device_train_batch_size,
            hub_model_id=cfg.huggingface.name,
            hub_private_repo=False,
            report_to=["wandb"] if logging_config.wandb else [],
            output_dir=cfg.logging.save_dir,
            save_strategy="no",
            lr_scheduler_type=train_config.lr_scheduler_type,
            num_train_epochs=train_config.epochs,
            max_steps=train_config.max_steps,
            logging_steps=1,
            max_seq_length=model_config.max_length,
            do_eval=val_dataset is not None,
            eval_strategy="steps",
            eval_steps=50,
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Training...")
    train_results = trainer.train()
    logger.info("Training complete!")
    logger.info(train_results)

    trainer.model.config.use_cache = True
    trainer.save_model(logging_config.save_dir + "/model")

    if cfg.huggingface.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
