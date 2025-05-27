import os
import wandb
import hydra
import uvicorn
import threading
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from src.classroom import Classroom, Conversation
from config.train_rl_model import RLModelTrainingConfig
from src.utils.utils import init_logger

logger = init_logger()

import warnings

warnings.filterwarnings("ignore")
load_dotenv()

lock = threading.Lock()

cs = ConfigStore.instance()
cs.store(name="config", node=RLModelTrainingConfig)

classroom: Classroom = None
config: RLModelTrainingConfig = None
app = FastAPI()


class ConversationSampleRequest(BaseModel):
    problems: List[str]
    answers: List[str]
    meta: dict = {}


class RewardRequest(BaseModel):
    conversations: list[str]


@app.post("/sample_conversations")
def sample_conversations(request: ConversationSampleRequest):
    global classroom, config

    problems = request.problems
    answers = request.answers
    meta = request.meta
    conversations = None
    with lock:
        conversations = classroom.sample_conversations(
            problems=problems, answers=answers, meta=meta
        )

    df_table = classroom.to_pd_latest()
    rewards = [classroom.get_end_rm_reward(c) for c in conversations]
    df_table["end_rm_reward"] = rewards
    rewards = [classroom.get_thinking_reward(c) for c in conversations]
    df_table["thinking_reward"] = rewards
    rewards = [classroom.get_end_of_conversation_reward(c) for c in conversations]
    df_table["end_of_conversation_reward"] = rewards
    rewards = [classroom.get_length_reward(c) for c in conversations]
    df_table["length_reward"] = rewards

    # sum of all rewards
    df_table["total_reward"] = (
        df_table["end_rm_reward"]
        + df_table["thinking_reward"]
        + df_table["end_of_conversation_reward"]
        + df_table["length_reward"]
    )
    df_table = df_table.astype(str)
    if config.logging.wandb:
        wandb.log(
            {
                f"batch_{len(classroom.conversation_sets)}": wandb.Table(
                    dataframe=df_table
                )
            }
        )

    return [c.get_trainable_representation() for c in conversations]


@app.post("/get_end_rm_reward")
def get_end_rm_reward(request: RewardRequest):
    global classroom
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    rewards = [classroom.get_end_rm_reward(c) for c in conversations]
    return rewards


@app.post("/get_thinking_reward")
def get_thinking_reward(request: RewardRequest):
    global classroom
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    rewards = [classroom.get_thinking_reward(c) for c in conversations]
    return rewards


@app.post("/get_end_of_conversation_reward")
def get_end_of_conversation_reward(request: RewardRequest):
    global classroom
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    rewards = [classroom.get_end_of_conversation_reward(c) for c in conversations]
    return rewards


@app.post("/get_length_reward")
def get_length_reward(request: RewardRequest):
    global classroom
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    rewards = [classroom.get_length_reward(c) for c in conversations]
    return rewards


@app.get("/wait_batch")
def wait_batch():
    # This endpoint waits (blocks) until the current batch (if any) is finished.
    with lock:
        return {"message": "Batch has been run."}


@hydra.main(config_path="config/train_rl", version_base=None)
def main(cfg: RLModelTrainingConfig):
    global classroom, config

    # We merge the config with the defaults
    default_config = OmegaConf.structured(RLModelTrainingConfig)

    # Merge loaded config with defaults
    cfg = OmegaConf.merge(
        default_config, cfg
    )  # Unspecified keys will use defaults from RLModelTrainingConfig

    config = cfg

    if cfg.logging.wandb:
        wandb.init(
            project=cfg.logging.wandb_project + "-server",
            name=cfg.logging.wandb_run_name,
            entity=cfg.logging.wandb_entity,
            group=cfg.logging.run_group,
            tags=cfg.logging.wandb_tags,
            config=OmegaConf.to_object(cfg),
        )

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    classroom = Classroom(
        cfg.student_model,
        cfg.teacher_model,
        cfg.judge_model,
        cfg.reward_model,
        cfg.generation,
        os.path.join(cfg.logging.save_dir, "policy"),
        log_file_path=None,  # hydra_cfg['runtime']['output_dir']
    )

    uvicorn.run(app, host="0.0.0.0", port=cfg.generation.server_port)


if __name__ == "__main__":
    main()
