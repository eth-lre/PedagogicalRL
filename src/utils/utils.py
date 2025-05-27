import os
import deepspeed
import torch
import logging
import torch
import psutil
import logging
import pynvml

pynvml.nvmlInit()


class GPUFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style="%", rank=0):
        super().__init__(fmt, datefmt, style)
        self.rank = int(str(rank))
        pynvml.nvmlInit()

    def format(self, record):
        if torch.cuda.is_available():
            gpu_id = self.rank % torch.cuda.device_count()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_mb = info.used / 1024**2
            total_mb = info.total / 1024**2
            pct = used_mb / total_mb * 100
            record.gpu = f"GPU{gpu_id}: {used_mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%)"
        else:
            record.gpu = "GPU: N/A"

        vm = psutil.virtual_memory()
        used_gb = vm.used / 1024**3
        total_gb = vm.total / 1024**3
        record.ram = f"RAM: {used_gb:.1f}/{total_gb:.1f} GB ({vm.percent:.1f}%)"

        record.rank = self.rank

        return super().format(record)


def init_logger(rank=None):
    if rank == None:
        rank = os.getenv("RANK", 0)
    logger = logging.getLogger("gpu_logger")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        fmt = (
            "%(asctime)s | rank:%(rank)s | %(name)s | %(levelname)s | "
            "%(gpu)s | %(ram)s | %(message)s"
        )
        formatter = GPUFormatter(fmt=fmt, rank=rank)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Prevent logs from propagating to the root logger
    logger.propagate = False

    return logger


def incremental_state_dict(model, batch_size=30):
    """
    Incrementally gathers a model's parameters into a state dict on CPU.
    This processes parameters in batches to reduce peak GPU memory usage.

    Args:
        model (torch.nn.Module): The model to gather parameters from.
        batch_size (int): Number of parameters to process per batch.

    Returns:
        dict: A state dict with parameters moved to CPU.
    """
    state_dict = {}

    # List of (name, parameter) pairs
    params = list(model.named_parameters())
    total = len(params)

    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batch = params[start:end]

        # Create a list of just the parameters in this batch.
        batch_params = [p for _, p in batch]

        # Use DeepSpeed's context manager to gather the partitioned parameters.
        with deepspeed.zero.GatheredParameters(batch_params, modifier_rank=0):
            for name, param in batch:

                # Clone and move the parameter data to CPU.
                state_dict[name] = param.data.clone().cpu()

        torch.cuda.empty_cache()

    return state_dict


################################################################################
# Helper functions for parsing and checking answers
################################################################################


def check_equal(given_answer, real_answer):
    try:
        if str(given_answer).strip().lower() == str(real_answer).strip().lower():
            return True
        return False
        # This below sometimes gives errors.
        # return verify(parse(str(real_answer)), parse(str(given_answer)))
    except Exception as e:
        return False


def extract_answer(solution):
    # Find the last occurrence of '\boxed{'
    last_boxed_start = solution.rfind("\\boxed{")
    if last_boxed_start == -1:
        return None  # No \boxed{} found

    # Start counting braces from the character after '\boxed{'
    start_index = last_boxed_start + len("\\boxed{")
    stack = [1]  # Initial brace count
    end_index = start_index

    # Iterate from start_index to find the matching closing brace
    for i in range(start_index, len(solution)):
        if solution[i] == "{":
            stack.append(1)
        elif solution[i] == "}":
            if stack:
                stack.pop()
                if not stack:
                    end_index = i
                    break

    if end_index <= start_index:
        return None  # No matching closing brace found

    return solution[start_index:end_index]


################################################################################
# Reward wrappers.
################################################################################
from src.vllm.client import (
    get_end_of_conversation_reward,
    get_end_rm_reward,
    get_length_reward,
    get_thinking_reward,
)


def construct_end_rm_reward_func(server_port: 8000):
    def end_rm_reward_func(completions, **kwargs):
        return get_end_rm_reward(conversations=completions, server_port=server_port)

    return end_rm_reward_func


def construct_thinking_reward_func(server_port: 8000):
    def thinking_reward_func(completions, **kwargs):
        return get_thinking_reward(conversations=completions, server_port=server_port)

    return thinking_reward_func


def construct_end_of_conversation_reward_func(server_port: 8000):
    def end_of_conversation_reward_func(completions, **kwargs):
        return get_end_of_conversation_reward(
            conversations=completions, server_port=server_port
        )

    return end_of_conversation_reward_func


def construct_length_reward_func(server_port: 8000):
    def length_reward_func(completions, **kwargs):
        return get_length_reward(conversations=completions, server_port=server_port)

    return length_reward_func
