# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

##########################################################################################
# Slightly modified from official TRL Library.
# - Changed the `_generate_and_score_completions` function to work with our vLLM server.
# - Added `_compute_assistant_mask` function to mask user turns for loss computation.
##########################################################################################


import torch

_orig_load = torch.load

# Fixes an error related to model loading.
def _unsafe_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)


torch.load = _unsafe_load

import os
import gc
import shutil, os, re
import time
import wandb
import deepspeed
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union

import datasets
import torch
import transformers
from accelerate.utils import (
    gather,
    gather_object,
    is_peft_model,
    set_seed,
)
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available

from trl.data_utils import maybe_apply_chat_template
from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
from trl.models import prepare_deepspeed
from trl.extras.profiling import profiling_decorator
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import (
    pad,
    selective_log_softmax,
)
from vllm import RequestOutput
from src.utils.utils import incremental_state_dict
from src.vllm.client import sample_conversations, wait_batch
from src.utils.shared_memory import create_shared_state_dict, get_shareable_version
from src.utils.utils import init_logger, _ForwardRedirection
from src.grpo.config import ClassroomGRPOConfig

logger = init_logger()


RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(
                self.num_samples, generator=self.generator
            ).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [
            indexes[i : i + self.batch_size]
            for i in range(0, len(indexes), self.batch_size)
        ]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class RepeatRandomSampler(RepeatSampler):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "RepeatRandomSampler is deprecated and will be removed in version 0.18. Use RepeatSampler instead.",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)


def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
        >>> x = torch.arange(12).reshape(6, 2)
        >>> y = torch.arange(6).reshape(6, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> split_tensor_dict(tensor_dict, 3)
        [
            {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
            {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
            {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
        ]
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: (
                tensor[i * chunk_size : (i + 1) * chunk_size]
                if tensor is not None
                else None
            )
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]


class ClassroomGRPOTrainer(Trainer):

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: str,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: ClassroomGRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
    ):
        ################################################################################
        # Checks
        ################################################################################
        if not isinstance(model, str):
            raise ValueError(
                "The `model` argument must be a string representing the model name. "
                f"Got {type(model)} instead."
            )
        processing_class = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        # We need that eos_token_id != pad_token_id. If pad_token_id is None we set it to the last token of the vocab.
        if (
            processing_class.pad_token_id is None
            or processing_class.pad_token_id == processing_class.eos_token_id
        ):
            processing_class.pad_token_id = processing_class.vocab_size - 1
            processing_class.pad_token = processing_class.convert_ids_to_tokens(
                processing_class.pad_token_id
            )
            logger.warning(
                f"Setting pad_token_id to {processing_class.pad_token_id} ({processing_class.pad_token}) "
                "because it was None or equal to eos_token_id."
            )

        self.server_port = args.vllm_server_port
        self.use_experimental_shared_memory = args.use_experimental_shared_memory

        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = ClassroomGRPOConfig(f"{model_name}-GRPO")

        ################################################################################
        # Model loading
        ################################################################################
        self.model_name_or_path = model

        # Loading the policy model
        model_init_kwargs = args.model_init_kwargs or {}

        torch_dtype = model_init_kwargs.get("torch_dtype")
        if (
            isinstance(torch_dtype, torch.dtype)
            or torch_dtype == "auto"
            or torch_dtype is None
        ):
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
            model_init_kwargs["torch_dtype"] = torch_dtype
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )

        # Disable caching if gradient checkpointing is enabled (not supported)
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, **model_init_kwargs
        )

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Loading the reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        else:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path, **model_init_kwargs
            )

        ################################################################################
        # Final setups
        ################################################################################
        self.reward_funcs = reward_funcs

        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.temperature = args.temperature

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = (
            args.epsilon_high if args.epsilon_high is not None else args.epsilon
        )
        self.epsilon = args.epsilon
        self.use_liger_loss = args.use_liger_loss
        if self.use_liger_loss:
            self._forward_redirection = _ForwardRedirection()

        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle.
        self._step = 0

        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list)}

        self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
            beta=self.beta,
            epsilon_low=self.epsilon_low,
            epsilon_high=self.epsilon_high,
            temperature=self.temperature,
            use_ref_model=self.ref_model is not None,
        )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.num_processes = self.accelerator.num_processes
        self.num_nodes = int(os.getenv("NNODES", "1"))
        self.node_id = self.accelerator.process_index // torch.cuda.device_count()
        logger.info(
            f"Node ID: {self.node_id}, Process ID: {self.accelerator.process_index}, Number of processes: {self.num_processes}"
        )

        # This means number of completions per process.
        self.global_batch_size = (
            self.args.per_device_train_batch_size
            * self.num_processes
            * self.args.gradient_accumulation_steps
        )

        # We need it for the global_batch_size to be divisible by the number of generations.
        if self.global_batch_size % self.num_generations != 0:
            raise ValueError(
                f"Global batch size {self.global_batch_size} is not divisible by the number of generations {self.num_generations}."
            )
        self.number_of_problems_per_batch = (
            self.global_batch_size // self.num_generations
        )

        set_seed(args.seed, device_specific=True)

        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "answer"]

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        dataloader_params = {
            "batch_size": self._train_batch_size
            * self.args.gradient_accumulation_steps,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self) -> Sampler:

        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.global_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.gradient_accumulation_steps,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(
        self, model: PreTrainedModel, args: GRPOConfig
    ) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs
            or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    ####################################################################################
    # Mask user turns, used for multi-turn RL Training
    ####################################################################################
    def _compute_assistant_mask(self, input_ids):

        if "llama" in self.model_name_or_path.lower():
            # Llama-3 chat-template tokens
            start_header_tok = self.processing_class.encode(
                "<|start_header_id|>", add_special_tokens=False
            )[0]
            end_header_tok = self.processing_class.encode(
                "<|end_header_id|>", add_special_tokens=False
            )[0]
            assistant_tok = self.processing_class.encode(
                "assistant", add_special_tokens=False
            )[0]
            eot_tok = self.processing_class.encode(
                "<|eot_id|>", add_special_tokens=False
            )[0]

            def compute_mask_for_sequence(seq):
                mask = []
                inside_msg = inside_asst = False
                tokens_in = 0
                for tok in seq:
                    if tok == start_header_tok:
                        inside_msg = True
                        inside_asst = False
                        tokens_in = 0
                        mask.append(0)
                    elif inside_msg and tok == assistant_tok:
                        inside_asst = True
                        tokens_in = 0
                        mask.append(0)
                    elif tok == end_header_tok:
                        mask.append(0)
                    elif tok == eot_tok:
                        mask.append(1 if inside_asst else 0)
                        inside_msg = inside_asst = False
                    elif inside_msg and inside_asst:
                        mask.append(1 if tokens_in >= 3 else 0)
                        tokens_in += 1
                    else:
                        mask.append(0)
                print(f"Number of tokens in: {tokens_in}")
                return mask

            if input_ids.dim() == 1:
                return torch.tensor(
                    compute_mask_for_sequence(input_ids.tolist()),
                    device=input_ids.device,
                )
            elif input_ids.dim() == 2:
                return torch.tensor(
                    [compute_mask_for_sequence(seq.tolist()) for seq in input_ids],
                    device=input_ids.device,
                )
            else:
                raise ValueError(
                    "Unsupported input_ids dimensions: expected 1D or 2D tensor."
                )
        else:
            # Get the special tokens.
            start_token = self.processing_class.apply_chat_template(
                [{"role": "system", "content": ""}]
            )[0]
            system_token = self.processing_class.encode("system")[0]
            user_token = self.processing_class.encode("user")[0]
            assistant_token = self.processing_class.encode("assistant")[0]
            eos_token = self.processing_class.eos_token_id

            def compute_mask_for_sequence(seq):
                """
                Given a 1D list (or tensor converted to list) of tokens,
                splits it into messages based on the start token and computes the mask.
                """
                mask = []
                inside_assistant_message = False
                inside_a_message = False
                tokens_in = 0
                for token in seq:
                    if token == start_token:
                        inside_a_message = True
                        mask.append(0)
                    elif inside_a_message and token == assistant_token:
                        inside_assistant_message = True
                        tokens_in = 0
                        mask.append(0)
                    elif token == eos_token:
                        mask.append(1 if inside_assistant_message else 0)
                        inside_a_message = False
                        inside_assistant_message = False
                    elif inside_a_message and inside_assistant_message:
                        mask.append(1 if tokens_in >= 3 else 0)
                        tokens_in += 1
                    else:
                        mask.append(0)

                assert len(mask) == len(seq)
                return mask

            # If input_ids is a 1D tensor (a single sequence).
            if input_ids.dim() == 1:
                # Convert to a list of ints if needed.
                seq = input_ids.tolist()
                mask = compute_mask_for_sequence(seq)
                return torch.tensor(mask, device=input_ids.device)

            # If input_ids is a 2D tensor (batch of sequences).
            elif input_ids.dim() == 2:
                batch_masks = []
                # Process each sequence in the batch independently.
                for seq_tensor in input_ids:
                    seq = seq_tensor.tolist()
                    mask = compute_mask_for_sequence(seq)
                    batch_masks.append(mask)
                # Convert the list of masks into a tensor.
                return torch.tensor(batch_masks, device=input_ids.device)

            else:
                raise ValueError(
                    "Unsupported input_ids dimensions: expected 1D or 2D tensor."
                )

    @profiling_decorator
    def _get_last_hidden_state(
        self, model, input_ids, attention_mask, logits_to_keep=None
    ):
        # unwrap the model to access the model.model
        unwrapped_model = self.accelerator.unwrap_model(model)
        last_hidden_state = unwrapped_model.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[
                :, -logits_to_keep:, :
            ]  # (B, logits_to_keep, H)
        return last_hidden_state

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(
        self, model, input_ids, attention_mask, logits_to_keep, batch_size=None
    ) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(
            0
        )  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]

            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                logits_to_keep=logits_to_keep + 1,
            ).logits
            logits = logits[
                :, :-1, :
            ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            logps = selective_log_softmax(
                logits, input_ids_batch
            )  # compute logprobs for the input tokens
            all_logps.append(logps)
        return torch.cat(all_logps, dim=0)

    @profiling_decorator
    def _prepare_inputs(
        self, accumulated_local_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:

        generate_every = self.args.gradient_accumulation_steps * self.num_iterations
        if self._step % generate_every == 0 or self._buffered_inputs is None:
            # self._buffered_inputs=None can occur when resuming from a checkpoint
            accumulated_local_batch = self._generate_and_score_completions(
                accumulated_local_batch
            )
            self._buffered_inputs = split_tensor_dict(
                accumulated_local_batch, self.args.gradient_accumulation_steps
            )
        inputs = self._buffered_inputs[
            self._step % self.args.gradient_accumulation_steps
        ]
        self._step += 1

        return inputs

    def _save_only_model(self, output_dir: str):
        """
        Save only the model, then delete any other checkpoint-* folders
        under the same `policy` directory so that only the latest remains.
        """
        CHECKPOINT_RE = re.compile(r"^checkpoint-\d+$")
        if self.accelerator.is_main_process:

            # Save current model
            os.makedirs(output_dir, exist_ok=True)
            # Prune older checkpoints
            policy_root = os.path.dirname(output_dir)  # …/policy
            for d in os.listdir(policy_root):
                if CHECKPOINT_RE.match(d):
                    shutil.rmtree(os.path.join(policy_root, d), ignore_errors=True)

        self.accelerator.wait_for_everyone()
        self.save_model(output_dir, _internal_call=True)

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        inputs = gather_object(inputs)
        # sort inputs by prompt
        inputs = sorted(inputs, key=lambda x: str(x))
        prompts = [x["prompt"] for x in inputs]
        answers = [x["answer"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        prompt_inputs = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        ############################################################################################################
        # vLLM Classroom generation
        # ############################################################################################################

        state_dict = None
        meta_info = None
        meta_info_shared = {}

        # If we are using shared memory.
        if self.use_experimental_shared_memory:
            state_dict = incremental_state_dict(self.model)
            if self.accelerator.is_local_main_process:
                logger.info("Creating shared memory")

                meta_info = create_shared_state_dict(state_dict)
                meta_info_shared = get_shareable_version(meta_info)

            # Free up memory
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Shared memory created")

        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()
        # We also sometimes save the model every self.args.save_policy_to_disk_every_n_steps steps.
        # We do this to avoid a bug with vllm when changing weights at tensor parallel=4.
        if self.state.global_step % self.args.save_policy_to_disk_every_n_steps == 0:
            logger.info(f"Saving the model to {self.args.output_dir}")
            self._save_only_model(
                os.path.join(
                    self.args.output_dir,
                    "policy",
                    f"checkpoint-{self.state.global_step}",
                )
            )
            self.accelerator.wait_for_everyone()
            logger.info(f"Model saved to {self.args.output_dir}")

        all_prompts_text = prompts_text
        for _ in range(5):
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1)
        
        # We only generate on main processes
        if self.accelerator.is_local_main_process:
            logger.info(f"Entered with {len(all_prompts_text)} prompts")
            ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
            local_answers = answers[:: self.num_generations]

            num_prompts_per_node = len(ordered_set_of_prompts) // self.num_nodes

            # We get our slice of the prompts
            start_slice = self.node_id * num_prompts_per_node
            end_slice = (self.node_id + 1) * num_prompts_per_node
            ordered_set_of_prompts = ordered_set_of_prompts[start_slice:end_slice]
            local_answers = local_answers[start_slice:end_slice]
            # we duplicate the answers by num_generations
            real_answers = []
            for answer in local_answers:
                for _ in range(self.num_generations):
                    real_answers.append(answer)

            logger.info(
                f"Generating completions for {len(ordered_set_of_prompts)} unique problems, with {self.num_generations} generations each and answers {real_answers}"
            )

            all_outputs: list[RequestOutput] = sample_conversations(
                ordered_set_of_prompts,
                answers=real_answers,
                meta=meta_info_shared,
                server_port=self.server_port,
                num_samples_per_problem=self.num_generations,
                tokenizer=self.tokenizer,
            )
            logger.info(f"Generated completions for {len(all_outputs)} prompts")

            if self.use_experimental_shared_memory:
                logger.info("Closing the shared memory")

                for meta in meta_info.values():
                    shm = meta["_shm_obj"]
                    shm.close()
                    shm.unlink()
        else:
            all_outputs = []
            if self.use_experimental_shared_memory:
                logger.info("Deleting state_dict memory")
                try:
                    for meta in meta_info.values():
                        shm = meta["_shm_obj"]
                        shm.close()
                        shm.unlink()
                except:
                    pass
                gc.collect()
                torch.cuda.empty_cache()
            time.sleep(2 * 60)
            logger.info("Waiting for the batch to be ready")
            wait_batch(
                server_port=self.server_port,
            )

        self.accelerator.wait_for_everyone()
        if self.use_experimental_shared_memory:
            logger.info("Closing the shared memory")
            try:
                for meta in meta_info.values():
                    shm = meta["_shm_obj"]
                    shm.close()
                    shm.unlink()
            except Exception as e:
                pass
            logger.info("Shared memory closed")
        self.accelerator.wait_for_everyone()
        all_outputs_ = gather_object(all_outputs)
        all_outputs = all_outputs_

        all_inputs = inputs
        all_prompts = prompts
        all_completion_ids = [
            list(output.outputs[0].token_ids) for output in all_outputs
        ]

        step_size = len(all_outputs) // self.num_processes
        start_slice = self.accelerator.process_index * step_size

        logger.info(f"Start slice: {start_slice}, Step size: {step_size}")

        outputs = all_outputs[start_slice : start_slice + step_size]
        prompts = prompts[start_slice : start_slice + step_size]
        prompts_text = prompts_text[start_slice : start_slice + step_size]
        prompt_ids = prompt_ids[start_slice : start_slice + step_size]
        prompt_mask = prompt_mask[start_slice : start_slice + step_size]
        inputs = inputs[start_slice : start_slice + step_size]
        completion_ids = all_completion_ids[start_slice : start_slice + step_size]

        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(
            completion_ids, padding_value=self.processing_class.pad_token_id
        )
        prompt_completion_ids = completion_ids

        # TODO: Technically problematic if pad_token_id is equal to eos_token_id.
        completion_mask = (completion_ids != self.processing_class.pad_token_id).int()
        attention_mask = completion_mask
        logits_to_keep = completion_ids.size(1) - 1

        batch_size = self.args.per_device_train_batch_size

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size,
                    )

        # Decode the generated completions
        all_completions_text = self.processing_class.batch_decode(
            all_completion_ids, skip_special_tokens=False
        )
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=False
        )
        completions = completions_text

        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        if self.num_nodes == 1:
            for i, reward_func in enumerate(self.reward_funcs):
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {
                    key: [example[key] for example in inputs] for key in keys
                }
                output_reward_func = reward_func(
                    prompts=prompts, completions=completions, **reward_kwargs
                )
                rewards_per_func[:, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device
                )
        else:
            # We check in which node this problem is.
            num_completions_per_node = len(all_outputs) // self.num_nodes

            rewards_per_func = torch.zeros(
                num_completions_per_node, len(self.reward_funcs), device=device
            )
            if not self.accelerator.is_local_main_process:
                rewards_per_func = (
                    torch.ones(
                        num_completions_per_node, len(self.reward_funcs), device=device
                    )
                    * -199
                )
            else:

                start_slice = self.node_id * num_completions_per_node
                end_slice = (self.node_id + 1) * num_completions_per_node

                for i, reward_func in enumerate(self.reward_funcs):
                    keys = [
                        key
                        for key in all_inputs[start_slice:end_slice][0]
                        if key not in ["prompt", "completion"]
                    ]
                    reward_kwargs = {
                        key: [
                            example[key]
                            for example in all_inputs[start_slice:end_slice]
                        ]
                        for key in keys
                    }
                    output_reward_func = reward_func(
                        prompts=all_prompts[start_slice:end_slice],
                        completions=all_completions_text[start_slice:end_slice],
                        **reward_kwargs,
                    )
                    rewards_per_func[:, i] = torch.tensor(
                        output_reward_func, dtype=torch.float32, device=device
                    )

        rewards_per_func = gather(rewards_per_func)

        rewards_per_func = rewards_per_func[(rewards_per_func != -199).any(dim=1)]

        rewards = (rewards_per_func).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = rewards - mean_grouped_rewards
        if self.args.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )
        self._metrics[mode]["completion_length"].append(completion_length)

        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}"].append(
                reward_per_func[i].item()
            )

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            table = {
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "answer": gather_object([x["answer"] for x in inputs]),
                "reward": rewards.tolist(),
            }

            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({f"completions_{self._step}": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }

    def compute_old_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        # input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        input_ids = completion_ids  # New: we only need the completions
        # attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        # attention_mask = completion_mask # New: we only need the completions
        attention_mask = (completion_ids != self.processing_class.pad_token_id).int()
        # we remove one token from completion_mask at the start.
        completion_mask = completion_mask[:, 1:]
        logits_to_keep = (
            completion_ids.size(1) - 1
        )  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            inputs["old_per_token_logps"]
            if self.num_iterations > 1
            else per_token_logps.detach()
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # We mask with assistant loss too.
        assistant_mask = self._compute_assistant_mask(completion_ids).int()[:, 1:]
        loss = (per_token_loss * completion_mask * assistant_mask).sum() / (
            assistant_mask * completion_mask
        ).sum()
        # loss = (per_token_loss * completion_mask).sum() / (completion_mask).sum()

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (
                (per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
            ).mean()
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).mean().item()
            )

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )
        torch.cuda.empty_cache()
        return loss

    @profiling_decorator
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        torch.cuda.empty_cache()

        if not self.use_liger_loss:
            return self.compute_old_loss(
                model, inputs, return_outputs, num_items_in_batch
            )
        
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = completion_ids

        attention_mask = (completion_ids != self.processing_class.pad_token_id).int()
        completion_mask = completion_mask
        logits_to_keep = completion_ids.size(1) - 1

        # We mask with assistant loss too.
        assistant_mask = self._compute_assistant_mask(completion_ids).int()

        # get the last hidden state of the model
        last_hidden_state = self._get_last_hidden_state(
            model, input_ids, attention_mask, logits_to_keep
        )
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        with deepspeed.zero.GatheredParameters(
            [unwrapped_model.lm_head.weight, unwrapped_model.lm_head.bias],
            modifier_rank=None,
        ):
            # compute loss and metrics using liger grpo loss
            loss, metrics = self._forward_redirection(
                self.model,
                unwrapped_model,
                self.liger_grpo_loss,
                last_hidden_state,
                unwrapped_model.lm_head.weight,
                completion_ids[:, 1:],
                (completion_mask * assistant_mask)[:, 1:],
                inputs["advantages"],
                inputs["ref_per_token_logps"],
                inputs["old_per_token_logps"],
            )

        # Extract metrics from the liger_grpo_loss output
        # KL divergence is the first metric when beta is non-zero
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]
        self.accelerator.wait_for_everyone()
        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).mean().item()
            )
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }  # average the metrics

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()