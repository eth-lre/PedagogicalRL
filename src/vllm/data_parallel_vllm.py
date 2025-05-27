import os
import gc
import math
import torch
from typing import List, Optional
from multiprocess import get_context
from multiprocess.queues import Empty
from huggingface_hub import snapshot_download
from src.utils.utils import init_logger

logger = init_logger()


class InferenceTask:
    GENERATE = "generate"
    REWARD = "reward"
    EMBEDDING = "embedding"
    CLASSIFY = "classify"


class ParallelvLLMInference:
    def __init__(
        self,
        model_path: str,
        n_instances: Optional[int] = None,
        gpus_per_instance: int = 2,
        gpu_memory_utilization: float = 0.5,
        max_model_len: int = 9000,
        max_num_seqs: int = 5,
        enforce_eager: bool = False,
        model_save_path: Optional[str] = None,
        use_lora: bool = False,
        load_and_unload: bool = True,
        max_number_of_instances: int = -1,
        inference_task: InferenceTask = InferenceTask.GENERATE,
        bits_and_bytes: bool = False,
        enable_sleep_mode: bool = True,
        from_0: bool = True,
        use_v0: bool = False,
        logging_enabled: bool = False,
        log_file_path: Optional[str] = None,
    ):
        self.model_path = model_path
        self.model_save_path = model_save_path
        self.total_gpus = torch.cuda.device_count()
        self.gpus_per_instance = gpus_per_instance
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.enforce_eager = enforce_eager
        self.use_lora = use_lora
        self.load_and_unload = load_and_unload
        self.inference_task = inference_task
        self.bits_and_bytes = bits_and_bytes
        self.enable_sleep_mode = enable_sleep_mode
        self.use_v0 = use_v0
        self.logging_enabled = logging_enabled
        self.log_file = (
            os.path.join(log_file_path, "input_prompts.txt") if log_file_path else ""
        )

        if self.load_and_unload and not self.enable_sleep_mode:
            raise ValueError("Cannot use load_and_unload without enabling sleep mode")

        logger.info(
            f"Total GPUs: {self.total_gpus}, using {self.gpus_per_instance} per instance"
        )

        # Try downloading the model to cache
        try:
            snapshot_download(self.model_path)
        except Exception as e:
            logger.warning(f"Could not download model: {e}; will load from cache/local")

        # Determine number of instances
        if n_instances:
            required = n_instances * gpus_per_instance
            if required > self.total_gpus:
                raise ValueError(
                    f"Need {required} GPUs, only {self.total_gpus} available"
                )
            self.n_instances = n_instances
        else:
            self.n_instances = max(1, self.total_gpus // self.gpus_per_instance)

        # Build GPU groups [[0,1], [2,3], ...]
        self.gpu_groups = [
            list(range(i, i + self.gpus_per_instance))
            for i in range(
                0, self.n_instances * self.gpus_per_instance, self.gpus_per_instance
            )
        ]
        if not from_0:
            self.gpu_groups = [
                [(self.total_gpus - 1 - gpu) for gpu in group]
                for group in self.gpu_groups
            ]

        if max_number_of_instances > 0:
            self.n_instances = min(max_number_of_instances, self.n_instances)
            self.gpu_groups = self.gpu_groups[: self.n_instances]
            logger.info(f"Limiting number of instances to {self.n_instances}")

        # Track last checkpoint ID seen
        self._last_reload_ckpt = None

        # Start worker processes
        self._start_workers()

    def _get_latest_checkpoint_id(self) -> Optional[int]:
        """Scan model_save_path for 'checkpoint-<n>' folders and return max(n), or None."""
        if self.model_save_path and os.path.isdir(self.model_save_path):
            ids = []
            for d in os.listdir(self.model_save_path):
                if d.startswith("checkpoint-"):
                    parts = d.split("-")
                    if parts[-1].isdigit():
                        ids.append(int(parts[-1]))
            if ids:
                return max(ids)
        return None

    def _start_workers(self):
        """Spawn worker processes and wait until they're READY."""
        self.ctx = get_context("spawn")
        self.task_queues = [self.ctx.Queue() for _ in range(self.n_instances)]
        self.result_queues = [self.ctx.Queue() for _ in range(self.n_instances)]
        self.processes = []

        for idx, gpu_group in enumerate(self.gpu_groups):
            p = self.ctx.Process(
                target=self._worker_loop,
                args=(
                    gpu_group,
                    self.task_queues[idx],
                    self.result_queues[idx],
                    self.inference_task,
                ),
            )
            p.start()
            self.processes.append(p)

        # Wait for each worker to signal "READY"
        for q in self.result_queues:
            q.get()

    def _reload_workers(self):
        latest = self._get_latest_checkpoint_id()
        if latest is not None:
            new_path = os.path.join(self.model_save_path, f"checkpoint-{latest}")
            logger.info(f"Reload: switching model_path to {new_path}")
            self.model_path = new_path
            self._last_reload_ckpt = latest

        self.cleanup()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        self._start_workers()
        torch.cuda.empty_cache()

    def run_batch(
        self, messages: List[str], sampling_params: dict, meta: Optional[dict] = None
    ):

        current = self._get_latest_checkpoint_id()
        if current is not None and current != self._last_reload_ckpt:
            logger.info(
                f"New checkpoint {current} detected (was {self._last_reload_ckpt}); reloading workers."
            )
            self._reload_workers()

        indexed = list(enumerate(messages))
        total = len(indexed)
        chunk_size = math.ceil(total / self.n_instances)
        chunks = [
            indexed[i * chunk_size : (i + 1) * chunk_size]
            for i in range(self.n_instances)
        ]

        for i, chunk in enumerate(chunks):
            self.task_queues[i].put((chunk, sampling_params, meta))

        results = []
        received = 0
        expected = total
        while received < expected:
            for q in self.result_queues:
                try:
                    batch = q.get(block=True, timeout=0.1)
                    if batch not in ("READY", "SLEEP_DONE"):
                        results.extend(batch)
                        received += len(batch)
                except Empty:
                    continue

        return [out for _, out in sorted(results, key=lambda x: x[0])]

    def sleep(self):
        for q in self.task_queues:
            q.put("SLEEP")
        for q in self.result_queues:
            resp = q.get()
            if resp != "SLEEP_DONE":
                logger.error("Unexpected response to SLEEP:", resp)
        logger.info("All workers are now asleep.")

    def cleanup(self):
        for q in self.task_queues:
            q.put(None)
        for p in self.processes:
            p.join()
        for q in (*self.task_queues, *self.result_queues):
            q.close()
        torch.cuda.empty_cache()
        logger.info("Cleaned up all resources")

    def _handle_reward_task(self, llm, prompts: List[str], tokenizer):
        prompts = [
            tokenizer.decode(tokenizer.encode(p)[: self.max_model_len - 1])
            for p in prompts
        ]
        return llm.encode(prompts)

    def _handle_embedding_task(self, llm, prompts: List[str]):
        return llm.embed(prompts)

    def _handle_classify_task(self, llm, prompts: List[str]):
        return llm.classify(prompts)

    def _handle_causallm_task(
        self,
        llm,
        prompts: List[str],
        sampling_params: dict,
        meta: Optional[dict],
        counter: int,
    ):
        if meta is not None:
            from ..utils.shared_memory import load_shared_state_dict

            state = load_shared_state_dict(meta).items()
            llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(
                state
            )
            return llm.chat(prompts, sampling_params=sampling_params)

        return llm.chat(prompts, sampling_params=sampling_params)

    def _worker_loop(
        self, gpu_group: List[int], task_queue, result_queue, inference_task
    ):
        import os
        import gc

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_group))

        if self.use_v0:
            os.environ["VLLM_USE_V1"] = "0"
        else:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        from vllm import LLM

        print(f"Worker on GPUs {gpu_group} initializing for task '{inference_task}'...")

        llm = LLM(
            model=self.model_path,
            task=str(inference_task),
            tensor_parallel_size=self.gpus_per_instance,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            enable_lora=self.use_lora,
            enforce_eager=self.enforce_eager,
            enable_prefix_caching=False,
            enable_sleep_mode=self.enable_sleep_mode,
            quantization="bitsandbytes" if self.bits_and_bytes else None,
            load_format="bitsandbytes" if self.bits_and_bytes else "auto",
        )
        tokenizer = llm.get_tokenizer()

        if self.load_and_unload:
            llm.sleep()

        result_queue.put("READY")
        counter = 0

        while True:
            task = task_queue.get()
            if task is None:
                break
            if task == "SLEEP":
                try:
                    llm.sleep()
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error during sleep: {e}")
                result_queue.put("SLEEP_DONE")
                continue

            if self.inference_task == InferenceTask.GENERATE:
                llm.wake_up()

            chunk, sampling_params, meta = task
            prompts = [p for _, p in chunk]

            if inference_task == InferenceTask.REWARD:
                outs = self._handle_reward_task(llm, prompts, tokenizer)
            elif inference_task == InferenceTask.EMBEDDING:
                outs = self._handle_embedding_task(llm, prompts)
            elif inference_task == InferenceTask.CLASSIFY:
                outs = self._handle_classify_task(llm, prompts)
            else:
                outs = self._handle_causallm_task(
                    llm, prompts, sampling_params, meta, counter
                )
                counter += 1

            gc.collect()
            torch.cuda.empty_cache()
            if self.load_and_unload:
                llm.sleep()

            result_queue.put([(idx, out) for (idx, _), out in zip(chunk, outs)])

        # Final cleanup
        from vllm.distributed.parallel_state import (
            destroy_model_parallel,
            destroy_distributed_environment,
        )
        import contextlib

        destroy_model_parallel()
        destroy_distributed_environment()
        del llm.llm_engine.model_executor
        del llm
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()
