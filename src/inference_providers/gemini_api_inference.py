################################################################
# Simple Inference class for Google Gemini API
################################################################

import os
import time
import random
import concurrent.futures

from dotenv import load_dotenv
import google.generativeai as genai
from vllm import SamplingParams, RequestOutput, CompletionOutput
from src.utils.utils import init_logger
logger = init_logger()

class GeminiInference:
    def __init__(self, model_name: str):
        load_dotenv()
        self.model_name = model_name
        primary_key = os.getenv("GEMINI_API_KEY")
        if not primary_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment")
        genai.configure(api_key=primary_key)

    def run_batch(self, conversations: list, sampling_params: SamplingParams, meta=None, max_retries=10000000):

        def _transform_history(conv):
            return [
                {"role": "user" if msg["role"] == "user" else "model", "parts": [{"text": msg["content"]}]}
                for msg in conv[:-1]
            ]

        def _execute_one(conversation):
            backoff = 1
            current_key = os.getenv("GEMINI_API_KEY")

            for attempt in range(1, max_retries + 1):
                try:
                    # re-configure for this key
                    genai.configure(api_key=current_key)

                    # build generation_config
                    gen_cfg = {
                        "temperature": sampling_params.temperature,
                        "top_p": sampling_params.top_p,
                        "top_k": getattr(sampling_params, "top_k", 0),
                        "max_output_tokens": sampling_params.max_tokens,
                        "response_mime_type": "text/plain",
                    }
                    model = genai.GenerativeModel(
                        model_name=self.model_name,
                        generation_config=gen_cfg,
                    )

                    # chat history + last user turn
                    history = _transform_history(conversation)
                    chat = model.start_chat(history=history)
                    last_user = conversation[-1]["content"]

                    # generate n completions
                    completion_outputs = []
                    for i in range(sampling_params.n):
                        resp = chat.send_message(last_user)
                        completion_outputs.append(
                            CompletionOutput(
                                index=i,
                                text=resp.text,
                                token_ids=[],
                                cumulative_logprob=0.0,
                                logprobs=[]
                            )
                        )

                    return RequestOutput(
                        request_id="",
                        prompt="",
                        outputs=completion_outputs,
                        prompt_token_ids=[],
                        prompt_logprobs=[],
                        finished=True
                    )

                except Exception as e:
                    logger.warning(f"Attempt {attempt} failed: {e}")
                    if attempt == max_retries:
                        logger.error(f"All {max_retries} attempts failed.")
                        return RequestOutput(
                            request_id="",
                            prompt="",
                            outputs=[CompletionOutput(i, None, [], 0.0, []) for i in range(sampling_params.n)],
                            prompt_token_ids=[],
                            prompt_logprobs=[],
                            finished=False
                        )

                    # backoff + jitter
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30) + random.uniform(0, 5)

                    fallbacks = [
                        k for k in os.environ
                        if k.startswith("GEMINI_API_KEY") and k != "GEMINI_API_KEY"
                    ]
                    if fallbacks:
                        chosen = random.choice(fallbacks)
                        current_key = os.getenv(chosen)
                        logger.warning(f"Switching to fallback key: {chosen}")

        outputs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=80) as executor:
            futures = [executor.submit(_execute_one, conv) for conv in conversations]
            for fut in futures:
                outputs.append(fut.result())

        return outputs

    def sleep(self):
        pass
