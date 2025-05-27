import requests
from typing import List
from vllm import RequestOutput, CompletionOutput
from transformers import PreTrainedTokenizer

####################################################################################################
# The following functions are used to interact with the vLLM server
# - For sampling conversations
# - For getting rewards
####################################################################################################


def sample_conversations(
    problems: List[str],
    answers: List[str],
    meta: dict = {},
    server_port: int = 8000,
    num_samples_per_problem: int = 1,
    tokenizer: PreTrainedTokenizer = None,
) -> List[RequestOutput]:
    server_url = f"http://localhost:{server_port}/sample_conversations"

    actual_problems = []
    for problem in problems:
        actual_problems.extend([problem] * num_samples_per_problem)
    answers = [str(answer) for answer in answers]
    response = requests.post(
        server_url, json={"problems": actual_problems, "meta": meta, "answers": answers}
    )
    response.raise_for_status()

    response_list = response.json()
    results: list[str] = [item for item in response_list]

    request_outputs = []
    for i in range(0, len(results)):
        request_output = RequestOutput(
            request_id="",
            prompt="",
            outputs=[
                CompletionOutput(
                    index=0,
                    text=tokenizer.apply_chat_template(results[i], tokenize=False),
                    token_ids=tokenizer.apply_chat_template(results[i], tokenize=True),
                    cumulative_logprob=0.0,
                    logprobs=[],
                )
            ],
            prompt_token_ids=[],
            prompt_logprobs=[],
            finished=True,
        )
        request_outputs.append(request_output)

    return request_outputs


def get_end_rm_reward(
    conversations: List[str],
    server_port: int = 8000,
) -> List[float]:
    server_url = f"http://localhost:{server_port}/get_end_rm_reward"

    response = requests.post(server_url, json={"conversations": conversations})
    response.raise_for_status()

    rewards = response.json()
    return rewards


def get_thinking_reward(
    conversations: List[str],
    server_port: int = 8000,
) -> List[float]:
    server_url = f"http://localhost:{server_port}/get_thinking_reward"

    response = requests.post(server_url, json={"conversations": conversations})
    response.raise_for_status()

    rewards = response.json()
    return rewards


def get_end_of_conversation_reward(
    conversations: List[str],
    server_port: int = 8000,
) -> List[float]:
    server_url = f"http://localhost:{server_port}/get_end_of_conversation_reward"

    response = requests.post(server_url, json={"conversations": conversations})
    response.raise_for_status()

    rewards = response.json()
    return rewards


def get_length_reward(
    conversations: List[str],
    server_port: int = 8000,
) -> List[float]:
    server_url = f"http://localhost:{server_port}/get_length_reward"

    response = requests.post(server_url, json={"conversations": conversations})
    response.raise_for_status()

    rewards = response.json()
    return rewards


####################################################################################################


def wait_batch(server_port: int = 8000):
    """
    Sends a request to the FastAPI server's /wait_batch endpoint.
    """
    server_url = f"http://localhost:{server_port}/wait_batch"

    response = requests.get(server_url)
    response.raise_for_status()

    return response.json()
