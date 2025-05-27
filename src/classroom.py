#####################################################################
# Main Classroom Logic Class. Here is where the rollouts are created.
#####################################################################

from functools import lru_cache
import re
import gc
import torch
import time
import json
import pandas as pd
from tqdm import tqdm
from enum import Enum
from typing import Dict, List
from random import choice
from jinja2 import Template
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm import PoolingOutput, SamplingParams, RequestOutput
from config.train_rl_model import (
    StudentModelConfig,
    TeacherModelConfig,
    JudgeModelConfig,
    RewardModelConfig,
    GenerationConfig,
)
from src.vllm.data_parallel_vllm import ParallelvLLMInference, InferenceTask
from src.utils.utils import check_equal, extract_answer
from src.inference_providers.open_router_inference import OpenRouterInference
from src.inference_providers.gemini_api_inference import GeminiInference
import logging

logger = logging.getLogger(__name__)


# Each conversation will have a small state machine to track the conversation state.
class ConversationState(Enum):
    START = 0
    TEACHER_TURN = 1
    STUDENT_TURN = 2
    JUDGE_TURN = 4
    GENERATE_SOLUTION = 5
    REWARD_TURN = 6
    END = 7

# This is the type of conversation we are having.
class ConversationType(Enum):
    GUIDED = 0
    ATTEMPTED = 1

# This is the decision the judge can make.
class JudgeDecision(Enum):
    OK = "OK"
    REJECT = "REJECT"

# This is the response from the judge. Which also includes the reasoning behind the decision.
class JudgeResponse(BaseModel):
    reasoning: str
    decision: JudgeDecision


@lru_cache(maxsize=1000)
def read_template(path: str) -> Template:
    return Template(open(path).read())


@lru_cache(maxsize=1)
def get_tokenizer(tokenizer_to_use: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(tokenizer_to_use)


class Conversation:
    def __init__(
        self,
        problem: str,
        answer: str,
        generation_cfg: GenerationConfig,
        forced_type: ConversationType = None,
        forced_student_name: str = None,
    ):
        self.problem = problem
        self.answer = answer
        self.generation_cfg = generation_cfg
        self.conversation = []  # list of dicts: {role: str, content: str}
        self.state = ConversationState.START
        problem_hash = hash(problem)
        self.type: ConversationType = (
            [ConversationType.GUIDED, ConversationType.ATTEMPTED][problem_hash % 2]
            if forced_type is None
            else forced_type
        )
        self.student_name = (
            generation_cfg.student_names[
                problem_hash % len(generation_cfg.student_names)
            ]
            if forced_student_name is None
            else forced_student_name
        )

        self.student_persona = list(
            generation_cfg.student_personas_prompts_paths.keys()
        )[
            problem_hash
            % len(list(generation_cfg.student_personas_prompts_paths.keys()))
        ]
        self.system_prompt_student = read_template(
            generation_cfg.student_personas_prompts_paths[self.student_persona]
        ).render(student_name=self.student_name, problem=problem)
        self.system_prompt_teacher = read_template(
            generation_cfg.teacher_prompt_path
        ).render(
            student_name=self.student_name,
            problem=problem,
            include_thinking=generation_cfg.use_thinking,
        )
        self.system_prompt_student_attempt = read_template(
            generation_cfg.student_initial_attempt_prompt_path
        ).render(problem=problem)
        self.initial_attempt_wrapper = read_template(
            generation_cfg.initial_attempt_wrapper_prompt_path
        )
        self.student_final_prompt = read_template(
            generation_cfg.student_final_prompt_path
        ).render()
        self.student_attempt = read_template(
            generation_cfg.student_attempt_prompt_path
        ).render(problem=problem)

        self.judge_evaluation_type = None
        self.judge_decisions: Dict[str, list[JudgeResponse]] = {}
        self.solutions: list[str] = []
        self.rewards: list[float] = []

        self.tokenizer = get_tokenizer(generation_cfg.tokenizer_to_use)

        self.initial_attempts = []
        self.initial_rewards = []

        self.failed_judges = False

        # SocraticLM special treatment.
        if (
            "teacher_message" in open(generation_cfg.teacher_prompt_path).read()
            and "teacher_message" in open(generation_cfg.teacher_prompt_path).read()
        ):
            self.system_prompt_teacher = read_template(
                generation_cfg.teacher_prompt_path
            ).render()
            start_user_message = read_template(
                generation_cfg.teacher_prompt_path
            ).render(problem=problem, user_message=True)
            teacher_start_message = read_template(
                generation_cfg.teacher_prompt_path
            ).render(teacher_message=True)
            self.conversation.append({"role": "student", "content": start_user_message})
            self.conversation.append(
                {"role": "teacher", "content": teacher_start_message}
            )
            self.state = ConversationState.STUDENT_TURN

    @classmethod
    def from_dataframe(
        cls, row: any, generation_cfg: GenerationConfig
    ) -> "Conversation":

        # Extract the answer (if not present, default to an empty string)
        answer = row.get("Answer", "")

        # Convert the 'Type' column to a ConversationType enum if available.
        forced_type = None
        type_val = row.get("Type")
        if isinstance(type_val, str):
            if type_val in ConversationType.__members__:
                forced_type = ConversationType[type_val]
        elif type_val in ConversationType.__members__:
            forced_type = ConversationType[type_val]

        # Initialize the Conversation instance.
        instance = cls(
            problem=row["Problem"],
            answer=answer,
            generation_cfg=generation_cfg,
            forced_type=forced_type,
            forced_student_name=row.get("Student Name"),
        )

        # Restore conversation list. If stored as string, assume JSON and load.
        conv_data = row.get("Conversation", [])
        if isinstance(conv_data, str):
            try:
                conv_data = eval(conv_data)
            except Exception as e:
                raise ValueError(f"Failed to load 'Conversation' field: {e}")
        instance.conversation = conv_data

        # Restore the conversation state.
        state_val = row.get("State")
        if isinstance(state_val, str):
            if state_val in ConversationState.__members__:
                instance.state = ConversationState[state_val]
        elif state_val in ConversationState.__members__:
            instance.state = ConversationState[state_val]

        # Restore student persona and name.
        instance.student_persona = row.get("Student Persona", instance.student_persona)

        # Restore judge decisions.
        jd_data = row.get("Judge Decisions", {})
        if isinstance(jd_data, str):
            try:
                jd_data = eval(jd_data)
            except Exception as e:
                raise ValueError(f"Failed to load 'Judge Decisions': {e}")
        jd = {}
        for key, decisions in jd_data.items():
            # If decisions is a string, load it as JSON.
            if isinstance(decisions, str):
                try:
                    decisions = eval(decisions)
                except Exception as e:
                    raise ValueError(
                        f"Failed to load judge decisions for key {key}: {e}"
                    )
            jd[key] = [
                JudgeResponse(
                    reasoning=d["reasoning"], decision=JudgeDecision[d["decision"]]
                )
                for d in decisions
            ]
        instance.judge_decisions = jd

        # Restore solutions.
        solutions = row.get("Solutions", [])
        if isinstance(solutions, str):
            try:
                solutions = eval(solutions)
            except Exception as e:
                raise ValueError(f"Failed to load 'Solutions': {e}")
        instance.solutions = solutions

        # Restore rewards.
        rewards = row.get("Rewards", [])
        if isinstance(rewards, str):
            try:
                rewards = eval(rewards)
            except Exception as e:
                raise ValueError(f"Failed to load 'Rewards': {e}")
        instance.rewards = rewards

        # Restore initial attempts.
        initial_attempts = row.get("Initial Attempts", [])
        if isinstance(initial_attempts, str):
            try:
                initial_attempts = eval(initial_attempts)
            except Exception as e:
                raise ValueError(f"Failed to load 'Initial Attempts': {e}")
        instance.initial_attempts = initial_attempts

        # Restore initial rewards.
        initial_rewards = row.get("Initial Rewards", [])
        if isinstance(initial_rewards, str):
            try:
                initial_rewards = eval(initial_rewards)
            except Exception as e:
                raise ValueError(f"Failed to load 'Initial Rewards': {e}")
        instance.initial_rewards = initial_rewards

        return instance

    def get_student_no_tutor_attempt(self):
        messages = [{"role": "user", "content": self.student_attempt}]
        return messages

    def start_conversation(self):

        # If we already started.
        if self.state != ConversationState.START:
            return

        if self.type == ConversationType.GUIDED:
            self.state = ConversationState.TEACHER_TURN
        else:
            self.state = ConversationState.STUDENT_TURN

    def _exceeded_max_tokens(self):
        return (
            sum(
                [
                    len(self.tokenizer.encode(message["content"]))
                    for message in self.conversation
                ]
            )
            > self.generation_cfg.max_tokens_in_conversation
        )

    def _hide_thinking(self, content: str):
        # We remove everything between <think> and </think>
        return re.sub(r"<think>.*?</think>", "", content, flags=re.S).replace(
            "<end_of_conversation>", ""
        )

    def _get_hidden_conversation(self):
        conversation = []
        for message in self.conversation:
            conversation.append(
                {
                    "role": message["role"],
                    "content": self._hide_thinking(message["content"]),
                }
            )
        return conversation

    def _get_conversation_from_teacher_perspective(self):
        conversation = []
        for message in self.conversation:
            if message["role"] == "teacher":
                conversation.append(
                    {"role": "assistant", "content": message["content"]}
                )
            else:
                conversation.append({"role": "user", "content": message["content"]})
        return conversation

    def _get_conversation_from_student_perspective(self):
        conversation = []
        for message in self.conversation:
            if message["role"] == "student":
                conversation.append(
                    {
                        "role": "assistant",
                        "content": self._hide_thinking(message["content"]),
                    }
                )
            else:
                conversation.append(
                    {"role": "user", "content": self._hide_thinking(message["content"])}
                )
        return conversation

    def get_conversation(self):
        if self.state == ConversationState.TEACHER_TURN:
            conversation = [
                {"role": "system", "content": self.system_prompt_teacher}
            ] + self._get_conversation_from_teacher_perspective()
            return conversation

        elif self.state == ConversationState.STUDENT_TURN:
            # If this is the first message in a guided conversation we request the student to start the conversation
            if self.type == ConversationType.ATTEMPTED and len(self.conversation) == 0:
                return [
                    {"role": "system", "content": self.system_prompt_student_attempt}
                ]
            conversation = [
                {"role": "system", "content": self.system_prompt_student}
            ] + self._get_conversation_from_student_perspective()
            return conversation

        elif self.state == ConversationState.JUDGE_TURN:
            remaining_judge_rules = list(
                set(self.generation_cfg.judges_rules_prompts_paths.keys())
                - set(self.judge_decisions.keys())
            )
            if len(remaining_judge_rules) == 0:
                raise ValueError(
                    "All judge rules have been evaluated, makes no sense we are at this state"
                )
            judge_rule = remaining_judge_rules[0]
            self.judge_evaluation_type = judge_rule
            return [
                {
                    "role": "user",
                    "content": Template(
                        open(
                            self.generation_cfg.judges_rules_prompts_paths[judge_rule]
                        ).read()
                    ).render(conversation=self._get_hidden_conversation()),
                }
            ]

        elif self.state == ConversationState.GENERATE_SOLUTION:
            conversation = [
                {"role": "system", "content": self.system_prompt_student}
            ] + self._get_conversation_from_student_perspective()
            conversation.append({"role": "user", "content": self.student_final_prompt})
            return conversation

    def add_message(self, content: str):
        if self.state == ConversationState.TEACHER_TURN:
            self.conversation.append({"role": "teacher", "content": content})
            self.state = ConversationState.STUDENT_TURN
            if (
                len(self.conversation) >= self.generation_cfg.max_turns
                or "<end_of_conversation>" in content
            ):
                self.state = ConversationState.JUDGE_TURN
        elif self.state == ConversationState.STUDENT_TURN:
            if self.type == ConversationType.ATTEMPTED and len(self.conversation) == 0:
                self.conversation.append(
                    {
                        "role": "student",
                        "content": self.initial_attempt_wrapper.render(attempt=content),
                    }
                )
                self.state = ConversationState.TEACHER_TURN
            else:
                self.conversation.append({"role": "student", "content": content})
                self.state = ConversationState.TEACHER_TURN
        if self._exceeded_max_tokens():
            self.state = ConversationState.JUDGE_TURN

        # If there is no judge messages in the config we skip to GENERATE_SOLUTION
        if (
            self.generation_cfg.number_judge_attempts == 0
            and self.state == ConversationState.JUDGE_TURN
        ):
            self.state = ConversationState.GENERATE_SOLUTION

    def add_judge_decisions(self, decisions: List[JudgeResponse]):
        if self.state != ConversationState.JUDGE_TURN:
            raise ValueError("We are not in the judge turn state")
        self.judge_decisions[self.judge_evaluation_type] = decisions
        for decision in decisions:
            if decision.decision == JudgeDecision.REJECT:
                self.failed_judges = True
                if not self.generation_cfg.ignore_rejected_judge:
                    self.state = ConversationState.END
                    return
        if len(self.judge_decisions) == len(
            self.generation_cfg.judges_rules_prompts_paths
        ):
            self.state = ConversationState.GENERATE_SOLUTION

    def add_solutions(self, solutions: List[str]):
        if self.state != ConversationState.GENERATE_SOLUTION:
            raise ValueError("We are not in the generate solution state")
        self.solutions = solutions
        self.state = ConversationState.REWARD_TURN

    def get_solutions_for_reward(self):
        chats = [
            self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "Please reason step by step, and put your final answer within \\boxed{}.",
                    },
                    {"role": "user", "content": self.problem},
                    {"role": "assistant", "content": solution},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            for solution in self.solutions
        ]
        return chats

    def get_initial_solutions_for_reward(self):
        chats = [
            self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "Please reason step by step, and put your final answer within \\boxed{}.",
                    },
                    {"role": "user", "content": self.problem},
                    {"role": "assistant", "content": solution},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            for solution in self.initial_attempts
        ]
        return chats

    def add_initial_rewards(self, rewards: List[float]):
        self.initial_rewards = rewards

    def add_rewards(self, rewards: List[float]):
        if self.state != ConversationState.REWARD_TURN:
            raise ValueError("We are not in the reward turn state")
        self.rewards = rewards
        self.state = ConversationState.END

    def add_initial_attempts(self, attempts: List[str]):
        self.initial_attempts = attempts

    def get_end_rm_reward(self):
        average_rm_reward = (
            sum(self.rewards) / len(self.rewards) if len(self.rewards) > 0 else None
        )
        return average_rm_reward

    def get_initial_rm_reward(self):
        average_rm_reward = (
            sum(self.initial_rewards) / len(self.initial_rewards)
            if len(self.initial_rewards) > 0
            else None
        )
        return average_rm_reward

    def get_thinking_reward(self):
        if len(self.rewards) == 0:
            return 0.0
        penalty_for_missing_closing_think = 0.0
        count_used_thinking, count_total = 0, 0
        for message in self.conversation:
            if message["role"] == "teacher":
                if message["content"].count("<think>") != message["content"].count(
                    "</think>"
                ):
                    penalty_for_missing_closing_think -= 0.5
                elif message["content"].count("<think>") > 0:
                    count_used_thinking += 1
                count_total += 1
        return (
            penalty_for_missing_closing_think
            + (count_used_thinking / count_total) * 0.5
        )

    def get_end_of_conversation_reward(self):
        if len(self.rewards) == 0:
            return 0.0

        return (
            0.1
            if any(
                "<end_of_conversation>" in message["content"]
                for message in self.conversation
            )
            else 0.0
        )

    def get_length_reward(self):
        texts = []
        for message in self.conversation:
            if message["role"] == "teacher":
                texts.append(message["content"])

        text_tokens_count = [len(self.tokenizer.encode(t)) for t in texts]

        return (
            -0.5
            if any(
                [
                    t >= self.generation_cfg.max_tokens_per_turn - 1
                    for t in text_tokens_count
                ]
            )
            else 0.0
        )

    def to_pd(self):
        return pd.DataFrame(
            [
                {
                    "State": self.state.name,
                    "Problem": self.problem,
                    "Answer": self.answer,
                    "Conversation": self.conversation,
                    "Type": self.type.name,
                    "Student Persona": self.student_persona,
                    "Student Name": self.student_name,
                    "Judge Decisions": {
                        key: [
                            {
                                "reasoning": decision.reasoning,
                                "decision": decision.decision.name,
                            }
                            for decision in decisions
                        ]
                        for key, decisions in self.judge_decisions.items()
                    },
                    "Solutions": self.solutions,
                    "Rewards": self.rewards,
                    "Initial Attempts": self.initial_attempts,
                    "Initial Rewards": self.initial_rewards,
                    "Conversation from student perspective": self._get_conversation_from_student_perspective(),
                }
            ]
        )

    def __str__(self):
        return self.to_pd().to_string()

    def __repr__(self):
        return self.to_pd().to_string()

    def get_trainable_representation(self):
        conversation = [
            {"role": "system", "content": self.system_prompt_teacher}
        ] + self._get_conversation_from_teacher_perspective()
        return conversation


class Classroom:
    def __init__(
        self,
        student_model_cfg: StudentModelConfig,
        teacher_model_cfg: TeacherModelConfig,
        judge_model_cfg: JudgeModelConfig,
        reward_model_cfg: RewardModelConfig,
        generation_cfg: GenerationConfig,
        model_save_path: str,
        log_file_path: str = None,
    ):

        self.student_model_cfg = student_model_cfg
        self.teacher_model_cfg = teacher_model_cfg
        self.judge_model_cfg = judge_model_cfg
        self.reward_model_cfg = reward_model_cfg
        self.generation_cfg = generation_cfg

        if self.teacher_model_cfg.use_openrouter:
            self.teacher_model = OpenRouterInference(
                self.teacher_model_cfg.model_name_or_path
            )
        elif self.teacher_model_cfg.use_gemini:
            self.teacher_model = GeminiInference(
                self.teacher_model_cfg.model_name_or_path
            )
        else:
            self.teacher_model = ParallelvLLMInference(
                model_path=teacher_model_cfg.model_name_or_path,
                gpus_per_instance=teacher_model_cfg.vllm.number_of_gpus_per_instance,
                gpu_memory_utilization=teacher_model_cfg.vllm.gpu_memory_utilization,
                max_model_len=teacher_model_cfg.vllm.max_length,
                max_num_seqs=teacher_model_cfg.vllm.max_num_seqs,
                model_save_path=model_save_path,
                use_lora=teacher_model_cfg.lora.enable,
                load_and_unload=teacher_model_cfg.vllm.load_and_unload,
                max_number_of_instances=teacher_model_cfg.vllm.max_number_of_instances,
                enable_sleep_mode=teacher_model_cfg.vllm.enable_sleep_mode,
                bits_and_bytes=teacher_model_cfg.vllm.bits_and_bytes,
                from_0=teacher_model_cfg.vllm.from_0,
                use_v0=teacher_model_cfg.vllm.use_v0,
                enforce_eager=teacher_model_cfg.vllm.enforce_eager,
                logging_enabled=log_file_path != None,
                log_file_path=log_file_path,
            )
        self.teacher_model.sleep()

        if self.student_model_cfg.use_openrouter:
            self.student_model = OpenRouterInference(
                self.student_model_cfg.model_name_or_path
            )
        elif self.student_model_cfg.use_gemini:
            self.student_model = GeminiInference(
                self.student_model_cfg.model_name_or_path
            )
        else:
            self.student_model = ParallelvLLMInference(
                model_path=student_model_cfg.model_name_or_path,
                gpus_per_instance=student_model_cfg.vllm.number_of_gpus_per_instance,
                gpu_memory_utilization=student_model_cfg.vllm.gpu_memory_utilization,
                max_model_len=student_model_cfg.vllm.max_length,
                max_num_seqs=student_model_cfg.vllm.max_num_seqs,
                model_save_path=None,
                load_and_unload=student_model_cfg.vllm.load_and_unload,
                max_number_of_instances=student_model_cfg.vllm.max_number_of_instances,
                bits_and_bytes=student_model_cfg.vllm.bits_and_bytes,
                enable_sleep_mode=student_model_cfg.vllm.enable_sleep_mode,
                from_0=student_model_cfg.vllm.from_0,
                use_v0=student_model_cfg.vllm.use_v0,
                enforce_eager=student_model_cfg.vllm.enforce_eager,
                logging_enabled=log_file_path != None,
                log_file_path=log_file_path,
            )
        self.student_model.sleep()

        if self.judge_model_cfg.use_openrouter:
            self.judge_model = OpenRouterInference(
                self.judge_model_cfg.model_name_or_path
            )
        elif self.judge_model_cfg.use_gemini:
            self.judge_model = GeminiInference(self.judge_model_cfg.model_name_or_path)
        else:
            self.judge_model = ParallelvLLMInference(
                model_path=judge_model_cfg.model_name_or_path,
                gpus_per_instance=judge_model_cfg.vllm.number_of_gpus_per_instance,
                gpu_memory_utilization=judge_model_cfg.vllm.gpu_memory_utilization,
                max_model_len=judge_model_cfg.vllm.max_length,
                max_num_seqs=judge_model_cfg.vllm.max_num_seqs,
                model_save_path=None,
                load_and_unload=judge_model_cfg.vllm.load_and_unload,
                max_number_of_instances=judge_model_cfg.vllm.max_number_of_instances,
                bits_and_bytes=judge_model_cfg.vllm.bits_and_bytes,
                enable_sleep_mode=judge_model_cfg.vllm.enable_sleep_mode,
                enforce_eager=judge_model_cfg.vllm.enforce_eager,
                from_0=judge_model_cfg.vllm.from_0,
                use_v0=judge_model_cfg.vllm.use_v0,
                logging_enabled=log_file_path != None,
                log_file_path=log_file_path,
            )
        self.judge_model.sleep()

        if self.reward_model_cfg.model_name_or_path not in ["None", "Answer"]:
            self.reward_model = ParallelvLLMInference(
                model_path=reward_model_cfg.model_name_or_path,
                gpus_per_instance=reward_model_cfg.vllm.number_of_gpus_per_instance,
                gpu_memory_utilization=reward_model_cfg.vllm.gpu_memory_utilization,
                max_model_len=reward_model_cfg.vllm.max_length,
                max_num_seqs=reward_model_cfg.vllm.max_num_seqs,
                model_save_path=None,
                load_and_unload=reward_model_cfg.vllm.load_and_unload,
                max_number_of_instances=reward_model_cfg.vllm.max_number_of_instances,
                bits_and_bytes=reward_model_cfg.vllm.bits_and_bytes,
                inference_task=InferenceTask.REWARD,
                enable_sleep_mode=reward_model_cfg.vllm.enable_sleep_mode,
                enforce_eager=reward_model_cfg.vllm.enforce_eager,
                from_0=reward_model_cfg.vllm.from_0,
                use_v0=reward_model_cfg.vllm.use_v0,
                logging_enabled=log_file_path != None,
                log_file_path=log_file_path,
            )
            self.reward_model.sleep()

        self.sampling_params_student = SamplingParams(
            temperature=student_model_cfg.vllm.temperature,
            top_k=student_model_cfg.vllm.top_k,
            top_p=student_model_cfg.vllm.top_p,
            max_tokens=generation_cfg.max_tokens_per_turn,
        )

        # Sadly this is too slow.
        # guided_decoding_params = GuidedDecodingParams(
        #   json=JudgeResponse.model_json_schema(),
        #   backend='lm-format-enforcer'
        # )

        self.sampling_params_judge = SamplingParams(
            temperature=judge_model_cfg.vllm.temperature,
            top_k=judge_model_cfg.vllm.top_k,
            top_p=judge_model_cfg.vllm.top_p,
            max_tokens=generation_cfg.max_tokens_per_judge_attempt,
            # guided_decoding=guided_decoding_params
        )

        self.sampling_params_student_solution = SamplingParams(
            n=generation_cfg.number_student_attempts,
            temperature=student_model_cfg.vllm.temperature,
            top_k=student_model_cfg.vllm.top_k,
            top_p=student_model_cfg.vllm.top_p,
            max_tokens=generation_cfg.max_tokens_per_student_attempt,
        )

        teacher_tokenizer = AutoTokenizer.from_pretrained(
            generation_cfg.tokenizer_to_use
        )
        thinking_tokens = teacher_tokenizer.encode("<think>", add_special_tokens=False)

        def force_thinking_processor(token_ids, logits):
            if len(token_ids) < len(thinking_tokens):
                logits[thinking_tokens[len(token_ids)]] = 10000
            return logits

        self.sampling_params_teacher = SamplingParams(
            temperature=teacher_model_cfg.vllm.temperature,
            top_k=teacher_model_cfg.vllm.top_k,
            top_p=teacher_model_cfg.vllm.top_p,
            max_tokens=generation_cfg.max_tokens_per_turn,
            logits_processors=(
                [force_thinking_processor] if generation_cfg.force_thinking else []
            ),
        )

        self.conversation_sets = []

    def _compute_rewards_from_prompts(
        self, prompts: List[str], answers: List[str]
    ) -> List[float]:
        if self.reward_model_cfg.model_name_or_path not in ["None", "Answer"]:
            responses = self.reward_model.run_batch(prompts, None)
            rewards = [
                output.outputs.data[-1].item() if hasattr(output, "outputs") else 1.0
                for output in responses
            ]
        elif self.reward_model_cfg.model_name_or_path == "Answer":
            extracted_answers = [extract_answer(prompt) for prompt in prompts]
            rewards = [
                1.0 if check_equal(answer, extracted_answer) else 0.0
                for answer, extracted_answer in zip(answers, extracted_answers)
            ]
        elif self.reward_model_cfg.model_name_or_path == "None":
            rewards = [0.0 for _ in prompts]
        return rewards

    def generate_next_teacher_utterances(
        self, conversations: List[Conversation], meta: dict = None
    ) -> List[str]:
        """
        Given a list of Conversation objects in TEACHER_TURN, generate the next teacher utterance
        for each and add it to the conversation.
        """
        if meta is None:
            meta = {}
        prompts = [conv.get_conversation() for conv in conversations]
        responses = self.teacher_model.run_batch(
            prompts, self.sampling_params_teacher, meta
        )
        teacher_utterances = [response.outputs[0].text for response in responses]
        for conv, utterance in zip(conversations, teacher_utterances):
            conv.add_message(utterance)
        return teacher_utterances

    def generate_next_student_utterances(
        self, conversations: List[Conversation]
    ) -> List[str]:
        """
        Given a list of Conversation objects in STUDENT_TURN, generate the next student utterance
        for each and add it to the conversation.
        """
        prompts = [conv.get_conversation() for conv in conversations]
        responses = self.student_model.run_batch(prompts, self.sampling_params_student)
        student_utterances = [response.outputs[0].text for response in responses]
        for conv, utterance in zip(conversations, student_utterances):
            conv.add_message(utterance)
        return student_utterances

    def sample_conversations(
        self,
        problems: List[str],
        answers: List[str],
        forced_type: ConversationType = None,
        meta: dict = {},
        compute_initial_attempt: bool = False,
    ) -> List[Conversation]:
        # If we force a certain type of conversation we set it here.
        if forced_type is None:
            if self.generation_cfg.forced_conversation_type == "guided":
                forced_type = ConversationType.GUIDED
            elif self.generation_cfg.forced_conversation_type == "attempt":
                forced_type = ConversationType.ATTEMPTED
            else:
                forced_type = None

        conversations = []
        for problem, answer in tqdm(
            zip(problems, answers),
            total=len(problems),
            desc="Initializing conversations",
        ):
            conversations.append(
                Conversation(problem, answer, self.generation_cfg, forced_type)
            )

        # Start the conversations.
        for conversation in conversations:
            conversation.start_conversation()

        # Only for eval we compute how good the model was initially.
        if compute_initial_attempt:
            logger.info(("=" * 10) + "Computing initial attempts" + ("=" * 10))
            messages = [
                conversation.get_student_no_tutor_attempt()
                for conversation in conversations
            ]
            responses = self.student_model.run_batch(
                messages, self.sampling_params_student_solution
            )
            for conversation, response in zip(conversations, responses):
                conversation.add_initial_attempts(
                    [output.text for output in response.outputs]
                )

            prompts_for_rewards = [
                conversation.get_initial_solutions_for_reward()
                for conversation in conversations
            ]
            lengths = [len(prompts) for prompts in prompts_for_rewards]

            all_prompts = [
                prompt for prompts in prompts_for_rewards for prompt in prompts
            ]
            all_answers = []
            for conversation in conversations:
                all_answers.extend(
                    [conversation.answer] * len(conversation.initial_attempts)
                )

            rewards = self._compute_rewards_from_prompts(all_prompts, all_answers)

            for conv in conversations:
                curr_len = lengths.pop(0)
                conv_rewards = rewards[:curr_len]
                conv.add_initial_rewards(conv_rewards)
                rewards = rewards[curr_len:]

        round_counter = 1

        # We now alternate between teacher and student turns until the conversation is not in the conversation student/teacher turn state
        while any(
            [
                conversation.state
                in [ConversationState.TEACHER_TURN, ConversationState.STUDENT_TURN]
                for conversation in conversations
            ]
        ):
            for state_to_process in [
                ConversationState.TEACHER_TURN,
                ConversationState.STUDENT_TURN,
            ]:
                logger.info(
                    ("=" * 10)
                    + f"Executing turn {round_counter}: {'Teacher' if state_to_process == ConversationState.TEACHER_TURN else 'Student'}"
                    + ("=" * 10)
                )

                start_time = time.time()
                # We get all conversations that are in the state_to_process state
                conversations_to_process = [
                    conversation
                    for conversation in conversations
                    if conversation.state == state_to_process
                ]
                if len(conversations_to_process) == 0:
                    continue

                # We get the messages from the conversations (not used further here since helper methods call get_conversation internally)
                messages = [
                    conversation.get_conversation()
                    for conversation in conversations_to_process
                ]

                # We get the responses from the model using our helper functions.
                if state_to_process == ConversationState.TEACHER_TURN:
                    self.generate_next_teacher_utterances(
                        conversations_to_process, meta
                    )
                else:
                    self.generate_next_student_utterances(conversations_to_process)

                # Next round counter.
                round_counter += 1

                logger.info(f"Took {time.time() - start_time} seconds.")

        # We can put both to sleep.
        self.teacher_model.sleep()
        self.student_model.sleep()

        # We now evaluate the judge rules
        logger.info(("=" * 10) + "Running judges" + ("=" * 10))
        start_time = time.time()

        num_attempts_required = self.generation_cfg.number_judge_attempts
        max_rounds = 5
        while any(
            [
                conversation.state == ConversationState.JUDGE_TURN
                for conversation in conversations
            ]
        ):
            logger.info(("=" * 15) + "Judges round" + ("=" * 15))
            # Select conversations in judge turn.
            conversations_to_process = [
                conv
                for conv in conversations
                if conv.state == ConversationState.JUDGE_TURN
            ]

            # Dictionary to collect valid JudgeResponse objects per conversation.
            valid_responses = {conv: [] for conv in conversations_to_process}

            for _judge_round in range(max_rounds):
                logger.info(
                    ("=" * 10) + f"Judges inner round {_judge_round}" + ("=" * 10)
                )
                pending = []  # List of tuples: (conversation, message)
                # For each conversation, schedule as many generations as are missing.
                for conv in conversations_to_process:
                    missing = num_attempts_required - len(valid_responses[conv])
                    if missing > 0:
                        for _ in range(missing):
                            pending.append((conv, conv.get_conversation()))

                if not pending:
                    break  # All conversations have enough valid responses.
                logger.info("Number of pending judge responses:" + str(len(pending)))

                # Run a batch for all pending messages.
                pending_messages = [msg for _, msg in pending]
                responses = self.judge_model.run_batch(
                    pending_messages, self.sampling_params_judge
                )

                # Map each response back to its conversation.
                for (conv, _), response in zip(pending, responses):
                    for output in response.outputs:
                        try:
                            # We only take stuff that is between { and }
                            out_text = output.text[
                                output.text.find("{") : output.text.rfind("}") + 1
                            ].replace("\\", "")
                            decision = JudgeResponse(
                                **json.loads(out_text, strict=False)
                            )
                            valid_responses[conv].append(decision)
                        except Exception as e:
                            continue

            # For any conversation still missing valid responses, add default decisions.
            for conv in conversations_to_process:
                while len(valid_responses[conv]) < num_attempts_required:
                    logger.warning(
                        "Judge decision ran out of attempts, adding default decision"
                    )
                    valid_responses[conv].append(
                        JudgeResponse(reasoning="max turns exceeded", decision="OK")
                    )
                conv.add_judge_decisions(valid_responses[conv])

        self.judge_model.sleep()
        logger.info(f"Took {time.time() - start_time} seconds.")

        # We now generate the solutions
        logger.info(("=" * 10) + "Sampling solutions" + ("=" * 10))
        start_time = time.time()
        conversations_to_process = [
            conversation
            for conversation in conversations
            if conversation.state == ConversationState.GENERATE_SOLUTION
        ]
        logger.info(
            f"Generating solutions for {len(conversations_to_process)} conversations"
        )

        if len(conversations_to_process) > 0:
            messages = [
                conversation.get_conversation()
                for conversation in conversations_to_process
            ]
            responses = self.student_model.run_batch(
                messages, self.sampling_params_student_solution
            )
            for conversation, response in zip(conversations_to_process, responses):
                conversation.add_solutions([output.text for output in response.outputs])

        self.student_model.sleep()
        logger.info(f"Took {time.time() - start_time} seconds.")

        # Compute rewards.
        logger.info(("=" * 10) + "Computing Rewards" + ("=" * 10))
        start_time = time.time()
        reward_convs = [
            conv
            for conv in conversations
            if conv.state == ConversationState.REWARD_TURN
        ]
        if reward_convs:
            all_prompts = []
            all_answers = []
            lengths = []
            for conv in reward_convs:
                prompts = conv.get_solutions_for_reward()
                lengths.append(len(prompts))
                all_prompts.extend(prompts)
                all_answers.extend([conv.answer] * len(prompts))
            rewards = self._compute_rewards_from_prompts(all_prompts, all_answers)
            for conv in reward_convs:
                curr_len = lengths.pop(0)
                conv_rewards = rewards[:curr_len]
                conv.add_rewards(conv_rewards)
                rewards = rewards[curr_len:]

        logger.info(f"Took {time.time() - start_time} seconds.")
        # Free memory
        gc.collect()
        torch.cuda.empty_cache()

        self.conversation_sets.append(conversations)
        return conversations

    def run_judges(self, conversations: List[Conversation]):
        """
        Given a list of Conversation objects in JUDGE_TURN, generate the next judge utterance
        for each and add it to the conversation.
        """
        # We now evaluate the judge rules
        logger.info(("=" * 10) + "Running judges" + ("=" * 10))
        start_time = time.time()

        num_attempts_required = self.generation_cfg.number_judge_attempts
        max_rounds = 5
        while any(
            [
                conversation.state == ConversationState.JUDGE_TURN
                for conversation in conversations
            ]
        ):
            logger.info(("=" * 15) + "Judges round" + ("=" * 15))
            # Select conversations in judge turn.
            conversations_to_process = [
                conv
                for conv in conversations
                if conv.state == ConversationState.JUDGE_TURN
            ]

            # Dictionary to collect valid JudgeResponse objects per conversation.
            valid_responses = {conv: [] for conv in conversations_to_process}

            for _judge_round in range(max_rounds):
                logger.info(
                    ("=" * 10) + f"Judges inner round {_judge_round}" + ("=" * 10)
                )
                pending = []  # List of tuples: (conversation, message)
                # For each conversation, schedule as many generations as are missing.
                for conv in conversations_to_process:
                    missing = num_attempts_required - len(valid_responses[conv])
                    if missing > 0:
                        for _ in range(missing):
                            pending.append((conv, conv.get_conversation()))

                if not pending:
                    break  # All conversations have enough valid responses.
                logger.info("Number of pending judge responses:" + str(len(pending)))

                # Run a batch for all pending messages.
                pending_messages = [msg for conv, msg in pending]
                responses = self.judge_model.run_batch(
                    pending_messages, self.sampling_params_judge
                )

                # Map each response back to its conversation.
                for (conv, _), response in zip(pending, responses):
                    for output in response.outputs:
                        try:
                            # We only take stuff that is between { and }
                            out_text = output.text[
                                output.text.find("{") : output.text.rfind("}") + 1
                            ].replace("\\", "")
                            decision = JudgeResponse(
                                **json.loads(out_text, strict=False)
                            )
                            valid_responses[conv].append(decision)
                        except Exception as e:
                            continue

            for conv in conversations_to_process:
                while len(valid_responses[conv]) < num_attempts_required:
                    logger.warning(
                        "Judge decision ran out of attempts, adding default decision"
                    )
                    valid_responses[conv].append(
                        JudgeResponse(reasoning="max turns exceeded", decision="OK")
                    )
                conv.add_judge_decisions(valid_responses[conv])

        self.judge_model.sleep()
        logger.info(f"Took {time.time() - start_time} seconds.")

    def to_pd_latest(self):
        return pd.concat(
            [conversation.to_pd() for conversation in self.conversation_sets[-1]]
        )

    def get_conversation_by_text(self, text: str):
        conversations = self.conversation_sets[-1]
        max_messages_overlap = 0
        conversation = None
        for conv in conversations:
            trainable_representation = conv.get_trainable_representation()
            messages_overlap = sum(
                [
                    len(message["content"])
                    for message in trainable_representation
                    if message["content"] in text
                ]
            )
            if messages_overlap > max_messages_overlap:
                max_messages_overlap = messages_overlap
                conversation = conv

        if max_messages_overlap == 0:
            raise ValueError("No conversation found")
        return conversation

    def get_end_rm_reward(self, conversation: Conversation):
        import os

        reward = conversation.get_end_rm_reward()
        if reward == None:
            conversations = [
                conv
                for conv in self.conversation_sets[-1]
                if conv.problem == conversation.problem
            ]
            rewards = [conv.get_end_rm_reward() for conv in conversations]
            rewards = [reward for reward in rewards if reward is not None]
            if os.getenv("USE_OLD_MIN", "0") == "1":
                minimum_reward = (
                    min(rewards) - self.generation_cfg.extra_penalty_for_rejected_judges
                    if len(rewards) > 0
                    else 0.0
                )
            else:
                minimum_reward = -self.generation_cfg.extra_penalty_for_rejected_judges

            return minimum_reward

        if conversation.failed_judges:
            reward -= self.generation_cfg.extra_penalty_for_rejected_judges
        return reward

    def get_thinking_reward(self, conversation: Conversation):
        return conversation.get_thinking_reward()

    def get_end_of_conversation_reward(self, conversation: Conversation):
        return conversation.get_end_of_conversation_reward()

    def get_length_reward(self, conversation: Conversation):
        return conversation.get_length_reward()
