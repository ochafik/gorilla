import subprocess
import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor

import requests
from bfcl.constant import RESULT_PATH
from bfcl.model_handler.base_handler import BaseHandler
from bfcl.model_handler.model_style import ModelStyle
from bfcl.model_handler.local_inference.constant import VLLM_PORT
from bfcl.model_handler.local_inference.server_handler import LocalServerHandler
from bfcl.model_handler.utils import (
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    func_doc_language_specific_pre_processing,
    system_prompt_pre_processing_chat_model,
)
from openai import OpenAI
from overrides import EnforceOverrides, final, override
from tqdm import tqdm


class OSSHandler(BaseHandler, EnforceOverrides):
    def __init__(self, model_name, temperature, dtype="bfloat16") -> None:
        super().__init__(model_name, temperature)
        self.model_name_huggingface = model_name
        self.model_style = ModelStyle.OSSMODEL
        self.dtype = dtype
        # Read from env vars with fallbacks
        self.vllm_host = os.getenv("VLLM_ENDPOINT", "localhost")
        self.vllm_port = os.getenv("VLLM_PORT", VLLM_PORT)

        self.base_url = f"http://{self.vllm_host}:{self.vllm_port}/v1"
        self.client = OpenAI(base_url=self.base_url, api_key="EMPTY")

        self.server_handler = None

    @override
    def decode_ast(self, result, language="Python"):
        return default_decode_ast_prompting(result, language)

    @override
    def decode_execute(self, result):
        return default_decode_execute_prompting(result)

    @override
    def before_batch(
        self,
        *,
        num_gpus: int,
        gpu_memory_utilization: float,
        backend: str,
        skip_server_setup: bool,
    ):
        """
        Batch inference for OSS models.
        """
        from transformers import AutoConfig, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_huggingface, trust_remote_code=True
        )

        config = AutoConfig.from_pretrained(
            self.model_name_huggingface, trust_remote_code=True
        )
        if hasattr(config, "max_position_embeddings"):
            self.max_context_length = config.max_position_embeddings
        elif self.tokenizer.model_max_length is not None:
            self.max_context_length = self.tokenizer.model_max_length
        else:
            if not hasattr(self, "max_context_length"):
                raise ValueError(
                    "Model does not have a max_position_embeddings attribute or tokenizer.model_max_length attribute. Please set the max_context_length attribute in the corresponding model handler."
                )
        print(f"Max context length: {self.max_context_length}")

        command = None
        if not skip_server_setup:
            if backend == "vllm":
                command = [
                    "vllm",
                    "serve",
                    str(self.model_name_huggingface),
                    "--port",
                    str(self.vllm_port),
                    "--dtype",
                    str(self.dtype),
                    "--tensor-parallel-size",
                    str(num_gpus),
                    "--gpu-memory-utilization",
                    str(gpu_memory_utilization),
                    "--trust-remote-code",
                ]
            elif backend == "sglang":
                command = [
                    "python",
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    str(self.model_name_huggingface),
                    "--port",
                    str(self.vllm_port),
                    "--dtype",
                    str(self.dtype),
                    "--tp",
                    str(num_gpus),
                    "--mem-fraction-static",
                    str(gpu_memory_utilization),
                    "--trust-remote-code",
                ]
            else:
                raise ValueError(f"Backend {backend} is not supported.")

            self.server_handler = LocalServerHandler(
                command=command,
                host=self.vllm_host,
                port=self.vllm_port,
                ready_path="/v1/models",
            )

            self.server_handler.wait_for_server_ready()

    @override
    def after_batch(self):
        self.monitor.terminate();
        self.monitor = None

    @override
    def inference(self, test_case: dict, include_input_log: bool, exclude_state_log: bool):
        assert type(test_case["function"]) is list

        if "multi_turn" in test_case["id"]:
            return self.inference_multi_turn_prompting(
                test_case, include_input_log, exclude_state_log
            )
        else:
            return self.inference_single_turn_prompting(
                test_case, include_input_log
            )

    #### Prompting methods ####

    def _format_prompt(self, messages, function):
        """
        Manually apply the chat template to construct the formatted prompt.
        This way, we can have full control over the final formatted prompt and is generally recommended for advanced use cases.
        """
        raise NotImplementedError(
            "OSS Models should implement their own prompt formatting."
        )

    @override
    def _query_prompting(self, inference_data: dict):
        # We use the OpenAI Completions API
        function: list[dict] = inference_data["function"]
        message: list[dict] = inference_data["message"]

        formatted_prompt: str = self._format_prompt(message, function)
        inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}

        # Tokenize the formatted prompt to get token count
        input_token_count = len(self.tokenizer.tokenize(formatted_prompt))

        # Determine the number of tokens to request. Cap it at 4096 if the model has a larger limit.
        if self.max_context_length < input_token_count + 2:
            # If the prompt is already at the max length, just request 1000 token, we will get an error anyway
            leftover_tokens_count = 1000
        else:
            leftover_tokens_count = min(
                4096,
                self.max_context_length - input_token_count - 2,
            )

        extra_body = {}
        if hasattr(self, "stop_token_ids"):
            extra_body["stop_token_ids"] = self.stop_token_ids
        if hasattr(self, "skip_special_tokens"):
            extra_body["skip_special_tokens"] = self.skip_special_tokens

        start_time = time.time()
        if len(extra_body) > 0:
            api_response = self.client.completions.create(
                model=self.model_name_huggingface,
                temperature=self.temperature,
                prompt=formatted_prompt,
                max_tokens=leftover_tokens_count,
                extra_body=extra_body,
            )
        else:
            api_response = self.client.completions.create(
                model=self.model_name_huggingface,
                temperature=self.temperature,
                prompt=formatted_prompt,
                max_tokens=leftover_tokens_count,
            )
        end_time = time.time()

        return api_response, end_time - start_time

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )

        return {"message": [], "function": functions}

    @override
    def _parse_query_response_prompting(self, api_response: any) -> dict:
        return {
            "model_responses": api_response.choices[0].text,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    @override
    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            {"role": "assistant", "content": model_response_data["model_responses"]}
        )
        return inference_data

    @override
    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        for execution_result, decoded_model_response in zip(
            execution_results, model_response_data["model_responses_decoded"]
        ):
            inference_data["message"].append(
                {
                    "role": "tool",
                    "name": decoded_model_response,
                    "content": execution_result,
                }
            )

        return inference_data
