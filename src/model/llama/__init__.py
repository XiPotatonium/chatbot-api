import logging
import sys
import io
import torch

from ...device import empty_cache
from ...common import ROLE_BOT, ROLE_USER, ROLE_SYSTEM
from ..model import Model, ChatModel
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, GenerationConfig, PreTrainedTokenizer
)
from peft import PeftModel
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Tuple, Union


class LlamaLoraModel(ChatModel):
    @classmethod
    def request_validator(
        cls, request: Mapping[str, Any], files: Dict[str, Tuple[bytes, str]]
    ):
        if request.get("stream", False):
            raise ValueError(f"Stream is not supported for {cls.__name__}")

        for req in request["messages"]:
            if "media" in req and len(req["media"]) != 0:
                raise ValueError(
                    f"multi-media input is not supported for {cls.__name__}"
                )
            if req["role"] == ROLE_BOT:
                raise ValueError(f"bot role is not supported for {cls.__name__}")

        if len(request["messages"]) == 1:
            pass
        elif len(request["messages"] == 2):
            if request["messages"][0]["role"] == ROLE_USER or request["messages"][1]["role"] == ROLE_SYSTEM:
                raise ValueError(f"Available messages form for {cls.__name__} is [system, user] or [user] or [system]")
        else:
            raise ValueError(f"Available messages form for {cls.__name__} is [system, user] or [user] or [system]")

    @classmethod
    def load(cls, cfg: MutableMapping[str, Any], device: torch.device) -> Model:
        path = cfg["model_path"]
        lora_path = cfg["lora_path"]
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            torch_dtype=torch.float16,
        )
        model.half()
        model.to(device)
        # logging.info(device)
        model.eval()
        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     model = torch.compile(model)
        return cls(tokenizer, model, device)

    def __init__(self, tokenizer: PreTrainedTokenizer, model: PeftModel, device: torch.device) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def delete(self):
        del self.model
        empty_cache(self.device)

    def generate(
            self,
            history: Iterator[Dict[str, Any]],
            max_tokens: int = 128,
            top_p: float = 0.75,
            top_k: int = 40,
            temperature: float = 0.1,
            beams: int = 4,
            **kwargs
    ):
        """default generation config comes from https://huggingface.co/spaces/tloen/alpaca-lora

        Args:
            history (Iterator[Dict[str, Any]]): _description_
            max_tokens (int, optional): _description_. Defaults to 128.
            top_p (float, optional): _description_. Defaults to 0.75.
            top_k (int, optional): _description_. Defaults to 40.
            temperature (float, optional): _description_. Defaults to 0.1.
            beams (int, optional): _description_. Defaults to 4.
        """
        def generate_prompt(instruction, input=None):
            if input:
                return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:"""
            else:
                return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""

        instruction = None
        query = None
        for info in history:
            if ROLE_SYSTEM in info:
                instruction = info[ROLE_SYSTEM]
            else:
                query = info[ROLE_USER]["content"]

        prompt = generate_prompt(instruction, query)
        # print(f"usr: {prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=beams,
            **kwargs,
        )
        generation_output = self.model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_tokens,
        )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)
        # print("bot: {}".format(output.split("### Response:")[1].strip()))
        output = output.split("### Response:")[1].strip()

        empty_cache(self.device)
        return [{"index": 0, "message": {"role": ROLE_BOT, "content": output}}]
