import sys
import io
import torch

from ...device import empty_cache
from ...common import ROLE_BOT, ROLE_USER, ROLE_SYSTEM
from ..model import Model, ChatModel
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    BlipImageProcessor,
    PreTrainedTokenizer,
)
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Tuple, Union


def iter_messages(
    messages: List[Dict[str, Any]], files: Dict[str, Tuple[bytes, str]]
) -> Iterator[Dict[str, Any]]:
    """Used in chatglm-based models to convert messages into chatglm inference history

    Args:
        messages (List[Dict[str, Any]]): _description_
        files (Dict[str, Tuple[bytes, str]]): _description_

    Yields:
        Iterator[Dict[str, Any]]: _description_
    """

    def _new_message():
        return {
            ROLE_USER: {"content": "", "media": []},
            ROLE_BOT: {"content": "", "media": []},
        }

    if len(messages) == 0:
        raise ValueError("messages is empty")
    message = _new_message()
    for raw_msg in messages:
        if raw_msg["role"] == ROLE_USER:
            if (
                len(message[ROLE_USER]["content"]) != 0
                or len(message[ROLE_USER]["media"]) != 0
            ):
                yield message
                message = _new_message()

            if "content" in raw_msg:
                message[ROLE_USER]["content"] = raw_msg["content"]
            if "media" in raw_msg:
                message[ROLE_USER]["media"] = [
                    files[fname] for fname in raw_msg["media"]
                ]
        elif raw_msg["role"] == ROLE_BOT:
            if (
                len(message[ROLE_BOT]["content"]) != 0
                or len(message[ROLE_BOT]["media"]) != 0
            ):
                yield message
                message = _new_message()

            if "content" in raw_msg:
                message[ROLE_BOT]["content"] = raw_msg["content"]
            if "media" in raw_msg:
                message[ROLE_BOT]["media"] = [
                    files[fname] for fname in raw_msg["media"]
                ]
        elif raw_msg["role"] == ROLE_SYSTEM:
            # flush last message and this system message
            if (
                len(message[ROLE_BOT]["content"]) != 0
                or len(message[ROLE_BOT]["media"]) != 0
                or len(message[ROLE_USER]["content"]) != 0
                or len(message[ROLE_USER]["media"]) != 0
            ):
                yield message
                message = _new_message()
            yield {ROLE_SYSTEM: raw_msg["content"]}
        else:
            raise ValueError(f"Unknown role: {raw_msg['role']}")

    if (
        len(message[ROLE_BOT]["content"]) != 0
        or len(message[ROLE_BOT]["media"]) != 0
        or len(message[ROLE_USER]["content"]) != 0
        or len(message[ROLE_USER]["media"]) != 0
    ):
        yield message


class ChatGLMModel(ChatModel):
    @classmethod
    def request_validator(
        cls, request: Mapping[str, Any], files: Dict[str, Tuple[bytes, str]]
    ):
        for req in request["messages"]:
            if "media" in req and len(req["media"]) != 0:
                raise ValueError(
                    f"multi-media input is not supported for {cls.__name__}"
                )
            if req["role"] == ROLE_SYSTEM:
                raise ValueError(f"system role is not supported for {cls.__name__}")

    @classmethod
    def load(cls, cfg: MutableMapping[str, Any], device: torch.device) -> Model:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_path"], trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            cfg["model_path"],
            trust_remote_code=True,
            # device_map="auto"
        )

        if device.type == "cpu":
            model = model.float()
        else:
            prec = cfg["prec"]
            if prec == "fp16":
                model = model.half()
            elif prec == "int4":
                model = model.half().quantize(4)
            elif prec == "int8":
                model = model.half().quantize(8)

        if "lora_path" in cfg:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, cfg["lora_path"])

        model.to(device)
        model.eval()
        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     model = torch.compile(model)
        return cls(tokenizer, model, device)

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        device: torch.device,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def delete(self):
        del self.model
        empty_cache(self.device)

    def generate(
        self,
        messages: Iterator[Dict[str, Any]],
        files: Dict[str, Tuple[bytes, str]],
        max_tokens: int = 4096,
        top_p: float = 0.7,
        temperature: float = 0.95,
        **kwargs,
    ):
        inference_history = []

        for info in iter_messages(messages, files):

            def convert(info: Dict[str, Any]):
                content = info["content"]
                return content

            inference_history.append(
                (convert(info[ROLE_USER]), convert(info[ROLE_BOT]))
            )
        query = inference_history.pop()
        query = query[0]

        output, _ = self.model.chat(
            self.tokenizer,
            query=query,
            history=inference_history,
            max_length=max_tokens,
            top_p=top_p,
            temperature=temperature,
        )
        empty_cache(self.device)
        return [{"index": 0, "message": {"role": ROLE_BOT, "content": output}}]

    # TODO: more generation configs
    def stream_generate(
        self,
        messages: Iterator[Dict[str, Any]],
        files: Dict[str, Tuple[bytes, str]],
        max_tokens: int = 2048,
        top_p: float = 0.7,
        temperature: float = 0.95,
        **kwargs,
    ):
        inference_history = []

        for info in iter_messages(messages, files):

            def convert(info: Dict[str, Any]):
                content = info["content"]
                return content

            inference_history.append(
                (convert(info[ROLE_USER]), convert(info[ROLE_BOT]))
            )
        query = inference_history.pop()
        query = query[0]

        last_output = ""
        yield [{"index": 0, "delta": {"role": ROLE_BOT}}]
        for i, (output, _) in enumerate(
            self.model.stream_chat(
                self.tokenizer,
                query=query,
                history=inference_history,
                max_length=max_tokens,
                top_p=top_p,
                temperature=temperature,
            )
        ):
            yield [{"index": 0, "delta": {"content": output[len(last_output) :]}}]
            last_output = output
        yield [{"index": 0, "delta": {}}]
        empty_cache(self.device)
