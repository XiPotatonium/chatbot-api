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
from PIL import Image


class ChatGLMModel(ChatModel):
    @classmethod
    def request_validator(
        cls, request: Mapping[str, Any], files: Dict[str, Tuple[bytes, str]]
    ):
        if not request.get("stream", False):
            # Currently ChatGLM only allow streaming
            raise ValueError(f"only stream is implemented for {cls.__name__}")

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
        model.to(device)
        model.eval()
        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     model = torch.compile(model)
        return cls(tokenizer, model, device)

    def __init__(self, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, device: torch.device) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def delete(self):
        del self.model
        empty_cache(self.device)

    # TODO: more generation configs
    def stream_generate(
        self,
        history: Iterator[Dict[str, Any]],
        max_tokens: int = 2048,
        top_p: float = 0.7,
        temperature: float = 0.95,
        **kwargs,
    ):
        inference_history = []

        for info in history:

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
