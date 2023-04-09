import sys
import io
import torch

from ...device import empty_cache
from ...common import ROLE_BOT, ROLE_USER, ROLE_SYSTEM
from ..model import Model, ChatModel
from transformers import (
    AutoModel,
    AutoTokenizer,
    BlipImageProcessor,
    PreTrainedTokenizer,
)
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Tuple, Union
from .modeling_blip2chatglm import Blip2ChatGLM, Blip2ForChatGLM, Blip2ChatGLMConfig
from .modeling_chatglm import ChatGLMForConditionalGeneration
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from PIL import Image


class Blip2ChatGLMModel(ChatModel):
    @classmethod
    def request_validator(
        cls, request: Mapping[str, Any], files: Dict[str, Tuple[bytes, str]]
    ):
        if not request.get("stream", False):
            # Currently Blip2ChatGLM only allow streaming
            raise ValueError(f"only stream is implemented for {cls.__name__}")
        for k, (_, mime) in files.items():
            if mime not in ["image/png", "image/jpeg"]:
                raise ValueError(f"unsupported media type {mime} for {cls.__name__}")
        for req in request["messages"]:
            if "media" in req and len(req["media"]) > 1:
                raise ValueError(
                    f"only one multi-media input is supported for {cls.__name__}"
                )
            if req["role"] == ROLE_SYSTEM:
                raise ValueError(f"system role is not supported for {cls.__name__}")

    @classmethod
    def load(cls, cfg: MutableMapping[str, Any], device: torch.device) -> Model:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["lm_path"], trust_remote_code=True
        )
        lm = ChatGLMForConditionalGeneration.from_pretrained(
            cfg["lm_path"],  # device_map="auto"
        )

        if device.type == "cpu":
            lm = lm.float()
        else:
            prec = cfg["prec"]
            if prec == "fp16":
                lm = lm.half()
            elif prec == "int4":
                lm = lm.half().quantize(4)
            elif prec == "int8":
                lm = lm.half().quantize(8)

        blip2 = Blip2ForChatGLM.from_pretrained(
            cfg["model_path"],
        )
        blip2_config = Blip2ChatGLMConfig.from_pretrained(cfg["model_path"])

        model = Blip2ChatGLM(blip2_config, blip2, lm)
        model.to(device)
        model.eval()

        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     model = torch.compile(model)

        image_size = model.blip2.config.vision_config.image_size
        image_processor = BlipImageProcessor(
            size={"height": image_size, "width": image_size},
            image_mean=OPENAI_CLIP_MEAN,
            image_std=OPENAI_CLIP_STD,
        )

        return cls(tokenizer, image_processor, model, device)

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        pixel_processor: BlipImageProcessor,
        model: Blip2ChatGLM,
        device: torch.device,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.pixel_processor = pixel_processor
        self.device = device

    def delete(self):
        del self.model
        empty_cache(self.device)

    def stream_generate(
        self,
        history: Iterator[Dict[str, Any]],
        max_tokens: int = 2048,
        top_p: float = 0.7,
        temperature: float = 0.95,
        **kwargs,
    ) -> Iterator[List[Dict[str, Any]]]:
        inference_history = []

        for info in history:

            def convert(info: Dict[str, Any]):
                content = info["content"]
                media = info["media"]
                if len(media) != 0:
                    data, mime = media[0]
                    pixel_values = self.pixel_processor(
                        Image.open(io.BytesIO(data)).convert("RGB"), return_tensors="pt"
                    ).pixel_values.to(self.device)
                    return (content, pixel_values)
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
