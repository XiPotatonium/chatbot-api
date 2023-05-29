import sys
import io
import torch

from ...device import empty_cache
from ...common import ROLE_BOT, ROLE_USER, ROLE_SYSTEM
from ..model import Model, ChatModel
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedModel,
    BlipImageProcessor,
    PreTrainedTokenizer,
)
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Tuple, Union
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
            if "media" in req:
                for img in req["media"]:
                    if isinstance(img, str):
                        # fname
                        pass
                    elif isinstance(img, (tuple, list)):
                        if len(img) != 2:
                            raise ValueError(
                                f"invalid media {img} for {cls.__name__}"
                            )
                        img, index = img
                        if not isinstance(img, str) or not isinstance(index, int) or index < 0:
                            raise ValueError(
                                f"invalid media {img} for {cls.__name__}"
                            )
                    else:
                        raise ValueError(f"invalid media {img} for {cls.__name__}")
            if req["role"] == ROLE_SYSTEM:
                raise ValueError(f"system role is not supported for {cls.__name__}")

    @classmethod
    def load(cls, cfg: MutableMapping[str, Any], device: torch.device) -> Model:
        if device == "cpu":
            lm_dtype = "fp32"
        else:
            lm_dtype = cfg["prec"]

        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_path"], trust_remote_code=True
        )
        model.setup_dtype(vision_encoder_dtype="fp16", lm_dtype=lm_dtype)

        if "lora_path" in cfg:
            from peft import PeftModel

            model.language_model = PeftModel.from_pretrained(
                model.language_model,
                cfg["lora_path"],
                # torch_dtype=torch.float16,
            )

        model.to(device)
        model.eval()

        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     logger.info("Use torch.compile")
        #     model = torch.compile(model)

        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_path"], trust_remote_code=True
        )

        image_size = model.config.vision_config.image_size
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
        model: PreTrainedModel,
        device: torch.device,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.pixel_processor = pixel_processor
        self.device = device

    def delete(self):
        del self.model
        empty_cache(self.device)

    def _prepare_input(
        self,
        messages: List[Dict[str, Any]],
        files: Dict[str, Tuple[bytes, str]],
    ):
        ret = []
        # DISCUSS: should we add a field to state to store chat history for inference?
        # PROS: save conversion time
        # CONS: memory consuming, especially for mm history
        role_mapping = {ROLE_USER: "问", ROLE_BOT: "答", ROLE_SYSTEM: "指令"}
        for message in messages:
            role = role_mapping[message["role"]]
            text = message["content"]
            medias = []
            for img in message.get("media", []):
                if isinstance(img, str):
                    index = 0       # position default at 0
                else:
                    img, index = img
                data, mime = files[img]
                pixel_values = self.pixel_processor(
                    Image.open(io.BytesIO(data)).convert("RGB"), return_tensors="pt"
                ).pixel_values.to(self.device)
                medias.append((pixel_values, index))

            ret.append((role, text, medias))
        return ret

    def stream_generate(
        self,
        messages: Iterator[Dict[str, Any]],
        files: Dict[str, Tuple[bytes, str]],
        max_tokens: int = 4096,
        top_p: float = 0.7,
        temperature: float = 0.95,
        **kwargs,
    ):
        messages = self._prepare_input(messages, files)

        last_output = ""
        yield [{"index": 0, "delta": {"role": ROLE_BOT}}]
        with torch.cuda.amp.autocast(enabled=True):
            for i, output in enumerate(
                self.model.stream_chat(
                    self.tokenizer,
                    messages=messages,
                    max_length=max_tokens,
                    top_p=top_p,
                    temperature=temperature,
                )
            ):
                yield [{"index": 0, "delta": {"content": output[len(last_output) :]}}]
                last_output = output
        yield [{"index": 0, "delta": {}}]
        empty_cache(self.device)
