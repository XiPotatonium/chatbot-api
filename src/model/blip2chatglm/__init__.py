import sys
from loguru import logger
import torch
from ...device import empty_cache
from ..model import Model, ChatModel
from transformers import AutoModel, AutoTokenizer, BlipImageProcessor, PreTrainedTokenizer
from typing import Any, Dict, Iterator, List, Tuple, Union
from .modeling_blip2chatglm import Blip2ChatGLM, Blip2ForChatGLM
from .modeling_chatglm import ChatGLMForConditionalGeneration
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from PIL import Image


class Blip2ChatGLMModel(ChatModel):
    @classmethod
    def load(cls):
        tokenizer = AutoTokenizer.from_pretrained(sym_tbl().cfg["lm_path"], trust_remote_code=True)
        lm = ChatGLMForConditionalGeneration.from_pretrained(
            sym_tbl().cfg["lm_path"], # device_map="auto"
        )

        if sym_tbl().device_info["device"] == "cpu":
            lm = lm.float()
        else:
            prec = sym_tbl().cfg["prec"]
            if prec == "fp16":
                lm = lm.half()
            elif prec == "int4":
                lm = lm.half().quantize(4)
            elif prec == "int8":
                lm = lm.half().quantize(8)

        blip2 = Blip2ForChatGLM.from_pretrained(sym_tbl().cfg["model_path"],)

        model = Blip2ChatGLM(blip2, lm)
        model.to(sym_tbl().device)
        model.eval()

        if torch.__version__ >= "2" and sys.platform != "win32":
            logger.info("Use torch.compile")
            model = torch.compile(model)

        image_size = model.blip2.config.vision_config.image_size
        image_processor = BlipImageProcessor(
            size={"height": image_size, "width": image_size}, image_mean=OPENAI_CLIP_MEAN, image_std=OPENAI_CLIP_STD
        )

        sym_tbl().model = cls(tokenizer, image_processor, model)

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            pixel_processor: BlipImageProcessor,
            model: Blip2ChatGLM
        ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.pixel_processor = pixel_processor

    def delete(self):
        del self.model
        empty_cache()

    def stream_generate(
            self,
            history: Iterator[Dict[str, Any]],
            max_tokens: int = 2048,
            top_p: float = 0.7,
            temperature: float = 0.95,
            **kwargs
    ):
        history = []
        # DISCUSS: should we add a field to state to store chat history for inference?
        # PROS: save conversion time
        # CONS: memory consuming, especially for mm history
        for info in state.history:
            def convert(info: Dict[str, Any]):
                text = info["text"]
                mm_type = info["mm_type"]
                if len(info["mm_type"]) != 0:
                    mm_path = state.folder / info["mm_path"]
                    if info["mm_type"] == "Image":
                        pixel_values = self.pixel_processor(
                            Image.open(mm_path).convert("RGB"), return_tensors="pt"
                        ).pixel_values.to(sym_tbl().device)
                        return (text, pixel_values)
                    else:
                        logger.warning(
                            f"{self.__class__.__name__} is a text-image model, but got {mm_type} input."
                            "The media is ignored and only the text is used."
                        )
                return text
            history.append((convert(info["query"]), convert(info["response"])))
        query = history.pop()
        instruction = state.history[-1]["query"]["instruction"]
        if len(instruction) != 0:
            logger.warning(f"{self.__class__.__name__} will ignore instruction {instruction}.")
        query = query[0]

        for i, (output, _) in enumerate(self.model.stream_chat(
            self.tokenizer, query=query, history=history,
            max_length=max_tokens,
            top_p=top_p,
            temperature=temperature
        )):
            if i == 0:
                yield append_response_binding(state, binding, output)
            else:
                yield update_response_binding(state, binding, output)
        state.history[-1]["response"]["text"] = output
        empty_cache()
