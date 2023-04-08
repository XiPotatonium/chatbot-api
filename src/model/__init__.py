from __future__ import annotations
import sys
from .blip2chatglm import Blip2ChatGLMModel
from .chatglm import ChatGLMModel
from .llama import LlamaLoraModel
from .model import ChatModel, Model, iter_messages

# __future__.annotations will become the default in Python 3.11
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, MutableMapping, Optional, Type
import json


__all__ = [
    "Model",
    "ChatModel",
    "iter_messages",
    "ModelMeta",
]


@dataclass
class ModelMeta:
    name: str
    cfg: MutableMapping[str, Any]
    cls: Type[Model]


MODEL_MAPPING = {}

for file in Path("cfgs").iterdir():
    with file.open('r', encoding="utf8") as rf:
        cfg = json.load(rf)
        MODEL_MAPPING[file.stem] = ModelMeta(
            file.stem,
            cfg,
            getattr(sys.modules[__name__], cfg["cls"]),
        )
