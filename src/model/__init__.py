from __future__ import annotations
from .blip2chatglm import Blip2ChatGLMModel
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
    "Blip2ChatGLMModel",
]


def _load_cfg(path: str):
    with Path(path).open("r", encoding="utf8") as rf:
        return json.load(rf)


@dataclass
class ModelMeta:
    name: str
    cfg: MutableMapping[str, Any]
    cls: Type[Model]


MODEL_MAPPING = {
    "blip2zh-chatglm-6b": ModelMeta(
        "blip2zh-chatglm-6b",
        _load_cfg("cfgs/blip2zh-chatglm-6b.json"),
        Blip2ChatGLMModel,
    ),
}
