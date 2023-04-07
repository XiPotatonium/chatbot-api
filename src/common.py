# __future__.annotations will become the default in Python 3.11
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, MutableMapping, Optional, Type
import torch
import json
from rich.console import Console
from .model import *


if TYPE_CHECKING:
    from .model import Model


def _load_cfg(path: str):
    with Path(path).open('r', encoding="utf8") as rf:
        return json.load(rf)


@dataclass
class ModelMeta:
    cfg: MutableMapping[str, Any]
    cls: Type[Model]


MODEL_MAPPING = {
    "blip2zh-chatglm-6b": ModelMeta(_load_cfg("cfgs/blip2zh-chatglm-6b.json"), Blip2ChatGLMModel),
}
