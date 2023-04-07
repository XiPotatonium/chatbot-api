from __future__ import annotations
from abc import abstractclassmethod, abstractmethod
from typing import Any, Dict, Iterator, List, MutableMapping
import torch


class Model:
    @abstractclassmethod
    def load(cls, cfg: MutableMapping[str, Any], device: torch.device) -> Model:
        pass

    @abstractmethod
    def delete(self):
        pass

    def generate(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """_summary_

        Raises:
            NotImplementedError: _description_

        Returns:
            List[Dict[str, Any]]: choices
        """
        raise NotImplementedError()

    def stream_generate(self, *args, **kwargs) -> Iterator[List[Dict[str, Any]]]:
        raise NotImplementedError()


class ChatModel(Model):
    def generate(self, history: Iterator[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError()

    def stream_generate(self, history: Iterator[Dict[str, Any]], **kwargs) -> Iterator[List[Dict[str, Any]]]:
        raise NotImplementedError()


def _new_message():
    return {
        "user": {"text": "", "mm": None, "mm_meta": {}},
        "assistance": {"text": "", "mm": None, "mm_meta": {}},
    }


def iter_messages(messages: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    if len(messages) == 0:
        raise ValueError("messages is empty")
    message = _new_message()
    for raw_msg in messages:
        if raw_msg["role"] == "user":
            if len(message["assistant"]["text"]) != 0 or len(message["assistant"]["mm_type"]) != 0:
                yield message
                message = _new_message()

            if isinstance(q, str):
                # text-only query
                message["user"]["text"] = q
            else:
                # mm query
                message["user"]["mm"] = q["mm"]
                message["user"][""] = q["alt_txt"]
        elif raw_msg["role"] == "assistant":
            if isinstance(r, str):
                message["assistant"]["text"] = r
            else:
                message["assistant"]["mm_path"] = r["name"]
                message["assistant"]["mm_type"] = r["alt_text"]
        elif raw_msg["role"] == "system":
            # flush last message and this system message
            if len(message["assistant"]["text"]) != 0 or len(message["assistant"]["mm_type"]) != 0:
                yield message
            yield {"system": raw_msg["content"]}
            # start a new one
            message = _new_message()
        else:
            raise ValueError(f"Unknown role: {raw_msg['role']}")

    yield message