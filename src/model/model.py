from __future__ import annotations
from abc import abstractclassmethod, abstractmethod
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Tuple
import torch
from ..common import ROLE_USER, ROLE_BOT, ROLE_SYSTEM


class Model:
    @abstractclassmethod
    def request_validator(cls, request: Mapping[str, Any], files: Dict[str, Tuple[bytes, str]]):
        pass

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
        ROLE_USER: {"content": "", "media": []},
        ROLE_BOT: {"content": "", "media": []},
    }


def iter_messages(messages: List[Dict[str, Any]], files: Dict[str, Tuple[bytes, str]]) -> Iterator[Dict[str, Any]]:
    if len(messages) == 0:
        raise ValueError("messages is empty")
    message = _new_message()
    for raw_msg in messages:
        if raw_msg["role"] == ROLE_USER:
            if len(message[ROLE_BOT]["content"]) != 0 or len(message[ROLE_BOT]["media"]) != 0:
                yield message
                message = _new_message()

            if "content" in raw_msg:
                message[ROLE_USER]["content"] = raw_msg["content"]
            if "media" in raw_msg:
                message[ROLE_USER]["media"] = [files[fname] for fname in raw_msg["media"]]
        elif raw_msg["role"] == ROLE_BOT:
            if "content" in raw_msg:
                message[ROLE_BOT]["content"] = raw_msg["content"]
            if "media" in raw_msg:
                message[ROLE_BOT]["media"] = [files[fname] for fname in raw_msg["media"]]
        elif raw_msg["role"] == ROLE_SYSTEM:
            # flush last message and this system message
            if len(message[ROLE_BOT]["content"]) != 0 or len(message[ROLE_BOT]["media"]) != 0:
                yield message
            yield {ROLE_SYSTEM: raw_msg["content"]}
            # start a new one
            message = _new_message()
        else:
            raise ValueError(f"Unknown role: {raw_msg['role']}")

    yield message