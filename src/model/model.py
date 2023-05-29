from __future__ import annotations
from abc import abstractclassmethod, abstractmethod
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Tuple
import torch


class Model:
    @abstractclassmethod
    def request_validator(
        cls, request: Mapping[str, Any], files: Dict[str, Tuple[bytes, str]]
    ):
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
    def generate(
        self, messages: Iterator[Dict[str, Any]], files: Dict[str, Tuple[bytes, str]] = {}, **kwargs
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError()

    def stream_generate(
        self, messages: Iterator[Dict[str, Any]], files: Dict[str, Tuple[bytes, str]] = {}, **kwargs
    ) -> Iterator[List[Dict[str, Any]]]:
        raise NotImplementedError()
