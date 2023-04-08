import itertools
from asyncio import Lock
import traceback
from typing import Any, Dict, Iterator, List, MutableMapping, Optional, Tuple, Type

import torch
from multiprocessing import Queue, Process
from fastapi import HTTPException
import logging

from .model import Model, ChatModel, iter_messages, ModelMeta


class ModelInternelException(Exception):
    def __init__(self, msg: str = "", *args: object) -> None:
        super().__init__(*args)
        self.msg = msg

    def __str__(self) -> str:
        return self.msg

    def __repr__(self) -> str:
        return self.msg


class ModelResource:
    def __init__(
        self,
        process: Process,
        in_q: Queue,
        out_q: Queue,
        request_validator: Optional[callable] = None,
    ) -> None:
        self.process = process
        self.lock = Lock()
        self._in_q = in_q
        self._out_q = out_q
        self.request_validator = request_validator

    async def predict(self, req: dict, files: Dict[str, Tuple[bytes, str]]):
        """_summary_

        Args:
            req (dict): _description_
            files (Dict[str, Tuple[bytes, str]]): key is filename, value is tuple of (file content, file type)

        Raises:
            ValueError: when input validator failed
            NotImplementedError: _description_
            RuntimeError: _description_

        Returns:
            _type_: _description_

        Yields:
            _type_: _description_
        """
        try:
            if self.request_validator is not None:
                self.request_validator(req, files)
        except Exception as e:
            raise HTTPException(
                400, f"Oops! An error occurred in validating your requests: {str(e)}"
            )
        async with self.lock:
            self._in_q.put((req, files))
            while True:
                resp = self._out_q.get()
                if resp is None:
                    break
                if isinstance(resp, ModelInternelException):
                    raise HTTPException(
                        400, f"Oops! An error occurred during inference: {resp.msg}"
                    )
                yield resp

    async def terminate(self):
        async with self.lock:
            self._in_q.put(None)
            self.process.join()


class ModelPool:
    def __init__(self) -> None:
        self.models = {}

    def load_models(
        self, models: List[Tuple[ModelMeta, int]], devices: Iterator[Dict[str, Any]]
    ):
        for meta, count in models:
            model_lst = []
            for i in range(count):
                device = next(devices)
                device = torch.device(device["device"])
                in_q = Queue()
                out_q = Queue()
                process = Process(target=model_worker, args=(in_q, out_q, meta, device))
                process.start()
                model_lst.append(
                    ModelResource(process, in_q, out_q, meta.cls.request_validator)
                )
                # logging.info(
                #     f"Loaded {meta.name} ({i + 1}/{count}) on {device} (pid={process.pid})"
                # )
            self.models[meta.name] = model_lst
        # may be too simple design
        self.models_iter = {
            name: itertools.cycle(model_lst) for name, model_lst in self.models.items()
        }

    async def terminate(self):
        for model_lst in self.models.values():
            for model in model_lst:
                await model.terminate()

    def acquire(self, modelname: str) -> ModelResource:
        return next(self.models_iter[modelname])


MODEL_POOL = ModelPool()


def model_worker(in_q: Queue, out_q: Queue, meta: ModelMeta, device: torch.device):
    model = meta.cls.load(meta.cfg, device)
    logging.info(f"Loaded {meta.name} on {device}")
    while True:
        data = in_q.get()
        if data is None:
            break
        req, files = data
        try:
            if issubclass(meta.cls, ChatModel):
                messages = iter_messages(req.pop("messages"), files)
            else:
                raise SystemError(f"Unknown model type: {meta.cls}")

            if req.get("stream", False):
                for i, choices in enumerate(model.stream_generate(messages, **req)):
                    out_q.put(choices)
            else:
                choices = model.generate(messages, **req)
                # logging.debug(choices)
                # currently mm output is not supported
                out_q.put(choices)
            out_q.put(None)
        except Exception as e:
            logging.error(traceback.format_exc())
            out_q.put(ModelInternelException(msg=str(e)))
