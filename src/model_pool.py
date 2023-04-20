import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import itertools
from asyncio import Lock
import traceback
from typing import Any, Dict, Iterator, List, MutableMapping, Optional, Tuple, Type

import torch
import multiprocessing as mp
from multiprocessing.connection import Connection
from collections import deque
from fastapi import HTTPException
import logging

from .device import DeviceMixin
from .model import Model, ChatModel, ModelMeta, MODEL_MAPPING


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
        process: mp.Process,
        conn: Connection,
        executor: ThreadPoolExecutor,
        request_validator: Optional[callable] = None,
    ) -> None:
        self.process = process
        self.lock = Lock()  # pipe should be protected by lock
        self._conn = conn
        self.executor = executor
        self.request_validator = request_validator

    async def generate(self, req: dict, files: Dict[str, Tuple[bytes, str]]):
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
            self._conn.send((req, files))
            while True:
                # resp = self._conn.recv()
                resp = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self._conn.recv
                )
                if resp is None:
                    break
                if isinstance(resp, ModelInternelException):
                    raise HTTPException(
                        400, f"Oops! An error occurred during inference: {resp.msg}"
                    )
                yield resp

    async def terminate(self):
        async with self.lock:
            self._conn.send(None)
            self.process.join()


class ModelPool(DeviceMixin):
    def __init__(self) -> None:
        super().__init__()

    def init(self, sched_config: Dict[str, Any]):
        self.sched_config = sched_config
        self.models = {modelname: [] for modelname in self.model_sched_configs.keys()}
        self.models_iter = {k: itertools.cycle(v) for k, v in self.models.items()}
        self.requests_history = {
            modelname: deque(
                maxlen=model_sched_config["create_threshold"]["n_requests"]
            )
            for modelname, model_sched_config in self.model_sched_configs.items()
        }
        self.executor = ThreadPoolExecutor(max_workers=8)
        # the async lock for the first model to avoid creating multiple models at the same time
        self.model_creating_locks = {
            modelname: Lock() for modelname in self.model_sched_configs.keys()
        }

    async def _load(self, modelname: str):
        meta = MODEL_MAPPING[modelname]
        logging.info(f"Try loading 1 instance of {modelname}...")
        async for d in self.alloc(count=1):
            # actually only 1 device,
            # NOTE: maybe anext in the future
            device = d
        device = torch.device(device["device"])
        p_conn, c_conn = mp.Pipe(duplex=True)
        process = mp.Process(target=model_worker, args=(c_conn, meta, device))
        process.start()
        resources = ModelResource(
            process, p_conn, self.executor, meta.cls.request_validator
        )
        self.models[modelname].append(resources)
        # logging.info(
        #     f"Loaded {meta.name} ({i + 1}/{count}) on {device} (pid={process.pid})"
        # )
        self.models_iter[modelname] = itertools.cycle(self.models[modelname])

    @property
    def model_sched_configs(self):
        return self.sched_config["models"]

    async def close_idle_models(self):
        check_period = self.sched_config["idle_check_period"]
        while True:
            for modelname, model_sched_config in self.model_sched_configs.items():
                req_history = self.requests_history[modelname]
                if len(self.models[modelname]) == 0:
                    continue
                cur_time = datetime.now()
                if (
                    len(req_history) == 0
                    or (cur_time - req_history[-1]).total_seconds()
                    > model_sched_config["idle_time"]
                ):
                    # close 1 instance of unused for a long period of time
                    # Based on https://docs.python.org/3/library/asyncio-sync.html
                    # Acquiring a lock is fair: the coroutine that proceeds will be the first coroutine that started waiting on the lock.
                    # Therefore, by removing the model we want to close in the resource list,
                    # the upcoming acquire will not get the model we want to close,
                    # and the requests which are waiting for the model we are going to close will not be effected
                    # since they will preceed before close_unused_models
                    res = next(self.models_iter[modelname])
                    self.models[modelname].remove(res)
                    self.models_iter[modelname] = itertools.cycle(
                        self.models[modelname]
                    )
                    await res.terminate()
                    logging.info(f"Shut down 1 instance of {modelname}.")
            await asyncio.sleep(check_period)

    async def terminate(self):
        for model_lst in self.models.values():
            for model in model_lst:
                await model.terminate()

    async def acquire(self, modelname: str) -> ModelResource:
        model_sched_config = self.model_sched_configs[modelname]
        cur_time = datetime.now()
        self.requests_history[modelname].append(cur_time)
        async with self.model_creating_locks[modelname]:
            # acquire a lock to avoid creating multiple models at the same time
            if len(self.models[modelname]) == 0:
                if model_sched_config["max_instances"] <= 0:
                    raise HTTPException(
                        400, f"Oops! Model {modelname} is not available now."
                    )
                # create if no instance
                await self._load(modelname)
            elif (
                len(self.models[modelname]) < model_sched_config["max_instances"]
                and self.get_free_gpu_number() > 0
                and (cur_time - self.requests_history[modelname][0]).total_seconds()
                < model_sched_config["create_threshold"]["delay"]
            ):
                # create if has free device and reach requests frequency threshold
                # remove all requests_history but last one to avoid too frequent creating
                last_req = self.requests_history[modelname].pop()
                self.requests_history[modelname].clear()
                self.requests_history[modelname].append(last_req)
                # NOTE: since we checked the free_gpu_number > 0, it is unlikely that the loading will await
                # So there is no efficency issue here
                await self._load(modelname)
        return next(self.models_iter[modelname])


MODEL_POOL = ModelPool()


def model_worker(conn: Connection, meta: ModelMeta, device: torch.device):
    model = meta.cls.load(meta.cfg, device)
    logging.info(f"Loaded {meta.name} on {device}")
    while True:
        data = conn.recv()
        if data is None:
            break
        req, files = data
        try:
            messages = req.pop("messages")
            kwargs = req.pop("kwargs", {})
            kwargs.update(req)

            if req.get("stream", False):
                for i, choices in enumerate(model.stream_generate(messages, files=files, **kwargs)):
                    conn.send(choices)
            else:
                choices = model.generate(messages, files=files, **kwargs)
                # logging.debug(choices)
                # currently mm output is not supported
                conn.send(choices)
            conn.send(None)
        except Exception as e:
            logging.error(traceback.format_exc())
            conn.send(ModelInternelException(msg=str(e)))
