import itertools
from asyncio import Lock
from typing import Any, List, MutableMapping, Tuple, Type

import torch
from .model import Model, ChatModel, iter_messages
from multiprocessing import Queue, Process


class ModelResource:
    def __init__(self, process: Process, in_q: Queue, out_q: Queue) -> None:
        self.process = process
        self.lock =  Lock()
        self._in_q = in_q
        self._out_q = out_q

    async def predict(self, req: dict):
        async with self.lock:
            self._in_q.put(req)
            while True:
                resp = self._out_q.get()
                if resp is None:
                    break
                raise NotImplementedError()
                if resp.get('error', None) is not None:
                    raise RuntimeError(resp['error'])
                if resp.get('choices', None) is not None:
                    return resp['choices']
                if resp.get('stream', None) is not None:
                    yield resp['stream']

    def terminate(self):
        with self.lock:
            self._in_q.put(None)
            self.process.join()


class ModelPool:

    def __init__(self, models: List[Tuple[str, Type[Model], MutableMapping[str, Any], List[torch.device]]]) -> None:
        self.models = {}
        for name, cls, cfg, devices in models:
            model_lst = []
            for device in devices:
                in_q = Queue()
                out_q = Queue()
                process = Process(target=model_worker, args=(in_q, out_q, cls, cfg, device))
                process.start()
                model_lst.append(ModelResource(process, in_q, out_q))
            self.models[name] = model_lst
        # may be too simple design
        self.models_iter = {
            name: itertools.cycle(model_lst) for name, model_lst in self.models.items()
        }

    def terminate(self):
        for model_lst in self.models.values():
            for model in model_lst:
                model.terminate()

    def acquire(self, modelname: str) -> ModelResource:
        return next(self.models_iter[modelname])


def model_worker(in_q: Queue, out_q: Queue, cls: Type[Model], cfg: MutableMapping[str, Any], device: torch.device):
    model = cls.load(cfg, device)
    allow_stream = cfg['allow_stream']
    while True:
        req = in_q.get()
        if req is None:
            break
        do_stream = req.get("stream", False)
        if do_stream and not allow_stream:
            out_q.put({'error': f'Streaming is not allowed for {cls}'})
            continue

        try:
            if cls == ChatModel:
                query = iter_messages(req.pop('messages'), allow_stream)
            else:
                raise SystemError(f'Unknown model type: {cls}')
            if do_stream:
                for i, choices in enumerate(model.generate(query, **req)):
                    raise NotImplementedError()
                    out_q.put()
            else:
                choices = model.generate(query, **req)
                # currently mm output is not supported
                out_q.put({
                        "choices": choices,
                    })
        except Exception as e:
            out_q.put({'error': str(e)})
