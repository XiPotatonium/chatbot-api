import os
import random
import time
from typing import Any, Dict, Iterator, List, Set
import importlib
import sys
import logging

import torch


class DeviceMixin:
    def __init__(self) -> None:
        if check_installed('pynvml'):
            self.device = "cuda"
            import pynvml
            pynvml.nvmlInit()
            self.systen_info = {
                "driver": pynvml.nvmlSystemGetCudaDriverVersion_v2()
            }
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                visible_devices = set(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
            else:
                visible_devices = set()
            all_cuda_device = list(range(pynvml.nvmlDeviceGetCount()))
            if len(visible_devices) == 0:
                # CUDA_VISIBLE_DEVICES not set by os.environ
                self.visible_devices = all_cuda_device
            else:
                self.visible_devices = list(visible_devices.intersection(all_cuda_device))
        else:
            raise SystemError("pynvml not installed. Currently only cuda device is supported")
        self.free_gpus = set()
        self.gpu_just_used = set()

    def detect_free_gpu(self, ):
        if self.device == "cuda":
            import pynvml

            free_gpus = set()
            # pick one gpu for training
            # detect free gpu if gpu queue is empty
            for index in self.visible_devices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = meminfo.used / 1024 / 1024   # MB
                if mem_used < 1500:
                    # add if gpu is free and not preserved (just allocated)
                    if index not in self.gpu_just_used:
                        free_gpus.add(index)
                else:
                    # remove from gpu_used list
                    # elaborate: If a gpu has been occupied by a process, the DeviceMixin will not preserve for it
                    self.gpu_just_used.discard(index)
            self.free_gpus = free_gpus
        else:
            raise NotImplementedError()

    def alloc(self, count: int = 1, waittime: float = 240.0) -> Iterator[Dict[str, Any]]:
        for _ in range(count):
            # pick one gpu for training
            while True:
                self.detect_free_gpu()
                if len(self.free_gpus) == 0:
                    logging.info("Visible: {}. Waiting for Free GPU ......".format(self.visible_devices))
                    time.sleep(waittime)
                else:
                    logging.info(f"Available device: {self.free_gpus}")
                    device_id = random.sample(list(self.free_gpus), k=1)[0]
                    self.free_gpus.discard(device_id)
                    self.gpu_just_used.add(device_id)
                    break

            if self.device == "cuda":
                import pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                yield {
                    "device": "cuda:{}".format(device_id),
                    "desc": str(pynvml.nvmlDeviceGetName(handle)),
                    **self.systen_info
                }
            else:
                raise NotImplementedError()

    def get_free_gpu_number(self) -> int:
        self.detect_free_gpu()
        return len(self.free_gpus)


def check_installed(package_name: str):
    if package_name in sys.modules:
        return True
    elif (spec := importlib.util.find_spec(package_name)) is not None:
        # If you choose to perform the actual import ...
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)
        return True
    else:
        return False


def empty_cache(device: torch.device):
    if device.type == "cpu":
        return
    with torch.cuda.device(device):
        # empty cache uses GPU 0 by default
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
