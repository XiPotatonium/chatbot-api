import json
from typing import List, Optional

from fastapi import Body, FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse
import logging
from pydantic import BaseModel
from pathlib import Path
import pydantic
from .model_pool import MODEL_POOL
from .model import MODEL_MAPPING
from .api import chat_completion
from .device import alloc


logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.include_router(chat_completion.router)


class Base(BaseModel):
    name: str
    point: Optional[float] = None
    is_accepted: Optional[bool] = False

    # @classmethod
    # def __get_validators__(cls):
    #     yield cls.validate_to_json

    # @classmethod
    # def validate_to_json(cls, value):
    #     if isinstance(value, str):
    #         return cls(**json.loads(value))
    #     return value

    @classmethod
    def __get_validators__(cls):
        yield cls._validate_from_json_string

    @classmethod
    def _validate_from_json_string(cls, value):
        if isinstance(value, str):
            return cls.validate(json.loads(value.encode()))
        return cls.validate(value)


@app.post("/debug")
def debug(files: List[UploadFile] = File(...), data: Base = Body(...)):
    return {"JSON Payload ": data, "Filenames": [file.filename for file in files]}


@app.on_event("startup")
async def startup_event():
    models = [
        (MODEL_MAPPING["blip2zh-chatglm-6b"], 1)
    ]
    total_models = sum(n for _, n in models)
    MODEL_POOL.load_models(models, alloc([[] for _ in range(total_models)]))
    logging.info("Server startup finished.")


@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Closing service...")
    logging.info("Closing models")
    await MODEL_POOL.terminate()


# io = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")
# gradio_app = gr.routes.App.create_app(io)
#
# app.mount("/", gradio_app)
