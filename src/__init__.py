import json
from dataclasses import dataclass
from typing import List, Optional

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from starlette.responses import FileResponse, PlainTextResponse, HTMLResponse
from pydantic import BaseModel
from pathlib import Path
import pydantic
from .model_pool import ModelPool
from .common import MODEL_MAPPING
from .api import chat_completion


MODEL_POOL = ModelPool()


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


@app.on_event("shutdown")
def shutdown_event():
    print("Closing service...")
    print("Closing models")
    MODEL_POOL.terminate()


# io = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")
# gradio_app = gr.routes.App.create_app(io)
#
# app.mount("/", gradio_app)
