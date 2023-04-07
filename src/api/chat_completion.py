import asyncio
import io
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, APIRouter, UploadFile, Form, File, Body, Depends
from fastapi.responses import StreamingResponse
from starlette.responses import FileResponse, PlainTextResponse, HTMLResponse
from pydantic import BaseModel
from pathlib import Path
import pydantic
from PIL import Image
from ..model_pool import ModelPool
from ..common import MODEL_MAPPING


router = APIRouter()


class ChatCompletionResponse(BaseModel):
    choices: List[dict]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[dict]
    stream: bool = False
    top_k: int = 5
    temperature: float = 1.0

    @classmethod
    def __get_validators__(cls):
        yield cls._validate_from_json_string

    @classmethod
    def _validate_from_json_string(cls, value):
        if isinstance(value, str):
            return cls.validate(json.loads(value.encode()))
        return cls.validate(value)


@router.post("/api/chat/completion")
async def chat_completion(
    image: Optional[UploadFile] = File(None),
    data: ChatCompletionRequest = Body(...),
):
    # model = MODEL_POOL.get(request.model)
    # if model is None:
    #     raise HTTPException(status_code=404, detail="model not found")
    if image is not None:
        # print(image)
        print(image.filename)
        contents = await image.read()
        # print(len(contents))
        pilimg = Image.open(io.BytesIO(contents))
        print(pilimg.size)
    if data.stream:
        print("streaming")
        all_choices = [
            [{"delta": {"role": "assistant"}}],
            [{"delta": {"content": "This"}}],
            [{"delta": {"content": " "}}],
            [{"delta": {"content": "is"}}],
            [{"delta": {"content": " "}}],
            [{"delta": {"content": "a"}}],
            [{"delta": {"content": " "}}],
            [{"delta": {"content": "debug"}}],
            [{"delta": {"content": " "}}],
            [{"delta": {"content": "message"}}],
            [{"delta": {"content": " "}}],
            [{"delta": {"content": "."}}],
            [{"delta": {}}],
        ]
        async def stream_generate():
            for choices in all_choices:
                yield json.dumps({"choices": choices}) + '\n'
                await asyncio.sleep(0.1)
        return StreamingResponse(stream_generate(), media_type='text/event-stream')
    else:
        choices = [{"message": {"role": "assistant", "content": "This is a debug message."}}]
        return ChatCompletionResponse(choices=choices)
