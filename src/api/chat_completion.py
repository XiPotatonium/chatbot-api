import asyncio
import io
import json
from dataclasses import dataclass
import logging
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, APIRouter, UploadFile, Form, File, Body, Depends
from fastapi.responses import StreamingResponse
from starlette.responses import FileResponse, PlainTextResponse, HTMLResponse
from pydantic import BaseModel
from ..model_pool import ModelPool, MODEL_POOL


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
    files: List[UploadFile] = File([]),
    data: ChatCompletionRequest = Body(...),
):
    try:
        model = MODEL_POOL.acquire(data.model)
    except KeyError:
        raise HTTPException(status_code=404, detail="model not found")

    if files is not None:
        files = {f.filename: (await f.read(), f.content_type) for f in files}
        # logging.info(files.keys())

    if data.stream:
        async def stream_generate():
            async for choices in model.predict(data.dict(), files):
                resp = {'choices': choices}
                # logging.debug(resp)
                yield json.dumps(resp) + '\n'
        return StreamingResponse(stream_generate(), media_type='text/event-stream')
    else:
        choices = [pred async for pred in model.predict(data.dict(), files)]
        assert len(choices) == 1
        return ChatCompletionResponse(choices=choices)
