import asyncio
import io
import json
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, APIRouter, UploadFile, Form, File, Body, Depends
from fastapi.responses import StreamingResponse
from starlette.responses import FileResponse, PlainTextResponse, HTMLResponse
from pydantic import BaseModel, Field
from ..model_pool import ModelPool, MODEL_POOL


router = APIRouter()


class ChatCompletionResponse(BaseModel):
    choices: List[dict]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[dict]
    stream: bool = Field(False)
    # NOTE: The following default values will not actually take effect since we exclude unset values in conversion.
    # field which is not specified by user will have default value defined in model.generate or model.stream_generate
    top_p: float = Field(1.0)
    temperature: float = Field(1.0)
    max_tokens: int = Field(512)
    kwargs: Dict[str, Any] = Field({})

    @classmethod
    def __get_validators__(cls):
        yield cls._validate_from_json_string

    @classmethod
    def _validate_from_json_string(cls, value):
        if isinstance(value, str):
            return cls.validate(json.loads(value.encode()))
        return cls.validate(value)


@router.post("/v1/mmchat/completions")
async def chat_completion(
    files: List[UploadFile] = File([]),
    data: ChatCompletionRequest = Body(...),
):
    try:
        model = await MODEL_POOL.acquire(data.model)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"model not found: {str(e)}")

    if files is None:
        files = {}
    else:
        files = {f.filename: (await f.read(), f.content_type) for f in files}
        # logging.info(files.keys())

    if data.stream:
        async def stream_generate():
            async for choices in model.generate(data.dict(exclude_unset=True), files):
                resp = ChatCompletionResponse(choices=choices)
                # logging.debug(resp)
                yield json.dumps(resp.dict()) + '\n'
        return StreamingResponse(stream_generate(), media_type='text/event-stream')
    else:
        choices = [pred async for pred in model.generate(data.dict(exclude_unset=True), files)]
        assert len(choices) == 1
        choices = choices[0]
        # logging.debug(choices)
        return ChatCompletionResponse(choices=choices).dict()


@router.post("/v1/chat/completions")
async def chat_completion(
    data: ChatCompletionRequest = Body(...),
):
    try:
        model = await MODEL_POOL.acquire(data.model)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"model not found: {str(e)}")

    if data.stream:
        async def stream_generate():
            async for choices in model.generate(data.dict(exclude_unset=True), {}):
                resp = ChatCompletionResponse(choices=choices)
                # logging.debug(resp)
                yield json.dumps(resp.dict()) + '\n'
        return StreamingResponse(stream_generate(), media_type='text/event-stream')
    else:
        choices = [pred async for pred in model.generate(data.dict(exclude_unset=True), {})]
        assert len(choices) == 1
        choices = choices[0]
        # logging.debug(choices)
        return ChatCompletionResponse(choices=choices).dict()
