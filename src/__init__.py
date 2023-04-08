import json
from typing import List, Optional

from fastapi import Body, FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse
import logging
from pathlib import Path
from .model_pool import MODEL_POOL
from .model import MODEL_MAPPING
from .api import chat_completion
from .device import alloc


logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

app.include_router(chat_completion.router)


@app.on_event("startup")
async def startup_event():
    models = [
        # (MODEL_MAPPING["blip2zh-chatglm-6b"], 1)
        (MODEL_MAPPING["chatglm-6b"], 1)
        # (MODEL_MAPPING["llama-7b-lora"], 1)
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
