import asyncio
import json
from typing import List, Optional

from fastapi import Body, FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse
import logging
from pathlib import Path
from .model_pool import MODEL_POOL
from .api import chat_completion


logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

app.include_router(chat_completion.router)


@app.on_event("startup")
async def startup_event():
    with Path("load_config.json").open('r', encoding="utf8") as rf:
        load_config = json.load(rf)
    MODEL_POOL.load_config = load_config
    asyncio.create_task(MODEL_POOL.close_idle_models())
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
