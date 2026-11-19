"""Fake OpenAI-compatible model server for local runs, Docker Compose and CI.

Speaks the part of vLLM's API the gateway uses (``GET /v1/models``,
``POST /v1/chat/completions`` with logprobs, runtime LoRA load and unload). It
recognises the task from the system prompt and answers from simple photo
statistics, so every cascade branch can be reached with a synthetic image:

    flat photo (almost one colour)  -> quality ["unrecognizable"], answerability unanswerable
    dark photo (mean brightness <50) -> quality ["dark"], answerability unanswerable
    anything else                    -> quality ["none"], answerable, "mock answer"

    uvicorn vizwiz.serve.mock_model:app --port 8000
    env: MOCK_ADAPTERS (comma list, default e5-multi-lora), MOCK_LATENCY_MS (default 0)
"""

from __future__ import annotations

import asyncio
import base64
import io
import math
import os
import time

from fastapi import FastAPI, HTTPException, Request
from PIL import Image, ImageStat

from vizwiz.data.formatting import SYSTEM_PROMPTS

BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def photo_stats(data_url: str) -> tuple[float, float]:
    """(mean, standard deviation) of the photo's brightness, 0-255."""
    _, _, payload = data_url.partition("base64,")
    img = Image.open(io.BytesIO(base64.b64decode(payload))).convert("L")
    st = ImageStat.Stat(img)
    return st.mean[0], st.stddev[0]


def mock_reply(task: str, mean: float, std: float) -> tuple[str, float]:
    """(reply text, P(unanswerable)) for a task and photo statistics."""
    flat, dark = std < 6, mean < 50
    if task == "quality":
        return ('["unrecognizable"]' if flat else '["dark"]' if dark else '["none"]'), 0.0
    if task == "answerability":
        p = 0.95 if flat else 0.8 if dark else 0.02
        return ("unanswerable" if p >= 0.5 else "answerable"), p
    return "mock answer", 0.0


def create_mock_app(adapters: list[str] | None = None, latency_ms: float = 0.0) -> FastAPI:
    adapters = list(["e5-multi-lora"] if adapters is None else adapters)  # load/unload change it
    task_of = {prompt: task for task, prompt in SYSTEM_PROMPTS.items()}
    app = FastAPI(title="vizwiz mock model")

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": m, "object": "model"}
                                           for m in [BASE_MODEL, *adapters]]}  # fmt: skip

    @app.post("/v1/load_lora_adapter")
    async def load(request: Request):
        body = await request.json()
        name, path = body.get("lora_name"), body.get("lora_path", "")
        if not os.path.exists(os.path.join(path, "adapter_config.json")):
            raise HTTPException(400, f"no adapter at {path!r}")
        if name in adapters:
            raise HTTPException(400, f"adapter {name!r} is already loaded")
        adapters.append(name)
        return f"Success: LoRA adapter {name!r} added successfully."

    @app.post("/v1/unload_lora_adapter")
    async def unload(request: Request):
        name = (await request.json()).get("lora_name")
        if name not in adapters:
            raise HTTPException(404, f"adapter {name!r} is not loaded")
        adapters.remove(name)
        return f"Success: LoRA adapter {name!r} removed successfully."

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        if body.get("model") not in [BASE_MODEL, *adapters]:
            raise HTTPException(404, f"model {body.get('model')!r} does not exist")
        msgs = body.get("messages", [])
        task = task_of.get(msgs[0].get("content")) if msgs else None
        image = next((p["image_url"]["url"] for m in msgs if isinstance(m.get("content"), list)
                      for p in m["content"] if p.get("type") == "image_url"), None)  # fmt: skip
        if task is None or image is None:
            raise HTTPException(400, "expected a known system prompt and one image")
        if latency_ms:
            await asyncio.sleep(latency_ms / 1000)
        text, p = mock_reply(task, *photo_stats(image))
        choice: dict = {"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": text}}  # fmt: skip
        if body.get("logprobs"):
            p = min(max(p, 1e-6), 1 - 1e-6)
            top = [{"token": "un", "logprob": math.log(p)},
                   {"token": "answer", "logprob": math.log(1 - p)}]  # fmt: skip
            first = top[0] if text.startswith("un") else top[1]
            choice["logprobs"] = {"content": [{**first, "top_logprobs": top}]}
        return {"id": "mock", "object": "chat.completion", "created": int(time.time()),
                "model": body["model"], "choices": [choice]}  # fmt: skip

    return app


app = create_mock_app(
    [a for a in os.environ.get("MOCK_ADAPTERS", "e5-multi-lora").split(",") if a],
    float(os.environ.get("MOCK_LATENCY_MS", "0")),
)
