"""FastAPI gateway: photo + question -> cascade -> spoken-style reply (Step 5.2).

    POST /v1/assist   multipart form: image (file), question (text)
    GET  /health      gateway and model server status
    GET  /adapters    adapters configured per stage and adapters the server has loaded
    GET  /metrics     Prometheus metrics

    uvicorn --factory vizwiz.serve.gateway:create_app --port 8080

Privacy: logs and metrics hold ids, statuses, timings and quality labels only, never
the photo, the question or the answer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from vizwiz.serve.cascade import Cascade
from vizwiz.serve.config import ServeConfig, load_serve_config
from vizwiz.serve.model_client import ModelClient, ModelError
from vizwiz.serve.preprocess import prepare_image, to_data_url

log = logging.getLogger("vizwiz.gateway")
LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 20)


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()  # per app, so tests can build several apps
        r = self.registry
        self.requests = Counter("vizwiz_requests", "Assist requests by outcome", ["status"],
                                registry=r)  # fmt: skip
        self.latency = Histogram("vizwiz_request_seconds", "End-to-end assist latency",
                                 buckets=LATENCY_BUCKETS, registry=r)  # fmt: skip
        self.stage = Histogram("vizwiz_stage_seconds", "Model call latency per stage", ["stage"],
                               buckets=LATENCY_BUCKETS, registry=r)  # fmt: skip
        self.retries = Counter("vizwiz_model_retries", "Model call retries", ["stage"],
                               registry=r)  # fmt: skip
        self.in_flight = Gauge("vizwiz_in_flight", "Assist requests in progress", registry=r)


def _prepare(data: bytes, cfg: ServeConfig) -> str:
    return to_data_url(prepare_image(data, cfg.max_pixels), cfg.jpeg_quality)


def create_app(cfg: ServeConfig | None = None, client: ModelClient | None = None) -> FastAPI:
    cfg = cfg or load_serve_config()
    if not log.handlers:  # one JSON object per line on stderr, next to uvicorn's own logs
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
    owns_client = client is None
    client = client or ModelClient(
        cfg.model_url,
        api_key=cfg.model_api_key,
        timeout_s=cfg.timeouts.model_call_s,
        retries=cfg.timeouts.retries,
        backoff_s=cfg.timeouts.backoff_s,
    )
    cascade = Cascade(client, cfg)
    metrics = Metrics()
    max_bytes = int(cfg.limits.max_upload_mb * 1024 * 1024)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        if owns_client:
            await client.aclose()

    app = FastAPI(title="VizWiz visual assistant", lifespan=lifespan)

    def fail(status: str, code: int, detail: str, rid: str):
        metrics.requests.labels(status).inc()
        log.info(json.dumps({"request_id": rid, "status": status}))
        raise HTTPException(code, detail, headers={"X-Request-ID": rid})

    @app.post("/v1/assist")
    async def assist(image: Annotated[UploadFile, File()], question: Annotated[str, Form()]):
        rid = uuid.uuid4().hex[:16]
        question = question.strip()
        if not question:
            fail("bad_request", 422, "question is empty", rid)
        if len(question) > cfg.limits.max_question_chars:
            fail("bad_request", 422, "question is too long", rid)
        data = await image.read(max_bytes + 1)
        if len(data) > max_bytes:
            fail("too_large", 413, f"photo is larger than {cfg.limits.max_upload_mb:g} MB", rid)
        try:
            url = await asyncio.to_thread(_prepare, data, cfg)
        except Exception:
            fail("bad_image", 415, "the upload is not a readable photo", rid)
        t0 = time.perf_counter()
        metrics.in_flight.inc()
        try:
            result = await cascade.run(url, question)
        except ModelError:
            fail("model_error", 503, "the model is unavailable, please try again", rid)
        finally:
            metrics.in_flight.dec()
        metrics.requests.labels(result.status).inc()
        metrics.latency.observe(time.perf_counter() - t0)
        for t in result.trace:
            metrics.stage.labels(t.stage).observe(t.latency_ms / 1000)
            if t.attempts > 1:
                metrics.retries.labels(t.stage).inc(t.attempts - 1)
        log.info(json.dumps({
            "request_id": rid, "status": result.status, "latency_ms": result.latency_ms,
            "issues": result.issues, "unanswerable_score": result.unanswerable_score,
            "stages": [{"stage": t.stage, "latency_ms": t.latency_ms, "used": t.used}
                       for t in result.trace],
        }))  # fmt: skip
        return JSONResponse({"request_id": rid, **result.to_dict()}, headers={"X-Request-ID": rid})

    @app.get("/health")
    async def health():
        try:
            loaded = set(await client.models())
        except ModelError:
            return JSONResponse({"status": "degraded", "model_server": "unreachable"}, 503)
        missing = sorted(set(cfg.adapters.values()) - loaded)
        body = {"status": "ok" if not missing else "degraded", "model_server": "ok",
                "mode": cfg.mode, "missing_adapters": missing}  # fmt: skip
        return JSONResponse(body, 200 if not missing else 503)

    @app.get("/adapters")
    async def adapters():
        try:
            loaded = await client.models()
        except ModelError as e:
            raise HTTPException(503, "model server unreachable") from e
        return {"configured": cfg.adapters, "loaded": loaded}

    @app.get("/metrics")
    async def prometheus():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app
