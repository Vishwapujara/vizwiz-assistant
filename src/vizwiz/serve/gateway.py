"""FastAPI gateway: photo + question -> cascade -> spoken-style reply (Step 5.2).

    POST /v1/assist   multipart form: image (file), question (text)
    GET  /health      gateway and model server status
    GET  /adapters    adapter per stage, adapters the server has loaded, registry status
    GET  /metrics     Prometheus metrics
    POST /adapters/{name}/load      {"version": "v1"}  load without serving   (admin)
    POST /adapters/{name}/promote   {"version": "v1"}  make it live            (admin)
    POST /adapters/{name}/rollback                     back to the previous    (admin)

Admin endpoints need the registry on and the header ``X-Admin-Token`` equal to
``VIZWIZ_ADMIN_TOKEN``; with no token set they are disabled.

``POST /v1/assist`` is open unless ``VIZWIZ_API_KEY`` is set; then it needs the header
``X-API-Key`` with that value (for a gateway on a public address, D-069).

    uvicorn --factory vizwiz.serve.gateway:create_app --port 8080

Privacy: logs and metrics hold ids, statuses, timings and quality labels only, never
the photo, the question or the answer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from vizwiz.serve.adapters import AdapterManager
from vizwiz.serve.cascade import Cascade
from vizwiz.serve.config import ServeConfig, load_serve_config
from vizwiz.serve.model_client import ModelClient, ModelError
from vizwiz.serve.preprocess import prepare_image, to_data_url
from vizwiz.serve.registry import Registry, RegistryError

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
        self.live = Gauge("vizwiz_adapter_live", "Adapter serving each stage", ["stage", "adapter"],
                          registry=r)  # fmt: skip

    def set_live(self, stages: dict[str, str]) -> None:
        self.live.clear()
        for stage, adapter in stages.items():
            self.live.labels(stage, adapter).set(1)


def _prepare(data: bytes, cfg: ServeConfig) -> str:
    return to_data_url(prepare_image(data, cfg.max_pixels), cfg.jpeg_quality, cfg.image_format)


def create_app(
    cfg: ServeConfig | None = None,
    client: ModelClient | None = None,
    registry: Registry | None = None,
    admin_token: str | None = None,
    api_key: str | None = None,
) -> FastAPI:
    cfg = cfg or load_serve_config()
    admin_token = admin_token or os.environ.get("VIZWIZ_ADMIN_TOKEN") or None
    api_key = api_key or os.environ.get("VIZWIZ_API_KEY") or None
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
    metrics.set_live(cascade.adapters)
    if registry is None and cfg.registry.enabled:
        from vizwiz.serve.registry import WandbRegistry

        registry = WandbRegistry(cfg.registry.project, cfg.registry.entity)
    manager = AdapterManager(cascade, client, registry, cfg) if registry is not None else None
    max_bytes = int(cfg.limits.max_upload_mb * 1024 * 1024)

    @asynccontextmanager
    async def lifespan(_app):
        if manager is not None:
            await manager.sync()
            metrics.set_live(cascade.adapters)
        yield
        if owns_client:
            await client.aclose()

    app = FastAPI(title="VizWiz visual assistant", lifespan=lifespan)
    app.state.manager, app.state.cascade, app.state.metrics = manager, cascade, metrics

    def fail(status: str, code: int, detail: str, rid: str):
        metrics.requests.labels(status).inc()
        log.info(json.dumps({"request_id": rid, "status": status}))
        raise HTTPException(code, detail, headers={"X-Request-ID": rid})

    @app.post("/v1/assist")
    async def assist(
        image: Annotated[UploadFile, File()],
        question: Annotated[str, Form()],
        x_api_key: Annotated[str | None, Header()] = None,
    ):
        rid = uuid.uuid4().hex[:16]
        if api_key and not (x_api_key and secrets.compare_digest(x_api_key, api_key)):
            fail("unauthorized", 401, "missing or wrong X-API-Key", rid)
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
        missing = sorted(set(cascade.adapters.values()) - loaded)
        body = {"status": "ok" if not missing else "degraded", "model_server": "ok",
                "mode": cfg.mode, "missing_adapters": missing}  # fmt: skip
        return JSONResponse(body, 200 if not missing else 503)

    @app.get("/adapters")
    async def adapters():
        try:
            loaded = await client.models()
        except ModelError as e:
            raise HTTPException(503, "model server unreachable") from e
        body = {"stages": dict(cascade.adapters), "loaded": loaded}
        if manager is not None:
            body["registry"] = manager.status()
        return body

    def admin(token: str | None) -> AdapterManager:
        if manager is None:
            raise HTTPException(404, "the adapter registry is not enabled")
        if not admin_token:
            raise HTTPException(403, "admin endpoints are disabled (no VIZWIZ_ADMIN_TOKEN)")
        if not token or not secrets.compare_digest(token, admin_token):
            raise HTTPException(401, "missing or wrong X-Admin-Token")
        return manager

    async def run_admin(action, *args):
        try:
            out = await action(*args)
        except RegistryError as e:
            raise HTTPException(409, str(e)) from e
        except ModelError as e:
            raise HTTPException(503, f"model server refused the change: {e}") from e
        metrics.set_live(cascade.adapters)
        log.info(
            json.dumps({"admin": action.__name__, "args": list(args), "stages": cascade.adapters})
        )
        return out

    @app.post("/adapters/{name}/load")
    async def load(name: str, version: Annotated[str, Body(embed=True)],
                   x_admin_token: Annotated[str | None, Header()] = None):  # fmt: skip
        v = await run_admin(admin(x_admin_token).load, name, version)
        return {"loaded": v.served_name, "metadata": v.metadata}

    @app.post("/adapters/{name}/promote")
    async def promote(name: str, version: Annotated[str, Body(embed=True)],
                      x_admin_token: Annotated[str | None, Header()] = None):  # fmt: skip
        out = await run_admin(admin(x_admin_token).promote, name, version)
        return {**out, "stages": dict(cascade.adapters)}

    @app.post("/adapters/{name}/rollback")
    async def rollback(name: str, x_admin_token: Annotated[str | None, Header()] = None):
        out = await run_admin(admin(x_admin_token).rollback, name)
        return {**out, "stages": dict(cascade.adapters)}

    @app.get("/metrics")
    async def prometheus():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app
