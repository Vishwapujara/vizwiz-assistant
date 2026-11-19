"""Async client for the model server's OpenAI-compatible API (vLLM or the mock).

One chat-completion call per cascade stage; the LoRA adapter is chosen by the
request's ``model`` field (vLLM multi-LoRA). Connection errors, 429 and 5xx are
retried with exponential backoff; anything else fails at once.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import httpx

RETRY_STATUS = {429, 500, 502, 503, 504}


class ModelError(RuntimeError):
    """The model server could not produce a reply (after retries)."""


@dataclass
class ModelReply:
    text: str
    first_top_logprobs: dict[str, float] = field(default_factory=dict)  # token -> logprob
    latency_ms: float = 0.0
    attempts: int = 1


class ModelClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 20.0,
        retries: int = 2,
        backoff_s: float = 0.2,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_s, transport=transport
        )
        self.retries, self.backoff_s = retries, backoff_s

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _request(self, method: str, path: str, **kw) -> tuple[httpx.Response, int]:
        delay = self.backoff_s
        for attempt in range(1, self.retries + 2):
            try:
                r = await self._http.request(method, path, **kw)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last = f"{type(e).__name__}"
            else:
                if r.status_code not in RETRY_STATUS:
                    if r.is_error:
                        raise ModelError(f"model server returned {r.status_code}")
                    return r, attempt
                last = f"HTTP {r.status_code}"
            if attempt <= self.retries:
                await asyncio.sleep(delay)
                delay *= 2
        raise ModelError(f"model server unavailable after {self.retries + 1} attempts ({last})")

    async def chat(
        self, adapter: str, messages: list[dict], max_tokens: int, top_logprobs: int = 0
    ) -> ModelReply:
        body: dict = {
            "model": adapter,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if top_logprobs:
            body |= {"logprobs": True, "top_logprobs": top_logprobs}
        t0 = time.perf_counter()
        r, attempts = await self._request("POST", "/v1/chat/completions", json=body)
        latency = (time.perf_counter() - t0) * 1000
        try:
            choice = r.json()["choices"][0]
            text = choice["message"]["content"] or ""
            first = {}
            content = (choice.get("logprobs") or {}).get("content") or []
            if content:
                first = {t["token"]: t["logprob"] for t in content[0].get("top_logprobs", [])}
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise ModelError("model server reply had an unexpected shape") from e
        return ModelReply(text, first, round(latency, 1), attempts)

    async def models(self) -> list[str]:
        r, _ = await self._request("GET", "/v1/models")
        return [m["id"] for m in r.json().get("data", [])]

    async def load_adapter(self, name: str, path: str) -> None:
        """vLLM runtime LoRA loading (server needs VLLM_ALLOW_RUNTIME_LORA_UPDATING=True)."""
        await self._request(
            "POST", "/v1/load_lora_adapter", json={"lora_name": name, "lora_path": path}
        )

    async def unload_adapter(self, name: str) -> None:
        await self._request("POST", "/v1/unload_lora_adapter", json={"lora_name": name})
