"""Gateway configuration: ``configs/serve.yaml`` plus environment overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from vizwiz.data.config import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "configs" / "serve.yaml"
STAGES = ("quality", "answerability", "answer")


class Rules(BaseModel):
    model_config = {"extra": "forbid"}
    answerability_threshold: float = Field(0.10, ge=0, le=1)  # chosen on val (D-063)
    unusable_labels: list[str] = []  # empty: quality gives tips only, no forced retake (D-064)
    max_tips: int = Field(2, ge=0)


class Timeouts(BaseModel):
    model_config = {"extra": "forbid"}
    model_call_s: float = Field(20, gt=0)
    retries: int = Field(2, ge=0)
    backoff_s: float = Field(0.2, ge=0)


class Limits(BaseModel):
    model_config = {"extra": "forbid"}
    max_upload_mb: float = Field(10, gt=0)
    max_question_chars: int = Field(500, gt=0)


class RegistryConfig(BaseModel):
    model_config = {"extra": "forbid"}
    enabled: bool = False  # off: ``adapters`` are names already loaded on the model server
    project: str = "vizwiz-assistant"
    entity: str | None = None  # W&B entity; default: the API key's default entity
    alias: str = "production"
    cache_dir: str = "/tmp/vizwiz-adapters"
    base_model: str = "Qwen/Qwen3-VL-4B-Instruct"  # promotion refuses other bases
    max_lora_rank: int = 16  # = the model server's --max-lora-rank


class ServeConfig(BaseModel):
    model_config = {"extra": "forbid"}
    model_url: str = "http://localhost:8000"
    model_api_key: str | None = None  # from the environment only, never from the YAML
    adapters: dict[str, str] = {s: "e5-multi-lora" for s in STAGES}
    max_pixels: int = 524_288
    jpeg_quality: int = Field(95, ge=50, le=100)
    # sequential: quality, then answerability, then answer (least GPU work)
    # parallel_checks: quality and answerability together, then answer (default)
    # speculative: all three together; the answer is dropped on a decline (lowest latency)
    mode: Literal["sequential", "parallel_checks", "speculative"] = "parallel_checks"
    rules: Rules = Rules()
    timeouts: Timeouts = Timeouts()
    limits: Limits = Limits()
    registry: RegistryConfig = RegistryConfig()

    def model_post_init(self, _context) -> None:
        missing = set(STAGES) - set(self.adapters)
        if missing:
            raise ValueError(f"adapters missing for stages: {sorted(missing)}")


def load_serve_config(path: Path | None = None) -> ServeConfig:
    path = Path(path or os.environ.get("VIZWIZ_SERVE_CONFIG") or DEFAULT_CONFIG)
    data = yaml.safe_load(path.read_text()) or {}
    data.pop("model_api_key", None)  # a key in the YAML would end up in git
    if url := os.environ.get("VIZWIZ_MODEL_URL"):
        data["model_url"] = url
    if key := os.environ.get("VIZWIZ_MODEL_API_KEY"):
        data["model_api_key"] = key
    return ServeConfig(**data)
