"""Training configuration: defaults from PROJECT_PLAN 8.1, a per-task YAML, and
command-line overrides (``--set key=value``), in that order."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from vizwiz.data.formatting import TASKS


class TrainConfig(BaseModel):
    model_config = {"extra": "forbid"}  # a typo in a YAML key is an error, not ignored

    task: Literal[TASKS]  # type: ignore[valid-type]
    model: str = "Qwen/Qwen3-VL-4B-Instruct"
    mode: Literal["lora", "qlora"] = "lora"
    # LoRA (fixed: the rank ablation is out of scope, PROJECT_PLAN 10)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Optimisation
    learning_rate: float = 1e-4
    warmup_ratio: float = Field(0.03, ge=0, lt=1)
    epochs: float = 1.0
    max_steps: int | None = None  # overrides epochs when set
    per_device_batch: int = 1  # fastest at 524k pixels in the L4 profile (D-050)
    grad_accum: int = 16  # effective batch = per_device_batch × grad_accum
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    seed: int = 1234
    # Data
    max_pixels: int = 524_288  # fixed by the D-047 rule (D-050)
    limit: int | None = None  # train examples (smoke tests)
    eval_limit: int = 500  # val examples for the periodic val loss
    num_workers: int = 4
    # Checkpointing and logging
    save_every_minutes: float | None = 10.0  # time-based checkpoints (owner's request)
    save_steps: int | None = None  # optional step-based checkpoints, in addition
    eval_steps: int = 200
    logging_steps: int = 10
    save_total_limit: int = 2

    @property
    def effective_batch(self) -> int:
        return self.per_device_batch * self.grad_accum


def parse_override(text: str) -> tuple[str, object]:
    """``key=value`` with the value parsed as YAML (so 2e-4, 10, true, null work)."""
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise ValueError(f"override must look like key=value, got {text!r}")
    return key.strip(), yaml.safe_load(value)


def load_train_config(
    path: Path, mode: str | None = None, overrides: list[str] | None = None
) -> TrainConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides or []:
        k, v = parse_override(item)
        data[k] = v
    if mode is not None:
        data["mode"] = mode
    return TrainConfig(**data)


def warmup_steps(total_steps: int, ratio: float) -> int:
    return max(1, round(total_steps * ratio)) if ratio > 0 else 0


def total_optimizer_steps(n_examples: int, cfg: TrainConfig) -> int:
    if cfg.max_steps:
        return cfg.max_steps
    per_epoch = -(-n_examples // cfg.effective_batch)  # ceil
    return max(1, round(per_epoch * cfg.epochs))


def save_due(now: float, last_save: float, minutes: float | None) -> bool:
    """True when at least ``minutes`` have passed since the last checkpoint."""
    return minutes is not None and now - last_save >= 60 * minutes


def last_checkpoint(run_dir: Path) -> Path | None:
    """Newest ``checkpoint-<step>`` directory that finished saving (has trainer_state)."""
    done = [
        p
        for p in run_dir.glob("checkpoint-*")
        if p.name.split("-")[-1].isdigit() and (p / "trainer_state.json").exists()
    ]
    return max(done, key=lambda p: int(p.name.split("-")[-1]), default=None)
