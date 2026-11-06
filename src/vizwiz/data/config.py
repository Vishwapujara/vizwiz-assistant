"""Typed loader for configs/data.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    seed: int = 1234
    val_fraction: float = Field(0.10, gt=0, lt=1)
    quality_k: int = Field(2, ge=1, le=5)
    quality_issues: dict[str, str]
    unrecognizable_label: str = "unrecognizable"
    answerability_labels: list[str] = ["answerable", "unanswerable"]

    @property
    def quality_labels(self) -> list[str]:
        """All quality output codes, in a fixed order, plus 'none'."""
        return [*self.quality_issues.values(), self.unrecognizable_label, "none"]


REPO_ROOT = Path(__file__).resolve().parents[3]  # src/vizwiz/data/config.py -> repo
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data.yaml"


def load_data_config(path: Path | str = DEFAULT_CONFIG) -> DataConfig:
    return DataConfig(**yaml.safe_load(Path(path).read_text()))
