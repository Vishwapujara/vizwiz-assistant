"""Environment check and Drive cache setup for Colab runtimes.

Run as a script so it always sees the freshly installed package:

    python -m vizwiz.env --drive-root /content/drive/MyDrive/vizwiz

It prints a report (Python, RAM, disk, GPU and what that GPU supports), creates the
cache directories on Drive, and saves the report as JSON under ``<drive-root>/logs``.
Works on CPU-only machines and without torch installed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# Sub-directories created under the Drive root. Datasets, weights and checkpoints live
# here so they survive Colab disconnects and are not re-downloaded every session.
CACHE_DIRS = ("hf_cache", "datasets", "checkpoints", "results", "logs")

GiB = 1024**3


@dataclass
class GpuInfo:
    name: str
    memory_gib: float
    compute_capability: tuple[int, int]

    @property
    def bf16(self) -> bool:
        # bf16 tensor cores arrived with Ampere (sm80). T4 is Turing (sm75).
        return self.compute_capability >= (8, 0)

    @property
    def flash_attention_2(self) -> bool:
        # FlashAttention-2 requires Ampere or newer.
        return self.compute_capability >= (8, 0)

    @property
    def fp8(self) -> bool:
        # Native FP8 tensor cores start at Ada (sm89), e.g. L4.
        return self.compute_capability >= (8, 9)


@dataclass
class EnvReport:
    timestamp: str
    python: str
    platform: str
    ram_gib: float | None
    disk_free_gib: dict[str, float]
    torch: str | None
    cuda: str | None
    gpu: dict | None
    role: str
    notes: list[str] = field(default_factory=list)


def classify_gpu(gpu: GpuInfo | None) -> tuple[str, list[str]]:
    """Return the role this runtime plays in the project, plus warnings."""
    if gpu is None:
        return "cpu: data download, audit, preprocessing only", []
    notes: list[str] = []
    if not gpu.bf16:
        notes.append("No bf16: use only for smoke tests (QLoRA / fp16), never real training.")
    if not gpu.flash_attention_2:
        notes.append("No FlashAttention-2: use attn_implementation='sdpa'.")
    if gpu.memory_gib < 20:
        notes.append("Under 20 GB: not enough for bf16 LoRA training or vLLM serving of 4B.")
    if gpu.bf16 and gpu.memory_gib >= 20:
        role = "main: training, full evaluation, vLLM serving, benchmarks"
    else:
        role = "smoke tests only"
    return role, notes


def detect_gpu() -> GpuInfo | None:
    """Detect the first GPU via torch if available, otherwise nvidia-smi."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return GpuInfo(
                props.name, round(props.total_memory / GiB, 1), (props.major, props.minor)
            )
        return None
    except ImportError:
        pass
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    name, mem_mib, cap = (part.strip() for part in out.splitlines()[0].split(","))
    major, minor = (int(x) for x in cap.split("."))
    return GpuInfo(name, round(float(mem_mib) / 1024, 1), (major, minor))


def total_ram_gib() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GiB, 1)
    except (ValueError, OSError, AttributeError):
        return None


def torch_versions() -> tuple[str | None, str | None]:
    try:
        import torch
    except ImportError:
        return None, None
    return torch.__version__, torch.version.cuda


def setup_cache_dirs(drive_root: Path) -> list[Path]:
    paths = [drive_root / name for name in CACHE_DIRS]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def build_report(disk_paths: list[Path]) -> EnvReport:
    gpu = detect_gpu()
    role, notes = classify_gpu(gpu)
    torch_v, cuda_v = torch_versions()
    disk = {}
    for path in disk_paths:
        if path.exists():
            disk[str(path)] = round(shutil.disk_usage(path).free / GiB, 1)
    gpu_dict = None
    if gpu is not None:
        gpu_dict = asdict(gpu) | {
            "bf16": gpu.bf16,
            "flash_attention_2": gpu.flash_attention_2,
            "fp8": gpu.fp8,
        }
    return EnvReport(
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        python=sys.version.split()[0],
        platform=platform.platform(),
        ram_gib=total_ram_gib(),
        disk_free_gib=disk,
        torch=torch_v,
        cuda=cuda_v,
        gpu=gpu_dict,
        role=role,
        notes=notes,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--drive-root", type=Path, default=None, help="Cache root on Drive")
    args = parser.parse_args(argv)

    disk_paths = [Path("/content"), Path.cwd()]
    if args.drive_root is not None:
        created = setup_cache_dirs(args.drive_root)
        disk_paths.append(args.drive_root)
        print("Cache dirs ready:")
        for path in created:
            print(f"  {path}")

    report = build_report(disk_paths)
    text = json.dumps(asdict(report), indent=2)
    print(text)

    if args.drive_root is not None:
        gpu_tag = (report.gpu or {}).get("name", "cpu").replace(" ", "_")
        stamp = report.timestamp.replace(":", "")
        out = args.drive_root / "logs" / f"env_{gpu_tag}_{stamp}.json"
        out.write_text(text)
        print(f"Saved report to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
