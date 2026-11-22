"""Precompute the dashboard's try-it answers (D-068).

Sends every sample photo + question in the samples manifest to a running gateway and
saves the full replies, so the dashboard can show real model answers with no GPU on.

    python -m vizwiz.serve.demo --gateway http://127.0.0.1:8090 \\
        --samples dashboard/public/samples/samples.json --out results/demo/answers.json

The samples are openly licensed photos (never VizWiz images); see the manifest.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import httpx


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True, cwd=Path(__file__).parent)  # fmt: skip
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


async def collect(client: httpx.AsyncClient, samples: list[dict], image_dir: Path) -> dict:
    """Ask the gateway about every sample, one at a time (latencies stay comparable)."""
    health = (await client.get("/health")).json()
    adapters = (await client.get("/adapters")).json().get("stages", {})
    answers = []
    for s in samples:
        path = image_dir / s["file"]
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        r = await client.post(
            "/v1/assist",
            files={"image": (path.name, path.read_bytes(), mime)},
            data={"question": s["question"]},
        )
        r.raise_for_status()
        body = r.json()
        answers.append({
            "id": s["id"], "status": body["status"], "message": body["message"],
            "answer": body["answer"], "issues": body["issues"],
            "unanswerable_score": body["unanswerable_score"], "tips": body["tips"],
            "latency_ms": body["latency_ms"],
        })  # fmt: skip
        print(f"DEMO {s['id']}: {body['status']} | {body['message']}", flush=True)
    return {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "mode": health.get("mode"),
        "adapters": adapters,
        "answers": answers,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gateway", required=True, help="gateway base URL")
    ap.add_argument("--samples", required=True, type=Path, help="samples.json manifest")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args(argv)

    samples = json.loads(args.samples.read_text())
    headers = {"X-API-Key": k} if (k := os.environ.get("VIZWIZ_API_KEY")) else {}

    async def run() -> dict:
        async with httpx.AsyncClient(
            base_url=args.gateway, timeout=args.timeout, headers=headers
        ) as c:
            return await collect(c, samples, args.samples.parent)

    report = asyncio.run(run())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"DEMO saved {len(report['answers'])} answers to {args.out}")


if __name__ == "__main__":
    main()
