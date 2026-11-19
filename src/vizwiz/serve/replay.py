"""Replay dataset questions through the served model path (Step 5.1, D-060).

For every VQA record: the photo is prepared exactly as the gateway does
(``prepare_image`` + JPEG), and all three cascade stages are called on the model
server, concurrently, with the gateway's own stage code. All three outputs are kept
for every question (the gateway would skip the answer on a decline), so the
decision rule can be tuned afterwards for any threshold without calling the model
again (``vizwiz.serve.tune``).

    python -m vizwiz.serve.replay --records <dir> --images <dir> --split val \\
        --model-url http://localhost:8000 --out <Drive>/runs/<id>/replay_val.jsonl

Output rows hold model output about dataset photos, so they stay on Drive (like
``predictions_*.jsonl``); only ``tune``'s aggregate summary is committed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from vizwiz.data.formatting import read_jsonl
from vizwiz.serve.cascade import Cascade
from vizwiz.serve.config import load_serve_config
from vizwiz.serve.model_client import ModelClient
from vizwiz.serve.preprocess import prepare_image, to_data_url


async def replay_one(cascade: Cascade, record: dict, path: Path, sem: asyncio.Semaphore) -> dict:
    cfg = cascade.cfg
    async with sem:
        url = await asyncio.to_thread(
            lambda: to_data_url(
                prepare_image(path, cfg.max_pixels), cfg.jpeg_quality, cfg.image_format
            )
        )
        t0 = time.perf_counter()
        (issues, tq), (score, ta), (answer, tans) = await asyncio.gather(
            cascade._quality(url),
            cascade._answerability(url, record["question"]),
            cascade._answer(url, record["question"]),
        )
    return {
        "id": record["id"],
        "issues": issues,
        "valid_json": tq.output["valid_json"],
        "score": score,  # full precision (R-009)
        "from_logprobs": ta.output["from_logprobs"],
        "answer": answer,
        "latency_ms": {"quality": tq.latency_ms, "answerability": ta.latency_ms,
                       "answer": tans.latency_ms,
                       "all": round((time.perf_counter() - t0) * 1000, 1)},
    }  # fmt: skip


async def replay(args) -> dict:
    from vizwiz.data.images import find_images

    cfg = load_serve_config(args.config)
    client = ModelClient(args.model_url, timeout_s=120, retries=3, backoff_s=1.0)
    cascade = Cascade(client, cfg)
    records = read_jsonl(args.records / f"vqa_{args.split}.jsonl")[: args.limit]
    image_path = find_images(args.images)
    done = set()
    if args.out.exists():  # resume after a disconnect
        done = {json.loads(line)["id"] for line in args.out.open()}
    todo = [r for r in records if r["id"] not in done]
    print(f"{len(records)} questions, {len(done)} already done, {len(todo)} to go")
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.perf_counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.out.open("a") as f:
            jobs = [replay_one(cascade, r, image_path[r["image"]], sem) for r in todo]
            for i, job in enumerate(asyncio.as_completed(jobs), 1):
                f.write(json.dumps(await job, ensure_ascii=False) + "\n")
                if i % 200 == 0 or i == len(todo):
                    f.flush()
                    print(
                        f"  {i}/{len(todo)}  {i / (time.perf_counter() - t0):.2f} q/s", flush=True
                    )
    finally:
        await client.aclose()
    wall = time.perf_counter() - t0
    summary = {"questions": len(todo), "wall_s": round(wall, 1),
               "questions_per_s": round(len(todo) / wall, 2) if todo else None,
               "concurrency": args.concurrency}  # fmt: skip
    print("REPLAY DONE:", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Replay VQA questions through the served path")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--model-url", default="http://localhost:8000")
    p.add_argument("--config", type=Path, help="serve.yaml (default: the committed one)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--limit", type=int)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    asyncio.run(replay(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
