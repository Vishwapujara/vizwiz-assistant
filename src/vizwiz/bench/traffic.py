"""Synthetic traffic for the local stack's dashboards (Step 5.4).

Sends generated photos (noise, dark noise, flat grey: one per cascade branch with
the mock model), never dataset images, at a steady rate.

    python -m vizwiz.bench.traffic --url http://localhost:8080 --rps 4 --duration 600
"""

from __future__ import annotations

import argparse
import asyncio
import io
import random
import time
from collections import Counter

import httpx
from PIL import Image

QUESTIONS = ("What is this?", "What colour is this shirt?", "What does the label say?",
             "Is the light on?", "How much is this?")  # fmt: skip
MIX = (("normal", 0.7), ("dark", 0.15), ("flat", 0.15))


def synthetic_photo(kind: str, rng: random.Random, size=(320, 240)) -> bytes:
    if kind == "flat":
        img = Image.new("RGB", size, (128, 128, 128))
    else:
        hi = 60 if kind == "dark" else 255
        img = Image.frombytes(
            "RGB", size, bytes(rng.randint(0, hi) for _ in range(size[0] * size[1] * 3))
        )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


async def run(url: str, rps: float, duration: float, seed: int = 0) -> Counter:
    rng = random.Random(seed)
    photos = {k: [synthetic_photo(k, rng) for _ in range(5)] for k, _ in MIX}
    counts: Counter = Counter()
    async with httpx.AsyncClient(base_url=url, timeout=30) as client:

        async def one():
            kind = rng.choices([k for k, _ in MIX], [w for _, w in MIX])[0]
            files = {"image": ("p.jpg", rng.choice(photos[kind]), "image/jpeg")}
            try:
                r = await client.post("/v1/assist", files=files,
                                      data={"question": rng.choice(QUESTIONS)})  # fmt: skip
                counts[r.json()["status"] if r.is_success else f"http_{r.status_code}"] += 1
            except httpx.HTTPError as e:
                counts[type(e).__name__] += 1

        tasks, t0, sent, report_at = set(), time.perf_counter(), 0, 30.0
        while (elapsed := time.perf_counter() - t0) < duration:
            if sent < elapsed * rps:
                task = asyncio.create_task(one())
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                sent += 1
            else:
                await asyncio.sleep(0.01)
            if elapsed >= report_at:
                print(f"{elapsed:5.0f}s  sent {sent}  {dict(counts)}", flush=True)
                report_at += 30
        await asyncio.gather(*tasks)
    return counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Synthetic traffic for the local stack")
    p.add_argument("--url", default="http://localhost:8080")
    p.add_argument("--rps", type=float, default=4)
    p.add_argument("--duration", type=float, default=600, help="seconds")
    args = p.parse_args(argv)
    print("TRAFFIC DONE:", dict(asyncio.run(run(args.url, args.rps, args.duration))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
