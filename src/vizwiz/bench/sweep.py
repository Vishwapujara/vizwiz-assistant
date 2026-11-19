"""Concurrency sweep against a running gateway with Locust (Step 6.1, D-065).

For each concurrency level: a short warm-up, then a measured Locust run. Around
the measured run the gateway's and the model server's Prometheus metrics are
read, so each row also has the outcome mix, mean time per cascade stage, and
the model server's peak KV-cache use.

    python -m vizwiz.bench.sweep --label e5-parallel --gateway http://127.0.0.1:8090 \\
        --model-server http://127.0.0.1:8000 --locust /content/locust-venv/bin/locust \\
        --samples /content/bench_samples.json --users 1 2 4 8 16 32 --out sweep.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

LOCUSTFILE = Path(__file__).with_name("locustfile.py")
PCTS = ("50%", "90%", "95%", "99%")


def parse_prometheus(text: str) -> dict[tuple[str, tuple], float]:
    """{(sample name, sorted labels): value} from the Prometheus text format."""
    from prometheus_client.parser import text_string_to_metric_families

    return {(s.name, tuple(sorted(s.labels.items()))): s.value
            for fam in text_string_to_metric_families(text) for s in fam.samples}  # fmt: skip


def scrape(url: str) -> dict:
    with urllib.request.urlopen(f"{url}/metrics", timeout=5) as r:
        return parse_prometheus(r.read().decode())


def delta(after: dict, before: dict, name: str, label: str) -> dict[str, float]:
    """Increase of a counter per value of one label between two scrapes."""
    out = {}
    for (n, labels), v in after.items():
        if n == name:
            key = dict(labels).get(label, "")
            out[key] = out.get(key, 0) + v - before.get((n, labels), 0)
    return out


def stage_means_ms(after: dict, before: dict) -> dict[str, float]:
    sums = delta(after, before, "vizwiz_stage_seconds_sum", "stage")
    counts = delta(after, before, "vizwiz_stage_seconds_count", "stage")
    return {s: round(1000 * sums[s] / counts[s], 1) for s in sums if counts.get(s)}


def read_locust_csv(prefix: str) -> dict:
    with open(f"{prefix}_stats.csv") as f:
        row = next(r for r in csv.DictReader(f) if r["Name"] == "assist")
    n = int(row["Request Count"])
    return {
        "requests": n,
        "failures": int(row["Failure Count"]),
        "rps": round(float(row["Requests/s"]), 3),
        "mean_ms": round(float(row["Average Response Time"]), 1),
        **{f"p{p.rstrip('%')}_ms": float(row[p]) for p in PCTS if row.get(p) not in (None, "N/A")},
    }


class KvSampler(threading.Thread):
    """Peak KV-cache use on the model server (vLLM gauge) during a run."""

    NAMES = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")

    def __init__(self, url: str | None):
        super().__init__(daemon=True)
        self.url, self.peak, self.stop = url, None, threading.Event()

    def run(self):
        while self.url and not self.stop.wait(1.0):
            try:
                m = scrape(self.url)
            except Exception:
                continue
            vals = [v for (n, _), v in m.items() if n in self.NAMES]
            if vals:
                self.peak = max(self.peak or 0.0, max(vals))


def locust(bin_: str, host: str, users: int, seconds: int, prefix: str, env: dict) -> None:
    cmd = [bin_, "-f", str(LOCUSTFILE), "--headless", "-u", str(users), "-r", str(users),
           "-t", f"{seconds}s", "--host", host, "--csv", prefix, "--only-summary",
           "--stop-timeout", "30", "--loglevel", "WARNING"]  # fmt: skip
    subprocess.run(cmd, env=env, check=False, capture_output=True, text=True, timeout=seconds + 120)


def run_level(args, users: int) -> dict:
    env = {**os.environ, "VIZWIZ_BENCH_SAMPLES": str(args.samples)}
    with tempfile.TemporaryDirectory() as tmp:
        locust(args.locust, args.gateway, users, args.warmup, f"{tmp}/warm", env)
        before, sampler = scrape(args.gateway), KvSampler(args.model_server)
        sampler.start()
        locust(args.locust, args.gateway, users, args.seconds, f"{tmp}/run", env)
        sampler.stop.set()
        time.sleep(1)
        after = scrape(args.gateway)
        stats = read_locust_csv(f"{tmp}/run")
    outcomes = {
        k: int(v) for k, v in delta(after, before, "vizwiz_requests_total", "status").items() if v
    }
    kv = None if sampler.peak is None else round(100 * sampler.peak, 1)
    return {"label": args.label, "users": users, **stats, "outcomes": outcomes,
            "stage_mean_ms": stage_means_ms(after, before), "kv_cache_peak_pct": kv}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Locust concurrency sweep against the gateway")
    p.add_argument("--label", required=True)
    p.add_argument("--gateway", default="http://127.0.0.1:8090")
    p.add_argument("--model-server", help="vLLM URL, for its KV-cache gauge")
    p.add_argument("--locust", default="locust")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--users", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--seconds", type=int, default=60)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for users in args.users:
        row = run_level(args, users)
        with args.out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"SWEEP {args.label} users={users}: rps={row['rps']} p50={row.get('p50_ms')} "
              f"p95={row.get('p95_ms')} p99={row.get('p99_ms')} failures={row['failures']} "
              f"kv_peak={row['kv_cache_peak_pct']}%", flush=True)  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
