"""Locust user for the gateway load test (Step 6.1, D-065).

Closed loop: each simulated user sends its next request as soon as the previous
reply arrives, so the number of users is the number of requests in flight.
Photos and questions come from ``VIZWIZ_BENCH_SAMPLES`` (a JSON list of
``{"path", "question"}`` written by the notebook from val records); the photos are
read once into memory, so disk speed does not enter the timings.

    locust -f locustfile.py --headless -u 8 -r 8 -t 60s --host http://127.0.0.1:8090 --csv out
"""

from __future__ import annotations

import json
import os
import random

from locust import HttpUser, constant, task

_SAMPLES: list[tuple[bytes, str]] = []


def samples() -> list[tuple[bytes, str]]:
    if not _SAMPLES:
        for s in json.load(open(os.environ["VIZWIZ_BENCH_SAMPLES"])):
            with open(s["path"], "rb") as f:
                _SAMPLES.append((f.read(), s["question"]))
    return _SAMPLES


class AssistUser(HttpUser):
    wait_time = constant(0)

    def on_start(self):
        self.rng = random.Random()
        self.samples = samples()

    @task
    def assist(self):
        photo, question = self.rng.choice(self.samples)
        with self.client.post("/v1/assist", files={"image": ("p.jpg", photo, "image/jpeg")},
                              data={"question": question}, name="assist",
                              catch_response=True, timeout=60) as r:  # fmt: skip
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
