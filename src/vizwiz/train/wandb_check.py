"""W&B login check (Step 3.1): proves the API key from Colab Secrets works.

    WANDB_API_KEY=... python -m vizwiz.train.wandb_check

Creates a tiny run in the ``vizwiz-assistant`` project, logs one value, and prints the
run URL. The key is read from the environment by the wandb library and never printed.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY is not set. Add it in Colab: key icon → Secrets → "
              "WANDB_API_KEY, with notebook access on.")  # fmt: skip
        return 1
    import wandb

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "vizwiz-assistant"),
        name="wandb-check",
        job_type="check",
        config={"purpose": "login check, Step 3.1"},
    )
    run.log({"check": 1})
    url, entity = run.url, run.entity
    run.finish()
    print(f"W&B OK: entity {entity}, run {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
