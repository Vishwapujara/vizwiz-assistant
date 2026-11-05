# Decisions and Roadblocks Log

Every project decision and every roadblock is recorded here, newest at the bottom of
each section. Entries are never deleted. If a decision is reversed, add a new entry
that supersedes the old one and mark the old one `Superseded by D-xxx`.

**How to add an entry**
- Decisions get the next `D-xxx` id. Roadblocks get the next `R-xxx` id.
- Always record *why*, the alternatives considered, and the plan section affected.
- A roadblock stays `Open` until it is resolved; the resolution usually creates a
  new decision entry, linked from the roadblock.

---

## Decisions

### D-001: Build the VizWiz visual assistant as the fine-tuning project
- **Date:** 2026-11-05
- **Status:** Accepted
- **Decision:** Build the VizWiz assistant (quality check → answerability → answer),
  following `PROJECT_PLAN.md`.
- **Why:** Real-world problem with real human-labeled data; strong story for both SDE
  and ML Engineer roles; fills the resume gap of hands-on model training and serving.
- **Alternatives considered:** Text-to-SQL (Spider/BIRD), contract clause extraction
  (CUAD), accented-speech Whisper fine-tuning (Common Voice), log anomaly detection
  (Loghub), receipt extraction (CORD/SROIE).
- **Plan sections:** all.

### D-002: Target both SDE and ML Engineer roles; middle scope
- **Date:** 2026-11-05
- **Status:** Accepted
- **Decision:** Keep 5 core experiments (E1 baseline, E2 fine-tuning, E3 LoRA vs QLoRA,
  E4 serving quantization, E5 cascade vs multi-task) plus optional E6 (resolution).
  Add required error analysis and a results write-up.
- **Why:** ML roles need baselines, gains, trade-offs, and error analysis; SDE roles
  need a working, tested, benchmarked serving system. Anything beyond this adds time
  with little hiring signal.
- **Alternatives considered:** Original 9-experiment plan (too long, needs large GPUs);
  systems-only plan (too thin for ML roles).
- **Plan sections:** 1.4, 9.3, 10, 14.

### D-003: Cut full fine-tuning, LoRA rank ablation, and 2B vs 4B comparison
- **Date:** 2026-11-05
- **Status:** Accepted
- **Decision:** Remove these experiments. Qwen3-VL-2B is kept only as a fallback model.
- **Why:** Full fine-tuning needs more than an L4; rank and size sweeps are low signal
  for hiring relative to their compute cost.
- **Plan sections:** 8.2, 10.

### D-004: Hardware is Colab T4 + L4 only; all GPU work runs in Colab
- **Date:** 2026-11-05
- **Status:** Accepted
- **Decision:** L4 (24 GB) for all real training, evaluation, and serving. T4 only for
  smoke tests (QLoRA/fp16 mode). No notebooks run locally. Every config must fit on L4.
- **Why:** This is the hardware the owner has. T4 lacks bf16, FlashAttention-2, and
  the compute capability many vLLM quantized kernels need.
- **Fallback order if 4B does not fit on L4:** lower max image pixels → QLoRA → 2B model.
- **Plan sections:** 0, 2, 8.1, 16.

### D-005: Code in the repo, Colab notebooks are thin launchers
- **Date:** 2026-11-05
- **Status:** Accepted
- **Decision:** All logic lives in the `vizwiz` Python package and scripts. Claude Code
  writes code and notebooks locally; the owner runs notebooks in Colab, following
  step-by-step instructions from Claude Code, and pastes back outputs and errors.
- **Why:** Claude Code cannot operate Colab. Keeping logic in the repo makes it
  testable, reviewable, and reproducible.
- **Plan sections:** 0, 2.3, 4.

### D-006: No Docker in Colab; Docker Compose for local + CI with a mock model server
- **Date:** 2026-11-05
- **Status:** Accepted
- **Decision:** In Colab, vLLM, the gateway, and Locust run as background processes on
  one L4 VM. Locally and in CI, Docker Compose runs the gateway against a mock
  OpenAI-compatible model server, plus Prometheus, Grafana, and the dashboard.
- **Why:** Colab does not support Docker. A mock server lets the whole stack be tested
  without a GPU.
- **Plan sections:** 2.2, 12.

### D-007: Record all decisions and roadblocks in this file
- **Date:** 2026-11-05
- **Status:** Accepted
- **Decision:** Every decision (including choices made during implementation, such as
  library versions or thresholds) and every roadblock is logged here.
- **Why:** Keeps a reasoning trail for interviews and the final write-up, and stops
  the same problem from being re-investigated.
- **Plan sections:** 0.

### D-008: Host the repo as a public GitHub repository
- **Date:** 2026-11-05
- **Status:** Accepted
- **Decision:** The code lives in a public GitHub repo, which Colab clones directly.
- **Why:** No token setup in Colab, and the repo is visible to recruiters. Safe because
  data, images, weights, and secrets are never committed (enforced by `.gitignore`).
- **Alternatives considered:** Private repo + GitHub token in Colab Secrets.
- **Plan sections:** 0, 2.2.

### D-009: Packaging and dependency layout (Step 0.1)
- **Date:** 2026-11-05
- **Status:** Accepted
- **Decision:** `src/` layout, hatchling build backend, Python `>=3.11` (local venv on
  3.11, CI on 3.11). Core dependencies are only `pydantic` and `pyyaml`; heavy libraries
  are split into extras: `data`, `train`, `serve`, `bench`, `dev`. GPU library versions
  are **not pinned yet**; they get pinned after the Step 0.3 compatibility check on L4.
- **Why:** CI and local dev stay fast because they never install torch/vLLM. Colab
  installs only the extras a notebook needs. Pinning before checking what works with
  Qwen3-VL on L4 would mean guessing.
- **Alternatives considered:** one flat requirements.txt (slow CI, installs GPU libs
  everywhere); Poetry/uv (extra tooling to install in Colab for little benefit).
- **Plan sections:** 3, 4.

### D-010: Lint and test tooling
- **Date:** 2026-11-05
- **Status:** Accepted
- **Decision:** `ruff` for lint (rules E, F, I, B, UP) and formatting, line length 100;
  `pytest`. CI runs `ruff check`, `ruff format --check`, and `pytest` on every push to
  `main` and on pull requests.
- **Plan sections:** 3, 13.

### D-011: No license chosen yet
- **Date:** 2026-11-05
- **Status:** Accepted (revisit before sharing the repo widely)
- **Decision:** The repo is published without a LICENSE file for now.
- **Why:** The license is the owner's choice. Without one, the code is visible but
  others have no right to reuse it, which is the safe default.
- **Plan sections:** none.

### D-012: Colab session setup and Drive layout (Step 0.2)
- **Date:** 2026-11-05
- **Status:** Accepted
- **Decision:** Every Colab session starts with the same setup cells: mount Drive,
  clone or hard-reset the repo to `origin/main` in `/content/vizwiz-assistant`
  (local disk), `pip install -e`, then run `python -m vizwiz.env`. Persistent files
  live under `MyDrive/vizwiz/{hf_cache,datasets,checkpoints,results,logs}`. Each
  environment report is saved as JSON in `logs/`.
- **Why:** Code on local disk is fast and always matches GitHub (the Colab copy is
  throwaway, never edited). Running checks as `python -m` in a new process avoids
  the stale-import problem after an editable install in a running kernel. Drive
  holds what must survive a disconnect.
- **Alternatives considered:** keeping the repo on Drive (slow, drifts from GitHub);
  pip cache on Drive (dropped: reads over the Drive mount are slower than re-downloading).
- **Plan sections:** 2.2, 4.

### D-013: Colab baseline environment; CI tests Python 3.11 and 3.13
- **Date:** 2026-11-05
- **Status:** Accepted
- **Observed on Colab T4 (`results/env/colab_t4.json`):** Python 3.13.16, torch
  2.11.0+cu130 (CUDA 13.0) preinstalled, Tesla T4 14.6 GB, compute capability 7.5,
  no bf16 / FlashAttention-2 / FP8 (matches plan assumptions), 51 GB RAM (High RAM),
  ~235 GB local disk, ~224 GB free on Drive.
- **Decision:** CI runs on Python 3.11 (minimum supported) and 3.13 (Colab's version).
  Environment reports from each runtime are committed under `results/env/`.
- **Why:** Code must be tested on the Python version it actually runs on. The reports
  are the reference for "what Colab gave us" when debugging version problems later.
- **Watch-out for Step 0.3:** vLLM wheels are built against a specific torch/CUDA
  version, so installing vLLM may replace Colab's preinstalled torch 2.11/CUDA 13.
- **Plan sections:** 2.1, 3, 13.

### D-014: Step 0.2 verified on T4 and L4 only; CPU run skipped
- **Date:** 2026-11-05
- **Status:** Accepted
- **Decision:** The setup notebook is verified on T4 and L4. The CPU runtime is not
  run separately; it gets exercised in Step 1.2 (data download on CPU).
- **Why:** The notebook has no GPU-specific code paths, so a CPU run adds no
  information and costs a session.
- **Plan sections:** 14 (Step 0.2).

### D-015: Step 0.2 complete; L4 memory budget is 22 GiB, not 24
- **Date:** 2026-11-05
- **Status:** Accepted
- **Observed on Colab L4 (`results/env/colab_l4.json`):** NVIDIA L4, 22.0 GiB usable,
  compute capability 8.9, bf16 / FlashAttention-2 / FP8 all available, 53 GB RAM.
  Same Python 3.13.16 and torch 2.11.0+cu130 as the T4 runtime, so one dependency
  set serves both.
- **Decision:** All memory planning (training batch size, image resolution, vLLM
  `gpu_memory_utilization`) uses 22 GiB as the L4 budget. Step 0.2 is done
  (verified on T4 and L4, see D-014).
- **Why:** "24 GB" is the marketed decimal size; the usable figure reported by the
  driver is what actually limits us.
- **Plan sections:** 2.1, 8.1, 12.

---

## Open decisions

Resolved items move to the Decisions section with a `D-xxx` id.

| Item | Resolve at | Notes |
|---|---|---|
| Hugging Face model cache on Drive vs local disk | Step 0.3 | Drive survives disconnects but may load slower than re-downloading; measure load time on L4 |
| Experiment tracker: W&B vs MLflow | Step 1.1 | W&B is simpler from Colab |
| Image resolution setting | Step 3.1 | From L4 memory/latency measurements |
| Quality-label threshold `k` | Step 1.3 | From audit data |
| 4-bit serving format: AWQ vs GPTQ | Step 4.1 | Whichever vLLM supports for Qwen3-VL on L4 |
| Cascade "unusable image" rule | Step 5.2 | Tuned on val |
| Final serving quantization format | After E4 | |

---

## Roadblocks

### R-001: `.gitignore` pattern hid our own `vizwiz.data` package
- **Date hit:** 2026-11-05
- **Step:** 0.1
- **Environment:** local, git
- **What happened:** `git status` showed `src/vizwiz/data/__init__.py` was not staged.
- **Root cause:** the pattern `data/` matches a `data` directory at any depth, so it
  also matched `src/vizwiz/data/`.
- **Options considered:** rename the package; anchor the pattern to the repo root.
- **Resolution:** anchored dataset/artifact patterns to the root (`/data/`,
  `/datasets/`, `/checkpoints/`, `/outputs/`, `/runs/`, `/predictions/`).
- **Status:** Resolved


<!--
Template:

### R-001: <short title>
- **Date hit:** YYYY-MM-DD
- **Step:** <plan step, e.g. 3.1>
- **Environment:** <Colab T4 / Colab L4 / local / CI>, <library versions if relevant>
- **What happened:** <symptom, exact error message>
- **Root cause:** <once known>
- **Options considered:** <list>
- **Resolution:** <what was done> → see D-xxx
- **Status:** Open | Resolved | Worked around
-->
