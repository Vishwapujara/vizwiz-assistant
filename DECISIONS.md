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

### D-016: Pinned, isolated GPU environment: vLLM 0.32.0 + transformers 5.19.0
- **Date:** 2026-11-05
- **Status:** Accepted (confirmed by the Step 0.3 rerun on 2026-11-06, see D-021)
- **Decision:** GPU work runs in a separate venv at `/content/venv`, built with `uv`
  from `requirements/colab-gpu.txt` every session: `vllm==0.32.0`,
  `transformers==5.19.0`, plus peft/accelerate/pillow. vLLM pulls in `torch==2.13.0`
  and `torchvision==0.28.0` (CUDA 13). Notebooks call `/content/venv/bin/python -m ...`.
  The full resolved set is saved as `freeze_*.txt` for each run.
- **Why:**
  - Colab's preinstalled torch is 2.11 (D-013) and vLLM 0.32.0 requires exactly
    torch 2.13.0, so vLLM cannot share Colab's kernel environment without replacing
    torch mid-session.
  - One environment for training *and* serving means the same transformers version
    (same chat template and image processor) is used in both, so there is no
    train/serve mismatch.
  - Colab updates its preinstalled packages without notice; a pinned venv makes runs
    reproducible regardless.
  - vLLM 0.32.0 (2026-10-18) instead of 0.33.0 (2026-11-02, 3 days old at the time):
    the newer release has had no time to receive bug-fix releases.
  - transformers 5.19.0 is the newest release inside vLLM 0.32.0's allowed range
    (`>=5.16.1,<5.20`).
- **Alternatives considered:** let `pip install vllm` replace Colab's torch in the
  kernel (fragile, breaks other preinstalled packages); separate venvs for training
  and serving (two transformers versions, risk of mismatch).
- **Sources:** PyPI metadata for vllm 0.32.0 / 0.33.0 and transformers; vLLM supported
  models page (Qwen3-VL: LoRA supported); Qwen3-VL-4B-Instruct model card (Apache-2.0).
- **Plan sections:** 3, 7.

### D-017: Validate vLLM LoRA support with a blank adapter
- **Date:** 2026-11-05
- **Status:** Accepted
- **Decision:** Step 0.3 builds a rank-16 LoRA adapter with PEFT on the language-model
  projections (`q/k/v/o/gate/up/down_proj`) without training it. `lora_B` starts at
  zero, so the adapter's output must exactly equal the base model's. vLLM serves it,
  and the check passes if the base and adapter answers are identical under greedy
  decoding.
- **Why:** This proves the whole adapter path (PEFT save format → vLLM load → apply at
  request time) works for Qwen3-VL before any training compute is spent. The check
  also confirms the target names hit no vision-tower modules (vision encoder frozen,
  plan Section 7).
- **Plan sections:** 7, 12.

### D-018: Keep the Hugging Face model cache on local disk, not Drive
- **Date:** 2026-11-06
- **Status:** Accepted
- **Observed (Step 0.3, L4, `results/env/compat_hf_l4.json`):** Qwen3-VL-4B-Instruct
  downloads from the Hub in 26.4 s and loads to the GPU in 5.6 s from local disk.
- **Decision:** `HF_HOME=/content/hf_cache` (local disk) in every notebook. Drive keeps
  only datasets, checkpoints, adapters, results and logs. The `hf_cache` folder on
  Drive is unused.
- **Why:** A fresh download costs under 30 s per session, which beats both the risk
  and the speed of reading ~9 GB through the Drive mount, and it saves Drive space.
- **Plan sections:** 2.2.

### D-019: transformers path verified on L4 (Step 0.3, part 1)
- **Date:** 2026-11-06
- **Status:** Accepted
- **Observed:** with torch 2.13.0 / transformers 5.19.0 / peft 0.21.2, the model loads
  as `Qwen3VLForConditionalGeneration` in bf16 with SDPA attention; peak GPU memory
  8.39 GiB for one image + question (425 prompt tokens); generation 1.84 s; the test
  label was read correctly ("12/2026"). The blank LoRA (r=16, q/k/v/o/gate/up/down)
  targets 252 modules, 0 in the vision tower; 33.0M trainable parameters (0.74%).
- **Decision:** the training stack versions above are confirmed. The serving half of
  D-016 waits on R-002.
- **Plan sections:** 7, 8.

### D-020: vLLM subprocess environment: venv bin on PATH, FlashInfer sampler off
- **Date:** 2026-11-06
- **Status:** Accepted (confirmed by the R-002 rerun)
- **Decision:** Every vLLM server we start gets `PATH=<venv>/bin:$PATH` and
  `VLLM_USE_FLASHINFER_SAMPLER=0`.
- **Why:** The PATH fix lets any tool the venv installs be found (root cause of R-002).
  Disabling the FlashInfer sampler avoids a CUDA JIT build at every session start:
  Colab wipes the compile cache each session, and the build depends on Colab's CUDA
  toolkit matching torch's CUDA 13. We decode greedily (temperature 0), so vLLM's
  PyTorch-native sampler gives the same outputs with no measurable cost.
- **Alternatives considered:** keep the FlashInfer sampler and cache its build on Drive
  (more moving parts for a component greedy decoding barely uses).
- **Plan sections:** 12.

### D-021: Step 0.3 passed; full environment locked
- **Date:** 2026-11-06
- **Status:** Accepted
- **Observed (Colab L4, rerun):**
  - transformers: model load 4.5 s (cached), generation 1.1 s, label read correctly
    ("12/2026"), peak GPU memory 8.39 GiB (`results/env/compat_hf_l4.json`).
  - vLLM 0.32.0: server healthy after 205 s (most of it torch.compile and CUDA graph
    capture, ~90 s together in the first log); served models: base + `blank` adapter.
    Base and blank-adapter answers identical ("12/2026", 8 tokens). Latency 0.53 s base,
    1.30 s adapter; the adapter number includes its first-use load, so it is not a
    benchmark (`results/env/compat_vllm_l4.json`).
  - The package set resolved identically on both runs (`results/env/freeze_l4.txt`).
- **Decision:** Pin `peft==0.21.2`, `accelerate==1.15.0`, `pillow==13.0.0` in
  `requirements/colab-gpu.txt`, and commit the full resolved set as a constraints file,
  `requirements/colab-gpu.lock.txt` (~200 packages). Every notebook installs with
  `-r colab-gpu.txt -c colab-gpu.lock.txt`.
- **Why:** Top-level pins alone still let transitive dependencies drift between
  sessions. The lock reproduces exactly the set that was tested. It only changes
  after a new compatibility run passes.
- **Known per-session cost:** ~3.5 min of vLLM startup (compile + graph capture),
  because Colab wipes `~/.cache/vllm`. Possible later optimisation: point the vLLM
  compile cache at Drive. Not worth it until Phase 5.
- **Plan sections:** 3, 12, 14 (Step 0.3 done).

### D-022: Split strategy: keep test = official val; add official VQA test as a secondary test
- **Date:** 2026-11-06
- **Status:** Accepted
- **New fact (Step 1.1):** the VQA test answers are now public (`VQA_test.json`), but
  the quality test labels are not.
- **Decision:**
  - Primary test = official val (4,319 VQA questions / 7,750 quality images). It is
    the only split labelled for all three tasks, so the cascade can be scored end to end.
  - Our val = ~10% of official train, image-grouped and stratified (unchanged).
  - Secondary test = official VQA test (8,000), touched only for final answering and
    answerability numbers, to compare with published VizWiz results.
- **Why:** A single labelled test set for every task keeps the cascade evaluation
  honest. The extra VQA test set costs nothing and gives numbers people recognise.
- **Alternatives considered:** test = official test for VQA and official val for
  quality. Rejected because then the VQA val and quality test would share images, and
  the cascade could not be scored on one set.
- **Plan sections:** 5.1, 5.2.

### D-023: Quality labels come from the challenge JSON files, not the CSV
- **Date:** 2026-11-06
- **Status:** Accepted
- **Decision:** Use `image_quality/annotations.zip` (per-image counts from 5
  annotators). Do not download `VizWiz_quality_issues_train_val_test.csv`.
- **Why:** The JSON files use the same image names as VQA (exact overlap: 20,000 of the
  20,523 train images, and all val and test images). The CSV uses an older naming
  scheme, a different split assignment, and contains crowd-worker IDs we have no need
  to hold.
- **Plan sections:** 5.1, 5.3.

### D-024: License position
- **Date:** 2026-11-06
- **Status:** Accepted
- **Decision:** VizWiz images and VQA annotations are CC BY 4.0. The quality page gives
  no license, so the quality labels are treated under the same terms (research and
  portfolio use, with attribution, no redistribution). The README carries the
  attribution from `docs/data_licenses.md`.
- **Plan sections:** 11.

### D-025: Experiment tracker: Weights & Biases
- **Date:** 2026-11-06
- **Status:** Accepted
- **Decision:** Log all training and evaluation runs to W&B (free personal account).
  Colab authenticates with `WANDB_API_KEY` stored in Colab Secrets, never in the repo
  or notebook. Each run logs config, git commit, dataset manifest hashes, GPU type,
  peak memory, metrics and the loss curve. Summary metrics JSON is still committed
  under `results/`, so the repo stays the source of truth if W&B is unavailable.
- **Why:** It is the most widely used tracker for deep learning / LLM fine-tuning; the
  Hugging Face Trainer and TRL log to it with `report_to="wandb"`. Runs are stored
  online as they happen, so a Colab disconnect loses nothing. Dashboards can be shared
  publicly for the portfolio.
- **Alternatives considered:** MLflow (common in enterprise / Databricks; needs a
  self-hosted UI from Colab and is harder to share); TensorBoard (no hosted run history).
- **Plan sections:** 3, 8.3.

### D-026: W&B for everything: tracking and the adapter registry
- **Date:** 2026-11-06
- **Status:** Accepted
- **Decision:** W&B is the only MLOps tool. Training and evaluation runs go to W&B
  experiment tracking (D-025). The adapter registry is W&B Registry: each trained
  adapter is logged as a versioned artifact with metadata (task, base model, rank,
  training run, metrics), and the `production` alias marks the live version per task.
  Promotion and rollback mean moving that alias. This replaces the SQLite registry
  in the original plan.
- **Consequences:**
  - The gateway needs internet and `WANDB_API_KEY` at startup to resolve adapters.
    That holds on Colab and on the laptop.
  - Code talks to a small `Registry` interface. Tests and GitHub Actions use an
    in-memory fake, so CI needs no network and no secrets.
- **Why:** One tool instead of two means less setup and one place to look; runs and
  adapters are linked automatically (each adapter version points to the run that
  produced it).
- **Alternatives considered:** MLflow Model Registry (self-hostable, so no network
  dependency at serving time, but a second tool to run); custom SQLite registry
  (original plan; no UI or lineage).
- **Plan sections:** 3, 4, 12.1, 14 (Step 5.3).

### D-027: Download design (Step 1.2)
- **Date:** 2026-11-06
- **Status:** Accepted
- **Decision:** `vizwiz.data.download` downloads straight to
  `MyDrive/vizwiz/datasets/raw/` on a CPU runtime, using only the standard library.
  - Partial files are kept as `*.part` and resumed with HTTP Range requests (the server
    answers `Range` with 206; checked 2026-11-06).
  - Every file is checked against the server's `Content-Length` and hashed with SHA-256
    while streaming. Hashes go to `checksums.json` after each file, and the committed
    copy lives at `results/data/checksums.json`.
  - Image zips are checked by counting `.jpg` entries (23,431 / 7,750 / 8,000).
  - Order: annotations first (small, needed soonest), then val, test, train.
  - The notebook ends with `drive.flush_and_unmount()` so writes reach Drive before
    the runtime is deleted.
- **Why:** The VizWiz server publishes no checksums, so the hashes from our first
  download become the provenance record: any later copy can be verified against them.
  Resume means a disconnect during the 11.3 GB train zip costs minutes, not the
  whole download. A CPU runtime avoids spending GPU credits on network I/O.
- **Alternatives considered:** download to local disk and then copy to Drive (a
  disconnect loses everything in flight); Hugging Face mirrors (unofficial, D-023 / Step 1.1).
- **Plan sections:** 5, 14 (Step 1.2).

### D-028: Images stay zipped on Drive and are unzipped to local disk each session
- **Date:** 2026-11-06
- **Status:** Accepted
- **Observed (Step 1.2):**
  - Download of all 7 files (18.96 GB) took about 21 minutes (val 208 s, test 387 s,
    train 651 s; 10–18 MB/s). All sizes matched the server, and every zip holds the
    expected images (after R-004).
  - Unzipping val (3.49 GB, 7,750 images) from Drive to local disk took 45.9 s, about
    76 MB/s (`results/data/extract_timings_val.json`).
- **Decision:** The zips on Drive are the single copy. Each GPU session unzips only the
  splits it needs into `/content/data/images`: about 2.5 min for train, ~4 min for all
  three (extrapolated from val). Preprocessed shards are deferred until a step actually
  needs them (Step 1.5 may store resized images if image loading turns out to be a
  training bottleneck).
- **Why:** ~4 minutes per session is cheap next to training runs measured in hours. It
  avoids reading ~40k small files through the Drive mount, avoids a second ~19 GB copy
  on Drive, and keeps the original, checksummed files as the only source.
- **Checksums:** `results/data/checksums.json` is the provenance record (the VizWiz
  server publishes none). `vizwiz.data.download verify` re-checks any copy against it.
- **Plan sections:** 2.2, 5, 14 (Step 1.2 done).

### D-029: Port the official VizWiz metric; verified for exact parity
- **Date:** 2026-11-06
- **Status:** Accepted
- **Decision:** `vizwiz.eval.vqa_metric` re-implements the official `vqaEval.py`
  normalisation and accuracy (leave one annotator out, `min(1, matches/3)`, mean of 10),
  quirks included: only predictions are normalised; `none` → `0`; the official period
  regex is kept as written.
- **Why:** The official code has no license, so it is ported rather than copied, and it
  pulls in caption-metric dependencies we do not need. Keeping the quirks makes our
  numbers comparable with published VizWiz results.
- **Verification:** 0 mismatches against the official functions on 151,181
  normalisation probes (every unique train/val answer, plus upper-case, trailing-period
  and article variants) and on 3,000 accuracy checks. A one-off parity script was run
  locally; the official code is not vendored.
- **Plan sections:** 5.4, 9.1 (most of Step 2.1's metric work is done early).

### D-030: Quality issue threshold k = 2 (confirmed by the audit)
- **Date:** 2026-11-06
- **Status:** Accepted
- **Decision:** A quality issue is positive when at least 2 of 5 annotators flagged it.
- **Why:** k = 1 calls 92% of images flawed (single-annotator noise). k = 3 shrinks
  obstruction, too-bright and too-dark to 1.7–2.8% of images, too few to learn well.
  k = 2 keeps every main class at ≥ 3.6% and still needs two people to agree.
  `OTH` (0.8% at k = 2) is a candidate to drop as a class (Step 1.4).
- **Plan sections:** 5.3, 17.

### D-031: Answerability is 2-way (answerable / unanswerable)
- **Date:** 2026-11-06
- **Status:** Accepted (owner's choice)
- **Decision:** The answerability adapter predicts `answerable` or `unanswerable`,
  straight from the official flag. Unusable photos are caught earlier in the cascade
  by the quality adapter (`unrecognizable` → retake guidance).
- **Why:** Every label stays an original human annotation. A third class would have
  to be stitched together from two datasets (about 8% of questions) and would
  duplicate the quality adapter's job.
- **Alternatives considered:** 3-way with `unsuitable_image` built from
  `unrecognizable` votes (R-005 option b).
- **Plan sections:** 5.3, 6.2, 6.5, 9.1.

### D-032: Drop the `OTH` quality class
- **Date:** 2026-11-06
- **Status:** Accepted (owner's choice)
- **Decision:** Quality labels are `blur, framing, rotation, dark, bright, obstruction,
  unrecognizable`, or `none`. `OTH` flags are ignored.
- **Why:** `OTH` is 0.8% of images at k = 2, too rare to learn, and "other" has no
  actionable retake advice.
- **Plan sections:** 5.3, 6.3.

### D-033: Answer target rule; answering adapter trains on answerable questions only
- **Date:** 2026-11-06
- **Status:** Accepted
- **Decision:** The target is the distinct human answer with the highest official
  accuracy when output verbatim; ties go to more votes, then the earliest-listed answer.
  For answerable questions, `unanswerable` is excluded as a target. The answering
  adapter trains only on answerable questions; in the cascade, a decline is scored as
  the answer `unanswerable`.
- **Why:** The metric normalises the prediction and compares it with raw answers, so
  "most frequent string" can pick an answer that scores 0 (audit finding 8). Choosing
  by the metric itself avoids that. Excluding `unanswerable` keeps the answering
  adapter from learning to decline, which would blur the two adapters' jobs.
- **Effect (Step 1.4 build):** the target differs from the most common answer for
  2,106 of 13,492 answerable train questions (15.6%): 1,856 where `unanswerable` was the
  most common answer, 246 ties for most common, and 4 where the metric picked a
  different answer. Mean target accuracy on answerable questions:
  87.3% (train), 84.9% (test).
- **Plan sections:** 5.3, 6.1.

### D-034: What the split build commits, and what stays on Drive
- **Date:** 2026-11-06
- **Status:** Accepted
- **Decision:** `python -m vizwiz.data.splits` writes:
  - record files `vqa_{split}.jsonl` and `quality_{split}.jsonl` with questions,
    answers and labels → Drive only (`datasets/processed/`), never git;
  - `results/splits/membership.json`: image names per split (IDs only) → committed;
  - `results/splits/files.json`: SHA-256 of each record file → committed;
  - `results/splits/label_stats.{json,md}`: aggregate counts → committed.
  The build is deterministic (sorted records, fixed seed per stratum, canonical JSON).
  The Colab run rebuilds with `--expect results/splits/files.json` and must print
  `REBUILD MATCHES COMMITTED HASHES`.
- **Why:** Anyone can confirm they have exactly our splits without the repo holding
  dataset text. Two local builds already produced identical hashes.
- **Splits (seed 1234):** train 21,559 images / 18,471 questions; val 2,395 / 2,052
  (stratified, 26.9% unanswerable, the same as train); test 7,750 / 4,319;
  test_vqa 8,000 / 8,000.
- **Plan sections:** 5.2, 13.

### D-035: One image-preprocessing function for training, evaluation and serving
- **Date:** 2026-11-06
- **Status:** Accepted
- **Decision:** `vizwiz.serve.preprocess.prepare_image(source, max_pixels)`:
  1. apply the EXIF orientation explicitly;
  2. convert to RGB;
  3. downscale (never upscale) to area ≤ `max_pixels`, aspect kept, both sides
     floored to multiples of 32.
  The `max_pixels` budget is the image-resolution knob; candidate budgets 262,144 /
  524,288 / 1,048,576 / native are measured in Step 1.5, and the choice is made in
  Step 3.1 from L4 memory and speed.
- **Why:** Qwen3-VL makes one visual token per 32×32 block (16-px patches, 2×2 merge),
  so rounding to 32 makes token count = (w/32)·(h/32): predictable memory and no
  second resize inside the processor. Phone photos often store rotation as EXIF
  metadata, and whether a given processor or vLLM applies it is not guaranteed; doing
  it ourselves removes a silent train/serve mismatch.
- **Plan sections:** 6, 7, 12.1.

### D-036: Prompts, targets and loss masking
- **Date:** 2026-11-06
- **Status:** Accepted
- **Decision:** `vizwiz.data.formatting`: a system prompt per task; user turn = image
  (+ question, except for quality); assistant turn = target (answer string /
  `answerable`|`unanswerable` / JSON list of quality labels). Labels are -100 for every
  prompt token; the target and the end-of-turn token are trained. The prompt is
  encoded separately with `add_generation_prompt=True`, and its ids must be an exact
  prefix of the full conversation's ids. This is checked for every example
  (`prefix_ok`).
- **Why:** Training on the end-of-turn token teaches the model to stop after a short
  answer. The prefix check catches chat templates that add hidden tokens to assistant
  turns (for example empty "thinking" blocks), which would otherwise shift the mask
  silently.
- **Plan sections:** 6.

### D-037: Rest of Phase 1 runs as one CPU notebook (`02_data_prep.ipynb`)
- **Date:** 2026-11-06
- **Status:** Accepted (owner's request)
- **Decision:** The image-dependent parts of Steps 1.3–1.5 (image statistics, image
  hashes and duplicate check, label-file rebuild check, token-length statistics) run
  in one Colab CPU session instead of one per step. The pinned GPU venv is used even
  on CPU, so token counts come from exactly the transformers version used in training.
  The planned T4 "dataloader smoke test" moves to Step 3.1, which runs on a GPU anyway.
- **Plan sections:** 14 (Steps 1.3–1.5).

### D-038: Keep copies of test photos out of training; carve val by file hash
- **Date:** 2026-11-06
- **Status:** Accepted
- **Context:** the image scan (R-006) found byte-identical photos under different
  names, including across the official splits.
- **Decision:**
  - Commit `results/splits/image_hashes.json` (image name → SHA-256 for all 39,704
    images; no image content). The split builder reads it.
  - Any train-pool image whose bytes also appear in test or test_vqa is excluded from
    training and listed in `results/splits/excluded.json`: 605 images.
  - val is carved with groups of identical files as the unit, so copies never
    straddle train and val.
  - test and test_vqa stay exactly as published, so our scores stay comparable with
    other VizWiz results. They share 65 identical files with each other; both are
    evaluation-only, so that is allowed and only reported.
  - `check_no_leakage` and the image scan fail if any file in train or val is
    identical to a file in another split.
- **Effect:** train 21,559 → 21,010 images (18,471 → 17,967 questions); val 2,395 →
  2,339 images (2,052 → 2,000 questions). Test splits unchanged. Verified: 0 duplicate
  groups touch train or val and another split.
- **Not handled:** near-duplicates (the same scene re-photographed, re-encoded files).
  Exact hashing cannot find them; noted as a limitation for the write-up.
- **Plan sections:** 14 (Step 1.4).

### D-039: Phase 1 complete
- **Date:** 2026-11-09
- **Status:** Accepted
- **Evidence (Colab CPU run of `02_data_prep.ipynb` at `e77bebb`):**
  `REBUILD MATCHES COMMITTED HASHES` (label files on Drive are byte-identical to the
  local build), `IMAGE HASHES MATCH COMMITTED`, `IMAGE SCAN OK` (0 missing images,
  0 duplicate groups touching train or val), `ALL PREFIX CHECKS PASSED` (2,400
  encodings). The three synthetic demos show that only the target and end-of-turn
  tokens are trained.
- **Committed:** `results/splits/image_stats.json`, `results/data/token_stats.json`;
  interpreted in `results/audit/findings.md` sections 9–10.
- **Carried forward:** image-resolution budget for Step 3.1. Measured cost: about 270
  tokens per example at 262k pixels, 530 at 524k, 1,010 at 1M, 1,970 at native.

### D-040: Metric definitions (Step 2.1)
- **Date:** 2026-11-10
- **Status:** Accepted
- **Decision** (`src/vizwiz/eval/metrics.py`, pure Python):
  - **Answering:** official VQA accuracy (D-029) on answerable questions only, with a
    95% bootstrap CI and slices by answer type, by each quality label, and by
    any-issue vs no-issue. The rate at which the model says "unanswerable" anyway is
    reported as `declined_pct`.
  - **Answerability**, matching the official `evaluate_unanswerability`: the positive
    class is `unanswerable`, and each prediction needs a **score**, P(unanswerable).
    Headline metric is average precision; we also report official F1 at score > 0.5,
    recall at precision 0.8 and 0.9, macro-F1 of the generated labels, a confusion
    matrix, invalid-output rate, and the borderline slice (answerable with 3–5
    "unanswerable" votes).
  - **Quality:** per-label precision, recall and F1; macro-F1 over the 7 issue labels
    (`none` excluded: it is the absence of issues); exact-set match; JSON validity.
    An invalid reply counts as predicting no issues, so bad formatting is penalised
    rather than skipped.
  - **CIs:** percentile bootstrap, 1,000 resamples, fixed seed.
- **Why the score:** average precision ranks questions by confidence, so a yes/no
  label alone cannot be scored the official way. At inference the score will come
  from the probabilities of the first generated token ("un…" vs "answer…"), so
  generating the label and scoring it cost one forward pass (Step 2.2).
- **Verified:** `average_precision` and F1 equal scikit-learn (used by the official
  code) on 2,980 random cases with ties: max difference 2.8e-16. scikit-learn is not
  a project dependency; the check ran in a scratch environment.
- **Plan sections:** 9.1, 14 (Step 2.1).

### D-041: Evaluation runner design (Step 2.2)
- **Date:** 2026-11-10
- **Status:** Accepted
- **Decision** (`src/vizwiz/eval/run_eval.py`):
  - **Offline vLLM** (`LLM.generate`) for batch evaluation, not the HTTP server: no
    port or start-up polling, and the same engine and kernels as serving. The server
    path is exercised in Phase 5.
  - Prompts are rendered with the model's own chat template from the same
    `build_messages` used in training (D-036). Images go through the shared
    `prepare_image` (D-035). Train/eval mismatch is ruled out by construction.
  - Greedy decoding; max new tokens 16 (answer), 4 (answerability), 48 (quality),
    sized from the token stats (answer p95 7, quality max 18).
  - Answerability score = P(unanswerable) from the top-20 logprobs of the first
    generated token, summed over the casings of each word's first token. If neither
    class appears, the generated label decides and the case is counted.
  - dtype from the GPU: bf16 on L4, fp16 on T4 (no bf16).
  - `--limit N` takes a fixed-seed sample per task for quick checks.
  - Image budget for evaluation defaults to 524,288 pixels (~500 visual tokens)
    **provisionally**: the final budget is chosen in Step 3.1, and E1 is rerun at
    that budget if it differs, so baseline and fine-tuned models are always compared
    at the same resolution.
  - Predictions stay on Drive; only `metrics.json` (aggregates, settings, versions,
    git commit, timing) is committed.
- **Tested:** end-to-end in CI with a fake generator (real record files, images in
  split subfolders, real metrics). vLLM on a T4 has not been run before: Step 0.3
  validated vLLM on the L4 only.
- **Plan sections:** 9.4, 14 (Step 2.2).

### D-042: Step 2.2 verified on T4 (100 examples per task)
- **Date:** 2026-11-10
- **Status:** Accepted
- **Evidence** (`results/eval/e1-check-t4/metrics.json`, commit `3b26e49`):
  vLLM 0.32.0 runs Qwen3-VL-4B on a T4 in fp16; engine start-up 418 s including the
  model download; about 2.4 examples/s. All three tasks produce metrics;
  `score_fallbacks` is 0 (the answerability score was always read from logprobs);
  quality replies were 100% valid JSON.
- **First signals** (100 examples, wide CIs; not the baseline):
  - answering 55.1% VQA accuracy (CI 46–64); 67% on photos with no quality issue vs
    50% with any issue, consistent with audit finding 4.
  - answerability AP 71.1, but **41% of replies were neither "answerable" nor
    "unanswerable"**, almost all on answerable questions: the base model does not
    follow the one-word format. AP and the official F1 use the score, so they are
    not affected; the generated-label metrics are.
  - quality macro-F1 23.9: over-predicts blur and dark, never predicts
    unrecognizable.
- **Plan sections:** 14 (Step 2.2).

### D-043: E1 run setup (Step 2.3)
- **Date:** 2026-11-10
- **Status:** Accepted
- **Decision:** E1 runs every test example (official val) for all three tasks on an
  L4, bf16, at the provisional 524,288-pixel budget (D-041). test_vqa (official test,
  D-022) is not part of E1: it is kept for the final comparison, so it is not looked
  at repeatedly during development.
- **Robustness:** `metrics.json` is written after each task, and a rerun with
  `--tasks` keeps the finished tasks, so a Colab disconnect costs at most one task.
- **Diagnostics:** answerability replies that are neither label are counted by kind
  (`empty`, `yes_no`, `sentence`, `other` = usually a direct answer). Categories
  only: committed metrics contain no model text about dataset questions.
- **Plan sections:** 14 (Step 2.3).

### D-044: E1 baseline recorded; Phase 2 complete
- **Date:** 2026-11-11
- **Status:** Accepted
- **Evidence:** `results/eval/e1-baseline/` (metrics.json + README tables), L4, bf16,
  commit `77dd18e`, 36 min for 15,003 examples.
- **Result:** answering 53.5 VQA accuracy [51.9, 55.2]; answerability AP 69.4
  [66.8, 72.4]; quality macro-F1 28.0 [26.9, 28.9].
- **What E2 must beat, and where:** (1) the 36.6% of answerability replies that
  ignore the format, and recall 0 at precision 0.9; (2) the 9-point accuracy gap
  on photos with quality issues; (3) quality labels that are near-useless for
  `unrecognizable` (F1 0.01) and `dark` (precision 0.07).
- **Note:** the 7 answerability score fallbacks (0.16%) use the generated label.
- **Plan sections:** 14 (Steps 2.3, Phase 2).

### D-045: Training entrypoint design; Step 3.1 runs on the L4 only
- **Date:** 2026-11-11
- **Status:** Accepted
- **Decision** (`src/vizwiz/train/`):
  - Hugging Face `Trainer` with a PEFT LoRA model, not TRL's SFT trainer: our
    dataset already produces masked labels (prefix-checked on every example), so
    `Trainer` only needs to run the loop. It brings checkpoint/resume, gradient
    accumulation, cosine schedule, and the W&B integration.
  - Config: plan 8.1 defaults in `TrainConfig`, per-task YAML in `configs/train/`,
    `--set key=value` overrides; unknown keys are errors. Starting point: lr 1e-4,
    3% warmup, 1 epoch, per-device batch 2 × accumulation 8 = effective batch 16.
  - Every run records git commit, full config, SHA-256 of the record files (and
    refuses to train if they differ from `files.json`), GPU, versions, timing and
    peak memory, in `settings.json` / `train_summary.json` and in W&B.
  - Resume is automatic: a rerun with the same run id continues from the newest
    complete checkpoint, and the W&B run id equals the run id so the curves join.
  - QLoRA: NF4 with double quantisation, compute in bf16 (fp16 on GPUs without
    bf16); the vision tower and `lm_head` stay in 16-bit.
  - Attention: SDPA. FlashAttention-2 needs the separate `flash-attn` package,
    which is not in the pinned environment; SDPA on an L4 already uses a fused
    kernel. Revisit only if profiling shows attention dominates.
  - `bitsandbytes` and `wandb` added to `requirements/colab-gpu.txt`, unpinned until
    the first passing run, then pinned from its freeze file.
- **Deviation from the plan:** the plan put the smoke test on a T4 (QLoRA, fp16).
  All real training runs on the L4 in bf16, and T4-specific fp16 problems would cost
  time without telling us anything about those runs. The smoke tests (QLoRA and
  LoRA) and the profile therefore all run in one L4 session (`05_train_setup.ipynb`).
- **Resume test:** `--crash-at-step 20` ends the process with `os._exit` right
  after the step-20 checkpoint (no cleanup, like a lost runtime); the rerun must
  report `resumed_from = checkpoint-20` and finish at step 30.
- **Plan sections:** 8, 14 (Step 3.1).

### D-046: Step 3.1 checks passed; environment re-locked
- **Date:** 2026-11-11
- **Status:** Accepted
- **Evidence** (`results/train/setup/`, commit `e6284f5`, L4): W&B login OK;
  QLoRA smoke run stopped on purpose after its step-20 checkpoint and resumed to
  step 30 (`resumed_from = checkpoint-20`, 10 steps in the second session); LoRA
  bf16 smoke run OK; LoRA on 252 language-model modules, none in vision.
- **Measured:** LoRA 1.61 examples/s at 524k pixels, 11.1 GB; QLoRA 1.17
  examples/s, 7.2 GB. Profile: throughput roughly halves each time the image budget
  doubles; memory fits everywhere (max 16.6 of 22 GB).
- **Environment:** `bitsandbytes==0.50.2` and `wandb==0.31.0` pinned; lock file
  regenerated from `results/env/freeze_train_l4.txt`. Only additions; every
  previously locked version is unchanged.
- **Image budget:** still open (Open decisions). Cost per epoch of all three
  adapters: 5.1 h at 262k pixels, 9.8 h at 524k, 18.6 h at 1M.
- **Plan sections:** 14 (Step 3.1).

### D-047: How the image budget is chosen (rule fixed before the measurement)
- **Date:** 2026-11-11
- **Status:** Accepted (owner chose "measure first")
- **Why measure:** training cost scales with the budget (D-046: 5.1 h / 9.8 h / 18.6
  h per epoch of the three adapters at 262k / 524k / 1M pixels); the accuracy side
  is unknown, and VizWiz has many questions about small printed text.
- **Measurement:** the untrained model on the **val** split (never test, so the
  test set stays untouched for the final comparisons), at 262,144 / 524,288 /
  1,048,576 pixels: answering on all 1,464 answerable val questions, answerability
  on all 2,000 val questions (`06_resolution_check.ipynb`). Differences are
  measured **paired** on the same questions with a bootstrap CI
  (`vizwiz.eval.compare`, new; reused for E2–E5).
- **Rule, written down before seeing any result:** choose the smallest budget
  whose answering accuracy is within **1.0 point** of the best budget's and whose
  answerability AP is within **1.5 points** of the best budget's (point estimates;
  CIs reported alongside).
- **Known limitation:** zero-shot sensitivity to resolution is a proxy; a
  fine-tuned model may use extra detail differently. E6 (optional) would measure
  it after training.
- **Plan sections:** 10 (E6), 14 (Step 3.1).

### D-048: Checkpoint every 10 minutes (time-based)
- **Date:** 2026-11-12
- **Status:** Accepted (owner's request)
- **Decision:** training saves a checkpoint every `save_every_minutes` = 10 of
  training time (a `TimedSave` callback sets the Trainer's save flag), instead of
  every fixed number of steps. Step speed varies 4× across image budgets (D-046), so
  a fixed step count would mean anything from ~10 to ~40 minutes between saves; a
  time interval caps the loss from a Colab disconnect at 10 minutes for every run.
  `save_steps` remains available as an extra trigger (the smoke test uses it).
- **Cost:** a checkpoint is the adapter plus optimizer state (~0.4 GB); the newest 2
  are kept (`save_total_limit`), so ~0.8 GB of Drive per run, and each save takes
  seconds, about 2–3% of training time.
- **Plan sections:** 8.3.

### D-049: Keep VizWiz; frame it around safety; two additions
- **Date:** 2026-11-12
- **Status:** Accepted (owner)
- **Context:** the owner asked whether VizWiz solves a real problem, given that
  frontier models (GPT, Claude) plus phone screen readers already answer photo
  questions, and compared it with other portfolio projects.
- **Decision:** keep the project. The real-world problem is not "answer photo
  questions" but **unreliable answers on unusable photos**: about 30% of blind
  users' questions cannot be answered from the photo they took (audit finding 1),
  and E1 caught none of them at 90% precision. The project is framed as: a
  self-hosted assistant that knows when it cannot see, tells the user how to retake
  the photo, and answers only when reliable, with measured accuracy, decline rate,
  cost and privacy.
- **Additions:**
  - **Frontier reference** (optional, after E2): a frontier vision API on 300–500
    *val* questions, compared on accuracy, decline behaviour on unanswerable
    questions, and cost per photo. Sends dataset photos to an external API (CC BY
    4.0 permits it); to be confirmed by the owner before running.
  - **Sensitivity-guided compression** (optional, in E4): measure which layers
    tolerate low precision and protect the sensitive ones, instead of uniform
    quantization. Decided at Phase 4 from the remaining GPU budget.
- **Unchanged:** scope, experiments E1–E5, and all phases.
- **Plan sections:** 1, 10, 15.

### D-050: Image budget fixed at 524,288 pixels (D-047 rule applied)
- **Date:** 2026-11-12
- **Status:** Accepted (pre-registered rule, D-047)
- **Measurement** (untrained model, val split, `06_resolution_check.ipynb`; paired
  95% CI of the difference vs 524k in brackets):

  | Budget (px) | Visual tokens | Answering VQA acc | Answerability AP |
  |---|---|---|---|
  | 262,144 | ~234 | 54.99 [-2.11, +0.29] | 64.96 [-1.17, +1.46] |
  | 524,288 | ~494 | 55.87 (reference) | 64.77 (reference) |
  | 1,048,576 | ~972 | 56.07 [-0.92, +1.30] | 65.69 [-0.16, +2.12] |

- **Rule applied:** best answering 56.07 and best AP 65.69 (both at 1M).
  262k is 1.08 points behind on answering, so it fails the 1.0-point limit (AP gap
  0.73 passes). 524k is 0.20 behind on answering and 0.92 behind on AP, so it passes
  both. The smallest passing budget is **524,288**.
- **Honest reading:** every paired CI contains 0, so on val the three budgets are
  not clearly different. 262k missed the limit by 0.08 points. The rule was fixed
  in advance, so it is applied as written and not re-argued after the result.
  1M would double the training time (18.6 h vs 9.8 h per epoch of the three
  adapters) for a difference within noise.
- **Consequences:**
  - `max_pixels` = 524,288 for training, evaluation and serving. This is the E1
    budget already, so **E1 does not need a rerun**.
  - Batch: `per_device_batch` 1 × `grad_accum` 16 (effective batch 16, unchanged).
    In the profile bs1 was the fastest setting at 524k (1.46 vs 1.41 ex/s at bs2),
    and its peak memory was 9.7 GB.
  - Remaining GPU estimate: about 31–33 h.
  - Answering on val is 55.87 zero-shot. This is the val reference that E2
    training curves are compared against; the reported E2 number is on test.
- **Plan sections:** 8.1, 14 (Step 3.1).

---

## Open decisions

Resolved items move to the Decisions section with a `D-xxx` id.

| Item | Resolve at | Notes |
|---|---|---|
| 4-bit serving format: AWQ vs GPTQ | Step 4.1 | Whichever vLLM supports for Qwen3-VL on L4 |
| Cascade "unusable image" rule | Step 5.2 | Tuned on val |
| Final serving quantization format | After E4 | |
| E3 (QLoRA vs LoRA): run or cut | After E2 | Keep if GPU budget allows (~2–4 h); E4 has priority |
| Frontier-model reference run | After E2 | D-049; needs owner's OK to send val photos to an API |
| Sensitivity-guided quantization in E4 | Phase 4 | D-049; from remaining GPU budget |

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


### R-002: vLLM 0.32.0 server exits during startup on L4
- **Date hit:** 2026-11-06
- **Step:** 0.3
- **Environment:** Colab L4, Python 3.13.16, torch 2.13.0, vllm 0.32.0,
  transformers 5.19.0 (`results/env/freeze_l4.txt`)
- **What happened:** `vllm serve` with `--enable-lora` exited before `/health`
  responded (`server_started: false` after 180 s). Report:
  `results/env/compat_vllm_l4_attempt1.json`; log: `results/env/vllm_server_l4_attempt1.log`.
  Model load, torch.compile (66 s), CUDA graph capture (25 s), and KV cache allocation
  (6.7 GiB, 48,816 tokens) all succeeded first; the crash was the last startup step.
- **Root cause:** `FileNotFoundError: 'ninja'`. At the end of startup, FlashInfer
  JIT-compiles its sampling kernel and runs the `ninja` build tool by name. `ninja`
  is installed in the venv (`ninja==1.13.2`), but the venv is used without being
  activated, so `/content/venv/bin` is not on `PATH` and the subprocess cannot find it.
- **Initial suspect ruled out:** `torchaudio==2.11.0` next to `torch==2.13.0` is
  mismatched, but nothing in the startup path imported it. Left as is; it becomes a
  problem only if something imports torchaudio.
- **Options considered:** (a) put the venv `bin` on `PATH` for the server; (b) disable
  the FlashInfer sampler so no JIT build happens; (c) activate the venv in every
  notebook shell line (easy to forget).
- **Resolution:** (a) + (b), in `vizwiz.compat.server_env()` → see D-020.
- **Status:** Resolved (rerun 2026-11-06: server started, `ok: true`,
  `results/env/compat_vllm_l4.json`)

### R-003: `uv venv` blocks on an interactive prompt when the venv already exists
- **Date hit:** 2026-11-06
- **Step:** 0.3 (rerun)
- **Environment:** Colab L4, same VM reused, so `/content/venv` survived from the first run
- **What happened:** cell 2 printed "A virtual environment already exists at `venv`.
  Do you want to replace it? [y/n]" and waited for input; the prompt was cancelled
  (`^C`). The following `uv pip install` then reused the existing venv, and the printed
  versions were correct, so no harm was done.
- **Root cause:** `/content` is not always wiped between runs (only when the runtime is
  deleted), and `uv venv` asks before overwriting.
- **Resolution:** `uv venv --clear` in the notebook, so it always rebuilds without
  asking. A clean rebuild every session keeps runs reproducible.
- **Status:** Resolved

### R-004: train.zip image count did not match the expected 23,431
- **Date hit:** 2026-11-06
- **Step:** 1.2
- **Environment:** Colab, `vizwiz.data.download fetch`
- **What happened:** the zip check reported `train.zip` with 23,954 `.jpg` files
  against an expected 23,431 (`ok: false`). The file size matched the server's
  `Content-Length` exactly.
- **Root cause:** my expected value was wrong. 23,431 is the number of train images in
  the *quality* (and Captions) dataset. The shared zip holds every train image of both
  tasks: VQA train 20,523 ∪ quality train 23,431, which share 20,000 → 23,954. Checked
  against the annotation files: the union is exactly 23,954 names, numbered with no
  gaps from 00000000 to 00023953.
- **Resolution:** expected count corrected in `SOURCES` and in the committed
  `results/data/checksums.json`; data doc updated. No re-download needed.
- **Lesson:** derive expected counts from the annotation files themselves, not from
  dataset web pages that describe only one task.
- **Status:** Resolved

### R-005: The plan assumed an "unsuitable image" answer that does not exist
- **Date hit:** 2026-11-06
- **Step:** 1.3
- **What happened:** Plan Section 5.3 derives the 3-way answerability label
  (`answerable` / `unanswerable` / `unsuitable_image`) "from the dataset's
  answerability fields and answer distribution". In this VizWiz version the only
  special answer is `unanswerable` (55,993 train answers); there is no `unsuitable`
  answer and no field for it.
- **Root cause:** an outdated assumption in the plan (older VizWiz releases separated
  the two).
- **Options considered:** (a) binary answerability only; (b) derive `unsuitable_image`
  from the quality annotations' `unrecognizable` votes on unanswerable questions
  (about 8% of questions at k = 2); (c) drop answerability and rely on the quality adapter.
- **Resolution:** option (a): 2-way answerability; unusable photos are handled by the
  quality adapter's `unrecognizable` label → see D-031.
- **Status:** Resolved

### R-006: Identical photo files appear in more than one split
- **Date hit:** 2026-11-06
- **Step:** 1.5 (image scan in `02_data_prep.ipynb`, Colab CPU)
- **What happened:** the scan printed `IMAGE SCAN FOUND PROBLEMS`: 1,017 groups of
  byte-identical files (2,255 files), 643 of them spanning splits. Checked against the
  annotations: 235 of 7,750 test images and 298 of 8,000 test_vqa images had an
  identical file in train; 31 test and 77 test_vqa questions were repeated verbatim
  on the identical photo in train. Our own val carve had 87 images identical to
  train images.
- **Root cause:** the same photo was submitted more than once (often with a new
  question); the official splits were not deduplicated by file content, and our carve
  split by file name only.
- **Options considered:** (a) ignore and report; (b) drop duplicated images from the
  test sets; (c) drop copies from training and keep the test sets as published.
- **Resolution:** option (c) → see D-038.
- **Status:** Resolved

### R-007: Label files were never written in Colab, and the notebook did not stop
- **Date hit:** 2026-11-06
- **Step:** 1.5 (`02_data_prep.ipynb`, cells 3 and 6)
- **Environment:** Colab CPU
- **What happened:** cell 6 crashed with `FileNotFoundError:
  .../datasets/processed/vqa_val.jsonl`.
- **Root cause:** two bugs. (1) `vizwiz.data.splits` defaulted `--config` to the
  relative path `configs/data.yaml`; Colab runs from `/content`, not the repo, so cell 3
  failed and wrote nothing. Local runs and tests ran from the repo root, which hid
  it. (2) Colab `!` commands do not stop the notebook on failure, so later cells ran
  on missing inputs.
- **Resolution:** all default paths are resolved from the package location
  (`REPO_ROOT`), not the working directory. Notebook commands now go through an
  `sh()` helper that streams output and raises on a non-zero exit, so a failed step
  stops the run.
- **Status:** Resolved

### R-008: Token-stats step looked for images in the wrong folder
- **Date hit:** 2026-11-06
- **Step:** 1.5 (`02_data_prep.ipynb`, cell 6)
- **Environment:** Colab CPU
- **What happened:** after the three synthetic demos passed (`prefix_ok=True` for all
  tasks), the cell stopped with `FileNotFoundError:
  /content/data/images/VizWiz_train_00010748.jpg`.
- **Root cause:** the zips unpack into per-split subfolders (`images/train/...`). The
  image scan finds files with a recursive search, but the token-stats code built the
  path as `images/<name>`. The unit tests used a flat folder, which hid it.
- **Resolution:** all image lookups go through `vizwiz.data.images.find_images`
  (recursive); test added for the subfolder layout. The new fail-fast `sh()` helper
  (R-007) stopped the run at the failing cell, as intended.
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
