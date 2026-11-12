# PROJECT PLAN: Visual Assistant for Blind Users (VizWiz)

Fine-tuned vision-language adapters, compressed and served from one shared base model.

This document is the single source of truth for building the project. It is written
for Claude Code (the coding assistant) and for the project owner (Vishwa).

**Target audience for the finished project:** new-grad SDE roles *and* ML Engineer
roles. The scope balances ML rigor (baselines, fine-tuning gains, error analysis,
trade-off experiments) with systems work (serving, gateway, testing, load testing,
cost). Anything that does not serve one of those two stories is out of scope.

**Hardware available:** Google Colab **T4** and **L4** GPUs only. All notebooks and
GPU work run in Colab. Nothing GPU-related runs locally.

---

## 0. Working agreement for Claude Code (read first)

1. **Work step by step, gated by confirmation.** Each phase is split into numbered
   steps. Complete ONE step, report what was done, show the deliverable, and **stop
   and wait for the owner's confirmation** before starting the next step. Never
   chain multiple steps without approval.
2. **Explain mechanisms, not just outcomes.** When making a design choice, state
   *why* it works (what it changes in the model, data, or system), not only
   *what* it achieves.
3. **Correct mistakes directly.** If earlier code, results, or assumptions in this
   plan turn out to be wrong, say so plainly and propose the fix.
4. **Verify before assuming.** Library versions, model support, GPU support,
   dataset licenses, and dataset statistics in this plan are best-known values and
   must be verified (marked **[VERIFY]**). Report any discrepancy before proceeding.
5. **Never commit data, images, model weights, or secrets** to git.
6. **Never display or publish raw VizWiz images** in public artifacts (demo,
   README, dashboard, write-up). See Section 11 (Privacy).
7. **Code lives in the repo; Colab only runs it.** Claude Code cannot operate
   Colab directly, and the owner runs no notebooks locally. All logic goes into the
   Python package and scripts. Colab notebooks are thin launchers that clone the
   repo, install it, and call scripts. Claude Code writes the notebooks; the owner
   runs them in Colab and reports back output (logs, metrics JSON, errors).
8. **Design for T4/L4 only.** Every config must fit on an L4 (24 GB). Anything that
   needs a bigger GPU is out of scope unless the owner explicitly approves it.
9. **Guide every Colab step explicitly.** For each (Colab) step, give the owner exact
   instructions: which runtime to pick (CPU/T4/L4, High RAM), which notebook to open,
   which cells to run, what output to expect, and what to paste back. Debug from the
   pasted output.
10. **Log decisions and roadblocks in `DECISIONS.md`.** Every decision (design
    choices, library versions, thresholds, fallbacks) gets a `D-xxx` entry with the
    reason and alternatives. Every roadblock (errors, unsupported features,
    out-of-memory, wrong assumptions in this plan) gets an `R-xxx` entry, kept open
    until resolved. Update the log in the same step the decision or roadblock happens.

---

## 1. Project summary

### 1.1 The real-world problem
Blind people photograph objects and ask questions ("What does this label say?",
"Is this milk expired?"). Their photos are frequently blurry, badly framed,
poorly lit, or obstructed, because they cannot visually check the shot. Most
vision-language models are trained on clean, well-framed web images and fail
on this distribution. Worse, they answer confidently even when the photo does
not contain the answer, which is dangerous (for example, medication questions).

### 1.2 What the system does
Given a photo and a spoken-style question, the system:
1. Detects whether the photo is usable and, if not, tells the user how to retake
   it ("The label is cut off. Move the phone to the left.").
2. Decides whether the question is answerable from the photo, and declines
   instead of guessing when it is not.
3. Answers the question briefly when it is answerable.

### 1.3 How it is built
- One small vision-language base model (Qwen3-VL-4B-Instruct) **[VERIFY]**.
- Separate LoRA adapters per task, trained with SFT on real human-labeled data.
- Base model quantized for serving; adapters swapped per request via vLLM multi-LoRA.
- A FastAPI gateway runs the task cascade, plus monitoring and load testing.

### 1.4 Goals
- **ML:** measurable improvement over the zero-shot base model on all three tasks,
  broken down by image-quality slice, with confidence intervals and error analysis.
- **ML:** quantified trade-offs: LoRA vs QLoRA; accuracy vs serving quantization;
  cascade vs single multi-task adapter.
- **Systems:** a tested, containerized gateway + multi-LoRA serving setup with
  latency, throughput, and cost-per-1K-requests benchmarks on an L4.
- **Communication:** a README and short results write-up that explains what worked,
  what failed, and why.

### 1.5 Non-goals
- Not a medical device; no clinical or safety-critical claims.
- No synthetic or model-generated training labels. All labels come from human
  annotations in the datasets.
- Not training a vision model from scratch.
- No full fine-tuning, no GPUs larger than an L4, no exhaustive hyperparameter sweeps.

---

## 2. Environment and hardware

### 2.1 Compute (Google Colab only)
| Runtime | Memory | Use for |
|---|---|---|
| CPU (High RAM) | n/a | Data download, audit, preprocessing, label/split creation |
| T4 | 16 GB, Turing (sm75), **no bf16** | Smoke tests on tiny subsets, QLoRA debugging, short eval runs on subsets |
| L4 | 24 GB, Ada (sm89), bf16 + FP8 | All real training runs, full evals, vLLM serving, benchmarks |

Rules:
- Always debug on CPU or T4 first. Move to L4 only after a smoke test passes.
- **T4 is not a target for real training or serving.** It lacks bf16 (fp16 training
  of this model may be unstable) and lacks the compute capability required by many
  vLLM quantized kernels (AWQ/GPTQ Marlin and FP8 need sm80+; vLLM itself may require
  newer GPUs) **[VERIFY]**. Use T4 for: code paths, data loading, masking checks,
  checkpoint/resume tests, and QLoRA (4-bit base) smoke tests.
- **L4 memory budget (to be measured in Step 3.1):** 4B model in bf16 is roughly 8–9 GB
  of weights. LoRA training must fit with gradient checkpointing, small per-device
  batch (1–2) plus gradient accumulation, and capped image resolution. If it does
  not fit, fall back in this order: lower max image pixels → QLoRA → 2B model.
  Report before switching.
- Enable **High RAM** for data processing.
- Log GPU type, peak GPU memory, and wall-clock time for every run, so compute
  cost can be reported.

### 2.2 Colab constraints
- Sessions disconnect and wipe the machine. **Every training run must checkpoint**
  to Google Drive (or the Hugging Face Hub) at regular step intervals and must be
  resumable from the latest checkpoint with the same command.
- Cache datasets, model weights, and pip wheels on Google Drive to avoid
  re-downloading every session.
- Keep runs short enough to finish within one session where possible; otherwise
  rely on resume.
- **Docker is not available in Colab.** In Colab, services (vLLM, gateway, Locust,
  Prometheus) run as background processes on the same VM, launched by the
  serving notebook. Docker/Docker Compose is used for local packaging and
  CI only (see Section 12.3).

### 2.3 Local machine (no notebooks, no GPU)
Used for: code editing, git, unit tests, linting, and building/running the Docker
Compose stack **against a mock model server** (no real model locally). Optionally,
the local gateway can point at the Colab vLLM server through a tunnel
(e.g., cloudflared/ngrok) for a live demo **[VERIFY Colab terms allow this]**.

---

## 3. Tech stack

Pin exact versions in `pyproject.toml` / `requirements*.txt` after verifying
compatibility with Qwen3-VL **and with T4/L4 GPUs** **[VERIFY all versions]**.

| Layer | Tools |
|---|---|
| Language | Python 3.11+ |
| Data | `datasets`, `pandas`, `Pillow`, `numpy` |
| Training | `torch`, `transformers`, `peft`, `trl` (SFTTrainer), `bitsandbytes` (QLoRA), `accelerate` |
| Optional speedups | `unsloth` (only if it supports Qwen3-VL on T4/L4) **[VERIFY]**. FlashAttention-2 needs sm80+: available on L4, **not on T4**; use SDPA on T4 |
| Experiment tracking | Weights & Biases (D-025); API key in Colab Secrets |
| Config | YAML configs + `pydantic` for validation |
| Quantization | `llm-compressor` (FP8 / AWQ / GPTQ for vLLM) **[VERIFY Qwen3-VL support]** |
| Serving | `vllm` (OpenAI-compatible server, multi-LoRA), run on L4 |
| Gateway | `fastapi`, `uvicorn`, `httpx` |
| Adapter registry | W&B Registry (adapters as versioned artifacts; `production` alias = live version), behind a small interface with an in-memory fake for tests (D-026) |
| Load testing | `locust` (headless mode in Colab) |
| Monitoring | Prometheus metrics endpoints (vLLM + gateway); Grafana in the local Compose stack |
| Dashboard | Streamlit (results tables, plots; reads committed `results/*.json`) |
| Packaging | Docker, Docker Compose (local + CI only) |
| CI | GitHub Actions (lint, unit tests, metric-gate check) |
| Quality | `ruff`, `pytest`, `mypy` (optional) |

---

## 4. Repository structure

```
vizwiz-assistant/
├── README.md
├── PROJECT_PLAN.md               # this file
├── pyproject.toml
├── configs/
│   ├── data.yaml                 # paths, split seeds, label thresholds
│   ├── train/
│   │   ├── answer_lora.yaml
│   │   ├── answerability_lora.yaml
│   │   ├── quality_lora.yaml
│   │   └── multitask_lora.yaml
│   ├── eval.yaml
│   └── serve.yaml
├── src/vizwiz/
│   ├── data/
│   │   ├── download.py           # fetch + checksum datasets
│   │   ├── audit.py              # statistics report
│   │   ├── labels.py             # label construction (answers, answerability, quality)
│   │   ├── splits.py             # train/val/test split creation + manifests
│   │   └── formatting.py         # chat-template formatting per task
│   ├── train/
│   │   ├── sft.py                # LoRA / QLoRA entrypoint
│   │   └── callbacks.py          # checkpoint-to-Drive, logging, GPU memory logging
│   ├── eval/
│   │   ├── vqa_metric.py         # official-style VizWiz accuracy + normalization
│   │   ├── answerability.py
│   │   ├── quality.py
│   │   ├── slices.py             # per-quality-slice breakdowns
│   │   ├── error_analysis.py     # failure categorization (text only, no images)
│   │   └── run_eval.py           # batch inference + metrics JSON
│   ├── quantize/
│   │   └── quantize.py
│   ├── serve/
│   │   ├── gateway.py            # FastAPI app implementing the cascade
│   │   ├── cascade.py
│   │   ├── registry.py           # adapter registry interface: W&B Registry + in-memory fake
│   │   ├── preprocess.py         # image resizing / encoding
│   │   ├── guidance.py           # quality issue -> fixed retake-guidance text
│   │   └── mock_model.py         # fake OpenAI-compatible server for local/CI tests
│   └── bench/
│       ├── locustfile.py
│       └── report.py
├── notebooks/                    # thin Colab launchers only (run by the owner in Colab)
│   ├── 00_setup_colab.ipynb      # Drive mount, clone, install, GPU check
│   ├── 00b_compat_check.ipynb    # L4: Step 0.3 compatibility check
│   ├── 01_data_download.ipynb    # CPU: download + checksums (Step 1.2)
│   ├── 02_data_prep.ipynb        # CPU: rebuild check, image scan, token stats (1.3–1.5)
│   ├── 03_eval_check.ipynb       # T4: run_eval.py on 100 examples per task (2.2)
│   ├── 04_e1_baseline.ipynb      # L4: E1 zero-shot baseline, full test set (2.3)
│   ├── 05_train_setup.ipynb      # L4: W&B check, QLoRA/LoRA smoke + resume, profile (3.1)
│   ├── 06_resolution_check.ipynb # L4: zero-shot val accuracy at 3 image sizes (3.1, D-047)
│   ├── 07_e2_answer.ipynb        # L4: train answering adapter, test eval, paired vs E1 (3.2, D-051)
│   ├── 02_train.ipynb            # T4 smoke / L4 real
│   ├── 03_eval.ipynb             # L4
│   ├── 04_quantize.ipynb         # L4
│   └── 05_serve_bench.ipynb      # L4: vLLM + gateway + Locust as background processes
├── docker/
│   ├── gateway.Dockerfile
│   └── docker-compose.yml        # gateway + mock model + Prometheus + Grafana + dashboard
├── monitoring/                   # prometheus.yml, grafana dashboards
├── dashboard/app.py              # Streamlit results dashboard
├── docs/
│   ├── data_licenses.md
│   └── results_writeup.md        # final write-up for recruiters / interviews
├── results/                      # metrics JSON + plots (small files only, committed)
├── tests/
└── .github/workflows/ci.yml
```

---

## 5. Data

### 5.1 Datasets **[VERIFY licenses, access terms, sizes]**
| Dataset | Content | Used for |
|---|---|---|
| VizWiz-VQA | Real photos taken by blind people + their spoken questions; ~10 crowdsourced answers per question; answerable/unanswerable annotations | Answering adapter, answerability adapter |
| VizWiz image quality issues | Per-image flags from multiple annotators: blur, framing, too dark, too bright, obstruction, rotation, other, no issue; plus a recognizability flag | Quality / retake-guidance adapter |

Verified VizWiz-VQA split sizes: train 20,523, val 4,319, test 8,000 (one question per
image). Quality annotations: 23,431 / 7,750 / 8,000 images, same image pool and naming.
**VQA test answers are now public** (`VQA_test.json`); **quality test labels are not**.
Details, URLs, and sizes: `docs/data_licenses.md`. Quality labels come from the
challenge JSON files, not the per-worker CSV (D-023).

### 5.2 Split strategy
- **Our test set = official val split.** Never used for training, tuning, or
  early stopping. Touch it only for final reported numbers. This is the only split
  that has labels for all three tasks.
- **Secondary VQA test = official VQA test split** (answers now public). Used only for
  final answering/answerability numbers that are comparable with published results (D-022).
- **Our val set = ~10% carved from official train**, stratified by
  answerability and by answer type, with a fixed seed.
- **Image-level grouping:** if an image appears in multiple tasks/datasets,
  all of its examples must land in the same split across ALL tasks, to prevent
  leakage between the quality and VQA adapters. Join datasets on image filename.
- Save split membership as manifests (`splits/*.jsonl` with image id + sha256
  hash of the image file) so splits are reproducible and verifiable.
- **Dev subset:** also save a small fixed subset (~500 examples per task from our
  val set) for fast iteration on T4/L4. Full val/test only for decisions and final numbers.

### 5.3 Label construction (human labels only)
- **Answer target (D-033):** the distinct human answer with the highest official VQA
  accuracy when output verbatim; ties go to more votes, then the earliest-listed
  answer. For answerable questions `unanswerable` is never the target. Rule and
  tests: `src/vizwiz/data/labels.py`.
- **Answerability label (2-way, D-031):** `answerable` / `unanswerable`, from the
  official `answerable` flag. VizWiz has no "unsuitable image" answer (R-005);
  unusable photos are the quality adapter's job. The number of `unanswerable`
  votes is kept for the borderline slice.
- **Quality issues (multi-label, D-030, D-032):** an issue is positive if at least
  k = 2 of 5 annotators flagged it. Labels: `blur`, `framing`, `rotation`, `dark`,
  `bright`, `obstruction`, `unrecognizable`; `["none"]` if nothing reaches k.
  `OTH` is dropped. Config: `configs/data.yaml`.
- **Retake guidance text is NOT generated by any model.** It is a fixed,
  hand-written mapping from each quality issue to an actionable instruction in
  `serve/guidance.py` (e.g., blur → "Hold the phone steady and try again.").
  The model predicts issue labels; the system maps labels to guidance.

### 5.4 Answer normalization
Ported from the official VizWiz evaluation code with identical behaviour (D-029):
`src/vizwiz/eval/vqa_metric.py`. Following the official metric, only predictions
are normalised; ground-truth answers are compared as stored.

---

## 6. Tasks, prompts, and output formats

All tasks use the base model's chat template with the image in the user turn.
Outputs are short and machine-parseable. Loss is computed **only on the
assistant (output) tokens**; prompt and image tokens are masked out.

### 6.1 Answering adapter
- Input: image + question
- Output: a short answer string (target = Section 5.3 answer target)
- Trained on answerable questions only (D-033). In the cascade, a decline counts as
  the answer `unanswerable` for official VQA scoring.
- Prompt (system): "Answer the question about the photo in a few words."

### 6.2 Answerability adapter
- Input: image + question
- Output: exactly one of `answerable` | `unanswerable` (D-031)

### 6.3 Quality adapter
- Input: image only
- Output: JSON list of labels from Section 5.3, e.g. `["blur","framing"]` or `["none"]`
- Use constrained/structured decoding at inference to guarantee valid JSON.

### 6.4 Multi-task adapter (for Experiment E5)
- One adapter trained on the union of 6.1–6.3, with a task tag in the prompt.

### 6.5 Inference cascade
```
image + question
  -> quality adapter: issues
       if issues make the image unusable (configurable rule, e.g. unrecognizable
       flag or severe blur/framing): return retake guidance (guidance.py)
  -> answerability adapter
       unanswerable: return a decline message
  -> answering adapter: return answer
```
The cascade rule thresholds live in `configs/serve.yaml`.

---

## 7. Model

- **Primary base:** Qwen3-VL-4B-Instruct. **Fallback:** Qwen3-VL-2B-Instruct (used
  only if 4B does not fit or is too slow on L4).
  **[VERIFY]** exact model IDs, license, transformers support, vLLM LoRA support
  for this architecture, vLLM support on L4, and support for LoRA on quantized
  VLM bases in vLLM.
- **Fallback** if verification fails: another small VLM with confirmed vLLM
  multi-LoRA support. Report options before switching.
- **Trainable parts:** LoRA on the language model's attention + MLP projections.
  **Vision encoder frozen. Multimodal projector frozen.**
- **Image resolution:** controlled via the processor's min/max pixel settings.
  Pick one setting in Step 3.1 that fits L4 memory and keeps per-example latency
  reasonable; it is the main memory knob, because vision tokens dominate sequence
  length.

---

## 8. Training

### 8.1 Default SFT hyperparameters (starting points; tune lightly on our val set)
| Parameter | Start value |
|---|---|
| LoRA rank r / alpha | 16 / 32 (fixed; no rank sweep) |
| LoRA dropout | 0.05 |
| Target modules | q, k, v, o, gate, up, down (language model only) |
| Learning rate | 1e-4 to 2e-4 |
| Schedule | cosine, ~3% warmup |
| Epochs | 1–3, select by val metric |
| Precision | bf16 on L4. T4: QLoRA (4-bit base, fp16 compute) smoke tests only |
| Attention | FlashAttention-2 on L4 if supported for this model; SDPA on T4 |
| Gradient checkpointing | on |
| Per-device batch | 1–2 |
| Effective batch size | 16–32 via gradient accumulation |
| Seed | fixed, logged |

### 8.2 Training modes
- `lora` (bf16 base, L4) — **main method**
- `qlora` (NF4 4-bit base via bitsandbytes) — comparison (E3); also the T4 smoke-test mode

### 8.3 Requirements
- One entrypoint `python -m vizwiz.train.sft --config configs/train/<task>.yaml --mode <lora|qlora>`.
- Checkpoint every N steps to Drive; resume from latest automatically.
- Log config, git commit hash, dataset manifest hashes, GPU type, peak memory,
  wall-clock time, and metrics to the tracker.
- Save only adapter weights.

---

## 9. Evaluation

### 9.1 Metrics
| Task | Primary metric | Secondary |
|---|---|---|
| Answering | VizWiz VQA accuracy: per question, `min(#annotators matching / 3, 1)`, averaged over the official annotator-subset scheme **[VERIFY against official eval code; prefer reusing it]** | Accuracy split by image-quality slice and answer type |
| Answerability | Recall on `unanswerable` at a fixed precision; official average precision; macro-F1 | Confusion matrix; borderline slice (3–5 `unanswerable` votes) |
| Quality | Per-issue F1, macro-F1 | JSON validity rate |
| Cascade (end-to-end) | Accuracy of final user-facing response on test set | **Rate of confident wrong answers on unanswerable questions** (headline safety number) |

### 9.2 Rigor
- Report 95% bootstrap confidence intervals for all headline metrics.
- Same test set, same normalization, same decoding settings (greedy) for all compared setups.
- Track and report latency per request alongside accuracy for every setup.

### 9.3 Error analysis (required, for ML-role interviews)
- After E2, sample ~100 failures per task from our val set and categorize them
  (e.g., text too small to read, wrong object, label noise / annotators disagree,
  quality issue missed). Record categories and counts in text only.
- Report which failure categories fine-tuning fixed and which remain.
- Never export or embed the images themselves.

### 9.4 Output
`run_eval.py` writes `results/<run_id>/metrics.json` plus per-example predictions
(predictions stay on Drive, not in git; only summary JSON is committed).

---

## 10. Experiments

Five core experiments plus one optional. All fit on an L4.

| ID | Experiment | Compares | Key question | Audience |
|---|---|---|---|---|
| E1 | Baseline | Zero-shot 4B base model | How bad is the off-the-shelf model on this distribution? | ML |
| E2 | Main fine-tuning | LoRA (bf16) per task vs E1, per quality slice | How much does fine-tuning help, and where? | ML |
| E3 | Fine-tuning method | LoRA (bf16) vs QLoRA (NF4), both on L4 | What does QLoRA's memory saving cost in accuracy and speed? | ML |
| E4 | Serving quantization | bf16 vs FP8 vs 4-bit (AWQ or GPTQ) base, adapters attached unmerged | Quality retention per task; memory; throughput | ML + SDE |
| E5 | Architecture | Cascade of 3 adapters vs 1 multi-task adapter | Accuracy and end-to-end latency trade-off | ML + SDE |
| E6 (optional) | Image resolution | 2–3 max-pixel settings | Accuracy vs latency (vision tokens dominate compute) | ML + SDE |

Note for E4: adapters trained on a bf16 (or NF4) base are applied to a
differently quantized base at serving time. Measure the resulting degradation
explicitly; this is a known source of accuracy loss. FP8 is natively supported on
L4 (Ada); AWQ/GPTQ kernel support for this model in vLLM must be verified **[VERIFY]**.

Cut from scope: full fine-tuning, LoRA rank ablation, 2B vs 4B comparison
(2B is only a fallback).

---

## 11. Privacy and ethics

- A meaningful fraction of VizWiz images contain private information (the
  photographers could not check what they captured). Therefore:
  - Never display raw VizWiz images in the public demo, README, dashboard, or write-up.
  - Use the owner's own photos (or clearly licensed images) for demos.
  - Do not upload VizWiz images to third-party APIs unless the dataset terms
    allow it **[VERIFY]**.
- Model card in README: intended use, limitations, failure modes (especially
  medication/label reading), and that the system is not a safety device.
- Prefer declining over guessing; report the confident-wrong-answer rate.

---

## 12. Serving system

### 12.1 Components
- **vLLM server (Colab L4):** OpenAI-compatible API, `--enable-lora`, `--max-loras`,
  `--max-lora-rank`, quantized base (per E4).
- **FastAPI gateway:** `POST /v1/assist` (image + question → final response +
  trace of which adapters ran and their outputs), `GET /health`,
  `GET /adapters`, `POST /adapters/{name}/load`, `GET /metrics` (Prometheus).
  Implements the cascade, image preprocessing, timeouts, retries on transient
  errors, and structured logging.
- **Adapter registry (W&B Registry, D-026):** each trained adapter is uploaded as a
  versioned W&B artifact with metadata: task, base model id, rank, training run,
  metrics summary. The `production` alias marks the live version per task;
  promotion and rollback = moving the alias. At startup the gateway resolves the
  `production` version for each task, downloads it, and loads it into vLLM. Code
  depends on a `Registry` interface; tests and CI use an in-memory fake, so they
  need no network or API key.
- **Mock model server:** a small fake OpenAI-compatible server returning canned
  outputs, so the gateway, cascade, and Compose stack are testable locally and in CI
  without a GPU.

### 12.2 Where things run
| Environment | Runs | Purpose |
|---|---|---|
| Colab L4 (`05_serve_bench.ipynb`) | vLLM + gateway + Locust (headless) as background processes on one VM | Real benchmarks and E4/E5 latency numbers |
| Local Docker Compose | gateway + mock model + Prometheus + Grafana + Streamlit dashboard | Packaging, demo of the full stack, monitoring dashboards |
| GitHub Actions | unit tests + gateway tests against mock model | CI |

In Colab, raw Prometheus metrics from vLLM and the gateway are scraped by the
benchmark script and saved to `results/bench/` as JSON/CSV, then plotted. Grafana
dashboards are built and demonstrated in the local Compose stack.

### 12.3 Benchmarks (Locust on Colab L4)
- Metrics: time to first token, end-to-end latency p50/p95/p99, throughput at a
  latency SLO (e.g., p95 < 2 s), GPU memory (weights + KV cache), adapter load time.
- Sweeps: concurrency (1 → 32, raise if the L4 handles it), number of active adapters.
- Baseline: serving three merged bf16 models, or a documented explanation of why
  they do not fit on one L4 (which is itself a result).
- Cost per 1,000 requests = L4 hourly price / requests per hour at the SLO
  (state the price assumption used).

---

## 13. MLOps and quality

- Every run is reproducible from: config file + git commit + dataset manifest hashes + seed.
- CI (GitHub Actions, CPU only): ruff, pytest (metric code, normalization,
  label construction, split leakage checks, cascade logic and gateway against the
  mock model), and a **metric gate** that compares committed
  `results/*/metrics.json` for a candidate adapter against thresholds in
  `configs/eval.yaml`. (CI has no GPU; the eval itself runs on Colab and its
  summary JSON is committed.)
- Required tests:
  - No image id appears in more than one split (across all tasks).
  - Answer normalization cases.
  - VQA accuracy computation on hand-made examples.
  - Guidance mapping covers every quality issue code.
  - Cascade routing for each branch (unusable photo, unanswerable, answered).

---

## 14. Phases and steps

Each step ends with a **deliverable** and a **STOP: confirm with owner**.
Steps marked (Colab) require the owner to run a notebook and report output.

### Phase 0: Repository and environment
- **0.1** Create repo skeleton (Section 4), `pyproject.toml`, ruff/pytest config, CI stub.
  Deliverable: empty but runnable package; CI green. STOP.
- **0.2** (Colab) Setup notebook: mount Drive, clone repo, install, print GPU type and
  compute capability, set cache dirs on Drive.
  Deliverable: `00_setup_colab.ipynb` working on CPU, T4, and L4. STOP.
- **0.3** (Colab, L4) Compatibility check: load Qwen3-VL-4B in transformers, run one
  inference; start vLLM with `--enable-lora` and send one request.
  Deliverable: confirmed versions pinned; go/no-go on the model choice. STOP.

### Phase 1: Data (complete, D-039)
- **1.1** Verify licenses/access terms for all datasets; choose experiment tracker.
  Deliverable: short findings note in `docs/data_licenses.md`. STOP.
- **1.2** (Colab, CPU) Download scripts with checksums, cached on Drive.
  Deliverable: datasets on Drive; checksum log. STOP.
- **1.3** (Colab, CPU) Data audit: counts per split; answerability distribution; answer
  frequency and annotator agreement; quality-issue counts for k = 1, 2, 3;
  image size distribution; overlap between datasets by image id.
  Deliverable: `results/audit/audit_report.md` (text and tables only). STOP.
- **1.4** Label construction + split creation + dev subset + manifests + leakage tests. Leakage is checked by file content (SHA-256), not only by name: copies of test photos are excluded from training (D-038).
  Deliverable: manifests, passing tests, label statistics. STOP.
- **1.5** Task formatting (chat template, masking) + image preprocessing; then one
  (Colab, CPU) run of `02_data_prep.ipynb` for all image-dependent Phase 1 work
  (D-037). Deliverable: printed synthetic formatted examples, token-length stats per
  image-size budget, image stats + hashes, label-file rebuild check. STOP.

### Phase 2: Evaluation harness and baseline (complete, D-044)
- **2.1** Implement metrics (reuse official VizWiz eval logic where possible) + unit tests. STOP.
- **2.2** (Colab, T4) Batch inference + `run_eval.py`; verify on a 100-example subset. STOP.
- **2.3** (Colab, L4) E1 baseline (4B zero-shot) on the full test set.
  Deliverable: baseline metrics JSON + per-slice table. STOP.

### Phase 3: Training
- **3.1** (Colab, L4 only, D-045) W&B login check first (`WANDB_API_KEY` from Colab Secrets,
  D-025); then SFT entrypoint; smoke test with 200 examples on T4
  (QLoRA mode); verify loss masking (only answer tokens contribute) and
  checkpoint/resume (kill the run, resume, confirm step count continues). Then
  measure peak memory and step time on L4 in bf16 LoRA mode and fix the image
  resolution setting. STOP. **Done** (D-046, D-050: 524,288 pixels).
- **3.2** (Colab, L4) Train answering adapter (E2). Evaluate. STOP.
- **3.3** (Colab, L4) Train answerability and quality adapters (E2). Evaluate. STOP.
- **3.4** Error analysis (Section 9.3) on all three adapters. STOP.
- **3.5** (Colab, L4) E3: QLoRA run for the answering adapter; compare with LoRA. STOP.
- **3.6** (Colab, L4) E5: train multi-task adapter; compare with cascade. STOP.

### Phase 4: Compression
- **4.1** (Colab, L4) Quantize base to FP8 and one 4-bit format (AWQ or GPTQ)
  **[VERIFY tool support for Qwen3-VL; vision encoder may need to stay in bf16]**. STOP.
- **4.2** (Colab, L4) E4: evaluate each adapter on each quantized base in vLLM. STOP.

### Phase 5: Serving
- **5.1** (Colab, L4) vLLM multi-LoRA server config; manual request tests per adapter. STOP.
- **5.2** FastAPI gateway + cascade + preprocessing + guidance mapping + mock model + tests. STOP.
- **5.3** Adapter registry (W&B Registry client + in-memory fake) + promote/rollback + load/unload endpoints. STOP.
- **5.4** Local Docker Compose: gateway + mock model + Prometheus + Grafana + dashboard. STOP.

### Phase 6: Benchmarking and reporting
- **6.1** (Colab, L4) Locust load tests and sweeps; save metrics; plots. STOP.
- **6.2** Cost analysis and merged-model baseline comparison. STOP.
- **6.3** Streamlit results dashboard; README with architecture diagram, results
  tables, model card, and reproduction instructions; `docs/results_writeup.md`. STOP.
- **6.4** Draft resume bullets: one SDE-focused version, one ML-focused version. STOP.

---

## 15. Optional extensions (only after Phase 6, if time permits)

1. **E6 image-resolution experiment** (Section 10).
2. **Dedicated CV image-quality classifier.** Train a small vision model
   (e.g., MobileNet/EfficientNet family, fits easily on T4) on the quality labels
   as a fast pre-filter before the VLM. Compare to the VLM quality adapter on
   accuracy, latency, and size.
3. **MCP server.** Expose the assistant as an MCP tool so agents can call it.
4. **Live demo** via a tunnel from the local gateway to a Colab vLLM session,
   using the owner's own photos.

---

## 16. Risks and mitigations

| Risk | Mitigation |
|---|---|
| vLLM lacks LoRA support for the chosen VLM, or for LoRA on its quantized base | Verify in Step 0.3; fall back to a supported VLM; report before switching |
| 4B LoRA training does not fit in L4 24 GB | Lower max image pixels → QLoRA → 2B model, in that order; report before switching |
| T4 incompatibilities (no bf16, no FlashAttention-2, unsupported vLLM kernels) | Use T4 only for smoke tests in QLoRA/fp16 mode; all real runs on L4 |
| Colab disconnects mid-run | Step-based checkpoints to Drive; auto-resume; tested in Step 3.1 |
| Compute units run out | Debug on CPU/T4; dev subset for iteration; log GPU-hours per run; core experiments only |
| Quantization degrades the vision encoder | Keep vision encoder in bf16; quantize language model only; measure |
| Class imbalance (rare quality issues, unanswerable) | Report per-class metrics; consider class-balanced sampling; tune thresholds on val |
| Label noise from annotator disagreement | Use agreement-based targets and official metric; report agreement statistics |
| Test-set leakage | Image-level split manifests + automated leakage tests |
| Privacy exposure | Section 11 rules; no raw dataset images in public outputs |
| Scope creep | Only the 5 core experiments before Phase 6 is done |

---

## 17. Open decisions (resolve when reached)

- ~~Image resolution setting~~ resolved: 524,288 pixels (D-050).
- 4-bit serving format: AWQ vs GPTQ (Step 4.1, whichever vLLM supports for this model on L4).
- Cascade "unusable image" rule (Step 5.2, tuned on val).
- Final serving quantization format (after E4).
