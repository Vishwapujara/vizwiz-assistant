# VizWiz Visual Assistant: results write-up

*A fine-tuned vision-language assistant for blind users that declines rather than
guesses, served from one GPU. All numbers are from the committed results; every
decision is logged in [`DECISIONS.md`](../DECISIONS.md).*

## 1. The problem

Blind and low-vision people photograph things and ask questions: "What does this
label say?", "Is the light on?". Their photos are often blurry, badly framed, dark
or of the wrong side of the object, because they cannot check the shot. Vision-
language models trained on clean web images do poorly on these photos. Worse,
they answer confidently when the photo does not contain the answer, which is
dangerous for questions about medication or food.

VizWiz is the standard dataset for this: real questions from blind users, each
answered by 10 people, plus per-photo quality annotations from 5 people. Between a
quarter and a third of its questions (27–32% by split) cannot be answered from
the photo.

**Goal:** a system that (1) notices bad photos and says how to retake them,
(2) declines unanswerable questions instead of guessing, and (3) answers briefly
otherwise. It must be trained and served on one 24 GB GPU (an NVIDIA L4 in Colab).

## 2. What I built

- **Model:** Qwen3-VL-4B-Instruct with LoRA adapters (rank 16, about 33M trainable
  parameters, under 1% of the model), trained with supervised fine-tuning on human
  labels only. The loss covers the answer tokens only.
- **Three tasks:**
  - photo quality: a JSON list from 7 issue labels;
  - answerability: `answerable` or `unanswerable`, with a probability read from
    the first token's log-probabilities;
  - short answer.
- **Cascade:** a FastAPI gateway runs the quality and answerability checks
  concurrently on a vLLM server, then either answers or declines with fixed,
  reviewed retake tips.
- **Operations:**
  - adapters are versioned in a W&B registry with live promote and rollback;
  - Prometheus metrics and a Grafana dashboard;
  - a Docker Compose stack with a mock model server, so the whole system runs on
    a laptop and in CI;
  - Locust load tests on the real GPU.

## 3. How results were kept honest

- **Held-out test set = the official VizWiz val split.** The official test split
  has no public answers. A separate val set was carved from official train,
  grouped by image, for every tuning decision. Each experiment touched test once.
- **Leakage found and fixed.** A file-hash scan found byte-identical photos across
  the official splits: 235 test photos also appear in train, sometimes with the
  same question. 605 such copies were removed from training; the test sets stay
  as published (R-006, D-038).
- **Paired statistics.** Every comparison is on the same test examples, with
  paired bootstrap 95% confidence intervals of the difference.
- **Rules fixed before results.** For each decision (image resolution, E5
  "matches", FP8 "acceptable", the decline threshold, served-vs-offline parity)
  the reading rule was written into the decision log *before* the run. That
  includes the one case where the rule went against the convenient answer (FP8,
  below).
- **Privacy.** VizWiz photos can contain private information. No dataset image or
  question text is stored in the repo, shown or logged; committed results are
  aggregates.

## 4. Results

### Fine-tuning works on all three tasks (test, paired against the untrained model)

| Task | Untrained (E1) | One adapter per task (E2) | Difference [95% CI] |
|---|---|---|---|
| Answering, VQA accuracy | 53.5 | 61.7 | +8.2 [+7.0, +9.3] |
| Answerability, average precision | 69.4 | 89.8 | +20.8 [+18.5, +22.9] |
| Photo quality, macro-F1 | 28.0 | 65.2 | +37.2 [+35.9, +38.7] |

What changed:
- The untrained model produced a valid answerability label only 63% of the time.
- It almost never detected "unrecognizable" photos (F1 0.01 → 0.74).
- Its answers were too long for the scoring.
- Training fixed all three. Wrong refusals on clear-cut answerable questions fell
  from 10.4% to 0.9%.

### One multi-task adapter matches three specialists (E5)

Same data (51,626 examples) and same compute, one adapter for all three tasks:
- answering **+0.8 [+0.2, +1.5]**;
- answerability +0.2 [−0.4, +0.7];
- quality +0.4 [−0.3, +1.1].

By the pre-set rule it matches on all three, so the served system uses one
adapter.

### FP8 serving: big efficiency gain, but it missed the bar (E4)

FP8 weights:
- cut weight memory 41% (8.6 → 5.05 GiB);
- added 55% request capacity;
- made inference 14% faster.

Answering and answerability did not change. Quality macro-F1 fell 0.6
[−1.2, −0.04], and the lower end missed the pre-set −1 point margin by 0.16. The
rule was applied as written: serving stays bf16, and FP8 is a documented option
for high load. The load test later showed the GPU is compute-bound, not
memory-bound, which makes FP8's extra cache room less valuable here anyway.

### The whole system (test, threshold chosen on val)

A decline counts as the answer "unanswerable" under the official metric.

| | Always answer | Threshold 0.5 | **Served (0.10)** |
|---|---|---|---|
| Official VizWiz accuracy | 47.1 | 70.9 | **77.7** |
| Wrong answers, share of all questions | 52.8% | 26.4% | **14.8%** |
| Unanswerable questions answered anyway | 100% | 19.5% | **2.7%** |
| Answerable questions declined | 0% | 10.1% | 28.6% |

The chosen operating point cuts wrong answers by more than a factor of three.
The price is declining 29% of answerable questions, each with retake tips. The
threshold is a config value and the full curve is published, so a product owner
can choose a different point.

### Serving on one L4

| | Value |
|---|---|
| Latency, one user | p50 0.61 s, p95 0.88 s |
| Throughput within p95 < 2 s | 2.92 requests/s (each = 3 model calls) |
| Peak throughput | 4.12 requests/s |
| Cost per 1,000 requests | $0.067 (GCP L4 at $0.707/h, fully busy) |
| Failures | 0 of 4,436 requests |

Three merged bf16 models would need about 25.5 GiB for weights alone, more than
the L4 holds. Multi-LoRA serves four adapters from one base for about +0.2 GiB.

## 5. What did not work, and what surprised me

1. **A pre-registered safety rule hurt.** Error analysis suggested forcing
   "please retake" whenever the quality stage says "unrecognizable". Measured on
   the served system, it lowered accuracy (val −1.6, test −0.7) and mostly
   declined questions the answerability check would have answered correctly. The
   answerability adapter already sees the photo. The rule was removed; quality now
   only gives tips (D-064).
2. **The served system disagreed with the evaluated one.** The gateway re-encoded
   photos as JPEG, and 8% of answers changed (greedy decoding flips on near-ties).
   Accuracy did not change measurably, but "the numbers I report are the numbers I
   serve" matters, so I tested lossless PNG. Agreement rose to 97–100% per stage
   for about 10 ms per request (D-066).
3. **The official metric rewards declining.** "Unanswerable" earns credit whenever
   3 of 10 annotators said so, which is common even on questions labelled
   answerable. That pushed the chosen threshold down to 0.10. I report wrong-answer
   and decline rates beside the metric, so the trade-off stays visible.
4. **Label ambiguity caps accuracy.** The answering adapter scores 89.6 when 7+ of
   10 annotators agree and 37.2 when at most 3 do. A third of test questions are
   low-agreement, so part of the remaining error is not fixable by the model.
5. **The plan was wrong about the data.** It assumed a third answerability class
   ("unsuitable image") that this VizWiz release does not have. Answerability
   became 2-way, and unusable photos are handled by the quality labels (R-005).
6. **Infrastructure.**
   - vLLM crashed at startup because a JIT build tool was not on `PATH` inside an
     unactivated venv (R-002).
   - Colab already uses port 8080.
   - Colab sessions end at random, so training saves a checkpoint every 10
     minutes and resumes automatically. Three of the four training runs resumed
     mid-run with no lost work.

## 6. Decisions I chose not to make

- **No QLoRA.** bf16 LoRA fits on the L4 (10 GB peak), and QLoRA would have cost
  speed for no needed memory (D-053).
- **No hyperparameter sweep or second training run.** The validation loss showed
  no sign of overfitting, and there was no evidence another run would beat the
  confidence intervals. The GPU time went to E5, which answered an architecture
  question instead (D-053).
- **4-bit quantization deferred.** FP8 needs no calibration and was tried first;
  the compute-bound result makes 4-bit a lower priority (D-057).

## 7. Limitations

- One training seed per configuration. Differences under about 1 point
  (E5 vs E2, FP8) are reported with their CIs, not as wins.
- Tested only on VizWiz-style photos and English questions. Reading text (58.1)
  and counting (53.1) are the weakest skills, and rare photo problems
  (obstruction, dark, bright) are detected less reliably (F1 0.45–0.58).
- About 1 in 7 questions still gets a wrong answer (30% of the answers given).
  **This is a research prototype, not a safety device.**
- The load generator ran on the same VM as the server. Costs assume a fully busy
  GPU at on-demand prices.

## 8. What I would do next

1. **Process the photo once per request, not three times.** Putting the photo
   before the task prompt would let the three stages share a cached prefix in vLLM.
   This needs retraining with that prompt order, and it targets the measured
   bottleneck.
2. **Improve text reading**, the most common and most important question type, for
   example with an OCR tool the model can call.
3. **Calibrate with users.** The right decline rate is a product question for blind
   users themselves, not only a metric question.
4. **Compare against a frontier model** on val as a reference point. This is
   deferred because it would send dataset photos to a third-party API (D-049).

## 9. Reproducing

Each GPU step is a Colab notebook that pins the environment, checks data hashes,
resumes after a disconnect, and records the git commit and package versions in
its results. The local stack (`docker compose up`) runs the gateway, a mock model,
Prometheus and Grafana without a GPU; CI tests it on every push. See the
[README](../README.md#reproduce).
