# E4: E5 adapter on an FP8 base vs bf16 (Steps 4.1–4.2, D-057, D-058)

The base model's weights are cast to FP8 by vLLM at load time (dynamic scales,
no calibration). The E5 multi-task adapter stays bf16 and is attached unmerged.
Same L4, same pipeline, same test examples as E5 in bf16.

## Accuracy (full test sets, paired against bf16)
| Task (metric) | bf16 | FP8 | FP8 − bf16 [95% CI] | Rule (lower end > −1) |
|---|---|---|---|---|
| Answering (VQA accuracy, n 2,934) | 62.56 | 62.25 | −0.31 [−0.83, +0.17] | pass |
| Answerability (AP, n 4,319) | 90.00 | 90.01 | +0.01 [−0.20, +0.22] | pass |
| Quality (macro-F1, n 7,750) | 65.59 | 64.98 | −0.62 [−1.16, −0.04] | **fail (narrowly)** |

- Quality drops by a small but real amount (the CI excludes 0). Most of it is the
  rare **bright** label (F1 0.58 → 0.53, 476 photos); **unrecognizable**, the
  label that matters most for safety, is unchanged (0.73 → 0.74).
- Answerability: recall at precision 0.9 64.6 → 63.5; invalid replies 0%.
  Quality JSON valid 100%.

## Speed and memory (`speed_memory.json`: same 1,000 answering questions, same session)
| | bf16 | FP8 | Change |
|---|---|---|---|
| Weights in GPU memory | 8.6 GiB | 5.05 GiB | −41% |
| Room for requests in progress (KV cache) | 6.72 GiB | 10.42 GiB | +55% |
| Requests of 4,096 tokens that fit at once | 11.9 | 18.5 | +55% |
| Photos per second (batch 64, offline) | 7.15 | 8.12 | +14% |

Full test runs with FP8 gave 8.0–8.2 photos/s per task (bf16: 7.0–7.3).

## Verdict (rule fixed before the run, D-057)
FP8 fails the pre-registered rule on quality by 0.16 points (CI lower end −1.16
vs the −1.0 margin), so **serving stays bf16 by default** (D-058). FP8 is kept
as a serving option (`--quantization fp8`) for high load: about 55% more
concurrent requests and 14% more throughput, at a cost of about 0.6 macro-F1 on
photo-quality labels and no measurable change in answering or answerability.
