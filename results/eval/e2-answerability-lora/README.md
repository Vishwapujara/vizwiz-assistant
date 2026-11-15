# E2: answerability adapter vs untrained model (Step 3.3, D-052)

Qwen3-VL-4B-Instruct plus a bf16 LoRA adapter (r 16) trained for 1 epoch on 17,967
training questions. Evaluated on the full test set: 4,319 questions, of which
1,385 are unanswerable (the positive class). Score = P(unanswerable) from the first
generated token (D-040). Same pipeline and questions as E1.

## Headline
| | E1 untrained | E2 adapter |
|---|---|---|
| Average precision (official) | 69.38 | **89.85** [88.27, 91.14] |
| Recall at precision 0.8 | 12.5 | **85.6** |
| Recall at precision 0.9 | 0.0 | **63.1** |
| F1 at 0.5 (official) | 69.5 | 80.9 |
| Invalid replies | 36.6% | 0.0% |

Paired difference in AP on the same 4,319 questions: **+20.8 [+18.5, +22.9]**
(`compare_answerability.json`; there E1 is 69.02 and E2 89.83, because the
prediction files rounded scores to 6 decimals, which creates ties: R-009).

**In words:** when the assistant is allowed to be wrong at most 1 time in 10 when
it says "I can't answer this", it now catches 63% of the unanswerable photos;
the untrained model caught none.

## Mistakes at the 0.5 cut-off
| True \ predicted | answerable | unanswerable |
|---|---|---|
| answerable (2,934) | 2,711 | 223 (7.6% wrongly declined) |
| unanswerable (1,385) | 284 (20.5% missed) | 1,101 |

The cut-off used by the assistant is tuned on val in Step 5.2, not fixed at 0.5.

**Borderline questions** (550 answerable questions where 3–5 of 10 annotators
said unanswerable): predicted unanswerable 27.3%, against 45.8% for E1.

## Training (`results/train/e2-answerability-lora/`)
- 1,123 optimizer steps on one L4. **Colab disconnected once**: the run resumed
  from the step-959 checkpoint and finished the remaining 164 steps (35 min).
- Train loss 0.055, final val loss 0.053: no sign of overfitting.
- Peak memory 10.1 GB; inference with the adapter 7.5 examples/s.
