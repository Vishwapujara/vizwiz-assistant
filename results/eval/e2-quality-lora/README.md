# E2: quality adapter vs untrained model (Step 3.3, D-052)

Qwen3-VL-4B-Instruct plus a bf16 LoRA adapter (r 16) trained for 1 epoch on 20,525
training photos. Evaluated on the full test set (7,750 photos): the model lists the
photo's problems as JSON. Same pipeline and photos as E1.

## Headline
| | E1 untrained | E2 adapter |
|---|---|---|
| Macro-F1 over 7 issue labels | 27.95 | **65.20** [63.80, 66.56] |
| Exact match (all labels right) | 4.8% | 50.5% |
| Valid JSON | 100% | 100% |

Paired difference on the same 7,750 photos: **+37.2 [+35.9, +38.7]**
(`compare_quality.json`).

## Per label (F1; precision and recall for E2)
| Label | Photos with it | E1 F1 | E2 F1 | E2 precision | E2 recall |
|---|---|---|---|---|---|
| framing | 4,348 | 0.41 | 0.82 | 0.80 | 0.84 |
| blur | 3,279 | 0.60 | 0.81 | 0.83 | 0.79 |
| unrecognizable | 1,231 | 0.01 | **0.74** | 0.84 | 0.66 |
| rotation | 1,325 | 0.24 | 0.65 | 0.76 | 0.57 |
| dark | 476 | 0.14 | 0.55 | 0.69 | 0.45 |
| bright | 476 | 0.35 | 0.55 | 0.72 | 0.44 |
| obstruction | 267 | 0.21 | 0.45 | 0.71 | 0.33 |

- **Unrecognizable** went from almost never detected (0.01) to 0.74. It is the label
  that matters most for the safety goal (the photo shows nothing usable).
- The rare labels (dark, bright, obstruction: 3–6% of photos) are the weakest, and
  their errors are mostly misses (recall 0.33–0.45), not false alarms. Training
  has fewer examples of them.
- Labels come from 5 annotators per photo, kept when at least 2 agree (Step 1.4,
  quality k = 2), so some disagreement is built into the ground truth.

Inference with the adapter: 7.1 photos/s.

## Training (`results/train/e2-quality-lora/`)
- 1,283 optimizer steps on one L4. **Colab disconnected once**: the run resumed
  from the step-764 checkpoint and finished the remaining 519 steps (1.8 h).
- Train loss 0.203, final val loss 0.178: no sign of overfitting.
- Peak memory 10.1 GB.
