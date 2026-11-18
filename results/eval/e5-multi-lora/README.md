# E5: one multi-task adapter vs three specialists (Step 3.6, D-055, D-056)

Qwen3-VL-4B-Instruct plus one bf16 LoRA adapter (r 16) trained for 1 epoch on the
union of all three tasks' training data (51,626 examples). Each example keeps its
task's system prompt, so the same prompts select the task at inference. Evaluated
on the full test sets, paired against the specialist adapters (E2) and the
untrained model (E1) on the same examples.

## Headline
| Task (metric) | E1 untrained | E2 specialist | E5 multi-task | E5 vs E2 [95% CI] |
|---|---|---|---|---|
| Answering (VQA accuracy, n 2,934) | 53.52 | 61.73 | **62.56** | +0.83 [+0.18, +1.53] |
| Answerability (AP, n 4,319) | 69.02 | 89.83 | **90.00** | +0.17 [-0.37, +0.70] |
| Quality (macro-F1, n 7,750) | 27.95 | 65.20 | **65.59** | +0.40 [-0.29, +1.10] |

By the rule fixed before the run (D-055), E5 **matches** the specialist on all
three tasks, and is slightly better on answering (the CI lies above 0). The
gain is under 1 point, so the honest summary is "one adapter instead of three,
with no loss".

## Other numbers (E2 → E5)
- Answerability: recall at precision 0.9 63.1 → 64.6; at 0.8 85.6 → 84.0;
  F1 at threshold 0.5 80.9 → 81.5. Invalid replies 0%.
- Quality: exact match 50.5% → 50.3%; valid JSON 100%. Per-label F1 within
  ±0.02 of the specialist (unrecognizable 0.73).
- Answering by answer type: yes/no 89.0, number 75.0, other 60.4. Clean photos
  69.3 vs photos with an issue 60.3.

## Training (`results/train/e5-multi-lora/`)
- 3,227 optimizer steps on one L4 across 3 Colab sessions with automatic resume
  (last session resumed from step 2,184). About 11.4 s per step, ~10 h in total,
  the same compute as the three specialists together.
- Train loss 0.220, final val loss 0.222. Peak memory 10.1 GB.

Inference: ~7 examples/s per task, the same as the specialists. The latency
side of the comparison (one adapter vs switching between three) is measured in
the serving benchmark (Phase 5).
