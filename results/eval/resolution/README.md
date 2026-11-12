# Image-budget check (Step 3.1, D-047 → D-050)

Untrained Qwen3-VL-4B-Instruct on the **val** split (1,464 answerable questions for
answering, 2,000 questions for answerability). Paired bootstrap 95% CI of the
difference against the 524,288-pixel run, on the same questions.

| Budget (px) | Visual tokens | Answering VQA acc | Δ vs 524k (95% CI) | Answerability AP | Δ vs 524k (95% CI) |
|---|---|---|---|---|---|
| 262,144 | ~234 | 54.99 | [-2.11, +0.29] | 64.96 | [-1.17, +1.46] |
| 524,288 | ~494 | 55.87 | reference | 64.77 | reference |
| 1,048,576 | ~972 | 56.07 | [-0.92, +1.30] | 65.69 | [-0.16, +2.12] |

Rule fixed before the run: the smallest budget within 1.0 point of the best answering
accuracy and within 1.5 points of the best AP. 262k fails on answering (1.08 behind);
**524k passes** (0.20 and 0.92 behind). All CIs contain 0.

Speed at 1M during this run: about 4 examples/s on the L4, against about 8 at 524k
in E1.
