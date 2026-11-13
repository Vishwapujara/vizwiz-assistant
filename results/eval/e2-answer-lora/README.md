# E2: answering adapter vs untrained model (Step 3.2, D-051)

Qwen3-VL-4B-Instruct plus a bf16 LoRA adapter (r 16) trained for 1 epoch on 13,134
answerable training questions. Evaluated on the full test set (2,934 answerable
questions), 524,288 px, greedy decoding, vLLM 0.32.0 on one L4. Same pipeline and
questions as E1.

## Headline
| | VQA accuracy | 95% CI |
|---|---|---|
| E1, untrained | 53.52 | [51.9, 55.2] |
| **E2, answering adapter** | **61.73** | [60.19, 63.20] |
| **Difference (paired, same questions)** | **+8.21** | **[+6.98, +9.34]** |

## By answer type
| Type | n | E1 | E2 |
|---|---|---|---|
| yes/no | 195 | 85.4 | 90.0 |
| number | 48 | 51.3 | 69.4 |
| other | 2,691 | 51.3 | 59.6 |

The "number" slice has only 48 questions, so its gain is not precise.

## By photo quality (a photo can have several labels)
| Label | n | E1 | E2 | Change |
|---|---|---|---|---|
| none (clean) | 731 | 60.1 | 68.1 | +8.0 |
| any issue | 2,203 | 51.3 | 59.6 | +8.3 |
| blur | 1,051 | 50.0 | 57.7 | +7.7 |
| framing | 1,595 | 50.7 | 59.4 | +8.7 |
| rotation | 533 | 47.6 | 54.7 | +7.1 |
| unrecognizable | 225 | 48.8 | 59.2 | +10.4 |
| dark | 116 | 49.3 | 55.2 | +5.9 |
| bright | 122 | 55.4 | 63.3 | +7.9 |
| obstruction | 59 | 59.5 | 64.9 | +5.4 |

The gain is about the same on clean and flawed photos, so the 8-point gap between
them remains: training taught the answer style and vocabulary, not seeing through
bad photos. Bad photos are the job of the answerability and quality adapters
(Step 3.3).

## Training (`results/train/e2-answer-lora/`)
- 821 optimizer steps, 2.8 h on one L4, no disconnects, peak memory 10.1 GB.
- Train loss 0.521, final val loss 0.487 (500 val questions): no sign of
  overfitting after 1 epoch.
- Inference with the adapter: 7.2 examples/s against about 8 without it.
- Declined answers on these answerable questions: 0.0% (E1 0.07%).
