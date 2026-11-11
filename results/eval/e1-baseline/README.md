# E1: zero-shot baseline (Step 2.3)

Untrained `Qwen/Qwen3-VL-4B-Instruct`, bf16, vLLM 0.32.0 on an L4, greedy decoding,
images at max 524,288 pixels (D-041), training prompts (D-036). Test split = official
VizWiz val, every example. Run at commit `77dd18e`; full numbers in `metrics.json`.
95% bootstrap CIs in brackets.

## Headline

| Task | n | Metric | E1 |
|---|---|---|---|
| Answering (answerable questions) | 2,934 | VQA accuracy | **53.5** [51.9, 55.2] |
| Answerability (positive = unanswerable) | 4,319 | average precision | **69.4** [66.8, 72.4] |
| Image quality (7 issue labels) | 7,750 | macro-F1 | **28.0** [26.9, 28.9] |

Answering accuracy is on answerable questions only (the answering adapter's job,
D-033), so it is not directly comparable with leaderboard numbers that score every
question, unanswerable ones included.

## Answering by slice

| Slice | n | VQA accuracy |
|---|---|---|
| yes/no | 195 | 85.4 |
| number | 48 | 51.3 |
| other | 2,691 | 51.3 |
| no quality issue | 731 | **60.1** |
| any quality issue | 2,203 | **51.3** |
| rotation | 533 | 47.6 |
| unrecognizable | 225 | 48.8 |
| dark | 116 | 49.3 |
| blur | 1,051 | 50.0 |
| framing | 1,595 | 50.7 |

Photos with a quality issue cost about 9 points of accuracy. Slices overlap: a photo
can have several issues.

## Answerability

| Metric | E1 |
|---|---|
| Average precision (official) | 69.4 (base rate of unanswerable: 32.1%) |
| Official F1 at score > 0.5 | 69.5 |
| Recall of unanswerable at precision 0.8 / 0.9 | 12.5 / 0.0 |
| Replies that were neither label | **36.6%** |
| Borderline answerable (3–5 "unanswerable" votes) called unanswerable | 45.8% of 550 |

- The base model **does not follow the one-word format** on answerable questions:
  1,511 of 2,934 answerable questions got a reply that was neither label, and
  1,503 of all 1,580 invalid replies were of kind `other`, which is usually a
  direct answer to the question.
- It cannot flag unanswerable questions confidently: at 90% precision it catches
  none. This matters for the safety goal (do not answer when the photo cannot
  support an answer).
- The score was read from logprobs for all but 7 of 4,319 questions.

## Image quality

| Label | support | precision | recall | F1 |
|---|---|---|---|---|
| blur | 3,279 | 0.43 | 0.97 | 0.60 |
| framing | 4,348 | 0.56 | 0.33 | 0.41 |
| bright | 476 | 0.44 | 0.29 | 0.35 |
| rotation | 1,325 | 0.35 | 0.18 | 0.24 |
| obstruction | 267 | 0.14 | 0.48 | 0.21 |
| dark | 476 | 0.07 | 0.96 | 0.14 |
| unrecognizable | 1,231 | 0.07 | 0.00 | 0.01 |

- Replies were valid JSON every time, but the label set was exactly right for only
  4.8% of photos.
- The model calls almost everything blurry and very often dark (recall near 1,
  low precision), and almost never says a photo is unrecognizable, even though 16%
  of test photos are.

## Speed (L4, offline batch of 64)

About 8 examples/s for every task (7.6–8.2), engine start-up 95 s once the model is
cached. Batch throughput only; per-request latency is measured in Phase 5/6.
