# Serving check: real model behind the gateway (Step 5.1, D-060, D-063, D-064)

E5 adapter (bf16) on a vLLM OpenAI-compatible server on one L4, behind the FastAPI
gateway. Every val (2,000) and test (4,319) VQA question was replayed through the
served path: the photo prepared exactly as the gateway does (524,288 pixels, JPEG
quality 95), and all three cascade stages called on the server. `tune_report.json`
holds aggregate numbers only; per-question rows stay on Drive.

## 1. Does serving change the model's replies? (val)
| Stage | Same output as offline eval | Bar (D-060) | Accuracy, served − offline [95% CI] |
|---|---|---|---|
| Answerability | 98.8% same decision at 0.5 | ≥ 98% ✅ | AP +0.08 [−0.18, +0.37] |
| Answer | 92.2% same answer | ≥ 95% ❌ | VQA accuracy −0.59 [−1.41, +0.18] |
| Quality | 90.1% same labels | ≥ 90% ✅ | macro-F1 −0.38 [−1.39, +0.65] |

The answer bar is missed: re-encoding the photo as JPEG changes greedy decoding
on near-ties. The differences cost no measurable accuracy (every CI includes 0).
Sending the photo losslessly (PNG) is tested at Step 6.1.

## 2. The whole cascade (test, threshold chosen on val)
Official VizWiz accuracy, where a decline counts as the answer "unanswerable".

| | Always answer | Threshold 0.5 | **Served: 0.10, tips only** | 0.10 + hard retake gate |
|---|---|---|---|---|
| VizWiz accuracy | 47.10 | 70.87 | **77.73** | 77.04 |
| Questions answered | 100% | 67.3% | 49.4% | 48.1% |
| Wrong answers (share of all questions) | 52.8% | 26.4% | **14.8%** | 14.6% |
| Answerable questions declined | 0% | 10.1% | 28.6% | 30.4% |
| Unanswerable questions answered | 100% | 19.5% | 2.7% | 2.5% |

- **Threshold 0.10** won on val by the pre-set rule (val 75.6 vs 70.2 at 0.5),
  and also without the retake gate (val 77.2).
- The metric makes declining cheap: "unanswerable" earns credit whenever ≥ 3 of 10
  annotators said so, which is common even on answerable questions. 0.10 fits the
  safety goal (wrong answers 53% → 15%) at the cost of declining often.
- **Retake gate dropped (D-064):** forcing "please retake" on `unrecognizable`
  photos lowered accuracy (val −1.6, test −0.7) and mostly declined answerable
  questions. Quality now gives tips only; the gate is one config line away.

Full val threshold sweep (0.05–0.95) in `tune_report.json` (`val.sweep`).
