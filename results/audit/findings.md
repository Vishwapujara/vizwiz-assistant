# Audit findings (Step 1.3)

What the numbers in [`audit_report.md`](audit_report.md) mean for the project.
Generated from annotations only; image-size statistics follow in Step 1.5.

## 1. Unanswerable questions are common, so the decline path matters
- 27% of train/test questions and 32% of val questions are officially unanswerable.
- A model that always tries to answer gets roughly a quarter of the dataset wrong by
  construction. This is the main motivation for the answerability adapter.
- Our test set (official val, D-022) has 5 points more unanswerable questions than
  train. Our val is carved from train, so it has train's 27%. A decline threshold
  tuned on val will meet a slightly different class balance at test time. Report
  per-class precision and recall, not only overall accuracy, so this shift stays visible.

## 2. The official "answerable" flag is a vote count
- In train, 4,896 of the 5,532 unanswerable questions (89%) have 6–10 of 10
  "unanswerable" answers; the other 636 have 3–5.
- Answerable questions have 0–5. 2,505 train questions (12.2%) are "answerable" even
  though 3–5 annotators said "unanswerable": these are borderline cases.
- We use the official flag as the label and report borderline cases as a slice.

## 3. No "unsuitable image" answer exists, so the 3-way label needs the quality data
- The only special answer string in this VizWiz version is `unanswerable`.
- The plan's third class, `unsuitable_image`, must come from the quality annotations'
  `unrecognizable` votes. At k = 2, 31% of unanswerable train questions
  (28% of val) have an image that ≥ 2 of 5 annotators called unrecognizable.
- That would make `unsuitable_image` about 8% of all questions. Workable, but small.
  Decided in Step 1.4.

## 4. Image quality strongly predicts unanswerability
- If ≥ 2 annotators call the image unrecognizable, 56% of its questions are
  unanswerable (val 63%), against 22% (val 27%) otherwise.
- With no quality issue at k = 2, only 11% are unanswerable (val 14%).
- This supports the cascade design: checking quality first catches a large share of
  unanswerable questions before the answerability adapter runs.

## 5. Quality issues: framing and blur dominate; some classes are rare
At k = 2 (train): framing 55%, blur 41%, rotated 17%, too dark 6%, too bright 5%,
obstruction 4%, other 0.8%; unrecognizable 15%; 77% of images have at least one issue.
- `OTH` (other) is under 1% at k ≥ 2, too rare and too vague to learn or to map to
  useful guidance. Candidate to drop as an output class (Step 1.4).
- k = 3 (a majority of 5) makes obstruction, too bright, and too dark 1.7–2.8%.
- **Recommendation: keep k = 2**, as planned. It is the lowest threshold that needs
  more than one annotator, and it keeps the rare classes learnable.

## 6. Annotators often disagree; the ceiling is 94%
- In 10% of questions, no answer gets 3 votes, and 11% have a tied top answer.
- Even a perfect model scores at most about 94% official VQA accuracy (the oracle).
  Results are reported against this ceiling, not against 100%.

## 7. Answers are short
- The median answer to an answerable question is 1 word. The prompt ("answer in a
  few words") and generation settings (small `max_new_tokens`) fit this.

## 8. The official normalisation does not match 0.91% of stored answers
- 444 of 48,727 unique answers change under the official normalisation. The metric
  compares a normalised prediction with raw ground truth, so those answers can never
  be matched verbatim.
- Training targets are therefore chosen as the candidate answer with the highest
  official accuracy (Step 1.4), not simply the most frequent string.
- Our metric port matches the official code exactly: 151,181 normalisation probes
  and 3,000 accuracy checks, 0 mismatches.

## 9. Images: mostly 1–2 MP portrait phone photos; 7–9% need EXIF rotation
Source: `results/splits/image_stats.json` (all 39,704 images, Colab CPU scan).
- 99% are portrait once upright. The two most common sizes, 1224×1632 and 968×1296,
  cover about 75% of train. Median 1.3 MP; p95 3.1 MP (5.0 MP in test_vqa).
- 6–9% per split store the photo sideways with EXIF orientation 6. A loader that
  ignores EXIF would feed these to the model rotated 90°, so preprocessing always
  applies it (D-035).
- A few files are tiny (smallest 10×13 px). Preprocessing never makes a side smaller
  than 32 px.
- 1,017 groups of byte-identical files exist, some across the official splits. 605
  train-pool copies of test images are excluded from training (R-006, D-038).

## 10. Image size sets the sequence length; text is a rounding error
Source: `results/data/token_stats.json` (200 val examples per task, Qwen3-VL processor).

| max_pixels | visual tokens (p50) | total tokens (p50, answer task) |
|---|---|---|
| 262,144 (≈512²) | 234 | 269 |
| 524,288 | 494 | 529 |
| 1,048,576 | 972 | 1,010 |
| native | 1,938 (max 4,860) | 1,971 |

- Prompt text plus target is only about 35–65 tokens. Targets are short: answer p50
  4 tokens (p95 7), answerability 4–5, quality p50 6 (p95 14), each including the
  end-of-turn tokens.
- Native resolution would cost 7–20× the tokens of the 262k budget, with a long tail
  (up to 4,860). A fixed budget makes memory predictable: this is the input for the
  image-resolution decision in Step 3.1.
- All 2,400 encodings passed the prompt-prefix check, so loss masking is exact.
- 12 of 2,400 encodings had a visual-token count that differs from
  `(w/32)·(h/32)`. The counts are 4 (answer) and 8 (quality): one and two images
  at all four budgets, which fits tiny images that the processor scales up to its
  minimum size. The formula is only used for estimates, so this has no effect on
  training.

