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
