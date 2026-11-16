# Error analysis, part 1: what fine-tuning fixed and what remains (Step 3.4, D-054)

Automatic breakdown of E1 (untrained) vs the three E2 adapters on the same test
examples. Full tables: `errors_test.md` (`errors_test.json` for the numbers).
Counts and rates only. Part 2 (manual review of failures, on val) is at Step 5.2.

## Answering (2,934 questions)
- **Net gain from training:** 348 questions fixed, 99 broken (net +249).
- **The largest single factor is human agreement, not the model.** When 7+ of the
  10 answerers gave the same answer, the adapter scores 89.6; when at most 3 did,
  37.2. A third of the test questions (967) are low-agreement, often because the
  question is ambiguous or the photo hard to read. Part of the remaining error is
  label noise that no model can remove.
- **By question kind:** colour is easiest (76.5), counting and numbers hardest
  (53.1, n = 83), reading text 58.1 (n = 327).
- **Remaining errors (1,097):** 357 got credit from exactly one answerer (near
  misses or alternative wordings); 725 matched no answerer; only 15 are yes/no
  flips. Answers got shorter (mean 1.98 → 1.57 words; over 3 words 10.2% → 4.1%),
  close to the targets' 1.78, so verbosity is no longer the problem.
- Photo issues cost about 8 points before and after training (see E2 README).

## Answerability (4,319 questions)
- **Net gain:** 652 decisions fixed, 216 broken.
- **Wrong refusals almost disappeared on clear-cut questions.** With 0 of 10
  "unanswerable" votes: 10.4% → 0.9% wrongly declined; 1–2 votes: 25.2% → 6.2%.
- **Errors concentrate on borderline questions** (3–5 of 10 votes): 24.6% of the
  answerable ones are still declined and 54.4% of the unanswerable ones (n = 160)
  are missed. Here the annotators themselves were split.
- **Clean-looking photos hide unanswerable questions.** On photos with no quality
  issue, 30.3% of unanswerable questions are missed (n = 119), against 21.6% on
  photos with issues: the problem there is not visible in photo quality (e.g. the
  object asked about is not in view).
- **Trade-off at the 0.5 cut-off:** E1 declined far more often, so it missed fewer
  unanswerable questions on some slices (unrecognizable photos 6.2% → 12.3%
  missed; dark 9.9% → 14.9%) while wrongly declining 3–4× as many answerable ones.
  The cut-off is tuned on val in Step 5.2, and the quality adapter's
  "unrecognizable" label (F1 0.74) can back up the answerability check on exactly
  these photos.

## Quality (7,750 photos)
- **False alarms on clean photos:** 91.6% → 24.9% of clean photos flagged with an
  issue. **Photos where every issue was missed:** 25.0% → 14.0%.
- **Most mistakes are misses, not mix-ups.** When the adapter misses a label it
  usually reports nothing extra instead (e.g. blur missed: 630 times nothing
  extra, 59 times framing instead). The adapter is cautious; it rarely confuses one
  problem for another.
- **More issues, harder photo:** exact match 75.1% with 0 issues, 56.4% with 1,
  38.4% with 2, 28.4% with 3 or more. Labels come from 5 annotators with a
  2-vote threshold, so multi-issue photos also have the noisiest labels.

## What this means for the next steps
1. **Cascade (Step 5.2):** combine both safety adapters: decline when the
   answerability score is high **or** the quality adapter says "unrecognizable".
   Tune the cut-off on val, and report wrong refusals and misses separately.
2. **Borderline questions** are the hard core; report them as their own slice in
   the final write-up instead of hiding them in the average.
3. **No further answering-adapter tuning** (D-053): much of the remaining error is
   human disagreement or a photo problem.
