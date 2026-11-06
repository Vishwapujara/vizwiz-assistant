# Data sources, licenses, and terms (Step 1.1)

Checked 2026-11-06 against the official pages at vizwiz.org and the files on the
official download server (`vizwiz.cs.colorado.edu`). Only annotation files were
downloaded for this check (to a scratch folder, not the repo). No images were downloaded.

## Datasets used

| Dataset | Official page | License stated on page | Used for |
|---|---|---|---|
| VizWiz-VQA (v. 2020-01-01) | https://vizwiz.org/tasks-and-datasets/vqa/ | CC BY 4.0 | Answering and answerability adapters |
| VizWiz image quality issues | https://vizwiz.org/tasks-and-datasets/image-quality-issues/ | **No license statement on the page** (see note) | Quality adapter |
| VizWiz images (shared by all VizWiz tasks) | same pages | CC BY 4.0 (VQA and Captions pages) | All tasks |

**Note on the quality annotations:** the image quality page releases the annotations
"for self evaluation" and states no license. The images they describe are the same
CC BY 4.0 VizWiz images. We treat the quality labels under the same terms: research and
portfolio use, with attribution, and no redistribution of the annotation files. Before
any commercial use, ask the dataset authors.

**What CC BY 4.0 requires of us:** credit the creators, link the license, and say if we
changed anything. It allows copying, adapting, and commercial use. The README carries
the attribution (below).

**Registration:** none. All files download directly over HTTPS.

## Files to download (Step 1.2)

| File | URL | Size |
|---|---|---|
| Train images | https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip | 11.30 GB |
| Val images | https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip | 3.49 GB |
| Test images | https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip | 3.98 GB |
| VQA annotations (train/val/test) | https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip | 1.7 MB |
| VQA test answers | https://vizwiz.cs.colorado.edu/VizWiz_all_answers/VQA_test.json | 8.2 MB |
| Quality annotations (train/val/test) | https://vizwiz.cs.colorado.edu/VizWiz_final/image_quality/annotations.zip | 0.3 MB |
| Official VQA API + eval code | https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/API.zip | (Step 2.1) |

Total ≈ **18.8 GB**, almost all images. Drive has ~224 GB free (D-013).

**Not used:** `VizWiz_all_answers/VizWiz_quality_issues_train_val_test.csv`. It holds
raw per-worker rows with crowd-worker IDs, uses an older image naming scheme
(`VizWiz_train_000000000000.jpg`, `VizWiz_v2_...`), and splits images differently from
the JSON files (26,949 / 4,271 / 7,993 vs 23,431 / 7,750 / 8,000).

## What the files contain (verified)

**One shared image pool.** The VQA and Captions pages link the *same* image zips. The
quality annotations cover all 39,181 images (23,431 train / 7,750 val / 8,000 test).
VQA questions exist for a subset of them.

**VQA** (`Annotations.zip` → `train.json`, `val.json`, `test.json`; plus `VQA_test.json`):

| Split | Questions | Fields | Answers public? |
|---|---|---|---|
| train | 20,523 | image, question, answerable (0/1), answer_type, answers[10] | yes |
| val | 4,319 | same | yes |
| test | 8,000 | `test.json`: image, question only. `VQA_test.json`: all fields | **yes, via `VQA_test.json`** |

Each `answers` entry holds `answer` (string) and `answer_confidence` (yes/no/maybe).
One question per image. File names look like `VizWiz_train_00000000.jpg`.

**Quality** (`annotations.zip` → `train.json`, `val.json`, `test.json`):
- Fields: `image`, `flaws` (counts 0–5 per issue: BLR blur, BRT too bright, DRK too
  dark, FRM framing, OBS obstruction, ROT rotated, OTH other, NON no issue), and
  `unrecognizable` (count 0–5). There are 5 annotators per image.
- The test split has **no labels** (image names only).
- Same naming scheme as VQA.

**Overlap between the tasks, by exact file name:**
- VQA train ∩ quality train: 20,000 of 20,523. The other 523 VQA train images have no
  quality labels.
- VQA val ∩ quality val: 4,319 of 4,319.
- VQA test ∩ quality test: 8,000 of 8,000.
- So the official splits already keep each image in a single split across tasks.

## Evaluation notes

- **VQA accuracy:** for each question, `min(#humans who gave the answer / 3, 1)`,
  averaged over all 10 subsets of 9 annotators. The page's prose and its figure
  describe this slightly differently, so Step 2.1 reuses the official code in `API.zip`.
- **Answerability (official):** average precision on the "answerable" confidence, via
  scikit-learn `average_precision_score`.
- **Evaluation servers (EvalAI) are deprecated.** Annotations are released for
  self-evaluation.

## Privacy rules for this data (from PROJECT_PLAN Section 11)

- Never commit or publish images, and never commit the annotation files themselves.
- Committed reports contain aggregate statistics only. No verbatim questions or
  answers from the dataset: blind users' questions can describe private content.
- The worker-ID CSV is not downloaded at all.

## Attribution (for the README)

> This project uses the VizWiz datasets (https://vizwiz.org), licensed under
> CC BY 4.0. VizWiz-VQA: D. Gurari et al., "VizWiz Grand Challenge: Answering Visual
> Questions from Blind People," CVPR 2018. Image quality issues: T. Chiu et al.,
> "Assessing Image Quality Issues for Real-World Problems," CVPR 2020. Labels are
> derived from the original annotations (see `src/vizwiz/data/labels.py`).
