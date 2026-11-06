"""Step 1.3 data audit: statistics from the VizWiz annotation files.

    python -m vizwiz.data.audit --annotations <dir> --out results/audit

``<dir>`` holds the extracted annotations:
    vqa/train.json, vqa/val.json, vqa/VQA_test.json, quality/train.json, quality/val.json

Writes ``audit.json`` (every number) and ``audit_report.md`` (tables). Both contain
aggregate statistics only: no questions, no answers (except the label word
"unanswerable"), no image names. Image-size statistics need the images and are added
by the Colab data-prep run (Step 1.5).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

from vizwiz.eval.vqa_metric import normalize_answer, vqa_accuracy

FLAWS = ["BLR", "BRT", "DRK", "FRM", "OBS", "ROT", "OTH", "NON"]
FLAW_NAMES = {
    "BLR": "blur",
    "BRT": "too bright",
    "DRK": "too dark",
    "FRM": "bad framing",
    "OBS": "obstruction",
    "ROT": "rotated",
    "OTH": "other",
    "NON": "no issue",
}
K_VALUES = (1, 2, 3)
UNANSWERABLE = "unanswerable"


def pct(n: int, d: int) -> float:
    return round(100 * n / d, 2) if d else 0.0


def load_annotations(root: Path) -> tuple[dict[str, list], dict[str, list]]:
    vqa = {
        "train": json.loads((root / "vqa" / "train.json").read_text()),
        "val": json.loads((root / "vqa" / "val.json").read_text()),
        "test": json.loads((root / "vqa" / "VQA_test.json").read_text()),
    }
    quality = {
        "train": json.loads((root / "quality" / "train.json").read_text()),
        "val": json.loads((root / "quality" / "val.json").read_text()),
    }
    return vqa, quality


def oracle_accuracy(answers: list[str]) -> float:
    """Best achievable official accuracy: score every distinct human answer, keep the best."""
    return max(vqa_accuracy(a, answers) for a in set(answers))


def vqa_stats(entries: list[dict]) -> dict:
    n = len(entries)
    answerable = sum(e["answerable"] for e in entries)
    types = Counter(e["answer_type"] for e in entries)
    conf = Counter(a["answer_confidence"] for e in entries for a in e["answers"])
    q_words = [len(e["question"].split()) for e in entries]

    top_share, unans_votes, ties, oracle = [], [], 0, []
    ans_words = []
    for e in entries:
        answers = [a["answer"] for a in e["answers"]]
        counts = Counter(answers).most_common()
        top_share.append(counts[0][1])
        if len(counts) > 1 and counts[1][1] == counts[0][1]:
            ties += 1
        unans_votes.append(answers.count(UNANSWERABLE))
        oracle.append(oracle_accuracy(answers))
        if e["answerable"]:
            ans_words.append(len(counts[0][0].split()))

    agree = Counter(top_share)
    return {
        "questions": n,
        "answerable": answerable,
        "answerable_pct": pct(answerable, n),
        "unanswerable": n - answerable,
        "unanswerable_pct": pct(n - answerable, n),
        "answer_type": {t: {"n": c, "pct": pct(c, n)} for t, c in types.most_common()},
        "answer_confidence_pct": {k: pct(v, sum(conf.values())) for k, v in conf.most_common()},
        "question_words": {
            "mean": round(mean(q_words), 1),
            "median": median(q_words),
            "max": max(q_words),
        },
        "top_answer_words_answerable": {
            "mean": round(mean(ans_words), 2),
            "median": median(ans_words),
            "pct_1_to_3_words": pct(sum(1 for w in ans_words if w <= 3), len(ans_words)),
        },
        "agreement": {
            "top_answer_votes_distribution": {str(k): agree[k] for k in sorted(agree)},
            "pct_top_answer_ge3_votes": pct(sum(1 for s in top_share if s >= 3), n),
            "pct_top_answer_ge5_votes": pct(sum(1 for s in top_share if s >= 5), n),
            "pct_tied_top_answer": pct(ties, n),
        },
        "oracle_vqa_accuracy_pct": round(100 * mean(oracle), 2),
        "unanswerable_votes_distribution": dict(sorted(Counter(unans_votes).items())),
    }


def normalisation_stats(vqa: dict[str, list]) -> dict:
    answers = {a["answer"] for split in ("train", "val") for e in vqa[split] for a in e["answers"]}
    changed = sum(normalize_answer(a) != a for a in answers)
    return {
        "unique_gt_answers_train_val": len(answers),
        "changed_by_normalization": changed,
        "changed_pct": pct(changed, len(answers)),
    }


def quality_stats(entries: list[dict]) -> dict:
    n = len(entries)
    per_k = {}
    for k in K_VALUES:
        flags = {f: sum(1 for e in entries if e["flaws"][f] >= k) for f in FLAWS}
        issues = [f for f in FLAWS if f != "NON"]
        any_issue = sum(1 for e in entries if any(e["flaws"][f] >= k for f in issues))
        multi = sum(1 for e in entries if sum(e["flaws"][f] >= k for f in issues) >= 2)
        unrec = sum(1 for e in entries if e["unrecognizable"] >= k)
        per_k[str(k)] = {
            "flags": {f: {"n": c, "pct": pct(c, n)} for f, c in flags.items()},
            "any_issue_pct": pct(any_issue, n),
            "two_plus_issues_pct": pct(multi, n),
            "unrecognizable": {"n": unrec, "pct": pct(unrec, n)},
        }
    return {
        "images": n,
        "unrecognizable_votes_distribution": dict(
            sorted(Counter(e["unrecognizable"] for e in entries).items())
        ),
        "per_k": per_k,
    }


def cross_stats(vqa_entries: list[dict], quality_entries: list[dict]) -> dict:
    """How VQA answerability relates to image quality, on images that have both labels."""
    q = {e["image"]: e for e in quality_entries}
    joined = [(e, q[e["image"]]) for e in vqa_entries if e["image"] in q]
    out: dict = {"vqa_questions": len(vqa_entries), "with_quality_labels": len(joined)}
    issues = [f for f in FLAWS if f != "NON"]
    for k in K_VALUES:
        unrec = [e for e, qe in joined if qe["unrecognizable"] >= k]
        rec = [e for e, qe in joined if qe["unrecognizable"] < k]
        clean = [e for e, qe in joined if not any(qe["flaws"][f] >= k for f in issues)]
        flawed = [e for e, qe in joined if any(qe["flaws"][f] >= k for f in issues)]
        unans = [qe for e, qe in joined if not e["answerable"]]
        out[f"k{k}"] = {
            "unanswerable_pct_if_unrecognizable": pct(
                sum(not e["answerable"] for e in unrec), len(unrec)
            ),
            "unanswerable_pct_if_recognizable": pct(
                sum(not e["answerable"] for e in rec), len(rec)
            ),
            "unanswerable_pct_if_no_issue": pct(
                sum(not e["answerable"] for e in clean), len(clean)
            ),
            "unanswerable_pct_if_any_issue": pct(
                sum(not e["answerable"] for e in flawed), len(flawed)
            ),
            # Candidate 3-way split for Step 1.4: unanswerable questions whose image is
            # unrecognizable -> "unsuitable_image"; the rest -> "unanswerable".
            "of_unanswerable_unrecognizable_n": sum(qe["unrecognizable"] >= k for qe in unans),
            "of_unanswerable_unrecognizable_pct": pct(
                sum(qe["unrecognizable"] >= k for qe in unans), len(unans)
            ),
        }
    return out


def run_audit(root: Path) -> dict:
    vqa, quality = load_annotations(root)
    return {
        "vqa": {split: vqa_stats(entries) for split, entries in vqa.items()},
        "normalization": normalisation_stats(vqa),
        "quality": {split: quality_stats(entries) for split, entries in quality.items()},
        "vqa_x_quality": {
            split: cross_stats(vqa[split], quality[split]) for split in ("train", "val")
        },
    }


# ---------------------------------------------------------------- report


def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def render_report(a: dict) -> str:
    v, q, x = a["vqa"], a["quality"], a["vqa_x_quality"]
    splits = ["train", "val", "test"]
    out = [
        "# Data audit (Step 1.3)",
        "",
        "Generated by `python -m vizwiz.data.audit` from the official VizWiz annotation",
        "files. Aggregate statistics only. Every number is also in `audit.json`.",
        "Image-size statistics are added in Step 1.5 (they need the images).",
        "",
        "## VQA: questions and answerability",
        "",
        _table(
            ["", *splits],
            [
                ["questions", *(v[s]["questions"] for s in splits)],
                ["answerable %", *(v[s]["answerable_pct"] for s in splits)],
                ["unanswerable %", *(v[s]["unanswerable_pct"] for s in splits)],
                *[
                    [
                        f"type: {t} %",
                        *(v[s]["answer_type"].get(t, {}).get("pct", 0) for s in splits),
                    ]
                    for t in ("other", "unanswerable", "yes/no", "number")
                ],
                ["question words (median)", *(v[s]["question_words"]["median"] for s in splits)],
                [
                    "answer words, answerable (median)",
                    *(v[s]["top_answer_words_answerable"]["median"] for s in splits),
                ],
            ],
        ),
        "",
        "## VQA: annotator agreement",
        "",
        "Each question has 10 crowd answers. *Oracle accuracy* is the best official VQA",
        "accuracy any single answer could get (the ceiling for a perfect model).",
        "",
        _table(
            ["", *splits],
            [
                [
                    "top answer has >= 3 votes %",
                    *(v[s]["agreement"]["pct_top_answer_ge3_votes"] for s in splits),
                ],
                [
                    "top answer has >= 5 votes %",
                    *(v[s]["agreement"]["pct_top_answer_ge5_votes"] for s in splits),
                ],
                ["tied top answer %", *(v[s]["agreement"]["pct_tied_top_answer"] for s in splits)],
                ["oracle VQA accuracy %", *(v[s]["oracle_vqa_accuracy_pct"] for s in splits)],
            ],
        ),
        "",
        "Top-answer vote count distribution (train):",
        "",
        _table(
            ["votes for top answer", "questions"],
            [[k, c] for k, c in v["train"]["agreement"]["top_answer_votes_distribution"].items()],
        ),
        "",
        "## Answer normalisation",
        "",
        f"{a['normalization']['changed_pct']}% of the "
        f"{a['normalization']['unique_gt_answers_train_val']:,} unique ground-truth answers"
        " (train + val) change under the official normalisation. The official metric",
        "compares a *normalised* prediction with *raw* ground truth, so a model that",
        "outputs one of these answers verbatim cannot match it. Training targets are chosen",
        "to maximise the official metric (Step 1.4).",
        "",
        "## Image quality issues",
        "",
        "5 annotators per image. An issue counts as present when at least *k* annotators",
        "flagged it. Percent of images, by k:",
        "",
    ]
    for split in ("train", "val"):
        rows = []
        for f in FLAWS:
            rows.append(
                [
                    f"{f} ({FLAW_NAMES[f]})",
                    *(q[split]["per_k"][str(k)]["flags"][f]["pct"] for k in K_VALUES),
                ]
            )
        rows.append(
            [
                "any issue (excl. NON)",
                *(q[split]["per_k"][str(k)]["any_issue_pct"] for k in K_VALUES),
            ]
        )
        rows.append(
            ["2+ issues", *(q[split]["per_k"][str(k)]["two_plus_issues_pct"] for k in K_VALUES)]
        )
        rows.append(
            [
                "unrecognizable",
                *(q[split]["per_k"][str(k)]["unrecognizable"]["pct"] for k in K_VALUES),
            ]
        )
        out += [
            f"**{split}** ({q[split]['images']:,} images)",
            "",
            _table(["", "k=1", "k=2", "k=3"], rows),
            "",
        ]

    out += [
        "## Answerability vs image quality",
        "",
        "On VQA questions whose image also has quality labels "
        f"(train {x['train']['with_quality_labels']:,} of {x['train']['vqa_questions']:,}; "
        f"val {x['val']['with_quality_labels']:,} of {x['val']['vqa_questions']:,}).",
        "",
        _table(
            ["train", "k=1", "k=2", "k=3"],
            [
                [label, *(x["train"][f"k{k}"][key] for k in K_VALUES)]
                for label, key in (
                    (
                        "unanswerable % if image unrecognizable",
                        "unanswerable_pct_if_unrecognizable",
                    ),
                    ("unanswerable % if image recognizable", "unanswerable_pct_if_recognizable"),
                    ("unanswerable % if no quality issue", "unanswerable_pct_if_no_issue"),
                    ("unanswerable % if any quality issue", "unanswerable_pct_if_any_issue"),
                    (
                        "% of unanswerable with unrecognizable image",
                        "of_unanswerable_unrecognizable_pct",
                    ),
                )
            ],
        ),
        "",
    ]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    audit = run_audit(args.annotations)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (args.out / "audit_report.md").write_text(render_report(audit))
    print(f"Wrote {args.out / 'audit.json'} and {args.out / 'audit_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
