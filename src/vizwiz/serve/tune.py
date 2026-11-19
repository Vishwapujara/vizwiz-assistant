"""Served-vs-offline parity and cascade threshold tuning (Step 5.1, D-060).

    python -m vizwiz.serve.tune --records <dir> --val-replay <file> \\
        --offline-val <run_dir> [--test-replay <file>] --report <file>

Parity: the served path (replay rows) against offline ``run_eval`` predictions for the
same photos, per stage. Tuning: for each answerability threshold, the whole
cascade's result on val; the threshold is picked by the rule fixed in D-060
(highest official VizWiz accuracy, a decline scored as the answer
"unanswerable"; ties -> fewer wrong answers), then applied unchanged to test.
Output: aggregate numbers only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from vizwiz.data.config import load_data_config
from vizwiz.data.formatting import read_jsonl
from vizwiz.eval.metrics import UNANSWERABLE, clean_answer, parse_quality
from vizwiz.eval.vqa_metric import normalize_answer, vqa_accuracy

THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]
RIGHT = 0.6  # "right" = VQA accuracy >= 0.6, as in the error analysis (D-054)


def _pct(x: float | None) -> float | None:
    return None if x is None else round(100 * x, 2)


def _mean(xs: list) -> float | None:
    return mean(xs) if xs else None


def parity(rows: list[dict], records: list[dict], offline_dir: Path) -> dict:
    """Agreement of served outputs with offline predictions on the same photos."""
    by_id = {r["id"]: r for r in rows}
    out: dict = {}
    f = offline_dir / "predictions_answerability.jsonl"
    if f.exists():
        pairs = [(by_id[p["id"]]["score"], p["score"]) for p in read_jsonl(f) if p["id"] in by_id]
        diffs = [abs(a - b) for a, b in pairs]
        out["answerability"] = {
            "n": len(pairs),
            "mean_abs_score_diff": round(mean(diffs), 4) if diffs else None,
            "max_abs_score_diff": round(max(diffs), 4) if diffs else None,
            "same_decision_at_0.5_pct": _pct(_mean([(a >= 0.5) == (b >= 0.5) for a, b in pairs])),
        }
    f = offline_dir / "predictions_answer.jsonl"
    if f.exists():
        pairs = [(by_id[p["id"]]["answer"], clean_answer(p["prediction"]))
                 for p in read_jsonl(f) if p["id"] in by_id]  # fmt: skip
        out["answer"] = {
            "n": len(pairs),
            "same_answer_pct": _pct(
                _mean([normalize_answer(a) == normalize_answer(b) for a, b in pairs])
            ),
        }
    f = offline_dir / "predictions_quality.jsonl"
    if f.exists():
        allowed = load_data_config().quality_labels
        image_of = {r["id"]: r["image"] for r in records}
        served = {image_of[i]: set(r["issues"]) for i, r in by_id.items() if i in image_of}
        pairs = []
        for p in read_jsonl(f):
            image = p["id"].removeprefix("quality/")
            if image in served:
                pairs.append((served[image], parse_quality(p["prediction"], allowed)[0]))
        out["quality"] = {
            "n": len(pairs),  # photos that are in both the VQA and the quality split
            "same_labels_pct": _pct(_mean([a == b for a, b in pairs])),
        }
    return out


def outcome(row: dict, threshold: float, unusable: set[str], use_quality: bool) -> str | None:
    """The reply the cascade would give: None = declined, else the answer."""
    if use_quality and set(row["issues"]) & unusable:
        return None
    if row["score"] >= threshold:
        return None
    return row["answer"] or None


def cascade_metrics(rows, records, threshold, unusable, use_quality=True) -> dict:
    rec = {r["id"]: r for r in records}
    acc, answered, wrong, wrongly_declined, missed = [], [], [], [], []
    for row in rows:
        r = rec[row["id"]]
        reply = outcome(row, threshold, unusable, use_quality)
        a = vqa_accuracy(UNANSWERABLE if reply is None else reply, r["answers"])
        acc.append(a)
        answered.append(reply is not None)
        if reply is not None:
            wrong.append(a < RIGHT)
        if r["answerability"] == UNANSWERABLE:
            missed.append(reply is not None)
        else:
            wrongly_declined.append(reply is None)
    n = len(rows)
    return {
        "threshold": threshold,
        "use_quality_rule": use_quality,
        "n": n,
        "vizwiz_accuracy": _pct(_mean(acc)),  # official metric, decline = "unanswerable"
        "answered_pct": _pct(_mean(answered)),
        "wrong_answers_pct_of_all": _pct(sum(wrong) / n) if n else None,
        "wrong_among_answered_pct": _pct(_mean(wrong)),
        "wrongly_declined_pct_of_answerable": _pct(_mean(wrongly_declined)),
        "missed_unanswerable_pct": _pct(_mean(missed)),
    }


def choose(sweep: list[dict]) -> dict:
    """D-060: highest VizWiz accuracy; ties -> fewer wrong answers -> lower threshold."""
    return max(sweep, key=lambda m: (m["vizwiz_accuracy"], -m["wrong_answers_pct_of_all"],
                                     -m["threshold"]))  # fmt: skip


def tune(val_rows, records_val, unusable, test_rows=None, records_test=None) -> dict:
    sweep = [cascade_metrics(val_rows, records_val, t, unusable) for t in THRESHOLDS]
    best = choose(sweep)
    t = best["threshold"]
    report = {
        "rule": "max official VizWiz accuracy on val (decline = 'unanswerable'), D-060",
        "unusable_labels": sorted(unusable),
        "chosen_threshold": t,
        "val": {
            "chosen": best,
            "without_quality_rule": cascade_metrics(val_rows, records_val, t, unusable, False),
            "never_decline": cascade_metrics(val_rows, records_val, 1.01, unusable, False),
            "at_0.5": cascade_metrics(val_rows, records_val, 0.5, unusable),
            "sweep": sweep,
        },
    }
    if test_rows is not None:
        report["test"] = {
            "chosen": cascade_metrics(test_rows, records_test, t, unusable),
            "without_quality_rule": cascade_metrics(test_rows, records_test, t, unusable, False),
            "never_decline": cascade_metrics(test_rows, records_test, 1.01, unusable, False),
            "at_0.5": cascade_metrics(test_rows, records_test, 0.5, unusable),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    from vizwiz.serve.config import load_serve_config

    p = argparse.ArgumentParser(description="Parity check and cascade threshold tuning")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--val-replay", type=Path, required=True)
    p.add_argument("--offline-val", type=Path, help="run_eval output dir on val (parity)")
    p.add_argument("--test-replay", type=Path)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args(argv)
    unusable = set(load_serve_config().rules.unusable_labels)
    rec_val = read_jsonl(args.records / "vqa_val.jsonl")
    val_rows = read_jsonl(args.val_replay)
    test_rows = rec_test = None
    if args.test_replay:
        test_rows, rec_test = (
            read_jsonl(args.test_replay),
            read_jsonl(args.records / "vqa_test.jsonl"),
        )
    report = tune(val_rows, rec_val, unusable, test_rows, rec_test)
    if args.offline_val:
        report["parity_val"] = parity(val_rows, rec_val, args.offline_val)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
