"""Paired comparison of evaluation runs on the same examples.

    python -m vizwiz.eval.compare --records <dir> --split val --task answer \\
        --runs <run_dir_a> <run_dir_b> ... --names a b ... [--reference a] --report <file>

Reads each run's ``predictions_<task>.jsonl`` (from ``run_eval``), keeps the examples
present in every run, and reports each run's metric, its difference from the
reference run, and a paired 95% bootstrap CI of that difference
(``metrics.paired_bootstrap_diff``). Metrics: VQA accuracy (answer), average
precision (answerability), macro-F1 (quality). Output is aggregates only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vizwiz.data.config import load_data_config
from vizwiz.data.formatting import read_jsonl, task_records
from vizwiz.eval.metrics import (
    UNANSWERABLE,
    average_precision,
    clean_answer,
    paired_bootstrap_diff,
    parse_quality,
    prf,
)
from vizwiz.eval.vqa_metric import vqa_accuracy

METRIC = {"answer": "vqa_accuracy", "answerability": "average_precision", "quality": "macro_f1"}


def load_predictions(run_dir: Path, task: str) -> dict[str, dict]:
    return {r["id"]: r for r in read_jsonl(run_dir / f"predictions_{task}.jsonl")}


def make_stat(task: str, records: list[dict], preds: list[list[dict]]):
    """stat(indices, run) for the task's headline metric, in percent."""
    if task == "answer":
        acc = [[vqa_accuracy(clean_answer(p[i]["prediction"]), r["answers"])
                for i, r in enumerate(records)] for p in preds]  # fmt: skip
        return lambda idx, run: 100 * sum(acc[run][i] for i in idx) / len(idx)
    if task == "answerability":
        y = [int(r["answerability"] == UNANSWERABLE) for r in records]
        s = [[p[i]["score"] for i in range(len(records))] for p in preds]

        def ap(idx, run):
            v = average_precision([y[i] for i in idx], [s[run][i] for i in idx])
            return None if v is None else 100 * v

        return ap
    cfg = load_data_config()
    issues = [lbl for lbl in cfg.quality_labels if lbl != "none"]
    gold = [set(r["labels"]) - {"none"} for r in records]
    pred = [[parse_quality(p[i]["prediction"], [*issues, "none"])[0]
             for i in range(len(records))] for p in preds]  # fmt: skip

    def macro_f1(idx, run):
        f1s = []
        for lbl in issues:
            tp = sum(1 for i in idx if lbl in pred[run][i] and lbl in gold[i])
            fp = sum(1 for i in idx if lbl in pred[run][i] and lbl not in gold[i])
            fn = sum(1 for i in idx if lbl not in pred[run][i] and lbl in gold[i])
            f1s.append(prf(tp, fp, fn)["f1"])
        return 100 * sum(f1s) / len(f1s)

    return macro_f1


def compare(
    records_dir: Path,
    split: str,
    task: str,
    runs: list[Path],
    names: list[str],
    reference: str,
    n_resamples: int = 1000,
) -> dict:
    vqa = read_jsonl(records_dir / f"vqa_{split}.jsonl")
    qfile = records_dir / f"quality_{split}.jsonl"
    quality = read_jsonl(qfile) if qfile.exists() else []
    by_id = {r["id"]: r for r in task_records(task, vqa, quality)}
    run_preds = [load_predictions(d, task) for d in runs]
    common = sorted(set(by_id).intersection(*run_preds))
    records = [by_id[i] for i in common]
    preds = [[p[i] for i in common] for p in run_preds]
    ref = names.index(reference)
    stat = make_stat(task, records, preds)
    rows = paired_bootstrap_diff(len(common), stat, len(runs), ref, n_resamples)
    out = {"task": task, "split": split, "metric": METRIC[task], "n": len(common),
           "reference": reference, "runs": {}}  # fmt: skip
    for name, row in zip(names, rows, strict=True):
        out["runs"][name] = {
            k: (round(v, 2) if isinstance(v, float) else [round(x, 2) for x in v]
                if isinstance(v, list) else v)
            for k, v in row.items()
        }  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Paired comparison of eval runs")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--task", required=True, choices=list(METRIC))
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--reference", help="name of the reference run (default: first)")
    p.add_argument("--report", type=Path)
    args = p.parse_args(argv)
    if len(args.names) != len(args.runs):
        p.error("--names must match --runs")
    result = compare(args.records, args.split, args.task, args.runs, args.names,
                     args.reference or args.names[0])  # fmt: skip
    text = json.dumps(result, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
