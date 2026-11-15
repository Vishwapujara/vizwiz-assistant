"""Error analysis (Step 3.4, PROJECT_PLAN 9.3). Aggregate counts only.

Compares a "before" run (E1, untrained) with the E2 adapters on the same examples
and breaks the mistakes down by things known for every example without looking at
the photos: question kind (keyword rules on the question), how much the 10 human
answerers agreed, the photo's quality labels, and the number of "unanswerable"
votes. The report holds counts and rates only: no question text, answers or ids.

    python -m vizwiz.eval.errors --records <Drive>/datasets/processed --split test \\
        --before <Drive>/runs/e1-baseline \\
        --after answer=<Drive>/runs/e2-answer-lora \\
                answerability=<Drive>/runs/e2-answerability-lora \\
                quality=<Drive>/runs/e2-quality-lora \\
        --report errors.json --markdown errors.md

Definitions
    right (answering)   VQA accuracy >= 0.6, i.e. at least 2 of the 10 answerers gave
                        the predicted answer. 0.3 = exactly one did ("one answerer").
    declines (answerability)  P(unanswerable) > 0.5, the official F1 rule, used for
                        both runs (E1 often did not reply in the asked format).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

from vizwiz.data.config import load_data_config
from vizwiz.data.formatting import read_jsonl, task_records
from vizwiz.eval.metrics import UNANSWERABLE, clean_answer, mean, parse_quality, pct
from vizwiz.eval.vqa_metric import vqa_accuracy

RIGHT = 0.6  # >= 2 of 10 answerers agree with the prediction

_KIND_RULES = (
    ("color", re.compile(r"\bcolou?rs?\b")),
    ("count_number", re.compile(
        r"\b(how many|how much|number|price|cost|amount|time|date|temperature|expir\w*)\b")),
    ("read_text", re.compile(
        r"\b(read|say|says|said|written|writing|text|label|title|brand|name|words?)\b")),
    ("identify", re.compile(r"^(what|what's|whats)\s*(is|are)?\s*(this|that|these|those|it)\b")),
)  # fmt: skip
QUESTION_KINDS = ("yes_no", *(k for k, _ in _KIND_RULES), "other")


def question_kind(question: str, answer_type: str) -> str:
    """Rough question category from keywords; first matching rule wins."""
    if answer_type == "yes/no":
        return "yes_no"
    q = question.lower().strip()
    for kind, rule in _KIND_RULES:
        if rule.search(q):
            return kind
    return "other"


def agreement(answers: list[str]) -> str:
    """How many of the answerers gave the most common answer."""
    top = Counter(a.strip().lower() for a in answers).most_common(1)[0][1]
    return "high (7+ of 10)" if top >= 7 else "medium (4-6)" if top >= 4 else "low (1-3)"


def votes_bucket(votes: int) -> str:
    return "0" if votes == 0 else "1-2" if votes <= 2 else "3-5" if votes <= 5 else "6-10"


def quality_group(labels: list[str] | None) -> str | None:
    if labels is None:
        return None
    return "clean photo" if labels == ["none"] else "photo issue"


def transitions(before: Iterable[bool], after: Iterable[bool]) -> dict[str, int]:
    """Counts of right/wrong changes between two runs (True = right)."""
    c = Counter((b, a) for b, a in zip(before, after, strict=True))
    return {
        "fixed": c[(False, True)],
        "broken": c[(True, False)],
        "still_wrong": c[(False, False)],
        "still_right": c[(True, True)],
    }


def slices(rows: list[dict], keys: Callable[[dict], Iterable[str]], stat) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        for k in keys(r):
            groups[k].append(r)
    return {k: {"n": len(v), **stat(v)} for k, v in sorted(groups.items())}


def words(text: str) -> int:
    return len(clean_answer(text).split())


# ------------------------------------------------------------------- answering


def answer_errors(records: list[dict], before: dict, after: dict) -> dict:
    rows = []
    for r in records:
        b, a = before[r["id"]]["prediction"], after[r["id"]]["prediction"]
        rows.append({
            "kind": question_kind(r["question"], r["answer_type"]),
            "agreement": agreement(r["answers"]),
            "quality": r.get("quality"),
            "acc_b": vqa_accuracy(clean_answer(b), r["answers"]),
            "acc_a": vqa_accuracy(clean_answer(a), r["answers"]),
            "words_b": words(b), "words_a": words(a),
            "target_words": words(r["target"]),
            "declined_a": clean_answer(a).lower() == UNANSWERABLE,
            "yes_no": r["answer_type"] == "yes/no",
        })  # fmt: skip

    def stat(v):
        return {
            "accuracy_before": pct(mean([x["acc_b"] for x in v])),
            "accuracy_after": pct(mean([x["acc_a"] for x in v])),
            "right_after_pct": pct(mean([x["acc_a"] >= RIGHT for x in v])),
        }

    wrong = [x for x in rows if x["acc_a"] < RIGHT]
    kinds = Counter(
        "declined" if x["declined_a"]
        else "yes/no flipped" if x["yes_no"]
        else "one answerer agreed" if x["acc_a"] > 0
        else "no answerer agreed, longer than target" if x["words_a"] > x["target_words"]
        else "no answerer agreed"
        for x in wrong
    )  # fmt: skip
    return {
        "n": len(rows),
        "transitions": transitions(
            [x["acc_b"] >= RIGHT for x in rows], [x["acc_a"] >= RIGHT for x in rows]
        ),  # fmt: skip
        "by_question_kind": slices(rows, lambda x: [x["kind"]], stat),
        "by_answerer_agreement": slices(rows, lambda x: [x["agreement"]], stat),
        "by_photo": slices(
            rows, lambda x: [g] if (g := quality_group(x["quality"])) else [], stat
        ),  # fmt: skip
        "by_quality_label": slices(rows, lambda x: x["quality"] or [], stat),
        "remaining_errors": {"n": len(wrong), "kinds": dict(kinds.most_common())},
        "answer_length_words": {
            "target_mean": round(mean([x["target_words"] for x in rows]), 2),
            "before_mean": round(mean([x["words_b"] for x in rows]), 2),
            "after_mean": round(mean([x["words_a"] for x in rows]), 2),
            "before_over_3_words_pct": pct(mean([x["words_b"] > 3 for x in rows])),
            "after_over_3_words_pct": pct(mean([x["words_a"] > 3 for x in rows])),
        },
    }


# --------------------------------------------------------------- answerability


def answerability_errors(records: list[dict], before: dict, after: dict) -> dict:
    rows = []
    for r in records:
        rows.append({
            "unans": r["answerability"] == UNANSWERABLE,
            "dec_b": float(before[r["id"]]["score"]) > 0.5,
            "dec_a": float(after[r["id"]]["score"]) > 0.5,
            "kind": question_kind(r["question"], r["answer_type"]),
            "votes": votes_bucket(r["unanswerable_votes"]),
            "quality": r.get("quality"),
        })  # fmt: skip

    def rates(v, key):
        ans = [x for x in v if not x["unans"]]
        una = [x for x in v if x["unans"]]
        return {
            "wrongly_declined_pct": pct(mean([x[key] for x in ans])) if ans else None,
            "missed_unanswerable_pct": pct(mean([not x[key] for x in una])) if una else None,
        }

    def stat(v):
        return {
            "unanswerable": sum(x["unans"] for x in v),
            "before": rates(v, "dec_b"),
            "after": rates(v, "dec_a"),
        }

    return {
        "n": len(rows),
        "overall": stat(rows),
        "transitions": transitions(
            [x["dec_b"] == x["unans"] for x in rows], [x["dec_a"] == x["unans"] for x in rows]
        ),  # fmt: skip
        "by_unanswerable_votes": slices(rows, lambda x: [x["votes"]], stat),
        "by_photo": slices(
            rows, lambda x: [g] if (g := quality_group(x["quality"])) else [], stat
        ),  # fmt: skip
        "by_quality_label": slices(rows, lambda x: x["quality"] or [], stat),
        "by_question_kind": slices(rows, lambda x: [x["kind"]], stat),
    }


# --------------------------------------------------------------------- quality


def quality_errors(records: list[dict], before: dict, after: dict, issues: list[str]) -> dict:
    allowed = [*issues, "none"]
    rows = []
    for r in records:
        rows.append({
            "gold": set(r["labels"]) - {"none"},
            "b": parse_quality(before[r["id"]]["prediction"], allowed)[0],
            "a": parse_quality(after[r["id"]]["prediction"], allowed)[0],
        })  # fmt: skip
    clean = [x for x in rows if not x["gold"]]
    bad = [x for x in rows if x["gold"]]

    def exact(v, k):
        return pct(mean([x[k] == x["gold"] for x in v])) if v else None

    missed_instead: dict[str, Counter] = {lbl: Counter() for lbl in issues}
    for x in rows:
        for lbl in x["gold"] - x["a"]:
            missed_instead[lbl].update(x["a"] - x["gold"] or {"nothing extra"})
    by_count = defaultdict(list)
    for x in rows:
        by_count[str(min(len(x["gold"]), 3)).replace("3", "3+")].append(x)
    return {
        "n": len(rows),
        "transitions_exact_match": transitions(
            [x["b"] == x["gold"] for x in rows], [x["a"] == x["gold"] for x in rows]
        ),  # fmt: skip
        "clean_photos": {
            "n": len(clean),
            "false_alarm_before_pct": pct(mean([bool(x["b"]) for x in clean])),
            "false_alarm_after_pct": pct(mean([bool(x["a"]) for x in clean])),
        },
        "photos_with_issues": {
            "n": len(bad),
            "all_issues_missed_before_pct": pct(mean([not (x["b"] & x["gold"]) for x in bad])),
            "all_issues_missed_after_pct": pct(mean([not (x["a"] & x["gold"]) for x in bad])),
        },
        "exact_match_by_number_of_issues": {
            k: {"n": len(v), "before_pct": exact(v, "b"), "after_pct": exact(v, "a")}
            for k, v in sorted(by_count.items())
        },
        "when_label_missed_model_said_instead": {
            lbl: dict(c.most_common(3)) for lbl, c in missed_instead.items()
        },
    }


# ------------------------------------------------------------------------ report


def _table(head: list[str], body: list[list]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join("" if c is None else str(c) for c in row) + " |" for row in body]
    return out + [""]


def to_markdown(rep: dict) -> str:
    lines = [f"# Error analysis, {rep['split']} split (Step 3.4)", "",
             "Generated by `python -m vizwiz.eval.errors`. Before = E1 (untrained), "
             "after = E2 adapter. Aggregate counts only.", ""]  # fmt: skip
    if a := rep.get("answer"):
        lines += [
            "## Answering",
            "",
            f"Right = at least 2 of 10 answerers gave the "
            f"predicted answer. Changes on {a['n']:,} questions: {a['transitions']}.",
            "",
        ]
        for key, title in (
            ("by_question_kind", "Question kind"),
            ("by_answerer_agreement", "Answerer agreement"),
            ("by_photo", "Photo"),
            ("by_quality_label", "Quality label"),
        ):
            lines += _table([title, "n", "accuracy before", "accuracy after", "right after %"],
                            [[k, v["n"], v["accuracy_before"], v["accuracy_after"],
                              v["right_after_pct"]] for k, v in a[key].items()])  # fmt: skip
        lines += [f"Remaining errors ({a['remaining_errors']['n']:,}):", ""]
        lines += _table(["kind", "n"], [[k, v] for k, v in a["remaining_errors"]["kinds"].items()])
        lines += ["Answer length (words): " + json.dumps(a["answer_length_words"]), ""]
    if b := rep.get("answerability"):
        lines += [
            "## Answerability",
            "",
            "Declines = P(unanswerable) > 0.5. Changes in "
            f"correct decisions on {b['n']:,} questions: {b['transitions']}.",
            "",
        ]
        for key, title in (
            ("by_unanswerable_votes", "'unanswerable' votes of 10"),
            ("by_photo", "Photo"),
            ("by_quality_label", "Quality label"),
            ("by_question_kind", "Question kind"),
        ):
            lines += _table(
                [title, "n", "unanswerable", "wrongly declined % (before → after)",
                 "missed unanswerable % (before → after)"],
                [[k, v["n"], v["unanswerable"],
                  f"{v['before']['wrongly_declined_pct']} → {v['after']['wrongly_declined_pct']}",
                  f"{v['before']['missed_unanswerable_pct']} → "
                  f"{v['after']['missed_unanswerable_pct']}"]
                 for k, v in b[key].items()])  # fmt: skip
    if q := rep.get("quality"):
        lines += ["## Quality", "", f"Exact-match changes on {q['n']:,} photos: "
                  f"{q['transitions_exact_match']}.", "",
                  f"- Clean photos ({q['clean_photos']['n']:,}) flagged with an issue: "
                  f"{q['clean_photos']['false_alarm_before_pct']}% → "
                  f"{q['clean_photos']['false_alarm_after_pct']}%",
                  f"- Photos with issues ({q['photos_with_issues']['n']:,}) where every issue "
                  f"was missed: {q['photos_with_issues']['all_issues_missed_before_pct']}% → "
                  f"{q['photos_with_issues']['all_issues_missed_after_pct']}%", ""]  # fmt: skip
        lines += _table(["issues on photo", "n", "exact match before %", "after %"],
                        [[k, v["n"], v["before_pct"], v["after_pct"]]
                         for k, v in q["exact_match_by_number_of_issues"].items()])  # fmt: skip
        lines += _table(["missed label", "what the adapter said instead (top 3)"],
                        [[k, json.dumps(v)] for k, v in
                         q["when_label_missed_model_said_instead"].items()])  # fmt: skip
    return "\n".join(lines)


def analyse(records_dir: Path, split: str, before: Path, after: dict[str, Path]) -> dict:
    cfg = load_data_config()
    issues = [lbl for lbl in cfg.quality_labels if lbl != "none"]
    vqa = read_jsonl(records_dir / f"vqa_{split}.jsonl")
    qfile = records_dir / f"quality_{split}.jsonl"
    quality = read_jsonl(qfile) if qfile.exists() else []
    rep: dict = {"split": split}
    for task, run in after.items():
        b = {p["id"]: p for p in read_jsonl(before / f"predictions_{task}.jsonl")}
        a = {p["id"]: p for p in read_jsonl(run / f"predictions_{task}.jsonl")}
        recs = [r for r in task_records(task, vqa, quality) if r["id"] in a and r["id"] in b]
        if task == "answer":
            rep[task] = answer_errors(recs, b, a)
        elif task == "answerability":
            rep[task] = answerability_errors(recs, b, a)
        else:
            rep[task] = quality_errors(recs, b, a, issues)
    return rep


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", nargs="+", required=True, metavar="TASK=RUN_DIR")
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--markdown", type=Path)
    args = p.parse_args(argv)
    after = {}
    for item in args.after:
        task, _, run = item.partition("=")
        if task not in ("answer", "answerability", "quality") or not run:
            p.error(f"--after expects task=run_dir, got {item!r}")
        after[task] = Path(run)
    rep = analyse(args.records, args.split, args.before, after)
    args.report.write_text(json.dumps(rep, indent=2) + "\n")
    if args.markdown:
        args.markdown.write_text(to_markdown(rep) + "\n")
    for task in after:
        print(f"[{task}] analysed {rep[task]['n']} examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
