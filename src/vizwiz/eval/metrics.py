"""Task metrics (Step 2.1). Pure Python, so they run anywhere and in CI.

Answering (PROJECT_PLAN 9.1)
    Official VizWiz VQA accuracy (``vqa_metric``), on answerable questions only
    (the answering adapter's job, D-033), sliced by answer type and image quality.

Answerability, matching the official ``evaluate_unanswerability``
    The positive class is ``unanswerable``. Each prediction carries a score: the
    model's probability that the question is unanswerable. The official code reports
    sklearn ``average_precision_score`` on that score and ``f1_score`` at score > 0.5.
    ``average_precision`` below reproduces sklearn exactly (step-wise, no
    interpolation, tied scores form one threshold). We add recall at fixed precision,
    macro-F1 of the generated labels, the confusion matrix, and the borderline slice.

Quality
    The model replies with a JSON array of labels. Per-label precision / recall / F1,
    macro-F1 over the issue labels, exact-set match, and the JSON validity rate.
    Invalid replies count as predicting no labels at all.

Confidence intervals (PROJECT_PLAN 9.2)
    ``bootstrap_ci``: percentile bootstrap over examples, fixed seed.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence

from vizwiz.eval.vqa_metric import vqa_accuracy

UNANSWERABLE = "unanswerable"
ANSWERABLE = "answerable"


# ---------------------------------------------------------------- generic helpers


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def pct(x: float | None, digits: int = 2) -> float | None:
    return None if x is None else round(100 * x, digits)


def bootstrap_ci(
    items: Sequence,
    stat: Callable[[Sequence], float | None],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI of ``stat(items)``: resample items with replacement."""
    if not items:
        return None
    rng = random.Random(seed)
    n = len(items)
    values = []
    for _ in range(n_resamples):
        v = stat([items[rng.randrange(n)] for _ in range(n)])
        if v is not None:
            values.append(v)
    if not values:
        return None
    values.sort()
    lo = values[int(alpha / 2 * (len(values) - 1))]
    hi = values[int((1 - alpha / 2) * (len(values) - 1))]
    return lo, hi


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "support": tp + fn}


# ---------------------------------------------------------------- ranking metrics


def _curve(y_true: Sequence[int], scores: Sequence[float]):
    """(precision, recall) at each distinct score threshold, highest score first."""
    pairs = sorted(zip(scores, y_true, strict=True), key=lambda t: -t[0])
    total_pos = sum(y_true)
    tp = fp = 0
    out = []
    for i, (s, y) in enumerate(pairs):
        tp += y
        fp += 1 - y
        if i + 1 == len(pairs) or pairs[i + 1][0] != s:  # end of a tie group
            out.append((tp / (tp + fp), tp / total_pos if total_pos else 0.0))
    return out


def average_precision(y_true: Sequence[int], scores: Sequence[float]) -> float | None:
    """Equal to sklearn.metrics.average_precision_score: sum of (R_n - R_{n-1}) * P_n."""
    if not sum(y_true):
        return None
    ap, prev_r = 0.0, 0.0
    for p, r in _curve(y_true, scores):
        ap += (r - prev_r) * p
        prev_r = r
    return ap


def recall_at_precision(
    y_true: Sequence[int], scores: Sequence[float], min_precision: float
) -> float:
    """Highest recall reachable with precision >= min_precision (0 if never reached)."""
    return max((r for p, r in _curve(y_true, scores) if p >= min_precision), default=0.0)


# ---------------------------------------------------------------- parsing model output


def clean_answer(text: str) -> str:
    """First non-empty line, stripped. The official normalisation does the rest."""
    for line in text.strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


INVALID_KINDS = ("empty", "yes_no", "sentence", "other")
_SENTENCE_STARTS = ("the", "this", "it", "there", "a", "an", "i", "yes,", "no,")


def invalid_kind(text: str) -> str:
    """Coarse category of a reply that is neither label. Categories only, so the
    committed metrics never contain model text about dataset questions."""
    words = text.strip().lower().split()
    if not words:
        return "empty"
    first = words[0].strip(".!,")
    if first in ("yes", "no") and len(words) == 1:
        return "yes_no"
    if words[0] in _SENTENCE_STARTS or first in _SENTENCE_STARTS:
        return "sentence"
    return "other"  # most often a direct answer to the question


def parse_answerability(text: str) -> str | None:
    t = text.strip().lower()
    if t.startswith(UNANSWERABLE) or t.startswith("not answerable"):
        return UNANSWERABLE
    if t.startswith(ANSWERABLE):
        return ANSWERABLE
    return None


def parse_quality(text: str, allowed: Sequence[str]) -> tuple[set[str], bool]:
    """(labels, valid). Valid = a JSON array of known labels; ``["none"]`` alone is
    no problems. Anything else is invalid and predicts no labels."""
    t = text.strip()
    if t.startswith("```"):  # tolerate a fenced code block
        t = t.strip("`").removeprefix("json").strip()
    try:
        value = json.loads(t)
    except json.JSONDecodeError:
        return set(), False
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        return set(), False
    labels = {v.strip().lower() for v in value}
    if not labels or not labels <= set(allowed):
        return set(), False
    if "none" in labels and len(labels) > 1:
        return set(), False
    return labels - {"none"}, True


# ---------------------------------------------------------------- per-task metrics
#
# Each takes a list of joined examples (ground-truth record fields + prediction
# fields) and returns a JSON-ready dict. Rates are in percent.


def answer_metrics(examples: list[dict], n_resamples: int = 1000) -> dict:
    """examples: {"prediction", "answers", "answer_type", "quality" (list | None)}."""
    for e in examples:
        e.setdefault("accuracy", vqa_accuracy(clean_answer(e["prediction"]), e["answers"]))

    def acc(items):
        return mean([e["accuracy"] for e in items])

    def sliced(key_fn):
        groups = defaultdict(list)
        for e in examples:
            for k in key_fn(e):
                groups[k].append(e)
        return {k: {"n": len(v), "accuracy": pct(acc(v))} for k, v in sorted(groups.items())}

    ci = bootstrap_ci(examples, acc, n_resamples)
    return {
        "n": len(examples),
        "vqa_accuracy": pct(acc(examples)),
        "vqa_accuracy_ci95": [pct(x) for x in ci] if ci else None,
        "declined_pct": pct(
            mean([clean_answer(e["prediction"]).lower() == UNANSWERABLE for e in examples])
        ),
        "by_answer_type": sliced(lambda e: [e["answer_type"]]),
        "by_quality_label": sliced(lambda e: e.get("quality") or []),
        "by_quality_any_issue": sliced(
            lambda e: [] if e.get("quality") is None else
            ["no_issue" if e["quality"] == ["none"] else "any_issue"]
        ),
    }  # fmt: skip


def answerability_metrics(
    examples: list[dict],
    precisions: Sequence[float] = (0.8, 0.9),
    n_resamples: int = 1000,
) -> dict:
    """examples: {"prediction" (text), "score" (P(unanswerable)), "answerability",
    "unanswerable_votes"}."""
    y = [int(e["answerability"] == UNANSWERABLE) for e in examples]
    s = [float(e["score"]) for e in examples]
    labels = [parse_answerability(e["prediction"]) for e in examples]

    def ap(items):
        return average_precision(
            [int(e["answerability"] == UNANSWERABLE) for e in items],
            [float(e["score"]) for e in items],
        )

    conf = Counter(
        (e["answerability"], lbl or "invalid") for e, lbl in zip(examples, labels, strict=True)
    )
    per_class = {}
    for c in (UNANSWERABLE, ANSWERABLE):
        tp = sum(
            1
            for e, lbl in zip(examples, labels, strict=True)
            if lbl == c and e["answerability"] == c
        )
        fp = sum(
            1
            for e, lbl in zip(examples, labels, strict=True)
            if lbl == c and e["answerability"] != c
        )
        fn = sum(
            1
            for e, lbl in zip(examples, labels, strict=True)
            if lbl != c and e["answerability"] == c
        )
        per_class[c] = {
            k: round(v, 4) if isinstance(v, float) else v for k, v in prf(tp, fp, fn).items()
        }
    official_f1 = prf(
        sum(1 for yi, si in zip(y, s, strict=True) if yi and si > 0.5),
        sum(1 for yi, si in zip(y, s, strict=True) if not yi and si > 0.5),
        sum(1 for yi, si in zip(y, s, strict=True) if yi and si <= 0.5),
    )["f1"]
    borderline = [
        (e, lbl)
        for e, lbl in zip(examples, labels, strict=True)
        if e["answerability"] == ANSWERABLE and 3 <= e["unanswerable_votes"] <= 5
    ]
    ci = bootstrap_ci(examples, ap, n_resamples)
    return {
        "n": len(examples),
        "positives_unanswerable": sum(y),
        "average_precision": pct(average_precision(y, s)),
        "average_precision_ci95": [pct(x) for x in ci] if ci else None,
        "official_f1_at_0.5": pct(official_f1),
        **{f"recall_at_precision_{p}": pct(recall_at_precision(y, s, p)) for p in precisions},
        "macro_f1_generated": pct(mean([per_class[c]["f1"] for c in per_class])),
        "per_class_generated": per_class,
        "invalid_output_pct": pct(mean([lbl is None for lbl in labels])),
        "invalid_output_kinds": dict(
            sorted(
                Counter(
                    invalid_kind(e["prediction"])
                    for e, lbl in zip(examples, labels, strict=True)
                    if lbl is None
                ).items()
            )
        ),
        "confusion": {f"{gt}->{pred}": n for (gt, pred), n in sorted(conf.items())},
        "borderline_answerable": {
            "n": len(borderline),
            "predicted_unanswerable_pct": pct(mean([lbl == UNANSWERABLE for _, lbl in borderline])),
        },
    }  # fmt: skip


def quality_metrics(
    examples: list[dict], issue_labels: Sequence[str], n_resamples: int = 1000
) -> dict:
    """examples: {"prediction" (text), "labels" (gold list, ["none"] = no issue)}.
    ``issue_labels``: every label except "none"."""
    allowed = [*issue_labels, "none"]
    parsed = [parse_quality(e["prediction"], allowed) for e in examples]
    gold = [set(e["labels"]) - {"none"} for e in examples]
    pred = [p for p, _ in parsed]

    def macro_f1(idx):
        f1s = []
        for lbl in issue_labels:
            tp = sum(1 for i in idx if lbl in pred[i] and lbl in gold[i])
            fp = sum(1 for i in idx if lbl in pred[i] and lbl not in gold[i])
            fn = sum(1 for i in idx if lbl not in pred[i] and lbl in gold[i])
            f1s.append(prf(tp, fp, fn)["f1"])
        return mean(f1s)

    idx = list(range(len(examples)))
    per_label = {}
    for lbl in issue_labels:
        tp = sum(1 for i in idx if lbl in pred[i] and lbl in gold[i])
        fp = sum(1 for i in idx if lbl in pred[i] and lbl not in gold[i])
        fn = sum(1 for i in idx if lbl not in pred[i] and lbl in gold[i])
        per_label[lbl] = {
            k: round(v, 4) if isinstance(v, float) else v for k, v in prf(tp, fp, fn).items()
        }
    ci = bootstrap_ci(idx, macro_f1, n_resamples)
    return {
        "n": len(examples),
        "macro_f1": pct(macro_f1(idx)),
        "macro_f1_ci95": [pct(x) for x in ci] if ci else None,
        "per_label": per_label,
        "exact_match_pct": pct(mean([p == g for p, g in zip(pred, gold, strict=True)])),
        "json_valid_pct": pct(mean([v for _, v in parsed])),
    }  # fmt: skip


def paired_bootstrap_diff(
    n: int,
    stat: Callable[[Sequence[int], int], float | None],
    n_runs: int,
    reference: int = 0,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> list[dict]:
    """Differences between runs evaluated on the same n examples, with paired CIs.

    ``stat(indices, run)`` computes a metric for one run on a resample of example
    indices. Every resample is shared by all runs, so example difficulty cancels out
    and the CI of the difference is much tighter than comparing two separate CIs.
    Returns one dict per run: point difference vs ``reference``, its CI, and the
    fraction of resamples in which the run beats the reference.
    """
    rng = random.Random(seed)
    full = list(range(n))
    point = [stat(full, r) for r in range(n_runs)]
    diffs: list[list[float]] = [[] for _ in range(n_runs)]
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        values = [stat(idx, r) for r in range(n_runs)]
        if any(v is None for v in values):
            continue
        for r in range(n_runs):
            diffs[r].append(values[r] - values[reference])
    out = []
    for r in range(n_runs):
        d = sorted(diffs[r])
        k = len(d)
        out.append({
            "value": point[r],
            "diff_vs_reference": None if point[r] is None else point[r] - point[reference],
            "diff_ci95": [d[int(alpha / 2 * (k - 1))], d[int((1 - alpha / 2) * (k - 1))]]
            if k else None,
            "p_better": sum(x > 0 for x in d) / k if k else None,
        })  # fmt: skip
    return out
