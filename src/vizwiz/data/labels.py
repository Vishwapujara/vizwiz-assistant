"""Label construction from the VizWiz annotations (human labels only).

Answer target (D-033)
    Among the distinct human answers, pick the one with the highest official VQA
    accuracy when output verbatim (the metric normalises the prediction and compares it
    with the raw answers). Ties: more votes first, then the earliest-listed answer. For
    answerable questions, ``unanswerable`` is never a target: declining is the
    answerability adapter's job (D-031).

Answerability (D-031)
    2 labels from the official flag: ``answerable`` / ``unanswerable``. The number of
    ``unanswerable`` votes is kept for the borderline slice (audit finding 2).

Quality (D-030, D-032)
    Each issue is positive when at least ``k`` of 5 annotators flagged it. ``OTH`` is
    dropped. ``unrecognizable`` is positive when at least ``k`` annotators said so. No
    positive issue gives ``["none"]``. The ``NON`` votes are not used directly.
"""

from __future__ import annotations

from collections import Counter

from vizwiz.data.config import DataConfig
from vizwiz.eval.vqa_metric import vqa_accuracy

UNANSWERABLE = "unanswerable"


def answer_target(answers: list[str], answerable: bool = True) -> tuple[str, float]:
    """Return (target answer, its official accuracy against these answers)."""
    counts = Counter(answers)
    first_index = {}
    for i, a in enumerate(answers):
        first_index.setdefault(a, i)
    candidates = [a for a in counts if not (answerable and a == UNANSWERABLE)]
    if not candidates:  # every answer is "unanswerable": only possible if not answerable
        candidates = list(counts)
    scored = [(vqa_accuracy(a, answers), counts[a], -first_index[a], a) for a in candidates]
    best = max(scored)
    return best[3], best[0]


def answerability_label(answerable: int) -> str:
    return "answerable" if answerable else "unanswerable"


def quality_labels(entry: dict, cfg: DataConfig) -> list[str]:
    """Quality output codes for one image, in config order."""
    k = cfg.quality_k
    out = [name for code, name in cfg.quality_issues.items() if entry["flaws"].get(code, 0) >= k]
    if entry.get("unrecognizable", 0) >= k:
        out.append(cfg.unrecognizable_label)
    return out or ["none"]


def vqa_record(entry: dict, split: str, quality: dict | None, cfg: DataConfig) -> dict:
    """One VQA question with every label the tasks need."""
    answers = [a["answer"] for a in entry["answers"]]
    answerable = bool(entry["answerable"])
    target, target_acc = answer_target(answers, answerable) if answerable else (UNANSWERABLE, None)
    most_common = Counter(answers).most_common(1)[0][0]
    return {
        "id": f"vqa/{entry['image']}",
        "image": entry["image"],
        "split": split,
        "question": entry["question"],
        "answers": answers,
        "answer_type": entry["answer_type"],
        "answerability": answerability_label(entry["answerable"]),
        "unanswerable_votes": answers.count(UNANSWERABLE),
        "target": target,
        "target_accuracy": target_acc,
        "target_is_most_common": target == most_common,
        "quality": quality_labels(quality, cfg) if quality else None,
    }


def quality_record(entry: dict, split: str, cfg: DataConfig) -> dict:
    return {
        "id": f"quality/{entry['image']}",
        "image": entry["image"],
        "split": split,
        "labels": quality_labels(entry, cfg),
        "unrecognizable_votes": entry["unrecognizable"],
    }
