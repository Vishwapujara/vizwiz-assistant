"""Step 1.4: build train / val / test splits and labelled record files.

    python -m vizwiz.data.splits --annotations <dir> --out <data dir> --results results/splits

Splits (D-022)
    train     official train minus the carved val
    val       ~10% of official train, image-grouped, stratified, fixed seed
    test      official val (the only split labelled for all three tasks)
    test_vqa  official VQA test (answers public; VQA tasks only, no quality labels)

Image grouping: the unit of splitting is the image, so every example built from one
image (its VQA question and its quality labels) lands in the same split.

Outputs
    <out>/vqa_{split}.jsonl, <out>/quality_{split}.jsonl
        Labelled records. They contain the dataset's questions and answers, so they stay
        on Drive and never go into git.
    <results>/membership.json     image names per split (IDs only, committed)
    <results>/files.json          SHA-256 of every record file, so a rebuild in Colab
                                  can be checked to be identical (committed)
    <results>/label_stats.json, label_stats.md   aggregate label statistics (committed)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from vizwiz.data.config import DataConfig, load_data_config
from vizwiz.data.labels import quality_record, vqa_record

SPLITS = ("train", "val", "test", "test_vqa")


def load_official(root: Path) -> dict:
    def read(*parts):
        return json.loads(root.joinpath(*parts).read_text())

    return {
        "vqa": {
            "train": read("vqa", "train.json"),
            "val": read("vqa", "val.json"),
            "test": read("vqa", "VQA_test.json"),
        },
        "quality": {
            "train": read("quality", "train.json"),
            "val": read("quality", "val.json"),
        },
    }


def stratum(image: str, vqa: dict, quality: dict, cfg: DataConfig) -> str:
    """Stratification key for carving val out of official train."""
    if image in vqa:
        return f"vqa:{vqa[image]['answer_type']}"  # answer_type also encodes answerability
    unrec = quality[image]["unrecognizable"] >= cfg.quality_k
    return f"quality_only:{'unrecognizable' if unrec else 'recognizable'}"


def carve_val(images: list[str], vqa: dict, quality: dict, cfg: DataConfig) -> set[str]:
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for image in sorted(images):
        by_stratum[stratum(image, vqa, quality, cfg)].append(image)
    val: set[str] = set()
    for key in sorted(by_stratum):
        group = by_stratum[key]
        random.Random(f"{cfg.seed}:{key}").shuffle(group)
        val.update(group[: round(len(group) * cfg.val_fraction)])
    return val


def assign_splits(official: dict, cfg: DataConfig) -> dict[str, set[str]]:
    vqa_tr = {e["image"]: e for e in official["vqa"]["train"]}
    q_tr = {e["image"]: e for e in official["quality"]["train"]}
    train_pool = sorted(set(vqa_tr) | set(q_tr))
    val = carve_val(train_pool, vqa_tr, q_tr, cfg)
    return {
        "train": set(train_pool) - val,
        "val": val,
        "test": {e["image"] for e in official["vqa"]["val"]}
        | {e["image"] for e in official["quality"]["val"]},
        "test_vqa": {e["image"] for e in official["vqa"]["test"]},
    }


def check_no_leakage(membership: dict[str, set[str]]) -> None:
    """Every image belongs to exactly one split."""
    seen: dict[str, str] = {}
    for split, images in membership.items():
        for image in images:
            if image in seen:
                raise ValueError(f"{image} is in both {seen[image]} and {split}")
            seen[image] = split


def build_records(official: dict, membership: dict[str, set[str]], cfg: DataConfig) -> dict:
    """Return {"vqa": {split: [records]}, "quality": {split: [records]}}."""
    split_of = {img: s for s, imgs in membership.items() for img in imgs}
    quality_all = {
        e["image"]: e for part in official["quality"].values() for e in part
    }  # train + val; test quality labels are hidden
    vqa_out: dict[str, list] = {s: [] for s in SPLITS}
    q_out: dict[str, list] = {s: [] for s in SPLITS if s != "test_vqa"}
    for part in official["vqa"].values():
        for e in part:
            s = split_of[e["image"]]
            vqa_out[s].append(vqa_record(e, s, quality_all.get(e["image"]), cfg))
    for part in official["quality"].values():
        for e in part:
            s = split_of[e["image"]]
            q_out[s].append(quality_record(e, s, cfg))
    for out in (vqa_out, q_out):
        for recs in out.values():
            recs.sort(key=lambda r: r["id"])
    return {"vqa": vqa_out, "quality": q_out}


def write_jsonl(path: Path, records: list[dict]) -> str:
    """Write deterministically and return the file's SHA-256."""
    data = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in records)
    raw = data.encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def label_stats(records: dict, cfg: DataConfig) -> dict:
    stats: dict = {"vqa": {}, "quality": {}}
    for split, recs in records["vqa"].items():
        ans = [r for r in recs if r["answerability"] == "answerable"]
        stats["vqa"][split] = {
            "questions": len(recs),
            "answerability": dict(Counter(r["answerability"] for r in recs)),
            "answer_type": dict(Counter(r["answer_type"] for r in recs).most_common()),
            "borderline_answerable_3to5_unans_votes": sum(
                3 <= r["unanswerable_votes"] <= 5 for r in ans
            ),
            "with_quality_labels": sum(r["quality"] is not None for r in recs),
            "target_mean_accuracy_answerable_pct": round(
                100 * mean(r["target_accuracy"] for r in ans), 2
            )
            if ans
            else None,
            "target_differs_from_most_common": sum(not r["target_is_most_common"] for r in ans),
        }
    for split, recs in records["quality"].items():
        counts = Counter(lbl for r in recs for lbl in r["labels"])
        stats["quality"][split] = {
            "images": len(recs),
            "labels": {lbl: counts.get(lbl, 0) for lbl in cfg.quality_labels},
            "labels_pct": {
                lbl: round(100 * counts.get(lbl, 0) / len(recs), 2) for lbl in cfg.quality_labels
            },
            "mean_labels_per_image": round(mean(len(r["labels"]) for r in recs), 2),
        }
    return stats


def render_stats(stats: dict, membership: dict[str, set[str]], cfg: DataConfig) -> str:
    v, q = stats["vqa"], stats["quality"]

    def row(label, values):
        return "| " + " | ".join([label, *(str(x) for x in values)]) + " |"

    lines = [
        "# Labels and splits (Step 1.4)",
        "",
        f"Generated by `python -m vizwiz.data.splits` (seed {cfg.seed}, val fraction "
        f"{cfg.val_fraction}, quality k = {cfg.quality_k}). Aggregate statistics only.",
        "",
        "## Images per split",
        "",
        "| split | images |",
        "|---|---|",
        *[f"| {s} | {len(membership[s]):,} |" for s in SPLITS],
        "",
        "## VQA questions",
        "",
        "| | " + " | ".join(SPLITS) + " |",
        "|---|" + "---|" * len(SPLITS),
        row("questions", (v[s]["questions"] for s in SPLITS)),
        row("answerable", (v[s]["answerability"].get("answerable", 0) for s in SPLITS)),
        row("unanswerable", (v[s]["answerability"].get("unanswerable", 0) for s in SPLITS)),
        row("borderline answerable (3–5 'unanswerable' votes)",
            (v[s]["borderline_answerable_3to5_unans_votes"] for s in SPLITS)),
        row("with quality labels", (v[s]["with_quality_labels"] for s in SPLITS)),
        row("target accuracy, answerable (mean %)",
            (v[s]["target_mean_accuracy_answerable_pct"] for s in SPLITS)),
        row("target ≠ most common answer",
            (v[s]["target_differs_from_most_common"] for s in SPLITS)),
        "",
        "## Quality labels (% of images)",
        "",
        "| label | train | val | test |",
        "|---|---|---|---|",
        *[row(lbl, (q[s]["labels_pct"][lbl] for s in ("train", "val", "test")))
          for lbl in cfg.quality_labels],
        row("mean labels per image",
            (q[s]["mean_labels_per_image"] for s in ("train", "val", "test"))),
        "",
    ]  # fmt: skip
    return "\n".join(lines)


def build(annotations: Path, out: Path, results: Path, cfg: DataConfig) -> dict:
    official = load_official(annotations)
    membership = assign_splits(official, cfg)
    check_no_leakage(membership)
    records = build_records(official, membership, cfg)

    out.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for task, by_split in records.items():
        for split, recs in by_split.items():
            name = f"{task}_{split}.jsonl"
            hashes[name] = {"records": len(recs), "sha256": write_jsonl(out / name, recs)}

    members = {s: sorted(membership[s]) for s in SPLITS}
    (results / "membership.json").write_text(json.dumps(members, indent=1) + "\n")
    (results / "files.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    stats = label_stats(records, cfg)
    (results / "label_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (results / "label_stats.md").write_text(render_stats(stats, membership, cfg))
    return {"hashes": hashes, "stats": stats}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build VizWiz splits and label files")
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="record files (Drive, not git)")
    p.add_argument("--results", type=Path, default=Path("results/splits"))
    p.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    p.add_argument("--expect", type=Path, help="files.json to compare against (rebuild check)")
    args = p.parse_args(argv)
    cfg = load_data_config(args.config)
    # Read the expected hashes before building: the build may overwrite that file.
    expected = json.loads(args.expect.read_text()) if args.expect else None
    result = build(args.annotations, args.out, args.results, cfg)
    for name, h in sorted(result["hashes"].items()):
        print(f"{name}: {h['records']} records  sha256 {h['sha256'][:16]}…")
    if expected is not None:
        same = expected == result["hashes"]
        print(
            "REBUILD MATCHES COMMITTED HASHES" if same else "REBUILD DIFFERS FROM COMMITTED HASHES"
        )
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
