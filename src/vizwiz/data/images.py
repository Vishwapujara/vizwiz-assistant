"""Image scan (Step 1.4 manifests + Step 1.3 image statistics), run in Colab on CPU.

    python -m vizwiz.data.images --images /content/data/images \\
        --membership results/splits/membership.json --results results/splits

For every image on disk:
- SHA-256 of the file bytes → ``image_hashes.json`` (image name → hash; committed,
  contains no image content). The split builder uses it to handle duplicate files
  (D-038), and anyone can check they hold exactly our files. ``--expect`` compares a
  fresh scan with the committed file.
- width, height, EXIF orientation → aggregate statistics in ``image_stats.json``.
- duplicate check: identical files under different names. A file in train or val that
  is identical to one in any other split is leakage and fails the scan. test and
  test_vqa may share files (both are evaluation-only); those are only counted.
  Images in no split are the excluded copies of test images (``excluded.json``).

Uses a process pool; reading ~40k JPEG headers and hashing ~19 GB is I/O-bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, quantiles

from PIL import Image

from vizwiz.data.config import REPO_ROOT

EXIF_ORIENTATION = 0x0112
ROTATING_ORIENTATIONS = {5, 6, 7, 8}  # orientations that swap width and height


def scan_one(path: str) -> dict:
    data = Path(path).read_bytes()
    with Image.open(path) as img:
        w, h = img.size
        orientation = img.getexif().get(EXIF_ORIENTATION, 1)
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "width": w,
        "height": h,
        "orientation": orientation,
    }


def find_images(root: Path) -> dict[str, Path]:
    """Image name → path, searching subfolders (the zips unpack into train/, val/, test/).
    Every caller resolves image paths through this, never ``root / name``."""
    return {p.name: p for p in root.rglob("*.jpg")}


def pct_summary(values: list[float]) -> dict:
    v = sorted(values)
    q = quantiles(v, n=100, method="inclusive") if len(v) > 1 else [v[0]] * 99
    return {
        "mean": round(mean(v), 1),
        "p5": round(q[4], 1),
        "p50": round(q[49], 1),
        "p95": round(q[94], 1),
        "min": v[0],
        "max": v[-1],
    }


TRAINING_SPLITS = {"train", "val"}


def analyse(scans: dict[str, dict], membership: dict[str, list[str]]) -> dict:
    """Return (stats, leaking duplicate groups)."""
    split_of = {img: s for s, imgs in membership.items() for img in imgs}
    by_hash: dict[str, list[str]] = defaultdict(list)
    for name, r in scans.items():
        by_hash[r["sha256"]].append(name)
    dup_groups = [sorted(v) for v in by_hash.values() if len(v) > 1]

    def splits_of(group):
        return {split_of.get(n, "excluded") for n in group}

    cross = [g for g in dup_groups if len(splits_of(g)) > 1]
    leaks = [g for g in cross if splits_of(g) & TRAINING_SPLITS]
    cross_pairs = Counter(
        tuple(sorted(splits_of(g))) for g in cross
    )  # which split pairs share files

    stats: dict = {"images_scanned": len(scans), "per_split": {}}
    for split, names in membership.items():
        rs = [scans[n] for n in names if n in scans]
        if not rs:
            continue
        # Upright size: EXIF orientations 5–8 mean the stored pixels are rotated 90°.
        uw = [r["height"] if r["orientation"] in ROTATING_ORIENTATIONS else r["width"] for r in rs]
        uh = [r["width"] if r["orientation"] in ROTATING_ORIENTATIONS else r["height"] for r in rs]
        mp = [r["width"] * r["height"] / 1e6 for r in rs]
        stats["per_split"][split] = {
            "images": len(rs),
            "megapixels": pct_summary(mp),
            "upright_width": pct_summary(uw),
            "upright_height": pct_summary(uh),
            "portrait_pct": round(
                100 * sum(h > w for w, h in zip(uw, uh, strict=True)) / len(rs), 2
            ),
            "exif_orientation_counts": dict(sorted(Counter(r["orientation"] for r in rs).items())),
            "exif_rotated_pct": round(100 * sum(r["orientation"] != 1 for r in rs) / len(rs), 2),
            "file_mb": pct_summary([r["bytes"] / 1e6 for r in rs]),
            "most_common_sizes": [
                [f"{w}x{h}", c]
                for (w, h), c in Counter((r["width"], r["height"]) for r in rs).most_common(5)
            ],
        }
    stats["duplicates"] = {
        "groups_of_identical_files": len(dup_groups),
        "files_in_groups": sum(len(g) for g in dup_groups),
        "groups_spanning_splits": len(cross),
        "groups_leaking_into_train_or_val": len(leaks),
        "spanning_split_pairs": {"+".join(k): v for k, v in sorted(cross_pairs.items())},
    }
    return stats, leaks


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Hash and measure every VizWiz image")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--membership", type=Path, default=REPO_ROOT / "results/splits/membership.json")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--expect", type=Path, help="committed image_hashes.json to compare with")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = p.parse_args(argv)

    membership = json.loads(args.membership.read_text())
    wanted = {n for names in membership.values() for n in names}
    found = find_images(args.images)
    missing = sorted(wanted - set(found))
    extra = sorted(set(found) - wanted)
    print(f"membership: {len(wanted)} images; found on disk: {len(found)}; "
          f"missing: {len(missing)}; not in any split (excluded): {len(extra)}")  # fmt: skip

    names = sorted(found)
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(scan_one, [str(found[n]) for n in names], chunksize=64))
    scans = dict(zip(names, results, strict=True))

    stats, cross = analyse(scans, membership)
    stats["missing_images"] = len(missing)
    stats["images_not_in_any_split"] = len(extra)
    args.results.mkdir(parents=True, exist_ok=True)
    hashes = {n: scans[n]["sha256"] for n in names}
    (args.results / "image_hashes.json").write_text(json.dumps(hashes, indent=0) + "\n")
    (args.results / "image_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    if cross:
        print(f"LEAKAGE: {len(cross)} groups of identical files touch train/val and another split.")
    same = True
    if args.expect:
        same = json.loads(args.expect.read_text()) == hashes
        print("IMAGE HASHES MATCH COMMITTED" if same else "IMAGE HASHES DIFFER FROM COMMITTED")
    ok = not missing and not cross and same
    print("IMAGE SCAN OK" if ok else "IMAGE SCAN FOUND PROBLEMS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
