"""Download VizWiz files to Drive with resume, size checks, and SHA-256 checksums.

    python -m vizwiz.data.download fetch   --root /content/drive/MyDrive/vizwiz/datasets
    python -m vizwiz.data.download verify  --root ... --checksums results/data/checksums.json
    python -m vizwiz.data.download extract --root ... --dest /content/data

``fetch`` downloads every file in SOURCES into ``<root>/raw``. Partial downloads are kept
as ``*.part`` and resumed with HTTP Range requests, so a Colab disconnect only costs the
bytes in flight. Each finished file is checked against the server's size, hashed, and
recorded in ``<root>/checksums.json``. Finished files are skipped on rerun.

``extract`` unzips images to fast local disk and annotations next to them, and times it
(the Drive-vs-local storage decision in DECISIONS.md is made from these numbers).

Sources and sizes: docs/data_licenses.md. No file here ever goes into git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

BASE = "https://vizwiz.cs.colorado.edu"
CHUNK = 8 * 1024 * 1024  # 8 MiB
USER_AGENT = "vizwiz-assistant-downloader/0.1"


@dataclass(frozen=True)
class Source:
    name: str  # local file name under <root>/raw
    url: str
    size: int  # bytes, from the server's Content-Length (checked 2026-11-06)
    kind: str  # "images" | "annotations" | "code"
    expected_entries: int | None = None  # files inside the zip, if known


SOURCES: tuple[Source, ...] = (
    Source(
        "vqa_annotations.zip",
        f"{BASE}/VizWiz_final/vqa_data/Annotations.zip",
        1_711_273,
        "annotations",
        3,
    ),
    Source("VQA_test.json", f"{BASE}/VizWiz_all_answers/VQA_test.json", 8_227_183, "annotations"),
    Source(
        "quality_annotations.zip",
        f"{BASE}/VizWiz_final/image_quality/annotations.zip",
        303_584,
        "annotations",
        3,
    ),
    Source("vqa_api.zip", f"{BASE}/VizWiz_final/vqa_data/API.zip", 185_601_325, "code"),
    Source("val.zip", f"{BASE}/VizWiz_final/images/val.zip", 3_488_913_457, "images", 7_750),
    Source("test.zip", f"{BASE}/VizWiz_final/images/test.zip", 3_975_272_799, "images", 8_000),
    # train.zip holds VQA ∪ quality train images: 20,523 + 23,431 - 20,000 shared (R-004).
    Source("train.zip", f"{BASE}/VizWiz_final/images/train.zip", 11_298_421_598, "images", 23_954),
)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000:
            return f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


def sha256_file(path: Path, hasher=None):
    h = hasher or hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h


def load_checksums(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_checksums(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def download_one(src: Source, raw: Path, checksums: dict, retries: int = 5) -> dict:
    """Download (or resume) one file. Returns its checksum record."""
    final = raw / src.name
    part = raw / (src.name + ".part")
    rec = checksums.get(src.name)
    if final.exists() and rec and final.stat().st_size == src.size == rec["size"]:
        print(f"[skip] {src.name}: already downloaded ({human(src.size)})")
        return rec

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > src.size:
            part.unlink()
            have = 0
        # Re-hash the bytes already on disk so the final hash covers the whole file.
        hasher = sha256_file(part) if have else hashlib.sha256()
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
            print(f"[resume] {src.name}: {human(have)} of {human(src.size)} already on disk")
        else:
            print(f"[start] {src.name}: {human(src.size)}")
        try:
            req = urllib.request.Request(src.url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                if have and resp.status != 206:
                    raise RuntimeError(f"server ignored Range request (HTTP {resp.status})")
                t0, got, last = time.monotonic(), have, 0.0
                with part.open("ab") as out:
                    while block := resp.read(CHUNK):
                        out.write(block)
                        hasher.update(block)
                        got += len(block)
                        now = time.monotonic()
                        if now - last > 30:
                            rate = (got - have) / max(now - t0, 1e-6)
                            print(
                                f"    {src.name}: {100 * got / src.size:5.1f}% "
                                f"({human(got)}) at {human(rate)}/s",
                                flush=True,
                            )
                            last = now
            size = part.stat().st_size
            if size != src.size:
                raise RuntimeError(f"size mismatch: got {size}, expected {src.size}")
            part.rename(final)
            elapsed = time.monotonic() - t0
            rec = {
                "size": size,
                "sha256": hasher.hexdigest(),
                "url": src.url,
                "seconds_last_attempt": round(elapsed, 1),
            }
            print(f"[done] {src.name}: sha256 {rec['sha256'][:16]}… in {elapsed:.0f}s")
            return rec
        except Exception as exc:  # network errors: keep the .part file and retry
            wait = min(60, 5 * attempt)
            print(f"[retry {attempt}/{retries}] {src.name}: {type(exc).__name__}: {exc}")
            if attempt == retries:
                raise
            time.sleep(wait)
    raise AssertionError("unreachable")


def check_zip(path: Path, expected: int | None, count_jpg_only: bool) -> dict:
    """Read the zip's central directory and count files (fast; does not decompress).

    Image zips are checked by their number of .jpg files, other zips by all files.
    """
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
    jpgs = sum(n.lower().endswith(".jpg") for n in names)
    info: dict = {"files": len(names), "jpg": jpgs}
    if expected is not None:
        info["expected"] = expected
        info["ok"] = (jpgs if count_jpg_only else len(names)) == expected
    return info


def cmd_fetch(root: Path, only: list[str] | None) -> int:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    ck_path = root / "checksums.json"
    checksums = load_checksums(ck_path)
    free = shutil.disk_usage(raw).free
    todo = [s for s in SOURCES if not only or s.name in only]
    need = sum(s.size for s in todo)
    print(f"Destination: {raw}  free: {human(free)}  to download (max): {human(need)}")
    if free < need * 1.05:
        print("Not enough free space on the destination.")
        return 1
    for src in todo:
        checksums[src.name] = download_one(src, raw, checksums)
        save_checksums(ck_path, checksums)  # save after every file
    print("\nZip contents:")
    ok = True
    for src in todo:
        if src.name.endswith(".zip"):
            info = check_zip(raw / src.name, src.expected_entries, src.kind == "images")
            checksums[src.name]["zip"] = info
            ok &= info.get("ok", True)
            print(f"  {src.name}: {info}")
    save_checksums(ck_path, checksums)
    print(f"\nChecksums: {ck_path}")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


def cmd_verify(root: Path, reference: Path | None) -> int:
    """Re-hash files on disk and compare with the recorded (or committed) checksums."""
    raw = root / "raw"
    ref = load_checksums(reference) if reference else load_checksums(root / "checksums.json")
    if not ref:
        print("No reference checksums found.")
        return 1
    ok = True
    for name, rec in sorted(ref.items()):
        path = raw / name
        if not path.exists():
            print(f"[missing] {name}")
            ok = False
            continue
        t0 = time.monotonic()
        digest = sha256_file(path).hexdigest()
        match = digest == rec["sha256"] and path.stat().st_size == rec["size"]
        ok &= match
        print(f"[{'ok' if match else 'MISMATCH'}] {name} ({time.monotonic() - t0:.0f}s)")
    print("ALL OK" if ok else "VERIFY FAILED")
    return 0 if ok else 1


def cmd_extract(root: Path, dest: Path, splits: list[str]) -> int:
    """Unzip annotations and the requested image splits to local disk, timing each step."""
    raw = root / "raw"
    dest.mkdir(parents=True, exist_ok=True)
    timings = {}
    for name, sub in (("vqa_annotations.zip", "vqa"), ("quality_annotations.zip", "quality")):
        with zipfile.ZipFile(raw / name) as zf:
            zf.extractall(dest / "annotations" / sub)
    shutil.copy2(raw / "VQA_test.json", dest / "annotations" / "vqa" / "VQA_test.json")
    for split in splits:
        t0 = time.monotonic()
        with zipfile.ZipFile(raw / f"{split}.zip") as zf:
            zf.extractall(dest / "images")
        secs = round(time.monotonic() - t0, 1)
        n = sum(1 for _ in (dest / "images").rglob(f"VizWiz_{split}_*.jpg"))
        timings[split] = {"seconds": secs, "jpg_files": n}
        print(f"[extract] {split}: {n} images in {secs}s")
    report = {"dest": str(dest), "timings": timings}
    print(json.dumps(report, indent=2))
    (root / "extract_timings.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download and verify VizWiz files")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--root", type=Path, required=True)
    f.add_argument("--only", nargs="*", help="file names from SOURCES (default: all)")
    v = sub.add_parser("verify")
    v.add_argument("--root", type=Path, required=True)
    v.add_argument("--checksums", type=Path, help="reference checksums (default: <root>)")
    e = sub.add_parser("extract")
    e.add_argument("--root", type=Path, required=True)
    e.add_argument("--dest", type=Path, required=True)
    e.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "fetch":
        return cmd_fetch(args.root, args.only)
    if args.cmd == "verify":
        return cmd_verify(args.root, args.checksums)
    return cmd_extract(args.root, args.dest, args.splits)


if __name__ == "__main__":
    sys.exit(main())
