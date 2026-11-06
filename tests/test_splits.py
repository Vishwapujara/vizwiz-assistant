import json

import pytest

from vizwiz.data.config import load_data_config
from vizwiz.data.splits import assign_splits, build, check_no_leakage

CFG = load_data_config("configs/data.yaml")
FLAWS = dict.fromkeys(["BLR", "BRT", "DRK", "FRM", "OBS", "ROT", "OTH", "NON"], 0)


def vqa(image, answerable=1, atype="other"):
    ans = ["red"] * 10 if answerable else ["unanswerable"] * 10
    return {
        "image": image,
        "question": "q",
        "answerable": answerable,
        "answer_type": atype if answerable else "unanswerable",
        "answers": [{"answer": a, "answer_confidence": "yes"} for a in ans],
    }


def qual(image, unrec=0):
    return {"image": image, "flaws": dict(FLAWS), "unrecognizable": unrec}


def official():
    tr = [f"VizWiz_train_{i:08d}.jpg" for i in range(200)]
    va = [f"VizWiz_val_{i:08d}.jpg" for i in range(20)]
    te = [f"VizWiz_test_{i:08d}.jpg" for i in range(10)]
    return {
        "vqa": {
            "train": [vqa(x, answerable=i % 4 != 0) for i, x in enumerate(tr[:150])],
            "val": [vqa(x) for x in va[:15]],
            "test": [vqa(x) for x in te],
        },
        "quality": {
            "train": [qual(x, unrec=(i % 5 == 0) * 3) for i, x in enumerate(tr[20:])],
            "val": [qual(x) for x in va],
        },
    }


def test_every_image_in_exactly_one_split():
    m = assign_splits(official(), CFG)
    check_no_leakage(m)
    assert len(m["train"]) + len(m["val"]) == 200
    assert 15 <= len(m["val"]) <= 25  # ~10%, rounded per stratum
    assert len(m["test"]) == 20 and len(m["test_vqa"]) == 10


def test_leakage_check_catches_duplicates():
    with pytest.raises(ValueError, match="both"):
        check_no_leakage({"train": {"a.jpg"}, "val": {"a.jpg"}})


def test_splits_are_deterministic():
    assert assign_splits(official(), CFG) == assign_splits(official(), CFG)


def test_build_groups_tasks_by_image_and_is_reproducible(tmp_path):
    root = tmp_path / "ann"
    (root / "vqa").mkdir(parents=True)
    (root / "quality").mkdir()
    o = official()
    (root / "vqa" / "train.json").write_text(json.dumps(o["vqa"]["train"]))
    (root / "vqa" / "val.json").write_text(json.dumps(o["vqa"]["val"]))
    (root / "vqa" / "VQA_test.json").write_text(json.dumps(o["vqa"]["test"]))
    (root / "quality" / "train.json").write_text(json.dumps(o["quality"]["train"]))
    (root / "quality" / "val.json").write_text(json.dumps(o["quality"]["val"]))
    r1 = build(root, tmp_path / "out1", tmp_path / "res1", CFG)
    r2 = build(root, tmp_path / "out2", tmp_path / "res2", CFG)
    assert r1["hashes"] == r2["hashes"]
    # An image's VQA record and quality record share a split.
    split_of = {}
    for name in r1["hashes"]:
        for line in (tmp_path / "out1" / name).read_text().splitlines():
            rec = json.loads(line)
            assert split_of.setdefault(rec["image"], rec["split"]) == rec["split"]
