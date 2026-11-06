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
    m, excluded = assign_splits(official(), CFG)
    check_no_leakage(m)
    assert excluded == set()
    assert len(m["train"]) + len(m["val"]) == 200
    assert 15 <= len(m["val"]) <= 25  # ~10%, rounded per stratum
    assert len(m["test"]) == 20 and len(m["test_vqa"]) == 10


def test_leakage_check_catches_duplicates():
    with pytest.raises(ValueError, match="both"):
        check_no_leakage({"train": {"a.jpg"}, "val": {"a.jpg"}})


def test_splits_are_deterministic():
    assert assign_splits(official(), CFG) == assign_splits(official(), CFG)


def fake_hashes(o, same_groups):
    """Unique hash per image, except images listed together in a group share one."""
    names = {e["image"] for part in o.values() for split in part.values() for e in split}
    h = {n: f"h-{n}" for n in names}
    for i, group in enumerate(same_groups):
        for n in group:
            h[n] = f"dup-{i}"
    return h


def test_copies_of_test_images_are_excluded_from_training():
    o = official()
    hashes = fake_hashes(o, [["VizWiz_train_00000003.jpg", "VizWiz_val_00000001.jpg"]])
    m, excluded = assign_splits(o, CFG, hashes)
    assert excluded == {"VizWiz_train_00000003.jpg"}
    assert "VizWiz_val_00000001.jpg" in m["test"]  # test set unchanged
    check_no_leakage(m, hashes)


def test_identical_train_pool_files_land_in_the_same_split():
    o = official()
    group = [f"VizWiz_train_{i:08d}.jpg" for i in range(30, 60)]  # one 30-file group
    hashes = fake_hashes(o, [group])
    m, _ = assign_splits(o, CFG, hashes)
    assert set(group) <= m["train"] or set(group) <= m["val"]
    check_no_leakage(m, hashes)


def test_leakage_check_catches_identical_files_across_splits():
    m = {"train": {"a.jpg"}, "val": set(), "test": {"b.jpg"}, "test_vqa": {"c.jpg"}}
    with pytest.raises(ValueError, match="identical files shared by train and test"):
        check_no_leakage(m, {"a.jpg": "x", "b.jpg": "x", "c.jpg": "y"})
    # test and test_vqa may share files: both are evaluation-only.
    m = {"train": {"a.jpg"}, "val": set(), "test": {"b.jpg"}, "test_vqa": {"c.jpg"}}
    check_no_leakage(m, {"a.jpg": "x", "b.jpg": "y", "c.jpg": "y"})


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
    assert (tmp_path / "res1" / "excluded.json").exists()
    # An image's VQA record and quality record share a split.
    split_of = {}
    for name in r1["hashes"]:
        for line in (tmp_path / "out1" / name).read_text().splitlines():
            rec = json.loads(line)
            assert split_of.setdefault(rec["image"], rec["split"]) == rec["split"]
