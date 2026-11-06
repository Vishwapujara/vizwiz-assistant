import json

from PIL import Image

from vizwiz.data.images import analyse, main, scan_one


def save(path, color, size=(64, 48)):
    Image.new("RGB", size, color).save(path, format="JPEG")


def test_scan_and_cross_split_duplicate_detection(tmp_path):
    imgs = tmp_path / "images"
    imgs.mkdir()
    save(imgs / "VizWiz_train_00000000.jpg", (255, 0, 0))
    save(imgs / "VizWiz_val_00000000.jpg", (255, 0, 0))  # identical bytes, other split
    save(imgs / "VizWiz_test_00000000.jpg", (0, 0, 255), size=(48, 64))
    membership = {
        "train": ["VizWiz_train_00000000.jpg"],
        "val": ["VizWiz_val_00000000.jpg"],
        "test": ["VizWiz_test_00000000.jpg"],
    }
    scans = {n: scan_one(str(imgs / n)) for names in membership.values() for n in names}
    stats, cross = analyse(scans, membership)
    assert stats["duplicates"]["groups_spanning_splits"] == 1
    assert stats["per_split"]["test"]["portrait_pct"] == 100.0
    (tmp_path / "m.json").write_text(json.dumps(membership))
    rc = main(["--images", str(imgs), "--membership", str(tmp_path / "m.json"),
               "--results", str(tmp_path / "res"), "--workers", "1"])  # fmt: skip
    assert rc == 1  # leakage found -> non-zero exit
    assert len(json.loads((tmp_path / "res" / "image_hashes.json").read_text())) == 3
