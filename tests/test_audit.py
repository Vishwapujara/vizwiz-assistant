import json

from vizwiz.data.audit import main, run_audit


def q(image, answerable, answers, atype="other"):
    return {
        "image": image,
        "question": "what is this",
        "answerable": answerable,
        "answer_type": atype,
        "answers": [{"answer": a, "answer_confidence": "yes"} for a in answers],
    }


def write_fixture(root):
    (root / "vqa").mkdir(parents=True)
    (root / "quality").mkdir()
    train = [
        q("VizWiz_train_00000000.jpg", 1, ["red"] * 6 + ["blue"] * 4),
        q("VizWiz_train_00000001.jpg", 0, ["unanswerable"] * 10, "unanswerable"),
    ]
    for name in ("train.json", "val.json", "VQA_test.json"):
        (root / "vqa" / name).write_text(json.dumps(train))
    flaws = dict.fromkeys(["BLR", "BRT", "DRK", "FRM", "OBS", "ROT", "OTH", "NON"], 0)
    quality = [
        {"image": "VizWiz_train_00000000.jpg", "flaws": {**flaws, "NON": 5}, "unrecognizable": 0},
        {"image": "VizWiz_train_00000001.jpg", "flaws": {**flaws, "BLR": 4}, "unrecognizable": 3},
    ]
    for name in ("train.json", "val.json"):
        (root / "quality" / name).write_text(json.dumps(quality))


def test_audit_numbers(tmp_path):
    write_fixture(tmp_path)
    a = run_audit(tmp_path)
    t = a["vqa"]["train"]
    assert t["questions"] == 2
    assert t["unanswerable_pct"] == 50.0
    assert t["oracle_vqa_accuracy_pct"] == 100.0
    k2 = a["quality"]["train"]["per_k"]["2"]
    assert k2["flags"]["BLR"]["n"] == 1
    assert k2["unrecognizable"]["n"] == 1
    x = a["vqa_x_quality"]["train"]["k2"]
    assert x["unanswerable_pct_if_unrecognizable"] == 100.0
    assert x["of_unanswerable_unrecognizable_pct"] == 100.0


def test_report_has_no_question_text(tmp_path):
    write_fixture(tmp_path / "ann")
    main(["--annotations", str(tmp_path / "ann"), "--out", str(tmp_path / "out")])
    report = (tmp_path / "out" / "audit_report.md").read_text()
    assert "what is this" not in report
    assert "VizWiz_train_" not in report
