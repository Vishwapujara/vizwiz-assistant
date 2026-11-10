import json
import math

import pytest
from PIL import Image

from vizwiz.eval.run_eval import (
    build_parser,
    class_ids_from,
    evaluate,
    sample_records,
    unanswerable_score,
)

# Toy tokenizer: first token = first 2 characters.
VOCAB: dict[str, int] = {}


def encode(word):
    return [VOCAB.setdefault(word[:2], len(VOCAB) + 1), 0]


def test_class_ids_cover_casings_and_do_not_overlap():
    ids = class_ids_from(encode)
    assert len(ids["unanswerable"]) == 3 and len(ids["answerable"]) == 3
    assert not ids["unanswerable"] & ids["answerable"]


def test_unanswerable_score_from_logprobs_and_fallback():
    ids = {"unanswerable": {1}, "answerable": {2}}
    s, ok = unanswerable_score({1: math.log(0.3), 2: math.log(0.1), 9: math.log(0.6)}, ids, "")
    assert ok and s == pytest.approx(0.75)
    assert unanswerable_score({9: 0.0}, ids, "Unanswerable") == (1.0, False)
    assert unanswerable_score({}, ids, "answerable") == (0.0, False)


def test_sample_records_is_deterministic_subset():
    recs = [{"id": f"r{i}"} for i in range(50)]
    a = sample_records(recs, 10, seed=1)
    assert a == sample_records(recs, 10, seed=1) and len(a) == 10
    assert sample_records(recs, None, 0) == recs


class FakeGenerator:
    """Answers 'red', says unanswerable when the question contains 'blurry', and
    always reports blur."""

    def class_token_ids(self):
        return {"unanswerable": {1}, "answerable": {2}}

    def generate(self, task, records, images):
        assert all(isinstance(im, Image.Image) and im.width % 32 == 0 for im in images)
        out = []
        for r in records:
            if task == "answer":
                text, lp = "Red.", {}
            elif task == "answerability":
                unans = "blurry" in r["question"]
                text = "unanswerable" if unans else "answerable"
                lp = {1: math.log(0.9 if unans else 0.2), 2: math.log(0.1 if unans else 0.8)}
            else:
                text, lp = '["blur"]', {}
            out.append({"text": text, "first_logprobs": lp, "tokens": 3})
        return out


def write_split(tmp_path):
    rec_dir, img_dir = tmp_path / "processed", tmp_path / "images" / "val"
    rec_dir.mkdir()
    img_dir.mkdir(parents=True)
    vqa, quality = [], []
    for i in range(6):
        name = f"VizWiz_val_{i:08d}.jpg"
        Image.new("RGB", (100, 140), (i * 40, 0, 0)).save(img_dir / name)
        unans = i % 3 == 0
        vqa.append({
            "id": f"vqa/{name}", "image": name, "question": "blurry?" if unans else "color?",
            "answers": ["unanswerable"] * 10 if unans else ["red"] * 10,
            "answer_type": "unanswerable" if unans else "other",
            "answerability": "unanswerable" if unans else "answerable",
            "unanswerable_votes": 10 if unans else 0, "target": "red",
            "quality": ["blur"] if i % 2 else ["none"],
        })  # fmt: skip
        quality.append({"id": f"quality/{name}", "image": name, "labels": ["blur"]})
    for task, recs in (("vqa", vqa), ("quality", quality)):
        (rec_dir / f"{task}_test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    return rec_dir, tmp_path / "images"


def test_evaluate_end_to_end(tmp_path):
    rec_dir, images = write_split(tmp_path)
    args = build_parser().parse_args(
        ["--records", str(rec_dir), "--images", str(images), "--run-id", "t",
         "--out", str(tmp_path / "run"), "--batch-size", "4", "--max-pixels", "4096"]
    )  # fmt: skip
    report = evaluate(FakeGenerator(), args, {"run_id": "t"})
    t = report["tasks"]
    assert t["answer"]["metrics"]["n"] == 4  # answerable questions only
    assert t["answer"]["metrics"]["vqa_accuracy"] == 100.0
    assert t["answerability"]["metrics"]["average_precision"] == 100.0
    assert t["answerability"]["timing"]["score_fallbacks"] == 0
    assert t["quality"]["metrics"]["per_label"]["blur"]["f1"] == 1.0
    saved = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert saved["tasks"].keys() == t.keys()
    rows = (tmp_path / "run" / "predictions_answerability.jsonl").read_text().splitlines()
    assert len(rows) == 6 and "score" in json.loads(rows[0])
