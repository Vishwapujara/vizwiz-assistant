import json

import pytest

from vizwiz.eval.compare import compare
from vizwiz.eval.metrics import paired_bootstrap_diff


def test_paired_bootstrap_detects_a_consistent_small_gain():
    # run 1 is right on 5 more of 400 examples than run 0, never worse.
    a = [1] * 200 + [0] * 200
    b = [1] * 205 + [0] * 195
    vals = [a, b]
    rows = paired_bootstrap_diff(400, lambda idx, r: sum(vals[r][i] for i in idx) / len(idx), 2)
    assert rows[1]["diff_vs_reference"] == pytest.approx(5 / 400)
    lo, hi = rows[1]["diff_ci95"]
    assert 0 <= lo <= hi  # paired CI excludes negative differences
    assert rows[1]["p_better"] > 0.9 and rows[0]["diff_vs_reference"] == 0


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_compare_answer_runs(tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    vqa = [{"id": f"q{i}", "answers": ["red"] * 10, "answerability": "answerable"}
           for i in range(20)]  # fmt: skip
    write(rec / "vqa_val.jsonl", vqa)
    for name, n_right in (("low", 10), ("high", 15)):
        d = tmp_path / name
        d.mkdir()
        preds = [{"id": f"q{i}", "prediction": "red" if i < n_right else "blue"} for i in range(20)]
        write(d / "predictions_answer.jsonl", preds)
    out = compare(rec, "val", "answer", [tmp_path / "low", tmp_path / "high"],
                  ["low", "high"], "low", n_resamples=200)  # fmt: skip
    assert out["n"] == 20
    assert out["runs"]["low"]["value"] == 50.0 and out["runs"]["high"]["value"] == 75.0
    assert out["runs"]["high"]["diff_vs_reference"] == 25.0
