import json

from vizwiz.serve.tune import cascade_metrics, choose, outcome, parity, tune


def rec(i, answerable=True):
    answers = ["red"] * 10 if answerable else ["unanswerable"] * 10
    return {"id": f"vqa/{i}", "image": f"img{i}.jpg", "answers": answers,
            "answerability": "answerable" if answerable else "unanswerable"}  # fmt: skip


def row(i, score, answer="red", issues=()):
    return {"id": f"vqa/{i}", "score": score, "answer": answer, "issues": list(issues)}


def test_outcome_rules():
    u = {"unrecognizable"}
    assert outcome(row(0, 0.1), 0.5, u, True) == "red"
    assert outcome(row(0, 0.6), 0.5, u, True) is None
    assert outcome(row(0, 0.1, issues=["unrecognizable"]), 0.5, u, True) is None
    assert outcome(row(0, 0.1, issues=["unrecognizable"]), 0.5, u, False) == "red"
    assert outcome(row(0, 0.1, answer=""), 0.5, u, True) is None


def test_cascade_metrics_and_choice():
    records = [rec(0), rec(1), rec(2, answerable=False), rec(3, answerable=False)]
    rows = [row(0, 0.2), row(1, 0.4, answer="blue"), row(2, 0.7), row(3, 0.3, answer="x")]
    m = cascade_metrics(rows, records, 0.5, set())
    # answered 0 (right), 1 (wrong), 3 (wrong); declined 2 (right: "unanswerable")
    assert m["vizwiz_accuracy"] == 50.0 and m["answered_pct"] == 75.0
    assert m["wrong_answers_pct_of_all"] == 50.0 and m["missed_unanswerable_pct"] == 50.0
    assert m["wrongly_declined_pct_of_answerable"] == 0.0
    best = choose([cascade_metrics(rows, records, t, set()) for t in (0.25, 0.5, 0.9)])
    assert best["threshold"] == 0.25  # declines 1 and 3 too: accuracy 75
    rep = tune(rows, records, set(), rows, records)
    assert rep["chosen_threshold"] == 0.25 and rep["test"]["chosen"]["vizwiz_accuracy"] == 75.0
    assert rep["val"]["never_decline"]["answered_pct"] == 100.0


def test_parity(tmp_path):
    records = [rec(0), rec(1)]
    rows = [row(0, 0.2, answer="Red", issues=["blur"]), row(1, 0.6, answer="blue")]
    (tmp_path / "predictions_answerability.jsonl").write_text(
        "\n".join(json.dumps({"id": f"vqa/{i}", "prediction": "x", "score": s})
                  for i, s in ((0, 0.25), (1, 0.4))) + "\n")  # fmt: skip
    (tmp_path / "predictions_answer.jsonl").write_text(
        json.dumps({"id": "vqa/0", "prediction": "red\n"}) + "\n")  # fmt: skip
    (tmp_path / "predictions_quality.jsonl").write_text(
        "\n".join(json.dumps({"id": f"quality/img{i}.jpg", "prediction": p})
                  for i, p in ((0, '["blur"]'), (1, '["dark"]'))) + "\n")  # fmt: skip
    p = parity(rows, records, tmp_path)
    assert p["answerability"]["n"] == 2 and p["answerability"]["max_abs_score_diff"] == 0.2
    assert p["answerability"]["same_decision_at_0.5_pct"] == 50.0
    assert p["answer"]["same_answer_pct"] == 100.0
    assert p["quality"] == {"n": 2, "same_labels_pct": 50.0}
