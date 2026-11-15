import json

from vizwiz.eval.errors import (
    agreement,
    analyse,
    question_kind,
    to_markdown,
    transitions,
    votes_bucket,
)


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_question_kind_rules():
    assert question_kind("Is the light on?", "yes/no") == "yes_no"
    assert question_kind("What color is this shirt?", "other") == "color"
    assert question_kind("How much does this cost?", "other") == "count_number"
    assert question_kind("Can you read this label?", "other") == "read_text"
    assert question_kind("What is this?", "other") == "identify"
    assert question_kind("Where am I?", "other") == "other"


def test_agreement_and_votes_buckets():
    assert agreement(["a"] * 7 + ["b"] * 3) == "high (7+ of 10)"
    assert agreement(["a"] * 4 + list("bcdefg")) == "medium (4-6)"
    assert agreement(list("abcdefghij")) == "low (1-3)"
    assert [votes_bucket(v) for v in (0, 2, 3, 5, 6)] == ["0", "1-2", "3-5", "3-5", "6-10"]


def test_transitions():
    assert transitions([False, True, False, True], [True, False, False, True]) == {
        "fixed": 1, "broken": 1, "still_wrong": 1, "still_right": 1}  # fmt: skip


def _vqa(i, answerable=True, quality=("none",)):
    return {"id": f"vqa/{i}", "question": "What color is it?", "answer_type": "other",
            "answers": ["red"] * 10 if answerable else ["unanswerable"] * 10,
            "answerability": "answerable" if answerable else "unanswerable",
            "unanswerable_votes": 0 if answerable else 10, "target": "red",
            "quality": list(quality)}  # fmt: skip


def test_analyse_all_tasks(tmp_path):
    rec, before, after = tmp_path / "rec", tmp_path / "e1", tmp_path / "e2"
    for d in (rec, before, after):
        d.mkdir()
    vqa = [_vqa(0), _vqa(1, quality=["blur"]), _vqa(2, answerable=False)]
    quality = [{"id": "quality/0", "labels": ["none"]}, {"id": "quality/1", "labels": ["blur"]}]
    write(rec / "vqa_test.jsonl", vqa)
    write(rec / "quality_test.jsonl", quality)
    # answer: E1 wrong on both (long sentence), E2 right on 0 only
    write(before / "predictions_answer.jsonl",
          [{"id": "vqa/0", "prediction": "It is a red shirt here"},
           {"id": "vqa/1", "prediction": "It is a red shirt here"}])  # fmt: skip
    write(
        after / "predictions_answer.jsonl",
        [{"id": "vqa/0", "prediction": "red"}, {"id": "vqa/1", "prediction": "blue"}],
    )
    # answerability: E1 declines nothing, E2 declines the unanswerable one
    write(
        before / "predictions_answerability.jsonl",
        [{"id": f"vqa/{i}", "prediction": "x", "score": 0.1} for i in range(3)],
    )
    write(after / "predictions_answerability.jsonl",
          [{"id": f"vqa/{i}", "prediction": "x", "score": 0.9 if i == 2 else 0.1}
           for i in range(3)])  # fmt: skip
    # quality: E1 flags blur on the clean photo; E2 gets both right
    write(before / "predictions_quality.jsonl",
          [{"id": "quality/0", "prediction": '["blur"]'},
           {"id": "quality/1", "prediction": '["dark"]'}])  # fmt: skip
    write(after / "predictions_quality.jsonl",
          [{"id": "quality/0", "prediction": "[]"},
           {"id": "quality/1", "prediction": '["blur"]'}])  # fmt: skip

    rep = analyse(rec, "test", before,
                  {"answer": after, "answerability": after, "quality": after})  # fmt: skip

    a = rep["answer"]
    assert a["n"] == 2 and a["transitions"]["fixed"] == 1 and a["transitions"]["still_wrong"] == 1
    assert a["by_photo"]["clean photo"]["accuracy_after"] == 100.0
    assert a["remaining_errors"]["kinds"] == {"no answerer agreed": 1}
    assert a["answer_length_words"]["before_over_3_words_pct"] == 100.0

    b = rep["answerability"]
    assert b["overall"]["before"]["missed_unanswerable_pct"] == 100.0
    assert b["overall"]["after"]["missed_unanswerable_pct"] == 0.0
    assert b["overall"]["after"]["wrongly_declined_pct"] == 0.0
    assert b["transitions"]["fixed"] == 1

    q = rep["quality"]
    assert q["clean_photos"]["false_alarm_before_pct"] == 100.0
    assert q["clean_photos"]["false_alarm_after_pct"] == 0.0
    assert q["photos_with_issues"]["all_issues_missed_before_pct"] == 100.0
    assert q["transitions_exact_match"]["fixed"] == 2

    md = to_markdown(rep)
    assert "## Answering" in md and "## Quality" in md and "What color" not in md
