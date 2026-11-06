from vizwiz.data.config import load_data_config
from vizwiz.data.labels import answer_target, quality_labels, vqa_record

CFG = load_data_config("configs/data.yaml")


def test_target_is_majority_when_clear():
    assert answer_target(["red"] * 6 + ["blue"] * 4)[0] == "red"


def test_target_prefers_metric_matchable_answer():
    # Predictions are normalised; ground truth is not. "Coca-Cola" becomes "coca cola",
    # which equals none of the raw answers, so it scores 0 even with the most votes.
    answers = ["Coca-Cola"] * 4 + ["soda"] * 3 + ["pop"] * 3
    target, acc = answer_target(answers)
    assert target == "soda"
    assert acc > 0


def test_equivalent_forms_tie_and_votes_decide():
    # Both forms normalise to "coca cola" and score the same; more votes wins.
    answers = ["Coca-Cola"] * 4 + ["coca cola"] * 3 + ["soda"] * 3
    assert answer_target(answers)[0] == "Coca-Cola"


def test_target_never_unanswerable_for_answerable_question():
    answers = ["unanswerable"] * 5 + ["milk"] * 3 + ["juice"] * 2
    assert answer_target(answers, answerable=True)[0] == "milk"


def test_target_tie_break_votes_then_order():
    # Both score 1.0 (>= 4 votes); "b" has more votes.
    assert answer_target(["a"] * 4 + ["b"] * 6)[0] == "b"
    # Equal votes and score: the earliest-listed answer wins.
    assert answer_target(["x"] * 5 + ["y"] * 5)[0] == "x"


def flaws(**kw):
    base = dict.fromkeys(["BLR", "BRT", "DRK", "FRM", "OBS", "ROT", "OTH", "NON"], 0)
    return {**base, **kw}


def test_quality_threshold_and_oth_dropped():
    entry = {"flaws": flaws(BLR=2, FRM=1, OTH=5), "unrecognizable": 2}
    assert quality_labels(entry, CFG) == ["blur", "unrecognizable"]


def test_quality_none_when_nothing_reaches_k():
    entry = {"flaws": flaws(BLR=1, NON=4), "unrecognizable": 1}
    assert quality_labels(entry, CFG) == ["none"]


def test_unanswerable_record_has_unanswerable_target():
    entry = {
        "image": "VizWiz_train_00000001.jpg",
        "question": "q",
        "answerable": 0,
        "answer_type": "unanswerable",
        "answers": [{"answer": "unanswerable"}] * 10,
    }
    rec = vqa_record(entry, "train", None, CFG)
    assert rec["answerability"] == "unanswerable"
    assert rec["target"] == "unanswerable"
    assert rec["quality"] is None
