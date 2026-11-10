import pytest

from vizwiz.eval.metrics import (
    answer_metrics,
    answerability_metrics,
    average_precision,
    bootstrap_ci,
    parse_answerability,
    parse_quality,
    quality_metrics,
    recall_at_precision,
)

ISSUES = ["blur", "framing", "rotation", "dark", "bright", "obstruction", "unrecognizable"]


def test_average_precision_known_values():
    # Hand-computed: ranking 1,0,1 -> AP = (1/2)*1 + (1/2)*(2/3)
    assert average_precision([1, 0, 1], [0.9, 0.8, 0.7]) == pytest.approx(0.5 + 1 / 3)
    assert average_precision([1, 1, 0], [0.9, 0.8, 0.1]) == pytest.approx(1.0)
    # Tied scores form one threshold: precision 2/4 at recall 1.
    assert average_precision([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    assert average_precision([0, 0], [0.1, 0.2]) is None


def test_recall_at_precision():
    y, s = [1, 1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6, 0.5]
    assert recall_at_precision(y, s, 1.0) == pytest.approx(2 / 3)
    assert recall_at_precision(y, s, 0.75) == pytest.approx(1.0)  # 3/4 at recall 1
    assert recall_at_precision([0, 1], [0.9, 0.1], 0.9) == 0.0


def test_parsers():
    assert parse_answerability("Unanswerable.") == "unanswerable"
    assert parse_answerability(" answerable") == "answerable"
    assert parse_answerability("I think so") is None
    assert parse_quality('["blur", "dark"]', [*ISSUES, "none"]) == ({"blur", "dark"}, True)
    assert parse_quality('["none"]', [*ISSUES, "none"]) == (set(), True)
    assert parse_quality('```json\n["blur"]\n```', [*ISSUES, "none"]) == ({"blur"}, True)
    assert parse_quality("blur", [*ISSUES, "none"]) == (set(), False)
    assert parse_quality('["sparkly"]', [*ISSUES, "none"]) == (set(), False)
    assert parse_quality('["none", "blur"]', [*ISSUES, "none"]) == (set(), False)


def test_answer_metrics_slices():
    ex = [
        {
            "prediction": "Red.",
            "answers": ["red"] * 10,
            "answer_type": "other",
            "quality": ["none"],
        },
        {
            "prediction": "blue",
            "answers": ["red"] * 10,
            "answer_type": "other",
            "quality": ["blur"],
        },
        {"prediction": "yes", "answers": ["yes"] * 10, "answer_type": "yes/no", "quality": None},
    ]
    m = answer_metrics(ex, n_resamples=50)
    assert m["vqa_accuracy"] == pytest.approx(66.67)
    assert m["by_answer_type"]["other"]["accuracy"] == 50.0
    assert m["by_quality_label"]["blur"]["accuracy"] == 0.0
    assert m["by_quality_any_issue"] == {
        "any_issue": {"n": 1, "accuracy": 0.0},
        "no_issue": {"n": 1, "accuracy": 100.0},
    }
    lo, hi = m["vqa_accuracy_ci95"]
    assert 0 <= lo <= hi <= 100


def test_answerability_metrics():
    def ex_(pred, score, gt, votes):
        return {
            "prediction": pred,
            "score": score,
            "answerability": gt,
            "unanswerable_votes": votes,
        }

    ex = [
        ex_("unanswerable", 0.9, "unanswerable", 9),
        ex_("answerable", 0.2, "answerable", 4),
        ex_("unanswerable", 0.6, "answerable", 3),
        ex_("hmm", 0.4, "unanswerable", 7),
    ]
    m = answerability_metrics(ex, n_resamples=50)
    assert m["positives_unanswerable"] == 2
    assert m["average_precision"] == pytest.approx(100 * (0.5 * 1 + 0.5 * 2 / 3), abs=0.01)
    assert m["invalid_output_pct"] == 25.0
    assert m["confusion"]["unanswerable->invalid"] == 1
    assert m["borderline_answerable"] == {"n": 2, "predicted_unanswerable_pct": 50.0}
    assert m["official_f1_at_0.5"] == pytest.approx(50.0)  # tp1 fp1 fn1


def test_quality_metrics():
    ex = [
        {"prediction": '["blur"]', "labels": ["blur"]},
        {"prediction": '["none"]', "labels": ["framing"]},
        {"prediction": "not json", "labels": ["none"]},
    ]
    m = quality_metrics(ex, ISSUES, n_resamples=50)
    assert m["per_label"]["blur"]["f1"] == 1.0
    assert m["per_label"]["framing"]["recall"] == 0.0
    assert m["json_valid_pct"] == pytest.approx(66.67)
    assert m["exact_match_pct"] == pytest.approx(66.67)  # invalid predicts {} == gold {}


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    items = [0.0] * 50 + [1.0] * 50
    a = bootstrap_ci(items, lambda xs: sum(xs) / len(xs), 200, seed=3)
    assert a == bootstrap_ci(items, lambda xs: sum(xs) / len(xs), 200, seed=3)
    assert a[0] < 0.5 < a[1]
