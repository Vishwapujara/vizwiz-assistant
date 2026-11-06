import pytest

from vizwiz.eval.vqa_metric import normalize_answer, vqa_accuracy


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Two", "2"),
        ("The Red Car.", "red car"),
        ("none", "0"),  # official quirk
        ("  yes\n", "yes"),
        ("dont know", "don’t know"),  # contraction map uses a curly apostrophe
        ("3.5", "3.5"),  # period before a digit is kept
        ("coca-cola", "coca cola"),  # '-' replaced by a space when not next to a space
        ("1,000", "1000"),  # comma between digits: punctuation removed, not spaced
        ("a b c", "b c"),
    ],
)
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected


def test_accuracy_full_credit_with_three_or_more_matches():
    gts = ["yes"] * 4 + ["no"] * 6
    # Leave-one-out: dropping a "yes" leaves 3 matches (1.0); dropping a "no" leaves 4 (1.0).
    assert vqa_accuracy("yes", gts) == pytest.approx(1.0)


def test_accuracy_partial_credit():
    gts = ["blue"] * 2 + ["green"] * 8
    # Dropping a "blue" (2 of 10): 1 match -> 1/3. Dropping a "green" (8 of 10): 2 -> 2/3.
    expected = (2 * (1 / 3) + 8 * (2 / 3)) / 10
    assert vqa_accuracy("Blue", gts) == pytest.approx(expected)


def test_accuracy_zero_and_single_match():
    gts = ["cat"] + ["dog"] * 9
    # Dropping the "cat": 0 matches. Dropping a "dog": 1 match -> 1/3.
    assert vqa_accuracy("cat", gts) == pytest.approx(9 * (1 / 3) / 10)
    assert vqa_accuracy("bird", gts) == 0.0


def test_ground_truth_is_not_normalised():
    # The official code compares raw ground-truth strings with the normalised prediction.
    assert vqa_accuracy("Two", ["two"] * 10) == 0.0
    assert vqa_accuracy("Two", ["2"] * 10) == pytest.approx(1.0)
