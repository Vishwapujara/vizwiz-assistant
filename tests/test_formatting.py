import json

import pytest
from PIL import Image

from vizwiz.data.formatting import (
    IGNORE_INDEX,
    SYSTEM_PROMPTS,
    build_messages,
    encode,
    target_text,
    task_records,
)

IMAGE_PAD = 999


class FakeProcessor:
    """Mimics apply_chat_template: renders text, image = 4 pad tokens, ids = chars."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict):
        ids = []
        for m in messages:
            ids += [ord(c) for c in f"<{m['role']}>"]
            content = m["content"]
            parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for part in parts:
                if part["type"] == "image":
                    ids += [IMAGE_PAD] * 4
                else:
                    ids += [ord(c) for c in part["text"]]
            ids += [ord("|")]  # end of turn
        if add_generation_prompt:
            ids += [ord(c) for c in "<assistant>"]
        return {"input_ids": [ids]}


VQA = [
    {"question": "what is it", "target": "milk", "answerability": "answerable", "image": "a"},
    {"question": "what", "target": "unanswerable", "answerability": "unanswerable", "image": "b"},
]
QUALITY = [{"labels": ["blur", "framing"], "image": "a"}]


def test_task_records():
    assert len(task_records("answer", VQA, QUALITY)) == 1
    assert len(task_records("answerability", VQA, QUALITY)) == 2
    assert len(task_records("quality", VQA, QUALITY)) == 1
    with pytest.raises(ValueError):
        task_records("nope", VQA, QUALITY)


def test_targets():
    assert target_text("answer", VQA[0]) == "milk"
    assert target_text("answerability", VQA[1]) == "unanswerable"
    assert json.loads(target_text("quality", QUALITY[0])) == ["blur", "framing"]


def test_quality_prompt_has_no_question_and_lists_labels():
    msgs = build_messages("quality", QUALITY[0], Image.new("RGB", (32, 32)), with_target=False)
    assert [p["type"] for p in msgs[1]["content"]] == ["image"]
    assert "obstruction" in SYSTEM_PROMPTS["quality"] and "none" in SYSTEM_PROMPTS["quality"]


@pytest.mark.parametrize("task,rec", [("answer", VQA[0]), ("quality", QUALITY[0])])
def test_only_target_tokens_are_trained(task, rec):
    enc = encode(FakeProcessor(), task, rec, Image.new("RGB", (32, 32)))
    assert enc["prefix_ok"]
    trained = [
        t for t, lbl in zip(enc["input_ids"], enc["labels"], strict=True) if lbl != IGNORE_INDEX
    ]
    assert "".join(map(chr, trained)) == target_text(task, rec) + "|"
    assert IMAGE_PAD not in trained
    assert enc["labels"][: enc["prompt_tokens"]] == [IGNORE_INDEX] * enc["prompt_tokens"]
