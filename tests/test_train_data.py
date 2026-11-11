"""Needs torch; skipped where it is not installed (CI). Run locally with torch CPU."""

import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from vizwiz.train.data import Collator, SFTDataset  # noqa: E402


class FakeProcessor:
    """apply_chat_template like Qwen: ids = chars, 4 pad tokens per image, plus pixels."""

    def apply_chat_template(self, messages, tokenize, return_dict, add_generation_prompt=False,
                            return_tensors=None):  # fmt: skip
        ids = []
        for m in messages:
            ids += [ord(c) for c in f"<{m['role']}>"]
            parts = m["content"] if isinstance(m["content"], list) else [
                {"type": "text", "text": m["content"]}]  # fmt: skip
            for part in parts:
                ids += [9] * 4 if part["type"] == "image" else [ord(c) for c in part["text"]]
            ids.append(ord("|"))
        if add_generation_prompt:
            ids += [ord(c) for c in "<assistant>"]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                "pixel_values": torch.randn(4, 8),
                "image_grid_thw": torch.tensor([[1, 4, 4]]),
            }
        return {"input_ids": [ids]}


def test_dataset_masks_prompt_and_collator_pads(tmp_path):
    Image.new("RGB", (64, 64)).save(tmp_path / "a.jpg")
    Image.new("RGB", (64, 64)).save(tmp_path / "b.jpg")
    recs = [
        {"id": "1", "image": "a.jpg", "question": "q?", "target": "milk",
         "answerability": "answerable"},
        {"id": "2", "image": "b.jpg", "question": "longer question?", "target": "a red car",
         "answerability": "answerable"},
    ]  # fmt: skip
    paths = {"a.jpg": tmp_path / "a.jpg", "b.jpg": tmp_path / "b.jpg"}
    ds = SFTDataset("answer", recs, paths, FakeProcessor(), max_pixels=4096)
    item = ds[0]
    trained = [
        t for t, lbl in zip(item["input_ids"][0], item["labels"][0], strict=True) if lbl != -100
    ]
    assert "".join(chr(t) for t in trained) == "milk|"
    batch = Collator(pad_token_id=0)([ds[0], ds[1]])
    n = batch["input_ids"].shape[1]
    assert batch["labels"].shape == (2, n) and batch["attention_mask"].shape == (2, n)
    short = ds[0]["input_ids"].shape[1]
    assert batch["labels"][0, short:].eq(-100).all() and batch["input_ids"][0, short:].eq(0).all()
    assert batch["attention_mask"][0, short:].eq(0).all()
    assert batch["pixel_values"].shape == (8, 8) and batch["image_grid_thw"].shape == (2, 3)
