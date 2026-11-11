"""Training examples and batching (Step 3.1).

Each example goes through the same pieces as evaluation and serving: the shared
image preprocessing (D-035) and the task prompts and chat template (D-036). The
processor is called twice per example: once on the full conversation (token ids,
pixel values, image grid) and once on the prompt alone, to find where the answer
starts. Prompt tokens get label -100; the prefix check from Step 1.5 runs on every
example and a mismatch is an error, so training can never silently learn from the
wrong positions.
"""

from __future__ import annotations

import random
from pathlib import Path

from vizwiz.data.formatting import IGNORE_INDEX, build_messages, task_records


def select_records(
    task: str, vqa: list[dict], quality: list[dict], limit: int | None, seed: int
) -> list[dict]:
    recs = task_records(task, vqa, quality)
    if limit is not None and limit < len(recs):
        recs = random.Random(seed).sample(recs, limit)
    return recs


class SFTDataset:
    """Map-style dataset (works with torch DataLoader without subclassing)."""

    def __init__(
        self,
        task: str,
        records: list[dict],
        image_path: dict[str, Path],
        processor,
        max_pixels: int | None,
    ):
        self.task = task
        self.records = records
        self.image_path = image_path
        self.processor = processor
        self.max_pixels = max_pixels

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        from vizwiz.serve.preprocess import prepare_image

        r = self.records[i]
        img = prepare_image(self.image_path[r["image"]], self.max_pixels)
        full = self.processor.apply_chat_template(
            build_messages(self.task, r, img, with_target=True),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt = self.processor.apply_chat_template(
            build_messages(self.task, r, img, with_target=False),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        ids = full["input_ids"][0]
        prompt_ids = prompt["input_ids"]
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        n = len(prompt_ids)
        if ids[:n].tolist() != list(prompt_ids) or n >= len(ids):
            raise ValueError(f"prompt is not a prefix of the conversation for {r['id']}")
        labels = ids.clone()
        labels[:n] = IGNORE_INDEX
        item = {k: v for k, v in full.items()}
        item["labels"] = labels.unsqueeze(0)
        return item


class Collator:
    """Right-pads every per-token tensor (same shape as ``input_ids``) and concatenates
    the image tensors (``pixel_values`` holds patches of all images in the batch,
    ``image_grid_thw`` one row per image)."""

    PAD_VALUES = {"labels": IGNORE_INDEX, "attention_mask": 0}

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        lengths = [f["input_ids"].shape[-1] for f in features]
        longest = max(lengths)
        batch = {}
        for key in features[0]:
            values = [f[key] for f in features]
            per_token = all(
                v.dim() == 2 and v.shape == (1, n) for v, n in zip(values, lengths, strict=True)
            )
            if per_token:
                pad = self.pad_token_id if key == "input_ids" else self.PAD_VALUES.get(key, 0)
                rows = [
                    torch.cat([v[0], v.new_full((longest - v.shape[-1],), pad)]) for v in values
                ]
                batch[key] = torch.stack(rows)
            else:
                batch[key] = torch.cat(values, dim=0)
        return batch
