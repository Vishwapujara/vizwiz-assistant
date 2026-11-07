"""Task formatting: records → chat messages → token ids with loss masking (Step 1.5).

Three tasks (PROJECT_PLAN Section 6), all using the model's own chat template:

    answer          image + question → short answer       (answerable questions only)
    answerability   image + question → answerable | unanswerable
    quality         image only       → JSON list of quality labels

Loss masking: the prompt (system text, image tokens, question, and the template's
"assistant turn starts" tokens) gets label -100, so only the answer tokens, including
the end-of-turn token that teaches the model to stop, contribute to the loss. The
prompt is tokenised on its own with ``add_generation_prompt=True``, and its ids must
be an exact prefix of the full conversation's ids. That is checked for every example,
because a template that inserts extra tokens (e.g. empty "thinking" blocks) would
silently train on the wrong positions.

    python -m vizwiz.data.formatting stats --records <dir> --images <dir> --split val ...
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable
from pathlib import Path
from statistics import mean, quantiles

from vizwiz.data.config import load_data_config

IGNORE_INDEX = -100
TASKS = ("answer", "answerability", "quality")
_QUALITY_LABELS = load_data_config().quality_labels

SYSTEM_PROMPTS = {
    "answer": "Answer the question about the photo in a few words.",
    "answerability": (
        "Decide whether the question can be answered from the photo. "
        "Reply with exactly one word: answerable or unanswerable."
    ),
    "quality": (
        "List the image quality problems in this photo as a JSON array, using only these "
        f"labels: {', '.join(lbl for lbl in _QUALITY_LABELS if lbl != 'none')}. "
        'If there are no problems, reply ["none"].'
    ),
}


def task_records(task: str, vqa: list[dict], quality: list[dict]) -> list[dict]:
    """Which records each task trains and evaluates on."""
    if task == "answer":
        return [r for r in vqa if r["answerability"] == "answerable"]
    if task == "answerability":
        return list(vqa)
    if task == "quality":
        return list(quality)
    raise ValueError(f"unknown task {task!r}")


def target_text(task: str, record: dict) -> str:
    if task == "answer":
        return record["target"]
    if task == "answerability":
        return record["answerability"]
    return json.dumps(record["labels"])


def build_messages(task: str, record: dict, image, with_target: bool) -> list[dict]:
    user: list[dict] = [{"type": "image", "image": image}]
    if task != "quality":
        user.append({"type": "text", "text": record["question"]})
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[task]},
        {"role": "user", "content": user},
    ]
    if with_target:
        messages.append({"role": "assistant", "content": target_text(task, record)})
    return messages


def _ids(processor, messages, add_generation_prompt: bool) -> list[int]:
    out = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
    )
    ids = out["input_ids"]
    if ids and isinstance(ids[0], list):  # batched output
        ids = ids[0]
    return list(ids)


def encode(processor, task: str, record: dict, image) -> dict:
    """Token ids and masked labels for one training example."""
    prompt_ids = _ids(processor, build_messages(task, record, image, False), True)
    full_ids = _ids(processor, build_messages(task, record, image, True), False)
    prefix_ok = full_ids[: len(prompt_ids)] == prompt_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {
        "input_ids": full_ids,
        "labels": labels,
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(full_ids) - len(prompt_ids),
        "prefix_ok": prefix_ok,
    }


# ---------------------------------------------------------------- stats (Colab, CPU)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def summarise(values: Iterable[int]) -> dict:
    v = sorted(values)
    if not v:
        return {}
    q = quantiles(v, n=100, method="inclusive") if len(v) > 1 else [v[0]] * 99
    return {
        "n": len(v),
        "mean": round(mean(v), 1),
        "p50": round(q[49], 1),
        "p95": round(q[94], 1),
        "max": v[-1],
    }


def demo_record() -> dict:
    """Synthetic record for printing a formatted example (no dataset text in logs)."""
    return {
        "question": "What does this label say?",
        "target": "whole milk",
        "answerability": "answerable",
        "labels": ["blur", "framing"],
    }


def run_stats(args) -> int:
    from transformers import AutoProcessor

    from vizwiz.compat import make_test_image
    from vizwiz.data.images import find_images
    from vizwiz.serve.preprocess import prepare_image, visual_tokens

    processor = AutoProcessor.from_pretrained(args.model)
    image_pad = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vqa = read_jsonl(args.records / f"vqa_{args.split}.jsonl")
    quality = read_jsonl(args.records / f"quality_{args.split}.jsonl")
    rng = random.Random(args.seed)
    image_path = find_images(args.images)  # zips unpack into per-split subfolders

    report: dict = {"model": args.model, "split": args.split, "sample": args.sample, "tasks": {}}

    # 1. One synthetic example per task: show the template and which tokens are trained.
    demo_img = make_test_image()
    for task in TASKS:
        enc = encode(processor, task, demo_record(), prepare_image(demo_img, 262_144))
        text = processor.tokenizer.decode(enc["input_ids"])
        trained = processor.tokenizer.decode(
            [t for t, lbl in zip(enc["input_ids"], enc["labels"], strict=True) if lbl != -100]
        )
        collapsed = text.replace("<|image_pad|>", "")
        print(f"\n=== demo: {task} (synthetic example) ===")
        print(collapsed)
        print(f"--- trained tokens: {trained!r}   prefix_ok={enc['prefix_ok']}")

    # 2. Token lengths on a sample of real examples, at each image-size budget.
    for task in TASKS:
        recs = task_records(task, vqa, quality)
        sample = rng.sample(recs, min(args.sample, len(recs)))
        per_budget = {}
        prefix_failures = 0
        token_formula_mismatches = 0
        for budget in args.max_pixels:
            total, prompt, target, vis = [], [], [], []
            for r in sample:
                img = prepare_image(image_path[r["image"]], budget or None)
                enc = encode(processor, task, r, img)
                prefix_failures += not enc["prefix_ok"]
                total.append(len(enc["input_ids"]))
                prompt.append(enc["prompt_tokens"])
                target.append(enc["target_tokens"])
                vis.append(sum(t == image_pad for t in enc["input_ids"]))
                # Differs only if the processor resizes again (e.g. its own pixel cap).
                token_formula_mismatches += vis[-1] != visual_tokens(img.width, img.height)
            per_budget[str(budget or "native")] = {
                "total_tokens": summarise(total),
                "visual_tokens": summarise(vis),
                "target_tokens": summarise(target),
            }
            print(
                f"[{task}] max_pixels={budget or 'native'}: {per_budget[str(budget or 'native')]}"
            )
        report["tasks"][task] = {
            "examples_in_split": len(recs),
            "by_max_pixels": per_budget,
            "prefix_failures": prefix_failures,
            "visual_token_formula_mismatches": token_formula_mismatches,
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved {args.report}")
    ok = all(t["prefix_failures"] == 0 for t in report["tasks"].values())
    print("ALL PREFIX CHECKS PASSED" if ok else "PREFIX CHECK FAILURES")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Task formatting and token statistics")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--records", type=Path, required=True)
    s.add_argument("--images", type=Path, required=True)
    s.add_argument("--split", default="val")
    s.add_argument("--sample", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument(
        "--max-pixels",
        type=int,
        nargs="+",
        default=[262_144, 524_288, 1_048_576, 0],
        help="image pixel budgets to measure; 0 = native size",
    )
    s.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    s.add_argument("--report", type=Path, required=True)
    args = p.parse_args(argv)
    return run_stats(args)


if __name__ == "__main__":
    raise SystemExit(main())
