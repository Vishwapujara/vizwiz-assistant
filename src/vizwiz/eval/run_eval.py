"""Batch inference + metrics (Step 2.2). Runs in Colab on a GPU, with vLLM.

    python -m vizwiz.eval.run_eval --records <dir> --images <dir> --split test \\
        --tasks answer answerability quality --limit 100 \\
        --out <Drive>/runs/<run_id> --run-id <run_id> [--adapter <dir>]

For each task: picks the records (``formatting.task_records``), prepares every image
with the shared preprocessing (D-035), builds the prompt with the model's own chat
template and the training prompts (D-036), and decodes greedily with vLLM.

Outputs
    <out>/predictions_<task>.jsonl   one line per example: id, model output, score.
                                     Model output about dataset questions, so it stays
                                     on Drive and never goes into git.
    <out>/metrics.json               aggregate metrics, timing, versions, settings.
                                     Summary only: this is the file that gets committed.

Answerability score (D-040): P(unanswerable) from the first generated token. The
probability mass on first tokens of "unanswerable" spellings is compared with the mass
on "answerable" spellings, read from the top-k logprobs of that one token. When
neither appears in the top-k, the generated label decides (1.0 / 0.0); those cases are
counted as ``score_fallbacks``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from vizwiz.data.config import REPO_ROOT, load_data_config
from vizwiz.data.formatting import TASKS, build_messages, read_jsonl, task_records
from vizwiz.eval.metrics import (
    UNANSWERABLE,
    answer_metrics,
    answerability_metrics,
    parse_answerability,
    quality_metrics,
)

DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
MAX_NEW_TOKENS = {"answer": 16, "answerability": 4, "quality": 48}
TOP_LOGPROBS = 20
CLASS_WORDS = {"unanswerable": "unanswerable", "answerable": "answerable"}


# ---------------------------------------------------------------- generator interface


class Generator(Protocol):
    def class_token_ids(self) -> dict[str, set[int]]: ...

    def generate(self, task: str, records: list[dict], images: list) -> list[dict]:
        """One dict per record: {"text", "first_logprobs" {token_id: logprob}, "tokens"}."""
        ...


def first_token_ids(encode, word: str) -> set[int]:
    """First token id of each casing of ``word`` (the reply starts a new turn, so no
    leading space)."""
    return {encode(v)[0] for v in {word, word.capitalize(), word.upper()}}


def class_ids_from(encode) -> dict[str, set[int]]:
    ids = {c: first_token_ids(encode, w) for c, w in CLASS_WORDS.items()}
    if ids["unanswerable"] & ids["answerable"]:
        raise ValueError(f"first tokens overlap, score would be meaningless: {ids}")
    return ids


def unanswerable_score(
    first_logprobs: dict[int, float], class_ids: dict[str, set[int]], text: str
) -> tuple[float, bool]:
    """(P(unanswerable), True if read from logprobs; False if fallback to the label)."""
    mass = {
        c: sum(math.exp(lp) for t, lp in first_logprobs.items() if t in ids)
        for c, ids in class_ids.items()
    }
    total = mass["unanswerable"] + mass["answerable"]
    if total > 0:
        return mass["unanswerable"] / total, True
    return (1.0 if parse_answerability(text) == UNANSWERABLE else 0.0), False


# ---------------------------------------------------------------- vLLM backend


class VLLMGenerator:
    def __init__(
        self,
        model: str,
        adapter: Path | None,
        max_model_len: int,
        gpu_memory_utilization: float,
        seed: int,
    ):
        import os

        from vizwiz.compat import server_env

        os.environ.update(server_env())  # R-002 / D-020, before vLLM is imported
        import torch
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self._sampling = SamplingParams
        major, _ = torch.cuda.get_device_capability()
        self.dtype = "bfloat16" if major >= 8 else "float16"  # T4 (7.5) has no bf16
        self.processor = AutoProcessor.from_pretrained(model)
        self.llm = LLM(
            model=model,
            dtype=self.dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1},
            enable_lora=adapter is not None,
            max_lora_rank=16,
            seed=seed,
        )
        self.lora = None
        if adapter is not None:
            from vllm.lora.request import LoRARequest

            self.lora = LoRARequest(adapter.name, 1, str(adapter))

    def class_token_ids(self) -> dict[str, set[int]]:
        tok = self.processor.tokenizer
        ids = class_ids_from(lambda w: tok.encode(w, add_special_tokens=False))
        shown = {c: sorted(tok.decode([i]) for i in v) for c, v in ids.items()}
        print(f"  answerability score uses first tokens {shown}")
        return ids

    def prompt(self, task: str, record: dict) -> str:
        messages = build_messages(task, record, None, with_target=False)
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, task: str, records: list[dict], images: list) -> list[dict]:
        params = self._sampling(
            temperature=0.0,
            max_tokens=MAX_NEW_TOKENS[task],
            logprobs=TOP_LOGPROBS if task == "answerability" else None,
        )
        inputs = [
            {"prompt": self.prompt(task, r), "multi_modal_data": {"image": img}}
            for r, img in zip(records, images, strict=True)
        ]
        outs = self.llm.generate(inputs, params, lora_request=self.lora, use_tqdm=False)
        results = []
        for o in outs:
            c = o.outputs[0]
            first = {}
            if c.logprobs:
                first = {tid: lp.logprob for tid, lp in c.logprobs[0].items()}
            results.append({"text": c.text, "first_logprobs": first, "tokens": len(c.token_ids)})
        return results


# ---------------------------------------------------------------- pipeline


def sample_records(records: list[dict], limit: int | None, seed: int) -> list[dict]:
    if limit is None or limit >= len(records):
        return list(records)
    return sorted(random.Random(seed).sample(records, limit), key=lambda r: r["id"])


def to_example(task: str, record: dict, pred: dict) -> dict:
    """Join ground truth and prediction into what the metric functions expect."""
    if task == "answer":
        keys = ("answers", "answer_type", "quality")
    elif task == "answerability":
        keys = ("answerability", "unanswerable_votes")
    else:
        keys = ("labels",)
    return {
        "prediction": pred["text"],
        **({"score": pred["score"]} if "score" in pred else {}),
        **{k: record[k] for k in keys},
    }


def compute_metrics(task: str, examples: list[dict], issue_labels: Sequence[str]) -> dict:
    if task == "answer":
        return answer_metrics(examples)
    if task == "answerability":
        return answerability_metrics(examples)
    return quality_metrics(examples, issue_labels)


def run_task(
    gen: Generator,
    task: str,
    records: list[dict],
    image_path: dict[str, Path],
    max_pixels: int | None,
    batch_size: int,
    out_file: Path,
) -> tuple[list[dict], dict]:
    from vizwiz.serve.preprocess import prepare_image

    class_ids = gen.class_token_ids() if task == "answerability" else None
    examples, fallbacks, gen_tokens, gen_seconds = [], 0, 0, 0.0
    t_start = time.perf_counter()
    with ThreadPoolExecutor(8) as pool, out_file.open("w") as f:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            images = list(
                pool.map(lambda r: prepare_image(image_path[r["image"]], max_pixels), batch)
            )
            t0 = time.perf_counter()
            preds = gen.generate(task, batch, images)
            gen_seconds += time.perf_counter() - t0
            for r, p in zip(batch, preds, strict=True):
                if class_ids is not None:
                    p["score"], from_lp = unanswerable_score(
                        p["first_logprobs"], class_ids, p["text"]
                    )
                    fallbacks += not from_lp
                gen_tokens += p["tokens"]
                row = {"id": r["id"], "prediction": p["text"], "tokens": p["tokens"]}
                if "score" in p:
                    row["score"] = round(p["score"], 6)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                examples.append(to_example(task, r, p))
            print(f"  [{task}] {len(examples)}/{len(records)}", flush=True)
    wall = time.perf_counter() - t_start
    timing = {
        "examples": len(records),
        "wall_s": round(wall, 1),
        "generate_s": round(gen_seconds, 1),
        "examples_per_s": round(len(records) / gen_seconds, 2) if gen_seconds else None,
        "generated_tokens_mean": round(gen_tokens / len(records), 2) if records else None,
    }
    if class_ids is not None:
        timing["score_fallbacks"] = fallbacks
    return examples, timing


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def evaluate(gen: Generator, args, settings: dict) -> dict:
    from vizwiz.data.images import find_images

    cfg = load_data_config()
    issue_labels = [lbl for lbl in cfg.quality_labels if lbl != "none"]
    vqa = read_jsonl(args.records / f"vqa_{args.split}.jsonl")
    quality = (
        read_jsonl(args.records / f"quality_{args.split}.jsonl")
        if (args.records / f"quality_{args.split}.jsonl").exists()
        else []
    )
    image_path = find_images(args.images)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {**settings, "tasks": {}}
    for task in args.tasks:
        records = sample_records(task_records(task, vqa, quality), args.limit, args.seed)
        if not records:
            print(f"[{task}] no records in split {args.split}; skipped")
            continue
        print(f"[{task}] {len(records)} examples")
        examples, timing = run_task(
            gen,
            task,
            records,
            image_path,
            args.max_pixels or None,
            args.batch_size,
            args.out / f"predictions_{task}.jsonl",
        )
        report["tasks"][task] = {
            "metrics": compute_metrics(task, examples, issue_labels),
            "timing": timing,
        }
        headline = {
            "answer": "vqa_accuracy",
            "answerability": "average_precision",
            "quality": "macro_f1",
        }[task]
        print(f"[{task}] {headline} = {report['tasks'][task]['metrics'][headline]}  {timing}")
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.out / 'metrics.json'}")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch inference and metrics")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    p.add_argument("--limit", type=int, help="examples per task (default: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-pixels", type=int, default=524_288, help="0 = native size")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", type=Path, help="LoRA adapter directory (default: base)")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out", type=Path, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from vizwiz.compat import package_versions

    t0 = time.perf_counter()
    gen = VLLMGenerator(
        args.model, args.adapter, args.max_model_len, args.gpu_memory_utilization, args.seed
    )
    load_s = round(time.perf_counter() - t0, 1)

    import torch

    settings = {
        "run_id": args.run_id,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "model": args.model,
        "adapter": args.adapter.name if args.adapter else None,
        "split": args.split,
        "limit": args.limit,
        "seed": args.seed,
        "max_pixels": args.max_pixels or None,
        "decoding": {"temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS},
        "dtype": gen.dtype,
        "gpu": torch.cuda.get_device_name(0),
        "engine_load_s": load_s,
        "versions": package_versions(),
    }
    evaluate(gen, args, settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
