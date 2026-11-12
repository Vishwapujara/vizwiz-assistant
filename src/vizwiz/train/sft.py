"""Supervised fine-tuning entrypoint (PROJECT_PLAN 8.3). Runs in Colab on a GPU.

    python -m vizwiz.train.sft train --config configs/train/answer.yaml --mode lora \\
        --records <Drive>/datasets/processed --images /content/data/images \\
        --out <Drive>/checkpoints --run-id <id> [--set key=value ...] [--wandb]

    python -m vizwiz.train.sft profile --config configs/train/answer.yaml \\
        --records ... --images ... --budgets 262144 524288 1048576 --batch-sizes 1 2 4 \\
        --report <file>

``train``
    LoRA on the language model only (vision tower frozen), bf16 on GPUs that support
    it. ``--mode qlora`` loads the base in 4-bit NF4 (bitsandbytes), vision tower and
    lm_head kept in 16-bit. Checkpoints every ``save_every_minutes`` (default 10) to
    ``<out>/<run-id>/``; a rerun with the same run id resumes from the newest complete checkpoint
    automatically. The final adapter (adapter weights only) goes to
    ``<out>/<run-id>/adapter``; ``train_summary.json`` records steps, timing, peak
    memory, and whether the run resumed. ``--crash-at-step N`` stops the process right
    after the checkpoint at step N, to test resuming.

``profile``
    Loads the model once and times a few optimizer steps for every image budget ×
    per-device batch size, recording peak GPU memory (OOM is recorded, not fatal).
    Input for fixing the image resolution (Step 3.1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from vizwiz.data.config import REPO_ROOT
from vizwiz.data.formatting import read_jsonl
from vizwiz.train.config import (
    TrainConfig,
    last_checkpoint,
    load_train_config,
    save_due,
    total_optimizer_steps,
    warmup_steps,
)
from vizwiz.train.data import Collator, SFTDataset, select_records

WANDB_PROJECT = "vizwiz-assistant"
CRASH_EXIT_CODE = 3


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_supports_bf16() -> bool:
    import torch

    return torch.cuda.get_device_capability()[0] >= 8


def load_model(cfg: TrainConfig):
    """(model with LoRA attached, processor, dtype name)."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from vizwiz.compat import LORA_TARGETS, VISION_MARKERS

    dtype = torch.bfloat16 if gpu_supports_bf16() else torch.float16
    kwargs: dict = {"dtype": dtype, "attn_implementation": "sdpa", "device_map": {"": 0}}
    if cfg.mode == "qlora":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["visual", "lm_head"],
        )
    model = AutoModelForImageTextToText.from_pretrained(cfg.model, **kwargs)
    model.config.use_cache = False
    if cfg.mode == "qlora":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    lora_names = [n for n, _ in model.named_modules() if n.endswith("lora_A")]
    in_vision = [n for n in lora_names if any(m in n for m in VISION_MARKERS)]
    if not lora_names or in_vision:
        raise RuntimeError(f"LoRA placement wrong: {len(lora_names)} modules, vision {in_vision}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA modules: {len(lora_names)} (none in vision); trainable {trainable:,} "
          f"of {total:,} ({100 * trainable / total:.2f}%)")  # fmt: skip
    processor = AutoProcessor.from_pretrained(cfg.model)
    return model, processor, str(dtype).removeprefix("torch.")


def load_split(records_dir: Path, split: str) -> tuple[list[dict], list[dict], dict]:
    files = {t: records_dir / f"{t}_{split}.jsonl" for t in ("vqa", "quality")}
    hashes = {p.name: sha256_file(p) for p in files.values()}
    return read_jsonl(files["vqa"]), read_jsonl(files["quality"]), hashes


def check_record_hashes(hashes: dict[str, str]) -> None:
    """The record files must be the committed ones (results/splits/files.json)."""
    expected = json.loads((REPO_ROOT / "results/splits/files.json").read_text())
    for name, digest in hashes.items():
        if expected[name]["sha256"] != digest:
            raise RuntimeError(f"{name} differs from results/splits/files.json")


# ---------------------------------------------------------------- train


def make_callbacks(crash_at_step: int | None, save_every_minutes: float | None):
    import torch
    from transformers import TrainerCallback

    class Stats(TrainerCallback):
        def __init__(self):
            self.step_times: list[float] = []
            self._t = None

        def on_step_begin(self, args, state, control, **kw):
            self._t = time.perf_counter()

        def on_step_end(self, args, state, control, **kw):
            if self._t is not None:
                self.step_times.append(time.perf_counter() - self._t)

        def on_log(self, args, state, control, logs=None, **kw):
            if logs is not None:
                logs["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    class CrashAt(TrainerCallback):
        def on_save(self, args, state, control, **kw):
            if crash_at_step is not None and state.global_step >= crash_at_step:
                print(f"--crash-at-step: stopping after the checkpoint at step "
                      f"{state.global_step} (simulated disconnect)", flush=True)  # fmt: skip
                os._exit(CRASH_EXIT_CODE)  # no cleanup, like a lost Colab runtime

    class TimedSave(TrainerCallback):
        """Ask the Trainer for a checkpoint once ``save_every_minutes`` have passed, so
        a lost Colab runtime costs at most that much training, at any speed."""

        def __init__(self):
            self.last = time.monotonic()

        def on_step_end(self, args, state, control, **kw):
            if save_due(time.monotonic(), self.last, save_every_minutes):
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kw):
            self.last = time.monotonic()

    return Stats(), TimedSave(), CrashAt()


def run_train(args) -> int:
    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    from vizwiz.compat import package_versions
    from vizwiz.data.images import find_images
    from vizwiz.eval.run_eval import git_commit

    cfg = load_train_config(args.config, args.mode, args.set)
    set_seed(cfg.seed)
    run_dir = args.out / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    resume = last_checkpoint(run_dir)
    print(f"run {args.run_id}: {cfg.task} / {cfg.mode}; "
          f"{'resuming from ' + resume.name if resume else 'starting fresh'}")  # fmt: skip

    train_vqa, train_q, train_hashes = load_split(args.records, "train")
    val_vqa, val_q, val_hashes = load_split(args.records, "val")
    check_record_hashes({**train_hashes, **val_hashes})
    train_recs = select_records(cfg.task, train_vqa, train_q, cfg.limit, cfg.seed)
    val_recs = select_records(cfg.task, val_vqa, val_q, cfg.eval_limit, cfg.seed)
    image_path = find_images(args.images)

    model, processor, dtype = load_model(cfg)
    train_ds = SFTDataset(cfg.task, train_recs, image_path, processor, cfg.max_pixels)
    val_ds = SFTDataset(cfg.task, val_recs, image_path, processor, cfg.max_pixels)
    steps = total_optimizer_steps(len(train_recs), cfg)
    bf16 = dtype == "bfloat16"

    settings = {
        "run_id": args.run_id,
        "git_commit": git_commit(),
        "config": cfg.model_dump(),
        "effective_batch": cfg.effective_batch,
        "train_examples": len(train_recs),
        "val_examples": len(val_recs),
        "optimizer_steps": steps,
        "record_sha256": {**train_hashes, **val_hashes},
        "dtype": dtype,
        "gpu": torch.cuda.get_device_name(0),
        "versions": package_versions(),
    }
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    report_to = "none"
    if args.wandb:
        import wandb

        os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
        wandb.init(
            project=os.environ["WANDB_PROJECT"],
            name=args.run_id,
            id=args.run_id.replace("/", "-"),  # same id on resume: the curve continues
            resume="allow",
            group=args.wandb_group,
            job_type="train",
            config=settings,
        )
        report_to = "wandb"

    targs = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=cfg.per_device_batch,
        per_device_eval_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps(steps, cfg.warmup_ratio),
        max_steps=steps,
        bf16=bf16,
        fp16=not bf16,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps or 10**9,  # time-based saving is the default
        save_total_limit=cfg.save_total_limit,
        report_to=report_to,
        run_name=args.run_id,
        seed=cfg.seed,
        data_seed=cfg.seed,
        dataloader_num_workers=cfg.num_workers,
        remove_unused_columns=False,
        label_names=["labels"],
        gradient_checkpointing=False,  # already enabled on the model in load_model
        optim="adamw_torch",
    )
    stats, timed_save, crash = make_callbacks(args.crash_at_step, cfg.save_every_minutes)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(processor.tokenizer.pad_token_id),
        callbacks=[stats, timed_save, crash],  # crash last: it runs after the save
    )
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    wall = time.perf_counter() - t0
    final_eval = trainer.evaluate()
    trainer.save_model(str(run_dir / "adapter"))  # PEFT model: adapter weights only

    timed = stats.step_times[2:] or stats.step_times  # skip warm-up steps
    summary = {
        "run_id": args.run_id,
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "resumed_from": resume.name if resume else None,
        "global_step": trainer.state.global_step,
        "optimizer_steps_planned": steps,
        "steps_this_session": len(stats.step_times),
        "train_loss": round(result.training_loss, 4),
        "final_val_loss": round(final_eval["eval_loss"], 4),
        "wall_s": round(wall, 1),
        "optimizer_step_s_median": round(statistics.median(timed), 3) if timed else None,
        "examples_per_s": round(cfg.effective_batch / statistics.median(timed), 2)
        if timed
        else None,
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "adapter_dir": str(run_dir / "adapter"),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.wandb:
        import wandb

        wandb.summary.update(summary)
        wandb.finish()
    return 0


# ---------------------------------------------------------------- profile


def run_profile(args) -> int:
    import torch
    from torch.utils.data import DataLoader

    from vizwiz.data.images import find_images

    cfg = load_train_config(args.config, args.mode, args.set)
    vqa, quality, hashes = load_split(args.records, "train")
    check_record_hashes(hashes)
    image_path = find_images(args.images)
    model, processor, dtype = load_model(cfg)
    collate = Collator(processor.tokenizer.pad_token_id)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    model.train()
    rows = []
    for budget in args.budgets:
        for bs in args.batch_sizes:
            n = bs * (args.steps + 2)
            recs = select_records(cfg.task, vqa, quality, n, cfg.seed)
            ds = SFTDataset(cfg.task, recs, image_path, processor, budget or None)
            dl = DataLoader(ds, batch_size=bs, collate_fn=collate, num_workers=cfg.num_workers)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            times, tokens = [], []
            row = {"max_pixels": budget or "native", "per_device_batch": bs}
            try:
                for batch in dl:
                    batch = {k: v.to("cuda") for k, v in batch.items()}
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    loss = model(**batch).loss
                    loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
                    tokens.append(int(batch["attention_mask"].sum()))
                timed = times[2:]
                step = statistics.median(timed)
                row.update(
                    status="ok",
                    step_s=round(step, 3),
                    examples_per_s=round(bs / step, 2),
                    tokens_per_example=round(statistics.mean(tokens) / bs),
                    peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                )
            except torch.cuda.OutOfMemoryError:
                row.update(status="oom")
                batch = loss = None  # drop references so the memory can be freed
                opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(row, flush=True)
            rows.append(row)
    report = {
        "task": cfg.task,
        "mode": cfg.mode,
        "dtype": dtype,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "steps_timed_per_setting": args.steps,
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.report}")
    return 0


# ---------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LoRA / QLoRA supervised fine-tuning")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("train", "profile"):
        s = sub.add_parser(name)
        s.add_argument("--config", type=Path, required=True)
        s.add_argument("--mode", choices=["lora", "qlora"])
        s.add_argument("--records", type=Path, required=True)
        s.add_argument("--images", type=Path, required=True)
        s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    t = sub.choices["train"]
    t.add_argument("--out", type=Path, required=True, help="checkpoint root (on Drive)")
    t.add_argument("--run-id", required=True)
    t.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    t.add_argument("--wandb-group")
    t.add_argument("--crash-at-step", type=int, help="test only: exit after this checkpoint")
    pr = sub.choices["profile"]
    pr.add_argument("--budgets", type=int, nargs="+", default=[262_144, 524_288, 1_048_576])
    pr.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    pr.add_argument("--steps", type=int, default=8, help="timed steps per setting")
    pr.add_argument("--report", type=Path, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_train(args) if args.cmd == "train" else run_profile(args)


if __name__ == "__main__":
    raise SystemExit(main())
