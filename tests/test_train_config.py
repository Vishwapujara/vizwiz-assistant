import pytest
from pydantic import ValidationError

from vizwiz.train.config import (
    last_checkpoint,
    load_train_config,
    parse_override,
    save_due,
    total_optimizer_steps,
    warmup_steps,
)
from vizwiz.train.sft import build_parser


def test_task_configs_load_with_plan_defaults():
    for task in ("answer", "answerability", "quality"):
        cfg = load_train_config(f"configs/train/{task}.yaml")
        assert cfg.task == task and cfg.mode == "lora"
        assert (cfg.lora_r, cfg.lora_alpha) == (16, 32)
        assert 16 <= cfg.effective_batch <= 32  # PROJECT_PLAN 8.1


def test_overrides_and_mode():
    cfg = load_train_config(
        "configs/train/answer.yaml", "qlora", ["learning_rate=2e-4", "limit=200", "max_steps=30"]
    )
    assert cfg.mode == "qlora" and cfg.learning_rate == 2e-4 and cfg.limit == 200
    assert parse_override("eval_limit=null") == ("eval_limit", None)
    with pytest.raises(ValueError):
        parse_override("no_equals_sign")
    with pytest.raises(ValidationError):  # typo in a key is an error
        load_train_config("configs/train/answer.yaml", None, ["learnin_rate=1"])


def test_step_arithmetic():
    cfg = load_train_config("configs/train/answer.yaml")  # effective batch 16
    assert total_optimizer_steps(13_134, cfg) == 821  # ceil(13134 / 16)
    cfg2 = load_train_config("configs/train/answer.yaml", None, ["epochs=2"])
    assert total_optimizer_steps(13_134, cfg2) == 1642
    cfg3 = load_train_config("configs/train/answer.yaml", None, ["max_steps=30"])
    assert total_optimizer_steps(13_134, cfg3) == 30
    assert warmup_steps(821, 0.03) == 25 and warmup_steps(10, 0.0) == 0


def test_last_checkpoint_ignores_incomplete(tmp_path):
    for step, complete in ((10, True), (20, True), (30, False)):
        d = tmp_path / f"checkpoint-{step}"
        d.mkdir()
        if complete:
            (d / "trainer_state.json").write_text("{}")
    assert last_checkpoint(tmp_path).name == "checkpoint-20"
    assert last_checkpoint(tmp_path / "missing") is None


def test_cli_parses():
    a = build_parser().parse_args(
        ["train", "--config", "c.yaml", "--records", "r", "--images", "i", "--out", "o",
         "--run-id", "x", "--set", "limit=5", "--crash-at-step", "20"]
    )  # fmt: skip
    assert a.cmd == "train" and a.set == ["limit=5"] and a.crash_at_step == 20
    b = build_parser().parse_args(
        ["profile", "--config", "c.yaml", "--records", "r", "--images", "i", "--report", "p"]
    )
    assert b.budgets == [262_144, 524_288, 1_048_576] and b.batch_sizes == [1, 2, 4]


def test_time_based_save_every_10_minutes_by_default():
    cfg = load_train_config("configs/train/answer.yaml")
    assert cfg.save_every_minutes == 10 and cfg.save_steps is None
    assert not save_due(now=599.0, last_save=0.0, minutes=10)
    assert save_due(now=600.0, last_save=0.0, minutes=10)
    assert not save_due(now=10_000.0, last_save=0.0, minutes=None)
