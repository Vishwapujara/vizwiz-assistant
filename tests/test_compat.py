from pathlib import Path

from vizwiz.compat import (
    LORA_TARGETS,
    VISION_MARKERS,
    build_parser,
    build_vllm_command,
    looks_correct,
)


def test_answer_check():
    assert looks_correct("EXP 12/2026")
    assert looks_correct("December 2026")
    assert not looks_correct("I can't read it")


def test_lora_targets_are_language_model_names():
    # Qwen3-VL vision modules are named attn.qkv / attn.proj / linear_fc1 / linear_fc2.
    for name in LORA_TARGETS:
        assert name.endswith("_proj")
        assert name not in {"qkv", "proj", "linear_fc1", "linear_fc2"}
        assert not any(v in name for v in VISION_MARKERS)


def test_vllm_command_enables_lora():
    cmd = build_vllm_command("some/model", Path("/tmp/lora"), 8000)
    assert cmd[1:3] == ["serve", "some/model"]
    assert "--enable-lora" in cmd
    assert cmd[cmd.index("--lora-modules") + 1] == "blank=/tmp/lora"


def test_parser_subcommands():
    parser = build_parser()
    assert parser.parse_args(["versions"]).cmd == "versions"
    args = parser.parse_args(["hf", "--lora-out", "/tmp/x"])
    assert args.lora_out == Path("/tmp/x")
