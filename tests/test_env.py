import json

from vizwiz.env import CACHE_DIRS, GpuInfo, classify_gpu, main

T4 = GpuInfo("Tesla T4", 15.0, (7, 5))
L4 = GpuInfo("NVIDIA L4", 22.0, (8, 9))


def test_t4_capabilities():
    assert not T4.bf16
    assert not T4.flash_attention_2
    assert not T4.fp8


def test_l4_capabilities():
    assert L4.bf16
    assert L4.flash_attention_2
    assert L4.fp8


def test_t4_is_smoke_test_only():
    role, notes = classify_gpu(T4)
    assert role == "smoke tests only"
    assert any("bf16" in n for n in notes)


def test_l4_is_main_runtime():
    role, notes = classify_gpu(L4)
    assert role.startswith("main")
    assert notes == []


def test_cpu_role():
    role, _ = classify_gpu(None)
    assert role.startswith("cpu")


def test_main_creates_dirs_and_report(tmp_path, capsys):
    assert main(["--drive-root", str(tmp_path)]) == 0
    for name in CACHE_DIRS:
        assert (tmp_path / name).is_dir()
    reports = list((tmp_path / "logs").glob("env_*.json"))
    assert len(reports) == 1
    data = json.loads(reports[0].read_text())
    assert {"python", "gpu", "role", "disk_free_gib"} <= data.keys()
