"""Step 0.3 compatibility check: Qwen3-VL in transformers and vLLM multi-LoRA on one GPU.

Run inside the pinned GPU environment (see notebooks/00b_compat_check.ipynb):

    python -m vizwiz.compat versions
    python -m vizwiz.compat hf   --lora-out /content/blank_lora --report hf.json
    python -m vizwiz.compat vllm --lora /content/blank_lora --report vllm.json --log vllm.log

``hf`` downloads the model, answers one question about a synthetic test image, and saves
a *blank* LoRA adapter (lora_B initialised to zero, so it does not change outputs).
``vllm`` starts a vLLM server with that adapter, sends the same request to the base model
and to the adapter, and checks both answers are identical. Identical answers prove vLLM
loaded the adapter for this architecture without corrupting the model.

No VizWiz images are used here: the test image is drawn in code.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from importlib import metadata
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
QUESTION = "What is the expiration date on this label? Answer in a few words."
# LoRA targets: language-model projections only. In Qwen3-VL the vision tower uses
# different names (attn.qkv, attn.proj, linear_fc1, linear_fc2), so these names never
# match it. hf_check() verifies that at runtime.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
VISION_MARKERS = ("visual", "vision")
PACKAGES = ["torch", "torchvision", "transformers", "vllm", "peft", "accelerate", "pillow"]


# ---------------------------------------------------------------- helpers


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def make_test_image():
    """A fake milk label. Drawn in code so no dataset image is ever involved."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (800, 500), (30, 90, 160))
    draw = ImageDraw.Draw(img)
    draw.rectangle((80, 80, 720, 420), fill=(250, 250, 245))
    big = ImageFont.load_default(size=72)
    small = ImageFont.load_default(size=56)
    draw.text((140, 140), "WHOLE MILK", fill=(20, 20, 20), font=big)
    draw.text((140, 290), "EXP 12/2026", fill=(180, 20, 20), font=small)
    return img


def image_data_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def looks_correct(answer: str) -> bool:
    a = answer.lower()
    return "12/2026" in a or ("dec" in a and "2026" in a)


def write_report(path: Path | None, data: dict) -> None:
    text = json.dumps(data, indent=2)
    print(text)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"Saved report to {path}")


# ---------------------------------------------------------------- hf


def hf_check(model_id: str, lora_out: Path, report: Path | None) -> int:
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    result: dict = {"model": model_id, "versions": package_versions()}

    t0 = time.perf_counter()
    snapshot_download(model_id)
    result["download_s"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
    )
    model.eval()
    torch.cuda.synchronize()
    result["load_s"] = round(time.perf_counter() - t0, 1)
    result["model_class"] = type(model).__name__

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": make_test_image()},
                {"type": "text", "text": QUESTION},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    result["prompt_tokens"] = int(inputs["input_ids"].shape[1])

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    torch.cuda.synchronize()
    result["generate_s"] = round(time.perf_counter() - t0, 2)
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    answer = processor.decode(new_tokens, skip_special_tokens=True).strip()
    result["answer"] = answer
    result["answer_looks_correct"] = looks_correct(answer)
    result["peak_gpu_mem_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)

    # Blank LoRA adapter: lora_B starts at zero, so base and adapter outputs must match.
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=LORA_TARGETS)
    peft_model = get_peft_model(model, config)
    targeted = [n for n, m in peft_model.named_modules() if hasattr(m, "lora_A")]
    vision_hits = [n for n in targeted if any(v in n for v in VISION_MARKERS)]
    result["lora_modules_targeted"] = len(targeted)
    result["lora_vision_modules_targeted"] = len(vision_hits)
    result["lora_example_module"] = targeted[0] if targeted else None
    trainable, total = peft_model.get_nb_trainable_parameters()
    result["lora_trainable_params"] = trainable
    result["lora_trainable_pct"] = round(100 * trainable / total, 3)
    peft_model.save_pretrained(lora_out)
    result["lora_saved_to"] = str(lora_out)

    ok = bool(answer) and not vision_hits and len(targeted) > 0
    result["ok"] = ok
    write_report(report, result)
    return 0 if ok else 1


# ---------------------------------------------------------------- vllm


def _post_json(url: str, payload: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def _wait_healthy(base: str, proc: subprocess.Popen, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"{base}/health", timeout=5)
            return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(5)
    return False


def build_vllm_command(model_id: str, lora: Path, port: int) -> list[str]:
    vllm_bin = Path(sys.executable).parent / "vllm"
    return [
        str(vllm_bin),
        "serve",
        model_id,
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.85",
        "--limit-mm-per-prompt",
        json.dumps({"image": 1, "video": 0}),
        "--enable-lora",
        "--max-loras",
        "3",
        "--max-lora-rank",
        "16",
        "--lora-modules",
        f"blank={lora}",
    ]


def vllm_check(model_id: str, lora: Path, report: Path | None, log: Path, port: int) -> int:
    base = f"http://127.0.0.1:{port}"
    cmd = build_vllm_command(model_id, lora, port)
    result: dict = {"model": model_id, "versions": package_versions(), "command": cmd}
    log.parent.mkdir(parents=True, exist_ok=True)

    with log.open("w") as log_file:
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        try:
            healthy = _wait_healthy(base, proc, timeout_s=900)
            result["startup_s"] = round(time.perf_counter() - t0, 1)
            result["server_started"] = healthy
            if not healthy:
                result["ok"] = False
                result["hint"] = f"Server did not become healthy; read the log: {log}"
                write_report(report, result)
                return 1

            result["served_models"] = [m["id"] for m in _get_json(f"{base}/v1/models")["data"]]
            url = image_data_url(make_test_image())
            answers = {}
            for name in (model_id, "blank"):
                payload = {
                    "model": name,
                    "temperature": 0,
                    "max_tokens": 32,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": url}},
                                {"type": "text", "text": QUESTION},
                            ],
                        }
                    ],
                }
                t1 = time.perf_counter()
                resp = _post_json(f"{base}/v1/chat/completions", payload)
                latency = round(time.perf_counter() - t1, 2)
                text = resp["choices"][0]["message"]["content"].strip()
                key = "base" if name == model_id else "lora_blank"
                answers[key] = {"answer": text, "latency_s": latency, "usage": resp.get("usage")}
            result["answers"] = answers
            result["base_answer_looks_correct"] = looks_correct(answers["base"]["answer"])
            result["lora_matches_base"] = (
                answers["base"]["answer"] == answers["lora_blank"]["answer"]
            )
            result["ok"] = "blank" in result["served_models"] and result["lora_matches_base"]
        except Exception as exc:  # report any failure instead of a bare traceback
            result["ok"] = False
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()

    result["log"] = str(log)
    write_report(report, result)
    return 0 if result["ok"] else 1


# ---------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 0.3 compatibility check")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("versions", help="Print installed package versions")

    hf = sub.add_parser("hf", help="transformers inference + blank LoRA adapter")
    hf.add_argument("--model", default=DEFAULT_MODEL)
    hf.add_argument("--lora-out", type=Path, required=True)
    hf.add_argument("--report", type=Path)

    vl = sub.add_parser("vllm", help="vLLM server with LoRA; base vs adapter check")
    vl.add_argument("--model", default=DEFAULT_MODEL)
    vl.add_argument("--lora", type=Path, required=True)
    vl.add_argument("--report", type=Path)
    vl.add_argument("--log", type=Path, default=Path("vllm_server.log"))
    vl.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "versions":
        print(json.dumps(package_versions(), indent=2))
        return 0
    if args.cmd == "hf":
        return hf_check(args.model, args.lora_out, args.report)
    return vllm_check(args.model, args.lora, args.report, args.log, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
