# VizWiz Visual Assistant

A visual assistant for blind users. Given a photo and a question, it:

1. checks whether the photo is usable, and if not, says how to retake it;
2. decides whether the question can be answered from the photo, and declines instead of guessing;
3. answers briefly when it can.

It is built from one small vision-language model (Qwen3-VL) with a separate LoRA adapter
for each task, fine-tuned on the human-labeled [VizWiz](https://vizwiz.org) datasets,
then quantized and served with vLLM multi-LoRA behind a FastAPI gateway.

> **Status:** early development. See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the plan
> and [`DECISIONS.md`](DECISIONS.md) for decisions and roadblocks.

## Privacy

VizWiz photos were taken by blind people who could not check what they captured, and
some contain private information. This repo never stores or displays dataset images.

## Development

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check . && pytest
```

GPU work (training, evaluation, serving) runs in Google Colab via the launchers in
`notebooks/`. Start every session with
[`00_setup_colab.ipynb`](https://colab.research.google.com/github/Vishwapujara/vizwiz-assistant/blob/main/notebooks/00_setup_colab.ipynb).
