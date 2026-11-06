"""Image preprocessing shared by training, evaluation, and serving.

One function so the model sees identically prepared images everywhere (D-035):

1. Apply the EXIF orientation. Phone photos often store "rotate 90°" as metadata
   rather than rotating the pixels; PIL ignores it unless asked, and model processors
   may or may not apply it. We always do, explicitly.
2. Convert to RGB.
3. Downscale (never upscale) so width × height ≤ ``max_pixels``, keeping the aspect
   ratio, and round both sides to a multiple of ``factor``.

Why a multiple of 32: Qwen3-VL cuts the image into 16×16 patches and merges 2×2
patches into one visual token, so each 32×32 block costs exactly one token. A side
that is not a multiple of 32 would be resized again by the processor; rounding here
keeps the token count predictable: tokens = (width / 32) × (height / 32).
"""

from __future__ import annotations

import io
import math
from pathlib import Path

from PIL import Image, ImageOps

FACTOR = 32  # Qwen3-VL: patch 16 × spatial merge 2
MIN_SIDE = FACTOR


def load_image(source: str | Path | bytes) -> Image.Image:
    img = Image.open(io.BytesIO(source)) if isinstance(source, bytes) else Image.open(source)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def target_size(width: int, height: int, max_pixels: int | None, factor: int = FACTOR):
    """(new_width, new_height): multiples of ``factor``, area ≤ max_pixels, aspect kept."""
    scale = 1.0
    if max_pixels is not None and width * height > max_pixels:
        scale = math.sqrt(max_pixels / (width * height))
    w = max(MIN_SIDE, math.floor(width * scale / factor) * factor)
    h = max(MIN_SIDE, math.floor(height * scale / factor) * factor)
    return w, h


def prepare_image(source, max_pixels: int | None, factor: int = FACTOR) -> Image.Image:
    img = load_image(source) if not isinstance(source, Image.Image) else source.convert("RGB")
    size = target_size(img.width, img.height, max_pixels, factor)
    if size != (img.width, img.height):
        img = img.resize(size, Image.Resampling.BICUBIC)
    return img


def visual_tokens(width: int, height: int, factor: int = FACTOR) -> int:
    """Visual tokens for an image already sized to multiples of ``factor``."""
    return (width // factor) * (height // factor)
