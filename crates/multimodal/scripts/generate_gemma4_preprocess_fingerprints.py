#!/usr/bin/env python3
"""Record Gemma 4 image-preprocessing fingerprints from the reference.

The reference is transformers' ``Gemma4ImageProcessor`` (the torchvision
backend, the processor vLLM runs; it also handles video frames there, at the
video processor's soft-token budget). This script runs it unmodified over a
fixed set of images and writes, per image and budget, the resized size, the
soft-token count, the output shapes and FNV-1a fingerprints of the exact
bytes the engine receives:

* ``fnv1a_f32`` — ``pixel_values`` as float32 little-endian, padding included;
* ``fnv1a_bf16`` — the same tensor cast to bfloat16, as little-endian u16 bits;
* ``fnv1a_positions_i64`` — ``image_position_ids`` as int64 little-endian.

Images are either synthetic (a seeded RGB pattern the Rust test regenerates
identically) or PNG files under the fixtures directory (PNG so that PIL and
the Rust ``image`` crate decode identical pixels). A multi-frame case pins the
stacking of video frames.

Usage (inside an environment with the engine's transformers and torch):
    python crates/multimodal/scripts/generate_gemma4_preprocess_fingerprints.py \
        [--output <fixtures>/golden/gemma4_preprocess_fingerprints.json]
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image
from transformers.models.gemma4 import image_processing_gemma4
from transformers.models.gemma4.image_processing_gemma4 import (
    Gemma4ImageProcessor,
    get_aspect_ratio_preserving_size,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGES_DIR = REPO_ROOT / "crates/multimodal/tests/fixtures/images"
DEFAULT_OUTPUT = (
    REPO_ROOT / "crates/multimodal/tests/fixtures/golden/gemma4_preprocess_fingerprints.json"
)

# The processor constants of the released checkpoints (also the class defaults).
PATCH_SIZE = 16
POOLING_KERNEL_SIZE = 3
IMAGE_SOFT_TOKENS = 280
VIDEO_SOFT_TOKENS = 70

# (name, width, height, seed): the Rust test rebuilds these with the same formula.
SYNTHETIC_CASES = [
    ("synthetic_64x64", 64, 64, 0),
    ("synthetic_96x128", 96, 128, 1),
    ("synthetic_200x150", 200, 150, 2),
    ("synthetic_320x240", 320, 240, 3),
    ("synthetic_512x512", 512, 512, 4),
    ("synthetic_640x480", 640, 480, 5),
    ("synthetic_1024x1024", 1024, 1024, 6),
    ("synthetic_1200x300", 1200, 300, 7),
    ("synthetic_300x1200", 300, 1200, 8),
    ("synthetic_1280x720", 1280, 720, 9),
    ("synthetic_1920x1080", 1920, 1080, 10),
    ("synthetic_3000x2000", 3000, 2000, 11),
    # Sizes that exercise the rounding and the one-block edge rule.
    ("synthetic_41x56", 41, 56, 12),
    ("synthetic_28x351", 28, 351, 13),
    ("synthetic_1x1", 1, 1, 14),
    ("synthetic_2x3", 2, 3, 15),
    ("synthetic_47x47", 47, 47, 16),
    ("synthetic_3000x10", 3000, 10, 17),
]

# PNG fixtures already in the repository.
FILE_CASES = [
    ("carrots", "deepseek_v41_carrots.png"),
    ("corn", "deepseek_v41_corn.png"),
]

# A clip of frames at the video budget: (name, width, height, seeds).
VIDEO_CASES = [
    ("frames_320x240", 320, 240, [20, 21, 22]),
    ("frames_640x360", 640, 360, [23, 24]),
]

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
FNV_MASK = (1 << 64) - 1


def fnv1a(data: bytes) -> str:
    h = FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & FNV_MASK
    return f"{h:016x}"


def seeded_image(width: int, height: int, seed: int) -> Image.Image:
    """The Rust tests' seeded pattern: R=(x*7+y*3)%256, G=(x*5+y*11)%256,
    B=(x+y*2)%256, each plus the seed with u8 wraparound."""
    y, x = np.mgrid[0:height, 0:width].astype(np.uint32)
    r = (x * 7 + y * 3) % 256
    g = (x * 5 + y * 11) % 256
    b = (x + y * 2) % 256
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint32)
    rgb = ((rgb + seed) % 256).astype(np.uint8)
    return Image.fromarray(rgb)


def processor(max_soft_tokens: int) -> Gemma4ImageProcessor:
    return Gemma4ImageProcessor(
        patch_size=PATCH_SIZE,
        pooling_kernel_size=POOLING_KERNEL_SIZE,
        max_soft_tokens=max_soft_tokens,
    )


def record(name: str, source: dict, images: list[Image.Image], max_soft_tokens: int) -> dict:
    out = processor(max_soft_tokens)(images=images, return_tensors="pt")
    pixel_values = out["pixel_values"]
    positions = out["image_position_ids"]
    assert pixel_values.dtype == torch.float32, pixel_values.dtype
    assert positions.dtype == torch.int64, positions.dtype
    max_patches = max_soft_tokens * POOLING_KERNEL_SIZE**2
    assert tuple(pixel_values.shape) == (len(images), max_patches, 3 * PATCH_SIZE**2)
    targets = []
    tokens = []
    for image in images:
        target_h, target_w = get_aspect_ratio_preserving_size(
            height=image.height,
            width=image.width,
            patch_size=PATCH_SIZE,
            max_patches=max_patches,
            pooling_kernel_size=POOLING_KERNEL_SIZE,
        )
        targets.append([target_h, target_w])
        tokens.append((target_h // PATCH_SIZE) * (target_w // PATCH_SIZE) // POOLING_KERNEL_SIZE**2)
    assert list(out["num_soft_tokens_per_image"]) == tokens, (
        name,
        out["num_soft_tokens_per_image"],
        tokens,
    )
    bf16 = pixel_values.to(torch.bfloat16).contiguous().view(torch.int16).numpy().astype("<i2")
    return {
        "name": name,
        "source": source,
        "width": images[0].width,
        "height": images[0].height,
        "max_soft_tokens": max_soft_tokens,
        "target_hw": targets if len(images) > 1 else targets[0],
        "num_soft_tokens": tokens if len(images) > 1 else tokens[0],
        "shape": list(pixel_values.shape),
        "positions_shape": list(positions.shape),
        "fnv1a_f32": fnv1a(pixel_values.contiguous().numpy().astype("<f4").tobytes()),
        "fnv1a_bf16": fnv1a(bf16.tobytes()),
        "fnv1a_positions_i64": fnv1a(positions.contiguous().numpy().astype("<i8").tobytes()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    opts = parser.parse_args()

    reference = Path(inspect.getsourcefile(image_processing_gemma4))
    cases = []
    for name, width, height, seed in SYNTHETIC_CASES:
        image = seeded_image(width, height, seed)
        for budget in (IMAGE_SOFT_TOKENS, VIDEO_SOFT_TOKENS):
            case = record(f"{name}_{budget}", {"seed": seed}, [image], budget)
            cases.append(case)
            print(f"{case['name']}: {case['num_soft_tokens']} tokens, target {case['target_hw']}")
    for name, file_name in FILE_CASES:
        path = IMAGES_DIR / file_name
        with Image.open(path) as source:
            image = source.convert("RGB")
        for budget in (IMAGE_SOFT_TOKENS, VIDEO_SOFT_TOKENS):
            case = record(f"{name}_{budget}", {"file": file_name}, [image], budget)
            cases.append(case)
            print(f"{case['name']}: {case['num_soft_tokens']} tokens, target {case['target_hw']}")
    video_cases = []
    for name, width, height, seeds in VIDEO_CASES:
        frames = [seeded_image(width, height, seed) for seed in seeds]
        case = record(name, {"seeds": seeds}, frames, VIDEO_SOFT_TOKENS)
        video_cases.append(case)
        print(f"{case['name']}: {case['num_soft_tokens']} tokens per frame, shape {case['shape']}")

    document = {
        "generator": Path(__file__).name,
        "reference": "transformers/models/gemma4/image_processing_gemma4.py (torchvision backend)",
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "processor": {
            "patch_size": PATCH_SIZE,
            "pooling_kernel_size": POOLING_KERNEL_SIZE,
            "image_max_soft_tokens": IMAGE_SOFT_TOKENS,
            "video_max_soft_tokens": VIDEO_SOFT_TOKENS,
        },
        "cases": cases,
        "video_cases": video_cases,
    }
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    opts.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"wrote {opts.output} ({len(cases)} image cases, {len(video_cases)} clip cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
