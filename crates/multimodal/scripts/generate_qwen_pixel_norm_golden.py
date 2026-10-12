#!/usr/bin/env python3
"""Reference pixel values of the Qwen vision families' image processors.

The reference image processors (the torchvision-backed ones the serving
engines run) are constructed locally from each family's preprocessor values,
so nothing is downloaded. A synthetic 448x448 gradient holding every byte
value in every channel goes through each processor twice: once with rescale
and normalization off, which gives the patch bytes, and once as configured,
which gives the normalized floats. From the two the per-byte table each
channel maps through is read off, and the normalized tensor is fingerprinted
as `f32` and as `bfloat16` (round to nearest even, what the engine casts it
to). The Rust integration test ``tests/qwen_pixel_norm_golden.rs`` checks its
own table and tensors against this JSON bit for bit.

448 is a multiple of both families' patch grids (28 and 32) and within both
pixel budgets, so no resize takes part.

Usage (from the repository root):
    python crates/multimodal/scripts/generate_qwen_pixel_norm_golden.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
import transformers
from PIL import Image

if int(transformers.__version__.split(".")[0]) >= 5:
    from transformers import Qwen2VLImageProcessor as ReferenceImageProcessor
else:
    from transformers import Qwen2VLImageProcessorFast as ReferenceImageProcessor

GOLDEN = "tests/fixtures/golden/qwen_pixel_norm.json"
WIDTH = HEIGHT = 448
MODELS = {
    "qwen2_vl": {
        "patch_size": 14,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1280 * 28 * 28,
    },
    "qwen3_vl": {
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "size": {"shortest_edge": 65536, "longest_edge": 16777216},
    },
}


def gradient_image() -> Image.Image:
    """Channel 0 is the row, channel 1 the column, channel 2 their sum, each modulo 256."""
    y, x = np.indices((HEIGHT, WIDTH), dtype=np.int32)
    pixels = np.stack([y % 256, x % 256, (x + y) % 256], axis=-1).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def fingerprint_bytes(data: bytes) -> str:
    value = 0xCBF29CE484222325
    for byte in data:
        value ^= byte
        value = value * 0x100000001B3 & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)


def byte_table(
    patch_bytes: np.ndarray, normalized: np.ndarray, channel_block: int
) -> list[list[str]]:
    table = []
    for channel in range(3):
        block = slice(channel * channel_block, (channel + 1) * channel_block)
        mapping: dict[int, np.float32] = {}
        for byte, value in zip(
            patch_bytes[:, block].ravel().tolist(), normalized[:, block].ravel().tolist()
        ):
            seen = mapping.setdefault(byte, value)
            if seen != value:
                raise SystemExit(f"channel {channel}: byte {byte} maps to {seen} and {value}")
        if len(mapping) != 256:
            raise SystemExit(f"channel {channel}: {len(mapping)} byte values seen, expected 256")
        table.append([f"{np.float32(mapping[byte]).view(np.uint32):08x}" for byte in range(256)])
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    crate = Path(__file__).resolve().parent.parent
    parser.add_argument("--out", type=Path, default=crate / GOLDEN)
    args = parser.parse_args()

    image = gradient_image()
    models = {}
    for name, values in MODELS.items():
        processor = ReferenceImageProcessor(**values)
        normalized = processor(images=image, return_tensors="np")
        raw = processor(images=image, do_rescale=False, do_normalize=False, return_tensors="np")
        pixel_values = np.ascontiguousarray(normalized["pixel_values"], dtype=np.float32)
        patch_bytes = raw["pixel_values"]
        patch_u8 = np.rint(patch_bytes).astype(np.uint8)
        if not np.array_equal(patch_u8.astype(np.float32), patch_bytes):
            raise SystemExit(f"{name}: un-normalized patches are not whole bytes")
        channel_block = values["temporal_patch_size"] * values["patch_size"] ** 2
        models[name] = {
            "preprocessor_config": values,
            "shape": list(pixel_values.shape),
            "image_grid_thw": normalized["image_grid_thw"][0].tolist(),
            "channel_block": channel_block,
            "table_bits": byte_table(patch_u8, pixel_values, channel_block),
            "fnv1a_patch_u8": fingerprint_bytes(patch_u8.tobytes()),
            "fnv1a_f32": fingerprint_bytes(pixel_values.tobytes()),
            "fnv1a_bf16": fingerprint_bytes(bf16_bits(pixel_values).tobytes()),
        }
        print(
            name, models[name]["shape"], models[name]["image_grid_thw"], models[name]["fnv1a_f32"]
        )

    golden = {
        "generator": "crates/multimodal/scripts/generate_qwen_pixel_norm_golden.py",
        "reference": {
            "processor": ReferenceImageProcessor.__name__,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
        },
        "image": {
            "width": WIDTH,
            "height": HEIGHT,
            "pixel": "channel 0 = row % 256, channel 1 = column % 256, channel 2 = (row + column) % 256",
        },
        "models": models,
    }
    args.out.write_text(json.dumps(golden, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
