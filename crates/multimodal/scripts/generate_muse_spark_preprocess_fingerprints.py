#!/usr/bin/env python3
"""Record Muse Spark image and video preprocessing fingerprints from the reference.

The reference is transformers' ``MuseGlimmerImageProcessor`` and
``MuseGlimmerVideoProcessor`` (the torchvision backend) with the released
checkpoints' processor config values, and ``MuseSparkProcessor``'s prompt
rewriting. This script runs them unmodified over a fixed set of images and
frame clips and writes, per case, the resized size, the grid, the token count,
the output shape, FNV-1a fingerprints of the exact bytes the engine receives
(``pixel_values`` as float32 little-endian and the same tensor cast to
bfloat16, as little-endian u16 bits) and, for clips, the sampled frame
indices the reference picks, the prompt stamps and the expansion string.

Images are either synthetic (a seeded RGB pattern the Rust test regenerates
identically) or PNG files under the fixtures directory (PNG so that PIL and
the Rust ``image`` crate decode identical pixels). Clips are synthetic frame
sequences with a stated source rate, sampled by the reference's own
``sample_frames``.

Usage (inside an environment with transformers >= 5.20 and torch):
    python crates/multimodal/scripts/generate_muse_spark_preprocess_fingerprints.py \
        [--output <fixtures>/golden/muse_spark_preprocess_fingerprints.json]
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image
from transformers.models.muse_glimmer import (
    image_processing_muse_glimmer,
    video_processing_muse_glimmer,
)
from transformers.models.muse_glimmer.image_processing_muse_glimmer import MuseGlimmerImageProcessor
from transformers.models.muse_glimmer.video_processing_muse_glimmer import MuseGlimmerVideoProcessor
from transformers.video_utils import VideoMetadata

REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGES_DIR = REPO_ROOT / "crates/multimodal/tests/fixtures/images"
DEFAULT_OUTPUT = (
    REPO_ROOT / "crates/multimodal/tests/fixtures/golden/muse_spark_preprocess_fingerprints.json"
)

# The released checkpoints' processor_config.json sections.
IMAGE_CONFIG = {
    "do_convert_rgb": True,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "max_image_tokens": 4096,
    "merge_size": 2,
    "patch_size": 14,
    "resample": 1,
    "rescale_factor": 0.00392156862745098,
    "temporal_patch_size": 2,
}
VIDEO_CONFIG = {
    "do_convert_rgb": True,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "do_sample_frames": True,
    "fps": 2.0,
    "frame_timestamp_format": "Time: {timestamp:.2f}s",
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "max_video_frame_tokens": 256,
    "merge_size": 2,
    "num_frames": 600,
    "patch_size": 14,
    "resample": 1,
    "rescale_factor": 0.00392156862745098,
    "temporal_patch_size": 2,
}

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
    # Over the image cap: the grid shrinks to the budget.
    ("synthetic_3000x2000", 3000, 2000, 11),
    # Sizes that exercise the rounding, the ties and the one-cell edge rule.
    ("synthetic_41x56", 41, 56, 12),
    ("synthetic_28x351", 28, 351, 13),
    ("synthetic_1x1", 1, 1, 14),
    ("synthetic_2x3", 2, 3, 15),
    ("synthetic_47x47", 47, 47, 16),
    ("synthetic_3000x10", 3000, 10, 17),
    ("synthetic_137x130", 137, 130, 18),
    ("synthetic_463x469", 463, 469, 19),
    ("synthetic_698x688", 698, 688, 20),
]

# PNG fixtures already in the repository.
FILE_CASES = [
    ("carrots", "deepseek_v41_carrots.png"),
    ("corn", "deepseek_v41_corn.png"),
]

# Synthetic clips: (name, width, height, source frames, source fps, seed base).
# The reference's sample_frames picks the frames; both sides pick them the same.
VIDEO_CASES = [
    ("clip_320x240_8fps_1s", 320, 240, 8, 8.0, 30),
    ("clip_640x360_24fps_5frames", 640, 360, 5, 24.0, 40),
    ("clip_1280x720_30fps_45frames", 1280, 720, 45, 30.0, 50),
    ("clip_96x96_1frame", 96, 96, 1, 25.0, 60),
    ("clip_200x300_12fps_31frames", 200, 300, 31, 12.0, 70),
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


def seeded_pixels(width: int, height: int, seed: int) -> np.ndarray:
    """The Rust tests' seeded pattern: R=(x*7+y*3)%256, G=(x*5+y*11)%256,
    B=(x+y*2)%256, each plus the seed with u8 wraparound."""
    y, x = np.mgrid[0:height, 0:width].astype(np.uint32)
    r = (x * 7 + y * 3) % 256
    g = (x * 5 + y * 11) % 256
    b = (x + y * 2) % 256
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint32)
    return ((rgb + seed) % 256).astype(np.uint8)


def seeded_image(width: int, height: int, seed: int) -> Image.Image:
    return Image.fromarray(seeded_pixels(width, height, seed))


def fingerprints(pixel_values: torch.Tensor) -> dict:
    assert pixel_values.dtype == torch.float32, pixel_values.dtype
    f32 = pixel_values.contiguous().numpy().astype("<f4").tobytes()
    bf16 = (
        pixel_values.to(torch.bfloat16)
        .view(torch.int16)
        .contiguous()
        .numpy()
        .astype("<i2")
        .tobytes()
    )
    return {"fnv1a_f32": fnv1a(f32), "fnv1a_bf16": fnv1a(bf16)}


def image_record(
    name: str, source: dict, image: Image.Image, processor: MuseGlimmerImageProcessor
) -> dict:
    out = processor(images=[image], return_tensors="pt")
    pixel_values = out["pixel_values"]
    grid = out["image_grid_thw"][0].tolist()
    tokens = int(np.prod(grid)) // processor.merge_size**2
    return {
        "name": name,
        "source": source,
        "width": image.width,
        "height": image.height,
        "resized_hw": [grid[1] * processor.patch_size, grid[2] * processor.patch_size],
        "grid_thw": grid,
        "tokens": tokens,
        "shape": list(pixel_values.shape),
        "expansion": "<|image_start|>" + "<|patch|>" * tokens + "<|image_end|>",
        **fingerprints(pixel_values),
    }


def video_record(
    name: str,
    width: int,
    height: int,
    frames: int,
    fps: float,
    seed: int,
    processor: MuseGlimmerVideoProcessor,
) -> dict:
    # As a tensor: the reference takes a one-frame array for an image batch.
    clip = torch.from_numpy(
        np.stack([seeded_pixels(width, height, seed + index) for index in range(frames)])
    )
    metadata = VideoMetadata(total_num_frames=frames, fps=fps)
    out = processor(
        videos=[clip], video_metadata=[metadata], return_tensors="pt", return_metadata=True
    )
    pixel_values = out["pixel_values_videos"]
    grid = out["video_grid_thw"][0].tolist()
    sampled = out["video_metadata"][0]
    indices = [int(i) for i in sampled.frames_indices]
    groups = grid[0]
    per_group = grid[1] * grid[2] // processor.merge_size**2
    timestamps = list(sampled.timestamps[:: processor.temporal_patch_size])[:groups]
    while len(timestamps) < groups:
        timestamps.append(timestamps[-1] if timestamps else 0.0)
    expansion = "<|vid_start|>"
    for g, ts in enumerate(timestamps):
        expansion += processor.frame_timestamp_format.format(timestamp=ts) + "<|video|>" * per_group
        expansion += "<|vid_frame_separator|>" if g < groups - 1 else "<|vid_end|>"
    return {
        "name": name,
        "source": {"seed": seed, "frames": frames, "fps": fps},
        "width": width,
        "height": height,
        "frame_indices": indices,
        "resized_hw": [grid[1] * processor.patch_size, grid[2] * processor.patch_size],
        "grid_thw": grid,
        "tokens": groups * per_group,
        "shape": list(pixel_values.shape),
        "stamps": [processor.frame_timestamp_format.format(timestamp=ts) for ts in timestamps],
        "expansion": expansion,
        **fingerprints(pixel_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)

    image_processor = MuseGlimmerImageProcessor(**IMAGE_CONFIG)
    video_processor = MuseGlimmerVideoProcessor(**VIDEO_CONFIG)
    cases = []
    for name, width, height, seed in SYNTHETIC_CASES:
        cases.append(
            image_record(name, {"seed": seed}, seeded_image(width, height, seed), image_processor)
        )
    for name, file in FILE_CASES:
        image = Image.open(IMAGES_DIR / file)
        image.load()
        cases.append(image_record(name, {"file": file}, image.convert("RGB"), image_processor))
    # One batch of two images: rows concatenate, grids stack.
    batch = [seeded_image(200, 150, 2), seeded_image(96, 128, 1)]
    out = image_processor(images=batch, return_tensors="pt")
    cases.append(
        {
            "name": "batch_200x150_96x128",
            "source": {"seeds": [2, 1], "sizes": [[200, 150], [96, 128]]},
            "width": 0,
            "height": 0,
            "resized_hw": [0, 0],
            "grid_thw": out["image_grid_thw"].tolist(),
            "tokens": [int(np.prod(g)) // 4 for g in out["image_grid_thw"].tolist()],
            "shape": list(out["pixel_values"].shape),
            "expansion": "",
            **fingerprints(out["pixel_values"]),
        }
    )
    video_cases = [video_record(*case, video_processor) for case in VIDEO_CASES]

    reference_sources = {
        "image_processing_muse_glimmer.py": hashlib.sha256(
            inspect.getsource(image_processing_muse_glimmer).encode()
        ).hexdigest(),
        "video_processing_muse_glimmer.py": hashlib.sha256(
            inspect.getsource(video_processing_muse_glimmer).encode()
        ).hexdigest(),
    }
    document = {
        "reference": (
            f"transformers {transformers.__version__} MuseGlimmerImageProcessor / "
            f"MuseGlimmerVideoProcessor (torchvision backend), torch {torch.__version__}"
        ),
        "reference_sha256": reference_sources,
        "image_config": IMAGE_CONFIG,
        "video_config": VIDEO_CONFIG,
        "cases": cases,
        "video_cases": video_cases,
    }
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(cases)} image cases and {len(video_cases)} video cases to {args.output}")


if __name__ == "__main__":
    main()
