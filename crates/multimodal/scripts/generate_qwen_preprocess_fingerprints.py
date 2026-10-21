#!/usr/bin/env python3
"""Generate compact reference fingerprints for Qwen preprocessing.

The reference processors are the torchvision-backed image and video processors
the serving engines run, constructed locally, so this script does not download
model weights or configuration. Its output is checked into the Rust integration
test as an external correctness oracle for resize, normalization, and patchify.
The resize oracle is torch's antialiased `uint8` bicubic reached through
torchvision; the document records the torch and torchvision versions it was
generated with, since that kernel's dispatch has changed across releases.
"""

import json

import numpy as np
import torch
import torchvision
from PIL import Image
from PIL import __version__ as pillow_version
from transformers import __version__ as transformers_version
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

if int(transformers_version.split(".")[0]) >= 5:
    from transformers import Qwen2VLImageProcessor
else:
    from transformers import Qwen2VLImageProcessorFast as Qwen2VLImageProcessor

CASES = ((37, 23), (259, 194))


def make_image(width: int, height: int, seed: int = 0) -> Image.Image:
    y, x = np.indices((height, width), dtype=np.uint32)
    pixels = np.stack(
        (
            (seed + x * 7 + y * 3) % 256,
            (seed + x * 5 + y * 11) % 256,
            (seed + x + y * 2) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def fingerprint_bytes(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    value = 0xCBF29CE484222325
    for byte in contiguous.tobytes():
        value ^= byte
        value = value * 0x100000001B3 & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def processor_cases(name: str, processor: Qwen2VLImageProcessor) -> list[dict]:
    results = []
    for width, height in CASES:
        output = processor(
            images=make_image(width, height),
            do_normalize=False,
            return_tensors="np",
        )
        values = output["pixel_values"]
        patch_u8 = np.rint(values * 255.0).astype(np.uint8)
        results.append(
            {
                "model": name,
                "width": width,
                "height": height,
                "shape": list(values.shape),
                "grid_thw": output["image_grid_thw"][0].tolist(),
                "fnv1a_patch_u8": fingerprint_bytes(patch_u8),
            }
        )
    return results


def qwen3_video_case(processor: Qwen3VLVideoProcessor) -> dict:
    width, height = 37, 35
    seeds = (3, 101, 177)
    # The reference video processor resizes the frames itself (its torchvision
    # kernel), then pads temporally and patchifies.
    frames = [make_image(width, height, seed) for seed in seeds]
    output = processor(
        videos=[frames],
        do_normalize=False,
        do_sample_frames=False,
        return_tensors="pt",
    )
    values = output["pixel_values_videos"].cpu().numpy()
    patch_u8 = np.rint(values * 255.0).astype(np.uint8)
    return {
        "model": "qwen3_vl",
        "width": width,
        "height": height,
        "frame_count": len(seeds),
        "shape": list(values.shape),
        "grid_thw": output["video_grid_thw"][0].tolist(),
        "fnv1a_patch_u8": fingerprint_bytes(patch_u8),
    }


def main() -> None:
    qwen2 = Qwen2VLImageProcessor(
        patch_size=14,
        merge_size=2,
        temporal_patch_size=2,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        resample=Image.Resampling.BICUBIC,
    )
    qwen3 = Qwen2VLImageProcessor(
        patch_size=16,
        merge_size=2,
        temporal_patch_size=2,
        min_pixels=65536,
        max_pixels=16777216,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        resample=Image.Resampling.BICUBIC,
    )
    qwen3_video = Qwen3VLVideoProcessor(
        patch_size=16,
        merge_size=2,
        temporal_patch_size=2,
        size={"shortest_edge": 4096, "longest_edge": 25165824},
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        resample=Image.Resampling.BICUBIC,
        do_sample_frames=False,
    )
    document = {
        "generator": "generate_qwen_preprocess_fingerprints.py",
        "transformers": transformers_version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pillow": pillow_version,
        "cases": processor_cases("qwen2_vl", qwen2) + processor_cases("qwen3_vl", qwen3),
        "video_cases": [qwen3_video_case(qwen3_video)],
    }
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
