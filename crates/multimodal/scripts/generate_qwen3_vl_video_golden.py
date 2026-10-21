#!/usr/bin/env python3
"""Reference shapes for the Qwen3-VL family's video path.

The reference video processor is constructed locally from the family's
preprocessor values, so nothing is downloaded. The clip is decoded with OpenCV,
sampled the way the media connector samples it (2 frames per second, at least 4
and at most 768, spread evenly from the first frame to the last), and run
through the processor with frame sampling off. The Rust integration test
``tests/qwen3_vl_video_golden.rs`` compares its own output's shapes, grid and
token count against the JSON this writes; the frame pixels themselves are not
part of the golden.

The clip fixture was made with
``ffmpeg -f lavfi -i testsrc=size=128x96:rate=8 -t 1 -c:v libx264 -preset veryslow
-crf 30 -pix_fmt yuv420p -an -movflags +faststart``.

Usage (from the repository root):
    python crates/multimodal/scripts/generate_qwen3_vl_video_golden.py
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import transformers
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

CLIP = "tests/fixtures/videos/testsrc_128x96_8fps_1s.mp4"
GOLDEN = "tests/fixtures/golden/qwen3_vl_video.json"
PREPROCESSOR_CONFIG = {
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "size": {"shortest_edge": 4096, "longest_edge": 25165824},
}
SAMPLE_FPS = 2.0
MIN_FRAMES = 4
MAX_FRAMES = 768


def decode(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise SystemExit(f"no frames decoded from {path}")
    return frames, fps


def sampled_indices(total: int, fps: float) -> list[int]:
    """The frames the media connector samples (`even_frame_indices` /
    `spread_evenly` in `src/media.rs`), step for step: the count is the clip's
    duration times the sample rate, rounded half away from zero as `f64::round`
    does (Python's `round` rounds half to even), `MAX_FRAMES` when the rate is
    unknown, clamped to `[MIN_FRAMES, MAX_FRAMES]` and NOT capped at the clip's
    frame count (a short clip repeats frames); the indices are
    `floor(i * (total - 1) / (count - 1))`."""
    if fps > 0:
        count = int(math.floor(total / fps * SAMPLE_FPS + 0.5))
    else:
        count = MAX_FRAMES
    count = min(max(count, MIN_FRAMES), MAX_FRAMES)
    if count <= 1:
        return [0]
    return [int(math.floor(i * (total - 1) / (count - 1))) for i in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    crate = Path(__file__).resolve().parent.parent
    parser.add_argument("--clip", type=Path, default=crate / CLIP)
    parser.add_argument("--out", type=Path, default=crate / GOLDEN)
    args = parser.parse_args()

    frames, fps = decode(args.clip)
    indices = sampled_indices(len(frames), fps)
    sampled = np.stack([frames[index] for index in indices])  # [T, H, W, C] uint8
    height, width = sampled.shape[1:3]

    processor = Qwen3VLVideoProcessor(**PREPROCESSOR_CONFIG)
    output = processor(videos=[sampled], do_sample_frames=False, return_tensors="np")
    pixel_values = output["pixel_values_videos"]
    grid = output["video_grid_thw"]
    merge = PREPROCESSOR_CONFIG["merge_size"]
    tokens = int(np.prod(grid[0])) // (merge * merge)

    golden = {
        "generator": "crates/multimodal/scripts/generate_qwen3_vl_video_golden.py",
        "reference": {
            "processor": type(processor).__name__,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "clip": args.clip.resolve().relative_to(crate).as_posix(),
        "clip_frames": len(frames),
        "clip_fps": fps,
        "sampled_indices": indices,
        "sampled_frames": len(indices),
        "frame": {"width": int(width), "height": int(height)},
        "preprocessor_config": PREPROCESSOR_CONFIG,
        "outputs": {
            "pixel_values_videos": {
                "shape": list(pixel_values.shape),
                "dtype": str(pixel_values.dtype),
            },
            "video_grid_thw": {"shape": list(grid.shape), "values": grid.tolist()},
        },
        "tokens_per_video": tokens,
    }
    args.out.write_text(json.dumps(golden, indent=2) + "\n")
    print(json.dumps(golden, indent=2))


if __name__ == "__main__":
    main()
