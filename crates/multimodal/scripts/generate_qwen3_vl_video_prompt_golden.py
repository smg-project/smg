#!/usr/bin/env python3
"""Reference prompt layout for the Qwen3-VL family's videos.

For a set of clip shapes (frame count and frame rate) the family's reference
video processor picks the frames (``sample_frames`` at its 2 fps rate, 4 to 768
frames, never more than the clip has, the last pick repeated up to a whole
temporal group) and the reference processor expands the bare ``<|video_pad|>``
into one stamped, marker-wrapped block per temporal group
(``replace_video_token``). Nothing is downloaded: both processors are
built locally from the family's preprocessor values, with a tokenizer that
knows only the marker tokens. The Rust test
``tests/qwen3_vl_video_prompt_golden.rs`` checks the media connector's frame
picks and the spec's prompt replacement against the JSON this writes.

Usage (from the repository root):
    python crates/multimodal/scripts/generate_qwen3_vl_video_prompt_golden.py
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import transformers
from tokenizers import Tokenizer, models
from transformers import Qwen2TokenizerFast
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
from transformers.video_utils import VideoMetadata

GOLDEN = "tests/fixtures/golden/qwen3_vl_video_prompt.json"
PREPROCESSOR_CONFIG = {
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}
VIDEO_SIZE = {"shortest_edge": 4096, "longest_edge": 25165824}
MARKERS = ["<|vision_start|>", "<|vision_end|>", "<|video_pad|>", "<|image_pad|>"]

# (name, frame count, frame rate, spatial grid (h, w) in patches): the clip
# shapes of the family's parity set plus the edges of the sampling rule (a
# clip shorter than the minimum, an odd pick, a fractional rate).
CASES = [
    ("1s_24fps", 24, 24.0, (4, 4)),
    ("1s_24fps_small_grid", 24, 24.0, (2, 2)),
    ("5s_24fps", 120, 24.0, (2, 2)),
    ("10s_30fps", 300, 30.0, (2, 2)),
    ("30s_24fps", 720, 24.0, (2, 2)),
    ("600s_24fps", 14400, 24.0, (2, 2)),
    ("1_frame_30fps", 1, 30.0, (2, 2)),
    ("3_frames_10fps", 3, 10.0, (2, 2)),
    ("25_frames_10fps", 25, 10.0, (2, 2)),
    ("9_frames_2_5fps", 9, 2.5, (2, 2)),
    ("1000_frames_29_97fps", 1000, 29.97, (2, 2)),
    ("16_frames_8fps", 16, 8.0, (2, 2)),
]


def marker_tokenizer() -> Qwen2TokenizerFast:
    """A tokenizer that knows the family's markers and nothing else."""
    vocab = {"<unk>": 0, **{marker: index + 1 for index, marker in enumerate(MARKERS)}}
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="<unk>")),
        unk_token="<unk>",
    )
    tokenizer.add_special_tokens({"additional_special_tokens": MARKERS})
    return tokenizer


def build_processor() -> Qwen3VLProcessor:
    image_processor = Qwen2VLImageProcessor(**PREPROCESSOR_CONFIG)
    video_processor = Qwen3VLVideoProcessor(size=VIDEO_SIZE, **PREPROCESSOR_CONFIG)
    return Qwen3VLProcessor(
        image_processor=image_processor,
        video_processor=video_processor,
        tokenizer=marker_tokenizer(),
    )


def case(
    processor: Qwen3VLProcessor, name: str, total: int, fps: float, grid_hw: tuple[int, int]
) -> dict:
    video_processor = processor.video_processor
    metadata = VideoMetadata(
        total_num_frames=total,
        fps=fps,
        duration=total / fps,
        frames_indices=list(range(total)),
        video_backend="opencv",
    )
    indices = [int(index) for index in video_processor.sample_frames(metadata=metadata)]
    metadata.frames_indices = indices
    temporal_patch_size = video_processor.temporal_patch_size
    grid_t = math.ceil(len(indices) / temporal_patch_size)
    grid_h, grid_w = grid_hw
    placeholder = processor.replace_video_token(
        {"video_grid_thw": np.array([[grid_t, grid_h, grid_w]]), "video_metadata": [metadata]},
        video_idx=0,
    )
    timestamps = processor._calculate_timestamps(indices, fps, temporal_patch_size)
    return {
        "name": name,
        "total_frames": total,
        "fps": fps,
        "frame_indices": indices,
        "grid_thw": [grid_t, grid_h, grid_w],
        "frame_seqlen": grid_h * grid_w // video_processor.merge_size**2,
        "timestamps": [f"{seconds:.1f}" for seconds in timestamps],
        "placeholder": placeholder,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default=GOLDEN, help="where to write the JSON")
    args = parser.parse_args()
    processor = build_processor()
    video_processor = processor.video_processor
    golden = {
        "transformers": transformers.__version__,
        "sampling": {
            "fps": video_processor.fps,
            "min_frames": video_processor.min_frames,
            "max_frames": video_processor.max_frames,
            "temporal_patch_size": video_processor.temporal_patch_size,
        },
        "markers": {
            "vision_start": processor.vision_start_token,
            "vision_end": processor.vision_end_token,
            "video_pad": processor.video_token,
        },
        "cases": [case(processor, *shape) for shape in CASES],
    }
    out = Path(args.golden)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(golden, indent=1) + "\n")
    for entry in golden["cases"]:
        print(
            f"{entry['name']}: {len(entry['frame_indices'])} frames {entry['frame_indices'][:6]}..., "
            f"grid_t {entry['grid_thw'][0]}, stamps {entry['timestamps'][:3]}...{entry['timestamps'][-1:]}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
