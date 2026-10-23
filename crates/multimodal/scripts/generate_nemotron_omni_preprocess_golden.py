#!/usr/bin/env python3
"""Golden values for Nemotron-H Omni image, video and audio preprocessing.

Reproduces vLLM's Nano-Nemotron-VL dynamic-resolution image tiler, its video
frame sizing and its ``ParakeetExtractor`` with torch on deterministic inputs
the Rust test can rebuild, and writes ``crates/multimodal/tests/fixtures/golden/
nemotron_omni_preprocess.json``: each image's patch grid and token count, each
video's (frames, rows, cols) and token count, and each audio's per-clip frame
counts and token count, plus sampled output values and sums the test compares
within a tolerance.

Run from the repository root: ``python crates/multimodal/scripts/generate_nemotron_omni_preprocess_golden.py``
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers.audio_utils import mel_filter_bank

FIXTURES = Path(__file__).resolve().parent.parent / "tests/fixtures"
OUT = FIXTURES / "golden/nemotron_omni_preprocess.json"

PATCH = 16
MIN_PATCHES, MAX_PATCHES = 1024, 13312
VIDEO_PATCHES, TUBELET = 1024, 2
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
SAMPLES = 512

SAMPLE_RATE, N_FFT, HOP, WIN, N_MELS = 16000, 512, 160, 400, 128
CLIP, TAIL_MIN = 30 * SAMPLE_RATE, round(0.1 * SAMPLE_RATE)


def seeded_image(width: int, height: int, seed: int) -> np.ndarray:
    """The Rust tests' ``make_seeded_image``."""
    y, x = np.mgrid[0:height, 0:width]
    channels = [(x * 7 + y * 3) % 256, (x * 5 + y * 11) % 256, (x + y * 2) % 256]
    return np.stack([(c + seed) % 256 for c in channels], axis=-1).astype(np.uint8)


def plan(width: int, height: int) -> tuple[int, int]:
    """vLLM ``DynamicResolutionImageTiler.process_media`` at the per-image maximum."""
    budget = MAX_PATCHES
    ch, cw = round(height / PATCH + 0.5), round(width / PATCH + 0.5)
    factor = min(math.sqrt(budget / (ch * cw)), 1.0)
    rows, cols = math.floor(factor * ch), math.floor(factor * cw)
    if budget > MIN_PATCHES and rows * cols < MIN_PATCHES:
        up = math.sqrt(MIN_PATCHES / (rows * cols))
        rows, cols = math.ceil(up * rows), math.ceil(up * cols)
    if rows % 2:
        rows = rows + 1 if (rows + 1) * cols <= budget else max(2, rows - 1)
    if cols % 2:
        cols = cols + 1 if rows * (cols + 1) <= budget else max(2, cols - 1)
    return rows, cols


def video_grid(width: int, height: int) -> tuple[int, int]:
    """vLLM ``_compute_aspect_preserving_size`` at the checkpoint's frame target."""
    aspect = width / max(height, 1)
    rows = max(round(math.sqrt(VIDEO_PATCHES / aspect)), 1)
    cols = max(round(math.sqrt(VIDEO_PATCHES * aspect)), 1)
    if (rows + rows % 2) * (cols + cols % 2) <= VIDEO_PATCHES:
        return rows + rows % 2, cols + cols % 2
    return max(2, rows - rows % 2), max(2, cols - cols % 2)


def image_patches(rgb: np.ndarray) -> tuple[int, int, torch.Tensor]:
    height, width, _ = rgb.shape
    rows, cols = plan(width, height)
    return rows, cols, resized_patches(rgb, rows, cols)


def resized_patches(rgb: np.ndarray, rows: int, cols: int) -> torch.Tensor:
    """vLLM ``_bicubic_resize_and_normalize`` then RADIO's patch flattening."""
    t = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(torch.float32)
    t = F.interpolate(
        t, size=(rows * PATCH, cols * PATCH), mode="bicubic", align_corners=False, antialias=True
    )
    t = (t / 255.0 - MEAN) / STD
    return t[0].view(3, rows, PATCH, cols, PATCH).permute(1, 3, 0, 2, 4).reshape(rows * cols, -1)


def audio_pcm(seconds: float, gain: float) -> np.ndarray:
    """Two tones and a low-level chirp scaled by ``gain``, computed in f64 and stored as f32."""
    t = np.arange(round(seconds * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    x = 0.3 * np.sin(2 * np.pi * 440.0 * t) + 0.2 * np.sin(2 * np.pi * 1234.5 * t + 0.3)
    x += 0.05 * np.sin(2 * np.pi * (200.0 + 900.0 * t) * t)
    return (gain * x).astype(np.float32)


def clip_sizes(length: int) -> list[int]:
    length = max(length, TAIL_MIN)
    sizes = [CLIP] * (length // CLIP)
    if length % CLIP:
        sizes.append(max(length % CLIP, TAIL_MIN))
    return sizes


def clip_tokens(frames: int) -> int:
    for _ in range(3):
        frames = math.floor((frames - 1) / 2 + 1)
    return frames


def parakeet_features(clip: torch.Tensor) -> torch.Tensor:
    """vLLM ``ParakeetExtractor`` for one clip: its ``samples // hop`` normalized frames."""
    x = torch.cat([clip[:1], clip[1:] - 0.97 * clip[:-1]])
    stft = torch.stft(
        x,
        N_FFT,
        hop_length=HOP,
        win_length=WIN,
        window=torch.hann_window(WIN, periodic=False),
        return_complex=True,
        pad_mode="constant",
    )
    filters = torch.from_numpy(
        mel_filter_bank(
            num_frequency_bins=N_FFT // 2 + 1,
            num_mel_filters=N_MELS,
            min_frequency=0.0,
            max_frequency=SAMPLE_RATE / 2,
            sampling_rate=SAMPLE_RATE,
            norm="slaney",
            mel_scale="slaney",
        ).T
    ).to(torch.float32)
    mel = torch.log(filters @ (stft.real.square() + stft.imag.square()) + 2**-24).T
    valid = len(clip) // HOP
    mel = mel[:valid]
    mean = mel.mean(0)
    std = ((mel - mean) ** 2).sum(0).div(valid - 1).sqrt()
    return (mel - mean) / (std + 1e-5)


def summary(values: torch.Tensor) -> dict:
    flat = values.reshape(-1).to(torch.float64)
    stride = max(1, flat.numel() // SAMPLES)
    return {
        "numel": flat.numel(),
        "sample_stride": stride,
        "samples": flat[::stride][:SAMPLES].tolist(),
        "sum": flat.sum().item(),
        "sum_sq": flat.square().sum().item(),
    }


def main() -> None:
    images = []
    for source in [
        {"width": 224, "height": 224, "seed": 0},
        {"width": 333, "height": 517, "seed": 9},
        {"width": 1000, "height": 700, "seed": 3},
        {"file": "deepseek_v41_corn.png"},
    ]:
        if "file" in source:
            rgb = np.asarray(Image.open(FIXTURES / "images" / source["file"]).convert("RGB"))
        else:
            rgb = seeded_image(source["width"], source["height"], source["seed"])
        rows, cols, patches = image_patches(rgb)
        images.append(
            {
                "source": source,
                "rows": rows,
                "cols": cols,
                "tokens": rows * cols // 4,
                **summary(patches),
            }
        )

    videos = []
    # Non-square with an odd frame count: the last tubelet repeats its frame engine-side.
    for width, height, seeds in [(333, 200, [1, 2, 3])]:
        rows, cols = video_grid(width, height)
        frames = [resized_patches(seeded_image(width, height, s), rows, cols) for s in seeds]
        tubelets = -(-len(seeds) // TUBELET)
        videos.append(
            {
                "source": {"width": width, "height": height, "seeds": seeds},
                "grid": [len(seeds), rows, cols],
                "tokens": tubelets * rows * cols // 4,
                **summary(torch.cat(frames)),
            }
        )

    audios = []
    # A full 30-second clip at normal and at quiet level, where the per-bin statistics
    # sum thousands of frames.
    for seconds, gain in [(1.0, 1.0), (2.37, 1.0), (0.05, 1.0), (30.0, 1.0), (30.0, 2e-3)]:
        pcm = torch.from_numpy(audio_pcm(seconds, gain))
        sizes = clip_sizes(len(pcm))
        pcm = F.pad(pcm, (0, sum(sizes) - len(pcm)))
        features, frames, start = [], [], 0
        for size in sizes:
            features.append(parakeet_features(pcm[start : start + size]))
            frames.append(size // HOP)
            start += size
        audios.append(
            {
                "seconds": seconds,
                "gain": gain,
                "clip_frames": frames,
                "tokens": sum(clip_tokens(f) for f in frames),
                **summary(torch.cat(features)),
            }
        )

    document = {
        "generator": "crates/multimodal/scripts/generate_nemotron_omni_preprocess_golden.py",
        "reference": "vLLM nano_nemotron_vl dynamic tiler, video sizing and ParakeetExtractor, reproduced with torch",
        "torch": torch.__version__,
        "images": images,
        "videos": videos,
        "audios": audios,
    }
    OUT.write_text(json.dumps(document, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
