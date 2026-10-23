#!/usr/bin/env python3
"""Generate the reference fingerprints for Muse Spark audio preprocessing.

Reference extractor
-------------------
Muse Spark consumes audio through ``MuseSparkFeatureExtractor``
(``transformers.models.muse_spark``): mono float32 samples at 24 kHz, no
spectrogram, every clip right-padded with ``padding_value`` (0.0) to a
multiple of ``samples_per_token`` (1,920, the product of the audio tower's
``patch_sizes``). The processor replaces each ``<|audio|>`` the chat template
wrote with ``<|soft_audio_start|>``, ``ceil(real samples / 1920)`` audio
tokens and ``<|soft_audio_end|>``. This script constructs the extractor
locally with the checkpoint's ``processor_config.json`` values, so it needs
neither model weights nor a network download, and feeds it deterministic
integer-formula waveforms the Rust test rebuilds bit for bit.

For a clip that is not at 24 kHz the extractor expects an already-resampled
waveform. For this family the resampled samples are the features themselves,
so the resampling step must round the same way everywhere; a floating-point
convolution does not (torchaudio's float32 result differs from its float64
one in most samples, by up to 1e-5, and exact cancellations come out as
rounding noise that differs between implementations). The oracle is
therefore torchaudio's filter (Hann-windowed sinc, width 6, rolloff 0.99)
evaluated exactly: samples on a 2^-40 grid of the clip's peak bound,
coefficients on a 2^-62 grid, an integer convolution, one rounding to
float32 -- the arithmetic ``audio/transforms.rs::bandlimited_resample_exact``
performs. ``exact_resample`` below is that arithmetic in numpy; torchaudio's
float64 result is reported next to it (``torchaudio_f64_differing``) as
evidence that it is the same filter.

Fingerprints
------------
Each case records the extractor's ``input_values`` row as an FNV-1a hash over
its little-endian float32 bytes, and the same row rounded to bfloat16 (the
dtype the gateway writes on the wire, round to nearest even) as an FNV-1a
hash over the little-endian 16-bit patterns. The Rust test matches both
exactly: one differing bit fails it.

Usage::

    python3 crates/multimodal/scripts/generate_muse_spark_audio_fingerprints.py \
        > crates/multimodal/tests/fixtures/golden/muse_spark_audio_fingerprints.json
"""

import json
import math

import numpy as np
import torch
import torchaudio.functional as AF
from transformers import MuseSparkFeatureExtractor

SAMPLE_RATE = 24_000
SAMPLES_PER_TOKEN = 1_920
PADDING_VALUE = 0.0

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x00000100000001B3
FNV_MASK = 0xFFFFFFFFFFFFFFFF


def fnv1a(data: bytes) -> str:
    value = FNV_OFFSET
    for byte in data:
        value ^= byte
        value = (value * FNV_PRIME) & FNV_MASK
    return f"{value:016x}"


def make_waveform(count: int, seed: int, amplitude: float) -> np.ndarray:
    """The integer formula the Rust golden test rebuilds: exact in float32."""
    if seed == 0:
        return np.zeros(count, dtype=np.float32)
    index = np.arange(count, dtype=np.uint64)
    raw = ((index * 73 + np.uint64(seed) * 977) % 65_536).astype(np.int64) - 32_768
    return (raw.astype(np.float32) / np.float32(32_768.0)) * np.float32(amplitude)


SAMPLE_BITS = 40
COEFFICIENT_BITS = 62
LOWPASS_FILTER_WIDTH = 6
ROLLOFF = 0.99


def exact_kernel(orig_freq: int, new_freq: int):
    """torchaudio's sinc kernel as integers on the 2^-62 grid: one row per
    output phase, evaluated with the C library's sin/cos in the same
    operation order as the Rust implementation."""
    gcd = math.gcd(orig_freq, new_freq)
    orig_freq //= gcd
    new_freq //= gcd
    base_freq = min(orig_freq, new_freq) * ROLLOFF
    width = math.ceil(LOWPASS_FILTER_WIDTH * orig_freq / base_freq)
    kernel_len = 2 * width + orig_freq
    scale = base_freq / orig_freq
    coefficient_scale = 2.0**COEFFICIENT_BITS
    kernels = np.zeros((new_freq, kernel_len), dtype=object)
    for phase in range(new_freq):
        for k in range(kernel_len):
            idx = (k - width) / orig_freq
            t = (idx - phase / new_freq) * base_freq
            t = min(max(t, -LOWPASS_FILTER_WIDTH), LOWPASS_FILTER_WIDTH)
            w = math.cos(t * math.pi / LOWPASS_FILTER_WIDTH / 2)
            window = w * w
            radians = t * math.pi
            sinc = 1.0 if radians == 0.0 else math.sin(radians) / radians
            coefficient = sinc * window * scale
            kernels[phase, k] = round(coefficient * coefficient_scale)
    return orig_freq, new_freq, width, kernels


def exact_resample(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """The filter evaluated exactly on the integer grids, rounded to float32 once."""
    if sample_rate == SAMPLE_RATE:
        return waveform
    waveform = waveform.astype(np.float32, copy=False)
    orig_freq, new_freq, width, kernels = exact_kernel(sample_rate, SAMPLE_RATE)
    kernel_len = kernels.shape[1]
    peak = float(np.max(np.abs(waveform))) if len(waveform) else 0.0
    peak_exponent = math.frexp(peak)[1] if peak != 0.0 else 0
    sample_scale = math.ldexp(1.0, SAMPLE_BITS - peak_exponent)
    quantized = np.rint(waveform.astype(np.float64) * sample_scale).astype(np.int64)
    padded = np.zeros(width + len(waveform) + kernel_len, dtype=np.int64)
    padded[width : width + len(waveform)] = quantized
    target_len = -(-len(waveform) * new_freq // orig_freq)
    blocks = -(-len(waveform) // orig_freq)
    # Integer dot products exactly: split the 2^62 coefficients into two
    # 31-bit limbs and the 2^40 samples into two 20-bit limbs so every partial
    # product fits int64 (2^51) and every partial sum of <= 2^12 taps too.
    assert kernel_len <= 4096
    coef_lo = np.vectorize(lambda c: int(c) & ((1 << 31) - 1), otypes=[np.int64])(kernels)
    coef_hi = np.vectorize(lambda c: int(c) >> 31, otypes=[np.int64])(kernels)
    # Sample window matrix: rows = blocks, columns = taps.
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel_len)[::orig_freq][:blocks]
    neg = windows < 0
    mag = np.abs(windows)
    s_lo = (mag & ((1 << 20) - 1)).astype(np.int64)
    s_hi = (mag >> 20).astype(np.int64)
    out = np.empty(blocks * new_freq, dtype=np.float32)
    output_scale = math.ldexp(1.0, peak_exponent - SAMPLE_BITS - COEFFICIENT_BITS)
    for phase in range(new_freq):
        clo = coef_lo[phase]
        chi = coef_hi[phase]
        sign = np.where(neg, -1, 1).astype(np.int64)
        ll = ((s_lo * clo) * sign).sum(axis=1, dtype=np.int64)
        lh = ((s_lo * chi) * sign).sum(axis=1, dtype=np.int64)
        hl = ((s_hi * clo) * sign).sum(axis=1, dtype=np.int64)
        hh = ((s_hi * chi) * sign).sum(axis=1, dtype=np.int64)
        for block in range(blocks):
            total = (
                int(ll[block])
                + (int(lh[block]) << 31)
                + (int(hl[block]) << 20)
                + (int(hh[block]) << 51)
            )
            out[block * new_freq + phase] = np.float32(float(total) * output_scale)
    return out[:target_len]


def torchaudio_f64(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """torchaudio's float64 evaluation, rounded to float32: the cross-check."""
    if sample_rate == SAMPLE_RATE:
        return waveform
    tensor = torch.from_numpy(waveform.astype(np.float32, copy=False)).to(torch.float64)
    return AF.resample(tensor, sample_rate, SAMPLE_RATE).to(torch.float32).numpy()


def resample(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    return exact_resample(waveform, sample_rate)


def extractor() -> MuseSparkFeatureExtractor:
    return MuseSparkFeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLE_RATE,
        padding_value=PADDING_VALUE,
        samples_per_token=SAMPLES_PER_TOKEN,
        return_attention_mask=True,
    )


def bf16_bits(row: np.ndarray) -> bytes:
    return torch.from_numpy(row).to(torch.bfloat16).view(torch.int16).numpy().tobytes()


def one_clip(fe: MuseSparkFeatureExtractor, waveform_24k: np.ndarray) -> dict:
    features = fe(waveform_24k, sampling_rate=SAMPLE_RATE, padding="longest", return_tensors="np")
    row = features["input_values"][0].astype(np.float32, copy=False)
    real = int(features["input_values_mask"][0].sum())
    tokens = -(-real // SAMPLES_PER_TOKEN)
    assert real == len(waveform_24k)
    assert len(row) == tokens * SAMPLES_PER_TOKEN
    assert np.array_equal(row[:real], waveform_24k)
    assert np.all(row[real:] == PADDING_VALUE)
    return {
        "resampled_samples": real,
        "tokens": tokens,
        "padded_samples": int(len(row)),
        "fnv1a_f32": fnv1a(row.tobytes()),
        "fnv1a_bf16": fnv1a(bf16_bits(row)),
    }


CLIPS = [
    # name, sample rate, samples, seed, amplitude
    ("silence_one_token", 24_000, 1_920, 0, 1.0),
    ("one_sample", 24_000, 1, 11, 1.0),
    ("one_token_minus_one", 24_000, 1_919, 12, 0.5),
    ("one_token", 24_000, 1_920, 13, 0.25),
    ("one_token_plus_one", 24_000, 1_921, 14, 1.0),
    ("three_tokens_and_five", 24_000, 3 * 1_920 + 5, 15, 0.75),
    ("two_seconds_native", 24_000, 48_000, 16, 1.0),
    ("one_second_16k", 16_000, 16_000, 21, 1.0),
    ("half_second_16k", 16_000, 8_000, 22, 0.5),
    ("one_sample_16k", 16_000, 1, 23, 1.0),
    ("one_second_8k", 8_000, 8_000, 31, 1.0),
    ("short_8k", 8_000, 123, 32, 1.0),
    ("one_second_44k1", 44_100, 44_100, 41, 1.0),
    ("half_second_44k1", 44_100, 22_050, 42, 0.5),
    ("one_second_48k", 48_000, 48_000, 51, 1.0),
    ("short_48k", 48_000, 1_000, 52, 1.0),
    ("ten_seconds_22k05", 22_050, 220_500, 61, 0.9),
]

BATCHES = [
    ("native_pair", ["three_tokens_and_five", "one_token"]),
    ("mixed_rates", ["one_second_16k", "short_8k", "one_second_48k"]),
]


def main() -> None:
    fe = extractor()
    clips = []
    rows = {}
    for name, sample_rate, samples, seed, amplitude in CLIPS:
        source = make_waveform(samples, seed, amplitude)
        waveform = resample(source, sample_rate)
        case = one_clip(fe, waveform)
        cross = torchaudio_f64(source, sample_rate)
        case["torchaudio_f64_differing"] = int((cross != waveform).sum())
        case["torchaudio_f64_max_abs"] = (
            float(np.max(np.abs(cross.astype(np.float64) - waveform))) if len(cross) else 0.0
        )
        features = fe(waveform, sampling_rate=SAMPLE_RATE, padding="longest", return_tensors="np")
        rows[name] = features["input_values"][0].astype(np.float32, copy=False)
        clips.append(
            {
                "name": name,
                "sample_rate": sample_rate,
                "samples": samples,
                "seed": seed,
                "amplitude": amplitude,
                **case,
            }
        )
    batches = []
    for name, members in BATCHES:
        flat = np.concatenate([rows[member] for member in members]).astype(np.float32, copy=False)
        batches.append(
            {
                "name": name,
                "clips": members,
                "samples_per_audio": [int(len(rows[member])) for member in members],
                "fnv1a_f32": fnv1a(flat.tobytes()),
                "fnv1a_bf16": fnv1a(bf16_bits(flat)),
            }
        )
    document = {
        "generator": "crates/multimodal/scripts/generate_muse_spark_audio_fingerprints.py",
        "reference": (
            "transformers MuseSparkFeatureExtractor "
            "(sampling_rate 24000, samples_per_token 1920, padding_value 0.0)"
        ),
        "resampler": (
            "torchaudio sinc_interp_hann filter (width 6, rolloff 0.99) evaluated exactly "
            "on 2^-40 sample / 2^-62 coefficient grids, rounded to float32 once"
        ),
        "torch": torch.__version__,
        "torchaudio": __import__("torchaudio").__version__,
        "transformers": __import__("transformers").__version__,
        "sample_rate": SAMPLE_RATE,
        "samples_per_token": SAMPLES_PER_TOKEN,
        "clips": clips,
        "batches": batches,
    }
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
