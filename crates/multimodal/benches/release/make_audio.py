#!/usr/bin/env python3
"""The WAV classes of the release comparison, written with the standard library alone.

Every clip is one second of a seeded signal (a 440 Hz tone, a rising tone and a little
broadband noise) repeated to the clip's length, so the files are the same bytes on every
machine. `audio_classes.json` beside this file names the classes by these file names.

    python3 make_audio.py --out <dir>

Uncompressed WAV only: 16-, 24- and 32-bit PCM, IEEE float and A-law; 8 to 96 kHz; one,
two and six channels; 0.1 s to 600 s. A compressed clip (FLAC, MP3, AAC, Opus, Vorbis)
needs an encoder and is not made here. Writes the files and `audio.json` (name, bytes,
SHA-256) into the directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import sys
from array import array
from pathlib import Path

SEED = 20261105
PCM, IEEE_FLOAT, ALAW = 1, 3, 6


def little_endian(values: array) -> bytes:
    """The array's bytes in the byte order WAV requires."""
    if sys.byteorder == "big":
        values.byteswap()
    return values.tobytes()


def second(rate: int, seed: int = SEED) -> list[int]:
    """One second of the signal as 16-bit samples: a tone, a rising tone, light noise."""
    noise = random.Random(seed + rate)
    samples = []
    for index in range(rate):
        t = index / rate
        value = (
            0.5 * math.sin(2 * math.pi * 440.0 * t)
            + 0.2 * math.sin(2 * math.pi * (200.0 + 900.0 * t) * t)
            + 0.05 * noise.uniform(-1.0, 1.0)
        )
        samples.append(max(-32768, min(32767, round(value * 32767 * 0.9))))
    return samples


def alaw(sample: int) -> int:
    """One 16-bit sample as a G.711 A-law byte."""
    sign = 0x80 if sample >= 0 else 0x00
    magnitude = min(abs(sample), 32767) >> 3
    if magnitude < 32:
        code = magnitude >> 1
    else:
        exponent = magnitude.bit_length() - 5
        code = (exponent << 4) | ((magnitude >> exponent) & 0x0F)
    return (sign | code) ^ 0x55


def encode(samples: list[int], kind: str) -> tuple[int, int, bytes]:
    """The samples in one sample format: its tag, its bits per sample, its bytes."""
    if kind == "pcm16":
        return PCM, 16, little_endian(array("h", samples))
    if kind == "pcm24":
        return (
            PCM,
            24,
            b"".join((value << 8).to_bytes(3, "little", signed=True) for value in samples),
        )
    if kind == "pcm32":
        return PCM, 32, little_endian(array("i", [value << 16 for value in samples]))
    if kind == "float32":
        return IEEE_FLOAT, 32, little_endian(array("f", [value / 32768.0 for value in samples]))
    if kind == "alaw":
        return ALAW, 8, bytes(alaw(value) for value in samples)
    raise ValueError(kind)


def wav(tag: int, bits: int, rate: int, channels: int, data: bytes) -> bytes:
    """A RIFF/WAVE file: the format chunk, a fact chunk where the tag is not PCM, the data."""
    block = channels * bits // 8
    fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * block, block, bits)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    if tag != PCM:
        fmt += struct.pack("<H", 0)
        chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
        chunks += b"fact" + struct.pack("<II", 4, len(data) // block)
    chunks += b"data" + struct.pack("<I", len(data)) + data + (b"\x00" if len(data) % 2 else b"")
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def clip(
    rate: int, seconds: float, channels: int = 1, kind: str = "pcm16", content: str = "signal"
) -> bytes:
    base = second(rate)
    if content == "silence":
        base = [0] * rate
    elif content == "dc":
        base = [max(-32768, min(32767, value // 2 + 8000)) for value in base]
    elif content == "clipping":
        base = [max(-32768, min(32767, value * 4)) for value in base]
    if channels > 1:
        # every channel is the same second, started a little later than the one before
        frames = []
        for index in range(rate):
            frames.extend(base[(index + 37 * channel) % rate] for channel in range(channels))
        base = frames
    tag, bits, data = encode(base, kind)
    block = channels * bits // 8
    whole, rest = divmod(round(seconds * rate), rate)
    return wav(tag, bits, rate, channels, data * whole + data[: rest * block])


CLIPS = [
    ("wav-16-16k-mono-0.1s.wav", {"rate": 16000, "seconds": 0.1}),
    ("wav-16-16k-mono-1s.wav", {"rate": 16000, "seconds": 1}),
    ("wav-16-16k-mono-30s.wav", {"rate": 16000, "seconds": 30}),
    ("wav-16-16k-mono-600s.wav", {"rate": 16000, "seconds": 600}),
    ("wav-16-16k-mono-clipping-30s.wav", {"rate": 16000, "seconds": 30, "content": "clipping"}),
    ("wav-16-16k-mono-dc-30s.wav", {"rate": 16000, "seconds": 30, "content": "dc"}),
    ("wav-16-16k-mono-silence-30s.wav", {"rate": 16000, "seconds": 30, "content": "silence"}),
    ("wav-16-8000-mono-30s.wav", {"rate": 8000, "seconds": 30}),
    ("wav-16-22050-mono-30s.wav", {"rate": 22050, "seconds": 30}),
    ("wav-16-44100-mono-30s.wav", {"rate": 44100, "seconds": 30}),
    ("wav-16-48000-mono-30s.wav", {"rate": 48000, "seconds": 30}),
    ("wav-16-96000-mono-30s.wav", {"rate": 96000, "seconds": 30}),
    ("wav-16-44100-stereo-30s.wav", {"rate": 44100, "seconds": 30, "channels": 2}),
    ("wav-16-48k-2ch-30s.wav", {"rate": 48000, "seconds": 30, "channels": 2}),
    ("wav-16-48k-6ch-30s.wav", {"rate": 48000, "seconds": 30, "channels": 6}),
    ("wav-24-16k-mono-30s.wav", {"rate": 16000, "seconds": 30, "kind": "pcm24"}),
    ("wav-32-16k-mono-30s.wav", {"rate": 16000, "seconds": 30, "kind": "pcm32"}),
    ("wav-f32-16k-mono-30s.wav", {"rate": 16000, "seconds": 30, "kind": "float32"}),
    ("wav-alaw-8k-mono-30s.wav", {"rate": 8000, "seconds": 30, "kind": "alaw"}),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", required=True, help="the directory the files are written to")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, options in CLIPS:
        data = clip(**options)
        (out / name).write_bytes(data)
        written.append(
            {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
        print(f"  {name}: {len(data)} B", file=sys.stderr)
    (out / "audio.json").write_text(json.dumps({"seed": SEED, "files": written}, indent=1) + "\n")
    print(f"make_audio: {len(written)} files in {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
