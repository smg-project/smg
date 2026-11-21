#!/usr/bin/env python3
"""The image classes of the release comparison, made from one seeded scene.

Every file is derived from one deterministic synthetic scene (two gradients, a checker,
a few blocks, light noise), so a resize and a colour conversion have structure to work
on and the files are the same on every machine that runs the same Pillow and numpy.
`classes.json` beside this file names the 19 standard image classes and the batch by
these file names; `classes_wide.json` names all 59.

    python3 make_images.py --out <dir>

Needs numpy and Pillow (the pins of `scripts/bench/multimodal/requirements.txt` wrote
the files the recorded results were measured on; `environment.json` lists the SHA-256
of the standard ones). Writes the files and `images.json` (name, bytes, SHA-256) into
the directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

SEED = 20261105
# An ICC profile carries its creation time in its header (bytes 24 to 35: year, month,
# day, hour, minute, second). The two files with a profile get the times the recorded
# ones carry, so they are the same bytes whenever they are made.
PROFILE_TIMES = {
    "jpeg-icc-srgb.jpg": (2026, 11, 6, 7, 42, 52),
    "png-icc-srgb.png": (2026, 11, 6, 7, 42, 53),
}


def scene(width: int, height: int, seed: int = SEED) -> np.ndarray:
    """A deterministic RGB8 scene: two gradients, a checker, a few blocks, light noise."""
    rng = np.random.default_rng(seed + width * 7919 + height)
    ys, xs = np.mgrid[0:height, 0:width]
    red = (xs * 255 // max(width - 1, 1)).astype(np.uint8)
    green = (ys * 255 // max(height - 1, 1)).astype(np.uint8)
    checker = (((xs // max(width // 16, 1)) + (ys // max(height // 16, 1))) % 2) * 255
    image = np.stack([red, green, checker.astype(np.uint8)], axis=-1)
    for _ in range(6):
        x0, y0 = rng.integers(0, max(width, 1)), rng.integers(0, max(height, 1))
        w, h = rng.integers(1, max(width // 4, 2)), rng.integers(1, max(height // 4, 2))
        image[y0 : y0 + h, x0 : x0 + w] = rng.integers(0, 256, size=3, dtype=np.uint8)
    noise = rng.integers(0, 12, size=image.shape, dtype=np.uint8)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def orientation(value: int) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = value
    return exif.tobytes()


def srgb_profile(name: str) -> bytes:
    profile = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    profile[24:36] = b"".join(part.to_bytes(2, "big") for part in PROFILE_TIMES[name])
    return bytes(profile)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", required=True, help="the directory the files are written to")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    def save(name: str, image: Image.Image, fmt: str, **options) -> None:
        path = out / name
        image.save(path, fmt, **options)
        data = path.read_bytes()
        written.append(
            {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
        print(f"  {name}: {len(data)} B", file=sys.stderr)

    def made(width: int, height: int, seed: int = SEED) -> Image.Image:
        return Image.fromarray(scene(width, height, seed))

    base = made(1024, 768)
    small = made(640, 480)
    rgba = base.convert("RGBA")
    rgba.putalpha(Image.fromarray(scene(1024, 768, SEED + 1)[:, :, 0].astype(np.uint8), "L"))
    palette = base.convert("P", palette=Image.ADAPTIVE, colors=64)

    # sizes: baseline JPEG 4:2:0 at quality 90, then PNG
    for width, height in [
        (1, 1),
        (2, 2),
        (7, 5),
        (16, 16),
        (64, 64),
        (256, 256),
        (640, 480),
        (1024, 768),
        (1920, 1080),
        (4096, 2304),
        (8192, 8192),
    ]:
        save(
            f"size-{width}x{height}-jpeg.jpg",
            made(width, height),
            "JPEG",
            quality=90,
            subsampling="4:2:0",
        )
    for width, height in [(2000, 40), (40, 2000), (4000, 15), (10000, 10), (3, 900)]:
        save(f"aspect-{width}x{height}-jpeg.jpg", made(width, height), "JPEG", quality=90)
    for width, height in [(256, 256), (1024, 768), (1920, 1080), (3840, 2160)]:
        save(f"size-{width}x{height}-png.png", made(width, height), "PNG", compress_level=6)

    # JPEG: sampling, scan order, colour space, profile, orientation
    save("jpeg-baseline-420-q90.jpg", base, "JPEG", quality=90, subsampling="4:2:0")
    save("jpeg-baseline-444-q95.jpg", base, "JPEG", quality=95, subsampling="4:4:4")
    save("jpeg-baseline-422-q85.jpg", base, "JPEG", quality=85, subsampling="4:2:2")
    save("jpeg-progressive-q90.jpg", base, "JPEG", quality=90, progressive=True)
    save("jpeg-optimized-q75.jpg", base, "JPEG", quality=75, optimize=True)
    save("jpeg-grayscale.jpg", base.convert("L"), "JPEG", quality=90)
    save("jpeg-cmyk.jpg", base.convert("CMYK"), "JPEG", quality=90)
    save(
        "jpeg-icc-srgb.jpg",
        base,
        "JPEG",
        quality=90,
        icc_profile=srgb_profile("jpeg-icc-srgb.jpg"),
    )
    save("jpeg-q100-444.jpg", base, "JPEG", quality=100, subsampling="4:4:4")
    for value in range(1, 9):
        save(
            f"jpeg-exif-orientation-{value}.jpg",
            small,
            "JPEG",
            quality=90,
            exif=orientation(value),
        )
    save(
        "jpeg-exif-orientation-6-tall.jpg", made(480, 800), "JPEG", quality=90, exif=orientation(6)
    )

    # PNG: colour types, bit depths, transparency, profile, orientation
    save("png-rgb8.png", base, "PNG", compress_level=6)
    save("png-rgba8.png", rgba, "PNG", compress_level=6)
    save("png-gray8.png", base.convert("L"), "PNG")
    save("png-gray16.png", Image.fromarray(scene(640, 480)[:, :, 1].astype(np.uint16) * 257), "PNG")
    save("png-palette.png", palette, "PNG")
    save("png-palette-trns.png", palette, "PNG", transparency=3)
    save("png-1bit.png", base.convert("1"), "PNG")
    save("png-la8.png", base.convert("LA"), "PNG")
    save("png-icc-srgb.png", base, "PNG", icc_profile=srgb_profile("png-icc-srgb.png"))
    save("png-exif-orientation-6.png", small, "PNG", exif=orientation(6))

    # WebP, GIF, BMP, TIFF, ICO
    save("webp-lossy-q80.webp", base, "WEBP", quality=80)
    save("webp-lossless.webp", base, "WEBP", lossless=True)
    save("webp-lossy-alpha.webp", rgba, "WEBP", quality=80)
    save("webp-exif-orientation-6.webp", small, "WEBP", quality=90, exif=orientation(6))
    save("gif-static.gif", base.convert("P", palette=Image.ADAPTIVE), "GIF")
    frames = [
        made(128, 96, SEED + index).convert("P", palette=Image.ADAPTIVE) for index in range(10)
    ]
    save(
        "gif-animated-10.gif",
        frames[0],
        "GIF",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    save("bmp-24.bmp", base, "BMP")
    save("tiff-uncompressed.tiff", base, "TIFF")
    save("tiff-lzw.tiff", base, "TIFF", compression="tiff_lzw")
    save("tiff-exif-orientation-3.tiff", small, "TIFF", exif=orientation(3))
    save("ico-64.ico", made(64, 64), "ICO")

    (out / "images.json").write_text(json.dumps({"seed": SEED, "files": written}, indent=1) + "\n")
    print(f"make_images: {len(written)} files in {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
