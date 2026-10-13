#!/usr/bin/env python3
"""Generate the decoded-RGB fixtures: small synthetic images in the sample
formats the pipeline has to convert, with the RGB bytes the serving engine's
own image loader hands its processor for each of them.

The loader (reproduced below) opens the file with Pillow, applies the EXIF
orientation and converts to RGB: an image with an alpha channel, an ``LA`` or
``PA`` mode or a ``transparency`` key is converted to RGBA and composited over
opaque white with a paste through the alpha mask; anything else goes through
``convert("RGB")``. A truecolour image is returned as stored even when it
names a transparency colour key, because its mode is already RGB.

The images land under ``tests/fixtures/images/decoded_rgb/``; the expected
bytes (raw interleaved RGB, row-major, base64 in the JSON) and the case list
in ``tests/fixtures/golden/decoded_rgb/manifest.json``.
``tests/decoded_rgb_reference.rs`` reads them. Regenerate with the Pillow
version the manifest records.
"""

import base64
import contextlib
import io
import json
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
IMAGES = FIXTURES / "images" / "decoded_rgb"
GOLDEN = FIXTURES / "golden" / "decoded_rgb"
WHITE = (255, 255, 255)


def rgba_to_rgb(image: Image.Image) -> Image.Image:
    converted = Image.new("RGB", image.size, WHITE)
    converted.paste(image, mask=image.split()[3])
    return converted


def has_transparency(image: Image.Image) -> bool:
    return image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info


def reference_load(data: bytes) -> Image.Image:
    """The loader's RGB for an encoded image, exactly as its processor gets it."""
    image = Image.open(io.BytesIO(data))
    with contextlib.suppress(Exception):
        image = ImageOps.exif_transpose(image)
    image.load()
    if image.mode == "RGB":
        return image
    if image.mode != "RGBA" and not has_transparency(image):
        return image.convert("RGB")
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return rgba_to_rgb(image)


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def png_16bit(samples: np.ndarray, colour_type: int, trns: bytes = b"") -> bytes:
    """A 16-bit PNG from (height, width[, channels]) uint16 samples; Pillow writes
    no 16-bit truecolour, so the file is assembled here (filter 0 rows)."""
    height, width = samples.shape[:2]
    rows = b"".join(b"\x00" + samples[y].astype(">u2").tobytes() for y in range(height))
    chunks = [png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 16, colour_type, 0, 0, 0))]
    if trns:
        chunks.append(png_chunk(b"tRNS", trns))
    chunks += [png_chunk(b"IDAT", zlib.compress(rows, 6)), png_chunk(b"IEND", b"")]
    return b"\x89PNG\r\n\x1a\n" + b"".join(chunks)


def sixteen_bit_samples() -> np.ndarray:
    """64 x 4 samples whose low bytes run through every rounding case."""
    y, x = np.indices((4, 64))
    return ((x * 1040 + y * 3 + 65) % 65536).astype(np.uint16)


def rgb16() -> bytes:
    v = sixteen_bit_samples()
    rgb = np.stack([v, 65535 - v, (v.astype(np.uint32) * 7 + 128) % 65536], axis=-1)
    return png_16bit(rgb.astype(np.uint16), 2)


def rgb16_trns() -> bytes:
    """16-bit truecolour with a colour key: the loader keeps the stored RGB."""
    v = sixteen_bit_samples()
    rgb = np.stack([v, 65535 - v, (v.astype(np.uint32) * 7 + 128) % 65536], axis=-1).astype(
        np.uint16
    )
    key = rgb[1, 5]
    return png_16bit(rgb, 2, struct.pack(">HHH", *(int(c) for c in key)))


CASES = (
    ("rgb16", "png", rgb16, "16-bit truecolour PNG: each sample's high byte"),
    (
        "rgb16_trns",
        "png",
        rgb16_trns,
        "16-bit truecolour PNG with a tRNS colour key: kept as stored",
    ),
)


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    GOLDEN.mkdir(parents=True, exist_ok=True)
    cases = []
    for name, ext, build, note in CASES:
        built = build()
        if isinstance(built, bytes):
            data = built
        else:
            image, save_kwargs = built
            buffer = io.BytesIO()
            image.save(buffer, format=ext.upper(), **save_kwargs)
            data = buffer.getvalue()
        (IMAGES / f"{name}.{ext}").write_bytes(data)
        opened = Image.open(io.BytesIO(data))
        mode, transparent = opened.mode, has_transparency(opened)
        rgb = np.asarray(reference_load(data), dtype=np.uint8)
        cases.append(
            {
                "name": name,
                "image": f"{name}.{ext}",
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "mode": mode,
                "transparent": transparent,
                "note": note,
                "expected_rgb_base64": base64.b64encode(rgb.tobytes()).decode("ascii"),
            }
        )
    manifest = {
        "generator": "crates/multimodal/scripts/generate_decoded_rgb_fixtures.py",
        "pillow": Image.__version__,
        "cases": cases,
    }
    (GOLDEN / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for case in cases:
        print(
            f"{case['name']}: {case['mode']} {case['width']}x{case['height']} transparent={case['transparent']}"
        )


if __name__ == "__main__":
    main()
