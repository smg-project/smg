#!/usr/bin/env python3
"""Generate the decoded-RGB fixtures: small synthetic images in the sample
formats the pipeline has to convert (transparency, 16-bit depth), with the RGB
bytes each of the serving engines' image loaders hands its processor for them.

Two loader rules are recorded per case, both with Pillow. The compositing
loader (reproduced below) opens the file, applies the EXIF orientation and
converts to RGB: an image with an alpha channel, an ``LA`` or ``PA`` mode or
a ``transparency`` key is converted to RGBA and composited over opaque white
with a paste through the alpha mask; anything else goes through
``convert("RGB")``; a truecolour image is returned as stored even when it
names a transparency colour key, because its mode is already RGB. The
dropping loader opens the file and calls ``convert("RGB")``, which keeps
every pixel's stored colour and ignores the alpha channel (the orientation
step is applied first here too: it is the gateway's own, before any policy).

The images land under ``tests/fixtures/images/decoded_rgb/``; the expected
bytes (raw interleaved RGB, row-major, base64 in the JSON: ``expected_rgb``
for the composite, ``expected_dropped_rgb`` for the drop) and the case list
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


def dropped_load(data: bytes) -> Image.Image:
    """The dropping loader's RGB: the stored colour, the alpha channel ignored."""
    image = Image.open(io.BytesIO(data))
    with contextlib.suppress(Exception):
        image = ImageOps.exif_transpose(image)
    image.load()
    return image if image.mode == "RGB" else image.convert("RGB")


def alpha_grid() -> tuple[Image.Image, dict]:
    """Every alpha value (rows) against 52 channel values (columns), three
    different values per pixel, so each blend rounding case is pinned."""
    values = np.array([*range(0, 255, 5), 255], dtype=np.uint16)
    y, x = np.indices((256, len(values)))
    v = values[x]
    rgba = np.stack([v, 255 - v, (v * 3) % 256, y], axis=-1).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA"), {}


def rgba_opaque() -> tuple[Image.Image, dict]:
    y, x = np.indices((16, 16))
    rgba = np.stack([x * 17, y * 17, (x + y) * 8, np.full_like(x, 255)], axis=-1)
    return Image.fromarray(rgba.astype(np.uint8), "RGBA"), {}


def la_blend() -> tuple[Image.Image, dict]:
    y, x = np.indices((64, 64))
    la = np.stack([x * 255 // 63, y * 255 // 63], axis=-1).astype(np.uint8)
    return Image.fromarray(la, "LA"), {}


def palette_trns() -> tuple[Image.Image, dict]:
    """Eight palette entries, a tRNS chunk for the first six (alpha 0, 1, 127,
    128, 254, 255); the last two are opaque by omission."""
    palette = [
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (40, 90, 160),
        (200, 150, 100),
        (255, 255, 0),
        (17, 34, 51),
    ]
    image = Image.new("P", (16, 16))
    image.putpalette([c for rgb in palette for c in rgb])
    y, x = np.indices((16, 16))
    image.putdata(((x + y) % 8).flatten().tolist())
    return image, {"transparency": bytes([0, 1, 127, 128, 254, 255])}


def gray_trns() -> tuple[Image.Image, dict]:
    """8-bit greyscale with a colour key: value 85 is transparent."""
    y, x = np.indices((16, 16))
    gray = ((x * 17 + y) % 256).astype(np.uint8)
    return Image.fromarray(gray, "L"), {"transparency": 85}


def rgb_trns() -> tuple[Image.Image, dict]:
    """Truecolour with a colour key: the loader keeps the stored RGB."""
    y, x = np.indices((16, 16))
    rgb = np.stack([x * 17, 255 - x * 17, (y // 8) * 255], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb, "RGB"), {"transparency": (85, 170, 0)}


def rgba_exif6() -> tuple[Image.Image, dict]:
    """RGBA with EXIF orientation 6 in an eXIf chunk: transposed, then
    composited."""
    y, x = np.indices((24, 16))
    rgba = np.stack([x * 16, y * 10, (x * y) % 256, (x * 16 + y) % 256], axis=-1)
    exif = Image.Exif()
    exif[0x0112] = 6
    return Image.fromarray(rgba.astype(np.uint8), "RGBA"), {"exif": exif.tobytes()}


def webp_rgba() -> tuple[Image.Image, dict]:
    y, x = np.indices((32, 32))
    rgba = np.stack([x * 8, y * 8, (x + y) * 4, x * 8], axis=-1).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA"), {"lossless": True, "exact": True}


def gif_trns() -> tuple[Image.Image, dict]:
    image = Image.new("P", (16, 16))
    image.putpalette([0, 0, 0, 200, 30, 30, 30, 200, 30, 30, 30, 200])
    y, x = np.indices((16, 16))
    image.putdata(((x + y) % 4).flatten().tolist())
    return image, {"transparency": 0}


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
    ("alpha_grid", "png", alpha_grid, "RGBA PNG: every alpha against 52 values per channel"),
    ("rgba_opaque", "png", rgba_opaque, "RGBA PNG whose alpha is 255 everywhere"),
    ("la_blend", "png", la_blend, "LA PNG: grey replicated, then composited"),
    ("palette_trns", "png", palette_trns, "palette PNG with per-index alpha in tRNS"),
    ("gray_trns", "png", gray_trns, "greyscale PNG with a tRNS colour key"),
    ("rgb_trns", "png", rgb_trns, "truecolour PNG with a tRNS colour key: kept as stored"),
    ("rgba_exif6", "png", rgba_exif6, "RGBA PNG with EXIF orientation 6"),
    ("webp_rgba", "webp", webp_rgba, "lossless WebP with alpha"),
    ("gif_trns", "gif", gif_trns, "GIF with a transparent index"),
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
        dropped = np.asarray(dropped_load(data), dtype=np.uint8)
        assert dropped.shape == rgb.shape
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
                "expected_dropped_rgb_base64": base64.b64encode(dropped.tobytes()).decode("ascii"),
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
