#!/usr/bin/env python3
"""Record DeepSeek-V4.1 image-preprocessing fingerprints from the reference.

The reference is the checkpoint's own ``inference/image_processor.py``
(``deepseek-ai/DeepSeek-V4.1-Flash``): ``load_image`` contain-fits the image
into a patch-aligned canvas with ``ImageOps.pad`` (gray 127), normalises
``x / 255`` then ``(x - 0.5) / 0.5`` in float32, casts to bfloat16 and cuts
14x14 ViT patches. This script runs that code unmodified over a fixed set of
images and writes, per image, the grids, the token count, and FNV-1a
fingerprints of the exact bytes:

* ``fnv1a_f32`` — the float32 patches *before* the bfloat16 cast, recomputed
  here with the same PIL call and the same float32 op order (the script
  asserts that casting this recomputation to bfloat16 reproduces the
  reference's tensor bit-for-bit, so the two never drift apart);
* ``fnv1a_bf16`` — the reference's bfloat16 patches as little-endian u16 bits,
  which is what the engine receives.

Images are either synthetic (a seeded RGB pattern the Rust test regenerates
identically) or PNG files under the fixtures directory (PNG so that PIL and
the Rust ``image`` crate decode identical pixels; the DeepSeek example JPEGs
are converted once with ``--convert-examples``).

Usage:
    python crates/multimodal/scripts/generate_deepseek_v41_preprocess_fingerprints.py \
        --checkpoint /path/to/DeepSeek-V4.1-Flash \
        [--convert-examples] [--output <fixtures>/golden/deepseek_v41_preprocess_fingerprints.json]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import PIL
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGES_DIR = REPO_ROOT / "crates/multimodal/tests/fixtures/images"
DEFAULT_OUTPUT = (
    REPO_ROOT / "crates/multimodal/tests/fixtures/golden/deepseek_v41_preprocess_fingerprints.json"
)

# (name, width, height, seed): the Rust test rebuilds these with the same formula.
SYNTHETIC_CASES = [
    ("synthetic_64x48", 64, 48, 0),
    ("synthetic_100x100", 100, 100, 3),
    ("synthetic_640x480", 640, 480, 7),
    ("synthetic_1024x768", 1024, 768, 11),
    ("synthetic_1920x1080", 1920, 1080, 17),
    ("synthetic_4000x3000", 4000, 3000, 23),
    ("synthetic_8000x1000", 8000, 1000, 29),
    ("synthetic_50x900", 50, 900, 31),
    # Sizes whose contain edge lands a hair below .5 (Python round() goes
    # down; an epsilon tie test would go up).
    ("synthetic_41x56", 41, 56, 37),
    ("synthetic_28x351", 28, 351, 41),
]

# PNG fixtures (converted from the checkpoint's inference/examples/images).
FILE_CASES = [
    ("carrots", "deepseek_v41_carrots.png"),
    ("corn", "deepseek_v41_corn.png"),
]

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3  # 2**40 + 0x1B3, the 64-bit FNV prime
FNV_MASK = (1 << 64) - 1


def fnv1a(data: bytes) -> str:
    h = FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & FNV_MASK
    return f"{h:016x}"


def seeded_image(width: int, height: int, seed: int) -> Image.Image:
    """The Rust tests' seeded pattern: R=(x*7+y*3)%256, G=(x*5+y*11)%256,
    B=(x+y*2)%256, each plus the seed with u8 wraparound."""
    y, x = np.mgrid[0:height, 0:width].astype(np.uint32)
    r = (x * 7 + y * 3) % 256
    g = (x * 5 + y * 11) % 256
    b = (x + y * 2) % 256
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint32)
    rgb = ((rgb + seed) % 256).astype(np.uint8)
    return Image.fromarray(rgb)


def png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def load_reference(checkpoint: Path):
    path = checkpoint / "inference" / "image_processor.py"
    spec = importlib.util.spec_from_file_location("deepseek_v41_image_processor", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, hashlib.sha256(path.read_bytes()).hexdigest()


def vision_args(checkpoint: Path) -> SimpleNamespace:
    config = json.loads((checkpoint / "config.json").read_text())
    vision = config["vision_config"]
    return SimpleNamespace(
        vision_patch_size=vision["patch_size"],
        vision_downsample_ratio=vision["downsample_ratio"],
        vision_max_n_token=vision["max_image_tokens"],
        vision_min_pixels=vision["min_pixels"],
        vision_max_wh_ratio=vision.get("max_wh_ratio"),
    )


def f32_patches_like_reference(module, image: Image.Image, args) -> np.ndarray:
    """The reference ``load_image`` up to (not including) the bfloat16 cast,
    with the same PIL call and float32 op order; returns (np, 3, p, p)."""
    from PIL import ImageOps

    p = args.vision_patch_size
    n_llm_h, n_llm_w, best_height, best_width = module.plan_image_grid(
        image.width, image.height, args
    )
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if (
        args.vision_max_wh_ratio is not None
        and image.width >= args.vision_max_wh_ratio * image.height
    ):
        transformed = image.resize((best_width, best_height))
    else:
        transformed = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = np.asarray(transformed, dtype=np.float32).transpose(2, 0, 1) / 255
    x = (x - 0.5) / 0.5
    assert x.dtype == np.float32
    patches = (
        x.reshape(3, n_vit_h, p, n_vit_w, p)
        .transpose(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, p, p)
    )
    return np.ascontiguousarray(patches, dtype=np.float32)


def record_case(module, args, name: str, source: dict, image: Image.Image) -> dict:
    data = png_bytes(image)
    ref_patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = module.load_image({"data": data}, args)
    assert ref_patches.dtype == torch.bfloat16
    f32 = f32_patches_like_reference(module, image.convert("RGB"), args)
    recomputed_bf16 = torch.from_numpy(f32).to(torch.bfloat16)
    if not torch.equal(recomputed_bf16, ref_patches):
        raise SystemExit(f"{name}: float32 recomputation does not reproduce the reference bf16")
    bf16_bits = ref_patches.contiguous().view(torch.int16).numpy().astype("<i2").tobytes()
    types = module.image_token_types(n_llm_h, n_llm_w).numpy().astype("<i8")
    return {
        "name": name,
        "source": source,
        "width": image.width,
        "height": image.height,
        "shape": list(ref_patches.shape),
        "vit_grid": [n_vit_h, n_vit_w],
        "llm_grid": [n_llm_h, n_llm_w],
        "num_tokens": int(module.num_image_tokens(n_llm_h, n_llm_w)),
        "fnv1a_types_i64": fnv1a(types.tobytes()),
        "fnv1a_f32": fnv1a(f32.astype("<f4").tobytes()),
        "fnv1a_bf16": fnv1a(bf16_bits),
    }


def convert_examples(checkpoint: Path) -> None:
    examples = checkpoint / "inference" / "examples" / "images"
    for stem, target in [("carrots", FILE_CASES[0][1]), ("corn", FILE_CASES[1][1])]:
        with Image.open(examples / f"{stem}.jpeg") as source:
            rgb = source.convert("RGB")
        rgb.save(IMAGES_DIR / target, format="PNG", optimize=True)
        print(f"converted {stem}.jpeg -> {target} ({rgb.width}x{rgb.height})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--convert-examples", action="store_true")
    opts = parser.parse_args()

    if opts.convert_examples:
        convert_examples(opts.checkpoint)

    module, reference_sha256 = load_reference(opts.checkpoint)
    args = vision_args(opts.checkpoint)

    cases = []
    for name, width, height, seed in SYNTHETIC_CASES:
        image = seeded_image(width, height, seed)
        cases.append(record_case(module, args, name, {"seed": seed}, image))
        print(f"{name}: {cases[-1]['num_tokens']} tokens, grid {cases[-1]['llm_grid']}")
    for name, file_name in FILE_CASES:
        path = IMAGES_DIR / file_name
        if not path.exists():
            raise SystemExit(f"missing fixture {path}; run with --convert-examples first")
        with Image.open(path) as source:
            image = source.convert("RGB")
        cases.append(record_case(module, args, name, {"file": file_name}, image))
        print(f"{name}: {cases[-1]['num_tokens']} tokens, grid {cases[-1]['llm_grid']}")

    document = {
        "generator": Path(__file__).name,
        "reference": "inference/image_processor.py of deepseek-ai/DeepSeek-V4.1-Flash",
        "reference_sha256": reference_sha256,
        "pillow": PIL.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "vision_config": vars(args),
        "cases": cases,
    }
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    opts.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"wrote {opts.output} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
