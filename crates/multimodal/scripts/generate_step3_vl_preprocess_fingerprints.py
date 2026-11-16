#!/usr/bin/env python3
"""Record Step3-VL image-preprocessing fingerprints from the reference.

The reference is the serving engine's own processor module for the family,
``vllm/transformers_utils/processors/step3_vl.py`` (the checkpoints ship no
``preprocessor_config.json``; the engine builds ``Step3VLImageProcessor`` from
its constants): a square pad for a thin image, the 3024 px cap, Pillow BILINEAR
resizes, the sliding-window crops, then torchvision's ``ToTensor``,
``Normalize`` and antialiased BILINEAR resize of the global view to 728 x 728
and of every crop to 504 x 504. This script runs that module unmodified over
seeded synthetic images (the pattern the Rust test regenerates) and writes,
per image, the crop count, the newline flags, the token count, the tensor
shapes and FNV-1a fingerprints of the exact bytes:

* ``fnv1a_f32`` / ``fnv1a_bf16`` of ``pixel_values`` (the global view) as
  little-endian float32 and as the bfloat16 bits of the round-to-nearest-even
  cast the engine applies;
* ``patches_fnv1a_f32`` / ``patches_fnv1a_bf16`` of ``patch_pixel_values``
  (the crops, flat) the same way.

The module is loaded from a checkout of the engine's source tree; it imports
one engine type for an annotation, which is stubbed.

Usage:
    python crates/multimodal/scripts/generate_step3_vl_preprocess_fingerprints.py \
        --engine-tree /path/to/engine \
        [--output <fixtures>/golden/step3_vl_preprocess_fingerprints.json]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    REPO_ROOT / "crates/multimodal/tests/fixtures/golden/step3_vl_preprocess_fingerprints.json"
)
REFERENCE = "vllm/transformers_utils/processors/step3_vl.py"

# (name, width, height, seed): the Rust test rebuilds these with the same formula.
# Every branch of the reference's rule: no crops under a 728 px long side at a
# gentle aspect ratio and exactly at the 727/728 boundary, crops of the short
# side past an aspect ratio of 1.5, the 504 window from 728 px, a side shorter
# than the window (black past the edge), the short-side window past an aspect
# ratio of 4, the square pad under 32 px, and the 3024 px cap with truncation.
SYNTHETIC_CASES = [
    ("synthetic_64x48", 64, 48, 0),
    ("synthetic_100x100", 100, 100, 3),
    ("synthetic_640x480", 640, 480, 7),
    ("synthetic_727x500", 727, 500, 9),
    ("synthetic_728x728", 728, 728, 11),
    ("synthetic_1024x768", 1024, 768, 13),
    ("synthetic_1920x1080", 1920, 1080, 17),
    ("synthetic_600x300", 600, 300, 19),
    ("synthetic_1000x300", 1000, 300, 23),
    ("row-of-36-windows-3024x84", 3024, 84, 29),
    ("synthetic_40x700", 40, 700, 31),
    ("synthetic_4000x15", 4000, 15, 37),
    ("synthetic_3x900", 3, 900, 41),
    ("synthetic_3100x2000", 3100, 2000, 43),
]


def make_seeded_image(width: int, height: int, seed: int) -> Image.Image:
    """The crate's seeded pattern: R=(x*7+y*3)%256, G=(x*5+y*11)%256, B=(x+y*2)%256, plus the seed."""
    xs = np.arange(width, dtype=np.uint32)[None, :]
    ys = np.arange(height, dtype=np.uint32)[:, None]
    r = ((xs * 7 + ys * 3) % 256 + seed) % 256
    g = ((xs * 5 + ys * 11) % 256 + seed) % 256
    b = ((xs + ys * 2) % 256 + seed) % 256
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def fnv1a(data: bytes) -> str:
    h = 0xCBF29CE484222325
    for byte in data:
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def fnv1a_f32(tensor: torch.Tensor) -> str:
    return fnv1a(tensor.detach().contiguous().to(torch.float32).numpy().astype("<f4").tobytes())


def fnv1a_bf16(tensor: torch.Tensor) -> str:
    bits = tensor.detach().contiguous().to(torch.float32).to(torch.bfloat16).view(torch.int16)
    return fnv1a(bits.numpy().astype("<i2").tobytes())


def load_reference(tree: Path):
    path = tree / REFERENCE
    source = path.read_bytes()
    for name in ("vllm", "vllm.tokenizers"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["vllm.tokenizers"].TokenizerLike = object
    spec = importlib.util.spec_from_file_location("reference_step3_vl", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, hashlib.sha256(source).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--engine-tree", required=True, type=Path, help="a checkout of the engine's source tree"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    torch.set_num_threads(1)
    module, reference_sha256 = load_reference(args.engine_tree)
    processor = module.Step3VLImageProcessor()
    cases = []
    for name, width, height, seed in SYNTHETIC_CASES:
        image = make_seeded_image(width, height, seed)
        out = processor(images=[image], return_tensors="pt")
        pixel_values = out["pixel_values"]
        patches = out["patch_pixel_values"]
        num_patches = int(out["num_patches"][0])
        newline_mask = [int(flag) for flag in out["patch_newline_mask"].tolist()]
        case = {
            "name": name,
            "source": {"seed": seed},
            "width": width,
            "height": height,
            "num_patches": num_patches,
            "newline_mask": newline_mask,
            "num_tokens": processor.get_num_image_tokens(width, height),
            "shape": list(pixel_values.shape),
            "patches_shape": list(patches.shape),
            "fnv1a_f32": fnv1a_f32(pixel_values),
            "fnv1a_bf16": fnv1a_bf16(pixel_values),
            "patches_fnv1a_f32": fnv1a_f32(patches),
            "patches_fnv1a_bf16": fnv1a_bf16(patches),
        }
        cases.append(case)
        print(
            f"{name}: crops {num_patches}, tokens {case['num_tokens']}, {case['fnv1a_f32']}",
            file=sys.stderr,
        )

    document = {
        "reference": REFERENCE,
        "reference_sha256": reference_sha256,
        "torch": torch.__version__,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(cases)} cases to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
