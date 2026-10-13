#!/usr/bin/env python3
"""Generate the smart_resize golden: the resize target the reference image
processors pick for a set of input shapes, including the degenerate ones (a
side under half the patch factor), where the order of the rounding, the
floor to one factor and the pixel-budget test decides the grid.

The Qwen families follow transformers' Qwen2-VL image processor (the Qwen3-VL
and Qwen3-Omni image paths resolve to the same class): each side rounds to a
multiple of the factor without a floor, so a thin image falls under
``min_pixels`` and both sides scale up from the original dimensions. The
MiniMax-M3 image processor (the hub's ``image_processor.py`` at the pinned
revision, passed with ``--minimax-m3-image-processor``) floors each rounded
side to one factor first. Both are recorded with their provenance.

Output: ``tests/fixtures/golden/qwen_smart_resize.json``, read by
``tests/qwen_smart_resize_golden.rs``.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from transformers import __version__ as transformers_version
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize as qwen_smart_resize

GOLDEN = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "golden" / "qwen_smart_resize.json"
)

# (height, width): the degenerate shapes first (a side under half the factor
# with the other side long), then the shapes around the factor and ordinary
# photographs, then two aspect ratios the reference refuses.
SHAPES = (
    (1, 100),
    (100, 1),
    (1, 150),
    (1, 199),
    (199, 1),
    (2, 300),
    (10, 100),
    (100, 10),
    (10, 1900),
    (12, 2400),
    (15, 2050),
    (16, 2048),
    (6000, 31),
    (9, 7),
    (13, 14),
    (27, 27),
    (28, 28),
    (100, 100),
    (400, 800),
    (500, 500),
    (640, 480),
    (1080, 1920),
    (3000, 3000),
    (4000, 3000),
    (100, 30000),
    (13, 3000),
)

# patch_size, merge_size, min_pixels, max_pixels: the public Qwen2-VL and
# Qwen3-VL preprocessor configurations and MiniMax-M3's defaults.
FAMILIES = {
    "qwen2_vl": (14, 2, 3136, 12_845_056),
    "qwen3_vl": (16, 2, 65_536, 16_777_216),
    "minimax_m3": (14, 2, 3136, 451_584),
}


def cases(resize, factor: int, min_pixels: int, max_pixels: int) -> list[dict]:
    out = []
    for height, width in SHAPES:
        case = {"height": height, "width": width}
        try:
            resized = resize(
                height, width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
            )
            case["resized"] = [int(resized[0]), int(resized[1])]
        except ValueError as error:
            case["error"] = str(error).split(",")[0]
        out.append(case)
    return out


def load_minimax_smart_resize(path: Path):
    spec = importlib.util.spec_from_file_location("minimax_m3_image_processor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.smart_resize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--minimax-m3-image-processor",
        type=Path,
        required=True,
        help="the hub's image_processor.py of MiniMax-M3 at the pinned revision",
    )
    args = parser.parse_args()
    minimax_smart_resize = load_minimax_smart_resize(args.minimax_m3_image_processor)
    families = {}
    for name, (patch, merge, min_pixels, max_pixels) in FAMILIES.items():
        resize = minimax_smart_resize if name == "minimax_m3" else qwen_smart_resize
        families[name] = {
            "patch_size": patch,
            "merge_size": merge,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "cases": cases(resize, patch * merge, min_pixels, max_pixels),
        }
    document = {
        "generator": "crates/multimodal/scripts/generate_qwen_smart_resize_golden.py",
        "transformers": transformers_version,
        "minimax_m3_image_processor_sha256": hashlib.sha256(
            args.minimax_m3_image_processor.read_bytes()
        ).hexdigest(),
        "families": families,
    }
    GOLDEN.write_text(json.dumps(document, indent=2) + "\n")
    for name, family in families.items():
        resized = sum("resized" in case for case in family["cases"])
        print(f"{name}: {resized} shapes resized, {len(family['cases']) - resized} refused")


if __name__ == "__main__":
    main()
