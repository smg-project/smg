"""Observe the fixed-image SDK source pair for the shared native verifier.

A5 supplies source_hashes/verify_installed only. The opted-in SDK patch installer
and its pinned patch payload belong to the separate specialized SDK change.
This observer never edits SDK files or executes an installation CLI.
"""

import hashlib
import importlib.util
import os
from pathlib import Path

ROW_ENV = "E2E_SGLANG_GPTOSS_MXFP4_BF16"
IMAGE = "lmsysorg/sglang:v0.5.21-rocm10-mi30x@sha256:4420a7e492fce623285c635b9a6a7e09ccf77a3c812006494e4414655d46662d"
FILES = {
    "srt/layers/quantization/__init__.py": (
        "13a9f194ce7693840529647e818fd784cbd221b7120ceafa98a986762a85512c",
        "70ae391b2635b04a613f5195169337d186c681467fb3b486dfc827224a98814e",
    ),
    "srt/layers/quantization/mxfp4.py": (
        "10692c68bc5e1d412fc4489d2c469e9916948d396f209b5ba6835c942676d9f9",
        "b0f90a0b1e539657e5d23ac4893550975136e5c46d7deb2e07e724f7d94f9e5c",
    ),
}
PATCH_SHA = "db59b663d2e4a0f87f6bd7f788ade509a60780bfa0d756b5c63eddb77fe4422e"


def enabled():
    value = os.environ.get(ROW_ENV, "0")
    if value not in ("0", "1"):
        raise RuntimeError(ROW_ENV + " must be 0 or 1")
    return value == "1"


def sdk_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        raise RuntimeError("SGLang package location is unavailable")
    return Path(spec.origin).parent.resolve()


def source_hashes(root):
    hashes = {}
    for name in FILES:
        path = root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or (not path.resolve().is_relative_to(root.resolve()))
        ):
            raise RuntimeError("Invalid SGLang SDK source: " + name)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def verify_installed(root=None):
    root = sdk_root() if root is None else root
    actual = source_hashes(root)
    expected = {name: pair[int(enabled())] for (name, pair) in FILES.items()}
    if actual != expected:
        raise RuntimeError("Unexpected or mixed SGLang MXFP4 SDK source pair: " + repr(actual))
    return {
        "state": "patched" if enabled() else "original",
        "patch_sha256": PATCH_SHA,
        "sdk_root": str(root),
        "source_sha256": actual,
    }


if __name__ == "__main__":
    raise SystemExit(
        "SDK source observer has no install CLI; use the separate specialized SDK producer"
    )
