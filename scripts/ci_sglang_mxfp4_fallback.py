#!/usr/bin/env python3
"""Install and verify the fixed-image GPT-OSS gfx942 compatibility patch.

The two SDK sources are checked before installation and in both cohort reports.
Only the original GPT-OSS worker opts into their BF16/generic Triton path.
"""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

ROW_ENV = "E2E_SGLANG_GPTOSS_MXFP4_BF16"
FILES = {
    "srt/layers/quantization/__init__.py": (
        "13a9f194ce7693840529647e818fd784cbd221b7120ceafa98a986762a85512c",
        "70ae391b2635b04a613f5195169337d186c681467fb3b486dfc827224a98814e",
    ),
    "srt/layers/quantization/mxfp4.py": (
        "10692c68bc5e1d412fc4489d2c469e9916948d396f209b5ba6835c942676d9f9",
        "77170fb5097e5623276e8a70201bb39aea1b07fd078fb75a589173ccad2fa2ce",
    ),
}
PATCH = Path(__file__).with_name("patches") / "sglang-0.5.21-gptoss-gfx942.patch"
PATCH_SHA = "f28f934faca6c84a8f66d8e3c364047ba4f9859cdcef07f31bbb1ed54a069f50"


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
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise RuntimeError("Invalid SGLang SDK source: " + name)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def verify_installed(root=None):
    root = sdk_root() if root is None else root
    actual = source_hashes(root)
    expected = {name: pair[int(enabled())] for name, pair in FILES.items()}
    if actual != expected:
        raise RuntimeError("Unexpected or mixed SGLang MXFP4 SDK source pair: " + repr(actual))
    return {
        "state": "patched" if enabled() else "original",
        "patch_sha256": PATCH_SHA,
        "sdk_root": str(root),
        "source_sha256": actual,
    }


def install(root):
    if not enabled():
        return verify_installed(root)
    actual = source_hashes(root)
    original = {name: pair[0] for name, pair in FILES.items()}
    patched = {name: pair[1] for name, pair in FILES.items()}
    if actual == patched:
        return verify_installed(root)
    if actual != original:
        raise RuntimeError("Refuse to patch unknown or mixed SGLang SDK sources")
    if hashlib.sha256(PATCH.read_bytes()).hexdigest() != PATCH_SHA:
        raise RuntimeError("SGLang compatibility patch bytes changed")
    # Validate both outputs on disposable copies before any installed SDK write.
    with TemporaryDirectory(prefix="smg-sglang-mxfp4-") as tmp:
        stage = Path(tmp)
        for name in FILES:
            target = stage / "sglang" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        subprocess.run(
            [
                "patch",
                "--batch",
                "--forward",
                "--fuzz=0",
                "-p2",
                "-d",
                str(stage),
                "-i",
                str(PATCH.resolve()),
            ],
            check=True,
            capture_output=True,
            timeout=10,
        )
        if source_hashes(stage / "sglang") != patched:
            raise RuntimeError("SGLang compatibility patch output differs")
        for name in FILES:
            target = root / name
            temporary = None
            try:
                with NamedTemporaryFile(
                    dir=target.parent, prefix=".smg-mxfp4-", delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    os.fchmod(stream.fileno(), target.stat().st_mode & 0o777)
                    stream.write((stage / "sglang" / name).read_bytes())
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    return verify_installed(root)


if __name__ == "__main__":
    from importlib.metadata import version

    from ci_verify_sglang_rocm_cohort import IMAGE

    if os.environ.get("SMG_SGLANG_ROCM_IMAGE") != IMAGE or version("sglang") != "0.5.21":
        raise SystemExit("Unsupported SGLang image/version for the checked MXFP4 patch")
    print(json.dumps(install(sdk_root()), sort_keys=True))
