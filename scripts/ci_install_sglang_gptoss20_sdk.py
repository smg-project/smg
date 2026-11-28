#!/usr/bin/env python3
"""Install the original pinned 20B SDK patch only in its active owned cohort.

The shared A5 SDK observer remains read-only. Native package installation,
prepared/final native checks and the original numerical probe remain separate.
This BF16 compatibility route does not establish MXFP4 accuracy or performance.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from ci_sglang_mxfp4_fallback import (
    FILES,
    IMAGE,
    PATCH_SHA,
    enabled,
    sdk_root,
    source_hashes,
    verify_installed,
)

PATCH = Path(__file__).with_name("patches") / "sglang-0.5.21-gptoss-gfx942.patch"


def require_profile():
    if (
        os.environ.get("E2E_ENGINE") != "sglang"
        or os.environ.get("E2E_GPU_VENDOR") != "amd"
        or not enabled()
        or os.environ.get("SMG_SGLANG_ROCM_IMAGE") != IMAGE
    ):
        raise RuntimeError("Explicit AMD SGLang fixed-image 20B profile required")


def owned_sdk_root():
    """Require the exact active private SDK origin before any SDK write."""
    require_profile()
    boot = sys.modules.get("ci_sglang_cohort")
    if boot is None or sys.modules.get("sitecustomize") is not boot:
        raise RuntimeError("Active owned cohort is required before SDK installation")
    prefix = Path(sys.prefix).resolve()
    manifest = Path(boot.__file__).with_name("smg-cohort.json")
    body = manifest.read_bytes()
    spec = json.loads(body)
    if (
        not manifest.resolve().is_relative_to(prefix)
        or hashlib.sha256(body).hexdigest() != boot.SMG_MANIFEST_SHA256
        or spec.get("interpreter_prefix") != str(prefix)
        or not spec.get("isolated_children")
    ):
        raise RuntimeError("Active private cohort manifest differs")
    source = Path(boot.SMG_COHORT["sglang"]["module_path"])
    root = sdk_root()
    if (
        source != source.resolve()
        or root != source.parent
        or root != prefix.parent / "sdk/sglang"
        or root == Path("/sgl-workspace/sglang/python/sglang")
    ):
        raise RuntimeError("SDK origin must be the active cohort's owned staged copy")
    return root


def install(root):
    if root != owned_sdk_root():
        raise RuntimeError("Foreign SDK install target")
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


def main(argv=None):
    """Export a fresh prepared/final source report without overwriting evidence."""
    from importlib.metadata import version

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--phase", choices=("prepared", "final"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.phase and not args.verify_only:
        parser.error("--phase requires --verify-only")
    # Reserve evidence before touching the SDK. An existing report is an error,
    # including a symlink, and cannot cause a second installation attempt.
    report = args.output.open("x", encoding="utf-8") if args.output else None
    record = {
        "status": "failed",
        "phase": args.phase or ("verify" if args.verify_only else "install"),
        "scope": "SDK source pair only; not numerical/native/model acceptance",
        "image": os.environ.get("SMG_SGLANG_ROCM_IMAGE"),
        "patch_sha256": PATCH_SHA,
    }
    try:
        require_profile()
        record["sglang_version"] = version("sglang")
        if record["sglang_version"] != "0.5.21":
            raise RuntimeError("Unsupported SGLang version for the checked MXFP4 patch")
        root = owned_sdk_root()
        record["sdk"] = verify_installed(root) if args.verify_only else install(root)
        record["status"] = "pass"
        return 0
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)[:2048]}
        if isinstance(error, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
            for stream in ("stdout", "stderr"):
                captured = getattr(error, stream, None)
                if captured is not None:
                    if isinstance(captured, bytes):
                        captured = captured.decode(errors="replace")
                    record["error"][stream] = captured[-1024:]
        return 1
    finally:
        body = json.dumps(record, sort_keys=True) + "\n"
        if report is not None:
            try:
                report.write(body)
            finally:
                report.close()
        print(body, end="")


if __name__ == "__main__":
    raise SystemExit(main())
