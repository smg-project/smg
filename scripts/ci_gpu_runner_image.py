#!/usr/bin/env python3
"""Identify prepared runner images and distinguish missing manifests from errors."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

BUILD_INPUTS = (
    "docker/ci-gpu-runner.Dockerfile",
    "scripts/ci_retry.sh",
    "scripts/ci_apt_mirror.sh",
    "scripts/ci_setup_python_venv.sh",
    "scripts/ci_prepared_backend_env.sh",
    "scripts/ci_ensure_python_headers.sh",
    "scripts/ci_install_vllm.sh",
    "scripts/ci_install_sglang.sh",
    "scripts/ci_install_flashinfer_jit_cache.sh",
)


def image_tag(root: Path, base_image: str, driver: str, force_run: str = "") -> str:
    if not re.fullmatch(r"[A-Za-z0-9._:/-]+@sha256:[0-9a-f]{64}", base_image):
        raise ValueError("BASE_IMAGE must be a digest-pinned image reference")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){0,2}", driver):
        raise ValueError("UV_CUDA_DRIVER_VERSION must be a target driver version")
    if force_run and not re.fullmatch(r"[0-9]+-[0-9]+", force_run):
        raise ValueError("forced build identity must be <run-id>-<run-attempt>")
    digest = hashlib.sha256(json.dumps([base_image, driver]).encode())
    for name in BUILD_INPUTS:
        digest.update(name.encode() + b"\0")
        digest.update((root / name).read_bytes() + b"\0")
    tag = f"ci-gpu-runner-{digest.hexdigest()}"
    return f"{tag}-run-{force_run}" if force_run else tag


def image_exists(image: str) -> bool:
    result = subprocess.run(
        ["docker", "manifest", "inspect", image], capture_output=True, text=True, timeout=60
    )
    if result.returncode == 0:
        return True
    error = result.stderr.strip()
    if error == f"no such manifest: {image}" or re.fullmatch(
        r"manifest unknown(?::[^\n]*)?", error, re.IGNORECASE
    ):
        return False
    raise RuntimeError(f"Cannot check image availability: {error or result.stdout.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    tag = commands.add_parser("tag")
    tag.add_argument("--base-image", required=True)
    tag.add_argument("--cuda-driver-version", required=True)
    tag.add_argument("--force-run", default="")
    exists = commands.add_parser("exists")
    exists.add_argument("image")
    args = parser.parse_args()
    try:
        if args.command == "tag":
            print(
                image_tag(
                    Path(__file__).resolve().parents[1],
                    args.base_image,
                    args.cuda_driver_version,
                    args.force_run,
                )
            )
        else:
            print("true" if image_exists(args.image) else "false")
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
