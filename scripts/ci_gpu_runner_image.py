#!/usr/bin/env python3
"""Identify prepared GPU runner/carrier images and distinguish missing manifests from errors."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

DEFAULT_CARRIER_BASE = (
    "ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55"
)

COMMON_INPUTS = (
    "docker/ci-gpu-runner.Dockerfile",
    "scripts/ci_retry.sh",
    "scripts/ci_apt_mirror.sh",
    "scripts/ci_setup_python_venv.sh",
    "scripts/ci_prepared_backend_env.sh",
    "scripts/ci_ensure_python_headers.sh",
)

BACKEND_INPUTS = {
    "vllm": COMMON_INPUTS
    + (
        "scripts/ci_install_vllm.sh",
        "scripts/ci_install_flashinfer_jit_cache.sh",
    ),
    "sglang": COMMON_INPUTS + ("scripts/ci_install_sglang.sh",),
    "trtllm": COMMON_INPUTS + ("scripts/ci_install_trtllm.sh",),
}

BUILD_INPUTS = COMMON_INPUTS + (
    "scripts/ci_install_vllm.sh",
    "scripts/ci_install_sglang.sh",
    "scripts/ci_install_trtllm.sh",
    "scripts/ci_install_flashinfer_jit_cache.sh",
)

DIGEST_REF_RE = re.compile(r"[A-Za-z0-9._:/-]+@sha256:[0-9a-f]{64}")


def image_tag(
    root: Path,
    base_image: str,
    driver: str,
    force_run: str = "",
    backend: str = "all",
    carrier_base_image: str = DEFAULT_CARRIER_BASE,
) -> str:
    """Compute a deterministic content-addressed GHCR tag for a prepared GPU image.

    When ``backend == "all"`` (the combined self-hosted runner image), hashes
    ``base_image``, ``carrier_base_image``, ``driver``, and every file in
    ``BUILD_INPUTS`` under the ``ci-gpu-runner-<sha256>`` prefix. When
    ``backend`` names a single carrier (``vllm``, ``sglang``, or ``trtllm``),
    hashes only the inputs affecting that carrier under
    ``ci-gpu-backend-<backend>-<sha256>``.
    """
    if backend not in ("all", *BACKEND_INPUTS):
        raise ValueError(f"Unsupported backend: {backend}")
    if not DIGEST_REF_RE.fullmatch(base_image):
        raise ValueError("BASE_IMAGE must be a digest-pinned image reference")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){0,2}", driver):
        raise ValueError("UV_CUDA_DRIVER_VERSION must be a target driver version")
    if force_run and not re.fullmatch(r"[0-9]+-[0-9]+", force_run):
        raise ValueError("forced build identity must be <run-id>-<run-attempt>")

    if backend == "all":
        if not DIGEST_REF_RE.fullmatch(carrier_base_image):
            raise ValueError("CARRIER_BASE_IMAGE must be a digest-pinned image reference")
        inputs = BUILD_INPUTS
        seed: list[str] = [base_image, carrier_base_image, driver]
        prefix = "ci-gpu-runner"
    else:
        inputs = BACKEND_INPUTS[backend]
        # Only vLLM resolves PyTorch wheels against UV_CUDA_DRIVER_VERSION.
        seed = [backend, base_image, driver] if backend == "vllm" else [backend, base_image]
        prefix = f"ci-gpu-backend-{backend}"

    digest = hashlib.sha256(json.dumps(seed).encode())
    for name in inputs:
        digest.update(name.encode() + b"\0")
        digest.update((root / name).read_bytes() + b"\0")
    tag = f"{prefix}-{digest.hexdigest()}"
    return f"{tag}-run-{force_run}" if force_run else tag


def image_exists(image: str) -> bool:
    """Return True if ``image`` exists in the registry, False only on missing manifest."""
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
    """CLI entry point for computing image tags and checking manifest existence."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    tag = commands.add_parser("tag")
    tag.add_argument("--base-image", required=True)
    tag.add_argument("--carrier-base-image", default=DEFAULT_CARRIER_BASE)
    tag.add_argument("--cuda-driver-version", required=True)
    tag.add_argument("--force-run", default="")
    tag.add_argument("--backend", default="all", choices=("all", *BACKEND_INPUTS))
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
                    args.backend,
                    args.carrier_base_image,
                )
            )
        else:
            print("true" if image_exists(args.image) else "false")
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
