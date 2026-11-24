"""Fail closed on READ image/source mismatches; this does not qualify native ABI."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
from pathlib import Path

# Reviewed engine source: https://github.com/chunfangamd/vllm/commit/
# f9e4ca9653da9bd731c8b5bf83483cb2083c8dc9 (public fork, not upstream merged).
# Upstream baseline: 595b1514e837ba38df7ae51c411807783e227b78.
# Source/material receipts and bounded native validation: docker/moriio-read/.
# These hashes identify source bytes, not native ABI or SMG PD qualification.
_SOURCE_DIR = "distributed/kv_transfer/kv_connector/v1/moriio"
SOURCE_SHA256 = {
    f"{_SOURCE_DIR}/moriio_common.py": "61754babd34a560810106f1cdccde4e538f7bb4ab65e2c9c45937c75d414fdf9",
    f"{_SOURCE_DIR}/moriio_connector.py": "1f0a5d592b37840ebad629e5a279a310f307d4ccf6291578bca21822c49aace0",
}
OLD_IMAGE_DIGEST = "749f6f3f944f12af49966ac523c1f8573e4b229594b541954bd5f879b4496b1f"
COMPONENT = r"[a-z0-9]+(?:[._-][a-z0-9]+)*"
IMAGE = re.compile(
    rf"(?:[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?(?::[0-9]+)?/)?"
    rf"{COMPONENT}(?:/{COMPONENT})*(?::[A-Za-z0-9_][A-Za-z0-9_.-]{{0,127}})?"
    r"@sha256:[0-9a-f]{64}"
)


def read_image(value: str) -> str:
    if not IMAGE.fullmatch(value):
        raise ValueError("MoRI READ requires SMG_AMD_MORIIO_READ_IMAGE as a digest-pinned image")
    if value.rsplit(":", 1)[-1] == OLD_IMAGE_DIGEST:
        raise ValueError(
            f"vllm/vllm-openai-rocm:v0.31.0@sha256:{OLD_IMAGE_DIGEST} "
            "does not implement the reviewed ACK-fenced READ hold"
        )
    return value


def needs_read_preflight(env: dict[str, str]) -> bool:
    backend = (
        env.get("E2E_KV_BACKEND", "").strip().lower()
        or env.get("E2E_VLLM_KV_BACKEND", "nixl").lower()
    )
    extra = {part.strip().lower() for part in env.get("E2E_VLLM_EXTRA_KV_BACKENDS", "").split(",")}
    if backend != "moriio" and "moriio" not in extra:
        return False
    mode = env.get("E2E_VLLM_MORIIO_MODE", "").strip().lower() or "read"
    if mode not in ("read", "write"):
        raise ValueError("E2E_VLLM_MORIIO_MODE must be read or write")
    return mode == "read"


def verify_source_root(root: Path) -> dict[str, str]:
    observed = {}
    for relative, wanted in SOURCE_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"MoRI READ source is missing: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != wanted:
            raise ValueError(
                f"MoRI READ source does not match reviewed ACK-fenced hold: {relative}"
            )
        observed[relative] = actual
    return observed


def verify_installed() -> dict[str, str]:
    # A top-level spec lookup does not import vllm or any GPU dependencies.
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None or not spec.submodule_search_locations:
        raise ValueError("Installed vllm package is unavailable")
    locations = list(spec.submodule_search_locations)
    if len(locations) != 1:
        raise ValueError("Ambiguous effective vllm package roots")
    root = Path(locations[0]).resolve()
    distribution = importlib.metadata.distribution("vllm")
    installed = Path(distribution.locate_file("vllm")).resolve()
    if root != installed or Path(spec.origin).resolve() != installed / "__init__.py":
        raise ValueError("Effective vllm is shadowing its installed distribution")
    return verify_source_root(root)


# This is invocation intent from the successful source preflight, not a claim
# that GPU transfers, MR lifetime or the held-peer restart has already passed.
QUALIFICATION_ENV = "E2E_MORIIO_READ_QUALIFICATION"


def _qualification_lane(env: dict[str, str]) -> bool:
    backend = (
        env.get("E2E_KV_BACKEND", "").strip().lower()
        or env.get("E2E_VLLM_KV_BACKEND", "nixl").lower()
    )
    return (
        env.get("E2E_RUNTIME", "").strip().lower() == "vllm"
        and env.get("E2E_GPU_VENDOR", "").strip().lower() == "amd"
        and backend == "moriio"
        and (env.get("E2E_VLLM_MORIIO_MODE", "").strip().lower() or "read") == "read"
        and (env.get("E2E_VLLM_MORIIO_BACKEND", "").strip().lower() or "rdma") == "rdma"
    )


def _qualification_exports(
    image: str, observed: dict[str, str], env: dict[str, str]
) -> dict[str, str]:
    if not _qualification_lane(env):
        return {}
    image = read_image(image)
    if observed != SOURCE_SHA256:
        raise ValueError("First READ qualification requires the verified source cohort")
    receipt = {
        "purpose": "first-strict-rdma-read-qualification-attempt",
        "image": image,
        "source_sha256": observed,
    }
    return {
        "MORIIO_READ_IMAGE": image,
        "MORI_DISABLE_AUTO_XGMI": "1",
        QUALIFICATION_ENV: json.dumps(receipt, sort_keys=True, separators=(",", ":")),
    }


def qualification_attempt(env: dict[str, str]) -> bool:
    raw = env.get(QUALIFICATION_ENV, "")
    if not raw or not _qualification_lane(env) or env.get("MORI_DISABLE_AUTO_XGMI") != "1":
        return False
    # A bare flag is insufficient: match the captured image and recheck the
    # effective installed source, including the existing anti-shadow check.
    receipt = json.loads(raw)
    expected = {
        "purpose": "first-strict-rdma-read-qualification-attempt",
        "image": read_image(env.get("MORIIO_READ_IMAGE", "")),
        "source_sha256": SOURCE_SHA256,
    }
    if receipt != expected or verify_installed() != SOURCE_SHA256:
        raise ValueError("First READ qualification receipt/source mismatch")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("image", "verify"))
    parser.add_argument("--image", required=True)
    parser.add_argument("--qualification-env", type=Path)
    args = parser.parse_args()
    if args.qualification_env and args.command != "verify":
        parser.error("--qualification-env requires verify")
    try:
        if args.qualification_env:
            # Clear inherited intent even when this invocation fails or is not
            # applicable. Actions reads this file after the step completes.
            with args.qualification_env.open("a") as output:
                output.write(f"{QUALIFICATION_ENV}=\n")
        if args.command == "image":
            print(read_image(args.image))
        elif needs_read_preflight(dict(os.environ)):
            read_image(args.image)
            observed = verify_installed()
            if args.qualification_env:
                exports = _qualification_exports(args.image, observed, dict(os.environ))
                with args.qualification_env.open("a") as output:
                    for name, value in exports.items():
                        output.write(f"{name}={value}\n")
            print(
                json.dumps(
                    {
                        "reviewed_READ_source_sha256": observed,
                        "scope": "Source cohort only; not native ABI or GPU qualification",
                    }
                )
            )
        else:
            print("MoRI READ source preflight does not apply to this backend/mode")
    except (ValueError, OSError, importlib.metadata.PackageNotFoundError) as error:
        parser.exit(1, f"ERROR: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
