#!/usr/bin/env python3
"""Deliver ordinary SGLang provider data; does not prepare or qualify a runtime."""

import argparse
import ast
import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKET = Path("scripts/ci_data/sglang-llama-provider.json")
DELIVERED = Path("e2e-logs/ordinary-provider/provider.json")
IMAGE = "lmsysorg/sglang:v0.5.21-rocm10-mi30x@sha256:4420a7e492fce623285c635b9a6a7e09ccf77a3c812006494e4414655d46662d"
LIMIT = 64 * 1024


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*arguments):
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, timeout=15
    ).stdout


def checkout(expected):
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise ValueError("Exact GitHub checkout SHA required")
    if git("rev-parse", "HEAD").decode().strip() != expected:
        raise ValueError("Producer/consumer checkout differs from GitHub SHA")


def read_packet(path, expected=None):
    path = Path(path).absolute()
    if path != path.resolve():
        raise ValueError("Aliased provider data path")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > LIMIT:
            raise ValueError("Bounded regular provider data required")
        body = stream.read(LIMIT + 1)
    if len(body) > LIMIT or len(body) != info.st_size:
        raise ValueError("Provider data size changed")
    actual = hashlib.sha256(body).hexdigest()
    if expected is not None and actual != expected:
        raise ValueError("Provider data digest differs")
    return json.loads(body), body, actual


def source_contract(packet):
    if packet.get("schema") != "smg-ordinary-sg-provider-v1" or packet.get("image") != IMAGE:
        raise ValueError("Ordinary fixed-image provider data required")
    tree = ast.parse((ROOT / "scripts/ci_sglang_llama_cohort.py").read_text())
    names = ast.literal_eval(
        next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "SOURCES" for t in node.targets)
        )
    )
    expected = {name: digest(ROOT / "scripts" / name) for name in names}
    if packet.get("source_sha256") != expected or packet.get("workflow_sha256") != digest(
        ROOT / ".github/workflows/e2e-gpu-job.yml"
    ):
        raise ValueError("Provider data does not match this checkout Source")


def bindings(path, sha256, artifact_id):
    if Path(path) != DELIVERED or str(path) != DELIVERED.as_posix():
        raise ValueError("Only the ordinary same-run artifact path is supported")
    if re.fullmatch(r"[0-9a-f]{64}", sha256 or "") is None:
        raise ValueError("Exact provider data digest required")
    if re.fullmatch(r"[1-9][0-9]*", artifact_id or "") is None:
        raise ValueError("Immutable same-run provider artifact ID required")
    if (
        os.environ.get("SMG_GPTOSS_COHORT_MANIFEST")
        or os.environ.get("E2E_SGLANG_GPTOSS_MXFP4_BF16", "0") != "0"
        or os.environ.get("SGLANG_GPTOSS_MXFP4_BF16_FALLBACK", "0") != "0"
    ):
        raise ValueError("Ordinary and GPT-OSS provider selections are exclusive")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("publish", "bindings", "read"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--artifact-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    checkout(args.checkout)
    if args.phase != "publish":
        bindings(args.input, args.sha256, args.artifact_id)
        if args.phase == "bindings":
            return 0
    if args.output is None:
        parser.error("--output is required for publish/read")
    packet, body, sha256 = read_packet(args.input, args.sha256)
    source_contract(packet)
    # Both ends compare the bytes with the immutable same-checkout Git object.
    # A matching filesystem path or HEAD metadata alone is insufficient.
    if body != git("show", args.checkout + ":" + PACKET.as_posix()):
        raise ValueError("Provider data differs from the tracked producer blob")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.phase == "publish":
        if args.input.absolute() != ROOT / PACKET:
            raise ValueError("Tracked same-checkout provider data required")
        with args.output.open("xb") as stream:
            stream.write(body)
        args.output.chmod(0o444)
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
            stream.write("provider_sha256=" + sha256 + "\n")
    else:
        args.input.chmod(0o444)
        with args.output.open("x") as stream:
            json.dump(
                {
                    "provider_sha256": sha256,
                    "producer_checkout": args.checkout,
                    "same_run_artifact_id": args.artifact_id,
                    "workflow_sha256": packet["workflow_sha256"],
                    "SDK_native_runtime_qualified": False,
                },
                stream,
            )
            stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
