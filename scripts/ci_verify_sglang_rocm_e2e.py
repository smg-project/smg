#!/usr/bin/env python3
"""Preserve the Torch cohort and verify editable/generated RPC origins."""

import argparse
import hashlib
import importlib
import importlib.metadata as md
import json
from pathlib import Path

CONSTRAINT = Path("/etc/sglang/constraints/torch-rocm.txt")
TORCH_PACKAGES = ("torch", "torchaudio", "torchvision")


def torch_snapshot():
    return {
        "packages": {
            name: {"version": md.version(name), "requires": md.requires(name)}
            for name in TORCH_PACKAGES
        },
        "constraint_sha256": hashlib.sha256(CONSTRAINT.read_bytes()).hexdigest(),
    }


def verify_torch_snapshot(before):
    if before != torch_snapshot():
        raise RuntimeError("ROCm Torch packages or constraints changed during setup")


def verify_rpc_sources(root):
    root = Path(root).resolve()
    generated = root / "crates/grpc_client/python/smg_grpc_proto/generated"
    expected = {
        "smg_grpc_proto": generated.parent / "__init__.py",
        "smg_grpc_servicer": root / "grpc_servicer/smg_grpc_servicer/__init__.py",
        "sglang": Path("/sgl-workspace/sglang/python/sglang/__init__.py"),
    }
    sources = {}
    for name, path in expected.items():
        actual = Path(importlib.import_module(name).__file__).resolve()
        if actual != path.resolve():
            raise RuntimeError(f"unexpected source origin for {name}: {actual}; expected {path}")
        sources[name] = str(actual)
    proto_root = root / "crates/grpc_client/proto"
    protos = sorted(proto_root.rglob("*.proto"))
    if not protos:
        raise RuntimeError("No repository RPC proto files found")
    companions = {}
    for proto in protos:
        if proto.is_symlink() or not proto.is_file():
            raise RuntimeError(f"unexpected proto source: {proto}")
        relative = proto.relative_to(proto_root).with_suffix("")
        for suffix in ("_pb2", "_pb2_grpc"):
            name = "smg_grpc_proto.generated." + ".".join(relative.parts) + suffix
            path = generated / relative.parent / (relative.name + suffix + ".py")
            actual = Path(importlib.import_module(name).__file__).resolve()
            if actual != path.resolve():
                raise RuntimeError(
                    f"unexpected generated origin for {name}: {actual}; expected {path}"
                )
            companions[name] = str(actual)
    return {
        "editable_sources": sources,
        "generated_modules": companions,
        "GPU_or_model_accepted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--snapshot", type=Path)
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.snapshot:
        args.snapshot.write_text(json.dumps(torch_snapshot()) + "\n")
    else:
        verify_torch_snapshot(json.loads(args.verify.read_text()))
        print(json.dumps(verify_rpc_sources(args.root), indent=2))


if __name__ == "__main__":
    main()
