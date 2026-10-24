"""Mooncake ROCm import identity; KV transfer is proved by original cases/logs."""

import base64
import csv
import hashlib
import importlib
import io
import json
from importlib.metadata import distribution
from pathlib import Path

from .owned import require

ENGINE = {
    "bytes": 4826641,
    "sha256": "18e5b44278f92388e0465398fe490619342646e93cc80f1d07fdbbb7f388134d",
}


def require_engine_mapped(children):
    require(
        any(
            value == ENGINE
            for child in children
            for path, value in child["loaded_native"].items()
            if path.endswith("/mooncake/engine.so")
        ),
        "owned Mooncake role lacks the accepted actual ROCm engine mapping",
    )


def capture(out):
    import torch
    from vllm.platforms import current_platform

    require(torch.version.hip and current_platform.is_rocm(), "ROCm runtime required")
    dist = distribution("mooncake-transfer-engine-rocm")
    require(dist.version == "0.3.13.post1", "Mooncake ROCm version differs")
    record = dist.read_text("RECORD")
    metadata = dist.read_text("METADATA")
    require(record and metadata, "Mooncake distribution evidence missing")
    (out / "mooncake.RECORD").write_text(record)
    (out / "mooncake.METADATA").write_text(metadata)
    path = Path(dist.locate_file("mooncake/engine.so")).resolve()
    require(path.stat().st_size == ENGINE["bytes"], "Mooncake engine size differs")
    raw = path.read_bytes()
    identity = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    require(identity == ENGINE, "Mooncake engine differs from captured749f bytes")
    rows = [row for row in csv.reader(io.StringIO(record)) if row[0] == "mooncake/engine.so"]
    digest = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
    require(
        len(rows) == 1 and rows[0][1:] == [digest, str(len(raw))], "Mooncake engine RECORD differs"
    )
    (out / "mooncake-source-before-native.json").write_text(
        json.dumps(
            {
                "distribution": str(dist._path),
                "version": dist.version,
                "engine": str(path),
                **identity,
                "ABI_or_transfer_proven": False,
            },
            indent=2,
        )
        + "\n"
    )
    module = importlib.import_module("mooncake.engine")
    require(Path(module.__file__).resolve() == path, "loaded Mooncake engine origin differs")
    require(
        callable(getattr(module, "TransferEngine", None)), "Mooncake TransferEngine unavailable"
    )
    maps = Path("/proc/self/maps").read_text()
    (out / "mooncake-native.maps").write_text(maps)
    require(
        any(
            line.split(maxsplit=5)[-1] == str(path) and "x" in line.split()[1]
            for line in maps.splitlines()
        ),
        "Mooncake extension is not actually mapped",
    )
    (out / "mooncake-canary.json").write_text(
        json.dumps(
            {
                "version": dist.version,
                "module": module.__name__,
                "engine": str(path),
                **identity,
                "native_import_succeeded": True,
                "agent_created": False,
                "KV_transfer_proven": False,
            },
            indent=2,
        )
        + "\n"
    )
