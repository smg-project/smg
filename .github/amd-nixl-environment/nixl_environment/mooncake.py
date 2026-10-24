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
from . import private_wheel

ENGINE = {
    "bytes": 3740816,
    "sha256": "710c03c60a8937a5a0bfbee14062c565715cd0bd886c0c4ca4c82c30b0325bc3",
}


def require_engine_mapped(children):
    private_wheel.require_mapped(children)


def capture(out):
    import torch
    from vllm.platforms import current_platform

    require(torch.version.hip and current_platform.is_rocm(), "ROCm runtime required")
    config, target = private_wheel.verify("canary")
    dist = distribution("mooncake-transfer-engine-rocm")
    require(dist.version == config["version"], "private Mooncake version differs")
    record = dist.read_text("RECORD")
    metadata = dist.read_text("METADATA")
    require(record and metadata, "Mooncake distribution evidence missing")
    (out / "mooncake.RECORD").write_text(record)
    (out / "mooncake.METADATA").write_text(metadata)
    relative = "mooncake/engine.cpython-312-x86_64-linux-gnu.so"
    path = Path(dist.locate_file(relative)).resolve()
    require(path == target / relative, "private Mooncake engine origin differs")
    require(path.stat().st_size == ENGINE["bytes"], "Mooncake engine size differs")
    raw = path.read_bytes()
    identity = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    require(identity == ENGINE, "Mooncake engine differs from captured749f bytes")
    rows = [row for row in csv.reader(io.StringIO(record)) if row[0] == relative]
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
