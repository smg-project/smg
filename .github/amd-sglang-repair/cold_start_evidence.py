"""Read only this job's fixed AITER cache and worker process group."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

TARGET = "mha_batch_prefill_bf16_nlogits_nbias_mask_nlse_ndropout_nqscale_nsink"
AITER = Path("/sgl-workspace/aiter")
SOURCE_PINS = {
    "aiter/jit/core.py": "ec2e1e51ddace53ec7999515bf8753fa76369915b29e834d81923916bec49e4f",
    "aiter/jit/utils/file_baton.py": "0483680d78c77caf3d69babbdb51a70019db8b624c44df75037025c63e4ab188",
    "aiter/jit/utils/cpp_extension.py": "88bc7ddf34a6c1294b3a39b6a957b4ea17c6f837ed53fd89ef581785f49bd9bb",
    "aiter/jit/optCompilerConfig.json": "c3d9d1093bd6489f0c85be9882b1d73207cb84c631a9628393dae996c688072e",
}
LIMIT = 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    require(path.stat().st_size <= 2 * 1024**3, "unexpected native cache size")
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(LIMIT), b""):
            value.update(block)
    return value.hexdigest()


class Collector:
    def __init__(self, output, root=AITER):
        self.output = output
        self.root = root.resolve()
        self.jit = self.root / "aiter/jit"
        self.output.mkdir()
        self.snapshots = 0

    def read(self, path):
        require(path.resolve().is_relative_to(self.root), "cache path escape")
        if not path.exists():
            return {"path": str(path), "exists": False}
        try:
            stat = path.stat()
            with path.open("rb") as source:
                raw = source.read(LIMIT + 1)
        except FileNotFoundError:
            return {"path": str(path), "exists": False, "raced": True}
        clipped = len(raw) > LIMIT
        raw = raw[:LIMIT]
        name = hashlib.sha256(raw).hexdigest()
        target = self.output / (name + ".bytes")
        if not target.exists():
            target.write_bytes(raw)
        return {
            "path": str(path),
            "exists": True,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "captured_bytes": len(raw),
            "sha256_of_captured_bytes": name,
            "truncated": clipped,
            "saved": target.name,
        }

    def sources(self):
        require(os.access(self.jit, os.W_OK), "unexpected fallback JIT cache")
        rows = {name: self.read(self.root / name) for name in SOURCE_PINS}
        (self.output / "source-identity.json").write_text(json.dumps(rows, indent=2) + "\n")
        for name, row in rows.items():
            require(
                row["exists"]
                and not row["truncated"]
                and row["sha256_of_captured_bytes"] == SOURCE_PINS[name],
                "fixed AITER source mismatch: " + name,
            )
        return rows

    def snapshot(self, label, worker):
        self.snapshots += 1
        require(self.snapshots <= 4, "cold-start evidence sample limit")
        build = self.jit / "build" / TARGET / "build"
        files = [
            self.jit / "build" / ("lock_" + TARGET),
            build / "lock",
            build / "build.ninja",
            build / ".ninja_log",
        ]
        rows = [self.read(path) for path in files]
        process = getattr(worker, "process", None)
        pgid = process.pid if process is not None else None
        listing = subprocess.run(
            ["ps", "-eo", "pid,ppid,pgid,stat,etimes,time,args"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        processes = []
        for line in listing.stdout.splitlines()[1:]:
            parts = line.split(None, 6)
            if len(parts) == 7 and pgid is not None and parts[2] == str(pgid):
                processes.append(line)
        objects = []
        for path in sorted(build.glob("*.o")):
            require(len(objects) < 1024, "unexpected object count")
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            objects.append({"name": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        maps = {}
        if label == "ready":
            for line in processes:
                pid = line.split()[0]
                try:
                    with Path("/proc", pid, "maps").open() as source:
                        text = source.read(4 * LIMIT + 1)
                    maps[pid] = {
                        "truncated": len(text) > 4 * LIMIT,
                        "selected": [
                            row for row in text[: 4 * LIMIT].splitlines() if str(self.jit) in row
                        ],
                    }
                except (FileNotFoundError, PermissionError):
                    maps[pid] = {"unavailable": True}
        record = {
            "label": label,
            "at_epoch": time.time(),
            "pgid": pgid,
            "files": rows,
            "process_group": processes,
            "objects": objects,
            "ready_loaded_JIT_maps": maps,
        }
        with (self.output / "snapshots.jsonl").open("a") as output:
            output.write(json.dumps(record) + "\n")
        return record

    def native_identity(self):
        path = self.jit / (TARGET + ".so")
        require(path.resolve().is_relative_to(self.root), "native cache path escape")
        if not path.is_file():
            return {"path": str(path), "exists": False}
        record = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
        (self.output / "target-native.json").write_text(json.dumps(record, indent=2) + "\n")
        return record


def guard_environment(environment):
    require(environment.get("MAX_JOBS") == "32", "prewarm MAX_JOBS must be 32")
    require(environment.get("CARGO_BUILD_JOBS") == "8", "prewarm Cargo jobs must be 8")
    require(environment.get("AITER_LOG_MORE") == "1", "compiler diagnostics must be enabled")
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]:
        require(environment.get(key) == "1", "unexpected prewarm setting: " + key)
    require(environment.get("E2E_ENGINE") == "sglang", "unexpected engine")
    require(environment.get("E2E_GPU_TIER") == "1", "prewarm needs original single GPU")
    require(not environment.get("AITER_JIT_DIR"), "do not replace the image's baked JIT cache")
    require(environment.get("AITER_REBUILD", "0") == "0", "forced cache rebuild is not allowed")
    require(environment.get("SGLANG_USE_AITER") == "1", "AITER must remain enabled")
    require(environment.get("E2E_CONNECTION_MODE", "") == "", "original connection override")
    require(environment.get("E2E_SGLANG_SERVICER_IMPL", "") == "", "original servicer override")
