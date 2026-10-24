"""Observe the unchanged admin/load oracles; never create requests or trace files."""
import gzip
import hashlib
import json
import stat
import threading
from pathlib import Path

from .process import require


def trace_artifacts(directory):
    root = directory.resolve(strict=True)
    pending, files, total, visited = [root], [], 0, 0
    cpu, gpu = False, False
    while pending:
        parent = pending.pop()
        for path in parent.iterdir():
            visited += 1
            require(visited <= 32 and len(path.relative_to(root).parts) <= 4, "trace traversal cap")
            mode = path.lstat().st_mode
            require(not stat.S_ISLNK(mode), "trace symlink is not owned evidence")
            if stat.S_ISDIR(mode):
                pending.append(path)
                continue
            require(stat.S_ISREG(mode) and len(files) < 8, "trace member/count differs")
            raw = path.read_bytes() if path.stat().st_size <= 64 * 1024**2 else None
            require(raw is not None, "trace byte cap")
            total += len(raw)
            require(total <= 128 * 1024**2, "trace aggregate cap")
            entry = {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            if path.name.endswith((".json", ".json.gz")):
                if path.name.endswith(".gz"):
                    with gzip.open(path, "rb") as stream:
                        decoded = stream.read(64 * 1024**2 + 1)
                    require(len(decoded) <= 64 * 1024**2, "decompressed trace cap")
                else:
                    decoded = raw
                events = json.loads(decoded).get("traceEvents", [])
                require(isinstance(events, list), "traceEvents is not a list")
                categories = {str(event.get("cat", "")) for event in events if isinstance(event, dict)}
                cpu |= "cpu_op" in categories
                gpu |= "kernel" in categories
                entry["categories"] = sorted(categories)
            files.append(entry)
    require(files and cpu and gpu, "requested directory lacks actual CPU and GPU trace events")
    return {"requested_directory": str(root), "files": files, "cpu_op": cpu, "GPU_kernel": gpu}


class Burst:
    def __init__(self, client):
        self.resource, self.lock = client.chat.completions, threading.Lock()
        self.original = self.resource.create
        self.started = self.completed = self.failed = 0

    def create(self, *args, **kwargs):
        require(kwargs.get("max_tokens") == 1536 and kwargs.get("temperature") == 0.0
                and kwargs.get("extra_body") == {"ignore_eos": True}, "original load request changed")
        with self.lock:
            self.started += 1
        try:
            result = self.original(*args, **kwargs)
        except BaseException:
            with self.lock:
                self.failed += 1
            raise
        with self.lock:
            self.completed += 1
        return result

    def start(self):
        self.resource.create = self.create

    def finish(self):
        self.resource.create = self.original
        require(self.started == self.completed == 32 and self.failed == 0, "original 32-request burst incomplete")
        return {"started": self.started, "completed": self.completed, "failed": self.failed}
