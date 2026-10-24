"""Exact, job-private actual4420 source overlay; no engine import or installation."""

import fcntl
import hashlib
import json
import os
import stat
import tempfile
import time
from pathlib import Path

from artifact_io import require

HERE = Path(__file__).resolve().parent
ROOT = Path("/sgl-workspace/sglang/python/sglang")
PATHS = {
    "sglang.srt.disaggregation.encoder.receiver": "srt/disaggregation/encoder/receiver.py",
    "sglang.srt.disaggregation.prefill": "srt/disaggregation/prefill.py",
    "sglang.srt.disaggregation.decode": "srt/disaggregation/decode.py",
}
HELPER = "sglang.srt.disaggregation._smg_epd_position_trace"
DELTA_HELPER = "sglang.srt.disaggregation._smg_epd_mrope_delta"
DELTA_PATHS = {
    "sglang.srt.disaggregation.utils": "srt/disaggregation/utils.py",
    "sglang.srt.managers.schedule_batch": "srt/managers/schedule_batch.py",
    "sglang.srt.model_executor.forward_batch_info": "srt/model_executor/forward_batch_info.py",
}
META = "srt/disaggregation/_smg_epd_position_manifest.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def manifest_path():
    return HERE / ("mrope_manifest.json" if os.environ.get("SMG_EPD_MROPE_DELTA_HANDOFF") == "1"
                   else "position_manifest.json")


def configuration():
    data = json.loads(manifest_path().read_text())
    require(data["version"] == 1 and data["root"] == str(ROOT), "overlay root/version")
    require(data["model_id"] == "Qwen/Qwen3.5-9B", "overlay model")
    delta = os.environ.get("SMG_EPD_MROPE_DELTA_HANDOFF") == "1"
    paths = PATHS | DELTA_PATHS if delta else PATHS
    require(len(data["modules"]) == len(paths), "overlay module count")
    require({row["module"]: row["path"] for row in data["modules"]} == paths, "overlay modules")
    if delta:
        require(data["delta_helper"]["module"] == DELTA_HELPER
                and data["delta_helper"]["source"] == "mrope_delta.py"
                and data["delta_helper"]["path"] == "srt/disaggregation/_smg_epd_mrope_delta.py",
                "delta helper identity")
    helper = data["helper"]
    require(helper["module"] == HELPER and helper["source"] == "position_trace.py"
            and helper["path"] == "srt/disaggregation/_smg_epd_position_trace.py", "overlay helper")
    return data


def expected_engine():
    require(os.environ.get("SMG_EPD_MROPE_DELTA_HANDOFF") != "1"
            or os.environ.get("SMG_EPD_POSITION_DIAGNOSTIC") == "1", "delta needs source witness")
    if os.environ.get("SMG_EPD_POSITION_DIAGNOSTIC") != "1":
        return {}
    data = configuration()
    result = {row["module"]: {"path": str(ROOT / row["path"]), "sha256": row["patched_sha256"]}
            for row in data["modules"]} | {
        HELPER: {"path": str(ROOT / data["helper"]["path"]), "sha256": data["helper"]["sha256"]}
    }
    if "delta_helper" in data:
        helper = data["delta_helper"]
        result[DELTA_HELPER] = {"path": str(ROOT / helper["path"]), "sha256": helper["sha256"]}
    return result


def read(path):
    require(not path.is_symlink() and path.parent.resolve() == path.parent, "unsafe overlay path")
    with path.open("rb") as stream:
        raw = stream.read(512 * 1024 + 1)
    require(len(raw) <= 512 * 1024, "overlay source cap")
    return raw


def atomic(path, raw):
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".smg-position-", delete=False) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), mode)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def apply(directory):
    require(os.environ.get("SMG_EPD_MROPE_DELTA_HANDOFF") != "1"
            or os.environ.get("SMG_EPD_POSITION_DIAGNOSTIC") == "1", "delta needs source witness")
    if os.environ.get("SMG_EPD_POSITION_DIAGNOSTIC") != "1":
        return
    data = configuration()
    require(ROOT.resolve() == ROOT, "overlay root is not physical")
    lock = ROOT / "srt/disaggregation/.smg-position-overlay.lock"
    require(not lock.is_symlink(), "unsafe overlay lock")
    with lock.open("a+b") as stream:
        deadline = time.monotonic() + 10
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                require(time.monotonic() < deadline, "overlay lock deadline")
                time.sleep(0.05)
        writes = []
        for row in data["modules"]:
            path = ROOT / row["path"]
            raw = read(path)
            require(sha(raw) in {row["original_sha256"], row["patched_sha256"]}, "overlay source identity")
            patch = HERE / row["patch"]
            require(patch.parent == HERE / "position_patches", "overlay patch path")
            patch_bytes = read(patch)
            require(sha(patch_bytes) == row["patch_sha256"], "overlay patch hash")
            if sha(raw) == row["original_sha256"]:
                text = raw.decode()
                for before, after in json.loads(patch_bytes):
                    require(text.count(before) == 1, "overlay anchor")
                    text = text.replace(before, after, 1)
                raw = text.encode()
            require(sha(raw) == row["patched_sha256"], "overlay patched identity")
            writes.append((path, raw))
        helper = data["helper"]
        helper_bytes = read(HERE / helper["source"])
        require(sha(helper_bytes) == helper["sha256"], "overlay helper source")
        additions = [(ROOT / helper["path"], helper_bytes),
                     (ROOT / META, manifest_path().read_bytes())]
        if "delta_helper" in data:
            extra = data["delta_helper"]
            raw = read(HERE / extra["source"])
            require(sha(raw) == extra["sha256"], "delta helper source")
            additions.append((ROOT / extra["path"], raw))
        for path, raw in additions:
            require(not path.is_symlink() and path.parent.resolve() == path.parent, "unsafe overlay addition")
            require(not path.exists() or read(path) == raw, "overlay addition already differs")
        for path, raw in additions + writes:
            atomic(path, raw)
            require(read(path) == raw, "overlay write differs")
    report = {"applied": True, "pid": os.getpid(), "modules": expected_engine(),
              "manifest_sha256": sha(manifest_path().read_bytes()),
              "runtime_events": "NOT_YET_VERIFIED"}
    (directory / f"position-overlay-{os.getpid()}.json").write_text(json.dumps(report, indent=2) + "\n")
