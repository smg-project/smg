"""Exact4420 job-container overlay; no engine imports or execution during apply."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat

HERE = Path(__file__).resolve().parent
TARGETS = {
    "/sgl-workspace/aiter/aiter/jit/utils/chip_info.py",
    "/sgl-workspace/sglang/python/sglang/srt/arg_groups/memory_hook.py",
    "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py",
}
HELPER = Path("/opt/venv/lib/python3.12/site-packages/smg_sg2_startup_trace.py")
sha = lambda data: hashlib.sha256(data).hexdigest()


def spec():
    data = json.loads((HERE / "startup-overlay.json").read_text())
    if set(data["files"]) != TARGETS or data["helper"]["path"] != str(HELPER):
        raise RuntimeError("exact three4420 source paths and helper required")
    return data


def read(path):
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode) or status.st_size > 512 * 1024:
        raise RuntimeError("bounded regular startup source required")
    return path.read_bytes()


def atomic(path, data, mode):
    temporary = path.with_name(path.name + f".sg2-startup-{os.getpid()}")
    created = False
    try:
        with temporary.open("xb") as stream:
            created = True
            os.fchmod(stream.fileno(), mode)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def apply(directory):
    marker = os.environ.get("E2E_JOB_MARKER", "")
    if not re.fullmatch(r"e2e-[0-9]+-[0-9]+-amd-mi300x-[12]", marker):
        raise RuntimeError("same-job startup overlay marker required")
    contract, directory = spec(), Path(directory)
    lock = directory / "sglang-startup-overlay.lock"
    with os.fdopen(os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), "r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        prepared = []
        for name, item in contract["files"].items():
            path = Path(name)
            raw = read(path)
            if sha(raw) == item["before_sha256"]:
                for change in item["changes"]:
                    before = change["before"].encode()
                    if raw.count(before) != 1:
                        raise RuntimeError("startup source anchor changed")
                    raw = raw.replace(before, change["after"].encode())
            if sha(raw) != item["after_sha256"]:
                raise RuntimeError("startup overlay refuses unknown source")
            prepared.append((path, raw, stat.S_IMODE(path.stat().st_mode)))
        helper = (HERE / "startup_trace.py").read_bytes()
        if sha(helper) != contract["helper"]["sha256"]:
            raise RuntimeError("startup helper source identity differs")
        if HELPER.exists() or HELPER.is_symlink():
            if sha(read(HELPER)) != contract["helper"]["sha256"]:
                raise RuntimeError("startup helper refuses an unrelated installed module")
        else:
            atomic(HELPER, helper, 0o644)
        for path, raw, mode in prepared:
            atomic(path, raw, mode)
        (directory / "sglang-startup").mkdir(exist_ok=True)
        verify(directory, "applied")


def verify(directory, stage="final"):
    contract, directory = spec(), Path(directory)
    for path, item in contract["files"].items():
        if sha(read(Path(path))) != item["after_sha256"]:
            raise RuntimeError("startup source replaced after overlay")
    if sha(read(HELPER)) != contract["helper"]["sha256"]:
        raise RuntimeError("startup helper replaced after overlay")
    evidence = directory / "sglang-startup"
    if evidence.is_symlink() or not evidence.is_dir():
        raise RuntimeError("startup artifact directory identity changed")
    files, total = {}, 0
    for path in evidence.iterdir():
        if not re.fullmatch(r"[0-9]+-[0-9]+(?:\.jsonl|-cu-(?:stdout|stderr)\.txt)", path.name):
            raise RuntimeError("unexpected startup evidence filename")
        raw = read(path)
        total += len(raw)
        if len(files) >= 512 or total > 32 * 1024**2:
            raise RuntimeError("startup artifact count/bytes cap")
        files[path.name] = {"bytes": len(raw), "sha256": sha(raw)}
    (directory / f"sglang-startup-overlay-{stage}.json").write_text(json.dumps({
        "source_manifest_sha256": sha((HERE / "startup-overlay.json").read_bytes()),
        "files": contract["files"], "helper": contract["helper"], "artifact_files": files,
        "runtime_origins": "HOLD_UNTIL_ACTUAL_MATCHED_EMITS", "model_or_RDMA_accepted": False,
    }, indent=2) + "\n")
