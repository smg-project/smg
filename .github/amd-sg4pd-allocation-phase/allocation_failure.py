"""One bounded, opt-in observation after the original seq-lens operation fails."""

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import resource
import stat
import sys
import threading
import time

FLAG = "SMG_SG4_PD_ALLOCATION_PHASE_DIAGNOSTIC"
TARGET = "/sgl-workspace/sglang/python/sglang/srt/managers/schedule_batch.py"
TARGET_SHA256 = "c136bb9a316144f17cc7108a7bb09d374bdd4afd063710be7ccdc8bdd9fe320d"
STATS = ("allocated_bytes.all.current", "reserved_bytes.all.current", "active_bytes.all.current",
         "inactive_split_bytes.all.current", "num_ooms", "num_alloc_retries")
ENV = ("SGLANG_MOONCAKE_CUSTOM_MEM_POOL", "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF",
       "PYTORCH_NO_CUDA_MEMORY_CACHING", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
       "ROCR_VISIBLE_DEVICES", "HSA_XNACK", "HSA_ENABLE_SDMA", "HSA_FORCE_FINE_GRAIN_PCIE",
       "HIP_FORCE_DEV_KERNARG", "AMD_SERIALIZE_KERNEL", "AMD_SERIALIZE_COPY")
_pid, _attempted, _dropped = os.getpid(), False, False
_lock = threading.Lock()


def _text(value):
    return value[:64].encode("ascii", errors="backslashreplace").decode()[:64] if type(value) is str else None


def _claim():
    global _pid, _attempted, _dropped, _lock
    pid = os.getpid()
    if _pid != pid:
        _pid, _attempted, _dropped, _lock = pid, False, False, threading.Lock()
    with _lock:
        if _attempted:
            return False
        _attempted = True  # Claim before queries; even a dropped observation is never retried in this PID.
        return True


def _origin(module, file):
    path = Path(file)
    actual = getattr(sys.modules.get(module), "__file__", None)
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode) or status.st_size > 512 * 1024 or actual != str(path):
        raise ValueError("bounded current loaded source required")
    return {"module": module, "file": str(path), "sys_modules_file": actual,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _ticks():
    with open("/proc/self/stat") as stream:
        raw = stream.read(8193)
    if len(raw) > 8192:
        raise ValueError("bounded process identity required")
    value = raw.rsplit(")", 1)[1].split()[19]
    if not value.isdigit():
        raise ValueError("process start ticks required")
    return value


def _query(function):
    try:
        return function()
    except BaseException as error:
        return {"query_error": _text(type(error).__name__)}


def _host_status():
    with open("/proc/self/status") as stream:
        raw = stream.read(16385)
    if len(raw) > 16384:
        raise ValueError("bounded host process status required")
    return {key: _text(value.strip()) for key, _, value in
            (line.partition(":") for line in raw.splitlines()) if key in ("VmLck", "VmRSS")}


def _stats(torch, index):
    values = torch.cuda.memory_stats(index)
    return {key: values.get(key) if type(values.get(key)) in (int, float) else None for key in STATS}


def observe(stage, module, file, original_error, torch, device, pin_memory):
    global _dropped
    if os.environ.get(FLAG) != "1":
        return
    try:
        if not _claim():
            return
        marker = os.environ.get("E2E_JOB_MARKER", "")
        if not re.fullmatch(r"e2e-[0-9]+-[0-9]+-amd-mi300x-[12]", marker):
            raise ValueError("same-job diagnostic marker required")
        runtime = sys.modules.get("sglang.srt.runtime_context")
        if runtime is None or runtime.get_disagg().disaggregation_mode != "prefill":
            return
        if torch is not sys.modules.get("torch") or not torch.cuda.is_initialized():
            return  # No query may initialize Torch or a device on this failure path.
        if stage not in ("seq-lens-cpu-pin", "seq-lens-to-device") or module != "sglang.srt.managers.schedule_batch" or file != TARGET:
            raise ValueError("exact seq-lens failure location required")
        origin, helper = _origin(module, file), _origin(__name__, __file__)
        if origin["sha256"] != TARGET_SHA256:
            raise ValueError("exact overlaid emitter source required")
        ticks, pid = _ticks(), os.getpid()
        host = {"RLIMIT_MEMLOCK_bytes": _query(lambda: resource.getrlimit(resource.RLIMIT_MEMLOCK)),
                "proc_status": _query(_host_status)}
        index = _query(torch.cuda.current_device)
        has_index = type(index) is int
        fraction = _query(lambda: torch.cuda.get_per_process_memory_fraction(index)) if has_index else {"not_queried": "device_query_failed"}
        stats = _query(lambda: _stats(torch, index)) if has_index else {"not_queried": "device_query_failed"}
        backend = _query(torch.cuda.memory.get_allocator_backend)
        if type(backend) is str:
            backend = _text(backend)
        row = {"kind": "allocation-failure", "stage": stage, "pid": pid, "ppid": os.getppid(),
               "start_ticks": ticks, "job_marker": marker, "role": "prefill", "monotonic": time.monotonic(),
               "origin": origin, "helper": helper, "Torch_initialized": True,
               "exception_type": _text(type(original_error).__name__), "device_argument": _text(str(device)),
               "pin_memory": pin_memory if type(pin_memory) is bool else None, "host": host, "current_device": index,
               "stream_or_capture_query": "not attempted: no proved non-priming getter in this scope",
               "per_process_memory_fraction": fraction, "allocator_backend": backend, "memory_stats": stats,
               "env": {key: {"value": _text(os.environ.get(key)), "original_chars": len(os.environ[key]) if key in os.environ else None} for key in ENV}}
        data = (json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n").encode()
        if len(data) > 4096 or ticks != _ticks() or pid != os.getpid():
            raise ValueError("bounded same-PID diagnostic required")
        directory = Path(os.environ["AMD_DEPENDENCY_DIRECTORY"]) / "sglang-startup"
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("precreated job diagnostic directory required")
        fd = os.open(directory / f"{pid}-{ticks}.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            status = os.fstat(stream.fileno())
            if not stat.S_ISREG(status.st_mode) or status.st_size + len(data) > 64 * 1024:
                raise ValueError("bounded regular diagnostic log required")
            stream.write(data)
        logging.getLogger("smg.sg2.startup").warning("SG2_STARTUP %s", data.decode().rstrip("\n"))
    except BaseException as error:
        if not _dropped:
            _dropped = True
            try:
                sys.stderr.write("SG2_ALLOCATION_FAILURE_DROP " + (_text(type(error).__name__) or "error") + "\n")
            except BaseException:
                pass  # Observation failure cannot replace the original exception.
