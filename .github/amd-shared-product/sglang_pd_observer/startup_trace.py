"""Opt-in startup evidence; no engine import, retry, allocation or synchronization."""

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import threading
import time

_lock = threading.Lock()
_pid, _memory_count, _cu_count, _dropped = None, 0, 0, False
ENV = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CU_NUM", "GPU_ARCHS")


def _drop(error):
    global _dropped
    if not _dropped:
        _dropped = True
        try:
            sys.stderr.write("SG2_STARTUP_DROP " + type(error).__name__ + "\n")
        except Exception:
            pass


def _scalar(value):
    return value if type(value) in (str, int, float, bool, type(None)) else None


def _origin(module, file):
    actual = getattr(sys.modules.get(module), "__file__", None)
    path = Path(file)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 512 * 1024:
        raise ValueError("bounded regular startup source required")
    return {"module": module, "file": str(path), "sys_modules_file": actual,
            "same_file": actual is not None and Path(actual).resolve() == path.resolve(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _emit(kind, stage, module, file, values, raw=None):
    global _pid, _memory_count, _cu_count, _dropped, _lock
    if os.environ.get("SMG_SG2_STARTUP_DIAGNOSTIC") != "1":
        return
    try:
        pid = os.getpid()
        if _pid != pid:
            _pid, _memory_count, _cu_count, _dropped, _lock = pid, 0, 0, False, threading.Lock()
        with _lock:
            if (kind == "memory" and _memory_count >= 12) or (kind == "cu" and _cu_count >= 1):
                return
            marker = os.environ.get("E2E_JOB_MARKER", "")
            if not re.fullmatch(r"e2e-[0-9]+-[0-9]+-amd-mi300x-[12]", marker):
                raise ValueError("same-job startup marker required")
            directory = Path(os.environ["AMD_DEPENDENCY_DIRECTORY"]) / "sglang-startup"
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("precreated artifact directory required")
            stat = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
            runtime = sys.modules.get("sglang.srt.runtime_context")
            try:
                role = _scalar(getattr(runtime.get_disagg(), "disaggregation_mode", None)) if runtime else None
            except Exception:
                role = None  # Argument resolution can precede publication of the runtime context.
            row = {"kind": kind, "stage": stage, "pid": pid, "ppid": os.getppid(),
                   "start_ticks": stat[19], "job_marker": marker, "monotonic": time.monotonic(),
                   "role": role,
                   "origin": _origin(module, file), "helper": _origin(__name__, __file__),
                   "env": {key: os.environ.get(key) for key in ENV}, **values}
            if raw is not None:
                row["raw"] = {}
                for name, text in raw.items():
                    prefix = text[:128 * 1024 + 1].encode("utf-8", errors="replace")
                    data = prefix[:128 * 1024]
                    target = directory / f"{pid}-{stat[19]}-cu-{name}.txt"
                    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(data)
                    row["raw"][name] = {"file": target.name, "saved_bytes": len(data),
                                        "original_chars": len(text), "truncated": len(text) > 128 * 1024 or len(prefix) > 128 * 1024,
                                        "saved_sha256": hashlib.sha256(data).hexdigest()}
            line = json.dumps(row, ensure_ascii=True, allow_nan=False)
            if len(line.encode()) > 4096:
                raise ValueError("startup event exceeds4096 bytes")
            fd = os.open(directory / f"{pid}-{stat[19]}.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "a") as stream:
                stream.write(line + "\n")
            _memory_count += kind == "memory"
            _cu_count += kind == "cu"
            logging.getLogger("smg.sg2.startup").warning("SG2_STARTUP %s", line)
    except Exception as error:
        _drop(error)


def cu(module, file, scope):
    if os.environ.get("SMG_SG2_STARTUP_DIAGNOSTIC") != "1":
        return
    if _pid == os.getpid() and _cu_count >= 1:
        return
    try:
        result, path = scope.get("result"), scope.get("rocminfo")
        executable = None
        if path:
            target = Path(path).resolve()
            executable = {"path": str(target), "bytes": target.stat().st_size}
            if target.stat().st_size <= 1024 * 1024:
                executable["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        values = scope.get("gpu_compute_units", [])
        _emit("cu", "original-query-result", module, file,
              {"executable": executable, "returncode": getattr(result, "returncode", None),
               "parsed_CU_list": values[:32], "parsed_count": len(values),
               "exception_type": type(scope["e"]).__name__ if "e" in scope else None},
              {"stdout": getattr(result, "stdout", "") or "", "stderr": getattr(result, "stderr", "") or ""})
    except Exception as error:
        _drop(error)  # Evidence cannot replace the original return, assertion or exception.


def memory(stage, module, file, scope, namespace):
    if os.environ.get("SMG_SG2_STARTUP_DIAGNOSTIC") != "1":
        return
    if _pid == os.getpid() and _memory_count >= 12:
        return
    try:
        fields = {key: _scalar(scope[key]) for key in
                  ("mem_fraction_static", "reserved_mem", "gpu_mem", "post_capture_kv_sizing") if key in scope}
        cfg = scope.get("cfg")
        if cfg is not None:
            fields["config"] = {key: _scalar(getattr(cfg, key, None)) for key in
                                ("mem_fraction_static", "chunked_prefill_size", "max_running_requests", "tp_size", "pp_size", "disaggregation_mode")}
            graph = scope.get("decode_cuda_graph_config", cfg.cuda_graph_config.decode)
            fields["decode_graph_max_bs"] = _scalar(graph.max_bs)
        for getter, keys in (("get_disagg", ("disaggregation_mode",)), ("get_device", ("gpu_id", "device")),
                             ("get_parallel", ("tp_rank", "tp_size", "pp_size"))):
            if getter in namespace:
                obj = namespace[getter]()
                fields[getter] = {key: _scalar(getattr(obj, key, None)) for key in keys}
        torch = namespace.get("torch") or sys.modules.get("torch")
        fields["Torch_initialized"] = torch is not None and torch.cuda.is_initialized()
        if fields["Torch_initialized"]:
            try:
                index = torch.cuda.current_device()
                free, total = torch.cuda.mem_get_info(index)
                fields["memory"] = {"device": index, "free": free, "total": total,
                                    "allocated": torch.cuda.memory_allocated(index), "reserved": torch.cuda.memory_reserved(index)}
            except Exception as error:
                fields["memory_query_error"] = type(error).__name__
        _emit("memory", stage, module, file, fields)
    except Exception as error:
        _drop(error)
