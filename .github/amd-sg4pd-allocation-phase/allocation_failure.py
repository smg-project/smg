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


_RUNTIME_STAGES = {"cu-before", "cu-return", "cu-error", "stream-left-before", "stream-left-return",
                   "stream-right-before", "stream-right-return", "stream-error", "futuremap-second-allocation-error"}
_runtime_pid, _runtime_calls, _runtime_seen = None, {}, set()
_runtime_lock = threading.Lock()
_RUNTIME_SOURCES = {'/sgl-workspace/aiter/aiter/jit/utils/chip_info.py': '5b8aedd33b5d081be4ceeb6fd211b5a77bbf7f5bdf0f6293304cef7ed0a5e7e6', '/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py': 'de5fa91beee5b42ccd918ef7586b2bc9122b3862cdfd62653a6e65f38cea1294', '/sgl-workspace/sglang/python/sglang/srt/managers/overlap_utils.py': 'bd2c86072fa4edf41fa5ac41fdc73c2840c61b7282c39ac30305443f26e41648'}


def runtime_event(stage, module, file, fields):
    """First CU/stream invocation per PID; observations cannot replace its result."""
    global _runtime_pid, _runtime_calls, _runtime_seen, _runtime_lock
    if os.environ.get(FLAG) != "1":
        return
    try:
        pid, call = os.getpid(), fields["observation_id"]
        if stage not in _RUNTIME_STAGES or type(call) is not int or file not in _RUNTIME_SOURCES:
            raise ValueError("exact runtime event required")
        if pid != _runtime_pid:
            _runtime_pid, _runtime_calls, _runtime_seen, _runtime_lock = pid, {}, set(), threading.Lock()
        group = "futuremap" if stage == "futuremap-second-allocation-error" else ("cu" if stage.startswith("cu-") else "stream")
        with _runtime_lock:
            _runtime_calls.setdefault(group, fields)
            if _runtime_calls[group] is not fields or stage in _runtime_seen or len(_runtime_seen) >= 8:
                return
            _runtime_seen.add(stage)
        marker = os.environ.get("E2E_JOB_MARKER", "")
        if not re.fullmatch(r"e2e-[0-9]+-[0-9]+-amd-mi300x-[12]", marker):
            raise ValueError("same-job marker required")
        origin = _origin(module, file)
        if origin["sha256"] != _RUNTIME_SOURCES[file]:
            raise ValueError("exact observed source required")
        torch = sys.modules.get("torch")
        initialized = torch is not None and torch.cuda.is_initialized()
        runtime = sys.modules.get("sglang.srt.runtime_context")
        role = _query(lambda: _text(str(runtime.get_disagg().disaggregation_mode))) if runtime else None
        ticks = _ticks()
        row = {"kind": "sg4-runtime-event", "stage": stage, "pid": pid, "start_ticks": ticks,
               "job_marker": marker, "role": role, "monotonic": time.monotonic(), "origin": origin,
               "helper": _origin(__name__, __file__), "Torch_initialized": initialized, "fields": dict(fields),
               "masks": {key: _text(os.environ.get(key)) for key in
                         ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")}}
        if stage in ("stream-error", "futuremap-second-allocation-error") and initialized:
            row["allocator_stats_after_error"] = _query(lambda: _stats(torch, None))
            row["allocator_backend_after_error"] = _query(torch.cuda.memory.get_allocator_backend)
            row["query_limit"] = "CPU allocator metadata only; possibly error-contaminated; no physical-memory query"
            if stage == "futuremap-second-allocation-error":
                row["allocator_context"] = "implicit current Torch allocator (index=None); agreement with requested_device is unproved"
        data = (json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n").encode()
        if len(data) > 4096 or ticks != _ticks():
            raise ValueError("bounded same-PID event required")
        directory = Path(os.environ["AMD_DEPENDENCY_DIRECTORY"]) / "sglang-startup"
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("precreated same-job directory required")
        fd = os.open(directory / f"{pid}-{ticks}.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            status = os.fstat(stream.fileno())
            if not stat.S_ISREG(status.st_mode) or status.st_size + len(data) > 64 * 1024:
                raise ValueError("bounded regular event file required")
            stream.write(data)
        logging.getLogger("smg.sg4.startup").warning("SG4_RUNTIME %s", data.decode().rstrip("\n"))
    except BaseException as error:
        try:
            sys.stderr.write("SG4_RUNTIME_EVENT_DROP " + (_text(type(error).__name__) or "error") + "\n")
        except BaseException:
            pass


_MEMORY_TARGET = "/sgl-workspace/sglang/python/sglang/srt/utils/common.py"
_MEMORY_SHA256 = "1d7f730cf63588278bf96dac091832b7620bd587868154c1dcd456d6a952127c"
_memory_pid, _memory_attempts, _memory_drop = os.getpid(), 0, False
_memory_lock = threading.Lock()


def _memory_properties(props, integrated):
    result = {"is_integrated": integrated if type(integrated) in (bool, int) else None}
    for name in ("name", "total_memory", "multi_processor_count", "gcnArchName",
                 "uuid", "pci_bus_id", "pci_device_id", "pci_domain_id"):
        try:
            value = getattr(props, name)
            if name == "uuid" and type(value) not in (str, bytes):
                raw = getattr(value, "bytes", None)
                value = bytes(raw).hex() if isinstance(raw, (list, tuple)) and len(raw) == 16 and all(type(x) is int and 0 <= x <= 255 for x in raw) else None
            if type(value) is bytes:
                value = value.hex()
            if type(value) is str:
                value = value[:128]
            result[name] = {"value": value if type(value) in (str, int, float, bool) else None,
                            "status": "observed-existing-CPU-property" if type(value) in (str, int, float, bool) else "unavailable-or-unrecorded"}
        except BaseException as error:
            result[name] = {"value": None, "status": "missing-or-read-error", "error_type": type(error).__name__}
    return result


def memory_query_event(module, file, device, gpu_id, distributed, empty_cache,
                       branch, integrated, props, raw_bytes, total_bytes,
                       returned_bytes, original_error):
    """Bounded existing-query values only; never issue a device or psutil query."""
    global _memory_pid, _memory_attempts, _memory_drop, _memory_lock
    if os.environ.get(FLAG) != "1":
        return
    try:
        pid = os.getpid()
        if pid != _memory_pid:
            _memory_pid, _memory_attempts, _memory_drop, _memory_lock = pid, 0, False, threading.Lock()
        with _memory_lock:
            if _memory_attempts >= 8:
                return
            _memory_attempts += 1  # Even a failed observation consumes this PID's slot.
            ordinal = _memory_attempts
        # Preserve the original exception identity before diagnostic metadata work.
        exception = None if original_error is None else {"type": type(original_error).__name__, "message": str(original_error)[:512]}
        marker = os.environ.get("E2E_JOB_MARKER", "")
        if not re.fullmatch(r"e2e-[0-9]+-[0-9]+-amd-mi300x-[12]", marker):
            raise ValueError("same-job memory-query marker required")
        if module != "sglang.srt.utils.common" or file != _MEMORY_TARGET or device != "cuda":
            raise ValueError("exact existing CUDA memory-query source required")
        origin = _origin(module, file)
        if origin["sha256"] != _MEMORY_SHA256:
            raise ValueError("exact current memory-query overlay required")
        runtime = sys.modules.get("sglang.srt.runtime_context")
        role = _query(lambda: _text(str(runtime.get_disagg().disaggregation_mode))) if runtime else None
        caller = None
        try:
            frame = sys._getframe(3)
            caller = {"file": frame.f_code.co_filename, "function": frame.f_code.co_name,
                      "line": frame.f_lineno}
            del frame
        except BaseException:
            pass
        scalar = lambda value: value if type(value) in (int, float, bool) else None
        ticks = _ticks()
        row = {"kind": "existing-memory-query", "ordinal": ordinal, "pid": pid,
               "start_ticks": ticks, "role": role, "job_marker": marker, "monotonic": time.monotonic(),
               "origin": origin, "helper": _origin(__name__, __file__), "caller": caller,
               "device_argument": device, "logical_gpu_id_argument": scalar(gpu_id),
               "distributed_argument": scalar(distributed), "empty_cache_argument": scalar(empty_cache),
               "branch": branch, "properties_from_existing_query": _memory_properties(props, integrated),
               "existing_query_raw_available_bytes_before_MIN": scalar(raw_bytes),
               "existing_query_raw_total_bytes": scalar(total_bytes),
               "bytes_after_existing_MIN_before_original_GiB_division": scalar(returned_bytes),
               "original_exception": exception,
               "masks": {key: _text(os.environ.get(key)) for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")},
               "new_device_or_psutil_queries": 0, "actual_resolved_pool_budget_observed": False,
               "limits": "First8 existing queries/PID only; raw query versus post-MIN value are distinct. Caller/source identities do not prove physical counters correct or any allocation succeeds."}
        data = (json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n").encode()
        if len(data) > 4096 or ticks != _ticks():
            raise ValueError("bounded same-PID memory event required")
        directory = Path(os.environ["AMD_DEPENDENCY_DIRECTORY"]) / "sglang-startup"
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("precreated same-job event directory required")
        fd = os.open(directory / f"{pid}-{ticks}.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            status = os.fstat(stream.fileno())
            if not stat.S_ISREG(status.st_mode) or status.st_size + len(data) > 64 * 1024:
                raise ValueError("bounded regular memory event file required")
            stream.write(data)
        logging.getLogger("smg.sg4.memory").warning("SG4_MEMORY %s", data.decode().rstrip("\n"))
    except BaseException as error:
        if not _memory_drop:
            _memory_drop = True
            try:
                sys.stderr.write("SG4_MEMORY_EVENT_DROP " + type(error).__name__ + "\n")
            except BaseException:
                pass


_POOL_SOURCES = {'/sgl-workspace/sglang/python/sglang/srt/mem_cache/kv_cache_configurator.py': 'a2493e2dcb2b1e1feb63665c2a896fe6c0079555a46d95953484eaef8600c122', '/sgl-workspace/sglang/python/sglang/srt/distributed/bootstrap.py': '6bda018bde5e3eb1f3b7665b39c4a1d26240c1f5c17cf7c286bc27756c9f2211'}
_pool_pid, _pool_seen = None, set()
_pool_lock = threading.Lock()


def pool_budget_event(stage, module, file, fields):
    """Existing consumed CPU locals only; no Torch, device or allocator query."""
    global _pool_pid, _pool_seen, _pool_lock
    if os.environ.get(FLAG) != "1":
        return
    try:
        if stage not in ("preload-B", "kv-profile", "pool-before-init") or file not in _POOL_SOURCES:
            raise ValueError("exact consumed pool stage required")
        if type(fields) is not dict or len(fields) > 16:
            raise ValueError("bounded scalar pool locals required")
        if any(type(k) is not str or len(k) > 64 or type(v) not in (str, int, float, bool, type(None)) for k, v in fields.items()):
            raise ValueError("CPU scalar locals only")
        pid = os.getpid()
        if pid != _pool_pid:
            _pool_pid, _pool_seen, _pool_lock = pid, set(), threading.Lock()
        with _pool_lock:
            if stage in _pool_seen or len(_pool_seen) >= 3:
                return
            _pool_seen.add(stage)
        marker = os.environ.get("E2E_JOB_MARKER", "")
        if not re.fullmatch(r"e2e-[0-9]+-[0-9]+-amd-mi300x-[12]", marker):
            raise ValueError("same-job pool marker required")
        origin = _origin(module, file)
        if origin["sha256"] != _POOL_SOURCES[file]:
            raise ValueError("exact installed pool source required")
        ticks = _ticks()
        runtime = sys.modules.get("sglang.srt.runtime_context")
        role = _query(lambda: _text(str(runtime.get_disagg().disaggregation_mode))) if runtime else None
        row = {"kind": "consumed-pool-budget", "stage": stage, "pid": pid,
               "start_ticks": ticks, "job_marker": marker, "monotonic": time.monotonic(),
               "role": role, "origin": origin, "helper": _origin(__name__, __file__),
               "fields": fields, "new_GPU_or_allocator_queries": 0,
               "masks": {key: _text(os.environ.get(key)) for key in
                         ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")},
               "limit": "Planning locals before pool init; not physical counter correctness or allocation success."}
        data = (json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n").encode()
        if len(data) > 4096 or ticks != _ticks():
            raise ValueError("bounded same-PID pool event required")
        directory = Path(os.environ["AMD_DEPENDENCY_DIRECTORY"]) / "sglang-startup"
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("precreated same-job pool directory required")
        fd = os.open(directory / f"{pid}-{ticks}.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            status = os.fstat(stream.fileno())
            if not stat.S_ISREG(status.st_mode) or status.st_size + len(data) > 64 * 1024:
                raise ValueError("bounded regular pool event file required")
            stream.write(data)
        logging.getLogger("smg.sg4.pool").warning("SG4_POOL %s", data.decode().rstrip("\n"))
    except BaseException as error:
        try:
            sys.stderr.write("SG4_POOL_EVENT_DROP " + type(error).__name__ + "\n")
        except BaseException:
            pass
