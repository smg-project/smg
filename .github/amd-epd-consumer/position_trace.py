"""Opt-in CPU metadata only; diagnostic failures never wrap engine operations."""

import hashlib
import json
import logging
import os
import re
import struct
import sys
import threading
import time
from array import array
from pathlib import Path

LOGGER = logging.getLogger("smg.epd.position")
LOGGER.setLevel(logging.WARNING)
LOGGER.propagate = False
if not LOGGER.handlers:
    LOGGER.addHandler(logging.StreamHandler())
LOCK = threading.RLock()
PID, REQUESTS, ORIGINS, USED, DROPPED = None, {}, {}, 0, False
IDENTITY = None


def bounded(path, cap):
    with Path(path).open("rb") as stream:
        raw = stream.read(cap + 1)
    if len(raw) > cap:
        raise ValueError("position identity read cap")
    return raw


def process(pid):
    raw = bounded(f"/proc/{pid}/stat", 8192).decode()
    fields = raw.rsplit(")", 1)[1].split()
    if int(raw.split("(", 1)[0]) != pid or not fields[19].isdigit():
        raise ValueError("position process identity")
    return {"pid": pid, "parent_pid": int(fields[1]), "start_ticks": fields[19]}


def loaded(module, path, expected_hash):
    actual = sys.modules.get(module)
    filename = getattr(actual, "__file__", None)
    if not filename or Path(filename).resolve() != path:
        raise ValueError("position actual loaded module path")
    if hashlib.sha256(bounded(path, 512 * 1024)).hexdigest() != expected_hash:
        raise ValueError("position actual loaded module hash")
    return {"module": module, "path": str(path), "sha256": expected_hash}


def child_identity(role, first_origin):
    manifest = json.loads(bounded(Path(__file__).with_name("_smg_epd_position_manifest.json"), 16384))
    anchor_raw = os.environ["SMG_EPD_POSITION_ANCHOR"]
    if len(anchor_raw) > 2048:
        raise ValueError("position launcher anchor cap")
    anchor = json.loads(anchor_raw)
    marker = os.environ["E2E_JOB_MARKER"]
    if (not re.fullmatch(r"[0-9a-f]{32}", anchor["nonce"])
            or not 1 <= len(marker) <= 256 or anchor["job_marker"] != marker
            or anchor["role"] != role or anchor["model_id"] != "Qwen/Qwen3.5-9B"
            or type(anchor["pid"]) is not int or anchor["pid"] <= 0
            or not re.fullmatch(r"[0-9]+", anchor["start_ticks"])):
        raise ValueError("position launcher anchor identity")
    me = process(os.getpid())
    if me["parent_pid"] != os.getppid():
        raise ValueError("position parent changed")
    chain, current = [], me
    for _ in range(16):
        chain.append(current)
        env = bounded(f"/proc/{current['pid']}/environ", 65536).split(b"\0")
        if ("E2E_JOB_MARKER=" + marker).encode() not in env:
            raise ValueError("position proc marker missing")
        if current["pid"] == anchor["pid"]:
            if current["start_ticks"] != anchor["start_ticks"]:
                raise ValueError("position launcher PID reused")
            break
        if current["parent_pid"] <= 1 or current["parent_pid"] in {r["pid"] for r in chain}:
            raise ValueError("position launcher ancestry missing")
        current = process(current["parent_pid"])
    else:
        raise ValueError("position ancestry depth cap")
    if process(me["pid"]) != me:
        raise ValueError("position emitting process changed")
    helper = manifest["helper"]
    helper_origin = loaded(helper["module"], Path(manifest["root"]) / helper["path"], helper["sha256"])
    if getattr(sys.modules[helper["module"]], "emit", None) is not emit:
        raise ValueError("position loaded helper binding")
    return {"kind": "smg_epd_position_identity_v2", **me, "role": role,
            "model_id": anchor["model_id"], "job_marker": marker, "launcher": anchor,
            "ancestry": chain, "helper": helper_origin, "first_origin": first_origin,
            "wall_time_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()}


def log_row(row):
    global USED
    text = json.dumps(row, separators=(",", ":"), allow_nan=False)
    size = len(text.encode()) + 1
    if size > 4096 or USED + size > 256 * 1024 - 512:
        raise ValueError("position diagnostic byte cap")
    USED += size
    LOGGER.warning("%s", text)


def ids(value):
    if not isinstance(value, (list, tuple, array)):
        return {"status": "NOT_CPU_IDS"}
    result = {"length": len(value)}
    if len(value) > 16384:
        return dict(result, status="OVER_CAP_NOT_HASHED")
    digest = hashlib.sha256(b"smg-epd-token-i64le-v1\0" + struct.pack("<Q", len(value)))
    for token in value:
        if type(token) is not int or not -(2**63) <= token < 2**63:
            return dict(result, status="UNSUPPORTED_NOT_HASHED")
        digest.update(struct.pack("<q", token))
    return dict(result, status="CPU_FULL", sha256=digest.hexdigest())


def tensor_meta(value, scalar=False):
    if value is None:
        return {"present": False}
    if type(value) is int:
        return {"present": True, "scalar": value} if scalar else {"present": True}
    device = getattr(value, "device", None)
    shape = tuple(getattr(value, "shape", ()))
    result = {"present": True, "shape": list(shape[:8]), "device": str(device),
              "dtype": str(getattr(value, "dtype", None))}
    if scalar:
        if getattr(device, "type", None) == "cpu" and shape in {(), (1,), (1, 1)}:
            number = value.item()  # CPU scalar only; never GPU synchronization.
            if type(number) is int:
                result["scalar"] = number
        else:
            result["scalar_status"] = "NOT_READ_GPU_OR_NONSCALAR"
    return result


def origin(module, filename):
    key = (module, filename)
    if key not in ORIGINS:
        manifest = json.loads(Path(__file__).with_name("_smg_epd_position_manifest.json").read_text())
        expected = next(row for row in manifest["modules"] if row["module"] == module)
        source = Path(filename).resolve()
        if source != Path(manifest["root"]) / expected["path"]:
            raise ValueError("position diagnostic origin path")
        ORIGINS[key] = loaded(module, source, expected["patched_sha256"])
    return ORIGINS[key]


def emit(stage, role, req, module, filename, *, embedding=None, scheduler=None, processor=None, lengths=None):
    global PID, REQUESTS, ORIGINS, USED, DROPPED, IDENTITY
    if (os.environ.get("SMG_EPD_POSITION_DIAGNOSTIC") != "1"
            or os.environ.get("SMG_EPD_POSITION_MODEL") != "Qwen/Qwen3.5-9B"):
        return
    try:
        with LOCK:
            pid = os.getpid()
            if PID != pid:
                PID, REQUESTS, ORIGINS, USED, DROPPED, IDENTITY = pid, {}, {}, 0, False, None
            if DROPPED:
                return
            rid = str(req.rid)
            if len(rid) > 4096:
                raise ValueError("position diagnostic RID cap")
            room = getattr(req, "bootstrap_room", None)
            if stage != "receiver_before" and stage != "receiver_after" and not (type(room) is int and room > 0):
                return  # Do not exhaust the bounded request sample on startup warmups.
            events = REQUESTS.get(rid, set())
            if stage in events:
                return  # Repeated allocation polling must not hide the later commit.
            if (rid not in REQUESTS and len(REQUESTS) >= 8) or len(events) >= 8:
                raise ValueError("position diagnostic request/event cap")
            if role not in {"prefill", "decode"}:
                raise ValueError("position diagnostic role")
            actual_origin = origin(module, filename)
            if IDENTITY is None:
                identity = child_identity(role, actual_origin)
                log_row(identity)
                IDENTITY = {"start_ticks": identity["start_ticks"], "identity_sha256":
                            hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
            cfg = getattr(scheduler, "model_config", None)
            mm = getattr(req, "mm_inputs", None) or getattr(req, "multimodal_inputs", None)
            row = {"kind": "smg_epd_position_v2", "stage": stage, "role": role, "pid": pid, **IDENTITY,
                   "rid": rid if re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", rid) else "sha256:" + hashlib.sha256(rid.encode()).hexdigest(),
                   "wall_time_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
                   "origin": actual_origin, "event": len(events) + 1,
                   "input_ids": ids(getattr(req, "input_ids", None)),
                   "origin_ids": ids(getattr(req, "origin_input_ids", None)),
                   "unpadded_ids": ids(getattr(req, "origin_input_ids_unpadded", None)),
                   "mm_present": mm is not None, "model_is_mrope": getattr(cfg, "model_is_mrope", None),
                   "mrope_positions": tensor_meta(getattr(mm, "mrope_positions", None)),
                   "mrope_delta": tensor_meta(getattr(mm, "mrope_position_delta", None), scalar=True),
                   "embedding": tensor_meta(embedding)}
            row["bootstrap_room"] = room if type(room) is int else None
            row["processor_class"] = (type(processor).__module__ + "." + type(processor).__qualname__) if processor is not None else None
            row["lengths"] = {key: value for key, value in (lengths or {}).items() if type(value) is int}
            REQUESTS[rid] = events | {stage}
            log_row(row)
    except Exception as error:
        # Only diagnostic work is caught. Original validation/compute/transfer remains outside.
        with LOCK:
            if not DROPPED:
                DROPPED = True
                try:
                    LOGGER.warning('{"kind":"smg_epd_position_drop_v2","pid":%d,"reason":"%s"}',
                                   os.getpid(), type(error).__name__)
                except Exception:
                    pass
