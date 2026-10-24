"""Opt-in CPU metadata only; diagnostic failures never wrap engine operations."""

import hashlib
import json
import logging
import os
import re
import struct
import threading
import time
from array import array
from pathlib import Path

LOGGER = logging.getLogger("smg.epd.position")
LOGGER.setLevel(logging.WARNING)
LOCK = threading.RLock()
PID, REQUESTS, ORIGINS, USED, DROPPED = None, {}, {}, 0, False


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
        with source.open("rb") as stream:
            raw = stream.read(512 * 1024 + 1)
        if len(raw) > 512 * 1024 or hashlib.sha256(raw).hexdigest() != expected["patched_sha256"]:
            raise ValueError("position diagnostic origin hash")
        ORIGINS[key] = {"module": module, "path": str(source), "sha256": expected["patched_sha256"]}
    return ORIGINS[key]


def emit(stage, role, req, module, filename, *, embedding=None, scheduler=None, processor=None, lengths=None):
    global PID, REQUESTS, ORIGINS, USED, DROPPED
    if (os.environ.get("SMG_EPD_POSITION_DIAGNOSTIC") != "1"
            or os.environ.get("SMG_EPD_POSITION_MODEL") != "Qwen/Qwen3.5-9B"):
        return
    try:
        with LOCK:
            pid = os.getpid()
            if PID != pid:
                PID, REQUESTS, ORIGINS, USED, DROPPED = pid, {}, {}, 0, False
            rid = str(req.rid)
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
            cfg = getattr(scheduler, "model_config", None)
            mm = getattr(req, "mm_inputs", None) or getattr(req, "multimodal_inputs", None)
            row = {"kind": "smg_epd_position_v1", "stage": stage, "role": role, "pid": pid,
                   "rid": rid if re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", rid) else "sha256:" + hashlib.sha256(rid.encode()).hexdigest(),
                   "wall_time_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
                   "origin": origin(module, filename), "event": len(events) + 1,
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
            text = json.dumps(row, separators=(",", ":"), allow_nan=False)
            size = len(text.encode())
            if size > 4096 or USED + size > 256 * 1024 - 512:
                raise ValueError("position diagnostic byte cap")
            REQUESTS[rid] = events | {stage}
            USED += size
            LOGGER.warning("%s", text)
    except Exception:
        # Only diagnostic work is caught. Original validation/compute/transfer remains outside.
        with LOCK:
            if not DROPPED:
                DROPPED = True
                try:
                    LOGGER.warning('{"kind":"smg_epd_position_drop","pid":%d}', os.getpid())
                except Exception:
                    pass
