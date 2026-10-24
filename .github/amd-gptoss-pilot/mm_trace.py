"""Opt-in CPU observations; absent/error/limit records never establish quality."""

import hashlib
import json
import logging
import os
import sys
import threading

_state = {}
_local = threading.local()
_origins = {}


def enabled():
    return (os.environ.get("SMG_QWEN_MM_TRACE") == "1"
            and os.environ.get("SMG_QWEN_MM_MODEL") == "Qwen/Qwen3-VL-8B-Instruct")


def state():
    if _state.get("pid") != os.getpid():
        _state.clear()
        _state.update(pid=os.getpid(), rids=[], counts={}, records=0, hash_bytes=0,
                      batch=0, limit=False)
    return _state


def emit(rid, stage, fields):
    current = state()
    rid = str(rid)
    reason = None
    if not rid or len(rid) > 160:
        reason = "RID_INVALID"
    elif rid not in current["rids"]:
        if len(current["rids"]) >= 9:
            reason = "RID_LIMIT"
        else:
            current["rids"].append(rid)
    if current["records"] >= 64 or current["counts"].get(rid, 0) >= 8:
        reason = "RECORD_LIMIT"
    if reason:
        if not current["limit"]:
            logging.getLogger(__name__).warning("SMG_QWEN_MM_TRACE_LIMIT %s", reason)
            current["limit"] = True
        return
    row = dict(schema="smg_qwen_mm_v1", pid=os.getpid(), thread=threading.get_ident(),
               rid=rid, stage=stage, **fields)
    raw = json.dumps(row, separators=(",", ":"), allow_nan=False)
    if len(raw.encode()) > 4096:
        raw = json.dumps(dict(schema="smg_qwen_mm_v1", pid=os.getpid(), rid=rid,
                              stage=stage, error="RECORD_CAP"))
    logging.getLogger(__name__).warning("SMG_QWEN_MM_TRACE %s", raw)
    current["records"] += 1
    current["counts"][rid] = current["counts"].get(rid, 0) + 1


def origin(module, path):
    key = (os.getpid(), path)
    if key not in _origins:
        with open(path, "rb") as stream:
            raw = stream.read(524289)
        if len(raw) > 524288:
            raise ValueError("SOURCE_CAP")
        _origins[key] = dict(module=module, path=path, sha256=hashlib.sha256(raw).hexdigest())
    return _origins[key]


def tensor(value):
    if value is None:
        return None
    shape = list(value.shape)
    result = dict(shape=shape, dtype=str(value.dtype), device=str(value.device))
    if value.device.type != "cpu":
        return dict(result, observation="NON_CPU_NOT_READ")
    size = value.numel() * value.element_size()
    current = state()
    if size > 8 * 1024**2 or current["hash_bytes"] + size > 64 * 1024**2:
        return dict(result, error="HASH_CAP")
    if not value.is_contiguous():
        return dict(result, error="NONCONTIGUOUS_NOT_COPIED")
    # Both checks above precede all value access. This is a CPU byte view, not H2D/D2H.
    raw = memoryview(value.detach().reshape(-1).view(sys.modules["torch"].uint8).numpy())
    current["hash_bytes"] += size
    return dict(result, bytes=size, sha256=hashlib.sha256(raw).hexdigest())


def grid(item):
    value = getattr(item, "image_grid_thw", None)
    if value is None or value.device.type != "cpu" or tuple(value.shape)[-1:] != (3,):
        raise ValueError("GRID_NOT_CPU_THW")
    if value.numel() > 12:
        raise ValueError("GRID_CAP")
    return value.tolist()


def item_info(item):
    offsets = item.offsets or []
    if len(offsets) > 4:
        raise ValueError("OFFSET_CAP")
    return dict(item_id=id(item), hash=item.hash, pad=item.pad_value,
                offsets=offsets, grid=grid(item))


def observe(rid, stage, module, path, operation):
    if not enabled():
        return
    try:
        emit(rid, stage, dict(source=origin(module, path), **operation()))
    except Exception as error:
        emit(rid, stage, dict(error=type(error).__name__ + ":" + str(error)[:160]))
