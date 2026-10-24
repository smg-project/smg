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
        _local.items, _local.batch = {}, 0
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
    current = state()
    if ((str(rid) not in current["rids"] and len(current["rids"]) >= 9)
            or current["records"] >= 64 or current["counts"].get(str(rid), 0) >= 8):
        emit(rid, stage, {})
        return
    try:
        emit(rid, stage, dict(source=origin(module, path), **operation()))
    except Exception as error:
        emit(rid, stage, dict(error=type(error).__name__ + ":" + str(error)[:160]))


def wire(rid, mm, model_path, module, path):
    def fields():
        if len(mm.mm_items) > 4:
            raise ValueError("ITEM_CAP")
        result = []
        for item in mm.mm_items:
            info = item_info(item)
            feature = item.feature
            if feature.device.type != "cpu":
                raise ValueError("WIRE_PIXELS_NOT_CPU")
            offset, slices = 0, []
            if len(info["grid"]) != len(info["offsets"]):
                raise ValueError("GRID_OFFSET_COUNT")
            for ordinal, (t, h, width) in enumerate(info["grid"]):
                length = t * h * width
                if min(t, h, width) <= 0 or offset + length > feature.shape[0]:
                    raise ValueError("GRID_PIXEL_RANGE")
                slices.append(dict(ordinal=ordinal, start=offset, rows=length,
                                   tensor=tensor(feature.narrow(0, offset, length))))
                offset += length
            if offset != feature.shape[0]:
                raise ValueError("GRID_PIXEL_TOTAL")
            result.append(dict(info, feature_shape=list(feature.shape),
                               feature_dtype=str(feature.dtype), ordinals=slices))
        return dict(model_path=str(model_path)[:256], items=result)
    observe(rid, "wire_cpu", module, path, fields)


def positions(*args):
    if not enabled():
        return
    try:
        _positions(*args)
    except Exception as error:
        emit("UNASSOCIATED", "positions_cpu", dict(error=type(error).__name__ + ":" + str(error)[:160]))


def _positions(reqs, mm_inputs, values, prefixes, lengths, model_path, module, path):
    current = state()
    current["batch"] += 1
    _local.batch, _local.items = current["batch"], {}
    if not (len(reqs) == len(mm_inputs) == len(values) == len(prefixes) == len(lengths)):
        emit("UNASSOCIATED", "positions_cpu", dict(error="BATCH_LENGTHS"))
        return
    for req, mm, value, prefix, length in zip(reqs[:9], mm_inputs, values, prefixes, lengths):
        if mm is None:
            continue
        rid = req.rid
        def fields():
            if len(mm.mm_items) > 4:
                raise ValueError("ITEM_CAP")
            items = [item_info(item) for item in mm.mm_items]
            for item in items:
                previous = _local.items.get(item["item_id"])
                owner = rid if previous is None or previous[0] == rid else None
                _local.items[item["item_id"]] = (owner, item)
            return dict(batch=_local.batch, model_path=str(model_path)[:256], items=items,
                        prefix=prefix, extend=length, table=tensor(value),
                        delta=tensor(mm.mrope_position_delta))
        observe(rid, "positions_cpu", module, path, fields)
    if len(reqs) > 9:
        emit("UNASSOCIATED", "positions_cpu", dict(error="BATCH_CAP"))


def selection(items, key, cached, expected_rows, branch, module, path):
    if not enabled():
        return
    state()  # A fork cannot reuse its parent's current-batch association.
    rid = "UNASSOCIATED"
    try:
        infos = [item_info(item) for item in items[:4]]
        links = [getattr(_local, "items", {}).get(info["item_id"]) for info in infos]
        if len(items) > 4 or not links or any(link is None or link[0] is None for link in links):
            raise ValueError("NO_CURRENT_BATCH_ITEM_LINK")
        if any(link[1] != info for link, info in zip(links, infos)):
            raise ValueError("CURRENT_ITEM_IDENTITY_CHANGED")
        rids = {str(link[0]) for link in links}
        if len(rids) != 1:
            raise ValueError("MULTIPLE_REQUEST_LINKS")
        rid = next(iter(rids))
        def fields():
            emb = cached.embedding if cached is not None else None
            shape = list(emb.shape) if emb is not None else None
            return dict(batch=_local.batch, branch=branch, key=key, hit=cached is not None,
                        expected_rows=expected_rows, items=infos, embedding_shape=shape,
                        embedding_device=str(emb.device) if emb is not None else None,
                        embedding_values="NOT_READ", association="current_thread_batch_items")
        observe(rid, "cache_selection", module, path, fields)
    except Exception as error:
        emit(rid, "cache_selection", dict(error=type(error).__name__ + ":" + str(error)[:160]))
