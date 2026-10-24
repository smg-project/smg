"""Default-off, scalar-only Qwen EPD metadata; no image features or GPU reads."""

import os

TAG = 0x4D525001


def enabled():
    return (os.environ.get("SMG_EPD_MROPE_DELTA_HANDOFF") == "1"
            and os.environ.get("SMG_EPD_POSITION_MODEL") == "Qwen/Qwen3.5-9B")


def require(ok, reason):
    if not ok:
        raise ValueError("EPD MROPE: " + reason)


def scalar(value):
    require(type(value) is int and -(2**31) <= value < 2**31, "delta must fit int32")
    return value


def metadata_row(row):
    require(tuple(row.shape) == (16,) and str(row.dtype) == "torch.int32"
            and getattr(row.device, "type", None) == "cpu", "expected CPU int32[16] row")


def write(row, req):
    if not enabled():
        return
    metadata_row(row)
    row[7], row[8] = 0, 0
    mm = req.multimodal_inputs
    delta = getattr(mm, "mrope_position_delta", None)
    if delta is None:
        require(row[4].item() == 0, "image handoff has no MROPE delta")
        return
    require(getattr(delta.device, "type", None) == "cpu" and delta.numel() == 1
            and str(delta.dtype) == "torch.int64", "expected CPU int64 scalar delta")
    value = scalar(delta.item())
    row[7], row[8] = TAG, value


def restore(row, req):
    if not enabled():
        return
    req.pd_mrope_delta = req.pd_mrope_prompt_len = None
    metadata_row(row)
    tag, value = row[7].item(), row[8].item()
    if tag == 0:
        require(value == 0 and row[4].item() == 0, "missing versioned image delta")
        return
    require(tag == TAG, "unknown metadata version")
    req.pd_mrope_delta = scalar(value)
    req.pd_mrope_prompt_len = len(req.origin_input_ids)


def request_delta(req):
    if not enabled() or getattr(req, "pd_mrope_delta", None) is None:
        return None
    require(type(req.pd_mrope_prompt_len) is int
            and req.pd_mrope_prompt_len == len(req.origin_input_ids), "prompt length changed")
    return scalar(req.pd_mrope_delta)


def decode_deltas(reqs):
    values = [request_delta(req) for req in reqs]
    return None if all(value is None for value in values) else [value or 0 for value in values]


def tail_delta(req, prefix, length):
    value = request_delta(req)
    if value is None:
        return 0
    require(type(prefix) is type(length) is int and length > 0
            and prefix >= req.pd_mrope_prompt_len, "prompt re-prefill needs full positions")
    return value


def reject_spec(reqs):
    require(decode_deltas(reqs) is None, "delta-only speculative decode is unsupported")
