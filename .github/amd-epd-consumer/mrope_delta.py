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
