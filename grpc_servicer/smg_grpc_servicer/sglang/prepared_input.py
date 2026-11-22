"""Version-sensitive native image layout boundary, without a media fallback."""

from __future__ import annotations

import inspect
from functools import cache

LAYOUT = "sglang_native_manifest_v1"
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_TOKENS = 131072
MAX_IMAGES = 32
_ITEM_FIELDS = {
    "identity",
    "pad_values",
    "processor_fingerprint",
    "offsets",
    "format",
    "feature",
    "precomputed_embeddings",
    "model_specific_data",
}
_OUTPUT_FIELDS = {
    "mm_items",
    "input_ids",
    "padded_input_ids",
    "im_token_id",
    "im_start_id",
    "im_end_id",
    "mrope_positions",
    "mrope_position_delta",
    "token_type_ids",
}


@cache
def supports_routing_labels():
    """Return whether the installed SGLang `MultimodalDataItem` exposes `routing_label`."""
    from sglang.srt.managers.req_types import MultimodalDataItem

    return "routing_label" in MultimodalDataItem.__struct_fields__


def verify_contract(scheduler):
    """Require a native vision configuration and the layout export hooks."""
    from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalProcessorOutput

    if getattr(scheduler.model_config.hf_config, "vision_config", None) is None:
        raise ValueError("managed EPD requires a vision model")
    scheduler_type = type(scheduler)
    for cls, fields in (
        (MultimodalDataItem, _ITEM_FIELDS),
        (MultimodalProcessorOutput, _OUTPUT_FIELDS),
    ):
        missing = fields - set(cls.__struct_fields__)
        if missing:
            raise ValueError(f"managed EPD missing {cls.__name__} fields: {sorted(missing)}")
    for method in (
        "handle_generate_request",
        "_add_request_to_queue",
        "_get_multimodal_inputs",
        "_maybe_compute_mrope_positions",
    ):
        if not callable(getattr(scheduler_type, method, None)):
            raise ValueError(f"managed EPD missing Scheduler.{method}")
    if "req" not in inspect.signature(scheduler_type._add_request_to_queue).parameters:
        raise ValueError("managed EPD queue hook signature changed")
    handle_params = inspect.signature(scheduler_type.handle_generate_request).parameters
    if "mm_input_error" not in handle_params and not any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in handle_params.values()
    ):
        raise ValueError("managed EPD requires handle_generate_request(mm_input_error=...)")


def _proto():
    """Lazily import generated `sglang_epd_pb2` protobuf stubs."""
    from smg_grpc_proto.generated import sglang_epd_pb2

    return sglang_epd_pb2


def _integers(value, max_elements):
    """Serialize a bounded int32/int64 tensor into a `sglang_epd.IntegerArray` message."""
    import torch

    if not isinstance(value, torch.Tensor) or value.dtype not in (torch.int32, torch.int64):
        raise ValueError("native position metadata must be an integer tensor")
    if value.numel() > max_elements or value.ndim > 3:
        raise ValueError("native integer metadata exceeds its bound")
    return _proto().IntegerArray(
        shape=list(value.shape), values=value.detach().cpu().reshape(-1).tolist()
    )


def export_layout(req, binding, *, require_mrope=True):
    """Observe finalized P input before bootstrap; copy metadata, not embedding storage."""
    pb = _proto()
    mm = req.multimodal_inputs
    if req.to_finish is not None or mm is None or req.rid != binding.prefill_rid:
        raise ValueError("managed P has no successful final multimodal layout")
    unpadded = list(req.origin_input_ids_unpadded)
    padded = list(req.origin_input_ids)
    if not unpadded or len(unpadded) != len(padded) or len(padded) > MAX_TOKENS:
        raise ValueError("managed P native prompt lengths are invalid")
    if not 0 < len(mm.mm_items) <= MAX_IMAGES:
        raise ValueError("managed P image count exceeds capability")
    output = pb.PreparedDecodeInput(
        version=1,
        binding=binding,
        unpadded_ids=unpadded,
        padded_ids=padded,
        image_token_id=mm.im_token_id,
        image_start_id=mm.im_start_id or 0,
        image_end_id=mm.im_end_id or 0,
    )
    if require_mrope or mm.mrope_positions is not None or mm.mrope_position_delta is not None:
        output.mrope_positions.CopyFrom(_integers(mm.mrope_positions, MAX_TOKENS * 3))
        output.mrope_delta.CopyFrom(_integers(mm.mrope_position_delta, 1))
    for index, item in enumerate(mm.mm_items):
        if item.modality.name != "IMAGE":
            raise ValueError("managed EPD only supports image media")
        data = item.model_specific_data
        if set(data) - {"image_grid_thw"}:
            raise ValueError(f"unsupported managed image metadata: {sorted(data)}")
        grid = data.get("image_grid_thw", [])
        if hasattr(grid, "tolist"):
            grid = grid.tolist()
        if isinstance(grid, list) and len(grid) == 1 and isinstance(grid[0], list):
            grid = grid[0]
        if (
            not isinstance(grid, list)
            or len(grid) not in (0, 3)
            or any(type(v) is not int or v <= 0 for v in grid)
        ):
            raise ValueError("missing managed image grid")
        output.images.add(
            occurrence=index,
            full_identity=item.identity,
            pad_values=item.pad_values,
            processor_fingerprint=item.processor_fingerprint,
            offsets=[pb.Offset(start=start, end=end) for start, end in item.offsets],
            native_format=item.format.value,
            grid_thw=grid,
            routing_label=getattr(item, "routing_label", None),
            native_hash=item.hash,
        )
    token_types = getattr(mm, "token_type_ids", None)
    if token_types is None:
        token_types = getattr(req, "token_type_ids", None)
    if token_types is not None:
        output.token_types.CopyFrom(_integers(token_types, MAX_TOKENS))
    return output


def decode_input(manifest, *, require_mrope=True):
    """Construct metadata-only native input; no media IO or processor is reachable."""
    import torch
    from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalProcessorOutput
    from sglang.srt.multimodal.modality import Modality, MultimodalInputFormat

    if not supports_routing_labels() and any(
        item.HasField("routing_label") for item in manifest.images
    ):
        raise ValueError("native engine does not support media routing labels")
    items = [
        MultimodalDataItem(
            modality=Modality.IMAGE,
            identity=bytes(item.full_identity),
            hash=item.native_hash,
            pad_value=item.pad_values[0],
            pad_values=tuple(item.pad_values),
            processor_fingerprint=item.processor_fingerprint
            if item.HasField("processor_fingerprint")
            else None,
            **({"routing_label": item.routing_label} if item.HasField("routing_label") else {}),
            offsets=[(o.start, o.end) for o in item.offsets],
            format=MultimodalInputFormat(item.native_format),
            feature=None,
            precomputed_embeddings=None,
            keep_device_embedding=False,
            model_specific_data={
                "image_grid_thw": torch.tensor([list(item.grid_thw)], dtype=torch.int64)
            }
            if item.grid_thw
            else {},
        )
        for item in manifest.images
    ]

    def tensor(array):
        return torch.tensor(array.values, dtype=torch.int64).reshape(tuple(array.shape))

    output = MultimodalProcessorOutput(
        mm_items=items,
        input_ids=list(manifest.unpadded_ids),
        padded_input_ids=list(manifest.padded_ids),
        im_token_id=manifest.image_token_id,
        im_start_id=manifest.image_start_id or None,
        im_end_id=manifest.image_end_id or None,
    )
    if manifest.HasField("mrope_positions"):
        output.mrope_positions = tensor(manifest.mrope_positions)
        output.mrope_position_delta = tensor(manifest.mrope_delta)
    elif require_mrope:
        raise ValueError("managed model requires native M-RoPE positions")
    if manifest.HasField("token_types"):
        output.token_type_ids = tensor(manifest.token_types)
    return output
