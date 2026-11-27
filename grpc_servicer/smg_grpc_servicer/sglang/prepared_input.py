"""Version-sensitive native image/video layout boundary, without a media fallback."""

from __future__ import annotations

import hashlib
import inspect

LAYOUT = "sglang_native_manifest_v1"
MAX_MANIFEST_BYTES = 32 * 1024 * 1024
MAX_TOKENS = 1_048_576
MAX_IMAGES = 1024
_ITEM_FIELDS = {
    "hash",
    "pad_value",
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


def _import_mm_item_types():
    """Import SGLang multimodal item/output types across v0.5.21 and newer layouts."""
    try:
        from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalProcessorOutput
    except ImportError:
        from sglang.srt.managers.schedule_batch import (
            MultimodalDataItem,
            MultimodalProcessorOutput,
        )
    return MultimodalDataItem, MultimodalProcessorOutput


def _import_mm_types():
    """Import SGLang multimodal item, output, modality, and format enums across releases."""
    MultimodalDataItem, MultimodalProcessorOutput = _import_mm_item_types()
    try:
        from sglang.srt.multimodal.modality import Modality, MultimodalInputFormat
    except ImportError:
        from sglang.srt.managers.schedule_batch import Modality, MultimodalInputFormat
    return MultimodalDataItem, MultimodalProcessorOutput, Modality, MultimodalInputFormat


def supports_routing_labels():
    """Return whether the installed SGLang MultimodalDataItem exposes ``routing_label``."""
    MultimodalDataItem, _ = _import_mm_item_types()
    return "routing_label" in MultimodalDataItem.__struct_fields__


def verify_contract(scheduler):
    """Require a native vision configuration and the layout export hooks."""
    MultimodalDataItem, MultimodalProcessorOutput = _import_mm_item_types()

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
    """Import the generated ``sglang_epd_pb2`` protobuf module lazily."""
    from smg_grpc_proto.generated import sglang_epd_pb2

    return sglang_epd_pb2


def _integers(value, max_elements):
    """Serialize a bounded int32/int64 tensor into a proto IntegerArray."""
    import torch

    if not isinstance(value, torch.Tensor) or value.dtype not in (torch.int32, torch.int64):
        raise ValueError("native position metadata must be an integer tensor")
    if value.numel() > max_elements or value.ndim > 3:
        raise ValueError("native integer metadata exceeds its bound")
    return _proto().IntegerArray(
        shape=list(value.shape), values=value.detach().cpu().reshape(-1).tolist()
    )


def _normalize_grid(grid):
    """Normalize an image/video ``grid_thw`` tensor or list into a 0- or 3-int list."""
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    if isinstance(grid, list) and len(grid) == 1 and isinstance(grid[0], list):
        grid = grid[0]
    if (
        not isinstance(grid, list)
        or len(grid) not in (0, 3)
        or any(type(v) is not int or v <= 0 for v in grid)
    ):
        raise ValueError("missing managed media grid")
    return grid


def _flatten_floats(raw):
    """Flatten optional video ``second_per_grid_ts`` metadata into a list of floats."""
    if raw is None:
        return []
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, (int, float)):
        return [float(raw)]
    if isinstance(raw, list):
        out = []
        for entry in raw:
            if isinstance(entry, list):
                out.extend(float(v) for v in entry)
            else:
                out.append(float(entry))
        return out
    raise ValueError("invalid managed video timing metadata")


def export_layout(req, binding, *, require_mrope=True):
    """Observe finalized P input before bootstrap; copy metadata, not embedding storage."""
    pb = _proto()
    mm = req.multimodal_inputs
    if req.to_finish is not None or mm is None or req.rid != binding.prefill_rid:
        raise ValueError("managed P has no successful final multimodal layout")
    unpadded, padded = list(req.origin_input_ids_unpadded), list(req.origin_input_ids)
    if not unpadded or len(unpadded) != len(padded):
        raise ValueError(
            f"managed P native prompt lengths are invalid: unpadded={len(unpadded)}, padded={len(padded)}"
        )
    if len(padded) > MAX_TOKENS:
        raise ValueError(
            f"managed P native prompt length ({len(padded)}) exceeds limit ({MAX_TOKENS})"
        )
    if not 0 < len(mm.mm_items) <= MAX_IMAGES:
        raise ValueError("managed P media count exceeds capability")
    output = pb.PreparedDecodeInput(
        version=1,
        binding=binding,
        unpadded_ids=unpadded,
        padded_ids=padded,
        image_token_id=mm.im_token_id or 0,
        image_start_id=mm.im_start_id or 0,
        image_end_id=mm.im_end_id or 0,
        video_token_id=getattr(mm, "video_token_id", None) or 0,
    )
    if require_mrope or mm.mrope_positions is not None or mm.mrope_position_delta is not None:
        output.mrope_positions.CopyFrom(_integers(mm.mrope_positions, MAX_TOKENS * 3))
        output.mrope_delta.CopyFrom(_integers(mm.mrope_position_delta, 1))
    for index, item in enumerate(mm.mm_items):
        if item.modality.name not in ("IMAGE", "VIDEO"):
            raise ValueError("managed EPD only supports image and video media")
        is_video = item.modality.name == "VIDEO"
        grid_key = "video_grid_thw" if is_video else "image_grid_thw"
        allowed_keys = {"video_grid_thw", "second_per_grid_ts"} if is_video else {"image_grid_thw"}
        data = item.model_specific_data
        if set(data) - allowed_keys:
            raise ValueError(f"unsupported managed media metadata: {sorted(data)}")
        raw_identity = getattr(item, "identity", None)
        if isinstance(raw_identity, (bytes, bytearray)) and len(raw_identity) == 32:
            full_identity = bytes(raw_identity)
        else:
            if item.hash is None or item.pad_value is None:
                raise ValueError("managed P media item lacks identity and hash/pad_value")
            full_identity = hashlib.sha256(
                f"smg-epd-v0521-item:{item.modality.name}:{item.hash}:{item.pad_value}".encode()
            ).digest()
        raw_pad_values = getattr(item, "pad_values", None)
        if raw_pad_values:
            pad_values = list(raw_pad_values)
        elif item.pad_value is not None:
            pad_values = [item.pad_value]
        else:
            raise ValueError("managed P media item lacks pad_value")
        output.images.add(
            occurrence=index,
            full_identity=full_identity,
            pad_values=pad_values,
            processor_fingerprint=getattr(item, "processor_fingerprint", None),
            offsets=[pb.Offset(start=start, end=end) for start, end in item.offsets],
            native_format=item.format.value,
            grid_thw=_normalize_grid(data.get(grid_key, [])),
            routing_label=getattr(item, "routing_label", None),
            native_hash=item.hash,
            modality=2 if is_video else 1,
            second_per_grid_ts=_flatten_floats(
                data.get("second_per_grid_ts") if is_video else None
            ),
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

    MultimodalDataItem, MultimodalProcessorOutput, Modality, MultimodalInputFormat = (
        _import_mm_types()
    )
    raw_item_fields = getattr(MultimodalDataItem, "__struct_fields__", None)
    item_fields = set(raw_item_fields) if raw_item_fields is not None else None

    if not supports_routing_labels() and any(
        item.HasField("routing_label") for item in manifest.images
    ):
        raise ValueError("native engine does not support media routing labels")
    items = []
    for item in manifest.images:
        is_video = getattr(item, "modality", 1) == 2
        grid_key = "video_grid_thw" if is_video else "image_grid_thw"
        model_specific_data = (
            {grid_key: torch.tensor([list(item.grid_thw)], dtype=torch.int64)}
            if item.grid_thw
            else {}
        )
        if is_video and getattr(item, "second_per_grid_ts", None):
            model_specific_data["second_per_grid_ts"] = torch.tensor(
                list(item.second_per_grid_ts), dtype=torch.float32
            )
        extra_item = {}
        if item_fields is None or "identity" in item_fields:
            extra_item["identity"] = bytes(item.full_identity)
        if item_fields is None or "pad_values" in item_fields:
            extra_item["pad_values"] = tuple(item.pad_values)
        if item_fields is None or "processor_fingerprint" in item_fields:
            extra_item["processor_fingerprint"] = (
                item.processor_fingerprint if item.HasField("processor_fingerprint") else None
            )
        if item.HasField("routing_label"):
            extra_item["routing_label"] = item.routing_label
        items.append(
            MultimodalDataItem(
                modality=Modality.VIDEO if is_video else Modality.IMAGE,
                hash=item.native_hash,
                pad_value=item.pad_values[0],
                offsets=[(o.start, o.end) for o in item.offsets],
                format=MultimodalInputFormat(item.native_format),
                feature=None,
                precomputed_embeddings=None,
                keep_device_embedding=False,
                model_specific_data=model_specific_data,
                **extra_item,
            )
        )

    def tensor(array):
        return torch.tensor(array.values, dtype=torch.int64).reshape(tuple(array.shape))

    extra_output = (
        {"video_token_id": getattr(manifest, "video_token_id", 0) or None}
        if "video_token_id" in MultimodalProcessorOutput.__struct_fields__
        else {}
    )
    output = MultimodalProcessorOutput(
        mm_items=items,
        input_ids=list(manifest.unpadded_ids),
        padded_input_ids=list(manifest.padded_ids),
        im_token_id=manifest.image_token_id or None,
        im_start_id=manifest.image_start_id or None,
        im_end_id=manifest.image_end_id or None,
        **extra_output,
    )
    if manifest.HasField("mrope_positions"):
        output.mrope_positions = tensor(manifest.mrope_positions)
        output.mrope_position_delta = tensor(manifest.mrope_delta)
    elif require_mrope:
        raise ValueError("managed model requires native M-RoPE positions")
    if manifest.HasField("token_types"):
        output.token_type_ids = tensor(manifest.token_types)
    return output
