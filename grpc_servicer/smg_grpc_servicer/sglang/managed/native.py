"""Version-sensitive native image layout boundary, without a media fallback.

Field compatibility does not prove Rust processor parity. Model/weight and
transport capabilities must be validated separately before registration.
"""

from __future__ import annotations

import hashlib
import inspect
import math

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
    "routing_label",
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


def verify_contract(model_type, item_type, output_type, scheduler_type):
    """Fail startup on an unknown model or missing native layout/hook boundary."""
    if model_type != "qwen3_vl":
        raise ValueError(f"managed EPD has no native layout adapter for {model_type}")
    for cls, fields in ((item_type, _ITEM_FIELDS), (output_type, _OUTPUT_FIELDS)):
        missing = fields - set(cls.__struct_fields__)
        if missing:
            raise ValueError(f"managed EPD missing {cls.__name__} fields: {sorted(missing)}")
    for method in (
        "handle_generate_request",
        "_add_request_to_queue",
        "_maybe_compute_mrope_positions",
    ):
        if not callable(getattr(scheduler_type, method, None)):
            raise ValueError(f"managed EPD missing Scheduler.{method}")
    if "req" not in inspect.signature(scheduler_type._add_request_to_queue).parameters:
        raise ValueError("managed EPD queue hook signature changed")


def _proto():
    """Load the new control contract only for managed workers."""
    from smg_grpc_proto.generated import sglang_managed_pb2

    return sglang_managed_pb2


def _integers(value, max_elements):
    """Copy small integer metadata, never a device pointer or floating tensor."""
    import torch

    if not isinstance(value, torch.Tensor) or value.dtype not in (torch.int32, torch.int64):
        raise ValueError("native position metadata must be an integer tensor")
    if value.numel() > max_elements or value.ndim > 3:
        raise ValueError("native integer metadata exceeds its bound")
    return _proto().IntegerArray(
        shape=list(value.shape), values=value.detach().cpu().reshape(-1).tolist()
    )


def digest(manifest):
    """Use the deterministic protobuf representation shared with the Rust validator."""
    unsigned = type(manifest)()
    unsigned.CopyFrom(manifest)
    unsigned.ClearField("integrity_digest")
    return hashlib.sha256(unsigned.SerializeToString(deterministic=True)).digest()


def export_layout(req, binding):
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
        mrope_positions=_integers(mm.mrope_positions, MAX_TOKENS * 3),
        mrope_delta=_integers(mm.mrope_position_delta, 1),
    )
    for index, item in enumerate(mm.mm_items):
        if item.modality.name != "IMAGE":
            raise ValueError("managed EPD only supports image media")
        data = item.model_specific_data
        if set(data) - {"image_grid_thw"}:
            raise ValueError(f"unsupported managed image metadata: {sorted(data)}")
        # Native Qwen EPD consumes the grid when computing M-RoPE and leaves
        # model_specific_data empty. Preserve that absence, without deriving
        # a replacement grid or running a processor on D.
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
            routing_label=item.routing_label,
            native_hash=item.hash,
        )
    token_types = getattr(req, "token_type_ids", None)
    if token_types is not None:
        output.token_types.CopyFrom(_integers(token_types, MAX_TOKENS))
    output.integrity_digest = digest(output)
    validate_layout(output, binding)
    return output


def _validate_array(value, limit):
    """Check shape before constructing a native tensor from integer metadata."""
    if (
        len(value.shape) > 3
        or math.prod(value.shape) != len(value.values)
        or len(value.values) > limit
    ):
        raise ValueError("invalid managed integer metadata shape")


def validate_layout(manifest, binding):
    """Validate the final scheduler boundary, including full native padding identity."""
    if manifest.ByteSize() > MAX_MANIFEST_BYTES:
        raise ValueError("managed manifest exceeds byte bound")
    if manifest.version != 1 or manifest.binding != binding or binding.layout_contract != LAYOUT:
        raise ValueError("managed manifest binding/version mismatch")
    if manifest.integrity_digest != digest(manifest):
        raise ValueError("managed manifest integrity mismatch")
    count = len(manifest.unpadded_ids)
    if not 0 < count <= MAX_TOKENS or len(manifest.padded_ids) != count:
        raise ValueError("invalid managed native prompt length")
    if not 0 < len(manifest.images) <= MAX_IMAGES or not manifest.image_token_id:
        raise ValueError("invalid managed image count/token")
    _validate_array(manifest.mrope_positions, MAX_TOKENS * 3)
    _validate_array(manifest.mrope_delta, 1)
    if list(manifest.mrope_positions.shape) != [3, count]:
        raise ValueError("missing managed native positions")
    if len(manifest.mrope_delta.values) != 1:
        raise ValueError("missing managed native position delta")
    if manifest.HasField("token_types"):
        _validate_array(manifest.token_types, MAX_TOKENS)
        if len(manifest.token_types.values) != count:
            raise ValueError("invalid managed token types")
    previous_end = -1
    for index, item in enumerate(manifest.images):
        if (
            item.occurrence != index
            or len(item.full_identity) != 32
            or (item.HasField("processor_fingerprint") and not item.processor_fingerprint)
        ):
            raise ValueError("invalid managed full media identity")
        if item.native_hash != int.from_bytes(item.full_identity[:8], "big"):
            raise ValueError("native hash differs from full media identity")
        if len(item.routing_label) > 256:
            raise ValueError("native routing label exceeds limit")
        if binding.media_routing_labels and (
            index >= len(binding.media_routing_labels)
            or item.routing_label != binding.media_routing_labels[index]
        ):
            raise ValueError("native routing label differs from frozen input binding")
        hashed = hashlib.sha256(b"sglang-mm-wide-pad-v1" + item.full_identity).digest()
        pads = [
            1000000 + (int.from_bytes(hashed[i : i + 8], "big") & ((1 << 62) - 1))
            for i in range(0, 32, 8)
        ]
        if (
            list(item.pad_values) != pads
            or len(item.grid_thw) not in (0, 3)
            or any(v <= 0 for v in item.grid_thw)
        ):
            raise ValueError("managed identity/padding/grid mismatch")
        if not item.offsets or item.native_format not in (1, 2, 3):
            raise ValueError("incomplete managed native image layout")
        for offset in item.offsets:
            if not previous_end < offset.start <= offset.end < count:
                raise ValueError("invalid managed image offsets")
            if any(v not in pads for v in manifest.padded_ids[offset.start : offset.end + 1]):
                raise ValueError("managed image sentinel mismatch")
            previous_end = offset.end


def decode_input(manifest, binding):
    """Construct metadata-only native input; no media IO or processor is reachable."""
    import torch
    from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalProcessorOutput
    from sglang.srt.multimodal.modality import Modality, MultimodalInputFormat

    validate_layout(manifest, binding)
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
            routing_label=item.routing_label if item.HasField("routing_label") else None,
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
    output = MultimodalProcessorOutput(
        mm_items=items,
        input_ids=list(manifest.unpadded_ids),
        padded_input_ids=list(manifest.padded_ids),
        im_token_id=manifest.image_token_id,
        im_start_id=manifest.image_start_id or None,
        im_end_id=manifest.image_end_id or None,
        mrope_positions=torch.tensor(
            list(manifest.mrope_positions.values), dtype=torch.int64
        ).reshape(list(manifest.mrope_positions.shape)),
        mrope_position_delta=torch.tensor(
            list(manifest.mrope_delta.values), dtype=torch.int64
        ).reshape(list(manifest.mrope_delta.shape)),
    )
    if manifest.HasField("token_types"):
        output.token_type_ids = torch.tensor(
            list(manifest.token_types.values), dtype=torch.int64
        ).reshape(list(manifest.token_types.shape))
    return output
