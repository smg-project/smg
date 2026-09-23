"""The prefill leg's account of the media it processed, for the decode leg.

With prefill/decode disaggregation only the prefill worker runs the media
processor; its result travels back on the completion in the shape a
router-preprocessed decode leg already accepts: expanded prompt ids and, per
modality, hashes, placeholders and the grid tensors, never pixels. Engine-free:
the processed input is read by shape, not by vLLM's types.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from smg_grpc_proto import vllm_engine_pb2
from smg_grpc_proto.generated import common_pb2

from smg_grpc_servicer.tensor_wire import PROTO_DTYPE_MAP

logger = logging.getLogger(__name__)

# The per-item tensors a decode leg needs besides pixels: the M-RoPE grids and
# the video timing that goes with them, in both spellings: `second_per_grid_ts`
# from the Qwen-VL processors, `video_second_per_grid` from the Qwen-Omni
# family and the router's own processors. The router's VLLM_MROPE_GRID_KEYS
# mirrors this list.
GRID_KEYS = ("image_grid_thw", "video_grid_thw", "second_per_grid_ts", "video_second_per_grid")

_MODALITIES = {"image": common_pb2.IMAGE, "video": common_pb2.VIDEO, "audio": common_pb2.AUDIO}
_WIRE_DTYPES = {dtype: name for name, dtype in PROTO_DTYPE_MAP.items()}


def tensor_to_proto(tensor: torch.Tensor) -> vllm_engine_pb2.TensorData:
    """A grid-sized tensor as inline wire bytes, widened to a dtype the wire
    names. Bytes go through a Python list: no numpy, and grids are tiny."""
    data = tensor.detach().cpu()
    if data.dtype not in _WIRE_DTYPES:
        data = data.to(torch.float32 if data.is_floating_point() else torch.int64)
    data = data.contiguous()
    return vllm_engine_pb2.TensorData(
        shape=list(data.shape),
        dtype=_WIRE_DTYPES[data.dtype],
        inline=bytes(data.flatten().view(torch.uint8).tolist()),
    )


def _first(prompt: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in prompt:
            return prompt[name]
    return None


def _data(elem: Any) -> torch.Tensor:
    return torch.as_tensor(getattr(elem, "data", elem))


def _im_token_id(prompt_token_ids: Sequence[int], ranges: Sequence[Any]) -> int | None:
    """The token embeddings land on, when a placeholder mixes it with others."""
    for r in ranges:
        is_embed = getattr(r, "is_embed", None)
        if is_embed is None:
            continue
        mask = torch.as_tensor(is_embed).flatten().bool()
        if bool(mask.all()) or not bool(mask.any()):
            continue
        first = int(mask.nonzero()[0])
        return int(prompt_token_ids[r.offset + first])
    return None


def build_media_identity(prompt: Mapping[str, Any]) -> vllm_engine_pb2.MediaIdentity | None:
    """The identity of a processed engine input, or None when it has no media
    or a grid tensor cannot be carried (the decode leg then reprocesses)."""
    hashes = _first(prompt, "mm_hashes", "multi_modal_hashes") or {}
    placeholders = _first(prompt, "mm_placeholders", "multi_modal_placeholders") or {}
    kwargs = _first(prompt, "mm_kwargs", "multi_modal_kwargs") or {}
    prompt_token_ids = list(prompt["prompt_token_ids"])
    if not hashes:
        return None

    batches = []
    for modality, modality_hashes in hashes.items():
        mm = vllm_engine_pb2.MultimodalInputs(
            modality=_MODALITIES.get(modality, common_pb2.MODALITY_UNSPECIFIED),
            mm_hashes=list(modality_hashes),
        )
        ranges = list(placeholders.get(modality, []) or [])
        mm.mm_placeholders.extend(
            vllm_engine_pb2.PlaceholderRange(offset=int(r.offset), length=int(r.length))
            for r in ranges
        )
        im_token_id = _im_token_id(prompt_token_ids, ranges)
        if im_token_id is not None:
            mm.im_token_id = im_token_id

        items = list(kwargs.get(modality, []) or [])
        for key in GRID_KEYS:
            elems = [item[key] for item in items if key in item]
            if not elems:
                continue
            tensors = [_data(elem) for elem in elems]
            if len(tensors) != len(items) or any(t.shape != tensors[0].shape for t in tensors):
                # Not one row per item: the batched layout the decode leg
                # reads cannot carry it, so no identity at all.
                logger.warning(
                    "media identity: %s grid tensor %s is not one row per item; "
                    "the decode leg will reprocess the media",
                    modality,
                    key,
                )
                return None
            mm.model_specific_tensors[key].CopyFrom(tensor_to_proto(torch.stack(tensors)))
            mm.batched_keys.append(key)
            if any(getattr(getattr(elem, "field", None), "keep_on_cpu", False) for elem in elems):
                mm.keep_on_cpu_keys.append(key)
        batches.append(mm)

    identity = vllm_engine_pb2.MediaIdentity(prompt_token_ids=prompt_token_ids)
    identity.mm_inputs.CopyFrom(batches[0])
    identity.extra_mm_inputs.extend(batches[1:])
    return identity
