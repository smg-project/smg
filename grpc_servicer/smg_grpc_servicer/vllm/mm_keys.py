"""Wire-key helpers for preprocessed multimodal payloads (engine-free).

The router serialises the primary encoder tensor in the proto's ``pixel_values``
field regardless of what the model's forward pops it as. Most vision models
take ``pixel_values``; DeepSeek-V4.1 takes ``patches``. The router names the
tensor through ``MultimodalInputs.encoder_input_key`` and uses the same name in
``batched_keys`` / ``flat_keys``, so the servicer must register the tensor under
that key for the field configs to line up.
"""

from __future__ import annotations

from smg_grpc_proto.generated import common_pb2

DEFAULT_ENCODER_INPUT_KEY = "pixel_values"


def mm_batches(request) -> list:
    """Every multimodal batch on a GenerateRequest, ``mm_inputs`` first and
    then ``extra_mm_inputs`` (a request mixing image and video carries one
    batch per modality). A stub built from an older proto has no
    ``extra_mm_inputs``; it reads as none.
    """
    batches = [request.mm_inputs] if request.HasField("mm_inputs") else []
    batches.extend(getattr(request, "extra_mm_inputs", ()))
    return batches


def describes_media_twice(request) -> bool:
    """Whether a request brings both media references and batches of its own.

    Only one of the two is ever read, so the other would be dropped without a
    word. A request asking for both has not said what it wants.
    """
    return request.HasField("media_refs") and bool(mm_batches(request))


def batches_missing_pixels(batches) -> list:
    """The batches carrying no encoder tensor of their own.

    Each modality is encoded from its own batch, so one batch standing in for
    another is not enough: whichever batch is listed here would leave its
    encoder with nothing to read.
    """
    return [batch for batch in batches if not batch.HasField("pixel_values")]


def mm_identity_hashes(batches) -> list:
    """Every batch's media hashes, in order.

    These name the media a request carries. Leaving a batch's hashes out would
    let two requests differing only in that batch's media read as the same one.
    """
    return [mm_hash for batch in batches for mm_hash in batch.mm_hashes]


def modality_name(mm_proto) -> str:
    """vLLM's name for the batch's modality: ``video`` or ``image``."""
    return "video" if mm_proto.modality == common_pb2.VIDEO else "image"


def primary_encoder_key(mm_proto) -> str:
    """The HF kwarg name for the tensor carried in ``mm_proto.pixel_values``.

    ``encoder_input_key`` is proto field 11; a ``smg-grpc-proto`` stub built
    from an older proto (releases up to 0.4.18) has no such attribute, and
    that must read as the default rather than break every multimodal request.
    """
    return getattr(mm_proto, "encoder_input_key", "") or DEFAULT_ENCODER_INPUT_KEY


def modality_key(key: str, is_video: bool) -> str:
    """vLLM routes video pixels through ``pixel_values_videos``; every other
    key (including a renamed primary tensor) is used as-is."""
    if is_video and key == DEFAULT_ENCODER_INPUT_KEY:
        return "pixel_values_videos"
    return key
