"""Wire-key helpers for preprocessed multimodal payloads (engine-free).

The router serialises the primary encoder tensor in the proto's ``pixel_values``
field regardless of what the model's forward pops it as. Most vision models
take ``pixel_values``; DeepSeek-V4.1 takes ``patches``. The router names the
tensor through ``MultimodalInputs.encoder_input_key`` and uses the same name in
``batched_keys`` / ``flat_keys``, so the servicer must register the tensor under
that key for the field configs to line up.
"""

from __future__ import annotations

DEFAULT_ENCODER_INPUT_KEY = "pixel_values"


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
