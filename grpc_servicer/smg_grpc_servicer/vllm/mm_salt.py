"""Engine-free multimodal helpers for tensor-stripped (PD decode) legs."""

import logging
from collections.abc import Sequence

logger = logging.getLogger(__name__)


def engine_accepts_mm_inputs(model_config) -> bool:
    """Whether the engine runs a vision encoder for pixel payloads.

    ``is_multimodal_model`` only reports the architecture: it stays true
    under ``--language-model-only``, which zeroes every modality limit and
    drops the vision encoder (encoder-cache budget 0), and it is true for
    ``enable_mm_embeds``-only models that ingest pre-computed embeddings but
    cannot encode pixels. The router reads this through the worker's
    ``supports_vision`` label to decide whether a decode leg may carry an mm
    payload, so the answer must come from vLLM's own check. That check moved
    between releases:

    - vLLM main: ``ModelConfig.supports_multimodal_inputs`` property;
    - vLLM 0.19-0.20 (the servicer's supported range): the equivalent
      ``MULTIMODAL_REGISTRY.supports_multimodal_inputs(model_config)``.
    """
    supports = getattr(model_config, "supports_multimodal_inputs", None)
    if supports is None:
        supports = _registry_supports_multimodal_inputs(model_config)
    if not supports:
        return False
    # vLLM can accept embeddings or audio while all visual inputs are
    # disabled. Neither implies an encoder for the pixel payloads that
    # supports_vision speaks for.
    return not _mm_embeds_leave_no_visual_encoder(model_config)


def _registry_supports_multimodal_inputs(model_config) -> bool:
    """vLLM 0.19-0.20 fallback: the check lives on the multimodal registry."""
    if not model_config.is_multimodal_model:
        return False
    try:
        from vllm.multimodal import MULTIMODAL_REGISTRY
    except ImportError:
        # Engine-free context (unit tests): the architecture is all we have.
        return True
    try:
        return bool(MULTIMODAL_REGISTRY.supports_multimodal_inputs(model_config))
    except ValueError:
        # No registered processor for this architecture: text-only.
        return False
    except Exception:
        # An unexpected probe failure keeps the previous
        # architecture-based answer rather than disabling a healthy
        # full-vision worker.
        logger.warning(
            "supports_multimodal_inputs probe failed; reporting the architecture's "
            "multimodal capability",
            exc_info=True,
        )
        return True


def _mm_embeds_leave_no_visual_encoder(model_config) -> bool:
    """Whether enabled mm embeds leave no image/video encoder enabled.

    True when ``enable_mm_embeds`` is on and every supported visual modality
    has an effective limit of 0. Explicit limits alone are insufficient:
    omitted modalities retain vLLM's default limit. Audio support does not
    imply that the engine accepts pixel payloads.
    """
    mm_config = getattr(model_config, "multimodal_config", None)
    if mm_config is None or not getattr(mm_config, "enable_mm_embeds", False):
        return False
    if getattr(mm_config, "language_model_only", False):
        return True
    get_limit = getattr(mm_config, "get_limit_per_prompt", None)
    if get_limit is None:
        return False
    try:
        from vllm.multimodal import MULTIMODAL_REGISTRY
    except ImportError:
        # Without the model's supported modalities, do not infer disabled
        # visual inputs from a potentially partial set of explicit limits.
        return False
    try:
        # Use the same modality set as vLLM's own capability check. Prefer
        # the public API, with a fallback for vLLM 0.19.
        get_info = getattr(MULTIMODAL_REGISTRY, "get_processing_info", None)
        if get_info is not None:
            info = get_info(model_config)
        else:
            info = MULTIMODAL_REGISTRY._create_processing_info(model_config, tokenizer=None)
        return all(
            get_limit(modality) == 0
            for modality in info.supported_mm_limits
            if modality in {"image", "video"}
        )
    except Exception:
        logger.warning(
            "Multimodal limits probe failed; preserving the engine's multimodal capability",
            exc_info=True,
        )
        return False


def has_preprocessed_mm_payload(mm_inputs) -> bool:
    """True when the payload carries tensors the preprocessed path can use.

    A grid-only payload (model-specific tensors, no pixels) is the PD decode
    leg's form; a bare identity payload (hashes only) is not preprocessable
    and falls back to the cache-salt path.
    """
    return mm_inputs.HasField("pixel_values") or bool(mm_inputs.model_specific_tensors)


def mm_identity_cache_salt(mm_hashes: Sequence[str]) -> str | None:
    """Fold per-image content hashes into a deterministic cache salt.

    The PD router strips multimodal tensors from the decode leg (the KV
    arrives via the P/D transfer), keeping only the per-image content hashes.
    Without tensors no ``mm_features`` can be built, so the engine's
    prefix-cache block hashes would carry no image identity — the identity
    rides ``cache_salt`` instead. Deterministic per image content: same-image
    reuse still hits the decode prefix cache, while different images behind
    the same text prefix no longer alias onto each other's KV.
    """
    if not mm_hashes:
        return None
    return "mm:" + ",".join(mm_hashes)
