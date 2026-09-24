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
    # vLLM counts enable_mm_embeds-only engines as accepting multimodal
    # inputs (they ingest embeddings), but they have no encoder for the pixel
    # payloads supports_vision speaks for.
    return not _mm_embeds_only(model_config)


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


def _mm_embeds_only(model_config) -> bool:
    """Whether the engine ingests only pre-computed embeddings (no encoder).

    True when ``enable_mm_embeds`` is on and every explicitly listed modality
    limit is 0 — including the ``--language-model-only`` case, where
    ``get_limit_per_prompt`` reads 0 for every modality.
    """
    mm_config = getattr(model_config, "multimodal_config", None)
    if mm_config is None or not getattr(mm_config, "enable_mm_embeds", False):
        return False
    limits = getattr(mm_config, "limit_per_prompt", None) or {}
    get_limit = getattr(mm_config, "get_limit_per_prompt", None)
    if get_limit is None:
        return False
    if getattr(mm_config, "language_model_only", False):
        return True
    return bool(limits) and all(get_limit(modality) == 0 for modality in limits)


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
