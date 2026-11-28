"""Engine-free multimodal helpers for tensor-stripped (PD decode) sides, and the cache salt."""

import hashlib
import logging
from collections.abc import Sequence

logger = logging.getLogger(__name__)

# The reserved prefix, and the domain tag of the digest that follows it.
_RESERVED_SALT_PREFIX = "mm:"
_IDENTITY_SALT_DOMAIN = b"smg.vllm.mm-identity.v1\0"

# The limits of vLLM's HTTP front end (`validate_cache_salt` of vLLM 0.31),
# which gRPC bypasses.
_MAX_CACHE_SALT_CHARS = 128
_FORBIDDEN_CACHE_SALT_CHARS = frozenset("@/\\\0")


def engine_accepts_mm_inputs(model_config) -> bool:
    """Whether the engine runs a vision encoder for pixel payloads.

    ``is_multimodal_model`` only reports the architecture: it stays true
    under ``--language-model-only``, which zeroes every modality limit and
    drops the vision encoder (encoder-cache budget 0), and it is true for
    ``enable_mm_embeds``-only models that ingest pre-computed embeddings but
    cannot encode pixels. The router reads this through the worker's
    ``supports_vision`` label to decide whether a decode side may carry an mm
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
    side's form; a bare identity payload (hashes only) is not preprocessable
    and falls back to the cache-salt path.
    """
    return mm_inputs.HasField("pixel_values") or bool(mm_inputs.model_specific_tensors)


def request_cache_salt(request) -> str | None:
    """The request salt, checked; None for a request without one.

    A stub built from an older proto has no ``cache_salt``; it reads as none.
    The error texts do not contain the salt. A salt can be a secret.
    """
    if "cache_salt" not in request.DESCRIPTOR.fields_by_name or not request.HasField("cache_salt"):
        return None
    salt = request.cache_salt
    if not salt:
        raise ValueError("cache_salt must not be empty")
    if len(salt) > _MAX_CACHE_SALT_CHARS:
        raise ValueError(f"cache_salt must be at most {_MAX_CACHE_SALT_CHARS} characters")
    if not _FORBIDDEN_CACHE_SALT_CHARS.isdisjoint(salt):
        raise ValueError("cache_salt must not contain '@', '/', '\\' or NUL")
    if salt.startswith(_RESERVED_SALT_PREFIX):
        raise ValueError("cache_salt must not start with the reserved prefix 'mm:'")
    return salt


def engine_cache_salt(request_salt: str | None, mm_hashes: Sequence[str]) -> str | None:
    """The engine's salt for a request salt and the media hashes of a tensor-less side.

    The PD router strips multimodal tensors from the decode side (the KV
    arrives via the P/D transfer), keeping only the per-image content hashes.
    Without tensors no ``mm_features`` can be built, so the engine's
    prefix-cache block hashes would carry no image identity — the identity
    rides ``cache_salt`` instead. Deterministic per image content: same-image
    reuse still hits the decode prefix cache, while different images behind
    the same text prefix no longer alias onto each other's KV.

    The digest covers the request salt too, so requests with different salts
    share no blocks. Its length is the same for any number of images. The Rust
    servicer has the same rule (``vllm_cache_salt.rs`` of
    ``engine_zmq_adapter``): change them together.
    """
    if not mm_hashes:
        return request_salt
    digest = hashlib.sha256(_IDENTITY_SALT_DOMAIN)
    if request_salt is None:
        digest.update(b"\0")
    else:
        digest.update(b"\1" + _length_prefixed(request_salt))
    for mm_hash in mm_hashes:
        digest.update(_length_prefixed(mm_hash))
    return _RESERVED_SALT_PREFIX + digest.hexdigest()


def _length_prefixed(text: str) -> bytes:
    data = text.encode()
    return len(data).to_bytes(8, "little") + data
