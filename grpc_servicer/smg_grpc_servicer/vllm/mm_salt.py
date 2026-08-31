"""Multimodal identity cache salt for tensor-stripped (PD decode) legs."""

from collections.abc import Sequence


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
