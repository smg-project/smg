"""The ``mm_item_limits`` label: an engine's own per-request media-count limits,
advertised through ``GetServerInfo`` for the Router's media pipeline to hold
requests to. The engine's front end checks those limits on the requests it
parses itself; the precomputed inputs the Router sends bypass it, so the Router
has to know them.

Engine-free at import: the readers take the parsed server arguments.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

# The `server_args` key (SGLang, TokenSpeed) and the `GetServerInfo` field (vLLM)
# the Router reads the label from.
MM_ITEM_LIMITS_KEY = "mm_item_limits"

# The modality names the Router's media pipeline counts, in label order.
MODALITIES = ("audio", "image", "video")


def format_item_limits(limits: Mapping[str, Any]) -> str:
    """``limits`` as the label: ``<modality>=<count>`` pairs in modality order
    (``image=8,video=2``) for the Router's modalities with a positive integral
    count; empty when there is none."""
    pairs = []
    for modality in MODALITIES:
        count = limits.get(modality)
        if isinstance(count, bool) or not isinstance(count, (int, float)):
            continue
        if count <= 0 or count != int(count):
            continue
        pairs.append(f"{modality}={int(count)}")
    return ",".join(pairs)


def sglang_item_limits(server_args: Any) -> str:
    """SGLang's ``--limit-mm-data-per-request`` (a JSON object of counts per
    modality, which its tokenizer manager refuses a request above) as the label;
    empty when the engine has no such limit."""
    raw = getattr(server_args, "limit_mm_data_per_request", None)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return ""
    if not isinstance(raw, Mapping):
        return ""
    return format_item_limits(raw)
