"""The lenient reads the engine configuration views share.

Each engine package has a ``config_view`` that reads the engine's own config
objects into typed sections (``smg_grpc_servicer.<engine>.config_view``).
What those sections read the same way lives here: a knob as a count, an HF
config's declarations when the config may be absent, and the classification
heads reconciled between their two spellings. Engine-free: the objects are
taken as given.
"""

from __future__ import annotations

from typing import Any


def positive_int(value: Any) -> int | None:
    """``value`` as a count above zero, else ``None`` (a bool is not a count)."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def attr_str(config: Any, name: str) -> str:
    """``config.<name>`` as text; ``""`` without a config, the attribute, or a value."""
    return str((getattr(config, name, "") or "") if config else "")


def attr_int(config: Any, name: str) -> int:
    """``config.<name>`` as an int; ``0`` without a config, the attribute, or a value."""
    return int((getattr(config, name, 0) or 0) if config else 0)


def attr_list(config: Any, name: str) -> list:
    """``config.<name>`` as a list; empty without a config, the attribute, or a value."""
    return list((getattr(config, name, []) or []) if config else [])


def classification_labels(hf_config: Any) -> tuple[Any, int]:
    """The classification heads an HF config declares, as ``(id2label, num_labels)``
    reconciled the way the Python servicer reports them: ``id2label`` from
    ``num_labels`` when only the count is declared, and the count from the
    labels when only they are; ``(None, 0)`` when neither is."""
    id2label = getattr(hf_config, "id2label", None) if hf_config is not None else None
    num_labels = (getattr(hf_config, "num_labels", 0) or 0) if hf_config is not None else 0
    if not id2label and num_labels:
        id2label = {index: f"LABEL_{index}" for index in range(num_labels)}
    elif id2label and not num_labels:
        num_labels = len(id2label)
    return id2label, int(num_labels or 0)
