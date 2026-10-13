"""The engine's configuration as the SGLang servicers read it.

The Python servicer and the Rust launcher advertise the same facts off
SGLang's ``ServerArgs`` and ``ModelConfig`` and read the same knobs (the
running window, the data-parallel width, the disaggregation role, the
KV-event publisher); the attribute names and what stands in when one is
absent live here, so an engine release that moves one is one fix.
Engine-free at import: the objects are taken as given and nothing from the
engine is imported.

A section is read when first asked for, once per view; a reader builds a
view where it reads. Every section is lenient: an absent attribute reads as
its default.
"""

from __future__ import annotations

import dataclasses
from functools import cached_property
from typing import Any


def _positive_int(value: Any) -> int | None:
    """``value`` as a count above zero, else ``None`` (a bool is not a count)."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


@dataclasses.dataclass(frozen=True)
class ServerArgsSection:
    """``ServerArgs``: the model's names and the knobs the servicers read."""

    model_path: str
    tokenizer_path: str | None
    served_model_name: str | None
    weight_version: str
    preferred_sampling_params: Any
    revision: Any
    # The operator's ``--max-running-requests`` when it is a count, else ``None``.
    max_running_requests: int | None
    # The data-parallel width when the args carry a count, else ``None``.
    dp_size: int | None
    enable_dp_attention: bool
    # The attention-DP width when the args carry it resolved, else ``None``.
    # The engine's newer releases rewrite ``--dp-size N --enable-dp-attention``
    # into ``attn_dp_size=N, dp_size=1, enable_dp_attention=False``; the pinned
    # release has no such attribute and keeps the flag as given.
    attn_dp_size: int | None
    # The attention context-parallel width when the args carry it (the newer
    # releases; it splits one attention group's TP width further), else ``None``.
    attn_cp_size: int | None
    kv_events_config: Any
    disaggregation_mode: str
    language_only: bool
    encoder_only: bool

    @classmethod
    def read(cls, server_args: Any) -> ServerArgsSection:
        return cls(
            model_path=str(getattr(server_args, "model_path", "") or ""),
            tokenizer_path=getattr(server_args, "tokenizer_path", None),
            served_model_name=getattr(server_args, "served_model_name", None),
            weight_version=str(getattr(server_args, "weight_version", None) or ""),
            preferred_sampling_params=getattr(server_args, "preferred_sampling_params", None),
            revision=getattr(server_args, "revision", None),
            max_running_requests=_positive_int(getattr(server_args, "max_running_requests", None)),
            dp_size=_positive_int(getattr(server_args, "dp_size", None)),
            enable_dp_attention=bool(getattr(server_args, "enable_dp_attention", False)),
            attn_dp_size=_positive_int(getattr(server_args, "attn_dp_size", None)),
            attn_cp_size=_positive_int(getattr(server_args, "attn_cp_size", None)),
            kv_events_config=getattr(server_args, "kv_events_config", None),
            disaggregation_mode=getattr(server_args, "disaggregation_mode", "null") or "null",
            language_only=bool(getattr(server_args, "language_only", False)),
            encoder_only=bool(getattr(server_args, "encoder_only", False)),
        )

    @property
    def attention_dp_width(self) -> int:
        """How many attention-DP workers ``--max-running-requests`` is split
        across, on either engine release: ``attn_dp_size`` when it is resolved
        above one, else ``dp_size`` under ``enable_dp_attention`` (the pinned
        release, where ``dp_size`` is the attention-DP width), else 1 (plain
        data parallelism runs the whole flag on every scheduler)."""
        if self.attn_dp_size is not None and self.attn_dp_size > 1:
            return self.attn_dp_size
        if self.enable_dp_attention:
            return self.dp_size or 1
        return 1

    @property
    def scheduler_count(self) -> int:
        """How many schedulers the engine runs, one load snapshot each: the
        data-parallel width times the attention-DP width the newer releases
        resolve (1 when absent, as on the pinned release, where ``dp_size``
        already counts every scheduler)."""
        return (self.dp_size or 1) * (self.attn_dp_size or 1)


@dataclasses.dataclass(frozen=True)
class ModelConfigSection:
    """``ModelConfig``: what the HF config declares and the model's shape, as
    the Python servicer derives them (the classification heads among them:
    ``id2label`` from ``num_labels`` when only the count is declared, and the
    count from the labels when only they are)."""

    hf_config: Any
    eos_token_ids: list[int]
    model_type: str
    architectures: list[str]
    pad_token_id: int
    bos_token_id: int
    id2label: Any
    num_labels: int
    context_len: int
    vocab_size: int
    is_generation: bool
    is_multimodal: bool

    @classmethod
    def read(cls, model_config: Any) -> ModelConfigSection:
        hf_config = getattr(model_config, "hf_config", None)
        eos = getattr(model_config, "hf_eos_token_id", None)
        if isinstance(eos, int):
            eos_token_ids = [eos]
        elif isinstance(eos, list | tuple | set | frozenset):
            eos_token_ids = sorted(int(value) for value in eos)
        else:
            eos_token_ids = []
        id2label = getattr(hf_config, "id2label", None) if hf_config is not None else None
        num_labels = (getattr(hf_config, "num_labels", 0) or 0) if hf_config is not None else 0
        if not id2label and num_labels:
            id2label = {index: f"LABEL_{index}" for index in range(num_labels)}
        elif id2label and not num_labels:
            num_labels = len(id2label)
        return cls(
            hf_config=hf_config,
            eos_token_ids=eos_token_ids,
            model_type=str((getattr(hf_config, "model_type", "") or "") if hf_config else ""),
            architectures=list(
                (getattr(hf_config, "architectures", []) or []) if hf_config else []
            ),
            pad_token_id=int((getattr(hf_config, "pad_token_id", 0) or 0) if hf_config else 0),
            bos_token_id=int((getattr(hf_config, "bos_token_id", 0) or 0) if hf_config else 0),
            id2label=id2label,
            num_labels=int(num_labels or 0),
            context_len=int(getattr(model_config, "context_len", 0) or 0),
            vocab_size=int(getattr(model_config, "vocab_size", 0) or 0),
            is_generation=bool(getattr(model_config, "is_generation", True)),
            is_multimodal=bool(getattr(model_config, "is_multimodal", False)),
        )


class EngineConfigView:
    """``ServerArgs`` and, when given, ``ModelConfig`` through the sections the
    servicers read, each read on first use."""

    def __init__(self, server_args: Any, model_config: Any = None):
        self._server_args = server_args
        self._model_config = model_config

    @cached_property
    def server_args(self) -> ServerArgsSection:
        return ServerArgsSection.read(self._server_args)

    @cached_property
    def model(self) -> ModelConfigSection:
        return ModelConfigSection.read(self._model_config)
