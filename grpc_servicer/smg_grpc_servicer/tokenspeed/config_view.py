"""The engine's configuration as the TokenSpeed servicers read it.

The Python servicer and the Rust launcher advertise the same facts off
TokenSpeed's ``ServerArgs`` and ``ModelConfig`` and read the same knobs
(the running window, the KV page size, the attention-DP width, the
reasoning parser, the KV-event publisher); the attribute names, the
upstream renames (``model_path`` -> ``model``, ``tokenizer_path`` ->
``tokenizer``) and what stands in when an attribute is absent live here, so
an engine release that moves one is one fix. Engine-free at import: the
objects are taken as given and nothing from the engine is imported.

A section is read when first asked for, once per view; a reader builds a
view where it reads. The sections are lenient (an absent attribute reads as
its default) except the vocabulary size, which the facts never invent.
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

    # ``model`` and ``tokenizer`` under either spelling; "" when neither is set.
    model: str
    tokenizer: str
    served_model_name: str | None
    preferred_sampling_params: str | None
    weight_version: str
    revision: Any
    reasoning_parser: Any
    # The attention-DP width (``mapping.attn.dp_size``) when the args carry one.
    attention_dp_size: int | None
    # The admission window, under TokenSpeed's spelling or the frontend's; 0 when unset.
    running_window: int
    page_size: int
    max_total_num_tokens: int
    kv_events_config: Any
    disaggregation_mode: str

    @classmethod
    def read(cls, server_args: Any) -> ServerArgsSection:
        mapping = getattr(server_args, "mapping", None)
        return cls(
            model=str(
                getattr(server_args, "model", None) or getattr(server_args, "model_path", "") or ""
            ),
            tokenizer=str(
                getattr(server_args, "tokenizer", None)
                or getattr(server_args, "tokenizer_path", "")
                or ""
            ),
            served_model_name=getattr(server_args, "served_model_name", None),
            preferred_sampling_params=getattr(server_args, "preferred_sampling_params", None),
            weight_version=str(getattr(server_args, "weight_version", "") or ""),
            revision=getattr(server_args, "revision", None),
            reasoning_parser=getattr(server_args, "reasoning_parser", None),
            attention_dp_size=_positive_int(
                getattr(getattr(mapping, "attn", None), "dp_size", None)
            ),
            running_window=int(
                getattr(server_args, "max_num_seqs", 0)
                or getattr(server_args, "max_running_requests", 0)
                or 0
            ),
            page_size=int(getattr(server_args, "page_size", 0) or 0),
            max_total_num_tokens=int(getattr(server_args, "max_total_num_tokens", 0) or 0),
            kv_events_config=getattr(server_args, "kv_events_config", None),
            disaggregation_mode=getattr(server_args, "disaggregation_mode", "null") or "null",
        )


@dataclasses.dataclass(frozen=True)
class ModelConfigSection:
    """``ModelConfig``: what the HF config declares and the model's shape. The
    vocabulary size is required of the config; the rest degrades to the
    proto's default."""

    hf_config: Any
    eos_token_ids: list[int]
    model_type: str
    architectures: list[str]
    pad_token_id: int
    bos_token_id: int
    context_len: int
    vocab_size: int
    dtype: Any

    @classmethod
    def read(cls, model_config: Any) -> ModelConfigSection:
        hf_config = getattr(model_config, "hf_config", None)
        eos = getattr(hf_config, "eos_token_id", None) if hf_config is not None else None
        if isinstance(eos, int):
            eos_token_ids = [eos]
        elif isinstance(eos, list):
            eos_token_ids = [int(value) for value in eos]
        else:
            eos_token_ids = []
        return cls(
            hf_config=hf_config,
            eos_token_ids=eos_token_ids,
            model_type=str((getattr(hf_config, "model_type", "") or "") if hf_config else ""),
            architectures=list(
                (getattr(hf_config, "architectures", []) or []) if hf_config else []
            ),
            pad_token_id=int((getattr(hf_config, "pad_token_id", 0) or 0) if hf_config else 0),
            bos_token_id=int((getattr(hf_config, "bos_token_id", 0) or 0) if hf_config else 0),
            context_len=int(getattr(model_config, "context_len", 0) or 0),
            vocab_size=int(model_config.vocab_size),
            dtype=getattr(model_config, "dtype", None),
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

    @cached_property
    def model_dtype(self) -> Any:
        """The model's dtype as the config spells it, ``None`` when it does
        not carry one (a lenient read for the request path, which never needs
        the facts)."""
        return getattr(self._model_config, "dtype", None)
