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

from smg_grpc_servicer.config_view import attr_int, attr_list, attr_str, positive_int

CONSTRAINED_DECODING_POLICIES = ("requested", "always")


def constrained_decoding_policy(value: Any) -> str:
    """The engine's declared constrained-decoding policy as the gateway spells
    it: ``always`` for a grammar on every turn, ``requested`` for one only when
    the request asks; ``""`` when the engine declares none, or something the
    gateway would not read (it then treats the worker as ``requested``)."""
    text = str(value or "").strip().lower()
    return text if text in CONSTRAINED_DECODING_POLICIES else ""


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
    # The engine's declared constrained-decoding policy, ``""`` when it declares none.
    constrained_decoding: str

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
            attention_dp_size=positive_int(
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
            constrained_decoding=constrained_decoding_policy(
                getattr(server_args, "constrained_decoding", None)
            ),
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
            model_type=attr_str(hf_config, "model_type"),
            architectures=attr_list(hf_config, "architectures"),
            pad_token_id=attr_int(hf_config, "pad_token_id"),
            bos_token_id=attr_int(hf_config, "bos_token_id"),
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
    def constrained_decoding(self) -> str:
        """The engine's declared policy (``constrained_decoding_policy``): the
        server args' when they carry one, else the model config's."""
        return self.server_args.constrained_decoding or constrained_decoding_policy(
            getattr(self._model_config, "constrained_decoding", None)
        )

    @cached_property
    def model_dtype(self) -> Any:
        """The model's dtype as the config spells it, ``None`` when it does
        not carry one (a lenient read for the request path, which never needs
        the facts)."""
        return getattr(self._model_config, "dtype", None)
