"""The engine's configuration as the servicers read it.

vLLM offers no API for the facts the servicers advertise beyond its config
attributes, and those move between releases: a section an older version does
not carry, an attribute a newer one adds, a name that changes. One reader
here knows the names and what stands in when they are absent;
:mod:`.model_info`, :mod:`.kv_transfer`, :mod:`.kv_events`, the Python
servicer and the Rust launcher (:mod:`.rust`) read typed sections off it, so
a vLLM rename is one fix, found by this module's tests and the Python
servicer's live lane. Engine-free at import: the config objects are taken as
given and nothing from vLLM is imported.

A section is read when first asked for, once per view; a reader builds a
view where it reads (a view holds a reference and nothing else until a
section is asked for). The sections are lenient (a missing section reads as
its defaults, a view over ``None`` reads every section so) except what the
servicers never invent: :attr:`model`, the facts of ``GetModelInfo``, fails
on a config without the model's name, context length or vocabulary;
:attr:`data_parallel_size` on a config without one; and :attr:`launch`, the
process manager's inputs, on a config missing any one of them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from functools import cached_property
from typing import Any


def _positive_int(value: Any) -> int | None:
    """``value`` as a count above zero, else ``None`` (a bool is not a count)."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def token_ids(value: Any) -> list[int]:
    """The token ids in an HF config field: one id, or a list of them;
    negative ids and booleans are not ids."""
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, int):
        return [value] if value >= 0 else []
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, int) and not isinstance(v, bool) and v >= 0]
    return []


def hf_eos_token_ids(model_config: Any) -> list[int]:
    """The EOS ids the model's HF config declares."""
    return token_ids(getattr(getattr(model_config, "hf_config", None), "eos_token_id", None))


def uses_mrope(model_config: Any) -> bool:
    """Whether the model positions tokens with M-RoPE (its media carry grids)."""
    return bool(getattr(model_config, "uses_mrope", False))


@dataclasses.dataclass(frozen=True)
class ModelSection:
    """``ModelConfig``: the facts ``GetModelInfo`` advertises. ``model``,
    ``max_model_len`` and the vocabulary are required of the config; the rest
    degrades to the proto's default."""

    model: str
    served_model_name: str
    tokenizer: str
    runner_type: str
    max_model_len: int
    vocab_size: int
    architectures: list[str]
    model_type: str
    hf_eos_token_ids: list[int]
    pad_token_id: int
    bos_token_id: int
    # The revision the tokenizer is fetched at: the tokenizer's own, else the model's.
    tokenizer_revision: str | None

    @classmethod
    def read(cls, model_config: Any) -> ModelSection:
        hf_config = getattr(model_config, "hf_config", None)
        served = getattr(model_config, "served_model_name", None) or model_config.model
        if isinstance(served, (list, tuple)):
            served = served[0] if served else model_config.model
        return cls(
            model=str(model_config.model),
            served_model_name=str(served),
            tokenizer=str(getattr(model_config, "tokenizer", None) or model_config.model),
            runner_type=getattr(model_config, "runner_type", "generate"),
            max_model_len=int(model_config.max_model_len),
            vocab_size=int(model_config.get_vocab_size()),
            architectures=[str(a) for a in (getattr(model_config, "architectures", None) or [])],
            model_type=str(getattr(hf_config, "model_type", None) or ""),
            hf_eos_token_ids=hf_eos_token_ids(model_config),
            pad_token_id=int(getattr(hf_config, "pad_token_id", None) or 0),
            bos_token_id=int(getattr(hf_config, "bos_token_id", None) or 0),
            tokenizer_revision=getattr(model_config, "tokenizer_revision", None)
            or getattr(model_config, "revision", None),
        )


@dataclasses.dataclass(frozen=True)
class MultimodalSection:
    """``ModelConfig.multimodal_config``, for a multimodal model: whether the
    engine normalizes pixels on device, and its per-prompt limit accessor
    (``get_limit_per_prompt``), ``None`` on a text model or a config without
    the accessor."""

    is_multimodal: bool
    device_do_normalize: bool
    limit_per_prompt: Callable[[str], Any] | None

    @classmethod
    def read(cls, model_config: Any) -> MultimodalSection:
        if not getattr(model_config, "is_multimodal_model", False):
            return cls(is_multimodal=False, device_do_normalize=False, limit_per_prompt=None)
        mm_config = getattr(model_config, "multimodal_config", None)
        get_limit = getattr(mm_config, "get_limit_per_prompt", None)
        return cls(
            is_multimodal=True,
            device_do_normalize=bool(getattr(mm_config, "mm_device_do_normalize", False)),
            limit_per_prompt=get_limit if callable(get_limit) else None,
        )


@dataclasses.dataclass(frozen=True)
class ParallelSection:
    """``ParallelConfig`` as a lenient read: the data-parallel width when the
    config carries a count (``None`` otherwise), the first rank this process
    serves (0 unless the config says otherwise), this node's rank within the
    data-parallel group (0: the head node) and the data-parallel backend
    (``mp`` unless the config says otherwise)."""

    data_parallel_size: int | None
    data_parallel_rank: int
    node_rank_within_dp: int
    data_parallel_backend: str

    @classmethod
    def read(cls, vllm_config: Any) -> ParallelSection:
        parallel = getattr(vllm_config, "parallel_config", None)
        return cls(
            data_parallel_size=_positive_int(getattr(parallel, "data_parallel_size", None)),
            data_parallel_rank=_positive_int(getattr(parallel, "data_parallel_rank", 0)) or 0,
            node_rank_within_dp=getattr(parallel, "node_rank_within_dp", 0),
            data_parallel_backend=getattr(parallel, "data_parallel_backend", "mp"),
        )


@dataclasses.dataclass(frozen=True)
class LaunchSection:
    """What vLLM's own process manager is started with on the head node, as
    ``run_headless`` reads it: every field required of the config (an engine
    this launcher cannot start this way fails here, by name)."""

    local_engine_count: int
    start_index: int
    master_ip: str
    rpc_port: int
    shutdown_timeout: Any

    @classmethod
    def read(cls, vllm_config: Any) -> LaunchSection:
        parallel = vllm_config.parallel_config
        return cls(
            local_engine_count=parallel.data_parallel_size_local,
            start_index=parallel.data_parallel_rank,
            master_ip=parallel.data_parallel_master_ip,
            rpc_port=parallel.data_parallel_rpc_port,
            shutdown_timeout=vllm_config.shutdown_timeout,
        )


@dataclasses.dataclass(frozen=True)
class CacheSection:
    """``CacheConfig``: the KV cache's dtype and block geometry, each ``None``
    when the config does not carry it."""

    cache_dtype: str | None
    block_size: int | None
    num_gpu_blocks: int | None

    @classmethod
    def read(cls, vllm_config: Any) -> CacheSection:
        cache = getattr(vllm_config, "cache_config", None)
        dtype = getattr(cache, "cache_dtype", None)
        return cls(
            cache_dtype=str(dtype) if dtype else None,
            block_size=_positive_int(getattr(cache, "block_size", None)),
            num_gpu_blocks=_positive_int(getattr(cache, "num_gpu_blocks", None)),
        )


# A config attribute that is not there at all, as opposed to one set to nothing.
_UNKNOWN = object()


@dataclasses.dataclass(frozen=True)
class StructuredOutputsSection:
    """``StructuredOutputsConfig``: the backend (``None`` when the section or
    the attribute is not there), the reasoning parser ("" for none, with
    ``reasoning_parser_known`` false when the section, or its parser attribute,
    is not there at all), and whether grammars run inside the reasoning."""

    backend: str | None
    reasoning_parser: str
    reasoning_parser_known: bool
    enable_in_reasoning: bool

    @classmethod
    def read(cls, vllm_config: Any) -> StructuredOutputsSection:
        structured = getattr(vllm_config, "structured_outputs_config", None)
        parser = (
            _UNKNOWN if structured is None else getattr(structured, "reasoning_parser", _UNKNOWN)
        )
        return cls(
            backend=None if structured is None else getattr(structured, "backend", None),
            reasoning_parser="" if parser is _UNKNOWN else str(parser or ""),
            reasoning_parser_known=parser is not _UNKNOWN,
            enable_in_reasoning=bool(getattr(structured, "enable_in_reasoning", False)),
        )


@dataclasses.dataclass(frozen=True)
class KvEventsSection:
    """``KVEventsConfig`` as given (``None`` for none), whether the engine
    publishes KV cache events, through which publisher (``SubscribeKvEvents``
    relays the ZMQ one), and where."""

    config: Any
    enable_kv_cache_events: bool
    publisher: Any
    endpoint: str
    replay_endpoint: str
    topic: str

    @property
    def relayed(self) -> bool:
        """A ZMQ publisher is on: ``SubscribeKvEvents`` relays it."""
        return self.config is not None and self.enable_kv_cache_events and self.publisher == "zmq"

    @classmethod
    def read(cls, vllm_config: Any) -> KvEventsSection:
        config = getattr(vllm_config, "kv_events_config", None)
        return cls(
            config=config,
            enable_kv_cache_events=bool(getattr(config, "enable_kv_cache_events", False)),
            publisher=getattr(config, "publisher", None),
            endpoint=str(getattr(config, "endpoint", "") or ""),
            replay_endpoint=str(getattr(config, "replay_endpoint", "") or ""),
            topic=str(getattr(config, "topic", "") or ""),
        )


@dataclasses.dataclass(frozen=True)
class KvTransferSection:
    """``KVTransferConfig``: the connector configuration as given (the
    connector and engine id are projected by :func:`.kv_transfer.resolve_pd_connector`)
    and the role; ``config`` is ``None`` without one."""

    config: Any
    role: str

    @classmethod
    def read(cls, vllm_config: Any) -> KvTransferSection:
        config = getattr(vllm_config, "kv_transfer_config", None)
        return cls(config=config, role=str(getattr(config, "kv_role", None) or ""))


@dataclasses.dataclass(frozen=True)
class PoolerSection:
    """``ModelConfig.pooler_config``: what vLLM's frontend merges into a pooling
    request's unset fields; absent on most generation models."""

    use_activation: bool | None
    dimensions: int | None

    @classmethod
    def read(cls, vllm_config: Any) -> PoolerSection:
        pooler = getattr(getattr(vllm_config, "model_config", None), "pooler_config", None)
        use_activation = getattr(pooler, "use_activation", None)
        dimensions = getattr(pooler, "dimensions", None)
        return cls(
            use_activation=None if use_activation is None else bool(use_activation),
            dimensions=None if dimensions is None else int(dimensions),
        )


class EngineConfigView:
    """``VllmConfig`` through the sections the servicers read, each read on
    first use. ``vllm_config`` may be ``None`` (an engine without one): every
    lenient section then reads as its defaults."""

    def __init__(self, vllm_config: Any):
        self._config = vllm_config

    @cached_property
    def model(self) -> ModelSection:
        """The model's facts; requires ``model_config`` with its name, context
        length and vocabulary."""
        return ModelSection.read(self._config.model_config)

    @cached_property
    def multimodal(self) -> MultimodalSection:
        return MultimodalSection.read(getattr(self._config, "model_config", None))

    @cached_property
    def model_dtype(self) -> str | None:
        """The model's dtype as the config spells it (``torch.bfloat16``), or
        ``None`` when it does not carry one."""
        dtype = getattr(getattr(self._config, "model_config", None), "dtype", None)
        return None if dtype is None else str(dtype)

    @cached_property
    def parallel(self) -> ParallelSection:
        return ParallelSection.read(self._config)

    @cached_property
    def data_parallel_size(self) -> int:
        """The data-parallel width the Rust server and ``GetServerInfo`` are
        told; required of the config (a reader that can do without one reads
        :attr:`parallel` instead)."""
        return int(self._config.parallel_config.data_parallel_size)

    @cached_property
    def max_num_seqs(self) -> int:
        """The scheduler's running window (``--max-num-seqs``); 0 when the
        config does not carry a count."""
        scheduler = getattr(self._config, "scheduler_config", None)
        return _positive_int(getattr(scheduler, "max_num_seqs", None)) or 0

    @cached_property
    def cache(self) -> CacheSection:
        return CacheSection.read(self._config)

    @cached_property
    def attention_backend(self) -> str | None:
        """The requested attention backend (``--attention-backend``) by name;
        ``None`` on the auto path, where the worker picks it."""
        attention = getattr(self._config, "attention_config", None)
        backend = getattr(attention, "backend", None) if attention is not None else None
        return None if backend is None else str(getattr(backend, "name", backend))

    @cached_property
    def structured_outputs(self) -> StructuredOutputsSection:
        return StructuredOutputsSection.read(self._config)

    @cached_property
    def kv_events(self) -> KvEventsSection:
        return KvEventsSection.read(self._config)

    @cached_property
    def kv_transfer(self) -> KvTransferSection:
        return KvTransferSection.read(self._config)

    @cached_property
    def pooler(self) -> PoolerSection:
        return PoolerSection.read(self._config)

    @cached_property
    def launch(self) -> LaunchSection:
        """The process manager's inputs; required of the config."""
        return LaunchSection.read(self._config)
