"""The engine facts both vLLM servicers advertise, read once off vLLM's config.

`GetModelInfo` and `GetServerInfo` answer from vLLM's `ModelConfig` and
`VllmConfig`. vLLM offers no API for these facts beyond the config attributes
themselves, so the reads live here, in one place, for the Python servicer
(which answers at request time from its AsyncLLM's config) and the Rust
servicer (whose launcher computes them before the headless engine starts,
since no Python runs on its request path). A vLLM change to any of these
attributes is then one fix, and the Python servicer's live lane catches it.

Engine-free at import: this module takes the config objects and imports
nothing from vLLM.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from smg_grpc_servicer import mm_shm
from smg_grpc_servicer.pd_pairing import pairing_protocol_from_env
from smg_grpc_servicer.vllm.kv_transfer import pairing_fields, resolve_pd_connector
from smg_grpc_servicer.vllm.mm_salt import engine_accepts_mm_inputs

logger = logging.getLogger(__name__)

# The generation-config sampling defaults `GetModelInfo` advertises.
SAMPLING_DEFAULT_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")


def filtered_sampling_defaults(params: dict | None) -> dict:
    """The advertised subset of a `get_diff_sampling_param()` result."""
    if not params:
        return {}
    return {key: params[key] for key in SAMPLING_DEFAULT_KEYS if params.get(key) is not None}


def default_sampling_params_json(model_config: Any) -> str:
    """`GetModelInfo.default_sampling_params_json`: the model's generation-config
    sampling defaults as compact JSON, empty when it declares none."""
    get_diff = getattr(model_config, "get_diff_sampling_param", None)
    if not callable(get_diff):
        return ""
    try:
        diff = get_diff() or {}
    except Exception:  # a missing or malformed generation config is not fatal
        logger.warning("Could not read the model's sampling defaults", exc_info=True)
        return ""
    filtered = filtered_sampling_defaults(diff)
    return json.dumps(filtered, separators=(",", ":")) if filtered else ""


def _ids(value: Any) -> list[int]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, int):
        return [value] if value >= 0 else []
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, int) and not isinstance(v, bool) and v >= 0]
    return []


def hf_eos_token_ids(model_config: Any) -> list[int]:
    """The EOS ids the model's HF config declares: what `GetModelInfo` advertises."""
    return _ids(getattr(getattr(model_config, "hf_config", None), "eos_token_id", None))


def eos_token_ids_with_generation_config(model_config: Any) -> list[int]:
    """The HF config's EOS ids first (the primary id the engine stops on), then
    the generation config's extras, deduplicated: the full set a request stops
    on, as vLLM's own frontend assembles it."""
    ids = hf_eos_token_ids(model_config)
    try_get = getattr(model_config, "try_get_generation_config", None)
    if callable(try_get):
        try:
            generation = try_get() or {}
        except Exception:  # a missing or malformed generation config is not fatal
            generation = {}
        for extra in _ids(generation.get("eos_token_id")):
            if extra not in ids:
                ids.append(extra)
    return ids


def supports_vision(model_config: Any) -> bool:
    """vLLM's own answer to whether the engine accepts multimodal input."""
    try:
        return bool(engine_accepts_mm_inputs(model_config))
    except Exception:  # an unexpected config shape must not take the servicer down
        logger.warning(
            "Could not determine multimodal support; reporting supports_vision=false",
            exc_info=True,
        )
        return False


def model_facts(model_config: Any) -> dict[str, Any]:
    """`GetModelInfo`'s fields, keyed as the proto names them."""
    hf_config = getattr(model_config, "hf_config", None)
    served = getattr(model_config, "served_model_name", None) or model_config.model
    if isinstance(served, (list, tuple)):
        served = served[0] if served else model_config.model
    return {
        "model_path": str(model_config.model),
        "is_generation": getattr(model_config, "runner_type", "generate") == "generate",
        "max_context_length": int(model_config.max_model_len),
        "vocab_size": int(model_config.get_vocab_size()),
        "supports_vision": supports_vision(model_config),
        "served_model_name": str(served),
        "tokenizer_path": str(getattr(model_config, "tokenizer", None) or model_config.model),
        "model_type": str(getattr(hf_config, "model_type", None) or ""),
        "architectures": [str(a) for a in (getattr(model_config, "architectures", None) or [])],
        "eos_token_ids": hf_eos_token_ids(model_config),
        "pad_token_id": int(getattr(hf_config, "pad_token_id", None) or 0),
        "bos_token_id": int(getattr(hf_config, "bos_token_id", None) or 0),
        "default_sampling_params_json": default_sampling_params_json(model_config),
    }


def mm_device_do_normalize(vllm_config: Any) -> bool:
    """Whether the engine rescales and normalizes pixels on device: vLLM's
    ``mm_device_do_normalize`` as its model config resolved it (off where the
    model class does not support it). Such an engine takes the pixels' own
    ``uint8`` bytes and would normalize anything else twice, so both servicers
    advertise it and whoever preprocesses sends raw pixels."""
    model_config = vllm_config.model_config
    if not getattr(model_config, "is_multimodal_model", False):
        return False
    mm_config = getattr(model_config, "multimodal_config", None)
    return bool(getattr(mm_config, "mm_device_do_normalize", False))


def running_window(vllm_config: Any) -> int:
    """The scheduler's running window (``--max-num-seqs``): how many requests
    the engine runs at once, which ``GetServerInfo`` advertises as
    ``max_num_seqs`` for the Router's PD admission gate and its fleet capacity
    accounting; 0 when the config does not carry it, which leaves both as
    they were."""
    scheduler = getattr(vllm_config, "scheduler_config", None)
    window = getattr(scheduler, "max_num_seqs", None)
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        return 0
    return int(window)


def server_facts(vllm_config: Any) -> dict[str, Any]:
    """`GetServerInfo`'s config-derived fields, keyed as the proto names them:
    the PD identity and pairing facts, the data-parallel size, the running
    window, this host's `/dev/shm` identity."""
    kv_config = getattr(vllm_config, "kv_transfer_config", None)
    kv_connector, kv_engine_id, kv_role = "", "", ""
    if kv_config is not None:
        # The effective PD engine_id; with DP the engine cores serve
        # `{id}_dp{rank}` and the router derives the suffix from the rank it
        # pins per request.
        kv_connector, kv_engine_id = resolve_pd_connector(kv_config)
        kv_role = getattr(kv_config, "kv_role", None) or ""
    return {
        "kv_connector": str(kv_connector or ""),
        "kv_role": str(kv_role),
        "kv_engine_id": str(kv_engine_id or ""),
        "data_parallel_size": int(vllm_config.parallel_config.data_parallel_size),
        "max_num_seqs": running_window(vllm_config),
        "shm_namespace_id": mm_shm.shm_namespace_id(),
        "pairing_protocol": pairing_protocol_from_env(),
        **pairing_fields(vllm_config),
    }
