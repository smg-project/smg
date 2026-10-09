"""The Rust request path of the TokenSpeed gRPC servicer, behind one flag.

``python -m smg_grpc_servicer.tokenspeed`` (what ``ts serve`` spawns) serves
``tokenspeed.grpc.scheduler.TokenSpeedScheduler`` from Python over TokenSpeed's
in-process ``AsyncLLM``. With ``SMG_TOKENSPEED_SERVICER_IMPL=rust`` the same
entrypoint hands the process to :func:`serve_rust` before an ``AsyncLLM`` or a
Python gRPC server exists: the contract is served by the Rust
:class:`smg.servicer.TokenSpeedGrpcServer` on a Rust-owned thread, and the
scheduler(s) run headless in a spawned child through TokenSpeed's own
``launch_scheduler_headless`` (what ``ts serve --headless`` runs), dialing the
servicer's same-host msgpack ZMQ handshake. Python keeps the lifecycle only.
The Router cannot tell the two implementations apart.

What the msgpack wire does not carry, the Rust path reports rather than
emulates: ``FlushCache`` and profiling answer UNIMPLEMENTED, ranked
``top_logprobs`` and prompt logprobs are refused, and PD/EPD disaggregation
(bootstrap fields) stays with the Python implementation.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import multiprocessing
import os
from collections.abc import Mapping
from typing import Any

from smg_grpc_servicer.pd_pairing import pairing_protocol_from_env
from smg_grpc_servicer.rust_lifecycle import (
    DEFAULT_DRAIN_SECS,
    DEFAULT_STARTUP_TIMEOUT_SECS,
    EngineProcess,
    default_socket_dir,
    env_float,
    free_port,
    resolve_tokenizer_dir,
    supervise,
)
from smg_grpc_servicer.tokenspeed.kv_events import resolve_kv_events_config
from smg_grpc_servicer.tokenspeed.loads import running_window
from smg_grpc_servicer.tokenspeed.redact import redact_secrets

logger = logging.getLogger(__name__)

SERVICER_IMPL_ENV = "SMG_TOKENSPEED_SERVICER_IMPL"
HANDSHAKE_PORT_ENV = "SMG_TOKENSPEED_SERVICER_HANDSHAKE_PORT"
DRAIN_SECS_ENV = "SMG_TOKENSPEED_SERVICER_DRAIN_SECS"
STARTUP_TIMEOUT_SECS_ENV = "SMG_TOKENSPEED_SERVICER_STARTUP_TIMEOUT_SECS"
# Set to 0/false/no/off to keep TokenSpeed's KV event publisher off when the
# launcher was given no --kv-events-config (see `default_kv_events_config`).
KV_EVENTS_ENV = "SMG_TOKENSPEED_SERVICER_KV_EVENTS"
_OFF_VALUES = ("0", "false", "no", "off")
# What a launcher without --kv-events-config gets under the Rust servicer:
# the ZMQ publisher, with TokenSpeed's own defaults for the rest (the
# endpoint, the topic).
DEFAULT_KV_EVENTS_CONFIG = '{"enable_kv_cache_events": true, "publisher": "zmq"}'
IMPLS = ("python", "rust")


def resolve_servicer_impl(environ: Mapping[str, str] | None = None) -> str:
    """Which implementation serves this process: ``$SMG_TOKENSPEED_SERVICER_IMPL``,
    else python."""
    source = os.environ if environ is None else environ
    value = str(source.get(SERVICER_IMPL_ENV) or "python").strip().lower()
    if value not in IMPLS:
        raise ValueError(f"{SERVICER_IMPL_ENV} must be one of {IMPLS}, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# What the Rust server advertises, from TokenSpeed's own config
# ---------------------------------------------------------------------------


def _model_config(server_args: Any) -> Any:
    """TokenSpeed's ``ModelConfig`` for the served model, built as ``AsyncLLM``
    builds its own (the facts below are what the Python servicer reads off
    that instance)."""
    from tokenspeed.runtime.configs.model_config import ModelConfig

    return ModelConfig(
        server_args.model,
        trust_remote_code=server_args.trust_remote_code,
        revision=server_args.revision,
        context_length=server_args.max_model_len,
        model_override_args=server_args.hf_overrides,
        dtype=server_args.dtype,
        quantization=server_args.quantization,
        server_args=server_args,
    )


def _dtype_name(dtype: Any) -> str:
    """``torch.bfloat16`` → ``bfloat16``, as the Python servicer reports it."""
    text = str(dtype or "")
    return text.removeprefix("torch.")


def model_facts(server_args: Any, model_config: Any | None = None) -> dict[str, Any]:
    """``TokenSpeedGrpcServer`` keyword arguments for ``GetModelInfo``: the
    Python servicer's ``GetModelInfo`` fields, computed up front."""
    from smg_grpc_servicer.tokenspeed.servicer import TokenSpeedSchedulerServicer

    model_config = model_config if model_config is not None else _model_config(server_args)
    hf_config = getattr(model_config, "hf_config", None)
    eos = getattr(hf_config, "eos_token_id", None) if hf_config is not None else None
    if isinstance(eos, int):
        eos_token_ids = [eos]
    elif isinstance(eos, list):
        eos_token_ids = [int(value) for value in eos]
    else:
        eos_token_ids = []
    supported_modalities = TokenSpeedSchedulerServicer._static_supported_modalities(
        model_config, hf_config
    )
    dtype = _dtype_name(getattr(model_config, "dtype", None))
    model_path = getattr(server_args, "model", None) or getattr(server_args, "model_path", "")
    tokenizer_path = getattr(server_args, "tokenizer", None) or getattr(
        server_args, "tokenizer_path", ""
    )
    return {
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path or model_path),
        "served_model_name": str(getattr(server_args, "served_model_name", None) or model_path),
        "model_type": str((getattr(hf_config, "model_type", "") or "") if hf_config else ""),
        "architectures": list((getattr(hf_config, "architectures", []) or []) if hf_config else []),
        "max_context_length": int(getattr(model_config, "context_len", 0) or 0),
        "vocab_size": int(getattr(model_config, "vocab_size", 0) or 0),
        "eos_token_ids": eos_token_ids,
        "pad_token_id": int((getattr(hf_config, "pad_token_id", 0) or 0) if hf_config else 0),
        "bos_token_id": int((getattr(hf_config, "bos_token_id", 0) or 0) if hf_config else 0),
        "default_sampling_params_json": str(
            getattr(server_args, "preferred_sampling_params", None) or ""
        ),
        "supports_vision": any(modality in (1, 3) for modality in supported_modalities),
        "supports_multimodal": bool(supported_modalities),
        "supported_modalities": [int(modality) for modality in supported_modalities],
        "model_dtype": dtype,
        "multimodal_encoder_dtype": dtype,
    }


def server_facts(server_args: Any) -> dict[str, Any]:
    """``TokenSpeedGrpcServer`` keyword arguments for ``GetServerInfo`` and
    ``GetLoads``: the server args as the Router's label source (with the
    attention-DP width and the operator's pairing protocol, as the Python
    servicer adds them), the admission window, the KV-event publisher."""
    from smg_grpc_servicer.tokenspeed.servicer import _make_json_serializable, _shm_namespace_id

    if dataclasses.is_dataclass(server_args) and not isinstance(server_args, type):
        args_dict = dataclasses.asdict(server_args)
    else:
        args_dict = dict(getattr(server_args, "__dict__", {}))
    dp_size = getattr(getattr(getattr(server_args, "mapping", None), "attn", None), "dp_size", None)
    if isinstance(dp_size, int) and dp_size > 1:
        args_dict["dp_size"] = dp_size
    pairing_protocol = pairing_protocol_from_env()
    if pairing_protocol:
        args_dict["pairing_protocol"] = pairing_protocol
    kv_events = resolve_kv_events_config(server_args)
    try:
        from tokenspeed.version import __version__ as tokenspeed_version
    except ImportError:  # the launcher's unit tests run without TokenSpeed
        tokenspeed_version = ""
    return {
        "server_args_json": json.dumps(
            redact_secrets(_make_json_serializable(args_dict)), sort_keys=True
        ),
        "scheduler_info_json": json.dumps({"shm_namespace_id": _shm_namespace_id()}),
        "tokenspeed_version": str(tokenspeed_version),
        "max_running_requests": running_window(server_args),
        "data_parallel_size": int(dp_size) if isinstance(dp_size, int) and dp_size > 0 else 1,
        "kv_events_endpoint": kv_events.endpoint if kv_events else "",
        "kv_events_replay_endpoint": kv_events.replay_endpoint if kv_events else "",
        "kv_events_topic": kv_events.topic if kv_events else "",
    }


def default_kv_events_config(
    server_args: Any, environ: Mapping[str, str] | None = None
) -> str | None:
    """Turn TokenSpeed's KV event publisher on when the launcher was given no
    ``--kv-events-config``; returns the configuration applied, or None.

    Cache-aware routing lives on the events ``SubscribeKvEvents`` relays, and
    TokenSpeed publishes none unless started with ``--kv-events-config
    '{"enable_kv_cache_events": true, "publisher": "zmq"}'``: a launcher
    without it got a router that routed blind, with one WARN per worker as
    the only trace. Under the Rust servicer the server args that lack the
    option get exactly that configuration before the facts and the headless
    scheduler are built from them, so publisher and relay agree. An explicit
    ``--kv-events-config`` is kept as given, off included;
    ``SMG_TOKENSPEED_SERVICER_KV_EVENTS=0`` keeps the publisher off without
    one.
    """
    source = os.environ if environ is None else environ
    if getattr(server_args, "kv_events_config", None):
        return None
    if str(source.get(KV_EVENTS_ENV, "")).strip().lower() in _OFF_VALUES:
        return None
    try:
        server_args.kv_events_config = DEFAULT_KV_EVENTS_CONFIG
    except AttributeError:  # frozen server args: left as they are
        logger.warning(
            "kv_events_config cannot be set on %s; KV event publishing stays as configured",
            type(server_args).__name__,
        )
        return None
    return DEFAULT_KV_EVENTS_CONFIG


# ---------------------------------------------------------------------------
# Headless scheduler: TokenSpeed's own launch, dialing this servicer
# ---------------------------------------------------------------------------


def headless_server_args(server_args: Any, *, handshake_port: int) -> Any:
    """The parsed server args re-aimed at the servicer's handshake: what
    ``ts serve --headless`` sets, plus the dial target."""
    headless = copy.copy(server_args)
    headless.zmq_msgpack = True
    headless.skip_tokenizer_init = True
    headless.data_parallel_address = "127.0.0.1"
    headless.data_parallel_rpc_port = int(handshake_port)
    headless.zmq_engine_index = 0
    return headless


def _run_headless(server_args: Any) -> None:
    """Child target: TokenSpeed's own headless launch."""
    from tokenspeed.runtime.entrypoints.engine import launch_scheduler_headless

    launch_scheduler_headless(server_args)


def launch_headless_scheduler(server_args: Any) -> EngineProcess:
    """Start the headless scheduler(s) in a spawned child: a fresh interpreter
    that inherits no Rust thread and never consults the servicer switch."""
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_run_headless, args=(server_args,), name="smg-headless-scheduler"
    )
    process.start()
    return EngineProcess(process)


def tokenizer_dir_for(server_args: Any) -> str | None:
    """The local tokenizer directory for the served model, or ``None``."""
    tokenizer = getattr(server_args, "tokenizer", None) or getattr(server_args, "model", "")
    return resolve_tokenizer_dir(str(tokenizer), getattr(server_args, "revision", None))


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def serve_rust(server_args: Any) -> int:
    """Serve this process's gRPC contract from Rust: start the Rust server,
    launch the headless scheduler(s) against its handshake, and supervise
    both until a signal or a failure. Returns the process exit code."""
    from smg.servicer import TokenSpeedGrpcServer, init_servicer_tracing

    init_servicer_tracing(os.environ.get("RUST_LOG") and None)
    if default_kv_events_config(server_args) is not None:
        logger.info(
            "KV event publishing enabled: no --kv-events-config was given, so TokenSpeed's ZMQ "
            "publisher is on with its default endpoint and SubscribeKvEvents relays it; pass "
            "--kv-events-config to configure it, or set %s=0 to leave it off",
            KV_EVENTS_ENV,
        )
    handshake_port = int(os.environ.get(HANDSHAKE_PORT_ENV) or 0) or free_port()
    socket_dir = default_socket_dir()
    os.makedirs(socket_dir, mode=0o700, exist_ok=True)
    host = getattr(server_args, "host", None) or "0.0.0.0"
    port = int(getattr(server_args, "port", 0) or 0)
    facts = {**model_facts(server_args), **server_facts(server_args)}
    if facts.get("kv_events_endpoint"):
        logger.info(
            "SubscribeKvEvents relays TokenSpeed's KV events from %s", facts["kv_events_endpoint"]
        )
    else:
        logger.warning(
            "SubscribeKvEvents is off (KV cache events disabled, or a publisher other than "
            "zmq): a cache-aware router sees nothing of this engine's cache"
        )
    engine_count = facts["data_parallel_size"]
    tokenizer_dir = tokenizer_dir_for(server_args)
    if tokenizer_dir is None:
        logger.warning(
            "No local tokenizer directory for %s; string stops will be refused",
            getattr(server_args, "model", ""),
        )
    server = TokenSpeedGrpcServer(
        bind_address=f"{host}:{port}",
        ipc_base_url=f"ipc://{socket_dir}/tokenspeed-servicer-{os.getpid()}",
        handshake_address=f"tcp://127.0.0.1:{handshake_port}",
        engine_count=engine_count,
        tokenizer_dir=tokenizer_dir,
        engine_startup_timeout_secs=env_float(
            STARTUP_TIMEOUT_SECS_ENV, DEFAULT_STARTUP_TIMEOUT_SECS
        ),
        **facts,
    )
    logger.info(
        "Rust TokenSpeed servicer listening on %s; headless scheduler(s) dial handshake port %d",
        server.address,
        handshake_port,
    )
    engine = launch_headless_scheduler(
        headless_server_args(server_args, handshake_port=handshake_port)
    )
    return await supervise(server, engine, drain_secs=env_float(DRAIN_SECS_ENV, DEFAULT_DRAIN_SECS))
