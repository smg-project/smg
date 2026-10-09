"""The Rust request path of the SGLang gRPC servicer, behind one flag.

``sglang.launch_server --grpc-mode`` hands the process to this package's
:func:`smg_grpc_servicer.sglang.server.serve_grpc`, which serves
``sglang.grpc.scheduler.SglangScheduler`` from Python over a request manager
in front of the scheduler. With ``SMG_SGLANG_SERVICER_IMPL=rust`` the same
entrypoint hands the process to :func:`serve_rust` before any of that exists:
the contract is served by the Rust :class:`smg.servicer.SglangGrpcServer` on a
Rust-owned thread, and the scheduler runs headless in a spawned child (this
package's :mod:`headless` launcher, with its SGLang plugin dialing the
servicer's same-host msgpack ZMQ handshake). Python keeps the lifecycle only.
The Router cannot tell the two implementations apart.

What the msgpack wire does not carry, the Rust path reports rather than
emulates: ``Embed``, ``FlushCache``, profiling, LoRA loading and the KV-event
relay answer UNIMPLEMENTED, prompt logprobs, hidden states and multimodal
inputs are refused, and PD/EPD disaggregation stays with the Python
implementation.
"""

from __future__ import annotations

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

logger = logging.getLogger(__name__)

SERVICER_IMPL_ENV = "SMG_SGLANG_SERVICER_IMPL"
HANDSHAKE_PORT_ENV = "SMG_SGLANG_SERVICER_HANDSHAKE_PORT"
DRAIN_SECS_ENV = "SMG_SGLANG_SERVICER_DRAIN_SECS"
STARTUP_TIMEOUT_SECS_ENV = "SMG_SGLANG_SERVICER_STARTUP_TIMEOUT_SECS"
# Set to 0/false/no/off to keep SGLang's KV event publisher off when the
# launcher was given no --kv-events-config (see `default_kv_events_config`).
KV_EVENTS_ENV = "SMG_SGLANG_SERVICER_KV_EVENTS"
_OFF_VALUES = ("0", "false", "no", "off")
# What a launcher without --kv-events-config gets under the Rust servicer:
# the ZMQ publisher, with SGLang's own defaults for the rest (the endpoint,
# the topic; no replay socket).
DEFAULT_KV_EVENTS_CONFIG = '{"publisher": "zmq"}'
IMPLS = ("python", "rust")


def resolve_servicer_impl(environ: Mapping[str, str] | None = None) -> str:
    """Which implementation serves this process: ``$SMG_SGLANG_SERVICER_IMPL``,
    else python."""
    source = os.environ if environ is None else environ
    value = str(source.get(SERVICER_IMPL_ENV) or "python").strip().lower()
    if value not in IMPLS:
        raise ValueError(f"{SERVICER_IMPL_ENV} must be one of {IMPLS}, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# What the Rust server advertises, from SGLang's own config
# ---------------------------------------------------------------------------


def _model_config(server_args: Any) -> Any:
    """SGLang's ``ModelConfig`` for the served model, as the Python servicer
    builds it."""
    from sglang.srt.configs.model_config import ModelConfig

    return ModelConfig.from_server_args(server_args)


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, str | int | float | bool):
        return obj
    if isinstance(obj, list | tuple | set):
        return [_json_safe(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): _json_safe(value) for key, value in obj.items()}
    return str(obj)


def _preferred_sampling_params(server_args: Any) -> dict[str, Any] | None:
    preferred = getattr(server_args, "preferred_sampling_params", None)
    if isinstance(preferred, str):
        try:
            preferred = json.loads(preferred)
        except json.JSONDecodeError:
            logger.warning("Failed to parse preferred_sampling_params JSON")
            return None
    return preferred if isinstance(preferred, dict) else None


def model_facts(server_args: Any, model_config: Any | None = None) -> dict[str, Any]:
    """``SglangGrpcServer`` keyword arguments for ``GetModelInfo``: the Python
    servicer's ``GetModelInfo`` fields, computed up front from the model
    config (the handshake supplies the scheduler's capacity figures later)."""
    model_config = model_config if model_config is not None else _model_config(server_args)
    hf_config = getattr(model_config, "hf_config", None)
    eos = getattr(model_config, "hf_eos_token_id", None)
    if isinstance(eos, int):
        eos_token_ids = [eos]
    elif isinstance(eos, list | tuple | set | frozenset):
        eos_token_ids = sorted(int(value) for value in eos)
    else:
        eos_token_ids = []
    # Classification heads, as the Python servicer derives them.
    id2label = getattr(hf_config, "id2label", None) if hf_config is not None else None
    num_labels = (getattr(hf_config, "num_labels", 0) or 0) if hf_config is not None else 0
    if not id2label and num_labels:
        id2label = {index: f"LABEL_{index}" for index in range(num_labels)}
    elif id2label and not num_labels:
        num_labels = len(id2label)
    defaults = dict(model_config.get_default_sampling_params() or {})
    preferred = _preferred_sampling_params(server_args)
    if preferred:
        defaults.update(preferred)
    model_path = str(getattr(server_args, "model_path", "") or "")
    context_len = int(getattr(model_config, "context_len", 0) or 0)
    return {
        "model_path": model_path,
        "tokenizer_path": str(getattr(server_args, "tokenizer_path", None) or model_path),
        "served_model_name": str(getattr(server_args, "served_model_name", None) or model_path),
        "is_generation": bool(getattr(model_config, "is_generation", True)),
        "model_type": str((getattr(hf_config, "model_type", "") or "") if hf_config else ""),
        "architectures": list((getattr(hf_config, "architectures", []) or []) if hf_config else []),
        "max_context_length": context_len,
        "max_req_input_len": context_len,
        "vocab_size": int(getattr(model_config, "vocab_size", 0) or 0),
        "eos_token_ids": eos_token_ids,
        "pad_token_id": int((getattr(hf_config, "pad_token_id", 0) or 0) if hf_config else 0),
        "bos_token_id": int((getattr(hf_config, "bos_token_id", 0) or 0) if hf_config else 0),
        "weight_version": str(getattr(server_args, "weight_version", None) or ""),
        "preferred_sampling_params": (
            json.dumps(preferred, separators=(",", ":"), sort_keys=True) if preferred else ""
        ),
        "default_sampling_params_json": (
            json.dumps(defaults, separators=(",", ":"), sort_keys=True) if defaults else ""
        ),
        "supports_vision": bool(getattr(model_config, "is_multimodal", False)),
        "id2label_json": json.dumps(_json_safe(id2label)) if id2label else "",
        "num_labels": int(num_labels or 0),
    }


def server_facts(server_args: Any) -> dict[str, Any]:
    """``SglangGrpcServer`` keyword arguments for ``GetServerInfo`` and
    ``GetLoads``: the server args as the Router's label source (with the
    operator's pairing protocol, as the Python servicer adds it), the
    admission window and the data-parallel width."""
    try:
        import msgspec

        is_struct = isinstance(server_args, msgspec.Struct)
    except ImportError:  # the launcher's unit tests run without SGLang's deps
        is_struct = False
    if is_struct:
        import msgspec

        args_dict = msgspec.structs.asdict(server_args)
    elif dataclasses.is_dataclass(server_args) and not isinstance(server_args, type):
        args_dict = dataclasses.asdict(server_args)
    else:
        args_dict = dict(getattr(server_args, "__dict__", {}))
    args_dict = _json_safe(args_dict)
    pairing_protocol = pairing_protocol_from_env()
    if pairing_protocol:
        args_dict["pairing_protocol"] = pairing_protocol
    try:
        from sglang.version import __version__ as sglang_version
    except ImportError:  # the launcher's unit tests run without SGLang
        sglang_version = ""
    dp_size = getattr(server_args, "dp_size", None)
    kv_events_endpoint, kv_events_replay_endpoint, kv_events_topic = kv_events_publisher(
        server_args
    )
    return {
        # allow_nan=False: a non-finite float would otherwise become `NaN` text,
        # which the Rust side rejects as a whole object.
        "server_args_json": json.dumps(args_dict, sort_keys=True, allow_nan=False),
        "scheduler_info_json": "{}",
        "sglang_version": str(sglang_version),
        "max_running_requests": int(getattr(server_args, "max_running_requests", None) or 0),
        "data_parallel_size": int(dp_size) if isinstance(dp_size, int) and dp_size > 0 else 1,
        "kv_events_endpoint": kv_events_endpoint,
        "kv_events_replay_endpoint": kv_events_replay_endpoint,
        "kv_events_topic": kv_events_topic,
    }


def kv_events_publisher(server_args: Any) -> tuple[str, str, str]:
    """The ZMQ KV-event publisher SGLang was told to run (``--kv-events-config``)
    as ``(endpoint, replay_endpoint, topic)``, with SGLang's defaults filled in;
    empty strings when events are off or the publisher is not ZMQ, and an empty
    replay endpoint when SGLang runs no replay socket. The Rust servicer relays
    this publisher on ``SubscribeKvEvents`` and asks the replay socket for gaps
    and for the batches published before its subscription joined."""
    raw = getattr(server_args, "kv_events_config", None)
    if not raw:
        return "", "", ""
    try:
        config = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        return "", "", ""
    if not isinstance(config, dict) or config.get("publisher", "null") != "zmq":
        return "", "", ""
    return (
        str(config.get("endpoint") or "tcp://*:5557"),
        str(config.get("replay_endpoint") or ""),
        str(config.get("topic") or ""),
    )


def default_kv_events_config(
    server_args: Any, environ: Mapping[str, str] | None = None
) -> str | None:
    """Turn SGLang's KV event publisher on when the launcher was given no
    ``--kv-events-config``; returns the configuration applied, or None.

    Cache-aware routing lives on the events ``SubscribeKvEvents`` relays, and
    SGLang publishes none unless started with ``--kv-events-config
    '{"publisher": "zmq"}'``: a launcher without it got a router that routed
    blind, with one WARN per worker as the only trace. Under the Rust
    servicer the server args that lack the option get exactly that
    configuration before the facts and the headless scheduler are built from
    them, so publisher and relay agree. An explicit ``--kv-events-config`` is
    kept as given, off included; ``SMG_SGLANG_SERVICER_KV_EVENTS=0`` keeps
    the publisher off without one.
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
# Headless scheduler: this package's launcher, dialing this servicer
# ---------------------------------------------------------------------------


def _run_headless(server_args: Any, handshake_address: str) -> None:
    """Child target: the headless launcher on the already-prepared args."""
    from smg_grpc_servicer.sglang.headless import run

    raise SystemExit(run(server_args, handshake_address))


def launch_headless_scheduler(server_args: Any, *, handshake_port: int) -> EngineProcess:
    """Start the headless scheduler ranks in a spawned child: a fresh
    interpreter that inherits no Rust thread and never consults the servicer
    switch."""
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_run_headless,
        args=(server_args, f"tcp://127.0.0.1:{int(handshake_port)}"),
        name="smg-headless-scheduler",
    )
    process.start()
    return EngineProcess(process)


def tokenizer_dir_for(server_args: Any) -> str | None:
    """The local tokenizer directory for the served model, or ``None``."""
    tokenizer = getattr(server_args, "tokenizer_path", None) or getattr(
        server_args, "model_path", ""
    )
    return resolve_tokenizer_dir(str(tokenizer), getattr(server_args, "revision", None))


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def refuse_disaggregation(server_args: Any) -> None:
    """PD/EPD stays with the Python servicer: the msgpack wire carries no
    bootstrap fields and this path starts no bootstrap server, so such a
    worker must not start rather than report SERVING and hang requests."""
    mode = getattr(server_args, "disaggregation_mode", "null") or "null"
    if mode != "null":
        raise ValueError(
            f"{SERVICER_IMPL_ENV}=rust does not support PD disaggregation "
            f"(disaggregation_mode={mode!r}); unset it for this worker"
        )
    if getattr(server_args, "language_only", False) or getattr(server_args, "encoder_only", False):
        raise ValueError(
            f"{SERVICER_IMPL_ENV}=rust does not support EPD disaggregation "
            "(language_only / encoder_only); unset it for this worker"
        )


async def serve_rust(server_args: Any) -> int:
    """Serve this process's gRPC contract from Rust: start the Rust server,
    launch the headless scheduler against its handshake, and supervise both
    until a signal or a failure. Returns the process exit code."""
    refuse_disaggregation(server_args)
    from smg.servicer import SglangGrpcServer, init_servicer_tracing

    init_servicer_tracing(None)
    if default_kv_events_config(server_args) is not None:
        logger.info(
            "KV event publishing enabled: no --kv-events-config was given, so SGLang's ZMQ "
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
            "SubscribeKvEvents relays SGLang's KV events from %s", facts["kv_events_endpoint"]
        )
    else:
        logger.warning(
            "SubscribeKvEvents is off (no ZMQ KV event publisher configured): a cache-aware "
            "router sees nothing of this engine's cache"
        )
    engine_count = facts["data_parallel_size"]
    if engine_count > 1:
        logger.info(
            "dp_size=%d on the Rust path: each rank dials the servicer with its own identity "
            "and control calls fan out per rank",
            engine_count,
        )
    tokenizer_dir = tokenizer_dir_for(server_args)
    if tokenizer_dir is None:
        logger.warning(
            "No local tokenizer directory for %s; GetTokenizer will be unavailable",
            getattr(server_args, "model_path", ""),
        )
    server = SglangGrpcServer(
        bind_address=f"{host}:{port}",
        ipc_base_url=f"ipc://{socket_dir}/sglang-servicer-{os.getpid()}",
        handshake_address=f"tcp://127.0.0.1:{handshake_port}",
        engine_count=engine_count,
        tokenizer_dir=tokenizer_dir,
        engine_startup_timeout_secs=env_float(
            STARTUP_TIMEOUT_SECS_ENV, DEFAULT_STARTUP_TIMEOUT_SECS
        ),
        **facts,
    )
    logger.info(
        "Rust SGLang servicer listening on %s; headless scheduler dials handshake port %d",
        server.address,
        handshake_port,
    )
    engine = launch_headless_scheduler(server_args, handshake_port=handshake_port)
    return await supervise(server, engine, drain_secs=env_float(DRAIN_SECS_ENV, DEFAULT_DRAIN_SECS))
