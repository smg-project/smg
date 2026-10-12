"""The Rust request path of the SGLang gRPC servicer, behind one flag.

``sglang.launch_server --grpc-mode`` hands the process to this package's
:func:`smg_grpc_servicer.sglang.server.serve_grpc`, which serves
``sglang.grpc.scheduler.SglangScheduler`` from Python over a request manager
in front of the scheduler. With ``--servicer-impl rust`` (a flag this package
adds to SGLang's parser as an SGLang plugin, see
:mod:`smg_grpc_servicer.sglang.plugin`; ``SMG_SGLANG_SERVICER_IMPL=rust`` is
the fallback the flag overrides) the same entrypoint hands the process to
:func:`serve_rust` before any of that exists:
the contract is served by the Rust :class:`smg.servicer.SglangGrpcServer` on a
Rust-owned thread, and the scheduler ranks are started from this same process
(SGLang's own spawner, the way this package's :mod:`headless` launcher starts
them), each with this package's SGLang plugin dialing the servicer's same-host
msgpack ZMQ handshake. Python keeps the lifecycle only. The Router cannot tell
the two implementations apart.

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
import signal
import threading
from collections.abc import Mapping, MutableMapping
from typing import Any

from smg_grpc_servicer.hostport import host_port
from smg_grpc_servicer.mm_item_limits import MM_ITEM_LIMITS_KEY, sglang_item_limits
from smg_grpc_servicer.pd_pairing import pairing_protocol_from_env

# The lifecycle's defaults, re-exported: this launcher's own documented defaults.
from smg_grpc_servicer.rust_lifecycle import (  # noqa: F401
    DEFAULT_DRAIN_SECS,
    DEFAULT_STARTUP_CEILING_SECS,
    DEFAULT_STARTUP_TIMEOUT_SECS,
    EngineProcess,
    EngineProcessGroup,
    FaultLatch,
    SchedulerProcesses,
    ServicerConfigError,
    default_socket_dir,
    ensure_spawn_start_method,
    free_port,
    on_fault_signal,
    read_launcher_knobs,
    resolve_tokenizer_dir,
    supervise,
)
from smg_grpc_servicer.structured_outputs import (
    grammar_start_labels,
    sglang_structured_outputs_start,
)

logger = logging.getLogger(__name__)

SERVICER_IMPL_ENV = "SMG_SGLANG_SERVICER_IMPL"
HANDSHAKE_PORT_ENV = "SMG_SGLANG_SERVICER_HANDSHAKE_PORT"
DRAIN_SECS_ENV = "SMG_SGLANG_SERVICER_DRAIN_SECS"
STARTUP_TIMEOUT_SECS_ENV = "SMG_SGLANG_SERVICER_STARTUP_TIMEOUT_SECS"
STARTUP_CEILING_SECS_ENV = "SMG_SGLANG_SERVICER_STARTUP_CEILING_SECS"
# Set to 0/false/no/off to keep SGLang's KV event publisher off when the
# launcher was given no --kv-events-config (see `default_kv_events_config`).
KV_EVENTS_ENV = "SMG_SGLANG_SERVICER_KV_EVENTS"
_OFF_VALUES = ("0", "false", "no", "off")
# What a launcher without --kv-events-config gets under the Rust servicer:
# the ZMQ publisher, with SGLang's own defaults for the rest (the endpoint,
# the topic; no replay socket).
DEFAULT_KV_EVENTS_CONFIG = '{"publisher": "zmq"}'
IMPLS = ("python", "rust")
# Where the implementation choice came from, for the startup log line.
SOURCE_FLAG = "flag"
SOURCE_ENV = "env"
SOURCE_DEFAULT = "default"


def resolve_servicer_impl(args: Any = None, environ: Mapping[str, str] | None = None) -> str:
    """Which implementation serves this process: the launcher's
    ``--servicer-impl`` when it carried one (``args.servicer_impl``, or the
    value this package's SGLang plugin parsed in this process), else
    ``$SMG_SGLANG_SERVICER_IMPL``, else python."""
    return servicer_impl_source(args, environ)[0]


def servicer_impl_source(
    args: Any = None, environ: Mapping[str, str] | None = None
) -> tuple[str, str]:
    """:func:`resolve_servicer_impl`'s answer with where it came from:
    ``flag``, ``env`` or ``default``.

    A decision made with the launcher's flag in hand is written back to the
    environment, so the scheduler processes this process starts and anything
    that reads only the variable agree with it when ``--servicer-impl python``
    overrides an exported ``rust``."""
    source = os.environ if environ is None else environ
    value = getattr(args, "servicer_impl", None) if args is not None else None
    if not value:
        from smg_grpc_servicer.sglang import plugin

        value = plugin.parsed_flag()
    origin = SOURCE_FLAG
    if not value:
        value = source.get(SERVICER_IMPL_ENV)
        origin = SOURCE_ENV if value else SOURCE_DEFAULT
    value = str(value or "python").strip().lower()
    if value not in IMPLS:
        raise ServicerConfigError(f"{SERVICER_IMPL_ENV} must be one of {IMPLS}, got {value!r}")
    if origin == SOURCE_FLAG and isinstance(source, MutableMapping):
        source[SERVICER_IMPL_ENV] = value
    return value, origin


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
    # The engine's own per-request media limits, flat, as the Python servicer
    # adds them: the Router's media pipeline holds requests to them.
    item_limits = sglang_item_limits(server_args)
    if item_limits:
        args_dict[MM_ITEM_LIMITS_KEY] = item_limits
    # The engine's reasoning parser and where it starts a grammar on a thinking
    # prompt, as the Python servicer adds them.
    args_dict.update(
        grammar_start_labels(server_args, sglang_structured_outputs_start(server_args))
    )
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
# Scheduler ranks: SGLang's own spawner from this process, dialing this servicer
# ---------------------------------------------------------------------------


def _run_headless(server_args: Any, handshake_address: str) -> None:
    """Child target: the headless launcher on the already-prepared args."""
    from smg_grpc_servicer.sglang.headless import run

    raise SystemExit(run(server_args, handshake_address))


def launch_headless_scheduler(server_args: Any, *, handshake_port: int) -> EngineProcess:
    """Start this package's headless launcher in a spawned child: a fresh
    interpreter that inherits no Rust thread and never consults the servicer
    switch. Only a caller off the main thread goes this way (see
    :func:`launch_scheduler_ranks`): the child is a second full Python/torch
    process per engine pod."""
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_run_headless,
        args=(server_args, f"tcp://127.0.0.1:{int(handshake_port)}"),
        name="smg-headless-scheduler",
    )
    process.start()
    return EngineProcess(process)


def launch_scheduler_ranks(
    server_args: Any, *, handshake_port: int
) -> EngineProcessGroup | EngineProcess:
    """Start the scheduler ranks from this process, the way this package's
    headless launcher (:func:`headless.run`) starts them: the server args
    prepared as SGLang's entrypoint prepares them, then SGLang's own spawner
    (``Engine._launch_scheduler_processes``, which knows the running
    version's rank layout), with this package's plugin dialing the servicer's
    handshake from each rank. This process has already imported SGLang for
    the servicer's own facts, so hosting the launch here costs no second
    interpreter, where the spawned launcher was a full Python/torch process
    per engine pod, idle once the ranks ran.

    The ranks are spawned (SGLang sets the start method), so they inherit no
    Rust thread. A rank that hits an exception signals its parent (SIGQUIT);
    SGLang's launch-phase handler for it kills the whole process tree, so the
    group's own handler takes its place and the servicer drains and exits
    through :func:`supervise`. Off the main thread, where no handler can be
    installed, the spawned launcher child is kept.
    """
    if threading.current_thread() is not threading.main_thread():
        logger.info(
            "Scheduler launch left to the headless launcher in a spawned process "
            "(not on the main thread)"
        )
        return launch_headless_scheduler(server_args, handshake_port=handshake_port)
    from sglang.srt.entrypoints.engine import Engine, _set_envs_and_config
    from sglang.srt.managers.scheduler import run_scheduler_process
    from sglang.srt.plugins import load_plugins
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import PortArgs
    from sglang.srt.utils import configure_logger

    from smg_grpc_servicer.sglang.zmq_plugin import ENGINE_INDEX_ENV, HANDSHAKE_ENV

    # The ranks inherit the environment; the plugin registers its hooks on it.
    os.environ[HANDSHAKE_ENV] = f"tcp://127.0.0.1:{int(handshake_port)}"
    os.environ[ENGINE_INDEX_ENV] = "0"
    configure_logger(server_args)
    server_args.resolve_once()
    _set_envs_and_config(server_args)
    ensure_spawn_start_method("SGLang scheduler")
    load_plugins()
    server_args.check_server_args()
    # This process fills the tokenizer manager's role for argument resolution
    # only; the ranks publish their own records when they start.
    publish(server_args, role="tokenizer")
    port_args = PortArgs.init_new(server_args)
    # The fault handler goes in before the spawner starts a rank: a rank that
    # fails during its start signals its parent at once.
    latch = FaultLatch()
    on_fault_signal(latch, signal.SIGQUIT, what="SGLang scheduler")
    init, processes = Engine._launch_scheduler_processes(
        server_args, port_args, run_scheduler_process
    )
    ranks = SchedulerProcesses(list(processes or []), ready=init.wait_for_ready)
    engine = latch.attach(EngineProcessGroup(ranks))
    ranks.watch_ready("SGLang scheduler", on_failure=engine.fault)
    logger.info(
        "Started %d scheduler rank process(es) from the servicer process: pids %s",
        len(ranks.processes),
        ", ".join(str(pid) for pid in ranks.pids),
    )
    return engine


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
        raise ServicerConfigError(
            f"{SERVICER_IMPL_ENV}=rust does not support PD disaggregation "
            f"(disaggregation_mode={mode!r}); unset it for this worker"
        )
    if getattr(server_args, "language_only", False) or getattr(server_args, "encoder_only", False):
        raise ServicerConfigError(
            f"{SERVICER_IMPL_ENV}=rust does not support EPD disaggregation "
            "(language_only / encoder_only); unset it for this worker"
        )


async def serve_rust(server_args: Any) -> int:
    """Serve this process's gRPC contract from Rust: start the Rust server,
    start the scheduler ranks against its handshake, and supervise both
    until a signal or a failure. Returns the process exit code."""
    refuse_disaggregation(server_args)
    # The launcher's own knobs first, validated together: a bad value is a
    # named error before any engine work and before the port is bound.
    knobs = read_launcher_knobs(
        handshake_port_env=HANDSHAKE_PORT_ENV,
        drain_secs_env=DRAIN_SECS_ENV,
        startup_timeout_secs_env=STARTUP_TIMEOUT_SECS_ENV,
        startup_ceiling_secs_env=STARTUP_CEILING_SECS_ENV,
        socket_dir=default_socket_dir(),
    )
    from smg.servicer import SglangGrpcServer, init_servicer_tracing

    init_servicer_tracing(None)
    if default_kv_events_config(server_args) is not None:
        logger.info(
            "KV event publishing enabled: no --kv-events-config was given, so SGLang's ZMQ "
            "publisher is on with its default endpoint and SubscribeKvEvents relays it; pass "
            "--kv-events-config to configure it, or set %s=0 to leave it off",
            KV_EVENTS_ENV,
        )
    handshake_port = knobs.handshake_port or free_port()
    socket_dir = knobs.socket_dir
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
    startup_timeout_secs = knobs.startup_timeout_secs
    startup_ceiling_secs = knobs.startup_ceiling_secs
    server = SglangGrpcServer(
        bind_address=host_port(host, port),
        ipc_base_url=f"ipc://{socket_dir}/sglang-servicer-{os.getpid()}",
        handshake_address=f"tcp://127.0.0.1:{handshake_port}",
        engine_count=engine_count,
        tokenizer_dir=tokenizer_dir,
        engine_startup_timeout_secs=startup_timeout_secs,
        engine_startup_ceiling_secs=startup_ceiling_secs,
        **facts,
    )
    logger.info(
        "Rust SGLang servicer listening on %s; scheduler ranks dial handshake port %d",
        server.address,
        handshake_port,
    )
    logger.info(
        "Scheduler startup bounds: %.0fs without a sign of life (%s), ceiling %s (%s); "
        "the handshake waits while the scheduler process is alive",
        startup_timeout_secs,
        STARTUP_TIMEOUT_SECS_ENV,
        f"{startup_ceiling_secs:.0f}s" if startup_ceiling_secs else "none",
        STARTUP_CEILING_SECS_ENV,
    )
    engine = launch_scheduler_ranks(server_args, handshake_port=handshake_port)
    return await supervise(server, engine, drain_secs=knobs.drain_secs)
