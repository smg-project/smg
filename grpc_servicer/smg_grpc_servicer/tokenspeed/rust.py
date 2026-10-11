"""The Rust request path of the TokenSpeed gRPC servicer, behind one flag.

``python -m smg_grpc_servicer.tokenspeed`` (what ``ts serve`` spawns) serves
``tokenspeed.grpc.scheduler.TokenSpeedScheduler`` from Python over TokenSpeed's
in-process ``AsyncLLM``. With ``--servicer-impl rust`` (this package's flag on
that launcher, see :func:`add_servicer_impl_argument`;
``SMG_TOKENSPEED_SERVICER_IMPL=rust`` is the fallback the flag overrides) the
same entrypoint hands the process to :func:`serve_rust` before an ``AsyncLLM``
or a Python gRPC server exists: the contract is served by the Rust
:class:`smg.servicer.TokenSpeedGrpcServer` on a Rust-owned thread, and the
scheduler rank processes are started from this same process (the way
TokenSpeed's own ``launch_scheduler_headless``, what ``ts serve --headless``
runs, starts them), dialing the servicer's same-host msgpack ZMQ handshake.
Python keeps the lifecycle only. The Router cannot tell the two
implementations apart.

What the msgpack wire does not carry, the Rust path reports rather than
emulates: ``FlushCache`` and profiling answer UNIMPLEMENTED, ranked
``top_logprobs`` and prompt logprobs are refused, and PD/EPD disaggregation
(bootstrap fields) stays with the Python implementation.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
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
from smg_grpc_servicer.pd_pairing import pairing_protocol_from_env
from smg_grpc_servicer.rust_lifecycle import (
    DEFAULT_DRAIN_SECS,
    DEFAULT_STARTUP_CEILING_SECS,
    DEFAULT_STARTUP_TIMEOUT_SECS,
    EngineProcess,
    EngineProcessGroup,
    FaultLatch,
    SchedulerProcesses,
    default_socket_dir,
    ensure_spawn_start_method,
    env_float,
    free_port,
    on_fault_signal,
    resolve_tokenizer_dir,
    supervise,
    wait_for_ready_reports,
)
from smg_grpc_servicer.structured_outputs import (
    grammar_start_labels,
    tokenspeed_structured_outputs_start,
)
from smg_grpc_servicer.tokenspeed.kv_events import resolve_kv_events_config
from smg_grpc_servicer.tokenspeed.loads import running_window
from smg_grpc_servicer.tokenspeed.redact import redact_secrets

logger = logging.getLogger(__name__)

SERVICER_IMPL_ENV = "SMG_TOKENSPEED_SERVICER_IMPL"
HANDSHAKE_PORT_ENV = "SMG_TOKENSPEED_SERVICER_HANDSHAKE_PORT"
DRAIN_SECS_ENV = "SMG_TOKENSPEED_SERVICER_DRAIN_SECS"
STARTUP_TIMEOUT_SECS_ENV = "SMG_TOKENSPEED_SERVICER_STARTUP_TIMEOUT_SECS"
STARTUP_CEILING_SECS_ENV = "SMG_TOKENSPEED_SERVICER_STARTUP_CEILING_SECS"
# Set to 0/false/no/off to keep TokenSpeed's KV event publisher off when the
# launcher was given no --kv-events-config (see `default_kv_events_config`).
KV_EVENTS_ENV = "SMG_TOKENSPEED_SERVICER_KV_EVENTS"
_OFF_VALUES = ("0", "false", "no", "off")
# What a launcher without --kv-events-config gets under the Rust servicer:
# the ZMQ publisher, with TokenSpeed's own defaults for the rest (the
# endpoint, the topic).
DEFAULT_KV_EVENTS_CONFIG = '{"enable_kv_cache_events": true, "publisher": "zmq"}'
IMPLS = ("python", "rust")
SERVICER_IMPL_FLAG = "--servicer-impl"
# Where the implementation choice came from, for the startup log line.
SOURCE_FLAG = "flag"
SOURCE_ENV = "env"
SOURCE_DEFAULT = "default"


def add_servicer_impl_argument(parser: argparse.ArgumentParser) -> bool:
    """Add ``--servicer-impl`` to a launcher's parser that lacks it; returns
    whether it was added."""
    actions = parser._option_string_actions  # noqa: SLF001 — argparse's registry of option strings
    if SERVICER_IMPL_FLAG in actions:
        return False
    parser.add_argument(
        SERVICER_IMPL_FLAG,
        dest="servicer_impl",
        choices=list(IMPLS),
        default=None,
        help=(
            "Which implementation serves the gRPC contract: the Python servicer (default) "
            "or the Rust one (smg.servicer.TokenSpeedGrpcServer, with the scheduler(s) "
            f"headless). Unset falls back to ${SERVICER_IMPL_ENV}."
        ),
    )
    return True


def resolve_servicer_impl(args: Any = None, environ: Mapping[str, str] | None = None) -> str:
    """Which implementation serves this process: ``args.servicer_impl`` when
    the launcher carried the flag, else ``$SMG_TOKENSPEED_SERVICER_IMPL``, else
    python."""
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
    origin = SOURCE_FLAG
    if not value:
        value = source.get(SERVICER_IMPL_ENV)
        origin = SOURCE_ENV if value else SOURCE_DEFAULT
    value = str(value or "python").strip().lower()
    if value not in IMPLS:
        raise ValueError(f"{SERVICER_IMPL_ENV} must be one of {IMPLS}, got {value!r}")
    if origin == SOURCE_FLAG and isinstance(source, MutableMapping):
        source[SERVICER_IMPL_ENV] = value
    return value, origin


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
    # The engine's reasoning parser and where it starts a grammar on a thinking
    # prompt, as the Python servicer adds them.
    args_dict.update(grammar_start_labels(server_args, tokenspeed_structured_outputs_start()))
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
# Scheduler ranks: TokenSpeed's own launch from this process, dialing this servicer
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
    """Start TokenSpeed's headless launcher in a spawned child: a fresh
    interpreter that inherits no Rust thread and never consults the servicer
    switch. Only the cases :func:`launch_scheduler_ranks` cannot serve go
    this way: the child is a second full Python/torch process per engine
    pod."""
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_run_headless, args=(server_args,), name="smg-headless-scheduler"
    )
    process.start()
    return EngineProcess(process)


def launch_scheduler_ranks(server_args: Any) -> EngineProcessGroup | EngineProcess:
    """Start the scheduler rank processes from this process, the way
    TokenSpeed's headless launcher (``launch_scheduler_headless``, what ``ts
    serve --headless`` runs) starts them: the environment and start method it
    sets, one process per rank of this node on TokenSpeed's own scheduler
    entry point (or its data-parallel controller), each with a pipe for its
    ready report. ``server_args`` are the headless ones
    (:func:`headless_server_args`), so the ranks dial the servicer's
    handshake. This process has already imported TokenSpeed for the
    servicer's own facts, so hosting the launch here costs no second
    interpreter, where the spawned launcher was a full Python/torch process
    per engine pod, idle once the ranks ran.

    The ranks are spawned (TokenSpeed sets the start method), so they inherit
    no Rust thread. A rank that hits an exception signals its parent (SIGUSR1,
    then exits on its own terms; SIGQUIT from its process-tree cleanup);
    TokenSpeed's launch-phase handler kills the whole process tree, so the
    group's own handler takes its place and the servicer drains and exits
    through :func:`supervise`. The spawned launcher child is kept for a
    TokenSpeed that lacks these entry points (they are the launcher's own
    internals) and for a caller off the main thread, where no handler can be
    installed.
    """
    if threading.current_thread() is not threading.main_thread():
        logger.info(
            "Scheduler launch left to TokenSpeed's headless launcher in a spawned process "
            "(not on the main thread)"
        )
        return launch_headless_scheduler(server_args)
    try:
        from tokenspeed.runtime.entrypoints import engine as launcher

        set_envs_and_config = launcher._set_envs_and_config
        scheduler_ranks = launcher._scheduler_ranks
        run_event_loop = launcher.run_event_loop
        run_data_parallel_controller_process = launcher.run_data_parallel_controller_process
        prepare_model_and_tokenizer = launcher.prepare_model_and_tokenizer
        configure_logger = launcher.configure_logger
        port_args_type = launcher.PortArgs
        memory_saver_type = launcher.TorchMemorySaverAdapter
    except (ImportError, AttributeError) as error:
        logger.warning(
            "Scheduler launch left to TokenSpeed's headless launcher in a spawned process: %s",
            error,
        )
        return launch_headless_scheduler(server_args)

    configure_logger(server_args)
    set_envs_and_config(server_args)
    ensure_spawn_start_method("TokenSpeed scheduler")
    port_args = port_args_type.init_new(server_args)
    server_args.model, server_args.tokenizer = prepare_model_and_tokenizer(
        server_args.model, server_args.tokenizer
    )
    # The processes are built first and started only once the fault handlers
    # are in: a rank that fails during its start signals its parent at once.
    processes: list = []
    readers: list = []
    writers: list = []
    starts: list = []
    if not server_args.mapping.attn.has_dp:
        memory_saver = memory_saver_type.create(enable=server_args.enable_memory_saver)
        for rank in scheduler_ranks(server_args):
            rank_args = copy.copy(server_args)
            rank_args.mapping = copy.deepcopy(server_args.mapping)
            rank_args.mapping.rank = rank
            reader, writer = multiprocessing.Pipe(duplex=False)
            processes.append(
                multiprocessing.Process(
                    target=run_event_loop,
                    args=(rank_args, port_args, writer),
                    name=f"smg-scheduler-rank-{rank}",
                )
            )
            readers.append(reader)
            writers.append(writer)
            starts.append(memory_saver.configure_subprocess)
    else:
        reader, writer = multiprocessing.Pipe(duplex=False)
        processes.append(
            multiprocessing.Process(
                target=run_data_parallel_controller_process,
                args=(server_args, port_args, writer),
                name="smg-scheduler-dp-controller",
            )
        )
        readers.append(reader)
        writers.append(writer)
        starts.append(contextlib.nullcontext)
    latch = FaultLatch()
    on_fault_signal(latch, signal.SIGUSR1, signal.SIGQUIT, what="TokenSpeed scheduler")
    for process, writer, around_start in zip(processes, writers, starts):
        with around_start():
            process.start()
        # The child holds its end of the pipe; this one closes, so a rank that
        # dies before reporting shows as EOF on the reader rather than as a
        # wait, and no descriptor pair stays open for the servicer's life.
        writer.close()
    ranks = SchedulerProcesses(processes, ready=lambda: wait_for_ready_reports(readers, processes))
    engine = latch.attach(EngineProcessGroup(ranks))
    ranks.watch_ready("TokenSpeed scheduler", on_failure=engine.fault)
    logger.info(
        "Started %d scheduler rank process(es) from the servicer process: pids %s",
        len(ranks.processes),
        ", ".join(str(pid) for pid in ranks.pids),
    )
    return engine


def tokenizer_dir_for(server_args: Any) -> str | None:
    """The local tokenizer directory for the served model, or ``None``."""
    tokenizer = getattr(server_args, "tokenizer", None) or getattr(server_args, "model", "")
    return resolve_tokenizer_dir(str(tokenizer), getattr(server_args, "revision", None))


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def serve_rust(server_args: Any) -> int:
    """Serve this process's gRPC contract from Rust: start the Rust server,
    start the scheduler ranks against its handshake, and supervise both
    until a signal or a failure. Returns the process exit code."""
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
    startup_timeout_secs = env_float(STARTUP_TIMEOUT_SECS_ENV, DEFAULT_STARTUP_TIMEOUT_SECS)
    startup_ceiling_secs = env_float(STARTUP_CEILING_SECS_ENV, DEFAULT_STARTUP_CEILING_SECS)
    server = TokenSpeedGrpcServer(
        bind_address=host_port(host, port),
        ipc_base_url=f"ipc://{socket_dir}/tokenspeed-servicer-{os.getpid()}",
        handshake_address=f"tcp://127.0.0.1:{handshake_port}",
        engine_count=engine_count,
        tokenizer_dir=tokenizer_dir,
        engine_startup_timeout_secs=startup_timeout_secs,
        engine_startup_ceiling_secs=startup_ceiling_secs,
        **facts,
    )
    logger.info(
        "Rust TokenSpeed servicer listening on %s; scheduler ranks dial handshake port %d",
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
    engine = launch_scheduler_ranks(
        headless_server_args(server_args, handshake_port=handshake_port)
    )
    return await supervise(server, engine, drain_secs=env_float(DRAIN_SECS_ENV, DEFAULT_DRAIN_SECS))
