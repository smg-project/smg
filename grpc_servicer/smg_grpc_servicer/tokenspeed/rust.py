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
import signal
import threading
from collections.abc import Mapping
from typing import Any

from smg_grpc_servicer.constrained_decoding import (
    grammar_start_labels,
    tokenspeed_constrained_decoding_start,
)
from smg_grpc_servicer.pd_pairing import pairing_protocol_from_env

# The lifecycle's defaults and the switch's choices, re-exported: this
# launcher's own documented values.
from smg_grpc_servicer.rust_lifecycle import (  # noqa: F401
    DEFAULT_DRAIN_SECS,
    DEFAULT_STARTUP_CEILING_SECS,
    DEFAULT_STARTUP_TIMEOUT_SECS,
    IMPLS,
    EngineProcess,
    EngineProcessGroup,
    FaultLatch,
    RustLauncher,
    SchedulerProcesses,
    apply_kv_events_default,
    ensure_spawn_start_method,
    on_fault_signal,
    resolve_tokenizer_dir,
    servicer_switch,
    wait_for_ready_reports,
)
from smg_grpc_servicer.tokenspeed.config_view import EngineConfigView
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
# What a launcher without --kv-events-config gets under the Rust servicer:
# the ZMQ publisher, with TokenSpeed's own defaults for the rest (the
# endpoint, the topic).
DEFAULT_KV_EVENTS_CONFIG = '{"enable_kv_cache_events": true, "publisher": "zmq"}'
SERVICER_IMPL_FLAG = "--servicer-impl"


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
    ``flag``, ``env`` or ``default`` (the shared
    :func:`~smg_grpc_servicer.rust_lifecycle.servicer_switch`, which writes a
    flag decision back to the environment the scheduler processes inherit)."""
    return servicer_switch(SERVICER_IMPL_ENV, args, environ)


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


def _config_section(config: Any, name: str) -> Any:
    """``config[name]`` or ``config.name``, ``None`` when absent."""
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def mrope_positions(hf_config: Any) -> bool | None:
    """Whether the language model's rotary positions are MRoPE, read off the
    checkpoint's text rope config: ``rope_scaling`` / ``rope_parameters`` (on the
    text config when the checkpoint nests one, else the top level) with an
    ``mrope_section``, or a rope type named ``mrope``. A checkpoint whose rope
    config is plain (``rope_type: default`` with no section) uses 1-D positions,
    so its vision grid tensors only lay out the patches. ``None`` when the
    config carries no rope section at all."""
    if hf_config is None:
        return None
    text = _config_section(hf_config, "text_config") or hf_config
    found = False
    for key in ("rope_scaling", "rope_parameters"):
        rope = _config_section(text, key)
        if rope is None:
            continue
        found = True
        if _config_section(rope, "mrope_section") is not None:
            return True
        kind = _config_section(rope, "rope_type") or _config_section(rope, "type")
        if isinstance(kind, str) and "mrope" in kind:
            return True
    return False if found else None


def model_facts(server_args: Any, model_config: Any | None = None) -> dict[str, Any]:
    """``TokenSpeedGrpcServer`` keyword arguments for ``GetModelInfo``: the
    Python servicer's ``GetModelInfo`` fields, computed up front."""
    from smg_grpc_servicer.tokenspeed.servicer import TokenSpeedSchedulerServicer

    model_config = model_config if model_config is not None else _model_config(server_args)
    view = EngineConfigView(server_args, model_config)
    args, model = view.server_args, view.model
    supported_modalities = TokenSpeedSchedulerServicer._static_supported_modalities(
        model_config, model.hf_config
    )
    dtype = _dtype_name(model.dtype)
    return {
        "model_path": args.model,
        "tokenizer_path": args.tokenizer or args.model,
        "served_model_name": str(args.served_model_name or args.model),
        "model_type": model.model_type,
        "architectures": model.architectures,
        "max_context_length": model.context_len,
        "vocab_size": model.vocab_size,
        "eos_token_ids": model.eos_token_ids,
        "pad_token_id": model.pad_token_id,
        "bos_token_id": model.bos_token_id,
        # The engine's declared constrained-decoding policy: off the server args,
        # else the model config, as the Python servicer reads it.
        "constrained_decoding": view.constrained_decoding,
        "default_sampling_params_json": str(args.preferred_sampling_params or ""),
        "supports_vision": any(modality in (1, 3) for modality in supported_modalities),
        "supports_multimodal": bool(supported_modalities),
        "supported_modalities": [int(modality) for modality in supported_modalities],
        "mrope_positions": mrope_positions(model.hf_config),
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
    dp_size = EngineConfigView(server_args).server_args.attention_dp_size
    if dp_size is not None and dp_size > 1:
        args_dict["dp_size"] = dp_size
    pairing_protocol = pairing_protocol_from_env()
    if pairing_protocol:
        args_dict["pairing_protocol"] = pairing_protocol
    # The engine's reasoning parser and where it starts a grammar on a thinking
    # prompt, as the Python servicer adds them.
    args_dict.update(grammar_start_labels(server_args, tokenspeed_constrained_decoding_start()))
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
        "data_parallel_size": dp_size or 1,
        "kv_events_endpoint": kv_events.endpoint if kv_events else "",
        "kv_events_replay_endpoint": kv_events.replay_endpoint if kv_events else "",
        "kv_events_topic": kv_events.topic if kv_events else "",
    }


def default_kv_events_config(
    server_args: Any, environ: Mapping[str, str] | None = None
) -> str | None:
    """Turn TokenSpeed's KV event publisher on when the launcher was given no
    ``--kv-events-config``; returns the configuration applied, or None: the
    shared rule (``rust_lifecycle.apply_kv_events_default``) with
    ``DEFAULT_KV_EVENTS_CONFIG``, applied to the server args before the facts
    and the headless scheduler are built from them."""
    return apply_kv_events_default(
        server_args, DEFAULT_KV_EVENTS_CONFIG, env_name=KV_EVENTS_ENV, environ=environ
    )


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
    args = EngineConfigView(server_args).server_args
    return resolve_tokenizer_dir(args.tokenizer or args.model, args.revision)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TokenSpeedLauncher(RustLauncher):
    """The TokenSpeed launcher's lifecycle: the facts come from TokenSpeed's
    model config and server args, and the scheduler rank processes start
    from this process on the headless server args."""

    engine = "TokenSpeed"
    started = "scheduler"
    counted = "DP rank"
    server_class = "TokenSpeedGrpcServer"
    ipc_prefix = "tokenspeed-servicer"
    handshake_port_env = HANDSHAKE_PORT_ENV
    drain_secs_env = DRAIN_SECS_ENV
    startup_timeout_secs_env = STARTUP_TIMEOUT_SECS_ENV
    startup_ceiling_secs_env = STARTUP_CEILING_SECS_ENV
    kv_events_env = KV_EVENTS_ENV
    without_tokenizer = "string stops will be refused"

    def kv_events_default(self, args: Any) -> str | None:
        return default_kv_events_config(args)

    def facts(self, args: Any) -> dict[str, Any]:
        return {**model_facts(args), **server_facts(args)}

    def tokenizer_dir(self, args: Any) -> str | None:
        return tokenizer_dir_for(args)

    def launch(self, args: Any, handshake_port: int, facts: Mapping[str, Any]) -> Any:
        return launch_scheduler_ranks(headless_server_args(args, handshake_port=handshake_port))


async def serve_rust(server_args: Any) -> int:
    """Serve this process's gRPC contract from Rust: start the Rust server,
    start the scheduler ranks against its handshake, and supervise both
    until a signal or a failure (:class:`TokenSpeedLauncher`). Returns the
    process exit code."""
    return await TokenSpeedLauncher().serve(server_args)
