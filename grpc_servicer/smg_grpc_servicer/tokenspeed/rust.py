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
``top_logprobs`` and prompt logprobs are refused. Prefill/decode
disaggregation is served: the request's bootstrap fields ride the wire as the
Python servicer forwards them, and a prefill launch starts the engine's KV
bootstrap server here as the engine's own frontend does
(:func:`kv_bootstrap_server`). The EPD prefill side is served the same way;
the EPD encode role is not (:func:`refuse_encode_role`): its worker takes
its requests over the engine's in-process channel only, which this path
does not have, so it stays with the Python implementation.
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
    grammar_backend_entry,
    grammar_start_labels,
    tokenspeed_constrained_decoding_start,
    tokenspeed_grammar_backend,
)
from smg_grpc_servicer.metrics_flag import add_metrics_arguments as _add_metrics_arguments
from smg_grpc_servicer.pd_pairing import pairing_protocol_from_env

# The lifecycle's defaults and the switch's flag and choices, re-exported:
# this launcher's own documented values.
from smg_grpc_servicer.rust_lifecycle import (  # noqa: F401
    DEFAULT_DRAIN_SECS,
    DEFAULT_PREFILL_TOKENS_PER_SEC,
    DEFAULT_SILENCE_TIMEOUT_SECS,
    DEFAULT_STARTUP_CEILING_SECS,
    DEFAULT_STARTUP_TIMEOUT_SECS,
    IMPLS,
    SERVICER_IMPL_FLAG,
    EngineProcess,
    EngineProcessGroup,
    FaultLatch,
    RustLauncher,
    SchedulerProcesses,
    ServicerConfigError,
    apply_kv_events_default,
    ensure_spawn_start_method,
    on_fault_signal,
    resolve_tokenizer_dir,
    servicer_switch,
    wait_for_ready_reports,
)
from smg_grpc_servicer.servicer_impl_flag import (
    add_servicer_impl_argument as _add_servicer_impl_argument,
)
from smg_grpc_servicer.tokenspeed import METRICS_HOST_ENV, METRICS_PORT_ENV
from smg_grpc_servicer.tokenspeed.config_view import EngineConfigView
from smg_grpc_servicer.tokenspeed.kv_events import resolve_kv_events_config
from smg_grpc_servicer.tokenspeed.loads import running_window
from smg_grpc_servicer.tokenspeed.redact import redact_secrets
from smg_grpc_servicer.tokenspeed.warmup import (
    DEFAULT_BUDGET_SECS,
    WARMUP_ENV,
    warmup_inputs,
    warmup_wire,
)

logger = logging.getLogger(__name__)

SERVICER_IMPL_ENV = "SMG_TOKENSPEED_SERVICER_IMPL"
HANDSHAKE_PORT_ENV = "SMG_TOKENSPEED_SERVICER_HANDSHAKE_PORT"
DRAIN_SECS_ENV = "SMG_TOKENSPEED_SERVICER_DRAIN_SECS"
STARTUP_TIMEOUT_SECS_ENV = "SMG_TOKENSPEED_SERVICER_STARTUP_TIMEOUT_SECS"
STARTUP_CEILING_SECS_ENV = "SMG_TOKENSPEED_SERVICER_STARTUP_CEILING_SECS"
SILENCE_TIMEOUT_SECS_ENV = "SMG_TOKENSPEED_SERVICER_SILENCE_TIMEOUT_SECS"
PREFILL_TOKENS_PER_SEC_ENV = "SMG_TOKENSPEED_SERVICER_PREFILL_TOKENS_PER_SEC"
# Set to 0/false/no/off to keep TokenSpeed's KV event publisher off when the
# launcher was given no --kv-events-config (see `default_kv_events_config`).
KV_EVENTS_ENV = "SMG_TOKENSPEED_SERVICER_KV_EVENTS"


def default_kv_events_json(endpoint: str) -> str:
    """What a launcher without --kv-events-config gets under the Rust servicer,
    as TokenSpeed's server args carry it (JSON text): the ZMQ publisher at
    ``endpoint``, with TokenSpeed's own defaults for the rest (the topic)."""
    return json.dumps({"enable_kv_cache_events": True, "publisher": "zmq", "endpoint": endpoint})


def add_servicer_impl_argument(parser: argparse.ArgumentParser) -> bool:
    """Add ``--servicer-impl`` to a launcher's parser that lacks it; returns
    whether it was added."""
    return _add_servicer_impl_argument(
        parser,
        help=(
            "Which implementation serves the gRPC contract: the Python servicer (default) "
            "or the Rust one (smg.servicer.TokenSpeedGrpcServer, with the scheduler(s) "
            f"headless). Unset falls back to ${SERVICER_IMPL_ENV}."
        ),
    )


def add_metrics_arguments(parser: argparse.ArgumentParser) -> bool:
    """Add ``--metrics-port`` and ``--metrics-host`` (the Rust servicer's
    Prometheus endpoint) to a launcher's parser that lacks them; returns
    whether they were added."""
    return _add_metrics_arguments(
        parser,
        port_env=METRICS_PORT_ENV,
        host_env=METRICS_HOST_ENV,
        server_class="smg.servicer.TokenSpeedGrpcServer",
    )


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
    text config when the checkpoint nests one, and the top level) with an
    ``mrope_section``, or a rope type named ``mrope``. A checkpoint whose rope
    config is plain (``rope_type: default`` with no section) uses 1-D positions,
    so its vision grid tensors only lay out the patches. ``None`` when the
    config carries no rope section at all."""
    if hf_config is None:
        return None
    # Both the nested text config and the top level are read: a config class
    # may fill a plain rope default on one while the checkpoint declares its
    # MRoPE section on the other, and either declaration means MRoPE.
    sections = [hf_config]
    text = _config_section(hf_config, "text_config")
    if text is not None:
        sections.insert(0, text)
    found = False
    for section in sections:
        for key in ("rope_scaling", "rope_parameters"):
            rope = _config_section(section, key)
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
    # The grammar backend the scheduler runs, resolved (an unset flag is the
    # checkpoint's default, which the engine may not have read yet): the Rust
    # servicer refuses a constraint that backend cannot enforce before the
    # scheduler, and judges nothing when the backend is not known.
    grammar_backend_entry(args_dict, tokenspeed_grammar_backend(server_args))
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


def cache_layer_labels(model_config: Any) -> tuple[str, ...]:
    """The served model's per-layer cache-group labels, read as TokenSpeed's
    cache recipes read them off its ``ModelConfig`` (the checkpoint's
    ``cache_layer_types``, else its ``layer_types``, on the config the engine
    holds, else on the text config nested in it); empty for a checkpoint that
    declares none, whose layers are all full attention."""
    hf_config = getattr(model_config, "hf_config", None)
    for config in (hf_config, getattr(hf_config, "text_config", None)):
        labels = getattr(config, "cache_layer_types", None) or getattr(config, "layer_types", None)
        if labels:
            return tuple(str(label) for label in labels)
    return ()


def kv_events_unpublishable(server_args: Any, model_config: Any = None) -> str | None:
    """Why TokenSpeed's KV event publisher cannot serve the model this process
    loads, or ``None`` when it can.

    TokenSpeed keeps a hybrid model's recurrent state (its Gated DeltaNet,
    Mamba or KDA layers) in State groups of its cache: blocks that hold one
    checkpoint each and span no tokens. Its KV-event drain walks every cache
    group of a published boundary for a token-span quantity, and a State
    group has none: the scheduler's assertion (``a State group has no token
    span``) ends the engine process at the first drain of a request's tail,
    under the fused scheduler as under the prefill and decode roles. Which
    layer labels get a State group is the engine's own vocabulary (the
    recipes' ``STATE_LAYER_TYPES``, over :func:`cache_layer_labels`), so the
    answer follows the engine installed: one without that vocabulary has no
    State groups, and a model whose labels are all history (full, sliding or
    sparse attention) is served. ``model_config`` is TokenSpeed's
    ``ModelConfig`` for the served model, built here when not given
    (:func:`_model_config`), after the vocabulary was found."""
    try:
        from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import STATE_LAYER_TYPES
    except ImportError as error:
        # An engine that does not spell its State-group vocabulary there (an
        # older one, with no State groups, or one that moved the module), or
        # no engine at all (the launcher's unit tests): the default is not
        # guarded, and the log says so, since a hybrid model on an engine
        # that moved the module would abort in the drain again.
        logger.info(
            "TokenSpeed's State-group vocabulary could not be read (%s): the KV-events default "
            "is not guarded against State groups on this engine",
            error,
        )
        return None
    if model_config is None:
        model_config = _model_config(server_args)
    labels = cache_layer_labels(model_config)
    state = [label for label in labels if label in STATE_LAYER_TYPES]
    if not state:
        return None
    return (
        f"{EngineConfigView(server_args).server_args.model} keeps recurrent state in "
        f"{len(state)} of its {len(labels)} layers ({', '.join(sorted(set(state)))}): State "
        "groups of TokenSpeed's cache, which span no tokens, and TokenSpeed's KV-event drain "
        "aborts the engine on a group with no token span"
    )


def default_kv_events_config(
    server_args: Any, environ: Mapping[str, str] | None = None, model_config: Any = None
) -> str | None:
    """Turn TokenSpeed's KV event publisher on when the launcher was given no
    ``--kv-events-config``; returns the configuration applied, or None: the
    shared rule (``rust_lifecycle.apply_kv_events_default``) with
    :func:`default_kv_events_json` at a free port of this host, one per
    attention-DP rank (each publishes at the port after its predecessor's),
    applied to the server args before the facts and the schedulers are built
    from them, unless the engine cannot serve the publisher for this model
    (:func:`kv_events_unpublishable`, on ``model_config`` when the caller has
    TokenSpeed's ``ModelConfig`` already): then the publisher stays off and
    the reason is logged once."""
    return apply_kv_events_default(
        server_args,
        default_kv_events_json,
        env_name=KV_EVENTS_ENV,
        environ=environ,
        ranks=EngineConfigView(server_args).server_args.attention_dp_size or 1,
        unless=lambda: kv_events_unpublishable(server_args, model_config),
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
# Disaggregation: the prefill side's bootstrap server, the roles this path serves
# ---------------------------------------------------------------------------


def kv_bootstrap_server(server_args: Any) -> Any | None:
    """Start the engine's KV bootstrap server for a prefill launch and return
    it; ``None`` for any other role.

    Under prefill/decode disaggregation the prefill side hosts the rendezvous
    the decode side discovers its KV at (``--disaggregation-bootstrap-port``).
    The engine's own frontend starts it when it is built, which the Python
    servicer inherits; this path builds no frontend, so the launcher starts
    the same class (the transfer backend's, through the engine's registry).
    It must be up before the prefill ranks start: each rank registers with
    it once, and a rank that misses it is a prefill no decode can find. The
    server runs on its own thread and binds the port on every interface;
    the launcher closes it when it ends."""
    args = EngineConfigView(server_args).server_args
    if args.disaggregation_mode != "prefill":
        return None
    from tokenspeed.runtime.pd.utils import KVClassType, TransferBackend, get_kv_class

    backend = TransferBackend(args.disaggregation_transfer_backend)
    server = get_kv_class(backend, KVClassType.BOOTSTRAP_SERVER)(args.disaggregation_bootstrap_port)
    logger.info(
        "Bootstrap server started for disaggregation mode on port %s (%s transfer)",
        args.disaggregation_bootstrap_port,
        backend.value,
    )
    return server


def close_kv_bootstrap_server(server: Any) -> None:
    """Stop a bootstrap server :func:`kv_bootstrap_server` started; a
    failure to stop it is a logged line at the end of the launcher's life,
    never the exit code."""
    if server is None:
        return
    try:
        server.close()
    except Exception:  # noqa: BLE001 -- the process is ending; nothing to recover
        logger.exception("The KV bootstrap server did not stop cleanly")


def refuse_encode_role(server_args: Any) -> None:
    """The EPD encode role stays with the Python servicer: the encode worker
    takes its requests over the engine's in-process channel, which this path
    does not build, and the msgpack wire it dials carries no encode message.
    Such a launch under the Rust path is a configuration error before any
    scheduler starts, not a servicer that reports SERVING and never encodes.
    The prefill and decode roles, and the EPD prefill side, are served."""
    if EngineConfigView(server_args).server_args.disaggregation_mode == "encode":
        raise ServicerConfigError(
            f"{SERVICER_IMPL_ENV}=rust does not serve the EPD encode role "
            "(disaggregation_mode='encode'): the encode worker takes its requests over the "
            "engine's in-process channel only; run this worker on the Python servicer "
            f"(unset {SERVICER_IMPL_ENV} or pass --servicer-impl python)"
        )


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
    silence_timeout_secs_env = SILENCE_TIMEOUT_SECS_ENV
    prefill_tokens_per_sec_env = PREFILL_TOKENS_PER_SEC_ENV
    kv_events_env = KV_EVENTS_ENV
    metrics_port_env = METRICS_PORT_ENV
    metrics_host_env = METRICS_HOST_ENV
    without_tokenizer = "string stops will be refused"
    warmup_env = WARMUP_ENV
    warmup_default_budget_secs = DEFAULT_BUDGET_SECS
    # TokenSpeed's ``ModelConfig`` for the served model, built on first use
    # and kept for the hooks after it: the KV-events default reads the cache's
    # layer labels off it, the facts the rest.
    _cached_model_config: Any = None
    # The prefill side's KV bootstrap server (:func:`kv_bootstrap_server`),
    # started right before the ranks and closed when the launcher ends.
    _bootstrap_server: Any = None

    def model_config(self, args: Any) -> Any:
        if self._cached_model_config is None:
            self._cached_model_config = _model_config(args)
        return self._cached_model_config

    def kv_events_default(self, args: Any) -> str | None:
        # The config is built here rather than in the facts: the facts need
        # it a hook later in any case, so an explicit configuration or the
        # opt-out, which do not read it, cost nothing extra.
        return default_kv_events_config(args, model_config=self.model_config(args))

    def facts(self, args: Any) -> dict[str, Any]:
        return {**model_facts(args, self.model_config(args)), **server_facts(args)}

    def tokenizer_dir(self, args: Any) -> str | None:
        return tokenizer_dir_for(args)

    def launch(self, args: Any, handshake_port: int, facts: Mapping[str, Any]) -> Any:
        # Before the ranks: a prefill rank registers with the server once.
        self._bootstrap_server = kv_bootstrap_server(args)
        return launch_scheduler_ranks(headless_server_args(args, handshake_port=handshake_port))

    async def serve(self, args: Any) -> int:
        try:
            return await super().serve(args)
        finally:
            close_kv_bootstrap_server(self._bootstrap_server)
            self._bootstrap_server = None

    def warmup_inputs(self, args: Any, facts: Mapping[str, Any]) -> Any:
        return warmup_inputs(args, facts)

    def warmup_wire(self, target: str) -> Any:
        return warmup_wire(target)


async def serve_rust(server_args: Any) -> int:
    """Serve this process's gRPC contract from Rust: start the Rust server,
    start the scheduler ranks against its handshake, and supervise both
    until a signal or a failure (:class:`TokenSpeedLauncher`). Returns the
    process exit code. An EPD encode launch is refused first
    (:func:`refuse_encode_role`)."""
    refuse_encode_role(server_args)
    return await TokenSpeedLauncher().serve(server_args)
