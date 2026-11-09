"""The Rust request path of the TokenSpeed gRPC servicer, the launcher's default.

``python -m smg_grpc_servicer.tokenspeed`` (what ``ts serve`` spawns) served
``tokenspeed.grpc.scheduler.TokenSpeedScheduler`` from Python over TokenSpeed's
in-process ``AsyncLLM``. Unless ``--servicer-impl python`` (this package's flag
on that launcher, see :func:`add_servicer_impl_argument`;
``SMG_TOKENSPEED_SERVICER_IMPL`` is the fallback the flag overrides) selects
that path, the same entrypoint hands the process to :func:`serve_rust` before
an ``AsyncLLM`` or a Python gRPC server exists: the contract is served by the Rust
:class:`smg.servicer.TokenSpeedGrpcServer` on a Rust-owned thread, and the
scheduler rank processes are started from this same process (the way
TokenSpeed's own ``launch_scheduler_headless``, what ``ts serve --headless``
runs, starts them), dialing the servicer's same-host msgpack ZMQ handshake.
Python keeps the lifecycle only. The Router cannot tell the two
implementations apart. This path is the default; the Python servicer is
selected with ``--servicer-impl python`` / ``SMG_TOKENSPEED_SERVICER_IMPL=python``
and is deprecated (the launcher logs its deprecation when it is chosen).

What the msgpack wire does not carry, the Rust path reports rather than
emulates: ``FlushCache`` and profiling ride the wire's control call on a
scheduler whose handshake declares its control reply, and answer UNIMPLEMENTED
on one that does not; ranked ``top_logprobs`` and prompt logprobs each ride to
schedulers whose handshakes all declare that one's output columns, and are
refused otherwise.
Prefill/decode disaggregation is served: the request's bootstrap fields ride
the wire as the Python servicer forwards them, and a prefill launch starts the
engine's KV bootstrap server here, ahead of the ranks, on an engine whose PD
registry hands the server class out (:func:`kv_bootstrap_server`); on an
engine whose headless launcher hosts its own, the prefill launch is left to
that launcher (:func:`prefill_bootstrap_server_is_the_launchers`). The EPD
prefill side is served the same way,
and the EPD encode role on an engine whose handshake lists the encode
request: the Rust servicer mounts the encoder service and hands each
``Encode`` to the worker over the wire, the launcher adds the worker's own
rendezvous address to the facts (:func:`encode_bootstrap_host`), an engine
whose wire predates the encode request is refused before anything starts
(:func:`refuse_encode_role_without_wire`: its encode rank never dials the
handshake, so nothing later could name the gap), and one that dials in but
lists no encode request fails the start by name. Multimodal items
of the Qwen3.5 lineage (an MRoPE family) are served: the Rust servicer
computes the engine's MRoPE position table itself from the checkpoint facts
:func:`mrope_layout` ships (the wire bypasses the engine's input processor,
which computes it on the Python path); items of other MRoPE families are
refused by name.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import importlib
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
    structural_tag_vocabulary_entry,
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
from smg_grpc_servicer.tokenspeed import (
    METRICS_HOST_ENV,
    METRICS_HOST_FLAG,
    METRICS_PORT_ENV,
    METRICS_PORT_FLAG,
)
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
            "Which implementation serves the gRPC contract: the Rust servicer (default; "
            "smg.servicer.TokenSpeedGrpcServer, with the scheduler(s) headless) or the "
            f"deprecated Python one. Unset falls back to ${SERVICER_IMPL_ENV}, then rust."
        ),
    )


def add_metrics_arguments(parser: argparse.ArgumentParser) -> bool:
    """Add ``--servicer-metrics-port`` and ``--servicer-metrics-host`` (the Rust servicer's
    Prometheus endpoint) to a launcher's parser that lacks them; returns
    whether they were added."""
    return _add_metrics_arguments(
        parser,
        port_env=METRICS_PORT_ENV,
        host_env=METRICS_HOST_ENV,
        server_class="smg.servicer.TokenSpeedGrpcServer",
        port_flag=METRICS_PORT_FLAG,
        host_flag=METRICS_HOST_FLAG,
    )


def resolve_servicer_impl(args: Any = None, environ: Mapping[str, str] | None = None) -> str:
    """Which implementation serves this process: ``args.servicer_impl`` when
    the launcher carried the flag, else ``$SMG_TOKENSPEED_SERVICER_IMPL``, else
    rust."""
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


def _optional_int(value: Any) -> int | None:
    """``int(value)`` for a config scalar, ``None`` when absent or not one."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def mrope_layout(hf_config: Any) -> dict[str, int | None]:
    """The checkpoint facts the Rust servicer's MRoPE position rule reads
    besides the request, as the engine's own input processor reads them:
    ``vision_config.spatial_merge_size`` (patches per position along each
    axis) and the top-level ``image_token_id`` / ``video_token_id`` every
    placeholder span is checked against. Each is ``None`` when the checkpoint
    has none; the rule needs all three and refuses the item otherwise."""
    vision = _config_section(hf_config, "vision_config") if hf_config is not None else None
    return {
        "spatial_merge_size": _optional_int(
            _config_section(vision, "spatial_merge_size") if vision is not None else None
        ),
        "image_token_id": _optional_int(
            _config_section(hf_config, "image_token_id") if hf_config is not None else None
        ),
        "video_token_id": _optional_int(
            _config_section(hf_config, "video_token_id") if hf_config is not None else None
        ),
    }


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
        **mrope_layout(model.hf_config),
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
    grammar_backend = tokenspeed_grammar_backend(server_args)
    grammar_backend_entry(args_dict, grammar_backend)
    # The vocabulary of structural-tag nodes that backend takes, when the
    # xgrammar installed beside the engine predates the current one: the
    # Router spells such a worker's structural tags in the older vocabulary.
    structural_tag_vocabulary_entry(args_dict, grammar_backend)
    # The encode role's worker names its own address in every encode request
    # (the Rust servicer reads it off these args, next to the bootstrap port).
    if EngineConfigView(server_args).server_args.disaggregation_mode == "encode":
        args_dict["encode_bootstrap_host"] = encode_bootstrap_host(server_args)
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


def kv_events_name_their_group() -> bool:
    """Whether the installed engine's scheduler publishes its KV events per
    cache group: its ``KVEvent.BlockStored`` carries ``group_idx``.

    Such an engine publishes a hybrid model's History chain, its State
    checkpoints and its sliding windows each under its own group, the layout
    the relay reads from the reference engine, so the publisher serves a
    hybrid model. An engine whose events carry no group publishes one event
    per boundary, and only while every group holds it: on a hybrid model
    that is the checkpoint boundaries alone, no chain a router can follow."""
    try:
        from tokenspeed_scheduler import tokenspeed_scheduler_ext as scheduler_ext
    except ImportError:
        return False
    stored = getattr(getattr(scheduler_ext, "KVEvent", None), "BlockStored", None)
    return hasattr(stored, "group_idx")


def kv_events_unpublishable(server_args: Any, model_config: Any = None) -> str | None:
    """Why TokenSpeed's KV event publisher cannot serve the model this process
    loads, or ``None`` when it can.

    TokenSpeed keeps a hybrid model's recurrent state (its Gated DeltaNet,
    Mamba or KDA layers) in State groups of its cache: blocks that hold one
    checkpoint each and span no tokens. An engine whose KV events name their
    cache group (:func:`kv_events_name_their_group`) publishes such a model's
    History chain beside its checkpoints, and is served without reading the
    model. On an engine whose events carry no group, a boundary is published
    only while every group holds it, so a hybrid model's history chain never
    reaches the router: the publisher stays off for such a model. Which layer
    labels get a State group is the engine's own vocabulary (the recipes'
    ``STATE_LAYER_TYPES``, over :func:`cache_layer_labels`), so the answer
    follows the engine installed: one without that vocabulary has no State
    groups, and a model whose labels are all history (full, sliding or sparse
    attention) is served. ``model_config`` is TokenSpeed's ``ModelConfig``
    for the served model, built here when not given (:func:`_model_config`),
    after the vocabulary was found."""
    if kv_events_name_their_group():
        return None
    try:
        from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import STATE_LAYER_TYPES
    except ImportError as error:
        # An engine that does not spell its State-group vocabulary there (an
        # older one, with no State groups, or one that moved the module), or
        # no engine at all (the launcher's unit tests): the default is not
        # guarded, and the log says so.
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
        "groups of TokenSpeed's cache, and this engine's KV-event feed publishes a boundary "
        "only while every group holds it, so the model's history chain would never reach the "
        "router (an engine whose KV events name their cache group publishes it)"
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
    switch. Only the cases :func:`launch_scheduler_ranks` cannot serve, and a
    prefill launch whose KV bootstrap server the engine's launcher hosts
    (:func:`prefill_bootstrap_server_is_the_launchers`), go this way: the
    child is a second full Python/torch process per engine pod."""
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
    internals), for a caller off the main thread, where no handler can be
    installed, and for a prefill launch whose KV bootstrap server the
    engine's launcher hosts, which :meth:`TokenSpeedLauncher.launch` leaves
    to that launcher before this function is reached
    (:func:`prefill_bootstrap_server_is_the_launchers`).
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


def prefill_bootstrap_server_is_the_launchers(server_args: Any) -> bool:
    """Whether a prefill launch's KV bootstrap server is the engine's own
    headless launcher's to host: the role is prefill and the engine's PD
    registry hands no bootstrap server class out. Since the engine's Rust
    control plane its launcher brings the server up itself ahead of the
    ranks, and the registry lost the class at the same commit; before that
    the registry hands the class out and no launcher hosts one. The
    registry's class is the capability read, so neither kind of engine is
    mistaken for the other."""
    args = EngineConfigView(server_args).server_args
    if args.disaggregation_mode != "prefill":
        return False
    pd_utils = importlib.import_module("tokenspeed.runtime.pd.utils")
    return getattr(getattr(pd_utils, "KVClassType", None), "BOOTSTRAP_SERVER", None) is None


def kv_bootstrap_server(server_args: Any) -> Any | None:
    """Start the engine's KV bootstrap server for a prefill launch and return
    it; ``None`` for any other role, and on an engine whose own headless
    launcher hosts it (:func:`prefill_bootstrap_server_is_the_launchers`).

    Under prefill/decode disaggregation the prefill side hosts the rendezvous
    the decode side discovers its KV at (``--disaggregation-bootstrap-port``).
    Who starts it is the engine's contract, not this launcher's launch path.
    Since the engine's Rust control plane its headless launcher brings the
    server up itself, and :meth:`TokenSpeedLauncher.launch` leaves a prefill
    launch on such an engine to that launcher: one started here too would
    bind the port twice. Before that the engine's frontend started it (which
    the Python servicer inherits), a headless launch had none, and the
    registry hands the class out: this launcher starts it, whichever way the
    ranks start (from this process or through the engine's spawned launcher).
    It must be up before the prefill ranks start: each rank registers with it
    once, and a rank that misses it is a prefill no decode can find. The
    server runs on its own thread and binds the port on every interface; the
    launcher closes it when it ends."""
    args = EngineConfigView(server_args).server_args
    if args.disaggregation_mode != "prefill" or prefill_bootstrap_server_is_the_launchers(
        server_args
    ):
        return None
    pd_utils = importlib.import_module("tokenspeed.runtime.pd.utils")
    backend = pd_utils.TransferBackend(args.disaggregation_transfer_backend)
    server = pd_utils.get_kv_class(backend, pd_utils.KVClassType.BOOTSTRAP_SERVER)(
        args.disaggregation_bootstrap_port
    )
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


def refuse_encode_role_without_wire(server_args: Any) -> None:
    """Refuse an EPD encode launch on an engine whose direct msgpack wire
    predates the encode request, before anything starts.

    On such an engine the encode rank reads the in-process channel only and
    never dials the servicer's handshake, so the handshake-time refusal of an
    engine that lists no encode request is never reached: the worker would
    wait in silence past its scheduler-startup bound (the engine rank's own
    start counts as its sign of life), SERVING nothing. The capability is read
    off the installed engine's wire module: the encode role's own READY
    builder (``zmq_wire.encode_ready_response``, the registration an encode
    rank dials in with on an engine whose ranks run its Rust control plane,
    where the encode request type lives in the wire crate, not in the Python
    module), or the request-type byte of a wire whose Python side carries the
    encode request itself (``zmq_wire.REQ_TYPE_ENCODE``). The other roles
    pass; an engine that is not installed (the launcher's unit tests) is left
    to the handshake-time check, while an installed engine whose wire module
    does not import is refused by name (a broken engine would otherwise wait
    in silence too)."""
    if EngineConfigView(server_args).server_args.disaggregation_mode != "encode":
        return
    try:
        zmq_wire = importlib.import_module("tokenspeed.runtime.engine.zmq_wire")
    except ImportError as error:
        if getattr(error, "name", None) == "tokenspeed":
            return  # no engine installed (the launcher's unit tests)
        raise ServicerConfigError(
            f"The Rust servicer cannot read the EPD encode capability off this engine "
            f"build ({error}): its wire module does not import; run this worker on the "
            f"deprecated Python servicer (--servicer-impl python / {SERVICER_IMPL_ENV}=python)"
        ) from None
    registers_encode_role = getattr(zmq_wire, "encode_ready_response", None) is not None
    if not registers_encode_role and getattr(zmq_wire, "REQ_TYPE_ENCODE", None) is None:
        raise ServicerConfigError(
            "The Rust servicer does not serve the EPD encode role on this engine build "
            "(disaggregation_mode='encode'): its direct msgpack wire carries no encode request, "
            "so the encode rank would never dial the servicer's handshake and the worker would "
            "wait in silence, serving nothing; run an engine whose wire carries the encode "
            "request, or this worker on the deprecated Python servicer (--servicer-impl python / "
            f"{SERVICER_IMPL_ENV}=python)"
        )


def encode_bootstrap_host(server_args: Any) -> str:
    """The encode worker's own address, which every encode request names as
    the host the prefill side pulls the embedding from: the engine's own
    reading of it (``get_local_ip_by_remote``), as the Python encoder
    servicer fills it. Empty when the engine's helper yields no address or
    cannot be imported: the Rust servicer refuses the launch by name then,
    rather than sending the worker a rendezvous no peer can reach (the
    launch's ``--host`` is usually an unspecified address)."""
    del server_args  # the address is the engine's reading, not the launch's --host
    try:
        from tokenspeed.runtime.utils.network import get_local_ip_by_remote
    except ImportError:
        return ""
    host = get_local_ip_by_remote()
    return str(host) if host else ""


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
    metrics_port_flag = METRICS_PORT_FLAG
    metrics_host_env = METRICS_HOST_ENV
    without_tokenizer = "string stops will be refused"
    warmup_env = WARMUP_ENV
    warmup_default_budget_secs = DEFAULT_BUDGET_SECS
    # TokenSpeed's ``ModelConfig`` for the served model, built on first use
    # and kept for the hooks after it: the KV-events default reads the cache's
    # layer labels off it, the facts the rest.
    _cached_model_config: Any = None
    # The prefill side's KV bootstrap server (:func:`kv_bootstrap_server`),
    # started before the ranks and closed when the launcher ends; ``None``
    # for the other roles and on an engine whose headless launcher hosts its
    # own.
    _bootstrap_server: Any = None

    def metrics_bind_address(
        self, args: Any, environ: Mapping[str, str] | None = None
    ) -> str | None:
        # ServerArgs.metrics_* belongs to the engine. The servicer's flags
        # are exported to its environment by the argument parser.
        return super().metrics_bind_address(
            argparse.Namespace(host=args.host, port=args.port), environ=environ
        )

    def model_config(self, args: Any) -> Any:
        if self._cached_model_config is None:
            self._cached_model_config = _model_config(args)
        return self._cached_model_config

    def kv_events_default(self, args: Any) -> str | None:
        # An encode worker runs the vision tower only, with no KV cache to
        # publish from: its server args are left as given (the Python server's
        # rule). Otherwise the config is built here rather than in the facts:
        # the facts need it a hook later in any case, so an explicit
        # configuration or the opt-out, which do not read it, cost nothing
        # extra.
        if EngineConfigView(args).server_args.disaggregation_mode == "encode":
            return None
        return default_kv_events_config(args, model_config=self.model_config(args))

    def facts(self, args: Any) -> dict[str, Any]:
        return {**model_facts(args, self.model_config(args)), **server_facts(args)}

    def tokenizer_dir(self, args: Any) -> str | None:
        return tokenizer_dir_for(args)

    def launch(self, args: Any, handshake_port: int, facts: Mapping[str, Any]) -> Any:
        headless = headless_server_args(args, handshake_port=handshake_port)
        if prefill_bootstrap_server_is_the_launchers(args):
            # The engine's headless launcher hosts the prefill side's KV
            # bootstrap server, so it starts the ranks, whatever entry points
            # the engine has: ranks started from this process would have no
            # server to register with. An in-process start on such an engine
            # brings the engine's own server up first (the launcher follow-up).
            logger.info(
                "Scheduler launch left to TokenSpeed's headless launcher in a spawned process "
                "(it hosts the prefill side's KV bootstrap server)"
            )
            return launch_headless_scheduler(headless)
        # Before the ranks, whichever way they start: a prefill rank registers
        # with the server once (``None`` for the other roles).
        self._bootstrap_server = kv_bootstrap_server(args)
        return launch_scheduler_ranks(headless)

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
    process exit code. The EPD encode role runs here too: the Rust servicer
    mounts the encoder service on an engine whose handshake lists the encode
    request, and fails the start by name on one that does not; an engine whose
    wire predates the encode request is refused first
    (:func:`refuse_encode_role_without_wire`)."""
    refuse_encode_role_without_wire(server_args)
    return await TokenSpeedLauncher().serve(server_args)
