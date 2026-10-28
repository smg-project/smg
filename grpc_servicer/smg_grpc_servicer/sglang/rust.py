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

Prefill/decode disaggregation is served as on the Python servicer: a request's
bootstrap fields (host, port, room) ride the scheduler's own request struct
over the wire, and the prefill side's bootstrap server starts in this process
before the ranks, where the Python servicer starts it; the planned warm-up
runs on a prefill or decode leg as on a single engine, its requests carrying
the engine's stand-in bootstrap, so the leg warms up without its peer and no
first request of a prompt-length class pays a kernel compile under the
Router's prefill-liveness bound. A leg of more than one scheduler rank is
served too: a prefill leg's request runs on the rank of its room's residue
(``bootstrap_room % dp_size``, where SGLang's prefill side requires it and
the decode side looks for the KV), the Router's pin yielding to it with a
warning, and a decode leg's on its pinned rank. Multimodal inputs reach the scheduler as its
own processor output over the wire. The language side of
encode/prefill disaggregation (``--language-only``) is served as behind
SGLang's own frontend: this process runs SGLang's own encoder bootstrap server
and receiver (:mod:`.rust_encode`), the Rust server hands each request's image
references to them before the request reaches the scheduler, and the request
carries what the dispatch set. What the msgpack wire does not carry, the Rust
path reports rather than emulates: LoRA loading answers UNIMPLEMENTED (as on
the Python servicer), an encoder-only launch is refused (SGLang's entrypoint
runs that role on its own encoder server, never on a servicer), and so is a
language-only launch on the ``zmq_to_tokenizer`` transfer backend, where the
frontend itself receives and processes the embeddings (neither servicer runs
that); ``Embed``, ``FlushCache``, profiling, the KV-event relay, prompt
logprobs and hidden states are served (hidden states on this path only: the
Python servicer refuses them by name).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import multiprocessing
import os
import signal
import threading
from collections.abc import Mapping
from typing import Any

from smg_grpc_servicer.constrained_decoding import (
    grammar_backend_entry,
    grammar_start_labels,
    sglang_constrained_decoding_start,
    sglang_grammar_backend,
    structural_tag_vocabulary_entry,
)
from smg_grpc_servicer.mm_item_limits import MM_ITEM_LIMITS_KEY, sglang_item_limits
from smg_grpc_servicer.pd_pairing import pairing_protocol_from_env

# The lifecycle's defaults and the switch's choices, re-exported: this
# launcher's own documented values.
from smg_grpc_servicer.rust_lifecycle import (  # noqa: F401
    DEFAULT_DRAIN_SECS,
    DEFAULT_PREFILL_TOKENS_PER_SEC,
    DEFAULT_SILENCE_TIMEOUT_SECS,
    DEFAULT_STARTUP_CEILING_SECS,
    DEFAULT_STARTUP_TIMEOUT_SECS,
    IMPLS,
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
)
from smg_grpc_servicer.sglang import METRICS_HOST_ENV, METRICS_PORT_ENV
from smg_grpc_servicer.sglang.config_view import EngineConfigView
from smg_grpc_servicer.sglang.warmup import WARMUP_ENV, media_warmup, warmup_inputs, warmup_wire

logger = logging.getLogger(__name__)

SERVICER_IMPL_ENV = "SMG_SGLANG_SERVICER_IMPL"
HANDSHAKE_PORT_ENV = "SMG_SGLANG_SERVICER_HANDSHAKE_PORT"
DRAIN_SECS_ENV = "SMG_SGLANG_SERVICER_DRAIN_SECS"
STARTUP_TIMEOUT_SECS_ENV = "SMG_SGLANG_SERVICER_STARTUP_TIMEOUT_SECS"
STARTUP_CEILING_SECS_ENV = "SMG_SGLANG_SERVICER_STARTUP_CEILING_SECS"
SILENCE_TIMEOUT_SECS_ENV = "SMG_SGLANG_SERVICER_SILENCE_TIMEOUT_SECS"
PREFILL_TOKENS_PER_SEC_ENV = "SMG_SGLANG_SERVICER_PREFILL_TOKENS_PER_SEC"
# Set to 0/false/no/off to keep SGLang's KV event publisher off when the
# launcher was given no --kv-events-config (see `default_kv_events_config`).
KV_EVENTS_ENV = "SMG_SGLANG_SERVICER_KV_EVENTS"


def default_kv_events_json(endpoint: str) -> str:
    """What a launcher without --kv-events-config gets under the Rust servicer,
    as SGLang's server args carry it (JSON text): the ZMQ publisher at
    ``endpoint``, with SGLang's own defaults for the rest (the topic; no
    replay socket)."""
    return json.dumps({"publisher": "zmq", "endpoint": endpoint})


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
    ``flag``, ``env`` or ``default`` (the shared
    :func:`~smg_grpc_servicer.rust_lifecycle.servicer_switch`, which writes a
    flag decision back to the environment the scheduler processes inherit).
    SGLang's ``ServerArgs`` keeps only its own fields, so the flag travels as
    the value this package's plugin parsed in this process."""
    from smg_grpc_servicer.sglang import plugin

    return servicer_switch(SERVICER_IMPL_ENV, args, environ, parsed_flag=plugin.parsed_flag())


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
    preferred = EngineConfigView(server_args).server_args.preferred_sampling_params
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
    view = EngineConfigView(server_args, model_config)
    args, model = view.server_args, view.model
    defaults = dict(model_config.get_default_sampling_params() or {})
    preferred = _preferred_sampling_params(server_args)
    if preferred:
        defaults.update(preferred)
    return {
        "model_path": args.model_path,
        "tokenizer_path": str(args.tokenizer_path or args.model_path),
        "served_model_name": str(args.served_model_name or args.model_path),
        "is_generation": model.is_generation,
        "model_type": model.model_type,
        "architectures": model.architectures,
        "max_context_length": model.context_len,
        "max_req_input_len": model.context_len,
        "vocab_size": model.vocab_size,
        "eos_token_ids": model.eos_token_ids,
        "pad_token_id": model.pad_token_id,
        "bos_token_id": model.bos_token_id,
        "weight_version": args.weight_version,
        "preferred_sampling_params": (
            json.dumps(preferred, separators=(",", ":"), sort_keys=True) if preferred else ""
        ),
        "default_sampling_params_json": (
            json.dumps(defaults, separators=(",", ":"), sort_keys=True) if defaults else ""
        ),
        "supports_vision": model.is_multimodal,
        "id2label_json": json.dumps(_json_safe(model.id2label)) if model.id2label else "",
        "num_labels": model.num_labels,
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
        grammar_start_labels(server_args, sglang_constrained_decoding_start(server_args))
    )
    # The grammar backend the scheduler runs, resolved (the raw argument is
    # None unless set): the Rust servicer refuses a constraint that backend
    # cannot enforce before the scheduler.
    grammar_backend = sglang_grammar_backend(server_args)
    grammar_backend_entry(args_dict, grammar_backend)
    # The vocabulary of structural-tag nodes that backend takes, when the
    # xgrammar installed beside the engine predates the current one: the
    # Router spells such a worker's structural tags in the older vocabulary.
    structural_tag_vocabulary_entry(args_dict, grammar_backend)
    try:
        from sglang.version import __version__ as sglang_version
    except ImportError:  # the launcher's unit tests run without SGLang
        sglang_version = ""
    args = EngineConfigView(server_args).server_args
    kv_events_endpoint, kv_events_replay_endpoint, kv_events_topic = kv_events_publisher(
        server_args
    )
    return {
        # allow_nan=False: a non-finite float would otherwise become `NaN` text,
        # which the Rust side rejects as a whole object.
        "server_args_json": json.dumps(args_dict, sort_keys=True, allow_nan=False),
        "scheduler_info_json": "{}",
        "sglang_version": str(sglang_version),
        "max_running_requests": args.max_running_requests or 0,
        # The schedulers that dial, on either engine release: the engine count
        # the lifecycle waits for, and the handshake's `data_parallel_size`.
        "data_parallel_size": args.scheduler_count,
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
    raw = EngineConfigView(server_args).server_args.kv_events_config
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
    ``--kv-events-config``; returns the configuration applied, or None: the
    shared rule (``rust_lifecycle.apply_kv_events_default``) with
    :func:`default_kv_events_json` at a free port of this host, one per
    scheduler (each publishes at the port after its predecessor's), applied
    to the server args before the facts and the schedulers are built from
    them."""
    return apply_kv_events_default(
        server_args,
        default_kv_events_json,
        env_name=KV_EVENTS_ENV,
        environ=environ,
        ranks=EngineConfigView(server_args).server_args.scheduler_count,
    )


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
    server_args: Any, *, handshake_port: int, encode_bridge: Any | None = None
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
        # The encode dispatch serves the Rust server of THIS process, so it
        # starts here whichever process runs the ranks (it publishes the
        # runtime context it reads when the spawned launch left none).
        if encode_bridge is not None:
            encode_bridge.start()
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
    # A prefill side's bootstrap server, before the ranks that register with
    # it; the ranks' handle holds it for their lifetime.
    bootstrap_server = start_bootstrap_server(server_args)
    # A language side's encode dispatch (its encoder bootstrap server and
    # receiver), where SGLang's own frontend starts them: after the runtime
    # context they read, before the ranks; held the same way.
    if encode_bridge is not None:
        encode_bridge.start()
    held = [server for server in (bootstrap_server, encode_bridge) if server is not None]
    port_args = PortArgs.init_new(server_args)
    # The fault handler goes in before the spawner starts a rank: a rank that
    # fails during its start signals its parent at once.
    latch = FaultLatch()
    on_fault_signal(latch, signal.SIGQUIT, what="SGLang scheduler")
    init, processes = Engine._launch_scheduler_processes(
        server_args, port_args, run_scheduler_process
    )
    ranks = SchedulerProcesses(
        list(processes or []),
        ready=init.wait_for_ready,
        holding=held,
    )
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
    args = EngineConfigView(server_args).server_args
    return resolve_tokenizer_dir(str(args.tokenizer_path or args.model_path), args.revision)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def refuse_epd_disaggregation(server_args: Any) -> None:
    """The EPD roles this process cannot serve are a configuration error
    before the server binds. ``encoder_only`` is not a servicer role on either
    path: SGLang's entrypoint hands an encoder-only launch to its own encoder
    server before this package is reached, so a launch that still arrives
    here is misrouted. ``language_only`` is served (the encode dispatch runs
    SGLang's own receiver in this process, :mod:`.rust_encode`) on the
    transfer backends whose scheduler ranks receive the embeddings
    themselves; under ``zmq_to_tokenizer`` the frontend is the receiver and
    processes the prompt itself, which neither servicer runs, so that launch
    must not start rather than report SERVING and refuse every image.
    Prefill/decode disaggregation is served (the wire carries the bootstrap
    fields; :func:`start_bootstrap_server`) on a leg of any number of
    scheduler ranks: the Rust servicer places a prefill leg's request on the
    rank of its room (``bootstrap_room % dp_size``, where the decode side looks
    for the KV) and a decode leg's on its pinned rank. One prefill shape is
    refused whatever its load-balance method and environment: several plain
    data-parallel replicas (``--dp-size N`` without attention DP), where every
    replica reports attention-DP rank 0 to the engine's KV sender; following
    the room, the sender fails each request whose room's residue is not 0, and
    under any other method or the engine's force-query switch it registers
    rank 0, so the decode side bootstraps against replica 0 for every request
    that ran elsewhere; attention DP (``--tp-size N --attn-dp-size N``) is the
    multi-rank prefill shape."""
    args = EngineConfigView(server_args).server_args
    if args.encoder_only:
        raise ServicerConfigError(
            f"{SERVICER_IMPL_ENV}=rust does not serve an encoder-only worker: the engine's "
            "entrypoint runs that role on its own encoder server, not on a servicer"
        )
    if args.language_only:
        from smg_grpc_servicer.sglang.rust_encode import (
            SCHEDULER_RECEIVE_BACKENDS,
            resolve_encoder_transfer_backend,
        )

        backend = resolve_encoder_transfer_backend(server_args)
        if backend not in SCHEDULER_RECEIVE_BACKENDS:
            raise ServicerConfigError(
                f"{SERVICER_IMPL_ENV}=rust does not serve a language-only worker on the "
                f"{backend!r} encoder transfer backend (the frontend receives and processes "
                "the embeddings itself there); use one of "
                f"{', '.join(SCHEDULER_RECEIVE_BACKENDS)}"
            )
    if (
        args.disaggregation_mode == "prefill"
        and args.scheduler_count > 1
        and args.attention_dp_width == 1
    ):
        raise ServicerConfigError(
            f"a prefill leg of {args.scheduler_count} plain data-parallel replicas is not served: "
            "each replica reports attention-DP rank 0 to the engine's KV sender, which fails every "
            "request whose bootstrap_room % dp_size is not 0 when it follows the room, and "
            "registers rank 0 for the decode side's bootstrap under any other load-balance "
            "method; run attention DP (--tp-size N --attn-dp-size N) or one rank per leg"
        )


def start_bootstrap_server(server_args: Any) -> Any | None:
    """The prefill side's KV bootstrap server, started where the Python
    servicer starts it (``serve_grpc``, before the schedulers): SGLang's
    ``start_disagg_service`` builds the transfer backend's server on the
    launcher's ``--host`` and ``--disaggregation-bootstrap-port`` in the
    process that fills the tokenizer manager's role, so the runtime context
    must be published first. The prefill ranks register their KV senders with
    it as they start and the decode ranks' receivers query it by room.
    ``None`` for every other role (a decode side starts none); otherwise the
    server, to be held for the ranks' lifetime (it serves on a daemon
    thread)."""
    if EngineConfigView(server_args).server_args.disaggregation_mode != "prefill":
        return None
    from sglang.srt.managers.disagg_service import start_disagg_service

    bootstrap_server = start_disagg_service()
    if bootstrap_server is not None:
        logger.info(
            "Bootstrap server started for disaggregation mode on %s:%s",
            getattr(server_args, "host", None),
            getattr(server_args, "disaggregation_bootstrap_port", None),
        )
    return bootstrap_server


class SglangLauncher(RustLauncher):
    """The SGLang launcher's lifecycle: the facts come from SGLang's model
    config and server args, and the scheduler ranks start from this
    process."""

    engine = "SGLang"
    started = "scheduler"
    counted = "DP rank"
    server_class = "SglangGrpcServer"
    ipc_prefix = "sglang-servicer"
    handshake_port_env = HANDSHAKE_PORT_ENV
    drain_secs_env = DRAIN_SECS_ENV
    startup_timeout_secs_env = STARTUP_TIMEOUT_SECS_ENV
    startup_ceiling_secs_env = STARTUP_CEILING_SECS_ENV
    silence_timeout_secs_env = SILENCE_TIMEOUT_SECS_ENV
    prefill_tokens_per_sec_env = PREFILL_TOKENS_PER_SEC_ENV
    kv_events_env = KV_EVENTS_ENV
    metrics_port_env = METRICS_PORT_ENV
    metrics_host_env = METRICS_HOST_ENV
    # The scheduler keeps its own tokenizer for stop strings.
    without_tokenizer = "GetTokenizer will be unavailable"
    warmup_env = WARMUP_ENV

    def kv_events_default(self, args: Any) -> str | None:
        return default_kv_events_config(args)

    def facts(self, args: Any) -> dict[str, Any]:
        return {**model_facts(args), **server_facts(args)}

    def tokenizer_dir(self, args: Any) -> str | None:
        return tokenizer_dir_for(args)

    # The language side's encode dispatch (``--language-only``), built for the
    # server and started with the ranks; ``None`` on every other worker.
    encode: Any | None = None

    def server_extras(self, args: Any, tokenizer_dir: str | None) -> dict[str, Any]:
        """A language-side worker's encode dispatch: SGLang's own receiver and
        encoder bootstrap server behind a bridge the Rust server drives per
        request (:mod:`.rust_encode`); nothing for every other worker."""
        from smg_grpc_servicer.sglang.rust_encode import (
            EncodeDispatchBridge,
            dispatches_from_the_frontend,
        )

        processor = os.environ.get("SMG_SGLANG_MM_PROCESSOR", "").strip().lower()
        if processor not in ("", "smg"):
            raise ValueError("SMG_SGLANG_MM_PROCESSOR must be empty or smg")
        if processor:
            if dispatches_from_the_frontend(args):
                raise ValueError(
                    "native media processing cannot be combined with encode disaggregation"
                )
            if not tokenizer_dir:
                raise ValueError("native media processing requires a tokenizer directory")
            try:
                concurrency = int(os.environ.get("SMG_SGLANG_MM_MAX_INFLIGHT", "64"))
            except ValueError as error:
                raise ValueError("SMG_SGLANG_MM_MAX_INFLIGHT must be positive integer") from error
            if concurrency <= 0:
                raise ValueError("SMG_SGLANG_MM_MAX_INFLIGHT must be positive integer")
            from smg_grpc_servicer.sglang.mm_defaults import configure_threads, encoder_dtype

            dtype = encoder_dtype(
                os.environ.get("SMG_SGLANG_MM_DTYPE"),
                lambda: _model_config(args).dtype,
            )
            configure_threads()
            return {
                "smg_media_processor": {
                    "model_dir": str(args.model_path),
                    "model_id": str(args.model_path),
                    "encoder_dtype": dtype,
                    "source": "env",
                    "max_inflight": concurrency,
                    "engine_item_limits": {
                        modality: int(count)
                        for entry in sglang_item_limits(args).split(",")
                        if entry
                        for modality, count in [entry.split("=", 1)]
                    },
                }
            }
        if not dispatches_from_the_frontend(args):
            return {}
        self.encode = EncodeDispatchBridge(args)
        logger.info(
            "Encode dispatch for the language-only worker: SGLang's own receiver in this "
            "process, %d encoder URL(s) configured, bootstrap port %s",
            len(self.encode.encoder_urls),
            getattr(args, "encoder_bootstrap_port", None),
        )
        return {"encode_dispatcher": self.encode}

    def launch(self, args: Any, handshake_port: int, facts: Mapping[str, Any]) -> Any:
        if self.encode is None:
            return launch_scheduler_ranks(args, handshake_port=handshake_port)
        return launch_scheduler_ranks(
            args, handshake_port=handshake_port, encode_bridge=self.encode
        )

    # The media leg's request material, built with the inputs for a
    # language-side worker whose frontend dispatches to the encoders; ``None``
    # on every other worker and where no prompt could be rendered.
    media: Any | None = None

    def warmup_inputs(self, args: Any, facts: Mapping[str, Any]) -> Any:
        """The plan's inputs; on a language-side worker whose frontend
        dispatches to the encoders, with the media leg when the model's chat
        template renders an image placeholder (the text ladder never reaches
        the scheduler's media path, whose first use costs a fresh worker
        seconds once)."""
        from smg_grpc_servicer.sglang.rust_encode import dispatches_from_the_frontend

        inputs = warmup_inputs(args, facts)
        if inputs is None or not dispatches_from_the_frontend(args):
            return inputs
        self.media = media_warmup(
            self.tokenizer_dir(args),
            trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
        )
        if self.media is None:
            return inputs
        logger.info(
            "Planned warm-up ends its shapes with one image through the encoders (the prompt the "
            "model's chat template renders around it, %d tokens)",
            len(self.media.input_ids),
        )
        return dataclasses.replace(inputs, media_leg=True)

    def warmup_wire(self, target: str) -> Any:
        return warmup_wire(target, media=self.media)


async def serve_rust(server_args: Any) -> int:
    """Serve this process's gRPC contract from Rust: start the Rust server,
    start the scheduler ranks against its handshake, and supervise both
    until a signal or a failure (:class:`SglangLauncher`). Returns the
    process exit code."""
    refuse_epd_disaggregation(server_args)
    return await SglangLauncher().serve(server_args)
