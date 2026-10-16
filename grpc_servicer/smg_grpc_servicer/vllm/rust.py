"""The Rust request path of the vLLM gRPC servicer, behind one flag.

Upstream vLLM's gRPC server (``vllm serve <model> --grpc`` /
``python -m vllm.entrypoints.grpc_server``) imports this package's servicer
classes and hosts them on an AsyncLLM. Setting ``SMG_VLLM_SERVICER_IMPL=rust``
(or ``--servicer-impl rust``, a flag this package adds to ``vllm serve``'s
parser as a vLLM general plugin, see :mod:`smg_grpc_servicer.vllm.plugin`) keeps that same
entrypoint but hands the process to :func:`serve_rust` before an AsyncLLM or
a Python gRPC server exists: the ``vllm.grpc.engine.VllmEngine`` contract is
served by the Rust :class:`smg.servicer.VllmGrpcServer` on a Rust-owned
thread, and the engine cores are started from this same process (vLLM's own
``CoreEngineProcManager``, built from the launcher's parsed namespace the way
``vllm serve --headless`` builds it), dialing the servicer's same-host ZMQ
handshake. Python keeps the lifecycle only. The Router cannot tell the two
implementations apart.

The switch runs ahead of the launcher's ``serve_grpc`` without a vLLM
change: the launcher imports this package's servicer classes at module
level, and that import installs the switch over ``serve_grpc`` (see
:mod:`smg_grpc_servicer.vllm.launcher_switch`). A launcher that carries the
check in its own source instead works the same::

    from smg_grpc_servicer.vllm import resolve_servicer_impl, serve_rust
    if resolve_servicer_impl(args) == "rust":
        raise SystemExit(await serve_rust(args))

Rust mode serves the whole contract: text generation, PD disaggregation
(connector KV-transfer params pass through both ways and ``GetServerInfo``
carries the pairing identity), Router-preprocessed media, worker-side media
processing (``media_refs``, through the same ``--mm-processor`` backends as
the Python servicer, see :mod:`smg_grpc_servicer.vllm.rust_media`, or through
smg's own pipeline with ``--mm-processor smg``, see :func:`smg_media_options`),
``Embed``, ``FlushCache``, ``GetTokenizer`` and ``SubscribeKvEvents``. The
Python implementation stays the default.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import dataclasses
import glob
import importlib.util
import json
import logging
import multiprocessing
import os
from collections.abc import Mapping
from typing import Any

# The lifecycle's defaults and the switch's choices, re-exported: this
# launcher's own documented values.
from smg_grpc_servicer.rust_lifecycle import (  # noqa: F401
    DEFAULT_DRAIN_SECS,
    DEFAULT_STARTUP_CEILING_SECS,
    DEFAULT_STARTUP_TIMEOUT_SECS,
    IMPLS,
    EngineProcess,
    EngineProcessGroup,
    RustLauncher,
    ServicerConfigError,
    apply_kv_events_default,
    resolve_tokenizer_dir,
    servicer_switch,
)
from smg_grpc_servicer.vllm.config_view import EngineConfigView
from smg_grpc_servicer.vllm.media_refs import FETCHABLE_MODALITIES
from smg_grpc_servicer.vllm.model_info import (
    constrained_decoding_start,
    engine_reasoning_parser,
    eos_token_ids_with_generation_config,
    mm_device_do_normalize,
    mm_item_limits,
    model_facts,
    server_facts,
)

logger = logging.getLogger(__name__)

SERVICER_IMPL_ENV = "SMG_VLLM_SERVICER_IMPL"
HANDSHAKE_PORT_ENV = "SMG_VLLM_SERVICER_HANDSHAKE_PORT"
DRAIN_SECS_ENV = "SMG_VLLM_SERVICER_DRAIN_SECS"
STARTUP_TIMEOUT_SECS_ENV = "SMG_VLLM_SERVICER_STARTUP_TIMEOUT_SECS"
STARTUP_CEILING_SECS_ENV = "SMG_VLLM_SERVICER_STARTUP_CEILING_SECS"
# Set to 0/false/no/off to keep vLLM's KV event publisher off when the
# launcher was given no --kv-events-config (see `default_kv_events_config`).
KV_EVENTS_ENV = "SMG_VLLM_SERVICER_KV_EVENTS"
# vLLM's own knob for how it starts its processes; the engine cores launched
# from this process are spawned unless the deployment chose otherwise.
MULTIPROC_METHOD_ENV = "VLLM_WORKER_MULTIPROC_METHOD"
# What upstream's gRPC entrypoint references when it carries the switch.
HOOK_SYMBOL = "resolve_servicer_impl"


def model_info_from_config(vllm_config: Any, *, log_stats: bool | None = None) -> dict[str, Any]:
    """`VllmGrpcServer` keyword arguments from vLLM's own config: the facts
    both servicers advertise (`model_facts`, `server_facts`), plus the inputs
    only the Rust servicer needs up front because no Python runs on its
    request path. ``log_stats`` is whether the engine cores are started with
    their scheduler statistics on (``--disable-log-stats`` not given): such
    an engine's wire reports every scheduler step, which the servicer's
    pushed load records say from the first record; ``None`` leaves the
    servicer to classify the wire by what it shows."""
    view = EngineConfigView(vllm_config)
    model_config = vllm_config.model_config
    facts = {**model_facts(model_config), **server_facts(vllm_config)}
    # `server_facts` carries the multimodal encoder dtype for the Python
    # servicer's GetServerInfo; the Rust servicer derives it from `model_dtype`
    # itself, by the same rule, and its binding takes no such keyword.
    facts.pop("multimodal_encoder_dtype", None)
    # KV-event publishing, as the Python servicer resolves it: only vLLM's ZMQ
    # publisher can be relayed; anything else leaves SubscribeKvEvents off.
    events = view.kv_events
    return {
        **facts,
        # The Rust servicer stops on the full set vLLM's frontend assembles
        # (the generation config's extras included) and advertises that set.
        "eos_token_ids": eos_token_ids_with_generation_config(model_config),
        # The pairing fields are absent when the config does not carry them;
        # the binding takes every keyword.
        "kv_cache_dtype": str(facts.get("kv_cache_dtype", "")),
        "attention_backend": str(facts.get("attention_backend", "")),
        "model_dtype": str(facts.get("model_dtype", "")),
        "block_size": int(facts.get("block_size", 0) or 0),
        "structured_outputs_backend": str(view.structured_outputs.backend or "auto"),
        "kv_events_endpoint": events.endpoint if events.relayed else "",
        "kv_events_replay_endpoint": events.replay_endpoint if events.relayed else "",
        "kv_events_topic": events.topic if events.relayed else "",
        # The pooler config vLLM's frontend merges into a pooling request's
        # unset fields (`PoolingParams._merge_default_parameters`); absent on
        # most generation models.
        "pooler_use_activation": view.pooler.use_activation,
        "pooler_dimensions": view.pooler.dimensions,
        "mm_device_do_normalize": mm_device_do_normalize(vllm_config),
        "mm_item_limits": mm_item_limits(vllm_config),
        "engine_reasoning_parser": engine_reasoning_parser(vllm_config),
        "constrained_decoding_start": constrained_decoding_start(vllm_config),
        "log_stats": log_stats,
    }


def default_kv_events_config(
    args: argparse.Namespace, environ: Mapping[str, str] | None = None
) -> Any | None:
    """Turn vLLM's KV event publisher on when the launcher was given no
    ``--kv-events-config``; returns the configuration applied, or None.

    The shared rule (``rust_lifecycle.apply_kv_events_default``) with vLLM's
    default, applied to the namespace before the engine args and the engine
    cores are built from it: the typed ``KVEventsConfig`` once ``vllm.config``
    is importable, else the JSON form the launcher's parser takes; vLLM's own
    defaults for the rest (the ZMQ endpoint and its port, the topic).
    """
    return apply_kv_events_default(
        args, _zmq_publisher_config, env_name=KV_EVENTS_ENV, environ=environ
    )


def _zmq_publisher_config() -> Any:
    """vLLM's configuration of its ZMQ KV event publisher: the typed form once
    ``vllm.config`` is importable, else as the launcher's parser takes it."""
    try:
        from vllm.config import KVEventsConfig

        return KVEventsConfig(enable_kv_cache_events=True, publisher="zmq")
    except ImportError:  # the launcher's parser hands the config over as JSON too
        return {"enable_kv_cache_events": True, "publisher": "zmq"}


def smg_media_options(vllm_config, settings, tokenizer_dir: str | None) -> dict[str, Any] | None:
    """What the binding builds smg's own media pipeline from (``--mm-processor
    smg``): the model's config directory and id, the pixel format the engine
    takes, its dtype, and the ``--mm-*`` caps. ``None`` when the served model
    takes no media, in which case the mode is ignored as the Python processors
    ignore it.

    An engine that normalizes pixels on device (vLLM's ``mm_device_do_normalize``,
    the default where the model supports it) takes raw ``uint8`` pixels and
    would normalize anything else twice; the pipeline writes raw pixels then.
    The engine's ``mm_processor_kwargs`` go along as overrides of the
    preprocessor config (less ``device``, which only says where vLLM's own
    processor would run); a knob the pipeline has no field for is refused
    at launch rather than silently ignored. The engine's own per-prompt media
    limits (``--limit-mm-per-prompt``) go along too: the pipeline refuses
    above them as the engine's own server does, whatever its own caps say. So
    does the engine's video frame budget (``--media-io-kwargs``
    ``video.num_frames``, else its loader's default), so a clip costs the same
    tokens as on the engine's own server for the families whose processor
    takes the loader's frames (unless ``SMG_VLLM_MM_MAX_VIDEO_FRAMES`` caps the
    budget below the engine's count, when the pipeline samples fewer frames
    than the engine's server would, as the cap intends); a sampling rule of
    the loader the pipeline cannot follow goes along by name
    (``video_loader_rule``), and the pipeline refuses it for those families
    only.
    """
    model_config = vllm_config.model_config
    if not getattr(model_config, "is_multimodal_model", False):
        logger.warning(
            "mm_processor=%s ignored: the served model is not multimodal", settings.processor
        )
        return None
    model_path = str(model_config.model)
    model_dir = (
        model_path
        if os.path.isfile(os.path.join(model_path, "config.json"))
        else tokenizer_dir or model_path
    )
    dtype = str(getattr(model_config, "dtype", "") or "").removeprefix("torch.")
    processor_kwargs = {
        key: value
        for key, value in (getattr(model_config, "mm_processor_kwargs", None) or {}).items()
        if key != "device"
    }
    return {
        "model_dir": model_dir,
        "model_id": model_path,
        "raw_pixels": mm_device_do_normalize(vllm_config),
        "encoder_dtype": dtype or "float32",
        "processor_kwargs_json": (
            json.dumps(processor_kwargs, sort_keys=True) if processor_kwargs else None
        ),
        "max_inflight": settings.max_inflight,
        "max_items": settings.max_items,
        "engine_item_limits": engine_item_limits(model_config),
        "max_item_bytes": settings.max_item_bytes,
        "video_frame_budget": video_frame_budget(model_config),
        "video_loader_rule": video_loader_rule(model_config),
        "source": settings.source,
    }


def engine_item_limits(model_config) -> dict[str, int] | None:
    """The engine's per-prompt media limits for the modalities the pipeline
    fetches, as vLLM resolved ``--limit-mm-per-prompt`` (its own server refuses
    a prompt above them); ``None`` on a config that does not carry them, which
    leaves the pipeline to its own caps."""
    mm_config = getattr(model_config, "multimodal_config", None)
    get_limit = getattr(mm_config, "get_limit_per_prompt", None)
    if not callable(get_limit):
        return None
    return {modality: int(get_limit(modality)) for modality in FETCHABLE_MODALITIES}


def video_frame_budget(model_config, environ: Mapping[str, str] | None = None) -> int | None:
    """The number of frames the engine's own loader samples a video to: its
    ``--media-io-kwargs`` ``video.num_frames`` when set (``0`` for a non-positive
    count, which vLLM reads as every frame), else the loader's default, under
    the same ``SMG_VLLM_MM_MAX_VIDEO_FRAMES`` cap the other processors apply
    (``clamp_video_frames``); ``None`` when neither is known, which leaves the
    pipeline's spec to its own constant. A sampling rule of the loader that the
    pipeline cannot follow is reported by ``video_loader_rule``, not here."""
    from smg_grpc_servicer.vllm.mm_processor import (
        DEFAULT_MAX_VIDEO_FRAMES,
        ENV_MAX_VIDEO_FRAMES,
        clamp_video_frames,
        env_int,
        vllm_default_video_frames,
    )

    source = os.environ if environ is None else environ
    mm_config = getattr(model_config, "multimodal_config", None)
    kwargs = getattr(mm_config, "media_io_kwargs", None) or {}
    if not isinstance(kwargs, Mapping):
        kwargs = {}
    default = vllm_default_video_frames()
    max_frames = env_int(source, ENV_MAX_VIDEO_FRAMES, DEFAULT_MAX_VIDEO_FRAMES, minimum=0)
    capped = clamp_video_frames(kwargs, max_frames, default) or {}
    video = capped.get("video") if isinstance(capped, Mapping) else None
    num_frames = video.get("num_frames") if isinstance(video, Mapping) else None
    if num_frames is None:
        num_frames = default
    if num_frames is None:
        return None
    return max(int(num_frames), 0)


def video_loader_rule(model_config, environ: Mapping[str, str] | None = None) -> str | None:
    """A sampling rule of the engine's video loader that the smg pipeline
    cannot follow, named the way the launcher was given it, or ``None``: a
    loader other than the default ``opencv`` (``video.video_backend`` in
    ``--media-io-kwargs``, else ``VLLM_VIDEO_LOADER_BACKEND``, in vLLM's own
    order of precedence), or a ``video.fps``
    above zero, which thins the frames by the clip's duration (vLLM's default
    ``-1`` and ``0`` do not). The pipeline refuses the rule at launch for a
    family that samples the way the loader does (Gemma 4); the families with a
    rate-based sampler of their own never followed the loader and start as
    before, so a fleet-wide loader setting does not block them."""
    source = os.environ if environ is None else environ
    mm_config = getattr(model_config, "multimodal_config", None)
    kwargs = getattr(mm_config, "media_io_kwargs", None) or {}
    video = kwargs.get("video") if isinstance(kwargs, Mapping) else None
    video = video if isinstance(video, Mapping) else {}
    rules: list[str] = []
    # vLLM takes the kwarg over the environment (`kwargs.pop("video_backend")
    # or envs.VLLM_VIDEO_LOADER_BACKEND`), so only the loader it ends up with
    # counts: a kwarg naming opencv silences a fleet-wide environment setting.
    backend = video.get("video_backend")
    loader = (source.get("VLLM_VIDEO_LOADER_BACKEND") or "").strip()
    if backend is not None:
        if str(backend).strip().lower() != "opencv":
            rules.append(f"--media-io-kwargs video.video_backend={backend}")
    elif loader and loader.lower() != "opencv":
        rules.append(f"VLLM_VIDEO_LOADER_BACKEND={loader}")
    fps = video.get("fps")
    if fps is not None:
        try:
            thins = float(fps) > 0
        except (TypeError, ValueError):
            thins = True
        if thins:
            rules.append(f"--media-io-kwargs video.fps={fps}")
    return "; ".join(rules) or None


# ---------------------------------------------------------------------------
# Headless engine: upstream's own engine-core launch, from the parsed namespace
# ---------------------------------------------------------------------------


def _fill_frontend_defaults(ns: argparse.Namespace) -> None:
    """``run_headless`` reads a few frontend fields (``reasoning_parser_plugin``,
    ...) that only the ``vllm serve`` parser defines; a bare
    ``python -m vllm.entrypoints.grpc_server`` namespace gets their defaults."""
    try:
        from vllm.entrypoints.launchers.cli_args import FrontendArgs
    except ImportError:
        try:  # older layout
            from vllm.entrypoints.openai.cli_args import FrontendArgs
        except ImportError:
            return
    if not dataclasses.is_dataclass(FrontendArgs):
        return
    for field in dataclasses.fields(FrontendArgs):
        if hasattr(ns, field.name):
            continue
        if field.default is not dataclasses.MISSING:
            setattr(ns, field.name, field.default)
        elif field.default_factory is not dataclasses.MISSING:
            setattr(ns, field.name, field.default_factory())
        else:
            setattr(ns, field.name, None)


def headless_namespace(
    args: argparse.Namespace, *, handshake_port: int, data_parallel_size: int
) -> argparse.Namespace:
    """The launcher's parsed namespace re-aimed at ``vllm serve --headless``.

    Every engine argument stays exactly as parsed (there is no argv round
    trip, so the ``vllm serve <model> --grpc`` and ``python -m`` entrypoints
    both work); the data-parallel group is pinned to this host and to the
    handshake port the Rust server bound, with every engine local."""
    ns = argparse.Namespace(**vars(args))
    _fill_frontend_defaults(ns)
    ns.model_tag = None  # `args.model` already carries the positional
    ns.grpc = False
    ns.headless = True
    ns.api_server_count = 0
    ns.data_parallel_size = data_parallel_size
    ns.data_parallel_size_local = data_parallel_size
    ns.data_parallel_address = "127.0.0.1"
    ns.data_parallel_rpc_port = handshake_port
    ns.data_parallel_start_rank = None
    ns.data_parallel_hybrid_lb = False
    ns.data_parallel_external_lb = False
    return ns


def _run_headless(ns: argparse.Namespace) -> None:
    """Child target: vLLM's own headless launch (`vllm serve --headless`)."""
    from vllm.entrypoints.cli.serve import run_headless

    run_headless(ns)


def launch_headless_engine(ns: argparse.Namespace) -> EngineProcess:
    """Start vLLM's headless launcher in a spawned child: a fresh interpreter
    that inherits no Rust thread and never consults the servicer switch again
    (it enters ``run_headless`` directly, not ``serve_grpc``). Only the cases
    the launcher alone serves go this way (see :func:`launch_engine_cores`):
    the child is a second full Python/torch process per engine pod."""
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_run_headless, args=(ns,), name="smg-headless-engine")
    process.start()
    return EngineProcess(process)


def launch_engine_cores(ns: argparse.Namespace) -> EngineProcessGroup | EngineProcess:
    """Start the engine cores from this process, the way vLLM's headless
    launcher (``run_headless``) starts them on the head node: the headless
    engine config from the same namespace, then ``CoreEngineProcManager``
    with every engine local and the servicer's handshake address. This
    process has already imported vLLM and torch for the servicer's own
    config, so hosting the launch here costs no second interpreter, where
    the spawned launcher was a full Python/torch process per engine pod,
    idle once the cores ran.

    The cores are started by vLLM's own process manager, spawned rather than
    forked: this process hosts the Rust runtime's threads, which no child may
    inherit (the Python servicer's process ends up spawning them too, vLLM
    forces it there once CUDA is initialized). The launcher keeps the cases
    only it serves: the worker ranks of a multi-node engine
    (``node_rank_within_dp > 0``) and the ray data-parallel backend go through
    :func:`launch_headless_engine` unchanged.
    """
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.usage.usage_lib import UsageContext

    engine_args = AsyncEngineArgs.from_cli_args(ns)
    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER, headless=True
    )
    view = EngineConfigView(vllm_config)
    parallel = view.parallel
    if parallel.node_rank_within_dp > 0 or parallel.data_parallel_backend == "ray":
        logger.info(
            "Engine launch left to vLLM's headless launcher in a spawned process "
            "(node rank %d within the data-parallel group, backend %s)",
            parallel.node_rank_within_dp,
            parallel.data_parallel_backend,
        )
        return launch_headless_engine(ns)

    from vllm.utils.network_utils import get_tcp_uri
    from vllm.v1.engine.utils import CoreEngineProcManager
    from vllm.v1.executor import Executor

    # vLLM's default start method is fork; an explicit choice is kept.
    if os.environ.setdefault(MULTIPROC_METHOD_ENV, "spawn") != "spawn":
        logger.warning(
            "%s=%s: the engine cores are forked from the servicer process",
            MULTIPROC_METHOD_ENV,
            os.environ[MULTIPROC_METHOD_ENV],
        )
    launch = view.launch
    manager = CoreEngineProcManager(
        local_engine_count=launch.local_engine_count,
        start_index=launch.start_index,
        local_start_index=0,
        vllm_config=vllm_config,
        local_client=False,
        handshake_address=get_tcp_uri(launch.master_ip, launch.rpc_port),
        executor_class=Executor.get_class(vllm_config),
        log_stats=not engine_args.disable_log_stats,
    )
    # The engine's own request-drain budget bounds the shutdown when one is
    # configured; otherwise the manager's best-effort grace applies.
    engines = EngineProcessGroup(manager, shutdown_timeout=launch.shutdown_timeout or None)
    logger.info(
        "Started %d engine core process(es) from the servicer process: pids %s",
        len(engines.pids),
        ", ".join(str(pid) for pid in engines.pids),
    )
    return engines


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class VllmLauncher(RustLauncher):
    """The vLLM launcher's lifecycle: the engine config, built once from the
    launcher's parsed namespace, serves the facts, the tokenizer lookup and
    the worker-side media options; the engine cores start from this
    process."""

    engine = "vLLM"
    started = "engine"
    counted = "engine"
    server_class = "VllmGrpcServer"
    ipc_prefix = "servicer"
    handshake_port_env = HANDSHAKE_PORT_ENV
    drain_secs_env = DRAIN_SECS_ENV
    startup_timeout_secs_env = STARTUP_TIMEOUT_SECS_ENV
    startup_ceiling_secs_env = STARTUP_CEILING_SECS_ENV
    kv_events_env = KV_EVENTS_ENV
    without_tokenizer = "string stops will be refused"

    def __init__(self) -> None:
        self.vllm_config: Any = None
        self.media: Any = None

    def prepare(self, args: argparse.Namespace) -> None:
        configure_logging()

    def kv_events_default(self, args: argparse.Namespace) -> Any | None:
        return default_kv_events_config(args)

    def facts(self, args: argparse.Namespace) -> dict[str, Any]:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.usage.usage_lib import UsageContext

        engine_args = AsyncEngineArgs.from_cli_args(args)
        self.vllm_config = engine_args.create_engine_config(
            usage_context=UsageContext.OPENAI_API_SERVER
        )
        # The engine cores are started with `log_stats=not disable_log_stats`
        # (`launch_engine_cores`); the servicer is told the same.
        disable_log_stats = getattr(engine_args, "disable_log_stats", None)
        return model_info_from_config(
            self.vllm_config,
            log_stats=None if disable_log_stats is None else not disable_log_stats,
        )

    def tokenizer_dir(self, args: argparse.Namespace) -> str | None:
        model = EngineConfigView(self.vllm_config).model
        return resolve_tokenizer_dir(model.tokenizer, model.tokenizer_revision)

    def server_extras(self, args: argparse.Namespace, tokenizer_dir: str | None) -> dict[str, Any]:
        """Worker-side media processing, from the same ``--mm-*`` settings
        (flags when the launcher has them, else the environment) as the
        Python servicer: the Python processors behind a bridge, or smg's own
        pipeline in Rust."""
        from smg_grpc_servicer.vllm.mm_processor import MODE_SMG, MmSettings
        from smg_grpc_servicer.vllm.rust_media import RustMediaBridge

        settings = MmSettings.from_args(args).resolve()
        smg_media = None
        if settings.processor == MODE_SMG:
            smg_media = smg_media_options(self.vllm_config, settings, tokenizer_dir)
        else:
            self.media = RustMediaBridge.build(
                self.vllm_config, settings, asyncio.get_running_loop()
            )
        logger.info(
            "Worker-side media processing: %s (source=%s)",
            MODE_SMG
            if smg_media is not None
            else self.media.name
            if self.media is not None
            else "off",
            settings.source if (smg_media is not None or self.media is not None) else "-",
        )
        return {"media_processor": self.media, "smg_media_processor": smg_media}

    def launch(
        self, args: argparse.Namespace, handshake_port: int, facts: Mapping[str, Any]
    ) -> Any:
        return launch_engine_cores(
            headless_namespace(
                args, handshake_port=handshake_port, data_parallel_size=facts["data_parallel_size"]
            )
        )

    def launched(self, engine: Any) -> None:
        logger.info("Launched the headless engine (pid %s)", engine.pid)
        if self.media is not None:
            self.media.start_warmup()


async def serve_rust(args: argparse.Namespace) -> int:
    """Serve this process's gRPC contract from Rust: start the Rust server,
    launch the engine cores from the same parsed namespace, and supervise
    both (:class:`VllmLauncher`). ``args`` is the upstream launcher's
    namespace (``--host``/``--port`` plus ``AsyncEngineArgs``, and the
    frontend fields under ``vllm serve``). Returns the process exit code."""
    return await VllmLauncher().serve(args)


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------


def resolve_servicer_impl(
    args: argparse.Namespace | None = None, environ: Mapping[str, str] | None = None
) -> str:
    """Which implementation serves this process: ``args.servicer_impl`` when the
    launcher exposes the flag, else ``$SMG_VLLM_SERVICER_IMPL``, else python
    (the shared :func:`~smg_grpc_servicer.rust_lifecycle.servicer_switch`). A
    decision made with the flag in hand is written back to the environment,
    so :func:`require_python_impl` (which has no flag) agrees with it when
    ``--servicer-impl python`` overrides an exported ``rust``."""
    return servicer_switch(SERVICER_IMPL_ENV, args, environ)[0]


def require_python_impl(environ: Mapping[str, str] | None = None) -> None:
    """Refuse to start the Python servicer when the flag asks for Rust. A
    launcher that gets this far ran without the switch; a silent fallback
    would report Rust coverage that never ran."""
    impl = resolve_servicer_impl(environ=environ)
    if impl != "python":
        raise RuntimeError(
            f"{SERVICER_IMPL_ENV}={impl}, but this process is starting the Python "
            "servicer: the switch this package installs over vLLM's gRPC launcher "
            "never ran (a launcher executed directly as __main__, or one that did "
            "not import smg_grpc_servicer.vllm.servicer before defining serve_grpc). "
            "Start through `vllm serve <model> --grpc`, or unset the flag; a silent "
            "Python fallback would report Rust coverage that never ran."
        )


def require_python_media_mode(
    args: argparse.Namespace | None = None, environ: Mapping[str, str] | None = None
) -> str:
    """The ``--mm-processor`` mode the Python servicer will serve, checked
    before the launcher builds the engine: ``smg`` is the Rust servicer's own
    pipeline, and the Python servicer's own refusal of it comes only once the
    engine exists, after the whole model loaded (under a restart policy a
    crash loop that pays a full load per turn). The mode is known from the
    flags and the environment here, so the switch refuses it now, as a
    :class:`~smg_grpc_servicer.rust_lifecycle.ServicerConfigError` with the
    servicer's own message."""
    from smg_grpc_servicer.vllm.mm_processor import MmSettings, require_python_servicer_mode

    settings = MmSettings.from_args(args) if args is not None else MmSettings()
    try:
        return require_python_servicer_mode(settings, os.environ if environ is None else environ)
    except ValueError as error:
        raise ServicerConfigError(str(error)) from None


def upstream_hook_installed(vllm_root: str | None = None) -> bool:
    """Whether the installed vLLM's gRPC entrypoint will consult this
    package's flag: it carries the check in its own source, or it imports
    this package's servicer modules at module level, where the switch
    installs itself (:mod:`smg_grpc_servicer.vllm.launcher_switch`). Scans
    the launcher source without importing vLLM, so a launcher (``smg serve``,
    the e2e harness) can refuse a Rust lane up front instead of starting
    workers that would run Python."""
    roots: list[str] = []
    if vllm_root is not None:
        roots = [vllm_root]
    else:
        try:
            spec = importlib.util.find_spec("vllm")
        except (ImportError, ValueError):
            spec = None
        if spec is not None:
            roots = list(spec.submodule_search_locations or [])
    for root in roots:
        launcher = os.path.join(root, "entrypoints", "launchers", "grpc_server.py")
        if os.path.isfile(launcher):
            candidates = [launcher]
        else:  # the launcher moved; scan the entrypoints tree
            candidates = glob.glob(os.path.join(root, "entrypoints", "**", "*.py"), recursive=True)
        for path in candidates:
            try:
                with open(path, encoding="utf-8") as handle:
                    source = handle.read()
            except OSError:
                continue
            if HOOK_SYMBOL in source or _imports_servicer_at_module_level(source):
                return True
    return False


_SWITCH_MODULES = (
    "smg_grpc_servicer.vllm.servicer",
    "smg_grpc_servicer.vllm.health_servicer",
)


_SWITCH_NAMES = ("VllmEngineServicer", "VllmHealthServicer")


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _imports_servicer_at_module_level(source: str) -> bool:
    """Whether a launcher imports a switch-installing module of this package
    at module level: directly, inside a top-level ``try``/``if``/``with``, or
    through the package's own lazy names (``from smg_grpc_servicer.vllm import
    VllmEngineServicer``, which imports the module). An ``if TYPE_CHECKING:``
    body never runs, and an import inside ``serve_grpc`` would run too late
    (the switch would bind over the next call, not the running one); neither
    counts."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    statements = list(tree.body)
    while statements:
        node = statements.pop()
        if isinstance(node, ast.ImportFrom):
            if node.module in _SWITCH_MODULES:
                return True
            if node.module == "smg_grpc_servicer.vllm" and any(
                alias.name in _SWITCH_NAMES for alias in node.names
            ):
                return True
        if isinstance(node, ast.Import) and any(
            alias.name in _SWITCH_MODULES for alias in node.names
        ):
            return True
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            statements.extend(node.orelse)
            continue
        if isinstance(node, (ast.Try, ast.If, ast.With)):
            for field in ("body", "orelse", "finalbody", "handlers"):
                for child in getattr(node, field, []) or []:
                    statements.extend(
                        getattr(child, "body", [child])
                        if isinstance(child, ast.ExceptHandler)
                        else [child]
                    )
    return False


def require_upstream_hook() -> None:
    """Fail a Rust lane whose vLLM would ignore the flag."""
    if not upstream_hook_installed():
        raise RuntimeError(
            f"{SERVICER_IMPL_ENV}=rust needs a vLLM whose gRPC launcher consults this "
            "package's flag: one that imports smg_grpc_servicer.vllm.servicer at module "
            "level (where the switch installs itself) or carries the check in its own "
            "serve_grpc. This installation's launcher does neither, so its workers would "
            "run the Python servicer; see grpc_servicer/README.md or select python."
        )


def configure_logging() -> None:
    """Route this package's logs through vLLM's handlers when vLLM configured
    them, else a plain stderr handler: the Rust path's lifecycle messages
    (listener, engine launch, drain) must not vanish into an unconfigured
    logger."""
    from smg_grpc_servicer.vllm import attach_vllm_logging

    attach_vllm_logging()
    package_logger = logging.getLogger("smg_grpc_servicer")
    if not package_logger.handlers and not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        )
