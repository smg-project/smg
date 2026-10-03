"""The Rust request path of the vLLM gRPC servicer, behind one flag.

Upstream vLLM's gRPC server (``vllm serve <model> --grpc`` /
``python -m vllm.entrypoints.grpc_server``) imports this package's servicer
classes and hosts them on an AsyncLLM. Setting ``SMG_VLLM_SERVICER_IMPL=rust``
(or ``--servicer-impl rust`` when the launcher exposes it) keeps that same
entrypoint but hands the process to :func:`serve_rust` before an AsyncLLM or
a Python gRPC server exists: the ``vllm.grpc.engine.VllmEngine`` contract is
served by the Rust :class:`smg.servicer.VllmGrpcServer` on a Rust-owned
thread, and the engine runs headless in a spawned child, launched through
vLLM's own ``run_headless`` from the launcher's parsed namespace, dialing the
servicer's same-host ZMQ handshake. Python keeps the lifecycle only. The
Router cannot tell the two implementations apart.

Upstream integration is one check at the top of its ``serve_grpc``::

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
import asyncio
import dataclasses
import glob
import importlib.util
import json
import logging
import multiprocessing
import os
from collections.abc import Mapping, MutableMapping
from typing import Any

from smg_grpc_servicer.rust_lifecycle import (
    DEFAULT_DRAIN_SECS,
    DEFAULT_STARTUP_TIMEOUT_SECS,
    EngineProcess,
    default_socket_dir,
    free_port,
    resolve_tokenizer_dir,
    supervise,
)
from smg_grpc_servicer.rust_lifecycle import env_float as _env_float
from smg_grpc_servicer.vllm.model_info import (
    eos_token_ids_with_generation_config,
    mm_device_do_normalize,
    model_facts,
    server_facts,
)

logger = logging.getLogger(__name__)

SERVICER_IMPL_ENV = "SMG_VLLM_SERVICER_IMPL"
HANDSHAKE_PORT_ENV = "SMG_VLLM_SERVICER_HANDSHAKE_PORT"
DRAIN_SECS_ENV = "SMG_VLLM_SERVICER_DRAIN_SECS"
STARTUP_TIMEOUT_SECS_ENV = "SMG_VLLM_SERVICER_STARTUP_TIMEOUT_SECS"
IMPLS = ("python", "rust")
# What upstream's gRPC entrypoint references when it carries the switch.
HOOK_SYMBOL = "resolve_servicer_impl"


def model_info_from_config(vllm_config: Any) -> dict[str, Any]:
    """`VllmGrpcServer` keyword arguments from vLLM's own config: the facts
    both servicers advertise (`model_facts`, `server_facts`), plus the inputs
    only the Rust servicer needs up front because no Python runs on its
    request path."""
    model_config = vllm_config.model_config
    facts = {**model_facts(model_config), **server_facts(vllm_config)}
    # KV-event publishing, as the Python servicer resolves it: only vLLM's ZMQ
    # publisher can be relayed; anything else leaves SubscribeKvEvents off.
    kv_events = getattr(vllm_config, "kv_events_config", None)
    kv_events_enabled = (
        kv_events is not None
        and bool(getattr(kv_events, "enable_kv_cache_events", False))
        and getattr(kv_events, "publisher", None) == "zmq"
    )
    structured = getattr(vllm_config, "structured_outputs_config", None)
    structured_backend = getattr(structured, "backend", None) if structured is not None else None
    # The pooler config vLLM's frontend merges into a pooling request's unset
    # fields (`PoolingParams._merge_default_parameters`); absent on most
    # generation models.
    pooler = getattr(model_config, "pooler_config", None)
    pooler_use_activation = getattr(pooler, "use_activation", None)
    pooler_dimensions = getattr(pooler, "dimensions", None)
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
        "structured_outputs_backend": str(structured_backend or "auto"),
        "kv_events_endpoint": str(getattr(kv_events, "endpoint", "") or "")
        if kv_events_enabled
        else "",
        "kv_events_replay_endpoint": (
            str(getattr(kv_events, "replay_endpoint", "") or "") if kv_events_enabled else ""
        ),
        "kv_events_topic": str(getattr(kv_events, "topic", "") or "") if kv_events_enabled else "",
        "pooler_use_activation": (
            bool(pooler_use_activation) if pooler_use_activation is not None else None
        ),
        "pooler_dimensions": int(pooler_dimensions) if pooler_dimensions is not None else None,
        "mm_device_do_normalize": mm_device_do_normalize(vllm_config),
    }


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
    at launch rather than silently ignored.
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
        "max_item_bytes": settings.max_item_bytes,
        "source": settings.source,
    }


# ---------------------------------------------------------------------------
# Headless engine: upstream's own launch, from the parsed namespace
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
    """Start the headless engine in a spawned child: a fresh interpreter that
    inherits no Rust thread and never consults the servicer switch again
    (it enters ``run_headless`` directly, not ``serve_grpc``)."""
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_run_headless, args=(ns,), name="smg-headless-engine")
    process.start()
    return EngineProcess(process)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def serve_rust(args: argparse.Namespace) -> int:
    """Serve this process's gRPC contract from Rust: start the Rust server,
    launch the headless engine from the same parsed namespace, and supervise
    both. ``args`` is the upstream launcher's namespace (``--host``/``--port``
    plus ``AsyncEngineArgs``, and the frontend fields under ``vllm serve``).
    Returns the process exit code."""
    from smg.servicer import VllmGrpcServer, init_servicer_tracing
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.usage.usage_lib import UsageContext

    configure_logging()
    engine_args = AsyncEngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER)
    info = model_info_from_config(vllm_config)
    model_config = vllm_config.model_config
    tokenizer_dir = resolve_tokenizer_dir(
        str(getattr(model_config, "tokenizer", None) or model_config.model),
        getattr(model_config, "tokenizer_revision", None)
        or getattr(model_config, "revision", None),
    )
    data_parallel_size = info["data_parallel_size"]
    handshake_port = _env_int(HANDSHAKE_PORT_ENV) or free_port()
    socket_dir = default_socket_dir()
    # Worker-side media processing, from the same `--mm-*` settings (flags
    # when the launcher has them, else the environment) as the Python servicer:
    # the Python processors behind a bridge, or smg's own pipeline in Rust.
    from smg_grpc_servicer.vllm.mm_processor import MODE_SMG, MmSettings
    from smg_grpc_servicer.vllm.rust_media import RustMediaBridge

    mm_settings = MmSettings.from_args(args).resolve()
    media = None
    smg_media = None
    if mm_settings.processor == MODE_SMG:
        smg_media = smg_media_options(vllm_config, mm_settings, tokenizer_dir)
    else:
        media = RustMediaBridge.build(vllm_config, mm_settings, asyncio.get_running_loop())
    logger.info(
        "Worker-side media processing: %s (source=%s)",
        MODE_SMG if smg_media is not None else media.name if media is not None else "off",
        mm_settings.source if (smg_media is not None or media is not None) else "-",
    )

    init_servicer_tracing()
    server = VllmGrpcServer(
        # `vllm serve` leaves host unset and upstream binds all interfaces then.
        bind_address=f"{getattr(args, 'host', None) or '0.0.0.0'}:{args.port}",
        # Per process, not per requested port: `--port 0` launchers would
        # otherwise share one path and unlink each other's sockets.
        ipc_base_url=f"ipc://{socket_dir}/servicer-{os.getpid()}",
        handshake_address=f"tcp://127.0.0.1:{handshake_port}",
        engine_count=data_parallel_size,
        tokenizer_dir=tokenizer_dir,
        engine_startup_timeout_secs=_env_float(
            STARTUP_TIMEOUT_SECS_ENV, DEFAULT_STARTUP_TIMEOUT_SECS
        ),
        media_processor=media,
        smg_media_processor=smg_media,
        **info,
    )
    logger.info(
        "Rust vLLM gRPC servicer listening on %s (engine handshake tcp://127.0.0.1:%d, %d engine(s))",
        server.address,
        handshake_port,
        data_parallel_size,
    )
    try:
        engine = launch_headless_engine(
            headless_namespace(
                args, handshake_port=handshake_port, data_parallel_size=data_parallel_size
            )
        )
    except BaseException:
        # The server owns the port and a runtime thread; a launch that fails
        # must not leave them to the process exit.
        server.set_serving(False)
        await asyncio.to_thread(server.stop, 5.0)
        raise
    logger.info("Launched the headless engine (pid %s)", engine.pid)
    if media is not None:
        media.start_warmup()
    return await supervise(
        server, engine, drain_secs=_env_float(DRAIN_SECS_ENV, DEFAULT_DRAIN_SECS)
    )


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------


def resolve_servicer_impl(
    args: argparse.Namespace | None = None, environ: Mapping[str, str] | None = None
) -> str:
    """Which implementation serves this process: ``args.servicer_impl`` when the
    launcher exposes the flag, else ``$SMG_VLLM_SERVICER_IMPL``, else python.

    A decision made with the launcher's flag in hand is written back to the
    environment, so :func:`require_python_impl` (which has no flag) agrees with
    it when ``--servicer-impl python`` overrides an exported ``rust``."""
    source = os.environ if environ is None else environ
    value = getattr(args, "servicer_impl", None) if args is not None else None
    if not value:
        value = source.get(SERVICER_IMPL_ENV) or "python"
    value = str(value).strip().lower()
    if value not in IMPLS:
        raise ValueError(f"{SERVICER_IMPL_ENV} must be one of {IMPLS}, got {value!r}")
    if args is not None and isinstance(source, MutableMapping):
        source[SERVICER_IMPL_ENV] = value
    return value


def require_python_impl(environ: Mapping[str, str] | None = None) -> None:
    """Refuse to start the Python servicer when the flag asks for Rust. A
    launcher that gets this far has no hook (or ignored it); a silent fallback
    would report Rust coverage that never ran."""
    impl = resolve_servicer_impl(environ=environ)
    if impl != "python":
        raise RuntimeError(
            f"{SERVICER_IMPL_ENV}={impl}, but this process is starting the Python "
            "servicer: the installed vLLM gRPC entrypoint does not consult "
            "smg_grpc_servicer.vllm.resolve_servicer_impl. Add the hook to its "
            "serve_grpc, or unset the flag."
        )


def upstream_hook_installed(vllm_root: str | None = None) -> bool:
    """Whether the installed vLLM's gRPC entrypoint consults this package's
    flag. Scans the launcher source without importing vLLM, so a launcher
    (``smg serve``, the e2e harness) can refuse a Rust lane up front instead
    of starting workers that would run Python."""
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
                    if HOOK_SYMBOL in handle.read():
                        return True
            except OSError:
                continue
    return False


def require_upstream_hook() -> None:
    """Fail a Rust lane whose vLLM would ignore the flag."""
    if not upstream_hook_installed():
        raise RuntimeError(
            f"{SERVICER_IMPL_ENV}=rust needs a vLLM whose gRPC entrypoint consults "
            "smg_grpc_servicer.vllm.resolve_servicer_impl; this installation's does not, "
            "so its workers would silently run the Python servicer. Install a vLLM with "
            "the hook (see grpc_servicer/README.md) or select the python implementation."
        )


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


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
