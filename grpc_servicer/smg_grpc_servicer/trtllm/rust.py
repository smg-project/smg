"""The Rust request path of the TensorRT-LLM gRPC servicer, behind one flag.

``trtllm-serve <model> --grpc --grpc-protocol smg`` serves
``trtllm.TrtllmService`` from Python, in-process over the engine's LLM API
(``tensorrt_llm.grpc.smg``). With ``SMG_TRTLLM_SERVICER_IMPL=rust`` that
entrypoint hands the process to :func:`serve_rust_from_engine` before an LLM
or a Python gRPC server exists, and ``python -m smg_grpc_servicer.trtllm
--servicer-impl rust`` reaches the same path: the contract is served by the
Rust :class:`smg.servicer.TrtllmGrpcServer` on a Rust-owned thread, and the
engine runs behind the shim beside its LLM API (``tensorrt_llm.grpc.smg.shim``
in the engine's tree, or the same package installed beside it as
``trtllm_smg_shim``), started from this process as a spawned child that dials
the servicer's same-host msgpack ZMQ handshake. Python keeps the lifecycle
only. The Router cannot tell the two implementations apart.

What the wire does not carry, the Rust path refuses by name rather than
drops: beam search, the logits-returning flags, KV-cache retention, prompt
tuning and lookahead configs. ``Embed`` and ``SubscribeKvEvents`` answer
UNIMPLEMENTED, as the engine's Python servicer answers them.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.metadata
import importlib.util
import json
import logging
import multiprocessing
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from smg_grpc_servicer.metrics_flag import add_metrics_arguments as _add_metrics_arguments

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
    RustLauncher,
    ServicerConfigError,
    resolve_tokenizer_dir,
    servicer_switch,
)
from smg_grpc_servicer.servicer_impl_flag import (
    add_servicer_impl_argument as _add_servicer_impl_argument,
)
from smg_grpc_servicer.trtllm import METRICS_HOST_ENV, METRICS_PORT_ENV, SERVICER_IMPL_ENV

logger = logging.getLogger(__name__)

HANDSHAKE_PORT_ENV = "SMG_TRTLLM_SERVICER_HANDSHAKE_PORT"
DRAIN_SECS_ENV = "SMG_TRTLLM_SERVICER_DRAIN_SECS"
STARTUP_TIMEOUT_SECS_ENV = "SMG_TRTLLM_SERVICER_STARTUP_TIMEOUT_SECS"
STARTUP_CEILING_SECS_ENV = "SMG_TRTLLM_SERVICER_STARTUP_CEILING_SECS"
SILENCE_TIMEOUT_SECS_ENV = "SMG_TRTLLM_SERVICER_SILENCE_TIMEOUT_SECS"
PREFILL_TOKENS_PER_SEC_ENV = "SMG_TRTLLM_SERVICER_PREFILL_TOKENS_PER_SEC"
# Reserved for the KV-event relay (no publisher on this wire yet).
KV_EVENTS_ENV = "SMG_TRTLLM_SERVICER_KV_EVENTS"

# The shim's package, by import: inside the engine's tree first, then the
# same package installed beside it as an overlay. The entry point is found by
# import, never by path.
SHIM_MODULES = ("tensorrt_llm.grpc.smg.shim", "trtllm_smg_shim")


@dataclasses.dataclass
class ServeArgs:
    """What the engine's gRPC entrypoint hands this launcher (its host, port,
    the LLM API's keyword arguments and the served name), plus this package's
    own flags; the lifecycle reads ``host``/``port`` for the bind address and
    ``metrics_port``/``metrics_host`` for the metrics endpoint."""

    model: str
    host: str = "0.0.0.0"
    port: int = 0
    llm_args: dict[str, Any] = dataclasses.field(default_factory=dict)
    served_model_name: str | None = None
    servicer_impl: str | None = None
    metrics_port: int | None = None
    metrics_host: str | None = None


def add_servicer_impl_argument(parser: argparse.ArgumentParser) -> bool:
    """Add ``--servicer-impl`` to a launcher's parser that lacks it; returns
    whether it was added."""
    return _add_servicer_impl_argument(
        parser,
        help=(
            "Which implementation serves the gRPC contract: the engine's Python servicer "
            "(default) or the Rust one (smg.servicer.TrtllmGrpcServer, the engine behind "
            f"its shim). Unset falls back to ${SERVICER_IMPL_ENV}."
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
        server_class="smg.servicer.TrtllmGrpcServer",
    )


def resolve_servicer_impl(args: Any = None, environ: Mapping[str, str] | None = None) -> str:
    """Which implementation serves this process: ``args.servicer_impl`` when
    the launcher carried the flag, else ``$SMG_TRTLLM_SERVICER_IMPL``, else
    python."""
    return servicer_impl_source(args, environ)[0]


def servicer_impl_source(
    args: Any = None, environ: Mapping[str, str] | None = None
) -> tuple[str, str]:
    """:func:`resolve_servicer_impl`'s answer with where it came from
    (``flag``, ``env`` or ``default``: the shared
    :func:`~smg_grpc_servicer.rust_lifecycle.servicer_switch`)."""
    return servicer_switch(SERVICER_IMPL_ENV, args, environ)


# ---------------------------------------------------------------------------
# What the Rust server advertises, from the engine's own arguments
# ---------------------------------------------------------------------------


def checkpoint_config(model: str, revision: str | None = None) -> dict[str, Any]:
    """The checkpoint's ``config.json`` as a dict, read without the engine:
    from ``model`` when it is a local directory, else from the local cache of
    the repository id, else empty (the handshake fills the window then)."""
    directory = model if Path(model).is_dir() else resolve_tokenizer_dir(model, revision)
    if not directory:
        return {}
    path = Path(directory) / "config.json"
    try:
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError) as error:
        logger.info("checkpoint config not read (%s): %s", path, error)
        return {}
    return config if isinstance(config, dict) else {}


def _config_int(config: Mapping[str, Any], *names: str) -> int:
    """The first of ``names`` found on the config or its nested text config,
    as an int; 0 when none is."""
    sections = [config, config.get("text_config") or {}]
    for section in sections:
        for name in names:
            value = section.get(name) if isinstance(section, Mapping) else None
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return 0


def _eos_token_ids(config: Mapping[str, Any]) -> list[int]:
    for section in (config, config.get("text_config") or {}):
        value = section.get("eos_token_id") if isinstance(section, Mapping) else None
        if isinstance(value, int) and not isinstance(value, bool):
            return [value]
        if isinstance(value, list) and all(isinstance(item, int) for item in value):
            return list(value)
    return []


def _llm_int(llm_args: Mapping[str, Any], name: str, default: int = 0) -> int:
    value = llm_args.get(name)
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def model_facts(args: ServeArgs) -> dict[str, Any]:
    """``TrtllmGrpcServer`` keyword arguments for ``GetModelInfo``: what the
    engine's Python servicer reads off the LLM API's arguments and tokenizer,
    computed up front from the arguments and the checkpoint's config. A
    bound the arguments do not state stays 0: the handshake's window fills
    it once the engine is up."""
    llm_args = args.llm_args
    config = checkpoint_config(args.model, llm_args.get("revision"))
    return {
        "model_path": args.model,
        "served_model_name": str(args.served_model_name or args.model),
        "max_input_len": _llm_int(llm_args, "max_input_len"),
        "max_seq_len": _llm_int(llm_args, "max_seq_len"),
        "max_batch_size": _llm_int(llm_args, "max_batch_size"),
        "vocab_size": _config_int(config, "vocab_size"),
        "hidden_size": _config_int(config, "hidden_size"),
        "num_layers": _config_int(config, "num_hidden_layers", "num_layers"),
        "num_heads": _config_int(config, "num_attention_heads", "num_heads"),
        "eos_token_ids": _eos_token_ids(config),
    }


def _engine_version() -> str:
    try:
        return importlib.metadata.version("tensorrt_llm")
    except importlib.metadata.PackageNotFoundError:
        return ""


def server_facts(args: ServeArgs) -> dict[str, Any]:
    """``TrtllmGrpcServer`` keyword arguments for ``GetServerInfo`` and the
    engine's policies: the backend and version, the parallel sizes, the
    guided-decoding backend the arguments resolve to (``none`` when unset:
    the engine enforces no grammar then), the admission window."""
    llm_args = args.llm_args
    backend = llm_args.get("guided_decoding_backend")
    return {
        "backend": str(llm_args.get("backend") or "pytorch"),
        "trtllm_version": _engine_version(),
        "tensor_parallel_size": _llm_int(llm_args, "tensor_parallel_size", 1),
        "pipeline_parallel_size": _llm_int(llm_args, "pipeline_parallel_size", 1),
        "context_parallel_size": _llm_int(llm_args, "context_parallel_size", 1),
        # One shim per engine process; the engine's own parallelism sits
        # behind it.
        "data_parallel_size": 1,
        "guided_decoding_backend": str(backend).strip().lower() if backend else "none",
        "max_running_requests": _llm_int(llm_args, "max_batch_size"),
    }


def tokenizer_dir_for(args: ServeArgs) -> str | None:
    """The local tokenizer directory for the served model, or ``None``."""
    llm_args = args.llm_args
    return resolve_tokenizer_dir(
        str(llm_args.get("tokenizer") or args.model), llm_args.get("revision")
    )


# ---------------------------------------------------------------------------
# The shim: the engine behind the LLM API, started from this process
# ---------------------------------------------------------------------------


def _module_present(name: str) -> str | None:
    """Why ``name`` cannot be imported, or ``None`` when it can, read WITHOUT
    importing it or its parents: the engine's package loads its runtime (and
    the accelerator libraries) on import, which belongs in the shim child,
    not in this process. A module already imported counts as present; else
    the top-level package is located by its spec and the dotted path below it
    on the package's search locations."""
    if name in sys.modules:
        return None
    top, _, rest = name.partition(".")
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ValueError) as error:
        return str(error)
    if spec is None:
        return f"no module named {top!r}"
    if not rest:
        return None
    locations = [str(location) for location in spec.submodule_search_locations or []]
    if not locations:
        return f"{top!r} is not a package"
    parts = rest.split(".")
    for location in locations:
        path = Path(location).joinpath(*parts)
        if (path / "__init__.py").is_file() or path.with_suffix(".py").is_file():
            return None
    return f"{name!r} is not under {top!r} ({', '.join(locations)})"


def locate_shim(modules: tuple[str, ...] | None = None) -> str:
    """The shim's package name, the first of ``modules`` (:data:`SHIM_MODULES`
    unless given) that is importable; a :class:`ServicerConfigError` naming
    them all when none is. The check imports nothing (see
    :func:`_module_present`): the engine is imported by the shim child only."""
    modules = tuple(modules) if modules is not None else SHIM_MODULES
    failures = []
    for name in modules:
        reason = _module_present(name)
        if reason is not None:
            failures.append(f"{name} ({reason})")
            continue
        return name
    raise ServicerConfigError(
        "the Rust servicer needs the engine-side shim, found by import as one of "
        + ", ".join(modules)
        + ": "
        + "; ".join(failures)
    )


def _run_shim(
    module: str,
    handshake_address: str,
    engine_index: int,
    llm_args: dict[str, Any],
    served_model_name: str | None,
) -> None:
    """Child target: the shim's own entry, in a spawned interpreter."""
    shim = importlib.import_module(module)
    raise SystemExit(
        shim.run(
            handshake_address=handshake_address,
            engine_index=engine_index,
            llm_args=llm_args,
            served_model_name=served_model_name,
        )
    )


def launch_shim(
    args: ServeArgs, *, handshake_port: int, module: str | None = None
) -> EngineProcess:
    """Start the shim in a spawned child (a fresh interpreter that inherits no
    Rust thread and never consults the servicer switch), dialing the
    servicer's handshake; the engine loads inside it."""
    module = module or locate_shim()
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_run_shim,
        args=(
            module,
            f"tcp://127.0.0.1:{int(handshake_port)}",
            0,
            dict(args.llm_args),
            args.served_model_name,
        ),
        name="smg-trtllm-shim",
    )
    process.start()
    logger.info("Started the engine shim (%s) as pid %s", module, process.pid)
    return EngineProcess(process)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TrtllmLauncher(RustLauncher):
    """The TensorRT-LLM launcher's lifecycle: the facts come from the LLM
    API's arguments and the checkpoint's config, and the shim starts from
    this process as a spawned child."""

    engine = "TensorRT-LLM"
    started = "engine"
    counted = "engine"
    server_class = "TrtllmGrpcServer"
    ipc_prefix = "trtllm-servicer"
    handshake_port_env = HANDSHAKE_PORT_ENV
    drain_secs_env = DRAIN_SECS_ENV
    startup_timeout_secs_env = STARTUP_TIMEOUT_SECS_ENV
    startup_ceiling_secs_env = STARTUP_CEILING_SECS_ENV
    silence_timeout_secs_env = SILENCE_TIMEOUT_SECS_ENV
    prefill_tokens_per_sec_env = PREFILL_TOKENS_PER_SEC_ENV
    kv_events_env = KV_EVENTS_ENV
    metrics_port_env = METRICS_PORT_ENV
    metrics_host_env = METRICS_HOST_ENV
    # The engine keeps its own tokenizer for stop strings.
    without_tokenizer = "GetTokenizer will be unavailable"

    def prepare(self, args: Any) -> None:
        # The shim is looked up before the port is bound: a missing package
        # is a configuration error, not an engine that never dials in.
        self.shim_module = locate_shim()

    def kv_events_default(self, args: Any) -> str | None:
        # No KV-event publisher on this wire yet: nothing to turn on.
        return None

    def facts(self, args: Any) -> dict[str, Any]:
        return {**model_facts(args), **server_facts(args)}

    def tokenizer_dir(self, args: Any) -> str | None:
        return tokenizer_dir_for(args)

    def launch(self, args: Any, handshake_port: int, facts: Mapping[str, Any]) -> Any:
        return launch_shim(args, handshake_port=handshake_port, module=self.shim_module)


async def serve_rust(args: ServeArgs) -> int:
    """Serve this process's gRPC contract from Rust: start the Rust server,
    start the shim against its handshake, and supervise both until a signal
    or a failure (:class:`TrtllmLauncher`). Returns the process exit code."""
    return await TrtllmLauncher().serve(args)


def serve_rust_from_engine(
    host: str,
    port: int,
    llm_args: Mapping[str, Any],
    served_model_name: str | None = None,
) -> int:
    """What the engine's own gRPC entrypoint calls when its switch says
    ``rust``: :func:`serve_rust` on the arguments it was given, blocking
    until the process is done. Returns the exit code."""
    import asyncio

    args = ServeArgs(
        model=str(llm_args.get("model", "")),
        host=host,
        port=int(port),
        llm_args=dict(llm_args),
        served_model_name=served_model_name,
    )
    return asyncio.run(serve_rust(args))
