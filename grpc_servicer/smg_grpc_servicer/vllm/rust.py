"""The Rust request path of the vLLM gRPC servicer, behind one flag.

Upstream vLLM's gRPC server (``vllm serve --grpc`` /
``python -m vllm.entrypoints.grpc_server``) imports this package's servicer
classes and hosts them on an AsyncLLM. Setting ``SMG_VLLM_SERVICER_IMPL=rust``
(or ``--servicer-impl rust`` when the launcher exposes it) keeps that same
entrypoint but hands the process to :func:`serve_rust` before an AsyncLLM or
a Python gRPC server exists: the engine runs headless (``vllm serve
--headless``) and dials a same-host ZMQ handshake, and the
``vllm.grpc.engine.VllmEngine`` contract is served by the Rust
:class:`smg.servicer.VllmGrpcServer` on a Rust-owned thread. Python keeps the
lifecycle only. The Router cannot tell the two implementations apart.

Upstream integration is one check at the top of its ``serve_grpc``::

    from smg_grpc_servicer.vllm import resolve_servicer_impl, serve_rust
    if resolve_servicer_impl(args) == "rust":
        raise SystemExit(await serve_rust(args))

Rust mode serves text generation. ``Embed``, ``FlushCache``, ``GetTokenizer``,
``SubscribeKvEvents`` and worker-side media processing answer UNIMPLEMENTED
there; the Python implementation stays the default.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from smg_grpc_servicer.pd_pairing import pairing_protocol_from_env

logger = logging.getLogger(__name__)

SERVICER_IMPL_ENV = "SMG_VLLM_SERVICER_IMPL"
HANDSHAKE_PORT_ENV = "SMG_VLLM_SERVICER_HANDSHAKE_PORT"
DRAIN_SECS_ENV = "SMG_VLLM_SERVICER_DRAIN_SECS"
IMPLS = ("python", "rust")
DEFAULT_DRAIN_SECS = 5.0
ENGINE_TERMINATE_SECS = 30.0
_POLL_SECS = 0.5

# Flags of the gRPC launcher itself; they never reach the engine. Each takes a value.
LAUNCHER_FLAGS = ("--host", "--port", "--servicer-impl")
# Flags the headless launch owns (a user copy would conflict with ours).
HEADLESS_OWNED_FLAGS = (
    "--model",
    "--data-parallel-size",
    "--data-parallel-size-local",
    "--data-parallel-address",
    "--data-parallel-rpc-port",
)
_BOOLEAN_FLAGS = frozenset({"--headless"})
# The `GetModelInfo.default_sampling_params_json` keys the Python servicer reports.
_SAMPLING_DEFAULT_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")


def strip_flags(argv: Sequence[str], flags: Sequence[str]) -> list[str]:
    """``argv`` without ``flags`` and their values (``--flag v`` or ``--flag=v``)."""
    drop = set(flags) | _BOOLEAN_FLAGS
    kept: list[str] = []
    skip_value = False
    for token in argv:
        if skip_value:
            skip_value = False
            continue
        name, has_value, _ = token.partition("=")
        if name in drop:
            skip_value = not has_value and name not in _BOOLEAN_FLAGS
            continue
        kept.append(token)
    return kept


def headless_engine_command(
    model: str,
    engine_argv: Sequence[str],
    *,
    handshake_port: int,
    data_parallel_size: int,
    python: str | None = None,
) -> list[str]:
    """The ``vllm serve --headless`` launch whose EngineCore(s) dial the
    servicer's ZMQ handshake: every data-parallel engine is local to this node,
    and the handshake port is the one the Rust server bound."""
    return [
        python or sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model,
        "--headless",
        "--data-parallel-size",
        str(data_parallel_size),
        "--data-parallel-size-local",
        str(data_parallel_size),
        "--data-parallel-address",
        "127.0.0.1",
        "--data-parallel-rpc-port",
        str(handshake_port),
        *engine_argv,
    ]


def _ids(value: Any) -> list[int]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, int):
        return [value] if value >= 0 else []
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, int) and not isinstance(v, bool) and v >= 0]
    return []


def eos_token_ids(model_config: Any) -> list[int]:
    """EOS ids in the order the Rust side expects: the model config's first
    (the primary id EngineCore stops on), then the generation config's extras."""
    hf_config = getattr(model_config, "hf_config", None)
    ids = _ids(getattr(hf_config, "eos_token_id", None))
    try_get = getattr(model_config, "try_get_generation_config", None)
    if callable(try_get):
        try:
            generation = try_get() or {}
        except Exception:  # a missing or malformed generation config is not fatal
            generation = {}
        for extra in _ids(generation.get("eos_token_id")):
            if extra not in ids:
                ids.append(extra)
    return ids


def default_sampling_params_json(model_config: Any) -> str:
    """The generation-config sampling defaults the Python servicer advertises."""
    get_diff = getattr(model_config, "get_diff_sampling_param", None)
    if not callable(get_diff):
        return ""
    try:
        diff = get_diff() or {}
    except Exception:
        return ""
    filtered = {
        key: diff[key] for key in _SAMPLING_DEFAULT_KEYS if key in diff and diff[key] is not None
    }
    return json.dumps(filtered, sort_keys=True) if filtered else ""


def model_info_from_config(vllm_config: Any) -> dict[str, Any]:
    """`VllmGrpcServer` keyword arguments from vLLM's own config, so
    `GetModelInfo` reports what the Python servicer would."""
    model_config = vllm_config.model_config
    hf_config = getattr(model_config, "hf_config", None)
    served = getattr(model_config, "served_model_name", None) or model_config.model
    if isinstance(served, (list, tuple)):
        served = served[0] if served else model_config.model
    return {
        "model_path": str(model_config.model),
        "served_model_name": str(served),
        "tokenizer_path": str(getattr(model_config, "tokenizer", None) or model_config.model),
        "is_generation": getattr(model_config, "runner_type", "generate") == "generate",
        "max_context_length": int(model_config.max_model_len),
        "vocab_size": int(model_config.get_vocab_size()),
        # Media processing stays with the Python servicer for now.
        "supports_vision": False,
        "model_type": str(getattr(hf_config, "model_type", None) or ""),
        "architectures": [str(a) for a in (getattr(model_config, "architectures", None) or [])],
        "eos_token_ids": eos_token_ids(model_config),
        "pad_token_id": int(getattr(hf_config, "pad_token_id", None) or 0),
        "bos_token_id": int(getattr(hf_config, "bos_token_id", None) or 0),
        "default_sampling_params_json": default_sampling_params_json(model_config),
        "data_parallel_size": int(vllm_config.parallel_config.data_parallel_size),
        "pairing_protocol": pairing_protocol_from_env(),
    }


def resolve_tokenizer_dir(tokenizer: str, revision: str | None = None) -> str | None:
    """A local directory holding ``tokenizer`` (a path, or a Hub id resolved
    through the local cache first); ``None`` when none can be found, in which
    case the Rust servicer refuses requests carrying string stops."""
    if os.path.isdir(tokenizer):
        return tokenizer
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning("huggingface_hub is not installed; cannot resolve tokenizer %r", tokenizer)
        return None
    patterns = ["*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja"]
    last_error: Exception | None = None
    for local_files_only in (True, False):
        try:
            return snapshot_download(
                tokenizer,
                revision=revision,
                allow_patterns=patterns,
                local_files_only=local_files_only,
            )
        except Exception as error:  # cache miss, offline, or an unknown repo
            last_error = error
    logger.warning("Could not resolve tokenizer %r to a local directory: %s", tokenizer, last_error)
    return None


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def default_socket_dir() -> str:
    return os.environ.get("SMG_ZMQ_SOCKET_DIR") or f"/tmp/smg-zmq-{os.getuid()}"


def terminate_engine(engine: Any, timeout: float = ENGINE_TERMINATE_SECS) -> None:
    """SIGTERM the headless engine (it tears down its own workers), then kill."""
    if engine.poll() is not None:
        return
    engine.terminate()
    try:
        engine.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("Headless engine did not exit within %.0fs; killing it", timeout)
        engine.kill()
        engine.wait()


async def supervise(
    server: Any,
    engine: Any,
    *,
    drain_secs: float = DEFAULT_DRAIN_SECS,
    stop_timeout: float = 5.0,
    stop_event: asyncio.Event | None = None,
    poll_secs: float = _POLL_SECS,
) -> int:
    """Run until a shutdown signal, an engine exit, or a server failure.

    Signals drain: health flips to NOT_SERVING at once (the Router stops
    routing here), in-flight streams get ``drain_secs`` to finish with the
    engine still up, then the server stops and the engine is terminated.
    Returns the process exit code.
    """
    loop = asyncio.get_running_loop()
    if stop_event is None:
        stop_event = asyncio.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop_event.set)
    exit_code = 0
    announced_ready = False
    try:
        while not stop_event.is_set():
            if not announced_ready and server.engine_ready:
                announced_ready = True
                logger.info("Engine connected; the servicer is SERVING")
            error = server.last_error
            if error or not server.running:
                logger.error("Rust servicer cannot serve: %s", error or "server exited")
                exit_code = 1
                break
            rc = engine.poll()
            if rc is not None:
                logger.error("Headless engine exited with code %s", rc)
                exit_code = 1
                break
            try:
                await asyncio.wait_for(stop_event.wait(), poll_secs)
            except TimeoutError:
                pass
    finally:
        try:
            server.set_serving(False)
        except Exception:
            logger.exception("Failed to mark the servicer as draining")
        if stop_event.is_set() and drain_secs > 0 and engine.poll() is None:
            logger.info("Draining for %.1fs before stopping", drain_secs)
            await asyncio.sleep(drain_secs)
        try:
            await asyncio.to_thread(server.stop, stop_timeout)
        except Exception:
            logger.exception("Failed to stop the Rust servicer cleanly")
        terminate_engine(engine)
    return exit_code


async def serve_rust(args: argparse.Namespace, engine_argv: Sequence[str] | None = None) -> int:
    """Serve this process's gRPC contract from Rust: launch the headless engine
    and the Rust server, then supervise both. ``args`` is the upstream
    launcher's namespace (``--host``/``--port`` + ``AsyncEngineArgs``);
    ``engine_argv`` defaults to this process's argv, which is what the headless
    engine re-parses. Returns the process exit code."""
    from smg.servicer import VllmGrpcServer, init_servicer_tracing
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.usage.usage_lib import UsageContext

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
    if engine_argv is None:
        engine_argv = sys.argv[1:]

    configure_logging()
    init_servicer_tracing()
    server = VllmGrpcServer(
        bind_address=f"{args.host}:{args.port}",
        ipc_base_url=f"ipc://{socket_dir}/servicer-{args.port}",
        handshake_address=f"tcp://127.0.0.1:{handshake_port}",
        engine_count=data_parallel_size,
        tokenizer_dir=tokenizer_dir,
        **info,
    )
    logger.info(
        "Rust vLLM gRPC servicer listening on %s (engine handshake tcp://127.0.0.1:%d, %d engine(s))",
        server.address,
        handshake_port,
        data_parallel_size,
    )
    command = headless_engine_command(
        args.model,
        strip_flags(engine_argv, LAUNCHER_FLAGS + HEADLESS_OWNED_FLAGS),
        handshake_port=handshake_port,
        data_parallel_size=data_parallel_size,
    )
    logger.info("Launching headless engine: %s", " ".join(command))
    engine = subprocess.Popen(command)
    return await supervise(
        server, engine, drain_secs=_env_float(DRAIN_SECS_ENV, DEFAULT_DRAIN_SECS)
    )


def resolve_servicer_impl(
    args: argparse.Namespace | None = None, environ: Mapping[str, str] | None = None
) -> str:
    """Which implementation serves this process: ``args.servicer_impl`` when the
    launcher exposes the flag, else ``$SMG_VLLM_SERVICER_IMPL``, else python."""
    value = getattr(args, "servicer_impl", None) if args is not None else None
    if not value:
        source = os.environ if environ is None else environ
        value = source.get(SERVICER_IMPL_ENV) or "python"
    value = str(value).strip().lower()
    if value not in IMPLS:
        raise ValueError(f"{SERVICER_IMPL_ENV} must be one of {IMPLS}, got {value!r}")
    return value


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


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
