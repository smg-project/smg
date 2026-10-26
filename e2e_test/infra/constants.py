"""Constants and enums for E2E test infrastructure."""

import os
import shutil
import socket
from collections.abc import Callable
from enum import StrEnum


class ConnectionMode(StrEnum):
    """Worker connection protocol."""

    HTTP = "http"
    GRPC = "grpc"
    ZMQ = "zmq"


class WorkerType(StrEnum):
    """Worker specialization type."""

    REGULAR = "regular"
    ENCODE = "encode"
    PREFILL = "prefill"
    DECODE = "decode"


class Runtime(StrEnum):
    """Inference runtime/backend."""

    SGLANG = "sglang"
    VLLM = "vllm"
    TRTLLM = "trtllm"
    MLX = "mlx"
    TOKENSPEED = "tokenspeed"
    # The scripted mock worker (`crates/mock_worker`): no model, the test
    # states the answer before it sends the request. CPU only.
    MOCK = "mock"
    OPENAI = "openai"
    XAI = "xai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


# Convenience sets
LOCAL_MODES = frozenset({ConnectionMode.HTTP, ConnectionMode.GRPC, ConnectionMode.ZMQ})
LOCAL_RUNTIMES = frozenset(
    {Runtime.SGLANG, Runtime.VLLM, Runtime.TRTLLM, Runtime.MLX, Runtime.TOKENSPEED, Runtime.MOCK}
)
CLOUD_RUNTIMES = frozenset({Runtime.OPENAI, Runtime.XAI, Runtime.GEMINI, Runtime.ANTHROPIC})

# Fixture parameter names (used in @pytest.mark.parametrize)
PARAM_SETUP_BACKEND = "setup_backend"
PARAM_BACKEND_ROUTER = "backend_router"
PARAM_MODEL = "model"

# Default model
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Default runtime for gRPC tests
DEFAULT_RUNTIME = "sglang"

# Environment variable names
ENV_MODELS = "E2E_MODELS"
ENV_BACKENDS = "E2E_BACKENDS"
ENV_MODEL = "E2E_MODEL"
ENV_RUNTIME = (
    "E2E_RUNTIME"  # Runtime for gRPC tests — one of Runtime.{SGLANG,VLLM,TRTLLM,TOKENSPEED}
)
ENV_CONNECTION_MODE = (
    "E2E_CONNECTION_MODE"  # Per-lane wire override — see get_connection_mode_override
)
ENV_ZMQ_ENGINE_COUNT = (
    "E2E_ZMQ_ENGINE_COUNT"  # DP engines per ZMQ worker (grouped vLLM launch; empty = 1)
)
# The gRPC servicer implementation of a lane's workers, per engine: python or
# rust; unset follows the servicer package's default — see get_servicer_impl.
ENV_VLLM_SERVICER_IMPL = "E2E_VLLM_SERVICER_IMPL"
ENV_SGLANG_SERVICER_IMPL = "E2E_SGLANG_SERVICER_IMPL"
ENV_TOKENSPEED_SERVICER_IMPL = "E2E_TOKENSPEED_SERVICER_IMPL"
ENV_MM_PROCESSING = (
    "E2E_MM_PROCESSING"  # Per-lane multimodal processing location — see get_mm_processing
)
ENV_VLLM_MM_PROCESSOR = "E2E_VLLM_MM_PROCESSOR"  # Worker-lane processor — see get_vllm_mm_processor
ENV_PARSER = "E2E_PARSER"  # The gateway's --parser for a lane — see get_parser
ENV_STARTUP_TIMEOUT = (
    "E2E_STARTUP_TIMEOUT"  # Floor on every startup wait — see effective_startup_timeout
)
ENV_GPU_OFFSET = "E2E_GPU_OFFSET"  # First GPU index a lane's workers take — see get_gpu_offset
ENV_GPU_VENDOR = "E2E_GPU_VENDOR"  # The GPU vendor a lane runs on — see get_gpu_vendor
ENV_SKIP_MODEL_POOL = "SKIP_MODEL_POOL"
ENV_SKIP_BACKEND_SETUP = "SKIP_BACKEND_SETUP"


# Runtime detection helpers
_RUNTIME_CACHE = None


def get_runtime() -> str:
    """Get the current test runtime (sglang or vllm).

    Returns:
        Runtime name from E2E_RUNTIME environment variable, defaults to "sglang".
    """
    global _RUNTIME_CACHE
    if _RUNTIME_CACHE is None:
        _RUNTIME_CACHE = os.environ.get(ENV_RUNTIME, DEFAULT_RUNTIME)
    return _RUNTIME_CACHE


def is_vllm() -> bool:
    """Check if tests are running with vLLM runtime.

    Returns:
        True if E2E_RUNTIME is "vllm", False otherwise.
    """
    return get_runtime() == "vllm"


def is_sglang() -> bool:
    """Check if tests are running with SGLang runtime.

    Returns:
        True if E2E_RUNTIME is "sglang", False otherwise.
    """
    return get_runtime() == "sglang"


def is_trtllm() -> bool:
    """Check if tests are running with TensorRT-LLM runtime.

    Returns:
        True if E2E_RUNTIME is "trtllm", False otherwise.
    """
    return get_runtime() == "trtllm"


def is_mlx() -> bool:
    """Check if tests are running with MLX runtime (Apple Silicon only).

    Returns:
        True if E2E_RUNTIME is "mlx", False otherwise.
    """
    return get_runtime() == "mlx"


def is_tokenspeed() -> bool:
    """Check if tests are running with TokenSpeed runtime.

    Returns:
        True if E2E_RUNTIME is "tokenspeed", False otherwise.
    """
    return get_runtime() == "tokenspeed"


def get_connection_mode_override() -> "ConnectionMode | None":
    """Per-lane wire-protocol override for local backends.

    Set ``E2E_CONNECTION_MODE`` to run the existing local test cases over a
    different wire (like ``E2E_RUNTIME`` picks the engine): a ``grpc``/``http``
    case then runs over that mode without a separate parametrize value. PD/EPD
    backends keep their own wire. Returns ``None`` when the var is unset or
    blank (the workflow always exports it and leaves it empty for non-override
    lanes); a set-but-unrecognized value is a misconfiguration and raises.
    """
    value = os.environ.get(ENV_CONNECTION_MODE)
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    valid = [mode.value for mode in ConnectionMode]
    try:
        return ConnectionMode(value.lower())
    except ValueError:
        raise ValueError(
            f"{ENV_CONNECTION_MODE}={value!r} is not a valid connection mode; use one of {valid}"
        ) from None


MM_PROCESSING_WORKER = "worker"


def get_mm_processing() -> str | None:
    """Per-lane multimodal processing location for local vLLM gRPC workers.

    Set ``E2E_MM_PROCESSING=worker`` to launch every vLLM gRPC worker with
    ``SMG_VLLM_MM_PROCESSOR=inprocess``: the worker advertises worker-side media
    processing and the gateway (left in its default ``auto`` mode) forwards
    media references instead of preprocessed tensors for models that opt in.
    Returns ``None`` when the var is unset or blank (the router path); a
    set-but-unrecognized value is a misconfiguration and raises.
    """
    value = os.environ.get(ENV_MM_PROCESSING, "").strip().lower()
    if not value:
        return None
    if value != MM_PROCESSING_WORKER:
        raise ValueError(
            f"{ENV_MM_PROCESSING}={value!r} is not a valid processing location; "
            f"use {MM_PROCESSING_WORKER!r} or leave it unset"
        )
    return value


SERVICER_IMPLS = ("python", "rust")
# Per engine: the lane's setting and the variable the servicer package's
# launchers read it under in the worker's environment.
SERVICER_IMPL_VARIABLES = {
    "vllm": (ENV_VLLM_SERVICER_IMPL, "SMG_VLLM_SERVICER_IMPL"),
    "sglang": (ENV_SGLANG_SERVICER_IMPL, "SMG_SGLANG_SERVICER_IMPL"),
    "tokenspeed": (ENV_TOKENSPEED_SERVICER_IMPL, "SMG_TOKENSPEED_SERVICER_IMPL"),
}
VLLM_MM_PROCESSORS = ("inprocess", "smg")


def get_parser() -> str | None:
    """The parser a lane starts the gateway with, its ``--parser`` flag.

    Set ``E2E_PARSER=legacy`` to read every chat completion with the per-model
    parsers; ``symphony`` pins Symphony for the models its registry knows.
    Unset or blank leaves the gateway on its default, symphony.
    """
    value = os.environ.get(ENV_PARSER, "").strip().lower()
    if not value:
        return None
    if value not in ("legacy", "symphony"):
        raise ValueError(f"{ENV_PARSER}={value!r} is not a parser; use legacy or symphony")
    return value


def effective_parser() -> str:
    """The parser the gateway reads with in this lane: ``get_parser()`` when ``E2E_PARSER``
    is set, else the gateway's own default, ``symphony``. A test that branches on the parser
    reads this, so it need not know the default; the command line still omits ``--parser``
    when the variable is blank (``Gateway._build_base_cmd``).
    """
    return get_parser() or "symphony"


def get_vllm_mm_processor() -> str:
    """The processor the worker lane launches vLLM gRPC workers with
    (``SMG_VLLM_MM_PROCESSOR``): ``inprocess`` (vLLM's own, the default) or
    ``smg`` (smg's pipeline inside the Rust servicer, so it needs
    ``E2E_VLLM_SERVICER_IMPL=rust``). A set-but-unrecognized value raises.
    """
    value = os.environ.get(ENV_VLLM_MM_PROCESSOR, "").strip().lower() or "inprocess"
    if value not in VLLM_MM_PROCESSORS:
        raise ValueError(
            f"{ENV_VLLM_MM_PROCESSOR}={value!r} is not a valid processor; "
            f"use one of {VLLM_MM_PROCESSORS}"
        )
    if value == "smg" and get_servicer_impl("vllm") != "rust":
        raise ValueError(
            f"{ENV_VLLM_MM_PROCESSOR}=smg needs the Rust servicer ({ENV_VLLM_SERVICER_IMPL}=rust, "
            "or the servicer package's default): smg's pipeline runs inside it"
        )
    return value


def get_servicer_impl_setting(engine: str) -> str | None:
    """The lane's own choice of gRPC servicer implementation for ``engine``'s
    workers: ``python`` or ``rust`` from ``E2E_<ENGINE>_SERVICER_IMPL``, or
    None when the variable is unset or blank and the lane leaves the choice to
    the servicer package's default. A set-but-unrecognized value raises.

    The worker command never changes: the choice travels to the engine's own
    entrypoint as ``SMG_<ENGINE>_SERVICER_IMPL`` in the worker's environment
    (``infra.worker``), and the Router and every test case stay the same.
    """
    setting, _variable = SERVICER_IMPL_VARIABLES[engine]
    value = os.environ.get(setting, "").strip().lower()
    if not value:
        return None
    if value not in SERVICER_IMPLS:
        raise ValueError(
            f"{setting}={value!r} is not a valid servicer impl; use one of {SERVICER_IMPLS}"
        )
    return value


def _package_servicer_switch() -> Callable[..., tuple[str, str]] | None:
    """The servicer package's own switch, the one its three launchers share
    (``smg_grpc_servicer.rust_lifecycle.servicer_switch``), or None when the
    package is not importable here. Imported late: the harness runs on hosts
    without it (its own unit tests), where no gRPC worker starts either."""
    try:
        from smg_grpc_servicer.rust_lifecycle import servicer_switch
    except ImportError:
        return None
    return servicer_switch


def get_servicer_impl_default(engine: str) -> str | None:
    """The implementation the servicer package selects for ``engine`` when a
    launch names none (no ``--servicer-impl``, no ``SMG_<ENGINE>_SERVICER_IMPL``):
    read from the package's own switch against an empty environment, so the
    harness and the workers it starts agree by construction. None when the
    package is not importable here."""
    switch = _package_servicer_switch()
    if switch is None:
        return None
    _setting, variable = SERVICER_IMPL_VARIABLES[engine]
    impl, _origin = switch(variable, environ={})
    return impl


def get_servicer_impl(engine: str) -> str | None:
    """Which implementation serves ``engine``'s gRPC contract on this lane: the
    lane's setting (:func:`get_servicer_impl_setting`), else the servicer
    package's default (:func:`get_servicer_impl_default`). None only when the
    lane sets nothing and the package is not importable here. The collection
    filters (``fixtures.hooks``), the skip conditions of the cases one
    implementation does not serve and the worker's hook check all read this,
    never a belief about what unset means."""
    setting = get_servicer_impl_setting(engine)
    if setting is not None:
        return setting
    return get_servicer_impl_default(engine)


def get_zmq_engine_count() -> int:
    """DP engines per ZMQ worker (grouped vLLM/TokenSpeed launch).

    Set ``E2E_ZMQ_ENGINE_COUNT`` to run a ZMQ lane with grouped workers: the
    worker launches that many engines on one socket set and the gateway's
    handshake awaits them all. Unset/blank means one engine per worker.
    """
    value = os.environ.get(ENV_ZMQ_ENGINE_COUNT, "").strip()
    if not value:
        return 1
    count = int(value)
    if count < 1:
        raise ValueError(f"{ENV_ZMQ_ENGINE_COUNT}={value!r} must be a positive integer")
    return count


def get_gpu_offset() -> int:
    """The first GPU index a lane's workers are placed on.

    Workers take GPUs sequentially from 0; on a multi-GPU host, set
    ``E2E_GPU_OFFSET`` to run several lanes side by side, each on its own
    GPUs. Every worker of the lane (regular and PD pools) shifts by it.
    Unset/blank means 0.
    """
    value = os.environ.get(ENV_GPU_OFFSET, "").strip()
    if not value:
        return 0
    offset = int(value)
    if offset < 0:
        raise ValueError(f"{ENV_GPU_OFFSET}={value!r} must be a non-negative integer")
    return offset


GPU_VENDORS = ("nvidia", "amd")


def get_gpu_vendor() -> str:
    """The GPU vendor a lane's workers run on: "nvidia" or "amd".

    The e2e job hands its ``gpu_vendor`` input down as ``E2E_GPU_VENDOR``.
    Unset/blank, the vendor is read from the SMI on the path: ``amd-smi``
    without ``nvidia-smi`` is a ROCm host, anything else counts as nvidia. A
    sweep reads it where an engine behaves differently per vendor (the PD
    topology sweep's sole-prefill restart over NIXL).
    """
    value = os.environ.get(ENV_GPU_VENDOR, "").strip().lower()
    if value:
        if value not in GPU_VENDORS:
            raise ValueError(
                f"{ENV_GPU_VENDOR}={value!r} is not a GPU vendor; use one of {GPU_VENDORS}"
            )
        return value
    if shutil.which("nvidia-smi") is None and shutil.which("amd-smi") is not None:
        return "amd"
    return "nvidia"


def get_startup_timeout() -> int | None:
    """``E2E_STARTUP_TIMEOUT`` in seconds, or ``None`` when unset/blank."""
    value = os.environ.get(ENV_STARTUP_TIMEOUT, "").strip()
    if not value:
        return None
    timeout = int(value)
    if timeout <= 0:
        raise ValueError(f"{ENV_STARTUP_TIMEOUT}={value!r} must be a positive number of seconds")
    return timeout


def effective_startup_timeout(spec_timeout: int) -> int:
    """A model's startup wait (its spec's ``startup_timeout`` or the default),
    raised to ``E2E_STARTUP_TIMEOUT`` when that is larger. The per-model bound
    assumes warm caches; on a fresh host an engine's first start JIT-compiles
    and autotunes kernels for minutes, which the env floor covers for every
    wait that gates on a loaded model (worker health, gateway readiness).
    An explicitly larger per-model bound is never shrunk.
    """
    return max(spec_timeout, get_startup_timeout() or 0)


ENV_VLLM_KV_BACKEND = "E2E_VLLM_KV_BACKEND"
# One KV transfer backend for every PD worker in the lane, whatever the
# engine; the per-engine variables below are the fallbacks.
ENV_KV_BACKEND = "E2E_KV_BACKEND"
ENV_SGLANG_TRANSFER_BACKEND = "E2E_SGLANG_TRANSFER_BACKEND"


def vllm_kv_backend() -> str:
    """KV transfer backend for vLLM PD workers: "nixl" (default), "mooncake" or "moriio"."""
    lane = os.environ.get(ENV_KV_BACKEND, "").strip().lower()
    return lane or os.environ.get(ENV_VLLM_KV_BACKEND, "nixl").lower()


def sglang_transfer_backend() -> str:
    """Disaggregation transfer backend for SGLang PD workers: "mooncake" (default) or "nixl"."""
    lane = os.environ.get(ENV_KV_BACKEND, "").strip().lower()
    return lane or os.environ.get(ENV_SGLANG_TRANSFER_BACKEND, "mooncake").lower()


# MoRI-IO (the "moriio" vLLM KV backend): READ or WRITE transfers, its
# transfer engine, and the address the workers advertise and SMG hands to
# their peers.
ENV_VLLM_MORIIO_MODE = "E2E_VLLM_MORIIO_MODE"
ENV_VLLM_MORIIO_BACKEND = "E2E_VLLM_MORIIO_BACKEND"
ENV_MORIIO_HOST = "E2E_MORIIO_HOST"


def vllm_moriio_mode() -> str:
    """MoRI-IO transfer mode for vLLM PD workers: "read" (default) or "write"."""
    mode = os.environ.get(ENV_VLLM_MORIIO_MODE, "").strip().lower() or "read"
    if mode not in ("read", "write"):
        raise ValueError(f"{ENV_VLLM_MORIIO_MODE} must be 'read' or 'write', not {mode!r}")
    return mode


def vllm_moriio_backend() -> str:
    """MoRI-IO transfer engine: "rdma" (default, through the NICs) or "xgmi" (GPU to GPU, one node)."""
    backend = os.environ.get(ENV_VLLM_MORIIO_BACKEND, "").strip().lower() or "rdma"
    if backend not in ("rdma", "xgmi"):
        raise ValueError(f"{ENV_VLLM_MORIIO_BACKEND} must be 'rdma' or 'xgmi', not {backend!r}")
    return backend


def moriio_host() -> str:
    """Address of this host for MoRI-IO: ``E2E_MORIIO_HOST``, else the default route's source."""
    host = os.environ.get(ENV_MORIIO_HOST, "").strip()
    if host:
        return host
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))  # sends nothing; only picks the route
        return s.getsockname()[0]


# Runtime display labels
RUNTIME_LABELS = {
    "sglang": "SGLang",
    "vllm": "vLLM",
    "trtllm": "TensorRT-LLM",
    "mlx": "MLX",
    "tokenspeed": "TokenSpeed",
    "mock": "Mock",
}

ENV_SHOW_ROUTER_LOGS = "SHOW_ROUTER_LOGS"
ENV_SHOW_WORKER_LOGS = "SHOW_WORKER_LOGS"

# Network
ENV_HOST = "E2E_HOST"  # The loopback the lane binds and dials — see default_host


def default_host() -> str:
    """The loopback the harness binds gateways and workers to and dials them at.

    ``127.0.0.1`` unless ``E2E_HOST`` names another address; ``E2E_HOST=::1`` runs
    a lane over IPv6 (bare literal, no brackets: it is passed to ``--host`` flags
    as is, and every URL goes through :func:`host_port`).
    """
    return os.environ.get(ENV_HOST, "").strip() or "127.0.0.1"


def host_port(host: str, port: int | str) -> str:
    """``host:port`` for a URL or a dial string: an IPv6 literal goes in brackets."""
    if ":" in host and not host.startswith("["):
        return f"[{host}]:{port}"
    return f"{host}:{port}"


DEFAULT_HOST = default_host()
BRAVE_MCP_PORT = int(os.environ.get("BRAVE_MCP_PORT") or 8080)
BRAVE_MCP_HOST = os.environ.get("BRAVE_MCP_HOST") or DEFAULT_HOST
BRAVE_MCP_URL = f"http://{host_port(BRAVE_MCP_HOST, BRAVE_MCP_PORT)}/mcp"

# In-process mock MCP server (see infra/mock_mcp_server.py). Bound to localhost
# on an auto-allocated port; host is exposed as a constant so tests and
# configuration helpers can share a single source of truth.
MOCK_MCP_HOST = DEFAULT_HOST

# Timeouts (seconds)
# A worker's start-up wait covers two phases before its health reads SERVING:
# the engine's own start (model load, graph capture, its kernel warm-ups) and
# the servicer's planned warm-up (smg_grpc_servicer.warmup), which holds the
# health until it ran and is bounded by its own budget, 300 s unless
# SMG_<ENGINE>_SERVICER_WARMUP says otherwise. The default bound is the sum, so
# a model whose engine start alone fit the old 300 s is not failed by the
# warm-up that now follows it (a 20B MoE on a slower accelerator took 2:36 of
# engine start and 2:45 of warm-up). A model's spec raises it further, and
# replaces it: such a spec is sized the same way, its engine's start plus the
# budget (``model_specs``, pinned by a test).
ENGINE_START_SECS = 300
SERVICER_WARMUP_BUDGET_SECS = 300  # smg_grpc_servicer.warmup.DEFAULT_BUDGET_SECS, pinned by a test
DEFAULT_STARTUP_TIMEOUT = ENGINE_START_SECS + SERVICER_WARMUP_BUDGET_SECS
DEFAULT_ROUTER_TIMEOUT = 60
HEALTH_CHECK_INTERVAL = 2  # Check every 2s (was 5s)

# Model loading configuration
INITIAL_GRACE_PERIOD = 30  # Wait before first health check (model loading time)
LAUNCH_STAGGER_DELAY = 10  # Delay between launching multiple workers (avoid I/O contention)

# Retry configuration
MAX_RETRY_ATTEMPTS = 6  # Max retries with exponential backoff (total ~63s: 1+2+4+8+16+32)

# Display formatting
LOG_SEPARATOR_WIDTH = 60  # Width for log separator lines (e.g., "="*60)
