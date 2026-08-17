"""Constants and enums for E2E test infrastructure."""

import os
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
    OPENAI = "openai"
    XAI = "xai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


# Convenience sets
LOCAL_MODES = frozenset({ConnectionMode.HTTP, ConnectionMode.GRPC, ConnectionMode.ZMQ})
LOCAL_RUNTIMES = frozenset(
    {Runtime.SGLANG, Runtime.VLLM, Runtime.TRTLLM, Runtime.MLX, Runtime.TOKENSPEED}
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
ENV_STARTUP_TIMEOUT = "E2E_STARTUP_TIMEOUT"
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


ENV_VLLM_KV_BACKEND = "E2E_VLLM_KV_BACKEND"


def vllm_kv_backend() -> str:
    """KV transfer backend for vLLM PD workers: "nixl" (default) or "mooncake"."""
    return os.environ.get(ENV_VLLM_KV_BACKEND, "nixl").lower()


# Runtime display labels
RUNTIME_LABELS = {
    "sglang": "SGLang",
    "vllm": "vLLM",
    "trtllm": "TensorRT-LLM",
    "mlx": "MLX",
    "tokenspeed": "TokenSpeed",
}

ENV_SHOW_ROUTER_LOGS = "SHOW_ROUTER_LOGS"
ENV_SHOW_WORKER_LOGS = "SHOW_WORKER_LOGS"

# Network
DEFAULT_HOST = "127.0.0.1"
BRAVE_MCP_PORT = int(os.environ.get("BRAVE_MCP_PORT") or 8080)
BRAVE_MCP_HOST = os.environ.get("BRAVE_MCP_HOST") or DEFAULT_HOST
BRAVE_MCP_URL = f"http://{BRAVE_MCP_HOST}:{BRAVE_MCP_PORT}/mcp"

# In-process mock MCP server (see infra/mock_mcp_server.py). Bound to localhost
# on an auto-allocated port; host is exposed as a constant so tests and
# configuration helpers can share a single source of truth.
MOCK_MCP_HOST = DEFAULT_HOST

# Timeouts (seconds)
DEFAULT_STARTUP_TIMEOUT = 300
DEFAULT_ROUTER_TIMEOUT = 60
HEALTH_CHECK_INTERVAL = 2  # Check every 2s (was 5s)

# Model loading configuration
INITIAL_GRACE_PERIOD = 30  # Wait before first health check (model loading time)
LAUNCH_STAGGER_DELAY = 10  # Delay between launching multiple workers (avoid I/O contention)

# Retry configuration
MAX_RETRY_ATTEMPTS = 6  # Max retries with exponential backoff (total ~63s: 1+2+4+8+16+32)

# Display formatting
LOG_SEPARATOR_WIDTH = 60  # Width for log separator lines (e.g., "="*60)
