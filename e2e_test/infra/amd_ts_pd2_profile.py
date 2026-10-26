"""Default-off original TokenSpeed PD2 semantics on an actual SGLang worker.

This is a functional reference profile, not a native TokenSpeed implementation.
It changes neither get_runtime() nor is_tokenspeed().
"""

from __future__ import annotations

import os

from .constants import (
    ConnectionMode,
    WorkerType,
    get_connection_mode_override,
    get_runtime,
    get_sglang_servicer_impl,
    sglang_transfer_backend,
)

PROFILE_ENV = "SMG_AMD_TS_PD2_REFERENCE"
QWEN_MODEL = "Qwen/Qwen3.5-9B"


def ts_pd2_reference_enabled() -> bool:
    """Only the explicit value 1 enables this experimental profile."""
    return os.environ.get(PROFILE_ENV) == "1"


def validate_ts_pd2_reference() -> None:
    """Reject another engine, tier, wire or KV route under this profile."""
    if not ts_pd2_reference_enabled():
        return
    if (
        get_runtime() != "sglang"
        or os.environ.get("E2E_ENGINE") != "sglang"
        or os.environ.get("E2E_GPU_TIER") != "2"
        or get_connection_mode_override() not in (None, ConnectionMode.GRPC)
        or get_sglang_servicer_impl() != "python"
        or sglang_transfer_backend() != "mooncake"
    ):
        raise ValueError(
            f"{PROFILE_ENV}=1 requires actual SGLang Python gRPC PD2 with Mooncake"
        )


def ts_pd2_reference_engine(engine: str | None) -> str | None:
    """Use original TS selection/model/skip markers, retaining the actual engine."""
    if ts_pd2_reference_enabled() and engine == "sglang":
        return "tokenspeed"
    return engine


def ts_pd2_sglang_args(
    model_id: str,
    tp_size: int,
    mode: ConnectionMode,
    worker_type: WorkerType,
    kv_backend: str,
) -> list[str]:
    """Map the original Qwen window/memory values onto the actual engine CLI."""
    if not ts_pd2_reference_enabled() or model_id != QWEN_MODEL:
        return []
    validate_ts_pd2_reference()
    if (
        tp_size != 1
        or mode != ConnectionMode.GRPC
        or worker_type not in (WorkerType.PREFILL, WorkerType.DECODE)
        or kv_backend != "mooncake"
    ):
        raise ValueError("The TS PD2 Qwen reference requires TP1 Mooncake P/D gRPC workers")
    # The original context test appends its own actual-engine 2048 override.
    # FA3 and TokenSpeed's layerwise flag are not SGLang/ROCm CLI settings.
    return ["--context-length", "8192", "--mem-fraction-static", "0.8"]
