"""Managed deployment defaults; native arguments remain authoritative for engine settings."""

from __future__ import annotations

import copy
import ctypes
import os
import sys
from pathlib import Path

PROCESSOR_CONTRACT = "smg-qwen3-vl-f32-v1"


def available_cpus():
    """Respect CPU affinity and the container quota when deriving preprocessing capacity."""
    count = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    )
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, int(quota) // int(period)))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    return max(1, count)


def complete(config, args, role):
    """Derive native contracts and finite limits instead of requiring duplicated deployment JSON."""
    # This adapter exports one complete Qwen3-VL layout from a single TP group.
    for field in ("nnodes", "pp_size", "dp_size", "attn_dp_size"):
        if getattr(args, field, 1) != 1:
            raise ValueError(f"managed EPD does not support native {field} != 1")
    if getattr(args, "enable_unified_memory", False):
        raise ValueError("managed EPD does not support unified-memory KV envelopes")
    supplied = copy.deepcopy(config)
    contract = supplied.pop("contract", {})
    revision = supplied.pop("model_revision", contract.get("model_revision")) or getattr(
        args, "revision", None
    )
    if not revision:
        raise ValueError("managed workers require a pinned model_revision or native --revision")
    advertise_host = supplied.pop("advertise_host", None) or args.host
    if advertise_host in ("0.0.0.0", "::"):
        from sglang.srt.utils.network import get_local_ip_auto

        advertise_host = get_local_ip_auto()
    dtype = args.dtype
    if dtype == "auto":
        from sglang.srt.configs.model_config import ModelConfig

        dtype = str(ModelConfig.from_server_args(args).dtype).removeprefix("torch.")
    derived = dict(
        model_revision=revision,
        processor_contract=PROCESSOR_CONTRACT,
        embedding_dtype=dtype,
        kv_backend=args.disaggregation_transfer_backend,
        embedding_backend=args.encoder_transfer_backend,
        topology_contract=f"single-host-tp{args.tp_size}",
        bootstrap_host=advertise_host,
        bootstrap_port=args.disaggregation_bootstrap_port,
        tp_size=args.tp_size,
    )
    # Existing explicit files are accepted, but cannot misadvertise native settings.
    for key in (
        "embedding_dtype",
        "embedding_backend",
        "kv_backend",
        "tp_size",
        "topology_contract",
    ):
        for source in (supplied, contract):
            advertised = source.get(key)
            if advertised is not None and advertised != derived[key]:
                raise ValueError(f"managed {key} differs from native runtime")
    if (
        contract.get("processor_contract", PROCESSOR_CONTRACT) != PROCESSOR_CONTRACT
        or supplied.get("processor_contract", PROCESSOR_CONTRACT) != PROCESSOR_CONTRACT
    ):
        raise ValueError("managed processor contract is determined by the validated adapter")
    if (
        role == "prefill"
        and contract.get("bootstrap_port", derived["bootstrap_port"]) != derived["bootstrap_port"]
    ):
        raise ValueError("managed bootstrap port differs from native prefill")
    derived.update(contract)
    defaults = dict(
        worker_id=f"{role}-{advertise_host}:{args.port}",
        max_active=64,
        # Completed operations retain their replay fence for up to 182 seconds.
        # Size that bounded history independently of concurrent native work:
        # 1024 records exhausted under sustained eight-request GPU workloads.
        max_records=16384,
        max_work=1000000,
        max_context_bytes=2 * 1024**3,
        retention_ms=60000,
        max_duration_ms=122000,
        max_clock_error_ms=1000,
    )
    if role == "encoder":
        defaults.update(derived)
        defaults.update(
            control_listen=f"0.0.0.0:{args.port + 10}",
            receive_endpoint=f"http://{advertise_host}:{args.port}",
            max_cpu=available_cpus(),
            max_artifact_bytes=64 * 1024**2,
            max_staging_bytes_per_request=32 * 1024**2,
            max_receivers=8,
            max_control_bytes=4 * 1024**2,
            max_images=32,
            drr_quantum=4096,
            aging_seconds=2,
            tenant_weights={},
            max_early_waiters=32,
            early_wait_seconds=5,
            load_interval_ms=100,
            max_http_connections=64,
            image_token_id=151655,
            processor_options={
                "model_dir": args.model_path,
                "model_id": args.model_path,
                "max_inflight": 2,
            },
            shutdown_timeout_seconds=30,
        )
    else:
        defaults.update(role=role, contract=derived)
    unknown = supplied.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"unknown managed configuration fields: {sorted(unknown)}")
    defaults.update(supplied)
    if role == "encoder":
        if "processor_options" not in supplied:
            defaults["processor_options"]["max_inflight"] = defaults["max_cpu"]
        receivers = defaults["max_active"] * defaults["max_receivers"]
        if "max_early_waiters" not in supplied:
            defaults["max_early_waiters"] = receivers
        if "max_http_connections" not in supplied:
            defaults["max_http_connections"] = receivers + 16
    return defaults


def encoder_staging_bytes(hidden_size, dtype):
    """Reserve the same token window across model widths using the native embedding dtype."""
    element_bytes = {"float16": 2, "bfloat16": 2, "float32": 4}.get(dtype)
    if type(hidden_size) is not int or hidden_size <= 0 or element_bytes is None:
        raise ValueError("unsupported managed embedding dimensions or dtype")
    return 4096 * hidden_size * element_bytes


def encoder_cpu_limit(config, batch_size):
    """Bound adaptive CPU supply by the actual host, native batch and owned memory."""
    processor_limit = config["processor_options"].get("max_inflight", config["max_cpu"])
    if type(processor_limit) is not int or processor_limit <= 0:
        raise ValueError("managed processor max_inflight must be a positive integer")
    return min(
        config["max_cpu"],
        processor_limit,
        available_cpus(),
        encoder_window(config)[0],
        2 * batch_size,
    )


def encoder_window(config):
    """Reserve all queued controls, then bound materialized artifacts and native sends."""
    # Native Mooncake sends all destinations from one registered GPU embedding;
    # it does not allocate one embedding copy per P rank. Reserve both the fused
    # batch output and its per-request GPU clone: native staging clones slices
    # so a slow receiver cannot pin the whole batch. ZMQ TensorWrapper can
    # materialize a CPU copy for every destination, so retain its full fanout.
    # Outstanding send bytes remain separately charged until native quiescence.
    send_copies = 1 if config.get("embedding_backend") == "mooncake" else config["max_receivers"]
    execution_bytes = config["max_artifact_bytes"] + config["max_staging_bytes_per_request"] * (
        1 + send_copies
    )
    capacity = min(
        config["max_active"],
        (config["max_context_bytes"] - config["max_active"] * config["max_control_bytes"])
        // execution_bytes,
    )
    if capacity <= 0:
        raise ValueError("managed E context limit cannot admit queued controls and native work")
    return capacity, execution_bytes


class Timeval(ctypes.Structure):
    """The platform libc timeval used by ntp_gettime."""

    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class NtpTimeval(ctypes.Structure):
    """glibc ntptimeval ABI, including its reserved storage."""

    _fields_ = [
        ("time", Timeval),
        ("maxerror", ctypes.c_long),
        ("esterror", ctypes.c_long),
        ("tai", ctypes.c_long),
        ("reserved", ctypes.c_long * 4),
    ]


def healthy(max_error_ms):
    """Do not confuse a wall timestamp with evidence that the node clock is synchronized."""
    if sys.platform != "linux" or ctypes.sizeof(ctypes.c_long) != 8 or max_error_ms <= 0:
        return False
    libc = ctypes.CDLL(None)
    query = getattr(libc, "ntp_gettime", None)
    if query is None:
        return False
    query.argtypes = [ctypes.POINTER(NtpTimeval)]
    query.restype = ctypes.c_int
    sample = NtpTimeval()
    state = query(ctypes.byref(sample))
    return 0 <= state < 5 and 0 <= sample.maxerror <= max_error_ms * 1000
