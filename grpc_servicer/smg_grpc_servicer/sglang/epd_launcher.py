"""SGLang EPD startup behind the native launcher."""

import asyncio
import json
import os
import signal
import tempfile
from pathlib import Path

from smg_grpc_servicer.hostport import host_port
from smg_grpc_servicer.rust_lifecycle import (
    DEFAULT_DRAIN_SECS,
    default_socket_dir,
    env_float,
    prepare_socket_dir,
)

from .epd_plugin import EVENTS_ENV, ROLE_ENV
from .rust import DRAIN_SECS_ENV

PROCESSOR_CONTRACT = "smg-image-grid-v1"
MAX_CONTEXT_BYTES = 48 * 1024**3
DEFAULT_MAX_ACTIVE = 256
DEFAULT_MAX_EXECUTION_CAPACITY = 8
MAX_IMAGE_TOKENS_PER_REQUEST = 32768
MAX_MEDIA_TOKENS_CEILING = 262144
_ARTIFACT_HEADER_BYTES = 4 * 1024**2
_MEMORY_BUDGET_NUMERATOR = 1
_MEMORY_BUDGET_DENOMINATOR = 2
_CGROUP_V2_MEMORY_MAX = Path("/sys/fs/cgroup/memory.max")
_CGROUP_V1_MEMORY_LIMIT = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
_PROC_MEMINFO = Path("/proc/meminfo")
_CGROUP_UNLIMITED_SENTINEL = 1 << 60
_DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}
_DTYPE_ALIASES = {"half": "float16", "float": "float32"}


def _parse_positive_int_arg(raw, flag_name):
    """Parse and validate a positive integer CLI argument value."""
    if isinstance(raw, bool) or isinstance(raw, float):
        raise ValueError(f"{flag_name} must be a positive integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{flag_name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{flag_name} must be a positive integer")
    return value


def resolve_max_active(args=None):
    """Derive encoder admission cap from native --max-running-requests, defaulting to 256."""
    raw = getattr(args, "max_running_requests", None)
    if raw is None:
        return DEFAULT_MAX_ACTIVE
    return _parse_positive_int_arg(raw, "--max-running-requests")


def resolve_max_media_tokens(args=None, model_config=None):
    """Derive per-request media token budget from --context-length or model_config.context_len."""
    raw_cli = getattr(args, "context_length", None)
    if raw_cli is not None:
        value = _parse_positive_int_arg(raw_cli, "--context-length")
        return max(MAX_IMAGE_TOKENS_PER_REQUEST, min(value, MAX_MEDIA_TOKENS_CEILING))
    raw_model = getattr(model_config, "context_len", None)
    if type(raw_model) is int and raw_model > 0:
        return max(MAX_IMAGE_TOKENS_PER_REQUEST, min(raw_model, MAX_MEDIA_TOKENS_CEILING))
    return MAX_IMAGE_TOKENS_PER_REQUEST


def _host_memory_bytes():
    """Detect total host RAM via POSIX sysconf, falling back to /proc/meminfo."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and phys_pages > 0:
            return int(page_size) * int(phys_pages)
    except (AttributeError, OSError, ValueError):
        pass
    try:
        for line in _PROC_MEMINFO.read_text().splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    kb = int(parts[1])
                    if kb > 0:
                        return kb * 1024
                break
    except (OSError, ValueError):
        pass
    return None


def available_memory_bytes():
    """Respect container cgroup v2/v1 memory limits, falling back to host RAM."""
    host_ram = _host_memory_bytes()
    try:
        raw = _CGROUP_V2_MEMORY_MAX.read_text().strip()
    except OSError:
        raw = None
    if raw is not None:
        if raw != "max":
            try:
                value = int(raw)
                if value > 0:
                    return min(value, host_ram) if host_ram is not None else value
            except ValueError:
                pass
        return host_ram
    try:
        value = int(_CGROUP_V1_MEMORY_LIMIT.read_text().strip())
        if 0 < value < _CGROUP_UNLIMITED_SENTINEL:
            return min(value, host_ram) if host_ram is not None else value
    except (OSError, ValueError):
        pass
    return host_ram


def resolve_max_context_bytes():
    """Derive encoder context memory budget as 50% of container/host RAM, defaulting to 48 GiB."""
    mem_bytes = available_memory_bytes()
    if mem_bytes is None or mem_bytes <= 0:
        return MAX_CONTEXT_BYTES
    return max(1, (mem_bytes * _MEMORY_BUDGET_NUMERATOR) // _MEMORY_BUDGET_DENOMINATOR)


def max_execution_capacity(max_batch_size=None):
    """Bound concurrent Python/GPU execution slots so bursts queue cleanly in Rust."""
    if max_batch_size is not None and not isinstance(max_batch_size, bool):
        value = int(max_batch_size)
        if value > 0:
            return value
    return DEFAULT_MAX_EXECUTION_CAPACITY


def resolve_max_cpu(execution_capacity, max_batch_size=None):
    """Resolve CPU preprocessing concurrency from SMG_SGLANG_MM_MAX_INFLIGHT or derived default."""
    raw = os.environ.get("SMG_SGLANG_MM_MAX_INFLIGHT")
    if raw is not None:
        try:
            value = int(raw)
        except ValueError as error:
            raise ValueError("SMG_SGLANG_MM_MAX_INFLIGHT must be positive integer") from error
        if value <= 0:
            raise ValueError("SMG_SGLANG_MM_MAX_INFLIGHT must be positive integer")
        return value
    batch_cap = 2 * max_execution_capacity(max_batch_size)
    return max(1, min(available_cpus(), execution_capacity, batch_cap))


def _normalize_dtype(value):
    dtype = str(value).removeprefix("torch.")
    return _DTYPE_ALIASES.get(dtype, dtype)


def available_cpus():
    """Respect CPU affinity and cgroup v2 quota when deriving preprocessing capacity."""
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


def complete(args, role):
    """Derive native contracts and finite limits instead of requiring duplicated deployment JSON."""
    restricted = (
        ("nnodes", "pp_size", "dp_size", "attn_dp_size")
        if role == "encoder"
        else ("nnodes", "pp_size")
    )
    for field in restricted:
        if getattr(args, field, 1) != 1:
            raise ValueError(f"managed EPD {role} does not support native {field} != 1")
    if getattr(args, "enable_unified_memory", False):
        raise ValueError("managed EPD does not support unified-memory KV envelopes")
    revision = getattr(args, "revision", None)
    if not revision:
        raise ValueError("managed workers require a pinned native --revision")
    advertise_host = args.host
    if advertise_host in ("0.0.0.0", "::"):
        from sglang.srt.utils.network import get_local_ip_auto

        advertise_host = get_local_ip_auto()
    dtype = args.dtype
    if dtype == "auto":
        from sglang.srt.configs.model_config import ModelConfig

        dtype = ModelConfig.from_server_args(args).dtype
    config = dict(
        role=role,
        model_revision=revision,
        processor_contract=PROCESSOR_CONTRACT,
        embedding_dtype=_normalize_dtype(dtype),
        kv_backend=args.disaggregation_transfer_backend,
        embedding_backend=args.encoder_transfer_backend,
        topology_contract=f"single-host-tp{args.tp_size}",
        bootstrap_host=advertise_host,
        bootstrap_port=args.disaggregation_bootstrap_port,
        tp_size=args.tp_size,
    )
    if role == "encoder":
        from smg_grpc_servicer.mm_item_limits import sglang_item_limits

        if not (1 <= args.port <= 65525):
            raise ValueError("managed encoder --port must be in 1..=65525 (control uses port + 10)")
        max_active, max_receivers = resolve_max_active(args), 8
        resolve_max_media_tokens(args)
        mm_item_limits = sglang_item_limits(args)
        engine_item_limits = {
            modality: int(count)
            for entry in mm_item_limits.split(",")
            if entry
            for modality, count in [entry.split("=", 1)]
        }
        config.update(
            max_active=max_active,
            max_context_bytes=resolve_max_context_bytes(),
            control_listen=host_port(args.host or "0.0.0.0", args.port + 10),
            receive_endpoint=f"http://{host_port(advertise_host, args.port)}",
            max_artifact_bytes=388 * 1024**2,
            max_staging_bytes_per_request=256 * 1024**2,
            max_receivers=max_receivers,
            max_control_bytes=256 * 1024**2,
            max_images=1024,
            mm_item_limits=mm_item_limits,
            max_http_connections=max_active * (max_receivers + 1) + 16,
            processor_options={
                "model_id": args.model_path,
                "engine_item_limits": engine_item_limits,
            },
            shutdown_timeout_seconds=env_float(DRAIN_SECS_ENV, DEFAULT_DRAIN_SECS, positive=True),
        )
    else:
        config.update(max_records=16384, max_duration_ms=122000)
    return config


def _cfg_first(obj, *keys, default=None):
    if obj is None:
        return default
    for key in keys:
        val = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
        if val is not None:
            return val
    return default


def _artifact_bytes_per_token(hf_config, dtype):
    """Return preprocessed pixel artifact bytes per visual token for the model vision geometry."""
    element_bytes = _DTYPE_BYTES.get(dtype)
    if element_bytes is None:
        raise ValueError("unsupported managed visual dtype")
    vision = getattr(hf_config, "vision_config", None)
    raw_patch = _cfg_first(vision, "patch_size", default=16)
    if isinstance(raw_patch, (dict, list, tuple)):
        vals = list(raw_patch.values() if isinstance(raw_patch, dict) else raw_patch)
        if not vals or any(type(v) is not int or v <= 0 for v in vals):
            raise ValueError("unsupported managed vision geometry")
        patch = max(vals)
    else:
        patch = raw_patch
    temporal = _cfg_first(vision, "temporal_patch_size", default=2)
    merge = _cfg_first(vision, "spatial_merge_size", "merge_size", default=2)
    channels = _cfg_first(vision, "in_chans", "in_channels", "num_channels", default=3)
    if any(type(v) is not int or v <= 0 for v in (patch, temporal, merge, channels)):
        raise ValueError("unsupported managed vision geometry")
    return channels * temporal * (patch * merge) ** 2 * element_bytes


def encoder_artifact_bytes(hf_config, dtype, max_tokens=MAX_IMAGE_TOKENS_PER_REQUEST):
    """Compute per-request preprocessed pixel-artifact budget from vision geometry and dtype."""
    per_token = _artifact_bytes_per_token(hf_config, dtype)
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("unsupported managed media token limit")
    return max_tokens * per_token + _ARTIFACT_HEADER_BYTES


def _staging_bytes_per_token(hidden_size, dtype):
    """Return staged embedding bytes per visual token for `hidden_size` and `dtype`."""
    element_bytes = _DTYPE_BYTES.get(dtype)
    if type(hidden_size) is not int or hidden_size <= 0 or element_bytes is None:
        raise ValueError("unsupported managed embedding dimensions or dtype")
    return hidden_size * element_bytes


def _encoder_embedding_width(encoder, hf_config):
    """Derive actual visual embedding width (accounting for DeepStack multi-layer outputs)."""
    base_hidden = encoder.model_config.hidden_size
    width = base_hidden
    vision = getattr(hf_config, "vision_config", None)
    deepstack = _cfg_first(vision, "deepstack_visual_indexes", default=None)
    if isinstance(deepstack, (list, tuple)) and deepstack:
        width = max(width, base_hidden * (1 + len(deepstack)))
    for attr in ("embedding_dim", "output_dim", "_embedding_dim"):
        val = getattr(encoder, attr, None)
        if type(val) is int and val > 0:
            width = max(width, val)
    return width


def encoder_staging_bytes(hidden_size, dtype, max_tokens=MAX_IMAGE_TOKENS_PER_REQUEST):
    """Compute per-request staging byte budget from embedding dimension and dtype."""
    per_token = _staging_bytes_per_token(hidden_size, dtype)
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("unsupported managed embedding dimensions or dtype")
    return max_tokens * per_token


def _control_reserve_bytes(config, budget_bytes=None):
    """Reserve shared queued-control memory inside the encoder context byte budget."""
    if budget_bytes is None:
        budget_bytes = config.get("max_context_bytes")
        if budget_bytes is None:
            budget_bytes = resolve_max_context_bytes()
    return min(
        config["max_active"] * config["max_control_bytes"],
        max(config["max_control_bytes"], budget_bytes // 8),
    )


def fit_auto_media_tokens(
    target_tokens,
    hf_config,
    vision_dtype,
    hidden_size,
    embedding_dtype,
    config,
    max_batch_size=None,
):
    """Fit auto-derived media tokens to preserve target execution slots within memory budget."""
    if target_tokens <= MAX_IMAGE_TOKENS_PER_REQUEST:
        return MAX_IMAGE_TOKENS_PER_REQUEST
    send_copies = 1 if config.get("embedding_backend") == "mooncake" else config["max_receivers"]
    bytes_per_token = 2 * _artifact_bytes_per_token(hf_config, vision_dtype) + (
        1 + send_copies
    ) * _staging_bytes_per_token(hidden_size, embedding_dtype)
    budget_bytes = config.get("max_context_bytes")
    if budget_bytes is None:
        budget_bytes = resolve_max_context_bytes()
    available = budget_bytes - _control_reserve_bytes(config, budget_bytes)
    exec_cap = config.get("max_execution_capacity", max_execution_capacity(max_batch_size))
    target_slots = max(1, min(config["max_active"], exec_cap))
    per_slot = available // target_slots
    fixed_overhead = 2 * _ARTIFACT_HEADER_BYTES
    if per_slot <= fixed_overhead:
        return MAX_IMAGE_TOKENS_PER_REQUEST
    fit_tokens = (per_slot - fixed_overhead) // bytes_per_token
    return max(MAX_IMAGE_TOKENS_PER_REQUEST, min(target_tokens, fit_tokens))


def encoder_window(config, max_batch_size=None):
    """Reserve queued controls, then bound materialized artifacts and native sends."""
    send_copies = 1 if config.get("embedding_backend") == "mooncake" else config["max_receivers"]
    execution_bytes = 2 * config["max_artifact_bytes"] + config["max_staging_bytes_per_request"] * (
        1 + send_copies
    )
    budget_bytes = config.get("max_context_bytes")
    if budget_bytes is None:
        budget_bytes = resolve_max_context_bytes()
    control_reserve = _control_reserve_bytes(config, budget_bytes)
    budget_cap = (budget_bytes - control_reserve) // execution_bytes
    if budget_cap <= 0:
        raise ValueError("managed E context limit cannot admit queued controls and native work")
    exec_cap = config.get("max_execution_capacity", max_execution_capacity(max_batch_size))
    capacity = min(config["max_active"], budget_cap, exec_cap)
    return capacity, execution_bytes


class ManagedBootstrap:
    def __init__(self, args):
        self.args, self.server = args, None

    def start(self):
        from sglang.srt.disaggregation.encoder.receiver import EncoderBootstrapServer

        self.server = EncoderBootstrapServer(
            host=self.args.host, port=self.args.encoder_bootstrap_port, urls=[]
        )

    def shutdown(self):
        if self.server is not None:
            self.server.close()


def launcher(config):
    """Build a ``SglangLauncher`` subclass configured for managed P or D startup."""
    from smg_grpc_servicer.sglang.rust import SglangLauncher

    class ManagedLauncher(SglangLauncher):
        ipc_prefix = "sglang-managed-servicer"
        warmup_env = ""

        def server_extras(self, args, tokenizer_dir):
            """Validate P/D CLI flags and attach serialized ``managed_config`` JSON."""
            if args.disaggregation_mode != config["role"]:
                raise ValueError("managed role must equal native disaggregation mode")
            if not args.language_only:
                raise ValueError("managed P/D requires --language-only")
            from .prepared_input import supports_routing_labels

            return {
                "managed_config": json.dumps(
                    {**config, "disable_routing_labels": not supports_routing_labels()}
                )
            }

        def launch(self, args, handshake_port, facts):
            """Attach the managed Prefill bootstrap listener, then delegate to ``SglangLauncher.launch``."""
            self.encode = ManagedBootstrap(args) if config["role"] == "prefill" else None
            return super().launch(args, handshake_port, facts)

    return ManagedLauncher()


def run_server(args):
    from smg_grpc_servicer.sglang.rust import resolve_servicer_impl

    if resolve_servicer_impl(args) != "rust":
        raise ValueError("managed EPD requires the Rust servicer")
    if args.encoder_only:
        return encoder_main(args)
    config = complete(args, args.disaggregation_mode)
    if config["role"] not in ("prefill", "decode"):
        raise ValueError("managed P/D requires native --disaggregation-mode prefill or decode")
    os.environ[ROLE_ENV] = config["role"]
    socket_dir = prepare_socket_dir(default_socket_dir())
    with tempfile.TemporaryDirectory(prefix="smg-managed-", dir=socket_dir) as directory:
        endpoint = f"ipc://{directory}/metadata.sock" if config["role"] == "prefill" else None
        config["metadata_endpoint"] = endpoint
        if endpoint:
            os.environ[EVENTS_ENV] = endpoint
        else:
            os.environ.pop(EVENTS_ENV, None)
        from .epd_plugin import load_role_hooks

        load_role_hooks()
        raise SystemExit(asyncio.run(launcher(config).serve(args)))


async def wait_encoder_shutdown(server, stopped):
    while not stopped.is_set():
        error = server.last_error()
        if error or not server.running():
            raise RuntimeError(f"Rust encoder control server stopped: {error or 'server exited'}")
        try:
            await asyncio.wait_for(stopped.wait(), 1)
        except asyncio.TimeoutError:  # noqa: UP041 - Python 3.10
            pass


async def serve(config, args):
    """Rust owns the E gRPC control plane; Python runs TP/GPU work and the P->E HTTP facade."""
    from smg.smg_rs import ManagedEncoderServer

    from smg_grpc_servicer.sglang.model_artifacts import encoder_artifact_dirs

    from .epd_encoder import launch_runtime, serve_native

    model_dir, tokenizer_dir = encoder_artifact_dirs(args)
    backend = launch_runtime(args, config["max_active"])
    server = native_task = shutdown_deadline = None
    stopped = asyncio.Event()
    previous = {}
    loop = asyncio.get_running_loop()
    try:
        if not hasattr(backend.encoder.preprocessor, "max_artifact_bytes"):
            raise RuntimeError("native rank zero did not install the managed Rust bridge")
        from smg_grpc_servicer.sglang.mm_defaults import encoder_dtype

        visual = getattr(backend.encoder.model, "visual", None)
        vision_dtype = encoder_dtype(
            os.environ.get("SMG_SGLANG_MM_DTYPE"),
            lambda: _normalize_dtype(getattr(visual, "dtype", config["embedding_dtype"])),
        )
        hf_config = backend.encoder.model_config.hf_config
        config.setdefault("max_context_bytes", resolve_max_context_bytes())
        max_batch_size = getattr(backend.scheduler, "max_batch_size", None)
        staging_width = _encoder_embedding_width(backend.encoder, hf_config)
        staging_dtype = _normalize_dtype(
            getattr(backend.encoder, "_embedding_dtype", None) or config["embedding_dtype"]
        )
        max_tokens = resolve_max_media_tokens(args, backend.encoder.model_config)
        if getattr(args, "context_length", None) is None:
            max_tokens = fit_auto_media_tokens(
                max_tokens,
                hf_config,
                vision_dtype,
                staging_width,
                staging_dtype,
                config,
                max_batch_size=max_batch_size,
            )
        config["max_staging_bytes_per_request"] = encoder_staging_bytes(
            staging_width,
            staging_dtype,
            max_tokens=max_tokens,
        )
        config["max_artifact_bytes"] = encoder_artifact_bytes(
            hf_config, vision_dtype, max_tokens=max_tokens
        )
        config["max_queued_control_bytes"] = _control_reserve_bytes(config)
        config["execution_capacity"], _ = encoder_window(config, max_batch_size=max_batch_size)
        config["max_cpu"] = resolve_max_cpu(
            config["execution_capacity"], max_batch_size=max_batch_size
        )
        config["processor_options"].update(
            model_dir=model_dir, max_inflight=config["max_cpu"], encoder_dtype=vision_dtype
        )
        config.update(
            model_path=args.model_path,
            tokenizer_path=tokenizer_dir,
            served_model_name=args.served_model_name or args.model_path,
            model_type=hf_config.model_type,
            http_listen=host_port(args.host or "0.0.0.0", args.port),
        )
        socket_dir = prepare_socket_dir(default_socket_dir())
        with tempfile.TemporaryDirectory(prefix="smg-encoder-", dir=socket_dir) as directory:
            config["native_commands"] = f"ipc://{directory}/commands.sock"
            config["native_events"] = f"ipc://{directory}/events.sock"
            server = ManagedEncoderServer(
                json.dumps(config),
                config["processor_options"],
                tokenizer_dir,
                config["max_artifact_bytes"],
            )
            stop_task = None
            try:
                backend.start()
                native_task = asyncio.create_task(serve_native(config, backend))
                previous = {
                    sig: signal.signal(
                        sig,
                        lambda _s, _f: loop.call_soon_threadsafe(
                            lambda: (server.drain(), stopped.set())
                        ),
                    )
                    for sig in (signal.SIGTERM, signal.SIGINT)
                }
                stop_task = asyncio.create_task(wait_encoder_shutdown(server, stopped))
                done, _ = await asyncio.wait(
                    (stop_task, native_task), return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    await task
            finally:
                if stop_task is not None:
                    stop_task.cancel()
                    await asyncio.gather(stop_task, return_exceptions=True)
                server.drain()
                shutdown_deadline = loop.time() + config["shutdown_timeout_seconds"]
                while server.active():
                    if loop.time() >= shutdown_deadline:
                        await stop_followers(backend)
                        os._exit(1)
                    await asyncio.sleep(0.05)
                await asyncio.to_thread(
                    server.stop, max(0.5, min(5.0, shutdown_deadline - loop.time()))
                )
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if shutdown_deadline is None:
            shutdown_deadline = loop.time() + config["shutdown_timeout_seconds"]

        def remaining():
            return max(5.0, shutdown_deadline - loop.time())

        try:
            if native_task is not None:
                native_task.cancel()
                await asyncio.wait_for(
                    asyncio.gather(native_task, return_exceptions=True), timeout=remaining()
                )
            await asyncio.wait_for(backend.stop(), timeout=remaining())
        except asyncio.TimeoutError:  # noqa: UP041 - Python 3.10
            await stop_followers(backend)
            os._exit(1)
        finally:
            await stop_followers(backend)


async def stop_followers(backend):
    for process in backend.tp_processes:
        if process.is_alive():
            process.terminate()
    for process in backend.tp_processes:
        await asyncio.to_thread(process.join, 1)
        if process.is_alive():
            process.kill()
            await asyncio.to_thread(process.join, 1)


def encoder_main(args):
    from smg_grpc_servicer.sglang.mm_defaults import configure_threads

    configure_threads()
    config = complete(args, "encoder")
    os.environ[ROLE_ENV] = "encoder"
    if args.encoder_transfer_backend == "mooncake":
        # Bound ZMQ ACK blocking when Prefill has already aborted instead of stalling release 180s.
        os.environ.setdefault("SGLANG_ENCODER_SEND_TIMEOUT", "15")
    from .epd_plugin import load_role_hooks

    load_role_hooks()
    if not args.encoder_only or args.language_only:
        raise ValueError("managed E requires native --encoder-only")
    if args.encoder_transfer_backend not in ("mooncake", "zmq_to_scheduler"):
        raise ValueError("managed E supports only Mooncake or scheduler ZMQ transfers")
    asyncio.run(serve(config, args))
