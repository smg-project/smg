"""SGLang EPD startup behind the native launcher."""

import asyncio
import json
import os
import signal
import tempfile
from pathlib import Path

from smg_grpc_servicer.hostport import host_port

from .epd_plugin import EVENTS_ENV, ROLE_ENV

PROCESSOR_CONTRACT = "smg-image-grid-v1"
MAX_CONTEXT_BYTES = 16 * 1024**3
MAX_IMAGE_TOKENS_PER_REQUEST = 32768
_DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}
_DTYPE_ALIASES = {"half": "float16", "float": "float32"}


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
    for field in ("nnodes", "pp_size", "dp_size", "attn_dp_size"):
        if getattr(args, field, 1) != 1:
            raise ValueError(f"managed EPD does not support native {field} != 1")
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
    dtype = _normalize_dtype(dtype)
    config = dict(
        role=role,
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
    if role == "encoder":
        if not (1 <= args.port <= 65525):
            raise ValueError("managed encoder --port must be in 1..=65525 (control uses port + 10)")
        max_active, max_receivers = 64, 8
        config.update(
            max_active=max_active,
            control_listen=host_port(args.host or "0.0.0.0", args.port + 10),
            receive_endpoint=f"http://{host_port(advertise_host, args.port)}",
            max_artifact_bytes=388 * 1024**2,
            max_staging_bytes_per_request=256 * 1024**2,
            max_receivers=max_receivers,
            max_control_bytes=64 * 1024**2,
            max_images=32,
            max_http_connections=max_active * max_receivers + 16,
            processor_options={"model_id": args.model_path},
            shutdown_timeout_seconds=30,
        )
    else:
        config.update(max_records=16384, max_duration_ms=122000)
    return config


def _cfg_get(obj, key):
    """Read `key` from either a dict-like config object or an attribute namespace."""
    if obj is None:
        return None
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def encoder_artifact_bytes(hf_config, dtype):
    """Compute per-request preprocessed pixel-artifact budget from vision geometry and dtype."""
    element_bytes = _DTYPE_BYTES.get(dtype)
    if element_bytes is None:
        raise ValueError("unsupported managed visual dtype")
    vision = getattr(hf_config, "vision_config", None)
    raw_patch = _cfg_get(vision, "patch_size")
    if raw_patch is None:
        patch = 16
    elif isinstance(raw_patch, (dict, list, tuple)):
        vals = list(raw_patch.values() if isinstance(raw_patch, dict) else raw_patch)
        if not vals or any(type(v) is not int or v <= 0 for v in vals):
            raise ValueError("unsupported managed vision geometry")
        patch = max(vals)
    else:
        patch = raw_patch
    temporal = _cfg_get(vision, "temporal_patch_size")
    if temporal is None:
        temporal = 2
    merge = next(
        (
            v
            for v in (_cfg_get(vision, "spatial_merge_size"), _cfg_get(vision, "merge_size"))
            if v is not None
        ),
        2,
    )
    channels = next(
        (
            v
            for v in (_cfg_get(vision, "in_chans"), _cfg_get(vision, "num_channels"))
            if v is not None
        ),
        3,
    )
    for value in (patch, temporal, merge, channels):
        if type(value) is not int or value <= 0:
            raise ValueError("unsupported managed vision geometry")
    return (
        MAX_IMAGE_TOKENS_PER_REQUEST * channels * temporal * (patch * merge) ** 2 * element_bytes
        + 4 * 1024**2
    )


def encoder_staging_bytes(hidden_size, dtype):
    """Compute per-request staging byte budget from embedding dimension and dtype."""
    element_bytes = _DTYPE_BYTES.get(dtype)
    if type(hidden_size) is not int or hidden_size <= 0 or element_bytes is None:
        raise ValueError("unsupported managed embedding dimensions or dtype")
    return MAX_IMAGE_TOKENS_PER_REQUEST * hidden_size * element_bytes


def encoder_window(config):
    """Reserve queued controls, then bound materialized artifacts and native sends."""
    send_copies = 1 if config.get("embedding_backend") == "mooncake" else config["max_receivers"]
    execution_bytes = 2 * config["max_artifact_bytes"] + config["max_staging_bytes_per_request"] * (
        1 + send_copies
    )
    capacity = min(
        config["max_active"],
        (MAX_CONTEXT_BYTES - config["max_active"] * config["max_control_bytes"]) // execution_bytes,
    )
    if capacity <= 0:
        raise ValueError("managed E context limit cannot admit queued controls and native work")
    return capacity, execution_bytes


class ManagedBootstrap:
    def __init__(self, args):
        self.args = args
        self.server = None

    def start(self):
        """Start SGLang's encoder bootstrap listener on prefill workers."""
        from sglang.srt.disaggregation.encoder.receiver import EncoderBootstrapServer

        self.server = EncoderBootstrapServer(
            host=self.args.host,
            port=self.args.encoder_bootstrap_port,
            urls=[],
        )

    def shutdown(self):
        """Close the encoder bootstrap listener if started."""
        if self.server is not None:
            self.server.close()


def launcher(config):
    """Build a `SglangLauncher` subclass configured with managed P/D contract metadata."""
    from smg_grpc_servicer.sglang.rust import SglangLauncher, launch_scheduler_ranks

    class ManagedLauncher(SglangLauncher):
        ipc_prefix = "sglang-managed-servicer"
        warmup_env = ""

        def server_extras(self, args, tokenizer_dir):
            """Validate managed P/D flags and serialize `managed_config` for the Rust servicer."""
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
            """Launch SGLang scheduler ranks alongside the prefill encoder bootstrap bridge."""
            from smg_grpc_servicer.sglang.scheduler_metrics import serve_scheduler_metrics

            bootstrap = ManagedBootstrap(args) if config["role"] == "prefill" else None
            engine = launch_scheduler_ranks(
                args, handshake_port=handshake_port, encode_bridge=bootstrap
            )
            self.scheduler_metrics = serve_scheduler_metrics(args)
            return engine

    return ManagedLauncher()


def run_server(args):
    """Entry point for `SMG_SGLANG_EPD=1`: dispatch to managed encoder or P/D Rust servicer."""
    from smg_grpc_servicer.sglang.rust import resolve_servicer_impl

    if resolve_servicer_impl(args) != "rust":
        raise ValueError("managed EPD requires the Rust servicer")
    if args.encoder_only:
        return encoder_main(args)
    config = complete(args, args.disaggregation_mode)
    if config["role"] not in ("prefill", "decode"):
        raise ValueError("managed P/D requires native --disaggregation-mode prefill or decode")
    os.environ[ROLE_ENV] = config["role"]
    with tempfile.TemporaryDirectory(prefix="smg-managed-") as directory:
        endpoint = f"ipc://{directory}/metadata.sock" if config["role"] == "prefill" else None
        config["metadata_endpoint"] = endpoint
        if endpoint:
            os.environ[EVENTS_ENV] = endpoint
        from .epd_plugin import load_role_hooks

        load_role_hooks()
        raise SystemExit(asyncio.run(launcher(config).serve(args)))


async def wait_encoder_shutdown(server, stopped):
    """Monitor the Rust encoder server thread until shutdown is signalled or it exits."""
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
    try:
        if not hasattr(backend.encoder.preprocessor, "max_artifact_bytes"):
            raise RuntimeError("native rank zero did not install the managed Rust bridge")
        visual = getattr(backend.encoder.model, "visual", None)
        vision_dtype = _normalize_dtype(getattr(visual, "dtype", config["embedding_dtype"]))
        hf_config = backend.encoder.model_config.hf_config
        config["max_staging_bytes_per_request"] = encoder_staging_bytes(
            backend.encoder.model_config.hidden_size, config["embedding_dtype"]
        )
        config["max_artifact_bytes"] = encoder_artifact_bytes(hf_config, vision_dtype)
        config["execution_capacity"], _ = encoder_window(config)
        config["max_cpu"] = min(
            available_cpus(),
            config["execution_capacity"],
            2 * backend.scheduler.max_batch_size,
        )
        config["processor_options"]["model_dir"] = model_dir
        config["processor_options"]["max_inflight"] = config["max_cpu"]
        config["processor_options"]["encoder_dtype"] = vision_dtype
        image_token_id = getattr(hf_config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(hf_config, "image_token_index", None)
        config.update(
            model_path=args.model_path,
            tokenizer_path=tokenizer_dir,
            served_model_name=args.served_model_name or args.model_path,
            model_type=hf_config.model_type,
            image_token_id=image_token_id,
            http_listen=host_port(args.host or "0.0.0.0", args.port),
        )
        with tempfile.TemporaryDirectory(prefix="smg-encoder-") as directory:
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
                loop = asyncio.get_running_loop()

                def stop(sig, frame):
                    loop.call_soon_threadsafe(lambda: (server.drain(), stopped.set()))

                previous = {
                    sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)
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
                shutdown_deadline = (
                    asyncio.get_running_loop().time() + config["shutdown_timeout_seconds"]
                )
                while server.active():
                    if asyncio.get_running_loop().time() >= shutdown_deadline:
                        await stop_followers(backend)
                        os._exit(1)
                    await asyncio.sleep(0.05)
                stop_wait = max(
                    0.5, min(5.0, shutdown_deadline - asyncio.get_running_loop().time())
                )
                await asyncio.to_thread(server.stop, stop_wait)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if shutdown_deadline is None:
            shutdown_deadline = (
                asyncio.get_running_loop().time() + config["shutdown_timeout_seconds"]
            )

        def remaining():
            # Post-drain cleanup gets its own small budget so a drain that finishes near
            # the deadline still exits cleanly while keeping total shutdown bounded.
            return max(5.0, shutdown_deadline - asyncio.get_running_loop().time())

        try:
            if native_task is not None:
                native_task.cancel()
                await asyncio.wait_for(
                    asyncio.gather(native_task, return_exceptions=True),
                    timeout=remaining(),
                )
            await asyncio.wait_for(backend.stop(), timeout=remaining())
        except asyncio.TimeoutError:  # noqa: UP041 - Python 3.10
            await stop_followers(backend)
            os._exit(1)
        finally:
            await stop_followers(backend)


async def stop_followers(backend):
    """Terminate and reap follower TP encoder processes during shutdown."""
    for process in backend.tp_processes:
        if process.is_alive():
            process.terminate()
    for process in backend.tp_processes:
        await asyncio.to_thread(process.join, 1)
        if process.is_alive():
            process.kill()
            await asyncio.to_thread(process.join, 1)


def encoder_main(args):
    """Validate encoder CLI arguments, install role hooks, and run the managed E server."""
    from smg_grpc_servicer.sglang.mm_defaults import configure_threads

    configure_threads()
    config = complete(args, "encoder")
    os.environ[ROLE_ENV] = "encoder"
    os.environ["SMG_SGLANG_MANAGED_PROCESSOR_CONTRACT"] = config["processor_contract"]
    os.environ["SMG_SGLANG_MANAGED_BATCH_BYTES"] = str(MAX_CONTEXT_BYTES)
    from .epd_plugin import load_role_hooks

    load_role_hooks()
    if not args.encoder_only or args.language_only:
        raise ValueError("managed E requires native --encoder-only")
    if args.encoder_transfer_backend not in ("mooncake", "zmq_to_scheduler"):
        raise ValueError("managed E supports only Mooncake or scheduler ZMQ transfers")
    asyncio.run(serve(config, args))
