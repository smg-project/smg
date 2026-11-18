"""Managed lifecycle behind native SGLang launch; the module entry remains a compatibility shim."""

import asyncio
import json
import os
import signal
import sys
import tempfile
import uuid

from .plugin import EVENTS_ENV, INSTANCE_ENV, ROLE_ENV, WORKER_ENV


def set_identity(config, role):
    """Give every native role a fresh process fence before plugins or TP ranks load."""
    config["generation"] = uuid.uuid4().hex
    os.environ[ROLE_ENV] = role
    os.environ[WORKER_ENV] = config["worker_id"]
    os.environ[INSTANCE_ENV] = config["generation"]


class ManagedBootstrap:
    """Own only native encoder registration; no frontend encode dispatcher or processor."""

    def __init__(self, args):
        """Defer receiver startup until native arguments and process identity are established."""
        self.args = args
        self.server = None

    def start(self):
        """Start the native receiver bootstrap before native scheduler ranks register."""
        from sglang.srt.disaggregation.encoder.receiver import EncoderBootstrapServer

        self.server = EncoderBootstrapServer(
            host=self.args.host,
            port=self.args.encoder_bootstrap_port,
            urls=[],
        )

    def shutdown(self):
        """Release the native bootstrap server with the scheduler process group."""
        if self.server is not None:
            self.server.close()


def launcher(config):
    """Construct a separate lifecycle without changing the default SGLang launcher."""
    from smg_grpc_servicer.sglang.rust import SglangLauncher, launch_scheduler_ranks

    class ManagedLauncher(SglangLauncher):
        ipc_prefix = "sglang-managed-servicer"
        warmup_env = ""

        def server_extras(self, args, tokenizer_dir):
            """Managed startup never constructs the legacy encode-dispatch bridge."""
            if args.disaggregation_mode != config["role"]:
                raise ValueError("managed role must equal native disaggregation mode")
            if not args.language_only:
                raise ValueError("managed P/D requires --language-only")
            return {"managed_config": json.dumps(config)}

        def launch(self, args, handshake_port, facts):
            """Use unchanged native TP/DP spawning and P's native receiver bootstrap."""
            bootstrap = ManagedBootstrap(args) if config["role"] == "prefill" else None
            return launch_scheduler_ranks(
                args, handshake_port=handshake_port, encode_bridge=bootstrap
            )

    return ManagedLauncher()


def main():
    """Load finite limits explicitly and assign a new process fence on every launch."""
    from sglang.srt.server_args import prepare_server_args

    return run_server(prepare_server_args(sys.argv[1:]))


def run_server(args):
    """Derive the role from native arguments; own initialization and shutdown exactly once."""
    from smg_grpc_servicer.sglang.rust import resolve_servicer_impl

    from .config import complete

    if resolve_servicer_impl(args) != "rust":
        raise ValueError("managed EPD requires the Rust servicer")
    if args.encoder_only:
        return encoder_main(args)
    config = complete(args, args.disaggregation_mode)
    if config["role"] not in ("prefill", "decode"):
        raise ValueError("managed P/D requires native --disaggregation-mode prefill or decode")
    set_identity(config, config["role"])
    # The temporary directory is kept until the native ranks and Rust servicer have exited.
    with tempfile.TemporaryDirectory(prefix="smg-managed-") as directory:
        endpoint = f"ipc://{directory}/metadata.sock" if config["role"] == "prefill" else None
        config["metadata_endpoint"] = endpoint
        if endpoint:
            os.environ[EVENTS_ENV] = endpoint
        from .plugin import load_role_hooks

        load_role_hooks()
        raise SystemExit(asyncio.run(launcher(config).serve(args)))


def encoder_stop_handler(loop, server, stopped):
    """Wake an idle native control loop when a process signal requests draining."""

    def request_stop():
        """Close admission and wake the launcher so native cleanup remains reachable."""
        server.drain()
        stopped.set()

    def stop(sig, frame):
        """Schedule event-loop shutdown; the signal handler does not release native resources."""
        loop.call_soon_threadsafe(request_stop)

    return stop


async def wait_encoder_shutdown(server, stopped):
    """Observe the existing Rust server thread's terminal state as other servicers do."""
    while not stopped.is_set():
        error = server.last_error()
        if error or not server.running():
            raise RuntimeError(f"Rust encoder control server stopped: {error or 'server exited'}")
        try:
            await asyncio.wait_for(stopped.wait(), 1)
        except asyncio.TimeoutError:  # noqa: UP041 - Python 3.10
            pass


async def serve(config, args):
    """Python launches native TP/GPU work; Rust owns both public E control listeners."""
    from smg.smg_rs import ManagedEncoderProcessor, ManagedEncoderServer

    from smg_grpc_servicer.sglang.model_artifacts import encoder_artifact_dirs

    from .config import encoder_cpu_limit, encoder_staging_bytes, encoder_window
    from .encoder_bridge import launch_runtime
    from .encoder_native import serve_native

    model_dir, tokenizer_dir = encoder_artifact_dirs(args)
    backend = launch_runtime(args, config["max_active"])
    server = native_task = None
    stopped = asyncio.Event()
    previous = {}
    try:
        if not hasattr(backend.encoder.preprocessor, "max_artifact_bytes"):
            raise RuntimeError("native rank zero did not install the managed Rust bridge")
        config["max_staging_bytes_per_request"] = encoder_staging_bytes(
            backend.encoder.model_config.hidden_size, config["embedding_dtype"]
        )
        config["max_cpu"] = encoder_cpu_limit(config, backend.scheduler.max_batch_size)
        config["execution_capacity"], _ = encoder_window(config)
        config["processor_options"]["model_dir"] = model_dir
        config["processor_options"]["max_inflight"] = config["max_cpu"]
        config["processor_options"]["encoder_dtype"] = str(
            backend.encoder.model.visual.dtype
        ).removeprefix("torch.")
        config.update(
            model_path=args.model_path,
            tokenizer_path=tokenizer_dir,
            served_model_name=args.served_model_name or args.model_path,
            model_type=backend.encoder.model_config.hf_config.model_type,
            http_listen=f"{args.host}:{args.port}",
        )
        processor = ManagedEncoderProcessor(
            config["processor_options"],
            tokenizer_dir,
            config["max_artifact_bytes"],
        )
        with tempfile.TemporaryDirectory(prefix="smg-encoder-") as directory:
            config["native_commands"] = f"ipc://{directory}/commands.sock"
            config["native_events"] = f"ipc://{directory}/events.sock"
            server = ManagedEncoderServer(json.dumps(config), processor)
            stop_task = None
            try:
                backend.start()
                native_task = asyncio.create_task(
                    serve_native(
                        config, backend, config["native_commands"], config["native_events"]
                    )
                )

                stop = encoder_stop_handler(asyncio.get_running_loop(), server, stopped)

                previous = {
                    sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)
                }
                stop_task = asyncio.create_task(wait_encoder_shutdown(server, stopped))
                done, _ = await asyncio.wait(
                    (stop_task, native_task), return_when=asyncio.FIRST_COMPLETED
                )
                if stop_task in done:
                    await stop_task
                if native_task in done:
                    await native_task
            finally:
                if stop_task is not None:
                    stop_task.cancel()
                    await asyncio.gather(stop_task, return_exceptions=True)
                server.drain()
                deadline = asyncio.get_running_loop().time() + config["shutdown_timeout_seconds"]
                while server.active():
                    if asyncio.get_running_loop().time() >= deadline:
                        await stop_followers(backend)
                        os._exit(1)
                    await asyncio.sleep(0.05)
                await asyncio.to_thread(server.stop, 5)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if native_task is not None:
            native_task.cancel()
            await asyncio.gather(native_task, return_exceptions=True)
        try:
            await backend.stop()
        finally:
            await stop_followers(backend)


async def stop_followers(backend):
    """Stop only this launcher's native TP child processes with bounded joins."""
    for process in backend.tp_processes:
        if process.is_alive():
            process.terminate()
    for process in backend.tp_processes:
        await asyncio.to_thread(process.join, 1)
        if process.is_alive():
            process.kill()
            await asyncio.to_thread(process.join, 1)


def encoder_main(args):
    """Require finite deployment limits and assign a fresh instance generation at startup."""
    from smg_grpc_servicer.sglang.mm_defaults import configure_threads

    from .config import complete, encoder_window

    # Initialize before native TP children or Rust preprocessing thread pools.
    configure_threads()
    config = complete(args, "encoder")
    set_identity(config, "encoder")
    os.environ["SMG_SGLANG_MANAGED_PROCESSOR_CONTRACT"] = config["processor_contract"]
    native_window, _ = encoder_window(config)
    os.environ["SMG_SGLANG_MANAGED_BATCH_BYTES"] = str(config["max_artifact_bytes"] * native_window)
    from .plugin import load_role_hooks

    load_role_hooks()
    if not args.encoder_only or args.language_only:
        raise ValueError("managed E requires native --encoder-only")
    if args.encoder_transfer_backend not in ("mooncake", "zmq_to_scheduler"):
        raise ValueError("managed E supports only Mooncake or scheduler ZMQ transfers")
    asyncio.run(serve(config, args))


if __name__ == "__main__":
    main()
