"""Explicit SMG encoder control service with the native receiver HTTP facade."""

import asyncio
import json
import logging
import os
import signal
import sys
import uuid
from contextlib import contextmanager

from .plugin import INSTANCE_ENV, ROLE_ENV, WORKER_ENV


@contextmanager
def shutdown_signals(server):
    """Let the owner drain before exiting; Uvicorn otherwise re-raises SIGTERM first."""
    previous = {
        sig: signal.signal(sig, server.handle_exit) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def receiver_error(error):
    """Preserve the native receiver's message field and timeout/admission status semantics."""
    import grpc

    from .encoder_service import Rejected

    if isinstance(error, TimeoutError):
        status = 504
    elif isinstance(error, Rejected):
        status = {
            grpc.StatusCode.RESOURCE_EXHAUSTED: 429,
            grpc.StatusCode.DEADLINE_EXCEEDED: 504,
        }.get(error.code, 400)
    else:
        status = 400
    return {"message": str(error) or "native receiver timeout"}, status


async def serve(config, args):
    """Bind bounded control listeners only after native TP ranks initialize successfully."""
    import grpc
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from smg.smg_rs import ManagedEncoderProcessor
    from smg_grpc_proto.generated import sglang_managed_pb2_grpc as rpc

    from .config import encoder_cpu_limit, encoder_staging_bytes
    from .encoder_bridge import launch_runtime
    from .encoder_service import EncoderService, Rejected

    backend = launch_runtime(args, config["max_active"])
    service = server = None
    try:
        # These are ceilings, not a fixed running concurrency. The registry starts
        # small and grows only when CPU supply leaves the native scheduler idle.
        config["max_staging_bytes_per_request"] = encoder_staging_bytes(
            backend.encoder.model_config.hidden_size, config["embedding_dtype"]
        )
        config["max_cpu"] = encoder_cpu_limit(config, backend.scheduler.max_batch_size)
        config["processor_options"]["max_inflight"] = config["max_cpu"]
        processor = ManagedEncoderProcessor(
            config["processor_options"],
            args.tokenizer_path or args.model_path,
            config["max_artifact_bytes"],
        )
        service = EncoderService(config, backend, processor)
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        @app.post("/{route}")
        async def receive(route: str, request: Request):
            """Expose only receiver controls; every call resolves an admitted managed operation."""
            kinds = {
                "scheduler_receive_meta_data": "metadata",
                "scheduler_receive_url": "register",
                "send": "send",
            }
            if route not in kinds:
                return JSONResponse(
                    {"message": "unsupported managed receiver operation"}, status_code=404
                )
            data = bytearray()
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data) > config["max_control_bytes"]:
                    return JSONResponse({"message": "control request too large"}, status_code=413)
            try:
                payload = json.loads(data)
                if not isinstance(payload, dict):
                    raise ValueError("receiver control must be an object")
                result = await service.facade(kinds[route], payload)
                return result or {"status": "ok"}
            except (Rejected, ValueError, TimeoutError) as error:
                body, status = receiver_error(error)
                return JSONResponse(body, status_code=status)

        server = grpc.aio.server(
            options=[("grpc.max_receive_message_length", config["max_control_bytes"])]
        )
        rpc.add_ManagedEncoderServicer_to_server(service, server)
        if not server.add_insecure_port(config["control_listen"]):
            raise RuntimeError("managed encoder control port could not be bound")
        service.start()
        await server.start()

        class ManagedHttpServer(uvicorn.Server):
            """Keep process termination behind the managed native-owner drain."""

            def handle_exit(self, sig, frame):
                """Close admission immediately, before HTTP observers finish their grace period."""
                service.draining = True
                super().handle_exit(sig, frame)

            def capture_signals(self):
                """Consume shutdown signals without Uvicorn re-raising them before cleanup."""
                return shutdown_signals(self)

        http = ManagedHttpServer(
            uvicorn.Config(
                app,
                host=args.host,
                port=args.port,
                limit_concurrency=config["max_http_connections"],
                # HTTP observers cannot indefinitely postpone the native-owner drain.
                timeout_graceful_shutdown=config["shutdown_timeout_seconds"],
            )
        )
        await http.serve()
    finally:
        await shutdown(service, server, backend, config.get("shutdown_timeout_seconds", 30))


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


async def shutdown(service, server, backend, timeout):
    """Drain owned work, or terminate the entire E process without false quiescence receipts."""
    if service is not None:
        service.draining = True
        for entry in service.entries.values():
            service._cancel(entry)

    async def drain():
        if service is not None:
            while any(not entry.terminal for entry in service.entries.values()):
                await asyncio.sleep(0.05)

    try:
        await asyncio.wait_for(drain(), timeout)
    except TimeoutError:
        logging.getLogger(__name__).error(
            "managed E drain expired; terminating owned native processes"
        )
        if server is not None:
            await server.stop(0)
        await stop_followers(backend)
        # Rank zero and its CUDA/send threads live here. Do not unwind Python
        # buffers under those threads or announce safe release while they run.
        os._exit(1)
    # Keep reconciliation reachable while native owners drain, then give terminal
    # events a bounded opportunity to reach their existing observers.
    if server is not None:
        await server.stop(1)
    if service is not None and service.dispatcher is not None:
        service.dispatcher.cancel()
        await asyncio.gather(service.dispatcher, return_exceptions=True)
    try:
        await backend.stop()
    finally:
        await stop_followers(backend)


def main():
    """Require finite deployment limits and assign a fresh instance generation at startup."""
    from sglang.srt.server_args import prepare_server_args

    from .config import complete, encoder_window

    args = prepare_server_args(sys.argv[1:])
    config = complete({}, args, "encoder")
    positive = (
        "max_records",
        "max_active",
        "max_work",
        "max_context_bytes",
        "retention_ms",
        "max_duration_ms",
        "max_clock_error_ms",
        "max_cpu",
        "max_artifact_bytes",
        "max_staging_bytes_per_request",
        "max_receivers",
        "max_control_bytes",
        "max_images",
        "drr_quantum",
        "aging_seconds",
        "max_early_waiters",
        "early_wait_seconds",
        "load_interval_ms",
        "max_http_connections",
    )
    if any(
        type(config.get(key)) not in (int, float) or not 0 < config[key] < 2**53 for key in positive
    ):
        raise ValueError("managed encoder requires finite positive resource and deadline limits")
    if any(
        type(config[key]) is not int
        for key in positive
        if key not in ("aging_seconds", "early_wait_seconds")
    ):
        raise ValueError(
            "managed encoder resource counts, bytes and millisecond limits must be integers"
        )
    if config["max_images"] > 32 or config["max_active"] > config["max_records"]:
        raise ValueError("invalid managed encoder admission bounds")
    if any(type(weight) is not int or weight <= 0 for weight in config["tenant_weights"].values()):
        raise ValueError("DRR weights must be positive integers")
    shutdown_timeout = config.get("shutdown_timeout_seconds", 30)
    if type(shutdown_timeout) not in (int, float) or not 0 < shutdown_timeout <= 300:
        raise ValueError("managed shutdown timeout must be in (0, 300] seconds")
    config["generation"] = uuid.uuid4().hex
    os.environ[ROLE_ENV] = "encoder"
    os.environ[WORKER_ENV] = config["worker_id"]
    os.environ[INSTANCE_ENV] = config["generation"]
    os.environ["SMG_SGLANG_MANAGED_PROCESSOR_CONTRACT"] = config["processor_contract"]
    native_window, _ = encoder_window(config)
    os.environ["SMG_SGLANG_MANAGED_BATCH_BYTES"] = str(config["max_artifact_bytes"] * native_window)
    from sglang.srt.plugins import load_plugins

    load_plugins()
    if not args.encoder_only or args.language_only:
        raise ValueError("managed E requires native --encoder-only")
    if args.encoder_transfer_backend not in ("mooncake", "zmq_to_scheduler"):
        raise ValueError("managed E supports only Mooncake or scheduler ZMQ transfers")
    asyncio.run(serve(config, args))


if __name__ == "__main__":
    main()
