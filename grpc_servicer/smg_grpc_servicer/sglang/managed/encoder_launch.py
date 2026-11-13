"""Explicit SMG encoder control service with the native receiver HTTP facade."""

import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path

from .plugin import INSTANCE_ENV, ROLE_ENV, WORKER_ENV


async def serve(config, args):
    """Bind bounded control listeners only after native TP ranks initialize successfully."""
    import grpc
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from smg.smg_rs import ManagedEncoderProcessor
    from smg_grpc_proto.generated import sglang_managed_pb2_grpc as rpc

    from .encoder_runtime import launch_runtime
    from .encoder_service import EncoderService, Rejected

    backend = launch_runtime(args, config["max_active"])
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
                {"error": "unsupported managed receiver operation"}, status_code=404
            )
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > config["max_control_bytes"]:
                return JSONResponse({"error": "control request too large"}, status_code=413)
        try:
            payload = json.loads(data)
            if not isinstance(payload, dict):
                raise ValueError("receiver control must be an object")
            result = await service.facade(kinds[route], payload)
            return result or {"status": "ok"}
        except Rejected as error:
            status = 429 if error.code == grpc.StatusCode.RESOURCE_EXHAUSTED else 400
            return JSONResponse({"error": str(error)}, status_code=status)
        except (ValueError, TimeoutError) as error:
            return JSONResponse({"error": str(error)}, status_code=400)

    server = grpc.aio.server(
        options=[("grpc.max_receive_message_length", config["max_control_bytes"])]
    )
    rpc.add_ManagedEncoderServicer_to_server(service, server)
    if not server.add_insecure_port(config["control_listen"]):
        raise RuntimeError("managed encoder control port could not be bound")
    service.start()
    await server.start()
    http = uvicorn.Server(
        uvicorn.Config(
            app, host=args.host, port=args.port, limit_concurrency=config["max_http_connections"]
        )
    )
    try:
        await http.serve()
    finally:
        # Drain real owners before stopping a scheduler that may own a TP collective.
        service.draining = True
        for entry in service.entries.values():
            service._cancel(entry)
        await server.stop(0)
        while any(not entry.terminal for entry in service.entries.values()):
            await asyncio.sleep(0.05)
        service.dispatcher.cancel()
        await asyncio.gather(service.dispatcher, return_exceptions=True)
        await backend.stop()


def main():
    """Require finite deployment limits and assign a fresh instance generation at startup."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--managed-config", required=True)
    managed, native = parser.parse_known_args()
    config = json.loads(Path(managed.managed_config).read_text())
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
    config["generation"] = uuid.uuid4().hex
    os.environ[ROLE_ENV] = "encoder"
    os.environ[WORKER_ENV] = config["worker_id"]
    os.environ[INSTANCE_ENV] = config["generation"]
    os.environ["SMG_SGLANG_MANAGED_PROCESSOR_CONTRACT"] = config["processor_contract"]
    os.environ["SMG_SGLANG_MANAGED_BATCH_BYTES"] = str(
        config["max_artifact_bytes"] * config["max_active"]
    )
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args

    load_plugins()
    args = prepare_server_args(native)
    if not args.encoder_only or args.language_only:
        raise ValueError("managed E requires native --encoder-only")
    if args.tp_size != config["tp_size"] or args.dtype != config["embedding_dtype"]:
        raise ValueError("native E TP size/dtype does not match the advertised managed contract")
    if args.encoder_transfer_backend not in ("mooncake", "zmq_to_scheduler"):
        raise ValueError("managed E supports only Mooncake or scheduler ZMQ transfers")
    asyncio.run(serve(config, args))


if __name__ == "__main__":
    main()
