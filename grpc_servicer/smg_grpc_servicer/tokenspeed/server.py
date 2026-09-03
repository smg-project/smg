"""Standalone TokenSpeed gRPC server."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading
import time
from concurrent import futures

import grpc
from grpc_health.v1 import health_pb2_grpc
from grpc_reflection.v1alpha import reflection
from smg_grpc_proto import tokenspeed_scheduler_pb2_grpc
from smg_grpc_proto.generated import tokenspeed_scheduler_pb2
from tokenspeed.runtime.utils.server_args import ServerArgs

from smg_grpc_servicer.tokenspeed.health_servicer import TokenSpeedHealthServicer
from smg_grpc_servicer.tokenspeed.scheduler_launcher import launch_engine
from smg_grpc_servicer.tokenspeed.servicer import TokenSpeedSchedulerServicer
from smg_grpc_servicer.worker_control_lifecycle import WorkerControlLifecycle

logger = logging.getLogger(__name__)

# Poll interval for the WorkerControl liveness watcher. Coarse on purpose:
# the Router's own health probing already covers fast detection, this only
# needs to stop the engine from outliving its control plane.
_WORKER_CONTROL_WATCH_INTERVAL_SECS = 1.0

# Match the other SMG servicers' 256 MiB default — a single oversized or
# malformed gRPC frame can otherwise trigger a multi-GiB transient
# allocation. VLM deployments that genuinely need bigger ``pixel_values``
# raise it via ``TOKENSPEED_GRPC_MAX_MESSAGE_BYTES``; the env value is
# clamped to gRPC's hard ceiling (``INT32_MAX`` = 2 GiB - 1).
_GRPC_DEFAULT_MAX_BYTES = 256 * 1024 * 1024
_GRPC_HARD_CEILING_BYTES = (1 << 31) - 1


def _grpc_max_message_bytes() -> int:
    """Return the configured gRPC message ceiling (send + receive use the same)."""
    raw = os.getenv("TOKENSPEED_GRPC_MAX_MESSAGE_BYTES")
    if not raw:
        return _GRPC_DEFAULT_MAX_BYTES
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "TOKENSPEED_GRPC_MAX_MESSAGE_BYTES=%r is not an int; falling back to %d",
            raw,
            _GRPC_DEFAULT_MAX_BYTES,
        )
        return _GRPC_DEFAULT_MAX_BYTES
    if value <= 0:
        logger.warning(
            "TOKENSPEED_GRPC_MAX_MESSAGE_BYTES=%d must be positive; falling back to %d",
            value,
            _GRPC_DEFAULT_MAX_BYTES,
        )
        return _GRPC_DEFAULT_MAX_BYTES
    if value > _GRPC_HARD_CEILING_BYTES:
        logger.warning(
            "TOKENSPEED_GRPC_MAX_MESSAGE_BYTES=%d exceeds gRPC ceiling %d; clamping",
            value,
            _GRPC_HARD_CEILING_BYTES,
        )
        return _GRPC_HARD_CEILING_BYTES
    return value


def _grpc_server_options(max_message_bytes: int) -> list[tuple[str, int]]:
    """Build gRPC server options for long-idle TokenSpeed generation streams."""
    return [
        ("grpc.max_send_message_length", max_message_bytes),
        ("grpc.max_receive_message_length", max_message_bytes),
        # Long EPD requests can spend minutes without sending DATA frames while
        # the Rust client still sends HTTP/2 keepalive pings.
        ("grpc.http2.min_ping_interval_without_data_ms", 10000),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.http2.max_ping_strikes", 0),
        ("grpc.keepalive_permit_without_calls", True),
    ]


class TokenSpeedEncodeDiscoveryServicer(TokenSpeedSchedulerServicer):
    """Scheduler discovery surface for encode-only workers."""

    async def Generate(self, request, context):
        await context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            "Generate is unavailable on TokenSpeed encode workers",
        )
        yield tokenspeed_scheduler_pb2.GenerateResponse()


async def serve_grpc(server_args: ServerArgs) -> None:
    """Run the TokenSpeed gRPC server until a shutdown signal is received."""

    logger.info("Launching TokenSpeed scheduler + AsyncLLM...")
    async_llm, scheduler_info = launch_engine(server_args)

    max_message_bytes = _grpc_max_message_bytes()
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=_grpc_server_options(max_message_bytes),
    )

    health_servicer = TokenSpeedHealthServicer(
        async_llm=async_llm,
        scheduler_info=scheduler_info,
    )
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    discovery_servicer = None

    if server_args.disaggregation_mode == "encode":
        # EPD encode worker: serve the vision-only encode loop via the encoder
        # service. ALSO mount a discovery-only scheduler surface so the
        # gateway's generic worker discovery (HealthCheck + GetModelInfo +
        # GetServerInfo, all over the TokenSpeedScheduler stub) can reach this
        # worker and register it.
        from smg_grpc_proto.generated import (
            tokenspeed_encoder_pb2,
            tokenspeed_encoder_pb2_grpc,
        )

        from smg_grpc_servicer.tokenspeed.encoder_servicer import (
            TokenSpeedEncoderServicer,
        )

        servicer = TokenSpeedEncoderServicer(
            async_llm=async_llm,
            server_args=server_args,
            scheduler_info=scheduler_info,
            health_servicer=health_servicer,
        )
        tokenspeed_encoder_pb2_grpc.add_TokenSpeedEncoderServicer_to_server(servicer, server)

        discovery_servicer = TokenSpeedEncodeDiscoveryServicer(
            async_llm=async_llm,
            server_args=server_args,
            scheduler_info=scheduler_info,
            health_servicer=health_servicer,
        )
        tokenspeed_scheduler_pb2_grpc.add_TokenSpeedSchedulerServicer_to_server(
            discovery_servicer, server
        )

        primary_service = tokenspeed_encoder_pb2.DESCRIPTOR.services_by_name[
            "TokenSpeedEncoder"
        ].full_name
    else:
        servicer = TokenSpeedSchedulerServicer(
            async_llm=async_llm,
            server_args=server_args,
            scheduler_info=scheduler_info,
            health_servicer=health_servicer,
        )
        tokenspeed_scheduler_pb2_grpc.add_TokenSpeedSchedulerServicer_to_server(servicer, server)
        primary_service = tokenspeed_scheduler_pb2.DESCRIPTOR.services_by_name[
            "TokenSpeedScheduler"
        ].full_name

    service_names = (
        primary_service,
        "grpc.health.v1.Health",
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    listen_addr = f"{server_args.host}:{server_args.port}"
    server.add_insecure_port(listen_addr)
    logger.info("TokenSpeed gRPC server listening on %s", listen_addr)

    await server.start()

    advertised_model = (
        getattr(server_args, "served_model_name", None)
        or getattr(server_args, "model", None)
        or getattr(server_args, "model_path", "")
    )
    model_ids = (
        list(advertised_model)
        if isinstance(advertised_model, (list, tuple))
        else [advertised_model]
    )
    try:
        worker_control = WorkerControlLifecycle.start_from_env(
            engine_type="tokenspeed",
            engine_distribution="tokenspeed",
            model_ids=model_ids,
            features=[
                "encode" if server_args.disaggregation_mode == "encode" else "generate",
                "abort",
            ],
            max_concurrent_requests=(
                getattr(server_args, "max_running_requests", 0)
                or getattr(server_args, "max_num_seqs", 0)
                or 0
            ),
            engine_attributes={
                "model_path": model_ids[0],
                "tokenizer_path": (
                    getattr(server_args, "tokenizer", None)
                    or getattr(server_args, "tokenizer_path", None)
                    or model_ids[0]
                ),
            },
        )
    except Exception:
        logger.exception("Failed to start the Worker control plane")
        health_servicer.set_not_serving()
        await server.stop(0)
        await servicer.shutdown()
        raise

    # Warmup on a background thread so the async server can handle the probe.
    # `shutdown_started` is how the two race: a signal arriving mid-warmup marks
    # the worker draining and NOT_SERVING, and without this the warmup would
    # happily flip both back to serving afterwards, pulling new requests into a
    # server that is already stopping.
    shutdown_started = threading.Event()
    warmup_thread = threading.Thread(
        target=_wait_and_warmup,
        args=(server_args, health_servicer, worker_control, shutdown_started),
        daemon=True,
    )
    warmup_thread.start()

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Received shutdown signal")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows and some exotic envs don't support loop.add_signal_handler.
            pass

    async def _watch_worker_control(lifecycle: WorkerControlLifecycle) -> None:
        """Treat a dead control plane as a shutdown signal.

        The Router reaches this worker through WorkerControl: discovery, health
        probes and drain all ride it. If that server exits, the gRPC port keeps
        accepting traffic the Router can no longer see or drain, so the useful
        failure is to go down with it rather than linger as an unreachable
        endpoint.
        """
        while not stop_event.is_set():
            await asyncio.sleep(_WORKER_CONTROL_WATCH_INTERVAL_SECS)
            if not lifecycle.running:
                logger.error(
                    "Worker control plane exited (%s) — shutting down the TokenSpeed gRPC server",
                    lifecycle.last_error or "no error reported",
                )
                stop_event.set()
                return

    control_watcher = (
        asyncio.create_task(_watch_worker_control(worker_control))
        if worker_control is not None
        else None
    )

    try:
        await stop_event.wait()
    finally:
        logger.info("Shutting down TokenSpeed gRPC server")
        shutdown_started.set()
        if control_watcher is not None:
            control_watcher.cancel()
        if worker_control is not None:
            try:
                worker_control.mark_draining()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to mark the Worker control plane as draining")
        health_servicer.set_not_serving()
        try:
            await servicer.shutdown()
        except Exception:  # noqa: BLE001
            logger.exception("servicer.shutdown() raised")
        if discovery_servicer is not None:
            try:
                await discovery_servicer.shutdown()
            except Exception:  # noqa: BLE001
                logger.exception("discovery_servicer.shutdown() raised")
        await server.stop(5.0)
        if worker_control is not None:
            try:
                worker_control.mark_not_serving()
                await asyncio.to_thread(worker_control.stop, 5.0)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to stop the Worker control plane cleanly")
        if warmup_thread.is_alive():
            warmup_thread.join(timeout=5.0)


def _wait_and_warmup(
    server_args: ServerArgs,
    health_servicer: TokenSpeedHealthServicer,
    worker_control: WorkerControlLifecycle | None = None,
    shutdown_started: threading.Event | None = None,
) -> None:
    """Probe the gRPC server until it can generate one token, then set SERVING.

    Hits the external port so the warmup exercises transport, proto codec,
    and scheduler IPC end-to-end.
    """

    def _mark_serving() -> None:
        if shutdown_started is not None and shutdown_started.is_set():
            logger.info("Shutdown started during warmup — leaving the worker NOT_SERVING")
            return
        health_servicer.set_serving()
        if worker_control is not None:
            try:
                worker_control.mark_serving()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to mark the Worker control plane as serving")

    if os.getenv("TOKENSPEED_SKIP_GRPC_WARMUP", "0").lower() in ("1", "true", "yes"):
        logger.info("TOKENSPEED_SKIP_GRPC_WARMUP=1 — skipping warmup")
        _mark_serving()
        return

    if server_args.disaggregation_mode == "encode":
        # Encode workers run the vision tower only — no LM. The warmup's
        # stub.Generate would route through the scheduler Generate path and drive
        # the LM, SIGUSR1-killing the encode TP group. Skip it for this role.
        logger.info("encode role — skipping Generate warmup (no LM)")
        _mark_serving()
        return

    # Wildcard bind hosts aren't routable as destinations; dial loopback instead.
    warmup_host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(server_args.host, server_args.host)
    grpc_url = f"{warmup_host}:{server_args.port}"
    max_message_bytes = _grpc_max_message_bytes()
    channel = grpc.insecure_channel(
        grpc_url,
        options=[
            ("grpc.max_send_message_length", max_message_bytes),
            ("grpc.max_receive_message_length", max_message_bytes),
        ],
    )
    stub = tokenspeed_scheduler_pb2_grpc.TokenSpeedSchedulerStub(channel)

    # GetModelInfo is the quickest confirmation the server is bound + the
    # engine is alive.
    deadline = time.time() + 180
    connected = False
    while time.time() < deadline:
        try:
            stub.GetModelInfo(
                tokenspeed_scheduler_pb2.GetModelInfoRequest(),
                timeout=5,
            )
            connected = True
            break
        except Exception as e:  # noqa: BLE001
            logger.debug("Warmup: GetModelInfo not ready yet: %s", e)
            time.sleep(1)

    if not connected:
        logger.error("TokenSpeed gRPC warmup failed: GetModelInfo never succeeded")
        channel.close()
        return

    # Generative only — warmup is a 1-token generate.
    warmup_ok = False
    try:
        warmup = tokenspeed_scheduler_pb2.GenerateRequest(
            request_id=f"WARMUP_{time.time()}",
            tokenized=tokenspeed_scheduler_pb2.TokenizedInput(
                input_ids=[0],
                original_text="warmup",
            ),
            sampling_params=tokenspeed_scheduler_pb2.SamplingParams(
                temperature=0.0,
                max_new_tokens=1,
            ),
            stream=False,
        )
        final = None
        for resp in stub.Generate(warmup, timeout=600):
            final = resp
        if final is None or not final.HasField("complete"):
            logger.warning(
                "Warmup Generate returned no Complete frame (last=%r)",
                final,
            )
        else:
            logger.info("Warmup generation succeeded")
            warmup_ok = True
    except Exception as e:  # noqa: BLE001
        logger.warning("TokenSpeed warmup failed: %s", e)
    finally:
        channel.close()

    if warmup_ok:
        _mark_serving()
        logger.info("TokenSpeed gRPC server is ready to serve")
    else:
        # Stays NOT_SERVING so K8s readiness keeps this worker out of rotation.
        logger.error("TokenSpeed gRPC warmup did not produce a complete frame")
