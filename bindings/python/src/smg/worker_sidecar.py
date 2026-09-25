"""Standalone two-tier SMG Worker sidecar.

One process per colocated vLLM or TokenSpeed engine. It runs the Rust
``WorkerControlServer``, which serves ``smg.worker.v1`` WorkerControl and
WorkerInference to the fleet Router and reaches the engine over the engine's
own gRPC endpoint or msgpack ZMQ IPC. ``smg serve --router-worker-mode smg``
launches one next to each engine; it also runs by hand as
``python -m smg.worker_sidecar``.

Exit paths: SIGTERM/SIGINT announces DRAINING, waits ``--drain-secs`` for
active streams, announces NOT_SERVING, stops the listener and exits 0. An
engine-transport failure reported by the Rust side (``last_error``) announces
NOT_SERVING, stops the listener and exits 1.
"""

from __future__ import annotations

import argparse
import logging
import signal
import socket
import threading
import time

from smg.worker import WorkerControlServer, init_tracing

logger = logging.getLogger("smg.worker_sidecar")

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# Python level name -> Rust tracing level.
_TRACING_LEVELS = {"debug": "debug", "info": "info", "warning": "warn", "error": "error"}
# Interval between `last_error` polls while waiting for a shutdown signal.
_HEALTH_POLL_SECS = 1.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m smg.worker_sidecar",
        description="Run the Rust SMG Worker next to a colocated inference engine",
    )
    parser.add_argument(
        "--bind-address",
        required=True,
        help="host:port the WorkerControl/WorkerInference gRPC listener binds",
    )
    parser.add_argument(
        "--worker-id",
        required=True,
        help="Identity this Worker reports to the Router",
    )
    parser.add_argument(
        "--engine-type",
        choices=["vllm", "tokenspeed"],
        required=True,
        help="Engine runtime behind this Worker",
    )
    parser.add_argument(
        "--engine-transport",
        choices=["grpc", "zmq"],
        default="grpc",
        help="How the Worker reaches the engine: its gRPC server or msgpack ZMQ IPC (default: grpc)",
    )
    parser.add_argument(
        "--engine-endpoint",
        required=True,
        help="Engine address: grpc://host:port for grpc, the ipc:// socket URL for zmq",
    )
    parser.add_argument(
        "--zmq-handshake-address",
        help=(
            "tcp:// address the Worker binds for the ZMQ engine handshake "
            "(default: derived from the ipc:// engine endpoint)"
        ),
    )
    parser.add_argument(
        "--engine-count",
        type=int,
        default=1,
        help="Engines sharing one ZMQ socket set, i.e. the engine-level data-parallel size (default: 1)",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        dest="model_ids",
        required=True,
        help="Model this Worker advertises; repeat for several. The first is also the engine model path",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=0,
        help="Admission bound across the whole engine group; 0 leaves the Worker unbounded (default: 0)",
    )
    parser.add_argument(
        "--drain-secs",
        type=float,
        default=5.0,
        help="Seconds active streams may finish after SIGTERM before the listener stops (default: 5)",
    )
    parser.add_argument(
        "--log-level",
        choices=sorted(_TRACING_LEVELS),
        default="info",
        help="Log level for the sidecar and the Rust Worker (default: info)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format=_LOG_FORMAT)
    init_tracing(_TRACING_LEVELS[args.log_level])
    if args.drain_secs < 0:
        raise ValueError("--drain-secs must be non-negative")
    if args.engine_count <= 0:
        raise ValueError("--engine-count must be positive")
    if args.max_concurrent_requests < 0:
        raise ValueError("--max-concurrent-requests must be non-negative")
    stopped = threading.Event()
    logger.info(
        "Starting SMG Worker %s: control listener %s, %s engine over %s at %s "
        "(%d engine(s), admission bound %d, models %s)",
        args.worker_id,
        args.bind_address,
        args.engine_type,
        args.engine_transport,
        args.engine_endpoint,
        args.engine_count,
        args.max_concurrent_requests,
        ",".join(args.model_ids),
    )
    # `token_only_wire` is not listed here: WorkerControlServer derives it from
    # engine_transport, so every entry point advertises it consistently.
    features = ["generate", "stream", "abort"]
    server = WorkerControlServer(
        bind_address=args.bind_address,
        worker_id=args.worker_id,
        engine_type=args.engine_type,
        hostname=socket.gethostname(),
        engine_endpoint=args.engine_endpoint,
        model_ids=args.model_ids,
        features=features,
        max_concurrent_requests=args.max_concurrent_requests,
        inference_enabled=True,
        engine_attributes={
            "model_path": args.model_ids[0],
            "tokenizer_path": args.model_ids[0],
        },
        engine_transport=args.engine_transport,
        zmq_handshake_address=args.zmq_handshake_address,
        engine_count=args.engine_count,
    )
    # The listener is up. Health reports STARTING until the Rust side has
    # connected the engine transport, then the SERVING announced here.
    server.set_health("serving", "ready")
    logger.info(
        "SMG Worker %s listening on %s; SERVING once the engine transport connects",
        args.worker_id,
        args.bind_address,
    )

    def stop(signum: int, _frame: object) -> None:
        if stopped.is_set():
            return
        logger.info(
            "SMG Worker %s received %s; draining for %.1fs",
            args.worker_id,
            signal.Signals(signum).name,
            args.drain_secs,
        )
        server.set_health("draining", "draining")
        stopped.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    # An engine-transport failure lands in `last_error` and nothing in this
    # process can recover it.
    while not stopped.wait(_HEALTH_POLL_SECS):
        error = server.last_error
        if error:
            logger.error("SMG Worker %s cannot serve: %s", args.worker_id, error)
            server.set_health("not_serving", error)
            server.stop(1.0)
            raise SystemExit(1)
    time.sleep(args.drain_secs)
    server.set_health("not_serving", "stopped")
    server.stop(max(1.0, args.drain_secs))
    logger.info("SMG Worker %s stopped after draining", args.worker_id)


if __name__ == "__main__":
    main()
