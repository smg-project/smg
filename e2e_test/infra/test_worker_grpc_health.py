"""Unit tests for what the harness takes as a gRPC worker's health answer.

A worker's port is chosen before the engine binds it, so what answers there
may be another process. The checks run real listeners on the loopback: a
gRPC server with the ``grpc.health.v1`` service, one without any service, and
an HTTP/2 server that answers every stream with ``404`` (the shape grpc reads
as UNIMPLEMENTED), the stranger that once passed for a decode worker.
"""

from __future__ import annotations

import os
import socket
import struct
import sys
import threading
import time
from collections.abc import Iterator
from concurrent import futures
from contextlib import contextmanager

import pytest
from infra.constants import DEFAULT_HOST, ENV_SHOW_WORKER_LOGS, ConnectionMode, host_port
from infra.worker import ENGINES_WITHOUT_GRPC_HEALTH, Worker

pytest.importorskip("grpc")
pytest.importorskip("grpc_health")

_FAMILY = socket.AF_INET6 if ":" in DEFAULT_HOST else socket.AF_INET
_HTTP2_PREFACE = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"
_SETTINGS, _HEADERS = 0x4, 0x1
_ACK, _END_STREAM, _END_HEADERS = 0x1, 0x1, 0x4
# HPACK indexed header field: static table entry 13 is ``:status: 404``.
_STATUS_404 = b"\x8d"


def _frame(kind: int, flags: int, stream_id: int, payload: bytes = b"") -> bytes:
    return (
        struct.pack("!I", len(payload))[1:]
        + bytes([kind, flags])
        + struct.pack("!I", stream_id & 0x7FFFFFFF)
        + payload
    )


def _recv_exactly(conn: socket.socket, buffer: bytearray, size: int) -> bytes:
    while len(buffer) < size:
        chunk = conn.recv(4096)
        if not chunk:
            raise ConnectionError("peer closed")
        buffer += chunk
    taken = bytes(buffer[:size])
    del buffer[:size]
    return taken


def _answer_404(conn: socket.socket) -> None:
    """Speak enough HTTP/2 to answer every request stream with a 404."""
    buffer = bytearray()
    if _recv_exactly(conn, buffer, len(_HTTP2_PREFACE)) != _HTTP2_PREFACE:
        return
    conn.sendall(_frame(_SETTINGS, 0, 0))
    while True:
        head = _recv_exactly(conn, buffer, 9)
        length = struct.unpack("!I", b"\x00" + head[:3])[0]
        kind, flags = head[3], head[4]
        stream_id = struct.unpack("!I", head[5:9])[0] & 0x7FFFFFFF
        _recv_exactly(conn, buffer, length)
        if kind == _SETTINGS and not flags & _ACK:
            conn.sendall(_frame(_SETTINGS, _ACK, 0))
        elif kind == _HEADERS:
            conn.sendall(_frame(_HEADERS, _END_HEADERS | _END_STREAM, stream_id, _STATUS_404))


def _serve_404(listener: socket.socket) -> None:
    while True:
        try:
            conn, _ = listener.accept()
        except OSError:
            return  # the listener is closed: the test is over
        with conn:
            conn.settimeout(5)
            try:
                _answer_404(conn)
            except (OSError, ConnectionError):
                continue


@contextmanager
def http2_404_listener() -> Iterator[int]:
    """An HTTP/2 server on a loopback port that answers every stream ``404``."""
    listener = socket.socket(_FAMILY, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((DEFAULT_HOST, 0))
    listener.listen(8)
    thread = threading.Thread(target=_serve_404, args=(listener,), daemon=True)
    thread.start()
    try:
        yield listener.getsockname()[1]
    finally:
        listener.close()
        thread.join(timeout=10)


@contextmanager
def grpc_listener(health: str | None) -> Iterator[int]:
    """A gRPC server on a loopback port: with the health service answering
    ``health`` (``"SERVING"``, ``"NOT_SERVING"``), or without any service at
    all when it is ``None``."""
    import grpc
    from grpc_health.v1 import health_pb2, health_pb2_grpc

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    if health is not None:
        status = getattr(health_pb2.HealthCheckResponse, health)

        class Health(health_pb2_grpc.HealthServicer):
            def Check(self, request, context):
                return health_pb2.HealthCheckResponse(status=status)

        health_pb2_grpc.add_HealthServicer_to_server(Health(), server)
    port = server.add_insecure_port(host_port(DEFAULT_HOST, 0))
    server.start()
    try:
        yield port
    finally:
        server.stop(0)


def _refused_port() -> int:
    with socket.socket(_FAMILY, socket.SOCK_STREAM) as probe:
        probe.bind((DEFAULT_HOST, 0))
        return probe.getsockname()[1]


def _worker(port: int, engine: str = "vllm") -> Worker:
    return Worker(
        model_id="org/model", engine=engine, port=port, gpu_ids=[0], mode=ConnectionMode.GRPC
    )


def test_a_serving_health_answer_is_the_worker():
    with grpc_listener("SERVING") as port:
        assert _worker(port).grpc_serving() is True


def test_a_not_serving_answer_is_not_yet():
    with grpc_listener("NOT_SERVING") as port:
        assert _worker(port).grpc_serving() is False


def test_a_refused_connection_is_not_yet():
    assert _worker(_refused_port()).grpc_serving(timeout=2) is False


@pytest.mark.parametrize("engine", ["vllm", "sglang", "tokenspeed"])
def test_an_http2_server_answering_404_is_not_the_worker(engine):
    """The stranger of the nightly: grpc reads its 404 as UNIMPLEMENTED and
    offers a ready channel to it; neither makes it this worker."""
    with http2_404_listener() as port:
        assert _worker(port, engine).grpc_serving() is False


def test_a_grpc_server_without_the_health_service_is_not_the_worker():
    with grpc_listener(None) as port:
        assert _worker(port).grpc_serving() is False


@pytest.mark.parametrize("engine", sorted(ENGINES_WITHOUT_GRPC_HEALTH))
def test_an_engine_without_the_health_service_passes_on_a_ready_channel(engine):
    """The scripted mock worker and TRT-LLM serve their own proto's HealthCheck
    and no grpc.health.v1; the harness keeps taking their ready channel."""
    with grpc_listener(None) as port:
        assert _worker(port, engine=engine).grpc_serving() is True


# The log of a worker that found its port taken when it came to bind it: the
# shape of an engine's gRPC server at start-up, after the model load.
PORT_TAKEN_SHAPE = r"""
import sys, time
time.sleep(1)
lines = [
    "[2026-01-01 00:00:49] gRPC startup failed, releasing scheduler resources",
    "Traceback (most recent call last):",
    '  File "grpc/_common.py", line 179, in validate_port_binding_result',
    "    raise RuntimeError(_ERROR_MESSAGE_PORT_BINDING_FAILED % address)",
    "RuntimeError: Failed to bind to address " + ADDRESS + "; set GRPC_VERBOSITY=debug"
    " environment variable to see detailed error message.",
]
print("\n".join(lines))
sys.exit(1)
"""


def test_a_worker_whose_port_a_stranger_holds_reports_its_death_at_once(tmp_path, monkeypatch):
    """A stranger on the worker's port never ends the wait as SERVING; the
    engine's death at bind time does, with the engine's own words, within
    seconds instead of the start-up bound."""
    monkeypatch.setenv(ENV_SHOW_WORKER_LOGS, "0")
    with http2_404_listener() as port:
        worker = Worker(
            model_id="org/model",
            engine="sglang",
            port=port,
            gpu_ids=[0],
            mode=ConnectionMode.GRPC,
            log_dir=str(tmp_path / "e2e-logs"),
        )
        address = host_port(DEFAULT_HOST, port)
        script = f"ADDRESS = {address!r}\n" + PORT_TAKEN_SHAPE
        worker.process = worker._spawn_process([sys.executable, "-c", script], env=dict(os.environ))
        started = time.perf_counter()
        try:
            with pytest.raises(RuntimeError, match="died during startup") as raised:
                worker.wait_ready(timeout=60)
        finally:
            if worker._log_file is not None:
                worker._log_file.close()
        elapsed = time.perf_counter() - started
    assert elapsed < 15, elapsed
    text = str(raised.value)
    assert (
        f'(exit status 1); the log says: "RuntimeError: Failed to bind to address {address}; '
        in text
    )
    assert text.endswith(f"; log file worker-org__model_sglang_grpc_{port}.log")
