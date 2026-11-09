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
from types import SimpleNamespace

import pytest
from infra import constants
from infra import worker as worker_module
from infra.constants import (
    DEFAULT_HOST,
    ENV_SHOW_WORKER_LOGS,
    ENV_TRTLLM_SERVICER_IMPL,
    ConnectionMode,
    host_port,
)
from infra.worker import Worker

pytest.importorskip("grpc")
pytest.importorskip("grpc_health")

_FAMILY = socket.AF_INET6 if ":" in DEFAULT_HOST else socket.AF_INET
_HTTP2_PREFACE = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"
_SETTINGS, _HEADERS = 0x4, 0x1
_ACK, _END_STREAM, _END_HEADERS = 0x1, 0x1, 0x4
# HPACK indexed header field: static table entry 13 is ``:status: 404``.
_STATUS_404 = b"\x8d"


@pytest.fixture(autouse=True)
def _hermetic_trt_lane(monkeypatch):
    """Native TRT unless this test selects a lane or package default."""
    monkeypatch.delenv(ENV_TRTLLM_SERVICER_IMPL, raising=False)
    monkeypatch.delenv("SMG_TRTLLM_SERVICER_IMPL", raising=False)
    monkeypatch.setattr(constants, "_package_servicer_switch", lambda: None)


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


def _worker(port: int, engine: str = "vllm", model_id: str = "org/model") -> Worker:
    return Worker(
        model_id=model_id, engine=engine, port=port, gpu_ids=[0], mode=ConnectionMode.GRPC
    )


def test_a_serving_health_answer_is_the_worker():
    with grpc_listener("SERVING") as port:
        assert _worker(port).grpc_serving() is True


def test_a_not_serving_answer_is_not_yet():
    with grpc_listener("NOT_SERVING") as port:
        assert _worker(port).grpc_serving() is False


def test_a_refused_connection_is_not_yet():
    assert _worker(_refused_port()).grpc_serving(timeout=2) is False


@pytest.mark.parametrize("engine", ["vllm", "sglang", "tokenspeed", "trtllm"])
def test_an_http2_server_answering_404_is_not_the_worker(engine, monkeypatch):
    """The stranger of the nightly: grpc reads its 404 as UNIMPLEMENTED and
    offers a ready channel to it; neither makes it this worker."""
    monkeypatch.setattr(worker_module, "get_model_spec", lambda model_id: {"model": _TRT_MODEL})
    with http2_404_listener() as port:
        assert _worker(port, engine).grpc_serving() is False


def test_a_grpc_server_without_the_health_service_is_not_the_worker():
    with grpc_listener(None) as port:
        assert _worker(port).grpc_serving() is False


def test_trt_does_not_accept_an_empty_grpc_server(monkeypatch):
    """A ready channel without native TRT RPCs cannot identify this worker."""
    monkeypatch.setattr(worker_module, "get_model_spec", lambda model_id: {"model": _TRT_MODEL})
    with grpc_listener(None) as port:
        assert _worker(port, engine="trtllm").grpc_serving() is False


def test_trt_does_not_accept_a_standard_health_only_server(monkeypatch):
    monkeypatch.setattr(worker_module, "get_model_spec", lambda model_id: {"model": _TRT_MODEL})
    with grpc_listener("SERVING") as port:
        assert _worker(port, engine="trtllm").grpc_serving() is False


_TRT_MODEL_KEY = "ci/logical-model-key"
_TRT_MODEL = "/qualified/models/exact-repository"


@pytest.fixture
def trt_model(monkeypatch):
    """The logical spec key differs from the model passed to TRT's launcher."""

    def spec(model_id):
        assert model_id == _TRT_MODEL_KEY
        return {"model": _TRT_MODEL}

    monkeypatch.setattr(worker_module, "get_model_spec", spec)


@contextmanager
def trt_listener(
    *,
    status: str = "OK",
    model_id: str = _TRT_MODEL,
    health_error: str | None = None,
    info_error: str | None = None,
    health_delay: float = 0,
    info_delay: float = 0,
):
    """Only the generated TRT service, on loopback; no engine imports."""
    import grpc
    from smg_grpc_proto import trtllm_service_pb2, trtllm_service_pb2_grpc

    calls = []

    class Native(trtllm_service_pb2_grpc.TrtllmServiceServicer):
        def HealthCheck(self, request, context):
            calls.append("HealthCheck")
            if health_delay:
                time.sleep(health_delay)
            if health_error:
                context.abort(getattr(grpc.StatusCode, health_error), "health fixture")
            return trtllm_service_pb2.HealthCheckResponse(status=status)

        def GetModelInfo(self, request, context):
            calls.append("GetModelInfo")
            if info_delay:
                time.sleep(info_delay)
            if info_error:
                context.abort(getattr(grpc.StatusCode, info_error), "model info fixture")
            # Zero optional metadata must not turn an identified, healthy
            # worker into NOT_SERVING while the upstream vocab bug is open.
            return trtllm_service_pb2.GetModelInfoResponse(model_id=model_id)

    with futures.ThreadPoolExecutor(max_workers=2) as pool:
        server = grpc.server(pool)
        trtllm_service_pb2_grpc.add_TrtllmServiceServicer_to_server(Native(), server)
        port = server.add_insecure_port(host_port(DEFAULT_HOST, 0))
        assert port > 0
        server.start()
        try:
            yield port, calls
        finally:
            server.stop(0).wait(timeout=2)


def _trt_worker(port: int) -> Worker:
    return _worker(port, engine="trtllm", model_id=_TRT_MODEL_KEY)


def test_trt_native_healthy_exact_launcher_model_passes(trt_model):
    with trt_listener() as (port, calls):
        assert _trt_worker(port).grpc_serving() is True
        assert calls == ["HealthCheck", "GetModelInfo"]


@pytest.mark.parametrize("status", ["", "NOT_SERVING", "Error: engine unavailable"])
def test_trt_native_non_ok_status_does_not_fetch_model_info(trt_model, status):
    with trt_listener(status=status) as (port, calls):
        assert _trt_worker(port).grpc_serving() is False
        assert calls == ["HealthCheck"]


@pytest.mark.parametrize("model_id", ["", "other/model", _TRT_MODEL_KEY, "exact-repository"])
def test_trt_rejects_missing_wrong_logical_key_or_basename_identity(trt_model, model_id):
    with trt_listener(model_id=model_id) as (port, _):
        assert _trt_worker(port).grpc_serving() is False


@pytest.mark.parametrize("error", ["UNIMPLEMENTED", "UNAVAILABLE", "DEADLINE_EXCEEDED"])
def test_trt_requires_a_successful_native_health_rpc(trt_model, error):
    with trt_listener(health_error=error) as (port, calls):
        assert _trt_worker(port).grpc_serving() is False
        assert calls == ["HealthCheck"]


@pytest.mark.parametrize("error", ["UNIMPLEMENTED", "UNAVAILABLE", "DEADLINE_EXCEEDED"])
def test_trt_requires_a_successful_native_model_info_rpc(trt_model, error):
    with trt_listener(info_error=error) as (port, calls):
        assert _trt_worker(port).grpc_serving() is False
        assert calls == ["HealthCheck", "GetModelInfo"]


def test_trt_two_real_rpcs_share_one_probe_budget(trt_model, monkeypatch):
    import grpc

    # Each reply fits a fresh 1s timeout, but the pair does not fit the
    # one probe deadline. A per-RPC timeout reset would incorrectly pass.
    with trt_listener(health_delay=0.6, info_delay=0.6) as (port, calls):
        target = host_port(DEFAULT_HOST, port)
        with grpc.insecure_channel(target) as channel:
            # Establish transport before measuring the RPC budget. The worker
            # must use this exact channel, not a second cold connection.
            grpc.channel_ready_future(channel).result(timeout=5.0)
            channels = []

            def create_channel(actual_target):
                assert actual_target == target
                channels.append(channel)
                return channel

            monkeypatch.setattr(grpc, "insecure_channel", create_channel)
            assert _trt_worker(port).grpc_serving(timeout=1.0) is False
            assert calls == ["HealthCheck", "GetModelInfo"]
            assert channels == [channel]


@pytest.mark.parametrize(
    ("status", "model_id", "info_error", "expected"),
    [
        ("OK", _TRT_MODEL, None, True),
        ("NOT_SERVING", _TRT_MODEL, None, False),
        ("OK", "wrong/model", None, False),
        ("OK", _TRT_MODEL, "UNIMPLEMENTED", False),
    ],
)
def test_trt_closes_real_channel_on_success_and_rejections(
    trt_model, monkeypatch, status, model_id, info_error, expected
):
    import grpc

    original_channel = grpc.insecure_channel
    channels = []

    class TrackedChannel:
        def __init__(self, channel):
            self.channel = channel
            self.closed = 0

        def __getattr__(self, name):
            return getattr(self.channel, name)

        def close(self):
            self.closed += 1
            self.channel.close()

    def create_channel(*args, **kwargs):
        channel = TrackedChannel(original_channel(*args, **kwargs))
        channels.append(channel)
        return channel

    monkeypatch.setattr(grpc, "insecure_channel", create_channel)
    with trt_listener(status=status, model_id=model_id, info_error=info_error) as (port, _):
        assert _trt_worker(port).grpc_serving() is expected
    assert len(channels) == 1
    assert channels[0].closed == 1


@pytest.mark.parametrize(
    ("budget", "health_elapsed", "info_elapsed", "expected_calls", "expected"),
    [
        (0.1, 0.03, 0.02, ["health", "info"], True),
        (0.1, 0.11, 0, ["health"], False),
        (0.1, 0.03, 0.08, ["health", "info"], False),
        (0, 0, 0, [], False),
    ],
)
def test_trt_preserves_absolute_deadline_and_closes_channel(
    trt_model, monkeypatch, budget, health_elapsed, info_elapsed, expected_calls, expected
):
    import grpc
    from smg_grpc_proto import trtllm_service_pb2_grpc

    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(worker_module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    calls = []
    timeouts = []
    channel = SimpleNamespace(closed=0)

    def close():
        channel.closed += 1

    channel.close = close
    monkeypatch.setattr(grpc, "insecure_channel", lambda *args, **kwargs: channel)

    class Stub:
        def HealthCheck(self, request, timeout):
            calls.append("health")
            timeouts.append(timeout)
            clock.now += health_elapsed
            return SimpleNamespace(status="OK")

        def GetModelInfo(self, request, timeout):
            calls.append("info")
            timeouts.append(timeout)
            clock.now += info_elapsed
            return SimpleNamespace(model_id=_TRT_MODEL)

    monkeypatch.setattr(trtllm_service_pb2_grpc, "TrtllmServiceStub", lambda channel: Stub())
    assert _trt_worker(1).grpc_serving(timeout=5.0, _deadline=100.0 + budget) is expected
    assert calls == expected_calls
    if timeouts:
        assert timeouts[0] == pytest.approx(budget)
    if len(timeouts) > 1:
        assert timeouts[1] == pytest.approx(budget - health_elapsed)
    assert channel.closed == 1


def test_trt_closes_channel_when_generated_proto_is_missing(monkeypatch):
    import grpc

    channel = SimpleNamespace(closed=0)

    def close():
        channel.closed += 1

    channel.close = close
    monkeypatch.setattr(grpc, "insecure_channel", lambda *args, **kwargs: channel)
    monkeypatch.setitem(sys.modules, "smg_grpc_proto", None)
    with pytest.raises(RuntimeError, match="TRT gRPC health check requires smg-grpc-proto"):
        _trt_worker(1).grpc_serving()
    assert channel.closed == 1


@pytest.mark.parametrize("reply_at_deadline", [False, True])
def test_trt_startup_probe_cannot_extend_original_budget(monkeypatch, reply_at_deadline):
    clock = SimpleNamespace(now=100.0)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(
        worker_module, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep)
    )
    worker = _trt_worker(1)
    probes = []

    def probe(timeout=5.0, *, _deadline=None):
        probes.append((timeout, _deadline))
        clock.now += timeout
        return reply_at_deadline

    monkeypatch.setattr(worker, "raise_if_exited", lambda: None)
    monkeypatch.setattr(worker, "grpc_serving", probe)
    with pytest.raises(TimeoutError, match="did not become healthy within 0.125s"):
        worker._wait_grpc_healthy(timeout=0.125)
    assert probes == [(0.125, 100.125)]
    assert clock.now == 100.125
    assert sleeps == []


def test_trt_startup_sleep_cannot_extend_original_budget(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(
        worker_module, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep)
    )
    worker = _trt_worker(1)
    probes = []

    def probe(timeout=5.0, *, _deadline=None):
        probes.append((timeout, _deadline))
        clock.now += 0.025
        return False

    monkeypatch.setattr(worker, "raise_if_exited", lambda: None)
    monkeypatch.setattr(worker, "grpc_serving", probe)
    with pytest.raises(TimeoutError):
        worker._wait_grpc_healthy(timeout=0.125)
    assert probes == [(0.125, 100.125)]
    assert sleeps == pytest.approx([0.1])
    assert clock.now == 100.125


def test_a_trtllm_worker_on_the_rust_servicer_needs_the_health_service(monkeypatch):
    """On the Rust servicer (``E2E_TRTLLM_SERVICER_IMPL=rust``) the worker
    hosts grpc.health.v1 like every engine servicer, so a ready channel
    without it is a stranger on the port, not this worker."""
    monkeypatch.setenv(ENV_TRTLLM_SERVICER_IMPL, "rust")
    with grpc_listener(None) as port:
        assert _worker(port, engine="trtllm").grpc_serving() is False
    with grpc_listener("SERVING") as port:
        assert _worker(port, engine="trtllm").grpc_serving() is True


def test_a_trtllm_rust_servicer_rejects_native_health_only(trt_model, monkeypatch):
    monkeypatch.setenv(ENV_TRTLLM_SERVICER_IMPL, "rust")
    with trt_listener() as (port, calls):
        assert _trt_worker(port).grpc_serving() is False
        assert calls == []


def test_a_trtllm_package_rust_default_uses_standard_health(monkeypatch):
    monkeypatch.setattr(
        constants,
        "_package_servicer_switch",
        lambda: lambda variable, environ=None, **_: ("rust", "default"),
    )

    def unexpected_model_lookup(_model_id):
        raise AssertionError("the Rust health service needs no native model lookup")

    monkeypatch.setattr(worker_module, "get_model_spec", unexpected_model_lookup)
    with grpc_listener("SERVING") as port:
        assert _worker(port, engine="trtllm").grpc_serving() is True


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


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(KeyError("missing launcher model"), id="key-error"),
        pytest.param(TypeError("malformed model spec"), id="type-error"),
    ],
)
def test_trt_model_lookup_error_is_not_a_readiness_retry(monkeypatch, error):
    import grpc
    from smg_grpc_proto import trtllm_service_pb2_grpc

    channel = SimpleNamespace(closed=0)
    calls = []

    def close():
        channel.closed += 1

    def bad_spec(model_id):
        assert model_id == _TRT_MODEL_KEY
        calls.append("spec")
        raise error

    def unexpected_rpc(channel):
        raise AssertionError("native RPC stub constructed before model configuration resolved")

    channel.close = close
    monkeypatch.setattr(grpc, "insecure_channel", lambda *args, **kwargs: channel)
    monkeypatch.setattr(worker_module, "get_model_spec", bad_spec)
    monkeypatch.setattr(trtllm_service_pb2_grpc, "TrtllmServiceStub", unexpected_rpc)
    with pytest.raises(type(error)) as raised:
        _trt_worker(1).grpc_serving()
    assert raised.value is error
    assert calls == ["spec"]
    assert channel.closed == 1


@pytest.mark.parametrize(
    "answered",
    [
        pytest.param("", id="empty"),
        pytest.param("other/model", id="foreign"),
        pytest.param(_TRT_MODEL_KEY, id="logical-key"),
        pytest.param("exact-repository", id="basename"),
    ],
)
def test_trt_timeout_names_actual_and_expected_model(trt_model, monkeypatch, answered):
    import grpc
    from smg_grpc_proto import trtllm_service_pb2_grpc

    clock = SimpleNamespace(now=100.0)
    sleeps = []
    calls = []
    channel = SimpleNamespace(closed=0)

    def sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds

    def close():
        channel.closed += 1

    class Stub:
        def HealthCheck(self, request, timeout):
            calls.append("health")
            assert timeout == pytest.approx(0.125)
            return SimpleNamespace(status="OK")

        def GetModelInfo(self, request, timeout):
            calls.append("info")
            assert timeout == pytest.approx(0.125)
            return SimpleNamespace(model_id=answered)

    channel.close = close
    monkeypatch.setattr(
        worker_module, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep)
    )
    monkeypatch.setattr(grpc, "insecure_channel", lambda *args, **kwargs: channel)
    monkeypatch.setattr(trtllm_service_pb2_grpc, "TrtllmServiceStub", lambda channel: Stub())
    worker = _trt_worker(1)
    monkeypatch.setattr(worker, "raise_if_exited", lambda: None)
    with pytest.raises(TimeoutError) as raised:
        worker._wait_grpc_healthy(timeout=0.125)
    assert str(raised.value) == (
        f"gRPC worker {_TRT_MODEL_KEY} on port 1 did not become healthy within 0.125s; "
        f"answered as {answered!r}, expected {_TRT_MODEL!r}"
    )
    assert calls == ["health", "info"]
    assert channel.closed == 1
    assert sleeps == [0.125]
    assert clock.now == 100.125


def test_trt_wait_does_not_reuse_a_previous_model_mismatch(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    probes = []

    def sleep(seconds):
        clock.now += seconds

    monkeypatch.setattr(
        worker_module, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep)
    )
    worker = _trt_worker(1)
    worker._last_trt_model_identity = ("old-answer", "old-expected")
    monkeypatch.setattr(worker, "raise_if_exited", lambda: None)

    def probe(timeout=5.0, *, _deadline=None):
        probes.append((timeout, _deadline))
        return False

    monkeypatch.setattr(worker, "grpc_serving", probe)
    with pytest.raises(TimeoutError) as raised:
        worker._wait_grpc_healthy(timeout=0.125)
    assert str(raised.value) == (
        f"gRPC worker {_TRT_MODEL_KEY} on port 1 did not become healthy within 0.125s"
    )
    assert worker._last_trt_model_identity is None
    assert probes == [(0.125, 100.125)]
    assert clock.now == 100.125


def test_trt_startup_returns_after_one_healthy_probe_before_deadline(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    sleeps = []
    probes = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(
        worker_module, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep)
    )
    worker = _trt_worker(1)
    worker.process = SimpleNamespace(pid=424242, returncode=None, poll=lambda: None)

    def probe(timeout=5.0, *, _deadline=None):
        probes.append((timeout, _deadline))
        clock.now += 0.025
        return True

    monkeypatch.setattr(worker, "grpc_serving", probe)
    worker.wait_ready(timeout=0.125)
    assert probes == [(0.125, 100.125)]
    assert clock.now == 100.025
    assert sleeps == []


@pytest.mark.parametrize("death_phase", ["initially-dead", "during-final-probe"])
def test_trt_startup_reports_real_launcher_death(tmp_path, monkeypatch, death_phase):
    clock = SimpleNamespace(now=100.0)
    sleeps = []
    probes = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(
        worker_module, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep)
    )
    worker = _trt_worker(1)
    process = SimpleNamespace(
        pid=424242, returncode=17 if death_phase == "initially-dead" else None
    )
    process.poll = lambda: process.returncode
    worker.process = process
    worker.log_dir = str(tmp_path)
    log_path = tmp_path / "trt-startup.log"
    log_path.write_text("RuntimeError: TRT launcher exited before readiness\n")
    worker.log_path = str(log_path)

    def probe(timeout=5.0, *, _deadline=None):
        assert process.returncode is None
        probes.append((timeout, _deadline))
        clock.now = _deadline
        process.returncode = 17
        return False

    # Only the probe is controlled. Keep real is_alive/raise_if_exited and
    # _death_report/read_log active, including their poll and log-file reads.
    monkeypatch.setattr(worker, "grpc_serving", probe)
    with pytest.raises(RuntimeError, match="died during startup") as raised:
        worker.wait_ready(timeout=0.125)
    assert str(raised.value) == (
        f"Worker {_TRT_MODEL_KEY} (PID 424242) died during startup (exit status 17); "
        'the log says: "RuntimeError: TRT launcher exited before readiness"; '
        "log file trt-startup.log"
    )
    assert process.returncode == 17
    if death_phase == "initially-dead":
        assert probes == []
        assert clock.now == 100.0
    else:
        assert probes == [(0.125, 100.125)]
        assert clock.now == 100.125
    assert sleeps == []
