"""A healthy worker must not let a fixture race static MCP registration."""

from __future__ import annotations

import ast
import asyncio
import logging
import os
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from infra import mcp_readiness


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _mock_metrics(monkeypatch, handler, clock=None):
    client_type = httpx.AsyncClient
    sleep = asyncio.sleep
    state = SimpleNamespace(closed=0, clients=[], timeout_budgets=[])

    class Body(httpx.AsyncByteStream):
        def __init__(self, content):
            self.content = content

        async def __aiter__(self):
            yield self.content

        async def aclose(self):
            state.closed += 1

    def respond(request):
        result = handler(request)
        if isinstance(result, tuple):
            status, body = result
        else:
            status, body = 200, result
        return httpx.Response(status, stream=Body(body.encode()))

    def client(**kwargs):
        value = client_type(transport=httpx.MockTransport(respond), **kwargs)
        state.clients.append(value)
        return value

    monkeypatch.setattr(mcp_readiness.httpx, "AsyncClient", client)
    if clock is not None:
        monkeypatch.setattr(mcp_readiness, "time", clock)

        async def advance(seconds):
            clock.sleep(seconds)
            await sleep(0)

        monkeypatch.setattr(mcp_readiness.asyncio, "sleep", advance)

        @asynccontextmanager
        async def virtual_timeout(seconds):
            # These cases test virtual budget arithmetic; the real HTTP cases
            # below exercise the actual asyncio cancellation scope and deadline.
            state.timeout_budgets.append(seconds)
            yield

        monkeypatch.setattr(mcp_readiness.asyncio, "timeout", virtual_timeout)
    return state


def _fixture():
    # Execute the actual fixture body without importing or collecting GPU tests.
    path = Path(__file__).parents[1] / "responses/test_builtin_tools.py"
    module = ast.parse(path.read_text())
    node = next(
        n
        for n in module.body
        if isinstance(n, ast.FunctionDef) and n.name == "gateway_with_mcp_config_grpc"
    )
    node.decorator_list = []
    namespace = {
        "os": os,
        "time": time,
        "logger": logging.getLogger(__name__),
        "pytest": pytest,
        "openai": SimpleNamespace(OpenAI=lambda **_: object()),
        "DEFAULT_ROUTER_TIMEOUT": 60,
        "wait_for_single_mcp_registration": mcp_readiness.wait_for_single_mcp_registration,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["gateway_with_mcp_config_grpc"]


@pytest.fixture
def fixture_parts(monkeypatch):
    import infra
    from infra import worker as worker_module

    state = SimpleNamespace(stopped=False, shutdown=False, started=False)
    gateway = SimpleNamespace(
        is_running=True,
        metrics_url="http://127.0.0.1:1",
        base_url="http://127.0.0.1:2",
        log_path=None,
    )

    def start(**kwargs):
        state.started = True
        state.start_args = kwargs

    def shutdown():
        state.shutdown = True
        gateway.is_running = False

    gateway.start = start
    gateway.shutdown = shutdown

    # Use the same absolute imports as the real fixture; prohibit an engine
    # launch if a future import change bypasses this CPU fixture's adapters.
    def forbid_process(*_, **__):
        pytest.fail("CPU fixture attempted to launch an actual engine process")

    monkeypatch.setattr(subprocess, "Popen", forbid_process)
    monkeypatch.setattr(infra, "Gateway", lambda: gateway)
    monkeypatch.setattr(
        worker_module,
        "start_workers",
        lambda *_, **__: [SimpleNamespace(base_url="grpc://127.0.0.1:3")],
    )
    monkeypatch.setattr(worker_module, "stop_workers", lambda _: setattr(state, "stopped", True))
    return gateway, state


def test_fixture_waits_for_registration_after_worker_health(fixture_parts):
    gateway, state = fixture_parts
    registration = SimpleNamespace(requests=0, ready=False)

    class Metrics(BaseHTTPRequestHandler):
        def do_GET(self):
            registration.requests += 1
            registration.ready = registration.requests >= 3
            self.send_response(200)
            self.end_headers()
            self.wfile.write(f"smg_mcp_servers_active {int(registration.ready)}\n".encode())

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Metrics)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    gateway.metrics_url = f"http://127.0.0.1:{server.server_port}"
    fixture = _fixture()(None, "unused.yaml")
    try:
        next(fixture)
        assert state.started
        assert registration.ready, "fixture yielded before static MCP registration"
        assert registration.requests == 3
    finally:
        fixture.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert state.shutdown and state.stopped
    assert not thread.is_alive()


@pytest.mark.parametrize(
    "body",
    [
        "",
        "# TYPE smg_mcp_servers_active gauge\n",
        "other_metric 1\n",
        "smg_mcp_servers_active 0\n",
        "smg_mcp_servers_active NaN\n",
        "smg_mcp_servers_active 1\nsmg_mcp_servers_active 1\n",
        "smg_mcp_servers_active malformed\n",
        "smg_mcp_servers_active 1 extra\n",
    ],
)
def test_missing_invalid_or_ambiguous_registration_times_out(monkeypatch, body):
    clock = Clock()
    state = _mock_metrics(monkeypatch, lambda _: body, clock)
    gateway = SimpleNamespace(is_running=True, metrics_url="http://127.0.0.1:1")
    with pytest.raises(TimeoutError, match="router-start deadline"):
        mcp_readiness.wait_for_single_mcp_registration(gateway, deadline=0.3)
    assert clock.now == pytest.approx(0.3)
    assert all(client.is_closed for client in state.clients)
    assert state.closed == 3


def test_a_late_success_cannot_reset_the_deadline(monkeypatch):
    clock = Clock()

    def get(request):
        assert request.extensions["timeout"]["read"] == pytest.approx(0.25)
        clock.now = 0.3
        return "smg_mcp_servers_active 1\n"

    state = _mock_metrics(monkeypatch, get, clock)
    with pytest.raises(TimeoutError, match="deadline"):
        mcp_readiness.wait_for_single_mcp_registration(
            SimpleNamespace(is_running=True, metrics_url="http://127.0.0.1:1"), deadline=0.25
        )
    assert state.closed == 1 and all(client.is_closed for client in state.clients)


@pytest.mark.parametrize("failure", ["http", "transport", "exit"])
def test_fixture_failure_cleans_up_owned_gateway_and_worker(monkeypatch, fixture_parts, failure):
    gateway, state = fixture_parts
    clock = Clock()
    fixture = _fixture()
    fixture.__globals__["time"] = clock

    def get(request):
        timeout = request.extensions["timeout"]["read"]
        assert 0 < timeout <= 1
        clock.sleep(timeout)
        if failure == "transport":
            raise httpx.ConnectError("private fixture")
        if failure == "exit":
            gateway.is_running = False
        return 503, ""

    metrics = _mock_metrics(monkeypatch, get, clock)
    with pytest.raises((TimeoutError, RuntimeError)):
        next(fixture(None, "unused.yaml"))
    assert state.shutdown and state.stopped
    assert clock.now <= 60
    assert all(client.is_closed for client in metrics.clients)


def test_router_deadline_starts_after_separately_budgeted_worker_startup(
    monkeypatch, fixture_parts
):
    from infra import worker as worker_module

    gateway, state = fixture_parts
    clock = Clock()
    fixture = _fixture()
    fixture.__globals__["time"] = clock

    def start_workers(*_, **__):
        clock.sleep(120)
        return [SimpleNamespace(base_url="grpc://127.0.0.1:3")]

    monkeypatch.setattr(worker_module, "start_workers", start_workers)

    def start(**kwargs):
        assert kwargs["timeout"] == 60
        clock.sleep(2)

    gateway.start = start

    def get(request):
        assert clock.now == 122 and request.extensions["timeout"]["read"] == 1
        return "smg_mcp_servers_active 1\n"

    _mock_metrics(monkeypatch, get, clock)
    generator = fixture(None, "unused.yaml")
    try:
        next(generator)
    finally:
        generator.close()
    assert state.shutdown and state.stopped


def test_registration_uses_only_the_router_start_time_left(monkeypatch, fixture_parts):
    gateway, state = fixture_parts
    clock = Clock()
    fixture = _fixture()
    fixture.__globals__["time"] = clock

    def start(**_):
        clock.sleep(59.75)

    def get(request):
        timeout = request.extensions["timeout"]["read"]
        assert timeout == pytest.approx(0.25)
        clock.sleep(timeout)
        return "smg_mcp_servers_active 1\n"

    gateway.start = start
    _mock_metrics(monkeypatch, get, clock)
    with pytest.raises(TimeoutError, match="deadline"):
        next(fixture(None, "unused.yaml"))
    assert clock.now == 60
    assert state.shutdown and state.stopped


def test_process_exit_cannot_be_hidden_by_a_successful_metric(monkeypatch):
    gateway = SimpleNamespace(is_running=True, metrics_url="http://127.0.0.1:1")

    def get(_):
        gateway.is_running = False
        return "smg_mcp_servers_active 1\n"

    metrics = _mock_metrics(monkeypatch, get)
    with pytest.raises(RuntimeError, match="exited"):
        mcp_readiness.wait_for_single_mcp_registration(gateway, deadline=time.monotonic() + 1)
    assert metrics.closed == 1 and all(client.is_closed for client in metrics.clients)


@pytest.mark.parametrize("slow_part", ["headers", "body"])
def test_real_slow_http_respects_total_deadline_and_cleans_up(
    fixture_parts, slow_part, record_property
):
    gateway, state = fixture_parts
    body = b"smg_mcp_servers_active 1\n"
    finished = threading.Event()
    disconnected = threading.Event()
    observations = SimpleNamespace(bytes_sent=0, requests=0)

    class SlowMetrics(BaseHTTPRequestHandler):
        def do_GET(self):
            observations.requests += 1
            headers = f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n\r\n".encode()
            try:
                if slow_part == "body":
                    self.connection.sendall(headers)
                for byte in headers if slow_part == "headers" else body:
                    self.connection.sendall(bytes([byte]))
                    observations.bytes_sent += 1
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                disconnected.set()
            finally:
                finished.set()

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), SlowMetrics)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01))
    thread.start()
    gateway.metrics_url = f"http://127.0.0.1:{server.server_port}"
    fixture = _fixture()
    fixture.__globals__["DEFAULT_ROUTER_TIMEOUT"] = 0.30
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="router-start deadline"):
            next(fixture(None, "unused.yaml"))
        elapsed = time.monotonic() - started
        record_property("slow_part", slow_part)
        record_property("original_budget_seconds", 0.30)
        record_property("elapsed_seconds", elapsed)
        assert elapsed <= 0.45, f"{slow_part} exceeded original total deadline: {elapsed}"
        assert state.shutdown and state.stopped
        assert finished.wait(0.2), "owned HTTP connection remained open after cancellation"
        assert disconnected.is_set()
        record_property("bytes_sent", observations.bytes_sent)
        record_property("owned_request_disconnected", disconnected.is_set())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert not thread.is_alive()
    assert observations.requests == 1


def test_oversized_metrics_fail_closed(monkeypatch):
    clock = Clock()
    state = _mock_metrics(monkeypatch, lambda _: "x" * (1024 * 1024 + 1), clock)
    with pytest.raises(TimeoutError, match="deadline"):
        mcp_readiness.wait_for_single_mcp_registration(
            SimpleNamespace(is_running=True, metrics_url="http://127.0.0.1:1"), deadline=0.3
        )
    assert all(client.is_closed for client in state.clients)


def test_gateway_exit_cancels_an_in_flight_request(monkeypatch):
    gateway = SimpleNamespace(is_running=True, metrics_url="http://127.0.0.1:1")
    client_type = httpx.AsyncClient
    closed = threading.Event()
    cancelled = threading.Event()

    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            gateway.is_running = False
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            yield b"smg_mcp_servers_active 1\n"

        async def aclose(self):
            closed.set()

    def client(**kwargs):
        return client_type(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=SlowBody())),
            **kwargs,
        )

    monkeypatch.setattr(mcp_readiness.httpx, "AsyncClient", client)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="exited"):
        mcp_readiness.wait_for_single_mcp_registration(gateway, deadline=started + 0.3)
    assert cancelled.is_set() and closed.is_set()


@pytest.mark.parametrize("failure", ["model_spec", "gateway_constructor", "worker_index"])
def test_initial_setup_failure_stops_created_workers(monkeypatch, fixture_parts, failure):
    import infra
    from infra import model_specs
    from infra import worker as worker_module

    _, state = fixture_parts

    def fail(*_, **__):
        raise RuntimeError(f"private setup {failure}")

    if failure == "model_spec":
        monkeypatch.setattr(model_specs, "get_model_spec", fail)
    elif failure == "gateway_constructor":
        monkeypatch.setattr(infra, "Gateway", fail)
    else:
        monkeypatch.setattr(worker_module, "start_workers", lambda *_, **__: [])

    with pytest.raises((RuntimeError, IndexError)):
        next(_fixture()(None, "unused.yaml"))
    assert state.stopped, "fixture did not release workers after initial setup failure"
    assert not state.shutdown, "fixture tried to shut down a gateway that was not constructed"


def test_registration_failure_preserves_private_router_log_without_dumping_it(
    monkeypatch, fixture_parts
):
    import infra
    from infra.gateway import Gateway

    _, state = fixture_parts
    monkeypatch.delenv("E2E_LOG_DIR", raising=False)
    gateway = Gateway(port=12345, prometheus_port=12346)
    sentinel = "private diagnostic content must stay in its log"
    log = gateway._open_log_file()
    log.write(sentinel)
    log.flush()
    log_path = Path(gateway.log_path)

    def started(**_):
        gateway._started = True

    def fail(*_, **__):
        raise TimeoutError("private registration failure")

    gateway.start = started
    monkeypatch.setattr(infra, "Gateway", lambda: gateway)
    fixture = _fixture()
    fixture.__globals__["wait_for_single_mcp_registration"] = fail
    try:
        with pytest.raises(TimeoutError, match="private registration failure") as error:
            next(fixture(None, "unused.yaml"))
        assert state.stopped and gateway._log_file is None
        assert log_path.is_file(), "failed MCP registration deleted the only temporary router log"
        notes = "\n".join(getattr(error.value, "__notes__", []))
        assert str(log_path) in notes
        assert sentinel not in str(error.value) and sentinel not in notes
    finally:
        gateway.shutdown()
        log_path.unlink(missing_ok=True)
