"""Unit tests for the gateway launch (no engine, no router: a stand-in script).

The stand-in binds the two ports the harness hands it and serves ``/workers``
and ``/readiness`` like the router, exits on request, or never registers the
last worker, so the three findings a readiness wait must report can be staged.
"""

from __future__ import annotations

import os
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest
from infra.constants import ENV_SHOW_ROUTER_LOGS, ENV_SHOW_WORKER_LOGS, ConnectionMode
from infra.gateway import Gateway
from infra.process_utils import ProcessExitedError, wait_for_health, wait_for_workers_ready
from infra.worker import Worker

FAKE_ROUTER = textwrap.dedent(
    '''
    """A stand-in router: binds its ports, serves /workers and /readiness."""

    import argparse
    import json
    import socket
    import sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--prometheus-port", type=int, required=True)
    parser.add_argument("--worker-urls", nargs="*", default=[])
    parser.add_argument("--fake", default="serve", choices=["serve", "exit", "hold-last"])
    parser.add_argument("--fake-dp", action="store_true", help="register rank-suffixed URLs")
    args, _ = parser.parse_known_args()

    if args.fake == "exit":
        print("Error starting router: the stand-in was told to exit", file=sys.stderr)
        sys.exit(7)

    metrics = socket.socket()
    metrics.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        metrics.bind(("127.0.0.1", args.prometheus_port))
    except OSError as e:
        print(
            "Error starting router: failed to bind metrics server on "
            f"127.0.0.1:{args.prometheus_port}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    metrics.listen(1)

    urls = args.worker_urls[:-1] if args.fake == "hold-last" else args.worker_urls

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/workers":
                workers = [
                    {"url": f"{u}@0", "dp_base_url": u} if args.fake_dp else {"url": u}
                    for u in urls
                ]
                body = {"total": len(urls), "workers": workers}
            elif self.path == "/readiness":
                body = {"status": "ready", "healthy_workers": len(urls)}
            elif self.path == "/health":
                body = {"status": "ok"}
            else:
                self.send_response(404)
                self.end_headers()
                return
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    try:
        server = HTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as e:
        print(f"Error starting router: {e}", file=sys.stderr)
        sys.exit(1)
    server.serve_forever()
    '''
)

WORKERS = ["http://127.0.0.1:1", "http://127.0.0.1:2"]


@pytest.fixture
def fake_router(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Launch the stand-in instead of ``smg.launch_router``, output captured."""
    script = tmp_path / "fake_router.py"
    script.write_text(FAKE_ROUTER, encoding="utf-8")
    monkeypatch.setenv(ENV_SHOW_ROUTER_LOGS, "0")
    monkeypatch.setattr(
        Gateway,
        "_build_base_cmd",
        lambda self: [
            sys.executable,
            str(script),
            "--port",
            str(self.port),
            "--prometheus-port",
            str(self.prometheus_port),
        ],
    )
    return script


def _squat(port: int) -> socket.socket:
    """Take ``port`` the way another process would after the harness reserved it."""
    squatter = socket.socket()
    squatter.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    squatter.bind(("127.0.0.1", port))
    squatter.listen(1)
    return squatter


def test_an_exited_router_is_reported_at_once_with_its_output(fake_router, tmp_path):
    gateway = Gateway()
    started = time.perf_counter()
    try:
        with pytest.raises(ProcessExitedError) as excinfo:
            gateway.start(
                worker_urls=WORKERS,
                model_path="m",
                timeout=30,
                extra_args=["--fake", "exit"],
                log_dir=str(tmp_path),
            )
    finally:
        gateway.shutdown()
    elapsed = time.perf_counter() - started
    assert elapsed < 10, f"waited {elapsed:.1f}s on a process that had exited"
    message = str(excinfo.value)
    assert "exited with code 7" in message
    assert "Error starting router: the stand-in was told to exit" in message
    log = tmp_path / f"router-{gateway.port}.log"
    assert str(log) in message
    assert "told to exit" in log.read_text(encoding="utf-8")
    assert gateway.process is None


@pytest.mark.parametrize("which", ["port", "prometheus_port"])
def test_a_port_taken_since_reservation_is_reallocated(fake_router, which):
    gateway = Gateway()
    taken = getattr(gateway, which)
    squatter = _squat(taken)
    try:
        gateway.start(worker_urls=WORKERS, model_path="m", timeout=15)
        assert gateway.is_running
        assert getattr(gateway, which) != taken
        assert gateway.base_url.endswith(f":{gateway.port}")
        assert gateway.metrics_url.endswith(f":{gateway.prometheus_port}")
    finally:
        gateway.shutdown()
        squatter.close()


def test_the_timeout_names_the_workers_still_missing(fake_router):
    gateway = Gateway()
    try:
        with pytest.raises(TimeoutError) as excinfo:
            gateway.start(
                worker_urls=WORKERS,
                model_path="m",
                timeout=4,
                extra_args=["--fake", "hold-last"],
            )
    finally:
        gateway.shutdown()
    message = str(excinfo.value)
    assert "workers: 1/2" in message
    assert f"missing: {WORKERS[1]}" in message
    assert "readiness: waiting for workers" in message


def test_the_timeout_names_a_data_parallel_worker_by_its_base_url(fake_router):
    gateway = Gateway()
    try:
        with pytest.raises(TimeoutError) as excinfo:
            gateway.start(
                worker_urls=WORKERS,
                model_path="m",
                timeout=4,
                extra_args=["--fake", "hold-last", "--fake-dp"],
            )
    finally:
        gateway.shutdown()
    # The first worker's rank registered as "<url>@0": only the second is missing.
    assert f"missing: {WORKERS[1]}, readiness" in str(excinfo.value)


def test_the_log_goes_under_e2e_log_dir_when_start_gets_no_log_dir(
    fake_router, tmp_path, monkeypatch
):
    log_dir = tmp_path / "e2e-logs"
    monkeypatch.setenv("E2E_LOG_DIR", str(log_dir))
    gateway = Gateway()
    try:
        with pytest.raises(ProcessExitedError):
            gateway.start(
                worker_urls=WORKERS, model_path="m", timeout=30, extra_args=["--fake", "exit"]
            )
    finally:
        gateway.shutdown()
    log = log_dir / f"router-{gateway.port}.log"
    assert gateway.log_path == str(log)
    assert "told to exit" in log.read_text(encoding="utf-8")


def test_without_a_log_dir_the_log_is_a_private_temp_file(fake_router, monkeypatch):
    monkeypatch.delenv("E2E_LOG_DIR", raising=False)
    # A failed launch keeps the file: the error names it.
    gateway = Gateway()
    try:
        with pytest.raises(ProcessExitedError) as excinfo:
            gateway.start(
                worker_urls=WORKERS, model_path="m", timeout=30, extra_args=["--fake", "exit"]
            )
    finally:
        gateway.shutdown()
    kept = Path(gateway.log_path)
    try:
        assert str(kept) in str(excinfo.value)
        assert kept.parent == Path(tempfile.gettempdir())
        assert kept.name.startswith(f"smg-router-{gateway.port}-")
        assert kept.name.endswith(".log")
        assert stat.S_IMODE(kept.stat().st_mode) == 0o600
        assert "told to exit" in kept.read_text(encoding="utf-8")
    finally:
        kept.unlink(missing_ok=True)
    # A launch that succeeded leaves nothing behind.
    gateway = Gateway()
    try:
        gateway.start(worker_urls=WORKERS, model_path="m", timeout=15)
        running = Path(gateway.log_path)
        assert running.exists()
        assert running.name.startswith(f"smg-router-{gateway.port}-")
    finally:
        gateway.shutdown()
    assert not running.exists()
    assert gateway.log_path is None


def test_readiness_waits_stop_when_the_process_exits():
    def exited() -> subprocess.Popen:
        proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(3)"])
        proc.wait(timeout=10)
        return proc

    started = time.perf_counter()
    with pytest.raises(ProcessExitedError, match="exited with code 3"):
        wait_for_health("http://127.0.0.1:9", timeout=30, process=exited())
    with pytest.raises(ProcessExitedError, match="exited with code 3"):
        wait_for_workers_ready("http://127.0.0.1:9", 1, timeout=30, process=exited())
    assert time.perf_counter() - started < 10


# A worker that dials the router in: the router binds the wire, so the worker
# has no port of its own to probe and loads its model inside the gateway's
# readiness wait. The launcher here dies before any ready line, as an engine
# whose install does not match its servicer's does.
DIALING_WORKER = "ipc:///tmp/smg-zmq/engine-31002"
DYING_LAUNCHER = (
    "import sys; print('INFO loading the engine', flush=True);"
    " raise ImportError(\"cannot import name 'serve' from 'engine'\")"
)


def _dialing_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: str) -> Worker:
    """A ZMQ-mode worker whose process is a stub running ``script`` through the
    harness's own output redirection."""
    monkeypatch.setenv(ENV_SHOW_WORKER_LOGS, "0")
    worker = Worker(
        model_id="org/model",
        engine="vllm",
        port=31002,
        gpu_ids=[0],
        mode=ConnectionMode.ZMQ,
        log_dir=str(tmp_path / "e2e-logs"),
    )
    worker.process = worker._spawn_process([sys.executable, "-c", script], env=dict(os.environ))
    return worker


def test_a_worker_that_dies_during_the_readiness_wait_ends_it_with_its_death_report(
    fake_router, tmp_path, monkeypatch
):
    # The stand-in holds the worker back, as the router never sees an engine
    # that died before dialling in; the wait used to run to its bound.
    worker = _dialing_worker(tmp_path, monkeypatch, DYING_LAUNCHER)
    gateway = Gateway()
    started = time.perf_counter()
    try:
        with pytest.raises(RuntimeError, match="died during startup") as excinfo:
            gateway.start(
                worker_urls=[DIALING_WORKER],
                workers=[worker],
                model_path="m",
                timeout=30,
                extra_args=["--fake", "hold-last"],
                log_dir=str(tmp_path),
            )
        elapsed = time.perf_counter() - started
        # The launch failed: the router is gone, as after a timeout.
        assert gateway.process is None
    finally:
        gateway.shutdown()
        if worker._log_file is not None:
            worker._log_file.close()
    # The exit ends the wait within one poll interval, not at the bound.
    assert elapsed < 10, f"waited {elapsed:.1f}s on a worker that had exited"
    text = str(excinfo.value)
    assert text == worker._death_report()
    assert text.startswith(
        f"Worker org/model (PID {worker.process.pid}) died during startup (exit status 1)"
    )
    assert "the log says: \"ImportError: cannot import name 'serve' from 'engine'\"" in text
    assert "loading the engine" not in text
    assert text.endswith("; log file worker-org__model_vllm_zmq_31002.log")


def test_the_readiness_wait_stops_when_a_launched_worker_exits(tmp_path, monkeypatch):
    worker = _dialing_worker(tmp_path, monkeypatch, "import sys; sys.exit(3)")
    worker.process.wait(timeout=30)
    started = time.perf_counter()
    try:
        with pytest.raises(RuntimeError, match=r"died during startup \(exit status 3\)"):
            wait_for_workers_ready("http://127.0.0.1:9", 1, timeout=30, workers=[worker])
    finally:
        if worker._log_file is not None:
            worker._log_file.close()
    assert time.perf_counter() - started < 10
