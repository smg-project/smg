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
from infra.constants import (
    CLOUD_REQUEST_TIMEOUT_SECS,
    ENV_SHOW_ROUTER_LOGS,
    ENV_SHOW_WORKER_LOGS,
    ConnectionMode,
)
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
    # The disaggregated flags: ``URL [BOOTSTRAP_PORT]`` per worker, repeatable.
    for role in ("--encode", "--prefill", "--decode"):
        parser.add_argument(role, nargs="+", action="append", default=[])
    parser.add_argument("--fake", default="serve", choices=["serve", "exit", "hold-last"])
    parser.add_argument("--fake-dp", action="store_true", help="register rank-suffixed URLs")
    parser.add_argument("--fake-dp-ranks", type=int, default=1, help="rank entries per worker")
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

    urls = args.worker_urls + [flag[0] for flag in args.encode + args.prefill + args.decode]
    urls = urls[:-1] if args.fake == "hold-last" else urls

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            # The router lists and counts rank entries: a data-parallel worker
            # is one entry per rank, "<url>@<rank>" with its base in dp_base_url.
            if args.fake_dp:
                entries = [
                    {"url": f"{u}@{rank}", "dp_base_url": u}
                    for u in urls
                    for rank in range(args.fake_dp_ranks)
                ]
            else:
                entries = [{"url": u} for u in urls]
            if self.path == "/workers":
                body = {"total": len(entries), "workers": entries}
            elif self.path == "/readiness":
                # Like the router's disaggregated readiness: held while no
                # worker of a role is registered; the healthy count is of
                # entries, as the router's.
                status = "ready" if urls else "waiting"
                body = {"status": status, "healthy_workers": len(entries)}
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


class _Started:
    """A launched worker as the disaggregated branches read one: its URL, its
    bootstrap port and the liveness check of the readiness wait."""

    def __init__(self, url: str, bootstrap_port: int | None = None) -> None:
        self.base_url = url
        self.worker_url = url
        self.bootstrap_port = bootstrap_port

    def raise_if_exited(self) -> None:
        """Alive for the whole wait."""


def test_a_pd_launch_waits_on_the_readiness_gate(fake_router):
    """The stand-in never registers the decode worker: the PD launch waits
    for readiness as the regular one does, and the timeout names the state."""
    gateway = Gateway()
    try:
        with pytest.raises(TimeoutError) as excinfo:
            gateway.start(
                prefill_workers=[_Started(WORKERS[0], bootstrap_port=8998)],
                decode_workers=[_Started(WORKERS[1])],
                timeout=4,
                extra_args=["--fake", "hold-last"],
            )
    finally:
        gateway.shutdown()
    message = str(excinfo.value)
    assert "workers: 1/2" in message
    assert f"missing: {WORKERS[1]}" in message
    assert "readiness: waiting for workers" in message


def test_an_empty_pd_launch_returns_on_health_alone(fake_router):
    """A disaggregated gateway started with no worker (a fleet registered
    through the API afterwards) returns on /health: its readiness needs each
    role, so there is nothing to wait for yet."""
    gateway = Gateway()
    try:
        gateway.start(prefill_workers=[], decode_workers=[], timeout=4)
    finally:
        gateway.shutdown()


def test_an_epd_launch_waits_on_the_readiness_gate(fake_router):
    """The same wait for the three-role gateway, the decode worker held."""
    gateway = Gateway()
    try:
        with pytest.raises(TimeoutError) as excinfo:
            gateway.start(
                encode_workers=[_Started("http://127.0.0.1:3")],
                prefill_workers=[_Started(WORKERS[0], bootstrap_port=8998)],
                decode_workers=[_Started(WORKERS[1])],
                timeout=4,
                extra_args=["--fake", "hold-last"],
            )
    finally:
        gateway.shutdown()
    message = str(excinfo.value)
    assert "workers: 2/3" in message
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


def test_the_wait_holds_for_a_missing_worker_behind_a_data_parallel_count(fake_router):
    """A data-parallel worker registers one entry per rank, so the first
    worker's two ranks satisfy a count of two while the second worker is
    still absent: the wait holds for the missing worker's URL, not for the
    count, and the timeout names it."""
    gateway = Gateway()
    try:
        with pytest.raises(TimeoutError) as excinfo:
            gateway.start(
                worker_urls=WORKERS,
                model_path="m",
                timeout=4,
                extra_args=["--fake", "hold-last", "--fake-dp", "--fake-dp-ranks", "2"],
            )
    finally:
        gateway.shutdown()
    message = str(excinfo.value)
    assert "workers: 1/2" in message
    assert f"missing: {WORKERS[1]}" in message


def test_a_data_parallel_fleet_releases_the_wait_once_each_worker_is_registered(fake_router):
    """Each worker registered with two ranks: four entries for two workers
    release the wait, since the gate is each worker's URL. A gate comparing
    the rank entries with the worker count would never release on a
    data-parallel fleet, the wait timing out at ``workers: 2/2``."""
    gateway = Gateway()
    try:
        gateway.start(
            worker_urls=WORKERS,
            model_path="m",
            timeout=15,
            extra_args=["--fake-dp", "--fake-dp-ranks", "2"],
        )
        assert gateway.is_running
    finally:
        gateway.shutdown()


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


def test_a_cloud_gateway_gets_a_budget_for_each_call_to_its_provider(monkeypatch):
    """A provider call that never answers ends at the budget as a gateway error
    the SDK clients retry on their own, instead of holding the test for the ten
    minutes of the SDK's own timeout (twice in one job: the job's cap)."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    args = Gateway()._build_cloud_args("openai", "memory")

    assert args[args.index("--request-timeout-secs") + 1] == str(CLOUD_REQUEST_TIMEOUT_SECS)
