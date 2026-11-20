"""SGLang's scheduler metrics on the Rust servicer.

Under ``--enable-metrics`` the scheduler ranks record into Prometheus's
multiprocess directory (``PROMETHEUS_MULTIPROC_DIR``), which SGLang's HTTP
sidecar serves on the Python servicer. The Rust path starts no sidecar, so the
launcher serves the directory on the sidecar's address. These tests run
``serve_rust`` with the Rust server, the scheduler launch and the supervisor
faked, write the directory from real child processes, and scrape over HTTP.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import subprocess
import sys
import types
import urllib.request
from types import SimpleNamespace

import pytest
from smg_grpc_servicer import rust_lifecycle
from smg_grpc_servicer.sglang import rust

# SGLang installs it; the launcher and the recording ranks below need it.
pytest.importorskip("prometheus_client")

# A scheduler rank's recording, as SGLang's collectors make it: a counter, and
# a gauge whose newest value wins across processes.
_RECORD = """
import sys
from prometheus_client import Counter, Gauge
Counter("sglang:evicted_tokens", "Evicted tokens").inc(int(sys.argv[1]))
Gauge("sglang:num_running_reqs", "Running", multiprocess_mode="mostrecent").set(int(sys.argv[2]))
"""


def _record(directory, *, evicted: int, running: int) -> None:
    subprocess.run(
        [sys.executable, "-c", _RECORD, str(evicted), str(running)],
        env={**os.environ, "PROMETHEUS_MULTIPROC_DIR": str(directory)},
        check=True,
        timeout=60,
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _scrape(port: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
        return response.read().decode()


def _server_args(**overrides):
    args = dict(
        model_path="org/model",
        tokenizer_path=None,
        served_model_name="served",
        host="127.0.0.1",
        port=30011,
        dp_size=1,
        enable_metrics=True,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def _serve(monkeypatch, tmp_path, args, *, during=lambda: None, exit_code: int = 0) -> int:
    """``serve_rust`` with the Rust server, the scheduler launch and the
    supervisor faked; ``during`` runs where the supervisor would."""

    class FakeServer:
        def __init__(self, **kwargs):
            self.address = kwargs["bind_address"]

    servicer = types.ModuleType("smg.servicer")
    servicer.SglangGrpcServer = FakeServer
    servicer.init_servicer_tracing = lambda *_args, **_kwargs: None
    package = types.ModuleType("smg")
    package.servicer = servicer
    monkeypatch.setitem(sys.modules, "smg", package)
    monkeypatch.setitem(sys.modules, "smg.servicer", servicer)
    monkeypatch.setenv(rust.HANDSHAKE_PORT_ENV, "24321")
    monkeypatch.setenv("SMG_ZMQ_SOCKET_DIR", str(tmp_path))
    monkeypatch.setattr(
        rust.SglangLauncher,
        "facts",
        lambda self, args: {"model_path": args.model_path, "data_parallel_size": 1},
    )
    monkeypatch.setattr(rust, "tokenizer_dir_for", lambda args: None)
    monkeypatch.setattr(
        rust, "launch_scheduler_ranks", lambda server_args, *, handshake_port: "engine"
    )

    async def supervise(server, engine, *, drain_secs, **_kwargs):
        during()
        return exit_code

    monkeypatch.setattr(rust_lifecycle, "supervise", supervise)
    return asyncio.run(rust.serve_rust(args))


def test_the_sidecar_port_serves_what_the_scheduler_ranks_record(monkeypatch, tmp_path):
    """Two ranks' records merge into one exposition, read at each scrape: a
    rank that records after the first scrape shows in the next."""
    directory = tmp_path / "metrics"
    directory.mkdir()
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(directory))
    _record(directory, evicted=4, running=5)
    sidecar = _free_port()
    scrapes: list[str] = []

    def during():
        scrapes.append(_scrape(sidecar))
        _record(directory, evicted=3, running=2)
        scrapes.append(_scrape(sidecar))

    _serve(monkeypatch, tmp_path, _server_args(port=sidecar - 1), during=during)

    first, second = scrapes
    assert "sglang:evicted_tokens_total 4.0" in first
    assert "sglang:num_running_reqs 5.0" in first
    assert "sglang:evicted_tokens_total 7.0" in second
    assert "sglang:num_running_reqs 2.0" in second


def _listening(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
    except ConnectionRefusedError:
        return False
    return True


def test_the_sidecar_port_closes_when_the_servicer_stops(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    sidecar = _free_port()
    seen: list[bool] = []

    code = _serve(
        monkeypatch,
        tmp_path,
        _server_args(port=sidecar - 1),
        during=lambda: seen.append(_listening(sidecar)),
        exit_code=3,
    )

    assert seen == [True]
    assert code == 3
    assert not _listening(sidecar)


def test_the_sidecar_port_flag_names_the_port(monkeypatch, tmp_path):
    """``--smg-http-sidecar-port`` moves the sidecar off ``--port + 1``, and the
    scheduler metrics go with it."""
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    _record(tmp_path, evicted=1, running=1)
    sidecar = _free_port()
    scrapes: list[str] = []

    _serve(
        monkeypatch,
        tmp_path,
        _server_args(port=30011, smg_http_sidecar_port=sidecar),
        during=lambda: scrapes.append(_scrape(sidecar)),
    )

    assert "sglang:evicted_tokens_total 1.0" in scrapes[0]


def test_nothing_listens_without_enable_metrics(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    sidecar = _free_port()
    seen: list[bool] = []

    _serve(
        monkeypatch,
        tmp_path,
        _server_args(port=sidecar - 1, enable_metrics=False),
        during=lambda: seen.append(_listening(sidecar)),
    )

    assert seen == [False]


@pytest.mark.parametrize("cause", ["the port is taken", "no metrics directory"])
def test_the_servicer_serves_on_when_the_scheduler_metrics_cannot_be_served(
    monkeypatch, tmp_path, caplog, cause
):
    """As SGLang's sidecar did: a failure to serve the metrics is logged and
    the gRPC servicer runs on."""
    if cause == "the port is taken":
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    else:
        monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        sidecar = taken.getsockname()[1]
        with caplog.at_level(logging.WARNING):
            code = _serve(monkeypatch, tmp_path, _server_args(port=sidecar - 1), exit_code=5)

    assert code == 5
    (warning,) = [r for r in caplog.records if "scheduler metrics" in r.getMessage()]
    assert warning.levelno == logging.WARNING
    assert "are not served" in warning.getMessage()


def test_a_sidecar_port_past_the_tcp_range_is_logged_and_the_servicer_serves_on(
    monkeypatch, tmp_path, caplog
):
    """--port 65535 puts the sidecar's default port past the TCP range."""
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    with caplog.at_level(logging.WARNING):
        code = _serve(monkeypatch, tmp_path, _server_args(port=65535), exit_code=5)

    assert code == 5
    (warning,) = [r for r in caplog.records if "scheduler metrics" in r.getMessage()]
    assert warning.levelno == logging.WARNING
    assert "outside the TCP range" in warning.getMessage()
