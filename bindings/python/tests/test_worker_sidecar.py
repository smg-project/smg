"""Behavior of the standalone two-tier Worker sidecar process."""

from __future__ import annotations

import importlib
import logging
import signal
import sys
import types
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

_BASE = [
    "--bind-address",
    "127.0.0.1:0",
    "--worker-id",
    "w0",
    "--engine-endpoint",
    "grpc://127.0.0.1:31000",
    "--model-id",
    "org/model",
]


@dataclass
class Harness:
    module: types.ModuleType = field(init=False)
    servers: list = field(default_factory=list)
    tracing_levels: list = field(default_factory=list)
    events: list = field(default_factory=list)
    # `last_error` value returned by successive polls (the last one repeats).
    errors: list = field(default_factory=lambda: [None])
    # Called with the poll count before each `last_error` read.
    on_poll: object = None


class FakeServer:
    """Records the lifecycle calls the sidecar makes on the Rust server."""

    def __init__(self, harness: Harness, **kwargs):
        self.harness = harness
        self.kwargs = kwargs
        self.polls = 0

    @property
    def last_error(self):
        self.polls += 1
        if self.harness.on_poll is not None:
            self.harness.on_poll(self.polls)
        errors = self.harness.errors
        return errors[min(self.polls, len(errors)) - 1]

    def set_health(self, state, message):
        self.harness.events.append(("health", state, message))

    def stop(self, timeout_secs):
        self.harness.events.append(("stop", timeout_secs))


@pytest.fixture
def sidecar(monkeypatch):
    """Import the sidecar against a stand-in for the native extension module."""
    harness = Harness()

    def make_server(**kwargs):
        server = FakeServer(harness, **kwargs)
        harness.servers.append(server)
        return server

    worker_module = types.ModuleType("smg.worker")
    worker_module.WorkerControlServer = make_server
    worker_module.init_tracing = harness.tracing_levels.append
    monkeypatch.setitem(sys.modules, "smg.worker", worker_module)
    sys.modules.pop("smg.worker_sidecar", None)
    harness.module = importlib.import_module("smg.worker_sidecar")
    monkeypatch.setattr(harness.module, "_HEALTH_POLL_SECS", 0)
    yield harness
    sys.modules.pop("smg.worker_sidecar", None)


def _run(harness: Harness, argv: list[str]) -> None:
    with patch.object(
        harness.module.time,
        "sleep",
        side_effect=lambda secs: harness.events.append(("sleep", secs)),
    ):
        harness.module.main(argv)


@pytest.mark.parametrize("transport", ["grpc", "zmq"])
def test_sglang_is_not_an_accepted_engine_type(sidecar, transport):
    """A Worker fronts vllm or tokenspeed only; argparse refuses anything else."""
    with pytest.raises(SystemExit) as exit_info:
        sidecar.module.main([*_BASE, "--engine-type", "sglang", "--engine-transport", transport])

    assert exit_info.value.code == 2
    assert sidecar.servers == []


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--drain-secs", "-1"], "drain-secs"),
        (["--engine-count", "0"], "engine-count"),
        (["--max-concurrent-requests", "-1"], "max-concurrent-requests"),
    ],
)
def test_invalid_numeric_arguments_fail_before_the_server_starts(sidecar, extra, message):
    with pytest.raises(ValueError, match=message):
        sidecar.module.main([*_BASE, "--engine-type", "vllm", *extra])

    assert sidecar.servers == []


def test_server_contract(sidecar):
    """Every entry point builds the same WorkerControlServer: inference on, the
    first model id doubling as the engine model and tokenizer path."""
    sidecar.errors = ["engine transport failed"]
    with pytest.raises(SystemExit):
        _run(
            sidecar,
            [
                *_BASE,
                "--engine-type",
                "tokenspeed",
                "--engine-transport",
                "zmq",
                "--zmq-handshake-address",
                "tcp://127.0.0.1:30500",
                "--engine-count",
                "2",
                "--model-id",
                "org/alias",
                "--max-concurrent-requests",
                "128",
            ],
        )

    (server,) = sidecar.servers
    assert server.kwargs == {
        "bind_address": "127.0.0.1:0",
        "worker_id": "w0",
        "engine_type": "tokenspeed",
        "hostname": server.kwargs["hostname"],
        "engine_endpoint": "grpc://127.0.0.1:31000",
        "model_ids": ["org/model", "org/alias"],
        "features": ["generate", "stream", "abort"],
        "max_concurrent_requests": 128,
        "inference_enabled": True,
        "engine_attributes": {"model_path": "org/model", "tokenizer_path": "org/model"},
        "engine_transport": "zmq",
        "zmq_handshake_address": "tcp://127.0.0.1:30500",
        "engine_count": 2,
    }
    assert server.kwargs["hostname"]


@pytest.mark.parametrize(
    ("flags", "tracing_level"),
    [([], "info"), (["--log-level", "debug"], "debug"), (["--log-level", "warning"], "warn")],
)
def test_rust_tracing_follows_the_log_level(sidecar, flags, tracing_level):
    sidecar.errors = ["engine transport failed"]
    with pytest.raises(SystemExit):
        _run(sidecar, [*_BASE, "--engine-type", "vllm", *flags])

    assert sidecar.tracing_levels == [tracing_level]


def test_engine_transport_failure_exits_nonzero_after_marking_not_serving(sidecar, caplog):
    """`last_error` is terminal: NOT_SERVING with the reason, listener stopped,
    exit status 1, no drain."""
    caplog.set_level(logging.INFO, logger="smg.worker_sidecar")
    sidecar.errors = [None, None, "zmq handshake timed out"]

    with pytest.raises(SystemExit) as exit_info:
        _run(sidecar, [*_BASE, "--engine-type", "vllm", "--engine-transport", "zmq"])

    assert exit_info.value.code == 1
    (server,) = sidecar.servers
    assert server.polls == 3
    assert sidecar.events == [
        ("health", "serving", "ready"),
        ("health", "not_serving", "zmq handshake timed out"),
        ("stop", 1.0),
    ]
    startup = caplog.records[0]
    assert startup.levelno == logging.INFO
    for detail in ("w0", "127.0.0.1:0", "vllm", "zmq", "grpc://127.0.0.1:31000"):
        assert detail in startup.getMessage()
    failure = caplog.records[-1]
    assert failure.levelno == logging.ERROR
    assert "zmq handshake timed out" in failure.getMessage()


def test_sigterm_drains_then_stops_in_order(sidecar, caplog):
    """SIGTERM: DRAINING at once, `--drain-secs` for active streams, then
    NOT_SERVING and the listener stop. A repeated signal changes nothing."""
    caplog.set_level(logging.INFO, logger="smg.worker_sidecar")

    def deliver_sigterm(poll_count):
        if poll_count == 2:
            handlers = dict(call.args for call in signaller.call_args_list)
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            handlers[signal.SIGTERM](signal.SIGTERM, None)

    sidecar.on_poll = deliver_sigterm
    with patch.object(sidecar.module.signal, "signal") as signaller:
        _run(sidecar, [*_BASE, "--engine-type", "vllm", "--drain-secs", "2.5"])

    assert {call.args[0] for call in signaller.call_args_list} == {signal.SIGINT, signal.SIGTERM}
    (server,) = sidecar.servers
    assert server.polls == 2
    assert sidecar.events == [
        ("health", "serving", "ready"),
        ("health", "draining", "draining"),
        ("sleep", 2.5),
        ("health", "not_serving", "stopped"),
        ("stop", 2.5),
    ]
    messages = [record.getMessage() for record in caplog.records]
    assert any("received SIGTERM" in message for message in messages)
    assert "SMG Worker w0 stopped after draining" in messages[-1]


def test_listener_stop_gets_at_least_one_second_after_a_short_drain(sidecar):
    def deliver_sigint(poll_count):
        handlers = dict(call.args for call in signaller.call_args_list)
        handlers[signal.SIGINT](signal.SIGINT, None)

    sidecar.on_poll = deliver_sigint
    with patch.object(sidecar.module.signal, "signal") as signaller:
        _run(sidecar, [*_BASE, "--engine-type", "vllm", "--drain-secs", "0.5"])

    assert sidecar.events[-3:] == [
        ("sleep", 0.5),
        ("health", "not_serving", "stopped"),
        ("stop", 1.0),
    ]
