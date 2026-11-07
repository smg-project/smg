"""CPU-only checks of the local backend's start when a ZMQ engine fails the
gateway's readiness gate, by dying in it or by running it to its bound: the
failure counts toward the fail-fast counter the pool keeps for workers that
fail their own gate; the gateway's own exit and the other wires' gate failures
do not."""

from importlib import import_module
from types import SimpleNamespace

import pytest
from infra.constants import ENV_STARTUP_TIMEOUT, ConnectionMode
from infra.process_utils import ProcessExitedError

setup = import_module("fixtures.setup_backend")

DEATH = RuntimeError(
    "Worker m (PID 7) died during startup (exit status 1); the log says: "
    '"ERROR mock_worker::zmq: zmq engine 0 handshake failed"; log file worker-m.log'
)
BOUND = TimeoutError("Timeout waiting for 1 worker(s) to connect after 600s")


@pytest.fixture
def local(monkeypatch):
    events = []
    monkeypatch.delenv(ENV_STARTUP_TIMEOUT, raising=False)
    monkeypatch.setattr(setup, "get_model_spec", lambda _: {"tp": 1, "startup_timeout": 10})
    monkeypatch.setattr(setup, "_worker_start_failures", {})
    worker = SimpleNamespace(base_url="ipc:///tmp/engine-1")
    monkeypatch.setattr(setup, "_start_workers_tracked", lambda **kwargs: [worker])
    monkeypatch.setattr(setup, "stop_workers", lambda ws: events.append(("stop", list(ws))))
    monkeypatch.setattr(setup, "set_class_workers", lambda ws: None)
    monkeypatch.setattr(setup, "clear_class_workers", lambda: None)
    monkeypatch.setattr(setup, "_make_openai_client", lambda _: None)
    gateway = SimpleNamespace(base_url="http://test", shutdown=lambda: events.append(("shutdown",)))

    def start(mode, error):
        def gateway_start(*args, **kwargs):
            raise error

        monkeypatch.setattr(setup, "_start_gateway", gateway_start)
        return setup._setup_local(
            "model", "path", "mock", mode, {}, dict(setup._GW_DEFAULTS), gateway, "zmq", None
        )

    return events, worker, start


@pytest.mark.parametrize("error", [DEATH, BOUND], ids=["death", "bound"])
def test_a_zmq_engine_that_fails_the_gateway_gate_counts_as_a_start_failure(local, error):
    events, worker, start = local
    with pytest.raises(type(error)) as caught:
        next(start(ConnectionMode.ZMQ, error))
    assert caught.value is error
    assert setup._worker_start_failures == {"mock": 1}
    # The class's teardown ran: the gateway is down and the engine reaped.
    assert events == [("shutdown",), ("stop", [worker])]


def test_the_gateways_own_exit_is_not_a_worker_start_failure(local):
    events, _, start = local
    with pytest.raises(ProcessExitedError):
        next(start(ConnectionMode.ZMQ, ProcessExitedError("Router at http://test exited")))
    assert setup._worker_start_failures == {}
    assert ("shutdown",) in events


def test_a_gate_failure_on_another_wire_is_the_pools_to_count(local):
    _, _, start = local
    with pytest.raises(RuntimeError):
        next(start(ConnectionMode.GRPC, DEATH))
    assert setup._worker_start_failures == {}
