"""CPU-only checks for the PD fleet's launch, readiness and cleanup contract."""

from importlib import import_module
from types import SimpleNamespace

import pytest
from infra.constants import ENV_STARTUP_TIMEOUT, ConnectionMode, WorkerType

setup = import_module("fixtures.setup_backend")


@pytest.fixture
def fleet(monkeypatch):
    events = []
    clock = [100.0]
    monkeypatch.delenv(ENV_STARTUP_TIMEOUT, raising=False)
    monkeypatch.setattr(setup, "get_model_spec", lambda _: {"tp": 1, "startup_timeout": 10})
    monkeypatch.setattr(setup.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(setup, "_worker_start_failures", {})

    def ready(worker, timeout):
        events.append(("ready", worker, timeout))
        clock[0] += 1

    def start(**kwargs):
        workers = []
        for index in range(kwargs["count"]):
            worker = SimpleNamespace(gpu_ids=[kwargs["gpu_offset"] + index])
            worker.wait_ready = lambda timeout, worker=worker: ready(worker, timeout)
            worker.stop = lambda worker=worker: events.append(("stop", worker))
            workers.append(worker)
            events.append(("start", worker, kwargs["wait_ready"]))
            clock[0] += 1
            if kwargs["wait_ready"]:
                ready(worker, 10)
        return workers

    monkeypatch.setattr(setup, "_start_workers_tracked", start)
    monkeypatch.setattr(setup, "_start_gateway", lambda *a, **kw: events.append(("gateway",)))
    monkeypatch.setattr(setup, "_wait_for_serving", lambda *a: events.append(("serving",)))
    monkeypatch.setattr(setup, "_make_openai_client", lambda _: None)
    gateway = SimpleNamespace(base_url="http://test", shutdown=lambda: events.append(("shutdown",)))
    config = {**setup._WORKER_DEFAULTS, "prefill": 2, "decode": 2}

    def create(**overrides):
        return setup._setup_pd(
            "model", "path", "vllm", ConnectionMode.HTTP, {**config, **overrides}, {}, gateway, None
        )

    return events, clock, create


def test_pd_default_spawns_all_legs_before_waiting(fleet):
    events, _, create = fleet
    backend = create()
    next(backend)
    assert [event[0] for event in events] == ["start"] * 4 + ["ready"] * 4 + [
        "gateway",
        "serving",
    ]
    assert all(event[2] is False for event in events[:4])
    assert [event[1].gpu_ids for event in events[:4]] == [[0], [1], [2], [3]]
    assert [event[2] for event in events[4:8]] == [10, 9, 8, 7]
    backend.close()
    assert [event[0] for event in events[-5:]] == ["shutdown"] + ["stop"] * 4


def test_pd_explicit_false_retains_serial_escape_hatch(fleet):
    events, _, create = fleet
    backend = create(parallel_start=False)
    next(backend)
    assert [event[0] for event in events[:8]] == ["start", "ready"] * 4
    assert all(event[2] is True for event in events[:8] if event[0] == "start")
    backend.close()


def test_pd_previous_pool_cleanup_does_not_consume_readiness_budget(fleet, monkeypatch):
    events, clock, create = fleet
    start = setup._start_workers_tracked

    def slow_start(**kwargs):
        # Pool acquisition can stop old workers before spawning replacements.
        clock[0] += 80
        workers = start(**kwargs)
        return workers

    monkeypatch.setattr(setup, "_start_workers_tracked", slow_start)
    backend = create()
    next(backend)
    assert len([event for event in events if event[0] == "ready"]) == 4
    assert setup._worker_start_failures == {}
    backend.close()


def test_pd_expired_readiness_deadline_stops_the_fleet(fleet, monkeypatch):
    events, clock, create = fleet
    start = setup._start_workers_tracked

    def slow_ready_start(**kwargs):
        workers = start(**kwargs)

        def ready(timeout):
            events.append(("ready", workers[0], timeout))
            clock[0] += timeout

        workers[0].wait_ready = ready
        return workers

    monkeypatch.setattr(setup, "_start_workers_tracked", slow_ready_start)
    with pytest.raises(TimeoutError):
        next(create())
    assert [event[0] for event in events] == ["start"] * 4 + ["ready", "shutdown"] + ["stop"] * 4
    assert setup._worker_start_failures == {"vllm": 1}


def test_pd_readiness_failure_stops_the_entire_fleet(fleet, monkeypatch):
    events, _, create = fleet
    start = setup._start_workers_tracked

    def failing_start(**kwargs):
        workers = start(**kwargs)
        if kwargs["worker_type"] == WorkerType.DECODE:

            def fail(timeout):
                raise RuntimeError("decode died")

            workers[0].wait_ready = fail
        return workers

    monkeypatch.setattr(setup, "_start_workers_tracked", failing_start)
    with pytest.raises(RuntimeError, match="decode died"):
        next(create())
    assert [event[1] for event in events if event[0] == "stop"] == [
        event[1] for event in events if event[0] == "start"
    ]
    assert setup._worker_start_failures == {"vllm": 1}


def test_partial_pd_leg_failure_stops_previously_started_workers(fleet, monkeypatch):
    events, _, _ = fleet
    start = setup._start_workers_tracked

    def fail_second(**kwargs):
        if kwargs["gpu_offset"] == 1:
            raise RuntimeError("second launch failed")
        return start(**kwargs)

    monkeypatch.setattr(setup, "_start_workers_tracked", fail_second)
    with pytest.raises(RuntimeError, match="second launch failed"):
        setup._start_pd_leg(
            model_id="model",
            engine="vllm",
            mode=ConnectionMode.HTTP,
            count=2,
            worker_type=WorkerType.PREFILL,
            log_dir=None,
            gpu_offset=0,
            wait_ready=False,
            tp=None,
            kv_backends=["nixl", "mooncake"],
        )
    assert [event[0] for event in events] == ["start", "stop"]
    assert events[0][1] is events[1][1]
