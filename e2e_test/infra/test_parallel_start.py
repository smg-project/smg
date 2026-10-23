"""CPU-only checks for worker launch ordering, startup bounds and cleanup."""

from itertools import count

import pytest
from infra import worker as worker_module
from infra.constants import ENV_STARTUP_TIMEOUT, WorkerType


@pytest.fixture
def launch(monkeypatch):
    events = []
    clock = [100.0]
    ports = count(20000)
    monkeypatch.delenv(ENV_STARTUP_TIMEOUT, raising=False)
    monkeypatch.setattr(worker_module, "get_gpu_offset", lambda: 0)
    monkeypatch.setattr(worker_module, "get_model_spec", lambda _: {"tp": 1})
    monkeypatch.setattr(worker_module, "detect_ib_device", lambda: None)
    monkeypatch.setattr(worker_module, "get_open_port", lambda: next(ports))
    monkeypatch.setattr(worker_module, "LAUNCH_STAGGER_DELAY", 0)
    monkeypatch.setattr(worker_module.time, "monotonic", lambda: clock[0])

    def start(worker, timeout, wait_ready):
        events.append(("start", worker, timeout, wait_ready))
        clock[0] += 1

    def ready(worker, timeout):
        events.append(("ready", worker, timeout))
        clock[0] += 4

    monkeypatch.setattr(worker_module.Worker, "start", start)
    monkeypatch.setattr(worker_module.Worker, "wait_ready", ready)
    monkeypatch.setattr(
        worker_module.Worker, "stop", lambda worker: events.append(("stop", worker))
    )
    return events, clock


def test_workers_load_before_any_readiness_wait(launch):
    events, _ = launch
    workers = worker_module.start_workers("model", engine="vllm", count=2, timeout=10)

    assert [event[0] for event in events] == ["start", "start", "ready", "ready"]
    assert [event[1] for event in events[:2]] == workers
    assert all(event[3] is False for event in events[:2])
    assert [event[2] for event in events[2:]] == [10, 6]


def test_gpu_assignment_and_launch_configuration_are_preserved(launch, monkeypatch):
    monkeypatch.setattr(worker_module, "get_gpu_offset", lambda: 5)
    workers = worker_module.start_workers(
        "model",
        engine="vllm",
        count=2,
        gpu_offset=2,
        gpus=3,
        tp=2,
        worker_type=WorkerType.PREFILL,
        kv_backend="nixl",
        extra_engine_args=["--max-model-len", "2048"],
        extra_env={"SMG_PAIRING_PROTOCOL": "test"},
    )

    assert [worker.gpu_ids for worker in workers] == [[7, 8, 9], [10, 11, 12]]
    assert len({worker.port for worker in workers}) == 2
    assert len({worker.bootstrap_port for worker in workers}) == 2
    for worker in workers:
        assert worker.tp == 2
        assert worker.worker_type == WorkerType.PREFILL
        assert worker.kv_backend == "nixl"
        assert worker.extra_engine_args == ["--max-model-len", "2048"]
        assert worker.extra_env == {"SMG_PAIRING_PROTOCOL": "test"}


def test_launch_stagger_remains_but_does_not_wait_for_health(launch, monkeypatch):
    events, clock = launch

    def sleep(seconds):
        events.append(("sleep", seconds))
        clock[0] += seconds

    monkeypatch.setattr(worker_module, "LAUNCH_STAGGER_DELAY", 2)
    monkeypatch.setattr(worker_module.time, "sleep", sleep)
    worker_module.start_workers("model", engine="vllm", count=2, timeout=10)

    assert [event[0] for event in events] == ["start", "sleep", "start", "ready", "ready"]
    assert [event[2] for event in events if event[0] == "ready"] == [10, 6]


def test_later_workers_get_the_full_load_budget_after_staggering(launch, monkeypatch):
    _, clock = launch

    def start(worker, timeout, wait_ready):
        worker.ready_at = clock[0] + 295

    def ready(worker, timeout):
        if worker.ready_at - clock[0] > timeout:
            raise TimeoutError("load budget shortened by launch staggering")
        clock[0] = max(clock[0], worker.ready_at)

    def sleep(seconds):
        clock[0] += seconds

    monkeypatch.setattr(worker_module.Worker, "start", start)
    monkeypatch.setattr(worker_module.Worker, "wait_ready", ready)
    monkeypatch.setattr(worker_module, "LAUNCH_STAGGER_DELAY", 10)
    monkeypatch.setattr(worker_module.time, "sleep", sleep)
    workers = worker_module.start_workers("model", engine="vllm", count=4, timeout=300)
    assert len(workers) == 4
    assert clock[0] == 425


def test_spawn_only_leaves_readiness_to_the_caller(launch):
    events, _ = launch
    worker_module.start_workers("model", engine="vllm", count=2, wait_ready=False)
    assert [event[0] for event in events] == ["start", "start"]


@pytest.mark.parametrize("phase", ["start", "ready"])
def test_failure_stops_every_attempted_worker(launch, monkeypatch, phase):
    events, _ = launch
    method = "start" if phase == "start" else "wait_ready"
    original = getattr(worker_module.Worker, method)
    calls = []

    def fail_second(worker, *args, **kwargs):
        calls.append(worker)
        original(worker, *args, **kwargs)
        if len(calls) == 2:
            raise RuntimeError("failed launch")

    monkeypatch.setattr(worker_module.Worker, method, fail_second)
    with pytest.raises(RuntimeError, match="failed launch"):
        worker_module.start_workers("model", engine="vllm", count=2)

    attempted = [event[1] for event in events if event[0] == "start"]
    stopped = [event[1] for event in events if event[0] == "stop"]
    assert stopped == attempted
    assert len(stopped) == 2


def test_expired_fleet_deadline_cleans_up_without_another_wait(launch, monkeypatch):
    events, clock = launch

    def ready(worker, timeout):
        events.append(("ready", worker, timeout))
        clock[0] += timeout

    monkeypatch.setattr(worker_module.Worker, "wait_ready", ready)
    with pytest.raises(TimeoutError, match="within 5s"):
        worker_module.start_workers("model", engine="vllm", count=2, timeout=5)

    assert [event[0] for event in events] == ["start", "start", "ready", "stop", "stop"]
    assert events[2][2] == 5


def test_model_timeout_and_environment_floor_apply_to_whole_fleet(launch, monkeypatch):
    events, _ = launch
    monkeypatch.setattr(worker_module, "get_model_spec", lambda _: {"tp": 1, "startup_timeout": 20})
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, "30")
    worker_module.start_workers("model", engine="vllm", count=2, timeout=10)
    assert [event[2] for event in events[:2]] == [30, 30]
    assert [event[2] for event in events[2:]] == [30, 26]
