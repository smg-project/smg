"""CPU-only tests of the paired ZMQ serving lifetime."""

from types import SimpleNamespace

import pytest
from infra import worker_pool


class _Gateway:
    def __init__(self):
        self.is_running = True
        self.healthy = True
        self.stopped = False
        self.base_url = self

    def start(self, **kwargs):
        self.started_with = kwargs

    def shutdown(self):
        self.stopped = True


@pytest.fixture
def pool(monkeypatch):
    launched, stopped = [], []

    def start_workers(**kwargs):
        worker = SimpleNamespace(alive=True, base_url="zmq://worker")
        worker.is_alive = lambda: worker.alive
        launched.append((kwargs, worker))
        return [worker]

    monkeypatch.setattr(worker_pool, "Gateway", _Gateway)
    monkeypatch.setattr(worker_pool, "start_workers", start_workers)
    monkeypatch.setattr(worker_pool, "stop_workers", lambda workers: stopped.extend(workers))

    def wait_ready(gateway, count, timeout):
        assert timeout == 5
        if not gateway.healthy:
            raise TimeoutError("engine unavailable")

    monkeypatch.setattr(worker_pool, "wait_for_workers_ready", wait_ready)
    cache = worker_pool.WorkerPool()
    yield cache, launched, stopped
    cache.cleanup()


def _acquire(pool, **overrides):
    kwargs = dict(
        model_id="model-id",
        model_path="model-path",
        engine="vllm",
        count=1,
        gpus=2,
        extra_engine_args=["--data-parallel-size", "2"],
        gateway_config={
            "policy": "round_robin",
            "timeout": 600,
            "extra_args": ["--history-backend", "memory"],
            "log_level": None,
            "log_dir": None,
        },
    )
    kwargs.update(overrides)
    return pool.acquire_zmq(**kwargs)


def test_identical_config_keeps_both_gateway_and_engine(pool):
    cache, launched, stopped = pool
    gateway = _acquire(cache)
    assert _acquire(cache) is gateway
    assert len(launched) == 1
    assert stopped == []
    assert gateway.started_with["worker_urls"] == [launched[0][1].base_url]
    cache.cleanup()
    cache.cleanup()
    assert gateway.stopped
    assert stopped == [launched[0][1]]


@pytest.mark.parametrize(
    "change",
    [
        {"model_id": "other-model"},
        {"model_path": "other-path"},
        {"engine": "tokenspeed"},
        {"count": 2},
        {"gpus": 1},
        {"extra_engine_args": ["--data-parallel-size", "1"]},
        {"log_dir": "/another-log-dir"},
    ],
)
def test_worker_configuration_changes_evict_pair(pool, change):
    cache, launched, stopped = pool
    old = _acquire(cache)
    assert _acquire(cache, **change) is not old
    assert old.stopped
    assert stopped == [launched[0][1]]
    assert len(launched) == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("policy", "cache_aware"),
        ("timeout", 900),
        ("extra_args", ["--tool-call-parser", "llama"]),
        ("log_level", "debug"),
        ("log_dir", "/gateway-logs"),
    ],
)
def test_gateway_configuration_changes_evict_pair(pool, field, value):
    cache, _, _ = pool
    old = _acquire(cache)
    config = {**old.started_with, field: value}
    assert _acquire(cache, gateway_config=config) is not old
    assert old.stopped


@pytest.mark.parametrize("failed_component", ["engine", "gateway_process", "gateway_health"])
def test_unhealthy_pair_is_replaced(pool, failed_component):
    cache, launched, stopped = pool
    old = _acquire(cache)
    if failed_component == "engine":
        launched[0][1].alive = False
    elif failed_component == "gateway_process":
        old.is_running = False
    else:
        old.healthy = False
    assert _acquire(cache) is not old
    assert old.stopped
    assert stopped == [launched[0][1]]


@pytest.mark.parametrize("mode", [worker_pool.ConnectionMode.HTTP, worker_pool.ConnectionMode.ZMQ])
def test_other_acquisition_releases_cached_pair_before_claiming_gpus(pool, mode):
    cache, launched, stopped = pool
    old = _acquire(cache)
    cache.acquire(model_id="model-id", engine="vllm", mode=mode)
    assert old.stopped
    assert stopped == [launched[0][1]]
    assert len(launched) == 2


def test_gateway_start_failure_releases_workers_and_leaves_slot_empty(pool, monkeypatch):
    cache, launched, stopped = pool

    def fail_start(self, **kwargs):
        raise TimeoutError("gateway did not become ready")

    monkeypatch.setattr(_Gateway, "start", fail_start)
    with pytest.raises(TimeoutError, match="ready"):
        _acquire(cache)
    assert stopped == [launched[0][1]]
    assert cache._key is None
    assert cache._gateway is None


def test_failed_class_discards_only_its_pair(pool):
    cache, _, stopped = pool
    old = _acquire(cache)
    newer = _acquire(cache, model_id="another-model")
    cache.discard_zmq(old)
    assert _acquire(cache, model_id="another-model") is newer
    cache.discard_zmq(newer)
    assert newer.stopped
    assert len(stopped) == 2


def test_closed_pool_cannot_acquire_zmq(pool):
    cache, _, _ = pool
    cache.cleanup()
    with pytest.raises(RuntimeError, match="closed"):
        _acquire(cache)
