"""CPU-only encode validation at the real pool and direct startup entrypoints.

Model lookup, ports, eviction and backend startup are stopped by test-owned
sentinels. Command-builder diagnostics live in test_epd_cmd_builders.py.
"""

import re
from types import SimpleNamespace

import pytest
from infra import worker, worker_pool
from infra.constants import ConnectionMode, WorkerType

_INVALID_PAIRS = [(engine, mode) for engine in ("sglang", "vllm") for mode in ConnectionMode] + [
    ("trtllm", ConnectionMode.GRPC),
    ("mlx", ConnectionMode.GRPC),
    ("mock", ConnectionMode.GRPC),
    ("tokenspeed", ConnectionMode.HTTP),
    ("tokenspeed", ConnectionMode.ZMQ),
]


def _pool(cached=True):
    pool = worker_pool.WorkerPool()
    if cached:
        pool._key = ("old-cache",)
        pool._workers = [SimpleNamespace(is_alive=lambda: True)]
        pool._gateway = object()
        pool._gateway_key = ("old-gateway",)
    return pool


def _forbidden(events, stage):
    def fail(*args, **kwargs):
        events.append(stage)
        pytest.fail(f"invalid encoder reached {stage}")

    return fail


def _assert_pool_refusal(monkeypatch, engine, mode, expected, cached=True):
    pool = _pool(cached)
    before = (pool._key, pool._workers, tuple(pool._workers), pool._gateway, pool._gateway_key)
    events = []
    monkeypatch.setattr(pool, "_evict_locked", _forbidden(events, "eviction"))
    monkeypatch.setattr(worker_pool, "start_workers", _forbidden(events, "worker preparation"))
    monkeypatch.setattr(worker, "get_model_spec", _forbidden(events, "model lookup"))
    monkeypatch.setattr(worker, "get_engine_port", _forbidden(events, "port preparation"))
    with pytest.raises(ValueError, match="^" + re.escape(expected) + "$") as caught:
        pool.acquire(
            model_id="never-prepare", engine=engine, mode=mode, worker_type=WorkerType.ENCODE
        )
    assert type(caught.value) is ValueError
    assert events == []
    assert pool._key is before[0]
    assert pool._workers is before[1]
    assert tuple(pool._workers) == before[2]
    assert pool._gateway is before[3]
    assert pool._gateway_key is before[4]


@pytest.mark.parametrize("engine,mode", _INVALID_PAIRS)
def test_invalid_encode_preserves_cached_workers(monkeypatch, engine, mode):
    expected = (
        "EPD encode workers currently require TokenSpeed over gRPC; "
        f"got engine={engine!r}, mode={mode.value!r}"
    )
    _assert_pool_refusal(monkeypatch, engine, mode, expected)


@pytest.mark.parametrize("engine,mode", _INVALID_PAIRS)
def test_invalid_encode_precedes_model_and_port_preparation(monkeypatch, engine, mode):
    expected = (
        "EPD encode workers currently require TokenSpeed over gRPC; "
        f"got engine={engine!r}, mode={mode.value!r}"
    )
    _assert_pool_refusal(monkeypatch, engine, mode, expected, cached=False)


def test_tokenspeed_grpc_encode_retains_uncached_route(monkeypatch):
    pool = _pool()
    events = []
    workers = [object()]
    expected = dict(
        model_id="encode-model",
        engine="tokenspeed",
        mode=ConnectionMode.GRPC,
        count=2,
        worker_type=WorkerType.ENCODE,
        timeout=7,
        log_dir="private-logs",
        gpu_offset=3,
        gpus=4,
        extra_engine_args=["--example"],
        wait_ready=False,
        tp=2,
        kv_backend="mooncake",
        extra_env={"EXAMPLE": "1"},
    )

    def evict():
        events.append("eviction")
        pool._key = None
        pool._workers = []

    def start(**kwargs):
        events.append(("start_workers", kwargs))
        return workers

    monkeypatch.setattr(pool, "_evict_locked", evict)
    monkeypatch.setattr(worker_pool, "start_workers", start)
    assert pool.acquire(**expected) is workers
    assert events == ["eviction", ("start_workers", expected)]


@pytest.mark.parametrize(
    "engine,mode,role",
    [
        ("vllm", ConnectionMode.HTTP, WorkerType.REGULAR),
        ("vllm", ConnectionMode.GRPC, WorkerType.REGULAR),
        ("trtllm", ConnectionMode.GRPC, WorkerType.REGULAR),
        ("mlx", ConnectionMode.HTTP, WorkerType.REGULAR),
        ("mock", ConnectionMode.HTTP, WorkerType.REGULAR),
        ("sglang", ConnectionMode.ZMQ, WorkerType.REGULAR),
        ("sglang", ConnectionMode.GRPC, WorkerType.PREFILL),
        ("vllm", ConnectionMode.HTTP, WorkerType.DECODE),
        ("tokenspeed", ConnectionMode.GRPC, WorkerType.PREFILL),
    ],
)
def test_other_roles_preserve_cache_and_uncached_routes(monkeypatch, engine, mode, role):
    pool = _pool()
    cached = pool._workers[0]
    pool._key = (engine, "control-model", mode, role, 1, None, None)
    events = []
    workers = [object()]

    def evict():
        events.append("eviction")
        pool._key = None
        pool._workers = []

    def start(**kwargs):
        events.append(("start_workers", kwargs["worker_type"]))
        return workers

    monkeypatch.setattr(pool, "_evict_locked", evict)
    monkeypatch.setattr(worker_pool, "start_workers", start)
    result = pool.acquire(model_id="control-model", engine=engine, mode=mode, worker_type=role)
    if role == WorkerType.REGULAR and mode != ConnectionMode.ZMQ:
        assert result == [cached]
        assert events == []
    else:
        assert result is workers
        assert events == ["eviction", ("start_workers", role)]


def _assert_direct_refusal(monkeypatch, engine, resolved, mode, expected):
    events = []

    def runtime():
        events.append("resolve_engine")
        return resolved

    monkeypatch.setattr(worker, "get_gpu_offset", lambda: 0)
    monkeypatch.setattr(worker, "get_runtime", runtime)
    for name in ("get_model_spec", "worker_startup_timeout", "get_engine_port", "Worker"):
        monkeypatch.setattr(worker, name, _forbidden(events, name))
    with pytest.raises(ValueError, match="^" + re.escape(expected) + "$") as caught:
        worker.start_workers(
            "never-prepare",
            engine=engine,
            mode=mode,
            worker_type=WorkerType.ENCODE,
            log_dir="private-logs",
        )
    assert type(caught.value) is ValueError
    assert events == (["resolve_engine"] if engine is None else [])


@pytest.mark.parametrize(
    "engine,resolved,mode",
    [(engine, engine, mode) for engine, mode in _INVALID_PAIRS]
    + [(None, "vllm", ConnectionMode.GRPC), (None, "tokenspeed", ConnectionMode.HTTP)],
)
def test_direct_invalid_encode_precedes_worker_preparation(monkeypatch, engine, resolved, mode):
    expected = (
        "EPD encode workers currently require TokenSpeed over gRPC; "
        f"got engine={resolved!r}, mode={mode.value!r}"
    )
    _assert_direct_refusal(monkeypatch, engine, resolved, mode, expected)


@pytest.mark.parametrize(
    "engine,resolved,mode,role",
    [
        ("tokenspeed", "tokenspeed", ConnectionMode.GRPC, WorkerType.ENCODE),
        (None, "tokenspeed", ConnectionMode.GRPC, WorkerType.ENCODE),
        ("sglang", "sglang", ConnectionMode.GRPC, WorkerType.REGULAR),
        ("vllm", "vllm", ConnectionMode.HTTP, WorkerType.REGULAR),
        ("vllm", "vllm", ConnectionMode.ZMQ, WorkerType.REGULAR),
        ("sglang", "sglang", ConnectionMode.HTTP, WorkerType.PREFILL),
        ("vllm", "vllm", ConnectionMode.GRPC, WorkerType.DECODE),
        ("trtllm", "trtllm", ConnectionMode.GRPC, WorkerType.REGULAR),
        ("mlx", "mlx", ConnectionMode.HTTP, WorkerType.REGULAR),
        ("mock", "mock", ConnectionMode.HTTP, WorkerType.REGULAR),
    ],
)
def test_direct_supported_requests_reach_model_lookup(monkeypatch, engine, resolved, mode, role):
    events = []

    def runtime():
        events.append("resolve_engine")
        return resolved

    def model_lookup(model_id):
        events.append(("model_lookup", model_id))
        raise LookupError("stop at original model lookup sentinel")

    monkeypatch.setattr(worker, "get_gpu_offset", lambda: 0)
    monkeypatch.setattr(worker, "get_runtime", runtime)
    monkeypatch.setattr(worker, "get_model_spec", model_lookup)
    with pytest.raises(LookupError, match="^stop at original model lookup sentinel$"):
        worker.start_workers(
            "supported-control-model",
            engine=engine,
            mode=mode,
            worker_type=role,
            log_dir="private-logs",
        )
    expected = ["resolve_engine"] if engine is None else []
    assert events == expected + [("model_lookup", "supported-control-model")]


@pytest.mark.parametrize(
    "engine,mode,expected_suffix",
    [
        ("sglang", ConnectionMode.GRPC, "got engine='sglang', mode='grpc'"),
        ("tokenspeed", None, "got engine='tokenspeed', mode=None"),
        ("vllm", "bad-wire", "got engine='vllm', mode='bad-wire'"),
    ],
)
def test_pool_diagnostics_preserve_exact_context(monkeypatch, engine, mode, expected_suffix):
    expected = "EPD encode workers currently require TokenSpeed over gRPC; " + expected_suffix
    _assert_pool_refusal(monkeypatch, engine, mode, expected)


@pytest.mark.parametrize(
    "engine,resolved,mode,expected_suffix",
    [
        ("sglang", "sglang", ConnectionMode.GRPC, "got engine='sglang', mode='grpc'"),
        ("tokenspeed", "tokenspeed", None, "got engine='tokenspeed', mode=None"),
        ("vllm", "vllm", "bad-wire", "got engine='vllm', mode='bad-wire'"),
        (None, "vllm", ConnectionMode.GRPC, "got engine='vllm', mode='grpc'"),
    ],
)
def test_direct_diagnostics_preserve_exact_context(
    monkeypatch, engine, resolved, mode, expected_suffix
):
    expected = "EPD encode workers currently require TokenSpeed over gRPC; " + expected_suffix
    _assert_direct_refusal(monkeypatch, engine, resolved, mode, expected)
