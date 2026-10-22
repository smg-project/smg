"""Actual fixture/pool/worker allocation with CPU-only lifecycle substitutes."""

import importlib
import itertools
from types import SimpleNamespace

import pytest
from infra import pd_fixture_observation as observation
from infra import worker, worker_pool
from infra.constants import ConnectionMode, WorkerType
from infra.pd_worker_observation import WorkerObservation

backend = importlib.import_module("fixtures.setup_backend")


@pytest.fixture
def fleet(monkeypatch, tmp_path):
    launched, stopped, calls = [], [], []
    monkeypatch.setattr(backend, "_worker_start_failures", {})
    spec = {"model": "/model", "tp": 1}
    for module in (backend, worker, observation):
        monkeypatch.setattr(module, "get_model_spec", lambda _: spec)
    monkeypatch.setattr(worker, "get_gpu_offset", lambda: 0)
    monkeypatch.setattr(worker, "detect_ib_device", lambda: None)
    monkeypatch.setattr(worker, "LAUNCH_STAGGER_DELAY", 0)
    ports = itertools.count(50000)
    monkeypatch.setattr(worker, "get_open_port", lambda: next(ports))
    monkeypatch.setattr(worker, "get_open_port_block", lambda _: next(ports))
    monkeypatch.setattr(worker, "reserve_port", lambda _: None)
    monkeypatch.setattr(worker.Worker, "wait_ready", lambda *a: None)

    def start(w, **kwargs):
        launched.append(w)
        if state.fail and w.worker_type == WorkerType.DECODE:
            raise RuntimeError("CPU startup failure")
        if w.scheduler_observation:
            env = w.scheduler_observation.child_environment(
                {}, engine=w.engine, mode=w.mode, worker_type=w.worker_type, command=w._build_cmd()
            )
            w.observer_instance = env["SMG_PD_TRACE_WORKER_INSTANCE"]

    def start_workers(**kwargs):
        calls.append(kwargs)
        return worker.start_workers(**kwargs)

    monkeypatch.setattr(worker.Worker, "start", start)
    monkeypatch.setattr(worker_pool, "start_workers", start_workers)
    for module in (worker, backend):
        monkeypatch.setattr(module, "stop_workers", lambda workers: stopped.extend(workers))
    monkeypatch.setattr(backend, "get_pool", lambda: state.pool)
    monkeypatch.setattr(backend, "_wait_for_serving", lambda *a: None)
    monkeypatch.setattr(backend, "_make_openai_client", lambda *a: "CPU client")
    monkeypatch.setattr(backend, "_start_gateway", lambda *a, **k: None)
    monkeypatch.setattr(backend, "_register_moriio_workers", lambda *a: None)
    state = SimpleNamespace(
        pool=worker_pool.WorkerPool(),
        launched=launched,
        stopped=stopped,
        calls=calls,
        spec=spec,
        fail=False,
    )
    state.config = {"run": "run", "attempt": "caller", "trace_dir": tmp_path}
    return state


def setup(fleet, **options):
    gateway = SimpleNamespace(shutdown=lambda: None, base_url="http://cpu")
    config = {
        "prefill": 2,
        "decode": 2,
        "parallel_start": True,
        "scheduler_observation": fleet.config,
        **options,
    }
    return backend._setup_pd("m", "/model", "vllm", ConnectionMode.HTTP, config, {}, gateway, None)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"parallel_start": False},
        {"prefill_kv": ["nixl", "mooncake"], "decode_kv": ["nixl", "mooncake"]},
    ],
)
def test_actual_fixture_pool_chain_assigns_all_ordinals_and_one_fleet_attempt(fleet, options):
    gen = setup(fleet, **options)
    next(gen)
    contexts = [w.scheduler_observation for w in fleet.launched]
    assert [c.worker for c in contexts] == ["prefill-0", "prefill-1", "decode-0", "decode-1"]
    assert len({c.attempt for c in contexts}) == 1 and contexts[0].attempt.startswith(
        "caller-fixture-"
    )
    assert {c.run for c in contexts} == {"run"}
    assert [w.gpu_ids for w in fleet.launched] == [[0], [1], [2], [3]]
    first = fleet.launched[0]
    instance = first.observer_instance
    first.start(wait_ready=False)
    assert first.observer_instance != instance and first.scheduler_observation is contexts[0]
    gen.close()
    assert {id(w) for w in fleet.launched} <= {id(w) for w in fleet.stopped}


def test_default_kwargs_and_new_fixture_attempts_remain_separate(fleet):
    gen = setup(fleet, scheduler_observation=None)
    next(gen)
    gen.close()
    assert all("scheduler_observations" not in call for call in fleet.calls)
    assert all(w.scheduler_observation is None for w in fleet.launched)
    fleet.launched.clear()
    attempts = []
    for _ in range(2):
        gen = setup(fleet)
        next(gen)
        attempts.append(fleet.launched[-1].scheduler_observation.attempt)
        gen.close()
    assert len(set(attempts)) == 2 and fleet.config["attempt"] == "caller"


@pytest.mark.parametrize(
    "options",
    [
        {"decode_args": ["--kv_transfer_config.kv_role=kv_producer"]},
        {"decode_args": ["--data_parallel_size", "2"]},
        {"decode_args": ["--pipeline-parallel-size=2"]},
        {"decode_tp": 2},
        {"decode_env": {"VLLM_DP_SIZE": "2"}},
        {"scheduler_observation": False},
        {"scheduler_observation": {}},
        {"decode_args": "--bad"},
        {"decode_env": {"bad": 1}},
    ],
)
def test_both_pools_preflight_before_first_worker(fleet, options):
    with pytest.raises(ValueError):
        next(setup(fleet, **options))
    assert fleet.launched == []


def test_model_parallel_override_is_not_hidden_from_preflight(fleet):
    fleet.spec["vllm_args"] = ["--decode-context-parallel-size=2"]
    with pytest.raises(ValueError, match="parallelism"):
        next(setup(fleet))
    assert fleet.launched == []


def test_moriio_effective_http_is_resolved_before_validation(fleet):
    config = {
        "prefill": 1,
        "decode": 1,
        "scheduler_observation": fleet.config,
        "prefill_kv": ["moriio"],
        "decode_kv": ["moriio"],
    }
    gen = backend._setup_pd(
        "m",
        "/model",
        "vllm",
        ConnectionMode.GRPC,
        config,
        {},
        SimpleNamespace(shutdown=lambda: None, base_url="http://cpu"),
        None,
    )
    assert next(gen)[0] == "pd_http"
    assert len(fleet.launched) == 2 and all(w.mode == ConnectionMode.HTTP for w in fleet.launched)
    gen.close()


def test_startup_exception_cleans_prefill_and_partial_decode(fleet):
    fleet.fail = True
    with pytest.raises(RuntimeError, match="CPU startup failure"):
        next(setup(fleet))
    assert {id(w) for w in fleet.launched} <= {id(w) for w in fleet.stopped}


def test_regular_pool_rejects_before_evicting_existing_workers(fleet):
    fleet.pool._key = ("existing",)
    with pytest.raises(ValueError):
        fleet.pool.acquire(
            model_id="m",
            engine="vllm",
            scheduler_observations=[WorkerObservation("r", "a", "w", fleet.config["trace_dir"])],
        )
    assert fleet.pool._key == ("existing",) and fleet.calls == []


@pytest.mark.parametrize("kind", ["count", "duplicate", "attempt"])
def test_direct_start_workers_rejects_ambiguous_context_lists(fleet, kind):
    a = WorkerObservation("r", "a", "w", fleet.config["trace_dir"])
    other = WorkerObservation("r", "b", "other", a.trace_dir)
    contexts = {"count": [a], "duplicate": [a, a], "attempt": [a, other]}[kind]
    with pytest.raises(ValueError):
        worker.start_workers(
            "m",
            engine="vllm",
            count=2,
            worker_type=WorkerType.PREFILL,
            scheduler_observations=contexts,
        )
    assert fleet.launched == []
