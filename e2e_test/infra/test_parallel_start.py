"""CPU-only checks for worker launch ordering, startup bounds and cleanup."""

from itertools import count

import pytest
from infra import constants as constants_module
from infra import worker as worker_module
from infra.constants import ENV_STARTUP_TIMEOUT, ConnectionMode, WorkerType
from infra.startup_profile_test_support import set_profile_lane


@pytest.fixture
def launch(monkeypatch):
    events = []
    clock = [100.0]
    ports = count(20000)
    monkeypatch.delenv(ENV_STARTUP_TIMEOUT, raising=False)
    monkeypatch.delenv("E2E_STARTUP_PROFILE", raising=False)
    monkeypatch.setattr(constants_module, "_STARTUP_PROFILE_CONTEXT", None)
    monkeypatch.setattr(constants_module, "_STARTUP_PROFILE_SELECTION", None)
    monkeypatch.setattr(worker_module, "get_gpu_offset", lambda: 0)
    monkeypatch.setattr(worker_module, "get_model_spec", lambda _: {"tp": 1})
    monkeypatch.setattr(worker_module, "detect_ib_device", lambda: None)
    monkeypatch.setattr(worker_module, "get_engine_port", lambda: next(ports))
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


def test_every_port_the_engine_binds_comes_from_the_engines_reservation(launch, monkeypatch):
    """The worker's port, a prefill's bootstrap port and a disaggregated
    TokenSpeed worker's dist-init port are bound by the engine after its model
    load: all three come from the reservation below the kernel's ephemeral
    range, not from a ``bind(0)`` number any process can be handed meanwhile."""
    monkeypatch.setattr(
        worker_module, "get_open_port", lambda *a, **k: pytest.fail("bind(0) port for an engine")
    )
    (worker,) = worker_module.start_workers(
        "model", engine="tokenspeed", worker_type=WorkerType.PREFILL, wait_ready=False
    )
    assert (worker.port, worker.bootstrap_port, worker.dist_init_port) == (20000, 20001, 20002)


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


@pytest.mark.parametrize(
    "profile,model,tp,bound,mode",
    [
        ("vllm-zmq-dp2-chat", "meta-llama/Llama-3.1-8B-Instruct", 1, 300, ConnectionMode.ZMQ),
        (
            "vllm-4gpu-chat",
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            4,
            1200,
            ConnectionMode.GRPC,
        ),
        ("vllm-4gpu-chat", "openai/gpt-oss-120b", 4, 600, ConnectionMode.GRPC),
        (
            "original1-vllm-chat-legacy",
            "meta-llama/Llama-3.2-1B-Instruct",
            1,
            300,
            ConnectionMode.GRPC,
        ),
    ],
)
def test_profile_value_reaches_worker_spawn_and_wait_arguments(
    launch,
    monkeypatch,
    profile,
    model,
    tp,
    bound,
    mode,
):
    events, _ = launch
    set_profile_lane(monkeypatch, profile)
    monkeypatch.setattr(
        worker_module, "get_model_spec", lambda _: {"tp": tp, "startup_timeout": 9999}
    )
    workers = worker_module.start_workers(model, engine="vllm", mode=mode)
    assert [event[0] for event in events] == ["start", "ready"]
    assert [event[2] for event in events] == [bound, bound]
    assert workers[0].gpu_ids == list(range(tp * (2 if mode == ConnectionMode.ZMQ else 1)))


@pytest.mark.parametrize(
    "options",
    [
        {"engine": "sglang"},
        {"mode": ConnectionMode.HTTP},
        {"tp": 2},
        {"gpus": 1},
        {"count": 2},
        {"count": True},
        {"count": 1.0},
        {"count": 0},
        {"worker_type": WorkerType.PREFILL},
        {"extra_engine_args": ["--tensor-parallel-size", "2"]},
        {"extra_engine_args": ["--tensor-parallel-size=2"]},
        {"extra_engine_args": ["-tp2"]},
        {"extra_engine_args": ["--data-parallel-size=1"]},
        {"extra_engine_args": ["--tensor_parallel_size=2"]},
        {"extra_engine_args": ["--data_parallel_size=1"]},
        {"extra_engine_args": ["--model", "different-model"]},
        {"extra_engine_args": ["--config=override.yaml"]},
        {"extra_env": {"SMG_VLLM_SERVICER_WARMUP": "off"}},
    ],
)
def test_profile_conflict_fails_before_any_fake_spawn(launch, monkeypatch, options):
    events, _ = launch
    set_profile_lane(monkeypatch, "vllm-zmq-dp2-chat")
    kwargs = {"engine": "vllm", "mode": ConnectionMode.ZMQ, **options}
    with pytest.raises(ValueError):
        worker_module.start_workers("meta-llama/Llama-3.1-8B-Instruct", **kwargs)
    assert events == []


def test_profile_readiness_failure_still_cleans_up(launch, monkeypatch):
    events, _ = launch
    set_profile_lane(monkeypatch, "vllm-4gpu-chat")
    monkeypatch.setattr(worker_module, "get_model_spec", lambda _: {"tp": 4})

    def fail(worker, timeout):
        assert timeout == 600
        raise TimeoutError("profile readiness failed")

    monkeypatch.setattr(worker_module.Worker, "wait_ready", fail)
    with pytest.raises(TimeoutError, match="profile readiness failed"):
        worker_module.start_workers("openai/gpt-oss-120b", engine="vllm", mode=ConnectionMode.GRPC)
    assert [event[0] for event in events] == ["start", "stop"]


def test_original1_exact_full_warmup_call_spawns_without_health_wait(launch, monkeypatch):
    events, _ = launch
    set_profile_lane(monkeypatch, "original1-vllm-chat-legacy")
    workers = worker_module.start_workers(
        "meta-llama/Llama-3.2-1B-Instruct",
        "vllm",
        mode=ConnectionMode.GRPC,
        count=1,
        wait_ready=False,
        extra_engine_args=[],
        extra_env={"SMG_VLLM_SERVICER_WARMUP": "full"},
    )
    assert [event[0] for event in events] == ["start"]
    assert events[0][2:] == (300, False)
    assert workers[0].extra_env == {"SMG_VLLM_SERVICER_WARMUP": "full"}
    assert workers[0].gpu_ids == [0]


@pytest.mark.parametrize(
    "change",
    [
        {"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
        {"mode": ConnectionMode.HTTP},
        {"mode": ConnectionMode.ZMQ},
        {"engine": "sglang"},
        {"worker_type": WorkerType.PREFILL},
        {"count": 2},
        {"tp": 2},
        {"gpus": 2},
        {"kv_backend": "nixl"},
        {"wait_ready": True},
        {"extra_engine_args": ["--log-level", "info"]},
        {"extra_env": {"SMG_VLLM_SERVICER_WARMUP": "fast"}},
        {"extra_env": {"SMG_VLLM_SERVICER_WARMUP": "off"}},
        {"extra_env": {"SMG_VLLM_SERVICER_WARMUP": "full", "E2E_PARSER": "legacy"}},
        {"extra_env": {"SMG_VLLM_SERVICER_WARMUP": "full", "UNRELATED": "value"}},
    ],
)
def test_original1_full_warmup_exception_rejects_changed_shape(launch, monkeypatch, change):
    events, _ = launch
    set_profile_lane(monkeypatch, "original1-vllm-chat-legacy")
    options = {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "engine": "vllm",
        "mode": ConnectionMode.GRPC,
        "count": 1,
        "wait_ready": False,
        "extra_engine_args": [],
        "extra_env": {"SMG_VLLM_SERVICER_WARMUP": "full"},
    }
    options.update(change)
    with pytest.raises(ValueError):
        worker_module.start_workers(**options)
    assert events == []


def test_other_startup_profile_still_rejects_full_warmup_override(launch, monkeypatch):
    events, _ = launch
    set_profile_lane(monkeypatch, "vllm-mooncake-pd2")
    with pytest.raises(ValueError, match="worker environment override"):
        worker_module.start_workers(
            "meta-llama/Llama-3.2-1B-Instruct",
            "vllm",
            mode=ConnectionMode.GRPC,
            count=1,
            wait_ready=False,
            extra_engine_args=[],
            extra_env={"SMG_VLLM_SERVICER_WARMUP": "full"},
        )
    assert events == []
