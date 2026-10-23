"""Real fixture/pool/Worker construction; process starts and health are not executed."""

from importlib import import_module
from itertools import count
from types import SimpleNamespace

import pytest
from infra import worker as worker_module
from infra.constants import ConnectionMode, WorkerType
from infra.worker import Worker
from infra.worker_pool import WorkerPool

setup = import_module("fixtures.setup_backend")
MODEL = "Qwen/Qwen3.5-9B"
RECEIVER_MODE = "SGLANG_ENCODER_MM_RECEIVER_MODE"


@pytest.fixture
def fleet(monkeypatch):
    workers = []
    ports = count(50100)
    pool = WorkerPool()
    monkeypatch.setattr(setup, "get_pool", lambda: pool)
    monkeypatch.setattr(worker_module, "get_engine_port", lambda: next(ports))
    monkeypatch.setattr(worker_module, "detect_ib_device", lambda: None)
    monkeypatch.setattr(worker_module, "LAUNCH_STAGGER_DELAY", 0)
    monkeypatch.delenv(RECEIVER_MODE, raising=False)

    def construct(worker, **kwargs):
        worker.command = worker._build_cmd()
        worker.environment = worker._build_env()
        workers.append(worker)

    monkeypatch.setattr(Worker, "start", construct)
    monkeypatch.setattr(Worker, "wait_ready", lambda *a: None)
    monkeypatch.setattr(setup, "_start_gateway", lambda *a, **kw: None)
    monkeypatch.setattr(setup, "_register_sglang_epd_encoders", lambda *a: None)
    monkeypatch.setattr(setup, "_make_openai_client", lambda _: None)
    gateway = SimpleNamespace(base_url="http://not-started", shutdown=lambda: None)

    def create(engine, topology=(1, 1, 1)):
        backend = setup._setup_epd(
            MODEL,
            MODEL,
            engine,
            ConnectionMode.GRPC,
            topology,
            dict(setup._GW_DEFAULTS),
            gateway,
            None,
        )
        try:
            next(backend)
        finally:
            backend.close()
        assert all(w.process is None for w in workers)
        return workers

    return create


@pytest.mark.parametrize("topology", [(1, 1, 1), (1, 2, 1), (2, 1, 1), (1, 1, 2)])
def test_sglang_epd_receives_external_embeddings_without_local_vision(fleet, topology):
    workers = fleet("sglang", topology)
    assert len(workers) == sum(topology)
    assert [w.gpu_ids for w in workers] == [[i] for i in range(sum(topology))]
    for worker in workers:
        cmd = worker.command
        assert "--language-model-only" not in cmd
        if worker.worker_type == WorkerType.ENCODE:
            assert "--encoder-only" in cmd and "--language-only" not in cmd
            assert RECEIVER_MODE not in worker.environment
        else:
            assert "--language-only" in cmd and "--encoder-only" not in cmd
            if worker.worker_type == WorkerType.PREFILL:
                assert cmd[cmd.index("--encoder-transfer-backend") + 1] == "zmq_to_scheduler"
                assert worker.environment[RECEIVER_MODE] == "grpc"
            else:
                assert "--encoder-transfer-backend" not in cmd
                assert RECEIVER_MODE not in worker.environment
            assert cmd[cmd.index("--disaggregation-transfer-backend") + 1] == "mooncake"


def test_tokenspeed_epd_does_not_receive_sglang_role_options(fleet):
    for worker in fleet("tokenspeed"):
        assert worker.extra_engine_args is None and worker.extra_env is None
        assert not any(
            arg in worker.command
            for arg in ("--language-only", "--language-model-only", "--encoder-transfer-backend")
        )
        assert RECEIVER_MODE not in worker.environment


@pytest.mark.parametrize("role", [WorkerType.PREFILL, WorkerType.DECODE])
def test_ordinary_sglang_pd_still_loads_its_own_vision(role, monkeypatch):
    monkeypatch.delenv(RECEIVER_MODE, raising=False)
    worker = Worker(
        model_id=MODEL,
        engine="sglang",
        port=50100,
        gpu_ids=[0],
        mode=ConnectionMode.GRPC,
        worker_type=role,
    )
    cmd = worker._build_cmd()
    assert "--language-only" not in cmd and "--language-model-only" not in cmd
    assert "--encoder-transfer-backend" not in cmd
    assert RECEIVER_MODE not in worker._build_env()
