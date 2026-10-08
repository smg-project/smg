"""Unit tests for vLLM PD workers on the MoRI-IO KV backend (no GPU, no wheel required)."""

from __future__ import annotations

import json

import pytest
from infra import process_utils
from infra.constants import ConnectionMode, WorkerType
from infra.worker import Worker

_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


@pytest.fixture(autouse=True)
def _moriio_lane(monkeypatch):
    monkeypatch.delenv("E2E_KV_BACKEND", raising=False)
    monkeypatch.delenv("E2E_VLLM_MORIIO_MODE", raising=False)
    monkeypatch.delenv("E2E_VLLM_MORIIO_BACKEND", raising=False)
    monkeypatch.setenv("E2E_VLLM_KV_BACKEND", "moriio")
    monkeypatch.setenv("E2E_MORIIO_HOST", "10.0.0.7")


def _worker(worker_type: WorkerType, tp: int | None = None) -> Worker:
    return Worker(
        model_id=_MODEL,
        engine="vllm",
        port=50100,
        gpu_ids=[0],
        mode=ConnectionMode.HTTP,
        worker_type=worker_type,
        tp=tp,
    )


def _kv_config(worker: Worker) -> dict:
    cmd = worker._build_cmd()
    return json.loads(cmd[cmd.index("--kv-transfer-config") + 1])


@pytest.mark.parametrize(
    "worker_type,role", [(WorkerType.PREFILL, "kv_producer"), (WorkerType.DECODE, "kv_consumer")]
)
def test_a_pd_worker_runs_the_moriio_connector(worker_type, role):
    config = _kv_config(_worker(worker_type))

    assert config["kv_connector"] == "MoRIIOConnector"
    assert config["kv_role"] == role
    extra = config["kv_connector_extra_config"]
    assert extra["host_ip"] == "10.0.0.7"
    assert extra["http_port"] == "50100"
    assert extra["read_mode"] is True
    assert extra["backend"] == "rdma"


def test_write_mode_turns_read_mode_off(monkeypatch):
    monkeypatch.setenv("E2E_VLLM_MORIIO_MODE", "write")

    extra = _kv_config(_worker(WorkerType.DECODE))["kv_connector_extra_config"]

    assert extra["read_mode"] is False


def test_the_xgmi_engine_can_be_chosen(monkeypatch):
    monkeypatch.setenv("E2E_VLLM_MORIIO_BACKEND", "xgmi")

    extra = _kv_config(_worker(WorkerType.PREFILL))["kv_connector_extra_config"]

    assert extra["backend"] == "xgmi"


def test_empty_settings_fall_back_to_the_defaults(monkeypatch):
    # A workflow input left empty still sets its variable, to "".
    monkeypatch.setenv("E2E_VLLM_MORIIO_MODE", "")
    monkeypatch.setenv("E2E_VLLM_MORIIO_BACKEND", "")

    extra = _kv_config(_worker(WorkerType.DECODE))["kv_connector_extra_config"]

    assert extra["read_mode"] is True
    assert extra["backend"] == "rdma"


def test_an_unknown_mode_is_refused(monkeypatch):
    monkeypatch.setenv("E2E_VLLM_MORIIO_MODE", "both")

    with pytest.raises(ValueError, match="E2E_VLLM_MORIIO_MODE"):
        _worker(WorkerType.DECODE)._build_cmd()


def test_registration_names_the_ports_the_engine_listens_on():
    worker = _worker(WorkerType.PREFILL, tp=2)
    extra = _kv_config(worker)["kv_connector_extra_config"]

    registration = worker.moriio_registration()

    assert registration["kv_connector"] == "MoRIIOConnector"
    assert registration["kv_role"] == "kv_producer"
    assert registration["connection_mode"] == "http"
    labels = registration["labels"]
    assert labels["moriio_handshake_port"] == extra["handshake_port"]
    assert labels["moriio_notify_port"] == extra["notify_port"]
    assert labels["moriio_host"] == "10.0.0.7"
    assert labels["moriio_mode"] == "read"
    assert labels["tp_size"] == "2"


def test_each_rank_gets_a_notify_port():
    worker = _worker(WorkerType.DECODE, tp=2)
    extra = _kv_config(worker)["kv_connector_extra_config"]

    notify = int(extra["notify_port"])

    assert set(worker._moriio_ports) == {int(extra["handshake_port"]), notify, notify + 1}


def test_the_ports_sit_below_the_kernels_ephemeral_range(monkeypatch):
    # The listeners bind only once the model has loaded; meanwhile the kernel
    # hands ephemeral ports to any process on the host.
    monkeypatch.setattr(process_utils, "_ephemeral_port_floor", lambda: 32768)
    worker = _worker(WorkerType.DECODE, tp=2)
    _kv_config(worker)

    assert all(20000 <= port < 32768 for port in worker._moriio_ports)


def test_a_restarted_worker_keeps_its_ports():
    worker = _worker(WorkerType.DECODE)
    first = _kv_config(worker)["kv_connector_extra_config"]

    second = _kv_config(worker)["kv_connector_extra_config"]

    assert second["handshake_port"] == first["handshake_port"]
    assert second["notify_port"] == first["notify_port"]


def test_the_engine_advertises_the_moriio_host_and_gets_no_nixl_port():
    worker = _worker(WorkerType.DECODE)
    worker._build_cmd()

    env = worker._build_env()

    assert env["VLLM_HOST_IP"] == "10.0.0.7"
    assert "VLLM_NIXL_SIDE_CHANNEL_PORT" not in env
