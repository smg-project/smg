"""Unit tests for vLLM PD workers on the MoRI-IO KV backend (no GPU, no wheel required)."""

from __future__ import annotations

import json

import pytest
from infra import process_utils
from infra import worker as worker_module
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


@pytest.mark.parametrize("tp", [1, 2])
def test_dp_group_launch_labels_and_ports_agree(tp):
    worker = _worker(WorkerType.PREFILL, tp=tp)
    worker.dp_size = 2
    worker.gpu_ids = list(range(2 * tp))
    config = _kv_config(worker)
    cmd = worker._build_cmd()
    for flag, value in {
        "--data-parallel-size": "2",
        "--data-parallel-size-local": "2",
        "--data-parallel-backend": "mp",
        "--data-parallel-address": "10.0.0.7",
        "--data-parallel-rpc-port": str(worker.moriio_dp_rpc_port),
    }.items():
        assert cmd[cmd.index(flag) + 1] == value
    assert worker.moriio_registration()["labels"]["dp_size"] == "2"
    extra = config["kv_connector_extra_config"]
    handshake, notify = int(extra["handshake_port"]), int(extra["notify_port"])
    expected = set(range(handshake, handshake + 2 * tp)) | set(range(notify, notify + 2 * tp))
    assert len(expected) == 4 * tp
    assert worker.moriio_dp_rpc_port not in expected
    assert set(worker._moriio_ports) == expected | {worker.moriio_dp_rpc_port}
    assert len(worker._moriio_ports) == 4 * tp + 1
    assert _kv_config(worker) == config  # a restart preserves every listener


def test_dp1_keeps_the_existing_launch_and_port_layout():
    worker = _worker(WorkerType.PREFILL, tp=2)
    cmd = worker._build_cmd()
    assert not any(arg.startswith("--data-parallel-") for arg in cmd)
    assert worker.moriio_dp_rpc_port is None
    assert len(worker._moriio_ports) == 3
    assert worker.moriio_registration()["labels"]["dp_size"] == "1"


def test_dp_pool_gpu_slices_and_role_ports_do_not_overlap(monkeypatch):
    monkeypatch.setattr(worker_module, "detect_ib_device", lambda: None)
    monkeypatch.setattr(worker_module, "get_gpu_offset", lambda: 0)
    monkeypatch.setattr(Worker, "start", lambda *_args, **_kwargs: None)
    prefill = worker_module.start_workers(
        _MODEL, engine="vllm", worker_type=WorkerType.PREFILL, tp=2, dp_size=2, wait_ready=False
    )[0]
    decode = worker_module.start_workers(
        _MODEL,
        engine="vllm",
        worker_type=WorkerType.DECODE,
        tp=1,
        dp_size=2,
        gpu_offset=4,
        wait_ready=False,
    )[0]
    assert prefill.gpu_ids == [0, 1, 2, 3]
    assert decode.gpu_ids == [4, 5]
    _kv_config(prefill)
    _kv_config(decode)
    assert set(prefill._moriio_ports).isdisjoint(decode._moriio_ports)


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_dp_sizes_are_rejected(size):
    worker = _worker(WorkerType.PREFILL)
    worker.dp_size = size
    with pytest.raises(ValueError, match="positive integer"):
        worker._build_cmd()


def test_dp_rejects_insufficient_gpus_and_conflicting_flags():
    worker = _worker(WorkerType.PREFILL, tp=2)
    worker.dp_size = 2
    with pytest.raises(ValueError, match="GPU slice"):
        worker._build_cmd()
    worker.gpu_ids = [0, 1, 2, 3]
    worker.extra_engine_args = ["--data-parallel-size=4"]
    with pytest.raises(ValueError, match="use dp_size"):
        worker._build_cmd()
    worker.extra_engine_args = None
    worker.moriio_handshake_port = 24000
    with pytest.raises(ValueError, match="allocated together"):
        worker._build_cmd()


@pytest.mark.parametrize(
    "backend,mode", [("nixl", ConnectionMode.HTTP), ("moriio", ConnectionMode.GRPC)]
)
def test_explicit_http_moriio_dp_does_not_change_other_launchers(backend, mode):
    worker = _worker(WorkerType.PREFILL)
    worker.dp_size, worker.kv_backend, worker.mode = 2, backend, mode
    with pytest.raises(ValueError, match="HTTP MoRI-IO"):
        worker._build_cmd()
