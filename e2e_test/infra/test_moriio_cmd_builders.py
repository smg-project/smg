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


def _worker(
    worker_type: WorkerType,
    tp: int | None = None,
    mode: ConnectionMode = ConnectionMode.HTTP,
) -> Worker:
    return Worker(
        model_id=_MODEL,
        engine="vllm",
        port=50100,
        gpu_ids=[0],
        mode=mode,
        worker_type=worker_type,
        tp=tp,
    )


def _kv_config(worker: Worker) -> dict:
    cmd = worker._build_cmd()
    return json.loads(cmd[cmd.index("--kv-transfer-config") + 1])


@pytest.mark.parametrize(
    "worker_type,role", [(WorkerType.PREFILL, "kv_producer"), (WorkerType.DECODE, "kv_consumer")]
)
@pytest.mark.parametrize("mode", [ConnectionMode.HTTP, ConnectionMode.GRPC])
def test_a_pd_worker_runs_the_moriio_connector(worker_type, role, mode):
    config = _kv_config(_worker(worker_type, mode=mode))

    assert config["kv_connector"] == "MoRIIOConnector"
    assert config["kv_role"] == role
    extra = config["kv_connector_extra_config"]
    assert extra["host_ip"] == "10.0.0.7"
    assert extra["http_port"] == "50100"
    assert extra["read_mode"] is True
    assert extra["backend"] == "rdma"
    if role == "kv_producer":
        # Sliced READ readers require this capability in the producer's
        # handshake before they can post I/O against its source allocation.
        assert extra.get("read_ack_fenced_hold") is True
    else:
        assert "read_ack_fenced_hold" not in extra


@pytest.mark.parametrize("worker_type", [WorkerType.PREFILL, WorkerType.DECODE])
def test_write_mode_turns_read_mode_off(monkeypatch, worker_type):
    monkeypatch.setenv("E2E_VLLM_MORIIO_MODE", "write")

    extra = _kv_config(_worker(worker_type))["kv_connector_extra_config"]

    assert extra["read_mode"] is False
    assert "read_ack_fenced_hold" not in extra


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


def _qualified_read_invocation(monkeypatch):
    from infra import worker as worker_module

    verifier = worker_module._moriio_read_image_verifier()
    invocation = {
        "E2E_RUNTIME": "vllm",
        "E2E_GPU_VENDOR": "amd",
        "E2E_KV_BACKEND": "",
        "E2E_VLLM_KV_BACKEND": "moriio",
        "E2E_VLLM_MORIIO_MODE": "read",
        "E2E_VLLM_MORIIO_BACKEND": "rdma",
    }
    image = "example.invalid/read@sha256:" + "a" * 64
    invocation.update(
        verifier._qualification_exports(image, dict(verifier.SOURCE_SHA256), invocation)
    )
    for name, value in invocation.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(verifier, "verify_installed", lambda: dict(verifier.SOURCE_SHA256))
    monkeypatch.setattr(worker_module, "_moriio_read_image_verifier", lambda: verifier)
    return verifier


@pytest.mark.parametrize("role", [WorkerType.PREFILL, WorkerType.DECODE])
def test_verified_invocation_keeps_strict_route_after_worker_overrides(monkeypatch, role):
    _qualified_read_invocation(monkeypatch)
    worker = _worker(role)
    worker.extra_env = {
        "MORI_DISABLE_AUTO_XGMI": "0",
        "E2E_VLLM_MORIIO_MODE": "write",  # An intentional wrong-protocol worker remains wrong.
        "SMG_PAIRING_PROTOCOL": "different-protocol",
    }
    env = worker._build_env()
    assert env["MORI_DISABLE_AUTO_XGMI"] == "1"
    assert env["E2E_VLLM_MORIIO_MODE"] == "write"
    assert env["SMG_PAIRING_PROTOCOL"] == "different-protocol"


@pytest.mark.parametrize(
    "override",
    [
        {"E2E_MORIIO_READ_QUALIFICATION": ""},
        {"E2E_VLLM_MORIIO_MODE": "write"},
        {"E2E_VLLM_MORIIO_BACKEND": "xgmi"},
        {"E2E_GPU_VENDOR": "nvidia"},
    ],
)
def test_unqualified_worker_route_and_mode_keep_their_original_values(monkeypatch, override):
    _qualified_read_invocation(monkeypatch)
    for name, value in override.items():
        monkeypatch.setenv(name, value)
    worker = _worker(WorkerType.PREFILL)
    worker.extra_env = {"MORI_DISABLE_AUTO_XGMI": "0"}
    env = worker._build_env()
    assert env["MORI_DISABLE_AUTO_XGMI"] == "0"
    config = _kv_config(worker)["kv_connector_extra_config"]
    assert config["read_mode"] is (override.get("E2E_VLLM_MORIIO_MODE") != "write")
    assert config["backend"] == override.get("E2E_VLLM_MORIIO_BACKEND", "rdma")


def test_worker_does_not_accept_intent_after_installed_source_changes(monkeypatch):
    verifier = _qualified_read_invocation(monkeypatch)

    def wrong_source():
        raise ValueError("installed source changed after preflight")

    monkeypatch.setattr(verifier, "verify_installed", wrong_source)
    with pytest.raises(ValueError, match="source changed"):
        _worker(WorkerType.PREFILL)._build_env()


@pytest.mark.parametrize("raw", [True, 12.5, -1, "01", None, 12, "12"])
def test_first_attempt_baseline_uses_actual_strict_parser_before_any_spawn(monkeypatch, raw):
    from types import SimpleNamespace

    from infra import worker as worker_module

    _qualified_read_invocation(monkeypatch)
    worker = _worker(WorkerType.PREFILL)
    calls = []
    monkeypatch.setattr(worker, "_build_cmd", lambda: ["fixture-no-server"])

    def fake_smi(cmd, **kwargs):
        if cmd[0] == "nvidia-smi":
            raise FileNotFoundError("fake AMD-only tool")
        data = (
            [{"gpu": 0, "node_id": 2}]
            if cmd[1] == "list"
            else [{"gpu": 0, "mem_usage": {"used_vram": {"value": raw}}}]
        )
        import subprocess

        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(data))

    def spawn(cmd, env):
        calls.append((cmd, env))
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(process_utils.subprocess, "run", fake_smi)
    monkeypatch.setattr(worker_module, "gpu_memory_used_mib", process_utils.gpu_memory_used_mib)
    monkeypatch.setattr(worker, "_spawn_process", spawn)
    if type(raw) is int and raw >= 0 or raw == "12":
        worker.start(wait_ready=False)
        assert worker._gpu_mem_baseline == {0: 12}
        assert len(calls) == 1
    else:
        with pytest.raises(ValueError, match="validated all-GPU baseline"):
            worker.start(wait_ready=False)
        assert calls == []
