import sys
import types

import pytest
from smg_grpc_servicer.worker_control_lifecycle import WorkerControlLifecycle


class FakeServer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.running = True
        self.last_error = None
        self.health = []
        self.stopped_with = None

    def set_health(self, state, message):
        self.health.append((state, message))

    def stop(self, timeout_secs):
        self.stopped_with = timeout_secs
        self.running = False


@pytest.fixture
def fake_smg(monkeypatch):
    worker_module = types.ModuleType("smg.worker")
    worker_module.WorkerControlServer = FakeServer
    smg_module = types.ModuleType("smg")
    smg_module.worker = worker_module
    monkeypatch.setitem(sys.modules, "smg", smg_module)
    monkeypatch.setitem(sys.modules, "smg.worker", worker_module)


def test_disabled_without_bind_address():
    lifecycle = WorkerControlLifecycle.start_from_env(
        engine_type="sglang",
        model_ids=["model"],
        features=["generate"],
        environ={},
    )
    assert lifecycle is None


def test_enabled_requires_distinct_advertised_endpoint():
    with pytest.raises(ValueError, match="SMG_WORKER_ENGINE_ENDPOINT"):
        WorkerControlLifecycle.start_from_env(
            engine_type="sglang",
            model_ids=["model"],
            features=["generate"],
            environ={"SMG_WORKER_CONTROL_BIND_ADDRESS": "0.0.0.0:31000"},
        )


def test_builds_shared_server_contract_and_drives_health(fake_smg):
    lifecycle = WorkerControlLifecycle.start_from_env(
        engine_type="sglang",
        model_ids=["model-a"],
        features=["generate", "abort"],
        max_concurrent_requests=32,
        engine_attributes={"model_path": "model-a"},
        environ={
            "SMG_WORKER_CONTROL_BIND_ADDRESS": "0.0.0.0:31000",
            "SMG_WORKER_ENGINE_ENDPOINT": "grpc://worker-a:32000",
            "SMG_WORKER_ID": "worker-a",
            "SMG_WORKER_INSTANCE_ID": "instance-a",
            "SMG_WORKER_ZONE": "zone-a",
        },
    )

    assert lifecycle is not None
    assert lifecycle.server.kwargs["engine_endpoint"] == "grpc://worker-a:32000"
    assert lifecycle.server.kwargs["model_ids"] == ["model-a"]
    assert lifecycle.server.kwargs["features"] == ["generate", "abort"]
    assert lifecycle.server.kwargs["max_concurrent_requests"] == 32
    assert lifecycle.server.kwargs["inference_enabled"] is False
    assert lifecycle.server.kwargs["engine_attributes"] == {"model_path": "model-a"}
    assert lifecycle.server.kwargs["engine_transport"] == "grpc"
    assert lifecycle.server.kwargs["zmq_handshake_address"] is None
    assert lifecycle.server.kwargs["engine_count"] == 1

    lifecycle.mark_serving()
    lifecycle.mark_draining()
    lifecycle.mark_not_serving("shutdown")
    lifecycle.stop(2.5)
    assert lifecycle.server.health == [
        ("serving", "ready"),
        ("draining", "draining"),
        ("not_serving", "shutdown"),
    ]
    assert lifecycle.server.stopped_with == 2.5


def test_enables_inference_adapter_explicitly(fake_smg):
    lifecycle = WorkerControlLifecycle.start_from_env(
        engine_type="sglang",
        model_ids=["model-a"],
        features=["generate", "abort"],
        environ={
            "SMG_WORKER_CONTROL_BIND_ADDRESS": "0.0.0.0:31000",
            "SMG_WORKER_ENGINE_ENDPOINT": "grpc://worker-a:32000",
            "SMG_WORKER_INFERENCE_ENABLED": "true",
        },
    )

    assert lifecycle is not None
    assert lifecycle.server.kwargs["inference_enabled"] is True


def test_configures_same_host_zmq_transport(fake_smg):
    lifecycle = WorkerControlLifecycle.start_from_env(
        engine_type="tokenspeed",
        model_ids=["model-a"],
        features=["generate", "abort"],
        environ={
            "SMG_WORKER_CONTROL_BIND_ADDRESS": "0.0.0.0:31000",
            "SMG_WORKER_ENGINE_ENDPOINT": "ipc:///tmp/smg-zmq/engine-a",
            "SMG_WORKER_INFERENCE_ENABLED": "true",
            "SMG_WORKER_ENGINE_TRANSPORT": "zmq",
            "SMG_WORKER_ZMQ_HANDSHAKE_ADDRESS": "tcp://127.0.0.1:30500",
            "SMG_WORKER_ENGINE_COUNT": "2",
        },
    )

    assert lifecycle is not None
    assert lifecycle.server.kwargs["engine_transport"] == "zmq"
    assert lifecycle.server.kwargs["zmq_handshake_address"] == "tcp://127.0.0.1:30500"
    assert lifecycle.server.kwargs["engine_count"] == 2


@pytest.mark.parametrize("value", ["0", "bad"])
def test_rejects_invalid_engine_count(fake_smg, value):
    with pytest.raises(ValueError, match="SMG_WORKER_ENGINE_COUNT"):
        WorkerControlLifecycle.start_from_env(
            engine_type="tokenspeed",
            model_ids=["model-a"],
            features=["generate"],
            environ={
                "SMG_WORKER_CONTROL_BIND_ADDRESS": "0.0.0.0:31000",
                "SMG_WORKER_ENGINE_ENDPOINT": "ipc:///tmp/smg-zmq/engine-a",
                "SMG_WORKER_ENGINE_COUNT": value,
            },
        )
