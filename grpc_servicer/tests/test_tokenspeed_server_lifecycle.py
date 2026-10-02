"""Engine-free tests for the TokenSpeed gRPC server's Worker control plane wiring."""

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


def _module(name, **attrs):
    module = ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


class FakeLifecycle:
    """WorkerControlLifecycle stand-in whose `last_error` follows a script."""

    def __init__(self, events, errors):
        self.events = events
        self.errors = list(errors)
        self.polls = 0
        self.running = True

    @property
    def last_error(self):
        self.polls += 1
        return self.errors[min(self.polls, len(self.errors)) - 1]

    def mark_serving(self):
        self.events.append("control-serving")

    def mark_draining(self):
        self.events.append("control-draining")

    def mark_not_serving(self, message="stopped"):
        self.events.append("control-not-serving")

    def stop(self, timeout_secs=5.0):
        self.events.append(f"control-stop:{timeout_secs}")


def _server_args(**overrides):
    args = {
        "disaggregation_mode": "null",
        "host": "127.0.0.1",
        "port": 0,
        "model": "/models/llama",
        "served_model_name": "llama-prod",
        "tokenizer": None,
        "max_running_requests": 0,
        "max_num_seqs": 16,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


@pytest.fixture
def harness(monkeypatch):
    """Load server.py against stand-ins for every runtime dependency and record
    the lifecycle calls it makes, in order."""
    events = []

    class GrpcServer:
        def add_insecure_port(self, address):
            events.append(f"bind:{address}")
            return 1

        async def start(self):
            events.append("server-start")

        async def stop(self, grace):
            events.append(f"server-stop:{grace}")

    class HealthServicer:
        def __init__(self, **_kwargs):
            pass

        def set_serving(self):
            events.append("health-serving")

        def set_not_serving(self):
            events.append("health-not-serving")

    class SchedulerServicer:
        def __init__(self, **_kwargs):
            pass

        async def shutdown(self):
            events.append("servicer-shutdown")

    descriptor = SimpleNamespace(
        services_by_name={
            "TokenSpeedScheduler": SimpleNamespace(
                full_name="tokenspeed.grpc.scheduler.TokenSpeedScheduler"
            )
        }
    )
    stubs = {
        "grpc": _module(
            "grpc",
            aio=SimpleNamespace(server=lambda *_a, **_k: GrpcServer()),
            StatusCode=SimpleNamespace(FAILED_PRECONDITION="failed-precondition"),
        ),
        "grpc_health": _module("grpc_health"),
        "grpc_health.v1": _module(
            "grpc_health.v1",
            health_pb2_grpc=SimpleNamespace(add_HealthServicer_to_server=lambda *_a: None),
        ),
        "grpc_reflection": _module("grpc_reflection"),
        "grpc_reflection.v1alpha": _module(
            "grpc_reflection.v1alpha",
            reflection=SimpleNamespace(
                enable_server_reflection=lambda *_a: None,
                SERVICE_NAME="grpc.reflection.v1alpha.ServerReflection",
            ),
        ),
        "smg_grpc_proto": _module(
            "smg_grpc_proto",
            tokenspeed_scheduler_pb2_grpc=SimpleNamespace(
                add_TokenSpeedSchedulerServicer_to_server=lambda *_a: None
            ),
        ),
        "smg_grpc_proto.generated": _module(
            "smg_grpc_proto.generated",
            tokenspeed_scheduler_pb2=SimpleNamespace(DESCRIPTOR=descriptor),
        ),
        "tokenspeed": _module("tokenspeed"),
        "tokenspeed.runtime": _module("tokenspeed.runtime"),
        "tokenspeed.runtime.utils": _module("tokenspeed.runtime.utils"),
        "tokenspeed.runtime.utils.server_args": _module(
            "tokenspeed.runtime.utils.server_args",
            ServerArgs=object,
        ),
        "smg_grpc_servicer.tokenspeed.health_servicer": _module(
            "smg_grpc_servicer.tokenspeed.health_servicer",
            TokenSpeedHealthServicer=HealthServicer,
        ),
        "smg_grpc_servicer.tokenspeed.scheduler_launcher": _module(
            "smg_grpc_servicer.tokenspeed.scheduler_launcher",
            launch_engine=lambda _args: (object(), {"status": "ready"}),
        ),
        "smg_grpc_servicer.tokenspeed.servicer": _module(
            "smg_grpc_servicer.tokenspeed.servicer",
            TokenSpeedSchedulerServicer=SchedulerServicer,
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = Path(__file__).parents[1] / "smg_grpc_servicer" / "tokenspeed" / "server.py"
    spec = importlib.util.spec_from_file_location(
        "test_tokenspeed_grpc_server_lifecycle", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "_wait_and_warmup", Mock())
    monkeypatch.setattr(module, "_WORKER_CONTROL_WATCH_INTERVAL_SECS", 0)

    return SimpleNamespace(module=module, events=events)


def _after_start(events):
    return events[events.index("server-start") + 1 :]


def test_engine_attributes_name_the_model_files_not_the_served_alias(harness, monkeypatch):
    """The Router resolves the tokenizer from `model_path`, so it carries the
    real model path while `model_ids` advertises the served name."""
    lifecycle = FakeLifecycle(harness.events, [None, "engine link lost"])
    started = {}

    def start_from_env(**kwargs):
        started.update(kwargs)
        return lifecycle

    monkeypatch.setattr(
        harness.module, "WorkerControlLifecycle", SimpleNamespace(start_from_env=start_from_env)
    )

    asyncio.run(harness.module.serve_grpc(_server_args()))

    assert started["engine_type"] == "tokenspeed"
    assert started["model_ids"] == ["llama-prod"]
    assert started["features"] == ["generate", "abort"]
    assert started["max_concurrent_requests"] == 16
    assert started["engine_attributes"] == {
        "model_path": "/models/llama",
        "tokenizer_path": "/models/llama",
    }
    assert lifecycle.polls == 2
    assert _after_start(harness.events) == [
        "control-draining",
        "health-not-serving",
        "servicer-shutdown",
        "server-stop:5.0",
        "control-not-serving",
        "control-stop:5.0",
    ]


def test_explicit_tokenizer_wins_over_the_model_path(harness, monkeypatch):
    lifecycle = FakeLifecycle(harness.events, ["engine link lost"])
    started = {}

    def start_from_env(**kwargs):
        started.update(kwargs)
        return lifecycle

    monkeypatch.setattr(
        harness.module, "WorkerControlLifecycle", SimpleNamespace(start_from_env=start_from_env)
    )

    asyncio.run(
        harness.module.serve_grpc(
            _server_args(served_model_name=None, tokenizer="/models/llama-tokenizer")
        )
    )

    assert started["model_ids"] == ["/models/llama"]
    assert started["engine_attributes"] == {
        "model_path": "/models/llama",
        "tokenizer_path": "/models/llama-tokenizer",
    }


def test_control_plane_rejection_is_a_startup_failure(harness, monkeypatch, caplog):
    """A WorkerControlServer the extension refuses to build is logged at ERROR,
    releases the gRPC server and servicer, and re-raises; warmup never starts."""
    caplog.set_level(logging.ERROR)

    def reject(**_kwargs):
        raise ValueError("engine_count must be positive")

    monkeypatch.setattr(
        harness.module, "WorkerControlLifecycle", SimpleNamespace(start_from_env=reject)
    )

    with pytest.raises(ValueError, match="engine_count must be positive"):
        asyncio.run(harness.module.serve_grpc(_server_args()))

    assert _after_start(harness.events) == [
        "health-not-serving",
        "server-stop:0",
        "servicer-shutdown",
    ]
    assert any(
        record.levelno == logging.ERROR
        and "Failed to start the Worker control plane" in record.getMessage()
        for record in caplog.records
    )
    harness.module._wait_and_warmup.assert_not_called()
