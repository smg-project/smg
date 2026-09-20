"""Engine-free tests for the SGLang gRPC server's lifecycle watcher and shutdown order."""

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


def _module(name, **attrs):
    module = ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


class FakeLifecycle:
    """WorkerControlLifecycle stand-in whose `last_error` follows a script."""

    def __init__(self, events, errors, on_poll=None):
        self.events = events
        self.errors = list(errors)
        self.on_poll = on_poll
        self.polls = 0
        self.running = True

    @property
    def last_error(self):
        self.polls += 1
        if self.on_poll is not None:
            self.on_poll(self.polls)
        return self.errors[min(self.polls, len(self.errors)) - 1]

    def mark_serving(self):
        self.events.append("control-serving")

    def mark_draining(self):
        self.events.append("control-draining")

    def mark_not_serving(self, message="stopped"):
        self.events.append("control-not-serving")

    def stop(self, timeout_secs=5.0):
        self.events.append(f"control-stop:{timeout_secs}")


def _server_args():
    return SimpleNamespace(
        disaggregation_mode="null",
        model_path="/models/m",
        served_model_name=None,
        tokenizer_path=None,
        context_length=None,
        is_embedding=False,
        host="127.0.0.1",
        port=0,
        ssl_certfile=None,
        ssl_keyfile=None,
        ssl_keyfile_password=None,
        skip_server_warmup=True,
        max_running_requests=32,
    )


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

    class PortArgs:
        @staticmethod
        def init_new(_server_args):
            raise AssertionError("test must replace the scheduler launch")

    descriptor = SimpleNamespace(
        services_by_name={
            "SglangScheduler": SimpleNamespace(full_name="sglang.grpc.scheduler.SglangScheduler")
        }
    )
    stubs = {
        "grpc": _module("grpc", aio=SimpleNamespace(server=lambda *_a, **_k: GrpcServer())),
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
        "sglang": _module("sglang"),
        "sglang.srt": _module("sglang.srt"),
        "sglang.srt.configs": _module("sglang.srt.configs"),
        "sglang.srt.configs.model_config": _module(
            "sglang.srt.configs.model_config",
            ModelConfig=SimpleNamespace(
                from_server_args=lambda _args: SimpleNamespace(hf_config=SimpleNamespace())
            ),
        ),
        "sglang.srt.disaggregation": _module("sglang.srt.disaggregation"),
        "sglang.srt.disaggregation.utils": _module(
            "sglang.srt.disaggregation.utils",
            FAKE_BOOTSTRAP_HOST="2.2.2.2",
            DisaggregationMode=object,
        ),
        "sglang.srt.managers": _module("sglang.srt.managers"),
        "sglang.srt.managers.disagg_service": _module(
            "sglang.srt.managers.disagg_service",
            start_disagg_service=lambda _args: None,
        ),
        "sglang.srt.runtime_context": _module(
            "sglang.srt.runtime_context",
            publish=lambda _args, role: None,
        ),
        "sglang.srt.server_args": _module(
            "sglang.srt.server_args",
            PortArgs=PortArgs,
            ServerArgs=object,
        ),
        "sglang.srt.utils": _module(
            "sglang.srt.utils",
            kill_process_tree=lambda _pid: None,
        ),
        "sglang.utils": _module(
            "sglang.utils",
            get_exception_traceback=lambda: "",
        ),
        "smg_grpc_proto": _module(
            "smg_grpc_proto",
            sglang_scheduler_pb2=SimpleNamespace(DESCRIPTOR=descriptor),
            sglang_scheduler_pb2_grpc=SimpleNamespace(
                add_SglangSchedulerServicer_to_server=lambda *_a: None
            ),
        ),
        "smg_grpc_servicer.sglang.health_servicer": _module(
            "smg_grpc_servicer.sglang.health_servicer",
            SGLangHealthServicer=HealthServicer,
        ),
        "smg_grpc_servicer.sglang.request_manager": _module(
            "smg_grpc_servicer.sglang.request_manager",
            GrpcRequestManager=object,
        ),
        "smg_grpc_servicer.sglang.scheduler_launcher": _module(
            "smg_grpc_servicer.sglang.scheduler_launcher",
            launch_scheduler_process_only=lambda **_kwargs: None,
            terminate_scheduler_processes=lambda _procs: events.append("schedulers"),
        ),
        "smg_grpc_servicer.sglang.servicer": _module(
            "smg_grpc_servicer.sglang.servicer",
            SGLangSchedulerServicer=SchedulerServicer,
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = Path(__file__).parents[1] / "smg_grpc_servicer" / "sglang" / "server.py"
    spec = importlib.util.spec_from_file_location("test_sglang_grpc_server_lifecycle", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    request_manager = SimpleNamespace(
        gracefully_exit=False,
        shutdown=AsyncMock(side_effect=lambda: events.append("manager-shutdown")),
    )
    scheduler_procs = [object()]

    async def launch(**_kwargs):
        return {"status": "ready"}, object(), scheduler_procs, request_manager

    monkeypatch.setattr(module, "_launch_scheduler_with_request_manager", launch)
    monkeypatch.setattr(module, "_wait_and_warmup_grpc", Mock())
    monkeypatch.setattr(module, "_LIFECYCLE_WATCH_INTERVAL_SECS", 0)

    return SimpleNamespace(
        module=module,
        events=events,
        request_manager=request_manager,
        scheduler_procs=scheduler_procs,
    )


def _after_start(events):
    return events[events.index("server-start") + 1 :]


def test_watcher_outlives_idle_ticks_and_shuts_down_in_order(harness, monkeypatch, caplog):
    """The watcher keeps polling through idle ticks; an engine-link failure
    surfacing later stops the server through the ordered shutdown path."""
    caplog.set_level(logging.ERROR)
    lifecycle = FakeLifecycle(harness.events, [None, None, None, "engine link lost"])
    started = {}

    def start_from_env(**kwargs):
        started.update(kwargs)
        return lifecycle

    monkeypatch.setattr(
        harness.module, "WorkerControlLifecycle", SimpleNamespace(start_from_env=start_from_env)
    )

    asyncio.run(harness.module.serve_grpc(_server_args()))

    assert lifecycle.polls == 4
    assert _after_start(harness.events) == [
        "control-draining",
        "health-not-serving",
        "server-stop:5.0",
        "control-not-serving",
        "control-stop:5.0",
        "servicer-shutdown",
        "schedulers",
    ]
    assert started["engine_type"] == "sglang"
    assert started["model_ids"] == ["/models/m"]
    assert started["features"] == ["generate", "abort"]
    assert started["max_concurrent_requests"] == 32
    assert started["engine_attributes"] == {
        "model_path": "/models/m",
        "tokenizer_path": "/models/m",
    }
    failure = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert failure and "engine link lost" in failure[0].getMessage()


def test_request_manager_exit_stops_the_server(harness, monkeypatch):
    def request_exit(poll_count):
        if poll_count == 2:
            harness.request_manager.gracefully_exit = True

    lifecycle = FakeLifecycle(harness.events, [None], on_poll=request_exit)
    monkeypatch.setattr(
        harness.module,
        "WorkerControlLifecycle",
        SimpleNamespace(start_from_env=lambda **_kwargs: lifecycle),
    )

    asyncio.run(harness.module.serve_grpc(_server_args()))

    assert lifecycle.polls == 2
    assert _after_start(harness.events) == [
        "control-draining",
        "health-not-serving",
        "server-stop:5.0",
        "control-not-serving",
        "control-stop:5.0",
        "servicer-shutdown",
        "schedulers",
    ]


def test_control_plane_rejection_is_a_startup_failure(harness, monkeypatch, caplog):
    """A WorkerControlServer the extension refuses to build is logged at ERROR,
    releases every owned resource and re-raises; warmup never starts."""
    caplog.set_level(logging.ERROR)

    def reject(**_kwargs):
        raise ValueError("engine_type sglang cannot serve inference")

    monkeypatch.setattr(
        harness.module, "WorkerControlLifecycle", SimpleNamespace(start_from_env=reject)
    )

    with pytest.raises(ValueError, match="cannot serve inference"):
        asyncio.run(harness.module.serve_grpc(_server_args()))

    assert _after_start(harness.events) == [
        "health-not-serving",
        "server-stop:0",
        "schedulers",
        "manager-shutdown",
    ]
    assert any(
        record.levelno == logging.ERROR and "gRPC startup failed" in record.getMessage()
        for record in caplog.records
    )
    harness.module._wait_and_warmup_grpc.assert_not_called()
