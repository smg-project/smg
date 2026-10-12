"""Engine-free tests for SGLang gRPC server startup ordering."""

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from smg_grpc_proto import sglang_scheduler_pb2 as pb


def _module(name, **attrs):
    module = ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


@pytest.fixture
def server_mod(monkeypatch):
    """Load server.py with lightweight stand-ins for optional runtime deps."""

    class PortArgs:
        @staticmethod
        def init_new(_server_args):
            raise AssertionError("test must replace PortArgs.init_new")

    class ServerArgs:
        pass

    stubs = {
        "grpc": _module("grpc", aio=SimpleNamespace()),
        "grpc_health": _module("grpc_health"),
        "grpc_health.v1": _module(
            "grpc_health.v1",
            health_pb2_grpc=SimpleNamespace(),
        ),
        "grpc_reflection": _module("grpc_reflection"),
        "grpc_reflection.v1alpha": _module(
            "grpc_reflection.v1alpha",
            reflection=SimpleNamespace(),
        ),
        "sglang": _module("sglang"),
        "sglang.srt": _module("sglang.srt"),
        "sglang.srt.configs": _module("sglang.srt.configs"),
        "sglang.srt.configs.model_config": _module(
            "sglang.srt.configs.model_config",
            ModelConfig=object,
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
            ServerArgs=ServerArgs,
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
            sglang_scheduler_pb2=SimpleNamespace(),
            sglang_scheduler_pb2_grpc=SimpleNamespace(),
        ),
        "smg_grpc_servicer.sglang.health_servicer": _module(
            "smg_grpc_servicer.sglang.health_servicer",
            SGLangHealthServicer=object,
        ),
        "smg_grpc_servicer.sglang.request_manager": _module(
            "smg_grpc_servicer.sglang.request_manager",
            GrpcRequestManager=object,
            init_request_tracing=lambda: False,
        ),
        "smg_grpc_servicer.sglang.scheduler_launcher": _module(
            "smg_grpc_servicer.sglang.scheduler_launcher",
            SchedulerStartError=type("SchedulerStartError", (RuntimeError,), {}),
            launch_scheduler_process_only=lambda **_kwargs: None,
            terminate_scheduler_processes=lambda _procs: None,
        ),
        "smg_grpc_servicer.sglang.servicer": _module(
            "smg_grpc_servicer.sglang.servicer",
            SGLangSchedulerServicer=object,
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = Path(__file__).parents[1] / "smg_grpc_servicer" / "sglang" / "server.py"
    spec = importlib.util.spec_from_file_location("test_sglang_grpc_server", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The exit over a dead engine without its guard: the guard would end the
    # test process after its grace.
    monkeypatch.setattr(module, "guarded_exit", lambda code, **_kwargs: SystemExit(code))
    return module


def test_request_manager_binds_before_scheduler_launch(monkeypatch, server_mod):
    events = []
    server_args = object()
    port_args = object()
    request_manager = SimpleNamespace(shutdown=AsyncMock())
    scheduler_info = {"status": "ready"}
    scheduler_procs = [object()]

    def init_ports(args):
        assert args is server_args
        events.append("ports")
        return port_args

    def create_request_manager(**kwargs):
        assert kwargs == {
            "server_args": server_args,
            "port_args": port_args,
            "bootstrap_server": "bootstrap",
        }
        events.append("request-manager")
        return request_manager

    def launch_scheduler(**kwargs):
        assert kwargs == {
            "server_args": server_args,
            "port_args": port_args,
        }
        events.append("scheduler")
        return scheduler_info, port_args, scheduler_procs

    monkeypatch.setattr(server_mod.PortArgs, "init_new", init_ports)
    monkeypatch.setattr(server_mod, "GrpcRequestManager", create_request_manager)
    monkeypatch.setattr(server_mod, "launch_scheduler_process_only", launch_scheduler)

    result = asyncio.run(
        server_mod._launch_scheduler_with_request_manager(
            server_args,
            bootstrap_server="bootstrap",
        )
    )

    assert events == ["ports", "request-manager", "scheduler"]
    assert result == (scheduler_info, port_args, scheduler_procs, request_manager)
    request_manager.shutdown.assert_not_awaited()


def test_scheduler_launch_failure_closes_bound_manager(monkeypatch, server_mod):
    events = []
    manager = None

    class BoundSocket:
        closed = False

        def close(self):
            self.closed = True

    class RequestManager:
        def __init__(self, **_kwargs):
            nonlocal manager
            manager = self
            self.socket = BoundSocket()
            self.asyncio_tasks = {asyncio.create_task(asyncio.Event().wait())}
            self.shutdown_called = False
            events.append("request-manager")

        async def shutdown(self):
            self.shutdown_called = True
            events.append("shutdown")
            for task in self.asyncio_tasks:
                task.cancel()
            await asyncio.gather(*self.asyncio_tasks, return_exceptions=True)
            self.socket.close()

    def fail_launch(**_kwargs):
        events.append("scheduler")
        raise RuntimeError("scheduler failed")

    monkeypatch.setattr(server_mod.PortArgs, "init_new", lambda _args: object())
    monkeypatch.setattr(server_mod, "GrpcRequestManager", RequestManager)
    monkeypatch.setattr(server_mod, "launch_scheduler_process_only", fail_launch)

    async def launch():
        with pytest.raises(RuntimeError, match="scheduler failed"):
            await server_mod._launch_scheduler_with_request_manager(object())

    asyncio.run(launch())

    assert events == ["request-manager", "scheduler", "shutdown"]
    assert manager.shutdown_called
    assert manager.socket.closed
    assert all(task.done() for task in manager.asyncio_tasks)


def test_scheduler_launch_failure_never_starts_serving(monkeypatch, server_mod):
    grpc_server_factory = Mock()
    health_servicer_factory = Mock()
    warmup_thread_factory = Mock()

    launcher = sys.modules["smg_grpc_servicer.sglang.scheduler_launcher"]

    async def fail_launch(**_kwargs):
        raise launcher.SchedulerStartError("scheduler failed")

    monkeypatch.setattr(server_mod, "_launch_scheduler_with_request_manager", fail_launch)
    monkeypatch.setattr(server_mod.grpc.aio, "server", grpc_server_factory, raising=False)
    monkeypatch.setattr(server_mod, "SGLangHealthServicer", health_servicer_factory)
    monkeypatch.setattr(server_mod.threading, "Thread", warmup_thread_factory)

    server_args = SimpleNamespace(disaggregation_mode="null")
    # smg-lab #292: an engine that did not start is a named exit, not a traceback.
    with pytest.raises(SystemExit) as raised:
        asyncio.run(server_mod.serve_grpc(server_args))
    assert raised.value.code == 1

    grpc_server_factory.assert_not_called()
    health_servicer_factory.assert_not_called()
    warmup_thread_factory.assert_not_called()


def test_generation_warmup_sends_valid_sampling_defaults(monkeypatch, server_mod):
    """Proto scalar defaults must not leak into SGLang sampling validation."""
    complete = Mock()
    complete.HasField.return_value = True
    stub = Mock()
    stub.GetModelInfo.return_value = pb.GetModelInfoResponse(is_generation=True)
    stub.Generate.return_value = [complete]
    channel = Mock()

    monkeypatch.setattr(
        server_mod.grpc,
        "insecure_channel",
        Mock(return_value=channel),
        raising=False,
    )
    monkeypatch.setattr(server_mod, "sglang_scheduler_pb2", pb)
    monkeypatch.setattr(
        server_mod,
        "sglang_scheduler_pb2_grpc",
        SimpleNamespace(SglangSchedulerStub=lambda _channel: stub),
    )
    monkeypatch.setattr(
        server_mod,
        "DisaggregationMode",
        SimpleNamespace(NULL=SimpleNamespace(value="null")),
    )
    monkeypatch.setattr(server_mod.time, "sleep", lambda _seconds: None)

    server_args = SimpleNamespace(
        host="127.0.0.1",
        port=30000,
        disaggregation_mode="null",
    )
    assert server_mod._execute_grpc_server_warmup(server_args) is True

    params = stub.Generate.call_args.args[0].sampling_params
    assert params.temperature == 0.0
    assert params.top_p == 1.0
    assert params.top_k == 1
    assert params.repetition_penalty == 1.0
    assert params.n == 1
    assert params.max_new_tokens == 8
    channel.close.assert_called_once()


def _serving_server(monkeypatch, server_mod, events, *, scheduler_exited):
    """`serve_grpc` up to the wait for the scheduler, every collaborator stood in
    for; the wait reports `scheduler_exited`."""
    scheduler_procs = [object(), object()]
    request_manager = object()

    async def stop_server(grace):
        events.append("server-stop")

    async def shutdown_servicer():
        events.append("servicer-shutdown")

    async def wait_for_shutdown(*_args):
        events.append("scheduler-wait")
        return scheduler_exited

    grpc_server = SimpleNamespace(
        add_insecure_port=Mock(),
        start=AsyncMock(),
        stop=AsyncMock(side_effect=stop_server),
    )
    health_servicer = SimpleNamespace(
        set_not_serving=Mock(side_effect=lambda: events.append("health"))
    )
    servicer = SimpleNamespace(shutdown=AsyncMock(side_effect=shutdown_servicer))
    warmup_thread = SimpleNamespace(start=Mock(), is_alive=Mock(return_value=False))

    async def launch(**_kwargs):
        return {}, object(), scheduler_procs, request_manager

    monkeypatch.setattr(server_mod, "_launch_scheduler_with_request_manager", launch)
    monkeypatch.setattr(
        server_mod,
        "ModelConfig",
        SimpleNamespace(from_server_args=Mock(return_value=object())),
    )
    monkeypatch.setattr(
        server_mod.grpc.aio, "server", Mock(return_value=grpc_server), raising=False
    )
    monkeypatch.setattr(server_mod, "SGLangHealthServicer", Mock(return_value=health_servicer))
    monkeypatch.setattr(
        server_mod.health_pb2_grpc, "add_HealthServicer_to_server", Mock(), raising=False
    )
    monkeypatch.setattr(server_mod, "SGLangSchedulerServicer", Mock(return_value=servicer))
    monkeypatch.setattr(
        server_mod.sglang_scheduler_pb2_grpc,
        "add_SglangSchedulerServicer_to_server",
        Mock(),
        raising=False,
    )
    monkeypatch.setattr(
        server_mod,
        "sglang_scheduler_pb2",
        SimpleNamespace(
            DESCRIPTOR=SimpleNamespace(
                services_by_name={
                    "SglangScheduler": SimpleNamespace(full_name="test.SglangScheduler")
                }
            )
        ),
    )
    monkeypatch.setattr(server_mod.reflection, "SERVICE_NAME", "reflection", raising=False)
    monkeypatch.setattr(server_mod.reflection, "enable_server_reflection", Mock(), raising=False)
    monkeypatch.setattr(server_mod.threading, "Thread", Mock(return_value=warmup_thread))
    monkeypatch.setattr(
        server_mod, "wait_for_scheduler_shutdown", AsyncMock(side_effect=wait_for_shutdown)
    )
    monkeypatch.setattr(
        server_mod,
        "terminate_scheduler_processes",
        Mock(side_effect=lambda _procs: events.append("scheduler-terminate")),
    )
    return SimpleNamespace(
        disaggregation_mode="null",
        host="127.0.0.1",
        port=50051,
        ssl_certfile=None,
        ssl_keyfile=None,
    )


def test_a_scheduler_exit_after_start_up_ends_the_server_with_exit_code_1(monkeypatch, server_mod):
    """smg-lab #292: when the watchdog reports that a scheduler process exited
    on its own, the server shuts down as on a signal and the servicer exits
    with code 1 instead of 0 (an orchestrator restarts it)."""
    events = []
    server_args = _serving_server(monkeypatch, server_mod, events, scheduler_exited=True)

    with pytest.raises(SystemExit) as raised:
        asyncio.run(server_mod.serve_grpc(server_args, model_info={}))

    assert raised.value.code == 1
    assert events == [
        "scheduler-wait",
        "health",
        "server-stop",
        "servicer-shutdown",
        "scheduler-terminate",
    ]


def test_a_signal_shutdown_still_returns_normally(monkeypatch, server_mod):
    events = []
    server_args = _serving_server(monkeypatch, server_mod, events, scheduler_exited=False)
    assert asyncio.run(server_mod.serve_grpc(server_args, model_info={})) is None
    assert events[0] == "scheduler-wait" and events[-1] == "scheduler-terminate"


def test_a_servicer_configuration_error_is_a_named_exit_not_a_traceback(
    monkeypatch, server_mod, caplog
):
    """smg-lab #292: a knob the servicer cannot use (the implementation
    selector here) ends the start with one ERROR line naming it and exit
    code 2, before any scheduler or engine work."""
    monkeypatch.setenv("SMG_SGLANG_SERVICER_IMPL", "bogus")
    launched = Mock()
    monkeypatch.setattr(server_mod, "_launch_scheduler_with_request_manager", launched)
    caplog.set_level(logging.ERROR)

    with pytest.raises(SystemExit) as raised:
        asyncio.run(server_mod.serve_grpc(SimpleNamespace(disaggregation_mode="null")))

    assert raised.value.code == 2
    assert "SMG_SGLANG_SERVICER_IMPL" in caplog.text and "bogus" in caplog.text
    launched.assert_not_called()


def test_server_waits_for_scheduler_shutdown(monkeypatch, server_mod):
    events = []
    scheduler_procs = [object(), object()]
    request_manager = object()

    async def stop_server(grace):
        assert grace == 5.0
        events.append("server-stop")

    async def shutdown_servicer():
        events.append("servicer-shutdown")

    async def wait_for_shutdown(*_args):
        events.append("scheduler-wait")

    grpc_server = SimpleNamespace(
        add_insecure_port=Mock(),
        start=AsyncMock(),
        stop=AsyncMock(side_effect=stop_server),
    )
    health_servicer = SimpleNamespace(
        set_not_serving=Mock(side_effect=lambda: events.append("health"))
    )
    servicer = SimpleNamespace(shutdown=AsyncMock(side_effect=shutdown_servicer))
    warmup_thread = SimpleNamespace(
        start=Mock(),
        is_alive=Mock(return_value=False),
    )
    wait_for_scheduler_shutdown = AsyncMock(side_effect=wait_for_shutdown)
    terminate_scheduler_processes = Mock(
        side_effect=lambda _procs: events.append("scheduler-terminate")
    )

    async def launch(**_kwargs):
        return {}, object(), scheduler_procs, request_manager

    model_config = object()
    monkeypatch.setattr(server_mod, "_launch_scheduler_with_request_manager", launch)
    monkeypatch.setattr(
        server_mod,
        "ModelConfig",
        SimpleNamespace(from_server_args=Mock(return_value=model_config)),
    )
    monkeypatch.setattr(
        server_mod.grpc.aio,
        "server",
        Mock(return_value=grpc_server),
        raising=False,
    )
    monkeypatch.setattr(
        server_mod,
        "SGLangHealthServicer",
        Mock(return_value=health_servicer),
    )
    monkeypatch.setattr(
        server_mod.health_pb2_grpc,
        "add_HealthServicer_to_server",
        Mock(),
        raising=False,
    )
    monkeypatch.setattr(
        server_mod,
        "SGLangSchedulerServicer",
        Mock(return_value=servicer),
    )
    monkeypatch.setattr(
        server_mod.sglang_scheduler_pb2_grpc,
        "add_SglangSchedulerServicer_to_server",
        Mock(),
        raising=False,
    )
    monkeypatch.setattr(
        server_mod,
        "sglang_scheduler_pb2",
        SimpleNamespace(
            DESCRIPTOR=SimpleNamespace(
                services_by_name={
                    "SglangScheduler": SimpleNamespace(full_name="test.SglangScheduler")
                }
            )
        ),
    )
    monkeypatch.setattr(server_mod.reflection, "SERVICE_NAME", "reflection", raising=False)
    monkeypatch.setattr(
        server_mod.reflection,
        "enable_server_reflection",
        Mock(),
        raising=False,
    )
    monkeypatch.setattr(
        server_mod.threading,
        "Thread",
        Mock(return_value=warmup_thread),
    )
    monkeypatch.setattr(
        server_mod,
        "wait_for_scheduler_shutdown",
        wait_for_scheduler_shutdown,
    )
    monkeypatch.setattr(
        server_mod,
        "terminate_scheduler_processes",
        terminate_scheduler_processes,
    )

    server_args = SimpleNamespace(
        disaggregation_mode="null",
        host="127.0.0.1",
        port=50051,
        ssl_certfile=None,
        ssl_keyfile=None,
    )
    asyncio.run(server_mod.serve_grpc(server_args, model_info={}))

    wait_for_scheduler_shutdown.assert_awaited_once()
    waited_processes, stop_event = wait_for_scheduler_shutdown.await_args.args
    assert waited_processes is scheduler_procs
    assert isinstance(stop_event, asyncio.Event)
    assert events == [
        "scheduler-wait",
        "health",
        "server-stop",
        "servicer-shutdown",
        "scheduler-terminate",
    ]
    grpc_server.start.assert_awaited_once_with()
    grpc_server.stop.assert_awaited_once_with(5.0)
    servicer.shutdown.assert_awaited_once_with()
    terminate_scheduler_processes.assert_called_once_with(scheduler_procs)


def test_post_launch_startup_failure_releases_owned_resources(monkeypatch, server_mod):
    request_manager = SimpleNamespace(shutdown=AsyncMock())
    scheduler_procs = [object(), object()]
    terminate_scheduler_processes = Mock()

    async def launch(**_kwargs):
        return {"status": "ready"}, object(), scheduler_procs, request_manager

    model_config_factory = Mock(side_effect=RuntimeError("model config failed"))
    monkeypatch.setattr(server_mod, "_launch_scheduler_with_request_manager", launch)
    monkeypatch.setattr(
        server_mod,
        "ModelConfig",
        SimpleNamespace(from_server_args=model_config_factory),
    )
    monkeypatch.setattr(
        server_mod,
        "terminate_scheduler_processes",
        terminate_scheduler_processes,
        raising=False,
    )

    server_args = SimpleNamespace(disaggregation_mode="null")
    with pytest.raises(RuntimeError, match="model config failed"):
        asyncio.run(server_mod.serve_grpc(server_args))

    request_manager.shutdown.assert_awaited_once_with()
    terminate_scheduler_processes.assert_called_once_with(scheduler_procs)


def test_failed_startup_cleanup_runs_in_reverse_ownership_order(monkeypatch, server_mod):
    events = []
    scheduler_procs = [object()]

    health_servicer = SimpleNamespace(
        set_not_serving=Mock(side_effect=lambda: events.append("health"))
    )

    async def stop_server(grace):
        assert grace == 0
        events.append("server")

    async def shutdown_manager():
        events.append("manager")

    server = SimpleNamespace(stop=AsyncMock(side_effect=stop_server))
    request_manager = SimpleNamespace(shutdown=AsyncMock(side_effect=shutdown_manager))
    terminate_scheduler_processes = Mock(side_effect=lambda _procs: events.append("schedulers"))
    monkeypatch.setattr(
        server_mod,
        "terminate_scheduler_processes",
        terminate_scheduler_processes,
    )

    asyncio.run(
        server_mod._cleanup_failed_startup(
            server,
            health_servicer,
            request_manager,
            scheduler_procs,
        )
    )

    assert events == ["health", "server", "schedulers", "manager"]
    terminate_scheduler_processes.assert_called_once_with(scheduler_procs)


def test_request_tracing_initializes_after_the_runtime_context_is_published(
    monkeypatch, server_mod
):
    """smg-lab#182: the Python servicer initializes SGLang's tracing in its
    own process (as SGLang's HTTP frontend does) once the runtime context is
    published and before anything else starts; the request manager then
    starts each request's trace from the Router's context."""
    events = []
    monkeypatch.setattr(server_mod, "publish", lambda _args, role: events.append(f"publish:{role}"))

    def init_tracing():
        events.append("tracing")
        return True

    async def stop_here(**_kwargs):
        events.append("launch")
        raise RuntimeError("stop here")

    monkeypatch.setattr(server_mod, "init_request_tracing", init_tracing)
    monkeypatch.setattr(server_mod, "_launch_scheduler_with_request_manager", stop_here)
    monkeypatch.delenv("SMG_SGLANG_SERVICER_IMPL", raising=False)
    with pytest.raises(RuntimeError, match="stop here"):
        asyncio.run(server_mod.serve_grpc(SimpleNamespace(disaggregation_mode="null")))
    assert events == ["publish:tokenizer", "tracing", "launch"]
