"""Engine-free tests for the TokenSpeed Python servicer's lifecycle (smg-lab
#292): the servicer exits non-zero when its engine dies, during the launch or
later, instead of running on as a worker that cannot serve."""

from __future__ import annotations

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


class _HardExit(BaseException):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


@pytest.fixture
def server_mod(monkeypatch):
    """Load the TokenSpeed server.py with stand-ins for the engine, gRPC and
    the proto package; the servicer classes are recording fakes."""

    class ServerArgs:
        pass

    class SchedulerServicer:
        instances: list = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.shutdown = AsyncMock()
            SchedulerServicer.instances.append(self)

    class HealthServicer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.set_not_serving = Mock()

    class EncoderServicer:
        """The encode worker's servicer: its shutdown takes no arguments."""

        instances: list = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.shutdown_calls = 0
            EncoderServicer.instances.append(self)

        async def shutdown(self):
            self.shutdown_calls += 1

    EncoderServicer.instances.clear()

    stubs = {
        "grpc": _module("grpc", aio=SimpleNamespace(), insecure_channel=Mock()),
        "grpc_health": _module("grpc_health"),
        "grpc_health.v1": _module(
            "grpc_health.v1",
            health_pb2_grpc=SimpleNamespace(add_HealthServicer_to_server=Mock()),
        ),
        "grpc_reflection": _module("grpc_reflection"),
        "grpc_reflection.v1alpha": _module(
            "grpc_reflection.v1alpha",
            reflection=SimpleNamespace(SERVICE_NAME="reflection", enable_server_reflection=Mock()),
        ),
        "smg_grpc_proto": _module(
            "smg_grpc_proto",
            tokenspeed_scheduler_pb2_grpc=SimpleNamespace(
                add_TokenSpeedSchedulerServicer_to_server=Mock(),
                TokenSpeedSchedulerStub=Mock(),
            ),
        ),
        "smg_grpc_proto.generated": _module(
            "smg_grpc_proto.generated",
            tokenspeed_scheduler_pb2=SimpleNamespace(
                DESCRIPTOR=SimpleNamespace(
                    services_by_name={
                        "TokenSpeedScheduler": SimpleNamespace(full_name="test.TokenSpeedScheduler")
                    }
                ),
                GenerateResponse=object,
            ),
            tokenspeed_encoder_pb2=SimpleNamespace(
                DESCRIPTOR=SimpleNamespace(
                    services_by_name={
                        "TokenSpeedEncoder": SimpleNamespace(full_name="test.TokenSpeedEncoder")
                    }
                ),
            ),
            tokenspeed_encoder_pb2_grpc=SimpleNamespace(
                add_TokenSpeedEncoderServicer_to_server=Mock(),
            ),
        ),
        "smg_grpc_servicer.tokenspeed.encoder_servicer": _module(
            "smg_grpc_servicer.tokenspeed.encoder_servicer",
            TokenSpeedEncoderServicer=EncoderServicer,
        ),
        "tokenspeed": _module("tokenspeed"),
        "tokenspeed.runtime": _module("tokenspeed.runtime"),
        "tokenspeed.runtime.utils": _module("tokenspeed.runtime.utils"),
        "tokenspeed.runtime.utils.server_args": _module(
            "tokenspeed.runtime.utils.server_args", ServerArgs=ServerArgs
        ),
        "smg_grpc_servicer.tokenspeed.health_servicer": _module(
            "smg_grpc_servicer.tokenspeed.health_servicer",
            TokenSpeedHealthServicer=HealthServicer,
        ),
        "smg_grpc_servicer.tokenspeed.scheduler_launcher": _module(
            "smg_grpc_servicer.tokenspeed.scheduler_launcher",
            launch_engine=lambda _args: (object(), {}),
        ),
        "smg_grpc_servicer.tokenspeed.servicer": _module(
            "smg_grpc_servicer.tokenspeed.servicer",
            TokenSpeedSchedulerServicer=SchedulerServicer,
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[1] / "smg_grpc_servicer" / "tokenspeed" / "server.py"
    spec = importlib.util.spec_from_file_location("test_tokenspeed_grpc_server", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The exit over a dead engine without its guard: the guard would end the
    # test process after its grace; the grace it was asked for is recorded.
    module._armed_graces = []  # noqa: SLF001 -- test hook

    def fake_guarded_exit(code, *, grace_secs=None):
        module._armed_graces.append(grace_secs)  # noqa: SLF001
        return SystemExit(code)

    monkeypatch.setattr(module, "guarded_exit", fake_guarded_exit)
    return module


class _FakeWatch:
    """The engine-process watch as the server drives it: `start` keeps the
    `on_exit` callback for the test to fire, `stop` is recorded."""

    instances: list = []

    def __init__(self, on_exit, **_kwargs):
        self.on_exit = on_exit
        self.started = False
        self.stopped = False
        self.terminated = 0
        _FakeWatch.instances.append(self)

    def start(self):
        self.started = True

    def freeze(self):
        self.frozen = True
        return 1

    def stop(self):
        self.stopped = True

    def terminate_seen(self):
        self.terminated += 1
        return 1


def _wire(monkeypatch, server_mod, events):
    _FakeWatch.instances.clear()
    monkeypatch.setattr(server_mod, "EngineProcessWatch", _FakeWatch)

    async def stop_server(grace):
        events.append(("server-stop", grace))

    grpc_server = SimpleNamespace(
        add_insecure_port=Mock(),
        start=AsyncMock(side_effect=lambda: events.append("server-start")),
        stop=AsyncMock(side_effect=stop_server),
    )
    monkeypatch.setattr(
        server_mod.grpc.aio, "server", Mock(return_value=grpc_server), raising=False
    )
    monkeypatch.setattr(
        server_mod.threading,
        "Thread",
        Mock(return_value=SimpleNamespace(start=Mock(), is_alive=Mock(return_value=False))),
    )
    return SimpleNamespace(host="127.0.0.1", port=50051, disaggregation_mode="null")


def test_an_engine_that_exits_after_start_up_ends_the_server_with_exit_code_1(
    monkeypatch, server_mod, caplog
):
    """The watch reports an engine process gone: the server stops as on a
    signal (health down, servicer shut down, server stopped) and the servicer
    exits with code 1, so an orchestrator restarts it."""
    events = []
    server_args = _wire(monkeypatch, server_mod, events)
    caplog.set_level(logging.ERROR)
    armed = server_mod._armed_graces  # noqa: SLF001

    async def exercise():
        task = asyncio.create_task(server_mod.serve_grpc(server_args))
        while "server-start" not in events:
            await asyncio.sleep(0)
        watch = _FakeWatch.instances[0]
        assert watch.started
        # The watch thread reports from outside the loop; here from inside,
        # which the thread-safe hand-over tolerates.
        watch.on_exit(4242, 1)
        await task

    with pytest.raises(SystemExit) as raised:
        asyncio.run(exercise())
    assert raised.value.code == 1
    assert _FakeWatch.instances[0].stopped
    assert events == ["server-start", ("server-stop", 5.0)]
    assert "engine process 4242 exited with code 1" in caplog.text
    # The guard's grace covers the servicer's own drain bound and the bounded stops.
    assert server_mod.ENGINE_EXIT_GRACE_SECS >= server_mod.SHUTDOWN_DRAIN_SECS + 5.0 + 5.0
    assert armed == [server_mod.ENGINE_EXIT_GRACE_SECS]


def test_a_signal_shutdown_still_returns_normally(monkeypatch, server_mod):
    events = []
    server_args = _wire(monkeypatch, server_mod, events)

    async def exercise():
        task = asyncio.create_task(server_mod.serve_grpc(server_args))
        while "server-start" not in events:
            await asyncio.sleep(0)
        # What the signal handler does.
        for handler_task in asyncio.all_tasks():
            pass
        stop = next(
            frame.f_locals["stop_event"]
            for frame in [task.get_coro().cr_frame]
            if "stop_event" in frame.f_locals
        )
        stop.set()
        return await task

    assert asyncio.run(exercise()) is None
    assert _FakeWatch.instances[0].stopped
    assert events == ["server-start", ("server-stop", 5.0)]


def _stop_event_of(task):
    return next(
        frame.f_locals["stop_event"]
        for frame in [task.get_coro().cr_frame]
        if "stop_event" in frame.f_locals
    )


def test_an_encode_worker_shuts_its_encoder_servicer_down(monkeypatch, server_mod, caplog):
    """In encode mode the servicer is the encoder servicer, whose shutdown takes
    no drain bound (encode is fire-and-forget): the bound goes to the scheduler
    servicer only, and the encoder's own tear-down runs on a signal."""
    events = []
    server_args = _wire(monkeypatch, server_mod, events)
    server_args.disaggregation_mode = "encode"
    caplog.set_level(logging.ERROR)
    encoder_servicer = sys.modules["smg_grpc_servicer.tokenspeed.encoder_servicer"]

    async def exercise():
        task = asyncio.create_task(server_mod.serve_grpc(server_args))
        while "server-start" not in events:
            await asyncio.sleep(0)
        _stop_event_of(task).set()
        return await task

    assert asyncio.run(exercise()) is None
    encoder = encoder_servicer.TokenSpeedEncoderServicer.instances[-1]
    assert encoder.shutdown_calls == 1, "the encoder servicer's own tear-down ran"
    assert "servicer.shutdown() raised" not in caplog.text
    assert events == ["server-start", ("server-stop", 5.0)]


def test_a_generation_worker_hands_its_servicer_the_drain_bound(monkeypatch, server_mod):
    events = []
    server_args = _wire(monkeypatch, server_mod, events)

    async def exercise():
        task = asyncio.create_task(server_mod.serve_grpc(server_args))
        while "server-start" not in events:
            await asyncio.sleep(0)
        _stop_event_of(task).set()
        return await task

    assert asyncio.run(exercise()) is None
    servicer_module = sys.modules["smg_grpc_servicer.tokenspeed.servicer"]
    scheduler_servicer = servicer_module.TokenSpeedSchedulerServicer.instances[-1]
    scheduler_servicer.shutdown.assert_awaited_once_with(
        drain_timeout_secs=server_mod.SHUTDOWN_DRAIN_SECS
    )


def test_an_engine_that_exits_during_the_launch_ends_the_process_at_once(
    monkeypatch, server_mod, caplog
):
    """While the launch still waits on the engine's ready report (the main
    thread is inside the engine's launcher), the watch ends the process: the
    other engine processes terminated, one ERROR line, exit code 1."""
    events = []
    server_args = _wire(monkeypatch, server_mod, events)
    caplog.set_level(logging.ERROR)

    def hard_exit(code):
        raise _HardExit(code)

    monkeypatch.setattr(server_mod.engine_watchdog, "exit_process", hard_exit)

    def launch_that_sees_the_engine_die(_args):
        # The watch thread fires while the launch is still waiting.
        _FakeWatch.instances[0].on_exit(4242, 0)
        raise AssertionError("the launch does not return")

    monkeypatch.setattr(server_mod, "launch_engine", launch_that_sees_the_engine_die)

    with pytest.raises(_HardExit) as raised:
        asyncio.run(server_mod.serve_grpc(server_args))
    assert raised.value.code == 1
    assert _FakeWatch.instances[0].terminated == 1
    assert "engine process 4242 exited with code 0" in caplog.text
    assert events == []


def test_an_engine_that_does_not_start_is_a_named_exit(monkeypatch, server_mod, caplog):
    events = []
    server_args = _wire(monkeypatch, server_mod, events)
    caplog.set_level(logging.ERROR)

    def failing_launch(_args):
        raise RuntimeError("launch_engine() returned no AsyncLLM")

    monkeypatch.setattr(server_mod, "launch_engine", failing_launch)
    with pytest.raises(SystemExit) as raised:
        asyncio.run(server_mod.serve_grpc(server_args))
    assert raised.value.code == 1
    watch = _FakeWatch.instances[0]
    assert watch.stopped and watch.terminated == 1
    # The traceback stays with the log line: the only clue to an unexpected failure.
    assert "The TokenSpeed engine did not start" in caplog.text
    assert "launch_engine() returned no AsyncLLM" in caplog.text
    assert events == []
