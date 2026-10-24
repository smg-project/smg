"""Engine-free tests for the TokenSpeed Python servicer's lifecycle (smg-lab
#292): the servicer exits non-zero when its engine dies, during the launch or
later, instead of running on as a worker that cannot serve."""

from __future__ import annotations

import asyncio
import importlib.util
import json
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


_SCHEDULER_PB2 = SimpleNamespace(
    DESCRIPTOR=SimpleNamespace(
        services_by_name={
            "TokenSpeedScheduler": SimpleNamespace(full_name="test.TokenSpeedScheduler")
        }
    ),
    GenerateResponse=object,
)


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
            # The load-reply conversion the Rust-path module imports reads
            # the scheduler protobuf through the package's lazy attribute.
            tokenspeed_scheduler_pb2=_SCHEDULER_PB2,
        ),
        "smg_grpc_proto.generated": _module(
            "smg_grpc_proto.generated",
            tokenspeed_scheduler_pb2=_SCHEDULER_PB2,
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
    # The server module imports the Rust-path module, which reads `grpc` at
    # import time: imported with the real `grpc` first, so the stand-in below
    # does not meet it (the whole test set has it cached by now; this file
    # alone would not).
    import smg_grpc_servicer.tokenspeed.rust  # noqa: F401

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


def test_the_python_servicer_turns_the_kv_event_publisher_on_by_default_too(
    monkeypatch, server_mod, caplog
):
    """A launcher without --kv-events-config left TokenSpeed publishing nothing
    under the Python servicer, while the Rust path turned the publisher on by
    itself: `serve_grpc` applies the same default to the server args before
    the engine and the servicer are built from them, with the same opt-out and
    the same log line; an encode worker, which has no KV cache, is left as
    given."""
    from smg_grpc_servicer.tokenspeed import rust as tokenspeed_rust

    monkeypatch.delenv(tokenspeed_rust.KV_EVENTS_ENV, raising=False)
    servicer_module = sys.modules["smg_grpc_servicer.tokenspeed.servicer"]

    def serve(configure):
        """One start up to a signal, every collaborator stood in for; what the
        engine launch and the server args ended up with."""
        events = []
        server_args = _wire(monkeypatch, server_mod, events)
        configure(server_args)
        seen = {}

        def launch_engine(args):
            seen["launched_with"] = getattr(args, "kv_events_config", None)
            return object(), {}

        monkeypatch.setattr(server_mod, "launch_engine", launch_engine)

        async def exercise():
            task = asyncio.create_task(server_mod.serve_grpc(server_args))
            while "server-start" not in events:
                await asyncio.sleep(0)
            _stop_event_of(task).set()
            return await task

        caplog.clear()
        with caplog.at_level(logging.INFO):
            assert asyncio.run(exercise()) is None
        servicer = servicer_module.TokenSpeedSchedulerServicer.instances[-1]
        assert servicer.kwargs["server_args"] is server_args
        return seen["launched_with"], getattr(server_args, "kv_events_config", None)

    # Nothing given: the engine starts with the publisher on, at a free port
    # of this host rather than TokenSpeed's fixed default, the servicer reads
    # the same server args, and the operator reads why.
    launched_with, applied = serve(lambda args: None)
    assert launched_with == applied
    config = json.loads(applied)
    assert config == {
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        "endpoint": config["endpoint"],
    }
    assert config["endpoint"].startswith("tcp://*:") and config["endpoint"] != "tcp://*:5557"
    assert (
        "KV event publishing enabled: no --kv-events-config was given, so TokenSpeed's ZMQ "
        "publisher is on at a port found free on this host and SubscribeKvEvents relays "
        f"it; pass --kv-events-config to name the endpoint, or set {tokenspeed_rust.KV_EVENTS_ENV}=0 to leave it off"
    ) in caplog.text
    # The opt-out: the server args reach the engine as parsed, and nothing is said.
    monkeypatch.setenv(tokenspeed_rust.KV_EVENTS_ENV, "false")
    assert serve(lambda args: None) == (None, None)
    assert "KV event publishing" not in caplog.text
    # An explicit configuration is kept as given, off included.
    monkeypatch.delenv(tokenspeed_rust.KV_EVENTS_ENV)
    given = '{"enable_kv_cache_events": false}'
    assert serve(lambda args: setattr(args, "kv_events_config", given)) == (given, given)
    # An encode worker runs the vision tower only: no KV cache, nothing to turn on.
    assert serve(lambda args: setattr(args, "disaggregation_mode", "encode")) == (None, None)
    assert "KV event publishing" not in caplog.text
    # A hybrid model keeps recurrent state in State groups, which span no
    # tokens, and TokenSpeed's KV-event drain aborts the engine on them: the
    # engine's own vocabulary of which layer labels get a State group, read
    # off the model config built here as the launcher builds it, leaves the
    # publisher off for such a model, and the operator reads why once.
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed.runtime.layers.attention.kv_cache.recipes.spec",
        _module(
            "tokenspeed.runtime.layers.attention.kv_cache.recipes.spec",
            STATE_LAYER_TYPES=frozenset({"linear_attention"}),
        ),
    )
    hybrid = SimpleNamespace(
        hf_config=SimpleNamespace(layer_types=["linear_attention", "full_attention"])
    )
    monkeypatch.setattr(tokenspeed_rust, "_model_config", lambda args: hybrid)
    assert serve(lambda args: setattr(args, "model", "org/hybrid")) == (None, None)
    assert (
        "KV event publishing left off: org/hybrid keeps recurrent state in 1 of its 2 layers "
        "(linear_attention): State groups of TokenSpeed's cache, which span no tokens, and "
        "TokenSpeed's KV-event drain aborts the engine on a group with no token span; pass "
        "--kv-events-config to turn the publisher on anyway"
    ) in caplog.text


def test_a_host_with_no_free_publisher_port_is_a_named_exit_on_the_python_path_too(
    monkeypatch, server_mod, caplog
):
    """The default's endpoint is a port found free on the host; a host with no
    free block of them is a configuration error, and the Python path answers
    it as the Rust path does: one ERROR line naming the knob and exit code 2,
    before the engine is launched, not a traceback."""
    from smg_grpc_servicer import rust_lifecycle
    from smg_grpc_servicer.tokenspeed import rust as tokenspeed_rust

    monkeypatch.delenv(tokenspeed_rust.KV_EVENTS_ENV, raising=False)
    refusal = "no block of 1 consecutive free ports for the KV event publishers"

    def no_block(ranks=1):
        raise rust_lifecycle.ServicerConfigError(
            f"{refusal} after 64 draws; pass --kv-events-config with an endpoint"
        )

    monkeypatch.setattr(rust_lifecycle, "free_kv_events_endpoint", no_block)
    server_args = _wire(monkeypatch, server_mod, [])
    launched = Mock()
    monkeypatch.setattr(server_mod, "launch_engine", launched)
    caplog.set_level(logging.ERROR)
    with pytest.raises(SystemExit) as raised:
        asyncio.run(server_mod.serve_grpc(server_args))
    assert raised.value.code == 2
    assert refusal in caplog.text and "--kv-events-config" in caplog.text
    launched.assert_not_called()


def test_a_failed_planned_warm_up_still_flips_the_worker_to_serving(
    monkeypatch, server_mod, caplog
):
    """The planned pass is an optimisation: when it raises past its own
    guards, the one-token warm-up still counts and the health flips to
    SERVING (a raise would otherwise leave the warm-up thread before it)."""
    caplog.set_level(logging.ERROR)
    final = Mock()
    final.HasField.return_value = True
    stub = Mock()
    stub.GetModelInfo.return_value = SimpleNamespace(max_context_length=4096, vocab_size=32000)
    stub.Generate.return_value = [final]
    channel = Mock()
    monkeypatch.setattr(server_mod.grpc, "insecure_channel", Mock(return_value=channel))
    monkeypatch.setattr(
        server_mod.tokenspeed_scheduler_pb2_grpc, "TokenSpeedSchedulerStub", lambda _channel: stub
    )
    monkeypatch.setattr(
        server_mod,
        "tokenspeed_scheduler_pb2",
        SimpleNamespace(
            GetModelInfoRequest=lambda: None,
            GenerateRequest=lambda **kwargs: SimpleNamespace(**kwargs),
            TokenizedInput=lambda **kwargs: SimpleNamespace(**kwargs),
            SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )

    def broken(_server_args, _facts):
        raise AttributeError("no such argument on this engine release")

    monkeypatch.setattr(server_mod, "warmup_inputs", broken)
    health = SimpleNamespace(set_serving=Mock())
    server_args = SimpleNamespace(host="127.0.0.1", port=30000, disaggregation_mode="null")

    server_mod._wait_and_warmup(server_args, health)

    health.set_serving.assert_called_once_with()
    channel.close.assert_called_once()
    assert any(
        "The planned warm-up failed; the server serves without it" in r.getMessage()
        for r in caplog.records
    )
