"""Readiness after a warm-up transport failure, through the real lifecycle.

The original full warm-up suite supplies its fake server/engine and actual
gRPC status type. No test substitutes a copied lifecycle or pass function.
"""

import asyncio

import grpc
import pytest
from smg_grpc_servicer import rust_lifecycle, warmup
from test_warmup import SMALL, _EngineOf, _RaisedStatus, _ServerOf, _supervised, _Wire


@pytest.mark.parametrize("async_wire", [False, True], ids=["blocking", "async"])
@pytest.mark.parametrize("flush_fails", [False, True], ids=["flush-accepted", "flush-deadline"])
def test_later_requests_and_flush_do_not_acknowledge_the_timed_out_request(async_wire, flush_fails):
    plan = warmup.plan_warmup(SMALL, "fast")
    timed_out = plan.steps[0].requests[0].request_id

    class Wire(_Wire):
        def generate(self, request, timeout):
            super().generate(request, timeout)
            if request.request_id == timed_out:
                raise _RaisedStatus(grpc.StatusCode.DEADLINE_EXCEEDED, "still running")

        def flush_cache(self):
            super().flush_cache()
            if flush_fails:
                raise TimeoutError("flush deadline")
            return True

    class AsyncWire(Wire):
        async def generate(self, request, timeout):
            return super().generate(request, timeout)

        async def flush_cache(self):
            return super().flush_cache()

    wire = AsyncWire() if async_wire else Wire()
    if async_wire:
        report = asyncio.run(warmup.run_warmup_async(plan, wire, budget_secs=30))
    else:
        report = warmup.run_warmup(plan, wire, budget_secs=30)
    assert len(wire.requests) == sum(len(step.requests) for step in plan.steps)
    assert report.covered == len(plan.steps) - 1
    assert report.errors == report.unrecovered_transport_errors == 1
    assert report.flushed is (not flush_fails)
    # Every later Generate can succeed; even an accepted flush carries no
    # receipt for the earlier request's cancellation or engine completion.
    code, server, engine = _supervised(lambda: report, stop_after=0.02)
    assert code == 1
    assert server.events == ["serving:False", "stop"]
    assert engine.events == ["terminate"]


def test_a_cancelled_generate_is_sticky_in_an_otherwise_running_async_pass():
    plan = warmup.plan_warmup(SMALL, "fast")
    cancelled = plan.steps[0].requests[0].request_id

    class Wire:
        def __init__(self):
            self.requests = []

        async def generate(self, request, timeout):
            self.requests.append(request.request_id)
            if request.request_id == cancelled:
                raise asyncio.CancelledError("engine consume cancelled")

        async def flush_cache(self):
            return True

    async def run():
        wire = Wire()
        report = await warmup.run_warmup_async(plan, wire, budget_secs=30)
        assert len(wire.requests) == sum(len(step.requests) for step in plan.steps)
        assert not (asyncio.all_tasks() - {asyncio.current_task()})
        return report

    report = asyncio.run(run())
    assert report.errors == report.unrecovered_transport_errors == 1
    assert report.flushed is True


def test_cancelling_the_parent_pass_propagates_and_closes_its_generate_tasks():
    class Wire:
        def __init__(self):
            self.started = asyncio.Event()
            self.active = 0
            self.flushes = 0
            self.closed = False

        async def generate(self, request, timeout):
            self.active += 1
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.active -= 1

        async def flush_cache(self):
            self.flushes += 1
            return True

        async def close(self):
            self.closed = True

    async def run():
        wire = Wire()
        task = asyncio.create_task(
            warmup.run_warmup_async(warmup.plan_warmup(SMALL, "fast"), wire, budget_secs=30)
        )
        try:
            await asyncio.wait_for(wire.started.wait(), 1.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await wire.close()
        assert task.cancelled()
        assert wire.active == wire.flushes == 0
        assert wire.closed
        assert not (asyncio.all_tasks() - {asyncio.current_task()})

    asyncio.run(run())


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("Generate deadline"),
        asyncio.TimeoutError("engine deadline"),  # noqa: UP041 -- distinct before Python 3.11
        ConnectionResetError("connection reset"),
        _RaisedStatus(grpc.StatusCode.DEADLINE_EXCEEDED),
        _RaisedStatus(grpc.StatusCode.UNAVAILABLE),
        _RaisedStatus(grpc.StatusCode.CANCELLED),
    ],
    ids=[
        "timeout",
        "async-timeout",
        "connection-reset",
        "grpc-deadline",
        "grpc-unavailable",
        "grpc-cancelled",
    ],
)
def test_a_direct_transport_exception_exits_instead_of_bypassing_the_report(error):
    def warm():
        raise error

    code, server, engine = _supervised(warm, stop_after=0.02)
    assert code == 1
    assert server.events == ["serving:False", "stop"]
    assert engine.events == ["terminate"]


@pytest.mark.parametrize(
    "error",
    [
        warmup.UnservedRequest("explicit terminal refusal"),
        RuntimeError("DEADLINE_EXCEEDED is text, not a transport status"),
        RuntimeError("the wire refused"),
    ],
    ids=["terminal-refusal", "no-string-matching", "ordinary-warmup-error"],
)
def test_an_ordinary_warmup_error_does_not_replace_current_engine_health(error):
    def warm():
        raise error

    code, server, engine = _supervised(warm, stop_after=0.02)
    assert code == 0
    assert server.events == ["ready:True", "serving:False", "stop"]
    assert engine.events == ["terminate"]


def test_terminal_refusals_and_a_refused_flush_remain_best_effort():
    class Refusing(_Wire):
        def generate(self, request, timeout):
            super().generate(request, timeout)
            raise warmup.UnservedRequest("INVALID_ARGUMENT: terminal shape refusal")

        def flush_cache(self):
            super().flush_cache()
            raise RuntimeError("UNIMPLEMENTED")

    report = warmup.run_warmup(warmup.plan_warmup(SMALL, "fast"), Refusing(), budget_secs=30)
    assert report.errors > 0
    assert report.unrecovered_transport_errors == 0
    assert report.flushed is False
    code, server, engine = _supervised(lambda: report, stop_after=0.02)
    assert code == 0
    assert server.events == ["ready:True", "serving:False", "stop"]
    assert engine.events == ["terminate"]


@pytest.mark.parametrize("failure", ["server-error", "server-exited", "engine-exited"])
def test_a_known_fatal_server_or_engine_never_gets_a_transient_readiness_release(failure):
    server, engine = _ServerOf(), _EngineOf()
    server.engine_ready = True

    def warm():
        # This runs on the supervisor's actual warm-up thread. The completed
        # report and fatal state are visible together on its next poll.
        if failure == "server-error":
            server.last_error = "fatal transport"
        elif failure == "server-exited":
            server.running = False
        else:
            engine.code = 9
        return warmup.WarmupReport(warmup.plan_warmup(SMALL, "fast"))

    async def run():
        return await asyncio.wait_for(
            rust_lifecycle.supervise(
                server,
                engine,
                warmup=warm,
                stop_event=asyncio.Event(),
                poll_secs=0.005,
                drain_secs=0.0,
                stop_timeout=0.1,
            ),
            1.0,
        )

    assert asyncio.run(run()) == 1
    assert server.events == ["serving:False", "stop"]
    assert engine.events == ([] if failure == "engine-exited" else ["terminate"])
