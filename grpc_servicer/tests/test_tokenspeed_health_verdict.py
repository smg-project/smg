"""Engine-free coverage of the TokenSpeed RPC's answer: a probe the engine
refuses at once is answered unhealthy at the wait's first step, not at the
deadline; one that gets no output and no push is answered at the deadline;
the next probe, answered by the engine, is healthy again."""

import asyncio
from types import SimpleNamespace

import pytest
import smg_grpc_servicer.health_probe as probe_wait
from smg_grpc_servicer.health_probe import PROBE_STEP_SECS

pytest.importorskip("msgspec")
pytest.importorskip("zmq")
pytest.importorskip("numpy")
pytest.importorskip("smg_grpc_proto")
from smg_grpc_proto.generated import tokenspeed_scheduler_pb2  # noqa: E402

DEADLINE = 20


class _Engine:
    """An engine whose answer to each probe is scripted: ``refuse`` raises at
    the request, ``silence`` never yields, ``answer`` yields one output."""

    def __init__(self, *script):
        self.script = list(script)
        self.gracefully_exit = False
        self.last_receive_tstamp = 0.0
        self.aborted = []

    async def generate_request(self, probe):
        turn = self.script.pop(0)
        if turn == "refuse":
            raise RuntimeError("the engine refused the request")
        if turn == "silence":
            await asyncio.Event().wait()
        yield {"output_ids": [1], "meta_info": {"finish_reason": None}}

    def abort_request(self, rid):
        self.aborted.append(rid)


@pytest.fixture
def bed(tokenspeed_servicer_mod, monkeypatch):
    mod = tokenspeed_servicer_mod
    clock = SimpleNamespace(now=100.0)
    steps = []

    async def poll(delay):
        # The wait's step: the probe's task gets its turns, the clock a second.
        steps.append(delay)
        for _ in range(5):
            await asyncio.sleep(0)
        clock.now += 1

    monkeypatch.setattr(mod, "HEALTH_CHECK_TIMEOUT", DEADLINE)
    monkeypatch.setattr(mod, "time", SimpleNamespace(time=lambda: clock.now))
    monkeypatch.setattr(probe_wait, "time", SimpleNamespace(time=lambda: clock.now))
    monkeypatch.setattr(probe_wait, "asyncio", SimpleNamespace(sleep=poll))
    monkeypatch.setattr(
        mod, "_lazy_generate_req_input", lambda: lambda **kwargs: SimpleNamespace(**kwargs)
    )

    def servicer(*script):
        s = mod.TokenSpeedSchedulerServicer.__new__(mod.TokenSpeedSchedulerServicer)
        s.async_llm = _Engine(*script)
        s._config = SimpleNamespace(server_args=SimpleNamespace(disaggregation_mode="null"))
        return s

    async def check(s):
        return await s.HealthCheck(tokenspeed_scheduler_pb2.HealthCheckRequest(), None)

    return SimpleNamespace(servicer=servicer, check=check, steps=steps)


def test_a_probe_the_engine_refuses_is_answered_at_the_first_step(bed):
    async def run():
        s = bed.servicer("refuse")
        response = await bed.check(s)
        assert response.healthy is False
        assert response.message == "Scheduler returned no output"
        assert bed.steps == [PROBE_STEP_SECS], "one step, not the deadline"
        (rid,) = s.async_llm.aborted
        assert rid.startswith("HEALTH_CHECK_")

    asyncio.run(run())


def test_a_probe_with_no_output_and_no_push_is_answered_at_the_deadline(bed):
    async def run():
        s = bed.servicer("silence")
        response = await bed.check(s)
        assert response.healthy is False
        assert response.message == f"Health check timeout after {DEADLINE}s"
        assert len(bed.steps) == DEADLINE
        for _ in range(5):
            await asyncio.sleep(0)
        assert asyncio.all_tasks() == {asyncio.current_task()}, "the probe was cancelled"
        assert len(s.async_llm.aborted) == 1

    asyncio.run(run())


def test_the_next_probe_the_engine_answers_is_healthy_again(bed):
    async def run():
        s = bed.servicer("refuse", "answer")
        assert (await bed.check(s)).healthy is False
        response = await bed.check(s)
        assert response.healthy is True
        assert response.message == "Health check passed"
        assert len(bed.steps) == 2, "one step each"
        assert len(s.async_llm.aborted) == 2

    asyncio.run(run())
