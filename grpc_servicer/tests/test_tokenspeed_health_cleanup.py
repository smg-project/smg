"""Engine-free regression coverage for the TokenSpeed RPC's health probe's
lifetime: the probe leaves the engine's stream at its first output and closes
it on the way out, so the engine's exit runs in the probe's own task, not in a
generator finalizer whose turn the garbage collector decides."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("msgspec")
pytest.importorskip("zmq")
pytest.importorskip("numpy")
pytest.importorskip("smg_grpc_proto")
from smg_grpc_proto.generated import tokenspeed_scheduler_pb2  # noqa: E402


class _AsyncLlm:
    """An engine whose stream notes the task that iterates it and the task
    that runs its exit."""

    def __init__(self):
        self.gracefully_exit = False
        self.last_receive_tstamp = 0.0
        self.iterated_in = None
        self.exited_in = None
        self.aborted = []

    async def generate_request(self, probe):
        self.iterated_in = asyncio.current_task()
        try:
            yield {"output_ids": [1], "meta_info": {"finish_reason": None}}
            raise AssertionError("the probe read past its first output")
        finally:
            self.exited_in = asyncio.current_task()

    def abort_request(self, rid):
        self.aborted.append(rid)


@pytest.fixture
def servicer(tokenspeed_servicer_mod, monkeypatch):
    mod = tokenspeed_servicer_mod
    servicer = mod.TokenSpeedSchedulerServicer.__new__(mod.TokenSpeedSchedulerServicer)
    servicer.async_llm = _AsyncLlm()
    servicer._config = SimpleNamespace(server_args=SimpleNamespace(disaggregation_mode="null"))
    monkeypatch.setattr(
        mod, "_lazy_generate_req_input", lambda: lambda **kwargs: SimpleNamespace(**kwargs)
    )

    async def poll(_delay):
        # The RPC's polling interval, cut to the turns the probe needs.
        for _ in range(5):
            await asyncio.sleep(0)

    # Control only the RPC's polling; the probe's task and stream are real.
    monkeypatch.setattr(
        mod, "asyncio", SimpleNamespace(create_task=asyncio.create_task, sleep=poll)
    )
    return servicer


def test_the_probe_closes_the_engines_stream_in_its_own_task(servicer):
    async def run():
        response = await servicer.HealthCheck(tokenspeed_scheduler_pb2.HealthCheckRequest(), None)
        engine = servicer.async_llm
        assert response.healthy
        assert engine.exited_in is not None, "the stream the probe left is closed"
        assert engine.exited_in is engine.iterated_in, "by the probe, not by a finalizer"
        assert len(engine.aborted) == 1 and engine.aborted[0].startswith("HEALTH_CHECK_")

    asyncio.run(run())
