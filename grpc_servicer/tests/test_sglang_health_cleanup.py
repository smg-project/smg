"""Engine-free regression coverage for the Python SGLang RPC's probe lifetime."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import smg_grpc_servicer.health_probe as probe_wait
from test_sglang_server_info import servicer_mod as _servicer_fixture

servicer_mod = _servicer_fixture


@pytest.fixture
def health_probe(monkeypatch, servicer_mod):
    manager_mod = sys.modules["smg_grpc_servicer.sglang.request_manager"]
    manager = manager_mod.GrpcRequestManager.__new__(manager_mod.GrpcRequestManager)
    manager.gracefully_exit = False
    manager.is_pause = False
    manager.last_receive_tstamp = 0.0
    manager.rid_to_state = {}
    manager.record_request_for_crash_dump = Mock()
    probes = []
    steps = []
    clock = SimpleNamespace(now=100.0)
    started, polling, resume = asyncio.Event(), asyncio.Event(), asyncio.Event()

    def register(obj, context):
        stats = Mock(finished_time=0)
        manager.rid_to_state[obj.rid] = SimpleNamespace(
            out_queue=asyncio.Queue(), output_ids=[0], time_stats=stats
        )

    async def send(obj):
        probes.append(asyncio.current_task())
        started.set()

    async def poll(delay):
        steps.append(delay)
        polling.set()
        await resume.wait()
        clock.now += 1

    manager._req_stats_init = register
    manager._send_to_scheduler = send
    service = SimpleNamespace(
        request_manager=manager,
        tokenizer=None,
        scheduler_info={"is_generation": True},
        server_args=SimpleNamespace(disaggregation_mode="null"),
    )
    monkeypatch.setattr(
        servicer_mod,
        "SGLSamplingParams",
        lambda **kw: SimpleNamespace(n=1, normalize=lambda **kw: None, **kw),
    )
    monkeypatch.setattr(servicer_mod, "TokenizedGenerateReqInput", SimpleNamespace)
    monkeypatch.setattr(servicer_mod, "TokenizedEmbeddingReqInput", SimpleNamespace)
    monkeypatch.setattr(
        servicer_mod, "DisaggregationMode", SimpleNamespace(NULL=SimpleNamespace(value="null"))
    )
    monkeypatch.setattr(servicer_mod, "HEALTH_CHECK_TIMEOUT", 1)
    # One clock for the RPC's start and the wait's deadline, and only the
    # wait's polling under control: the request manager uses real asyncio queues.
    monkeypatch.setattr(servicer_mod, "time", SimpleNamespace(time=lambda: clock.now))
    monkeypatch.setattr(probe_wait, "time", SimpleNamespace(time=lambda: clock.now))
    monkeypatch.setattr(probe_wait, "asyncio", SimpleNamespace(sleep=poll))
    monkeypatch.setattr(
        servicer_mod,
        "asyncio",
        SimpleNamespace(create_task=asyncio.create_task, gather=asyncio.gather),
    )

    async def start():
        started.clear()
        polling.clear()
        resume.clear()
        clock.now += 10
        rpc = asyncio.create_task(
            servicer_mod.SGLangSchedulerServicer.HealthCheck(service, None, None)
        )
        await asyncio.wait_for(started.wait(), 2)
        await asyncio.wait_for(polling.wait(), 2)
        return rpc

    def assert_clean():
        assert all(probe.done() for probe in probes)
        assert manager.rid_to_state == {}
        assert manager.get_server_info()["active_requests"] == 0

    return SimpleNamespace(
        manager=manager,
        service=service,
        probes=probes,
        steps=steps,
        clock=clock,
        resume=resume,
        start=start,
        assert_clean=assert_clean,
    )


@pytest.mark.parametrize("is_generation", [True, False])
@pytest.mark.parametrize("outcome", ["signal", "terminal", "timeout", "cancel"])
def test_health_probe_cleanup_on_every_rpc_exit(health_probe, outcome, is_generation):
    async def run():
        h = health_probe
        h.service.scheduler_info["is_generation"] = is_generation
        rpc = await h.start()
        state = next(iter(h.manager.rid_to_state.values()))
        if outcome == "terminal":
            state.out_queue.put_nowait({"finished": True})
            await asyncio.wait_for(asyncio.shield(h.probes[-1]), 2)
        if outcome in ("signal", "terminal"):
            # A busy scheduler's HealthCheckOutput without a rid only updates
            # liveness. It does not deliver anything to this request's queue.
            h.manager.last_receive_tstamp = h.clock.now + 0.5
        if outcome == "cancel":
            rpc.cancel()
            with pytest.raises(asyncio.CancelledError):
                await rpc
        else:
            h.resume.set()
            response = await asyncio.wait_for(rpc, 2)
            assert response.healthy is (outcome != "timeout")
        h.assert_clean()
        state.time_stats.set_finished_time.assert_called_once()

    asyncio.run(run())


def test_cancelled_busy_probes_do_not_survive_later_successes(health_probe):
    async def run():
        h = health_probe
        for _ in range(3):
            rpc = await h.start()
            h.manager.last_receive_tstamp = h.clock.now + 0.5
            rpc.cancel()
            with pytest.raises(asyncio.CancelledError):
                await rpc
            h.assert_clean()
        for _ in range(2):
            rpc = await h.start()
            h.manager.last_receive_tstamp = h.clock.now + 0.5
            h.resume.set()
            assert (await asyncio.wait_for(rpc, 2)).healthy
            h.assert_clean()
        assert len(h.probes) == 5

    asyncio.run(run())


def test_terminal_probe_finishes_before_rpc_cancellation(health_probe):
    """A probe that receives its own terminal output finishes itself, and its
    request state is gone the moment the probe is done: the probe closes the
    stream it leaves, so nothing waits for a generator finalizer, whose turn
    the garbage collector decides."""

    async def run():
        h = health_probe
        rpc = await h.start()
        state = next(iter(h.manager.rid_to_state.values()))
        state.out_queue.put_nowait({"finished": True})
        await asyncio.wait_for(asyncio.shield(h.probes[-1]), 2)
        h.assert_clean()
        rpc.cancel()
        with pytest.raises(asyncio.CancelledError):
            await rpc
        h.assert_clean()

    asyncio.run(run())


def test_rpc_cancellation_during_probe_cleanup_propagates(health_probe, monkeypatch):
    async def run():
        h = health_probe
        cleaning = asyncio.Event()
        generate = h.manager.generate_request

        async def delayed_cleanup(**kwargs):
            try:
                async for response in generate(**kwargs):
                    yield response
            finally:
                cleaning.set()
                await asyncio.Event().wait()

        monkeypatch.setattr(h.manager, "generate_request", delayed_cleanup)
        rpc = await h.start()
        h.manager.last_receive_tstamp = h.clock.now + 0.5
        h.resume.set()
        await asyncio.wait_for(cleaning.wait(), 2)
        assert not rpc.done()
        rpc.cancel()
        with pytest.raises(asyncio.CancelledError):
            await rpc
        h.assert_clean()

    asyncio.run(run())
