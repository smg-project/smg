"""Engine-free coverage of the Python SGLang RPC's answer, through the request
manager's own request path: a probe the scheduler refuses at once is answered
unhealthy at the wait's first step, not at the deadline; one that gets no
output and no push is answered at the deadline; the next probe, answered by
the scheduler, is healthy again."""

import asyncio

import pytest
from smg_grpc_servicer.health_probe import PROBE_STEP_SECS
from test_sglang_health_cleanup import health_probe as _health_probe_fixture
from test_sglang_server_info import servicer_mod as _servicer_fixture

servicer_mod = _servicer_fixture
health_probe = _health_probe_fixture

DEADLINE = 20


@pytest.fixture
def h(health_probe, servicer_mod, monkeypatch):
    # The lifetime fixture's deadline is one step away; the answers under test
    # must come well before it. Its clock moves a second a step.
    monkeypatch.setattr(servicer_mod, "HEALTH_CHECK_TIMEOUT", DEADLINE)
    return health_probe


def _refusing(h):
    """The scheduler refuses the probe at its submission: the manager's send
    raises, as it does when its socket to the scheduler is gone. Returns the
    send it replaced."""
    send = h.manager._send_to_scheduler

    async def refuse(obj):
        await send(obj)
        raise RuntimeError("the scheduler refused the request")

    h.manager._send_to_scheduler = refuse
    return send


def test_a_probe_the_scheduler_refuses_is_answered_at_the_first_step(h):
    async def run():
        _refusing(h)
        rpc = await h.start()
        await asyncio.wait_for(asyncio.shield(h.probes[-1]), 2)
        assert h.probes[-1].result() is False
        h.resume.set()
        response = await asyncio.wait_for(rpc, 2)
        assert response.healthy is False
        assert response.message == "Health check probe failed"
        assert h.steps == [PROBE_STEP_SECS], "one step, not the deadline"
        h.assert_clean()

    asyncio.run(run())


def test_a_probe_with_no_output_and_no_push_is_answered_at_the_deadline(h):
    async def run():
        rpc = await h.start()
        h.resume.set()
        response = await asyncio.wait_for(rpc, 2)
        assert response.healthy is False
        assert response.message == f"Health check timeout after {DEADLINE}s"
        assert len(h.steps) == DEADLINE
        h.assert_clean()

    asyncio.run(run())


def test_the_next_probe_the_scheduler_answers_is_healthy_again(h):
    async def run():
        send = _refusing(h)
        rpc = await h.start()
        h.resume.set()
        assert (await asyncio.wait_for(rpc, 2)).healthy is False
        h.assert_clean()

        h.manager._send_to_scheduler = send
        rpc = await h.start()
        state = next(iter(h.manager.rid_to_state.values()))
        state.out_queue.put_nowait({"finished": True})
        await asyncio.wait_for(asyncio.shield(h.probes[-1]), 2)
        h.resume.set()
        response = await asyncio.wait_for(rpc, 2)
        assert response.healthy is True
        assert response.message == "Health check passed"
        assert len(h.steps) == 2, "one step each"
        h.assert_clean()

    asyncio.run(run())
