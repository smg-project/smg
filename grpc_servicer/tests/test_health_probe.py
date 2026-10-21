"""The two rules the servicers' health answers share. The readiness probes'
stall rule: silent beyond the threshold with requests pending is stuck; said
once per stall, the recovery once. The deep health RPC's wait for its probe: a
push after the start or the probe's own output is healthy, the probe's failure
is unhealthy at once, neither by the deadline is unhealthy then."""

from __future__ import annotations

import asyncio
import logging
import time

from smg_grpc_servicer.health_probe import (
    PROBE_STEP_SECS,
    STUCK_SCHEDULER_THRESHOLD_SECS,
    StallReport,
    await_probe,
)


def _lines(caplog, level):
    return [r.getMessage() for r in caplog.records if r.levelno == level]


def test_silence_with_pending_work_is_a_stall_and_idle_silence_is_not(caplog):
    report = StallReport(logging.getLogger("probe-under-test"))
    long_ago = time.time() - 2 * STUCK_SCHEDULER_THRESHOLD_SECS
    with caplog.at_level(logging.DEBUG):
        assert report.stuck(long_ago, pending=0) is False
        assert report.stuck(time.time(), pending=3) is False
        assert report.stuck(long_ago, pending=3) is True
    (warning,) = _lines(caplog, logging.WARNING)
    assert warning.startswith("Scheduler not responsive: ")
    assert warning.endswith("s since last receive, 3 pending requests")


def test_a_stall_is_said_once_and_its_recovery_once(caplog):
    report = StallReport(logging.getLogger("probe-under-test"))
    long_ago = time.time() - 600
    with caplog.at_level(logging.DEBUG):
        for _ in range(4):
            assert report.stuck(long_ago, pending=1) is True
        assert report.stuck(time.time(), pending=1) is False
        assert report.stuck(time.time(), pending=1) is False
        assert report.stuck(long_ago, pending=1) is True
    assert len(_lines(caplog, logging.WARNING)) == 2
    assert _lines(caplog, logging.INFO) == ["Scheduler is responsive again"]


def test_the_threshold_is_the_report_s_own():
    report = StallReport(logging.getLogger("probe-under-test"), threshold_secs=1.0)
    assert report.stuck(time.time() - 2.0, pending=1) is True
    assert StallReport(logging.getLogger("probe-under-test")).stuck(time.time() - 2.0, 1) is False


def _awaited(scenario, *, timeout=5.0, step=0.01):
    """``await_probe`` over ``scenario(started, probe)``, which sets the
    probe's state and returns the time of the scheduler's last push; with the
    wall time the wait took and the probe as the wait left it."""

    async def run():
        started = time.time()
        probe = asyncio.get_running_loop().create_future()
        pushed_at = scenario(started, probe)
        t0 = time.monotonic()
        verdict = await await_probe(
            probe, started=started, last_receive=lambda: pushed_at, timeout=timeout, step=step
        )
        return verdict, time.monotonic() - t0, probe

    return asyncio.run(run())


def test_a_push_after_the_start_is_healthy_within_a_step():
    verdict, took, probe = _awaited(lambda started, probe: started + 0.001)
    assert verdict is True
    assert took < 1.0
    assert not probe.done(), "the probe is the caller's to cancel"


def test_a_push_during_the_wait_is_read_at_the_next_step():
    async def run():
        started = time.time()
        probe = asyncio.get_running_loop().create_future()
        pushes = {"at": started - 1.0}

        async def scheduler_speaks():
            await asyncio.sleep(0.03)
            pushes["at"] = time.time()

        asyncio.get_running_loop().create_task(scheduler_speaks())
        return await await_probe(
            probe, started=started, last_receive=lambda: pushes["at"], timeout=5.0, step=0.01
        )

    assert asyncio.run(run()) is True


def test_the_probe_s_own_verdict_answers_when_nothing_was_pushed():
    def refused(started, probe):
        probe.set_result(False)
        return started - 1.0  # the last push predates the probe

    verdict, took, _ = _awaited(refused)
    assert verdict is False
    assert took < 1.0, "the failure is answered at the first step, not the deadline"

    def answered(started, probe):
        probe.set_result(True)
        return started - 1.0

    assert _awaited(answered)[0] is True


def test_neither_by_the_deadline_is_none_and_the_probe_is_left_to_the_caller():
    verdict, took, probe = _awaited(lambda started, probe: started - 1.0, timeout=0.05)
    assert verdict is None
    assert took >= 0.05
    assert not probe.done()


def test_the_step_is_half_a_second():
    # A push, or a probe that fails at once, is answered at most a step late.
    assert PROBE_STEP_SECS == 0.5
