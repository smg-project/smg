"""The stall rule the readiness probes share: silent beyond the threshold with
requests pending is stuck; said once per stall, the recovery once."""

from __future__ import annotations

import logging
import time

from smg_grpc_servicer.health_probe import STUCK_SCHEDULER_THRESHOLD_SECS, StallReport


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
