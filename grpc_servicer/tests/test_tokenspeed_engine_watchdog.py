"""Engine-free tests for the TokenSpeed Python servicer's watch over its engine
processes (smg-lab #292)."""

from __future__ import annotations

import multiprocessing
import threading
import time
from types import SimpleNamespace

from smg_grpc_servicer.tokenspeed import engine_watchdog


def _live_then_exit(secs: float, code: int) -> None:
    time.sleep(secs)
    raise SystemExit(code)


def test_an_engine_process_that_exits_is_reported_once_with_its_code():
    reported: list[tuple[int, int | None]] = []
    done = threading.Event()

    def on_exit(pid: int, code: int | None) -> None:
        reported.append((pid, code))
        done.set()

    watch = engine_watchdog.EngineProcessWatch(on_exit, poll_secs=0.02)
    watch.start()
    child = multiprocessing.Process(target=_live_then_exit, args=(0.3, 3))
    child.start()
    try:
        assert done.wait(10.0), "the watch did not report the exited engine process"
        watch.join(2.0)
        assert not watch.is_alive()
        assert reported == [(child.pid, 3)]
    finally:
        child.join(5.0)


def test_a_living_engine_is_not_reported_and_stop_ends_the_watch():
    reported: list = []
    watch = engine_watchdog.EngineProcessWatch(reported.append, poll_secs=0.02)
    child = multiprocessing.Process(target=time.sleep, args=(30,))
    child.start()
    try:
        watch.start()
        time.sleep(0.2)
        assert reported == [] and watch.poll() is None
        watch.stop()
        watch.join(2.0)
        assert not watch.is_alive()
        assert watch.terminate_seen() == 1
    finally:
        child.terminate()
        child.join(5.0)


def test_freeze_admits_no_later_children():
    """Once the launch has returned, a process pool or a manager the servicer
    starts later is not the engine: it is neither watched nor reported."""
    rank = SimpleNamespace(pid=4242, exitcode=None, terminate=lambda: None)
    children = [rank]
    watch = engine_watchdog.EngineProcessWatch(lambda *_: None, children=lambda: list(children))
    assert watch.freeze() == 1
    later = SimpleNamespace(pid=4343, exitcode=0, terminate=lambda: None)
    children.append(later)
    assert watch.poll() is None
    rank.exitcode = 1
    assert watch.poll() == (4242, 1)


def test_the_watch_reads_the_children_it_is_given():
    process = SimpleNamespace(pid=4242, exitcode=None, terminated=0)
    process.terminate = lambda: setattr(process, "terminated", process.terminated + 1)
    watch = engine_watchdog.EngineProcessWatch(lambda *_: None, children=lambda: [process])
    assert watch.poll() is None
    process.exitcode = -9
    assert watch.poll() == (4242, -9)
    # A process already gone is not terminated again; a live one is.
    assert watch.terminate_seen() == 0
    process.exitcode = None
    assert watch.terminate_seen() == 1 and process.terminated == 1
