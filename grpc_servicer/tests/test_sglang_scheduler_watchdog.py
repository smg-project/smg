"""Engine-free tests for the SGLang scheduler-process watchdog."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

_SERVICER_ROOT = Path(__file__).resolve().parent.parent / "smg_grpc_servicer" / "sglang"
_WATCHDOG_PATH = _SERVICER_ROOT / "scheduler_watchdog.py"


def _load_watchdog() -> ModuleType:
    assert _WATCHDOG_PATH.exists(), "the scheduler watchdog module is missing"
    spec = importlib.util.spec_from_file_location(
        "_smg_sglang_scheduler_watchdog_under_test",
        _WATCHDOG_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    watchdog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watchdog)
    return watchdog


@dataclass
class _Process:
    exitcode: int | None
    terminate_calls: int = 0

    def terminate(self) -> None:
        self.terminate_calls += 1


def _watchdog_tasks(watchdog: ModuleType) -> list[asyncio.Task]:
    current_task = asyncio.current_task()
    return [
        task
        for task in asyncio.all_tasks()
        if task is not current_task and task.get_name() == watchdog.WATCHDOG_TASK_NAME
    ]


def test_all_exited_schedulers_request_shutdown(caplog):
    watchdog = _load_watchdog()
    processes = [_Process(exitcode=0), _Process(exitcode=-9)]
    caplog.set_level(logging.ERROR)

    async def exercise():
        stop_event = asyncio.Event()
        await watchdog.wait_for_scheduler_shutdown(
            processes,
            stop_event,
            poll_interval_seconds=0,
        )

        assert stop_event.is_set()
        assert not _watchdog_tasks(watchdog)

    asyncio.run(exercise())

    assert "[0, -9]" in caplog.text
    assert "non-zero" in caplog.text
    assert [process.terminate_calls for process in processes] == [0, 0]


def test_partial_scheduler_exit_does_not_request_shutdown():
    watchdog = _load_watchdog()

    async def exercise():
        stop_event = asyncio.Event()
        task = asyncio.create_task(
            watchdog.watch_scheduler_processes(
                [_Process(exitcode=1), _Process(exitcode=None)],
                stop_event,
                poll_interval_seconds=3600,
            )
        )
        await asyncio.sleep(0)

        assert not stop_event.is_set()
        assert not task.done()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())


def test_normal_shutdown_cancels_and_awaits_watchdog(monkeypatch):
    watchdog = _load_watchdog()
    process = _Process(exitcode=None)

    async def exercise():
        stop_event = asyncio.Event()
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def blocking_watchdog(*args, **kwargs):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        monkeypatch.setattr(watchdog, "watch_scheduler_processes", blocking_watchdog)
        waiter = asyncio.create_task(
            watchdog.wait_for_scheduler_shutdown([process], stop_event),
        )
        await started.wait()

        stop_event.set()
        await waiter

        assert cancelled.is_set()
        assert process.terminate_calls == 0
        assert not _watchdog_tasks(watchdog)

    asyncio.run(exercise())


def test_cancelling_waiter_cancels_and_awaits_watchdog(monkeypatch):
    watchdog = _load_watchdog()

    async def exercise():
        stop_event = asyncio.Event()
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def blocking_watchdog(*args, **kwargs):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        monkeypatch.setattr(watchdog, "watch_scheduler_processes", blocking_watchdog)
        waiter = asyncio.create_task(
            watchdog.wait_for_scheduler_shutdown([_Process(exitcode=None)], stop_event),
        )
        await started.wait()

        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        assert cancelled.is_set()
        assert not _watchdog_tasks(watchdog)

    asyncio.run(exercise())


def test_empty_scheduler_list_waits_for_normal_shutdown():
    watchdog = _load_watchdog()

    async def exercise():
        stop_event = asyncio.Event()
        waiter = asyncio.create_task(
            watchdog.wait_for_scheduler_shutdown([], stop_event),
        )
        await asyncio.sleep(0)

        assert not waiter.done()

        stop_event.set()
        await waiter
        assert not _watchdog_tasks(watchdog)

    asyncio.run(exercise())
