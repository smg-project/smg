"""Lifecycle helpers for watching SGLang scheduler processes."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from contextlib import suppress
from typing import Protocol

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_SECONDS = 1.0
WATCHDOG_TASK_NAME = "sglang-scheduler-exit-watchdog"


class SchedulerProcess(Protocol):
    """Subset of ``multiprocessing.Process`` used by the watchdog."""

    @property
    def exitcode(self) -> int | None: ...


async def watch_scheduler_processes(
    scheduler_processes: Sequence[SchedulerProcess],
    stop_event: asyncio.Event,
    *,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
) -> bool:
    """Request server shutdown once a scheduler process has exited on its own:
    one exited rank (of a tensor-parallel group, or the only one) leaves an
    engine that cannot serve. Returns whether that happened (``False`` when
    the stop event was set first, by a signal)."""
    if not scheduler_processes:
        return False

    while not stop_event.is_set():
        exit_codes = [process.exitcode for process in scheduler_processes]
        if any(exit_code is not None for exit_code in exit_codes):
            if any(exit_code not in (None, 0) for exit_code in exit_codes):
                logger.error(
                    "A scheduler process exited; non-zero exit codes detected: %s. "
                    "Shutting down gRPC server",
                    exit_codes,
                )
            else:
                logger.error(
                    "A scheduler process exited (exit codes %s); shutting down gRPC server",
                    exit_codes,
                )
            stop_event.set()
            return True

        await asyncio.sleep(poll_interval_seconds)
    return False


async def wait_for_scheduler_shutdown(
    scheduler_processes: Sequence[SchedulerProcess],
    stop_event: asyncio.Event,
    *,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
) -> bool:
    """Wait for a signal or a scheduler exit while owning the watchdog task;
    returns whether a scheduler process exited on its own (the servicer then
    exits non-zero)."""
    if not scheduler_processes:
        await stop_event.wait()
        return False

    watchdog_task = asyncio.create_task(
        watch_scheduler_processes(
            scheduler_processes,
            stop_event,
            poll_interval_seconds=poll_interval_seconds,
        ),
        name=WATCHDOG_TASK_NAME,
    )
    try:
        await stop_event.wait()
    finally:
        watchdog_task.cancel()
        with suppress(asyncio.CancelledError):
            await watchdog_task
    return not watchdog_task.cancelled() and bool(watchdog_task.result())
