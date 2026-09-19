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
) -> None:
    """Request server shutdown after every scheduler process has exited."""
    if not scheduler_processes:
        return

    while not stop_event.is_set():
        exit_codes = [process.exitcode for process in scheduler_processes]
        if all(exit_code is not None for exit_code in exit_codes):
            if any(exit_code != 0 for exit_code in exit_codes):
                logger.error(
                    "All scheduler processes exited; non-zero exit codes detected: %s. "
                    "Shutting down gRPC server",
                    exit_codes,
                )
            else:
                logger.warning(
                    "All scheduler processes exited with codes %s; shutting down gRPC server",
                    exit_codes,
                )
            stop_event.set()
            return

        await asyncio.sleep(poll_interval_seconds)


async def wait_for_scheduler_shutdown(
    scheduler_processes: Sequence[SchedulerProcess],
    stop_event: asyncio.Event,
    *,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
) -> None:
    """Wait for a signal or scheduler exit while owning the watchdog task."""
    if not scheduler_processes:
        await stop_event.wait()
        return

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
