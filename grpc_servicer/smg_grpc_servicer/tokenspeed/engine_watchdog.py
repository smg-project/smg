"""The TokenSpeed Python servicer's watch over its engine processes.

TokenSpeed's own launcher (``_launch_subprocesses``, behind
:func:`smg_grpc_servicer.tokenspeed.scheduler_launcher.launch_engine`)
starts the scheduler rank processes (or its data-parallel controller) as
``multiprocessing`` children of this process and keeps them to itself. A
rank that dies during its start left the launch waiting on a ready report
that never came, and one that died later left a gRPC server with nothing
behind it: the servicer ran on as a worker that could not serve. This
thread notes the children this process has while the engine starts and
runs, and reports the first one that exits on its own.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import sys
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Short, so a rank that dies early in its start is seen alive at least once
# (a child that exits before any poll saw it is reaped unseen).
DEFAULT_POLL_SECS = 0.1


class EngineProcessWatch(threading.Thread):
    """A daemon thread that polls this process's ``multiprocessing`` children
    and calls ``on_exit(pid, exit_code)`` once, for the first one seen alive
    that has exited; :meth:`stop` ends it (an orderly shutdown terminates the
    engine itself). ``children`` is the source of the processes
    (``multiprocessing.active_children`` by default). The engine's processes
    all start inside the launch: :meth:`freeze`, called once the launch has
    returned, stops admitting new children, so a process pool or a manager the
    servicer starts later is not mistaken for the engine."""

    def __init__(
        self,
        on_exit: Callable[[int, int | None], None],
        *,
        poll_secs: float = DEFAULT_POLL_SECS,
        children: Callable[[], list[Any]] = multiprocessing.active_children,
    ):
        super().__init__(name="smg-engine-process-watch", daemon=True)
        self._on_exit = on_exit
        self._poll_secs = poll_secs
        self._children = children
        self._stopping = threading.Event()
        self._frozen = threading.Event()
        self._seen: dict[int, Any] = {}

    def run(self) -> None:
        while not self._stopping.is_set():
            exited = self.poll()
            if exited is not None:
                self._on_exit(*exited)
                return
            self._stopping.wait(self._poll_secs)

    def poll(self) -> tuple[int, int | None] | None:
        """Note the engine processes alive now (until :meth:`freeze`); the pid
        and exit code of the first one noted earlier that has exited, else
        ``None``."""
        if not self._frozen.is_set():
            try:
                for process in self._children():
                    if process.pid is not None:
                        self._seen.setdefault(process.pid, process)
            except RuntimeError:  # the children set changed under the iteration
                return None
        for pid, process in self._seen.items():
            code = process.exitcode
            if code is not None:
                return pid, code
        return None

    def terminate_seen(self) -> int:
        """SIGTERM every engine process noted that is still alive; how many."""
        count = 0
        for process in self._seen.values():
            if process.exitcode is None:
                try:
                    process.terminate()
                    count += 1
                except (OSError, ValueError, AttributeError):
                    pass
        return count

    def freeze(self) -> int:
        """Admit no further children: the engine's processes are the ones seen
        so far. Returns how many are watched."""
        self.poll()
        self._frozen.set()
        return len(self._seen)

    def stop(self) -> None:
        self._stopping.set()


def exit_process(code: int) -> None:
    """End the process now, the log handlers flushed: for an engine that died
    while the launch still waits on it (the main thread is inside the engine's
    launcher, so nothing can be drained or shut down in order)."""
    for handler in logging.getLogger().handlers:
        handler.flush()
    sys.stderr.flush()
    os._exit(code)
