"""The Python-owned lifecycle around a Rust servicer, shared by the engine
launchers: a free port and socket directory for the handshake, the engine as
a ``Popen``-shaped handle (one spawned headless engine, the engine cores a
manager started from this process, or the scheduler processes this process
started itself), its termination, and the supervision loop that drains on a
signal and exits with the engine or the server. Nothing here knows which
engine it supervises.
"""

from __future__ import annotations

import asyncio
import glob
import logging
import multiprocessing
import os
import signal
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DRAIN_SECS = 5.0
ENGINE_TERMINATE_SECS = 30.0
# Bound on silence during the engine's ZMQ handshake. An engine's start
# includes model load, kernel JIT and graph capture, and nothing crosses the
# wire between its first handshake message and its last; a server that takes
# ``note_engine_alive`` reports hears from ``supervise`` that the engine
# process is alive on every poll, so a healthy start that outlasts this bound
# still completes. A dead engine never waits this long: ``supervise`` polls
# the engine process.
DEFAULT_STARTUP_TIMEOUT_SECS = 1800.0
# Bound on the whole start however alive the engine is: an engine that stays
# alive without ever completing its handshake. Large checkpoints have taken
# over half an hour to load on a cold start; four hours leaves room for the
# storage path. 0 lifts the ceiling.
DEFAULT_STARTUP_CEILING_SECS = 4 * 3600.0
_POLL_SECS = 0.5
# Grace a scheduler process this process started gets between SIGTERM and
# SIGKILL at shutdown, and how long the kill is waited for.
SCHEDULER_GRACE_SECS = 10.0
SCHEDULER_KILL_SECS = 2.0
# How often the wait for the scheduler processes' ready reports looks for one
# that died during its start instead.
READY_POLL_SECS = 5.0
TRACK_INTERVAL_SECS = 2.0  # how often the supervisor's poll refreshes the engine's descendants

# What makes a directory a tokenizer directory to the Rust tokenizer loader.
TOKENIZER_FILE_GLOBS = (
    "tokenizer.json",
    "tokenizer.model",
    "tiktoken.model",
    "*.tiktoken",
    "vocab.json",
)
# The files worth fetching for one: the tokenizer and its configs, never weights.
TOKENIZER_DOWNLOAD_PATTERNS = ["*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja"]


def holds_tokenizer(directory: str) -> bool:
    """Whether ``directory`` holds a tokenizer file the Rust loader reads."""
    return any(glob.glob(os.path.join(directory, pattern)) for pattern in TOKENIZER_FILE_GLOBS)


def resolve_tokenizer_dir(tokenizer: str, revision: str | None = None) -> str | None:
    """A local directory holding ``tokenizer`` (a path, or a Hub id resolved
    through the local cache first, then the Hub); ``None`` when none can be
    found, in which case the Rust servicer refuses requests carrying string
    stops.

    The cache is only trusted once it holds a tokenizer file: the engine
    downloads the same repo in parallel, and a snapshot it has started to
    populate (its ``config.json`` is there, its ``tokenizer.json`` not yet)
    satisfies a local-only lookup while loading nothing."""
    if os.path.isdir(tokenizer):
        return tokenizer
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning("huggingface_hub is not installed; cannot resolve tokenizer %r", tokenizer)
        return None
    last_error: Exception | None = None
    for local_files_only in (True, False):
        try:
            directory = snapshot_download(
                tokenizer,
                revision=revision,
                allow_patterns=TOKENIZER_DOWNLOAD_PATTERNS,
                local_files_only=local_files_only,
            )
        except Exception as error:  # cache miss, offline, or an unknown repo
            last_error = error
            continue
        if holds_tokenizer(directory):
            return directory
        last_error = FileNotFoundError(f"{directory} holds no tokenizer file")
    logger.warning("Could not resolve tokenizer %r to a local directory: %s", tokenizer, last_error)
    return None


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def default_socket_dir() -> str:
    return os.environ.get("SMG_ZMQ_SOCKET_DIR") or f"/tmp/smg-zmq-{os.getuid()}"


class EngineProcess:
    """A ``Popen``-shaped view of the spawned headless-engine process, so the
    lifecycle loop and :func:`terminate_engine` need no second code path.

    The engine stays in the launcher's process group: whoever supervises the
    launcher (``ts serve``, ``vllm serve``, the e2e harness) cleans up by
    group and must keep reaching the engine's workers. What a group cannot
    cover is a launcher that dies before its workers do (killed partway
    through its own teardown, or crashed at startup): the workers are
    re-parented and would keep the GPU. So the engine's descendants are
    tracked while it runs, and a kill, or the cleanup after it exited on its
    own, reaches every one that is still the process it was.
    """

    def __init__(self, process: Any):
        self._process = process
        self._descendants: dict[int, str] = {}  # pid -> start time, against PID reuse
        self._tracked_at = 0.0

    @property
    def pid(self) -> int | None:
        return self._process.pid

    def poll(self) -> int | None:
        code = self._process.exitcode
        if code is None and time.monotonic() - self._tracked_at >= TRACK_INTERVAL_SECS:
            self.track()
        return code

    def track(self) -> None:
        """Record the engine's current descendants (one pass over ``/proc``)."""
        pid = self.pid
        if pid is None:
            return
        self._tracked_at = time.monotonic()
        for child in _descendants(pid):
            start = _start_time(child)
            if start is not None:
                self._descendants[child] = start

    def terminate(self) -> None:
        self.track()
        self._process.terminate()

    def kill(self) -> None:
        self.track()
        self._process.kill()
        self.kill_descendants()

    def kill_descendants(self) -> int:
        """SIGKILL every tracked descendant that is still the same process;
        returns how many were signalled."""
        killed = 0
        for child, start in list(self._descendants.items()):
            del self._descendants[child]
            if _start_time(child) != start:
                continue
            try:
                os.kill(child, signal.SIGKILL)
                killed += 1
            except ProcessLookupError:
                pass
        return killed

    def wait(self, timeout: float | None = None) -> int:
        self._process.join(timeout)
        code = self._process.exitcode
        if code is None:
            raise subprocess.TimeoutExpired("headless engine", timeout or 0)
        return code


class EngineProcessGroup:
    """A ``Popen``-shaped view of the engine processes a manager started from
    this process (vLLM's ``CoreEngineProcManager``: its ``processes`` and its
    ``shutdown(timeout)``), so :func:`supervise` and :func:`terminate_engine`
    need no second code path.

    Every process is tracked as an :class:`EngineProcess`, so the workers an
    engine core spawns (its TP ranks) are reaped the same way when a core is
    killed or exits on its own. ``terminate`` goes through the manager, which
    signals the cores and force-kills what is still alive after the grace
    period; it runs once. ``kill`` reaches the processes directly.
    """

    def __init__(self, manager: Any, *, shutdown_timeout: float | None = None):
        self._manager = manager
        self._shutdown_timeout = shutdown_timeout
        self._engines = [EngineProcess(process) for process in manager.processes]
        self._exit_code: int | None = None
        self._shut_down = False

    @property
    def pid(self) -> int | None:
        """The first core's pid; :attr:`pids` lists them all."""
        return self._engines[0].pid if self._engines else None

    @property
    def pids(self) -> list[int | None]:
        return [engine.pid for engine in self._engines]

    def poll(self) -> int | None:
        """``None`` while every core is alive, else the exit code of the first
        core seen exiting. That exit ends the engine: the manager shuts the
        other cores down (what vLLM's own liveness monitor does)."""
        codes = [engine.poll() for engine in self._engines]
        if self._exit_code is None:
            # Only a code that was found is stored: a fault reported from the
            # ready thread or a signal handler between the check and the store
            # must not be written over with None.
            code = next((code for code in codes if code is not None), None)
            if code is not None:
                self._exit_code = code
        if self._exit_code is not None and None in codes:
            self._shutdown()
        return self._exit_code

    def fault(self, code: int = 1) -> None:
        """A failure reported out of band (a signal from a process that then
        exits on its own terms): the group ends with ``code``, and the next
        poll shuts the processes still alive down. The first report counts."""
        if self._exit_code is None:
            self._exit_code = code

    def track(self) -> None:
        for engine in self._engines:
            engine.track()

    def terminate(self) -> None:
        self.track()
        self._shutdown(self._shutdown_timeout)

    def _shutdown(self, timeout: float | None = None) -> None:
        if self._shut_down:
            return
        self._shut_down = True
        self._manager.shutdown(timeout=timeout)

    def kill(self) -> None:
        for engine in self._engines:
            engine.kill()

    def kill_descendants(self) -> int:
        return sum(engine.kill_descendants() for engine in self._engines)

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        for engine in self._engines:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            engine.wait(remaining)
        code = self.poll()
        assert code is not None
        return code


class SchedulerProcesses:
    """The manager an :class:`EngineProcessGroup` wraps when this process
    started the scheduler processes itself (the engine's own per-process
    entry point under ``multiprocessing``, spawned): the ``processes``, and
    the ``shutdown`` that terminates every live one and kills what outlives
    the grace period.

    ``ready`` blocks until every process has reported its start on the pipe
    it was given, or raises; :meth:`watch_ready` runs it on a thread, so the
    reports are drained (a process never blocks on its report) and a failed
    start is logged at once, while the supervisor keeps polling the
    processes themselves.
    """

    def __init__(
        self,
        processes: list,
        *,
        ready: Callable[[], Any] | None = None,
        grace_secs: float = SCHEDULER_GRACE_SECS,
    ):
        self.processes = list(processes)
        self._ready = ready
        self._grace_secs = grace_secs
        self.ready_thread: threading.Thread | None = None
        self.ready_error: Exception | None = None

    @property
    def pids(self) -> list[int | None]:
        return [process.pid for process in self.processes]

    def watch_ready(
        self, what: str, *, on_failure: Callable[[int], None] | None = None
    ) -> threading.Thread | None:
        """Wait for the ready reports on a daemon thread; logs the outcome. A
        failed start is handed to ``on_failure`` (the group's ``fault``, so
        its next poll shuts the processes down and the supervisor exits): the
        processes may well stay alive after a bad report, and the supervisor
        would otherwise keep reporting the engine alive until its startup
        ceiling."""
        if self._ready is None:
            return None

        def wait() -> None:
            try:
                self._ready()
            except Exception as error:
                self.ready_error = error
                logger.error("%s: the scheduler processes did not start: %s", what, error)
                if on_failure is not None:
                    on_failure(1)
                return
            logger.info("%s: %d scheduler process(es) ready", what, len(self.processes))

        self.ready_thread = threading.Thread(target=wait, name="smg-scheduler-ready", daemon=True)
        self.ready_thread.start()
        return self.ready_thread

    def shutdown(self, timeout: float | None = None) -> None:
        """SIGTERM every live process, wait up to ``timeout`` (the grace
        period by default) for all of them, SIGKILL what is still alive."""
        grace = self._grace_secs if timeout is None else timeout
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        deadline = time.monotonic() + grace
        for process in self.processes:
            process.join(max(0.0, deadline - time.monotonic()))
        for process in self.processes:
            if process.is_alive():
                logger.warning(
                    "Scheduler process %s did not exit within %.0fs; killing it", process.pid, grace
                )
                process.kill()
                process.join(SCHEDULER_KILL_SECS)


def wait_for_ready_reports(
    readers: list, processes: list, *, poll_secs: float = READY_POLL_SECS
) -> list[dict]:
    """Collect one ready report per pipe (a scheduler process reports
    ``{"status": "ready", ...}`` on the pipe it was started with, once its
    model is loaded), polling so a process that died during its start is
    noticed instead of waited on for ever. Returns the reports in pipe order;
    raises when a report says otherwise or a process is gone."""
    reports: list[dict] = []
    for index, reader in enumerate(readers):
        while not reader.poll(poll_secs):
            dead = next((process for process in processes if not process.is_alive()), None)
            if dead is not None:
                raise RuntimeError(
                    f"scheduler process {dead.pid} exited with {dead.exitcode} before it was ready"
                )
        try:
            report = reader.recv()
        except EOFError as error:
            raise RuntimeError(
                f"scheduler process {processes[index].pid} closed its pipe before it was ready"
            ) from error
        if not isinstance(report, dict) or report.get("status") != "ready":
            raise RuntimeError(f"scheduler process {index} failed to start: {report!r}")
        reports.append(report)
    return reports


class FaultLatch:
    """The target of a fault handler installed before the process group
    exists: the handlers go in ahead of the first process start (a process
    that fails during its start signals its parent right away, and a signal
    at its default disposition would end the servicer), the group is built
    once the processes are started, and a fault reported in between is
    passed on when the group is attached."""

    def __init__(self):
        self._code: int | None = None
        self._group: EngineProcessGroup | None = None

    def fault(self, code: int = 1) -> None:
        if self._group is not None:
            self._group.fault(code)
        elif self._code is None:
            self._code = code

    def attach(self, group: EngineProcessGroup) -> EngineProcessGroup:
        """Hand faults on to ``group`` from now on, and the one already
        reported, if any, at once. Returns the group."""
        self._group = group
        if self._code is not None:
            group.fault(self._code)
        return group


def ensure_spawn_start_method(what: str) -> None:
    """The scheduler processes must be spawned, never forked: this process
    runs the Rust server's threads, and a fork can leave the child deadlocked
    on a lock one of them held. The engine's own launch setup sets the method;
    this keeps the guarantee in our hands should a version stop doing so."""
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        logger.warning("%s: the multiprocessing start method was not spawn; setting it", what)
        multiprocessing.set_start_method("spawn", force=True)


def on_fault_signal(target: Any, *signals: int, what: str) -> bool:
    """Make the signals a scheduler process sends its parent when it hits an
    exception end ``target`` (an :class:`EngineProcessGroup`, or the
    :class:`FaultLatch` that stands in for one before the processes start:
    the group's next poll reports exit code 1 and shuts the other processes
    down) instead of this process: their default disposition would take the
    servicer down with the engine, and an engine's own launch-phase handler
    kills the whole process tree. Handlers can only be installed on the main
    thread; returns whether they were."""
    if threading.current_thread() is not threading.main_thread():
        return False

    def on_signal(signum: int, _frame: Any) -> None:
        logger.error(
            "%s: a scheduler process signalled a failure (signal %d); stopping", what, signum
        )
        target.fault(1)

    for sig in signals:
        signal.signal(sig, on_signal)
    return True


def _descendants(pid: int) -> list[int]:
    """Every live descendant of ``pid``, from one pass over ``/proc`` (the
    per-task ``children`` files are not available on every kernel)."""
    parent_of: dict[int, int] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as handle:
                parent_of[int(entry)] = int(handle.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
    found: list[int] = []
    pending = [pid]
    while pending:
        parent = pending.pop()
        for child, child_parent in parent_of.items():
            if child_parent == parent and child not in found:
                found.append(child)
                pending.append(child)
    return found


def _start_time(pid: int) -> str | None:
    """The process's start time from ``/proc/<pid>/stat`` (``None`` once it is
    gone), so a recycled PID is never mistaken for the process we recorded."""
    try:
        with open(f"/proc/{pid}/stat") as handle:
            fields = handle.read().rsplit(")", 1)[1].split()
    except OSError:
        return None
    return fields[19]


def terminate_engine(engine: Any, timeout: float = ENGINE_TERMINATE_SECS) -> None:
    """SIGTERM the headless engine (it tears down its own workers), then kill;
    workers that outlived it either way are killed too."""
    if engine.poll() is None:
        engine.terminate()
        try:
            engine.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("Headless engine did not exit within %.0fs; killing it", timeout)
            engine.kill()
            engine.wait()
    stragglers = engine.kill_descendants()
    if stragglers:
        logger.warning(
            "Killed %d engine worker process(es) that outlived the headless engine", stragglers
        )


async def supervise(
    server: Any,
    engine: Any,
    *,
    drain_secs: float = DEFAULT_DRAIN_SECS,
    stop_timeout: float = 5.0,
    stop_event: asyncio.Event | None = None,
    poll_secs: float = _POLL_SECS,
) -> int:
    """Run until a shutdown signal, an engine exit, or a server failure.

    While the engine's handshake runs, every poll that finds the engine
    process alive is reported to a server that takes such reports
    (``note_engine_alive``), so the server's startup bound counts silence
    from a living engine rather than time since the last handshake message.

    Signals drain: health flips to NOT_SERVING at once (the Router stops
    routing here), in-flight streams get ``drain_secs`` to finish with the
    engine still up, then the server stops and the engine is terminated.
    Returns the process exit code.
    """
    loop = asyncio.get_running_loop()
    note_engine_alive = getattr(server, "note_engine_alive", None)
    if stop_event is None:
        stop_event = asyncio.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop_event.set)
    exit_code = 0
    announced_ready = False
    try:
        while not stop_event.is_set():
            if not announced_ready and server.engine_ready:
                announced_ready = True
                logger.info("Engine connected; the servicer is SERVING")
            error = server.last_error
            if error or not server.running:
                logger.error("Rust servicer cannot serve: %s", error or "server exited")
                exit_code = 1
                break
            rc = engine.poll()
            if rc is not None:
                logger.error("Headless engine exited with code %s", rc)
                exit_code = 1
                break
            if not announced_ready and note_engine_alive is not None:
                note_engine_alive()
            try:
                await asyncio.wait_for(stop_event.wait(), poll_secs)
            except asyncio.TimeoutError:  # noqa: UP041 -- distinct from the builtin before 3.11
                pass
    finally:
        try:
            server.set_serving(False)
        except Exception:
            logger.exception("Failed to mark the servicer as draining")
        if stop_event.is_set() and drain_secs > 0 and engine.poll() is None:
            logger.info("Draining for %.1fs before stopping", drain_secs)
            await asyncio.sleep(drain_secs)
        try:
            await asyncio.to_thread(server.stop, stop_timeout)
        except Exception:
            logger.exception("Failed to stop the Rust servicer cleanly")
        terminate_engine(engine)
    return exit_code
