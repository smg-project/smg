"""The Python-owned lifecycle around a Rust servicer, shared by the vLLM and
TokenSpeed launchers: a free port and socket directory for the handshake, the
engine as a ``Popen``-shaped handle (one spawned headless engine, or the
engine cores a manager started from this process), its termination, and the
supervision loop that drains on a signal and exits with the engine or the
server. Nothing here knows which engine it supervises.
"""

from __future__ import annotations

import asyncio
import glob
import logging
import os
import signal
import socket
import subprocess
import time
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DRAIN_SECS = 5.0
ENGINE_TERMINATE_SECS = 30.0
# Bound on the engine's ZMQ handshake. An engine's start includes model load,
# kernel JIT and graph capture; a cold kernel cache has taken over ten minutes.
# A dead engine never waits this long: ``supervise`` polls the engine process.
DEFAULT_STARTUP_TIMEOUT_SECS = 1800.0
_POLL_SECS = 0.5
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
            self._exit_code = next((code for code in codes if code is not None), None)
        if self._exit_code is not None and None in codes:
            self._shutdown()
        return self._exit_code

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

    Signals drain: health flips to NOT_SERVING at once (the Router stops
    routing here), in-flight streams get ``drain_secs`` to finish with the
    engine still up, then the server stops and the engine is terminated.
    Returns the process exit code.
    """
    loop = asyncio.get_running_loop()
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
