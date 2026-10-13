"""The Python-owned lifecycle around a Rust servicer, shared by the engine
launchers: the servicer switch (which implementation serves the process, from
the launcher's flag or the engine's variable), the launcher's knobs, the
KV-events default (the engine's publisher on unless the launcher or its
variable says otherwise), a free port and socket directory for the handshake,
the engine as a ``Popen``-shaped handle (one spawned headless engine, the
engine cores a manager started from this process, or the scheduler processes
this process started itself), its termination, the supervision loop that
drains on a signal and exits with the engine or the server, and
:class:`RustLauncher`, the one sequence from the knobs to the supervision that
every launcher runs with its engine's hooks filled in. Nothing here knows
which engine it supervises.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import faulthandler
import glob
import importlib
import logging
import math
import multiprocessing
import os
import signal
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

from smg_grpc_servicer.hostport import host_port

logger = logging.getLogger(__name__)

DEFAULT_DRAIN_SECS = 5.0
ENGINE_TERMINATE_SECS = 30.0
# How long the Rust server gets to stop: at the end of the supervision loop,
# and when the engine launch behind an already-bound server failed.
SERVER_STOP_SECS = 5.0
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


SOCKET_DIR_ENV = "SMG_ZMQ_SOCKET_DIR"
# The exit code of a servicer that refuses to start over a knob it cannot use
# (argparse's convention for a usage error); an engine that died is 1.
CONFIG_ERROR_EXIT_CODE = 2


class ServicerConfigError(ValueError):
    """A knob of the servicer (an environment variable, or the flag that
    stands for one) holds a value the servicer cannot use. The message names
    the variable and the value, so an operator can act on it without a
    traceback; the entry points answer it with :data:`CONFIG_ERROR_EXIT_CODE`
    (:func:`configuration_exit`)."""


def env_float(
    name: str,
    default: float,
    *,
    environ: Mapping[str, str] | None = None,
    minimum: float | None = None,
    positive: bool = False,
) -> float:
    """``float(environ[name])``, ``default`` when the variable is unset or
    empty. A value that is not a finite number, below ``minimum``, or not
    above zero under ``positive`` is a :class:`ServicerConfigError` naming the
    variable."""
    source = os.environ if environ is None else environ
    raw = source.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ServicerConfigError(f"{name}={raw!r} is not a number") from None
    if math.isnan(value) or math.isinf(value):
        raise ServicerConfigError(f"{name}={raw!r} is not a finite number")
    if positive and value <= 0:
        raise ServicerConfigError(f"{name}={raw!r} must be greater than 0")
    if minimum is not None and value < minimum:
        raise ServicerConfigError(f"{name}={raw!r} must be at least {minimum:g}")
    return value


def env_port(name: str, *, environ: Mapping[str, str] | None = None) -> int | None:
    """A TCP port from ``environ[name]``, 1 to 65535; ``None`` when the
    variable is unset, empty or 0 (the caller picks a free port). Anything
    else is a :class:`ServicerConfigError` naming the variable."""
    source = os.environ if environ is None else environ
    raw = source.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ServicerConfigError(f"{name}={raw!r} is not a port number") from None
    if value == 0:
        return None
    if not 1 <= value <= 65535:
        raise ServicerConfigError(f"{name}={raw!r} must be between 1 and 65535")
    return value


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def default_socket_dir(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    return source.get(SOCKET_DIR_ENV) or f"/tmp/smg-zmq-{os.getuid()}"


def prepare_socket_dir(path: str) -> str:
    """``path`` as the servicer's IPC socket directory, created (mode 0700)
    when it is missing. A path that exists and is not a directory, or that
    cannot be created, is a :class:`ServicerConfigError` naming
    ``SMG_ZMQ_SOCKET_DIR``: found before the gRPC port is bound, not by the
    engine's first connection."""
    try:
        os.makedirs(path, mode=0o700, exist_ok=True)
    except FileExistsError:
        raise ServicerConfigError(
            f"{SOCKET_DIR_ENV}={path!r} exists and is not a directory"
        ) from None
    except OSError as error:
        raise ServicerConfigError(
            f"{SOCKET_DIR_ENV}={path!r} cannot be created: {error.strerror or error}"
        ) from None
    return path


@dataclasses.dataclass(frozen=True)
class LauncherKnobs:
    """The launcher's own knobs, read and validated together
    (:func:`read_launcher_knobs`) before any engine work and before the Rust
    server binds its port."""

    handshake_port: int | None
    drain_secs: float
    startup_timeout_secs: float
    startup_ceiling_secs: float
    socket_dir: str


def read_launcher_knobs(
    *,
    handshake_port_env: str,
    drain_secs_env: str,
    startup_timeout_secs_env: str,
    startup_ceiling_secs_env: str,
    socket_dir: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> LauncherKnobs:
    """Every knob of a Rust-servicer launcher at once, each failure a
    :class:`ServicerConfigError` naming its variable: the handshake port (1 to
    65535, or a free one), the drain on a signal (0 s or more), the startup
    silence bound (above 0 s), the startup ceiling (0 lifts it), the IPC socket
    directory (``socket_dir``, else ``$SMG_ZMQ_SOCKET_DIR``, else the
    default; created). Nothing is read later: a bad value is refused before
    the gRPC port is bound and before the engine starts."""
    return LauncherKnobs(
        handshake_port=env_port(handshake_port_env, environ=environ),
        drain_secs=env_float(drain_secs_env, DEFAULT_DRAIN_SECS, environ=environ, minimum=0.0),
        startup_timeout_secs=env_float(
            startup_timeout_secs_env,
            DEFAULT_STARTUP_TIMEOUT_SECS,
            environ=environ,
            positive=True,
        ),
        startup_ceiling_secs=env_float(
            startup_ceiling_secs_env, DEFAULT_STARTUP_CEILING_SECS, environ=environ, minimum=0.0
        ),
        socket_dir=prepare_socket_dir(
            default_socket_dir(environ) if socket_dir is None else socket_dir
        ),
    )


def configuration_exit(error: ServicerConfigError, log: logging.Logger = logger) -> SystemExit:
    """The ``SystemExit`` an entry point raises for a configuration error: one
    ERROR line naming the knob and exit code :data:`CONFIG_ERROR_EXIT_CODE`,
    no traceback."""
    log.error("servicer configuration error: %s", error)
    return SystemExit(CONFIG_ERROR_EXIT_CODE)


# ---------------------------------------------------------------------------
# The servicer switch: which implementation serves this process
# ---------------------------------------------------------------------------

IMPLS = ("python", "rust")
# Where the implementation choice came from, for the startup log line.
SOURCE_FLAG = "flag"
SOURCE_ENV = "env"
SOURCE_DEFAULT = "default"


def servicer_switch(
    env_name: str,
    args: Any = None,
    environ: Mapping[str, str] | None = None,
    *,
    parsed_flag: str | None = None,
) -> tuple[str, str]:
    """Which implementation serves this process and where the choice came
    from, as ``(impl, origin)``: the launcher's ``--servicer-impl``
    (``args.servicer_impl``, else ``parsed_flag``, the value a launcher's
    plugin kept when the engine's namespace carries only the engine's own
    fields), else ``environ[env_name]``, else ``python``; the origin is
    ``flag``, ``env`` or ``default``. A value outside :data:`IMPLS` is a
    :class:`ServicerConfigError` naming the variable.

    A decision made with the flag in hand is written back to the
    environment, so the engine processes this process starts and anything
    that reads only the variable (the Python servicer's guard against a
    switch that never ran) agree with it when ``--servicer-impl python``
    overrides an exported ``rust``. A value that came from the environment
    or from the default is not written: what the engine inherits there is
    what the operator exported, spelled as they exported it, and nothing
    when they exported nothing."""
    source = os.environ if environ is None else environ
    value = getattr(args, "servicer_impl", None) if args is not None else None
    value = value or parsed_flag
    origin = SOURCE_FLAG
    if not value:
        value = source.get(env_name)
        origin = SOURCE_ENV if value else SOURCE_DEFAULT
    value = str(value or "python").strip().lower()
    if value not in IMPLS:
        raise ServicerConfigError(f"{env_name} must be one of {IMPLS}, got {value!r}")
    if origin == SOURCE_FLAG and isinstance(source, MutableMapping):
        source[env_name] = value
    return value, origin


# The values of a launcher's KV-events variable (``RustLauncher.kv_events_env``)
# that keep the engine's KV event publisher off when the launcher was given no
# --kv-events-config.
KV_EVENTS_OFF_VALUES = ("0", "false", "no", "off")


def apply_kv_events_default(
    target: Any,
    default: Any,
    *,
    env_name: str,
    environ: Mapping[str, str] | None = None,
) -> Any | None:
    """Turn the engine's KV event publisher on when the launcher was given no
    ``--kv-events-config``; returns the configuration applied, or ``None``.

    Cache-aware routing lives on the events ``SubscribeKvEvents`` relays, and
    no engine publishes any unless started with a ``--kv-events-config`` that
    names its ZMQ publisher: a launcher without it got a router that routed
    blind, with one WARN per worker as the only trace. Under the Rust servicer
    ``target`` (the parsed launcher namespace or the engine's server args,
    whichever the engine builds its configuration from) gets ``default`` (the
    engine's ZMQ-publisher configuration, or a callable building it) as its
    ``kv_events_config`` before the facts and the engine are built from it, so
    publisher and relay agree. An explicit configuration, anything but ``None``
    or an empty string, is kept as given, off included; ``env_name`` set to
    one of ``KV_EVENTS_OFF_VALUES`` keeps the publisher off without one; a
    ``target`` that cannot be written (frozen server args) is left as it is,
    with a warning.
    """
    source = os.environ if environ is None else environ
    given = getattr(target, "kv_events_config", None)
    if given is not None and given != "":
        return None
    if str(source.get(env_name, "")).strip().lower() in KV_EVENTS_OFF_VALUES:
        return None
    config = default() if callable(default) else default
    try:
        target.kv_events_config = config
    except AttributeError:  # frozen server args: left as they are
        logger.warning(
            "kv_events_config cannot be set on %s; KV event publishing stays as configured",
            type(target).__name__,
        )
        return None
    return config


# How long a servicer that decided to exit over its engine may spend in the
# interpreter's shutdown before the process is ended anyway.
EXIT_GRACE_SECS = 15.0


def _end_process(code: int, grace_secs: float) -> None:
    """The guard fired: say so, record where every thread stands (the cause
    of the hang, for the next reader of the log), terminate the children the
    interpreter's own exit would have joined, then end the process. Every
    step before the exit is best effort: the exit is the last resort and
    happens whatever the others could do (a dump to a closed stderr raises)."""
    try:
        logger.error(
            "The servicer's shutdown did not finish within %.0f s; ending the process with code %d",
            grace_secs,
            code,
        )
        for handler in logging.getLogger().handlers:
            with contextlib.suppress(Exception):
                handler.flush()
        with contextlib.suppress(Exception):
            faulthandler.dump_traceback(all_threads=True)
        for child in multiprocessing.active_children():
            with contextlib.suppress(Exception):
                child.terminate()
    except BaseException:  # noqa: BLE001 -- nothing may stand between the guard and the exit
        pass
    finally:
        os._exit(code)


def guarded_exit(code: int, *, grace_secs: float = EXIT_GRACE_SECS) -> SystemExit:
    """The ``SystemExit`` a Python servicer raises to end over its engine
    (the engine died, or never started), with a guard: the orderly exit runs
    first (the drain, the launcher's own clean-up, the interpreter's
    shutdown), and should that hang (a thread that never joins, a messaging
    context waiting on an unclosed socket: a servicer was seen stuck there
    for minutes with its engine already gone), a daemon timer ends the
    process with ``code`` after ``grace_secs``. Arm it the moment the exit is
    decided, ahead of the shutdown steps, with a grace that covers the
    bounded ones."""
    timer = threading.Timer(grace_secs, _end_process, args=(code, grace_secs))
    timer.daemon = True
    timer.start()
    return SystemExit(code)


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


def mark_not_serving(server: Any) -> None:
    """Health flips to NOT_SERVING (the Router stops routing here); a failure
    to flip it is logged, never raised: the caller's own outcome is the one
    that propagates."""
    try:
        server.set_serving(False)
    except Exception:
        logger.exception("Failed to mark the servicer as not serving")


async def stop_server(server: Any, timeout: float) -> None:
    """Stop the Rust server off the event loop, within ``timeout``; a failure
    to stop is logged, never raised, for the same reason."""
    try:
        await asyncio.to_thread(server.stop, timeout)
    except Exception:
        logger.exception("Failed to stop the Rust servicer cleanly")


async def launch_or_stop(server: Any, launch: Callable[[], Any]) -> Any:
    """Start the engine behind an already-bound server: ``launch()``'s handle,
    or, when the launch raises, that error after the server was marked not
    serving and stopped, each step best effort. The server owns the port and
    a runtime thread; a launch that fails must not leave them to the process
    exit, and the launch error is what the entry point reports. A
    cancellation of this task while the stop runs waits for the stop too:
    the port is released either way."""
    try:
        return launch()
    except BaseException:
        mark_not_serving(server)
        stopping = asyncio.create_task(stop_server(server, SERVER_STOP_SECS))
        try:
            await asyncio.shield(stopping)
        except asyncio.CancelledError:
            await asyncio.shield(stopping)
            raise
        raise


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
    stop_timeout: float = SERVER_STOP_SECS,
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
        mark_not_serving(server)
        if stop_event.is_set() and drain_secs > 0 and engine.poll() is None:
            logger.info("Draining for %.1fs before stopping", drain_secs)
            await asyncio.sleep(drain_secs)
        await stop_server(server, stop_timeout)
        terminate_engine(engine)
    return exit_code


# ---------------------------------------------------------------------------
# The launcher: one lifecycle for every engine's Rust servicer
# ---------------------------------------------------------------------------


class RustLauncher:
    """The lifecycle of a Rust servicer as every engine's launcher runs it, in
    :meth:`serve`: the launcher's knobs first, the Rust server bound, the
    engine started from this process against the server's handshake, both
    supervised until a signal or a failure. What differs per engine is named
    here and filled by a subclass: the engine's words in the log lines, its
    environment variables, the facts it advertises, where its tokenizer
    lives, and how its processes start.

    One instance serves one process. The hooks run in the order ``serve``
    lists them, so a hook may keep on the instance what a later one needs
    (the engine config the facts were read from).
    """

    # The engine's name in the log lines; the word for what the handshake
    # waits for (its "engine" or its "scheduler"); and the word for what the
    # engine count counts: the handshake identities (a data-parallel engine of
    # vLLM, a DP rank of a scheduler whose TP ranks share one identity).
    engine: str
    started: str
    counted: str
    # The ``smg.servicer`` class that serves this engine's contract, by name:
    # the binding is imported when ``serve`` runs, after the knobs are read.
    server_class: str
    # The IPC socket name prefix under the socket directory.
    ipc_prefix: str
    # This launcher's own environment variables (:func:`read_launcher_knobs`).
    handshake_port_env: str
    drain_secs_env: str
    startup_timeout_secs_env: str
    startup_ceiling_secs_env: str
    kv_events_env: str
    # What the servicer loses without a local tokenizer directory, for the warning.
    without_tokenizer: str

    def prepare(self, args: Any) -> None:
        """Engine-specific set-up after the knobs are read and the binding is
        imported, before the first log line (nothing by default)."""

    def kv_events_default(self, args: Any) -> Any | None:
        """Turn the engine's KV event publisher on when the launcher was given
        no ``--kv-events-config``: the configuration applied, or ``None``
        (``apply_kv_events_default`` with the engine's own default)."""
        raise NotImplementedError

    def facts(self, args: Any) -> dict[str, Any]:
        """The Rust server's keyword arguments for ``GetModelInfo`` and
        ``GetServerInfo``; ``serve`` reads ``data_parallel_size``,
        ``kv_events_endpoint`` and ``model_path`` from them."""
        raise NotImplementedError

    def tokenizer_dir(self, args: Any) -> str | None:
        """A local directory holding the served model's tokenizer, or ``None``."""
        raise NotImplementedError

    def bind_address(self, args: Any) -> str:
        """Where the Rust server listens: the launcher's host and port (a
        launcher that leaves host unset binds every interface, as upstream's
        own server does then)."""
        host = getattr(args, "host", None) or "0.0.0.0"
        return host_port(host, int(getattr(args, "port", 0) or 0))

    def server_extras(self, args: Any, tokenizer_dir: str | None) -> dict[str, Any]:
        """Further keyword arguments of the Rust server (none by default)."""
        return {}

    def launch(self, args: Any, handshake_port: int, facts: Mapping[str, Any]) -> Any:
        """Start the engine against the server's handshake port: the
        ``Popen``-shaped handle :func:`supervise` polls."""
        raise NotImplementedError

    def launched(self, engine: Any) -> None:
        """Once the engine started, before the supervision (nothing by default)."""

    def announce_capabilities(self, facts: Mapping[str, Any], tokenizer_dir: str | None) -> None:
        """What a cache-aware router and a tokenizer-backed feature get from
        this servicer: whether SubscribeKvEvents relays a publisher, and what
        is lost without a local tokenizer directory."""
        if facts.get("kv_events_endpoint"):
            logger.info(
                "SubscribeKvEvents relays %s's KV events from %s",
                self.engine,
                facts["kv_events_endpoint"],
            )
        else:
            logger.warning(
                "SubscribeKvEvents is off (no ZMQ KV event publisher configured): a cache-aware "
                "router sees nothing of this engine's cache"
            )
        if tokenizer_dir is None:
            logger.warning(
                "No local tokenizer directory for %s; %s",
                facts.get("model_path", ""),
                self.without_tokenizer,
            )

    def announce(
        self, server: Any, knobs: LauncherKnobs, handshake_port: int, engine_count: int
    ) -> None:
        """The two lines an operator reads at the start: where the server
        listens and what dials its handshake, and the startup bounds with the
        variables that set them."""
        logger.info(
            "Rust %s servicer listening on %s (%s handshake tcp://127.0.0.1:%d, %d %s(s))",
            self.engine,
            server.address,
            self.started,
            handshake_port,
            engine_count,
            self.counted,
        )
        logger.info(
            "%s startup bounds: %.0fs without a sign of life (%s), ceiling %s (%s); "
            "the handshake waits while the %s process is alive",
            self.started.capitalize(),
            knobs.startup_timeout_secs,
            self.startup_timeout_secs_env,
            f"{knobs.startup_ceiling_secs:.0f}s" if knobs.startup_ceiling_secs else "none",
            self.startup_ceiling_secs_env,
            self.started,
        )

    async def serve(self, args: Any) -> int:
        """Serve this process's gRPC contract from Rust. Returns the process
        exit code."""
        # The launcher's own knobs first, validated together: a bad value is a
        # named error before any engine work and before the port is bound.
        knobs = read_launcher_knobs(
            handshake_port_env=self.handshake_port_env,
            drain_secs_env=self.drain_secs_env,
            startup_timeout_secs_env=self.startup_timeout_secs_env,
            startup_ceiling_secs_env=self.startup_ceiling_secs_env,
            socket_dir=default_socket_dir(),
        )
        servicer = importlib.import_module("smg.servicer")
        self.prepare(args)
        servicer.init_servicer_tracing()
        if self.kv_events_default(args) is not None:
            logger.info(
                "KV event publishing enabled: no --kv-events-config was given, so %s's ZMQ "
                "publisher is on with its default endpoint and SubscribeKvEvents relays it; pass "
                "--kv-events-config to configure it, or set %s=0 to leave it off",
                self.engine,
                self.kv_events_env,
            )
        facts = self.facts(args)
        engine_count = facts["data_parallel_size"]
        tokenizer_dir = self.tokenizer_dir(args)
        self.announce_capabilities(facts, tokenizer_dir)
        handshake_port = knobs.handshake_port or free_port()
        server = getattr(servicer, self.server_class)(
            bind_address=self.bind_address(args),
            # Per process, not per requested port: `--port 0` launchers would
            # otherwise share one path and unlink each other's sockets.
            ipc_base_url=f"ipc://{knobs.socket_dir}/{self.ipc_prefix}-{os.getpid()}",
            handshake_address=f"tcp://127.0.0.1:{handshake_port}",
            engine_count=engine_count,
            tokenizer_dir=tokenizer_dir,
            engine_startup_timeout_secs=knobs.startup_timeout_secs,
            engine_startup_ceiling_secs=knobs.startup_ceiling_secs,
            **self.server_extras(args, tokenizer_dir),
            **facts,
        )
        self.announce(server, knobs, handshake_port, engine_count)
        engine = await launch_or_stop(server, lambda: self.launch(args, handshake_port, facts))
        self.launched(engine)
        return await supervise(server, engine, drain_secs=knobs.drain_secs)
