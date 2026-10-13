"""Unit tests for the lifecycle helpers the Rust servicer launchers share."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
import types

import pytest
from smg_grpc_servicer import rust_lifecycle


def test_a_local_directory_is_the_tokenizer_directory(tmp_path):
    assert rust_lifecycle.resolve_tokenizer_dir(str(tmp_path)) == str(tmp_path)


def test_a_cached_snapshot_counts_only_once_it_holds_a_tokenizer(tmp_path, monkeypatch):
    """The engine downloads the same repo in parallel; a snapshot it has
    started (config present, tokenizer not yet) must not satisfy the local
    lookup, the Hub attempt that follows fetches the tokenizer files."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    calls: list[bool] = []

    def fake_snapshot_download(repo, revision=None, allow_patterns=None, local_files_only=False):
        calls.append(local_files_only)
        assert "*.json" in allow_patterns and "*.safetensors" not in str(allow_patterns)
        if not local_files_only:
            (snapshot / "tokenizer.json").write_text("{}")
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    assert rust_lifecycle.resolve_tokenizer_dir("org/model") == str(snapshot)
    assert calls == [True, False]

    # Once the tokenizer is cached the local lookup is enough.
    calls.clear()
    assert rust_lifecycle.resolve_tokenizer_dir("org/model") == str(snapshot)
    assert calls == [True]


def test_an_unresolvable_tokenizer_is_none(tmp_path, monkeypatch):
    def failing(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=failing)
    )
    assert rust_lifecycle.resolve_tokenizer_dir("org/missing") is None


@pytest.mark.parametrize("name", ["tokenizer.json", "tokenizer.model", "x.tiktoken", "vocab.json"])
def test_holds_tokenizer_recognises_each_loader_input(tmp_path, name):
    assert not rust_lifecycle.holds_tokenizer(str(tmp_path))
    (tmp_path / name).write_text("")
    assert rust_lifecycle.holds_tokenizer(str(tmp_path))


class _PopenAsSpawned:
    """A ``multiprocessing``-shaped view over a ``Popen`` for the lifecycle helpers."""

    def __init__(self, popen: subprocess.Popen):
        self._popen = popen

    @property
    def pid(self) -> int:
        return self._popen.pid

    @property
    def exitcode(self) -> int | None:
        return self._popen.poll()

    def join(self, timeout: float | None = None) -> None:
        try:
            self._popen.wait(timeout)
        except subprocess.TimeoutExpired:
            pass

    def terminate(self) -> None:
        self._popen.terminate()

    def kill(self) -> None:
        self._popen.kill()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as stat:  # a zombie is as dead as we need
            return stat.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False  # reaped: before the open (ENOENT), or between it and the read (ESRCH)


def _worker_pid(pid_file, deadline_secs: float = 10) -> int:
    deadline = time.monotonic() + deadline_secs
    while not pid_file.exists() or not pid_file.read_text().strip():
        assert time.monotonic() < deadline, "the launcher never spawned its worker"
        time.sleep(0.05)
    return int(pid_file.read_text())


def _wait_dead(pid: int, what: str, deadline_secs: float = 10) -> None:
    deadline = time.monotonic() + deadline_secs
    while _alive(pid):
        assert time.monotonic() < deadline, f"the {what} outlived the engine cleanup"
        time.sleep(0.05)


def test_kill_reaches_the_workers_the_engine_spawned(tmp_path):
    """A kill of the headless engine also kills the workers it spawned, while
    everything stays in the launcher's process group."""
    pid_file = tmp_path / "worker.pid"
    launcher = subprocess.Popen(["bash", "-c", f"sleep 300 & echo $! > {pid_file}; wait"])
    worker_pid = _worker_pid(pid_file)
    engine = rust_lifecycle.EngineProcess(_PopenAsSpawned(launcher))
    assert os.getpgid(launcher.pid) == os.getpgid(0)
    assert engine.poll() is None  # tracks the worker

    engine.kill()

    assert engine.wait(timeout=10) == -signal.SIGKILL
    _wait_dead(worker_pid, "worker")


def test_cleanup_after_the_engine_exited_kills_the_workers_it_left(tmp_path):
    """An engine that crashed leaves re-parented workers; terminate_engine
    kills the ones it had seen while the engine was alive."""
    pid_file = tmp_path / "worker.pid"
    launcher = subprocess.Popen(
        ["bash", "-c", f"sleep 300 & echo $! > {pid_file}; sleep 2; exit 3"]
    )
    worker_pid = _worker_pid(pid_file)
    engine = rust_lifecycle.EngineProcess(_PopenAsSpawned(launcher))
    assert engine.poll() is None  # tracks the worker while the launcher lives
    assert engine.wait(timeout=10) == 3
    assert _alive(worker_pid)

    rust_lifecycle.terminate_engine(engine)

    _wait_dead(worker_pid, "orphaned worker")


def test_a_recycled_pid_is_not_killed(tmp_path):
    """Only a tracked process whose start time still matches is signalled."""
    launcher = subprocess.Popen(["sleep", "300"])
    engine = rust_lifecycle.EngineProcess(_PopenAsSpawned(launcher))
    engine.track()
    engine._descendants[launcher.pid] = "not-its-start-time"
    assert engine.kill_descendants() == 0
    assert _alive(launcher.pid)
    launcher.kill()
    launcher.wait()


class _ManagerOf:
    """vLLM's process manager as the group sees it, over spawned-shaped launchers."""

    def __init__(self, launchers: list[subprocess.Popen]):
        self.processes = [_PopenAsSpawned(launcher) for launcher in launchers]
        self.shutdowns: list[float | None] = []

    def shutdown(self, timeout: float | None = None) -> None:
        self.shutdowns.append(timeout)
        for process in self.processes:
            if process.exitcode is None:
                process.terminate()


def test_group_kill_reaches_the_workers_of_every_core(tmp_path):
    """Each engine core's workers are tracked and killed with it, as for the
    single spawned engine."""
    pid_files = [tmp_path / "a.pid", tmp_path / "b.pid"]
    launchers = [
        subprocess.Popen(["bash", "-c", f"sleep 300 & echo $! > {pid_file}; wait"])
        for pid_file in pid_files
    ]
    workers = [_worker_pid(pid_file) for pid_file in pid_files]
    engines = rust_lifecycle.EngineProcessGroup(_ManagerOf(launchers))
    assert engines.pids == [launcher.pid for launcher in launchers]
    assert engines.poll() is None  # tracks every core's workers

    engines.kill()

    assert engines.wait(timeout=10) == -signal.SIGKILL
    for worker in workers:
        _wait_dead(worker, "worker")


def test_one_cores_exit_ends_the_group_and_the_cleanup_reaps_every_worker(tmp_path):
    """A core that exits takes the engine down: the manager shuts the other
    cores down and `terminate_engine` kills the workers they left."""
    pid_files = [tmp_path / "a.pid", tmp_path / "b.pid"]
    launchers = [
        subprocess.Popen(["bash", "-c", f"sleep 300 & echo $! > {pid_files[0]}; sleep 1; exit 3"]),
        subprocess.Popen(["bash", "-c", f"sleep 300 & echo $! > {pid_files[1]}; wait"]),
    ]
    workers = [_worker_pid(pid_file) for pid_file in pid_files]
    manager = _ManagerOf(launchers)
    engines = rust_lifecycle.EngineProcessGroup(manager)
    assert engines.poll() is None
    deadline = time.monotonic() + 10
    while engines.poll() is None:
        assert time.monotonic() < deadline, "the first core never exited"
        time.sleep(0.05)
    assert engines.poll() == 3 and manager.shutdowns == [None]
    assert engines.wait(timeout=10) == 3  # the first exit code; the sibling was terminated

    rust_lifecycle.terminate_engine(engines)

    for worker in workers:
        _wait_dead(worker, "orphaned worker")


class _FakeRank:
    """A ``multiprocessing``-shaped scheduler process this process started."""

    def __init__(self, pid: int, *, stubborn: bool = False):
        self.pid = pid
        self.exitcode: int | None = None
        self.events: list[str] = []
        self._stubborn = stubborn  # ignores SIGTERM; only a kill ends it

    def is_alive(self) -> bool:
        return self.exitcode is None

    def terminate(self) -> None:
        self.events.append("terminate")
        if not self._stubborn:
            self.exitcode = -signal.SIGTERM

    def kill(self) -> None:
        self.events.append("kill")
        self.exitcode = -signal.SIGKILL

    def join(self, timeout=None) -> None:
        self.events.append("join")


def _signals(rank: _FakeRank) -> list[str]:
    return [event for event in rank.events if event != "join"]


def test_scheduler_processes_shutdown_terminates_then_kills_what_stays():
    """Every live process gets SIGTERM; one that outlives the grace period is
    killed; one that had already exited is left alone."""
    ranks = [_FakeRank(1), _FakeRank(2, stubborn=True), _FakeRank(3)]
    ranks[2].exitcode = 0
    manager = rust_lifecycle.SchedulerProcesses(ranks, grace_secs=0.01)
    assert manager.pids == [1, 2, 3]

    manager.shutdown()

    assert [_signals(rank) for rank in ranks] == [["terminate"], ["terminate", "kill"], []]
    assert all(rank.exitcode is not None for rank in ranks)


def test_scheduler_processes_wait_for_the_ready_reports_on_a_thread(caplog):
    """The ready wait runs off the caller's thread and logs its outcome: the
    count when every process reported, the error when the start failed."""
    threads: list[threading.Thread] = []

    def ready():
        threads.append(threading.current_thread())

    manager = rust_lifecycle.SchedulerProcesses([_FakeRank(1), _FakeRank(2)], ready=ready)
    with caplog.at_level("INFO", logger=rust_lifecycle.logger.name):
        thread = manager.watch_ready("test scheduler")
        assert thread is not None
        thread.join(5)
    assert threads == [thread] and thread is not threading.main_thread()
    assert "test scheduler: 2 scheduler process(es) ready" in caplog.text
    assert manager.ready_error is None

    def failing_ready():
        raise RuntimeError("rank 1 died")

    failing = rust_lifecycle.SchedulerProcesses([_FakeRank(1)], ready=failing_ready)
    with caplog.at_level("ERROR", logger=rust_lifecycle.logger.name):
        failing.watch_ready("test scheduler").join(5)
    assert str(failing.ready_error) == "rank 1 died"
    assert "test scheduler: the scheduler processes did not start: rank 1 died" in caplog.text
    # No ready wait: nothing to watch.
    assert rust_lifecycle.SchedulerProcesses([_FakeRank(1)]).watch_ready("x") is None


def test_wait_for_ready_reports_reads_one_report_per_pipe():
    pipes = [multiprocessing.Pipe(duplex=False) for _ in range(2)]
    ranks = [_FakeRank(1), _FakeRank(2)]
    for index, (_reader, writer) in enumerate(pipes):
        writer.send({"status": "ready", "rank": index})
    reports = rust_lifecycle.wait_for_ready_reports(
        [reader for reader, _writer in pipes], ranks, poll_secs=0.01
    )
    assert reports == [{"status": "ready", "rank": 0}, {"status": "ready", "rank": 1}]


def test_wait_for_ready_reports_notices_a_failed_start():
    """A process that died before reporting, a closed pipe, and a report that
    is not `ready` each end the wait with the reason."""
    reader, writer = multiprocessing.Pipe(duplex=False)
    dead = _FakeRank(7)
    dead.exitcode = -signal.SIGKILL
    with pytest.raises(RuntimeError, match="process 7 exited with -9 before it was ready"):
        rust_lifecycle.wait_for_ready_reports([reader], [dead], poll_secs=0.01)

    reader, writer = multiprocessing.Pipe(duplex=False)
    writer.close()
    with pytest.raises(RuntimeError, match="process 8 closed its pipe before it was ready"):
        rust_lifecycle.wait_for_ready_reports([reader], [_FakeRank(8)], poll_secs=0.01)

    reader, writer = multiprocessing.Pipe(duplex=False)
    writer.send({"status": "error"})
    with pytest.raises(RuntimeError, match="process 0 failed to start"):
        rust_lifecycle.wait_for_ready_reports([reader], [_FakeRank(9)], poll_secs=0.01)


def test_a_fault_ends_the_group_and_shuts_the_other_processes_down():
    """`fault` reports a failure a process signalled out of band: the next poll
    returns the code and terminates the processes still alive; the fault is
    the group's exit code even when a process exits with another one later."""
    ranks = [_FakeRank(1), _FakeRank(2)]
    group = rust_lifecycle.EngineProcessGroup(
        rust_lifecycle.SchedulerProcesses(ranks, grace_secs=0.01)
    )
    assert group.poll() is None

    group.fault(1)

    assert group.poll() == 1
    assert [_signals(rank) for rank in ranks] == [["terminate"], ["terminate"]]
    assert group.wait() == 1
    ranks[0].exitcode = 3
    assert group.poll() == 1
    group.fault(5)  # a second report changes nothing
    assert group.poll() == 1


def test_a_fault_signal_ends_the_group_instead_of_the_process(caplog):
    """The handler `on_fault_signal` installs turns the signal into a fault of
    the group (the signal's default disposition would end this process); it
    can only be installed on the main thread."""
    ranks = [_FakeRank(1), _FakeRank(2)]
    group = rust_lifecycle.EngineProcessGroup(
        rust_lifecycle.SchedulerProcesses(ranks, grace_secs=0.01)
    )
    previous = signal.getsignal(signal.SIGUSR2)
    try:
        assert rust_lifecycle.on_fault_signal(group, signal.SIGUSR2, what="test scheduler")
        with caplog.at_level("ERROR", logger=rust_lifecycle.logger.name):
            signal.raise_signal(signal.SIGUSR2)
        assert group.poll() == 1
        assert [_signals(rank) for rank in ranks] == [["terminate"], ["terminate"]]
        assert (
            "test scheduler: a scheduler process signalled a failure "
            f"(signal {int(signal.SIGUSR2)}); stopping" in caplog.text
        )

        other = rust_lifecycle.EngineProcessGroup(
            rust_lifecycle.SchedulerProcesses([_FakeRank(3)], grace_secs=0.01)
        )
        installed: list[bool] = []
        thread = threading.Thread(
            target=lambda: installed.append(
                rust_lifecycle.on_fault_signal(other, signal.SIGUSR2, what="off the main thread")
            )
        )
        thread.start()
        thread.join(5)
        assert installed == [False]
        signal.raise_signal(signal.SIGUSR2)  # still the first group's handler
        assert other.poll() is None
    finally:
        signal.signal(signal.SIGUSR2, previous if previous is not None else signal.SIG_DFL)


def test_a_failed_ready_wait_ends_the_group():
    """A start that fails after the processes are up (a report that is not
    `ready`, a rank that died) is a fault of the group, not only a log line:
    the next poll returns 1 and the live processes are terminated, so the
    supervisor exits instead of reporting the engine alive until its ceiling."""
    ranks = [_FakeRank(1), _FakeRank(2)]
    manager = rust_lifecycle.SchedulerProcesses(
        ranks, ready=lambda: (_ for _ in ()).throw(RuntimeError("rank 2 reported an error"))
    )
    group = rust_lifecycle.EngineProcessGroup(manager)

    manager.watch_ready("test scheduler", on_failure=group.fault).join(5)

    assert str(manager.ready_error) == "rank 2 reported an error"
    assert group.poll() == 1
    assert [_signals(rank) for rank in ranks] == [["terminate"], ["terminate"]]


def test_a_fault_latch_carries_a_signal_sent_before_the_group_exists(caplog):
    """The handlers go in before the first process starts, so they target a
    latch: a signal that arrives before the group is attached ends the group
    the moment it is; one after the attach goes straight through."""
    latch = rust_lifecycle.FaultLatch()
    previous = signal.getsignal(signal.SIGUSR2)
    try:
        assert rust_lifecycle.on_fault_signal(latch, signal.SIGUSR2, what="test scheduler")
        with caplog.at_level("ERROR", logger=rust_lifecycle.logger.name):
            signal.raise_signal(signal.SIGUSR2)
        ranks = [_FakeRank(1), _FakeRank(2)]
        group = rust_lifecycle.EngineProcessGroup(
            rust_lifecycle.SchedulerProcesses(ranks, grace_secs=0.01)
        )
        assert group.poll() is None
        assert latch.attach(group) is group
        assert group.poll() == 1
        assert [_signals(rank) for rank in ranks] == [["terminate"], ["terminate"]]

        later = rust_lifecycle.EngineProcessGroup(
            rust_lifecycle.SchedulerProcesses([_FakeRank(3)], grace_secs=0.01)
        )
        rust_lifecycle.FaultLatch().attach(later)
        assert later.poll() is None  # nothing was reported to that latch
        attached = rust_lifecycle.FaultLatch()
        attached.attach(later)
        attached.fault(4)
        assert later.poll() == 4
    finally:
        signal.signal(signal.SIGUSR2, previous if previous is not None else signal.SIG_DFL)


def test_ensure_spawn_start_method_forces_spawn_only_when_it_is_not_set(monkeypatch, caplog):
    """The engine's own setup sets the start method; the guard only acts, with
    a warning, when a version stops doing so (a fork from a process running
    the Rust server's threads can deadlock the child)."""
    calls: list[tuple] = []
    monkeypatch.setattr(
        rust_lifecycle.multiprocessing, "get_start_method", lambda allow_none=False: "spawn"
    )
    monkeypatch.setattr(
        rust_lifecycle.multiprocessing,
        "set_start_method",
        lambda method, force=False: calls.append((method, force)),
    )
    rust_lifecycle.ensure_spawn_start_method("test scheduler")
    assert calls == []

    monkeypatch.setattr(
        rust_lifecycle.multiprocessing, "get_start_method", lambda allow_none=False: None
    )
    with caplog.at_level("WARNING", logger=rust_lifecycle.logger.name):
        rust_lifecycle.ensure_spawn_start_method("test scheduler")
    assert calls == [("spawn", True)]
    assert "test scheduler: the multiprocessing start method was not spawn; setting it" in (
        caplog.text
    )


# ── the launcher knobs (smg-lab #292) ──────────────────────────────────────────


def test_env_float_names_the_variable_on_a_value_it_cannot_use():
    ConfigError = rust_lifecycle.ServicerConfigError
    assert rust_lifecycle.env_float("X_DRAIN", 5.0, environ={}) == 5.0
    assert rust_lifecycle.env_float("X_DRAIN", 5.0, environ={"X_DRAIN": " "}) == 5.0
    assert rust_lifecycle.env_float("X_DRAIN", 5.0, environ={"X_DRAIN": "2.5"}) == 2.5
    with pytest.raises(ConfigError, match="X_DRAIN='abc' is not a number"):
        rust_lifecycle.env_float("X_DRAIN", 5.0, environ={"X_DRAIN": "abc"})
    with pytest.raises(ConfigError, match="X_DRAIN='-5' must be at least 0"):
        rust_lifecycle.env_float("X_DRAIN", 5.0, environ={"X_DRAIN": "-5"}, minimum=0.0)
    with pytest.raises(ConfigError, match="X_START='-1' must be greater than 0"):
        rust_lifecycle.env_float("X_START", 1.0, environ={"X_START": "-1"}, positive=True)
    with pytest.raises(ConfigError, match="X_START='nan' is not a finite number"):
        rust_lifecycle.env_float("X_START", 1.0, environ={"X_START": "nan"})
    assert issubclass(ConfigError, ValueError)


def test_env_port_takes_a_port_or_none_and_refuses_the_rest():
    ConfigError = rust_lifecycle.ServicerConfigError
    assert rust_lifecycle.env_port("X_PORT", environ={}) is None
    assert rust_lifecycle.env_port("X_PORT", environ={"X_PORT": "0"}) is None
    assert rust_lifecycle.env_port("X_PORT", environ={"X_PORT": "24321"}) == 24321
    with pytest.raises(ConfigError, match="X_PORT='abc' is not a port number"):
        rust_lifecycle.env_port("X_PORT", environ={"X_PORT": "abc"})
    with pytest.raises(ConfigError, match="X_PORT='70000' must be between 1 and 65535"):
        rust_lifecycle.env_port("X_PORT", environ={"X_PORT": "70000"})


def test_prepare_socket_dir_creates_the_directory_and_refuses_a_file(tmp_path):
    ConfigError = rust_lifecycle.ServicerConfigError
    created = rust_lifecycle.prepare_socket_dir(str(tmp_path / "sockets"))
    assert os.path.isdir(created) and (os.stat(created).st_mode & 0o777) == 0o700
    assert rust_lifecycle.prepare_socket_dir(created) == created
    a_file = tmp_path / "a-file"
    a_file.write_text("not a directory")
    with pytest.raises(ConfigError, match="SMG_ZMQ_SOCKET_DIR=.*exists and is not a directory"):
        rust_lifecycle.prepare_socket_dir(str(a_file))
    with pytest.raises(ConfigError, match="SMG_ZMQ_SOCKET_DIR=.*cannot be created"):
        rust_lifecycle.prepare_socket_dir(str(a_file / "below"))


def test_read_launcher_knobs_reads_every_knob_once_with_its_default(tmp_path):
    names = dict(
        handshake_port_env="X_HS",
        drain_secs_env="X_DRAIN",
        startup_timeout_secs_env="X_START",
        startup_ceiling_secs_env="X_CEIL",
    )
    socket_dir = {"SMG_ZMQ_SOCKET_DIR": str(tmp_path / "sockets")}
    knobs = rust_lifecycle.read_launcher_knobs(**names, environ=socket_dir)
    assert knobs == rust_lifecycle.LauncherKnobs(
        handshake_port=None,
        drain_secs=rust_lifecycle.DEFAULT_DRAIN_SECS,
        startup_timeout_secs=rust_lifecycle.DEFAULT_STARTUP_TIMEOUT_SECS,
        startup_ceiling_secs=rust_lifecycle.DEFAULT_STARTUP_CEILING_SECS,
        socket_dir=str(tmp_path / "sockets"),
    )
    assert os.path.isdir(tmp_path / "sockets")
    knobs = rust_lifecycle.read_launcher_knobs(
        **names,
        socket_dir=str(tmp_path / "given"),
        environ={"X_HS": "24321", "X_DRAIN": "0", "X_START": "42", "X_CEIL": "0", **socket_dir},
    )
    assert (knobs.handshake_port, knobs.drain_secs) == (24321, 0.0)
    assert (knobs.startup_timeout_secs, knobs.startup_ceiling_secs) == (42.0, 0.0)
    assert knobs.socket_dir == str(tmp_path / "given") and os.path.isdir(knobs.socket_dir)
    for name, value, text in [
        ("X_HS", "70000", "must be between 1 and 65535"),
        ("X_HS", "abc", "is not a port number"),
        ("X_DRAIN", "-5", "must be at least 0"),
        ("X_DRAIN", "abc", "is not a number"),
        ("X_START", "abc", "is not a number"),
        ("X_START", "-1", "must be greater than 0"),
        ("X_CEIL", "-1", "must be at least 0"),
    ]:
        with pytest.raises(rust_lifecycle.ServicerConfigError, match=f"{name}='{value}' {text}"):
            rust_lifecycle.read_launcher_knobs(**names, environ={name: value, **socket_dir})


def test_configuration_exit_is_one_error_line_and_exit_code_2(caplog):
    caplog.set_level(logging.ERROR)
    error = rust_lifecycle.ServicerConfigError("X_DRAIN='abc' is not a number")
    exit_ = rust_lifecycle.configuration_exit(error)
    assert isinstance(exit_, SystemExit)
    assert exit_.code == rust_lifecycle.CONFIG_ERROR_EXIT_CODE == 2
    assert "servicer configuration error: X_DRAIN='abc' is not a number" in caplog.text


def test_guarded_exit_ends_a_hung_shutdown_after_the_grace(monkeypatch):
    """The exit over a dead engine is a normal SystemExit first; a shutdown
    that hangs past the grace is ended by the guard with the same code."""
    ended = threading.Event()
    codes: list[int] = []

    def end_process(code: int, grace_secs: float) -> None:
        codes.append(code)
        assert grace_secs == 0.05
        ended.set()

    monkeypatch.setattr(rust_lifecycle, "_end_process", end_process)
    exit_ = rust_lifecycle.guarded_exit(1, grace_secs=0.05)
    assert isinstance(exit_, SystemExit) and exit_.code == 1
    assert ended.wait(5.0) and codes == [1]
    # The guard is a daemon timer: it never keeps a finished interpreter alive.
    guard = [t for t in threading.enumerate() if isinstance(t, threading.Timer)]
    assert all(t.daemon for t in guard)


def test_the_guard_says_it_fired_and_terminates_the_children(monkeypatch, caplog):
    """When the guard ends the process it says so, dumps every thread's stack
    (the cause of the hang, for the next reader) and terminates the children
    the interpreter's own exit would have joined, before ending the process."""
    terminated: list[int] = []
    child = types.SimpleNamespace(pid=4242, terminate=lambda: terminated.append(4242))
    monkeypatch.setattr(rust_lifecycle.multiprocessing, "active_children", lambda: [child])
    dumped: list[bool] = []
    monkeypatch.setattr(
        rust_lifecycle.faulthandler,
        "dump_traceback",
        lambda all_threads=False: dumped.append(all_threads),
    )
    ended: list[int] = []
    monkeypatch.setattr(rust_lifecycle.os, "_exit", ended.append)
    caplog.set_level(logging.ERROR)
    rust_lifecycle._end_process(1, 15.0)
    assert "did not finish within 15 s; ending the process with code 1" in caplog.text
    assert dumped == [True] and terminated == [4242] and ended == [1]


def test_the_guard_ends_the_process_even_when_its_report_fails(monkeypatch):
    """The exit is the last resort: a stack dump that raises (a closed stderr)
    or a child that cannot be terminated never keeps the process alive."""

    def failing_dump(all_threads=False):
        raise RuntimeError("sys.stderr is None")

    monkeypatch.setattr(rust_lifecycle.faulthandler, "dump_traceback", failing_dump)
    child = types.SimpleNamespace(
        pid=4242, terminate=lambda: (_ for _ in ()).throw(OSError("gone"))
    )
    monkeypatch.setattr(rust_lifecycle.multiprocessing, "active_children", lambda: [child])
    ended: list[int] = []
    monkeypatch.setattr(rust_lifecycle.os, "_exit", ended.append)
    rust_lifecycle._end_process(1, 15.0)
    assert ended == [1]


def test_launch_or_stop_stops_the_server_only_when_the_launch_raises():
    events: list[str] = []

    class Server:
        def set_serving(self, serving: bool) -> None:
            events.append(f"serving:{serving}")

        def stop(self, timeout: float) -> None:
            events.append(f"stop:{timeout}")

    server = Server()
    assert asyncio.run(rust_lifecycle.launch_or_stop(server, lambda: "engine")) == "engine"
    assert events == []

    def failing_launch():
        raise RuntimeError("engine would not start")

    with pytest.raises(RuntimeError, match="engine would not start"):
        asyncio.run(rust_lifecycle.launch_or_stop(server, failing_launch))
    assert events == ["serving:False", f"stop:{rust_lifecycle.SERVER_STOP_SECS}"]


def test_launch_or_stop_reports_the_launch_error_when_the_stop_itself_fails(caplog):
    """The stop after a failed launch is best effort, step by step: a server
    that cannot flip its health is still stopped, a server that cannot stop
    is logged, and the launch error is the one that propagates (the entry
    point's exit code depends on its type)."""
    events: list[str] = []

    class Server:
        def __init__(self, *, serving_fails: bool, stop_fails: bool):
            self.serving_fails = serving_fails
            self.stop_fails = stop_fails

        def set_serving(self, serving: bool) -> None:
            events.append(f"serving:{serving}")
            if self.serving_fails:
                raise RuntimeError("health flip failed")

        def stop(self, timeout: float) -> None:
            events.append(f"stop:{timeout}")
            if self.stop_fails:
                raise RuntimeError("stop failed")

    def failing_launch():
        raise ValueError("engine would not start")

    caplog.set_level(logging.ERROR)
    for server in (
        Server(serving_fails=True, stop_fails=False),
        Server(serving_fails=False, stop_fails=True),
        Server(serving_fails=True, stop_fails=True),
    ):
        events.clear()
        with pytest.raises(ValueError, match="engine would not start"):
            asyncio.run(rust_lifecycle.launch_or_stop(server, failing_launch))
        assert events == ["serving:False", f"stop:{rust_lifecycle.SERVER_STOP_SECS}"]
    assert "Failed to mark the servicer as not serving" in caplog.text
    assert "Failed to stop the Rust servicer cleanly" in caplog.text


def test_a_cancelled_launch_task_still_stops_the_server():
    """The stop after a failed launch finishes even when the launch's own task
    is cancelled meanwhile (the entry point's loop shutting down): the port
    is released before the cancellation propagates."""
    stop_entered = threading.Event()
    stop_may_finish = threading.Event()
    events: list[str] = []

    class Server:
        def set_serving(self, serving: bool) -> None:
            events.append(f"serving:{serving}")

        def stop(self, timeout: float) -> None:
            stop_entered.set()
            stop_may_finish.wait(5)
            events.append(f"stop:{timeout}")

    def failing_launch():
        raise RuntimeError("engine would not start")

    async def run():
        task = asyncio.create_task(rust_lifecycle.launch_or_stop(Server(), failing_launch))
        await asyncio.to_thread(stop_entered.wait, 5)
        task.cancel()
        stop_may_finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        events.append("cancelled")

    asyncio.run(run())
    assert events == ["serving:False", f"stop:{rust_lifecycle.SERVER_STOP_SECS}", "cancelled"]
