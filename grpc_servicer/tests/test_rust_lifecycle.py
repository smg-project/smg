"""Unit tests for the lifecycle helpers the Rust servicer launchers share."""

from __future__ import annotations

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
