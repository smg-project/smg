"""Unit tests for the lifecycle helpers the Rust servicer launchers share."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
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
    except FileNotFoundError:
        return False


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
