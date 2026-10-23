"""CPU-only checks of the harness's handling of an engine that dies under its
launcher: which processes are the engine's own, the teardown a worker runs
once its launcher has already exited, and the kill that deals the fault."""

from __future__ import annotations

import io
import signal
import subprocess
from pathlib import Path

import pytest
from infra import process_utils
from infra import worker as worker_module
from infra.constants import ConnectionMode
from infra.process_utils import live_child_processes, live_process_group_members
from infra.worker import Worker

# A launcher (100) leading its own session, its engine core (101) and a
# sidecar (104) as direct children, the core's rank worker (102) one level
# down, a zombie child awaiting reaping (103) and an unrelated process (200).
_TREE = {
    100: ("python3", "S", 1, 100),
    101: ("VLLM::EngineCore", "R", 100, 100),
    102: ("VLLM::Worker_TP0", "R", 101, 100),
    103: ("python3", "Z", 100, 100),
    104: ("mm (side) car", "S", 100, 100),
    200: ("bash", "S", 1, 200),
}


@pytest.fixture
def proc_root(tmp_path: Path) -> str:
    for pid, (comm, state, ppid, pgrp) in _TREE.items():
        entry = tmp_path / str(pid)
        entry.mkdir()
        # The first fields of /proc/<pid>/stat: pid (comm) state ppid pgrp session ...
        (entry / "stat").write_text(f"{pid} ({comm}) {state} {ppid} {pgrp} {pgrp} 0 -1 4194560 0\n")
    (tmp_path / "self").mkdir()
    (tmp_path / "cpuinfo").write_text("processor : 0\n")
    return str(tmp_path)


def test_the_engines_processes_are_the_launchers_live_children(proc_root):
    assert sorted(live_child_processes(100, proc_root)) == [101, 104]
    assert live_child_processes(101, proc_root) == [102]
    assert live_child_processes(102, proc_root) == []
    assert live_child_processes(999, proc_root) == []


def test_the_group_is_every_live_process_the_launcher_leads(proc_root):
    assert sorted(live_process_group_members(100, proc_root)) == [100, 101, 102, 104]
    assert live_process_group_members(200, proc_root) == [200]
    assert live_process_group_members(300, proc_root) == []


def test_without_a_proc_tree_no_process_counts_as_a_child(tmp_path):
    assert live_child_processes(100, str(tmp_path / "absent")) == []


class _FakeLauncher:
    """The ``Popen`` of a launcher: alive until ``returncode`` is set."""

    def __init__(self, pid: int, returncode: int | None = None):
        self.pid = pid
        self.returncode = returncode
        self.signals: list[str] = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("launcher", timeout or 0)
        return self.returncode

    def terminate(self):
        self.signals.append("terminate")

    def kill(self):
        self.signals.append("kill")


@pytest.fixture
def teardown(monkeypatch):
    """The teardown's side effects, recorded instead of run."""
    calls: list[tuple] = []
    monkeypatch.setattr(
        worker_module,
        "wait_for_process_group_exit",
        lambda pgid, timeout, kill_after: calls.append(("group", pgid)) or [],
    )
    monkeypatch.setattr(
        worker_module,
        "wait_for_gpu_memory_release",
        lambda gpu_ids, baseline, timeout: calls.append(("memory", baseline)) or {},
    )
    monkeypatch.setattr(
        worker_module.os, "killpg", lambda pgid, sig: calls.append(("killpg", pgid, sig))
    )
    monkeypatch.setattr(process_utils, "release_port", lambda port: calls.append(("port", port)))
    return calls


def _started_worker(launcher: _FakeLauncher) -> Worker:
    worker = Worker(model_id="m", engine="vllm", port=50222, gpu_ids=[0], mode=ConnectionMode.GRPC)
    worker.process = launcher  # type: ignore[assignment]
    worker._pgid = launcher.pid
    worker._gpu_mem_baseline = {0: 10}
    worker._log_file = io.StringIO()
    return worker


def test_a_launcher_that_exited_on_its_own_is_still_torn_down_once(teardown):
    launcher = _FakeLauncher(4242, returncode=1)
    worker = _started_worker(launcher)

    worker.stop()

    assert teardown == [("group", 4242), ("memory", {0: 10}), ("port", 50222)]
    assert launcher.signals == []  # nothing left to signal
    assert worker._log_file is None
    assert worker._pgid is None

    teardown.clear()
    worker.stop()
    assert teardown == []  # the second stop finds nothing to do


def test_a_live_launcher_is_signalled_by_its_group_then_torn_down(teardown, monkeypatch):
    launcher = _FakeLauncher(4242)
    worker = _started_worker(launcher)
    # The group's SIGTERM ends the launcher; ``wait`` then returns at once.
    monkeypatch.setattr(
        worker_module.os,
        "killpg",
        lambda pgid, sig: (
            teardown.append(("killpg", pgid, sig)),
            setattr(launcher, "returncode", 0),
        ),
    )

    worker.stop()

    assert teardown == [
        ("killpg", 4242, signal.SIGTERM),
        ("group", 4242),
        ("memory", {0: 10}),
        ("port", 50222),
    ]
    assert launcher.signals == []


def test_kill_engine_signals_the_launchers_children_and_leaves_the_launcher(monkeypatch):
    launcher = _FakeLauncher(4242)
    worker = _started_worker(launcher)
    sent: list[tuple[int, int]] = []

    def kill(pid, sig):
        if pid == 4244:
            raise ProcessLookupError  # exited between the scan and the signal
        sent.append((pid, sig))

    monkeypatch.setattr(worker_module, "live_child_processes", lambda pid: [4243, 4244, 4245])
    monkeypatch.setattr(worker_module.os, "kill", kill)

    assert worker.kill_engine() == [4243, 4245]
    assert sent == [(4243, signal.SIGKILL), (4245, signal.SIGKILL)]
    assert launcher.signals == []

    launcher.returncode = 1
    assert worker.kill_engine() == []
