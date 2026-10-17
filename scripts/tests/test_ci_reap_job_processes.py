"""Tests for scripts/ci_reap_job_processes.py: the teardown that ends what a
job started, by the marker in the processes' environment. Against a fake
``/proc`` tree (what is matched, what is left alone, the signals in order) and
against real child processes."""

from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "ci_reap_job_processes.py"
MARKER = "e2e-123-1-runner-a"


def _load():
    spec = importlib.util.spec_from_file_location("ci_reap_job_processes", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["ci_reap_job_processes"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


def _proc(
    root: Path,
    pid: int,
    *,
    ppid: int,
    environ: dict[str, str] | None,
    state: str = "S",
    command: str = "python -m worker",
) -> None:
    """One process of a fake ``/proc``: its ``stat`` (comm with a space and
    parentheses, as a real one may have), ``cmdline`` and ``environ``; no
    ``environ`` file stands for another user's process."""
    directory = root / str(pid)
    directory.mkdir()
    (directory / "stat").write_text(
        f"{pid} (py (thon)) {state} {ppid} {pid} {pid} 0 -1 4194560 0\n"
    )
    (directory / "cmdline").write_bytes(command.replace(" ", "\0").encode() + b"\0")
    if environ is not None:
        entries = (f"{key}={value}".encode() + b"\0" for key, value in environ.items())
        (directory / "environ").write_bytes(b"".join(entries))


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A runner at the end of a job: the teardown step's shell and the script
    itself carry the marker, so does a worker left behind with its child and
    a zombie, another job's worker carries its own, a lookalike value and an
    unreadable process carry nothing that counts."""
    root = tmp_path / "proc"
    root.mkdir()
    (root / "self").mkdir()  # the non-pid entries of /proc are passed over
    (root / "cpuinfo").write_text("")
    _proc(root, 1, ppid=0, environ={}, command="init")
    _proc(root, 50, ppid=1, environ={}, command="Runner.Worker")
    _proc(root, 60, ppid=50, environ={"E2E_JOB_MARKER": MARKER}, command="bash step.sh")
    _proc(root, 61, ppid=60, environ={"E2E_JOB_MARKER": MARKER}, command="python3 reap.py")
    _proc(root, 100, ppid=1, environ={"HOME": "/h", "E2E_JOB_MARKER": MARKER}, command="worker")
    _proc(root, 101, ppid=100, environ={"E2E_JOB_MARKER": MARKER}, command="engine core")
    _proc(root, 102, ppid=100, environ={"E2E_JOB_MARKER": MARKER}, state="Z", command="defunct")
    _proc(root, 200, ppid=1, environ={"E2E_JOB_MARKER": "e2e-123-1-runner-b"}, command="other")
    _proc(root, 300, ppid=1, environ={"NOTE": f"E2E_JOB_MARKER={MARKER}"}, command="lookalike")
    _proc(root, 400, ppid=1, environ=None, command="not readable")
    return root


def test_the_marked_processes_are_the_live_ones_carrying_exactly_this_jobs_marker(mod, tree):
    found = mod.marked_processes(MARKER, proc_root=str(tree))
    assert [process.pid for process in found] == [60, 61, 100, 101]
    assert found[2] == mod.Process(pid=100, ppid=1, command="worker")
    # The script's own chain up to the runner is what the teardown leaves alone.
    assert mod.ancestors(str(tree), 61) == {61, 60, 50}
    ours = mod.marked_processes(MARKER, proc_root=str(tree), exclude=mod.ancestors(str(tree), 61))
    assert [process.pid for process in ours] == [100, 101]


def test_a_dry_run_lists_the_leftovers_and_ends_nothing(mod, tree):
    sent: list[tuple[int, int]] = []
    lines: list[str] = []
    found = mod.reap(
        MARKER,
        proc_root=str(tree),
        dry_run=True,
        kill=lambda pid, sig: sent.append((pid, sig)),
        self_pid=61,
        log=lines.append,
    )
    assert [process.pid for process in found] == [100, 101]
    assert sent == []
    assert lines == [
        "would end pid 100 (parent 1): worker",
        "would end pid 101 (parent 100): engine core",
    ]


def test_the_teardown_terminates_then_kills_what_ignores_the_term(mod, tree):
    """Both leftovers get SIGTERM; the child exits on it, the worker ignores it
    and gets SIGKILL once the grace is up; the shell, the script, the other
    job's worker and the zombie are never signalled."""
    sent: list[tuple[int, int]] = []

    def kill(pid: int, sig: int) -> None:
        sent.append((pid, sig))
        if (pid, sig) == (101, signal.SIGTERM) or sig == signal.SIGKILL:
            for name in ("stat", "cmdline", "environ"):
                (tree / str(pid) / name).unlink()
            (tree / str(pid)).rmdir()

    lines: list[str] = []
    found = mod.reap(
        MARKER,
        proc_root=str(tree),
        grace=0.4,
        kill=kill,
        sleep=lambda seconds: None,
        self_pid=61,
        log=lines.append,
    )
    assert [process.pid for process in found] == [100, 101]
    assert sent == [(100, signal.SIGTERM), (101, signal.SIGTERM), (100, signal.SIGKILL)]
    assert "still alive after 0.4s: [100]; sending SIGKILL" in lines
    assert lines[-1] == "ended 2 of 2 process(es) of E2E_JOB_MARKER=" + MARKER
    assert not (tree / "100").exists() and not (tree / "101").exists()
    assert (tree / "60").exists() and (tree / "200").exists() and (tree / "102").exists()


def test_a_survivor_of_the_kill_is_reported_as_a_warning(mod, tree):
    lines: list[str] = []
    mod.reap(
        MARKER,
        proc_root=str(tree),
        grace=0.2,
        kill=lambda pid, sig: None,
        sleep=lambda seconds: None,
        self_pid=61,
        log=lines.append,
    )
    assert "::warning::survived SIGKILL: [100, 101]" in lines
    assert lines[-1] == "ended 0 of 2 process(es) of E2E_JOB_MARKER=" + MARKER


def test_nothing_to_end_is_said_so(mod, tree):
    lines: list[str] = []
    assert mod.reap("e2e-nobody", proc_root=str(tree), self_pid=61, log=lines.append) == []
    assert lines == ["no process carries E2E_JOB_MARKER=e2e-nobody"]


def test_the_command_line_takes_the_marker_and_refuses_an_empty_one(mod, tree, capsys):
    assert mod.main(["--dry-run", "--proc-root", str(tree), MARKER]) == 0
    out = capsys.readouterr().out
    assert "would end pid 100 (parent 1): worker" in out and "pid 200" not in out
    with pytest.raises(SystemExit) as refused:
        mod.main(["--proc-root", str(tree), " "])
    assert refused.value.code == 2


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="reads the real /proc")
def test_real_children_carrying_the_marker_are_ended_and_others_are_not(mod):
    marker = f"e2e-test-{os.getpid()}-{time.time_ns()}"
    marked = [
        subprocess.Popen(["sleep", "300"], env={**os.environ, "E2E_JOB_MARKER": marker})
        for _ in range(2)
    ]
    other = subprocess.Popen(["sleep", "300"], env={**os.environ, "E2E_JOB_MARKER": marker + "-x"})
    try:
        lines: list[str] = []
        found = mod.reap(marker, grace=5.0, log=lines.append)
        assert sorted(process.pid for process in found) == sorted(p.pid for p in marked)
        for process in marked:
            assert process.wait(timeout=5) == -signal.SIGTERM
        assert other.poll() is None
        assert lines[-1] == f"ended 2 of 2 process(es) of E2E_JOB_MARKER={marker}"
    finally:
        other.kill()
        other.wait(timeout=5)
        for process in marked:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
