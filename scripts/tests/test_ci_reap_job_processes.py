"""Tests for scripts/ci_reap_job_processes.py: the teardown that ends what a
job started, by the marker in the processes' environment. Against a fake
``/proc`` tree (what is matched, what is left alone, the signals in order) and
against real child processes."""

from __future__ import annotations

import importlib.util
import os
import select
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


def test_teardown_never_signals_init_with_an_inherited_job_marker(mod, tree):
    (tree / "1" / "environ").write_bytes(f"E2E_JOB_MARKER={MARKER}\0".encode())
    assert 1 in [process.pid for process in mod.marked_processes(MARKER, proc_root=str(tree))]
    sent: list[tuple[int, int]] = []

    def kill(pid: int, sig: int) -> None:
        sent.append((pid, sig))
        if (pid, sig) in ((101, signal.SIGTERM), (100, signal.SIGKILL)):
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
    assert all(pid != 1 for pid, _ in sent)
    assert not (tree / "100").exists() and not (tree / "101").exists()
    assert mod.alive(str(tree), 1)
    assert (tree / "1" / "environ").read_bytes() == f"E2E_JOB_MARKER={MARKER}\0".encode()
    assert all((tree / str(pid)).exists() for pid in (50, 60, 61, 102, 200, 300, 400))
    assert lines[-1] == "ended 2 of 2 process(es) of E2E_JOB_MARKER=" + MARKER


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
def test_real_children_carrying_the_marker_are_ended_and_others_are_not(mod, tmp_path):
    marker = f"e2e-test-{os.getpid()}-{time.time_ns()}"
    marked = [
        subprocess.Popen(["sleep", "300"], env={**os.environ, "E2E_JOB_MARKER": marker})
        for _ in range(2)
    ]
    other = subprocess.Popen(["sleep", "300"], env={**os.environ, "E2E_JOB_MARKER": marker + "-x"})
    try:
        view = tmp_path / "legacy-owned-proc-view"
        view.mkdir()
        pids = {1, *mod.ancestors("/proc", os.getpid()), *(p.pid for p in marked), other.pid}
        for pid in pids:
            (view / str(pid)).symlink_to(Path("/proc") / str(pid), target_is_directory=True)
        lines: list[str] = []
        found = mod.reap(marker, proc_root=str(view), grace=5.0, log=lines.append)
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


STRICT_UID = 31415


def _strict_process(root, pid, *, ppid=1, uid=STRICT_UID, birth=100, marker=MARKER):
    directory = root / str(pid)
    directory.mkdir(exist_ok=True)
    fields = ["S", str(ppid), *(["0"] * 17), str(birth)]
    (directory / "stat").write_text(f"{pid} (worker) {' '.join(fields)}\n")
    (directory / "status").write_text(f"Uid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
    (directory / "environ").write_bytes(f"E2E_JOB_MARKER={marker}\0".encode())


@pytest.fixture
def strict_tree(tmp_path):
    root = tmp_path / "strict-proc"
    root.mkdir()
    for pid, ppid in [(1, 0), (50, 1), (60, 50), (61, 60), (100, 1), (101, 100)]:
        _strict_process(root, pid, ppid=ppid, birth=1000 + pid)
    _strict_process(root, 200, marker=MARKER + "-other", birth=1200)
    _strict_process(root, 300, uid=STRICT_UID + 1, birth=1300)
    # Clearly foreign users are excluded without reading their environments.
    (root / "300" / "environ").unlink()
    return root


class _Pidfds:
    def __init__(self, root):
        self.root = root
        self.opened = {}
        self.closed = set()
        self.sent = []
        self.now = 0.0
        self.on_open = lambda pid: None
        self.on_signal = lambda pid, sig: self.remove(pid)
        self.on_sleep = lambda: None

    def remove(self, pid):
        directory = self.root / str(pid)
        if directory.exists():
            for child in directory.iterdir():
                child.unlink()
            directory.rmdir()

    def open(self, pid, flags):
        assert flags == 0
        fd = 1000 + len(self.opened)
        self.opened[fd] = pid
        self.on_open(pid)
        return fd

    def send(self, fd, sig, info, flags):
        assert fd in self.opened and fd not in self.closed
        assert info is None and flags == 0
        pid = self.opened[fd]
        self.sent.append((pid, sig))
        self.on_signal(pid, sig)

    def close(self, fd):
        assert fd in self.opened and fd not in self.closed
        self.closed.add(fd)

    def exited(self, fd):
        assert fd in self.opened and fd not in self.closed
        return not (self.root / str(self.opened[fd])).exists()

    def sleep(self, seconds):
        self.now += seconds
        self.on_sleep()

    def run(self, mod, **overrides):
        kwargs = dict(
            proc_root=str(self.root),
            grace=0.4,
            pidfd_open=self.open,
            pidfd_send_signal=self.send,
            pidfd_exited=self.exited,
            close=self.close,
            sleep=self.sleep,
            clock=lambda: self.now,
            self_pid=61,
            uid=STRICT_UID,
            log=lambda message: None,
        )
        kwargs.update(overrides)
        return mod.reap_strict(MARKER, **kwargs)


def test_strict_ends_only_bound_owned_processes_and_closes_all_handles(mod, strict_tree):
    runtime = _Pidfds(strict_tree)

    def signal_owned(pid, sig):
        if pid == 101 or sig == signal.SIGKILL:
            runtime.remove(pid)

    runtime.on_signal = signal_owned
    assert runtime.run(mod)
    assert runtime.sent == [(100, signal.SIGTERM), (101, signal.SIGTERM), (100, signal.SIGKILL)]
    assert runtime.closed == set(runtime.opened)
    assert all((strict_tree / str(pid)).exists() for pid in (1, 50, 60, 61, 200, 300))


@pytest.mark.parametrize("change", ["birth", "marker", "uid"])
def test_strict_rechecks_identity_after_open_before_any_signal(mod, strict_tree, change):
    runtime = _Pidfds(strict_tree)

    def replace(pid):
        if pid != 100:
            return
        _strict_process(
            strict_tree,
            pid,
            birth=9999 if change == "birth" else 1100,
            marker=MARKER + "-replacement" if change == "marker" else MARKER,
            uid=STRICT_UID + 1 if change == "uid" else STRICT_UID,
        )

    runtime.on_open = replace
    assert not runtime.run(mod)
    # Reject the changed identity while still cleaning independently verified peers.
    assert all(pid != 100 for pid, _ in runtime.sent)
    assert (101, signal.SIGTERM) in runtime.sent
    assert runtime.closed == set(runtime.opened)


def test_strict_reused_pid_does_not_abort_verified_peers_kill_grace(mod, strict_tree):
    runtime = _Pidfds(strict_tree)
    timed = []

    def replace(pid):
        if pid == 100:
            _strict_process(strict_tree, 100, birth=9999)

    def on_signal(pid, sig):
        timed.append((pid, sig, runtime.now))
        if pid == 101 and sig == signal.SIGKILL:
            runtime.remove(pid)

    runtime.on_open = replace
    runtime.on_signal = on_signal
    assert not runtime.run(mod)
    assert all(pid != 100 for pid, _, _ in timed)
    assert any(pid == 101 and sig == signal.SIGKILL and t >= 0.4 for pid, sig, t in timed)
    assert runtime.closed == set(runtime.opened)


@pytest.mark.parametrize("missing", ["pidfd_open", "pidfd_send_signal"])
def test_strict_missing_pidfd_api_has_no_numeric_fallback(mod, strict_tree, monkeypatch, missing):
    runtime = _Pidfds(strict_tree)
    target = mod.os if missing == "pidfd_open" else mod.signal
    monkeypatch.setattr(target, missing, None, raising=False)
    numeric = []
    monkeypatch.setattr(mod.os, "kill", lambda *args: numeric.append(args))
    monkeypatch.setattr(mod, "reap", lambda *args, **kwargs: pytest.fail("legacy fallback"))
    assert not runtime.run(mod, **{missing: None})
    assert runtime.sent == [] and runtime.opened == {} and numeric == []


def test_strict_changed_marker_after_term_is_not_killed_or_reported_clean(mod, strict_tree):
    runtime = _Pidfds(strict_tree)

    def signal_owned(pid, sig):
        if pid == 100 and sig == signal.SIGTERM:
            (strict_tree / "100" / "environ").write_bytes(b"E2E_JOB_MARKER=other\0")
        else:
            runtime.remove(pid)

    runtime.on_signal = signal_owned
    assert not runtime.run(mod)
    assert (100, signal.SIGKILL) not in runtime.sent
    assert runtime.closed == set(runtime.opened)


def test_strict_bound_exit_during_ownership_read_keeps_cleaning_peers(
    mod, strict_tree, monkeypatch
):
    runtime = _Pidfds(strict_tree)
    original_owned = mod._strict_owned
    original_exited = runtime.exited
    ready = set()

    def exit_during_read(proc_root, pid, *args, **kwargs):
        if pid == 100 and pid in runtime.opened.values():
            ready.add(pid)
            stat = strict_tree / str(pid) / "stat"
            stat.write_bytes(stat.read_bytes().replace(b") S ", b") Z "))
            return None
        return original_owned(proc_root, pid, *args, **kwargs)

    monkeypatch.setattr(mod, "_strict_owned", exit_during_read)
    assert runtime.run(
        mod, pidfd_exited=lambda fd: runtime.opened[fd] in ready or original_exited(fd)
    )
    assert (strict_tree / "100" / "stat").exists()  # Unreaped Z can remain visible.
    assert all(pid != 100 for pid, _ in runtime.sent)
    assert (101, signal.SIGTERM) in runtime.sent
    assert runtime.closed == set(runtime.opened)


@pytest.mark.parametrize("transition", ["zombie", "empty", "permission"])
def test_strict_transient_exit_observation_does_not_abort_other_worker(
    mod, strict_tree, monkeypatch, transition
):
    runtime = _Pidfds(strict_tree)
    original_exited = runtime.exited
    original_open = open
    transitioning = set()
    ready = set()
    signal_times = []

    def write_state(state):
        stat = strict_tree / "100" / "stat"
        text = stat.read_bytes()
        split = text.rfind(b")") + 2
        stat.write_bytes(text[:split] + state + text[split + 1 :])

    def proc_open(path, *args, **kwargs):
        if (
            transition == "permission"
            and 100 in transitioning
            and str(path) == str(strict_tree / "100" / "environ")
        ):
            raise PermissionError("exit in progress")
        return original_open(path, *args, **kwargs)

    def on_signal(pid, sig):
        signal_times.append((pid, sig, runtime.now))
        if pid == 100:
            transitioning.add(pid)
            write_state(b"Z" if transition == "zombie" else b"D")
            if transition == "empty":
                (strict_tree / "100" / "environ").write_bytes(b"")
        elif sig == signal.SIGKILL:
            runtime.remove(pid)

    def finish_group():
        if runtime.now >= 1.0:
            ready.add(100)
            write_state(b"Z")

    monkeypatch.setattr("builtins.open", proc_open)
    runtime.on_signal = on_signal
    runtime.on_sleep = finish_group
    assert runtime.run(
        mod, pidfd_exited=lambda fd: runtime.opened[fd] in ready or original_exited(fd)
    )
    assert (100, signal.SIGKILL) not in runtime.sent
    assert any(pid == 101 and sig == signal.SIGKILL and t >= 0.4 for pid, sig, t in signal_times)
    assert runtime.closed == set(runtime.opened)


def test_strict_live_empty_environment_is_not_a_negative_marker_observation(mod, strict_tree):
    runtime = _Pidfds(strict_tree)
    (strict_tree / "100" / "environ").write_bytes(b"")
    assert not runtime.run(mod)
    assert all(pid != 100 for pid, _ in runtime.sent)
    assert (101, signal.SIGTERM) in runtime.sent


def test_strict_survivors_are_failure_after_bounded_term_and_kill(mod, strict_tree):
    runtime = _Pidfds(strict_tree)
    runtime.on_signal = lambda pid, sig: None
    assert not runtime.run(mod)
    assert runtime.sent == [
        (100, signal.SIGTERM),
        (101, signal.SIGTERM),
        (100, signal.SIGKILL),
        (101, signal.SIGKILL),
    ]
    assert runtime.now <= 20.4 + 1e-9
    assert runtime.closed == set(runtime.opened)


def test_strict_complete_rescan_finds_a_late_reparented_child(mod, strict_tree):
    runtime = _Pidfds(strict_tree)

    def signal_owned(pid, sig):
        if pid == 100:
            _strict_process(strict_tree, 102, ppid=1, birth=1102)
        runtime.remove(pid)

    runtime.on_signal = signal_owned
    assert runtime.run(mod)
    assert (102, signal.SIGTERM) in runtime.sent
    assert runtime.closed == set(runtime.opened)


@pytest.mark.parametrize("file", ["environ", "status", "stat"])
def test_strict_incomplete_owned_observation_is_failure_without_signal(mod, strict_tree, file):
    runtime = _Pidfds(strict_tree)
    path = strict_tree / "100" / file
    if file == "environ":
        path.unlink()  # The PID/stat remains: this is not process disappearance.
    else:
        path.write_text("malformed\n")
    assert not runtime.run(mod)
    assert all(pid != 100 for pid, _ in runtime.sent)
    assert (101, signal.SIGTERM) in runtime.sent
    assert runtime.closed == set(runtime.opened)


def test_strict_reused_numeric_pid_during_rescan_is_not_retargeted(mod, strict_tree):
    runtime = _Pidfds(strict_tree)

    def signal_owned(pid, sig):
        runtime.remove(pid)
        if pid == 100:
            _strict_process(strict_tree, 100, birth=9999)

    runtime.on_signal = signal_owned
    assert not runtime.run(mod)
    assert runtime.sent.count((100, signal.SIGTERM)) == 1
    assert (100, signal.SIGKILL) not in runtime.sent
    assert runtime.closed == set(runtime.opened)


def test_strict_ancestor_observation_failure_never_signals(mod, strict_tree):
    runtime = _Pidfds(strict_tree)
    (strict_tree / "60" / "stat").write_text("malformed\n")
    assert not runtime.run(mod)
    assert runtime.sent == [] and runtime.opened == {}


def test_strict_cli_rejects_fake_proc_and_dry_run(mod, strict_tree):
    for args in [
        ["--strict", "--proc-root", str(strict_tree), MARKER],
        ["--strict", "--dry-run", MARKER],
    ]:
        with pytest.raises(SystemExit) as refused:
            mod.main(args)
        assert refused.value.code == 2


def test_strict_failure_is_nonzero_while_legacy_default_still_reports(mod, monkeypatch):
    monkeypatch.setattr(mod, "reap_strict", lambda *args, **kwargs: False)
    monkeypatch.setattr(mod, "reap", lambda *args, **kwargs: [])
    assert mod.main(["--strict", MARKER]) == 1
    assert mod.main([MARKER]) == 0


@pytest.mark.parametrize("leaf", ["stat", "status", "environ"])
def test_strict_open_read_exit_race_keeps_cleaning_other_workers(
    mod, strict_tree, monkeypatch, leaf
):
    import builtins

    runtime = _Pidfds(strict_tree)
    original = builtins.open
    target = strict_tree / "100" / leaf
    triggered = []

    class ReapedAfterOpen:
        def __enter__(self):
            return self

        def read(self, *args):
            triggered.append(True)
            runtime.remove(100)
            raise ProcessLookupError(3, "reaped after opening procfs file")

        def __exit__(self, *args):
            return False

    def opened(path, *args, **kwargs):
        if os.fspath(path) == str(target) and not triggered:
            return ReapedAfterOpen()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", opened)
    assert runtime.run(mod)
    assert triggered and (101, signal.SIGTERM) in runtime.sent
    assert all(pid != 100 for pid, _ in runtime.sent)
    assert runtime.closed == set(runtime.opened)


def test_strict_esrch_for_a_still_live_ownership_file_fails_closed(mod, strict_tree, monkeypatch):
    import builtins

    original = builtins.open

    def opened(path, *args, **kwargs):
        if os.fspath(path) == str(strict_tree / "100" / "status"):
            raise ProcessLookupError(3, "identity is unresolved, not confirmed gone")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", opened)
    runtime = _Pidfds(strict_tree)
    assert not runtime.run(mod)
    assert all(pid != 100 for pid, _ in runtime.sent)
    assert (101, signal.SIGTERM) in runtime.sent


def test_strict_raw_comm_bytes_do_not_prevent_uid_and_marker_checks(mod, strict_tree):
    for pid in (100, 300):
        p = strict_tree / str(pid) / "stat"
        p.write_bytes(p.read_bytes().replace(b"(worker)", b"(raw\xff\xfe)"))
        p = strict_tree / str(pid) / "status"
        p.write_bytes(b"Name:\traw\xff\xfe\n" + p.read_bytes())
    runtime = _Pidfds(strict_tree)
    assert runtime.run(mod)
    assert (100, signal.SIGTERM) in runtime.sent
    assert all(pid != 300 for pid, _ in runtime.sent)


def test_strict_fork_chain_cannot_turn_incomplete_scans_into_quiet_success(
    mod, strict_tree, monkeypatch
):
    runtime = _Pidfds(strict_tree)
    runtime.remove(101)
    original = mod._strict_owned
    transitions = []

    def after_enumeration(root, pid, uid, entry, **kwargs):
        if pid in (100, 102) and pid not in transitions:
            runtime.remove(pid)
            _strict_process(strict_tree, pid + 2 if pid == 100 else 103, birth=2000 + pid)
            transitions.append(pid)
        return original(root, pid, uid, entry, **kwargs)

    monkeypatch.setattr(mod, "_strict_owned", after_enumeration)
    assert runtime.run(mod)
    assert transitions == [100, 102]
    assert (103, signal.SIGTERM) in runtime.sent
    assert not (strict_tree / "103").exists()


def test_strict_final_census_cannot_certify_success_after_deadline(mod, strict_tree, monkeypatch):
    runtime = _Pidfds(strict_tree)
    runtime.remove(100)
    runtime.remove(101)
    original = mod._strict_census
    calls = []

    def slow_final_census(root, on_unresolved):
        found = original(root, on_unresolved)
        calls.append(True)
        if len(calls) == 4:
            runtime.now = 20.5
        return found

    monkeypatch.setattr(mod, "_strict_census", slow_final_census)
    assert not runtime.run(mod)
    assert len(calls) == 4 and runtime.sent == []


def test_strict_late_worker_gets_its_own_term_grace(mod, strict_tree):
    runtime = _Pidfds(strict_tree)
    timed = []

    def on_signal(pid, sig):
        timed.append((pid, sig, runtime.now))
        if pid == 100 and sig == signal.SIGKILL:
            _strict_process(strict_tree, 102, birth=1102)
        if pid == 101 or sig == signal.SIGKILL:
            runtime.remove(pid)

    runtime.on_signal = on_signal
    assert runtime.run(mod)
    term = next(t for pid, sig, t in timed if pid == 102 and sig == signal.SIGTERM)
    kill = next(t for pid, sig, t in timed if pid == 102 and sig == signal.SIGKILL)
    assert term >= 0.4 and kill - term >= 0.4 - 1e-9


def test_strict_fixed_deadline_does_not_shorten_a_late_workers_grace(mod, strict_tree):
    runtime = _Pidfds(strict_tree)
    runtime.on_signal = lambda pid, sig: None

    def late_birth():
        if runtime.now >= 21.0 and not (strict_tree / "102").exists():
            _strict_process(strict_tree, 102, birth=1102)

    runtime.on_sleep = late_birth
    assert not runtime.run(mod, grace=2.0)
    assert (102, signal.SIGTERM) in runtime.sent
    assert (102, signal.SIGKILL) not in runtime.sent
    assert runtime.now <= 22.0 + 1e-9


@pytest.mark.skipif(
    not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"),
    reason="requires Linux process pidfds",
)
def test_strict_zombie_main_thread_cannot_hide_live_threads(mod, tmp_path):
    child = None
    fd = None
    marker = f"e2e-zombie-main-{os.getpid()}-{time.time_ns()}"
    try:
        child = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                "import ctypes,threading,time; "
                "threading.Thread(target=lambda:time.sleep(30)).start(); "
                "print('READY',flush=True); ctypes.CDLL(None).pthread_exit(None)",
            ],
            env={
                **{key: os.environ[key] for key in ("LD_LIBRARY_PATH",) if key in os.environ},
                "PATH": "/usr/bin:/bin",
                "E2E_JOB_MARKER": marker,
            },
            stdout=subprocess.PIPE,
        )
        fd = os.pidfd_open(child.pid, 0)
        ready, _, _ = select.select([child.stdout], [], [], 5)
        assert ready and child.stdout.readline() == b"READY\n"
        limit = time.monotonic() + 5
        while mod._strict_stat("/proc", child.pid)[0] != "Z" and time.monotonic() < limit:
            time.sleep(0.01)
        assert mod._strict_stat("/proc", child.pid)[0] == "Z"
        assert not mod._pidfd_exited(fd)
        view = tmp_path / "owned-proc"
        view.mkdir()
        for pid in set(mod._strict_ancestors("/proc", os.getpid())) | {child.pid}:
            (view / str(pid)).symlink_to(Path("/proc") / str(pid), target_is_directory=True)
        assert not mod.reap_strict(marker, proc_root=str(view), grace=0.1)
        assert not mod._pidfd_exited(fd)
    finally:
        if fd is not None:
            try:
                signal.pidfd_send_signal(fd, signal.SIGKILL)
            except ProcessLookupError:
                pass
            finally:
                os.close(fd)
        elif child is not None and child.poll() is None:
            child.kill()
        if child is not None:
            child.wait(timeout=5)
            if child.stdout is not None:
                child.stdout.close()


@pytest.mark.skipif(
    not callable(getattr(os, "pidfd_open", None))
    or not callable(getattr(signal, "pidfd_send_signal", None)),
    reason="Linux pidfd APIs are unavailable; strict refusal is tested separately",
)
def test_real_pidfd_cleanup_within_explicit_ownership_view(mod, tmp_path):
    """Real pidfds and proc identities; this does not qualify a full-host scan."""
    marker = f"e2e-strict-{os.getpid()}-{time.time_ns()}"
    view = tmp_path / "real-proc-view"
    view.mkdir()
    children = []
    try:
        owned = subprocess.Popen(["sleep", "300"], env={**os.environ, "E2E_JOB_MARKER": marker})
        children.append(owned)
        other = subprocess.Popen(
            ["sleep", "300"], env={**os.environ, "E2E_JOB_MARKER": marker + "-other"}
        )
        children.append(other)
        protected = mod._strict_ancestors("/proc", os.getpid())
        assert os.getpid() in protected
        for pid in {1, *protected, owned.pid, other.pid}:
            target = Path("/proc") / str(pid)
            link = view / str(pid)
            link.symlink_to(target, target_is_directory=True)
            assert link.is_symlink() and link.resolve(strict=True) == target.resolve(strict=True)
        assert mod._strict_ancestors(str(view), os.getpid()) == protected
        owned_before = mod._strict_stat("/proc", owned.pid)
        other_before = mod._strict_stat("/proc", other.pid)
        assert owned_before is not None and other_before is not None
        lines = []
        assert mod.reap_strict(marker, proc_root=str(view), grace=2.0, log=lines.append)
        assert owned.wait(timeout=5) == -signal.SIGTERM
        assert other.poll() is None
        other_after = mod._strict_stat("/proc", other.pid)
        assert other_after is not None and other_after[2] == other_before[2]
        assert mod._strict_ancestors(str(view), os.getpid()) == protected
        identity = f"pid {owned.pid} uid {os.getuid()} birth {owned_before[2]}"
        assert any(identity in line and "strict signal" in line for line in lines)
    finally:
        for process in children:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


def test_strict_same_uid_unmarked_unreadable_process_holds_without_signal(
    mod, strict_tree, monkeypatch
):
    runtime = _Pidfds(strict_tree)
    _strict_process(strict_tree, 400, birth=1400)
    unreadable = strict_tree / "400" / "environ"
    unreadable.write_bytes(b"HOME=/unmarked\0")
    assert b"E2E_JOB_MARKER=" not in unreadable.read_bytes()
    real_open = open
    denied = []

    def deny_unmarked_environment(file, *args, **kwargs):
        if file == str(unreadable):
            denied.append(file)
            raise PermissionError(13, "Permission denied", file)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", deny_unmarked_environment)
    lines = []
    assert not runtime.run(mod, log=lines.append)
    assert denied and set(denied) == {str(unreadable)}
    assert all(pid != 400 for pid, _ in runtime.sent)
    assert {(100, signal.SIGTERM), (101, signal.SIGTERM)} <= set(runtime.sent)
    assert runtime.closed == set(runtime.opened)
    assert runtime.now <= 20.4 + 1e-9
    assert any("PermissionError" in line and str(unreadable) in line for line in lines)


def test_strict_allows_owned_exit_fifteen_seconds_after_kill(mod, strict_tree):
    runtime = _Pidfds(strict_tree)
    killed_at = {}

    def on_signal(pid, sig):
        if sig == signal.SIGKILL:
            killed_at[pid] = runtime.now

    def on_sleep():
        for pid, killed in list(killed_at.items()):
            if runtime.now - killed >= 15.0:
                runtime.remove(pid)

    runtime.on_signal = on_signal
    runtime.on_sleep = on_sleep
    assert runtime.run(mod, grace=10.0)
    assert 25.0 <= runtime.now < 30.0
    assert runtime.closed == set(runtime.opened)
    assert all(runtime.sent.count((pid, signal.SIGKILL)) == 1 for pid in (100, 101))


def test_strict_survivor_after_twenty_second_kill_wait_still_fails(mod, strict_tree):
    runtime = _Pidfds(strict_tree)
    runtime.on_signal = lambda pid, sig: None
    assert not runtime.run(mod, grace=10.0)
    assert 30.0 <= runtime.now <= 30.001
    assert runtime.closed == set(runtime.opened)
    assert all(runtime.sent.count((pid, signal.SIGKILL)) == 1 for pid in (100, 101))


def test_strict_stalled_clock_still_stops_without_unverified_success(mod, strict_tree):
    runtime = _Pidfds(strict_tree)
    runtime.on_signal = lambda pid, sig: None
    sleeps = []
    assert not runtime.run(mod, grace=10.0, sleep=sleeps.append)
    assert 0 < len(sleeps) <= 200
    assert runtime.now == 0.0
    assert not any(sig == signal.SIGKILL for _pid, sig in runtime.sent)
    assert runtime.closed == set(runtime.opened)


@pytest.mark.parametrize("unknown_pid", [99, 102])
def test_strict_initial_unobservable_pid_does_not_abandon_healthy_peers(
    mod, strict_tree, unknown_pid
):
    runtime = _Pidfds(strict_tree)
    _strict_process(strict_tree, unknown_pid, birth=2000 + unknown_pid)
    (strict_tree / str(unknown_pid) / "environ").unlink()
    timed = []

    def on_signal(pid, sig):
        timed.append((pid, sig, runtime.now))
        if pid == 100 or sig == signal.SIGKILL:
            runtime.remove(pid)

    runtime.on_signal = on_signal
    assert not runtime.run(mod)  # The unknown identity still prevents clean.
    assert all(pid != unknown_pid for pid, _, _ in timed)
    assert (100, signal.SIGTERM) in runtime.sent
    assert any(pid == 101 and sig == signal.SIGKILL and t >= 0.4 for pid, sig, t in timed)
    assert runtime.now <= 20.4 + 1e-9
    assert runtime.closed == set(runtime.opened)


@pytest.mark.parametrize("environment", [b"\0", b"\0" * 32])
def test_strict_nul_only_environment_never_establishes_quiet_cleanup(mod, strict_tree, environment):
    runtime = _Pidfds(strict_tree)
    runtime.remove(101)
    (strict_tree / "100" / "environ").write_bytes(environment)
    assert not runtime.run(mod)
    assert runtime.sent == []
    assert runtime.now <= 20.4 + 1e-9
    assert runtime.closed == set(runtime.opened)


@pytest.mark.parametrize("file", ["environ", "stat"])
def test_strict_initially_unresolved_exit_can_finish_cleanly(mod, strict_tree, file):
    runtime = _Pidfds(strict_tree)
    _strict_process(strict_tree, 102, birth=1102)
    (strict_tree / "102" / file).write_bytes(b"")
    runtime.on_sleep = lambda: runtime.remove(102)
    assert runtime.run(mod)
    assert all(pid != 102 for pid, _ in runtime.sent)
    assert {(100, signal.SIGTERM), (101, signal.SIGTERM)} <= set(runtime.sent)
    assert runtime.closed == set(runtime.opened)


@pytest.mark.parametrize("pid", [99, 102])
def test_strict_unreadable_census_keeps_cleaning_verified_peers(mod, strict_tree, monkeypatch, pid):
    runtime = _Pidfds(strict_tree)
    _strict_process(strict_tree, pid, birth=2000 + pid)
    unreadable = strict_tree / str(pid) / "stat"
    original = open

    def denied(path, *args, **kwargs):
        if os.fspath(path) == str(unreadable):
            raise PermissionError(13, "census stat unreadable", str(unreadable))
        return original(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", denied)
    lines = []
    assert not runtime.run(mod, log=lines.append)
    assert all(target != pid for target, _ in runtime.sent)
    assert {(100, signal.SIGTERM), (101, signal.SIGTERM)} <= set(runtime.sent)
    assert runtime.closed == set(runtime.opened)
    assert sum("PermissionError" in line and str(pid) in line for line in lines) == 1


@pytest.mark.parametrize("preserve_environment", [False, True])
def test_strict_real_retitled_process_retains_marker_only_with_spt_noenv(
    mod, tmp_path, preserve_environment
):
    pytest.importorskip("setproctitle")
    import select

    marker = f"spt-{os.getpid()}-{time.monotonic_ns()}"
    env = {**os.environ, "E2E_JOB_MARKER": marker}
    env.pop("SPT_NOENV", None)
    if preserve_environment:
        env["SPT_NOENV"] = "1"
    child = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-c",
            "import setproctitle,time; setproctitle.setproctitle('smg::router'); "
            "print('READY',flush=True); time.sleep(60)",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert select.select([child.stdout], [], [], 5)[0], "retitled child did not start"
        assert child.stdout.readline() == "READY\n"
        observed = Path(f"/proc/{child.pid}/environ").read_bytes()
        assert (f"E2E_JOB_MARKER={marker}".encode() in observed) is preserve_environment
        # Real procfs and pidfds, with discovery restricted to this test's child
        # and protected ancestors. Never expose unrelated host PIDs to cleanup.
        view = tmp_path / "proc"
        view.mkdir()
        for pid in {1, child.pid, *mod._strict_ancestors("/proc", os.getpid())}:
            (view / str(pid)).symlink_to(f"/proc/{pid}", target_is_directory=True)
        assert mod.reap_strict(marker, grace=0.1, proc_root=str(view)) is preserve_environment
        if preserve_environment:
            assert child.wait(timeout=5) == -signal.SIGTERM
        else:
            assert child.poll() is None
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
