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

    def sleep(self, seconds):
        self.now += seconds
        self.on_sleep()

    def run(self, mod, **overrides):
        kwargs = dict(
            proc_root=str(self.root),
            grace=0.4,
            pidfd_open=self.open,
            pidfd_send_signal=self.send,
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
    assert runtime.sent == []
    assert runtime.closed == set(runtime.opened)


@pytest.mark.parametrize("missing", ["pidfd_open", "pidfd_send_signal"])
def test_strict_missing_pidfd_api_has_no_numeric_fallback(mod, strict_tree, monkeypatch, missing):
    runtime = _Pidfds(strict_tree)
    target = mod.os if missing == "pidfd_open" else mod.signal
    monkeypatch.setattr(target, missing, None, raising=False)
    numeric = []
    monkeypatch.setattr(mod.os, "kill", lambda *args: numeric.append(args))
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
    assert runtime.sent == []
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
    assert denied == [str(unreadable)]
    assert runtime.sent == [] and runtime.opened == {} and runtime.closed == set()
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
