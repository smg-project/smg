"""Tests for the GPU guard of scripts/ci_killall_sglang.sh (``nuke_gpus``), the
verdict a job gets before its tests start. Driven by a stand-in nvidia-smi on
PATH (scripts/tests/fixtures/gpu_guard/) playing the three recorded shapes:
cards full with no visible process (a previous job's engine in another pid
namespace), cards busy with a visible process, cards free; and by a scratch
/proc tree for the holders this container can see."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "ci_killall_sglang.sh"
SHIMS = Path(__file__).resolve().parent / "fixtures" / "gpu_guard"
NODE = "203.0.113.7"  # a documentation address, never a node
THIS_JOB = "e2e-222-1-runner-new"
OLD_JOB = "e2e-111-1-runner-old"
# A pid no host can hold (the kernel's pid limit is 4194304 on 64-bit), so the
# kill step, which ends every holder of a device file the scratch /proc names,
# signals nothing of the machine the tests run on.
GHOST = 4_194_305
READING = re.compile(r"^gpu guard: (\d+)s: (.*)$", re.M)
REPORT_HEADER = f"gpu guard: this job: runner 'runner-new', marker '{THIS_JOB}'"


class Guard:
    """One guard run: the shims on PATH, the stand-in's state, a scratch /proc
    tree, the knobs in the environment."""

    def __init__(self, tmp_path: Path) -> None:
        self.state = tmp_path / "smi"
        self.state.mkdir()
        (self.state / "cards").write_text("0 GPU-aaaa\n1 GPU-bbbb\n")
        self.proc = tmp_path / "proc"
        self.proc.mkdir()
        self.bin = tmp_path / "bin"
        shutil.copytree(SHIMS, self.bin)

    def busy(self, mib: int = 80327, cards: tuple[int, ...] = (0, 1)) -> None:
        (self.state / "used").write_text("".join(f"{card} {mib}\n" for card in cards))

    def free_after(self, seconds: int) -> None:
        (self.state / "free_after").write_text(str(int(time.time()) + seconds))

    def visible(self, pid: int) -> None:
        (self.state / "visible_pid").write_text(str(pid))

    def listed_app(
        self, card: int, pid: int, name: str = "/usr/bin/python3", mib: int = 80327
    ) -> None:
        with (self.state / "other_pids").open("a") as f:
            f.write(f"{card} {pid} {name} {mib}\n")

    def container(self, line: str) -> None:
        with (self.state / "containers").open("a") as f:
            f.write(line + "\n")

    def process(self, pid: int, command: str, *, marker: str | None, device: str | None) -> None:
        """A process of the scratch /proc: its command line, its environment
        (with a job marker or not) and, when ``device`` is given, an fd on it."""
        directory = self.proc / str(pid)
        (directory / "fd").mkdir(parents=True)
        (directory / "cmdline").write_bytes(command.replace(" ", "\0").encode() + b"\0")
        environ = {"HOME": "/home/runner"}
        if marker:
            environ["E2E_JOB_MARKER"] = marker
        (directory / "environ").write_bytes(
            b"".join(f"{key}={value}".encode() + b"\0" for key, value in environ.items())
        )
        (directory / "fd" / "0").symlink_to("/dev/null")
        if device:
            (directory / "fd" / "3").symlink_to(device)

    def run(self, wait: int, poll: int = 1) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_SMI_STATE": str(self.state),
            "CI_GPU_GUARD_PROC_ROOT": str(self.proc),
            "CI_GPU_GUARD_WAIT_SECS": str(wait),
            "CI_GPU_GUARD_POLL_SECS": str(poll),
            "NODE_IP": NODE,
            "RUNNER_NAME": "runner-new",
            "E2E_JOB_MARKER": THIS_JOB,
        }
        started = time.monotonic()
        result = subprocess.run(
            ["bash", str(SCRIPT), "nuke_gpus"], env=env, capture_output=True, text=True, timeout=120
        )
        result.elapsed = time.monotonic() - started  # type: ignore[attr-defined]
        return result


@pytest.fixture
def guard(tmp_path: Path) -> Guard:
    return Guard(tmp_path)


def error_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith("::error::")]


def test_free_cards_pass_on_the_first_reading(guard: Guard) -> None:
    result = guard.run(wait=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert error_lines(result.stdout) == []
    assert READING.findall(result.stdout) == []
    assert "gpu guard:" not in result.stdout
    assert result.elapsed < 10
    assert (guard.state / "calls").read_text().count("--query-gpu=index,memory.used") == 1


def test_full_cards_with_no_visible_process_pass_once_the_memory_drains(guard: Guard) -> None:
    guard.busy()
    guard.listed_app(0, 4242)
    guard.free_after(5)
    result = guard.run(wait=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "No running processes found" in result.stdout
    # the reading's text; its second is bash's SECONDS, whole seconds from the
    # second the loop started in, so it may read 0 or 1
    assert READING.findall(result.stdout)[0][1] == "GPU 0: 80327 MiB used; GPU 1: 80327 MiB used"
    # the holders, once, at the first dirty reading
    assert result.stdout.count(REPORT_HEADER) == 1
    assert (
        "gpu guard: GPU 0: pid 4242 (/usr/bin/python3) holds 80327 MiB; "
        "not in this job's pid namespace" in result.stdout
    )
    assert re.search(r"^gpu guard: every GPU clean after [1-9]\d*s$", result.stdout, re.M)
    assert error_lines(result.stdout) == []
    assert 3 <= result.elapsed < 20


def test_full_cards_that_never_drain_fail_after_the_window_naming_an_outside_holder(
    guard: Guard,
) -> None:
    guard.busy()
    guard.listed_app(0, 4242)
    guard.listed_app(1, 4243)
    guard.container("0123abcd bench:latest bench-1 up 3 hours")
    result = guard.run(wait=4)
    assert result.returncode == 1
    assert 4 <= result.elapsed < 20
    seconds = [int(at) for at, _ in READING.findall(result.stdout)]
    # what the window guarantees: a reading at or past 4 s and one per second,
    # less the one SECONDS skips when the loop's own time crosses a boundary
    assert seconds[-1] >= 4 and len(seconds) >= 4
    (error,) = error_lines(result.stdout)
    assert error.startswith(f"::error::GPU not clean on node '{NODE}' after ")
    assert error.endswith(
        ", held by a process outside this job's view (another pid namespace: "
        "a previous job's engine still exiting, or a leaked context):"
    )
    lines = result.stdout.splitlines()
    assert lines[lines.index(error) + 1 :] == ["GPU 0: 80327 MiB used", "GPU 1: 80327 MiB used"]
    assert (
        "gpu guard: GPU 1: pid 4243 (/usr/bin/python3) holds 80327 MiB; "
        "not in this job's pid namespace" in result.stdout
    )
    assert (
        "gpu guard: containers of this runner's docker: 0123abcd bench:latest bench-1 up 3 hours"
        in result.stdout
    )
    # the holders at the first dirty reading and again at the verdict
    assert result.stdout.count(REPORT_HEADER) == 2


def test_a_visible_holder_is_ended_and_the_cards_pass(guard: Guard) -> None:
    holder = subprocess.Popen(["sleep", "300"])
    try:
        guard.busy()
        guard.visible(holder.pid)
        result = guard.run(wait=30)
        assert result.returncode == 0, result.stdout + result.stderr
        assert holder.wait(timeout=10) == -signal.SIGKILL
        assert f" {holder.pid} " in result.stdout  # the process table listed it
        assert error_lines(result.stdout) == []
        assert result.elapsed < 10
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait()


def test_a_holder_of_a_device_file_is_ended_through_proc(guard: Guard) -> None:
    """The kill step finds the holders of a device file in /proc, as the
    report below it does: a holder the process table never listed ends."""
    holder = subprocess.Popen(["sleep", "300"])
    try:
        guard.process(holder.pid, "sleep 300", marker=THIS_JOB, device="/dev/nvidia0")
        result = guard.run(wait=30)
        assert result.returncode == 0, result.stdout + result.stderr
        assert holder.wait(timeout=10) == -signal.SIGKILL
        assert "No running processes found" in result.stdout
        assert error_lines(result.stdout) == []
        assert result.elapsed < 10
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait()


def test_the_kill_step_installs_nothing() -> None:
    """A package install inside the test step charges the mirror's speed to
    the step's budget (8 min 23 s for lsof on a slow mirror, two jobs red):
    the script runs no package manager and needs no lsof."""
    commands = [
        line for line in SCRIPT.read_text().splitlines() if not line.lstrip().startswith("#")
    ]
    assert [line for line in commands if "apt-get" in line or "lsof" in line] == []


def test_a_holder_in_this_job_view_is_named_with_the_job_that_started_it(guard: Guard) -> None:
    guard.busy(cards=(0,))
    guard.process(GHOST, "python -m engine.server --port 1", marker=OLD_JOB, device="/dev/nvidia0")
    guard.process(778, "bash run.sh", marker=THIS_JOB, device=None)
    guard.listed_app(0, GHOST, name="python")
    result = guard.run(wait=1)
    assert result.returncode == 1
    assert (
        f"gpu guard: GPU 0: pid {GHOST} (python) holds 80327 MiB; in this job's view: "
        f"python -m engine.server --port 1 E2E_JOB_MARKER={OLD_JOB}" in result.stdout
    )
    assert (
        f"gpu guard: pid {GHOST} holds a GPU device file: python -m engine.server --port 1 "
        f"E2E_JOB_MARKER={OLD_JOB}" in result.stdout
    )
    assert "pid 778" not in result.stdout
    (error,) = error_lines(result.stdout)
    assert error.endswith(f", held by pid {GHOST}:")
    assert result.stdout.splitlines()[-1] == "GPU 0: 80327 MiB used"


def test_a_zero_window_keeps_the_single_reading(guard: Guard) -> None:
    guard.busy()
    result = guard.run(wait=0)
    assert result.returncode == 1
    assert len(READING.findall(result.stdout)) == 1
    assert result.stdout.count(REPORT_HEADER) == 1
    (error,) = error_lines(result.stdout)
    assert re.match(
        rf"::error::GPU not clean on node '{re.escape(NODE)}' after \d+s, held by ", error
    )
    assert result.elapsed < 10


def test_an_unanswered_memory_query_fails_the_guard(guard: Guard) -> None:
    (guard.state / "fail_queries").touch()
    result = guard.run(wait=5)
    assert result.returncode == 1
    assert error_lines(result.stdout) == [f"::error::Unable to query GPU memory on node '{NODE}'."]
    assert result.elapsed < 10


def test_deadline_crossing_after_print_requires_a_new_memory_read(
    guard: Guard, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-read after crossing the deadline and report the terminal sample time."""
    guard.busy()
    guard.listed_app(0, 4242)
    guard.listed_app(1, 4243)
    guard.container("0123abcd bench:latest bench-1 up 3 hours")
    hook = tmp_path / "guard-clock.sh"
    hook.write_text(
        """unset SECONDS
SECONDS=0
SMG_TEST_CLOCK_ADVANCED=0
sleep() { :; }
echo() {
    builtin echo "$@"
    local smg_echo_status=$?
    case "${1-}" in
        "gpu guard: 0s: "*)
            if [ "$smg_echo_status" -eq 0 ] && [ "$SMG_TEST_CLOCK_ADVANCED" -eq 0 ]; then
                SMG_TEST_CLOCK_ADVANCED=1
                SECONDS=1
            fi
            ;;
        "gpu guard: 1s: "*)
            if [ "$smg_echo_status" -eq 0 ] && [ "$SMG_TEST_CLOCK_ADVANCED" -eq 1 ]; then
                SMG_TEST_CLOCK_ADVANCED=2
                SECONDS=2
            fi
            ;;
    esac
    return "$smg_echo_status"
}
"""
    )
    # BASH_ENV affects only the owned child and its shims, and monkeypatch restores it.
    # Once unset, SECONDS is an ordinary variable: no wall-clock delay drives this case.
    monkeypatch.setenv("BASH_ENV", str(hook))
    result = guard.run(wait=1)
    assert result.returncode == 1, result.stdout + result.stderr
    readings = READING.findall(result.stdout)
    assert readings and readings[0][0] == "0", result.stdout
    assert len(readings) >= 2, result.stdout
    assert int(readings[-1][0]) >= 1, result.stdout
    # An extra printed line alone is insufficient: the guard must query memory again.
    calls = (guard.state / "calls").read_text()
    assert calls.count("--query-gpu=index,memory.used") >= 2, calls
    assert result.stdout.count(REPORT_HEADER) == 2, result.stdout
    for card, pid in ((0, 4242), (1, 4243)):
        assert (
            f"gpu guard: GPU {card}: pid {pid} (/usr/bin/python3) holds 80327 MiB; "
            "not in this job's pid namespace" in result.stdout
        )
    assert (
        "gpu guard: containers of this runner's docker: 0123abcd bench:latest bench-1 up 3 hours"
        in result.stdout
    )
    (error,) = error_lines(result.stdout)
    assert error.startswith(f"::error::GPU not clean on node '{NODE}' after ")
    verdict_time = re.search(r" after (\d+)s,", error)
    assert verdict_time is not None, error
    assert int(verdict_time.group(1)) == int(readings[-1][0]), result.stdout
    assert error.endswith(
        ", held by a process outside this job's view (another pid namespace: "
        "a previous job's engine still exiting, or a leaked context):"
    )
    lines = result.stdout.splitlines()
    assert lines[lines.index(error) + 1 :] == ["GPU 0: 80327 MiB used", "GPU 1: 80327 MiB used"]


def test_poll_sleep_is_capped_by_the_remaining_window(
    guard: Guard, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cap a polling sleep after processing has consumed part of the window."""
    guard.busy()
    guard.listed_app(0, 4242)
    guard.listed_app(1, 4243)
    guard.container("0123abcd bench:latest bench-1 up 3 hours")
    startup_sleeps = tmp_path / "startup-sleeps"
    poll_sleeps = tmp_path / "poll-sleeps"
    hook = tmp_path / "guard-capped-sleep-clock.sh"
    hook.write_text(
        """unset SECONDS
SECONDS=0
SMG_TEST_READING_SEEN=0
sleep() {
    if [ "$SMG_TEST_READING_SEEN" -eq 0 ]; then
        builtin printf '%s\\n' "$*" >> "$SMG_TEST_STARTUP_SLEEP_LOG"
    else
        builtin printf '%s\\n' "$*" >> "$SMG_TEST_POLL_SLEEP_LOG"
        SECONDS=3
    fi
}
echo() {
    builtin echo "$@"
    local smg_echo_status=$?
    case "${1-}" in
        "gpu guard: 0s: "*)
            if [ "$smg_echo_status" -eq 0 ] && [ "$SMG_TEST_READING_SEEN" -eq 0 ]; then
                SMG_TEST_READING_SEEN=1
                SECONDS=1
            fi
            ;;
    esac
    return "$smg_echo_status"
}
"""
    )
    # The startup grace is recorded separately; only a sleep after a reading polls.
    monkeypatch.setenv("SMG_TEST_STARTUP_SLEEP_LOG", str(startup_sleeps))
    monkeypatch.setenv("SMG_TEST_POLL_SLEEP_LOG", str(poll_sleeps))
    monkeypatch.setenv("BASH_ENV", str(hook))
    result = guard.run(wait=3, poll=5)
    assert result.returncode == 1, result.stdout + result.stderr
    assert startup_sleeps.read_text().splitlines() == ["2"]
    assert poll_sleeps.read_text().splitlines() == ["2"]
    readings = READING.findall(result.stdout)
    assert [int(second) for second, _ in readings] == [0, 3], result.stdout
    calls = (guard.state / "calls").read_text()
    assert calls.count("--query-gpu=index,memory.used") == 2, calls
    assert result.stdout.count(REPORT_HEADER) == 2, result.stdout
    for card, pid in ((0, 4242), (1, 4243)):
        assert (
            f"gpu guard: GPU {card}: pid {pid} (/usr/bin/python3) holds 80327 MiB; "
            "not in this job's pid namespace" in result.stdout
        )
    assert (
        "gpu guard: containers of this runner's docker: 0123abcd bench:latest bench-1 up 3 hours"
        in result.stdout
    )
    (error,) = error_lines(result.stdout)
    assert error.startswith(f"::error::GPU not clean on node '{NODE}' after 3s, ")
    assert error.endswith(
        ", held by a process outside this job's view (another pid namespace: "
        "a previous job's engine still exiting, or a leaked context):"
    )
    lines = result.stdout.splitlines()
    assert lines[lines.index(error) + 1 :] == ["GPU 0: 80327 MiB used", "GPU 1: 80327 MiB used"]
