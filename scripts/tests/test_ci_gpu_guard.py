"""Tests for the GPU guard of scripts/ci_killall_sglang.sh (``nuke_gpus``), the
verdict a job gets before its tests start. Driven by a stand-in nvidia-smi on
PATH (scripts/tests/fixtures/gpu_guard/) playing the three recorded shapes:
cards full with no visible process (a previous job's engine in another pid
namespace), cards busy with a visible process, cards free; and by a scratch
/proc tree for the holders this container can see."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
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

    def __init__(self, tmp_path: Path, *, real_holder: bool = False) -> None:
        self.real_holder = real_holder
        self.state = tmp_path / "smi"
        self.state.mkdir()
        (self.state / "cards").write_text("0 GPU-aaaa\n1 GPU-bbbb\n")
        self.proc = tmp_path / "proc"
        self.proc.mkdir()
        self.bin = tmp_path / "bin"
        shutil.copytree(SHIMS, self.bin)
        # Signals are recorded, never forwarded unless this fixture explicitly
        # registered its own live Popen child with exact /proc identity. All
        # package/network commands and all other PIDs remain fully fake.
        smi = self.bin / "nvidia-smi"
        smi.write_text(
            smi.read_text().replace('"/proc/$1/stat"', '"$CI_GPU_GUARD_PROC_ROOT/$1/stat"')
        )
        self.signals = self.state / "signals"
        self.allowed_children = self.state / "allowed-children.json"
        self.allowed_children.write_text("{}\n")
        self.allowed_children.chmod(0o600)
        self.apt_calls = self.state / "apt-calls"
        self.network_calls = self.state / "network-calls"
        scripts = {
            "kill": f"#!{sys.executable}\n"
            + r"""import json
import os
from pathlib import Path
import signal
import sys

state = Path(os.environ["FAKE_SMI_STATE"])
args = sys.argv[1:]
with (state / "signals").open("a") as stream:
    stream.write(" ".join(args) + "\n")
allowed = json.loads((state / "allowed-children.json").read_text())
for text in args:
    if not text.isdecimal():
        continue
    pid = int(text)
    scratch = Path(os.environ["CI_GPU_GUARD_PROC_ROOT"]) / text / "stat"
    if scratch.is_file():
        scratch.write_text(f"{pid} (fixture) Z 0\n")
    identity = allowed.get(text)
    if args[:1] != ["-9"] or identity is None:
        continue
    # pidfd pins the process before identity is re-read, so PID reuse cannot
    # redirect the eventual signal. Nothing outside the explicit allowlist
    # reaches this OS path; the original synthetic tests do not read /proc.
    fd = None
    try:
        fd = os.pidfd_open(pid)
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        parent = identity["ppid"]
        parent_fields = Path(f"/proc/{parent}/stat").read_text().rsplit(")", 1)[1].split()
        if (pid <= 1 or fields[0] == "Z" or int(fields[1]) != parent
                or fields[19] != identity["start_ticks"]
                or parent_fields[19] != identity["parent_start_ticks"]):
            continue
        signal.pidfd_send_signal(fd, signal.SIGKILL)
        with (state / "forwarded-signals").open("a") as stream:
            stream.write(f"-9 {pid} {identity['start_ticks']} {parent}\n")
    except (ProcessLookupError, FileNotFoundError):
        pass
    finally:
        if fd is not None:
            os.close(fd)
""",
            "apt-get": """#!/bin/bash
printf '%s\n' "$*" >> "$FAKE_SMI_STATE/apt-calls"
exit 97
""",
            "sudo": '#!/bin/bash\nexec "$@"\n',
            "fuser": "#!/bin/bash\nexit 1\n",
            "curl": '#!/bin/bash\nprintf "curl %s\\n" "$*" >> "$FAKE_SMI_STATE/network-calls"\nexit 98\n',
            "wget": '#!/bin/bash\nprintf "wget %s\\n" "$*" >> "$FAKE_SMI_STATE/network-calls"\nexit 98\n',
        }
        for name, source in scripts.items():
            executable = self.bin / name
            executable.write_text(source)
            executable.chmod(0o755)
        self.apt_root = tmp_path / "empty-apt"
        self.apt_root.mkdir()  # no apt.conf.d: the old mirror helper returns without a probe
        self.bootstrap = tmp_path / "guard-isolation.sh"

    def register_holder(self, pid: int) -> None:
        """Register the explicit process() holder only in the scoped real-child test."""
        assert self.real_holder and pid > 1 and pid != os.getpid()
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        assert int(fields[1]) == os.getpid() and fields[0] != "Z"
        parent = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()
        assert json.loads(self.allowed_children.read_text()) == {}
        self.allowed_children.write_text(
            json.dumps(
                {
                    str(pid): {
                        "ppid": os.getpid(),
                        "start_ticks": fields[19],
                        "parent_start_ticks": parent[19],
                    }
                }
            )
            + "\n"
        )

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
        if self.real_holder:
            self.register_holder(pid)
        directory = self.proc / str(pid)
        (directory / "fd").mkdir(parents=True)
        (directory / "stat").write_text(f"{pid} (fixture) S 0\n")
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
        original_hook = os.environ.get("BASH_ENV")
        self.bootstrap.write_text(
            (f"source {shlex.quote(original_hook)}\n" if original_hook else "")
            + f'kill() {{ {shlex.quote(str(self.bin / "kill"))} "$@"; }}\n'
            + """command() {
    if [ "${1-}" = -v ]; then
        case " ${FAKE_GUARD_MISSING_TOOLS:-} " in *" ${2-} "*) return 1 ;; esac
    fi
    builtin command "$@"
}
"""
        )
        env.update(BASH_ENV=str(self.bootstrap), CI_APT_ROOT=str(self.apt_root))
        started = time.monotonic()
        result = subprocess.run(
            ["bash", str(SCRIPT), "nuke_gpus"], env=env, capture_output=True, text=True, timeout=120
        )
        result.elapsed = time.monotonic() - started  # type: ignore[attr-defined]
        return result


@pytest.fixture
def guard(tmp_path: Path, request: pytest.FixtureRequest) -> Guard:
    # Only this unchanged upstream test asks the production cleanup to end
    # its own real child. All other existing tests retain a fake-only guard.
    return Guard(
        tmp_path,
        real_holder=request.node.name == "test_a_holder_of_a_device_file_is_ended_through_proc",
    )


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
    pid = 4242
    guard.process(pid, "python -m fixture.engine", marker=THIS_JOB, device="/dev/nvidia0")
    guard.busy()
    guard.visible(pid)
    result = guard.run(wait=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (guard.proc / str(pid) / "stat").read_text() == f"{pid} (fixture) Z 0\n"
    assert f"-9 {pid}" in guard.signals.read_text().splitlines()
    assert f" {pid} " in result.stdout  # the process table listed it
    assert error_lines(result.stdout) == []
    assert result.elapsed < 10


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


@pytest.mark.parametrize("fuser_present", [True, False], ids=["fuser-present", "no-holder-tools"])
def test_cleanup_needs_no_apt_even_when_lsof_is_missing(guard: Guard, monkeypatch, fuser_present):
    # Keep an executable lsof trap so old-code control runs cannot fall through
    # to a host utility, while command -v models its absence on the real runner.
    missing = "lsof" if fuser_present else "lsof fuser"
    monkeypatch.setenv("FAKE_GUARD_MISSING_TOOLS", missing)
    result = guard.run(wait=0)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not guard.apt_calls.exists(), "cleanup attempted a forbidden runtime apt install"
    assert not guard.network_calls.exists(), "cleanup attempted a forbidden network probe"
    assert error_lines(result.stdout) == []


def test_cleanup_records_only_scratch_device_holders_without_signalling(guard: Guard):
    guard.process(4242, "python -m fixture.engine", marker=THIS_JOB, device="/dev/nvidia0")
    guard.process(4243, "python -m fixture.other", marker=OLD_JOB, device="/dev/nvidia1")
    guard.process(4244, "python -m fixture.cpu", marker=THIS_JOB, device=None)
    result = guard.run(wait=0)
    assert result.returncode == 0, result.stdout + result.stderr
    assert guard.signals.read_text().splitlines() == ["-9", "-9 4242 4243"]
    assert " S " in (guard.proc / "4244" / "stat").read_text()


@pytest.mark.parametrize("identity", ["unregistered", "wrong-start", "wrong-parent"])
def test_signal_recorder_never_forwards_an_unregistered_or_wrong_identity(
    guard: Guard, identity: str
) -> None:
    holder = subprocess.Popen(["sleep", "300"])
    try:
        guard.real_holder = True  # Explicit opt-in only for identity-denial controls.
        guard.process(holder.pid, "sleep 300", marker=THIS_JOB, device="/dev/nvidia0")
        allowed = json.loads(guard.allowed_children.read_text())
        if identity == "unregistered":
            allowed = {}
        elif identity == "wrong-start":
            allowed[str(holder.pid)]["start_ticks"] = "0"
        else:
            allowed[str(holder.pid)]["ppid"] = 1
        guard.allowed_children.write_text(json.dumps(allowed))
        result = subprocess.run(
            [str(guard.bin / "kill"), "-9", str(holder.pid)],
            env={
                **os.environ,
                "FAKE_SMI_STATE": str(guard.state),
                "CI_GPU_GUARD_PROC_ROOT": str(guard.proc),
            },
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert holder.poll() is None
        assert not (guard.state / "forwarded-signals").exists()
        assert f"-9 {holder.pid}" in guard.signals.read_text().splitlines()
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait()
