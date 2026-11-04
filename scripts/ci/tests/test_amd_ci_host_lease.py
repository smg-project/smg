"""Exercise the installed-holder protocol on temporary locks, without GPUs."""

import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "amd_ci_hold_host_lease.sh"


def until(predicate, seconds=6):
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "bounded fixture wait expired"
        time.sleep(0.02)


def run_case(directory, logger_body, closed_stderr=False, blocked=False):
    directory.mkdir()
    bindir = directory / "bin"
    bindir.mkdir()
    logger = bindir / "logger"
    logger.write_text("#!/bin/bash\n" + logger_body)
    logger.chmod(0o755)
    events = directory / "events"
    env = os.environ.copy()
    env.update(
        PATH=str(bindir) + ":/usr/bin:/bin",
        TEST_EVENTS=str(events),
        TEST_LOGGER_PID=str(directory / "logger-pid"),
    )
    owner = subprocess.Popen(["sleep", "40"])
    start = Path(f"/proc/{owner.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    lock, ready = directory / "lock", directory / "ready"
    args = ["bash", str(SCRIPT), "--lock-path", str(lock), str(owner.pid), start, str(ready), "5"]
    if closed_stderr:
        args = ["bash", "-c", 'exec 2>&-; exec "$@"', "closed-stderr", *args]
    holder = subprocess.Popen(
        args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        if blocked:
            pidfile = directory / "logger-pid"
            until(pidfile.exists)
            logger_pid = int(pidfile.read_text())
            inode = lock.stat().st_ino
            for fd in Path(f"/proc/{logger_pid}/fd").iterdir():
                try:
                    assert fd.stat().st_ino != inode, "logger retained the lease"
                except FileNotFoundError:
                    pass
            holder.kill()
            holder.wait(timeout=2)
            assert holder.returncode == -signal.SIGKILL
            assert subprocess.run(["flock", "-n", str(lock), "true"]).returncode == 0
            time.sleep(4)  # bounded timeout must reap the intentionally stuck logger
            procstat = Path(f"/proc/{logger_pid}/stat")
            assert not procstat.exists() or procstat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
            return {"blocked_logger_holds_no_lock": True, "bounded_logger_exit": True}
        until(ready.exists)
        assert ready.read_text().strip() == "acquired"
        inode = lock.stat().st_ino
        assert holder.poll() is None
        assert subprocess.run(["flock", "-n", str(lock), "true"]).returncode != 0
        busy_args = [
            "bash",
            str(SCRIPT),
            "--lock-path",
            str(lock),
            str(owner.pid),
            start,
            str(directory / "busy"),
            "0",
        ]
        if closed_stderr:
            busy_args = ["bash", "-c", 'exec 2>&-; exec "$@"', "closed-stderr", *busy_args]
        busy = subprocess.run(busy_args, env=env, capture_output=True, timeout=5)
        assert busy.returncode == 75
        assert (directory / "busy").read_text().strip() == "busy"
        owner.terminate()
        owner.wait(timeout=2)
        holder.communicate(timeout=6)
        assert holder.returncode == 0
        assert lock.stat().st_ino == inode
        assert subprocess.run(["flock", "-n", str(lock), "true"]).returncode == 0
        gone = subprocess.run(
            [
                "bash",
                str(SCRIPT),
                "--lock-path",
                str(lock),
                str(owner.pid),
                start,
                str(directory / "gone"),
                "0",
            ],
            env=env,
            capture_output=True,
            timeout=3,
        )
        assert gone.returncode == 1
        assert (directory / "gone").read_text().strip() == "owner-gone"
        if events.exists():
            text = events.read_text()
            assert all(
                "event=" + word in text
                for word in ["waiting", "acquired", "released", "wait-timeout"]
            )
        return {"ready_protocol_and_rc_0_75_1": True, "same_inode": True}
    finally:
        if owner.poll() is None:
            owner.terminate()
            owner.wait(timeout=2)
        if holder.poll() is None:
            holder.kill()
            holder.communicate(timeout=2)


with tempfile.TemporaryDirectory(prefix="lease-audit-tests-") as scratch:
    root = Path(scratch)
    results = {
        "normal": run_case(root / "normal", 'printf "%s\\n" "$*" >> "$TEST_EVENTS"\n'),
        "logger_failure_closed_stderr": run_case(root / "failure", "exit 1\n", True),
        "blocked_logger_holder_death": run_case(
            root / "blocked",
            """
if [[ "$*" == *event=acquired* ]]; then
  echo $$ > "$TEST_LOGGER_PID"
  trap '' TERM
  sleep 20
fi
exit 0
""",
            blocked=True,
        ),
    }
print(json.dumps({"PASS": True, "results": results, "actual_GPU_or_shared_lock": False}))
