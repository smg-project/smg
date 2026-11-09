"""Exercise the installed-holder protocol on temporary locks, without GPUs."""

import json
import os
import signal
import stat
import subprocess
import sys
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


def run_case(directory, logger_body, closed_stderr=False, blocked=False, require_events=False):
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

            def logger_started():
                try:
                    return pidfile.read_text().strip().isdigit()
                except FileNotFoundError:
                    return False

            until(logger_started)
            logger_pid = int(pidfile.read_text())
            info = lock.stat()
            identity = info.st_dev, info.st_ino
            for fd in Path(f"/proc/{logger_pid}/fd").iterdir():
                try:
                    info = fd.stat()
                    assert (info.st_dev, info.st_ino) != identity, "logger retained the lease"
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
        if require_events:
            assert events.is_file(), "normal audit log missing"
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


def missing_lock_alias_cases(directory):
    directory.mkdir()
    owner = subprocess.Popen(["true"])
    try:
        procstat = Path(f"/proc/{owner.pid}/stat")
        until(lambda: procstat.read_text().rsplit(")", 1)[1].split()[0] == "Z")
        birth = procstat.read_text().rsplit(")", 1)[1].split()[19]
        owner.wait(timeout=2)
        observations = []
        for spelling in ["literal", "dot", "parent", "relative"]:
            case = directory / spelling
            case.mkdir()
            (case / "subdirectory").mkdir()
            lock = case / "lock"
            ready = {
                "literal": str(lock),
                "dot": str(case) + "/./lock",
                "parent": str(case) + "/subdirectory/../lock",
                "relative": "./lock",
            }[spelling]
            assert not lock.exists()
            result = subprocess.run(
                ["bash", str(SCRIPT), "--lock-path", str(lock), str(owner.pid), birth, ready, "0"],
                cwd=case,
                capture_output=True,
                timeout=4,
            )
            observation = {
                "spelling": spelling,
                "returncode": result.returncode,
                "lock_created": lock.exists(),
            }
            print("MISSING_LOCK_ALIAS_OBSERVATION " + json.dumps(observation), flush=True)
            assert result.returncode == 2, "missing-lock alias must reject before publishing status"
            assert b"ready path aliases the lease lock" in result.stderr, result.stderr
            assert not lock.exists(), "owner-gone status created the lease lock"
            assert not list(case.glob("lock.tmp.*"))
            observations.append(observation)
        return observations
    finally:
        owner.wait(timeout=2)


def lifecycle_cases(directory):
    directory.mkdir()
    bindir = directory / "bin"
    bindir.mkdir()
    logger = bindir / "logger"
    logger.write_text("#!/bin/sh\nexit 0\n")
    logger.chmod(0o700)
    env = {**os.environ, "PATH": str(bindir) + ":/usr/bin:/bin"}
    owner = subprocess.Popen(["sleep", "30"])
    start = Path(f"/proc/{owner.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    lock, ready = directory / "lock", directory / "ready"
    holder = subprocess.Popen(
        ["bash", str(SCRIPT), "--lock-path", str(lock), str(owner.pid), start, str(ready), "2"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        until(ready.exists)
        assert ready.read_text().strip() == "acquired"
        children = Path(f"/proc/{holder.pid}/task/{holder.pid}/children")

        def sleeping_child():
            for child in children.read_text().split():
                try:
                    if Path(f"/proc/{child}/cmdline").read_bytes().split(b"\0")[:2] == [
                        b"sleep",
                        b"2",
                    ]:
                        return True
                except FileNotFoundError:
                    pass
            return False

        until(sleeping_child)
        holder.kill()
        holder.wait(timeout=2)
        assert subprocess.run(["flock", "-n", str(lock), "true"]).returncode == 0
        identity = lock.stat().st_dev, lock.stat().st_ino
        alias = directory / "alias"
        alias.hardlink_to(lock)
        collision = subprocess.run(
            ["bash", str(SCRIPT), "--lock-path", str(lock), str(owner.pid), start, str(alias), "0"],
            env=env,
            capture_output=True,
            timeout=4,
        )
        assert collision.returncode == 2
        assert b"ready path aliases the lease lock" in collision.stderr, collision.stderr
        assert (lock.stat().st_dev, lock.stat().st_ino) == identity
        assert alias.samefile(lock)
        # Keep /./ in the literal argument: Path joining would normalize it.
        literal_alias = str(directory) + "/./lock"
        assert literal_alias != str(lock)
        lock.write_bytes(b"private-lock-sentinel\n")
        before_alias = lock.stat().st_dev, lock.stat().st_ino, lock.read_bytes()
        spelling_collision = subprocess.run(
            [
                "bash",
                str(SCRIPT),
                "--lock-path",
                str(lock),
                str(owner.pid),
                start,
                literal_alias,
                "0",
            ],
            env=env,
            capture_output=True,
            timeout=4,
        )
        assert spelling_collision.returncode == 2
        assert b"ready path aliases the lease lock" in spelling_collision.stderr, (
            spelling_collision.stderr
        )
        assert (lock.stat().st_dev, lock.stat().st_ino, lock.read_bytes()) == before_alias
        assert alias.samefile(lock) and alias.read_bytes() == before_alias[2]
    finally:
        owner.terminate()
        owner.wait(timeout=2)
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=2)
    zombie = subprocess.Popen(["true"])
    try:
        statfile = Path(f"/proc/{zombie.pid}/stat")
        until(lambda: statfile.read_text().rsplit(")", 1)[1].split()[0] == "Z")
        start = statfile.read_text().rsplit(")", 1)[1].split()[19]
        stopped = subprocess.run(
            [
                "bash",
                str(SCRIPT),
                "--lock-path",
                str(lock),
                str(zombie.pid),
                start,
                str(directory / "gone"),
                "0",
            ],
            env=env,
            capture_output=True,
            timeout=3,
        )
        assert stopped.returncode == 1
        assert (directory / "gone").read_text().strip() == "owner-gone"
    finally:
        zombie.wait(timeout=2)
    return {
        "polling_child_does_not_retain_lock": True,
        "ready_alias_rejected_without_inode_change": True,
        "zombie_owner_rejected": True,
        "literal_same_dentry_alias_rejected_content_preserved": True,
        "missing_lock_aliases_rejected": missing_lock_alias_cases(directory / "missing-lock"),
    }


def ready_tempfile_cases(directory):
    directory.mkdir()
    bindir = directory / "bin"
    bindir.mkdir()
    logger = bindir / "logger"
    logger.write_text("#!/bin/sh\nexit 0\n")
    logger.chmod(0o700)
    outcomes = []
    for index, (mask, collision_kind) in enumerate(
        [("022", "symlink"), ("077", "symlink"), ("022", "regular")]
    ):
        case = directory / str(index)
        case.mkdir()
        sentinel, lock, ready = case / "sentinel", case / "lock", case / "ready"
        sentinel.write_bytes(b"private-sentinel-unchanged\n")
        lock.write_bytes(b"private-lease-unchanged\n")
        sentinel_before = sentinel.stat().st_dev, sentinel.stat().st_ino, sentinel.read_bytes()
        lock_before = lock.stat().st_dev, lock.stat().st_ino, lock.read_bytes()
        owner = subprocess.Popen(["true"])
        try:
            procstat = Path(f"/proc/{owner.pid}/stat")
            until(lambda: procstat.read_text().rsplit(")", 1)[1].split()[0] == "Z")
            birth = procstat.read_text().rsplit(")", 1)[1].split()[19]
            env = {**os.environ, "PATH": str(bindir) + ":/usr/bin:/bin"}
            # exec preserves the wrapper PID, reproducing READY.tmp.$$ exactly.
            wrapper = r"""umask "$1"
if [ "$2" = symlink ]; then
  ln -s -- "$3" "$5.tmp.$$"
else
  printf 'preexisting-regular\n' > "$5.tmp.$$"
fi
exec bash "$6" --lock-path "$4" "$7" "$8" "$5" 0
"""
            holder = subprocess.Popen(
                [
                    "bash",
                    "-c",
                    wrapper,
                    "ready-fixture",
                    mask,
                    collision_kind,
                    str(sentinel),
                    str(lock),
                    str(ready),
                    str(SCRIPT),
                    str(owner.pid),
                    birth,
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = holder.communicate(timeout=4)
            old_temporary = Path(str(ready) + ".tmp." + str(holder.pid))
            observation = {
                "umask": mask,
                "collision": collision_kind,
                "returncode": holder.returncode,
                "sentinel_unchanged": (
                    sentinel.stat().st_dev,
                    sentinel.stat().st_ino,
                    sentinel.read_bytes(),
                )
                == sentinel_before,
                "lock_unchanged": (lock.stat().st_dev, lock.stat().st_ino, lock.read_bytes())
                == lock_before,
                "ready_is_symlink": ready.is_symlink(),
            }
            print("READY_TEMPFILE_OBSERVATION " + json.dumps(observation), flush=True)
            assert observation["sentinel_unchanged"], (
                "preexisting status symlink clobbered private sentinel"
            )
            assert observation["lock_unchanged"], "status publication changed private lease"
            assert holder.returncode == 1, (stdout, stderr)
            assert stat.S_ISREG(ready.lstat().st_mode) and not ready.is_symlink()
            assert ready.read_text().strip() == "owner-gone"
            assert stat.S_IMODE(ready.stat().st_mode) == (0o666 & ~int(mask, 8))
            if collision_kind == "symlink":
                assert old_temporary.is_symlink() and old_temporary.samefile(sentinel)
            else:
                assert old_temporary.read_bytes() == b"preexisting-regular\n"
            assert set(case.glob("ready.tmp.*")) == {old_temporary}
            outcomes.append(observation)
        finally:
            owner.wait(timeout=2)
    return {"exclusive_regular_status": True, "private_cases": outcomes}


def partial_write_failure_case(directory):
    directory.mkdir()
    bindir = directory / "bin"
    bindir.mkdir()
    logger = bindir / "logger"
    logger.write_text("#!/bin/sh\nexit 0\n")
    logger.chmod(0o700)
    lock, ready = directory / "lock", directory / "ready"
    lock.write_bytes(b"private-lock-unchanged\n")
    ready.write_bytes(b"previous-ready-unchanged\n")
    lock_before = lock.stat().st_dev, lock.stat().st_ino, lock.read_bytes()
    ready_before = ready.stat().st_dev, ready.stat().st_ino, ready.read_bytes()
    owner = subprocess.Popen(["true"])
    try:
        procstat = Path(f"/proc/{owner.pid}/stat")
        until(lambda: procstat.read_text().rsplit(")", 1)[1].split()[0] == "Z")
        birth = procstat.read_text().rsplit(")", 1)[1].split()[19]
        # Only this private process overrides printf. It writes partial data to
        # the actual redirected status FD and returns a controlled failure.
        wrapper = r"""printf() {
  if [[ $# == 2 && $1 == '%s\n' && $2 == owner-gone ]]; then
    builtin printf 'controlled-partial-status\n'
    return 23
  fi
  builtin printf "$@"
}
export -f printf
exec bash "$1" --lock-path "$2" "$3" "$4" "$5" 0
"""
        failed = subprocess.run(
            [
                "bash",
                "-c",
                wrapper,
                "partial-status-fixture",
                str(SCRIPT),
                str(lock),
                str(owner.pid),
                birth,
                str(ready),
            ],
            env={**os.environ, "PATH": str(bindir) + ":/usr/bin:/bin"},
            capture_output=True,
            timeout=4,
        )
        leftovers = sorted(directory.glob("ready.tmp.*"))
        observation = {
            "returncode": failed.returncode,
            "ready_unchanged": (ready.stat().st_dev, ready.stat().st_ino, ready.read_bytes())
            == ready_before,
            "lock_unchanged": (lock.stat().st_dev, lock.stat().st_ino, lock.read_bytes())
            == lock_before,
            "owned_leftovers": [
                {
                    "name": path.name,
                    "partial_contents": (path / "status").read_text()
                    if (path / "status").is_file()
                    else None,
                }
                for path in leftovers
            ],
        }
        print("PARTIAL_WRITE_OBSERVATION " + json.dumps(observation), flush=True)
        assert failed.returncode == 1
        assert observation["ready_unchanged"] and observation["lock_unchanged"]
        assert not leftovers, "failed status write leaked owned temporary directory"
        return {"controlled_partial_write_rc1_cleanup": True, "observation": observation}
    finally:
        owner.wait(timeout=2)


with tempfile.TemporaryDirectory(prefix="lease-audit-tests-") as scratch:
    root = Path(scratch)
    if sys.argv[1:] == ["--missing-lock-only"]:
        results = {"missing_lock_aliases": missing_lock_alias_cases(root / "missing-lock")}
    elif sys.argv[1:] == ["--partial-write-only"]:
        results = {"partial_write_cleanup": partial_write_failure_case(root / "partial-write")}
    elif sys.argv[1:] == ["--ready-tempfile-only"]:
        results = {"ready_tempfiles": ready_tempfile_cases(root / "ready-tempfiles")}
    else:
        assert not sys.argv[1:]
        results = {
            "lifecycle": lifecycle_cases(root / "lifecycle"),
            "normal": run_case(
                root / "normal", 'printf "%s\\n" "$*" >> "$TEST_EVENTS"\n', require_events=True
            ),
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
            "ready_tempfiles": ready_tempfile_cases(root / "ready-tempfiles"),
            "partial_write_cleanup": partial_write_failure_case(root / "partial-write"),
        }
print(json.dumps({"PASS": True, "results": results, "actual_GPU_or_shared_lock": False}))
