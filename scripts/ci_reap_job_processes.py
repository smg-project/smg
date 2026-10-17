#!/usr/bin/env python3
"""End every process that still carries a job's marker in its environment.

A CI job exports ``E2E_JOB_MARKER`` (a value unique to the job) before it
starts anything, so every process the job starts, and every child of those
(an engine's cores, a scheduler's ranks), inherits the marker. At the end of
the job, whatever still carries it is a leftover: an engine that outlived a
cancelled or timed-out test step, a gateway whose teardown never ran. Such a
process keeps its GPU memory and its listening ports, and the next job on the
runner starts against them. This script ends them: SIGTERM to each, a bounded
wait, SIGKILL to what remains. It reads ``/proc`` (a process whose environment
it cannot read belongs to another user, not to the job), leaves itself and its
ancestors alone (the step's shell carries the marker too), and prints what it
ended; ``--dry-run`` only prints. It exits 0 either way: a teardown reports,
it does not fail the job on its own.

Usage: ci_reap_job_processes.py [--dry-run] [--grace SECS] [--proc-root DIR] MARKER
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import signal
import sys
import time
from collections.abc import Callable, Iterable

MARKER_ENV = "E2E_JOB_MARKER"
DEFAULT_GRACE_SECS = 10.0
POLL_SECS = 0.2


@dataclasses.dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    command: str


def _stat(proc_root: str, pid: int) -> tuple[str, int] | None:
    """``(state, ppid)`` of ``pid``; ``None`` when it is gone."""
    try:
        with open(
            os.path.join(proc_root, str(pid), "stat"), encoding="utf-8", errors="replace"
        ) as f:
            stat = f.read()
    except OSError:
        return None
    # ``pid (comm) state ppid ...``: comm may hold spaces or parentheses, so
    # the fields are read after the last closing parenthesis.
    fields = stat.rpartition(")")[2].split()
    if len(fields) < 2:
        return None
    return fields[0], int(fields[1])


def _carries(proc_root: str, pid: int, entry: bytes) -> bool:
    """Whether ``pid``'s environment holds exactly the entry ``NAME=value``."""
    try:
        with open(os.path.join(proc_root, str(pid), "environ"), "rb") as f:
            environ = f.read()
    except OSError:  # gone, or another user's: not the job's
        return False
    return entry in environ.split(b"\0")


def _command(proc_root: str, pid: int) -> str:
    try:
        with open(os.path.join(proc_root, str(pid), "cmdline"), "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""


def alive(proc_root: str, pid: int) -> bool:
    """Whether ``pid`` still runs (a zombie awaiting its parent's wait does not)."""
    stat = _stat(proc_root, pid)
    return stat is not None and stat[0] != "Z"


def ancestors(proc_root: str, pid: int) -> set[int]:
    """``pid`` and its parent chain, up to (not including) init."""
    chain: set[int] = set()
    while pid > 1 and pid not in chain:
        chain.add(pid)
        stat = _stat(proc_root, pid)
        if stat is None:
            break
        pid = stat[1]
    return chain


def marked_processes(
    marker: str, *, proc_root: str = "/proc", exclude: Iterable[int] = ()
) -> list[Process]:
    """The live processes whose environment carries ``E2E_JOB_MARKER=marker``,
    in pid order, the ``exclude``d ones left out."""
    entry = f"{MARKER_ENV}={marker}".encode()
    excluded = set(exclude)
    found: list[Process] = []
    for name in os.listdir(proc_root):
        if not name.isdigit() or int(name) in excluded:
            continue
        pid = int(name)
        stat = _stat(proc_root, pid)
        if stat is None or stat[0] == "Z" or not _carries(proc_root, pid, entry):
            continue
        found.append(Process(pid=pid, ppid=stat[1], command=_command(proc_root, pid)))
    return sorted(found, key=lambda process: process.pid)


def _signal(pids: Iterable[int], sig: int, kill: Callable[[int, int], None], log) -> None:
    for pid in sorted(pids):
        try:
            kill(pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            log(f"::warning::cannot signal pid {pid}: permission denied")


def _wait(pids: set[int], seconds: float, proc_root: str, sleep) -> set[int]:
    """The pids of ``pids`` still alive after up to ``seconds``."""
    for _ in range(max(int(seconds / POLL_SECS), 1)):
        pids = {pid for pid in pids if alive(proc_root, pid)}
        if not pids:
            break
        sleep(POLL_SECS)
    return {pid for pid in pids if alive(proc_root, pid)}


def reap(
    marker: str,
    *,
    proc_root: str = "/proc",
    grace: float = DEFAULT_GRACE_SECS,
    dry_run: bool = False,
    kill: Callable[[int, int], None] = os.kill,
    sleep: Callable[[float], None] = time.sleep,
    self_pid: int | None = None,
    log: Callable[[str], None] = print,
) -> list[Process]:
    """End every process marked with ``marker`` but this one and its ancestors:
    SIGTERM to each, up to ``grace`` seconds for them to exit, SIGKILL to the
    rest. Returns the processes found (ended; in a dry run, only listed)."""
    me = os.getpid() if self_pid is None else self_pid
    found = marked_processes(marker, proc_root=proc_root, exclude=ancestors(proc_root, me))
    if not found:
        log(f"no process carries {MARKER_ENV}={marker}")
        return found
    for process in found:
        verb = "would end" if dry_run else "ending"
        log(f"{verb} pid {process.pid} (parent {process.ppid}): {process.command}")
    if dry_run:
        return found
    pending = {process.pid for process in found}
    _signal(pending, signal.SIGTERM, kill, log)
    survivors = _wait(pending, grace, proc_root, sleep)
    if survivors:
        log(f"still alive after {grace:g}s: {sorted(survivors)}; sending SIGKILL")
        _signal(survivors, signal.SIGKILL, kill, log)
        survivors = _wait(survivors, 2.0, proc_root, sleep)
        if survivors:
            log(f"::warning::survived SIGKILL: {sorted(survivors)}")
    log(f"ended {len(found) - len(survivors)} of {len(found)} process(es) of {MARKER_ENV}={marker}")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("marker", help=f"the value of {MARKER_ENV} the job exported")
    parser.add_argument("--dry-run", action="store_true", help="list the processes, end none")
    parser.add_argument(
        "--grace",
        type=float,
        default=DEFAULT_GRACE_SECS,
        help=f"seconds between SIGTERM and SIGKILL (default {DEFAULT_GRACE_SECS:g})",
    )
    parser.add_argument("--proc-root", default="/proc", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.marker.strip():
        parser.error("the marker is empty")
    reap(args.marker, proc_root=args.proc_root, grace=args.grace, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
