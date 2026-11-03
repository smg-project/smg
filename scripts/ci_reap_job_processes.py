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

Strict mode uses Linux pidfds, requires complete ownership observations and
returns nonzero for an unresolved identity or survivor. The default remains
legacy best-effort behavior. Process cleanup does not prove physical release.

Usage: ci_reap_job_processes.py [--strict | --dry-run] [--grace SECS] MARKER
"""

from __future__ import annotations

import argparse
import dataclasses
import math
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
    found = marked_processes(marker, proc_root=proc_root, exclude=ancestors(proc_root, me) | {1})
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


@dataclasses.dataclass(frozen=True)
class OwnedProcess:
    pid: int
    ppid: int
    uid: int
    starttime: int


def _strict_stat(proc_root: str, pid: int) -> tuple[str, int, int] | None:
    try:
        with open(os.path.join(proc_root, str(pid), "stat"), encoding="utf-8") as stream:
            text = stream.read(8193)
    except FileNotFoundError:
        return None
    fields = text.rpartition(")")[2].split()
    if len(text) > 8192 or text.partition("(")[0].strip() != str(pid) or len(fields) < 20:
        raise ValueError(f"cannot establish stat identity for pid {pid}")
    ppid, starttime = int(fields[1]), int(fields[19])
    if ppid < 0 or starttime < 0:
        raise ValueError(f"invalid stat identity for pid {pid}")
    return fields[0], ppid, starttime


def _strict_owned(proc_root: str, pid: int, uid: int, entry: bytes) -> OwnedProcess | None:
    before = _strict_stat(proc_root, pid)
    if before is None or before[0] == "Z":
        return None
    try:
        with open(os.path.join(proc_root, str(pid), "status"), encoding="utf-8") as stream:
            status = stream.read(65537)
        rows = [line.split()[1:] for line in status.splitlines() if line.startswith("Uid:")]
        if len(status) > 65536 or len(rows) != 1 or len(rows[0]) != 4:
            raise ValueError(f"cannot establish UID for pid {pid}")
        uids = tuple(int(value) for value in rows[0])
        if any(value < 0 for value in uids):
            raise ValueError(f"invalid UID for pid {pid}")
        if uids != (uid,) * 4:
            if uid not in uids:
                return None  # Clearly foreign UID: never read its environment or signal it.
            raise ValueError(f"mixed UID ownership for pid {pid}")
        with open(os.path.join(proc_root, str(pid), "environ"), "rb") as stream:
            environ = stream.read(1048577)
        if len(environ) > 1048576:
            raise ValueError(f"incomplete environment observation for pid {pid}")
    except FileNotFoundError:
        if _strict_stat(proc_root, pid) is None:
            return None
        raise ValueError(f"ownership files disappeared for live pid {pid}") from None
    after = _strict_stat(proc_root, pid)
    if after is None or after[0] == "Z":
        return None
    if before[2] != after[2]:
        raise ValueError(f"PID birth changed during observation: {pid}")
    markers = [value for value in environ.split(b"\0") if value.startswith(b"E2E_JOB_MARKER=")]
    if entry not in markers:
        return None
    if markers != [entry]:
        raise ValueError(f"ambiguous marker for pid {pid}")
    return OwnedProcess(pid, after[1], uid, after[2])


def _strict_ancestors(proc_root: str, pid: int) -> dict[int, tuple[int, int]]:
    chain: dict[int, tuple[int, int]] = {}
    while pid > 1:
        if pid in chain:
            raise ValueError("cannot establish the cleanup process ancestor chain")
        before = _strict_stat(proc_root, pid)
        after = _strict_stat(proc_root, pid)
        if before is None or after is None or before[1:] != after[1:]:
            raise ValueError(f"unresolved ancestor identity: {pid}")
        chain[pid] = after[1:]
        pid = after[1]
    return chain


def reap_strict(
    marker: str,
    *,
    proc_root: str = "/proc",
    grace: float = DEFAULT_GRACE_SECS,
    pidfd_open=None,
    pidfd_send_signal=None,
    close: Callable[[int], None] = os.close,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    self_pid: int | None = None,
    uid: int | None = None,
    log: Callable[[str], None] = print,
) -> bool:
    """Signal only verified marker/UID/birth identities through bound pidfds.

    Injectable APIs and proc_root are for CPU fixtures. The production strict
    CLI permits only /proc. True means two complete empty observations in this
    ownership domain; it does not prove GPU/container/service release.
    """
    opener = getattr(os, "pidfd_open", None) if pidfd_open is None else pidfd_open
    sender = (
        getattr(signal, "pidfd_send_signal", None)
        if pidfd_send_signal is None
        else pidfd_send_signal
    )
    if not callable(opener) or not callable(sender):
        log("::error::strict cleanup requires pidfd_open and pidfd_send_signal")
        return False
    if not marker.strip() or not math.isfinite(grace) or grace < 0:
        log("::error::invalid strict cleanup marker or grace")
        return False
    me = os.getpid() if self_pid is None else self_pid
    owner_uid = os.getuid() if uid is None else uid
    entry = f"{MARKER_ENV}={marker}".encode()
    term_deadline = clock() + grace
    deadline = term_deadline + 2.0
    handles: dict[int, tuple[OwnedProcess, int]] = {}
    seen: dict[int, int] = {}
    sent: set[tuple[int, int, int]] = set()
    clean = False

    def retire(pid: int) -> None:
        _, fd = handles.pop(pid)
        close(fd)

    def still_owned(process: OwnedProcess) -> bool:
        current = _strict_owned(proc_root, process.pid, owner_uid, entry)
        if current is None:
            stat = _strict_stat(proc_root, process.pid)
            if stat is None or stat[0] == "Z":
                return False
            raise ValueError(f"tracked ownership changed or is unobservable: {process.pid}")
        if (current.uid, current.starttime) != (process.uid, process.starttime):
            raise ValueError(f"tracked UID or PID birth changed: {process.pid}")
        return True

    try:
        protected = _strict_ancestors(proc_root, me)
        quiet = 0
        for _ in range(64):
            if _strict_ancestors(proc_root, me) != protected:
                raise ValueError("cleanup process ancestor chain changed")
            observed: list[OwnedProcess] = []
            for name in sorted(os.listdir(proc_root)):
                if not name.isdigit() or int(name) in protected or int(name) <= 1:
                    continue
                process = _strict_owned(proc_root, int(name), owner_uid, entry)
                if process is not None:
                    observed.append(process)
            for process in observed:
                if process.pid in seen and seen[process.pid] != process.starttime:
                    raise ValueError(f"previously observed numeric PID was reused: {process.pid}")
                seen[process.pid] = process.starttime
                if process.pid not in handles:
                    if clock() >= deadline:
                        raise TimeoutError("strict cleanup deadline before pidfd acquisition")
                    try:
                        fd = opener(process.pid, 0)
                    except ProcessLookupError:
                        continue
                    handles[process.pid] = (process, fd)
                    # The pidfd and the post-open proc observation must bind the
                    # same identity. No signal is sent if the numeric PID changed.
                    if not still_owned(process):
                        retire(process.pid)
            for pid, (process, fd) in list(handles.items()):
                if not still_owned(process):
                    retire(pid)
                    continue
                signals = [signal.SIGTERM]
                if clock() >= term_deadline:
                    signals.append(signal.SIGKILL)
                for sig in signals:
                    key = (pid, process.starttime, sig)
                    if key in sent:
                        continue
                    if not still_owned(process):
                        retire(pid)
                        break
                    if clock() >= deadline:
                        raise TimeoutError("strict cleanup signal deadline reached")
                    try:
                        sender(fd, sig, None, 0)
                    except ProcessLookupError:
                        retire(pid)
                        break
                    sent.add(key)
                    log(f"strict signal {sig} pid {pid} uid {owner_uid} birth {process.starttime}")
            if clock() > deadline:
                raise TimeoutError("strict cleanup observation deadline reached")
            quiet = quiet + 1 if not observed and not handles else 0
            if quiet == 2:
                clean = True
                break
            remaining = deadline - clock()
            if remaining <= 0:
                break
            sleep(min(POLL_SECS, remaining))
        if not clean:
            log(f"::error::strict cleanup unresolved/surviving owned pids: {sorted(handles)}")
    except (OSError, ValueError, TimeoutError) as error:
        log(f"::error::strict cleanup incomplete: {type(error).__name__}: {error}")
    finally:
        for _, fd in handles.values():
            try:
                close(fd)
            except OSError as error:
                clean = False
                log(f"::error::cannot close strict pidfd: {type(error).__name__}")
    if clean:
        log("strict owned-process observations are empty; physical resource release is separate")
    return clean


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("marker", help=f"the value of {MARKER_ENV} the job exported")
    parser.add_argument("--dry-run", action="store_true", help="list the processes, end none")
    parser.add_argument("--strict", action="store_true", help="require verified pidfd cleanup")
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
    if args.strict:
        if args.proc_root != "/proc":
            parser.error("--strict requires the actual /proc")
        if args.dry_run:
            parser.error("--strict cannot certify cleanup in a dry run")
        return 0 if reap_strict(args.marker, grace=args.grace) else 1
    reap(args.marker, proc_root=args.proc_root, grace=args.grace, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
