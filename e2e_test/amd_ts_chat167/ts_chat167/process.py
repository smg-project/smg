"""Exact reviewed P4 direct-process identity primitive; no TP4 worker rules reused."""
import os
from pathlib import Path

def require(condition, message):
    if not condition:
        raise ValueError(message)

def process_identity(process, title=None):
    require(process is not None and process.poll() is None, "owned process not alive")
    root = Path("/proc") / str(process.pid)
    fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    require(int(fields[1]) == os.getpid(), "process is not a child of this pytest harness")
    argv = [x.decode() for x in (root / "cmdline").read_bytes().split(b"\0") if x]
    launch = list(process.args)
    require(argv == launch or (title is not None and argv == [title]), "actual argv/title differs")
    marker = ("E2E_JOB_MARKER=" + os.environ["E2E_JOB_MARKER"]).encode()
    require(marker in (root / "environ").read_bytes().split(b"\0"), "process marker differs")
    return {
        "pid": process.pid,
        "parent_pid": int(fields[1]),
        "start_ticks": fields[19],
        "argv": argv,
        "launch_argv_intent": launch,
        "marker_visible_in_proc": True,
    }
