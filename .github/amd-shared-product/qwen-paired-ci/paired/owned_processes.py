"""Finite Linux /proc observations and cleanup for this runner's nonce only."""

import hashlib
import os
from pathlib import Path
import signal
import time

KEY = "SMG_QWEN_AB_NONCE"


def identity(pid):
    root = Path("/proc") / str(pid)
    try:
        text = (root / "stat").read_text()
        tail = text[text.rfind(")") + 2:].split()
        with (root / "environ").open("rb") as stream:
            env = stream.read(262145)
        with (root / "cmdline").open("rb") as stream:
            argv = stream.read(16385)
        if len(env) > 262144 or len(argv) > 16384:
            raise ValueError("proc identity cap")
        return {"pid": pid, "ppid": int(tail[1]), "pgrp": int(tail[2]),
                "start_ticks": int(tail[19]), "state": tail[0],
                "argv": argv.rstrip(b"\0").decode(errors="replace").split("\0"),
                "env": dict(item.decode(errors="replace").split("=", 1)
                            for item in env.split(b"\0") if b"=" in item)}
    except (FileNotFoundError, ProcessLookupError):
        return None


def owned(nonce):
    result = []
    for root in Path("/proc").iterdir():
        if root.name.isdigit():
            try:
                row = identity(int(root.name))
                if row and row["env"].get(KEY) == nonce and row["state"] != "Z":
                    result.append(row)
            except PermissionError:
                continue
    return result


def public(row):
    return {**{k: v for k, v in row.items() if k != "env"},
            "marker": row["env"].get(KEY),
            "gpu_masks": {k: row["env"].get(k) for k in
                          ["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"]}}


def signal_exact(row, signum, nonce):
    now = identity(row["pid"])
    if now is None or now["state"] == "Z":
        return
    if now["start_ticks"] != row["start_ticks"] or now["env"].get(KEY) != nonce:
        raise RuntimeError("refuse signal after PID/nonce identity change")
    os.kill(row["pid"], signum)


def signal_group(leader, signum, nonce):
    current = identity(leader["pid"])
    if current is None or current["state"] == "Z":
        return False
    if current["start_ticks"] != leader["start_ticks"] or current["pgrp"] != current["pid"]:
        raise RuntimeError("refuse changed process-group leader")
    members = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                stat = (path / "stat").read_text()
                fields = stat[stat.rfind(")") + 2:].split()
                if int(fields[2]) != current["pid"]:
                    continue
                row = identity(int(path.name))
                if row and row["state"] != "Z":
                    members.append(row)
            except (FileNotFoundError, ProcessLookupError):
                continue
    if not members or any(row["env"].get(KEY) != nonce for row in members):
        raise RuntimeError("refuse group with unowned member")
    os.killpg(current["pid"], signum)
    return True


def cleanup(nonce, deadline, leaders):
    actions = []
    # Individual identities cover descendants that change process group/session.
    # Never signal a guessed process-group number after its launcher disappeared.
    for sig, grace in [(signal.SIGTERM, 5), (signal.SIGKILL, 5)]:
        rows = owned(nonce)
        for row in rows:
            if row["pgrp"] == row["pid"]:
                try:
                    if signal_group(row, sig, nonce):
                        actions.append({"signal": int(sig), "owned_process_group": row["pid"], "start_ticks": row["start_ticks"]})
                except ProcessLookupError:
                    pass
        for row in rows:
            try:
                signal_exact(row, sig, nonce)
                actions.append({"signal": int(sig), **public(row)})
            except ProcessLookupError:
                pass
        end = min(deadline, time.monotonic() + grace)
        while owned(nonce) and time.monotonic() < end:
            time.sleep(min(0.1, max(0, end - time.monotonic())))
        for child in leaders:
            if child.poll() is not None:
                child.wait()
        if not owned(nonce):
            return {"passed": True, "signals": actions, "remaining": []}
        if time.monotonic() >= deadline:
            break
    return {"passed": False, "signals": actions,
            "remaining": [public(row) for row in owned(nonce)]}


def mapped_native(pid, expected):
    names = set()
    for line in (Path("/proc") / str(pid) / "maps").read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and "smg_rs" in fields[5]:
            names.add(fields[5])
    if len(names) != 1:
        raise RuntimeError("gateway must map exactly one SMG native")
    name = names.pop()
    if name.endswith(" (deleted)"):
        raise RuntimeError("gateway native deleted")
    path = Path(name)
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if sha != expected:
        raise RuntimeError("loaded gateway native SHA differs")
    return {"pid": pid, "path": str(path), "sha256": sha}
