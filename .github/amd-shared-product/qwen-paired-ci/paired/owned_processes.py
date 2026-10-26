"""Finite Linux /proc observations and cleanup for this runner's nonce only."""

import hashlib
import os
from pathlib import Path
import signal
import time

KEY = "SMG_QWEN_AB_NONCE"


def stat_identity(pid):
    try:
        text = (Path("/proc") / str(pid) / "stat").read_text()
        tail = text[text.rfind(")") + 2:].split()
        return {"pid": pid, "ppid": int(tail[1]), "pgrp": int(tail[2]),
                "start_ticks": int(tail[19]), "state": tail[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def identity(pid):
    root = Path("/proc") / str(pid)
    try:
        row = stat_identity(pid)
        if row is None:
            return None
        if row["state"] == "Z":
            return {**row, "argv": [], "env": {}}
        try:
            with (root / "environ").open("rb") as stream:
                env = stream.read(262145)
        except PermissionError:
            after = stat_identity(pid)
            if after is None:
                return None
            if after["state"] == "Z" and after["start_ticks"] == row["start_ticks"]:
                return {**after, "argv": [], "env": {}}
            raise
        with (root / "cmdline").open("rb") as stream:
            argv = stream.read(16385)
        if len(env) > 262144 or len(argv) > 16384:
            raise ValueError("proc identity cap")
        after = stat_identity(pid)
        if after is None:
            return None
        if after["start_ticks"] != row["start_ticks"]:
            raise ValueError("proc birth identity changed while reading")
        if after["state"] == "Z":
            return {**after, "argv": [], "env": {}}
        return {**after,
                "argv": argv.rstrip(b"\0").decode(errors="replace").split("\0"),
                "env": dict(item.decode(errors="replace").split("=", 1)
                            for item in env.split(b"\0") if b"=" in item)}
    except (FileNotFoundError, ProcessLookupError):
        return None


def scan_owned(nonce, groups=(), known=None):
    rows, errors = [], []
    known = known or {}
    for root in Path("/proc").iterdir():
        if not root.name.isdigit():
            continue
        pid = int(root.name)
        before = stat_identity(pid)
        if before is None or before["state"] == "Z":
            continue
        scoped = (before["pgrp"] in groups or
                  known.get(pid, {}).get("start_ticks") == before["start_ticks"])
        try:
            row = identity(pid)
        except PermissionError as error:
            after = stat_identity(pid)
            if scoped and after and after["state"] != "Z":
                errors.append({"stage": "scan_owned", "process": after,
                               "error_type": type(error).__name__, "error": str(error)[:512]})
            continue
        if row is None or row["state"] == "Z":
            continue
        if row["env"].get(KEY) == nonce:
            rows.append(row)
        elif scoped:
            errors.append({"stage": "scan_owned", "process": before,
                           "error_type": "OwnershipError", "error": "live scoped member has different nonce"})
    return rows, errors


def owned(nonce):
    return scan_owned(nonce)[0]


def public(row):
    return {**{k: v for k, v in row.items() if k != "env"},
            "marker": row["env"].get(KEY),
            "gpu_masks": {k: row["env"].get(k) for k in
                          ["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"]}}


def launcher_snapshot(child, nonce):
    row, error = None, None
    try:
        row = identity(child.pid)
    except (OSError, ValueError) as failure:
        error = {"type": type(failure).__name__, "message": str(failure)[:512]}
    stat, stat_error = None, None
    try:
        stat = stat_identity(child.pid)
    except OSError as failure:
        stat_error = {"type": type(failure).__name__, "message": str(failure)[:512]}
    code = child.poll()
    terminal = code is not None
    if terminal:
        code = child.wait()
    return {"pid": child.pid, "process": public(row) if row else None,
            "stat": stat, "stat_error": stat_error,
            "identity_error": error, "returncode": code,
            "checks": {"nonce": row["env"].get(KEY) == nonce if row and row["state"] != "Z" else None,
                       "pgrp": stat["pgrp"] == child.pid if stat else None},
            "terminal_reaped": terminal,
            "live_identity_verified": bool(not terminal and row and row["state"] != "Z" and
                                           row["env"].get(KEY) == nonce and row["pgrp"] == child.pid)}


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
    actions, errors = [], []
    known = {child.pid: child.smg_identity for child in leaders
             if getattr(child, "smg_identity", None) is not None}
    groups = {child.pid for child in leaders}

    def remember(error):
        if error not in errors and len(errors) < 32:
            errors.append(error)

    def scan():
        rows, unknown = scan_owned(nonce, groups, known)
        for row in rows:
            known[row["pid"]] = row
        for error in unknown:
            remember(error)
        return rows, unknown

    for sig, grace in [(signal.SIGTERM, 5), (signal.SIGKILL, 5)]:
        rows, unknown = scan()
        for row in rows:
            if row["pgrp"] == row["pid"]:
                try:
                    if signal_group(row, sig, nonce):
                        actions.append({"signal": int(sig), "owned_process_group": row["pid"], "start_ticks": row["start_ticks"]})
                except ProcessLookupError:
                    pass
                except (PermissionError, RuntimeError) as error:
                    remember({"stage": "signal_group", "process": public(row),
                              "error_type": type(error).__name__, "error": str(error)[:512]})
        for row in rows:
            try:
                signal_exact(row, sig, nonce)
                actions.append({"signal": int(sig), **public(row)})
            except ProcessLookupError:
                pass
            except (PermissionError, RuntimeError) as error:
                remember({"stage": "signal_exact", "process": public(row),
                          "error_type": type(error).__name__, "error": str(error)[:512]})
        end = min(deadline, time.monotonic() + grace)
        while time.monotonic() < end:
            remaining, unknown = scan()
            if not remaining and not unknown:
                break
            time.sleep(min(0.1, max(0, end - time.monotonic())))
        for child in leaders:
            if child.poll() is not None:
                child.wait()
        remaining, unknown = scan()
        if not remaining and not unknown:
            return {"passed": not errors, "signals": actions, "remaining": [], "errors": errors}
        if time.monotonic() >= deadline:
            break
    return {"passed": False, "signals": actions,
            "remaining": [public(row) for row in remaining], "unresolved": unknown, "errors": errors}


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
