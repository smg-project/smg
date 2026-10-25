"""Wait only for periodic metrics from the original nominal P roles; no requests."""

import hashlib
import json
import os
import re
import stat
import time
from pathlib import Path

from .mooncake_topology import CASE
from .owned import require

LIMIT = 12.0
BYTE_CAP = 1024 * 1024


def identity(row):
    proc = Path("/proc") / str(row["pid"])
    first = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    require(first[0] != "Z" and (int(first[1]), first[19]) ==
            (row["parent_pid"], row["start_ticks"]), "flush process identity changed")
    marker = ("E2E_JOB_MARKER=" + os.environ["E2E_JOB_MARKER"]).encode()
    require(marker in (proc / "environ").read_bytes().split(b"\0"), "flush marker differs")
    require([x.decode() for x in (proc / "cmdline").read_bytes().split(b"\0") if x] ==
            row["argv"], "flush argv differs")
    path = Path(row["log"])
    require(not path.is_symlink() and stat.S_ISREG(path.stat().st_mode), "flush log is not regular")
    require((proc / "fd/1").resolve() == path.resolve(), "flush stdout path differs")
    fd_info = (proc / "fd/1").stat()
    current = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    require(current[0] != "Z" and (current[1], current[19]) == (first[1], first[19]),
            "flush process changed while reading ownership")
    info = path.stat()
    require((fd_info.st_dev, fd_info.st_ino) == (info.st_dev, info.st_ino), "flush stdout inode differs")
    return path, info


def positive(raw, pid):
    prefix = rb"^\(MainProcess pid=" + str(pid).encode() + rb"\) .*\[metrics\.py:103\] KV Transfer metrics: Num successful transfers=([1-9][0-9]*),"
    return [int(m.group(1)) for line in raw.splitlines(keepends=True)
            if line.endswith(b"\n") and (m := re.search(prefix, line))]


def selected(runtime):
    gateways = [e["identity"] for e in runtime.events if e["nodeid"] == CASE
                and e["kind"] == "gateway" and e.get("identity", {}).get("prefill_urls")]
    require(gateways and len(gateways[-1]["prefill_urls"]) ==
            len(gateways[-1]["decode_urls"]) == 2, "flush requires original 2P2D gateway")
    urls = gateways[-1]["prefill_urls"]
    rows = {}
    for e in runtime.events:
        row = e.get("identity", {})
        if e["nodeid"] == CASE and e["kind"] == "worker" and row.get("URL") in urls:
            rows[row["URL"]] = row
    require(set(rows) == set(urls), "flush missing P identities")
    for row in rows.values():
        require(row["role"] == "prefill" and row["TP"] == "1" and
                row["kv_config"]["kv_connector"] == "MooncakeConnector" and
                row["parent_pid"] == os.getpid(), "flush P role differs")
    require(len({(r["pid"], r["start_ticks"]) for r in rows.values()}) == 2,
            "flush P identities overlap")
    return list(rows.values())


def existing(runtime, rows):
    sources = {str(Path(r["log"]).resolve()): r for r in rows}
    offsets, streams = {}, {}
    for e in runtime.events:
        if e["kind"] != "log":
            continue
        name = e["artifact"]
        require(Path(name).name == name and name not in {".", ".."}, "flush artifact name")
        start = offsets.get(name, 0)
        offsets[name] = start + e["bytes"]
        source = str(Path(e["source"]).resolve())
        if e["nodeid"] != CASE or source not in sources:
            continue
        require(0 < e["bytes"] <= 16 * 1024**2, "flush existing increment cap")
        with (runtime.directory / name).open("rb") as stream:
            stream.seek(start)
            raw = stream.read(e["bytes"])
        require(len(raw) == e["bytes"] and hashlib.sha256(raw).hexdigest() == e["sha256"],
                "flush existing log SHA differs")
        data = streams.setdefault(source, bytearray())
        data.extend(raw)
        require(len(data) <= 32 * 1024**2, "flush existing source cap")
    return any(positive(raw, sources[path]["pid"]) for path, raw in streams.items())


def observe(runtime, seconds=LIMIT):
    require(runtime.nodeid == CASE and 0 < seconds <= LIMIT, "flush scope or time bound")
    attempt = getattr(runtime, "passive_flush_attempts", 0) + 1
    require(attempt <= 3, "flush exceeds original reruns2")
    runtime.passive_flush_attempts = attempt
    receipt = runtime.directory / ("mooncake-periodic-flush-attempt-" + str(attempt) + ".json")
    require(not receipt.exists(), "flush receipt already exists")
    started = time.monotonic()
    deadline = started + seconds
    result = {"started_monotonic": started, "deadline_monotonic": deadline,
              "max_seconds": seconds, "scope": "same-P periodic interval aggregate; not lifetime or per-RID",
              "requests_added": 0, "status": "FAIL", "roles": [], "reads": []}
    states = []
    try:
        rows = selected(runtime)
        for row in rows:
            path, info = identity(row)
            states.append({"row": row, "path": path, "inode": (info.st_dev, info.st_ino),
                           "offset": 0, "buffer": b"", "bytes": 0})
            offsets = [v for k, v in runtime.offsets.items() if Path(k).resolve() == path.resolve()]
            require(len(offsets) == 1 and offsets[0][0] == info.st_ino and
                    offsets[0][1] <= info.st_size, "flush captured baseline differs")
            states[-1]["offset"] = offsets[0][1]
            result["roles"].append({"pid": row["pid"], "start_ticks": row["start_ticks"],
                                    "role": "prefill", "URL": row["URL"], "log": str(path),
                                    "inode": list(states[-1]["inode"]), "start_offset": states[-1]["offset"]})
        if existing(runtime, rows):
            require(time.monotonic() <= deadline, "flush deadline")
            result["status"] = "ALREADY_OBSERVED_IN_ORIGINAL_CASE"
            return
        while time.monotonic() < deadline:
            found = False
            for state in states:
                require(time.monotonic() < deadline, "flush deadline")
                path, info = identity(state["row"])
                require((info.st_dev, info.st_ino) == state["inode"] and
                        info.st_size >= state["offset"], "flush log rotated or truncated")
                with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
                    opened = os.fstat(stream.fileno())
                    require((opened.st_dev, opened.st_ino) == state["inode"], "flush open inode differs")
                    stream.seek(state["offset"])
                    raw = stream.read(min(65536, BYTE_CAP - state["bytes"] + 1))
                if not raw:
                    continue
                start = state["offset"]
                state["offset"] += len(raw)
                state["bytes"] += len(raw)
                fragment = runtime.directory / ("metrics-flush-" + str(attempt) + "-" + str(state["row"]["pid"]) + ".log")
                with fragment.open("ab") as stream:
                    stream.write(raw)
                result["reads"].append({"pid": state["row"]["pid"], "start": start,
                                        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                                        "raw_fragment": fragment.name})
                require(state["bytes"] <= BYTE_CAP, "flush byte cap")
                state["buffer"] += raw
                found = bool(positive(state["buffer"], state["row"]["pid"])) or found
                state["buffer"] = state["buffer"].rsplit(b"\n", 1)[-1]
                require(len(state["buffer"]) <= 65536, "flush line cap")
                identity(state["row"])
            runtime.capture_logs()
            require(not runtime.errors, "flush capture errors")
            if found:
                require(time.monotonic() <= deadline, "flush deadline")
                for state in states:
                    identity(state["row"])
                result["status"] = "OBSERVED_AFTER_CALL_BEFORE_NEXT_CASE"
                return
            time.sleep(min(0.25, max(0, deadline - time.monotonic())))
        raise ValueError("flush timed out without new positive P interval metrics")
    except BaseException as error:
        result["error"] = str(error)[:2048]
        raise
    finally:
        result["ended_monotonic"] = time.monotonic()
        result["end_offsets"] = {str(s["row"]["pid"]): s["offset"] for s in states}
        with receipt.open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
