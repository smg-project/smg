"""Bind a live fixture-owned gateway to the new producer's mapped Rust extension."""

import hashlib
import json
import os
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(gateway, directory):
    process = gateway.process
    require(process is not None and process.poll() is None, "gateway is not alive")
    root = Path("/proc") / str(process.pid)
    fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    require(int(fields[1]) == os.getpid(), "gateway is not a fixture Popen child")
    argv = [part.decode() for part in (root / "cmdline").read_bytes().split(b"\0") if part]
    launch = list(process.args)
    require(argv == launch or argv == ["smg::router"], "gateway argv/title differs")
    require(launch[1:3] == ["-m", "smg.launch_router"], "original gateway launcher differs")
    marker = os.environ["E2E_JOB_MARKER"]
    require(
        ("E2E_JOB_MARKER=" + marker).encode() in (root / "environ").read_bytes().split(b"\0"),
        "gateway marker differs",
    )
    mapped = set()
    for line in (root / "maps").read_text().splitlines():
        values = line.split(maxsplit=5)
        if len(values) == 6 and "x" in values[1] and Path(values[5]).name.startswith("smg_rs"):
            mapped.add(Path(values[5]).resolve())
    require(len(mapped) == 1, "missing/ambiguous actually loaded gateway native")
    path = mapped.pop()
    expected = json.loads(Path(__file__).with_name("inputs.json").read_text())["gateway_native"]
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    require(
        path.stat().st_size == expected["bytes"] and digest == expected["sha256"],
        "gateway native differs from new dbc4 producer",
    )
    stdout = (root / "fd/1").stat()
    log = Path(gateway.log_path).stat()
    require(
        (stdout.st_dev, stdout.st_ino) == (log.st_dev, log.st_ino), "gateway stdout/log differs"
    )
    record = {
        "pid": process.pid,
        "parent_pid": int(fields[1]),
        "start_ticks": fields[19],
        "argv": argv,
        "launch_argv": launch,
        "job_marker": marker,
        "native_path": str(path),
        "native_bytes": path.stat().st_size,
        "native_sha256": digest,
        "log": str(gateway.log_path),
    }
    target = directory / "gateways"
    target.mkdir(exist_ok=True)
    (target / f"{process.pid}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def from_fixture(gateway):
    directory = os.environ.get("E2E_EPD_RPC_PROOF_DIRECTORY")
    if directory:
        verify(gateway, Path(directory))
