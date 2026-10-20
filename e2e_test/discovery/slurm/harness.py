"""The Slurm-discovery e2e's own pieces: the state the stand-in `squeue` and
`scontrol` answer from, and the flags that point a gateway at them. The rest
is in e2e_test/discovery/common.py."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..common import Gateway

FAKE_BIN = Path(__file__).parent / "fake_bin"


class FakeSlurm:
    """The state file the stand-in commands answer from, and the record of
    every call they took. A test edits `state` and calls `write`, as Slurm's
    own state would change between polls."""

    def __init__(self, directory: Path):
        self.path = directory / "slurm-state.json"
        self.calls_path = directory / "slurm-calls.jsonl"
        self.calls_path.touch()
        self.state: dict = {
            "user": "smg",
            "cluster": "e2e",
            "jobs": [],
            "hostlists": {},
            "nodes": {},
            "fail": [],
        }
        self.write()

    def write(self) -> None:
        """Replace the state at once, so a command never reads half of it."""
        staged = self.path.with_suffix(".staged")
        staged.write_text(json.dumps(self.state))
        staged.replace(self.path)

    def node(self, name: str, address: str = "", hostname: str = "") -> None:
        self.state["nodes"][name] = {"address": address, "hostname": hostname}

    def job(self, job_id: int, name: str, hostlist: str, nodes: list[str], **fields) -> None:
        """A job on `nodes`, which Slurm writes as `hostlist`. Other fields
        (state, user, account, partition, start_time, restart_cnt) default as
        the fake squeue defaults them."""
        self.state["hostlists"][hostlist] = nodes
        self.state["jobs"] = [job for job in self.state["jobs"] if job["id"] != job_id]
        self.state["jobs"].append({"id": job_id, "name": name, "nodes": hostlist, **fields})

    def update_job(self, job_id: int, **fields) -> None:
        for job in self.state["jobs"]:
            if job["id"] == job_id:
                job.update(fields)

    def remove_job(self, job_id: int) -> None:
        self.state["jobs"] = [job for job in self.state["jobs"] if job["id"] != job_id]

    def calls(self, command: str | None = None) -> list[dict]:
        lines = self.calls_path.read_text().splitlines()
        calls = [json.loads(line) for line in lines if line]
        return [call for call in calls if command is None or call["command"] == command]

    def env(self) -> dict[str, str]:
        """The gateway's environment: the stand-ins first on its PATH."""
        return {
            **os.environ,
            "PATH": f"{FAKE_BIN}{os.pathsep}{os.environ.get('PATH', '')}",
            "SMG_FAKE_SLURM": str(self.path),
            "SMG_FAKE_SLURM_CALLS": str(self.calls_path),
        }


def slurm_gateway(slurm: FakeSlurm, port: int, log_path: Path) -> Gateway:
    """A gateway, not yet started, discovering Slurm workers on `port`. A
    test adds the job-name mappings and any scope when it starts it."""
    return Gateway(
        lambda check_interval_secs: [
            "--discovery-provider",
            "slurm",
            "--slurm-port",
            str(port),
            "--discovery-check-interval-secs",
            str(check_interval_secs),
        ],
        log_path,
        env=slurm.env(),
    )
