"""The real-Slurm smoke's pieces: the container running a one-node Slurm, and
the gateway running inside it. The rest is in e2e_test/discovery_common.py."""

from __future__ import annotations

import subprocess
import time
import uuid
from pathlib import Path

import requests

from ..discovery_common import Gateway, free_port

# Where the gateway listens inside the container, and where it logs.
GATEWAY_PORT = 3000
GATEWAY_LOG = "/var/log/smg/gateway.log"


class SlurmNode:
    """One container from e2e_test/slurm_smoke/Dockerfile: munge, slurmctld and
    one slurmd, as root. Commands run in it through `docker exec`."""

    def __init__(self, image: str, logs: Path | None):
        self.image = image
        self.logs = logs
        self.name = f"smg-slurm-smoke-{uuid.uuid4().hex[:8]}"
        # The host port the gateway's port is published on.
        self.gateway_port = free_port()

    def start(self) -> None:
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                self.name,
                "--hostname",
                "slurmnode",
                "-p",
                f"127.0.0.1:{self.gateway_port}:{GATEWAY_PORT}",
                self.image,
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        deadline = time.monotonic() + 120
        while "idle" not in self.exec("sinfo", "-h", "-n", "slurmnode", "-o", "%T", check=False):
            assert time.monotonic() < deadline, f"the node never became idle: {self.exec('sinfo')}"
            time.sleep(1)

    def exec(self, *args: str, check: bool = True, detach: bool = False) -> str:
        command = ["docker", "exec", *(["-d"] if detach else []), self.name, *args]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if check:
            assert completed.returncode == 0, f"{args}: {completed.stderr}"
        return completed.stdout

    def submit(self, name: str, command: str) -> int:
        """Submit a batch job running `command`; its output goes to the logs."""
        out = self.exec(
            *("sbatch", "--parsable", "-J", name),
            *("-o", "/var/log/slurm/job-%j.out", "--wrap", command),
        )
        return int(out.strip().split(";")[0])

    def record(self, step: str) -> None:
        """Keep Slurm's raw answers at this step beside the logs."""
        if self.logs is None:
            return
        self.logs.mkdir(parents=True, exist_ok=True)
        for name, args in [
            ("squeue", ("squeue", "--json")),
            ("nodes", ("scontrol", "--json", "show", "nodes")),
        ]:
            (self.logs / f"{step}-{name}.json").write_text(self.exec(*args, check=False))

    def stop(self) -> None:
        if self.logs is not None:
            self.logs.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["docker", "cp", f"{self.name}:/var/log", str(self.logs / "container")],
                capture_output=True,
                timeout=120,
            )
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, timeout=120)


class ContainerGateway(Gateway):
    """The smg in the container, reached through its published port."""

    def __init__(self, node: SlurmNode):
        super().__init__(lambda _: [], Path("/dev/null"))
        self.node = node

    def start(self, *extra_args: str, check_interval_secs: int = 2) -> None:
        flags = [
            *("--host", "0.0.0.0", "--port", str(GATEWAY_PORT)),
            *("--discovery-provider", "slurm"),
            *("--discovery-check-interval-secs", str(check_interval_secs)),
            *("--policy", "round_robin"),
            *("--health-check-interval-secs", "1"),
            *("--health-failure-threshold", "2"),
            *("--health-success-threshold", "1"),
            *("--drain-settle-secs", "0"),
            *("--worker-startup-timeout-secs", "10"),
            *extra_args,
        ]
        self.node.exec(
            "sh",
            "-c",
            f"exec smg {' '.join(flags)} > {GATEWAY_LOG} 2>&1",
            detach=True,
        )
        self.base_url = f"http://127.0.0.1:{self.node.gateway_port}"
        deadline = time.monotonic() + 60
        while True:
            try:
                if requests.get(f"{self.base_url}/health", timeout=2).ok:
                    return
            except requests.RequestException:
                pass
            assert time.monotonic() < deadline, f"smg never became healthy; {self.log_tail()}"
            time.sleep(0.5)

    def stop(self) -> None:
        self.node.exec("pkill", "-x", "smg", check=False)

    def log_tail(self, lines: int = 40) -> str:
        return self.node.exec("tail", "-n", str(lines), GATEWAY_LOG, check=False)
