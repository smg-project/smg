"""The Slurm-discovery e2e's own pieces: a real Slurm 23.11 cluster in
containers, and the gateway running on its controller. The rest is in
e2e_test/discovery/common.py."""

from __future__ import annotations

import shlex
import subprocess
import time
import uuid
from pathlib import Path

import requests

from ..common import MODEL, Gateway, free_port

# The nodes of the cluster (slurm.conf), each a container reached by its name.
NODES = [f"node{n}" for n in range(1, 7)]
# Where every worker listens, on its node.
WORKER_PORT = 8000
# Where the gateway listens inside the controller.
GATEWAY_PORT = 3000


def worker_url(node: str) -> str:
    return f"http://{node}:{WORKER_PORT}"


def worker_command(*, delay_secs: int = 0) -> str:
    """A job's command: one mock-worker on every node of its allocation,
    started after `delay_secs`."""
    serve = (
        f"srun --overlap --ntasks-per-node=1 mock-worker --host 0.0.0.0"
        f" --http-base-port {WORKER_PORT} --http-count 1 --model {MODEL}"
    )
    return f"sleep {delay_secs}; exec {serve}" if delay_secs else f"exec {serve}"


class SlurmCluster:
    """A controller (munge, slurmctld) and the nodes (munge, slurmd), one
    container each on a network of their own, from the image built from this
    directory's Dockerfile. Commands run on the controller through
    `docker exec`."""

    def __init__(self, image: str, logs: Path | None):
        self.image = image
        self.logs = logs
        self.prefix = f"smg-slurm-{uuid.uuid4().hex[:8]}"
        self.network = self.prefix
        self.controller = f"{self.prefix}-slurmctl"
        # The host port the gateway's port is published on.
        self.gateway_port = free_port()

    def _docker(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        completed = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=180)
        if check:
            assert completed.returncode == 0, f"docker {args}: {completed.stderr}"
        return completed

    def start(self) -> None:
        self._docker("network", "create", self.network)
        # slurmd keeps retrying until the controller answers, so the nodes
        # go first; the controller resolves every node's name when it starts.
        for node in NODES:
            self._docker(
                *("run", "-d", "--privileged", "--name", f"{self.prefix}-{node}"),
                *("--hostname", node, "--network", self.network, "--network-alias", node),
                self.image,
                "node",
            )
        self._docker(
            *("run", "-d", "--name", self.controller, "--hostname", "slurmctl"),
            *("--network", self.network, "--network-alias", "slurmctl"),
            *("-p", f"127.0.0.1:{self.gateway_port}:{GATEWAY_PORT}"),
            self.image,
            "controller",
        )
        self.wait_for_idle_nodes()

    def node_states(self) -> dict[str, str]:
        lines = self.exec("sinfo", "-h", "-N", "-o", "%N %T", check=False).splitlines()
        return dict(line.split() for line in lines if len(line.split()) == 2)

    def wait_for_idle_nodes(self, within: float = 180.0) -> None:
        deadline = time.monotonic() + within
        while (states := self.node_states()) != {node: "idle" for node in NODES}:
            assert time.monotonic() < deadline, f"nodes never became idle: {states}"
            time.sleep(1)

    def wait_for_responding_nodes(self, within: float = 180.0) -> None:
        """Every node answering the controller again, busy or not: after a
        controller restart."""
        deadline = time.monotonic() + within
        while True:
            states = self.node_states()
            if set(states) == set(NODES) and all(
                state in ("idle", "allocated", "mixed") for state in states.values()
            ):
                return
            assert time.monotonic() < deadline, f"nodes never answered again: {states}"
            time.sleep(1)

    def exec(self, *args: str, check: bool = True, detach: bool = False) -> str:
        command = ["exec", *(["-d"] if detach else []), self.controller, *args]
        completed = self._docker(*command, check=False)
        if check:
            assert completed.returncode == 0, f"{args}: {completed.stderr}"
        return completed.stdout

    def submit(
        self,
        name: str,
        command: str,
        *,
        nodes: str,
        user: str | None = None,
        account: str | None = None,
        partition: str | None = None,
    ) -> int:
        """Submit a batch job named `name` on the nodes `nodes` (a hostlist),
        running `command`; its output goes to the logs."""
        sbatch = [
            *("sbatch", "--parsable", "-J", name, "-w", nodes),
            *("-N", str(len(self.exec("scontrol", "show", "hostnames", nodes).split()))),
            *("-o", "/var/log/jobs/job-%j.out"),
            *(("-A", account) if account else ()),
            *(("-p", partition) if partition else ()),
            *("--wrap", command),
        ]
        if user:
            sbatch = ["runuser", "-u", user, "--", *sbatch]
        out = self.exec("sh", "-c", f"cd /tmp && {shlex.join(sbatch)}")
        return int(out.strip().split(";")[0])

    def job_state(self, job: int) -> str:
        return self.exec("squeue", "-h", "-j", str(job), "-o", "%T", check=False).strip()

    def wait_for_state(self, job: int, state: str, within: float = 120.0) -> None:
        deadline = time.monotonic() + within
        while self.job_state(job) != state:
            assert time.monotonic() < deadline, f"job {job} never reached {state}"
            time.sleep(1)

    def cancel_all(self) -> None:
        """Cancel every job, and wait for the queue to drain and the nodes to
        be idle, so a test starts on an empty cluster."""
        for user in ("root", "alice"):
            self.exec("scancel", "-u", user, check=False)
        deadline = time.monotonic() + 120
        while self.exec("squeue", "-h", check=False).strip():
            assert time.monotonic() < deadline, f"jobs left: {self.exec('squeue')}"
            time.sleep(1)
        self.wait_for_idle_nodes()

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
        containers = [self.controller, *(f"{self.prefix}-{node}" for node in NODES)]
        if self.logs is not None:
            self.logs.mkdir(parents=True, exist_ok=True)
            for container in containers:
                self._docker("cp", f"{container}:/var/log", str(self.logs / container), check=False)
        self._docker("rm", "-f", *containers, check=False)
        self._docker("network", "rm", self.network, check=False)


class ControllerGateway(Gateway):
    """The smg on the controller, reached through its published port."""

    def __init__(self, cluster: SlurmCluster):
        super().__init__(lambda _: [], Path("/dev/null"))
        self.cluster = cluster
        # Each test's gateway logs to its own file, kept across a restart.
        self.log = f"/var/log/smg/gateway-{uuid.uuid4().hex[:8]}.log"

    def start(self, *extra_args: str, check_interval_secs: int = 2) -> None:
        flags = [
            *("--host", "0.0.0.0", "--port", str(GATEWAY_PORT)),
            *("--discovery-provider", "slurm", "--slurm-port", str(WORKER_PORT)),
            *("--discovery-check-interval-secs", str(check_interval_secs)),
            *("--policy", "round_robin"),
            *("--health-check-interval-secs", "1"),
            *("--health-failure-threshold", "2"),
            *("--health-success-threshold", "1"),
            *("--drain-settle-secs", "0"),
            *("--worker-startup-timeout-secs", "10"),
            *extra_args,
        ]
        self.cluster.exec(
            "sh", "-c", f"exec smg {shlex.join(flags)} >> {self.log} 2>&1", detach=True
        )
        self.base_url = f"http://127.0.0.1:{self.cluster.gateway_port}"
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
        self.cluster.exec("pkill", "-x", "smg", check=False)
        # The next start binds the same port.
        deadline = time.monotonic() + 30
        while self.cluster.exec("pgrep", "-x", "smg", check=False).strip():
            assert time.monotonic() < deadline, "smg did not exit"
            time.sleep(0.2)

    def log_tail(self, lines: int = 40) -> str:
        return self.cluster.exec("tail", "-n", str(lines), self.log, check=False)

    def wait_for_log(self, needle: str, what: str, within: float = 60.0) -> None:
        deadline = time.monotonic() + within
        while needle not in self.cluster.exec("cat", self.log, check=False):
            assert time.monotonic() < deadline, f"timed out waiting for {what}; {self.log_tail()}"
            time.sleep(0.5)
