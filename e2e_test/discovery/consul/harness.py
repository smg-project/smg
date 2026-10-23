"""The Consul-discovery e2e's own pieces: a real Consul agent in a container,
and gateways discovering from it. The rest is in e2e_test/discovery/common.py."""

from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
import time
import uuid
from pathlib import Path

import requests

from ..common import CONVERGE, Gateway, free_port

# The service every test's workers are instances of.
SERVICE = "smg-worker"
# The management token of an agent started with ACLs on.
MANAGEMENT_TOKEN = "e2e-management"
# The files of an agent serving HTTPS, by the variable that carries each into
# its container.
SERVER_TLS_FILES = {
    "E2E_CA": "consul-agent-ca.pem",
    "E2E_CERT": "dc1-server-consul-0.pem",
    "E2E_KEY": "dc1-server-consul-0-key.pem",
}


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    completed = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=180)
    if check:
        assert completed.returncode == 0, f"docker {args}: {completed.stderr}"
    return completed


class ConsulAgent:
    """A dev-mode Consul agent in a container (all its state in memory, so a
    restart starts it empty), its HTTP API published on a local port.

    Instances are registered through the catalog API, as nodes without an
    agent are, so a test sets their health itself. With `acl`, ACLs are on
    and deny by default; with `tls_dir`, the API is HTTPS only and asks for a
    client certificate, the files in `tls_dir` made by `make_certificates`.
    """

    def __init__(
        self,
        image: str,
        logs: Path | None,
        *,
        acl: bool = False,
        tls_dir: Path | None = None,
    ):
        self.image = image
        self.logs = logs
        self.acl = acl
        self.tls_dir = tls_dir
        self.name = f"smg-consul-{uuid.uuid4().hex[:8]}"
        self.port = free_port()
        scheme = "https" if tls_dir else "http"
        self.address = f"{scheme}://127.0.0.1:{self.port}"
        self.session = requests.Session()
        if acl:
            self.session.headers["X-Consul-Token"] = MANAGEMENT_TOKEN
        if tls_dir:
            self.session.verify = str(tls_dir / "consul-agent-ca.pem")
            self.session.cert = (
                str(tls_dir / "dc1-client-consul-0.pem"),
                str(tls_dir / "dc1-client-consul-0-key.pem"),
            )

    def local_config(self) -> dict:
        config: dict = {}
        if self.acl:
            config["acl"] = {
                "enabled": True,
                "default_policy": "deny",
                "tokens": {"initial_management": MANAGEMENT_TOKEN, "agent": MANAGEMENT_TOKEN},
            }
        if self.tls_dir:
            # gRPC, which dev mode would serve over TLS too, is not used.
            config["ports"] = {"http": -1, "https": 8500, "grpc": -1, "grpc_tls": -1}
            config["tls"] = {
                "defaults": {
                    "ca_file": "/certs/consul-agent-ca.pem",
                    "cert_file": "/certs/dc1-server-consul-0.pem",
                    "key_file": "/certs/dc1-server-consul-0-key.pem",
                    "verify_incoming": True,
                },
            }
        return config

    def start(self) -> None:
        run = ["run", "-d", "--name", self.name, "-p", f"127.0.0.1:{self.port}:8500"]
        if config := self.local_config():
            run += ["-e", f"CONSUL_LOCAL_CONFIG={json.dumps(config)}"]
        agent = ["agent", "-dev", "-client", "0.0.0.0", "-node", "agent-node"]
        if self.tls_dir:
            # The certificates arrive in the environment and are written in
            # the container, so no mount (and no host's file labels) stands
            # between the agent and them; then the image's entrypoint runs.
            writes = []
            for variable, name in SERVER_TLS_FILES.items():
                run += ["-e", f"{variable}={(self.tls_dir / name).read_text()}"]
                writes.append(f'printf "%s\\n" "${variable}" > /certs/{name}')
            script = " && ".join(["mkdir -p /certs", *writes, 'exec docker-entrypoint.sh "$@"'])
            run += ["--entrypoint", "sh", self.image, "-c", script, "sh", *agent]
        else:
            run += [self.image, *agent]
        docker(*run)
        self.wait_until_up()

    def wait_until_up(self, within: float = 60.0) -> None:
        deadline = time.monotonic() + within
        while True:
            try:
                leader = self.session.get(f"{self.address}/v1/status/leader", timeout=2)
                if leader.ok and ":" in leader.json():
                    return
            except (requests.RequestException, ValueError):
                pass
            assert time.monotonic() < deadline, f"Consul never came up: {self.log_tail()}"
            time.sleep(0.5)

    def log(self, *args: str) -> str:
        """What the agent wrote, on stdout and stderr alike."""
        logs = docker("logs", *args, self.name, check=False)
        return logs.stdout + logs.stderr

    def log_tail(self, lines: int = 30) -> str:
        return self.log("--tail", str(lines))

    def stop_container(self) -> None:
        """Stop Consul, leaving the container to start again."""
        docker("stop", "-t", "1", self.name)

    def start_container(self) -> None:
        """Start the stopped Consul again: empty, as dev mode keeps nothing."""
        docker("start", self.name)
        self.wait_until_up()

    def remove(self) -> None:
        if self.logs is not None:
            self.logs.mkdir(parents=True, exist_ok=True)
            (self.logs / f"{self.name}.log").write_text(self.log())
        docker("rm", "-f", self.name, check=False)

    def put(self, path: str, body: dict) -> requests.Response:
        response = self.session.put(f"{self.address}{path}", json=body, timeout=10)
        assert response.ok, f"PUT {path}: {response.status_code} {response.text}"
        return response

    def register(
        self,
        node: str,
        service_id: str,
        port: int,
        *,
        node_address: str = "127.0.0.1",
        address: str = "",
        meta: dict[str, str] | None = None,
        status: str = "passing",
    ) -> None:
        """Register, or update, one instance on `node`, its one check in
        `status`."""
        self.put(
            "/v1/catalog/register",
            {
                "Node": node,
                "Address": node_address,
                "Service": {
                    "ID": service_id,
                    "Service": SERVICE,
                    "Address": address,
                    "Port": port,
                    "Meta": meta or {},
                },
                "Check": {
                    "CheckID": f"{service_id}-engine",
                    "Name": "engine",
                    "Status": status,
                    "ServiceID": service_id,
                },
            },
        )

    def deregister(self, node: str, service_id: str) -> None:
        self.put("/v1/catalog/deregister", {"Node": node, "ServiceID": service_id})

    def clear(self) -> None:
        """Deregister every node but the agent's own."""
        nodes = self.session.get(f"{self.address}/v1/catalog/nodes", timeout=10).json()
        for node in nodes:
            if node["Node"] != "agent-node":
                self.put("/v1/catalog/deregister", {"Node": node["Node"]})

    def token(self, rules: str) -> str:
        """A new ACL token holding `rules`."""
        name = f"policy-{uuid.uuid4().hex[:8]}"
        self.put("/v1/acl/policy", {"Name": name, "Rules": rules})
        return self.put("/v1/acl/token", {"Policies": [{"Name": name}]}).json()["SecretID"]


def make_certificates(image: str, parent: Path) -> Path:
    """A CA, a server certificate (for localhost and 127.0.0.1 among others)
    and a client certificate, made by Consul's own `consul tls` commands, in
    a new directory in `parent`, readable by the agent's user and the test's."""
    # Written out as a tar on stdout, so nothing is mounted or copied.
    script = (
        "mkdir -p /certs && cd /certs && consul tls ca create >&2"
        " && consul tls cert create -server >&2 && consul tls cert create -client >&2"
        " && chmod a+r * && tar -cf - ."
    )
    made = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "sh", image, "-c", script],
        capture_output=True,
        timeout=180,
    )
    assert made.returncode == 0, f"consul tls: {made.stderr.decode(errors='replace')}"
    directory = parent / "certs"
    with tarfile.open(fileobj=io.BytesIO(made.stdout)) as archive:
        archive.extractall(directory, filter="data")
    return directory


def consul_gateway(
    agent: ConsulAgent, log_path: Path, env: dict[str, str] | None = None
) -> Gateway:
    """A gateway, not yet started, discovering the instances of `SERVICE`
    from `agent`, with `env`'s Consul settings and none from this process."""
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("CONSUL_")
    }
    return Gateway(
        lambda check_interval_secs: [
            *("--discovery-provider", "consul"),
            *("--consul-service", SERVICE),
            *("--consul-address", agent.address),
            *("--discovery-check-interval-secs", str(check_interval_secs)),
        ],
        log_path,
        env={**environment, **(env or {})},
    )


def wait_until(what: str, predicate, within: float = CONVERGE) -> None:
    deadline = time.monotonic() + within
    while not predicate():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.25)
