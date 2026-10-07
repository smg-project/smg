"""Fixtures for the kind-cluster service discovery e2e.

The session owns one kind cluster and one smg process; tests drive the
cluster with kubectl and assert through the gateway's /workers endpoint.
Gated behind SMG_KIND_E2E=1 (and skipped when kind/kubectl are absent) so
regular e2e sessions never attempt cluster creation. Linux only: the host
must reach the kind node IP to probe hostNetwork worker ports.

Run locally:
    pip install -e e2e_test && cargo build -p smg && \
        SMG_KIND_E2E=1 pytest e2e_test/kind_discovery -m kind
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest
import requests

CLUSTER = "smg-discovery-e2e"
SMG_PORT = 3009
FALLBACK_PORT = 28090
HERE = Path(__file__).parent


def kubectl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["kubectl", *args], check=check, capture_output=True, text=True)


class KindGateway:
    """Handle on the smg process under test plus /workers assertions."""

    def __init__(self, binary: Path):
        self.binary = binary
        self.base_url = f"http://127.0.0.1:{SMG_PORT}"
        self.log_path = Path(tempfile.mkstemp(prefix="smg-kind-e2e-", suffix=".log")[1])
        self.proc: subprocess.Popen | None = None

    def start(self, extra_args: tuple[str, ...] = ()) -> None:
        log = open(self.log_path, "a")
        self.proc = subprocess.Popen(
            [
                str(self.binary),
                "--host",
                "127.0.0.1",
                "--port",
                str(SMG_PORT),
                # The legacy spelling; in_cluster.yaml uses the tagged
                # --discovery-provider kubernetes, so a run covers both.
                "--service-discovery",
                "--selector",
                "app=smg-kind-e2e",
                "--service-discovery-port",
                str(FALLBACK_PORT),
                "--service-discovery-namespace",
                "default",
                "--policy",
                "round_robin",
                *extra_args,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=10)

    def restart(self, extra_args: tuple[str, ...] = ()) -> None:
        self.stop()
        self.start(extra_args)

    def workers(self) -> list[dict]:
        try:
            payload = requests.get(f"{self.base_url}/workers", timeout=5).json()
        except requests.RequestException:
            return []
        return payload.get("workers", payload if isinstance(payload, list) else [])

    def worker_count(self) -> int:
        return len(self.workers())

    def pod_uid_of_port(self, port: int) -> str | None:
        for worker in self.workers():
            if worker.get("url", "").endswith(f":{port}"):
                return worker.get("labels", {}).get("smg.ai/pod-uid")
        return None

    def wait_for_count(self, expect: int, what: str, timeout: float = 120.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # A dead gateway must never satisfy an expected-zero count.
            assert self.proc is not None and self.proc.poll() is None, (
                f"smg process died while waiting for {what}"
            )
            if self.worker_count() == expect:
                return
            time.sleep(1)
        raise AssertionError(f"timed out waiting for {what}; workers: {self.workers()}")

    def log_contains(self, needle: str) -> bool:
        return needle in self.log_path.read_text()


@pytest.fixture(scope="session")
def kind_cluster():
    if os.environ.get("SMG_KIND_E2E") != "1":
        pytest.skip("kind e2e disabled (set SMG_KIND_E2E=1)")
    for tool in ("kind", "kubectl"):
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} not installed")

    subprocess.run(
        ["kind", "create", "cluster", "--name", CLUSTER, "--wait", "120s"],
        check=True,
    )
    try:
        for image_env in ("SMG_MOCK_IMAGE", "SMG_GATEWAY_IMAGE"):
            image = os.environ.get(image_env)
            if image:
                subprocess.run(
                    ["kind", "load", "docker-image", image, "--name", CLUSTER],
                    check=True,
                )
        kubectl("apply", "-f", str(HERE / "manifests.yaml"))
        kubectl(
            "wait",
            "--for=condition=Ready",
            "pod/multi-engine-0",
            "pod/single-engine-0",
            "--timeout=300s",
        )
        yield
    finally:
        subprocess.run(
            ["kind", "delete", "cluster", "--name", CLUSTER],
            check=False,
            capture_output=True,
        )


@pytest.fixture(scope="session")
def gateway(kind_cluster):
    binary = Path(os.environ.get("SMG_BIN", "target/debug/smg")).resolve()
    assert binary.exists(), f"smg binary not found at {binary} (cargo build -p smg first)"
    gw = KindGateway(binary)
    gw.start()
    try:
        yield gw
    finally:
        gw.stop()
        print(f"\n=== smg log tail ({gw.log_path}) ===")
        lines = gw.log_path.read_text().splitlines()
        print("\n".join(lines[-40:]))
