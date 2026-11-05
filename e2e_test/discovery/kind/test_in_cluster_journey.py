"""Operator-journey e2e: smg deployed **inside** the cluster.

Everything here runs the way a user runs it: smg is a Deployment using
in-cluster ServiceAccount auth with the minimal RBAC from in_cluster.yaml
(the reference manifests), engines are a plain Deployment whose pods each
run four servers on 8080-8083 over the pod network (no hostNetwork), and
the cluster is driven with kubectl apply / scale / rollout restart.

Beyond pass/fail invariants, each step records convergence timings into a
cluster report printed at the end — the "what goes well / what doesn't"
characterization for a real cluster.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import requests

HERE = Path(__file__).parent
FORWARD_PORT = 3109
ENGINE_PORTS = (8080, 8081, 8082, 8083)

needs_images = pytest.mark.skipif(
    not (os.environ.get("SMG_MOCK_IMAGE") and os.environ.get("SMG_GATEWAY_IMAGE")),
    reason="SMG_MOCK_IMAGE and SMG_GATEWAY_IMAGE not set",
)


def kubectl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["kubectl", *args], check=True, capture_output=True, text=True)


class InClusterGateway:
    """Assertion helper talking to the in-cluster smg via port-forward."""

    def __init__(self):
        self.base_url = f"http://127.0.0.1:{FORWARD_PORT}"
        self.report: list[tuple[str, float, str]] = []

    def workers(self) -> list[dict]:
        try:
            payload = requests.get(f"{self.base_url}/workers", timeout=5).json()
        except requests.RequestException:
            return []
        return payload.get("workers", payload if isinstance(payload, list) else [])

    def worker_urls(self) -> set[str]:
        return {w["url"] for w in self.workers()}

    def wait_for_count(self, expect: int, what: str, timeout: float = 180.0) -> float:
        """Wait for exactly `expect` workers; returns elapsed seconds."""
        start = time.monotonic()
        deadline = start + timeout
        while time.monotonic() < deadline:
            if len(self.workers()) == expect:
                elapsed = time.monotonic() - start
                self.report.append((what, elapsed, f"converged to {expect}"))
                return elapsed
            time.sleep(1)
        raise AssertionError(f"timed out waiting for {what}; workers: {sorted(self.worker_urls())}")

    def print_report(self) -> None:
        print("\n=== IN-CLUSTER JOURNEY REPORT (what the cluster observed) ===")
        for what, elapsed, note in self.report:
            print(f"  {elapsed:7.1f}s  {what}  [{note}]")


def engine_pod_ips(label: str = "app=engines-incluster") -> set[str]:
    out = kubectl("get", "pods", "-l", label, "-o", "json").stdout
    return {
        item["status"]["podIP"]
        for item in json.loads(out)["items"]
        if item["metadata"].get("deletionTimestamp") is None and item["status"].get("podIP")
    }


def _wait_until_gone(label: str, timeout: float = 120.0) -> None:
    """Block until no Pod matching `label` is listed as live."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if not engine_pod_ips(label):
            return
        time.sleep(1)
    raise AssertionError(f"timed out waiting for pods matching {label!r} to go away")


def expected_urls(pod_ips: set[str]) -> set[str]:
    return {f"http://{ip}:{port}" for ip in pod_ips for port in ENGINE_PORTS}


@pytest.fixture(scope="class")
def incluster(kind_cluster):
    kubectl("apply", "-f", str(HERE / "in_cluster.yaml"))
    kubectl("rollout", "status", "deployment/smg-gateway", "--timeout=180s")
    kubectl("rollout", "status", "deployment/engines", "--timeout=180s")

    forward = subprocess.Popen(
        ["kubectl", "port-forward", "svc/smg-gateway", f"{FORWARD_PORT}:3009"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    gw = InClusterGateway()
    deadline = time.monotonic() + 30
    while True:
        try:
            requests.get(f"{gw.base_url}/health", timeout=2)
            break
        except requests.RequestException:
            assert time.monotonic() < deadline, "port-forward never became ready"
            time.sleep(1)
    try:
        yield gw
    finally:
        gw.print_report()
        forward.kill()
        subprocess.run(
            ["kubectl", "delete", "-f", str(HERE / "in_cluster.yaml"), "--wait=false"],
            check=False,
            capture_output=True,
        )
        subprocess.run(
            ["kubectl", "delete", "deployment", "engines-typo", "--wait=false"],
            check=False,
            capture_output=True,
        )


@pytest.mark.kind
@needs_images
class TestInClusterUserJourney:
    def test_deploy_registers_every_engine_of_every_pod(self, incluster):
        incluster.wait_for_count(8, "initial deploy (2 pods x 4 engines)")
        assert incluster.worker_urls() == expected_urls(engine_pod_ips())

        # In-cluster data plane: a request through the gateway reaches a
        # discovered engine over the pod network.
        response = requests.post(
            f"{incluster.base_url}/v1/chat/completions",
            json={
                "model": "incluster-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
            timeout=15,
        )
        assert response.status_code == 200, response.text

    def test_scale_up_and_down_like_an_operator(self, incluster):
        kubectl("scale", "deployment/engines", "--replicas=4")
        kubectl("rollout", "status", "deployment/engines", "--timeout=180s")
        incluster.wait_for_count(16, "scale 2 -> 4 replicas")
        assert incluster.worker_urls() == expected_urls(engine_pod_ips())

        kubectl("scale", "deployment/engines", "--replicas=1")
        incluster.wait_for_count(4, "scale 4 -> 1 replicas")
        assert incluster.worker_urls() == expected_urls(engine_pod_ips())

    def test_rollout_restart_swaps_fleet_without_stale_workers(self, incluster):
        kubectl("scale", "deployment/engines", "--replicas=3")
        kubectl("rollout", "status", "deployment/engines", "--timeout=180s")
        incluster.wait_for_count(12, "pre-rollout fleet of 3")
        old_ips = engine_pod_ips()

        start = time.monotonic()
        floor = 12
        kubectl("rollout", "restart", "deployment/engines")
        while True:
            floor = min(floor, len(incluster.workers()))
            status = subprocess.run(
                ["kubectl", "rollout", "status", "deployment/engines", "--timeout=1s"],
                capture_output=True,
                text=True,
            )
            if status.returncode == 0:
                break
            assert time.monotonic() - start < 300, "rollout never completed"

        incluster.wait_for_count(12, "post-rollout convergence")
        new_ips = engine_pod_ips()
        assert incluster.worker_urls() == expected_urls(new_ips)
        assert not (expected_urls(old_ips - new_ips) & incluster.worker_urls()), (
            "stale workers survived the rollout"
        )
        incluster.report.append(
            ("rollout restart capacity floor", float(floor), "min workers during roll (of 12)")
        )

    def test_misconfigured_deployment_falls_back_and_is_diagnosable(self, incluster):
        manifest = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: engines-typo
spec:
  replicas: 1
  selector:
    matchLabels:
      app: engines-incluster
      variant: typo
  template:
    metadata:
      labels:
        app: engines-incluster
        variant: typo
      annotations:
        smg.ai/worker-ports: "80eight0,8081"
    spec:
      containers:
        - name: engines
          image: smg-mock-worker:e2e
          imagePullPolicy: Never
          args: ["--host", "0.0.0.0", "--http-base-port", "8080", "--http-count", "4"]
          readinessProbe:
            httpGet: { path: /health, port: 8080 }
            initialDelaySeconds: 1
            periodSeconds: 2
"""
        subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=manifest,
            text=True,
            check=True,
            capture_output=True,
        )
        kubectl("rollout", "status", "deployment/engines-typo", "--timeout=180s")

        # The invalid annotation must fall back to the single configured
        # port — the typo pod contributes exactly one worker, not four and
        # not zero, regardless of the healthy fleet's current size.
        typo_ips = engine_pod_ips("app=engines-incluster,variant=typo")
        assert typo_ips, "typo pod never got an IP"
        expected = expected_urls(engine_pod_ips()) - {
            f"http://{ip}:{port}" for ip in typo_ips for port in ENGINE_PORTS[1:]
        }
        self._wait_for_urls(incluster, expected, "typo deployment falls back to one worker")

        # The operator must be able to see why from the gateway's own logs.
        logs = kubectl("logs", "deployment/smg-gateway").stdout
        assert "invalid smg.ai/worker-ports annotation" in logs, (
            "misconfiguration is not diagnosable from gateway logs"
        )

        kubectl("delete", "deployment", "engines-typo", "--wait=true")
        # `--wait` covers the Deployment object only: its Pod is garbage
        # collected asynchronously and can still be listed, without a
        # deletionTimestamp, right after the call returns. Snapshotting the
        # fleet here would fold the doomed typo Pod into `expected` at four
        # ports -- a set the gateway can never produce, since that Pod only
        # ever contributed one worker.
        _wait_until_gone("app=engines-incluster,variant=typo")
        self._wait_for_urls(
            incluster,
            expected_urls(engine_pod_ips()),
            "typo deployment removed",
        )

    @staticmethod
    def _wait_for_urls(incluster, expected: set[str], what: str, timeout: float = 120.0) -> None:
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if incluster.worker_urls() == expected:
                incluster.report.append((what, time.monotonic() - start, f"{len(expected)} urls"))
                return
            time.sleep(1)
        raise AssertionError(
            f"timed out waiting for {what}; expected {sorted(expected)}, "
            f"got {sorted(incluster.worker_urls())}"
        )
