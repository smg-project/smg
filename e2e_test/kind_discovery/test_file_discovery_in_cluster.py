"""File discovery inside the cluster: the worker manifest is a ConfigMap.

The gateway Pod mounts the ConfigMap as its --discovery-file and has no
ServiceAccount token, so everything it registers comes from the file. The
test publishes and edits the manifest the way an operator would, with
`kubectl apply`; the kubelet projects each update into the Pod by atomically
swapping the mounted files, the kind of replacement file discovery expects.
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
FORWARD_PORT = 3110
ENGINE_PORT = 8080
CONFIGMAP = "smg-file-workers"
# The kubelet projects a ConfigMap update within about a minute. Touching the
# Pod in `publish_manifest` usually brings that down to seconds.
PROPAGATION = 180.0

needs_images = pytest.mark.skipif(
    not (os.environ.get("SMG_MOCK_IMAGE") and os.environ.get("SMG_GATEWAY_IMAGE")),
    reason="SMG_MOCK_IMAGE and SMG_GATEWAY_IMAGE not set",
)


def kubectl(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["kubectl", *args], input=stdin, check=True, capture_output=True, text=True
    )


def engine_ips() -> list[str]:
    out = kubectl("get", "pods", "-l", "app=engines-file", "-o", "json").stdout
    return sorted(
        item["status"]["podIP"]
        for item in json.loads(out)["items"]
        if item["metadata"].get("deletionTimestamp") is None and item["status"].get("podIP")
    )


def manifest_for(ips: list[str]) -> str:
    workers = [{"id": f"engine-{ip}", "url": f"http://{ip}:{ENGINE_PORT}"} for ip in ips]
    return json.dumps({"version": 1, "workers": workers})


def urls_for(ips: list[str]) -> set[str]:
    return {f"http://{ip}:{ENGINE_PORT}" for ip in ips}


def publish_manifest(manifest: str) -> None:
    """Publish the manifest as an operator would: as the gateway's ConfigMap."""
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIGMAP},
        "data": {"workers.json": manifest},
    }
    kubectl("apply", "-f", "-", stdin=json.dumps(configmap))
    # The kubelet refreshes a ConfigMap volume when it syncs the Pod, which
    # it does periodically or on a Pod update. Touching the Pod asks for
    # that sync now.
    kubectl(
        "annotate",
        "pod",
        "-l",
        "app=smg-gateway-file",
        f"e2e.smg.ai/manifest-revision={time.time_ns()}",
        "--overwrite",
    )


class FileGateway:
    """Assertion helper talking to the in-cluster file-discovery gateway."""

    def __init__(self):
        self.base_url = f"http://127.0.0.1:{FORWARD_PORT}"

    def workers(self) -> list[dict] | None:
        """The registered workers, or None when the gateway could not be
        asked; an unanswered request must never read as an empty fleet."""
        try:
            response = requests.get(f"{self.base_url}/workers", timeout=5)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            return None
        return payload.get("workers", payload if isinstance(payload, list) else [])

    def worker_urls(self) -> set[str] | None:
        workers = self.workers()
        return None if workers is None else {w["url"] for w in workers}

    def wait_for_urls(self, expected: set[str], what: str) -> None:
        deadline = time.monotonic() + PROPAGATION
        while time.monotonic() < deadline:
            if self.worker_urls() == expected:
                return
            time.sleep(1)
        raise AssertionError(
            f"timed out waiting for {what}; expected {sorted(expected)}, got {self.worker_urls()}"
        )

    @staticmethod
    def wait_for_log(needle: str, what: str) -> None:
        deadline = time.monotonic() + PROPAGATION
        while time.monotonic() < deadline:
            if needle in kubectl("logs", "deployment/smg-gateway-file").stdout:
                return
            time.sleep(2)
        raise AssertionError(f"timed out waiting for {what}: no {needle!r} in the gateway log")


@pytest.fixture(scope="class")
def file_gateway(kind_cluster):
    kubectl("apply", "-f", str(HERE / "file_in_cluster.yaml"))
    kubectl("rollout", "status", "deployment/smg-gateway-file", "--timeout=180s")
    kubectl("rollout", "status", "deployment/engines-file", "--timeout=180s")

    forward = subprocess.Popen(
        ["kubectl", "port-forward", "svc/smg-gateway-file", f"{FORWARD_PORT}:3009"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    gw = FileGateway()
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
        forward.kill()
        for args in (
            ["delete", "-f", str(HERE / "file_in_cluster.yaml"), "--wait=false"],
            ["delete", "configmap", CONFIGMAP, "--ignore-not-found"],
        ):
            subprocess.run(["kubectl", *args], check=False, capture_output=True)


@pytest.mark.kind
@needs_images
class TestFileDiscoveryInCluster:
    def test_gateway_waits_for_a_manifest_published_after_it(self, file_gateway):
        # Started before its manifest existed: nothing registered, and the
        # gateway says why rather than treating the absence as no workers.
        file_gateway.wait_for_log("cannot read discovery manifest", "the missing manifest reported")
        assert file_gateway.worker_urls() == set()

        ips = engine_ips()
        assert len(ips) == 2, ips
        publish_manifest(manifest_for(ips))
        file_gateway.wait_for_urls(urls_for(ips), "the manifest published after startup")

        workers = file_gateway.workers()
        assert workers is not None, "the gateway stopped answering"
        for worker in workers:
            labels = worker.get("labels", {})
            assert labels.get("smg.ai/discovery-provider") == "file", worker
            assert labels.get("smg.ai/discovery-id", "").startswith("engine-"), worker

        # A request through the gateway reaches a file-discovered engine.
        response = requests.post(
            f"{file_gateway.base_url}/v1/chat/completions",
            json={"model": "file-model", "messages": [{"role": "user", "content": "hi"}]},
            timeout=15,
        )
        assert response.status_code == 200, response.text

    def test_manifest_edits_remove_and_add_workers(self, file_gateway):
        ips = engine_ips()
        publish_manifest(manifest_for(ips[:1]))
        file_gateway.wait_for_urls(
            urls_for(ips[:1]), "the worker dropped from the manifest removed"
        )

        publish_manifest(manifest_for(ips))
        file_gateway.wait_for_urls(urls_for(ips), "the worker listed again registered")

    def test_invalid_manifest_keeps_workers_and_is_diagnosable(self, file_gateway):
        before = file_gateway.worker_urls()
        assert before, "expected the previous test's workers"

        publish_manifest(
            json.dumps(
                {
                    "version": 1,
                    "workers": [{"url": "http://10.0.0.1:8080", "worker_typ": "regular"}],
                }
            )
        )
        # The log line proves the bad manifest reached the Pod and was
        # rejected, so the unchanged fleet below is not just a slow kubelet.
        file_gateway.wait_for_log("unknown field `worker_typ`", "the invalid manifest rejected")
        assert file_gateway.worker_urls() == before

        publish_manifest(manifest_for([]))
        file_gateway.wait_for_urls(set(), "an empty manifest removing every worker")
