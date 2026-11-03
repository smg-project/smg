"""Real-cluster service discovery e2e against a kind cluster.

Covers the behaviors the fake-API-server integration tests cannot: real
API-server watches and RBAC scoping, kubelet-driven readiness conditions,
kubectl-applied mutations, and actual graceful/force deletion semantics.
Tests run in definition order and each converges the cluster back to the
5-worker baseline (4 annotated multi-engine ports + 1 fallback), except
the final teardown test.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest
import requests

HERE = Path(__file__).parent


def kubectl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["kubectl", *args], check=True, capture_output=True, text=True)


DECOY_MANIFEST = """
apiVersion: v1
kind: Pod
metadata:
  name: decoy-0
  namespace: decoy
  labels:
    app: smg-kind-e2e
  annotations:
    smg.ai/worker-ports: "28777"
spec:
  restartPolicy: Never
  containers:
    - name: sleeper
      image: python:3.12-alpine
      command: ["sleep", "3600"]
"""


def toggle_engine_health(pod: str, probe_port: int, state: str) -> None:
    kubectl(
        "exec",
        pod,
        "--",
        "python",
        "-c",
        "import urllib.request; urllib.request.urlopen("
        f"'http://127.0.0.1:{probe_port}/set_health?state={state}')",
    )


@pytest.mark.kind
class TestKindDiscovery:
    def test_initial_registration_and_ownership_labels(self, gateway):
        gateway.wait_for_count(5, "initial registration (4 annotated + 1 fallback)")
        workers = gateway.workers()
        assert all(w.get("labels", {}).get("smg.ai/pod-uid") for w in workers)
        multi = [
            w for w in workers if w.get("labels", {}).get("smg.ai/pod-name") == "multi-engine-0"
        ]
        assert len(multi) == 4

    def test_burst_batch_apply_and_delete(self, gateway):
        kubectl("apply", "-f", str(HERE / "burst.yaml"))
        kubectl(
            "wait",
            "--for=condition=Ready",
            "pod/burst-0",
            "pod/burst-1",
            "pod/burst-2",
            "--timeout=180s",
        )
        gateway.wait_for_count(8, "burst registration")
        kubectl("delete", "-f", str(HERE / "burst.yaml"), "--wait=false")
        gateway.wait_for_count(5, "burst removal")

    def test_namespace_scoping_ignores_foreign_pod(self, gateway):
        kubectl("create", "namespace", "decoy")
        subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=DECOY_MANIFEST,
            text=True,
            check=True,
            capture_output=True,
        )
        kubectl(
            "wait",
            "--for=condition=Ready",
            "pod/decoy-0",
            "-n",
            "decoy",
            "--timeout=180s",
        )
        time.sleep(10)
        assert not gateway.log_contains("28777"), (
            "namespace scoping violated: discovery touched the decoy pod's port"
        )
        assert gateway.worker_count() == 5

    def test_annotation_edit_reconciles_port_set(self, gateway):
        kubectl(
            "annotate",
            "pod",
            "multi-engine-0",
            "smg.ai/worker-ports=28080,28081",
            "--overwrite",
        )
        gateway.wait_for_count(3, "workers removed for dropped ports")
        kubectl(
            "annotate",
            "pod",
            "multi-engine-0",
            "smg.ai/worker-ports=28080,28081,28082,28083",
            "--overwrite",
        )
        gateway.wait_for_count(5, "workers restored for re-added ports")

    def test_readiness_flip_removes_and_restores_workers(self, gateway):
        toggle_engine_health("multi-engine-0", 28080, "down")
        kubectl(
            "wait",
            "--for=condition=Ready=false",
            "pod/multi-engine-0",
            "--timeout=60s",
        )
        drain_deadline = time.monotonic() + 30
        while True:
            multi = [
                worker
                for worker in gateway.workers()
                if worker.get("labels", {}).get("smg.ai/pod-name") == "multi-engine-0"
            ]
            if not any(worker.get("status") == "ready" for worker in multi):
                break
            assert time.monotonic() < drain_deadline, (
                f"unready pod workers remained routable: {multi}"
            )
            time.sleep(0.25)

        response = requests.post(
            f"{gateway.base_url}/v1/chat/completions",
            json={
                "model": "kind-e2e-model",
                "messages": [{"role": "user", "content": "drain probe"}],
            },
            timeout=10,
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "served-by-28090"

        gateway.wait_for_count(1, "four workers removed while their pod is unready")
        remaining = gateway.workers()
        assert len(remaining) == 1
        assert remaining[0].get("labels", {}).get("smg.ai/pod-name") == "single-engine-0"

        deletion_timestamp = kubectl(
            "get",
            "pod/multi-engine-0",
            "-o",
            "jsonpath={.metadata.deletionTimestamp}",
        ).stdout
        assert not deletion_timestamp, "unready pod must still exist"

        toggle_engine_health("multi-engine-0", 28080, "up")
        kubectl(
            "wait",
            "--for=condition=Ready",
            "pod/multi-engine-0",
            "--timeout=60s",
        )
        gateway.wait_for_count(5, "workers restored after readiness recovery")

    def test_gateway_restart_rebuilds_registry(self, gateway):
        gateway.restart()
        gateway.wait_for_count(5, "registry rebuilt from live cluster after restart")

    def test_pod_recreation_changes_ownership_uid(self, gateway):
        old_uid = gateway.pod_uid_of_port(28090)
        assert old_uid, "missing uid label before recreation"
        kubectl("delete", "pod", "single-engine-0", "--wait=true")
        gateway.wait_for_count(4, "removal of deleted pod's worker")
        kubectl("apply", "-f", str(HERE / "manifests.yaml"))
        kubectl(
            "wait",
            "--for=condition=Ready",
            "pod/single-engine-0",
            "--timeout=180s",
        )
        gateway.wait_for_count(5, "re-registration after recreation")
        new_uid = gateway.pod_uid_of_port(28090)
        assert new_uid and new_uid != old_uid, (
            f"recreated pod kept stale uid label (old={old_uid} new={new_uid})"
        )

    def test_completions_route_to_discovered_workers(self, gateway):
        gateway.wait_for_count(5, "baseline before data-plane check")
        # Health promotion lags registration; wait for the first 200 before
        # asserting fan-out.
        deadline = time.monotonic() + 60
        while True:
            probe = requests.post(
                f"{gateway.base_url}/v1/chat/completions",
                json={
                    "model": "kind-e2e-model",
                    "messages": [{"role": "user", "content": "warmup"}],
                },
                timeout=10,
            )
            if probe.status_code == 200:
                break
            assert time.monotonic() < deadline, f"no worker became routable: {probe.text}"
            time.sleep(1)
        served_by: set[str] = set()
        for i in range(10):
            response = requests.post(
                f"{gateway.base_url}/v1/chat/completions",
                json={
                    "model": "kind-e2e-model",
                    "messages": [{"role": "user", "content": f"probe {i}"}],
                },
                timeout=10,
            )
            assert response.status_code == 200, response.text
            content = response.json()["choices"][0]["message"]["content"]
            assert content.startswith("served-by-"), content
            served_by.add(content)
        # Round-robin over the discovered fleet: several distinct engines
        # (pod ports) must actually serve traffic.
        assert len(served_by) >= 2, f"traffic pinned to one engine: {served_by}"

    def test_graceful_then_force_delete(self, gateway):
        kubectl("delete", "pod", "single-engine-0", "--wait=false")
        gateway.wait_for_count(4, "removal after graceful delete")
        kubectl(
            "delete",
            "pod",
            "multi-engine-0",
            "--grace-period=0",
            "--force",
        )
        gateway.wait_for_count(0, "removal after force delete")


# ========== mock-worker fleet scenarios (need SMG_MOCK_IMAGE) ==========

MOCK_IMAGE = os.environ.get("SMG_MOCK_IMAGE")
needs_mock_image = pytest.mark.skipif(
    not MOCK_IMAGE, reason="SMG_MOCK_IMAGE not set (build Dockerfile.mock-worker)"
)


def mock_worker_pod(
    name: str,
    args: list[str],
    annotations: dict[str, str],
    probe_port: int,
    probe_kind: str = "http",
    extra_labels: dict[str, str] | None = None,
) -> str:
    labels = {"app": "smg-kind-e2e", **(extra_labels or {})}
    label_lines = "\n".join(f"    {k}: {v}" for k, v in labels.items())
    annotation_lines = "\n".join(f'    {k}: "{v}"' for k, v in annotations.items())
    if probe_kind == "http":
        probe = f"httpGet: {{ path: /health, port: {probe_port} }}"
    else:
        probe = f"tcpSocket: {{ port: {probe_port} }}"
    arg_list = ", ".join(f'"{a}"' for a in args)
    return f"""
apiVersion: v1
kind: Pod
metadata:
  name: {name}
  labels:
{label_lines}
  annotations:
{annotation_lines}
spec:
  hostNetwork: true
  restartPolicy: Never
  containers:
    - name: engines
      image: {MOCK_IMAGE}
      imagePullPolicy: Never
      args: [{arg_list}]
      readinessProbe:
        {probe}
        initialDelaySeconds: 1
        periodSeconds: 2
  volumes: []
"""


def apply_manifest(manifest: str) -> None:
    subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=manifest,
        text=True,
        check=True,
        capture_output=True,
    )


@pytest.mark.kind
@needs_mock_image
class TestKindMockWorkerFleet:
    """Scale and protocol-mix scenarios on the real mock-worker binary."""

    SCALE_PORTS = [29000 + i for i in range(40)]

    def test_forty_engine_pod_registers_and_unwinds(self, gateway):
        gateway.wait_for_count(0, "clean slate before fleet scenarios")
        apply_manifest(
            mock_worker_pod(
                "fleet-0",
                [
                    "--host",
                    "0.0.0.0",
                    "--http-base-port",
                    "29000",
                    "--http-count",
                    "40",
                    "--model",
                    "kind-e2e-model",
                ],
                {"smg.ai/worker-ports": ",".join(str(p) for p in self.SCALE_PORTS)},
                probe_port=29000,
            )
        )
        kubectl("wait", "--for=condition=Ready", "pod/fleet-0", "--timeout=180s")
        gateway.wait_for_count(40, "forty workers from one pod", timeout=180)
        urls = {w["url"].rsplit(":", 1)[-1] for w in gateway.workers()}
        assert urls == {str(p) for p in self.SCALE_PORTS}
        kubectl("delete", "pod", "fleet-0", "--grace-period=0", "--force")
        gateway.wait_for_count(0, "fleet unwound", timeout=180)

    def test_grpc_workers_register_with_grpc_mode(self, gateway):
        apply_manifest(
            mock_worker_pod(
                "grpc-0",
                [
                    "--host",
                    "0.0.0.0",
                    "--grpc-base-port",
                    "29500",
                    "--grpc-count",
                    "2",
                    "--model",
                    "kind-e2e-model",
                ],
                {"smg.ai/worker-ports": "29500,29501"},
                probe_port=29500,
                probe_kind="tcp",
            )
        )
        kubectl("wait", "--for=condition=Ready", "pod/grpc-0", "--timeout=180s")
        gateway.wait_for_count(2, "grpc workers registered", timeout=120)
        for worker in gateway.workers():
            assert worker.get("connection_mode") == "grpc", worker
        kubectl("delete", "pod", "grpc-0", "--grace-period=0", "--force")
        gateway.wait_for_count(0, "grpc pod unwound")


@pytest.mark.kind
@needs_mock_image
class TestKindPDDisaggregation:
    """PD prefill/decode mix with per-engine bootstrap alignment."""

    def test_pd_mix_with_aligned_bootstrap_ports(self, gateway):
        apply_manifest(
            mock_worker_pod(
                "prefill-0",
                [
                    "--host",
                    "0.0.0.0",
                    "--http-base-port",
                    "29600",
                    "--http-count",
                    "2",
                    "--model",
                    "kind-e2e-model",
                ],
                {
                    "smg.ai/worker-ports": "29600,29601",
                    "sglang.ai/bootstrap-port": "29700,29701",
                    "smg.ai/kv-connector": "MooncakeConnector",
                    "smg.ai/kv-engine-id": "prefill-0,prefill-1",
                },
                probe_port=29600,
                extra_labels={"role": "prefill"},
            )
        )
        apply_manifest(
            mock_worker_pod(
                "decode-0",
                [
                    "--host",
                    "0.0.0.0",
                    "--http-base-port",
                    "29610",
                    "--http-count",
                    "2",
                    "--model",
                    "kind-e2e-model",
                ],
                {
                    "smg.ai/worker-ports": "29610,29611",
                    "smg.ai/kv-connector": "NixlConnector",
                    "smg.ai/kv-engine-id": "decode-0,decode-1",
                },
                probe_port=29610,
                extra_labels={"role": "decode"},
            )
        )
        kubectl("wait", "--for=condition=Ready", "pod/prefill-0", "pod/decode-0", "--timeout=180s")

        gateway.restart(
            (
                "--pd-disaggregation",
                "--prefill-selector",
                "app=smg-kind-e2e",
                "--prefill-selector",
                "role=prefill",
                "--decode-selector",
                "app=smg-kind-e2e",
                "--decode-selector",
                "role=decode",
            )
        )
        gateway.wait_for_count(4, "PD fleet registered", timeout=120)

        by_port = {w["url"].rsplit(":", 1)[-1]: w for w in gateway.workers()}
        assert by_port["29600"]["worker_type"] == "prefill"
        assert by_port["29601"]["worker_type"] == "prefill"
        assert by_port["29610"]["worker_type"] == "decode"
        assert by_port["29611"]["worker_type"] == "decode"
        # Aligned bootstrap list: each prefill engine gets its own port.
        assert by_port["29600"].get("bootstrap_port") == 29700
        assert by_port["29601"].get("bootstrap_port") == 29701
        assert "bootstrap_port" not in by_port["29610"]
        assert by_port["29600"]["kv_connector"] == "MooncakeConnector"
        assert by_port["29600"]["kv_engine_id"] == "prefill-0"
        assert by_port["29601"]["kv_engine_id"] == "prefill-1"
        assert by_port["29610"]["kv_connector"] == "NixlConnector"
        assert by_port["29610"]["kv_engine_id"] == "decode-0"
        assert by_port["29611"]["kv_engine_id"] == "decode-1"

        kubectl("delete", "pod", "prefill-0", "decode-0", "--grace-period=0", "--force")
        gateway.wait_for_count(0, "PD fleet unwound")


@pytest.mark.kind
@needs_mock_image
class TestKindEPDDisaggregation:
    """EPD encode/prefill/decode mix, each role found by its own selector.

    The pods run gRPC mock-workers: EPD routing is the gRPC router's, over
    TokenSpeed engines, and mock-worker's gRPC server is a TokenSpeed
    scheduler, so each pod registers as the gRPC TokenSpeed worker a real EPD
    engine would.

    Registration only; no request is served through the EPD pipeline. The
    gRPC router tokenizes every request itself and mock-worker has no
    tokenizer to offer (GetTokenizer is unimplemented; its tokenizer path is
    the model id), and the one leg that is EPD's own, the encode worker's
    TokenSpeedEncoder.Encode call, is not a service mock-worker hosts. A text
    request would not reach the encode pool anyway: only image items are
    assigned to encode workers, so it would prove the PD path, not EPD.
    """

    def test_epd_mix_registers_each_role_into_its_own_pool(self, gateway):
        apply_manifest(
            mock_worker_pod(
                "epd-encode-0",
                [
                    "--host",
                    "0.0.0.0",
                    "--grpc-base-port",
                    "29800",
                    "--grpc-count",
                    "2",
                    "--model",
                    "kind-e2e-model",
                ],
                {
                    "smg.ai/worker-ports": "29800,29801",
                    "sglang.ai/bootstrap-port": "29900,29901",
                    "smg.ai/kv-connector": "MooncakeConnector",
                    "smg.ai/kv-engine-id": "encode-0,encode-1",
                },
                probe_port=29800,
                probe_kind="tcp",
                extra_labels={"role": "encode"},
            )
        )
        apply_manifest(
            mock_worker_pod(
                "epd-prefill-0",
                [
                    "--host",
                    "0.0.0.0",
                    "--grpc-base-port",
                    "29810",
                    "--grpc-count",
                    "2",
                    "--model",
                    "kind-e2e-model",
                ],
                {
                    "smg.ai/worker-ports": "29810,29811",
                    "sglang.ai/bootstrap-port": "29910,29911",
                    "smg.ai/kv-connector": "MooncakeConnector",
                    "smg.ai/kv-engine-id": "prefill-0,prefill-1",
                },
                probe_port=29810,
                probe_kind="tcp",
                extra_labels={"role": "prefill"},
            )
        )
        # The decode pod carries a bootstrap annotation too, as a template
        # shared across roles would: the gateway must drop it, since only
        # encode and prefill workers host a rendezvous for their partners.
        apply_manifest(
            mock_worker_pod(
                "epd-decode-0",
                [
                    "--host",
                    "0.0.0.0",
                    "--grpc-base-port",
                    "29820",
                    "--grpc-count",
                    "2",
                    "--model",
                    "kind-e2e-model",
                ],
                {
                    "smg.ai/worker-ports": "29820,29821",
                    "sglang.ai/bootstrap-port": "29920",
                    "smg.ai/kv-connector": "MooncakeConnector",
                    "smg.ai/kv-engine-id": "decode-0,decode-1",
                },
                probe_port=29820,
                probe_kind="tcp",
                extra_labels={"role": "decode"},
            )
        )
        kubectl(
            "wait",
            "--for=condition=Ready",
            "pod/epd-encode-0",
            "pod/epd-prefill-0",
            "pod/epd-decode-0",
            "--timeout=180s",
        )

        gateway.restart(
            (
                "--epd-disaggregation",
                "--encode-selector",
                "app=smg-kind-e2e",
                "--encode-selector",
                "role=encode",
                "--prefill-selector",
                "app=smg-kind-e2e",
                "--prefill-selector",
                "role=prefill",
                "--decode-selector",
                "app=smg-kind-e2e",
                "--decode-selector",
                "role=decode",
            )
        )
        gateway.wait_for_count(6, "EPD fleet registered", timeout=120)

        by_port = {w["url"].rsplit(":", 1)[-1]: w for w in gateway.workers()}
        expected_types = {
            "29800": "encode",
            "29801": "encode",
            "29810": "prefill",
            "29811": "prefill",
            "29820": "decode",
            "29821": "decode",
        }
        assert {port: w["worker_type"] for port, w in by_port.items()} == expected_types
        for worker in by_port.values():
            assert worker.get("connection_mode") == "grpc", worker
            assert worker.get("runtime_type") == "tokenspeed", worker
            assert worker["kv_connector"] == "MooncakeConnector", worker
        # Aligned bootstrap lists: each encode and prefill engine gets its own
        # port; decode gets none, whatever its pod says.
        assert by_port["29800"].get("bootstrap_port") == 29900
        assert by_port["29801"].get("bootstrap_port") == 29901
        assert by_port["29810"].get("bootstrap_port") == 29910
        assert by_port["29811"].get("bootstrap_port") == 29911
        assert "bootstrap_port" not in by_port["29820"]
        assert "bootstrap_port" not in by_port["29821"]
        assert by_port["29800"]["kv_engine_id"] == "encode-0"
        assert by_port["29801"]["kv_engine_id"] == "encode-1"
        assert by_port["29810"]["kv_engine_id"] == "prefill-0"
        assert by_port["29811"]["kv_engine_id"] == "prefill-1"
        assert by_port["29820"]["kv_engine_id"] == "decode-0"
        assert by_port["29821"]["kv_engine_id"] == "decode-1"
        # Prefill and decode share one pairing key, so the gateway would pair
        # them for a request's KV handoff.
        pairings = {
            by_port[port].get("pd_pairing") for port in ("29810", "29811", "29820", "29821")
        }
        assert len(pairings) == 1 and None not in pairings, pairings

        # Registration is not readiness: every role's workers must also pass the
        # gateway's own gRPC health checks.
        deadline = time.monotonic() + 60
        while True:
            workers = gateway.workers()
            if len(workers) == 6 and all(w.get("status") == "ready" for w in workers):
                break
            assert time.monotonic() < deadline, f"EPD workers never became ready: {workers}"
            time.sleep(1)

        # Encode is a pool of its own: listed and in the total, but in none
        # of the regular, prefill or decode counts. An encode pod that fell
        # into the regular pool would show here as a regular worker.
        listing = requests.get(f"{gateway.base_url}/workers", timeout=5).json()
        assert listing["total"] == 6
        assert listing["stats"] == {
            "prefill_count": 2,
            "decode_count": 2,
            "regular_count": 0,
        }

        kubectl(
            "delete",
            "pod",
            "epd-encode-0",
            "epd-prefill-0",
            "epd-decode-0",
            "--grace-period=0",
            "--force",
        )
        gateway.wait_for_count(0, "EPD fleet unwound")
