"""Real-Slurm smoke: a job a real Slurm runs is discovered, replaced when it
is requeued, and removed when it is cancelled. See conftest.py."""

from __future__ import annotations

import uuid

from ..discovery_common import MODEL, served_by

ID = "smg.ai/discovery-id"
PROVIDER = "smg.ai/discovery-provider"
# The port of the worker the job starts, on the node's NodeAddr.
WORKER_PORT = 8000
WORKER_URL = f"http://127.0.0.1:{WORKER_PORT}"
# A requeued job is held for about two minutes before it runs again.
REQUEUE = 420.0


def restart_count(discovery_id: str) -> str:
    """The restart count in a Slurm worker's id:
    cluster/job/start time/restart count/node/type."""
    return discovery_id.split("/")[3]


def test_a_real_job_is_discovered_replaced_on_requeue_and_removed_on_cancel(slurm_node, gateway):
    release = slurm_node.exec("squeue", "--version").split()[-1]
    assert release.startswith("24.11"), f"the image runs Slurm {release}"

    name = f"smg-smoke-{uuid.uuid4().hex[:8]}"
    gateway.start("--slurm-job-name", name, "--slurm-port", str(WORKER_PORT))
    job = slurm_node.submit(
        name,
        f"exec mock-worker --host 127.0.0.1 --http-base-port {WORKER_PORT}"
        f" --http-count 1 --model {MODEL}",
    )

    # The running job's node becomes a worker the gateway routes to.
    (worker,) = gateway.wait_for_urls({WORKER_URL}, "the job's worker", within=180)
    first = worker["labels"][ID]
    assert worker["labels"][PROVIDER] == "slurm", worker
    assert first.startswith(f"smoke/{job}/"), first
    assert restart_count(first) == "0", first
    assert served_by(gateway, 3), "no request was served"
    slurm_node.record("1-running")

    # Requeued, the job leaves RUNNING and its worker goes; when it runs again
    # it is a new instance at the same address.
    slurm_node.exec("scontrol", "requeue", str(job))
    gateway.wait_for_urls(set(), "the requeued job's worker removed", within=120)
    slurm_node.record("2-requeued")
    (worker,) = gateway.wait_for(
        "the requeued job running again as a new instance",
        lambda workers: (
            [w["url"] for w in workers] == [WORKER_URL] and workers[0]["labels"][ID] != first
        ),
        within=REQUEUE,
    )
    second = worker["labels"][ID]
    assert second.startswith(f"smoke/{job}/"), second
    assert restart_count(second) == "1", second
    slurm_node.record("3-restarted")

    # Cancelled, it leaves the fleet empty.
    slurm_node.exec("scancel", str(job))
    gateway.wait_for_urls(set(), "the cancelled job's worker removed", within=120)
    slurm_node.record("4-cancelled")
