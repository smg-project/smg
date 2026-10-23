"""Slurm-discovery e2e against a real Slurm 23.11 cluster (see conftest.py).

The scenarios: a multi-node allocation's lifecycle, a requeue replacing the
worker, a controller that goes away and comes back while a job's engine starts
late, a gateway restart, and the job-name role mappings with a user, account
and partition scope. What a small real cluster cannot stage (a node address
changing under a running job, a federation's other clusters, a node with no
address) is covered by the unit tests on recorded Slurm output.
"""

from __future__ import annotations

import time
import uuid

from ..common import served_by
from .harness import worker_command, worker_url

PROVIDER = "smg.ai/discovery-provider"
ID = "smg.ai/discovery-id"
# A requeued job is held for about two minutes before it runs again.
REQUEUE = 420.0


def unique(name: str) -> str:
    return f"{name}-{uuid.uuid4().hex[:6]}"


def ids_by_url(workers: list[dict]) -> dict[str, str]:
    return {w["url"]: w.get("labels", {}).get(ID, "") for w in workers}


def id_parts(discovery_id: str) -> list[str]:
    """cluster/job/start time/restart count/node/type"""
    return discovery_id.split("/")


def test_an_allocation_registers_its_nodes_and_leaves_with_them(slurm, gateway):
    name = unique("smg-worker")
    gateway.start("--slurm-job-name", name)

    first = slurm.submit(name, worker_command(), nodes="node[1-2]")
    slurm.wait_for_state(first, "RUNNING")
    # Slurm reports the allocation compressed; scontrol expands it.
    assert slurm.exec("squeue", "-h", "-j", str(first), "-o", "%N").strip() == "node[1-2]"
    workers = gateway.wait_for_urls(
        {worker_url("node1"), worker_url("node2")}, "both nodes of the allocation"
    )
    assert all(w["labels"][PROVIDER] == "slurm" for w in workers), workers
    for url, discovery_id in ids_by_url(workers).items():
        cluster, job, _, restarts, node, kind = id_parts(discovery_id)
        assert (cluster, job, restarts, kind) == ("e2e", str(first), "0", "regular"), url
        assert url == worker_url(node), discovery_id
    assert served_by(gateway, 4), "no request was served"
    slurm.record("allocation")

    second = slurm.submit(name, worker_command(), nodes="node3")
    gateway.wait_for_urls(
        {worker_url("node1"), worker_url("node2"), worker_url("node3")},
        "a second allocation joining",
    )

    slurm.exec("scancel", str(first))
    gateway.wait_for_urls({worker_url("node3")}, "the first allocation's nodes leaving")

    slurm.exec("scancel", str(second))
    gateway.wait_for_urls(set(), "an empty queue clearing the fleet")


def test_a_requeue_replaces_the_worker(slurm, gateway):
    name = unique("smg-worker")
    gateway.start("--slurm-job-name", name)
    job = slurm.submit(name, worker_command(), nodes="node1")
    (worker,) = gateway.wait_for_urls({worker_url("node1")}, "the job's worker")
    first = worker["labels"][ID]
    assert id_parts(first)[3] == "0", first

    # Requeued, the job leaves RUNNING and its worker goes; when it runs again
    # it is a new instance at the same address.
    slurm.exec("scontrol", "requeue", str(job))
    gateway.wait_for_urls(set(), "the requeued job's worker removed")
    slurm.record("requeued")
    (worker,) = gateway.wait_for(
        "the requeued job running again as a new instance",
        lambda workers: (
            [w["url"] for w in workers] == [worker_url("node1")]
            and workers[0]["labels"][ID] != first
        ),
        within=REQUEUE,
    )
    second = id_parts(worker["labels"][ID])
    assert second[1] == str(job) and second[3] == "1", second


def test_a_lost_controller_keeps_the_fleet_and_a_late_engine_registers(slurm, gateway):
    name = unique("smg-worker")
    gateway.start("--slurm-job-name", name)
    slurm.submit(name, worker_command(), nodes="node1")
    gateway.wait_for_urls({worker_url("node1")}, "the first job's worker")

    # With slurmctld down, squeue fails: the poll is rejected and the fleet
    # stays as it was.
    slurm.exec("pkill", "-x", "slurmctld")
    gateway.wait_for_log("Slurm poll failed", "the failed poll reported")
    time.sleep(6)
    assert {w["url"] for w in gateway.workers() or []} == {worker_url("node1")}

    # Back, it still holds the job, and takes a new one whose engine starts
    # late: no worker until it listens, then one with Slurm unchanged.
    slurm.exec("slurmctld")
    slurm.wait_for_responding_nodes()
    late = slurm.submit(name, worker_command(delay_secs=25), nodes="node2")
    slurm.wait_for_state(late, "RUNNING")
    time.sleep(10)
    assert {w["url"] for w in gateway.workers() or []} == {worker_url("node1")}
    gateway.wait_for_urls({worker_url("node1"), worker_url("node2")}, "the late engine registered")


def test_a_restarted_gateway_rebuilds_the_fleet(slurm, gateway):
    name = unique("smg-worker")
    gateway.start("--slurm-job-name", name)
    slurm.submit(name, worker_command(), nodes="node[1-2]")
    urls = {worker_url("node1"), worker_url("node2")}
    before = ids_by_url(gateway.wait_for_urls(urls, "the fleet"))

    gateway.stop()
    gateway.start("--slurm-job-name", name)
    after = ids_by_url(gateway.wait_for_urls(urls, "the fleet rebuilt"))
    assert after == before


def test_job_names_map_to_roles_within_the_scope(slurm, gateway):
    """Two prefill job names and a decode one, beside jobs the mapping or the
    scope leaves out, each with a running engine that would register if it were
    let through: a name no flag maps, another user's job, and one in another
    partition. The gateway's squeue runs with --me, and under --json Slurm
    23.11 ignores it, as every filter but --user, so it lists them all; only
    the mapped jobs of this user, account and partition get workers, with the
    bootstrap port on prefill alone. The account is configured in another case
    than the one Slurm stores."""
    prefill_a, prefill_b, decode = unique("prefill-a"), unique("prefill-b"), unique("decode")
    scope = {"account": "acct-e2e", "partition": "debug"}
    slurm.submit(prefill_a, worker_command(), nodes="node1", **scope)
    slurm.submit(prefill_b, worker_command(), nodes="node2", **scope)
    slurm.submit(decode, worker_command(), nodes="node3", **scope)
    slurm.submit(unique("other"), worker_command(), nodes="node4", **scope)
    slurm.submit(decode, worker_command(), nodes="node5", user="alice", **scope)
    slurm.submit(decode, worker_command(), nodes="node6", account="acct-e2e", partition="other")

    gateway.start(
        "--pd-disaggregation",
        *("--slurm-prefill-job-name", prefill_a),
        *("--slurm-prefill-job-name", prefill_b),
        *("--slurm-decode-job-name", decode),
        *("--slurm-bootstrap-port", "8998"),
        *("--slurm-account", "ACCT-E2E"),
        *("--slurm-partition", "debug"),
    )
    expected = {worker_url("node1"), worker_url("node2"), worker_url("node3")}
    workers = gateway.wait_for_urls(expected, "the mapped jobs' workers")
    slurm.record("roles")
    roles = {w["url"]: (w["worker_type"], w.get("bootstrap_port")) for w in workers}
    assert roles == {
        worker_url("node1"): ("prefill", 8998),
        worker_url("node2"): ("prefill", 8998),
        worker_url("node3"): ("decode", None),
    }
    # Every decoy runs, so its absence is the provider's filtering.
    assert len(slurm.exec("squeue", "-h", "-t", "RUNNING", "-o", "%i").split()) == 6
    time.sleep(6)
    assert {w["url"] for w in gateway.workers() or []} == expected
