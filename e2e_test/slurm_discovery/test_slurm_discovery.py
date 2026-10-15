"""CLI-contract e2e for Slurm discovery.

The real gateway polls stand-in `squeue` and `scontrol` commands, which
answer as Slurm 23.11 does: `squeue`'s filters are accepted but not applied,
so every job in the state comes back and the provider's own filtering is what
these tests watch. The scenarios: a multi-node allocation's lifecycle, a
requeue and a node address change, a failed node lookup and an engine that
starts late, a gateway restart, and the job-name role mappings with a user,
account and partition scope.
"""

from __future__ import annotations

import time

from ..discovery_common import MODEL, served_by

PROVIDER = "smg.ai/discovery-provider"
ID = "smg.ai/discovery-id"
WORKER = ("--slurm-job-name", "smg-worker")


def ids_by_url(workers: list[dict]) -> dict[str, str]:
    return {w["url"]: w.get("labels", {}).get(ID, "") for w in workers}


def test_an_allocation_registers_its_nodes_and_leaves_with_them(gateway, slurm, node_engines):
    node1, node2, node3 = node_engines("127.0.0.2", "127.0.0.3", "127.0.0.4")
    # node1 by its NodeAddr, beside a hostname that resolves nowhere; node2
    # by its NodeHostname, with no NodeAddr reported.
    slurm.node("node1", address="127.0.0.2", hostname="node1.invalid")
    slurm.node("node2", hostname="127.0.0.3")
    slurm.node("node3", address="127.0.0.4")
    slurm.job(1, "smg-worker", "node[1-2]", ["node1", "node2"])
    slurm.write()
    gateway.start(*WORKER)

    workers = gateway.wait_for_urls({node1.url, node2.url}, "both nodes of the allocation")
    assert all(w["labels"][PROVIDER] == "slurm" for w in workers), workers
    assert sorted(ids_by_url(workers).values()) == [
        "e2e/1/1700000000/0/node1/regular",
        "e2e/1/1700000000/0/node2/regular",
    ]
    assert served_by(gateway, 10) == {node1.served_by, node2.served_by}
    hostnames = [call["argv"] for call in slurm.calls("scontrol") if call["argv"][0] == "show"]
    assert ["show", "hostnames", "node[1-2]"] in hostnames

    slurm.job(2, "smg-worker", "node3", ["node3"])
    slurm.write()
    gateway.wait_for_urls({node1.url, node2.url, node3.url}, "a second allocation joining")

    slurm.remove_job(1)
    slurm.write()
    gateway.wait_for_urls({node3.url}, "the first allocation's nodes leaving with its job")

    slurm.state["jobs"] = []
    slurm.write()
    gateway.wait_for_urls(set(), "an empty job list clearing the fleet")


def test_a_requeue_and_an_address_change_replace_the_worker(gateway, slurm, node_engines):
    first, moved = node_engines("127.0.0.2", "127.0.0.3")
    slurm.node("node1", address="127.0.0.2")
    slurm.job(1, "smg-worker", "node1", ["node1"])
    slurm.write()
    gateway.start(*WORKER)
    (worker,) = gateway.wait_for_urls({first.url}, "the job's worker")
    before = worker["labels"][ID]

    # Requeued: the same job id, started again.
    slurm.update_job(1, restart_cnt=1, start_time=1_700_000_100)
    slurm.write()
    workers = gateway.wait_for(
        "the requeued job's new instance",
        lambda workers: ids_by_url(workers) == {first.url: "e2e/1/1700000100/1/node1/regular"},
    )
    assert workers[0]["labels"][ID] != before

    # The node's address changes: its worker follows it.
    slurm.node("node1", address="127.0.0.3")
    slurm.write()
    gateway.wait_for_urls({moved.url}, "the worker at the node's new address")


def test_a_failed_lookup_keeps_the_fleet_and_a_late_engine_registers(gateway, slurm, node_engines):
    (node1,) = node_engines("127.0.0.2")
    slurm.node("node1", address="127.0.0.2")
    slurm.node("node2", address="127.0.0.3")
    slurm.job(1, "smg-worker", "node1", ["node1"])
    slurm.write()
    gateway.start(*WORKER)
    gateway.wait_for_urls({node1.url}, "the first job's worker")

    # The node lookup fails while a second job appears: the poll is
    # rejected whole, so neither change applies and nothing is removed.
    slurm.state["fail"] = ["nodes"]
    slurm.job(2, "smg-worker", "node2", ["node2"])
    slurm.write()
    gateway.wait_for_log("Slurm poll failed", "the failed node lookup reported")
    failed = len(slurm.calls("scontrol"))
    deadline = time.monotonic() + 30
    while len(slurm.calls("scontrol")) < failed + 4:
        assert time.monotonic() < deadline, "the gateway stopped polling"
        time.sleep(0.25)
    assert {w["url"] for w in gateway.workers() or []} == {node1.url}

    # Restored: node2 is a candidate now, but nothing listens there, so it
    # gets no worker.
    slurm.state["fail"] = []
    slurm.write()
    gateway.wait_for_log("Slurm lists 2 worker(s)", "the restored poll taking both jobs")
    time.sleep(3)
    assert {w["url"] for w in gateway.workers() or []} == {node1.url}

    # Its engine starts, with Slurm unchanged: a later poll registers it.
    (node2,) = node_engines("127.0.0.3")
    gateway.wait_for_urls({node1.url, node2.url}, "the late engine registered")


def test_a_restarted_gateway_rebuilds_the_fleet(gateway, slurm, node_engines):
    node1, node2 = node_engines("127.0.0.2", "127.0.0.3")
    slurm.node("node1", address="127.0.0.2")
    slurm.node("node2", address="127.0.0.3")
    slurm.job(7, "smg-worker", "node[1-2]", ["node1", "node2"], start_time=1_700_000_007)
    slurm.write()
    gateway.start(*WORKER)
    before = ids_by_url(gateway.wait_for_urls({node1.url, node2.url}, "the fleet"))

    gateway.restart(*WORKER)
    after = ids_by_url(gateway.wait_for_urls({node1.url, node2.url}, "the fleet rebuilt"))
    assert after == before


def test_job_names_map_to_roles_within_the_scope(gateway, slurm, port, mock_worker):
    """Two prefill job names, a decode one, and jobs the mapping or the scope
    leaves out. Only the mapped jobs of user `svc` in account `acct` and
    partition `gpu` get workers, with the bootstrap port on prefill alone."""
    addresses = ["127.0.0.2", "127.0.0.3", "127.0.0.4", "127.0.0.5", "127.0.0.6", "127.0.0.7"]
    for address in addresses:
        mock_worker(
            *("--host", address, "--http-base-port", str(port), "--http-count", "1"),
            *("--model", MODEL),
            ready_port=port,
            ready_host=address,
        )
    for n, address in enumerate(addresses, start=1):
        slurm.node(f"node{n}", address=address)
    scope = {"user": "svc", "account": "acct", "partition": "gpu"}
    slurm.job(1, "smg-prefill-a", "node1", ["node1"], **scope)
    slurm.job(2, "smg-prefill-b", "node2", ["node2"], **scope)
    slurm.job(3, "smg-decode", "node3", ["node3"], **scope)
    # A name no flag maps, another user's job, and one in another partition.
    slurm.job(4, "smg-other", "node4", ["node4"], **scope)
    slurm.job(5, "smg-decode", "node5", ["node5"], **{**scope, "user": "alice"})
    slurm.job(6, "smg-decode", "node6", ["node6"], **{**scope, "partition": "cpu"})
    slurm.write()
    gateway.start(
        "--pd-disaggregation",
        *("--slurm-prefill-job-name", "smg-prefill-a"),
        *("--slurm-prefill-job-name", "smg-prefill-b"),
        *("--slurm-decode-job-name", "smg-decode"),
        *("--slurm-bootstrap-port", "8998"),
        *("--slurm-user", "svc"),
        *("--slurm-account", "acct"),
        *("--slurm-partition", "gpu"),
    )

    prefill_a, prefill_b, decode = (f"http://{address}:{port}" for address in addresses[:3])
    workers = gateway.wait_for_urls({prefill_a, prefill_b, decode}, "the mapped jobs' workers")
    roles = {w["url"]: (w["worker_type"], w.get("bootstrap_port")) for w in workers}
    assert roles == {
        prefill_a: ("prefill", 8998),
        prefill_b: ("prefill", 8998),
        decode: ("decode", None),
    }

    assert slurm.calls("squeue")[-1]["argv"] == [
        "--json",
        "--states=RUNNING",
        "--name=smg-decode,smg-prefill-a,smg-prefill-b",
        "--user=svc",
        "--account=acct",
        "--partition=gpu",
    ]
