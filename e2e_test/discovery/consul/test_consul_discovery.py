"""Consul-discovery e2e against a real Consul agent (see conftest.py).

The scenarios: passing instances becoming workers, and a failing check taking
one away, as Consul reports it; metadata configuring PD workers, an invalid
instance left out alone, and an edit updating a worker in place; a lost Consul
keeping the fleet, and one that comes back empty read afresh; a gateway
restart; ACLs (no token, a token that cannot read the nodes, one that can);
the agent's default token reading for a gateway with none, the last instance
leaving included; and HTTPS with a client certificate. What a real agent does
not do on demand
(an index going backwards or answered as 0) is covered by the unit tests
against a scripted Consul.

Every gateway here runs with the harness's one-second periodic pass. The
pass never reads Consul: it reconciles the list the last answer gave, so a
change still reaches the gateway only as the answer to a blocking query. What
the pass does is retry what an answer's own pass could not apply, such as a
removal that arrives while the instance's registration job is still in
flight; with the pass switched off (an hour, once), that removal waited for
the next change and the case timed out.
"""

from __future__ import annotations

import time

from ..common import MODEL, free_port_range, served_by
from .harness import SERVICE, consul_gateway

PROVIDER = "smg.ai/discovery-provider"
ID = "smg.ai/discovery-id"


def urls(gateway) -> set[str]:
    return {w["url"] for w in gateway.workers() or []}


def ids(workers: list[dict]) -> dict[str, str]:
    return {w["url"]: w.get("labels", {}).get(ID, "") for w in workers}


def test_passing_instances_are_workers_as_consul_reports_them(agent, gateway, engines):
    """Every change below reaches the gateway as the answer to a blocking
    query; the periodic pass only applies what an answer's pass had to hold
    back (see the module docstring)."""
    a, b = engines(2)
    gateway.start()
    agent.register("node-a", "worker-a", a.port)
    agent.register("node-b", "worker-b", b.port)
    workers = gateway.wait_for_urls({a.url, b.url}, "both passing instances")
    assert all(w["labels"][PROVIDER] == "consul" for w in workers), workers
    assert ids(workers) == {a.url: "dc1/node-a/worker-a", b.url: "dc1/node-b/worker-b"}
    assert served_by(gateway, 4) == {a.served_by, b.served_by}

    agent.register("node-b", "worker-b", b.port, status="critical")
    gateway.wait_for_urls({a.url}, "the failing instance removed")
    # Warning is not passing either.
    agent.register("node-b", "worker-b", b.port, status="warning")
    time.sleep(3)
    assert urls(gateway) == {a.url}
    agent.register("node-b", "worker-b", b.port)
    gateway.wait_for_urls({a.url, b.url}, "the instance passing again")

    agent.deregister("node-a", "worker-a")
    gateway.wait_for_urls({b.url}, "the deregistered instance removed")


def test_metadata_configures_pd_workers_and_an_edit_updates_one(agent, gateway, mock_worker):
    base = free_port_range(3)
    mock_worker(
        *("--host", "127.0.0.1", "--http-base-port", str(base), "--http-count", "3"),
        *("--model", MODEL),
        ready_port=base + 2,
    )
    prefill, decode, invalid = base, base + 1, base + 2

    def url(port: int) -> str:
        return f"http://127.0.0.1:{port}"

    def register_prefill(bootstrap_port: int) -> None:
        # Its own address stands before its node's, where nothing listens.
        agent.register(
            "gpu-1",
            "prefill-0",
            prefill,
            node_address="192.0.2.1",
            address="127.0.0.1",
            meta={
                "smg_worker_type": "prefill",
                "smg_bootstrap_port": str(bootstrap_port),
                "smg_kv_connector": "MooncakeConnector",
                "smg_kv_role": "kv_producer",
                "smg_kv_engine_id": "prefill-0",
                "team": "inference",
            },
        )

    register_prefill(29700)
    agent.register("gpu-2", "decode-0", decode, meta={"smg_worker_type": "decode"})
    agent.register("gpu-3", "verifier-0", invalid, meta={"smg_worker_type": "verifier"})
    gateway.start("--pd-disaggregation")

    workers = gateway.wait_for_urls({url(prefill), url(decode)}, "the PD fleet")
    by_url = {w["url"]: w for w in workers}
    assert by_url[url(prefill)]["worker_type"] == "prefill", by_url
    assert by_url[url(prefill)].get("bootstrap_port") == 29700, by_url
    assert by_url[url(prefill)].get("kv_connector") == "MooncakeConnector", by_url
    assert by_url[url(prefill)].get("kv_role") == "kv_producer", by_url
    assert by_url[url(prefill)].get("kv_engine_id") == "prefill-0", by_url
    assert by_url[url(decode)]["worker_type"] == "decode", by_url
    assert by_url[url(decode)].get("bootstrap_port") is None, by_url
    gateway.wait_for_log(
        "left out instance dc1/gpu-3/verifier-0: smg_worker_type 'verifier'",
        "the invalid instance reported",
    )

    # An edit updates the worker in place: the same instance, a new port.
    before = ids(workers)[url(prefill)]
    register_prefill(29701)
    workers = gateway.wait_for(
        "the new bootstrap port",
        lambda workers: any(
            w["url"] == url(prefill) and w.get("bootstrap_port") == 29701 for w in workers
        ),
    )
    assert ids(workers)[url(prefill)] == before


def test_a_model_id_in_the_metadata_names_the_worker_model(agent, gateway, engines):
    (a,) = engines(1)
    agent.register("node-a", "worker-a", a.port, meta={"smg_model_id": "consul-named"})
    gateway.start()
    gateway.wait_for(
        "the worker under the model Consul names",
        lambda workers: [w.get("model_id") for w in workers] == ["consul-named"],
    )


def test_a_lost_consul_keeps_the_fleet_and_an_empty_one_is_read_afresh(agent, gateway, engines):
    (a,) = engines(1)
    agent.register("node-a", "worker-a", a.port)
    gateway.start()
    gateway.wait_for_urls({a.url}, "the worker")

    agent.stop_container()
    gateway.wait_for_log("Consul query failed", "the failed query reported")
    time.sleep(3)
    assert urls(gateway) == {a.url}

    # Dev mode keeps nothing: back, Consul lists no instance, and that answer
    # stands; registered again, the instance returns.
    agent.start_container()
    gateway.wait_for_urls(set(), "the empty Consul read")
    agent.register("node-a", "worker-a", a.port)
    gateway.wait_for_urls({a.url}, "the instance registered again")


def test_a_restarted_gateway_rebuilds_the_fleet(agent, gateway, engines):
    a, b = engines(2)
    agent.register("node-a", "worker-a", a.port)
    agent.register("node-b", "worker-b", b.port)
    gateway.start()
    before = ids(gateway.wait_for_urls({a.url, b.url}, "the fleet"))
    gateway.restart()
    after = ids(gateway.wait_for_urls({a.url, b.url}, "the fleet rebuilt"))
    assert after == before


def test_acls_apply_nothing_without_a_token_that_reads_the_nodes(acl_agent, engines, tmp_path):
    (a,) = engines(1)
    acl_agent.register("node-a", "worker-a", a.port)
    service = f'service "{SERVICE}" {{ policy = "read" }}'
    service_only = acl_agent.token(service)
    reader = acl_agent.token(f'{service}\nnode_prefix "" {{ policy = "read" }}')

    # Consul answers both with an empty list. With no token it is applied, and
    # warned about; the service-only one Consul marks as filtered, and it is
    # rejected. Neither registers the instance.
    for name, env, needle in [
        ("no-token", {}, "no ACL token is set"),
        ("service-only", {"CONSUL_HTTP_TOKEN": service_only}, "node:read"),
    ]:
        gateway = consul_gateway(acl_agent, tmp_path / f"{name}.log", env=env)
        try:
            gateway.start()
            gateway.wait_for_log(needle, f"the {name} answer rejected")
            assert gateway.workers() == []
        finally:
            gateway.stop()

    log = tmp_path / "reader.log"
    gateway = consul_gateway(acl_agent, log, env={"CONSUL_HTTP_TOKEN": reader})
    try:
        gateway.start()
        gateway.wait_for_urls({a.url}, "the instance a reader reads")
    finally:
        gateway.stop()
    assert reader not in log.read_text()


def test_the_agents_default_token_reads_for_a_gateway_with_none(acl_agent, engines, tmp_path):
    """Consul reads a request with no token through the agent's default
    token. Under default deny the gateway then sees the instances, and the
    last one leaving empties the fleet rather than being kept."""
    (a,) = engines(1)
    acl_agent.register("node-a", "worker-a", a.port)
    reader = acl_agent.token(
        f'service "{SERVICE}" {{ policy = "read" }}\nnode_prefix "" {{ policy = "read" }}'
    )
    acl_agent.put("/v1/agent/token/default", {"Token": reader})
    gateway = consul_gateway(acl_agent, tmp_path / "default-token.log")
    try:
        gateway.start()
        gateway.wait_for_urls({a.url}, "the instance read through the default token")
        acl_agent.deregister("node-a", "worker-a")
        gateway.wait_for_urls(set(), "the last instance leaving")
    finally:
        gateway.stop()
        acl_agent.session.put(
            f"{acl_agent.address}/v1/agent/token/default", json={"Token": ""}, timeout=10
        )


def test_https_needs_the_client_certificate_consul_asks_for(tls_agent, tls_dir, engines, tmp_path):
    (a,) = engines(1)
    tls_agent.register("node-a", "worker-a", a.port)
    ca = {"CONSUL_CACERT": str(tls_dir / "consul-agent-ca.pem")}

    gateway = consul_gateway(tls_agent, tmp_path / "no-client-cert.log", env=ca)
    try:
        gateway.start()
        gateway.wait_for_log("Consul query failed", "the handshake refused")
        assert gateway.workers() == []
    finally:
        gateway.stop()

    client = {
        "CONSUL_CLIENT_CERT": str(tls_dir / "dc1-client-consul-0.pem"),
        "CONSUL_CLIENT_KEY": str(tls_dir / "dc1-client-consul-0-key.pem"),
    }
    gateway = consul_gateway(tls_agent, tmp_path / "client-cert.log", env={**ca, **client})
    try:
        gateway.start()
        gateway.wait_for_urls({a.url}, "the instance read over HTTPS")
    finally:
        gateway.stop()
