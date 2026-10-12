"""Host-process e2e for file discovery.

The Kubernetes lane's scenarios (e2e_test/kind_discovery/test_kind_discovery.py),
as they apply to a manifest: registration and provenance, round-robin over
the discovered fleet, one edit adding or removing several workers, a gateway
restart, a new instance at a reused address, an engine failing its health
checks, a forty-worker fleet, gRPC endpoints, and PD metadata. These reread
every second, as a gateway without a working watch would.

The watcher's own scenarios run with no periodic reread due for an hour, so
a change that applies was read because the gateway saw it happen: a
replaced manifest, an in-place rewrite, a manifest created after startup,
and an invalid edit and its fix.
"""

from __future__ import annotations

import time

from .harness import MODEL, NO_REREAD_SECS, free_port_range, served_by

# How long a watched change may take to apply: the event settling, one read,
# and registration. Far below NO_REREAD_SECS, so no periodic reread can pass
# for the watcher.
WATCHED = 15.0

PROVIDER = "smg.ai/discovery-provider"
ID = "smg.ai/discovery-id"


def ids(workers: list[dict]) -> list[str]:
    return sorted(w.get("labels", {}).get(ID, "") for w in workers)


def ready(workers: list[dict], url: str) -> bool:
    return any(w["url"] == url and w.get("status") == "ready" for w in workers)


def test_registration_provenance_and_round_robin(gateway, manifest, engines):
    a, b = engines(2)
    manifest.write([{"id": "engine-a", "url": a.url}, {"id": "engine-b", "url": b.url}])
    gateway.start()

    workers = gateway.wait_for_urls({a.url, b.url}, "both manifest workers registered")
    assert ids(workers) == ["engine-a", "engine-b"]
    assert all(w["labels"][PROVIDER] == "file" for w in workers), workers

    # Round-robin over the discovered fleet: both engines serve traffic.
    assert served_by(gateway, 10) == {f"served-by-{a.port}", f"served-by-{b.port}"}


def test_one_edit_adds_and_removes_several_workers(gateway, manifest, engines):
    urls = [engine.url for engine in engines(4)]
    manifest.write([{"url": urls[0]}])
    gateway.start()
    workers = gateway.wait_for_urls({urls[0]}, "the first worker registered")
    # Without an id the endpoint names the instance.
    assert ids(workers) == [urls[0]]

    manifest.write([{"url": url} for url in urls])
    gateway.wait_for_urls(set(urls), "three workers added by one edit")

    manifest.write([{"url": urls[3]}])
    gateway.wait_for_urls({urls[3]}, "three workers removed by one edit")


def test_gateway_restart_rebuilds_the_registry_from_the_manifest(gateway, manifest, engines):
    a, b = engines(2)
    manifest.write([{"url": a.url}, {"url": b.url}])
    gateway.start()
    gateway.wait_for_urls({a.url, b.url}, "both workers registered")

    gateway.restart()
    gateway.wait_for_urls({a.url, b.url}, "the registry rebuilt after a restart")


def test_a_new_id_at_the_same_url_replaces_the_worker(gateway, manifest, engines):
    (engine,) = engines(1)
    manifest.write([{"id": "engine-a", "url": engine.url}])
    gateway.start()
    gateway.wait_for("engine-a registered", lambda workers: ids(workers) == ["engine-a"])

    # A producer signals a new instance at a reused address by its id. The
    # old registration is replaced, not kept beside the new one.
    manifest.write([{"id": "engine-a-v2", "url": engine.url}])
    gateway.wait_for(
        "the new instance replacing the old",
        lambda workers: ids(workers) == ["engine-a-v2"],
    )


def test_an_unhealthy_engine_leaves_rotation_and_returns(gateway, manifest, engines):
    a, b = engines(2)
    manifest.write([{"url": a.url}, {"url": b.url}])
    gateway.start()
    gateway.wait_for_urls({a.url, b.url}, "both workers registered")
    assert served_by(gateway, 10) == {f"served-by-{a.port}", f"served-by-{b.port}"}

    # The manifest has no readiness signal: the health checker takes a
    # failing engine out, and the reread brings it back once it recovers.
    a.healthy = False
    gateway.wait_for(
        "the failing engine out of rotation", lambda workers: not ready(workers, a.url)
    )
    assert served_by(gateway, 6) == {f"served-by-{b.port}"}

    a.healthy = True
    gateway.wait_for("the engine back once healthy", lambda workers: ready(workers, a.url))
    assert f"served-by-{a.port}" in served_by(gateway, 10)


def test_a_forty_worker_manifest_registers_and_unwinds(gateway, manifest, mock_worker):
    base = free_port_range(40)
    mock_worker(
        *("--host", "127.0.0.1", "--http-base-port", str(base), "--http-count", "40"),
        *("--model", MODEL),
        ready_port=base + 39,
    )
    urls = {f"http://127.0.0.1:{port}" for port in range(base, base + 40)}
    manifest.write([{"url": url} for url in sorted(urls)])
    gateway.start()
    gateway.wait_for_urls(urls, "forty workers from one manifest")

    manifest.write([])
    gateway.wait_for_urls(set(), "an empty manifest unwinding all forty")


def test_grpc_urls_register_in_grpc_mode(gateway, manifest, mock_worker):
    base = free_port_range(2)
    mock_worker(
        *("--host", "127.0.0.1", "--grpc-base-port", str(base), "--grpc-count", "2"),
        *("--model", MODEL),
        ready_port=base + 1,
    )
    manifest.write([{"url": f"grpc://127.0.0.1:{port}"} for port in (base, base + 1)])
    gateway.start()

    workers = gateway.wait_for("both gRPC workers registered", lambda workers: len(workers) == 2)
    assert all(w.get("connection_mode") == "grpc" for w in workers), workers


def test_pd_metadata_reaches_the_registered_workers(gateway, manifest, mock_worker):
    base = free_port_range(4)
    mock_worker(
        *("--host", "127.0.0.1", "--http-base-port", str(base), "--http-count", "4"),
        *("--model", MODEL),
        ready_port=base + 3,
    )
    prefill_0, prefill_1, decode_0, decode_1 = (f"http://127.0.0.1:{base + i}" for i in range(4))
    manifest.write(
        [
            {
                "url": prefill_0,
                "worker_type": "prefill",
                "bootstrap_port": 29700,
                "kv_connector": "MooncakeConnector",
                "kv_role": "kv_producer",
                "kv_engine_id": "prefill-0",
            },
            {
                "url": prefill_1,
                "worker_type": "prefill",
                "bootstrap_port": 29701,
                "kv_connector": "MooncakeConnector",
                "kv_role": "kv_producer",
                "kv_engine_id": "prefill-1",
            },
            {
                "url": decode_0,
                "worker_type": "decode",
                "kv_connector": "NixlConnector",
                "kv_role": "kv_consumer",
                "kv_engine_id": "decode-0",
            },
            {
                "url": decode_1,
                "worker_type": "decode",
                "kv_connector": "NixlConnector",
                "kv_role": "kv_consumer",
                "kv_engine_id": "decode-1",
            },
        ]
    )
    gateway.start("--pd-disaggregation")

    workers = gateway.wait_for_urls({prefill_0, prefill_1, decode_0, decode_1}, "the PD fleet")
    by_url = {w["url"]: w for w in workers}
    for url, worker_type, bootstrap_port, connector, role, engine_id in [
        (prefill_0, "prefill", 29700, "MooncakeConnector", "kv_producer", "prefill-0"),
        (prefill_1, "prefill", 29701, "MooncakeConnector", "kv_producer", "prefill-1"),
        (decode_0, "decode", None, "NixlConnector", "kv_consumer", "decode-0"),
        (decode_1, "decode", None, "NixlConnector", "kv_consumer", "decode-1"),
    ]:
        worker = by_url[url]
        assert worker["worker_type"] == worker_type, worker
        assert worker.get("bootstrap_port") == bootstrap_port, worker
        assert worker.get("kv_connector") == connector, worker
        assert worker.get("kv_role") == role, worker
        assert worker.get("kv_engine_id") == engine_id, worker


def test_an_in_place_edit_takes_effect(gateway, manifest, mock_worker):
    """Fixing a record in place, keeping its id, reaches the registry. The
    case that motivated it: a prefill worker's bootstrap port corrected."""
    base = free_port_range(2)
    mock_worker(
        *("--host", "127.0.0.1", "--http-base-port", str(base), "--http-count", "2"),
        *("--model", MODEL),
        ready_port=base + 1,
    )
    prefill, decode = f"http://127.0.0.1:{base}", f"http://127.0.0.1:{base + 1}"

    def write(bootstrap_port: int) -> None:
        manifest.write(
            [
                {
                    "id": "prefill-a",
                    "url": prefill,
                    "worker_type": "prefill",
                    "bootstrap_port": bootstrap_port,
                },
                {"id": "decode-a", "url": decode, "worker_type": "decode"},
            ]
        )

    def prefill_registrations(workers: list[dict]) -> list[tuple[str, int | None]]:
        return [
            (w.get("labels", {}).get(ID, ""), w.get("bootstrap_port"))
            for w in workers
            if w["url"] == prefill
        ]

    write(29700)
    gateway.start("--pd-disaggregation")
    gateway.wait_for(
        "the prefill worker on bootstrap port 29700",
        lambda workers: prefill_registrations(workers) == [("prefill-a", 29700)],
    )

    write(29711)
    workers = gateway.wait_for(
        "the corrected bootstrap port registered",
        lambda workers: prefill_registrations(workers) == [("prefill-a", 29711)],
    )
    assert {w["url"] for w in workers} == {prefill, decode}


def wait_for_log(gateway, needle: str, what: str) -> None:
    deadline = time.monotonic() + WATCHED
    while needle not in gateway.log_path.read_text():
        assert time.monotonic() < deadline, f"timed out waiting for {what}; {gateway.log_tail()}"
        time.sleep(0.25)


def test_changes_apply_as_they_happen(gateway, manifest, engines):
    a, b, c = engines(3)
    manifest.write([{"url": a.url}])
    gateway.start(check_interval_secs=NO_REREAD_SECS)
    gateway.wait_for_urls({a.url}, "the startup read registering a")

    # A writer renaming a complete file over the manifest.
    manifest.write([{"url": a.url}, {"url": b.url}])
    gateway.wait_for_urls({a.url, b.url}, "the replaced manifest adding b", within=WATCHED)

    # An editor overwriting the manifest itself.
    manifest.rewrite_in_place([{"url": b.url}, {"url": c.url}])
    gateway.wait_for_urls({b.url, c.url}, "the rewritten manifest swapping a for c", within=WATCHED)

    manifest.write([])
    gateway.wait_for_urls(set(), "the empty manifest clearing the fleet", within=WATCHED)


def test_a_manifest_created_after_startup_applies_when_it_appears(gateway, manifest, engines):
    (engine,) = engines(1)
    gateway.start(check_interval_secs=NO_REREAD_SECS)
    wait_for_log(gateway, "cannot read discovery manifest", "the startup read finding no manifest")
    assert gateway.workers() == []

    manifest.write([{"url": engine.url}])
    gateway.wait_for_urls({engine.url}, "the new manifest's worker registered", within=WATCHED)


def test_an_invalid_edit_is_rejected_at_once_and_its_fix_applies(gateway, manifest, engines):
    a, b = engines(2)
    manifest.write([{"url": a.url}])
    gateway.start(check_interval_secs=NO_REREAD_SECS)
    gateway.wait_for_urls({a.url}, "the startup read registering a")

    manifest.write([{"url": b.url, "worker_typ": "regular"}])
    # The log line proves the watcher read the bad edit, so the unchanged
    # fleet is the provider keeping the last good manifest.
    wait_for_log(gateway, "unknown field `worker_typ`", "the invalid edit rejected")
    assert {w["url"] for w in gateway.workers() or []} == {a.url}

    manifest.write([{"url": b.url}])
    gateway.wait_for_urls({b.url}, "the fixed manifest applied", within=WATCHED)
