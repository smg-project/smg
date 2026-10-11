"""CPU unit tests for the vendor-conformance suite logic (no engine, no gateway).

Runs in the harness unit-test job (``pytest --noconftest``) alongside
``e2e_test/infra`` / ``e2e_test/fixtures``. Uses the real checked-in
baselines for the loader/selection/skeleton invariants and fabricated
mini-clusters for the judgment matrix.
"""

from __future__ import annotations

import json

import pytest
from vendor_compat import conformance

from vendor_probe import compat_diff

# ---------------------------------------------------------------------------
# representative selection
# ---------------------------------------------------------------------------


def test_representatives_small_cluster_returns_all():
    assert conformance.representatives(["b", "a"]) == ["a", "b"]


def test_representatives_caps_and_spreads():
    ids = [f"p{i:02d}" for i in range(10)]
    picks = conformance.representatives(ids)
    assert picks == ["p00", "p05", "p09"]


def test_representatives_deterministic_under_input_order():
    ids = [f"p{i:02d}" for i in range(7)]
    assert conformance.representatives(ids) == conformance.representatives(list(reversed(ids)))


def test_selection_full_mode_takes_every_member(monkeypatch):
    monkeypatch.delenv(conformance.FULL_ENV, raising=False)
    partial = conformance.selection("anthropic")
    monkeypatch.setenv(conformance.FULL_ENV, "1")
    full = conformance.selection("anthropic")
    by_id = conformance.cluster_by_id("anthropic")
    for cid, members in full.items():
        assert len(members) == by_id[cid]["probe_count"]
        assert set(partial[cid]) <= set(members)
        assert len(partial[cid]) <= conformance.REPRESENTATIVES_PER_CLUSTER


# ---------------------------------------------------------------------------
# probe subset construction (real probe matrix + real baseline)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_replay_probes_closure_and_order(provider):
    probes, sel, missing = conformance.replay_probes(provider)
    ids = [p["id"] for p in probes]
    id_set = set(ids)
    # every selected-and-present probe made it in
    for chosen in sel.values():
        for pid in chosen:
            assert pid in id_set or pid in missing
    # dependency closure: everything a selected probe waits on is included,
    # and appears before it (order_and_wire's topological order)
    pos = {pid: i for i, pid in enumerate(ids)}
    for p in probes:
        for dep in p["_waits"]:
            assert dep in id_set
            assert pos[dep] < pos[p["id"]]
    # baseline and probe set are from the same commit today: nothing stale
    assert missing == []


def test_representative_volume_is_bounded():
    for provider in ("openai", "anthropic"):
        probes, sel, _ = conformance.replay_probes(provider)
        selected = sum(len(v) for v in sel.values())
        assert selected <= conformance.REPRESENTATIVES_PER_CLUSTER * len(sel)
        # dependency closure adds curated-chain probes but must not balloon
        assert len(probes) < selected + 300


# ---------------------------------------------------------------------------
# baseline fidelity: skeletons and synthesized records reproduce signatures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_skeletons_round_trip_field_paths(provider):
    for c in conformance.clusters(provider):
        if c["kind"] != "json":
            continue
        body = compat_diff.skeleton_from_paths(c["field_paths"])
        got = compat_diff.field_paths(body) if body is not None else set()
        assert got == set(c["field_paths"]), c["cluster_id"]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_synth_records_reproduce_cluster_keys(provider):
    clusters = list(conformance.clusters(provider))
    records = compat_diff.synth_vendor_records(clusters)
    for c in clusters:
        key = compat_diff.vendor_cluster_key(records[c["members"][0]["id"]], provider)
        assert key[1] == c["status"], c["cluster_id"]
        assert list(key[2]) == (c.get("event_types") or c.get("field_paths") or []), c["cluster_id"]


def test_allowlist_covers_only_known_sigs():
    for provider in ("openai", "anthropic"):
        sigs = {c["sig_hash"] for c in conformance.clusters(provider)}
        for sig, entry in conformance.allowlist(provider).items():
            assert sig in sigs, f"allowlist entry {entry['cluster_id']} orphaned"


# ---------------------------------------------------------------------------
# judgment matrix (fabricated mini-clusters)
# ---------------------------------------------------------------------------

_PATHS = ["id", "object", "status", "usage", "usage.total_tokens"]


def _mini_cluster(status=200, sig="feedfeedfeedfeed"):
    return {
        "cluster_id": "T000",
        "sig_hash": sig,
        "kind": "json",
        "status": status,
        "field_paths": list(_PATHS),
        "probe_count": 2,
        "members": [
            {"id": "t.probe.a", "category": "unit", "expect": "ok"},
            {"id": "t.probe.b", "category": "unit", "expect": "ok"},
        ],
        "exemplar_probe": "t.probe.a",
        "exemplar_kind": "skeleton",
    }


def _smg_record(pid, status=200, body=None):
    return {
        "probe_id": pid,
        "recorded": True,
        "response": {"status": status, "body": body},
    }


def _matching_body():
    return {"id": "x", "object": "response", "status": "completed", "usage": {"total_tokens": 1}}


def _no_allowlist(monkeypatch):
    monkeypatch.setattr(conformance, "allowlist", lambda provider: {})


def test_judge_conformant(monkeypatch):
    _no_allowlist(monkeypatch)
    cluster = _mini_cluster()
    records = {m["id"]: _smg_record(m["id"], body=_matching_body()) for m in cluster["members"]}
    out = conformance.judge_cluster("openai", cluster, ["t.probe.a", "t.probe.b"], records)
    assert out["disposition"] == "conformant"
    assert out["verdict"] == "exact"


def test_judge_new_divergence_s1(monkeypatch):
    _no_allowlist(monkeypatch)
    cluster = _mini_cluster()
    records = {
        "t.probe.a": _smg_record("t.probe.a", status=500, body={"error": "boom"}),
        "t.probe.b": _smg_record("t.probe.b", body=_matching_body()),
    }
    out = conformance.judge_cluster("openai", cluster, ["t.probe.a", "t.probe.b"], records)
    assert out["disposition"] == "new-divergence"
    assert out["verdict"] == "S1"
    assert out["s1_kinds"] == ["server-error"]


def test_judge_expected_divergence_when_allowlisted(monkeypatch):
    cluster = _mini_cluster()
    monkeypatch.setattr(
        conformance,
        "allowlist",
        lambda provider: {cluster["sig_hash"]: {"verdict": "S1", "note": "grandfathered"}},
    )
    records = {
        "t.probe.a": _smg_record("t.probe.a", status=500, body={"error": "boom"}),
        "t.probe.b": _smg_record("t.probe.b", body=_matching_body()),
    }
    out = conformance.judge_cluster("openai", cluster, ["t.probe.a", "t.probe.b"], records)
    assert out["disposition"] == "expected-divergence"


def test_judge_severity_escalation(monkeypatch):
    cluster = _mini_cluster()
    monkeypatch.setattr(
        conformance,
        "allowlist",
        lambda provider: {cluster["sig_hash"]: {"verdict": "S2", "note": ""}},
    )
    records = {
        "t.probe.a": _smg_record("t.probe.a", status=500, body={"error": "boom"}),
        "t.probe.b": _smg_record("t.probe.b", body=_matching_body()),
    }
    out = conformance.judge_cluster("openai", cluster, ["t.probe.a", "t.probe.b"], records)
    assert out["disposition"] == "severity-escalation"


def _s1_server_error_allowlist(cluster):
    return {cluster["sig_hash"]: {"verdict": "S1", "s1_kinds": ["server-error"], "note": ""}}


def test_judge_s1_kind_outside_the_allowlisted_kinds_escalates(monkeypatch):
    # grandfathered: the gateway answered 5xx where the vendor gave 404 ...
    cluster = _mini_cluster(status=404)
    monkeypatch.setattr(
        conformance, "allowlist", lambda provider: _s1_server_error_allowlist(cluster)
    )
    # ... but now it accepts what the vendor rejects: S1 of another kind
    records = {m["id"]: _smg_record(m["id"], body=_matching_body()) for m in cluster["members"]}
    out = conformance.judge_cluster("openai", cluster, ["t.probe.a", "t.probe.b"], records)
    assert out["verdict"] == "S1"
    assert out["s1_kinds"] == ["accepts-invalid"]
    assert out["disposition"] == "severity-escalation"
    message = conformance.failure_message("openai", cluster, out)
    assert "['server-error'] but now also ['accepts-invalid']" in message


def test_judge_s1_kind_within_the_allowlisted_kinds_is_expected(monkeypatch):
    cluster = _mini_cluster(status=404)
    monkeypatch.setattr(
        conformance, "allowlist", lambda provider: _s1_server_error_allowlist(cluster)
    )
    records = {
        m["id"]: _smg_record(m["id"], status=500, body={"error": "boom"})
        for m in cluster["members"]
    }
    out = conformance.judge_cluster("openai", cluster, ["t.probe.a", "t.probe.b"], records)
    assert out["s1_kinds"] == ["server-error"]
    assert out["disposition"] == "expected-divergence"


def test_judge_no_coverage_when_replay_missing(monkeypatch):
    _no_allowlist(monkeypatch)
    cluster = _mini_cluster()
    out = conformance.judge_cluster("openai", cluster, ["t.probe.a", "t.probe.b"], {})
    assert out["disposition"] == "no-coverage"


def test_field_drift_is_s2_not_s1(monkeypatch):
    _no_allowlist(monkeypatch)
    cluster = _mini_cluster()
    body = _matching_body()
    del body["usage"]  # vendor-only paths: usage, usage.total_tokens
    records = {m["id"]: _smg_record(m["id"], body=body) for m in cluster["members"]}
    out = conformance.judge_cluster("openai", cluster, ["t.probe.a", "t.probe.b"], records)
    assert out["disposition"] == "new-divergence"
    assert out["verdict"] == "S2"


def test_replay_health_problem_flags_an_empty_replay():
    health = conformance.replay_health({"p": {"probe_id": "p", "skipped": True}})
    assert health["attempted"] == 0
    assert "nothing was compared" in conformance.replay_health_problem(health, 0.5)


def test_replay_health_problem_flags_a_dead_gateway_and_passes_a_live_one():
    live = _smg_record("a", body={})
    dead = {"probe_id": "b", "recorded": False, "error": "connect"}
    unhealthy = conformance.replay_health({"a": live, "b": dead, "c": dict(dead, probe_id="c")})
    assert "1/3" in conformance.replay_health_problem(unhealthy, 0.5)
    healthy = conformance.replay_health({"a": live, "b": dict(live, probe_id="b")})
    assert conformance.replay_health_problem(healthy, 0.5) is None


def test_triage_line_is_valid_allowlist_json(monkeypatch):
    _no_allowlist(monkeypatch)
    cluster = _mini_cluster()
    records = {
        "t.probe.a": _smg_record("t.probe.a", status=500, body={"error": "boom"}),
        "t.probe.b": _smg_record("t.probe.b", body=_matching_body()),
    }
    out = conformance.judge_cluster("openai", cluster, ["t.probe.a", "t.probe.b"], records)
    line = conformance.triage_allowlist_line("openai", cluster, out)
    entry = json.loads(line)
    assert entry["provider"] == "openai"
    assert entry["sig_hash"] == cluster["sig_hash"]
    assert entry["verdict"] == "S1"
    assert entry["note"].startswith("TO-TRIAGE")
    msg = conformance.failure_message("openai", cluster, out)
    assert "TO-TRIAGE" in msg
    assert "t.probe.a" in msg
