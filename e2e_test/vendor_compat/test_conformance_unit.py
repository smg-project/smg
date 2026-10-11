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

from vendor_probe import compat_diff, genmatrix, runner

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


@pytest.mark.parametrize("provider", ["openai", "anthropic", "openai-chat"])
def test_skeletons_round_trip_field_paths(provider):
    for c in conformance.clusters(provider):
        if c["kind"] != "json":
            continue
        body = compat_diff.skeleton_from_paths(c["field_paths"])
        got = compat_diff.field_paths(body) if body is not None else set()
        assert got == set(c["field_paths"]), c["cluster_id"]


@pytest.mark.parametrize("provider", ["openai", "anthropic", "openai-chat"])
def test_synth_records_reproduce_cluster_keys(provider):
    clusters = list(conformance.clusters(provider))
    records = compat_diff.synth_vendor_records(clusters)
    for c in clusters:
        key = compat_diff.vendor_cluster_key(records[c["members"][0]["id"]], provider)
        assert key[1] == c["status"], c["cluster_id"]
        assert list(key[2]) == (c.get("event_types") or c.get("field_paths") or []), c["cluster_id"]


def test_chat_stream_core_ignores_how_the_model_filled_its_chunks():
    # the vendor sends role with the first tool_calls delta and ends with
    # finish_reason tool_calls; a text model sends role alone and ends with stop
    vendor = [
        "delta.role+tool_calls",
        "delta.tool_calls",
        "delta.tool_calls",
        "delta.empty.finish:tool_calls",
        "usage",
        "[DONE]",
    ]
    text = [
        "delta.role",
        "delta.content",
        "delta.content",
        "delta.empty.finish:stop",
        "usage",
        "[DONE]",
    ]
    divergent, detail = compat_diff.compare_events(vendor, text, "openai-chat")
    assert not divergent, detail
    assert (
        detail["vendor_core"]
        == detail["smg_core"]
        == ["delta.role", "finish.finish:<terminal>", "usage", "[DONE]"]
    )
    # a chunk that carries content and reasoning at once is content-bound too
    mixed = [
        "delta.role",
        "delta.reasoning",
        "delta.content+reasoning",
        "delta.content",
        "delta.empty.finish:stop",
        "[DONE]",
    ]
    assert not compat_diff.compare_events(mixed, text[:4] + ["[DONE]"], "openai-chat")[0]


def test_chat_stream_core_keeps_structural_differences():
    vendor = ["delta.role", "delta.content", "delta.empty.finish:stop", "usage", "[DONE]"]
    # a finish class outside stop/length/tool_calls is structural, so is a missing usage chunk
    assert compat_diff.compare_events(
        vendor,
        ["delta.role", "delta.content", "delta.empty.finish:content_filter", "usage", "[DONE]"],
        "openai-chat",
    )[0]
    assert compat_diff.compare_events(
        vendor, ["delta.role", "delta.content", "delta.empty.finish:stop", "[DONE]"], "openai-chat"
    )[0]
    assert compat_diff.chat_core_name("delta.content+usage") == "+usage"
    assert compat_diff.chat_core_name("delta.content+") is None


def test_chat_stream_core_flags_a_role_repeated_on_every_chunk():
    vendor = [
        "delta.role",
        "delta.content",
        "delta.content",
        "delta.empty.finish:stop",
        "usage",
        "[DONE]",
    ]
    # role on every content chunk collapses to one run that still carries role;
    # a client accumulating the deltas concatenates the repeated role strings
    repeated = ["delta.role+content"] * 3 + ["delta.empty.finish:stop", "usage", "[DONE]"]
    divergent, detail = compat_diff.compare_events(vendor, repeated, "openai-chat")
    assert divergent, detail
    assert detail["vendor_core"][0] == "delta.role"
    assert detail["smg_core"][0] == "delta.role.repeated"
    assert compat_diff.chat_core_name("delta.role+") == "delta.role.repeated"
    assert compat_diff.chat_core_name("delta.role+tool_calls+") == "delta.role.repeated"
    # a multi-choice stream sends one role chunk per choice index, back to back
    # or not; the request's n tells the two apart
    per_index = [
        "delta.role",
        "delta.role",
        "delta.content",
        "delta.empty.finish:stop+",
        "usage",
        "[DONE]",
    ]
    one_choice = ["delta.role", "delta.empty.finish:stop", "usage", "[DONE]"]
    assert compat_diff.compare_events(per_index, one_choice, "openai-chat")[0]
    multi = compat_diff.compare_events(per_index, one_choice, "openai-chat", multi_choice=True)
    assert not multi[0], multi[1]
    assert compat_diff.request_choices({"request": {"body": {"n": 2}}}) == 2
    assert compat_diff.request_choices({"request": {"body": {"n": 1}}}) == 1
    assert compat_diff.request_choices({"request": {"body": None}}) == 1
    assert compat_diff.request_choices(None) == 1

    # classify reads n from the replayed request
    def sse(names):
        return {
            "status": 200,
            "sse_events": [{"event": n, "data": "", "parsed": {}} for n in names],
        }

    vrec = {"probe_id": "t.stream", "response": sse(per_index)}
    srec = {
        "probe_id": "t.stream",
        "recorded": True,
        "request": {"body": {"n": 2}},
        "response": sse(one_choice),
    }
    assert (
        compat_diff.classify("t.stream", vrec, srec, "openai-chat", frozenset())["severity"] != "S3"
    )
    srec["request"]["body"]["n"] = 1
    assert (
        compat_diff.classify("t.stream", vrec, srec, "openai-chat", frozenset())["severity"] == "S3"
    )


def test_allowlist_covers_only_known_sigs():
    for provider in ("openai", "anthropic", "openai-chat"):
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
    assert "append to" in msg


def test_triage_line_for_an_allowlisted_cluster_keeps_what_the_entry_grandfathered(monkeypatch):
    cluster = _mini_cluster()
    prior = {
        "verdict": "S1",
        "s1_kinds": ["accepts-invalid"],
        "first_seen": "2026-10-09",
        "note": "adjudicated: the gateway accepts what the vendor rejects",
    }
    monkeypatch.setattr(conformance, "allowlist", lambda provider: {cluster["sig_hash"]: prior})
    records = {
        m["id"]: _smg_record(m["id"], status=500, body={"error": "boom"})
        for m in cluster["members"]
    }
    out = conformance.judge_cluster("openai", cluster, ["t.probe.a", "t.probe.b"], records)
    assert out["disposition"] == "severity-escalation"
    entry = json.loads(conformance.triage_allowlist_line("openai", cluster, out))
    # load_allowlist keeps the last line per sig_hash: the pasted line replaces
    # the adjudicated one, so it must carry the kinds, date and note it had
    assert entry["s1_kinds"] == ["accepts-invalid", "server-error"]
    assert entry["first_seen"] == "2026-10-09"
    assert entry["note"].startswith("TO-TRIAGE")
    assert prior["note"] in entry["note"]
    msg = conformance.failure_message("openai", cluster, out)
    assert "replace the existing entry" in msg


def _gate_cluster(kinds, sig="feedfeedfeedfeed", verdict="S1"):
    return {
        "cluster_id": "C000",
        "sig_hash": sig,
        "verdict": verdict,
        "s1_kinds": list(kinds),
        "size": 1,
        "categories": ["unit"],
        "example_probes": ["t.probe.a"],
    }


def test_cli_gate_applies_the_allowlisted_s1_kinds_ceiling():
    entries = [{"class": "divergence"}]
    listed = {"feedfeedfeedfeed": {"verdict": "S1", "s1_kinds": ["accepts-invalid"]}}
    violations, _ = compat_diff.run_gate([_gate_cluster(["server-error"])], entries, listed, 0.0)
    assert [v["kind"] for v in violations] == ["s1-kind-escalation"]
    assert violations[0]["s1_kinds"] == ["server-error"]
    assert violations[0]["allowlisted_s1_kinds"] == ["accepts-invalid"]
    # within the listed kinds it is the expected divergence; an entry without
    # kinds keeps the rank-only ceiling (same reading as the conformance suite)
    within = compat_diff.run_gate([_gate_cluster(["accepts-invalid"])], entries, listed, 0.0)
    assert within[0] == []
    rank_only = {"feedfeedfeedfeed": {"verdict": "S1"}}
    assert compat_diff.run_gate([_gate_cluster(["server-error"])], entries, rank_only, 0.0)[0] == []


def test_write_allowlist_keeps_the_adjudication_of_a_persisting_cluster(tmp_path):
    path = tmp_path / "known_divergences.jsonl"
    prior = dict(
        _gate_cluster([], verdict="S2"),
        provider="openai",
        first_seen="2026-10-09",
        note="adjudicated: field drift, tracked",
    )
    other = dict(prior, provider="anthropic", sig_hash="cafecafecafecafe", note="theirs")
    path.write_text(json.dumps(prior) + "\n" + json.dumps(other) + "\n", encoding="utf-8")
    clusters = [
        _gate_cluster([], verdict="S2"),
        _gate_cluster(["server-error"], sig="beefbeefbeefbeef"),
    ]
    assert compat_diff.write_allowlist(path, "openai", clusters, "2026-10-11") == 2
    by_sig = {
        e["sig_hash"]: e for e in map(json.loads, path.read_text(encoding="utf-8").splitlines())
    }
    assert by_sig["feedfeedfeedfeed"]["note"] == prior["note"]
    assert by_sig["feedfeedfeedfeed"]["first_seen"] == "2026-10-09"
    assert by_sig["beefbeefbeefbeef"]["note"] == ""
    assert by_sig["beefbeefbeefbeef"]["first_seen"] == "2026-10-11"
    assert by_sig["cafecafecafecafe"]["note"] == "theirs"


def test_write_allowlist_carries_the_adjudication_from_a_previous_allowlist(tmp_path):
    # the refresh flow writes a new date's file, which does not exist yet: the
    # carry-over comes from the previous dated allowlist (--allowlist)
    old = tmp_path / "old" / "known_divergences.jsonl"
    old.parent.mkdir()
    prior = dict(
        _gate_cluster([], verdict="S2"),
        provider="openai",
        first_seen="2026-10-09",
        note="adjudicated: field drift, tracked",
    )
    other = dict(prior, provider="anthropic", sig_hash="cafecafecafecafe", note="theirs")
    old.write_text(json.dumps(prior) + "\n" + json.dumps(other) + "\n", encoding="utf-8")
    new = tmp_path / "new" / "known_divergences.jsonl"
    new.parent.mkdir()
    clusters = [_gate_cluster([], verdict="S2"), _gate_cluster(["server-error"], sig="beef" * 4)]
    carried = compat_diff.load_allowlist(old, "openai")
    assert compat_diff.write_allowlist(new, "openai", clusters, "2026-10-11", prior=carried) == 2
    by_sig = {
        e["sig_hash"]: e for e in map(json.loads, new.read_text(encoding="utf-8").splitlines())
    }
    assert set(by_sig) == {"feedfeedfeedfeed", "beef" * 4}
    assert by_sig["feedfeedfeedfeed"]["note"] == prior["note"]
    assert by_sig["feedfeedfeedfeed"]["first_seen"] == "2026-10-09"
    assert by_sig["beef" * 4]["note"] == ""
    assert by_sig["beef" * 4]["first_seen"] == "2026-10-11"
    # a first run without the previous allowlist blanks the note and stamps
    # first_seen; a rerun with it restores both (per-field merge, earliest date)
    blank = new.parent / "blank.jsonl"
    compat_diff.write_allowlist(blank, "openai", clusters, "2026-10-11")
    first = {e["sig_hash"]: e for e in map(json.loads, blank.read_text().splitlines())}
    assert first["feedfeedfeedfeed"]["note"] == ""
    assert first["feedfeedfeedfeed"]["first_seen"] == "2026-10-11"
    compat_diff.write_allowlist(blank, "openai", clusters, "2026-10-11", prior=carried)
    rerun = {e["sig_hash"]: e for e in map(json.loads, blank.read_text().splitlines())}
    assert rerun["feedfeedfeedfeed"]["note"] == prior["note"]
    assert rerun["feedfeedfeedfeed"]["first_seen"] == "2026-10-09"


def test_generated_mutations_stream_when_the_mutated_body_streams():
    probes = genmatrix._gen_mutations(
        "t", "gen-mutation", genmatrix.O_EP, genmatrix.O_BASE, genmatrix.O_CTX, genmatrix.O_FIELDS
    )
    streaming = [p for p in probes if p["body"].get("stream") is True]
    assert streaming  # the stream_options mutations and their unknown-sibling probe
    assert all(p["stream"] for p in streaming)
    assert not any(p["stream"] for p in probes if p["body"].get("stream") is not True)


def test_stream_probe_rejected_before_the_stream_keeps_the_json_envelope():
    probe = {"stream": True}
    envelope = {"error": {"type": "invalid_request_error", "message": "max_tokens: -1"}}
    headers = {"content-type": "application/json"}
    response, parsed = runner.shape_response(probe, 400, headers, json.dumps(envelope))
    assert parsed == envelope
    assert response["body"] == envelope
    assert "sse_events" not in response
    assert runner.fingerprint_sig({"response": response}, "anthropic")[0] == "json"
    # an event stream is still parsed as one, as is a response without a declared type
    sse = 'event: message_start\ndata: {"type": "message_start"}\n\n'
    response, _ = runner.shape_response(probe, 200, {"Content-Type": "text/event-stream"}, sse)
    assert [e["event"] for e in response["sse_events"]] == ["message_start"]
    response, _ = runner.shape_response(probe, 200, {}, sse)
    assert len(response["sse_events"]) == 1
    # a non-streaming probe is unchanged
    response, parsed = runner.shape_response({"stream": False}, 200, headers, '{"id": 1}')
    assert parsed == {"id": 1}
    assert response["body"] == {"id": 1}
