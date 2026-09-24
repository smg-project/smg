"""Tests for scripts/ci_node_health.py.

Everything here runs without network: Prometheus and GitHub are replaced by
fakes that return canned API payloads.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "ci_node_health.py"


def _load():
    spec = importlib.util.spec_from_file_location("ci_node_health", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["ci_node_health"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


NOW = datetime(2026, 9, 20, 6, 17, tzinfo=UTC)
H100 = {"10.0.121.183", "10.0.98.28"}


def _sample(metric: dict, value: float) -> dict:
    return {"metric": metric, "value": [NOW.timestamp(), str(value)]}


class FakeProm:
    def __init__(self, table: dict[str, list[dict]]):
        self.table = table
        self.queries: list[str] = []

    def query(self, promql: str) -> list[dict]:
        self.queries.append(promql)
        return self.table.get(promql, [])


def _promql(mod, key: str) -> str:
    return next(pc.promql for pc in mod.PROM_CHECKS if pc.check.key == key)


# --- node_of ---------------------------------------------------------------


def test_node_of_prefers_node_then_hostname_then_instance(mod):
    assert mod.node_of({"node": "10.0.1.1", "Hostname": "x"}) == "10.0.1.1"
    assert mod.node_of({"Hostname": "10.0.1.2"}) == "10.0.1.2"
    assert mod.node_of({"kubernetes_node": "10.0.1.4"}) == "10.0.1.4"
    assert mod.node_of({"instance": "10.0.1.3:9100"}) == "10.0.1.3"
    assert mod.node_of({}) == ""


# --- Prometheus evaluators ---------------------------------------------------


def test_h100_nodes_uses_gpu_capacity(mod):
    prom = FakeProm(
        {
            mod.H100_NODES_QUERY: [
                _sample({"node": "10.0.121.183"}, 8),
                _sample({"node": "10.0.98.28"}, 8),
            ]
        }
    )
    assert mod.h100_nodes(prom) == H100


def test_prom_findings_filters_to_h100_nodes(mod):
    q = _promql(mod, "node_cordoned")
    prom = FakeProm(
        {
            q: [
                _sample({"node": "10.0.98.28"}, 1),
                _sample({"node": "10.0.108.220"}, 1),  # an A10 node: ignored
            ]
        }
    )
    findings = mod.prom_findings(prom, H100)
    assert [(f.check.key, f.scope) for f in findings] == [("node_cordoned", "10.0.98.28")]


def test_prom_findings_detail_uses_metric_labels(mod):
    prom = FakeProm(
        {
            _promql(mod, "npd_condition"): [
                _sample({"node": "10.0.121.183", "condition": "GpuEcc"}, 1)
            ],
            _promql(mod, "gpu_xid"): [_sample({"Hostname": "10.0.121.183", "gpu": "3"}, 2)],
            _promql(mod, "disk_full"): [
                _sample({"instance": "10.0.121.183:9100", "mountpoint": "/"}, 0.91)
            ],
        }
    )
    details = {f.check.key: f.detail for f in mod.prom_findings(prom, H100)}
    assert details["npd_condition"] == "GpuEcc"
    assert details["gpu_xid"] == "gpu3: 2 XID change(s) in the last hour"
    assert details["disk_full"] == "/ 91% used"


def test_prom_findings_empty_when_healthy(mod):
    assert mod.prom_findings(FakeProm({}), H100) == []


def test_every_prom_check_has_a_registered_check(mod):
    for pc in mod.PROM_CHECKS:
        assert mod.CHECKS[pc.check.key] is pc.check
        assert pc.check.severity in {"CRIT", "WARN"}


def test_prom_query_raises_prom_error_on_failure(mod, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(mod.urllib.request, "urlopen", boom)
    with pytest.raises(mod.PromError):
        mod.Prom("http://127.0.0.1:1").query("up")


# --- GitHub evaluators -------------------------------------------------------


class FakeGitHub:
    """Serves canned GET payloads keyed by path; records writes."""

    def __init__(self, table: dict[str, object]):
        self.table = table
        self.writes: list[tuple[str, str, dict]] = []

    @staticmethod
    def key(path: str, params: dict | None) -> str:
        params = params or {}
        for name in ("status", "state"):
            if name in params:
                return f"{path}?{name}={params[name]}"
        return path

    def get(self, path: str, params: dict | None = None):
        page = int((params or {}).get("page", 1))
        payload = self.table.get(self.key(path, params), [])
        if isinstance(payload, dict):
            key = next(k for k in ("workflow_runs", "jobs", "runners") if k in payload)
            items = payload[key] if page == 1 else []
            return {key: items}
        return payload if page == 1 else []

    def paginate(self, path, key, params=None, max_pages=3):
        payload = self.get(path, dict(params or {}, page=1))
        return payload[key] if key else payload

    def post(self, path, body):
        self.writes.append(("POST", path, body))
        return {"number": 999, "html_url": "https://github.com/x/y/issues/999"}

    def patch(self, path, body):
        self.writes.append(("PATCH", path, body))
        return {}


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _job(name, label, created, started=None, status="completed", url="https://j"):
    return {
        "name": name,
        "labels": [label],
        "created_at": _ts(created),
        "started_at": _ts(started) if started else None,
        "status": status,
        "html_url": url,
    }


def _run(run_id, created, name="PR Test"):
    return {
        "id": run_id,
        "name": name,
        "created_at": _ts(created),
        "html_url": f"https://r/{run_id}",
    }


NO_ACTIVE_RUNS = {
    "actions/runs?status=queued": {"workflow_runs": []},
    "actions/runs?status=in_progress": {"workflow_runs": []},
}
NO_ISSUES = {"issues?state=open": [], "issues?state=closed": []}


def test_parse_ts_reads_github_timestamps(mod):
    assert mod.parse_ts("2026-09-20T05:22:31Z") == datetime(2026, 9, 20, 5, 22, 31, tzinfo=UTC)


def test_queue_wait_uses_job_created_to_started(mod):
    gh = FakeGitHub(
        {
            "actions/runs": {"workflow_runs": [_run(1, NOW)]},
            "actions/runs/1/jobs": {
                "jobs": [
                    _job(
                        "e2e-4gpu / run",
                        "4-gpu-h100",
                        NOW - timedelta(minutes=90),
                        NOW - timedelta(minutes=20),
                    ),
                    _job(
                        "e2e-1gpu / run",
                        "1-gpu-h100",
                        NOW - timedelta(minutes=60),
                        NOW - timedelta(minutes=10),
                    ),
                    _job(
                        "pre-commit",
                        "k8s-runner-cpu",
                        NOW - timedelta(minutes=60),
                        NOW - timedelta(minutes=59),
                    ),
                ]
            },
            **NO_ACTIVE_RUNS,
        }
    )
    findings, stats = mod.github_findings(gh, NOW)
    assert sorted(stats.waits_min) == [50.0, 70.0]  # the CPU job is not counted
    assert stats.sampled_runs == 1
    assert [f.check.key for f in findings] == ["queue_wait"]
    assert findings[0].detail.startswith("p50 60 min")


def test_skipped_jobs_do_not_dilute_queue_wait(mod):
    skipped = _job("e2e-4gpu / run", "4-gpu-h100", NOW, NOW)
    skipped["conclusion"] = "skipped"
    gh = FakeGitHub(
        {
            "actions/runs": {"workflow_runs": [_run(1, NOW)]},
            "actions/runs/1/jobs": {
                "jobs": [
                    skipped,
                    _job("e2e-1gpu / run", "1-gpu-h100", NOW - timedelta(minutes=60), NOW),
                ]
            },
            **NO_ACTIVE_RUNS,
        }
    )
    _, stats = mod.github_findings(gh, NOW)
    assert stats.waits_min == [60.0]


def test_queued_h100_job_in_old_in_progress_run_is_starved(mod):
    # The run started 5h ago (outside the 2h wait-statistics window) and one
    # H100 job has been queued for 3h: the starvation check must still see it.
    old = NOW - timedelta(hours=5)
    gh = FakeGitHub(
        {
            "actions/runs": {"workflow_runs": []},
            "actions/runs?status=queued": {"workflow_runs": []},
            "actions/runs?status=in_progress": {"workflow_runs": [_run(1, old)]},
            "actions/runs/1/jobs": {
                "jobs": [
                    _job("build", "k8s-runner-cpu", old, old, status="completed"),
                    _job("e2e-4gpu / run", "4-gpu-h100", NOW - timedelta(hours=3), status="queued"),
                    _job(
                        "e2e-1gpu / run", "1-gpu-h100", NOW - timedelta(minutes=5), status="queued"
                    ),
                ]
            },
        }
    )
    findings, stats = mod.github_findings(gh, NOW)
    assert stats.queued_h100 == 2  # both queued H100 jobs are counted
    assert stats.active_runs == 1
    assert [(f.check.key, f.scope) for f in findings] == [("runner_starved", "4-gpu-h100")]
    assert "queued 180 min" in findings[0].detail


def test_run_in_both_snapshots_is_scanned_once_as_in_progress(mod):
    # A run that moves from queued to in_progress between the two listings
    # shows up in both. It must be scanned once, and its age must not make it
    # a stale queued run.
    old = NOW - timedelta(days=2)
    both = _run(7, old)
    gh = FakeGitHub(
        {
            "actions/runs": {"workflow_runs": []},
            "actions/runs?status=queued": {"workflow_runs": [both]},
            "actions/runs?status=in_progress": {"workflow_runs": [both]},
            "actions/runs/7/jobs": {
                "jobs": [_job("e2e / run", "4-gpu-h100", NOW - timedelta(hours=2), status="queued")]
            },
        }
    )
    findings, stats = mod.github_findings(gh, NOW)
    assert stats.active_runs == 1 and stats.queued_h100 == 1
    assert [f.check.key for f in findings] == ["runner_starved"]


def test_github_findings_never_calls_the_runners_endpoint(mod):
    # GET actions/runners needs repository Administration permission, which the
    # workflow GITHUB_TOKEN cannot be granted.
    calls: list[str] = []
    gh = FakeGitHub({"actions/runs": {"workflow_runs": []}, **NO_ACTIVE_RUNS})
    original = gh.get

    def spy(path, params=None):
        calls.append(path)
        return original(path, params)

    gh.get = spy
    mod.github_findings(gh, NOW)
    assert calls and all(not c.startswith("actions/runners") for c in calls)


def test_stale_queued_runs_older_than_a_day(mod):
    old = NOW - timedelta(days=3)
    fresh = NOW - timedelta(hours=2)
    gh = FakeGitHub(
        {
            "actions/runs": {"workflow_runs": []},
            "actions/runs?status=queued": {
                "workflow_runs": [_run(5, old, "Nightly tau2"), _run(6, fresh)]
            },
            "actions/runs?status=in_progress": {"workflow_runs": []},
            # the zombie run's job would count as starved; stale wins instead
            "actions/runs/5/jobs": {"jobs": [_job("bench", "4-gpu-h100", old, status="queued")]},
            "actions/runs/6/jobs": {"jobs": []},
        }
    )
    findings, stats = mod.github_findings(gh, NOW)
    assert [(f.check.key, f.scope) for f in findings] == [("stale_queued_runs", "github")]
    assert "Nightly tau2 run 5 queued 3d" in findings[0].detail
    assert stats.active_runs == 1  # only the fresh queued run was scanned


# --- issue reconciliation ----------------------------------------------------


def _finding(mod, key, scope, detail="d"):
    return mod.Finding(mod.CHECKS[key], scope, detail)


def _issue(mod, number, key, first_seen, last_seen, last_comment=None, nodes=None):
    marker = {
        "key": key,
        "first_seen": first_seen.isoformat(),
        "last_seen": last_seen.isoformat(),
        "last_comment": last_comment.isoformat() if last_comment else None,
        "nodes": {n: first_seen.isoformat() for n in (nodes or [])},
    }
    return {
        "number": number,
        "body": f"old body\n\n{mod.MARKER_PREFIX}{json.dumps(marker)}{mod.MARKER_SUFFIX}",
        "title": "old title",
    }


def test_parse_issue_reads_marker_and_ignores_foreign_issues(mod):
    issue = _issue(
        mod,
        7,
        "node_cordoned",
        NOW - timedelta(hours=3),
        NOW - timedelta(hours=1),
        nodes=["10.0.98.28"],
    )
    state = mod.parse_issue(issue)
    assert state and state.number == 7 and state.key == "node_cordoned"
    assert state.nodes == {"10.0.98.28": NOW - timedelta(hours=3)}
    assert mod.parse_issue({"number": 8, "body": "a human wrote this", "title": "x"}) is None


def test_body_round_trips_through_marker(mod):
    check = mod.CHECKS["node_cordoned"]
    findings = [_finding(mod, "node_cordoned", "10.0.98.28", "unschedulable for over 2h")]
    state = mod.IssueState(0, "node_cordoned", NOW, NOW, None, {"10.0.98.28": NOW})
    body = mod.render_body(check, findings, state, NOW, "https://run/1")
    assert check.meaning in body and check.first_action in body
    assert "| 10.0.98.28 | unschedulable for over 2h |" in body
    assert "https://run/1" in body
    parsed = mod.parse_issue({"number": 1, "body": body, "title": ""})
    assert parsed and parsed.nodes == {"10.0.98.28": NOW}


def test_render_title_lists_scopes(mod):
    check = mod.CHECKS["gpu_xid"]
    fs = [
        _finding(mod, "gpu_xid", "10.0.1.2"),
        _finding(mod, "gpu_xid", "10.0.1.1"),
        _finding(mod, "gpu_xid", "10.0.1.2"),
    ]
    assert mod.render_title(check, fs) == "[node-health][CRIT] GPU XID error: 10.0.1.1, 10.0.1.2"


def test_plan_ops_creates_issue_for_new_finding(mod):
    active = mod.group_by_check([_finding(mod, "node_cordoned", "10.0.98.28")])
    ops = mod.plan_ops([], active, NOW, "https://run/1")
    assert [(o.kind, o.key) for o in ops] == [("create", "node_cordoned")]
    assert ops[0].title.startswith("[node-health][WARN] H100 node cordoned: 10.0.98.28")
    assert mod.MARKER_PREFIX in ops[0].body


def test_plan_ops_updates_open_issue_without_comment(mod):
    first = NOW - timedelta(hours=5)
    state = mod.parse_issue(
        _issue(mod, 7, "node_cordoned", first, NOW - timedelta(hours=1), nodes=["10.0.98.28"])
    )
    active = mod.group_by_check(
        [
            _finding(mod, "node_cordoned", "10.0.98.28"),
            _finding(mod, "node_cordoned", "10.0.69.208"),
        ]
    )
    ops = mod.plan_ops([state], active, NOW, "https://run/2")
    assert [(o.kind, o.number) for o in ops] == [("update", 7)]
    parsed = mod.parse_issue({"number": 7, "body": ops[0].body, "title": ""})
    assert parsed.first_seen == first  # first_seen survives updates
    assert parsed.last_seen == NOW
    assert parsed.nodes["10.0.98.28"] == first  # existing node keeps its first-seen
    assert parsed.nodes["10.0.69.208"] == NOW  # new node gets now


def test_plan_ops_closes_state_finding_only_after_two_missed_runs(mod):
    recent = mod.parse_issue(
        _issue(mod, 1, "node_cordoned", NOW - timedelta(hours=4), NOW - timedelta(minutes=60))
    )
    old = mod.parse_issue(
        _issue(mod, 2, "disk_full", NOW - timedelta(hours=4), NOW - timedelta(minutes=95))
    )
    ops = mod.plan_ops([recent, old], {}, NOW, "https://run/3")
    assert [(o.kind, o.number) for o in ops] == [("close", 2)]


def test_plan_ops_never_closes_event_findings(mod):
    xid = mod.parse_issue(
        _issue(mod, 3, "gpu_xid", NOW - timedelta(days=2), NOW - timedelta(days=2))
    )
    assert mod.plan_ops([xid], {}, NOW, "https://run/4") == []


def test_plan_ops_comments_on_event_at_most_daily(mod):
    fresh = mod.parse_issue(
        _issue(
            mod,
            3,
            "gpu_xid",
            NOW - timedelta(hours=2),
            NOW - timedelta(hours=1),
            last_comment=NOW - timedelta(hours=2),
        )
    )
    stale = mod.parse_issue(
        _issue(
            mod,
            4,
            "gpu_xid",
            NOW - timedelta(days=2),
            NOW - timedelta(days=1),
            last_comment=NOW - timedelta(days=2),
        )
    )
    active = mod.group_by_check(
        [_finding(mod, "gpu_xid", "10.0.1.1", "gpu0: 1 XID change(s) in the last hour")]
    )
    ops_fresh = mod.plan_ops([fresh], active, NOW, "https://run/5")
    assert [o.kind for o in ops_fresh] == ["update"]
    ops_stale = mod.plan_ops([stale], active, NOW, "https://run/5")
    assert [o.kind for o in ops_stale] == ["update", "comment"]
    assert "gpu0: 1 XID change(s)" in ops_stale[1].body
    parsed = mod.parse_issue({"number": 4, "body": ops_stale[0].body, "title": ""})
    assert parsed.last_comment == NOW


def test_plan_ops_ignores_issues_with_unknown_keys(mod):
    weird = mod.parse_issue(
        _issue(mod, 9, "retired_check", NOW - timedelta(days=9), NOW - timedelta(days=9))
    )
    assert mod.plan_ops([weird], {}, NOW, "https://run/6") == []


def test_plan_ops_leaves_unevaluated_checks_open(mod):
    # Prometheus was down: disk_full was not evaluated, so its issue must not
    # be closed as "resolved" even though it has been absent for two runs.
    disk = mod.parse_issue(
        _issue(mod, 2, "disk_full", NOW - timedelta(hours=4), NOW - timedelta(minutes=95))
    )
    stale = mod.parse_issue(
        _issue(mod, 3, "stale_queued_runs", NOW - timedelta(hours=4), NOW - timedelta(minutes=95))
    )
    evaluated = set(mod.CHECKS) - {pc.check.key for pc in mod.PROM_CHECKS}
    ops = mod.plan_ops([disk, stale], {}, NOW, "https://run/7", evaluated=evaluated)
    assert [(o.kind, o.number) for o in ops] == [("close", 3)]


def test_plan_ops_does_not_recreate_event_issue_closed_within_grace(mod):
    active = mod.group_by_check([_finding(mod, "gpu_xid", "10.0.1.1")])
    just_closed = {"gpu_xid": {"10.0.1.1": NOW - timedelta(minutes=20)}}
    assert mod.plan_ops([], active, NOW, "https://run/8", recently_closed=just_closed) == []
    long_ago = {"gpu_xid": {"10.0.1.1": NOW - timedelta(hours=2)}}
    ops = mod.plan_ops([], active, NOW, "https://run/8", recently_closed=long_ago)
    assert [o.kind for o in ops] == ["create"]
    # state checks are never suppressed by a recent close
    cordon = mod.group_by_check([_finding(mod, "node_cordoned", "10.0.1.1")])
    closed = {"node_cordoned": {"10.0.1.1": NOW - timedelta(minutes=20)}}
    ops = mod.plan_ops([], cordon, NOW, "https://run/8", recently_closed=closed)
    assert [o.kind for o in ops] == ["create"]


def test_plan_ops_recreates_event_issue_for_a_new_scope_during_grace(mod):
    # 10.0.1.1's XID issue was closed 20 minutes ago; a fresh XID on 10.0.1.2
    # must still open an issue, listing only the new node.
    active = mod.group_by_check(
        [_finding(mod, "gpu_xid", "10.0.1.1"), _finding(mod, "gpu_xid", "10.0.1.2")]
    )
    closed = {"gpu_xid": {"10.0.1.1": NOW - timedelta(minutes=20)}}
    ops = mod.plan_ops([], active, NOW, "https://run/9", recently_closed=closed)
    assert [(o.kind, o.title) for o in ops] == [
        ("create", "[node-health][CRIT] GPU XID error: 10.0.1.2")
    ]


def test_render_body_links_docs_when_given(mod):
    check = mod.CHECKS["node_cordoned"]
    state = mod.IssueState(0, "node_cordoned", NOW, NOW, None, {})
    body = mod.render_body(check, [], state, NOW, "https://run/1", "https://docs/readme#x")
    assert "[how this monitor works](https://docs/readme#x)" in body
    assert "docs/superpowers" not in body


@pytest.mark.parametrize("closed_minutes", [20, 60])
def test_plan_ops_preserves_event_acknowledgement_on_update(mod, closed_minutes):
    """An open issue must respect the same per-node grace window as creation."""
    first = NOW - timedelta(days=2)
    state = mod.parse_issue(
        _issue(mod, 7, "gpu_xid", first, first, last_comment=first, nodes=["10.0.1.2"])
    )
    active = mod.group_by_check(
        [_finding(mod, "gpu_xid", "10.0.1.1"), _finding(mod, "gpu_xid", "10.0.1.2")]
    )
    closed = {"gpu_xid": {"10.0.1.1": NOW - timedelta(minutes=closed_minutes)}}
    ops = mod.plan_ops([state], active, NOW, "https://run/10", recently_closed=closed)
    assert [o.kind for o in ops] == ["update", "comment"]
    parsed = mod.parse_issue({"number": 7, "body": ops[0].body})
    assert parsed.nodes["10.0.1.2"] == first
    for text in (ops[0].title, ops[0].body, ops[1].body):
        assert ("10.0.1.1" in text) == (closed_minutes == 60)
    assert ("10.0.1.1" in parsed.nodes) == (closed_minutes == 60)


def test_plan_ops_skips_update_when_all_event_scopes_are_suppressed(mod):
    """Acknowledged findings must not refresh or comment on an open event issue."""
    first = NOW - timedelta(days=2)
    state = mod.parse_issue(
        _issue(mod, 7, "gpu_xid", first, first, last_comment=first, nodes=["10.0.1.2"])
    )
    active = mod.group_by_check([_finding(mod, "gpu_xid", "10.0.1.1")])
    closed = {"gpu_xid": {"10.0.1.1": NOW - timedelta(minutes=20)}}
    assert mod.plan_ops([state], active, NOW, "https://run/11", recently_closed=closed) == []
    assert state.last_seen == first
    assert state.last_comment == first
    assert state.nodes == {"10.0.1.2": first}


@pytest.mark.parametrize("nodes", [None, 42, "10.0.1.1", ["10.0.1.1"]])
def test_issue_states_ignores_closed_markers_with_invalid_nodes(mod, nodes):
    """Malformed closed markers must not crash or contribute suppression scopes."""
    marker = json.dumps({"key": "gpu_xid", "nodes": nodes})
    malformed = {
        "number": 2,
        "closed_at": _ts(NOW),
        "body": f"{mod.MARKER_PREFIX}{marker}{mod.MARKER_SUFFIX}",
    }
    valid = _issue(mod, 3, "gpu_xid", NOW, NOW, nodes=["10.0.1.2"])
    valid["closed_at"] = _ts(NOW - timedelta(minutes=30))
    gh = FakeGitHub({"issues?state=open": [], "issues?state=closed": [malformed, valid]})
    states, recently_closed = mod.issue_states(gh)
    assert states == []
    assert recently_closed == {"gpu_xid": {"10.0.1.2": NOW - timedelta(minutes=30)}}


def test_issue_states_reads_open_state_and_latest_close_per_key(mod):
    open_issue = _issue(mod, 1, "node_cordoned", NOW - timedelta(hours=3), NOW - timedelta(hours=1))
    closed_new = dict(
        _issue(
            mod,
            2,
            "gpu_xid",
            NOW - timedelta(days=1),
            NOW - timedelta(days=1),
            nodes=["10.0.1.1", "10.0.1.2"],
        )
    )
    closed_new["closed_at"] = _ts(NOW - timedelta(minutes=30))
    auto_closed_marker = json.dumps({"key": "gpu_xid", "closed": "x"})
    closed_old = {
        "number": 3,
        "closed_at": _ts(NOW - timedelta(days=2)),
        "body": f"Resolved.\n\n{mod.MARKER_PREFIX}{auto_closed_marker}{mod.MARKER_SUFFIX}",
    }
    human = {"number": 4, "closed_at": _ts(NOW), "body": "not ours"}
    gh = FakeGitHub(
        {"issues?state=open": [open_issue], "issues?state=closed": [closed_old, closed_new, human]}
    )
    states, recently_closed = mod.issue_states(gh)
    assert [s.number for s in states] == [1]
    closed_at = NOW - timedelta(minutes=30)
    assert recently_closed == {"gpu_xid": {"10.0.1.1": closed_at, "10.0.1.2": closed_at}}


def test_apply_ops_dry_run_writes_nothing(mod, capsys):
    gh = FakeGitHub({})
    ops = [
        mod.Op("create", None, "node_cordoned", "t", "b"),
        mod.Op("close", 5, "disk_full", "", "b"),
    ]
    mod.apply_ops(gh, ops, dry_run=True)
    assert gh.writes == []
    out = capsys.readouterr().out
    assert "create" in out and "close #5" in out


def test_apply_ops_performs_writes(mod):
    gh = FakeGitHub({})
    ops = [
        mod.Op("create", None, "node_cordoned", "t", "b"),
        mod.Op("update", 7, "node_cordoned", "t2", "b2"),
        mod.Op("comment", 7, "gpu_xid", "", "c"),
        mod.Op("close", 5, "disk_full", "", "b5"),
    ]
    mod.apply_ops(gh, ops, dry_run=False)
    assert gh.writes == [
        ("POST", "issues", {"title": "t", "body": "b", "labels": [mod.ISSUE_LABEL]}),
        ("PATCH", "issues/7", {"title": "t2", "body": "b2"}),
        ("POST", "issues/7/comments", {"body": "c"}),
        ("PATCH", "issues/5", {"body": "b5", "state": "closed", "state_reason": "completed"}),
    ]


# --- summary and main --------------------------------------------------------


def test_fleet_rows_one_per_h100_node_with_defaults(mod):
    prom = FakeProm(
        {
            mod.FLEET_QUERIES["ready"]: [_sample({"node": "10.0.121.183"}, 1)],
            mod.FLEET_QUERIES["cordoned"]: [_sample({"node": "10.0.98.28"}, 1)],
            mod.FLEET_QUERIES["root_pct"]: [_sample({"instance": "10.0.121.183:9100"}, 0.53)],
            mod.FLEET_QUERIES["runner_pods"]: [_sample({"node": "10.0.121.183"}, 4)],
        }
    )
    rows = mod.fleet_rows(prom, H100)
    assert [r["node"] for r in rows] == ["10.0.121.183", "10.0.98.28"]
    a, b = rows
    assert a["ready"] == 1.0 and a["cordoned"] is None
    assert a["root_pct"] == 0.53 and a["runner_pods"] == 4.0
    assert b["ready"] is None and b["cordoned"] == 1.0


def test_render_summary_mentions_findings_and_queue(mod):
    rows = [
        {
            "node": "10.0.98.28",
            "ready": None,
            "cordoned": 1.0,
            "gpus": 8.0,
            "npd_true": None,
            "npd_unknown": None,
            "root_pct": 0.1,
            "raid_pct": 0.0,
            "runner_pods": None,
            "xid_24h": 0.0,
            "oom_24h": None,
        }
    ]
    stats = mod.QueueStats(waits_min=[5.0, 70.0], queued_h100=1, sampled_runs=3, active_runs=2)
    findings = [_finding(mod, "node_cordoned", "10.0.98.28", "unschedulable for over 2h")]
    text = mod.render_summary(rows, stats, findings, NOW)
    assert "| 10.0.98.28 |" in text
    assert "p50 38 min" in text and "p95 " in text
    assert "3 recent runs" in text and "2 queued or in-progress runs" in text
    assert "node_cordoned" in text and "10.0.98.28" in text


def test_main_dry_run_reports_monitor_blind_when_prometheus_is_down(mod, monkeypatch, capsys):
    class DeadProm:
        def __init__(self, *a, **k):
            pass

        def query(self, promql):
            raise mod.PromError("down")

    monkeypatch.setattr(mod, "Prom", DeadProm)
    monkeypatch.setattr(
        mod,
        "GitHub",
        lambda *a, **k: FakeGitHub(
            {"actions/runs": {"workflow_runs": []}, **NO_ACTIVE_RUNS, **NO_ISSUES}
        ),
    )
    rc = mod.main(["--dry-run", "--repo", "x/y"])
    assert rc == 1
    assert "monitor_blind" in capsys.readouterr().out
