#!/usr/bin/env python3
"""Hourly health monitor for the H100 CI nodes.

Reads Prometheus (node-problem-detector, DCGM, node-exporter, kube-state-metrics)
and the GitHub Actions API, evaluates the checks documented in
scripts/k8s-runner-resources/README.md (section "CI node health monitor"), and
keeps one GitHub issue per active check under the ``ci-node-health`` label. Slack is fed by
the GitHub Slack app subscribed to that label, so the only notifications are
"issue opened" and "issue closed".

Runs on the in-cluster ``k8s-runner-cpu`` runner, which reaches Prometheus over
its ClusterIP without auth. Needs no kubeconfig and no secret beyond GITHUB_TOKEN.

Usage:
    ci_node_health.py [--dry-run] [--prom-url URL] [--repo OWNER/NAME]

Environment:
    GITHUB_TOKEN         required to write issues (reads work without it)
    GITHUB_REPOSITORY    default for --repo
    PROM_URL             default for --prom-url
    GITHUB_STEP_SUMMARY  when set, the fleet summary is appended there
    GITHUB_SERVER_URL, GITHUB_REPOSITORY, GITHUB_RUN_ID  build the run link
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

ISSUE_LABEL = "ci-node-health"
H100_LABELS = ("1-gpu-h100", "2-gpu-h100", "4-gpu-h100")
CLOSE_AFTER = timedelta(minutes=90)
COMMENT_EVERY = timedelta(hours=24)
EVENT_REOPEN_GRACE = timedelta(hours=1)  # the gpu_xid lookback window
QUEUE_WINDOW = timedelta(hours=2)
MAX_RECENT_RUNS = 40  # one jobs request per run: bounds the API fan-out
STARVED_AFTER = timedelta(minutes=60)
STALE_RUN_AFTER = timedelta(hours=24)
QUEUE_WAIT_P50_MIN = 30.0
MARKER_PREFIX = "<!-- ci-node-health "
MARKER_SUFFIX = " -->"
DOCS_PATH = "scripts/k8s-runner-resources/README.md#ci-node-health-monitor"
DEFAULT_PROM_URL = "http://prometheus-kube-prometheus-prometheus.monitoring.svc:9090"


# --- check registry ----------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """One monitored condition: what it means and what to do first."""

    key: str
    severity: str  # "CRIT" or "WARN"
    title: str
    meaning: str
    first_action: str
    event: bool = False  # event findings never auto-close


@dataclass(frozen=True)
class Finding:
    """A check that fired for one scope (node, runner label, or \"github\")."""

    check: Check
    scope: str  # node name, runner label, or "github"
    detail: str


CHECKS: dict[str, Check] = {
    c.key: c
    for c in (
        Check(
            "node_not_ready",
            "CRIT",
            "H100 node not Ready",
            "The kubelet stopped reporting. Every runner pod on the node is dead and its "
            "jobs will time out.",
            "`kubectl describe node <node>`. If it does not recover in 15 minutes, cordon it "
            "and open an OCI ticket with the serial from the "
            "`oci.oraclecloud.com/host.serial_number` label.",
        ),
        Check(
            "npd_condition",
            "CRIT",
            "NPD hardware condition",
            "node-problem-detector's OKE GPU/RDMA plugin flagged the node. NPD only sets the "
            "condition; nothing cordons the node on its own.",
            "`kubectl cordon <node>`, read the condition message in `kubectl describe node`, "
            "and open an OCI hardware ticket.",
        ),
        Check(
            "gpu_xid",
            "CRIT",
            "GPU XID error",
            "The NVIDIA driver logged an XID on this GPU in the last hour. Lanes on it may "
            "fail or hang. This issue is closed by a human, not by the monitor.",
            "`dmesg | grep -i xid` on the node. XID 79, 48, 95 mean reset or RMA: cordon "
            "first. Close this issue after the GPU is reset or the XID is confirmed benign.",
            event=True,
        ),
        Check(
            "gpu_row_remap_failure",
            "CRIT",
            "GPU row-remap failure",
            "DCGM reports a failed HBM row remap. The GPU needs a reset or an RMA.",
            "Cordon the node and open an OCI hardware ticket.",
        ),
        Check(
            "disk_full",
            "CRIT",
            "Node disk over 85%",
            "Root (runner emptyDirs and dind storage) or /raid (model cache) is nearly full. "
            "The kubelet starts evicting pods at 85 to 90%.",
            "Root: delete finished runner pods and run `crictl rmi --prune` on the node. "
            "/raid: prune `hub/` snapshots that are not in `e2e_test/infra/model_specs.py`.",
        ),
        Check(
            "runner_starved",
            "CRIT",
            "GPU runner label starved",
            "A job has waited over an hour for a *-h100 runner, or no runner for that label "
            "is registered online. PR #2283 is what this looks like when nobody notices.",
            "`kubectl get autoscalingrunnersets -n actions-runner-system`, then the listener "
            "pod logs and `kubectl get events -n actions-runner-system | grep FailedScheduling`.",
        ),
        Check(
            "monitor_blind",
            "CRIT",
            "Monitor cannot reach Prometheus",
            "Prometheus did not answer, so every node check was skipped this run.",
            "`kubectl -n monitoring get pods | grep prometheus`. The monitor recovers on its own.",
        ),
        Check(
            "node_cordoned",
            "WARN",
            "H100 node cordoned",
            "The node has been unschedulable for over two hours. Its 8 GPUs are out of the CI "
            "pool while 4-GPU lanes queue.",
            "If intentional, assign this issue to yourself and leave it open. Otherwise "
            "`kubectl uncordon <node>`.",
        ),
        Check(
            "queue_wait",
            "WARN",
            "GPU jobs waiting for runners",
            "The median wait for a *-h100 runner exceeded 30 minutes over the last two hours.",
            "Check scale-set occupancy in the run summary. Free GPUs (cordoned nodes, stuck "
            "runner pods) or raise `maxRunners` in `scripts/k8s-runner-resources/`.",
        ),
        Check(
            "npd_unknown",
            "WARN",
            "NPD condition Unknown",
            "The node-problem-detector plugin timed out or refused to run for most of the last "
            "six hours, so a real fault on this node would go unnoticed.",
            "`kubectl -n monitoring logs <npd pod on the node> | grep -iE 'timeout|not RDMA'`.",
        ),
        Check(
            "gpu_mem_leak",
            "WARN",
            "GPU memory held by no pod",
            "A GPU has had over 2 GiB allocated with no pod attached for 30 minutes: an orphan "
            "process left by a killed runner.",
            "`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` on the node and "
            "kill the orphan.",
        ),
        Check(
            "gpu_hot",
            "WARN",
            "GPU over 85C",
            "Sustained temperature near the H100 throttle point.",
            "Check clocks and power with DCGM on the node. Open an OCI ticket if it persists "
            "while idle.",
        ),
        Check(
            "stale_queued_runs",
            "WARN",
            "Workflow runs queued over 24h",
            "Runs that will never be picked up (dead runner label, offline runner) sit as "
            "queued and later report `cancelled`, which hides the real cause.",
            "`gh run cancel <id>`; fix the `runs-on` label if the runner no longer exists.",
        ),
    )
}


# --- Prometheus --------------------------------------------------------------


class PromError(RuntimeError):
    """Prometheus was unreachable or rejected the query."""


class Prom:
    """Minimal Prometheus HTTP API v1 client (instant queries only)."""

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def query(self, promql: str) -> list[dict]:
        """Instant query; returns the result vector or raises PromError."""
        data = urllib.parse.urlencode({"query": promql}).encode()
        req = urllib.request.Request(f"{self.url}/api/v1/query", data=data)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.load(resp)
        except (OSError, ValueError) as exc:
            raise PromError(f"prometheus query failed: {exc}") from exc
        if payload.get("status") != "success":
            raise PromError(f"prometheus error: {payload.get('error', payload)}")
        return payload["data"]["result"]


def node_of(metric: dict) -> str:
    """Node name from whichever label the exporter used."""
    for key in ("node", "Hostname", "kubernetes_node"):
        if metric.get(key):
            return metric[key]
    instance = metric.get("instance", "")
    return instance.rsplit(":", 1)[0] if ":" in instance else instance


@dataclass(frozen=True)
class PromCheck:
    """A check evaluated from one PromQL instant query."""

    check: Check
    promql: str
    detail: Callable[[dict, float], str]


H100_NODES_QUERY = 'max by (node) (kube_node_status_capacity{resource="nvidia_com_gpu"}) == 8'

NPD_CONDITIONS = (
    "GpuEcc|GpuRowRemap|GpuBus|GpuCount|RdmaLink|RdmaLinkFlapping|RdmaWpaAuth|RdmaRttcc|"
    "KernelDeadlock|ReadonlyFilesystem"
)

PROM_CHECKS: tuple[PromCheck, ...] = (
    PromCheck(
        CHECKS["node_not_ready"],
        'max by (node) (kube_node_status_condition{condition="Ready",status="true"}) == 0',
        lambda m, v: "Ready is not True",
    ),
    PromCheck(
        CHECKS["npd_condition"],
        "max by (node, condition) (kube_node_status_condition"
        f'{{condition=~"{NPD_CONDITIONS}",status="true"}}) == 1',
        lambda m, v: m["condition"],
    ),
    PromCheck(
        CHECKS["gpu_xid"],
        "max by (Hostname, gpu) (changes(DCGM_FI_DEV_XID_ERRORS[1h])) > 0",
        lambda m, v: f"gpu{m['gpu']}: {int(v)} XID change(s) in the last hour",
    ),
    PromCheck(
        CHECKS["gpu_row_remap_failure"],
        "max by (Hostname, gpu) (DCGM_FI_DEV_ROW_REMAP_FAILURE) > 0",
        lambda m, v: f"gpu{m['gpu']}: row-remap failure",
    ),
    PromCheck(
        CHECKS["disk_full"],
        "max by (instance, mountpoint) (1 - "
        'node_filesystem_avail_bytes{mountpoint=~"/|/raid"} / '
        'node_filesystem_size_bytes{mountpoint=~"/|/raid"}) > 0.85',
        lambda m, v: f"{m['mountpoint']} {v * 100:.0f}% used",
    ),
    PromCheck(
        CHECKS["node_cordoned"],
        "max by (node) (min_over_time(kube_node_spec_unschedulable[2h])) == 1",
        lambda m, v: "unschedulable for over 2h",
    ),
    PromCheck(
        CHECKS["npd_unknown"],
        "max by (node, condition) (avg_over_time(kube_node_status_condition"
        f'{{condition=~"{NPD_CONDITIONS}",status="unknown"}}[6h])) > 0.5',
        lambda m, v: f"{m['condition']} Unknown {v * 100:.0f}% of the last 6h",
    ),
    # A subquery, not min_over_time: when a GPU gets attributed to a pod, its
    # pod="" series goes stale but its old samples stay inside a [30m] range
    # vector, so min_over_time flagged GPUs that a new job had just picked up.
    # Counting one-minute steps only sees the GPU while it is unattributed.
    PromCheck(
        CHECKS["gpu_mem_leak"],
        'count_over_time((max by (Hostname, gpu) (DCGM_FI_DEV_FB_USED{pod=""}) > 2048)'
        "[30m:1m]) >= 28",
        lambda m, v: (
            f"gpu{m['gpu']}: over 2 GiB used with no pod at {int(v)} of the last 30 "
            "one-minute checks"
        ),
    ),
    PromCheck(
        CHECKS["gpu_hot"],
        "max by (Hostname, gpu) (min_over_time(DCGM_FI_DEV_GPU_TEMP[15m])) > 85",
        lambda m, v: f"gpu{m['gpu']}: {v:.0f}C for 15m",
    ),
)


def h100_nodes(prom: Prom) -> set[str]:
    """Nodes that expose 8 GPUs, which is the H100 shape in this cluster."""
    return {node_of(s["metric"]) for s in prom.query(H100_NODES_QUERY)}


def prom_findings(prom: Prom, nodes: set[str]) -> list[Finding]:
    """Run every PromQL check and keep the samples that belong to H100 nodes."""
    findings: list[Finding] = []
    for pc in PROM_CHECKS:
        for sample in prom.query(pc.promql):
            node = node_of(sample["metric"])
            if node not in nodes:
                continue
            value = float(sample["value"][1])
            findings.append(Finding(pc.check, node, pc.detail(sample["metric"], value)))
    return findings


# --- GitHub ------------------------------------------------------------------


class GitHubError(RuntimeError):
    """The GitHub API was unreachable or returned an error."""


class GitHub:
    """Minimal GitHub REST client scoped to one repository."""

    def __init__(self, repo: str, token: str | None, timeout: float = 30.0) -> None:
        self.base = f"https://api.github.com/repos/{repo}"
        self.token = token
        self.timeout = timeout

    def _request(self, method: str, path: str, params: dict | None, body: dict | None):
        url = f"{self.base}/{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(body).encode() if body is not None else None
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            raise GitHubError(f"{method} {path} -> {exc.code}: {exc.read()[:200]!r}") from exc
        except (OSError, ValueError) as exc:
            raise GitHubError(f"{method} {path} failed: {exc}") from exc

    def get(self, path: str, params: dict | None = None):
        """GET a repository-relative path."""
        return self._request("GET", path, params, None)

    def post(self, path: str, body: dict):
        """POST a JSON body to a repository-relative path."""
        return self._request("POST", path, None, body)

    def patch(self, path: str, body: dict):
        """PATCH a JSON body to a repository-relative path."""
        return self._request("PATCH", path, None, body)

    def paginate(
        self, path: str, key: str | None, params: dict | None = None, max_pages: int = 3
    ) -> list:
        """Collect up to max_pages pages of 100 items; key selects the array field."""
        items: list = []
        base = dict(params or {}, per_page=100)
        for page in range(1, max_pages + 1):
            payload = self.get(path, dict(base, page=page))
            batch = payload[key] if key else payload
            items.extend(batch)
            if len(batch) < 100:
                break
        return items


def parse_ts(value: str) -> datetime:
    """Parse a GitHub API timestamp (RFC 3339 with a trailing Z)."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass
class QueueStats:
    """Runner-queue numbers shown in the job summary."""

    waits_min: list[float] = field(default_factory=list)
    queued_h100: int = 0
    sampled_runs: int = 0
    active_runs: int = 0


def _h100_labels(job: dict) -> list[str]:
    return [lab for lab in job.get("labels") or [] if lab in H100_LABELS]


def _jobs(gh: GitHub, run: dict) -> list:
    return gh.paginate(f"actions/runs/{run['id']}/jobs", "jobs", max_pages=1)


def github_findings(gh: GitHub, now: datetime) -> tuple[list[Finding], QueueStats]:
    """Evaluate the runner-queue checks from the GitHub Actions API.

    Request budget per run: one runs listing plus at most MAX_RECENT_RUNS jobs
    requests for the wait statistics, then two run listings (queued and
    in-progress) with one jobs request each for the starvation check. Every
    listing reads a single page of 100, so the strict worst case is about 245
    requests (1 + 40 + 2 + 200) and a normal hour costs well under 80, out of
    the 1,000 per hour GITHUB_TOKEN gets for the repository.
    """
    findings: list[Finding] = []
    stats = QueueStats()

    # Wait statistics: H100 jobs that started in recently created runs. A
    # job's created_at is set when it becomes eligible (after its needs), so
    # started_at - created_at is the time spent waiting for a runner.
    since = (now - QUEUE_WINDOW).strftime("%Y-%m-%dT%H:%M:%SZ")
    recent = gh.paginate("actions/runs", "workflow_runs", {"created": f">={since}"}, max_pages=1)[
        :MAX_RECENT_RUNS
    ]
    stats.sampled_runs = len(recent)
    for run in recent:
        for job in _jobs(gh, run):
            if job.get("conclusion") == "skipped":
                continue  # never waited for a runner; would dilute the median
            if _h100_labels(job) and job.get("started_at") and job.get("created_at"):
                wait = parse_ts(job["started_at"]) - parse_ts(job["created_at"])
                stats.waits_min.append(wait.total_seconds() / 60)

    # Starvation: H100 jobs still queued in any unfinished run, however old
    # the run is. Runs queued for over a day are reported as stale instead.
    # The two listings are sequential snapshots: a run that starts between
    # them appears in both, and the in-progress record wins.
    queued_runs = gh.paginate("actions/runs", "workflow_runs", {"status": "queued"}, max_pages=1)
    in_progress = gh.paginate(
        "actions/runs", "workflow_runs", {"status": "in_progress"}, max_pages=1
    )
    in_progress_ids = {r["id"] for r in in_progress}
    queued_only = [r for r in queued_runs if r["id"] not in in_progress_ids]
    stale = [r for r in queued_only if now - parse_ts(r["created_at"]) >= STALE_RUN_AFTER]
    stale_ids = {r["id"] for r in stale}
    active_runs = [r for r in queued_only if r["id"] not in stale_ids] + in_progress
    stats.active_runs = len(active_runs)
    for run in active_runs:
        for job in _jobs(gh, run):
            labels = _h100_labels(job)
            if not labels or job.get("status") != "queued":
                continue
            stats.queued_h100 += 1
            age = now - parse_ts(job["created_at"])
            if age >= STARVED_AFTER:
                minutes = age.total_seconds() / 60
                findings.append(
                    Finding(
                        CHECKS["runner_starved"],
                        labels[0],
                        f"{job['name']} queued {minutes:.0f} min ({job['html_url']})",
                    )
                )

    for run in stale:
        age = now - parse_ts(run["created_at"])
        findings.append(
            Finding(
                CHECKS["stale_queued_runs"],
                "github",
                f"{run['name']} run {run['id']} queued {age.days}d ({run['html_url']})",
            )
        )

    if stats.waits_min:
        p50 = statistics.median(stats.waits_min)
        if p50 > QUEUE_WAIT_P50_MIN:
            findings.append(
                Finding(
                    CHECKS["queue_wait"],
                    "github",
                    f"p50 {p50:.0f} min over {len(stats.waits_min)} jobs in the last 2h",
                )
            )
    return findings, stats


# --- issue reconciliation ----------------------------------------------------


@dataclass
class IssueState:
    """What the marker in an open monitor issue remembers between runs."""

    number: int
    key: str
    first_seen: datetime
    last_seen: datetime
    last_comment: datetime | None
    nodes: dict[str, datetime]


@dataclass(frozen=True)
class Op:
    """One planned issue operation."""

    kind: str  # "create" | "update" | "comment" | "close"
    number: int | None
    key: str
    title: str
    body: str


def _marker(body: str | None) -> dict | None:
    """The JSON marker the monitor leaves at the end of an issue body."""
    body = body or ""
    start = body.rfind(MARKER_PREFIX)
    if start < 0:
        return None
    end = body.find(MARKER_SUFFIX, start)
    if end < 0:
        return None
    try:
        marker = json.loads(body[start + len(MARKER_PREFIX) : end])
    except ValueError:
        return None
    return marker if isinstance(marker, dict) else None


def parse_issue(issue: dict) -> IssueState | None:
    """Rebuild the state of an open monitor issue from its marker."""
    marker = _marker(issue.get("body"))
    if marker is None:
        return None
    try:
        return IssueState(
            number=int(issue["number"]),
            key=marker["key"],
            first_seen=datetime.fromisoformat(marker["first_seen"]),
            last_seen=datetime.fromisoformat(marker["last_seen"]),
            last_comment=(
                datetime.fromisoformat(marker["last_comment"])
                if marker.get("last_comment")
                else None
            ),
            nodes={n: datetime.fromisoformat(t) for n, t in marker.get("nodes", {}).items()},
        )
    except (KeyError, TypeError, ValueError):
        return None


def group_by_check(findings: list[Finding]) -> dict[str, list[Finding]]:
    """Findings keyed by check key: one issue per key."""
    grouped: dict[str, list[Finding]] = {}
    for f in findings:
        grouped.setdefault(f.check.key, []).append(f)
    return grouped


def _scopes(findings: list[Finding]) -> list[str]:
    """Sorted, de-duplicated scopes of a group of findings."""
    return sorted({f.scope for f in findings})


def render_title(check: Check, findings: list[Finding]) -> str:
    """Issue title: severity, check title, and the affected scopes."""
    return f"[node-health][{check.severity}] {check.title}: {', '.join(_scopes(findings))}"


def _fmt(dt: datetime) -> str:
    """Human timestamp for issue bodies and summaries."""
    return dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def render_body(
    check: Check,
    findings: list[Finding],
    state: IssueState,
    now: datetime,
    run_url: str,
    docs_url: str = "",
) -> str:
    """Issue body: meaning, per-scope table, first action, links, and the marker."""
    rows = "\n".join(
        f"| {f.scope} | {f.detail} | {_fmt(state.nodes.get(f.scope, now))} |"
        for f in sorted(findings, key=lambda f: (f.scope, f.detail))
    )
    marker = {
        "key": check.key,
        "first_seen": state.first_seen.isoformat(),
        "last_seen": state.last_seen.isoformat(),
        "last_comment": state.last_comment.isoformat() if state.last_comment else None,
        "nodes": {n: t.isoformat() for n, t in state.nodes.items()},
    }
    lifecycle = (
        "Closed by a human once the GPU is reset or the XID is confirmed benign."
        if check.event
        else "Closed automatically once the condition has been clear for two consecutive runs."
    )
    links = f"[monitor run]({run_url})"
    if docs_url:
        links += f" · [how this monitor works]({docs_url})"
    return (
        f"**H100 CI node health** · {check.severity} · {check.title}\n\n"
        f"{check.meaning}\n\n"
        f"| Scope | Detail | First seen |\n|---|---|---|\n{rows}\n\n"
        f"**First action:** {check.first_action}\n\n"
        f"{lifecycle}\n\n"
        f"Last checked {_fmt(now)} · {links}\n\n"
        f"{MARKER_PREFIX}{json.dumps(marker, sort_keys=True)}{MARKER_SUFFIX}\n"
    )


def plan_ops(
    open_states: list[IssueState],
    active: dict[str, list[Finding]],
    now: datetime,
    run_url: str,
    docs_url: str = "",
    evaluated: set[str] | None = None,
    recently_closed: dict[str, dict[str, datetime]] | None = None,
) -> list[Op]:
    """Diff active findings against open issues and return the issue operations.

    ``evaluated`` lists the check keys that actually ran this time; issues for
    other keys are left alone rather than closed (a Prometheus outage must not
    "resolve" a disk-full issue). ``recently_closed`` maps a check key to the
    scopes of its closed issues and when each was closed; an event finding for
    a scope closed less than EVENT_REOPEN_GRACE ago is not recreated, since the
    lookback still sees the same event. A new scope always opens an issue.
    """
    ops: list[Op] = []
    by_key = {s.key: s for s in open_states}
    recently_closed = recently_closed or {}

    for key, findings in active.items():
        check = CHECKS[key]
        if check.event:
            closed = recently_closed.get(key, {})
            findings = [
                f
                for f in findings
                if f.scope not in closed or now - closed[f.scope] >= EVENT_REOPEN_GRACE
            ]
            if not findings:
                continue
        state = by_key.get(key)
        if state is None:
            nodes = {scope: now for scope in _scopes(findings)}
            fresh = IssueState(0, key, now, now, now if check.event else None, nodes)
            ops.append(
                Op(
                    "create",
                    None,
                    key,
                    render_title(check, findings),
                    render_body(check, findings, fresh, now, run_url, docs_url),
                )
            )
            continue
        state.last_seen = now
        state.nodes = {scope: state.nodes.get(scope, now) for scope in _scopes(findings)}
        comment = check.event and (
            state.last_comment is None or now - state.last_comment >= COMMENT_EVERY
        )
        if comment:
            state.last_comment = now
        ops.append(
            Op(
                "update",
                state.number,
                key,
                render_title(check, findings),
                render_body(check, findings, state, now, run_url, docs_url),
            )
        )
        if comment:
            lines = "\n".join(f"- {f.scope}: {f.detail}" for f in findings)
            ops.append(
                Op(
                    "comment",
                    state.number,
                    key,
                    "",
                    f"Still firing at {_fmt(now)} ([run]({run_url})):\n{lines}",
                )
            )

    for state in open_states:
        check = CHECKS.get(state.key)
        if state.key in active or check is None or check.event:
            continue
        if evaluated is not None and state.key not in evaluated:
            continue
        if now - state.last_seen >= CLOSE_AFTER:
            body = (
                f"Resolved: clear for two consecutive runs as of {_fmt(now)} "
                f"([run]({run_url})).\n\n"
                f"{MARKER_PREFIX}{json.dumps({'key': state.key, 'closed': now.isoformat()})}"
                f"{MARKER_SUFFIX}\n"
            )
            ops.append(Op("close", state.number, state.key, "", body))
    return ops


def apply_ops(gh: GitHub, ops: list[Op], dry_run: bool) -> None:
    """Perform (or, in dry-run mode, print) the planned issue operations."""
    for op in ops:
        label = f"{op.kind} #{op.number}" if op.number else op.kind
        mode = "dry-run" if dry_run else "apply"
        print(f"[{mode}] {label} {op.key}: {op.title or op.body[:80]!r}")
        if dry_run:
            continue
        if op.kind == "create":
            gh.post("issues", {"title": op.title, "body": op.body, "labels": [ISSUE_LABEL]})
        elif op.kind == "update":
            gh.patch(f"issues/{op.number}", {"title": op.title, "body": op.body})
        elif op.kind == "comment":
            gh.post(f"issues/{op.number}/comments", {"body": op.body})
        elif op.kind == "close":
            gh.patch(
                f"issues/{op.number}",
                {"body": op.body, "state": "closed", "state_reason": "completed"},
            )


def ensure_label(gh: GitHub, dry_run: bool) -> None:
    """Create the issue label once; 422 means it already exists."""
    if dry_run:
        return
    try:
        gh.post(
            "labels",
            {
                "name": ISSUE_LABEL,
                "color": "d93f0b",
                "description": "Automated H100 CI node health findings",
            },
        )
    except GitHubError as exc:
        if "422" not in str(exc):
            raise


def issue_states(gh: GitHub) -> tuple[list[IssueState], dict[str, dict[str, datetime]]]:
    """Open monitor issues as state, plus the latest close time per check key and scope."""
    open_states = []
    for issue in gh.paginate("issues", None, {"labels": ISSUE_LABEL, "state": "open"}):
        if "pull_request" in issue:
            continue
        state = parse_issue(issue)
        if state is not None:
            open_states.append(state)
    recently_closed: dict[str, dict[str, datetime]] = {}
    closed = gh.paginate(
        "issues",
        None,
        {"labels": ISSUE_LABEL, "state": "closed", "sort": "updated", "direction": "desc"},
        max_pages=1,
    )
    for issue in closed:
        marker = _marker(issue.get("body"))
        key = marker.get("key") if marker else None
        nodes = marker.get("nodes", {}) if marker else {}
        if not isinstance(key, str) or not isinstance(nodes, dict) or not issue.get("closed_at"):
            continue
        closed_at = parse_ts(issue["closed_at"])
        per_scope = recently_closed.setdefault(key, {})
        for scope in nodes:
            if scope not in per_scope or closed_at > per_scope[scope]:
                per_scope[scope] = closed_at
    return open_states, recently_closed


# --- fleet summary -----------------------------------------------------------

FLEET_QUERIES: dict[str, str] = {
    "ready": 'max by (node) (kube_node_status_condition{condition="Ready",status="true"})',
    "cordoned": "max by (node) (kube_node_spec_unschedulable)",
    "gpus": 'max by (node) (kube_node_status_allocatable{resource="nvidia_com_gpu"})',
    "npd_true": (
        "sum by (node) (max by (node, condition) "
        f'(kube_node_status_condition{{condition=~"{NPD_CONDITIONS}",status="true"}}))'
    ),
    "npd_unknown": (
        "sum by (node) (max by (node, condition) "
        f'(kube_node_status_condition{{condition=~"{NPD_CONDITIONS}",status="unknown"}}))'
    ),
    "root_pct": (
        "max by (instance) (1 - "
        'node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"})'
    ),
    "raid_pct": (
        "max by (instance) (1 - "
        'node_filesystem_avail_bytes{mountpoint="/raid"} / '
        'node_filesystem_size_bytes{mountpoint="/raid"})'
    ),
    "runner_pods": (
        "count by (node) (max by (pod, node) "
        '(kube_pod_info{namespace="actions-runner-system", pod=~".*-runner-.*"}))'
    ),
    "xid_24h": "sum by (Hostname) (max by (Hostname, gpu) (changes(DCGM_FI_DEV_XID_ERRORS[24h])))",
    "oom_24h": 'sum by (kubernetes_node) (increase(problem_counter{reason="OOMKilling"}[24h]))',
}


def fleet_rows(prom: Prom, nodes: set[str]) -> list[dict]:
    """One summary row per H100 node; missing metrics stay None."""
    rows = {node: {"node": node, **{k: None for k in FLEET_QUERIES}} for node in sorted(nodes)}
    for column, promql in FLEET_QUERIES.items():
        for sample in prom.query(promql):
            node = node_of(sample["metric"])
            if node in rows:
                rows[node][column] = float(sample["value"][1])
    return list(rows.values())


def _cell(value: float | None, fmt: str = "{:.0f}") -> str:
    """Table cell for an optional number."""
    return "-" if value is None else fmt.format(value)


def _pct(value: float | None) -> str:
    """Table cell for an optional ratio, as a percentage."""
    return "-" if value is None else f"{value * 100:.0f}%"


def _p95(values: list[float]) -> float:
    """Nearest-rank 95th percentile."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def render_summary(
    rows: list[dict], stats: QueueStats, findings: list[Finding], now: datetime
) -> str:
    """Markdown for the job summary: fleet table, queue numbers, active findings."""
    lines = [
        f"## H100 CI node health · {_fmt(now)}",
        "",
        "| Node | Ready | Cordoned | GPUs | NPD true | NPD unknown | root | /raid | "
        "runner pods | XID 24h | OOM 24h |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['node']} | {'yes' if r['ready'] else 'NO'} | {'yes' if r['cordoned'] else '-'} | "
            f"{_cell(r['gpus'])} | {_cell(r['npd_true'])} | {_cell(r['npd_unknown'])} | "
            f"{_pct(r['root_pct'])} | {_pct(r['raid_pct'])} | {_cell(r['runner_pods'])} | "
            f"{_cell(r['xid_24h'])} | {_cell(r['oom_24h'])} |"
        )
    lines.append("")
    if stats.waits_min:
        lines.append(
            f"Runner queue wait, last 2h ({len(stats.waits_min)} H100 jobs): "
            f"p50 {statistics.median(stats.waits_min):.0f} min, "
            f"p95 {_p95(stats.waits_min):.0f} min. Queued now: {stats.queued_h100}."
        )
    else:
        lines.append(f"No H100 jobs started in the last 2h. Queued now: {stats.queued_h100}.")
    lines.append(
        f"Wait statistics from {stats.sampled_runs} recent runs (cap {MAX_RECENT_RUNS}); "
        f"{stats.active_runs} queued or in-progress runs scanned for starvation."
    )
    lines.append("")
    if findings:
        lines.append("### Active findings")
        for f in sorted(findings, key=lambda f: (f.check.severity, f.check.key, f.scope)):
            lines.append(f"- **{f.check.severity}** `{f.check.key}` {f.scope}: {f.detail}")
    else:
        lines.append("No active findings.")
    lines.append("")
    return "\n".join(lines)


# --- main --------------------------------------------------------------------


def _run_url() -> str:
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    return f"{server}/{repo}/actions/runs/{run_id}" if repo and run_id else server


def _docs_url(repo: str) -> str:
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    return f"{server}/{repo}/blob/main/{DOCS_PATH}"


def _write_summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text)
    else:
        print(text)


def main(argv: list[str] | None = None) -> int:
    """Run every check once and reconcile the issues. Exit 1 if Prometheus was down, 2 on GitHub errors."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="print issue ops, write nothing")
    parser.add_argument("--prom-url", default=os.environ.get("PROM_URL", DEFAULT_PROM_URL))
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    args = parser.parse_args(argv)
    if not args.repo:
        parser.error("--repo or GITHUB_REPOSITORY is required")

    now = datetime.now(UTC)
    run_url = _run_url()
    prom = Prom(args.prom_url)
    gh = GitHub(args.repo, os.environ.get("GITHUB_TOKEN"))

    findings: list[Finding] = []
    rows: list[dict] = []
    evaluated = set(CHECKS)
    prom_ok = True
    try:
        nodes = h100_nodes(prom)
        findings.extend(prom_findings(prom, nodes))
        rows = fleet_rows(prom, nodes)
    except PromError as exc:
        prom_ok = False
        evaluated -= {pc.check.key for pc in PROM_CHECKS}
        print(f"::error::{exc}")
        findings.append(Finding(CHECKS["monitor_blind"], "prometheus", str(exc)[:200]))

    try:
        gh_findings, stats = github_findings(gh, now)
        findings.extend(gh_findings)
        ensure_label(gh, args.dry_run)
        open_states, recently_closed = issue_states(gh)
        ops = plan_ops(
            open_states,
            group_by_check(findings),
            now,
            run_url,
            docs_url=_docs_url(args.repo),
            evaluated=evaluated,
            recently_closed=recently_closed,
        )
        apply_ops(gh, ops, args.dry_run)
    except GitHubError as exc:
        print(f"::error::{exc}")
        return 2

    _write_summary(render_summary(rows, stats, findings, now))
    return 0 if prom_ok else 1


if __name__ == "__main__":
    sys.exit(main())
