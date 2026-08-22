"""CPU-side logic for the vendor-truth conformance e2e suite.

The suite replays probes from ``vendor_probe`` against the lane's real
gateway + engine and asserts each vendor behavior cluster (distilled into the
checked-in baselines under ``vendor_probe/baselines/``) still holds
structurally: status class, field-path skeleton, SSE event-type sequence,
error-envelope shape. All normalization and severity logic is
``vendor_probe.compat_diff``'s — this module only handles representative
selection, replay orchestration, and mapping cluster verdicts onto pytest
outcomes.

Everything here is CPU-pure except :func:`run_replay` (network against a
running gateway); the pure parts are unit-tested in
``test_conformance_unit.py`` without a GPU.

Selection: up to ``REPRESENTATIVES_PER_CLUSTER`` member probes per cluster
(first / middle / last of the sorted member ids — deterministic, spreads
across the id-ordered categories), plus the transitive dependency closure so
placeholder references (``{{probe#path}}``) resolve. ``VENDOR_COMPAT_FULL=1``
replays every member instead (nightly / dispatch use).

Verdict → outcome:
- ``exact`` / ``benign`` / ``mock-limited`` (content-bound)  -> pass
- divergent + allowlisted in ``known_divergences.jsonl``     -> xfail
- divergent + NOT allowlisted (or worse than allowlisted)    -> FAIL, with a
  ready-to-paste ``TO-TRIAGE`` allowlist line in the failure message
- no comparable replay (gated/skipped/stale probes)          -> skip
"""

from __future__ import annotations

import asyncio
import datetime
import functools
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    # Mirror e2e_test/conftest.py's sys.path setup for the repo-root
    # vendor_probe package (works under --noconftest unit runs too).
    sys.path.insert(0, str(REPO_ROOT))

from vendor_probe import compat_diff, genmatrix, runner  # noqa: E402
from vendor_probe.probes import anthropic_messages, openai_responses  # noqa: E402

BASELINES_ROOT = REPO_ROOT / "vendor_probe" / "baselines"
ALLOWLIST_NAME = "known_divergences.jsonl"
FULL_ENV = "VENDOR_COMPAT_FULL"
REPRESENTATIVES_PER_CLUSTER = 3

_DIVERGENT = ("S1", "S2", "S3", "S4")
_NOT_COMPARABLE = ("replay-gap", "replay-transport", "config-limited")


# ---------------------------------------------------------------------------
# Baseline + allowlist loading (checked-in data, cached per session)
# ---------------------------------------------------------------------------


def baseline_date() -> str:
    """Newest dated baseline directory (lexicographic = chronological)."""
    dates = sorted(p.name for p in BASELINES_ROOT.iterdir() if p.is_dir())
    if not dates:
        raise FileNotFoundError(f"no baseline directories under {BASELINES_ROOT}")
    return dates[-1]


def baseline_dir(provider: str) -> Path:
    return BASELINES_ROOT / baseline_date() / provider


@functools.cache
def clusters(provider: str) -> tuple:
    return tuple(compat_diff.load_baseline(baseline_dir(provider)))


@functools.cache
def cluster_by_id(provider: str) -> dict:
    return {c["cluster_id"]: c for c in clusters(provider)}


def cluster_ids(provider: str) -> list:
    return [c["cluster_id"] for c in clusters(provider)]


@functools.cache
def allowlist(provider: str) -> dict:
    path = BASELINES_ROOT / baseline_date() / ALLOWLIST_NAME
    return compat_diff.load_allowlist(path, provider)


# ---------------------------------------------------------------------------
# Probe selection
# ---------------------------------------------------------------------------


def is_full_run() -> bool:
    return os.environ.get(FULL_ENV, "").lower() in ("1", "true", "yes")


@functools.cache
def full_matrix(provider: str) -> tuple:
    curated = openai_responses.PROBES if provider == "openai" else anthropic_messages.PROBES
    return tuple(list(curated) + genmatrix.generate(provider, budget=0))


def representatives(member_ids, limit: int = REPRESENTATIVES_PER_CLUSTER) -> list:
    """Deterministic spread pick: first / middle / last of the sorted ids."""
    ids = sorted(member_ids)
    if len(ids) <= limit:
        return ids
    picks = [ids[0], ids[len(ids) // 2], ids[-1]]
    out = []
    for pid in picks:
        if pid not in out:
            out.append(pid)
    return out[:limit]


def selection(provider: str) -> dict:
    """{cluster_id: [probe ids to replay+judge]} for the current mode."""
    full = is_full_run()
    sel = {}
    for c in clusters(provider):
        ids = [m["id"] for m in c["members"]]
        sel[c["cluster_id"]] = sorted(ids) if full else representatives(ids)
    return sel


def replay_probes(provider: str):
    """Return (ordered probe subset incl. dependency closure, selection,
    member ids missing from the current probe matrix)."""
    matrix = list(full_matrix(provider))
    ordered, by_id = runner.order_and_wire(matrix)
    sel = selection(provider)
    wanted = set()
    for ids in sel.values():
        wanted.update(ids)
    missing = sorted(pid for pid in wanted if pid not in by_id)
    need = {pid for pid in wanted if pid in by_id}
    stack = list(need)
    while stack:
        pid = stack.pop()
        for dep in by_id[pid]["_waits"]:
            if dep not in need:
                need.add(dep)
                stack.append(dep)
    probes = [p for p in ordered if p["id"] in need]
    return probes, sel, missing


# ---------------------------------------------------------------------------
# Replay execution (the only networked part)
# ---------------------------------------------------------------------------


def run_replay(
    provider: str,
    base_url: str,
    model_name: str,
    *,
    concurrency: int = 16,
    timeout: float = 120.0,
    stream_timeout: float = 240.0,
):
    """Replay the selected probes against a running SMG gateway.

    ``@MODEL`` / ``@MODEL_CLASSIC`` resolve to the lane's served model;
    gated vendor models (``@MODEL_CUA`` etc.) keep their vendor names so the
    gateway's unknown-model envelope is exercised, exactly like the recorded
    replays. Returns ``(records_by_probe_id, selection, missing_member_ids)``.
    """
    probes, sel, missing = replay_probes(provider)
    overrides = {
        "SMG_BASE_URL": base_url,
        "OPENAI_MODEL": model_name,
        "OPENAI_MODEL_CLASSIC": model_name,
        "ANTHROPIC_MODEL": model_name,
    }
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        os.environ.update(overrides)
        adapter = runner.build_adapter(f"smg-{provider}")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    cfg = {
        "concurrency": concurrency,
        "timeout": timeout,
        "stream_timeout": stream_timeout,
        "max_retries": 0,
    }
    records = asyncio.run(runner.run_live(adapter, probes, cfg))
    return {r["probe_id"]: r for r in records}, sel, missing


def replay_health(records: dict) -> dict:
    """Fraction of attempted (non-skipped) probes that got an HTTP status."""
    attempted = [r for r in records.values() if not r.get("skipped")]
    responded = [r for r in attempted if compat_diff.get_status(r) is not None]
    return {
        "total": len(records),
        "attempted": len(attempted),
        "responded": len(responded),
        "responded_fraction": (len(responded) / len(attempted)) if attempted else 0.0,
    }


# ---------------------------------------------------------------------------
# Per-cluster judgment
# ---------------------------------------------------------------------------


def judge_cluster(provider: str, cluster: dict, member_ids, records: dict) -> dict:
    """Classify the replayed members of one baseline cluster.

    Returns ``{disposition, verdict, s1_kinds, entries, allowlisted}`` where
    disposition is one of ``conformant`` / ``expected-divergence`` /
    ``new-divergence`` / ``severity-escalation`` / ``no-coverage``.
    """
    synth = compat_diff.synth_vendor_records([cluster])
    entries = [
        compat_diff.classify(pid, synth[pid], records.get(pid), provider, frozenset())
        for pid in member_ids
        if pid in synth
    ]
    comparable = [e for e in entries if e["class"] not in _NOT_COMPARABLE]
    sevs = [e["severity"] for e in comparable if e["severity"] in _DIVERGENT]
    if not comparable:
        verdict = "no-coverage"
    elif sevs:
        verdict = compat_diff.worst_severity(sevs)
    elif any(e["class"] == "benign" for e in comparable):
        verdict = "benign"
    elif any(e["class"] == "mock-limited" for e in comparable):
        verdict = "mock-limited"
    else:
        verdict = "exact"

    allowed = allowlist(provider).get(cluster.get("sig_hash"))
    if verdict in _DIVERGENT:
        if allowed is None:
            disposition = "new-divergence"
        elif compat_diff._SEV_RANK[verdict] < compat_diff._SEV_RANK.get(allowed.get("verdict"), 99):
            disposition = "severity-escalation"
        else:
            disposition = "expected-divergence"
    elif verdict == "no-coverage":
        disposition = "no-coverage"
    else:
        disposition = "conformant"

    s1_kinds = sorted({e.get("s1_kind") for e in comparable if e.get("s1_kind")})
    return {
        "disposition": disposition,
        "verdict": verdict,
        "s1_kinds": s1_kinds,
        "entries": entries,
        "allowlisted": allowed,
    }


def triage_allowlist_line(provider: str, cluster: dict, outcome: dict) -> str:
    """Ready-to-paste allowlist entry for a newly-divergent cluster.

    Marked TO-TRIAGE on purpose: adding it silences the failure but the note
    keeps the debt visible until someone adjudicates the divergence.
    """
    divergent = [e for e in outcome["entries"] if e["class"] == "divergence"]
    entry = {
        "provider": provider,
        "sig_hash": cluster.get("sig_hash"),
        "cluster_id": cluster["cluster_id"],
        "verdict": outcome["verdict"],
        "s1_kinds": outcome["s1_kinds"],
        "size": cluster["probe_count"],
        "categories": sorted({m.get("category") for m in cluster["members"] if m.get("category")})[
            :5
        ],
        "example_probes": [e["probe_id"] for e in divergent[:3]],
        "first_seen": datetime.date.today().isoformat(),
        "note": "TO-TRIAGE: surfaced by the real-engine conformance lane; adjudicate "
        "before treating as accepted divergence",
    }
    return json.dumps(entry, ensure_ascii=False, sort_keys=True)


def failure_message(provider: str, cluster: dict, outcome: dict) -> str:
    """Human-oriented failure text: what diverged, on which probes, and the
    TO-TRIAGE line to paste if the divergence is adjudicated as accepted."""
    divergent = [e for e in outcome["entries"] if e["class"] == "divergence"]
    lines = [
        f"{outcome['disposition']} in vendor cluster {cluster['cluster_id']} "
        f"({provider}, sig {cluster.get('sig_hash')}, vendor status {cluster.get('status')}, "
        f"{cluster['probe_count']} probes): verdict {outcome['verdict']}"
        + (f" {outcome['s1_kinds']}" if outcome["s1_kinds"] else "")
    ]
    if outcome["disposition"] == "severity-escalation" and outcome["allowlisted"]:
        lines.append(
            f"allowlisted as {outcome['allowlisted'].get('verdict')} but now "
            f"{outcome['verdict']} — the divergence got worse"
        )
    for e in divergent[:3]:
        notes = "; ".join(compat_diff.scrub(n) for n in e["notes"][:3])
        lines.append(
            f"  probe {e['probe_id']}: vendor {e['status_vendor']} vs "
            f"SMG {e['status_smg']} — {notes}"
        )
        for key in ("missing_paths", "extra_paths"):
            if e.get(key):
                lines.append(f"    {key}: {e[key][:8]}")
        if e.get("events"):
            lines.append(f"    vendor core events: {e['events'].get('vendor_core')}")
            lines.append(f"    smg core events:    {e['events'].get('smg_core')}")
    lines.append(
        "If adjudicated as an accepted divergence, append to "
        f"vendor_probe/baselines/{baseline_date()}/{ALLOWLIST_NAME}:"
    )
    lines.append("  " + triage_allowlist_line(provider, cluster, outcome))
    return "\n".join(lines)
