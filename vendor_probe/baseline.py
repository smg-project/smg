#!/usr/bin/env python3
"""Distill a raw vendor-probe recording dir into a checked-in behavior baseline.

The raw recordings (``results.jsonl`` + friends, ~20MB/provider) stay out of
git; this tool collapses them into the per-cluster ground truth the compat gate
needs (``compat_diff --baseline``), a few hundred KB per provider:

For each vendor fingerprint cluster (same key ``compat_diff`` aggregates by:
status + sorted field-path inventory for JSON bodies, status + run-collapsed
SSE event-type sequence for streams) the baseline stores:

- ``cluster_id`` (``B000``... by descending size, then signature — stable for a
  given recording) and ``sig_hash`` (content hash of the signature — stable
  across recordings, the identity the regression allowlist keys on),
- the fingerprint signature itself (``status`` + ``field_paths`` | ``event_types``),
- ``probe_count`` + ``members`` (probe id, category, expect),
- one normalized exemplar: volatile values scrubbed exactly as
  ``compat_diff`` normalizes (ids / timestamps / model names / token counts →
  placeholders; volatile integers → 0). Bodies are stored verbatim for error
  clusters (4xx/5xx — the envelope values feed the S4 comparison) and for
  small bodies; large 2xx bodies keep only the field-path skeleton (the
  signature already carries it, and the differ never compares 2xx values).

Only recorded, non-transport-failure probes are distilled — the same filter
``compat_diff`` applies to the vendor side.

Usage:
  python -m vendor_probe.baseline \
      --results results/openai --provider openai \
      --out vendor_probe/baselines/2026-08-21/openai \
      --probe-set-sha 0ec065a1 --run-id 32506359628 --date 2026-08-21

Output: ``baseline.jsonl`` (one cluster per line, deterministic: sorted
members, sorted clusters, sorted JSON keys) + ``manifest.json`` (provenance:
probe-set git sha, recording run id, models observed, counts, cost).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vendor_probe import compat_diff
else:
    from . import compat_diff

# Integer values under these key shapes are volatile (timestamps, token
# counts, latency). The differ never compares 2xx body values, so zeroing them
# only serves exemplar determinism across recording runs.
_VOLATILE_INT_KEY = re.compile(
    r"(^created$|_at$|_tokens?$|_token_count$|_ms$|^sequence_number$|^event_count$)"
)

_DEFAULT_MAX_EXEMPLAR_BYTES = 8192


def normalize_value(value, key: str | None = None):
    """Scrub volatile content from an exemplar body, preserving JSON shape."""
    if isinstance(value, str):
        return compat_diff.scrub(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and key is not None and _VOLATILE_INT_KEY.search(key):
        return 0
    if isinstance(value, dict):
        return {k: normalize_value(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_value(v, key) for v in value]
    return value


def _sig_payload(key):
    """Serializable form of a compat_diff vendor_cluster_key."""
    kind, status, third = key
    return {"kind": kind, "status": status, "sig": list(third)}


def sig_hash(key) -> str:
    payload = json.dumps(_sig_payload(key), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_clusters(records: dict, provider: str, max_exemplar_bytes: int) -> list:
    grouped: dict = {}
    for pid in sorted(records):
        rec = records[pid]
        if rec.get("skipped") or not rec.get("recorded") or rec.get("transport_failure"):
            continue
        key = compat_diff.vendor_cluster_key(rec, provider)
        grouped.setdefault(key, []).append(rec)

    def sort_key(item):
        key, members = item
        return (-len(members), json.dumps(_sig_payload(key), sort_keys=True))

    clusters = []
    for i, (key, members) in enumerate(sorted(grouped.items(), key=sort_key)):
        kind, status, third = key
        cluster = {
            "cluster_id": f"B{i:03d}",
            "sig_hash": sig_hash(key),
            "kind": kind,
            "status": status,
            "probe_count": len(members),
            "members": [
                {
                    "id": m["probe_id"],
                    "category": m.get("category"),
                    "expect": m.get("expect"),
                }
                for m in members  # already sorted by probe id
            ],
        }
        if kind == "sse":
            cluster["event_types"] = list(third)
        else:
            cluster["field_paths"] = list(third)
        _attach_exemplar(cluster, members, max_exemplar_bytes)
        clusters.append(cluster)
    return clusters


def _attach_exemplar(cluster, members, max_exemplar_bytes):
    """Pick the lexicographically-first member as exemplar and normalize it."""
    exemplar = members[0]
    cluster["exemplar_probe"] = exemplar["probe_id"]
    if cluster["kind"] == "sse":
        # The signature IS the exemplar for streams: the differ only compares
        # event-type sequences, and members share the collapsed sequence.
        cluster["exemplar_kind"] = "sse"
        return
    body = compat_diff.get_body(exemplar)
    normalized = normalize_value(body)
    size = len(json.dumps(normalized, ensure_ascii=False, sort_keys=True))
    status = cluster["status"] or 0
    # Error bodies feed the S4 envelope comparison (error.type/code/param and
    # message wording) — always keep them. 2xx bodies are only ever compared
    # by field path, so large ones keep just the skeleton.
    if status >= 400 or size <= max_exemplar_bytes:
        cluster["exemplar_kind"] = "body"
        cluster["exemplar_body"] = normalized
    else:
        cluster["exemplar_kind"] = "skeleton"


def collect_models(records: dict) -> dict:
    requested, observed = set(), set()
    for rec in records.values():
        req_body = (rec.get("request") or {}).get("body")
        if isinstance(req_body, dict) and isinstance(req_body.get("model"), str):
            requested.add(req_body["model"])
        body = compat_diff.get_body(rec)
        if isinstance(body, dict) and isinstance(body.get("model"), str):
            observed.add(body["model"])
    return {"requested": sorted(requested), "observed": sorted(observed)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="distill probe recordings into a compat baseline")
    ap.add_argument("--results", required=True, help="raw recording dir (results.jsonl[.gz])")
    ap.add_argument("--provider", required=True, choices=["openai", "anthropic"])
    ap.add_argument("--out", required=True, help="baseline output dir")
    ap.add_argument("--probe-set-sha", default=None, help="git sha of the probe set recorded")
    ap.add_argument("--run-id", default=None, help="recording workflow run id")
    ap.add_argument("--date", default=None, help="baseline date (YYYY-MM-DD, default today)")
    ap.add_argument(
        "--max-exemplar-bytes",
        type=int,
        default=_DEFAULT_MAX_EXEMPLAR_BYTES,
        help="2xx exemplar bodies larger than this store only the field-path skeleton",
    )
    args = ap.parse_args(argv)

    results_dir = Path(args.results)
    records = compat_diff.load_results(results_dir)
    clusters = build_clusters(records, args.provider, args.max_exemplar_bytes)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = out_dir / "baseline.jsonl"
    with baseline_path.open("w", encoding="utf-8") as f:
        for c in clusters:
            f.write(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n")

    distilled = sum(c["probe_count"] for c in clusters)
    exemplar_kinds = {}
    for c in clusters:
        exemplar_kinds[c["exemplar_kind"]] = exemplar_kinds.get(c["exemplar_kind"], 0) + 1

    manifest = {
        "provider": args.provider,
        "date": args.date or datetime.date.today().isoformat(),
        "probe_set_git_sha": args.probe_set_sha,
        "recording_run_id": args.run_id,
        "models": collect_models(records),
        "probes_total": len(records),
        "probes_distilled": distilled,
        "clusters": len(clusters),
        "exemplar_kinds": exemplar_kinds,
        "max_exemplar_bytes": args.max_exemplar_bytes,
        "baseline_bytes": baseline_path.stat().st_size,
    }
    summary_path = results_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        manifest["recording"] = {
            k: summary.get(k)
            for k in (
                "recorded",
                "skipped",
                "transport_failures",
                "expected_error_recorded",
                "fingerprint_clusters_total",
                "tokens",
                "estimated_cost_usd",
            )
        }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )

    print(
        json.dumps(
            {
                "provider": args.provider,
                "clusters": len(clusters),
                "probes_distilled": distilled,
                "exemplar_kinds": exemplar_kinds,
                "baseline_bytes": manifest["baseline_bytes"],
                "out": str(out_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
