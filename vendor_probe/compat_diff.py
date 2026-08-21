#!/usr/bin/env python3
"""Cluster-level structural compatibility differ: vendor truth vs SMG replay.

Consumes two probe-recording dirs produced by ``vendor_probe.runner``
(``results.jsonl`` or ``results.jsonl.gz`` + ``summary.json``) — one recorded
against the real vendor API, one replayed against a local SMG gateway — and
emits a ranked markdown report plus a machine-readable ``diff.jsonl``.

Per probe it compares:
  (a) HTTP status: exact value and status class (2xx/4xx/...),
  (b) top-level + nested field-path inventory of JSON bodies (vendor fields
      SMG omits vs SMG fields the vendor lacks, direction marked),
  (c) SSE event-type sequences (run-length collapsed so token-count-dependent
      delta repetition does not read as drift),
  (d) 4xx error envelope shape: error.type / error.code / error.param and
      whether the mutated field path is named in the message.

Because the replay backend is a mock worker (canned tokens), CONTENT-dependent
differences are classified ``mock-limited``, never as divergences: which output
item types the model produced (reasoning / function_call / tool_use blocks),
whether generation hit max_output_tokens (``incomplete_details``), annotation /
logprob payloads, and the number of streamed delta events. STRUCTURE — status
codes, envelope skeletons, param echoes, lifecycle event framing, validation
envelopes — is compared for real.

Probes aggregate into the vendor-side fingerprint clusters (status + field-path
inventory, or status + collapsed event sequence), so ~12K probes report as a
few hundred behaviors.

Usage:
  python -m vendor_probe.compat_diff \
      --vendor results/openai --smg results/smg-openai \
      --provider openai --out compat/openai

Baseline mode diffs a replay against a checked-in distillation (built by
``vendor_probe.baseline``) instead of raw vendor recordings, and gates on the
known-divergences allowlist: any divergent cluster whose signature is not
allowlisted (or that got *worse* than its allowlisted severity) exits 1.

  python -m vendor_probe.compat_diff \
      --baseline vendor_probe/baselines/2026-08-21/openai \
      --smg results/smg-openai --provider openai --out compat/openai \
      --allowlist vendor_probe/baselines/2026-08-21/known_divergences.jsonl

``--write-allowlist FILE`` regenerates the allowlist entries for this provider
from the current run (grandfathering today's divergences); it replaces only
this provider's lines and suppresses the failing exit.

Severity ranking (worst first):
  S1  status mismatch (wrong status class; accepting what the vendor rejects
      or rejecting what the vendor accepts)
  S2  missing/extra response fields (structural inventory drift)
  S3  SSE event-sequence drift (lifecycle framing, not delta counts)
  S4  error-envelope shape/wording drift (type/code/param naming, whether the
      offending field is named)
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# =============================================================================
# volatile-value + content-dependence tables
# =============================================================================

# Body compare is field-path based (values never compared), which inherently
# normalizes ids / timestamps / model names / token counts. Value-level
# comparison happens only inside 4xx envelopes (error.type/code/param), where
# these regexes scrub volatile substrings before equality checks.
_VOLATILE_RES = [
    (re.compile(r"\b(resp|msg|conv|rs|fc|ws|ig|cu|lc|mcp|msg_batch)_[A-Za-z0-9_-]{6,}"), "<id>"),
    (re.compile(r"\breq_[A-Za-z0-9]{6,}"), "<req>"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "<uuid>"),
    (re.compile(r"\b\d{9,}\b"), "<ts>"),
    (re.compile(r"\b(gpt|claude|codex|computer-use|o[134])[A-Za-z0-9.\-]*"), "<model>"),
    (re.compile(r"\b\d+ tokens?\b"), "<n> tokens"),
]

# Vendor-only field paths under these prefixes reflect what the model chose to
# generate (canned mock tokens cannot reproduce them) — classified
# mock-limited, not missing. SMG-only paths are still reported.
# NOTE: skeleton fields the vendor emits on every response (usage.*_details,
# stop_sequence: null, cache counters) are deliberately NOT here — their
# presence is structural even when their value is content-bound.
_CONTENT_DEPENDENT_PATH_PREFIXES = {
    "openai": (
        "output[]",  # item mix: reasoning / function_call / message / web_search…
        "incomplete_details",  # only present when generation hit a cap
    ),
    "anthropic": (
        "content[]",  # block mix: text / tool_use / thinking
        "usage.server_tool_use",  # present only when server tools ran
    ),
}

# SSE event types that only appear for specific generated content (tool calls,
# reasoning, annotations, refusals). Present-in-vendor-only => mock-limited;
# core lifecycle events missing => S3 divergence.
_CONTENT_DEPENDENT_EVENTS = {
    "openai": (
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
        "response.output_text.annotation.added",
        "response.refusal.delta",
        "response.refusal.done",
        "response.web_search_call.in_progress",
        "response.web_search_call.searching",
        "response.web_search_call.completed",
        "response.file_search_call.in_progress",
        "response.file_search_call.searching",
        "response.file_search_call.completed",
        "response.image_generation_call.in_progress",
        "response.image_generation_call.generating",
        "response.image_generation_call.partial_image",
        "response.image_generation_call.completed",
        "response.mcp_call_arguments.delta",
        "response.mcp_call_arguments.done",
        "response.mcp_call.in_progress",
        "response.mcp_call.completed",
        "response.mcp_call.failed",
        "response.mcp_list_tools.in_progress",
        "response.mcp_list_tools.completed",
        "response.mcp_list_tools.failed",
        # item-level framing repeats per generated item; the mock produces a
        # different item count, so added/done arity is content-bound too.
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        # whether any text was emitted at all is content-bound: a reasoning
        # model can burn the whole budget before producing output_text.
        "response.output_text.delta",
        "response.output_text.done",
    ),
    "anthropic": (
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "signature_delta",
        "thinking_delta",
        "input_json_delta",
    ),
}

# Event names whose *repetition count* is token-count-bound. Runs are collapsed
# for everyone; these are also deduped against arity mismatches entirely.
_DELTAISH = re.compile(r"(\.delta$|^ping$|^content_block_delta$)")

_SEVERITY_ORDER = ["S1", "S2", "S3", "S4", "benign", "mock-limited", "exact"]
_SEV_RANK = {s: i for i, s in enumerate(_SEVERITY_ORDER)}


# =============================================================================
# loading
# =============================================================================
def load_results(dir_path: Path) -> dict:
    """Load results.jsonl(.gz) as {probe_id: record}."""
    plain = dir_path / "results.jsonl"
    gz = dir_path / "results.jsonl.gz"
    if plain.exists():
        fh = plain.open("r", encoding="utf-8")
    elif gz.exists():
        fh = gzip.open(gz, "rt", encoding="utf-8")
    else:
        raise FileNotFoundError(f"no results.jsonl(.gz) in {dir_path}")
    out = {}
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["probe_id"]] = rec
    return out


# =============================================================================
# fingerprint primitives (shared shape with runner.py, self-contained here)
# =============================================================================
def field_paths(obj, prefix=""):
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            out.add(p)
            out |= field_paths(v, p)
    elif isinstance(obj, list):
        for v in obj:
            out |= field_paths(v, prefix + "[]")
    return out


def event_names(record, provider) -> list:
    resp = record.get("response") or {}
    names = []
    for ev in resp.get("sse_events") or []:
        if provider == "anthropic":
            parsed = ev.get("parsed") or {}
            name = parsed.get("type") or ev.get("event")
        else:
            name = ev.get("event")
        if not name:
            # Nameless events are sentinels or malformed frames; the [DONE]
            # sentinel in particular is a real framing difference to surface.
            name = "[DONE]" if (ev.get("data") or "").strip() == "[DONE]" else "?"
        names.append(name)
    return names


# Terminal Responses events whose choice is content-bound against a canned
# mock: the vendor hits max_output_tokens (incomplete) where the mock stops
# naturally (completed). Both collapse to <terminal> in the core comparison;
# response.failed / error do NOT — failing differently is structural.
_TERMINAL_EQUIV = {"response.completed", "response.incomplete"}


def collapse_runs(names: list) -> tuple:
    """a b b b c -> (a, b+, c): delta repetition must not read as drift."""
    out: list = []
    for n in names:
        if out and (out[-1] == n or out[-1] == n + "+"):
            out[-1] = n + "+"
        else:
            out.append(n)
    return tuple(out)


def scrub(text: str) -> str:
    for rx, repl in _VOLATILE_RES:
        text = rx.sub(repl, text)
    return text


def is_streaming(record) -> bool:
    return "sse_events" in ((record or {}).get("response") or {})


def get_body(record):
    return ((record or {}).get("response") or {}).get("body")


def get_status(record):
    return ((record or {}).get("response") or {}).get("status")


def vendor_cluster_key(record, provider):
    """Vendor-side behavior cluster: status + structural fingerprint."""
    status = get_status(record)
    if is_streaming(record):
        return ("sse", status, collapse_runs(event_names(record, provider)))
    body = get_body(record)
    paths = tuple(sorted(field_paths(body))) if isinstance(body, (dict, list)) else ()
    return ("json", status, paths)


# =============================================================================
# baseline mode: load a distilled baseline and synthesize vendor-side records
# =============================================================================
def skeleton_from_paths(paths):
    """Rebuild a minimal JSON value whose field-path inventory equals ``paths``.

    Inverse of :func:`field_paths` up to values: every dict key on a path
    exists, arrays are materialized as single-element lists, leaves become 0.
    """
    if not paths:
        return None

    def node():
        return {"depth": 0, "children": {}}

    root = node()
    for path in paths:
        cur = root
        for comp in path.split("."):
            name, depth = comp, 0
            while name.endswith("[]"):
                name = name[:-2]
                depth += 1
            nxt = cur["children"].setdefault(name, node())
            nxt["depth"] = max(nxt["depth"], depth)
            cur = nxt

    def wrap(n):
        val = build(n)
        for _ in range(n["depth"]):
            val = [val]
        return val

    def build(n):
        if not n["children"]:
            return 0
        if "" in n["children"]:  # leading "[]": the value at this level is a list
            return wrap(n["children"][""])
        return {name: wrap(child) for name, child in n["children"].items()}

    return build(root)


def load_baseline(dir_path: Path) -> list:
    path = dir_path / "baseline.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no baseline.jsonl in {dir_path}")
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def synth_vendor_records(baseline_clusters: list) -> dict:
    """Reconstruct per-probe vendor records that classify() and the cluster
    key reproduce exactly from the stored signatures/exemplars."""
    records = {}
    for c in baseline_clusters:
        if c["kind"] == "sse":
            events: list = []
            for name in c["event_types"]:
                if name == "[DONE]":
                    events.append({"event": None, "data": "[DONE]", "parsed": None})
                else:
                    # event + parsed.type both set so OpenAI- and
                    # Anthropic-style event_names() read the same name back.
                    events.append({"event": name, "data": "", "parsed": {"type": name}})
            response = {"status": c["status"], "sse_events": events}
        else:
            if c.get("exemplar_kind") == "body":
                body = c.get("exemplar_body")
            else:
                body = skeleton_from_paths(c.get("field_paths") or [])
            response = {"status": c["status"], "body": body}
        for m in c["members"]:
            records[m["id"]] = {
                "probe_id": m["id"],
                "category": m.get("category"),
                "expect": m.get("expect"),
                "recorded": True,
                "response": response,
            }
    return records


def baseline_key_index(baseline_clusters: list, records: dict, provider: str) -> dict:
    """Map the recomputed vendor cluster key -> baseline cluster metadata."""
    index = {}
    mismatches = []
    for c in baseline_clusters:
        key = vendor_cluster_key(records[c["members"][0]["id"]], provider)
        stored = tuple(c.get("event_types") or c.get("field_paths") or [])
        if key[1] != c["status"] or key[2] != stored:
            mismatches.append(c["cluster_id"])
        index[key] = c
    if mismatches:
        print(
            f"::warning::baseline signature reconstruction drifted for "
            f"{len(mismatches)} clusters: {mismatches[:5]}",
            file=sys.stderr,
        )
    return index


# =============================================================================
# regression gate (baseline mode)
# =============================================================================
def load_allowlist(path: Path, provider: str) -> dict:
    """{sig_hash: allowlist entry} for this provider."""
    allow: dict = {}
    if not path.exists():
        return allow
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("provider") == provider:
                allow[entry["sig_hash"]] = entry
    return allow


def run_gate(clusters, entries, allow: dict, min_coverage: float):
    """Return (violations, coverage). A violation is a NEW divergent cluster
    (signature not allowlisted), a severity escalation beyond the allowlisted
    severity, or replay coverage too low to trust a green result."""
    violations = []
    for c in clusters:
        if c["verdict"] not in ("S1", "S2", "S3", "S4"):
            continue
        sig = c.get("sig_hash")
        allowed = allow.get(sig)
        if allowed is None:
            violations.append(
                {
                    "kind": "new-divergent-cluster",
                    "cluster_id": c["cluster_id"],
                    "sig_hash": sig,
                    "verdict": c["verdict"],
                    "size": c["size"],
                    "categories": c["categories"][:5],
                    "example_probes": c["example_probes"][:3],
                }
            )
        elif _SEV_RANK[c["verdict"]] < _SEV_RANK.get(allowed.get("verdict"), 99):
            violations.append(
                {
                    "kind": "severity-escalation",
                    "cluster_id": c["cluster_id"],
                    "sig_hash": sig,
                    "verdict": c["verdict"],
                    "allowlisted_verdict": allowed.get("verdict"),
                    "size": c["size"],
                    "example_probes": c["example_probes"][:3],
                }
            )
    uncovered = ("replay-gap", "replay-transport")
    usable = sum(1 for e in entries if e["class"] not in uncovered)
    coverage = usable / len(entries) if entries else 0.0
    if coverage < min_coverage:
        violations.append(
            {
                "kind": "low-replay-coverage",
                "coverage": round(coverage, 4),
                "min_coverage": min_coverage,
                "note": "too few baseline probes produced comparable SMG responses; "
                "a green gate would be meaningless",
            }
        )
    return violations, coverage


def write_allowlist(path: Path, provider: str, clusters, stamp: str):
    """Write this provider's divergent clusters as allowlist lines, keeping
    other providers' existing lines. Idempotent."""
    keep = []
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and json.loads(line).get("provider") != provider:
                    keep.append(json.loads(line))
    fresh = [
        {
            "provider": provider,
            "sig_hash": c["sig_hash"],
            "cluster_id": c["cluster_id"],
            "verdict": c["verdict"],
            "s1_kinds": c["s1_kinds"],
            "size": c["size"],
            "categories": c["categories"][:5],
            "example_probes": c["example_probes"][:3],
            "first_seen": stamp,
            "note": "",
        }
        for c in clusters
        if c["verdict"] in ("S1", "S2", "S3", "S4") and c.get("sig_hash")
    ]
    lines = sorted(
        keep + fresh,
        key=lambda e: (e["provider"], _SEV_RANK.get(e["verdict"], 99), e["cluster_id"]),
    )
    with path.open("w", encoding="utf-8") as fh:
        for e in lines:
            fh.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")
    return len(fresh)


# =============================================================================
# error-envelope extraction
# =============================================================================
def error_envelope(body, provider):
    """Normalize a 4xx envelope into comparable fields; None if not an error."""
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if not isinstance(err, dict):
        return None
    return {
        "shape": tuple(sorted(field_paths(body))),
        "type": err.get("type"),
        "code": err.get("code"),
        "param": err.get("param"),
        "message": err.get("message") if isinstance(err.get("message"), str) else None,
    }


_MUT_ID = re.compile(r"\.(?:gen\.)?mut\.([A-Za-z0-9_\-]+)\.[a-z\-]+$")


def mutated_field_tokens(probe_id: str) -> list:
    """openai.gen.mut.tools0-name.wrong-type-number -> ["tools", "name"]."""
    m = _MUT_ID.search(probe_id)
    if not m:
        return []
    slug = m.group(1)
    toks = [re.sub(r"\d+$", "", t) for t in slug.split("-")]
    return [t for t in toks if t]


def message_names_field(message: str | None, tokens: list) -> bool | None:
    if message is None or not tokens:
        return None
    msg = message.lower()
    return all(t.lower() in msg for t in tokens if len(t) > 1)


# =============================================================================
# per-probe comparison
# =============================================================================
def split_paths(vendor_paths, smg_paths, provider):
    """Partition inventory differences into structural vs mock-limited."""
    prefixes = _CONTENT_DEPENDENT_PATH_PREFIXES[provider]

    def content_dep(p):
        return any(p == pre or p.startswith(pre + ".") or p.startswith(pre) for pre in prefixes)

    missing = sorted(vendor_paths - smg_paths)  # vendor has, SMG omits
    extra = sorted(smg_paths - vendor_paths)  # SMG has, vendor lacks
    missing_struct = [p for p in missing if not content_dep(p)]
    missing_mock = [p for p in missing if content_dep(p)]
    # SMG-only paths under content-dependent prefixes are also mock-shaped
    # (the mock generated a plain message where the vendor generated tools).
    extra_struct = [p for p in extra if not content_dep(p)]
    extra_mock = [p for p in extra if content_dep(p)]
    return missing_struct, extra_struct, missing_mock, extra_mock


def compare_events(v_names, s_names, provider):
    """Compare collapsed event sequences; returns (divergent, detail)."""
    v_seq, s_seq = collapse_runs(v_names), collapse_runs(s_names)
    if v_seq == s_seq:
        return False, None
    content_events = set(_CONTENT_DEPENDENT_EVENTS[provider])

    def strip(seq):
        out = []
        for n in seq:
            base = n.rstrip("+")
            if base in content_events:
                continue
            out.append("<terminal>" if base in _TERMINAL_EQUIV else n)
        return tuple(out)

    v_core, s_core = strip(v_seq), strip(s_seq)
    detail = {
        "vendor_events": list(v_seq),
        "smg_events": list(s_seq),
        "vendor_core": list(v_core),
        "smg_core": list(s_core),
    }
    if v_core == s_core:
        return False, detail  # differences confined to content-bound events

    # Delta-run arity differences on the core sequence are content-bound too.
    def drop_deltaish(seq):
        return tuple(n.rstrip("+") for n in seq if not _DELTAISH.search(n.rstrip("+")))

    if drop_deltaish(v_core) == drop_deltaish(s_core):
        return False, detail
    return True, detail


def classify(pid, vrec, srec, provider, extension_allowlist):
    """Return a diff entry dict for one probe."""
    entry = {
        "probe_id": pid,
        "category": vrec.get("category"),
        "expect": vrec.get("expect"),
        "status_vendor": get_status(vrec),
        "status_smg": get_status(srec),
        "class": None,
        "severity": None,
        "notes": [],
    }
    v_status, s_status = entry["status_vendor"], entry["status_smg"]

    # --- replay availability ---------------------------------------------
    if srec is None:
        entry["class"] = "replay-gap"
        entry["notes"].append("probe absent from SMG replay")
        return entry
    if srec.get("skipped"):
        entry["class"] = "replay-gap"
        entry["notes"].append(f"SMG replay skipped: {srec.get('error')}")
        return entry
    if srec.get("transport_failure") and s_status is None:
        entry["class"] = "replay-transport"
        entry["notes"].append(f"SMG transport failure: {srec.get('error')}")
        return entry

    # Auth is disabled in the local replay (SMG's serving auth is Bearer-only
    # while the Anthropic adapter authenticates via x-api-key, so enabling it
    # would 401 the whole Anthropic matrix). Vendor 401s are therefore not
    # comparable — config-limited, never a divergence.
    if v_status == 401:
        entry["class"] = "config-limited"
        entry["notes"].append("vendor 401 (auth); replay runs with auth disabled")
        return entry

    findings = []  # list of (severity, note)

    # --- (a) status ------------------------------------------------------
    status_exact = v_status == s_status
    status_class = (
        v_status is not None and s_status is not None and (v_status // 100) == (s_status // 100)
    )
    if not status_class:
        if s_status is not None and s_status >= 500:
            entry["s1_kind"] = "server-error"
            note = f"SMG 5xx on a request the vendor answered {v_status}"
        elif v_status is not None and v_status < 400 <= (s_status or 0):
            entry["s1_kind"] = "rejects-valid"
            note = f"SMG rejects ({s_status}) what the vendor accepts ({v_status})"
        else:
            entry["s1_kind"] = "accepts-invalid"
            note = f"SMG {s_status} where vendor rejected with {v_status}"
        findings.append(
            ("S1", f"status class mismatch: vendor {v_status} vs SMG {s_status}; {note}")
        )
    elif not status_exact:
        # same class, different code (e.g. 400 vs 422): envelope-level drift
        findings.append(("S4", f"status code differs within class: {v_status} vs {s_status}"))

    v_stream, s_stream = is_streaming(vrec), is_streaming(srec)
    mock_limited_notes = []

    if v_stream != s_stream:
        # A stream answered as JSON (or vice versa) is a framing divergence,
        # except when the non-2xx side is an error body (errors are JSON).
        if status_class and v_status == 200:
            findings.append(
                (
                    "S3",
                    f"framing mismatch: vendor {'SSE' if v_stream else 'JSON'} "
                    f"vs SMG {'SSE' if s_stream else 'JSON'}",
                )
            )
    elif v_stream:
        # --- (c) SSE event-type sequence ---------------------------------
        divergent, detail = compare_events(
            event_names(vrec, provider), event_names(srec, provider), provider
        )
        if detail:
            entry["events"] = detail
        if divergent:
            findings.append(("S3", "SSE core event sequence drift"))
        elif detail:
            mock_limited_notes.append("SSE differences confined to content-bound events")
    else:
        # --- (b) field-path inventory ------------------------------------
        v_body, s_body = get_body(vrec), get_body(srec)
        if isinstance(v_body, (dict, list)) or isinstance(s_body, (dict, list)):
            v_paths = field_paths(v_body) if isinstance(v_body, (dict, list)) else set()
            s_paths = field_paths(s_body) if isinstance(s_body, (dict, list)) else set()
            miss_st, extra_st, miss_mock, extra_mock = split_paths(v_paths, s_paths, provider)
            extra_benign = [p for p in extra_st if p in extension_allowlist]
            extra_st = [p for p in extra_st if p not in extension_allowlist]
            if miss_st:
                entry["missing_paths"] = miss_st
            if extra_st:
                entry["extra_paths"] = extra_st
            if extra_benign:
                entry["benign_extra_paths"] = extra_benign
            if miss_mock or extra_mock:
                entry["mock_limited_paths"] = {
                    "vendor_only": miss_mock,
                    "smg_only": extra_mock,
                }
                mock_limited_notes.append("content-dependent body paths differ")
            if (miss_st or extra_st) and v_status == 200 and status_class:
                findings.append(
                    (
                        "S2",
                        f"field inventory drift: {len(miss_st)} vendor-only, "
                        f"{len(extra_st)} smg-only paths",
                    )
                )

    # --- (d) error envelope ---------------------------------------------
    if v_status is not None and 400 <= v_status < 500 and s_status is not None:
        v_env = error_envelope(get_body(vrec), provider)
        s_env = error_envelope(get_body(srec), provider)
        env = {"vendor": v_env, "smg": s_env}
        if v_env and s_env:
            toks = mutated_field_tokens(pid)
            env["vendor_names_field"] = message_names_field(v_env["message"], toks)
            env["smg_names_field"] = message_names_field(s_env["message"], toks)
            for k in ("type", "code", "param"):
                if scrub(str(v_env[k])) != scrub(str(s_env[k])):
                    findings.append(("S4", f"error.{k} differs: {v_env[k]!r} vs {s_env[k]!r}"))
                    break
            if (
                env["vendor_names_field"] is True
                and env["smg_names_field"] is False
                and status_class
            ):
                findings.append(("S4", "SMG error message does not name the offending field"))
        elif v_env and not s_env and status_class:
            findings.append(("S4", "SMG 4xx body is not a structured error envelope"))
        entry["error_envelope"] = {
            "vendor": _env_brief(v_env),
            "smg": _env_brief(s_env),
            "vendor_names_field": env.get("vendor_names_field"),
            "smg_names_field": env.get("smg_names_field"),
        }

    # --- verdict ----------------------------------------------------------
    if findings:
        worst = min(findings, key=lambda f: _SEV_RANK[f[0]])
        entry["class"] = "divergence"
        entry["severity"] = worst[0]
        entry["notes"].extend(n for _, n in findings)
    elif entry.get("benign_extra_paths"):
        entry["class"] = "benign"
        entry["severity"] = "benign"
        entry["notes"].append("SMG-only fields are documented extensions")
    elif mock_limited_notes:
        entry["class"] = "mock-limited"
        entry["severity"] = "mock-limited"
        entry["notes"].extend(mock_limited_notes)
    else:
        entry["class"] = "exact"
        entry["severity"] = "exact"
    return entry


def _env_brief(env):
    if not env:
        return None
    return {
        "type": env["type"],
        "code": env["code"],
        "param": env["param"],
        "message": (env["message"] or "")[:200],
    }


# =============================================================================
# aggregation + report
# =============================================================================
def worst_severity(sevs):
    return min(sevs, key=lambda s: _SEV_RANK.get(s, len(_SEVERITY_ORDER)))


def aggregate(entries, vendor_records, provider):
    clusters = defaultdict(lambda: {"probes": [], "severities": Counter(), "classes": Counter()})
    for e in entries:
        key = vendor_cluster_key(vendor_records.get(e["probe_id"]), provider)
        c = clusters[key]
        c["probes"].append(e)
        c["classes"][e["class"]] += 1
        if e["severity"]:
            c["severities"][e["severity"]] += 1
    out = []
    s1_rank = {"server-error": 0, "rejects-valid": 1, "accepts-invalid": 2}
    for i, (key, c) in enumerate(sorted(clusters.items(), key=lambda kv: -len(kv[1]["probes"]))):
        skip_classes = ("replay-gap", "replay-transport", "config-limited")
        comparable = [e for e in c["probes"] if e["class"] not in skip_classes]
        sevs = [e["severity"] for e in comparable if e["severity"]]
        cls_counter = c["classes"]
        if not comparable:
            verdict = "config-limited" if cls_counter.get("config-limited") else "replay-gap"
        elif any(e["class"] == "divergence" for e in comparable):
            verdict = worst_severity([s for s in sevs if s in ("S1", "S2", "S3", "S4")])
        elif any(e["class"] == "benign" for e in comparable):
            verdict = "benign"
        elif any(e["class"] == "mock-limited" for e in comparable):
            verdict = "mock-limited"
        else:
            verdict = "exact"
        s1_kinds = sorted(
            {e.get("s1_kind") for e in comparable if e.get("s1_kind")},
            key=lambda k: s1_rank.get(k, 9),
        )
        out.append(
            {
                "cluster_id": f"C{i:03d}",
                "key": key,
                "kind": key[0],
                "vendor_status": key[1],
                "size": len(c["probes"]),
                "verdict": verdict,
                "s1_kinds": s1_kinds,
                "s1_rank": s1_rank.get(s1_kinds[0], 9) if s1_kinds else 9,
                "classes": dict(cls_counter),
                "severities": dict(c["severities"]),
                "example_probes": [e["probe_id"] for e in c["probes"][:5]],
                "categories": sorted({e["category"] for e in c["probes"] if e["category"]}),
                "entries": c["probes"],
            }
        )
    return out


def _md_escape(s):
    return str(s).replace("|", "\\|")


def render_report(provider, vendor_dir, smg_dir, entries, clusters):
    n = Counter(e["class"] for e in entries)
    sev = Counter(e["severity"] for e in entries if e["class"] == "divergence")
    cluster_verdicts = Counter(c["verdict"] for c in clusters)

    lines = []
    a = lines.append
    a(f"# SMG structural compatibility report — {provider}")
    a("")
    a(f"- vendor truth: `{vendor_dir}`")
    a(f"- SMG replay:   `{smg_dir}`")
    if any(c.get("sig_hash") for c in clusters):
        a("- cluster ids: baseline (stable `Bnnn`, allowlist keys on `sig_hash`)")
    a(f"- probes compared: {len(entries)}")
    a(
        f"- probe verdicts: {n.get('exact', 0)} exact, {n.get('benign', 0)} benign, "
        f"{n.get('mock-limited', 0)} mock-limited, {n.get('divergence', 0)} divergent, "
        f"{n.get('config-limited', 0)} config-limited, "
        f"{n.get('replay-gap', 0)} replay-gap, {n.get('replay-transport', 0)} transport"
    )
    a(
        "- divergence severities: "
        + ", ".join(f"{k}={sev[k]}" for k in ("S1", "S2", "S3", "S4") if sev.get(k))
    )
    a(
        f"- vendor behavior clusters: {len(clusters)} — "
        + ", ".join(f"{v} {k}" for k, v in sorted(cluster_verdicts.items()))
    )
    a("")
    a(
        "Severity legend: S1 status-class mismatch > S2 field-inventory drift > "
        "S3 SSE core-sequence drift > S4 error-envelope drift. `mock-limited` = "
        "comparison meaningless against a canned mock worker (content-dependent)."
    )
    a("")

    a("## Divergent clusters (ranked)")
    a("")
    a("`at-sev` = members exhibiting the cluster's worst severity (cluster")
    a("verdicts are worst-member; the rest of the cluster may be milder).")
    a("")
    a(
        "| rank | cluster | sev | at-sev | size | vendor status | categories | example probes | detail |"
    )
    a("|---|---|---|---|---|---|---|---|---|")
    ranked = [c for c in clusters if c["verdict"] in ("S1", "S2", "S3", "S4")]
    ranked.sort(key=lambda c: (_SEV_RANK[c["verdict"]], c["s1_rank"], -c["size"]))
    for i, c in enumerate(ranked, 1):
        div = [e for e in c["entries"] if e["class"] == "divergence"]
        # Lead with the worst-severity members so an S1 cluster's examples and
        # detail column explain the S1, not the accompanying field drift.
        worst = [e for e in div if e["severity"] == c["verdict"]] or div
        notes = Counter()
        for e in worst:
            for note in e["notes"]:
                notes[scrub(note)] += 1
        top_notes = "; ".join(f"{k} (x{v})" for k, v in notes.most_common(3))
        sev_label = c["verdict"]
        if c["s1_kinds"]:
            sev_label += f" ({'/'.join(c['s1_kinds'])})"
        examples = [e["probe_id"] for e in worst[:3]]
        a(
            f"| {i} | {c['cluster_id']} | {sev_label} | {len(worst)} | {c['size']} "
            f"| {c['vendor_status']} | {_md_escape(', '.join(c['categories'][:3]))} "
            f"| {_md_escape(', '.join(examples))} | {_md_escape(top_notes)} |"
        )
    a("")

    a("## Aggregate field-inventory drift (S2 rollup)")
    a("")
    a("Structural field-path differences across all compared 200-vs-200 probes")
    a("(count = probes exhibiting the difference).")
    a("")
    miss, extra = Counter(), Counter()
    for e in entries:
        if e["class"] != "divergence":
            continue
        if e["status_vendor"] != 200 or e["status_smg"] != 200:
            continue  # inventory diffs are only meaningful 200-vs-200
        for p in e.get("missing_paths", []):
            miss[p] += 1
        for p in e.get("extra_paths", []):
            extra[p] += 1
    a("Vendor fields SMG omits:")
    a("")
    for p, cnt in miss.most_common(40):
        a(f"- `{p}` (x{cnt})")
    a("")
    a("SMG fields the vendor lacks:")
    a("")
    for p, cnt in extra.most_common(40):
        a(f"- `{p}` (x{cnt})")
    a("")

    a("## Benign clusters (documented SMG extensions)")
    a("")
    for c in (c for c in clusters if c["verdict"] == "benign"):
        a(f"- {c['cluster_id']} ({c['size']} probes): {', '.join(c['example_probes'][:3])}")
    a("")

    a("## Mock-limited clusters")
    a("")
    a("Content-dependent behavior the canned mock cannot reproduce; replay is")
    a("structurally consistent everywhere else in these clusters.")
    a("")
    for c in (c for c in clusters if c["verdict"] == "mock-limited"):
        a(
            f"- {c['cluster_id']} ({c['size']} probes, vendor {c['vendor_status']}): "
            f"{', '.join(c['example_probes'][:3])}"
        )
    a("")

    a("## Exact-match clusters")
    a("")
    exact = [c for c in clusters if c["verdict"] == "exact"]
    a(f"{len(exact)} clusters match exactly after normalization:")
    for c in exact:
        a(
            f"- {c['cluster_id']} ({c['size']} probes, vendor {c['vendor_status']}): "
            f"{', '.join(c['example_probes'][:3])}"
        )
    a("")
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="vendor-vs-SMG structural compat differ")
    ap.add_argument("--vendor", default=None, help="vendor truth results dir (raw recordings)")
    ap.add_argument(
        "--baseline",
        default=None,
        help="checked-in baseline dir (vendor_probe/baselines/<date>/<provider>) "
        "to diff against instead of raw vendor recordings",
    )
    ap.add_argument("--smg", required=True, help="SMG replay results dir")
    ap.add_argument(
        "--provider",
        required=True,
        choices=["openai", "anthropic"],
        help="protocol family (drives SSE parsing + content-dependence tables)",
    )
    ap.add_argument("--out", required=True, help="output dir (report.md + diff.jsonl)")
    ap.add_argument(
        "--extensions",
        default=None,
        help="optional JSON file: list of SMG-only field paths documented as extensions",
    )
    ap.add_argument(
        "--allowlist",
        default=None,
        help="known-divergences jsonl (baseline mode): exit 1 on divergent clusters "
        "whose sig_hash is not listed, or whose severity got worse",
    )
    ap.add_argument(
        "--write-allowlist",
        default=None,
        help="baseline mode: (re)write this provider's allowlist lines from the "
        "current run (grandfather today's divergences); suppresses the failing exit",
    )
    ap.add_argument(
        "--min-replay-coverage",
        type=float,
        default=0.8,
        help="baseline mode: gate fails when fewer than this fraction of baseline "
        "probes produced comparable SMG responses",
    )
    args = ap.parse_args(argv)

    if bool(args.vendor) == bool(args.baseline):
        ap.error("exactly one of --vendor / --baseline is required")

    smg_dir = Path(args.smg)
    smg_records = load_results(smg_dir)
    extension_allowlist = set()
    if args.extensions:
        extension_allowlist = set(json.loads(Path(args.extensions).read_text()))

    key_index = None
    if args.baseline:
        vendor_dir = Path(args.baseline)
        baseline_clusters = load_baseline(vendor_dir)
        vendor_records = synth_vendor_records(baseline_clusters)
        key_index = baseline_key_index(baseline_clusters, vendor_records, args.provider)
    else:
        vendor_dir = Path(args.vendor)
        vendor_records = load_results(vendor_dir)

    entries = []
    for pid, vrec in vendor_records.items():
        if vrec.get("skipped") or not vrec.get("recorded"):
            continue  # no vendor truth to compare against
        if vrec.get("transport_failure"):
            continue  # vendor-side 429/5xx is not ground truth
        entries.append(
            classify(pid, vrec, smg_records.get(pid), args.provider, extension_allowlist)
        )

    clusters = aggregate(entries, vendor_records, args.provider)
    if key_index is not None:
        # Relabel with the baseline's stable ids so reports and the allowlist
        # agree across runs.
        for c in clusters:
            bc = key_index.get(c["key"])
            if bc is not None:
                c["cluster_id"] = bc["cluster_id"]
                c["sig_hash"] = bc["sig_hash"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "diff.jsonl").open("w") as f:
        for c in clusters:
            for e in c["entries"]:
                e2 = dict(e)
                e2["cluster_id"] = c["cluster_id"]
                e2["cluster_verdict"] = c["verdict"]
                if c.get("sig_hash"):
                    e2["cluster_sig_hash"] = c["sig_hash"]
                f.write(json.dumps(e2, ensure_ascii=False) + "\n")
    report = render_report(args.provider, vendor_dir, smg_dir, entries, clusters)
    (out_dir / "report.md").write_text(report)

    n = Counter(e["class"] for e in entries)
    cv = Counter(c["verdict"] for c in clusters)
    summary = {
        "provider": args.provider,
        "probes_compared": len(entries),
        "probe_classes": dict(n),
        "clusters": len(clusters),
        "cluster_verdicts": dict(cv),
        "report": str(out_dir / "report.md"),
    }

    # --- regression gate (baseline mode only) -----------------------------
    exit_code = 0
    if args.baseline and (args.allowlist or args.write_allowlist):
        if args.write_allowlist:
            stamp = None
            manifest_path = vendor_dir / "manifest.json"
            if manifest_path.exists():
                stamp = json.loads(manifest_path.read_text()).get("date")
            if not stamp:
                stamp = datetime.date.today().isoformat()
            written = write_allowlist(Path(args.write_allowlist), args.provider, clusters, stamp)
            summary["allowlist_written"] = written
        allow = load_allowlist(Path(args.allowlist), args.provider) if args.allowlist else {}
        violations, coverage = run_gate(clusters, entries, allow, args.min_replay_coverage)
        gate = {
            "provider": args.provider,
            "allowlisted_clusters": len(allow),
            "divergent_clusters": sum(
                1 for c in clusters if c["verdict"] in ("S1", "S2", "S3", "S4")
            ),
            "replay_coverage": round(coverage, 4),
            "violations": violations,
        }
        (out_dir / "gate.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False))
        summary["gate"] = {k: gate[k] for k in gate if k != "violations"}
        summary["gate"]["violations"] = len(violations)
        if violations and not args.write_allowlist:
            for v in violations:
                print(f"GATE VIOLATION: {json.dumps(v, ensure_ascii=False)}", file=sys.stderr)
            exit_code = 1

    print(json.dumps(summary, indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
