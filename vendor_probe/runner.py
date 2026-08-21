#!/usr/bin/env python3
"""Vendor ground-truth probe runner.

Dual-target: the same probe matrix runs against the real OpenAI / Anthropic
APIs today and can replay against an SMG gateway later by swapping the
provider adapter (base_url + auth + model injection). Records raw request /
response bytes, SSE transcripts, structural fingerprints, and a summary so a
later diff tool can audit SMG's compatibility.

Usage:
  python -m vendor_probe.runner --provider openai   --out results/openai
  python -m vendor_probe.runner --provider anthropic --out results/anthropic
  python -m vendor_probe.runner --provider openai --dry-run    # no network

Env (defaults are cheap):
  OPENAI_API_KEY, OPENAI_MODEL=gpt-5-nano, OPENAI_MODEL_CLASSIC=gpt-4.1-nano,
  OPENAI_MODEL_CUA=computer-use-preview, OPENAI_MODEL_SHELL=codex-mini-latest,
  OPENAI_VECTOR_STORE_ID (optional), OPENAI_PROBE_IMAGES=1 (opt-in cost),
  OPENAI_BASE_URL=https://api.openai.com
  ANTHROPIC_API_KEY, ANTHROPIC_MODEL=claude-haiku-4-5,
  ANTHROPIC_VERSION=2023-06-01, ANTH_PROBE_EXPENSIVE=1 (opt-in cost),
  ANTHROPIC_BASE_URL=https://api.anthropic.com
  SMG_BASE_URL, SMG_API_KEY (for smg-openai / smg-anthropic replay targets)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import httpx
except ImportError:  # pragma: no cover - dry-run works without httpx
    httpx = None

# Import probe matrices (support both `-m vendor_probe.runner` and direct run).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vendor_probe import genmatrix
    from vendor_probe.probes import anthropic_messages, openai_responses
else:
    from . import genmatrix
    from .probes import anthropic_messages, openai_responses

# --- response header allowlist ----------------------------------------------
_HEADER_EXACT = {
    "content-type",
    "retry-after",
    "x-request-id",
    "request-id",
    "openai-version",
    "openai-processing-ms",
    "openai-organization",
    "anthropic-organization-id",
}
_HEADER_PREFIX = ("x-ratelimit-", "anthropic-ratelimit-", "openai-")

# rough $/1M token pricing for cost estimate (best-effort by model substring)
_PRICING = [
    ("gpt-5-nano", 0.05, 0.40),
    ("gpt-4.1-nano", 0.10, 0.40),
    ("gpt-5-mini", 0.25, 2.00),
    ("gpt-4.1-mini", 0.40, 1.60),
    ("claude-haiku", 1.00, 5.00),
    ("claude-sonnet", 3.00, 15.00),
    ("claude-opus", 15.00, 75.00),
]


class SkipProbe(Exception):
    """Raised when a probe cannot run (missing optional resource / gated)."""


# =============================================================================
# provider adapters
# =============================================================================
@dataclass
class Adapter:
    name: str
    base_url: str
    auth_headers: dict
    sentinels: dict  # sentinel -> real model name (always resolvable)
    optional: dict  # sentinel -> value or None (None => skip probe)
    probes: list
    extra_version_header: dict = field(default_factory=dict)

    def all_sentinels(self):
        d = dict(self.sentinels)
        d.update(self.optional)
        return d


def _openai_adapter(base_env, key, model, model_classic):
    return Adapter(
        name="openai",
        base_url=os.environ.get(base_env, "https://api.openai.com").rstrip("/"),
        auth_headers={"Authorization": f"Bearer {key}"} if key else {},
        sentinels={
            "@MODEL": model,
            "@MODEL_CLASSIC": model_classic,
            "@MODEL_CUA": os.environ.get("OPENAI_MODEL_CUA", "computer-use-preview"),
            "@MODEL_SHELL": os.environ.get("OPENAI_MODEL_SHELL", "codex-mini-latest"),
        },
        optional={"@VECTOR_STORE_ID": os.environ.get("OPENAI_VECTOR_STORE_ID")},
        probes=openai_responses.PROBES,
    )


def _anthropic_adapter(base_env, key, model):
    version = os.environ.get("ANTHROPIC_VERSION", "2023-06-01")
    hdr = {"anthropic-version": version}
    if key:
        hdr["x-api-key"] = key
    return Adapter(
        name="anthropic",
        base_url=os.environ.get(base_env, "https://api.anthropic.com").rstrip("/"),
        auth_headers=hdr,
        sentinels={"@MODEL": model},
        optional={},
        probes=anthropic_messages.PROBES,
    )


def build_adapter(provider: str) -> Adapter:
    o_model = os.environ.get("OPENAI_MODEL", "gpt-5-nano")
    o_classic = os.environ.get("OPENAI_MODEL_CLASSIC", "gpt-4.1-nano")
    a_model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
    if provider == "openai":
        return _openai_adapter(
            "OPENAI_BASE_URL", os.environ.get("OPENAI_API_KEY"), o_model, o_classic
        )
    if provider == "anthropic":
        return _anthropic_adapter(
            "ANTHROPIC_BASE_URL", os.environ.get("ANTHROPIC_API_KEY"), a_model
        )
    if provider == "smg-openai":
        # SMG's OpenAI-compatible surface. Bearer auth optional.
        a = _openai_adapter("SMG_BASE_URL", os.environ.get("SMG_API_KEY"), o_model, o_classic)
        a.name = "smg-openai"
        return a
    if provider == "smg-anthropic":
        a = _anthropic_adapter("SMG_BASE_URL", os.environ.get("SMG_API_KEY"), a_model)
        a.name = "smg-anthropic"
        return a
    raise ValueError(f"unknown provider {provider!r}")


# =============================================================================
# placeholder + sentinel resolution
# =============================================================================
_REF_RE = re.compile(r"\{\{([^}]+)\}\}")
_WHOLE_REF_RE = re.compile(r"^\{\{([^}]+)\}\}$")
_TOK_RE = re.compile(r"^([A-Za-z_]\w*)?(?:\[(\d+|\?\w+)\])?$")


def get_path(root, path):
    cur = root
    for tok in path.split("."):
        m = _TOK_RE.match(tok)
        if not m:
            raise KeyError(f"bad path token {tok!r} in {path!r}")
        name, idx = m.group(1), m.group(2)
        if name:
            cur = cur[name]
        if idx is not None:
            if idx.startswith("?"):
                t = idx[1:]
                cur = next(x for x in cur if isinstance(x, dict) and x.get("type") == t)
            else:
                cur = cur[int(idx)]
    return cur


def refs_in(value) -> set:
    out = set()

    def walk(v):
        if isinstance(v, str):
            for m in _REF_RE.finditer(v):
                spec = m.group(1)
                pid = spec.split("#", 1)[0]
                if pid != "self":
                    out.add(pid)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)

    walk(value)
    return out


def make_resolver(sub_sentinel, sub_ref):
    def resolve_str(s):
        m = _WHOLE_REF_RE.match(s)
        if m:
            return sub_ref(m.group(1))  # may be non-string
        s = _REF_RE.sub(lambda mo: str(sub_ref(mo.group(1))), s)
        return sub_sentinel(s)

    def walk(v):
        if isinstance(v, str):
            return resolve_str(v)
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        return v

    return walk


def sentinel_subber(adapter: Adapter, dry: bool):
    table = adapter.all_sentinels()
    keys = sorted(table, key=len, reverse=True)

    def sub(s):
        for k in keys:
            if k in s:
                val = table[k]
                if val is None:
                    if dry:
                        val = f"__{k.strip('@')}__"
                    else:
                        raise SkipProbe(f"missing {k}")
                s = s.replace(k, val)
        return s

    return sub


# =============================================================================
# fingerprints
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


def stream_event_names(sse_events, provider):
    names = []
    for ev in sse_events:
        if provider.endswith("anthropic"):
            parsed = ev.get("parsed") or {}
            names.append(parsed.get("type") or ev.get("event") or "?")
        else:
            names.append(ev.get("event") or "?")
    return ">".join(names)


def fingerprint_sig(record, provider):
    """Hashable behavior signature; identical sigs = one behavior cluster."""
    resp = record.get("response") or {}
    if "sse_events" in resp:
        return ("sse", resp.get("status"), stream_event_names(resp["sse_events"], provider))
    body = resp.get("body")
    paths = tuple(sorted(field_paths(body))) if isinstance(body, (dict, list)) else ()
    return ("json", resp.get("status"), paths)


# =============================================================================
# HTTP execution
# =============================================================================
def capture_headers(headers):
    out = {}
    for k, v in headers.items():
        lk = k.lower()
        if lk in _HEADER_EXACT or lk.startswith(_HEADER_PREFIX):
            out[k] = v
    return out


def parse_sse(text):
    """Parse an SSE stream body into ordered event dicts."""
    events = []
    event_name = None
    data_lines = []
    for raw in text.split("\n"):
        line = raw.rstrip("\r")
        if line == "":
            if data_lines or event_name is not None:
                data = "\n".join(data_lines)
                parsed = None
                try:
                    parsed = json.loads(data)
                except Exception:
                    pass
                seq = parsed.get("sequence_number") if isinstance(parsed, dict) else None
                events.append(
                    {"event": event_name, "data": data, "parsed": parsed, "sequence_number": seq}
                )
            event_name = None
            data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    return events


def _retry_after(headers, attempt):
    ra = headers.get("retry-after") or headers.get("Retry-After")
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    return min(30.0, (2**attempt) + random.random())


async def _send_once(client, method, url, headers, body, stream, timeout):
    kwargs = {"headers": headers, "timeout": timeout}
    if isinstance(body, str):
        kwargs["content"] = body
    elif body is not None:
        kwargs["json"] = body
    if stream:
        async with client.stream(method, url, **kwargs) as resp:
            chunks = []
            async for chunk in resp.aiter_text():
                chunks.append(chunk)
            resp_headers = capture_headers(resp.headers)
            return resp.status_code, resp_headers, None, "".join(chunks)
    resp = await client.request(method, url, **kwargs)
    return resp.status_code, capture_headers(resp.headers), resp, resp.text


async def execute(client, adapter, probe, recordings, cfg):
    """Run a single probe, with retries and optional polling. Returns a record."""
    pid = probe["id"]
    rec = {
        "probe_id": pid,
        "category": probe["category"],
        "request": {"method": probe["method"], "path": probe["endpoint"], "body": None},
        "response": None,
        "timing_ms": None,
        "attempt_count": 0,
        "error": None,
        "expect": probe["expect"],
        "recorded": False,
        "transport_failure": False,
        "skipped": False,
        "note": probe.get("note"),
    }

    # cost/gating skips
    note = probe.get("note") or ""
    if "OPENAI_PROBE_IMAGES" in note and os.environ.get("OPENAI_PROBE_IMAGES") != "1":
        rec["skipped"] = True
        rec["error"] = "gated: set OPENAI_PROBE_IMAGES=1"
        return rec
    if "ANTH_PROBE_EXPENSIVE" in note and os.environ.get("ANTH_PROBE_EXPENSIVE") != "1":
        rec["skipped"] = True
        rec["error"] = "gated: set ANTH_PROBE_EXPENSIVE=1"
        return rec

    # cascade skip if a dependency did not produce a usable recording
    for dep in probe["_waits"]:
        drec = recordings.get(dep)
        if drec is None or drec.get("skipped") or drec.get("_root", {}).get("body") is None:
            rec["skipped"] = True
            rec["error"] = f"dependency {dep} unavailable"
            return rec

    # resolve sentinels + placeholders
    def sub_ref(spec):
        did, path = spec.split("#", 1)
        root = recordings[pid]["_root"] if did == "self" else recordings[did]["_root"]
        return get_path(root, path)

    sub_sentinel = sentinel_subber(adapter, dry=False)
    resolve = make_resolver(sub_sentinel, sub_ref)
    try:
        endpoint = resolve(probe["endpoint"])
        body = resolve(probe["body"]) if probe["body"] is not None else None
        extra_headers = resolve(probe["headers"]) if probe["headers"] else {}
    except SkipProbe as e:
        rec["skipped"] = True
        rec["error"] = str(e)
        return rec
    except Exception as e:  # resolution failure counts as skip, not transport
        rec["skipped"] = True
        rec["error"] = f"resolve error: {e}"
        return rec

    rec["request"]["path"] = endpoint
    rec["request"]["body"] = body
    url = adapter.base_url + endpoint

    headers = {"content-type": "application/json"}
    headers.update(adapter.auth_headers)
    headers.update(adapter.extra_version_header)
    for k, v in (extra_headers or {}).items():
        if v is None:
            headers.pop(k, None)
            headers.pop(k.lower(), None)
            for hk in list(headers):
                if hk.lower() == k.lower():
                    headers.pop(hk)
        else:
            headers[k] = v

    timeout = cfg["stream_timeout"] if probe["stream"] else cfg["timeout"]
    t0 = time.time()
    last_exc = None
    status = resp_headers = text = None
    for attempt in range(cfg["max_retries"] + 1):
        rec["attempt_count"] = attempt + 1
        try:
            status, resp_headers, _resp, text = await _send_once(
                client, probe["method"], url, headers, body, probe["stream"], timeout
            )
        except Exception as e:  # transport-level
            last_exc = repr(e)
            if attempt < cfg["max_retries"]:
                await asyncio.sleep(min(30.0, (2**attempt) + random.random()))
                continue
            rec["timing_ms"] = int((time.time() - t0) * 1000)
            rec["error"] = f"transport: {last_exc}"
            rec["transport_failure"] = True
            return rec
        # retry only transient server-side statuses; 4xx are recordings
        if status == 429 or status >= 500:
            if attempt < cfg["max_retries"]:
                await asyncio.sleep(_retry_after(resp_headers, attempt))
                continue
        break

    rec["timing_ms"] = int((time.time() - t0) * 1000)
    response = {"status": status, "headers": resp_headers}
    parsed_body = None
    if probe["stream"]:
        response["sse_events"] = parse_sse(text or "")
    else:
        try:
            parsed_body = json.loads(text) if text else None
        except Exception:
            parsed_body = None
        response["body"] = parsed_body if parsed_body is not None else text
    rec["response"] = response
    rec["recorded"] = True
    rec["_root"] = {"body": parsed_body if isinstance(parsed_body, dict) else None}
    if status == 429 or (status is not None and status >= 500):
        rec["transport_failure"] = True

    # background polling (records status snapshots)
    poll = probe.get("poll")
    if poll and isinstance(parsed_body, dict) and parsed_body.get("id"):
        await _run_poll(client, adapter, poll, recordings, rec, headers, cfg)

    return rec


async def _run_poll(client, adapter, poll, recordings, rec, headers, cfg):
    def sub_ref(spec):
        did, path = spec.split("#", 1)
        root = rec["_root"] if did == "self" else recordings[did]["_root"]
        return get_path(root, path)

    resolve = make_resolver(sentinel_subber(adapter, dry=False), sub_ref)
    try:
        path = resolve(poll["path"])
    except Exception as e:
        rec.setdefault("response", {})["poll_error"] = f"resolve: {e}"
        return
    url = adapter.base_url + path
    snapshots = []
    if poll.get("cancel_first"):
        try:
            c = await client.request(
                "POST", url + "/cancel", headers=headers, timeout=cfg["timeout"]
            )
            snapshots.append(
                {"kind": "cancel", "status": c.status_code, "body": _safe_json(c.text)}
            )
        except Exception as e:
            snapshots.append({"kind": "cancel", "error": repr(e)})
    deadline = time.time() + poll.get("cap_s", 90)
    interval = poll.get("interval_s", 2)
    terminal = set(poll.get("until_status", []))
    while time.time() < deadline:
        try:
            g = await client.request("GET", url, headers=headers, timeout=cfg["timeout"])
        except Exception as e:
            snapshots.append({"kind": "get", "error": repr(e)})
            break
        gb = _safe_json(g.text)
        st = gb.get("status") if isinstance(gb, dict) else None
        snapshots.append(
            {"kind": "get", "status": g.status_code, "response_status": st, "body": gb}
        )
        if st in terminal:
            break
        await asyncio.sleep(interval)
    rec["response"]["poll_snapshots"] = snapshots


def _safe_json(text):
    try:
        return json.loads(text)
    except Exception:
        return text


# =============================================================================
# orchestration
# =============================================================================
def order_and_wire(probes):
    """Attach _waits (deps + placeholder refs) and topologically sort."""
    by_id = {p["id"]: p for p in probes}
    for p in probes:
        waits = set()
        if p["depends_on"]:
            waits.add(p["depends_on"])
        waits |= refs_in(p["endpoint"])
        waits |= refs_in(p["body"])
        waits |= refs_in(p["headers"])
        waits.discard(p["id"])
        p["_waits"] = sorted(w for w in waits if w in by_id)
    # topo sort (Kahn); stable by declaration order
    indeg = {p["id"]: len(p["_waits"]) for p in probes}
    ready = [p for p in probes if indeg[p["id"]] == 0]
    order = []
    dependents = {p["id"]: [] for p in probes}
    for p in probes:
        for w in p["_waits"]:
            dependents[w].append(p["id"])
    ready_ids = [p["id"] for p in ready]
    while ready_ids:
        pid = ready_ids.pop(0)
        order.append(pid)
        for d in dependents[pid]:
            indeg[d] -= 1
            if indeg[d] == 0:
                ready_ids.append(d)
    if len(order) != len(probes):
        cyc = [pid for pid, d in indeg.items() if d > 0]
        raise ValueError(f"dependency cycle among: {cyc}")
    return [by_id[i] for i in order], by_id


async def run_live(adapter, probes, cfg):
    if httpx is None:
        raise RuntimeError("httpx is required for live runs (pip install httpx)")
    ordered, by_id = order_and_wire(probes)
    recordings = {}
    events = {p["id"]: asyncio.Event() for p in probes}
    sem = asyncio.Semaphore(cfg["concurrency"])
    limits = httpx.Limits(max_connections=cfg["concurrency"] * 2)
    async with httpx.AsyncClient(limits=limits, follow_redirects=True) as client:

        async def run_one(probe):
            for dep in probe["_waits"]:
                await events[dep].wait()
            async with sem:
                try:
                    rec = await execute(client, adapter, probe, recordings, cfg)
                except Exception as e:  # never let one probe kill the run
                    rec = {
                        "probe_id": probe["id"],
                        "category": probe["category"],
                        "error": f"runner error: {e!r}",
                        "recorded": False,
                        "transport_failure": True,
                        "skipped": False,
                        "_root": {"body": None},
                    }
            recordings[probe["id"]] = rec
            events[probe["id"]].set()

        await asyncio.gather(*(run_one(by_id[p["id"]]) for p in ordered))
    return [recordings[p["id"]] for p in ordered]


# =============================================================================
# dry-run validation
# =============================================================================
def run_dry(adapter, probes):
    ordered, by_id = order_and_wire(probes)  # raises on cycle
    sub_sentinel = sentinel_subber(adapter, dry=True)
    problems = []
    resolved = []
    ids = set(by_id)
    for p in probes:
        # validate every referenced probe id exists
        for spec_src in (p["endpoint"], p["body"], p["headers"]):
            for pid in refs_in(spec_src):
                if pid not in ids:
                    problems.append(f"{p['id']}: unknown ref {pid}")
        sub_ref = lambda spec: f"<<{spec}>>"  # noqa: E731
        resolve = make_resolver(sub_sentinel, sub_ref)
        try:
            ep = resolve(p["endpoint"])
            body = resolve(p["body"]) if p["body"] is not None else None
            hdr = resolve(p["headers"]) if p["headers"] else None
            json.dumps({"endpoint": ep, "body": body, "headers": hdr})
        except Exception as e:
            problems.append(f"{p['id']}: resolve/serialize error: {e}")
            continue
        resolved.append(
            {
                "id": p["id"],
                "category": p["category"],
                "method": p["method"],
                "endpoint": ep,
                "stream": p["stream"],
                "waits": p["_waits"],
                "expect": p["expect"],
            }
        )
    return ordered, resolved, problems


# =============================================================================
# reporting
# =============================================================================
def price_for(model):
    if not model:
        return (0.0, 0.0)
    for sub, i, o in _PRICING:
        if sub in model:
            return (i, o)
    return (0.0, 0.0)


def build_summary(adapter, records):
    cats = {}
    sigs = {}  # category -> set of behavior signatures (recorded probes only)
    in_tok = out_tok = 0
    cost = 0.0
    n_recorded = n_transport = n_skipped = n_expected_err = 0
    for r in records:
        c = cats.setdefault(
            r["category"],
            {
                "total": 0,
                "recorded": 0,
                "transport_failures": 0,
                "skipped": 0,
                "expected_error_recorded": 0,
            },
        )
        c["total"] += 1
        if r.get("skipped"):
            c["skipped"] += 1
            n_skipped += 1
            continue
        if r.get("transport_failure"):
            c["transport_failures"] += 1
            n_transport += 1
        if r.get("recorded"):
            c["recorded"] += 1
            n_recorded += 1
            sigs.setdefault(r["category"], set()).add(fingerprint_sig(r, adapter.name))
        resp = r.get("response") or {}
        status = resp.get("status")
        if r.get("expect") == "error" and status and 400 <= status < 500:
            c["expected_error_recorded"] += 1
            n_expected_err += 1
        body = resp.get("body")
        if isinstance(body, dict):
            usage = body.get("usage") or {}
            it = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            ot = usage.get("output_tokens") or usage.get("completion_tokens") or 0
            model = body.get("model")
            pi, po = price_for(model)
            in_tok += it
            out_tok += ot
            cost += (it / 1e6) * pi + (ot / 1e6) * po
    total = len(records)
    ran = total - n_skipped
    # distinct behaviors per category: the volume-vs-signal metric — how many
    # fingerprint clusters the recorded probes collapse into
    for cat, c in cats.items():
        c["fingerprint_clusters"] = len(sigs.get(cat, ()))
    return {
        "provider": adapter.name,
        "total_probes": total,
        "recorded": n_recorded,
        "skipped": n_skipped,
        "transport_failures": n_transport,
        "expected_error_recorded": n_expected_err,
        "transport_failure_rate": round(n_transport / ran, 4) if ran else 0.0,
        "fingerprint_clusters_total": len(set().union(*sigs.values())) if sigs else 0,
        "tokens": {"input": in_tok, "output": out_tok},
        "estimated_cost_usd": round(cost, 4),
        "categories": cats,
    }


def write_outputs(out_dir: Path, adapter, records):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w") as f:
        for r in records:
            clean = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
    with (out_dir / "fingerprints.jsonl").open("w") as f:
        for r in records:
            resp = r.get("response") or {}
            fp = {
                "probe_id": r["probe_id"],
                "category": r.get("category"),
                "status": resp.get("status"),
                "skipped": r.get("skipped", False),
            }
            if "sse_events" in resp:
                fp["event_type_sequence"] = stream_event_names(resp["sse_events"], adapter.name)
                fp["event_count"] = len(resp["sse_events"])
            else:
                body = resp.get("body")
                fp["field_paths"] = (
                    sorted(field_paths(body)) if isinstance(body, (dict, list)) else []
                )
            f.write(json.dumps(fp, ensure_ascii=False) + "\n")
    summary = build_summary(adapter, records)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# =============================================================================
# CLI
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="Vendor ground-truth probe runner")
    ap.add_argument(
        "--provider", required=True, choices=["openai", "anthropic", "smg-openai", "smg-anthropic"]
    )
    ap.add_argument("--out", default=None, help="output dir (default results/<provider>)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--filter", default=None, help="regex over probe ids")
    ap.add_argument(
        "--tier",
        choices=["curated", "generated", "all"],
        default="curated",
        help="curated = hand-written matrix; generated = genmatrix tier; all = both",
    )
    ap.add_argument(
        "--budget", type=int, default=0, help="generated-tier probe budget (0 = unlimited)"
    )
    ap.add_argument(
        "--max-probes", type=int, default=0, help="cap total probes after filtering (0 = no cap)"
    )
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--stream-timeout", type=float, default=180.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--model", default=None, help="override @MODEL")
    ap.add_argument("--model-classic", default=None, help="override @MODEL_CLASSIC")
    args = ap.parse_args(argv)

    if args.model:
        os.environ["OPENAI_MODEL"] = args.model
        os.environ["ANTHROPIC_MODEL"] = args.model
    if args.model_classic:
        os.environ["OPENAI_MODEL_CLASSIC"] = args.model_classic

    adapter = build_adapter(args.provider)
    family = "openai" if adapter.name.endswith("openai") else "anthropic"
    probes = []
    if args.tier in ("curated", "all"):
        probes.extend(adapter.probes)
    if args.tier in ("generated", "all"):
        probes.extend(genmatrix.generate(family, budget=args.budget))
    ids = [p["id"] for p in probes]
    if len(set(ids)) != len(ids):
        seen, dups = set(), set()
        for i in ids:
            (dups if i in seen else seen).add(i)
        print(f"duplicate probe ids: {sorted(dups)[:10]}", file=sys.stderr)
        return 2
    if args.filter:
        rx = re.compile(args.filter)
        probes = [p for p in probes if rx.search(p["id"])]
    if args.max_probes > 0:
        probes = probes[: args.max_probes]
    if not probes:
        print("no probes matched", file=sys.stderr)
        return 2

    out_dir = Path(args.out or f"results/{args.provider}")

    if args.dry_run:
        ordered, resolved, problems = run_dry(adapter, probes)
        cats = {}
        for r in resolved:
            cats[r["category"]] = cats.get(r["category"], 0) + 1
        print(
            f"[dry-run] provider={adapter.name} probes={len(probes)} "
            f"resolved={len(resolved)} problems={len(problems)}"
        )
        print(f"[dry-run] categories: {json.dumps(cats)}")
        for r in resolved[-8:]:
            print(f"  {r['method']:4} {r['endpoint']}  <- {r['id']} (waits={r['waits']})")
        for prob in problems:
            print(f"  PROBLEM: {prob}", file=sys.stderr)
        return 1 if problems else 0

    cfg = {
        "concurrency": args.concurrency,
        "timeout": args.timeout,
        "stream_timeout": args.stream_timeout,
        "max_retries": args.max_retries,
    }
    records = asyncio.run(run_live(adapter, probes, cfg))
    summary = write_outputs(out_dir, adapter, records)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
