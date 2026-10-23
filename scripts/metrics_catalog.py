#!/usr/bin/env python3
"""Render the metrics catalog and the label reference from the sources.

The gateway, the library crates and the engine servicer describe every
Prometheus family where they register it (`describe_counter!`,
`describe_gauge!`, `describe_histogram!`, or the servicer's
`Registrar::describe_*`) and record it through the `counter!`, `gauge!` and
`histogram!` macros or a registrar handle. This script reads those sites and
renders:

* `docs/observability/metrics-catalog.md`: every family with its type, label
  keys, bucket edges (histograms), the recorder that emits it with the modules
  that call the recorder (the request point), and the HELP text;
* `docs/observability/metric-labels.md`: every label key with the families
  that carry it and the value sets the code fixes;
* the flag table between the `<!-- generated: observability-flags -->` markers
  of `docs/observability/probes-and-knobs.md`, from the gateway's clap
  definitions.

Nothing here is hand-maintained per family: a family added to the code shows
up on the next run, and `--check` fails when the checked-in pages are behind
the sources, so a CI job can hold the pages to the code.

Usage:
    python3 -I scripts/metrics_catalog.py            # rewrite the pages
    python3 -I scripts/metrics_catalog.py --check    # exit 1 when a page is stale
    python3 -I scripts/metrics_catalog.py --print    # the catalog on stdout
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOTS = ("model_gateway/src", "crates")
MAIN_RS = REPO / "model_gateway/src/main.rs"
DOCS = REPO / "docs/observability"
CATALOG = DOCS / "metrics-catalog.md"
LABELS = DOCS / "metric-labels.md"
KNOBS = DOCS / "probes-and-knobs.md"
FLAGS_BEGIN = "<!-- generated: observability-flags -->"
FLAGS_END = "<!-- /generated: observability-flags -->"
COMMAND = "python3 -I scripts/metrics_catalog.py"

# Families are grouped by name prefix; the second column says when a
# deployment has the group at all (the shape that creates the series).
GROUPS: list[tuple[str, str, str]] = [
    ("smg_http_", "HTTP front", "every gateway"),
    ("smg_router_", "Router request path", "every gateway; `token_type`-labelled series on gRPC backends only"),
    ("smg_admission_", "Admission", "every gateway (`--max-concurrent-requests` sets the token count)"),
    ("smg_scheduler_", "Priority scheduler", "gateways with the priority scheduler on"),
    ("smg_worker_", "Workers, health and circuit breaker", "every gateway with at least one worker"),
    ("smg_workers_", "Workers, health and circuit breaker", "every gateway with at least one worker"),
    ("smg_engine_", "Engine load reports", "workers that report load (gRPC `GetLoads` polls or pushed load records; `--engine-metrics` forces the polls)"),
    ("smg_kv_event_", "KV events", "gRPC workers whose engine publishes KV cache events"),
    ("smg_kv_events_", "KV events", "gRPC workers whose engine publishes KV cache events"),
    ("smg_kv_index_", "KV index", "cache-aware routing on a KV-event fleet"),
    ("smg_kv_role", "KV events", "gRPC workers whose engine publishes KV cache events"),
    ("smg_cache_", "Cache-aware policy", "the cache-aware policy"),
    ("smg_consistent_hashing_", "Other policies", "the named policy"),
    ("smg_manual_", "Other policies", "the named policy"),
    ("smg_prefix_hash_", "Other policies", "the named policy"),
    ("smg_policy_", "Other policies", "every gateway"),
    ("smg_routing_key_", "Other policies", "every gateway"),
    ("smg_pd_", "Prefill/decode", "prefill/decode (PD, EPD) deployments"),
    ("smg_symphony_", "Symphony tables", "gRPC backends reading tool calls and reasoning through a Symphony table"),
    ("smg_tokenizer_", "Tokenizer", "gRPC backends (the gateway tokenizes)"),
    ("smg_detokenize_", "Tokenizer", "gRPC backends (the gateway detokenizes)"),
    ("smg_mm_", "Multimodal", "gateways that process media (images, video, audio)"),
    ("smg_mcp_", "MCP tools", "requests that call MCP tools"),
    ("smg_history_", "Response storage", "the in-memory history backend"),
    ("smg_responses_", "Response storage", "the Responses API with a store"),
    ("smg_discovery_", "Service discovery", "`--service-discovery` providers"),
    ("smg_rl_", "RL control plane", "the RL control-plane routes"),
    ("smg_tokio_", "Runtime", "every gateway"),
    ("smg_allocator_", "Runtime", "builds with the jemalloc statistics feature (the release binaries)"),
    ("smg_log_", "Runtime", "every gateway"),
    ("smg_servicer_", "Engine servicer", "the Rust servicer with `--metrics-port` (its own `/metrics`, not the gateway's)"),
    ("router_mesh_", "Mesh", "`--mesh-*` high-availability peers"),
    ("router_rl_", "Mesh", "`--mesh-*` high-availability peers"),
    ("router_lb_", "Mesh", "`--mesh-*` high-availability peers"),
]
GROUP_ORDER = [
    "HTTP front", "Router request path", "Admission", "Priority scheduler",
    "Workers, health and circuit breaker", "Engine load reports", "KV events", "KV index",
    "Cache-aware policy", "Other policies", "Prefill/decode", "Symphony tables", "Tokenizer",
    "Multimodal", "MCP tools", "Response storage", "Service discovery", "RL control plane",
    "Runtime", "Engine servicer", "Mesh", "Other",
]

# What a label key means. Keys the code uses and this table does not know
# are rendered with an empty meaning, which the label page makes visible.
LABEL_MEANING: dict[str, str] = {
    "model": "the served model id as the gateway knows it; `other` once the bounded set of client-supplied names is full, `unknown` when the request named none",
    "worker": "the worker's URL (the registration address); series labelled this way leave `/metrics` with the worker",
    "worker_type": "`regular`, `prefill`, `decode`, `encode` (role), or `http`/`grpc` where the family counts by connection",
    "endpoint": "the API family of the request (`chat`, `completions`, `generate`, `responses`, `messages`, `embeddings`, `rerank`, `classify`, `realtime*`, `audio_transcriptions`, `decisions`, `systemone`, or `other`)",
    "router_type": "which router served the request: `http` (proxying), `grpc` (the tokenizing pipeline) or `openai` (external providers)",
    "backend_type": "`regular`, `pd` (prefill/decode), `external` or `harmony`",
    "connection_mode": "how the gateway reaches the worker: `http`, `grpc`, `zmq`, `websocket`, `webrtc`",
    "streaming": "`true`/`false`: the request asked for a stream",
    "token_type": "`input`/`output` (prompt vs generated tokens)",
    "error_type": "the router's failure class for the request",
    "error_code": "the API error code of the response body (`model_not_found`, `worker_overload_protection_shed`, ...), empty on success",
    "status_code": "the HTTP status of the response",
    "path": "the matched route template (not the raw URI), or the body path (`streamed`/`buffered`) on the body-path family",
    "method": "the HTTP method",
    "dp_rank": "the engine's data-parallel rank the report came from",
    "policy": "the routing policy that picked the worker",
    "class": "the priority class of the request",
    "victim_class": "the class preempted",
    "by_class": "the class that preempted it",
    "requested_class": "the class the request asked for",
    "effective_class": "the class it was clamped to",
    "tenant": "the tenant the request was keyed to",
    "stage": "the pipeline stage at which the event happened (`selection`, `dispatch`, ...)",
    "outcome": "how the operation ended (the family's HELP names the values)",
    "reason": "why it happened (the family's HELP names the values)",
    "result": "`success`, `error`, `timeout`, `not_found`, or `allowed`/`rejected` on the rate limiter, `hit`/`miss` on caches",
    "kind": "the sub-kind the family counts (the HELP names the values)",
    "table": "the Symphony table that read the output (a table name, `response_template`, `other`, or `none`)",
    "disposition": "what the gateway did with a KV event batch (`applied`, `stale`, `snapshot`, `load_only`)",
    "op": "the KV event operation (`stored`, `removed`, `snapshot`)",
    "layer": "the tokenizer cache layer (`L0` whole inputs, `L1` prefixes)",
    "index": "which KV index answered the lookup",
    "attempt": "the retry attempt number",
    "from": "the circuit-breaker state left",
    "to": "the circuit-breaker state entered",
    "gt": "the lower edge of an age bucket (seconds, exclusive)",
    "le": "the upper edge of an age bucket (seconds, inclusive)",
    "sink": "the log sink (`stdout`, `file`)",
    "mode": "the load-monitor tick decision, or the connector mode of a PD transfer",
    "branch": "the branch a policy took for the decision",
    "tree": "`string` or `token`: the cache-aware tree kind",
    "rank": "the servicer's engine rank the request ran on (`other` for a rank the launcher did not announce)",
    "wire": "the engine wire the servicer speaks (vllm, sglang, tokenspeed)",
    "finish_reason": "how the stream ended (`stop`, `length`, `abort`, ...)",
    "code": "the gRPC status code of the failure, or the HTTP status of a scrape",
    "state": "the one-hot lifecycle state",
    "family": "the metric family an observation was dropped from",
    "provider": "the discovery provider (`kubernetes`, `consul`, `file`, `slurm`, `static`, `manual`)",
    "source": "where the routing key came from",
    "role": "the KV-event role of the gateway",
    "type": "the worker type as registered",
    "transport": "the multimodal tensor transport",
    "format": "the media format",
    "codec": "the media codec",
    "operation": "the storage operation (`get`, `put`, `delete`, `list`)",
    "backend": "the storage backend",
    "status": "the outcome status",
    "connector": "the PD KV connector",
    "server": "the MCP server",
    "tool": "the MCP tool name (bounded; `other` past the cap)",
    "peer": "the mesh peer",
    "direction": "the direction of a transfer or sync",
    "phase": "the PD phase",
    "key": "the routing key",
    "tokenizer": "the tokenizer id",
    "priority": "the request priority",
    "error": "the gRPC status code the stream failed with (`out_of_range`, `unavailable`, `data_loss`, ...)",
    "leg": "the side of a prefill/decode pair whose rendezvous failed (the worker type of that side)",
    "runtime": "the engine runtime of the worker the request went to, as the worker registered it",
    "step": "which step of a store did not complete (`record`: nothing stored; `linking`: the record stored, its conversation link not)",
    "store": "the mesh replication store (one per replicated table)",
    "timeout": "the timeout that closed the connection (`header_read`, `idle`)",
    "tool_name": "the MCP tool invoked (bounded; `other` past the cap)",
}

# The label keys whose values the code fixes as constants, by the prefix of
# the constant's name in the shared `metrics_labels` module.
VALUE_SETS: dict[str, str] = {
    "ROUTER_": "router_type",
    "BACKEND_": "backend_type",
    "CONNECTION_": "connection_mode",
    "ENDPOINT_": "endpoint",
    "WORKER_": "worker_type",
    "TOKEN_": "token_type",
    "STREAMING_": "streaming",
    "RESULT_": "result",
    "KV_CONNECTOR_": "mode (PD KV connector)",
    "DISCOVERY_": "provider",
    "DB_OP_": "operation",
    "STORAGE_": "kind (storage)",
    "MODEL_": "model (sentinel)",
}

# Recorder names too generic to attribute callers to by name alone.
GENERIC_FN = {
    "new", "default", "drop", "set", "get", "record", "observe", "render", "call", "from",
    "into", "update", "run", "start", "stop", "clear", "tick", "poll", "apply", "build",
    "with", "inner", "handle", "init", "install", "reset", "sample", "flush", "retire",
    "register", "remove", "insert", "len", "name", "labels", "key", "counter", "gauge",
    "histogram", "rank", "describe", "describe_counter", "describe_gauge", "describe_histogram",
}

# Zero-series registrations: a family emitted only here is published at start
# so dashboards find the series before the first event.
STARTUP_FNS = {"init_startup_series", "init_series", "init_worker_series", "init_zero_series"}


@dataclass
class Source:
    path: Path
    rel: str
    text: str  # test modules and full-line comments removed
    consts: dict[str, str] = field(default_factory=dict)


@dataclass
class Family:
    name: str
    kind: str = ""  # counter | gauge | histogram
    help: str = ""
    described_at: list[str] = field(default_factory=list)
    labels: set[str] = field(default_factory=set)
    recorders: dict[tuple[str, str], None] = field(default_factory=dict)  # (fn, rel) in order
    startup_only: bool = True
    buckets: str = ""


# ----------------------------------------------------------------------------
# Source reading


def is_test_file(path: Path) -> bool:
    name = path.name
    return (
        "tests" in path.parts
        or "benches" in path.parts
        or "target" in path.parts
        or name == "tests.rs"
        or name.endswith("_test.rs")
        or name.endswith("_tests.rs")
    )


TEST_MOD_RE = re.compile(r"(?m)^[ \t]*#\[cfg\(test\)\]\s*\n[ \t]*(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+\s*(\{|;)")


def strip_tests_and_comments(text: str) -> str:
    """The code without its `#[cfg(test)]` modules and full-line comments.

    Every `#[cfg(test)] mod x { … }` block goes, wherever it sits in the file
    (the brace-balanced span), as does a `#[cfg(test)] mod x;` line.
    """
    while True:
        m = TEST_MOD_RE.search(text)
        if not m:
            break
        if m.group(1) == ";":
            text = text[: m.start()] + text[m.end():]
        else:
            text = text[: m.start()] + text[balanced(text, m.end() - 1):]
    return "\n".join("" if re.match(r"\s*//", line) else line for line in text.split("\n"))


CONST_RE = re.compile(
    r"(?:pub(?:\([^)]*\))?\s+)?const\s+([A-Z][A-Z0-9_]*)\s*:\s*&(?:'static\s+)?str\s*=\s*\"([^\"]+)\""
)


def read_sources() -> list[Source]:
    sources = []
    for root in SOURCE_ROOTS:
        for path in sorted((REPO / root).rglob("*.rs")):
            if is_test_file(path):
                continue
            raw = path.read_text(encoding="utf-8", errors="replace")
            text = strip_tests_and_comments(raw)
            rel = path.relative_to(REPO).as_posix()
            consts = {m.group(1): m.group(2) for m in CONST_RE.finditer(text)}
            sources.append(Source(path, rel, text, consts))
    if len(sources) < 100:
        sys.exit(f"only {len(sources)} Rust sources under {SOURCE_ROOTS}; the tree moved?")
    return sources


# ----------------------------------------------------------------------------
# Rust snippets: balanced parentheses, argument lists, string literals


def balanced(text: str, open_at: int) -> int:
    """Index just past the parenthesis group that opens at `open_at`."""
    depth = 0
    i = open_at
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"':
            i = skip_string(text, i)
            continue
        if c == "'" and i + 2 < n and text[i + 2] == "'":
            i += 3
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def skip_string(text: str, start: int) -> int:
    """Index just past the string literal that opens at `start`."""
    i = start + 1
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == '"':
            return i + 1
        i += 1
    return n


def split_args(inner: str) -> list[str]:
    args = []
    depth = 0
    cur = []
    i = 0
    n = len(inner)
    while i < n:
        c = inner[i]
        if c == '"':
            j = skip_string(inner, i)
            cur.append(inner[i:j])
            i = j
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    tail = "".join(cur).strip()
    if tail:
        args.append(tail)
    return args


def literal(arg: str) -> str | None:
    """The value of a Rust string literal argument, continuations folded."""
    arg = arg.strip()
    if not arg.startswith('"') or not arg.endswith('"'):
        return None
    body = arg[1:-1]
    body = re.sub(r"\\\n\s*", "", body)
    body = body.replace('\\"', '"').replace("\\\\", "\\").replace("\\n", " ")
    return re.sub(r"\s+", " ", body).strip()


def resolve_name(expr: str, source: Source, all_consts: dict[str, list[tuple[str, str]]]) -> str | None:
    """A family name from a literal or a constant path."""
    expr = expr.strip()
    lit = literal(expr)
    if lit is not None:
        return lit
    m = re.fullmatch(r"(?:[A-Za-z_][A-Za-z0-9_]*::)*([A-Z][A-Z0-9_]*)(?:\.into\(\))?", expr)
    if not m:
        m = re.fullmatch(r"String::from\(((?:[A-Za-z_][A-Za-z0-9_]*::)*[A-Z][A-Z0-9_]*)\)", expr)
        if not m:
            return None
        expr = m.group(1)
        m = re.fullmatch(r"(?:[A-Za-z_][A-Za-z0-9_]*::)*([A-Z][A-Z0-9_]*)", expr)
    ident = m.group(1)
    if ident in source.consts:
        return source.consts[ident]
    candidates = all_consts.get(ident, [])
    if len(candidates) == 1:
        return candidates[0][1]
    hint = expr.split("::")[-2] if "::" in expr else ""
    for rel, value in candidates:
        if hint and hint in rel:
            return value
    return candidates[0][1] if candidates else None


FN_RE = re.compile(r"(?m)^[ \t]*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:const\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)")


def enclosing_fn(text: str, at: int) -> str:
    last = ""
    for m in FN_RE.finditer(text, 0, at):
        last = m.group(1)
    return last


def enclosing_fn_span(text: str, at: int) -> tuple[int, int]:
    """The text span of the function that contains `at` (to its closing brace, roughly)."""
    start = 0
    for m in FN_RE.finditer(text, 0, at):
        start = m.start()
    brace = text.find("{", start)
    if brace < 0:
        return start, len(text)
    return start, balanced(text, brace)


def line_of(text: str, at: int) -> int:
    return text.count("\n", 0, at) + 1


# ----------------------------------------------------------------------------
# Describe sites

DESCRIBE_RE = re.compile(r"\bdescribe_(counter|gauge|histogram)!?\s*\(")
EMIT_MACRO_RE = re.compile(r"(?<![A-Za-z0-9_])(counter|gauge|histogram)!\s*\(")
HANDLE_CALL_RE = re.compile(r"\.(counter|gauge|histogram)\(\s*(names::[A-Z][A-Z0-9_]*)\s*,")
KEY_BUILDER_RE = re.compile(r"\bkey\(\s*([A-Z][A-Z0-9_]*)\s*,\s*&\[(.*?)\]\s*\)", re.S)
LABEL_PAIR_RE = re.compile(r"\"([a-z][a-z0-9_]*)\"\s*=>")
TUPLE_KEY_RE = re.compile(r"\(\s*\"([a-z][a-z0-9_]*)\"\s*,")
LABEL_NEW_RE = re.compile(r"Label::(?:new|from_static_parts)\(\s*\"([a-z][a-z0-9_]*)\"")
GAUGE_HIST_RE = re.compile(r"GaugeHistogramVec::new\(\s*\"([a-z][a-z0-9_]+)\"")


def collect(sources: list[Source]) -> tuple[dict[str, Family], list[str]]:
    all_consts: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for s in sources:
        for k, v in s.consts.items():
            all_consts[k].append((s.rel, v))
    families: dict[str, Family] = {}
    warnings: list[str] = []

    def fam(name: str) -> Family:
        return families.setdefault(name, Family(name))

    for s in sources:
        text = s.text
        for m in DESCRIBE_RE.finditer(text):
            end = balanced(text, m.end() - 1)
            args = split_args(text[m.end(): end - 1])
            if not args or args[0].startswith("&self") or "KeyName::" in args[0] or args[0] in ("key", "name"):
                continue  # the definition or the inner call of a describe method, not a site
            name = resolve_name(args[0], s, all_consts)
            if name is None:
                warnings.append(f"{s.rel}:{line_of(text, m.start())}: describe with an unresolved name {args[0]!r}")
                continue
            help_text = None
            for arg in reversed(args):
                help_text = literal(arg)
                if help_text is not None:
                    break
            f = fam(name)
            f.kind = m.group(1)
            f.help = help_text or ""
            f.described_at.append(f"{s.rel}:{line_of(text, m.start())}")

        for m in EMIT_MACRO_RE.finditer(text):
            if text[max(0, m.start() - 9): m.start()].endswith("describe_"):
                continue
            end = balanced(text, m.end() - 1)
            inner = text[m.end(): end - 1]
            args = split_args(inner)
            if not args:
                continue
            name = resolve_name(args[0], s, all_consts)
            if name is None:
                warnings.append(f"{s.rel}:{line_of(text, m.start())}: {m.group(1)}! with a dynamic name {args[0][:40]!r}")
                continue
            f = fam(name)
            keys = set(LABEL_PAIR_RE.findall(inner))
            if len(args) > 1 and not keys:
                # labels passed as a value: the keys the enclosing function builds
                a, b = enclosing_fn_span(text, m.start())
                keys = set(LABEL_NEW_RE.findall(text[a:b])) | set(TUPLE_KEY_RE.findall(text[a:b]))
            f.labels |= keys
            fn = enclosing_fn(text, m.start())
            f.recorders.setdefault((fn, s.rel), None)
            if fn not in STARTUP_FNS:
                f.startup_only = False

        # Servicer-style registrar handles: `.counter(names::X, &[("k", v)])`
        # with the registrar's fixed labels.
        if HANDLE_CALL_RE.search(text):
            fixed: set[str] = set()
            lab = re.search(r"fn labels\(&self[^{]*\{", text)
            if lab:
                a, b = lab.start(), balanced(text, text.find("{", lab.start()))
                fixed = set(LABEL_NEW_RE.findall(text[a:b]))
            for m in HANDLE_CALL_RE.finditer(text):
                name = resolve_name(m.group(2), s, all_consts)
                if name is None:
                    continue
                end = balanced(text, text.rfind("(", 0, m.end()))
                rest = text[m.end(): end - 1]
                keys = set(TUPLE_KEY_RE.findall(rest))
                ident = re.fullmatch(r"\s*([a-z_]+)\s*", rest)
                if not keys and ident:
                    # `labels` bound in the enclosing function: its own tuples only
                    a, b = enclosing_fn_span(text, m.start())
                    binding = re.search(r"let\s+" + ident.group(1) + r"\b[^=]*=\s*&\[(.*?)\];", text[a:b], re.S)
                    if binding:
                        keys = set(TUPLE_KEY_RE.findall(binding.group(1)))
                f = fam(name)
                f.labels |= keys | fixed
                fn = enclosing_fn(text, m.start())
                f.recorders.setdefault((fn, s.rel), None)
                f.startup_only = False

        # Key builders (`Key::from_parts(name, all)`): `key(CONST, &[("k", v)])`
        # with the fixed labels of the builder.
        if "Key::from_parts(" in text:
            at = text.find("Key::from_parts(")
            a, b = enclosing_fn_span(text, at)
            fixed = set(LABEL_NEW_RE.findall(text[a:b]))
            for m in KEY_BUILDER_RE.finditer(text):
                name = resolve_name(m.group(1), s, all_consts)
                if name is None:
                    continue
                f = fam(name)
                f.labels |= set(TUPLE_KEY_RE.findall(m.group(2))) | fixed
                fn = enclosing_fn(text, m.start())
                f.recorders.setdefault((fn, s.rel), None)
                f.startup_only = False

        # Gauge histograms: the family named at construction, the bucket
        # labels added by the type.
        for m in GAUGE_HIST_RE.finditer(text):
            f = fam(m.group(1))
            f.recorders.setdefault((enclosing_fn(text, m.start()) or "static", s.rel), None)
            f.startup_only = False
            for other in sources:
                if "struct GaugeHistogramVec" in other.text:
                    f.labels |= set(LABEL_NEW_RE.findall(other.text))

    return families, warnings


# ----------------------------------------------------------------------------
# Callers of the recorders: the request point


def short(rel: str) -> str:
    if rel.startswith("model_gateway/src/"):
        return rel[len("model_gateway/src/"):]
    m = re.match(r"crates/([^/]+)/src/(.*)", rel)
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    return rel


def is_metrics_module(rel: str) -> bool:
    return rel.endswith("/metrics.rs") or "/observability/" in rel


def find_callers(families: dict[str, Family], sources: list[Source]) -> dict[str, list[str]]:
    names = sorted({fn for f in families.values() for fn, _ in f.recorders if fn and fn not in GENERIC_FN})
    callers: dict[str, set[str]] = defaultdict(set)
    if not names:
        return {}
    pattern = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(map(re.escape, names)) + r")\s*\(")
    defn = re.compile(r"\bfn\s+(" + "|".join(map(re.escape, names)) + r")\b")
    # A recorder name defined in several files outside the metrics modules
    # (an `error`, a `new`) cannot be attributed by name: its callers are
    # skipped. A single outside definition is a same-named wrapper: its own
    # file is skipped, its callers count.
    definers: dict[str, set[str]] = defaultdict(set)
    for s in sources:
        if not is_metrics_module(s.rel):
            for name in set(defn.findall(s.text)):
                definers[name].add(s.rel)
    ambiguous = {name for name, files in definers.items() if len(files) > 1}
    for s in sources:
        if is_metrics_module(s.rel):
            continue
        for m in pattern.finditer(s.text):
            name = m.group(1)
            if name in ambiguous or s.rel in definers.get(name, ()):
                continue
            callers[name].add(short(s.rel))
    return {k: sorted(v) for k, v in callers.items()}


# ----------------------------------------------------------------------------
# Histogram buckets

NUM_RE = re.compile(r"-?\d[\d_]*(?:\.[\d_]+)?(?:e-?\d+)?")


def numbers(expr: str) -> list[float]:
    return [float(x.replace("_", "")) for x in NUM_RE.findall(expr)]


def fmt_num(x: float) -> str:
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    s = f"{x:.10f}".rstrip("0").rstrip(".")
    return s


def edges_summary(edges: list[float]) -> str:
    if not edges:
        return ""
    if len(edges) <= 6:
        return ", ".join(fmt_num(e) for e in edges)
    return f"{fmt_num(edges[0])}, {fmt_num(edges[1])} … {fmt_num(edges[-2])}, {fmt_num(edges[-1])} ({len(edges)} edges)"


def bucket_sets(sources: list[Source]) -> tuple[dict[str, list[float]], dict[str, str], dict[str, str]]:
    """Named edge sets, exact-name assignments and suffix assignments."""
    sets: dict[str, list[float]] = {}
    const_re = re.compile(r"const\s+([A-Z][A-Z0-9_]*BUCKETS)\s*:\s*&\[f64\]\s*=\s*&\[(.*?)\];", re.S)
    for s in sources:
        for m in const_re.finditer(s.text):
            sets[m.group(1)] = numbers(m.group(2))
    full: dict[str, str] = {}
    suffix: dict[str, str] = {}
    for s in sources:
        text = s.text
        # The gateway's override table.
        ov = re.search(r"fn histogram_bucket_overrides\b", text)
        if ov:
            a, b = ov.start(), balanced(text, text.find("{", ov.start()))
            body = text[a:b]
            default = re.search(r"unwrap_or_else\(\|\|\s*\{\s*vec!\[(.*?)\]", body, re.S)
            if default:
                sets["REQUEST_DURATION_BUCKETS (default of --prometheus-duration-buckets)"] = numbers(default.group(1))
            matchers: dict[str, tuple[str, str]] = {}
            for m in re.finditer(r"let\s+(\w+)\s*=\s*Matcher::(Full|Suffix)\((.*?)\);", body, re.S):
                arg = m.group(3).strip()
                lit = re.search(r"String::from\(\"([^\"]+)\"\)", arg)
                if lit:
                    matchers[m.group(1)] = (m.group(2), lit.group(1))
                else:
                    consts = {k: v for src in sources for k, v in src.consts.items()}
                    name = resolve_name(arg, s, {k: [("", v)] for k, v in consts.items()})
                    if name:
                        matchers[m.group(1)] = (m.group(2), name)
            for m in re.finditer(r"\(\s*(\w+)\s*,\s*([A-Za-z_:\.\(\)]+?)(?:\.to_vec\(\))?(?:\.clone\(\))?\s*,?\s*\)", body):
                var, expr = m.group(1), m.group(2)
                if var not in matchers:
                    continue
                set_name = expr.split("::")[-1].replace(".clone()", "")
                if set_name == "duration_bucket":
                    set_name = "REQUEST_DURATION_BUCKETS (default of --prometheus-duration-buckets)"
                mode, target = matchers[var]
                (full if mode == "Full" else suffix)[target] = set_name
        # The servicer's per-family table.
        for m in re.finditer(r"\(\s*(names::[A-Z][A-Z0-9_]*)\s*,\s*(&?[A-Za-z_]+)\s*,?\s*\)", text):
            consts = {k: [("", v)] for src in sources for k, v in src.consts.items()}
            name = resolve_name(m.group(1), s, consts)
            if name and name.startswith("smg_servicer_"):
                full[name] = m.group(2).lstrip("&")
    return sets, full, suffix


def assign_buckets(families: dict[str, Family], sets: dict[str, list[float]], full: dict[str, str], suffix: dict[str, str]) -> None:
    for f in families.values():
        if f.kind != "histogram":
            continue
        set_name = full.get(f.name)
        if set_name is None:
            for suf, name in suffix.items():
                if f.name.endswith(suf):
                    set_name = name
                    break
        if set_name is None:
            f.buckets = "recorder default (rendered as a summary with quantiles)"
        elif set_name == "token_buckets":
            f.buckets = "1-2-5 series up to the model's context length"
        elif set_name in sets:
            f.buckets = f"{edges_summary(sets[set_name])}: `{set_name.split(' ')[0]}`"
        else:
            f.buckets = f"`{set_name}`"


# ----------------------------------------------------------------------------
# Label value sets


def value_sets(sources: list[Source]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for s in sources:
        if "metrics_labels" not in s.text or not s.rel.endswith("external_router/src/metrics.rs"):
            continue
        m = re.search(r"pub mod metrics_labels\s*\{", s.text)
        if not m:
            continue
        body = s.text[m.start(): balanced(s.text, s.text.find("{", m.start()))]
        for c in CONST_RE.finditer(body):
            for prefix, key in VALUE_SETS.items():
                if c.group(1).startswith(prefix):
                    if c.group(2) not in out[key]:
                        out[key].append(c.group(2))
                    break
    return out


# ----------------------------------------------------------------------------
# The gateway's flags (clap) for the probes/knobs page

FLAG_HEADINGS = {"Logging", "Prometheus Metrics", "Tracing (OpenTelemetry)", "Health Checks", "Profiling"}
FLAG_FIELDS = {"health_check_port", "admin_port", "admin_host", "engine_metrics", "log_mm_timing", "shutdown_grace_period_secs"}


def flags_table() -> str:
    text = MAIN_RS.read_text(encoding="utf-8")
    rows = []
    pattern = re.compile(
        r"((?:[ \t]*///[^\n]*\n)+)[ \t]*#\[arg\((.*?)\)\]\s*\n(?:[ \t]*#\[arg\((.*?)\)\]\s*\n)?[ \t]*(?:pub(?:\([^)]*\))?\s+)?([a-z_][a-z0-9_]*)\s*:",
        re.S,
    )
    for m in pattern.finditer(text):
        doc, attrs, attrs2, fieldname = m.groups()
        attrs = attrs + ("," + attrs2 if attrs2 else "")
        heading = re.search(r"help_heading\s*=\s*\"([^\"]+)\"", attrs)
        heading = heading.group(1) if heading else ""
        if heading not in FLAG_HEADINGS and fieldname not in FLAG_FIELDS:
            continue
        long = re.search(r"long\s*=\s*\"([^\"]+)\"", attrs)
        flag = "--" + (long.group(1) if long else fieldname.replace("_", "-"))
        default = re.search(r"default_value(?:_t)?\s*=\s*([^,)]+)", attrs)
        default_text = default.group(1).strip().strip('"') if default else ""
        if "SetTrue" in attrs and not default_text:
            default_text = "false"
        docs = " ".join(line.strip()[3:].strip() for line in doc.strip().split("\n"))
        docs = docs.replace("|", "\\|")
        rows.append((heading or "Worker Configuration", flag, default_text or "unset", docs))
    rows.sort(key=lambda r: (list(FLAG_HEADINGS | {"Worker Configuration"}).index(r[0]) if False else r[0], r[1]))
    out = ["| Flag | Default | Help heading | What it does |", "|---|---|---|---|"]
    for heading, flag, default, docs in rows:
        out.append(f"| `{flag}` | `{default}` | {heading} | {docs} |")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Rendering


def group_of(name: str) -> tuple[str, str]:
    for prefix, group, when in GROUPS:
        if name.startswith(prefix):
            return group, when
    return "Other", ""


def md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_catalog(families: dict[str, Family], callers: dict[str, list[str]], sets: dict[str, list[float]]) -> str:
    by_group: dict[str, list[Family]] = defaultdict(list)
    for f in families.values():
        by_group[group_of(f.name)[0]].append(f)
    described = [f for f in families.values() if f.kind]
    histograms = sum(1 for f in described if f.kind == "histogram")
    counters = sum(1 for f in described if f.kind == "counter")
    gauges = sum(1 for f in described if f.kind == "gauge")
    undescribed = sorted(f.name for f in families.values() if not f.kind)
    unrecorded = sorted(f.name for f in families.values() if f.kind and not f.recorders)
    lines = [
        "# Metrics catalog",
        "",
        f"Generated by `{COMMAND}` from the registration and recording sites in the sources; do not edit by hand (`--check` holds this page to the code).",
        "",
        f"{len(described)} families: {counters} counters, {gauges} gauges, {histograms} histograms. "
        "A family is on `/metrics` once something recorded it, except the pre-registered ones (marked *zero at start*), which are published at process start so dashboards and alerts find the series before the first event.",
        "",
        "How to read a row:",
        "",
        "- **Labels**: the label keys the recorder sets; the [label reference](metric-labels.md) says what each key means and which values are fixed.",
        "- **Buckets** (histograms): the edge set the exporter uses; a family without one renders as a summary (quantile lines). The sets are listed at the end.",
        "- **Recorded by**: the recorder function and the modules that call it (the request point); `gateway` paths are under `model_gateway/src/`, `crate:path` under `crates/<crate>/src/`.",
        "- **Help**: the `# HELP` text of the scrape.",
        "",
        "Groups and when a deployment has them:",
        "",
        "| Group | Families | Present when |",
        "|---|---|---|",
    ]
    seen_when = {}
    for prefix, group, when in GROUPS:
        seen_when.setdefault(group, when)
    for group in GROUP_ORDER:
        fams = by_group.get(group)
        if not fams:
            continue
        lines.append(f"| [{group}](#{anchor(group)}) | {len(fams)} | {seen_when.get(group, '')} |")
    lines.append("")
    for group in GROUP_ORDER:
        fams = by_group.get(group)
        if not fams:
            continue
        lines += [f"## {group}", ""]
        has_hist = any(f.kind == "histogram" for f in fams)
        header = "| Family | Type | Labels |" + (" Buckets |" if has_hist else "") + " Recorded by | Help |"
        sep = "|---|---|---|" + ("---|" if has_hist else "") + "---|---|"
        lines += [header, sep]
        for f in sorted(fams, key=lambda x: x.name):
            labels = ", ".join(f"`{k}`" for k in sorted(f.labels)) or "—"
            recorded = render_recorders(f, callers)
            kind = f.kind or "*(not described)*"
            if f.startup_only and f.recorders:
                kind += ", zero at start"
            elif any(fn in STARTUP_FNS for fn, _ in f.recorders):
                kind += " (zero at start)"
            row = f"| `{f.name}` | {kind} | {labels} |"
            if has_hist:
                row += f" {md_cell(f.buckets) if f.kind == 'histogram' else ''} |"
            row += f" {md_cell(recorded)} | {md_cell(f.help)} |"
            lines.append(row)
        lines.append("")
    if undescribed:
        lines += ["## Families recorded without a description", "", "These names are recorded but no `describe_*` site gives them a HELP text; the scrape shows them without one.", ""]
        lines += [f"- `{n}`" for n in undescribed] + [""]
    if unrecorded:
        lines += ["## Families described but never recorded", "", "Described names with no recording site this script can see (a dynamic name, or a description that outlived its recorder).", ""]
        lines += [f"- `{n}`" for n in unrecorded] + [""]
    lines += ["## Bucket edge sets", "", "| Set | Edges (seconds unless the family says otherwise) |", "|---|---|"]
    for name in sorted(sets):
        lines.append(f"| `{name.split(' ')[0]}` | {', '.join(fmt_num(e) for e in sets[name])} |")
    lines.append("")
    return "\n".join(lines)


def anchor(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def render_recorders(f: Family, callers: dict[str, list[str]]) -> str:
    parts = []
    for fn, rel in f.recorders:
        if fn in STARTUP_FNS:
            continue
        where = short(rel)
        calls = callers.get(fn, [])
        if fn in GENERIC_FN or not fn or fn == "static":
            parts.append(f"`{where}`")
            continue
        text = f"`{fn}` ({where})"
        if calls:
            shown = calls[:3]
            more = len(calls) - len(shown)
            text += " ← " + ", ".join(f"`{c}`" for c in shown) + (f" +{more}" if more > 0 else "")
        parts.append(text)
    seen = []
    for p in parts:
        if p not in seen:
            seen.append(p)
    return "; ".join(seen) if seen else "pre-registered only"


def render_labels(families: dict[str, Family], values: dict[str, list[str]]) -> str:
    by_key: dict[str, list[str]] = defaultdict(list)
    for f in families.values():
        for k in f.labels:
            by_key[k].append(f.name)
    lines = [
        "# Metric labels",
        "",
        f"Generated by `{COMMAND}`; do not edit by hand.",
        "",
        "Every label key a family carries, what it means, and the families that carry it. "
        "Label values that identify a worker are its registration URL, so per-worker series come and go with the fleet (the gateway retires a worker's series when the worker is removed). "
        "Client-controlled values (`model`, MCP `tool`) are bounded: past the cap they collapse to `other`.",
        "",
        "| Key | Meaning | Families |",
        "|---|---|---|",
    ]
    for key in sorted(by_key):
        fams = sorted(by_key[key])
        shown = ", ".join(f"`{n}`" for n in fams[:6])
        if len(fams) > 6:
            shown += f" and {len(fams) - 6} more"
        lines.append(f"| `{key}` | {md_cell(LABEL_MEANING.get(key, ''))} | {len(fams)}: {shown} |")
    lines += ["", "## Fixed value sets", "", "Values the code fixes as constants (the shared label module of the router crate); a value outside the set is a bug, not a new category.", "", "| Key | Values |", "|---|---|"]
    for key in sorted(values):
        lines.append(f"| `{key}` | {', '.join(f'`{v}`' for v in values[key])} |")
    lines.append("")
    lines += [
        "## Families by label key count",
        "",
        "| Labels | Families |",
        "|---|---|",
    ]
    counts: dict[int, int] = defaultdict(int)
    for f in families.values():
        counts[len(f.labels)] += 1
    for n in sorted(counts):
        lines.append(f"| {n} | {counts[n]} |")
    lines.append("")
    return "\n".join(lines)


def splice_flags(page: str, table: str) -> str:
    a = page.find(FLAGS_BEGIN)
    b = page.find(FLAGS_END)
    if a < 0 or b < 0 or b < a:
        sys.exit(f"{KNOBS}: the {FLAGS_BEGIN} … {FLAGS_END} markers are missing")
    return page[: a + len(FLAGS_BEGIN)] + "\n" + table + "\n" + page[b:]


# ----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="exit 1 when a page differs from a fresh render")
    parser.add_argument("--print", action="store_true", help="print the catalog instead of writing")
    parser.add_argument("--quiet", action="store_true", help="no warnings on stderr")
    args = parser.parse_args()

    sources = read_sources()
    families, warnings = collect(sources)
    callers = find_callers(families, sources)
    sets, full, suffix = bucket_sets(sources)
    assign_buckets(families, sets, full, suffix)
    values = value_sets(sources)

    catalog = render_catalog(families, callers, sets)
    labels = render_labels(families, values)
    knobs_page = KNOBS.read_text(encoding="utf-8") if KNOBS.exists() else ""
    knobs = splice_flags(knobs_page, flags_table()) if knobs_page else ""

    if not args.quiet:
        for w in warnings:
            print(f"note: {w}", file=sys.stderr)
        print(f"{sum(1 for f in families.values() if f.kind)} described families, {len(families)} named, from {len(sources)} sources", file=sys.stderr)

    if args.print:
        sys.stdout.write(catalog)
        return 0

    outputs = {CATALOG: catalog, LABELS: labels}
    if knobs:
        outputs[KNOBS] = knobs
    if args.check:
        stale = 0
        for path, text in outputs.items():
            current = path.read_text(encoding="utf-8") if path.exists() else ""
            if current != text:
                stale += 1
                print(f"stale: {path.relative_to(REPO)} (run `{COMMAND}`)", file=sys.stderr)
                diff = difflib.unified_diff(current.splitlines(), text.splitlines(), "checked in", "fresh render", lineterm="", n=1)
                for i, line in enumerate(diff):
                    if i >= 40:
                        print("  …", file=sys.stderr)
                        break
                    print(f"  {line}", file=sys.stderr)
        return 1 if stale else 0
    DOCS.mkdir(parents=True, exist_ok=True)
    for path, text in outputs.items():
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(REPO)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
