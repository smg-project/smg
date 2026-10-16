#!/usr/bin/env python3
"""Fold a `perf script` dump (recorded with -g) of the gateway into the
tables a clone/allocation hunt needs: the top self symbols, the share of
cycles inside the allocator, memcpy/memmove and Clone impls, and the sites
behind them (the first frame outside the allocator or libc, with its caller).
Usage: perf script -i perf.data | perf_sites.py [--top 20] [--json out]"""

import argparse
import json
import re
import sys

FRAME = re.compile(r"^\s+([0-9a-f]+)\s+(.*?)(?:\+0x[0-9a-f]+)?\s+\((.*)\)\s*$")
# jemalloc's own symbols (the vendored, prefixed build is statically linked) and the Rust shims above it.
ALLOC_LEAF = re.compile(
    r"^(_rjem_|je_|tcache|arena_|extent_|rtree_|sz_|do_rallocx|imalloc|ifree|large_|pages_|prof_|"
    r"malloc_mutex|base_|emap_|edata_|eset_|pac_|pa_|hpa_|sec_|bin_|ecache_|san_|tsd_|psset_|"
    r"malloc$|free$|calloc$|realloc$|posix_memalign$|aligned_alloc$)"
)
ALLOC = re.compile(
    ALLOC_LEAF.pattern + r"|__rust_(alloc|dealloc|realloc)|^alloc::alloc::|alloc::raw_vec::|RawVec|"
    r"<alloc::alloc::Global|box_new_uninit|finish_grow|__rjem"
)
COPY = re.compile(r"^(__)?mem(cpy|move|set)|copy_nonoverlapping|copy_from_slice")
ATOMIC = re.compile(r"^__aarch64_(ldadd|cas|swp|ldclr|ldeor|ldset)|^__sync_|^__atomic_")
CLONE = re.compile(
    r"as core::clone::Clone>::clone|::clone$|::to_owned|::to_vec|::to_string|"
    r"<str as alloc::string::ToString>|String::from|::into_owned"
)
LIBC = re.compile(r"^(libc|ld-linux|\[kernel|\[unknown|\[vdso)")
# Where the self cycles are, by the leaf frame: every sample lands in exactly one bucket (first match wins).
SELF_BUCKETS = [
    ("kernel", lambda sym, dso: dso.startswith("[kernel")),
    ("allocator (jemalloc)", lambda sym, dso: bool(ALLOC_LEAF.search(sym))),
    ("memcpy/memmove/memset", lambda sym, dso: bool(COPY.search(sym))),
    ("atomics (outlined LSE helpers)", lambda sym, dso: bool(ATOMIC.search(sym))),
    ("libc other", lambda sym, dso: dso.startswith("libc") or dso.endswith("libc.so.6")),
    ("tokio/mio", lambda sym, dso: sym.startswith(("tokio::", "<tokio::", "mio::", "<mio::"))),
    (
        "tokenizer",
        lambda sym, dso: (
            "llm_tokenizer" in sym
            or "tokenizers::" in sym
            or "fancy_regex" in sym
            or "regex_automata" in sym
        ),
    ),
    (
        "serde/serde_json",
        lambda sym, dso: "serde_json" in sym or "serde_core" in sym or "serde::" in sym,
    ),
    (
        "hyper/h2/http/tower/axum",
        lambda sym, dso: sym.startswith(
            (
                "hyper",
                "<hyper",
                "h2::",
                "<h2::",
                "http::",
                "<http::",
                "tower",
                "<tower",
                "axum",
                "<axum",
            )
        ),
    ),
    ("tonic/prost", lambda sym, dso: sym.startswith(("tonic", "<tonic", "prost", "<prost"))),
    (
        "kv_index/radix/policies",
        lambda sym, dso: "kv_index" in sym or "radix_tree" in sym or "smg::policies" in sym,
    ),
    (
        "symphony/minijinja/template",
        lambda sym, dso: (
            "symphony" in sym
            or "minijinja" in sym
            or "response_template" in sym
            or "chat_template" in sym
        ),
    ),
    ("multimodal", lambda sym, dso: "llm_multimodal" in sym or "multimodal" in sym),
    ("smg router/pipeline", lambda sym, dso: "smg::routers" in sym),
    ("smg middleware/server", lambda sym, dso: "smg::middleware" in sym or "smg::server" in sym),
    ("smg other", lambda sym, dso: sym.startswith(("smg::", "<smg::"))),
    ("openai_protocol", lambda sym, dso: "openai_protocol" in sym),
    (
        "metrics/tracing/otel",
        lambda sym, dso: sym.startswith(
            ("metrics", "<metrics", "tracing", "<tracing", "opentelemetry", "<opentelemetry")
        ),
    ),
    (
        "drop glue / outlined",
        lambda sym, dso: "drop_glue" in sym or sym.startswith("OUTLINED_FUNCTION"),
    ),
    (
        "core/alloc std",
        lambda sym, dso: sym.startswith(
            (
                "core::",
                "<core::",
                "alloc::",
                "<alloc::",
                "std::",
                "<std::",
                "hashbrown",
                "<hashbrown",
            )
        ),
    ),
]
# A sample counts once per crate group it touches anywhere in its stack.
INCLUSIVE = [
    ("tokenizer", re.compile(r"llm_tokenizer|tokenizers::|fancy_regex|regex_automata")),
    ("serde/serde_json", re.compile(r"serde_json|serde_core|serde::")),
    ("hyper/h2 (http edge + grpc transport)", re.compile(r"\bh2::|hyper::|hyper_util")),
    ("tonic/prost", re.compile(r"tonic::|prost::")),
    ("kv_index/radix/policies", re.compile(r"kv_index|radix_tree|smg::policies")),
    ("worker_selection", re.compile(r"worker_selection")),
    ("request_execution/client", re.compile(r"request_execution|grpc::client|backend_client")),
    (
        "streaming/response processing",
        re.compile(r"streaming::|response_processing|response_collection"),
    ),
    (
        "preparation/request_building (template, tokenize)",
        re.compile(r"preparation|request_building|chat_utils"),
    ),
    (
        "symphony/minijinja/template",
        re.compile(r"symphony|minijinja|response_template|chat_template"),
    ),
    ("multimodal", re.compile(r"llm_multimodal|multimodal")),
    ("middleware (auth, tenant, concurrency, logging)", re.compile(r"smg::middleware")),
    ("metrics/tracing/otel", re.compile(r"metrics::|tracing::|opentelemetry")),
    ("Clone impls within 6 frames of the leaf", CLONE),
]


def parse(stream):
    sample = None
    for line in stream:
        if not line.strip():
            if sample is not None:
                yield sample
            sample = None
            continue
        if line[0] not in (" ", "\t"):
            sample = {"frames": []}
            continue
        if sample is None:
            continue
        m = FRAME.match(line.rstrip("\n"))
        if m:
            sample["frames"].append((m.group(2), m.group(3)))
    if sample is not None:
        yield sample


def site_after(frames, pattern, skip_libc=True):
    """The first frame not matching `pattern` (nor libc/kernel), with its caller."""
    found = False
    for i, (sym, dso) in enumerate(frames):
        if pattern.search(sym) or (skip_libc and LIBC.search(dso)):
            found = True
            continue
        if not found:
            return None
        caller = frames[i + 1][0] if i + 1 < len(frames) else ""
        return (sym, caller)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json")
    a = ap.parse_args()
    total = 0
    self_counts, bucket_counts, incl_counts = {}, {}, {}
    alloc_sites, copy_sites, clone_sites = {}, {}, {}
    for s in parse(sys.stdin):
        frames = s["frames"]
        if not frames:
            continue
        total += 1
        leaf, leaf_dso = frames[0]
        self_counts[leaf] = self_counts.get(leaf, 0) + 1
        bucket = next((name for name, test in SELF_BUCKETS if test(leaf, leaf_dso)), "other")
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        for name, pat in INCLUSIVE:
            window = frames[:6] if pat is CLONE else frames
            if any(pat.search(sym) for sym, _ in window):
                incl_counts[name] = incl_counts.get(name, 0) + 1
        if ALLOC_LEAF.search(leaf) or (
            LIBC.search(leaf_dso) and len(frames) > 1 and ALLOC.search(frames[1][0])
        ):
            site = site_after(frames, ALLOC)
            if site:
                alloc_sites[site] = alloc_sites.get(site, 0) + 1
        if COPY.search(leaf):
            site = site_after(frames, COPY)
            if site:
                copy_sites[site] = copy_sites.get(site, 0) + 1
        for i, (sym, _) in enumerate(frames[:8]):
            if CLONE.search(sym):
                caller = frames[i + 1][0] if i + 1 < len(frames) else ""
                key = (sym, caller)
                clone_sites[key] = clone_sites.get(key, 0) + 1
                break
    if total == 0:
        print("no samples")
        return 1

    def pct(n):
        return 100.0 * n / total

    out = {"samples": total}
    print(f"samples: {total}\n")
    print("## Self cycles by area (the leaf frame; every sample in one bucket)\n")
    print("| area | % of samples |\n|---|---|")
    out["self_by_area"] = {}
    for name, n in sorted(bucket_counts.items(), key=lambda kv: -kv[1]):
        out["self_by_area"][name] = pct(n)
        print(f"| {name} | {pct(n):.1f} |")
    print("\n## Inclusive by crate group (a sample counts once per group anywhere in its stack)\n")
    print("| group | % of samples |\n|---|---|")
    out["inclusive"] = {}
    for name, _ in INCLUSIVE:
        n = incl_counts.get(name, 0)
        out["inclusive"][name] = pct(n)
        print(f"| {name} | {pct(n):.1f} |")
    print(f"\n## Top {a.top} self symbols\n\n| % | symbol |\n|---|---|")
    out["self"] = [
        (pct(n), s) for s, n in sorted(self_counts.items(), key=lambda kv: -kv[1])[: a.top]
    ]
    for p, s in out["self"]:
        print(f"| {p:.2f} | `{s[:120]}` |")
    for title, sites, key in (
        (
            "allocation sites (cycles inside the allocator, by the caller that allocates)",
            alloc_sites,
            "alloc_sites",
        ),
        (
            "copy sites (cycles in memcpy/memmove/memset, by the caller that copies)",
            copy_sites,
            "copy_sites",
        ),
        (
            "Clone impls on the stack (by the clone frame and its caller)",
            clone_sites,
            "clone_sites",
        ),
    ):
        share = pct(sum(sites.values()))
        print(
            f"\n## Top {a.top} {title}: {share:.1f} % of samples in total\n\n| % | site | caller |\n|---|---|---|"
        )
        out[key] = [
            (pct(n), s[0], s[1]) for s, n in sorted(sites.items(), key=lambda kv: -kv[1])[: a.top]
        ]
        out[key + "_total_pct"] = share
        for p, site, caller in out[key]:
            print(f"| {p:.2f} | `{site[:110]}` | `{caller[:110]}` |")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
