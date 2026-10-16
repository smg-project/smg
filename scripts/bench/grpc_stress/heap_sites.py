#!/usr/bin/env python3
"""Diff two jemalloc heap profiles (heap_v2, written with prof_accum:true by
POST /heap_profile at the start and the end of a window) into the allocation
sites of the window: objects and bytes allocated per stack, unbiased for the
sampling interval the way jeprof does, symbolized with addr2line against the
gateway binary and grouped by the first frame outside the allocator.
Usage: heap_sites.py --bin smg --start a.heap --end b.heap [--requests N] [--top 20] [--json out]"""

import argparse
import json
import math
import re
import subprocess
import sys

SKIP = re.compile(
    r"^(prof_|_rjem_|je_|imalloc|ifree|do_rallocx|arena_|tcache|large_|extent_|sz_|malloc_default|"
    r"malloc$|calloc$|realloc$|posix_memalign$|aligned_alloc$|__rust_|__rg_|__rdl_|rust_alloc)|"
    r"^alloc::alloc::|alloc::raw_vec|RawVec|<alloc::alloc::Global|box_new_uninit|finish_grow|"
    r"alloc::vec::Vec<.*>::(reserve|push|extend|with_capacity|from_iter|insert)|"
    r"alloc::string::String::(push|reserve|from|with_capacity)|<alloc::string::String as core::clone|"
    r"<alloc::vec::Vec<.*> as (alloc::vec::spec|core::clone|core::iter|alloc::vec::in_place)|<T as alloc::slice::|"
    r"<\[T\]>::to_vec|<str as alloc::str::|<str>::to_owned|<str>::to_string|<alloc::boxed::Box<.*>::new|"
    r"<alloc::sync::Arc<.*>::new|alloc::collections|hashbrown::|<hashbrown::|^std::thread|^core::|"
    r"<core::|BytesMut::|<bytes::|bytes::|indexmap::|<indexmap::|serde_json::value::ser|"
    r"<serde_json::|serde_json::de|serde_core::"
)


def parse_heap(path):
    interval = None
    stacks = {}
    maps = []
    with open(path) as f:
        lines = f.read().splitlines()
    i = 0
    cur = None
    while i < len(lines):
        line = lines[i]
        if line.startswith("heap_v2/"):
            interval = int(line.split("/", 1)[1].split()[0])
        elif line.startswith("@ "):
            cur = tuple(int(a, 16) for a in line[2:].split())
        elif line.strip().startswith("t*:") and cur is not None:
            m = re.match(r"\s*t\*:\s*(\d+):\s*(\d+)\s*\[\s*(\d+):\s*(\d+)\s*\]", line)
            if m:
                stacks[cur] = tuple(int(x) for x in m.groups())
            cur = None
        elif line.startswith("MAPPED_LIBRARIES:"):
            maps = lines[i + 1 :]
            break
        i += 1
    return interval, stacks, maps


def load_base(maps, binary):
    for line in maps:
        parts = line.split()
        if len(parts) >= 6 and parts[5].endswith(binary.rsplit("/", 1)[-1]):
            start = int(parts[0].split("-")[0], 16)
            offset = int(parts[2], 16)
            return start - offset
    return 0


def unbias(objs, bytes_, interval):
    if objs <= 0 or bytes_ <= 0 or not interval:
        return objs, bytes_
    ratio = 1.0 / (1.0 - math.exp(-(bytes_ / objs) / interval))
    return objs * ratio, bytes_ * ratio


def symbolize(binary, addrs):
    """Function names for the addresses, from the binary's symbol table (`nm -C -n`, read once; a
    bisect per address). Function granularity is what the site attribution needs, and it stays fast
    on a large binary with full debug info, which addr2line reads again for every batch."""
    import bisect

    proc = subprocess.run(
        ["nm", "-C", "-n", "--defined-only", binary], capture_output=True, text=True, check=False
    )
    starts, names = [], []
    for line in proc.stdout.splitlines():
        parts = line.split(" ", 2)
        if len(parts) == 3 and parts[1].lower() in ("t", "w", "i"):
            try:
                starts.append(int(parts[0], 16))
            except ValueError:
                continue
            names.append(parts[2])
    out = {}
    for addr in addrs:
        i = bisect.bisect_right(starts, addr) - 1
        out[addr] = [names[i]] if i >= 0 else [hex(addr)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--requests", type=float, default=0)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json")
    a = ap.parse_args()
    interval, s0, maps0 = parse_heap(a.start)
    _, s1, maps1 = parse_heap(a.end)
    base = load_base(maps1 or maps0, a.bin)
    deltas = {}
    live = {}
    for stack, (co1, cb1, ao1, ab1) in s1.items():
        co0, cb0, ao0, ab0 = s0.get(stack) or (0, 0, 0, 0)
        do, db = ao1 - ao0, ab1 - ab0
        if do > 0 or db > 0:
            deltas[stack] = unbias(do, db, interval)
        lo, lb = co1 - co0, cb1 - cb0
        if lo != 0 or lb != 0:
            live[stack] = unbias(abs(lo), abs(lb), interval) if lb > 0 else (lo, lb)
    for stack, (co0, cb0, _, _) in s0.items():
        if stack not in s1 and (co0 or cb0):
            live[stack] = (-co0, -cb0)
    addrs = {addr for stack in list(deltas) + list(live) for addr in stack}
    syms = symbolize(
        a.bin, {addr - base - 1 for addr in addrs}
    )  # return addresses: the call is one byte back

    def frames(stack):
        names = []
        for addr in stack:
            chain = syms.get(addr - base - 1) or [hex(addr)]
            names.extend(chain)
        return names

    groups = {}
    for stack, (do, db) in deltas.items():
        names = frames(stack)
        site = next((n for n in names if not SKIP.search(n)), names[0] if names else "?")
        idx = names.index(site) if site in names else 0
        key = (site, " <- ".join(names[idx + 1 : idx + 3]))
        g = groups.setdefault(key, [0.0, 0.0])
        g[0] += do
        g[1] += db
    tot_o = sum(g[0] for g in groups.values())
    tot_b = sum(g[1] for g in groups.values())
    print(f"sample interval: {interval} bytes; stacks with new allocations: {len(deltas)}")
    print(
        f"allocated in the window (unbiased estimate): {tot_o:,.0f} objects, {tot_b / 2**20:,.1f} MiB",
        end="",
    )
    if a.requests:
        print(
            f"; per request: {tot_o / a.requests:,.1f} objects, {tot_b / a.requests / 1024:,.1f} KiB",
            end="",
        )
    print("\n")
    out = {"interval": interval, "objects": tot_o, "bytes": tot_b, "by_bytes": [], "by_objects": []}
    for title, idx, key in (("by bytes", 1, "by_bytes"), ("by objects", 0, "by_objects")):
        print(
            f"## Top {a.top} allocation sites {title}\n\n"
            "| % bytes | % objects | KiB/req | objs/req | site | callers |\n|---|---|---|---|---|---|"
        )
        for (site, callers), (do, db) in sorted(groups.items(), key=lambda kv: -kv[1][idx])[
            : a.top
        ]:
            per_b = db / a.requests / 1024 if a.requests else 0.0
            per_o = do / a.requests if a.requests else 0.0
            out[key].append({"site": site, "callers": callers, "objects": do, "bytes": db})
            print(
                f"| {100 * db / tot_b if tot_b else 0:.1f} | {100 * do / tot_o if tot_o else 0:.1f} | "
                f"{per_b:.2f} | {per_o:.2f} | `{site[:100]}` | `{callers[:120]}` |"
            )
        print()
    # What stayed allocated: the in-use (live) bytes per site at the end minus at the start of the window.
    growth = {}
    for stack, (lo, lb) in live.items():
        names = frames(stack)
        site = next((n for n in names if not SKIP.search(n)), names[0] if names else "?")
        idx = names.index(site) if site in names else 0
        key = (site, " <- ".join(names[idx + 1 : idx + 3]))
        g = growth.setdefault(key, [0.0, 0.0])
        g[0] += lo
        g[1] += lb
    net = sum(g[1] for g in growth.values())
    print(
        f"## Live bytes at the end minus at the start of the window: net {net / 2**20:+,.1f} MiB "
        f"(top {a.top} sites by growth)\n"
    )
    print("| live MiB delta | live objects delta | site | callers |\n|---|---|---|---|")
    out["live_growth"] = []
    for (site, callers), (lo, lb) in sorted(growth.items(), key=lambda kv: -kv[1][1])[: a.top]:
        out["live_growth"].append({"site": site, "callers": callers, "objects": lo, "bytes": lb})
        print(f"| {lb / 2**20:+.2f} | {lo:+,.0f} | `{site[:100]}` | `{callers[:120]}` |")
    print()
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
