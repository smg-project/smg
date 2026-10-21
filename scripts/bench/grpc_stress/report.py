#!/usr/bin/env python3
"""Turn the cells of a run into the tables: per cell the gateway's CPU per
request and per streamed frame over the sampled window (from the sampler's
utime+stime), the RSS/anon before and after with the slope over the window
(a leak shows as a slope), the allocator gauges, the client's latency
percentiles and its status/error counts, the shed counters from the metrics;
then the mean over the repeats of each cell name (`<name>-r<k>`).
Standard library only: `report.py <run dir | cell dirs...> [--json out.json] [--md out.md]`."""

import argparse
import csv
import glob
import json
import os
import re
import sys


def read_samples(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: (float(v) if v not in ("", None) else None) for k, v in r.items()})
            except ValueError:
                continue
    return rows


def slope_per_min(points):
    """Least-squares slope of (t seconds, value) in value units per minute."""
    pts = [(t, v) for t, v in points if v is not None]
    if len(pts) < 3:
        return None
    n = len(pts)
    mt = sum(t for t, _ in pts) / n
    mv = sum(v for _, v in pts) / n
    den = sum((t - mt) ** 2 for t, _ in pts)
    if den == 0:
        return None
    return sum((t - mt) * (v - mv) for t, v in pts) / den * 60.0


def prom_sum(text, name, label=None):
    """Sum of a metric's samples, optionally keyed by one label's value."""
    total = 0.0
    by = {}
    for line in text.splitlines():
        if not line.startswith(name):
            continue
        head, _, value = line.rpartition(" ")
        if head.split("{", 1)[0] != name:
            continue
        try:
            v = float(value)
        except ValueError:
            continue
        total += v
        if label:
            m = re.search(label + r'="([^"]*)"', head)
            key = m.group(1) if m else ""
            by[key] = by.get(key, 0.0) + v
    return (total, by) if label else total


def first_last(rows, key):
    vals = [(r["t"], r[key]) for r in rows if r.get(key) is not None]
    if not vals:
        return None, None, vals
    return vals[0][1], vals[-1][1], vals


def analyze(cell_dir):
    cj = json.load(open(os.path.join(cell_dir, "cell.json")))
    summary_path = os.path.join(cell_dir, "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    rows = (
        read_samples(os.path.join(cell_dir, "samples.csv"))
        if os.path.exists(os.path.join(cell_dir, "samples.csv"))
        else []
    )
    w0 = cj["t0"] + cj["warmup"]
    w1 = w0 + cj["duration"]
    win = [r for r in rows if w0 <= r["t"] <= w1]
    out = {
        "cell": cj["name"],
        "dir": cell_dir,
        "window_s": cj["duration"],
        "loadavg_end": cj.get("loadavg_end", ""),
    }
    out["host_busy_pct"] = (cj.get("busy") or {}).get("host_busy_pct")
    out["gw_cores_busy_pct"] = (cj.get("busy") or {}).get("gw_cores_busy_pct")
    out["rps_target"] = summary.get("rate_target")
    out["rps"] = summary.get("achieved_rps")
    out["sampled"] = summary.get("sampled", 0)
    out["ok"] = summary.get("ok")
    out["failed"] = summary.get("failed")
    out["skipped"] = summary.get("skipped")
    out["unfinished"] = summary.get("unfinished")
    out["statuses"] = summary.get("statuses", {})
    out["errors"] = summary.get("errors", {})
    out["inflight_max"] = summary.get("inflight_max")
    for k in ("ttfb_ms", "ttft_ms", "e2e_ms", "gap_us"):
        q = summary.get(k, {})
        out[k] = {p: q.get(p) for p in ("p50", "p90", "p99", "max")}
    out["frames_per_request"] = summary.get("frames_per_request")
    frames = summary.get("frames_total", 0) or 0
    if len(win) >= 2:
        cpu = win[-1]["cpu_s"] - win[0]["cpu_s"]
        span = win[-1]["t"] - win[0]["t"]
        out["cpu_s"] = cpu
        out["cores"] = cpu / span if span > 0 else None
        out["cpu_ms_per_req"] = cpu * 1000.0 / out["sampled"] if out["sampled"] else None
        out["cpu_us_per_frame"] = cpu * 1e6 / frames if frames else None
        a0, a1, apts = first_last(win, "rssanon_kb")
        out["rss_anon_mb_start"] = a0 / 1024 if a0 is not None else None
        out["rss_anon_mb_end"] = a1 / 1024 if a1 is not None else None
        s = slope_per_min(apts)
        out["rss_anon_slope_mb_per_min"] = s / 1024 if s is not None else None
        r0, r1, _ = first_last(win, "vmrss_kb")
        out["rss_mb_start"] = r0 / 1024 if r0 is not None else None
        out["rss_mb_end"] = r1 / 1024 if r1 is not None else None
        out["rss_mb_max"] = (
            max((r["vmrss_kb"] for r in rows if r.get("vmrss_kb") is not None), default=0) / 1024
        )
        c0, c1, cpts = first_last(win, "cg_anon")
        out["cg_anon_mb_start"] = c0 / 2**20 if c0 is not None else None
        out["cg_anon_mb_end"] = c1 / 2**20 if c1 is not None else None
        s = slope_per_min(cpts)
        out["cg_anon_slope_mb_per_min"] = s / 2**20 if s is not None else None
        m0, m1, _ = first_last(win, "smg_allocator_allocated_bytes")
        out["alloc_allocated_mb_start"] = m0 / 2**20 if m0 is not None else None
        out["alloc_allocated_mb_end"] = m1 / 2**20 if m1 is not None else None
        _, res1, _ = first_last(win, "smg_allocator_resident_bytes")
        out["alloc_resident_mb_end"] = res1 / 2**20 if res1 is not None else None
        out["threads_end"] = win[-1].get("threads")
        out["fds_end"] = win[-1].get("fds")
    try:
        before = open(os.path.join(cell_dir, "before.prom")).read()
        after = open(os.path.join(cell_dir, "after.prom")).read()
        out["shed_admission"] = prom_sum(after, "smg_admission_queue_rejected_total") - prom_sum(
            before, "smg_admission_queue_rejected_total"
        )
        out["shed_overload"] = prom_sum(after, "smg_worker_overload_shed_total") - prom_sum(
            before, "smg_worker_overload_shed_total"
        )
        out["router_errors"] = prom_sum(after, "smg_router_request_errors_total") - prom_sum(
            before, "smg_router_request_errors_total"
        )
        _, sb = prom_sum(before, "smg_http_responses_total", "status_code")
        _, sa = prom_sum(after, "smg_http_responses_total", "status_code")
        out["http_status_delta"] = {
            k: sa.get(k, 0) - sb.get(k, 0) for k in set(sa) | set(sb) if sa.get(k, 0) - sb.get(k, 0)
        }
    except OSError:
        pass
    return out


def fmt(v, digits=2):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


COLUMNS = [
    ("cell", "cell", None),
    ("rps", "rps", 0),
    ("ok", "ok", None),
    ("failed", "failed", None),
    ("cpu_ms_per_req", "CPU ms/req", 3),
    ("cpu_us_per_frame", "CPU us/frame", 1),
    ("cores", "cores", 2),
    ("ttfb_p50", "TTFB p50", 1),
    ("ttfb_p99", "TTFB p99", 1),
    ("ttft_p50", "TTFT p50", 1),
    ("ttft_p99", "TTFT p99", 1),
    ("e2e_p99", "e2e p99", 0),
    ("gap_p50", "gap p50 us", 0),
    ("gap_p99", "gap p99 us", 0),
    ("rss_anon_mb_start", "anon MB start", 0),
    ("rss_anon_mb_end", "anon MB end", 0),
    ("rss_anon_slope_mb_per_min", "anon MB/min", 2),
    ("rss_mb_max", "RSS MB max", 0),
    ("alloc_allocated_mb_end", "jemalloc live MB", 0),
    ("shed", "shed", None),
    ("host_busy_pct", "host busy %", 0),
]


def flat(c):
    f = dict(c)
    for k in ("ttfb", "ttft", "e2e"):
        f[k + "_p50"] = c.get(k + "_ms", {}).get("p50")
        f[k + "_p99"] = c.get(k + "_ms", {}).get("p99")
    f["gap_p50"] = c.get("gap_us", {}).get("p50")
    f["gap_p99"] = c.get("gap_us", {}).get("p99")
    f["shed"] = int((c.get("shed_admission") or 0) + (c.get("shed_overload") or 0))
    return f


def table(cells):
    lines = ["| " + " | ".join(h for _, h, _ in COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for c in cells:
        f = flat(c)
        lines.append(
            "| "
            + " | ".join(
                fmt(f.get(k), d if d is not None else 2)
                if not isinstance(f.get(k), str)
                else f.get(k)
                for k, _, d in COLUMNS
            )
            + " |"
        )
    return "\n".join(lines)


def aggregate(cells):
    groups = {}
    for c in cells:
        name = re.sub(r"-r\d+$", "", c["cell"])
        groups.setdefault(name, []).append(flat(c))
    rows = []
    for name, members in groups.items():
        agg = {"cell": f"{name} (n={len(members)})"}
        for k, _, _ in COLUMNS[1:]:
            vals = [m.get(k) for m in members if isinstance(m.get(k), (int, float))]
            agg[k] = sum(vals) / len(vals) if vals else None
            if k in ("cpu_ms_per_req", "cpu_us_per_frame") and len(vals) > 1:
                agg[k + "_range"] = (min(vals), max(vals))
        rows.append(agg)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--md")
    a = ap.parse_args()
    dirs = []
    for p in a.paths:
        if os.path.exists(os.path.join(p, "cell.json")):
            dirs.append(p)
        else:
            dirs += sorted(
                d
                for d in glob.glob(os.path.join(p, "cells", "*"))
                if os.path.exists(os.path.join(d, "cell.json"))
            )
    cells = [analyze(d) for d in dirs]
    md = ["## Cells", "", table(cells), "", "## Mean over repeats", "", table(aggregate(cells))]
    text = "\n".join(md)
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")
    if a.json:
        json.dump({"cells": cells, "mean": aggregate(cells)}, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
