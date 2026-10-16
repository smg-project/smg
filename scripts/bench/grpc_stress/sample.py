#!/usr/bin/env python3
"""Sample a gateway process once a second: CPU time, RSS (anon/file), threads,
file descriptors, its cgroup's memory.current and anon bytes when a cgroup is
given, and every --metrics-every seconds the allocator gauges and request
counters from its Prometheus endpoint. One CSV row per sample, until SIGTERM.
Standard library only."""

import argparse
import csv
import os
import signal
import sys
import time
import urllib.request

CLK_TCK = os.sysconf("SC_CLK_TCK")
METRICS = [
    "smg_allocator_allocated_bytes",
    "smg_allocator_active_bytes",
    "smg_allocator_resident_bytes",
    "smg_allocator_metadata_bytes",
    "smg_http_requests_total",
    "smg_router_requests_total",
    "smg_admission_queue_rejected_total",
    "smg_worker_overload_shed_total",
    "smg_kv_index_blocks_live",
    "smg_kv_index_arena_bytes",
]
FIELDS = [
    "t",
    "cpu_s",
    "vmrss_kb",
    "rssanon_kb",
    "rssfile_kb",
    "threads",
    "fds",
    "cg_current",
    "cg_anon",
    "cg_file",
] + METRICS


def proc_sample(pid):
    with open(f"/proc/{pid}/stat") as f:
        parts = f.read().rsplit(")", 1)[1].split()
    cpu_s = (int(parts[11]) + int(parts[12])) / CLK_TCK  # utime + stime
    status = {}
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            k, _, v = line.partition(":")
            status[k] = v.strip()

    def kb(k):
        return int(status.get(k, "0 kB").split()[0])

    try:
        fds = len(os.listdir(f"/proc/{pid}/fd"))
    except OSError:
        fds = ""
    return cpu_s, kb("VmRSS"), kb("RssAnon"), kb("RssFile"), int(status.get("Threads", 0)), fds


def cgroup_sample(path):
    if not path:
        return "", "", ""
    try:
        with open(os.path.join(path, "memory.current")) as f:
            current = int(f.read())
        anon = file = ""
        with open(os.path.join(path, "memory.stat")) as f:
            for line in f:
                k, v = line.split()
                if k == "anon":
                    anon = int(v)
                elif k == "file":
                    file = int(v)
        return current, anon, file
    except OSError:
        return "", "", ""


def metrics_sample(url):
    out = {m: "" for m in METRICS}
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            text = r.read().decode()
    except Exception:
        return out
    sums = {m: 0.0 for m in METRICS}
    seen = set()
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in sums:
            try:
                sums[name] += float(line.rsplit(" ", 1)[1])
                seen.add(name)
            except ValueError:
                pass
    for m in seen:
        out[m] = sums[m]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cgroup", default="")
    ap.add_argument("--metrics", default="")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--metrics-every", type=float, default=5.0)
    a = ap.parse_args()
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    last_metrics = -1e9
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        f.flush()
        next_t = time.time()
        while not stop["flag"]:
            now = time.time()
            try:
                cpu_s, vmrss, anon, file, threads, fds = proc_sample(a.pid)
            except (OSError, IndexError, ValueError):
                break
            row = [
                f"{now:.3f}",
                f"{cpu_s:.3f}",
                vmrss,
                anon,
                file,
                threads,
                fds,
                *cgroup_sample(a.cgroup),
            ]
            if a.metrics and now - last_metrics >= a.metrics_every:
                m = metrics_sample(a.metrics)
                last_metrics = now
                row += [m[k] for k in METRICS]
            else:
                row += [""] * len(METRICS)
            w.writerow(row)
            f.flush()
            next_t += a.interval
            delay = next_t - time.time()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.time()
    return 0


if __name__ == "__main__":
    sys.exit(main())
