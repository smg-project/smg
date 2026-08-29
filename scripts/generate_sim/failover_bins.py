#!/usr/bin/env python3
"""Time-sliced failover analysis: bin follow-up cache ratios around the
index-replica kill instant recorded in meta.json.

Usage: failover_bins.py <scenario-run-dir> [bin_secs]
"""
import glob
import json
import sys


def main():
    base = sys.argv[1]
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    bins = {}
    seeds = sorted(glob.glob(f"{base}/*/seed-*/"))
    observed = 0
    errors = requests = 0
    for sd in seeds:
        meta = json.load(open(sd + "meta.json"))
        killed = meta.get("index_killed_at_ms")
        if not killed:
            print(f"WARN: kill not observed in {sd}")
            continue
        observed += 1
        for line in open(sd + "requests.jsonl"):
            rec = json.loads(line)
            requests += 1
            if not (200 <= rec.get("status", 0) < 300):
                errors += 1
            if rec.get("turn", 1) < 2:
                continue
            prompt = rec.get("prompt_tokens") or 0
            cached = rec.get("cached_tokens") or 0
            if not prompt:
                continue
            offset = (rec["start_ms"] - killed) / 1000.0
            b = int(offset // width) * width
            slot = bins.setdefault(b, {"p": 0, "c": 0, "src": {}})
            slot["p"] += prompt
            slot["c"] += cached
            src = rec.get("index_source") or "none"
            slot["src"][src] = slot["src"].get(src, 0) + 1
    print(f"kill observed in {observed}/{len(seeds)} seeds; errors {errors}/{requests}")
    for b in sorted(bins):
        slot = bins[b]
        total = sum(slot["src"].values())
        top = ", ".join(
            f"{k} {v / total:.0%}"
            for k, v in sorted(slot["src"].items(), key=lambda kv: -kv[1])[:3]
        )
        print(f"{b:+6d}s  followup cached {slot['c'] / slot['p']:.4f}  {top}")


if __name__ == "__main__":
    main()
