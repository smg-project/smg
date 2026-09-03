#!/usr/bin/env python3
"""Refit every selected engine from a checkpoint on disk, through SMG.

    python examples/rl/refit_from_disk.py --smg http://127.0.0.1:30000 \
        --model-path /ckpt/step-42 --weight-version 42 --selector engine=sglang

Sequence: pause_generation -> update_weights_from_disk -> continue_generation,
each as one fan-out, then one /generate through SMG to confirm the engine
reports the new meta_info.weight_version.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

from smg.rl import RL


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--smg", required=True, help="SMG base URL, e.g. http://127.0.0.1:30000")
    ap.add_argument(
        "--model-path", required=True, help="checkpoint directory visible to every engine"
    )
    ap.add_argument("--weight-version", required=True, help="version string to stamp, e.g. 42")
    ap.add_argument("--selector", default="engine=sglang", help="which workers to refit")
    ap.add_argument("--api-key", default=None, help="SMG control-plane key, if configured")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    rl = RL(args.smg, api_key=args.api_key, timeout=args.timeout)
    workers = rl.workers()
    print(f"{len(workers)} worker(s) registered; selector={args.selector!r}")

    print("pause_generation ...")
    # SGLang requires a JSON body on these routes; a bodyless POST is a 400.
    rl.fanout("pause_generation", {}, selector=args.selector)
    print("update_weights_from_disk ...")
    res = rl.fanout(
        "update_weights_from_disk",
        {"model_path": args.model_path, "weight_version": args.weight_version, "flush_cache": True},
        selector=args.selector,
    )
    for wid, r in res.results.items():
        print(f"  {wid}: HTTP {r.status} in {r.latency_ms} ms -> {json.dumps(r.body)[:120]}")
    print("continue_generation ...")
    rl.fanout("continue_generation", {}, selector=args.selector)

    req = urllib.request.Request(
        f"{args.smg}/generate",
        data=json.dumps({"text": "1+1=", "sampling_params": {"max_new_tokens": 4}}).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read())
    got = str(out.get("meta_info", {}).get("weight_version"))
    print(f"/generate reported weight_version={got}")
    if got != str(args.weight_version):
        print("MISMATCH: engine did not report the new version", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
