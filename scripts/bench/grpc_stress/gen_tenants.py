#!/usr/bin/env python3
"""Write a tenant rate-limit YAML for N header-resolved tenants
(tenant-0..tenant-N-1, as stress_load --tenants sends them), each with its own
limits high enough not to shed at the cell's rate, so the cost measured is
the per-tenant accounting, not rejection."""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--rpm", type=int, default=120_000)
    ap.add_argument("--tpm", type=int, default=100_000_000)
    a = ap.parse_args()
    lines = [
        "default_policy:",
        f"  tokens_per_minute: {a.tpm}",
        f"  requests_per_minute: {a.rpm}",
        "tenants:",
    ]
    for i in range(a.count):
        lines += [
            f'  - tenant_key: "header:tenant-{i}"',
            f"    tokens_per_minute: {a.tpm - i}",
            f"    requests_per_minute: {a.rpm - i}",
        ]
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(a.out, a.count, "tenants")
    return 0


if __name__ == "__main__":
    sys.exit(main())
