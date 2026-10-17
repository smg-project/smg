#!/usr/bin/env python3
"""Renders the throughput row of a Symphony change from criterion's saved baselines.

Usage: throughput_row.py --filter PREFIX LABEL=CRITERION_DIR [LABEL=CRITERION_DIR ...]

Each side is a label and the criterion directory its bench binary wrote (`criterion/` under the
build's target directory, or the `CRITERION_HOME` the README's recipe gives both sides: one
directory can serve several sides, each under its own label); its baselines are the ones named
LABEL or LABEL-<n>, one per repeat.
For every bench whose id starts with PREFIX the table has, per side, the median over the repeats
of the bench's mean, in nanoseconds per byte, per token, per event or per iteration as the bench
declares its throughput, with the spread of the repeats, and the change of every side after the
first against the first.
"""

import argparse
import json
import pathlib
import statistics
import sys

# Baseline names criterion writes on its own: `base` is the default of `--save-baseline` (any
# plain run), `new` is rewritten by every run, `change` by a comparison. A side under one of them
# would take a stray run for a repeat.
RESERVED_LABELS = ("base", "new", "change")


def unit_of(full_id, throughput):
    if "Bytes" in throughput:
        return "ns/byte", throughput["Bytes"]
    if "Elements" in throughput:
        name = "ns/event" if full_id.startswith("adapt/") else "ns/token"
        return name, throughput["Elements"]
    return "ns", 1


def baselines(root, label):
    """The benches under `root` with a baseline of `label`: id -> (unit, [ns per unit, per repeat])."""
    out = {}
    for meta_path in sorted(pathlib.Path(root).rglob("benchmark.json")):
        name = meta_path.parent.name
        suffix = name[len(label) + 1 :]
        if name != label and not (name.startswith(label + "-") and suffix.isdigit()):
            continue
        meta = json.loads(meta_path.read_text())
        estimates = json.loads((meta_path.parent / "estimates.json").read_text())
        unit, per = unit_of(meta["full_id"], meta.get("throughput") or {})
        out.setdefault(meta["full_id"], (unit, []))[1].append(
            estimates["mean"]["point_estimate"] / per
        )
    return out


def repeats(benches, ids):
    """The repeat count of `benches` over the benches shown, or its range when they differ."""
    counts = sorted({len(benches[full_id][1]) for full_id in ids if full_id in benches})
    if not counts:
        return 0
    return counts[0] if len(counts) == 1 else f"{counts[0]}-{counts[-1]}"


def cell(values):
    median = statistics.median(values)
    if len(values) == 1:
        return f"{median:.1f}", median
    return f"{median:.1f} ({min(values):.1f}-{max(values):.1f})", median


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--filter", default="", help="a prefix of the bench ids to show")
    parser.add_argument("sides", nargs="+", metavar="LABEL=CRITERION_DIR")
    args = parser.parse_args()
    sides = []
    for side in args.sides:
        label, _, root = side.partition("=")
        if not root:
            sys.exit(f"throughput_row: {side!r} is not LABEL=CRITERION_DIR")
        if label in RESERVED_LABELS:
            sys.exit(
                f"throughput_row: {label!r} is a name criterion writes itself; "
                "label the side by hand (main, pr)"
            )
        sides.append((label, baselines(root, label)))
    ids = sorted(
        {full_id for _, benches in sides for full_id in benches if full_id.startswith(args.filter)}
    )
    if not ids:
        sys.exit("throughput_row: no bench matches; the baselines are missing or the filter is off")
    header = ["bench", "unit"] + [f"{label} ({repeats(b, ids)})" for label, b in sides]
    header += [f"{label} vs {sides[0][0]}" for label, _ in sides[1:]]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for full_id in ids:
        unit = next(b[full_id][0] for _, b in sides if full_id in b)
        cells, medians = [], []
        for _, benches in sides:
            if full_id in benches:
                text, median = cell(benches[full_id][1])
            else:
                text, median = "-", None
            cells.append(text)
            medians.append(median)
        changes = []
        for median in medians[1:]:
            if median is None or medians[0] is None:
                changes.append("-")
            else:
                changes.append(f"{(median / medians[0] - 1) * 100:+.1f}%")
        print("| " + " | ".join([full_id, unit] + cells + changes) + " |")


if __name__ == "__main__":
    main()
