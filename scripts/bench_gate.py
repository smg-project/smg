#!/usr/bin/env python3
"""Regression gate for a criterion benchmark: the pull request against its base, in one job.

Two bench binaries, the base branch's and the pull request's (both built in the same
checkout), run interleaved N times each on the same runner, every run with a fresh criterion
output directory, so that one run yields one median per benchmark id. Per id the gate
compares the median of the N pull-request runs with the median of the N base runs and fails
only when both hold:

1. the change is worse than the id's threshold, and
2. the two sets of runs do not overlap: the slowest base run is faster than the fastest
   pull-request run. With no real change, N interleaved runs of each side fall in that
   order with probability 1 / C(2N, N): one in 20 at N = 3, one in 70 at N = 4, one in
   252 at N = 5. This is what makes the threshold noise-aware: a runner hiccup that
   lands on one run cannot fail the job on its own.

A change beyond the threshold whose runs overlap is reported as suspect, not failed. An id
no gate pattern matches is reported, never failed. A gated id the base has and the pull
request lacks fails: a benchmark that is removed or renamed on purpose is excluded from the
group pattern that matches its old id (a negative lookahead such as
`^(?!latency/encode_4KB$)(encode_throughput|latency)/`) in the same change.

The repeat count, the id filter and the thresholds come from a TOML file
(.github/bench-gates/<bench>.toml); the measured spreads that justify each threshold are
documented there.

Usage:
    bench_gate.py build --out FILE [--name BENCH] -- cargo bench ... --no-run
        Run the cargo command and copy the bench executable it reports to FILE.
    bench_gate.py run --gates FILE --base BIN --pr BIN --out DIR [--cwd DIR] [--repeats N]
        Run both binaries interleaved, write DIR/results.json and DIR/gate.md, compare.
    bench_gate.py compare --gates FILE RESULTS_JSON [--summary FILE]
        Re-evaluate a saved results.json.

The markdown table goes to stdout and, when set, to $GITHUB_STEP_SUMMARY (or --summary).
Exit code 1 means a gated regression, 2 a usage or run error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

SIDES = ("base", "pr")


# --------------------------------------------------------------------------- gate file


@dataclass
class Gate:
    pattern: str
    threshold: float
    note: str = ""

    def matches(self, bench_id: str) -> bool:
        return re.search(self.pattern, bench_id) is not None


@dataclass
class Gates:
    repeats: int = 3
    filter: str | None = None
    bench_args: list[str] = field(default_factory=list)
    gates: list[Gate] = field(default_factory=list)

    def threshold_for(self, bench_id: str) -> Gate | None:
        """The first gate whose pattern matches wins; None means report only."""
        for gate in self.gates:
            if gate.matches(bench_id):
                return gate
        return None


GATE_FILE_KEYS = {"repeats", "filter", "bench_args", "gate"}
GATE_ENTRY_KEYS = {"pattern", "threshold", "note"}


def load_gates(path: Path) -> Gates:
    """The gate file, checked: an unknown key, a value of the wrong type, a `[[gate]]` without
    `pattern` or `threshold`, a pattern or filter that does not compile, a threshold outside
    (0, 1) or a repeat count under 1 is a `ValueError` (exit 2, a usage error), never a silent
    default or a traceback."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    unknown = set(raw) - GATE_FILE_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}; known: {sorted(GATE_FILE_KEYS)}")
    repeats = raw.get("repeats", 3)
    if isinstance(repeats, bool) or not isinstance(repeats, int):
        raise ValueError(f"{path}: repeats must be an integer, got {repeats!r}")
    if repeats < 1:
        raise ValueError(f"{path}: repeats must be at least 1")
    filter_ = raw.get("filter")
    if filter_ is not None and not isinstance(filter_, str):
        raise ValueError(f"{path}: filter must be a string, got {filter_!r}")
    bench_args = raw.get("bench_args", [])
    if not isinstance(bench_args, list):
        raise ValueError(f"{path}: bench_args must be a list of strings, got {bench_args!r}")
    gates = Gates(repeats=repeats, filter=filter_, bench_args=[str(arg) for arg in bench_args])
    if gates.filter is not None:
        try:
            re.compile(gates.filter)
        except re.error as err:
            raise ValueError(f"{path}: filter {gates.filter!r} does not compile: {err}") from err
    entries = raw.get("gate", [])
    if not isinstance(entries, list):
        raise ValueError(
            f"{path}: gate must be an array of tables ([[gate]]), got a {type(entries).__name__}"
        )
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or "pattern" not in entry or "threshold" not in entry:
            raise ValueError(f"{path}: [[gate]] entry {index} needs pattern and threshold")
        unknown = set(entry) - GATE_ENTRY_KEYS
        if unknown:
            raise ValueError(f"{path}: [[gate]] entry {index}: unknown keys {sorted(unknown)}")
        pattern, threshold = entry["pattern"], entry["threshold"]
        if not isinstance(pattern, str):
            raise ValueError(
                f"{path}: [[gate]] entry {index}: pattern must be a string, got {pattern!r}"
            )
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(
                f"{path}: [[gate]] entry {index}: threshold must be a number, got {threshold!r}"
            )
        threshold = float(threshold)
        if not 0 < threshold < 1:
            raise ValueError(f"{path}: threshold {threshold} must be a fraction in (0, 1)")
        try:
            re.compile(pattern)
        except re.error as err:
            raise ValueError(f"{path}: pattern {pattern!r} does not compile: {err}") from err
        gates.gates.append(Gate(pattern, threshold, str(entry.get("note", ""))))
    return gates


# --------------------------------------------------------------------------- build


def select_bench_executable(messages: list[dict], name: str | None) -> str:
    """The executable of the one bench target among cargo's JSON messages."""
    found: dict[str, str] = {}
    for message in messages:
        if message.get("reason") != "compiler-artifact" or not message.get("executable"):
            continue
        target = message.get("target", {})
        if "bench" not in target.get("kind", []):
            continue
        if name is not None and target.get("name") != name:
            continue
        found[target.get("name", "?")] = message["executable"]
    if len(found) != 1:
        wanted = f" named {name}" if name else ""
        raise ValueError(
            f"expected one bench executable{wanted}, cargo reported {sorted(found) or 'none'}"
        )
    return next(iter(found.values()))


def cmd_build(args: argparse.Namespace) -> int:
    cargo = list(args.cargo)
    if cargo and cargo[0] == "--":
        cargo = cargo[1:]
    if not cargo:
        print("build: give the cargo command after --", file=sys.stderr)
        return 2
    if "--no-run" not in cargo:
        cargo.append("--no-run")
    cargo.append("--message-format=json-render-diagnostics")
    print("+", " ".join(cargo), flush=True)
    proc = subprocess.run(cargo, stdout=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0:
        print(f"build: cargo exited with {proc.returncode}", file=sys.stderr)
        return 2
    messages = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            messages.append(json.loads(line))
    try:
        executable = select_bench_executable(messages, args.name)
    except ValueError as err:
        print(f"build: {err}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(executable, out)
    out.chmod(out.stat().st_mode | 0o111)
    print(f"build: {executable} -> {out}", flush=True)
    return 0


# --------------------------------------------------------------------------- run


def collect(home: Path) -> dict[str, float]:
    """Per benchmark id, the median (ns per iteration) criterion wrote under HOME."""
    medians: dict[str, float] = {}
    for benchmark_json in sorted(home.rglob("benchmark.json")):
        if benchmark_json.parent.name != "new":
            continue
        estimates_json = benchmark_json.with_name("estimates.json")
        if not estimates_json.exists():
            continue
        with benchmark_json.open() as handle:
            bench_id = json.load(handle)["full_id"]
        with estimates_json.open() as handle:
            medians[bench_id] = float(json.load(handle)["median"]["point_estimate"])
    return medians


def run_once(
    binary: Path,
    home: Path,
    cwd: Path,
    gates: Gates,
    extra_args: list[str],
    log_path: Path,
) -> dict[str, float]:
    home.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "--bench", "--noplot", *gates.bench_args, *extra_args]
    if gates.filter:
        cmd.append(gates.filter)
    env = dict(os.environ, CRITERION_HOME=str(home))
    with log_path.open("w") as log:
        log.write("+ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"{binary} exited with {proc.returncode}; see {log_path}")
    medians = collect(home)
    if not medians:
        raise RuntimeError(f"{binary} wrote no criterion estimates under {home}; see {log_path}")
    return medians


def interleaved_order(repeats: int) -> list[tuple[int, str]]:
    """Run i alternates which side goes first, so neither side always warms the other up."""
    order = []
    for i in range(1, repeats + 1):
        sides = SIDES if i % 2 == 1 else tuple(reversed(SIDES))
        order.extend((i, side) for side in sides)
    return order


def cmd_run(args: argparse.Namespace) -> int:
    gates = load_gates(Path(args.gates))
    repeats = args.repeats or gates.repeats
    # Absolute: the bench binaries run under --cwd and get this path through the environment.
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cwd = Path(args.cwd).resolve() if args.cwd else Path.cwd()
    binaries = {"base": Path(args.base).resolve(), "pr": Path(args.pr).resolve()}
    extra_args = [arg for arg in args.bench_args if arg != "--"]
    results = {
        "gates": str(args.gates),
        "repeats": repeats,
        "filter": gates.filter,
        "bench_args": gates.bench_args + extra_args,
        "order": [],
        "base": {"binary": str(binaries["base"]), "runs": [], "seconds": []},
        "pr": {"binary": str(binaries["pr"]), "runs": [], "seconds": []},
    }
    for i, side in interleaved_order(repeats):
        home = out / side / f"run-{i}"
        started = time.monotonic()
        medians = run_once(
            binaries[side], home, cwd, gates, extra_args, out / side / f"run-{i}.log"
        )
        seconds = time.monotonic() - started
        results[side]["runs"].append(medians)
        results[side]["seconds"].append(round(seconds, 1))
        results["order"].append(f"{side}-{i}")
        print(f"[run {i}/{repeats}] {side}: {len(medians)} ids in {seconds:.0f} s", flush=True)
    results_path = out / "results.json"
    with results_path.open("w") as handle:
        json.dump(results, handle, indent=1, sort_keys=True)
    print(f"results: {results_path}", flush=True)
    return report(results, gates, out / "gate.md", args.summary)


# --------------------------------------------------------------------------- compare


@dataclass
class Row:
    bench_id: str
    base: list[float]
    pr: list[float]
    gate: Gate | None
    verdict: str = ""

    @property
    def base_median(self) -> float | None:
        return statistics.median(self.base) if self.base else None

    @property
    def pr_median(self) -> float | None:
        return statistics.median(self.pr) if self.pr else None

    @property
    def change(self) -> float | None:
        if not self.base or not self.pr:
            return None
        return self.pr_median / self.base_median - 1.0

    @property
    def spread(self) -> float | None:
        """The larger of the two sides' (max - min) / median: the noise of this job."""
        spreads = [
            (max(values) - min(values)) / statistics.median(values)
            for values in (self.base, self.pr)
            if values
        ]
        return max(spreads) if spreads else None

    @property
    def separated(self) -> bool:
        """Every pull-request run slower than every base run (or the reverse)."""
        if not self.base or not self.pr:
            return False
        return min(self.pr) > max(self.base) or max(self.pr) < min(self.base)

    @property
    def fails(self) -> bool:
        return self.verdict in ("regression", "missing")


def judge(row: Row) -> str:
    if not row.pr:
        return "missing" if row.gate else "missing (report only)"
    if not row.base:
        return "new"
    if row.gate is None:
        return "report only"
    change = row.change
    if change > row.gate.threshold:
        return "regression" if row.separated else "suspect (runs overlap)"
    if change < -row.gate.threshold and row.separated:
        return "improvement"
    return "ok"


def compare(results: dict, gates: Gates) -> list[Row]:
    per_side: dict[str, dict[str, list[float]]] = {side: {} for side in SIDES}
    for side in SIDES:
        for run in results[side]["runs"]:
            for bench_id, value in run.items():
                per_side[side].setdefault(bench_id, []).append(value)
    ids = sorted(set(per_side["base"]) | set(per_side["pr"]))
    rows = []
    for bench_id in ids:
        row = Row(
            bench_id,
            per_side["base"].get(bench_id, []),
            per_side["pr"].get(bench_id, []),
            gates.threshold_for(bench_id),
        )
        row.verdict = judge(row)
        rows.append(row)
    return rows


def fmt_time(ns: float, unit_ns: float) -> str:
    unit, scale = "ns", 1.0
    for candidate, candidate_scale in (("s", 1e9), ("ms", 1e6), ("µs", 1e3)):
        if unit_ns >= candidate_scale:
            unit, scale = candidate, candidate_scale
            break
    return f"{ns / scale:.4g} {unit}"


def fmt_side(values: list[float], unit_ns: float) -> str:
    if not values:
        return "-"
    median = statistics.median(values)
    if len(values) == 1:
        return fmt_time(median, unit_ns)
    return f"{fmt_time(median, unit_ns)} ({fmt_time(min(values), unit_ns)} - {fmt_time(max(values), unit_ns)})"


def fmt_pct(value: float | None, signed: bool = True) -> str:
    if value is None:
        return "-"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def markdown(results: dict, rows: list[Row]) -> str:
    repeats = results["repeats"]
    failed = [row for row in rows if row.fails]
    suspect = [row for row in rows if row.verdict.startswith("suspect")]
    verdict = "FAIL" if failed else "PASS"
    lines = [
        f"### Benchmark gate: {verdict}",
        "",
        f"Medians of {repeats} interleaved runs per side on this runner, (min - max) per side; "
        "spread = the larger side's (max - min) / median. A gated id fails when its change is "
        "worse than the threshold and every pull-request run is slower than every base run.",
        "",
        "| benchmark | base | pull request | change | spread | threshold | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        unit_ns = row.base_median or row.pr_median or 1.0
        threshold = fmt_pct(row.gate.threshold, signed=False) if row.gate else "-"
        mark = {"regression": "FAIL ", "missing": "FAIL ", "improvement": "faster "}.get(
            row.verdict, ""
        )
        lines.append(
            f"| {row.bench_id} | {fmt_side(row.base, unit_ns)} | {fmt_side(row.pr, unit_ns)} "
            f"| {fmt_pct(row.change)} | {fmt_pct(row.spread, signed=False)} | {threshold} "
            f"| {mark}{row.verdict} |"
        )
    lines.append("")
    gated = sum(1 for row in rows if row.gate is not None)
    lines.append(
        f"{len(rows)} ids, {gated} gated, {len(failed)} failed, {len(suspect)} suspect "
        f"(beyond the threshold, runs overlap: not failed)."
    )
    for row in failed:
        if row.verdict == "missing":
            lines.append(
                f"- `{row.bench_id}` is gated but the pull request's run has no such id: "
                "restore the benchmark, or exclude the old id from the gate pattern that "
                "matches it (a negative lookahead in the group pattern)."
            )
        else:
            lines.append(
                f"- `{row.bench_id}`: {fmt_pct(row.change)} against a {fmt_pct(row.gate.threshold, signed=False)} "
                f"threshold; base runs {[round(v) for v in row.base]} ns, pull request runs "
                f"{[round(v) for v in row.pr]} ns."
            )
    return "\n".join(lines) + "\n"


def report(results: dict, gates: Gates, gate_md: Path | None, summary: str | None) -> int:
    rows = compare(results, gates)
    text = markdown(results, rows)
    print(text, end="", flush=True)
    if gate_md is not None:
        gate_md.write_text(text)
    summary = summary or os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as handle:
            handle.write(text)
    return 1 if any(row.fails for row in rows) else 0


def cmd_compare(args: argparse.Namespace) -> int:
    gates = load_gates(Path(args.gates))
    with open(args.results) as handle:
        results = json.load(handle)
    return report(results, gates, None, args.summary)


# --------------------------------------------------------------------------- main


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build a bench binary and copy it to --out")
    build.add_argument("--out", required=True, help="where the bench executable is copied")
    build.add_argument("--name", help="the bench target when the command builds several")
    build.add_argument("cargo", nargs=argparse.REMAINDER, help="the cargo bench command")
    build.set_defaults(func=cmd_build)

    run = sub.add_parser("run", help="run base and PR binaries interleaved, then compare")
    run.add_argument("--gates", required=True, help="the TOML gate file")
    run.add_argument("--base", required=True, help="the base branch's bench binary")
    run.add_argument("--pr", required=True, help="the pull request's bench binary")
    run.add_argument("--out", required=True, help="output directory (results.json, logs)")
    run.add_argument("--cwd", help="working directory for the bench binaries")
    run.add_argument("--repeats", type=int, help="override the gate file's repeat count")
    run.add_argument("--summary", help="markdown file to append to (default: GITHUB_STEP_SUMMARY)")
    run.add_argument("bench_args", nargs=argparse.REMAINDER, help="extra bench arguments")
    run.set_defaults(func=cmd_run)

    cmp_ = sub.add_parser("compare", help="re-evaluate a saved results.json")
    cmp_.add_argument("--gates", required=True, help="the TOML gate file")
    cmp_.add_argument("--summary", help="markdown file to append to (default: GITHUB_STEP_SUMMARY)")
    cmp_.add_argument("results", help="results.json written by run")
    cmp_.set_defaults(func=cmd_compare)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        return args.func(args)
    except (RuntimeError, ValueError, TypeError, OSError) as err:
        print(f"bench_gate: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
