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

The gate file also names the bench sources: the files that feed the gated binary beyond the
crate it measures (the bench file and what it includes by path). The bench profile links one
codegen unit under thin LTO, so a row added to that file changes the inlining of the code the
other rows time, and the gated rows move with the measured crate untouched (+32 % on the
Symphony prose rows from a bench-file-only change, 2026-11-08). The base side is therefore
built with the pull request's bench sources (`build --sources-from`), so the two binaries
differ in the measured code alone, and the report names the case where those sources do not
build on the base. The rule for a change: a gated bench binary's sources stay byte-identical
unless the change is to what they measure, and a new row goes to a bench target of its own.

Four controls keep the verdict about the code. Binaries of the same bytes pass without a run:
no code change can move a row. Each side is measured from a fresh copy of its binary under the
output directory, through paths and criterion homes of one length: a file's cached pages decide
its speed on 1 to 6 µs rows (one binary read 9 to 20 % slower than its byte-identical copy on
whichever side it stood, 2026-11-09), so neither side carries its file's history; the copies go
after the runs, the output directory being the workflow's artifact. An id that fails after the
interleaved runs is measured once more per side before the verdict, and that round is read
within itself (a process that runs a few ids sits on levels of its own), so one burst on the
runner cannot fail it alone: a burst re-measures near zero and a regression near its size, so a
round under half the threshold makes the failure a suspect. The host's load before every run
is recorded,
and `--busy-above` makes the gate wait for a quieter host and then refuse to judge (exit 2,
never a failure) rather than read another job's load as a regression.

Usage:
    bench_gate.py build --out FILE [--name BENCH] [--gates FILE [--sources-from REV]]
                   -- cargo bench ... --no-run
        Run the cargo command and copy the bench executable it reports to FILE. With a gate
        file, record the sha256 of each bench source it names in FILE.sources.json; with a
        revision too, build with those files as REV has them, and fall back to the checkout's
        own when they do not build, saying so in the record.
    bench_gate.py run --gates FILE --base BIN --pr BIN --out DIR [--cwd DIR] [--repeats N]
                  [--recheck N] [--busy-above LOAD [--busy-wait S]] [--cpu N]
        Run both binaries interleaved, write DIR/results.json and DIR/gate.md, compare.
    bench_gate.py compare --gates FILE RESULTS_JSON [--summary FILE]
        Re-evaluate a saved results.json.

The markdown table goes to stdout and, when set, to $GITHUB_STEP_SUMMARY (or --summary).
Exit code 1 means a gated regression, 2 a usage or run error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
# The on-disk names of the two sides, of one length: the criterion homes, the logs and the
# copies the binaries run from, so the two processes have the same shape.
SIDE_DIRS = {"base": "base", "pr": "head"}


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
    sources: list[str] = field(default_factory=list)
    gates: list[Gate] = field(default_factory=list)

    def threshold_for(self, bench_id: str) -> Gate | None:
        """The first gate whose pattern matches wins; None means report only."""
        for gate in self.gates:
            if gate.matches(bench_id):
                return gate
        return None


GATE_FILE_KEYS = {"repeats", "filter", "bench_args", "sources", "gate"}
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
    sources = raw.get("sources", [])
    if not isinstance(sources, list) or not all(
        isinstance(source, str) and source for source in sources
    ):
        raise ValueError(f"{path}: sources must be a list of file paths, got {sources!r}")
    gates = Gates(
        repeats=repeats,
        filter=filter_,
        bench_args=[str(arg) for arg in bench_args],
        sources=sources,
    )
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


def run_cargo(cargo: list[str]) -> tuple[int, list[dict]]:
    """The cargo command's exit code and the JSON messages it printed."""
    print("+", " ".join(cargo), flush=True)
    proc = subprocess.run(cargo, stdout=subprocess.PIPE, text=True, check=False)
    messages = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            messages.append(json.loads(line))
    return proc.returncode, messages


def sources_at(paths: list[str], rev: str) -> dict[str, bytes | None]:
    """The listed files as the commit `rev` has them, None for one it does not have."""
    verify = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if verify.returncode != 0:
        raise ValueError(f"--sources-from {rev}: not a commit of this repository")
    contents: dict[str, bytes | None] = {}
    for path in paths:
        shown = subprocess.run(
            ["git", "show", f"{rev}:./{path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        contents[path] = shown.stdout if shown.returncode == 0 else None
    return contents


def place_sources(contents: dict[str, bytes | None]) -> dict[str, bytes | None]:
    """Write each file as given (remove it for None) and return what was there, for the way
    back."""
    before: dict[str, bytes | None] = {}
    for path, data in contents.items():
        file = Path(path)
        before[path] = file.read_bytes() if file.exists() else None
        if data is None:
            if file.exists():
                file.unlink()
        else:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(data)
    return before


def digests(paths: list[str]) -> dict[str, str | None]:
    """The sha256 of each listed file as it is now, None for one that is absent."""
    return {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None
        for path in paths
    }


def sources_record_path(binary: Path) -> Path:
    """Where `build` records the bench sources a binary was built from, beside it."""
    return binary.with_name(binary.name + ".sources.json")


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
    gates = load_gates(Path(args.gates)) if args.gates else None
    sources = gates.sources if gates else []
    if args.sources_from and not sources:
        print("build: --sources-from needs a --gates file that names sources", file=sys.stderr)
        return 2
    record: dict | None = None
    if args.sources_from:
        # The base side carries the pull request's bench sources, so the two binaries differ
        # in the measured code alone; the checkout is its own again before anything else runs.
        before = place_sources(sources_at(sources, args.sources_from))
        try:
            built = digests(sources)
            code, messages = run_cargo(cargo)
        finally:
            place_sources(before)
        record = {"rev": args.sources_from, "applied": code == 0, "files": built}
        if code != 0:
            record["reason"] = (
                f"The bench sources of {args.sources_from} did not build here "
                f"(cargo exited with {code})"
            )
            print(f"build: {record['reason']}, so the checkout's own are built", file=sys.stderr)
            record["files"] = digests(sources)
            code, messages = run_cargo(cargo)
    else:
        code, messages = run_cargo(cargo)
        if gates is not None:
            record = {"rev": None, "applied": None, "files": digests(sources)}
    if code != 0:
        print(f"build: cargo exited with {code}", file=sys.stderr)
        return 2
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
    if record is not None:
        sources_record_path(out).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        print(f"build: {len(sources)} bench sources recorded in {sources_record_path(out)}")
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
    bench_filter: str | None = None,
    cpu: int | None = None,
) -> dict[str, float]:
    home.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "--bench", "--noplot", *gates.bench_args, *extra_args]
    bench_filter = gates.filter if bench_filter is None else bench_filter
    if bench_filter:
        cmd.append(bench_filter)
    if cpu is not None:
        cmd = ["taskset", "-c", str(cpu), *cmd]
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


def round_sides(i: int) -> tuple[str, ...]:
    """The sides in the order run i takes them: odd runs base first, even runs the pull request
    first, so neither side always warms the other up."""
    return SIDES if i % 2 == 1 else tuple(reversed(SIDES))


def interleaved_order(repeats: int) -> list[tuple[int, str]]:
    """Every run of the gate, in order."""
    return [(i, side) for i in range(1, repeats + 1) for side in round_sides(i)]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_binary(binary: Path, copy: Path) -> Path:
    """`copy`, a fresh copy of `binary`: the side runs from pages this copy caches, through a
    path of the same length as the other side's."""
    copy.parent.mkdir(parents=True, exist_ok=True)
    if copy.is_symlink() or copy.exists():
        copy.unlink()
    shutil.copy2(binary, copy)
    return copy


def host_load() -> float | None:
    """The host's one-minute load average, None where /proc/loadavg is not readable."""
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def quiet_host(limit: float, wait_s: float, poll_s: float = 15.0) -> tuple[float | None, bool]:
    """Wait up to `wait_s` for the one-minute load to be at most `limit`: the load read last
    and whether the host is quiet (a host without a readable load counts as quiet)."""
    deadline = time.monotonic() + wait_s
    while True:
        load = host_load()
        if load is None or load <= limit:
            return load, True
        if time.monotonic() >= deadline:
            return load, False
        time.sleep(min(poll_s, max(0.0, deadline - time.monotonic())))


class BusyHost(RuntimeError):
    """The host stayed busier than `--busy-above` allows: the gate does not judge."""


def cmd_run(args: argparse.Namespace) -> int:
    gates = load_gates(Path(args.gates))
    repeats = args.repeats or gates.repeats
    # Absolute: the bench binaries run under --cwd and get this path through the environment.
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cwd = Path(args.cwd).resolve() if args.cwd else Path.cwd()
    binaries = {"base": Path(args.base).resolve(), "pr": Path(args.pr).resolve()}
    if args.cpu is not None and shutil.which("taskset") is None:
        print("run: --cpu needs taskset on this host", file=sys.stderr)
        return 2
    extra_args = [arg for arg in args.bench_args if arg != "--"]
    sources: dict[str, dict | None] = {}
    for side, binary in binaries.items():
        record = sources_record_path(binary)
        sources[side] = json.loads(record.read_text()) if record.exists() else None
    hashes = {side: sha256_of(binary) for side, binary in binaries.items()}
    results = {
        "gates": str(args.gates),
        "repeats": repeats,
        "filter": gates.filter,
        "bench_args": gates.bench_args + extra_args,
        "sources": sources,
        "binaries": {side: {"path": str(binaries[side]), "sha256": hashes[side]} for side in SIDES},
        "identical": hashes["base"] == hashes["pr"],
        "order": [],
        "base": {"binary": str(binaries["base"]), "runs": [], "seconds": [], "load": []},
        "pr": {"binary": str(binaries["pr"]), "runs": [], "seconds": [], "load": []},
    }
    results_path = out / "results.json"

    def save() -> None:
        with results_path.open("w") as handle:
            json.dump(results, handle, indent=1, sort_keys=True)

    if results["identical"]:
        # The same bytes on both sides: no code change can move a row, nothing to measure.
        save()
        print(f"results: {results_path}", flush=True)
        return report(results, gates, out / "gate.md", args.summary)
    # Each side from a fresh copy of its binary: a file's cached pages are not a side. The two
    # processes have one shape: paths and criterion homes of one length, the same environment.
    copies = {side: copy_binary(binaries[side], out / SIDE_DIRS[side] / "bench") for side in SIDES}

    def run(i: int, side: str, bench_filter: str | None, label: str) -> None:
        load = host_load()
        if args.busy_above is not None:
            load, quiet = quiet_host(args.busy_above, args.busy_wait)
            if not quiet:
                results["not_judged"] = (
                    f"host busy: the one-minute load {load:.1f} stayed above {args.busy_above:g} "
                    f"for {args.busy_wait:g} s before {label}"
                )
                raise BusyHost(results["not_judged"])
        side_dir = out / SIDE_DIRS[side]
        started = time.monotonic()
        medians = run_once(
            copies[side],
            side_dir / f"run-{i}",
            cwd,
            gates,
            extra_args,
            side_dir / f"run-{i}.log",
            bench_filter=bench_filter,
            cpu=args.cpu,
        )
        seconds = time.monotonic() - started
        results[side]["runs"].append(medians)
        results[side]["seconds"].append(round(seconds, 1))
        results[side]["load"].append(load)
        results["order"].append(f"{side}-{i}")
        print(f"[{label}] {side}: {len(medians)} ids in {seconds:.0f} s", flush=True)

    try:
        try:
            for i, side in interleaved_order(repeats):
                run(i, side, None, f"run {i}/{repeats}")
            # An id that fails after the full rounds is measured once more per side before
            # the verdict, in the interleaved order: one burst on the runner cannot fail it
            # alone. Every `--recheck` round runs the same ids, fixed here, so the median of
            # the rounds reads the same whichever round came first.
            failing = sorted(
                row.bench_id for row in compare(results, gates) if row.verdict == "regression"
            )
            if failing and args.recheck > 0:
                results["recheck"] = {"ids": failing, "rounds": 0}
                exact = "^(" + "|".join(re.escape(bench_id) for bench_id in failing) + ")$"
                for extra in range(1, args.recheck + 1):
                    i = repeats + extra
                    results["recheck"]["rounds"] = extra
                    for side in round_sides(i):
                        run(i, side, exact, f"recheck {extra}/{args.recheck}")
        except BusyHost as busy:
            save()
            print(f"run: not judged, {busy}", file=sys.stderr)
            report(results, gates, out / "gate.md", args.summary)
            return 2
        save()
        print(f"results: {results_path}", flush=True)
        return report(results, gates, out / "gate.md", args.summary)
    finally:
        # The copies served the runs. The output directory is what the workflows upload,
        # and two bench binaries do not belong in that artifact.
        for copy in copies.values():
            copy.unlink(missing_ok=True)


# --------------------------------------------------------------------------- compare


@dataclass
class Row:
    bench_id: str
    base: list[float]
    pr: list[float]
    gate: Gate | None
    verdict: str = ""
    # The re-measured rounds, (base, pull request) per round: a process that runs the
    # failing ids alone sits on levels of its own, so these are never pooled with the
    # full rounds; each round is read within itself.
    rechecks: list[tuple[float, float]] = field(default_factory=list)

    @property
    def base_median(self) -> float | None:
        return statistics.median(self.base) if self.base else None

    @property
    def pr_median(self) -> float | None:
        return statistics.median(self.pr) if self.pr else None

    @property
    def change(self) -> float | None:
        """The relative change of the medians against the base median's magnitude (an
        idle-subtracted count can sit below zero: a rise stays a rise); None without both
        sides, or when the base median is zero (a count that was zero before: the absolute
        values tell)."""
        if not self.base or not self.pr or not self.base_median:
            return None
        return (self.pr_median - self.base_median) / abs(self.base_median)

    @property
    def spread(self) -> float | None:
        """The larger of the two sides' (max - min) / median: the noise of this job."""
        spreads = []
        for values in (self.base, self.pr):
            if not values:
                continue
            median = statistics.median(values)
            if median:
                spreads.append((max(values) - min(values)) / abs(median))
            elif max(values) != min(values):
                return None
            else:
                spreads.append(0.0)
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

    @property
    def recheck_change(self) -> float | None:
        """The median of the re-measured rounds' own changes (a round with a zero base:
        +inf for a rise, -inf otherwise); None without a re-measurement."""
        if not self.rechecks:
            return None
        changes = []
        for base, pr in self.rechecks:
            if base:
                changes.append((pr - base) / abs(base))
            else:
                changes.append(math.inf if pr > base else -math.inf)
        return statistics.median(changes)

    @property
    def confirmed(self) -> bool | None:
        """Whether the re-measured rounds agree with the failure. Their job is to rule out
        a burst, not to measure the change again: a burst re-measures near zero and a
        regression near its size, so the bar is half the threshold (a zero base: any
        rise) on the median of the rounds; None without a re-measurement."""
        change = self.recheck_change
        if change is None:
            return None
        threshold = self.gate.threshold if self.gate else 0.0
        return change > threshold / 2


def judge(row: Row) -> str:
    if not row.pr:
        return "missing" if row.gate else "missing (report only)"
    if not row.base:
        return "new"
    if row.gate is None:
        return "report only"
    change = row.change
    if change is None:
        # A zero base median: the absolute head values tell. Above zero, every
        # head run above every base run is a regression and an overlap is
        # suspect, as for a change beyond the threshold; below zero and
        # separated, an improvement.
        if row.pr_median > 0:
            return regression_or_suspect(row)
        if row.pr_median < 0 and row.separated:
            return "improvement"
        return "ok"
    if change > row.gate.threshold:
        return regression_or_suspect(row)
    if change < -row.gate.threshold and row.separated:
        return "improvement"
    return "ok"


def regression_or_suspect(row: Row) -> str:
    """Beyond the threshold: a regression when every pull-request run is worse than every
    base run and the re-measured round, read within itself, is beyond half the threshold;
    otherwise suspect."""
    if not row.separated:
        return "suspect (runs overlap)"
    if row.confirmed is False:
        return "suspect (not confirmed by the re-measurement)"
    return "regression"


def compare(results: dict, gates: Gates) -> list[Row]:
    # The first `repeats` runs of a side are the full rounds; the runs after them are the
    # re-measured rounds, kept apart from the full rounds and paired by round.
    repeats = results.get("repeats") or len(results["base"]["runs"])
    per_side: dict[str, dict[str, list[float]]] = {side: {} for side in SIDES}
    extra: dict[str, dict[str, dict[int, float]]] = {side: {} for side in SIDES}
    for side in SIDES:
        for i, run in enumerate(results[side]["runs"]):
            for bench_id, value in run.items():
                if i < repeats:
                    per_side[side].setdefault(bench_id, []).append(value)
                else:
                    extra[side].setdefault(bench_id, {})[i] = value
    ids = sorted(set(per_side["base"]) | set(per_side["pr"]))
    rows = []
    for bench_id in ids:
        base_extra = extra["base"].get(bench_id, {})
        pr_extra = extra["pr"].get(bench_id, {})
        rounds = sorted(set(base_extra) & set(pr_extra))
        row = Row(
            bench_id,
            per_side["base"].get(bench_id, []),
            per_side["pr"].get(bench_id, []),
            gates.threshold_for(bench_id),
            rechecks=[(base_extra[i], pr_extra[i]) for i in rounds],
        )
        row.verdict = judge(row)
        rows.append(row)
    # (3) A run the busy host stopped: the rows it left one-sided are not missing ids.
    if results.get("not_judged"):
        for row in rows:
            if not row.base or not row.pr:
                row.verdict = "not judged (the run stopped)"
    return rows


def fmt_time(ns: float, unit_ns: float) -> str:
    unit, scale = "ns", 1.0
    for candidate, candidate_scale in (("s", 1e9), ("ms", 1e6), ("µs", 1e3)):
        if unit_ns >= candidate_scale:
            unit, scale = candidate, candidate_scale
            break
    return f"{ns / scale:.4g} {unit}"


def fmt_value(value: float, unit_ns: float, unit: str | None) -> str:
    """A criterion median is nanoseconds and is scaled to a time unit; a results file
    that names its own `unit` (an empty string for a bare count: the ids then carry the
    unit in their names) prints its values as they are, with that unit."""
    if unit is None:
        return fmt_time(value, unit_ns)
    return f"{value:.4g} {unit}".rstrip()


def fmt_side(values: list[float], unit_ns: float, unit: str | None = None) -> str:
    if not values:
        return "-"
    median = statistics.median(values)
    if len(values) == 1:
        return fmt_value(median, unit_ns, unit)
    return (
        f"{fmt_value(median, unit_ns, unit)} "
        f"({fmt_value(min(values), unit_ns, unit)} - {fmt_value(max(values), unit_ns, unit)})"
    )


def fmt_run(value: float) -> str:
    """One run's value: six significant digits below a million, above it the plain integer
    with separators (a millisecond row in nanoseconds reads `1,234,567`, never `1.23e+06`)."""
    return f"{value:.6g}" if abs(value) < 1e6 else f"{value:,.0f}"


def fmt_runs(values: list[float]) -> str:
    """The per-run values of one side, compact."""
    return "[" + ", ".join(fmt_run(value) for value in values) + "]"


def fmt_rechecks(row: Row, unit_suffix: str) -> str:
    """The re-measured rounds of a row, each read within itself."""
    parts = []
    for base, pr in row.rechecks:
        change = (pr - base) / abs(base) if base else None
        parts.append(
            f"base {fmt_run(base)}{unit_suffix}, pull request {fmt_run(pr)}{unit_suffix} "
            f"({fmt_pct(change)})"
        )
    return "; re-measured " + "; ".join(parts) if parts else ""


def fmt_pct(value: float | None, signed: bool = True) -> str:
    if value is None:
        return "-"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def sources_note(sources: dict | None) -> str:
    """One line on the bench sources the two binaries were built from, when both builds
    recorded them: byte-identical, so the rows compare the measured code alone, or the files
    that differ and why the base side does not carry the pull request's."""
    if not sources or any(sources.get(side) is None for side in SIDES):
        return ""
    base, pr = sources["base"]["files"], sources["pr"]["files"]
    files = sorted(set(base) | set(pr))
    if not files:
        return ""
    differing = [path for path in files if base.get(path) != pr.get(path)]
    named = f"Bench sources ({len(files)} named by the gate file):"
    if not differing:
        return (
            f"{named} byte-identical in the two binaries, so the rows compare the measured "
            "code alone."
        )
    listed = ", ".join(f"`{path}`" for path in differing)
    verb = "differs" if len(differing) == 1 else "differ"
    why = ""
    if sources["base"].get("applied") is False:
        reason = sources["base"].get("reason", "the pull request's were not applied")
        why = f" {reason}, so the base's binary carries its own."
    return (
        f"{named} {listed} {verb} between the two binaries.{why} A gated row that moves "
        "without a change to what it measures is the binary's inlining of a neighbour: a row "
        "this pull request adds goes to a bench target of its own."
    )


def markdown(results: dict, rows: list[Row]) -> str:
    repeats = results["repeats"]
    # Criterion results carry nanoseconds; another producer (crates/gateway_perf)
    # names the unit of its values, "" when the id names it. Lower is better either way.
    unit = results.get("unit")
    if results.get("identical"):
        sha = results["binaries"]["base"]["sha256"]
        return (
            "### Benchmark gate: PASS (identical binaries)\n\n"
            f"The base and pull-request bench binaries are the same bytes (sha256 {sha[:16]}...): "
            "no code change can move a row, so nothing was run.\n"
        )
    failed = [row for row in rows if row.fails]
    suspect = [row for row in rows if row.verdict.startswith("suspect")]
    recheck = results.get("recheck") or {}
    rechecked = set(recheck.get("ids", []))
    not_judged = results.get("not_judged")
    verdict = "NOT JUDGED" if not_judged else ("FAIL" if failed else "PASS")
    lines = [
        f"### Benchmark gate: {verdict}",
        "",
        f"Medians of {repeats} interleaved runs per side on this runner, (min - max) per side; "
        "spread = the larger side's (max - min) / median. A gated id fails when its change is "
        "worse than the threshold and every pull-request run is worse (higher) than every base run.",
        "",
    ]
    if not_judged:
        lines += [
            f"Not judged: {not_judged}. The runs below are the ones that completed; "
            "run the gate again on a quiet host.",
            "",
        ]
    if rechecked:
        lines += [
            f"Re-measured before the verdict, {recheck.get('rounds', 1)} more run(s) per side in the "
            f"interleaved order: {', '.join(f'`{bench_id}`' for bench_id in sorted(rechecked))}.",
            "",
        ]
    if any(results.get(side, {}).get("load") for side in SIDES):
        shown = ", ".join(
            f"{side} {[round(v, 1) for v in results[side]['load'] if v is not None]}"
            for side in SIDES
        )
        lines += [f"Host load (one minute) before each run: {shown}.", ""]
    note = sources_note(results.get("sources"))
    if note:
        lines += [note, ""]
    lines += [
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
            f"| {row.bench_id} | {fmt_side(row.base, unit_ns, unit)} | {fmt_side(row.pr, unit_ns, unit)} "
            f"| {fmt_pct(row.change)} | {fmt_pct(row.spread, signed=False)} | {threshold} "
            f"| {mark}{row.verdict} |"
        )
    lines.append("")
    gated = sum(1 for row in rows if row.gate is not None)
    lines.append(
        f"{len(rows)} ids, {gated} gated, {len(failed)} failed, {len(suspect)} suspect "
        "(beyond the threshold, the runs overlap or the re-measurement did not confirm: "
        "not failed)."
    )
    unit_suffix = " ns" if unit is None else (f" {unit}" if unit else "")
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
                f"threshold; base runs {fmt_runs(row.base)}{unit_suffix}, pull request runs "
                f"{fmt_runs(row.pr)}{unit_suffix}{fmt_rechecks(row, unit_suffix)}."
            )
    # The per-run values of every suspect or re-measured row, so a bimodal set is visible here.
    for row in rows:
        if row.verdict.startswith("suspect") or (row.bench_id in rechecked and not row.fails):
            lines.append(
                f"- `{row.bench_id}`: {row.verdict}, {fmt_pct(row.change)}; base runs "
                f"{fmt_runs(row.base)}{unit_suffix}, pull request runs "
                f"{fmt_runs(row.pr)}{unit_suffix}{fmt_rechecks(row, unit_suffix)}."
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
    if results.get("not_judged"):
        return 2
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
    build.add_argument("--gates", help="the TOML gate file: its bench sources are recorded")
    build.add_argument(
        "--sources-from",
        metavar="REV",
        help="build with the gate file's bench sources as this commit has them "
        "(the base side carries the pull request's)",
    )
    build.add_argument("cargo", nargs=argparse.REMAINDER, help="the cargo bench command")
    build.set_defaults(func=cmd_build)

    run = sub.add_parser("run", help="run base and PR binaries interleaved, then compare")
    run.add_argument("--gates", required=True, help="the TOML gate file")
    run.add_argument("--base", required=True, help="the base branch's bench binary")
    run.add_argument("--pr", required=True, help="the pull request's bench binary")
    run.add_argument("--out", required=True, help="output directory (results.json, logs)")
    run.add_argument("--cwd", help="working directory for the bench binaries")
    run.add_argument("--repeats", type=int, help="override the gate file's repeat count")
    run.add_argument(
        "--recheck",
        type=int,
        default=1,
        metavar="N",
        help="more runs per side for the ids that fail, before the verdict (default 1, 0 for none)",
    )
    run.add_argument("--cpu", type=int, metavar="N", help="pin the bench processes to this CPU")
    run.add_argument(
        "--busy-above",
        type=float,
        metavar="LOAD",
        help="wait for the host's one-minute load to be at most LOAD before a run, "
        "then refuse to judge",
    )
    run.add_argument(
        "--busy-wait",
        type=float,
        default=600.0,
        metavar="S",
        help="how long --busy-above waits for a quiet host (default 600 s)",
    )
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
