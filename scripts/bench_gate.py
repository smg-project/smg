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
no code change can move a row. So do two that load the same image, the bytes the loader maps
with the build id left out: a change outside a bench's code still moves its executable's
symbol names, debug sections and build id, and such a pair would be timed against itself. Each
side is measured from a fresh copy of its binary under the output directory, through paths and
criterion homes of one length: a file's cached pages decide its speed on 1 to 6 µs rows (one
binary read 9 to 20 % slower than its byte-identical copy on whichever side it stood,
2026-11-09), so neither side carries its file's history; the copies go after the runs, the
output directory being the workflow's artifact. An id that fails after the interleaved runs is
measured once more per side before the verdict, and that round is read within itself (a process
that runs a few ids sits on levels of its own), so one burst on the runner cannot fail it
alone: a burst re-measures near zero and a regression near its size, so a round under half the
threshold makes the failure a suspect. The host's load before every run is recorded, and
`--busy-above` makes the gate wait for a quieter host and then refuse to judge (exit 2, never a
failure) rather than read another job's load as a regression.

A fifth control tells where code sits from what it does. The functions of a bench executable
move when code ahead of them grows, and the same machine code further on in its page can read
slower by time alone: +10.4 % on a 330 ns row with every run apart, the row's instruction
count equal on both sides (2026-11-15). With `count = true` in the gate file, an id that still
fails after its re-measurement is counted on both executables: the bench under valgrind's
callgrind in the harness's profile mode, one profile after each batch of iterations. The same
instruction count in every batch both ran after the first is the same work. The id is then a
placement, reported with the count and the two places and not failed, when two more facts
hold: the function the harness spent the most in starts at another address in the pull
request's executable, and the change is within twice the id's threshold. The same count with
that function where it was leaves the failure, with the count in its line. An id whose change
is beyond twice its threshold is counted too, though no count makes it a placement: a count
that reads level asks for the second measurement below. Two counts that differ are each taken
once more, and neither is a placement: the count moved, or it is unsteady on an executable
(its work follows the clock, a seed or the allocator's state) and says nothing of work. A
count that was not read (no valgrind on the host, one batch alone) leaves the failure too. The
count ends by the run's budget: `run_budget`, the seconds from the run's start that a gate
file asking the count must name, below its gate step's bound by the report's room. No count
starts with less than 30 s of it left, and one that starts is capped by what is left. The ids
are counted in name order: the id whose turn the budget cut and the ids behind it keep the
time's verdict, their lines saying so, and the report is written. The
count cannot see a change that runs the same instructions slower, a load that misses or a
branch that stops predicting: where the function moved too, or for an id that times a whole
pass on any level count, such a row reads as a placement up to twice its threshold, by name,
in the summary. A placement is still a row to read.

An id that times a whole pass, many functions over many pages as a decision of the policy
engine is, is marked in its gate entry: `whole_pass = true`. Its time follows which of the
pass's code lines share a set of the processor's op cache, and every change to the executable
draws that again. A change to the pass itself moves its count by a few instructions, so the
same count is not there to read: a pull request read two decide ids 11.9 and 15.2 % slower on
6 and 2 instructions more an iteration, 0.03 and 0.04 % of their counts, and the same two
sources built without the alignment read them 3.8 and 5.1 % faster (2026-11-16). For a marked
id the placement's first fact is a level count: the two counts within half a percent of each
other (`LEVEL_COUNT`) at the last batch that differs among those all four counts ran, each
executable's own two counts the same. A batch a second count did not run grants no level
count, and it refuses one where the first counts differ last there, further apart than the
level: work that grows with the iterations can read under the level one batch early. No
function's address is asked, since any line of the pass can move. The ceiling of twice the
threshold holds as for every id, and a count that moved further leaves the failure as before:
work inserted into the decision read the same two ids 16.2 and 11.3 % slower on counts 3.1 and
3.4 % higher. Half a percent of a count carries a twentieth of a 10 % threshold by itself: the
rest of such a time is not work.

A sixth control asks the clock again where the count is level. The full rounds and the
re-measured round run from one copy of each executable, and share whatever that copy's pages
carry: four pull requests failed on ids of 110 to 270 ns that no line of theirs reaches, the
same instruction count on both executables at the same address, the re-measured round
agreeing, and each passed when its job ran again (2026-11-18). An id that still fails on a
count that reads level is therefore measured again before it fails: the gate's N interleaved
runs a side once more, from a fresh copy of each executable, read within itself. A count reads
level when it is the same on both executables, or when each executable was counted twice and
the four counts lie within `LEVEL_COUNT` of each other at every batch where two of them last
differ. A count that moved and one that is not steady are one noise in two states, and inside
the level neither carries a failure by the clock. The id fails where the second measurement
too is beyond its threshold with every pull-request run worse than every base run, and is
reported as not repeated otherwise. No threshold moves, and a count beyond the level fails as
before. An id whose second measurement the run's budget has no room for keeps the first: it
starts only with room for all of its runs, by the longest run of the failing ids so far.

Usage:
    bench_gate.py build --out FILE [--name BENCH] [--gates FILE [--sources-from REV]]
                   -- cargo bench ... --no-run
        Run the cargo command and copy the bench executable it reports to FILE. With a gate
        file, record the sha256 of each bench source it names in FILE.sources.json; with a
        revision too, build with those files as REV has them, and fall back to the checkout's
        own when they do not build, saying so in the record.
    bench_gate.py run --gates FILE --base BIN --pr BIN --out DIR [--cwd DIR] [--repeats N]
                  [--recheck N] [--busy-above LOAD [--busy-wait S]] [--cpu N]
        Run both binaries interleaved, write DIR/results.json and DIR/gate.md, compare. With
        `count = true` in the gate file, the ids that still fail are counted first (valgrind).
    bench_gate.py compare --gates FILE RESULTS_JSON [--summary FILE]
        Re-evaluate a saved results.json.

The markdown table goes to stdout and, when set, to $GITHUB_STEP_SUMMARY (or --summary).
Exit code 1 means a gated regression, 2 a usage or run error.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import statistics
import struct
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
    # The id times a whole pass: at the count a level count places it (`LEVEL_COUNT`).
    whole_pass: bool = False

    def matches(self, bench_id: str) -> bool:
        return re.search(self.pattern, bench_id) is not None


# No count starts with less than this left of the run's budget (`run_budget` in the gate file,
# seconds from the run's start): a count is the harness's one-second warm-up under an emulator,
# and one the budget cuts short only fails. A step's bound that ends the script takes the
# report with it, so a gate file sets its budget below that bound by the report's room.
COUNT_FLOOR_S = 30.0


@dataclass
class Gates:
    repeats: int = 3
    filter: str | None = None
    bench_args: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    # An id that fails by time is counted on both executables before the verdict.
    count: bool = False
    # Seconds from the run's start by which the count ends; a gate file that counts names it.
    run_budget: float | None = None
    gates: list[Gate] = field(default_factory=list)

    def threshold_for(self, bench_id: str) -> Gate | None:
        """The first gate whose pattern matches wins; None means report only."""
        for gate in self.gates:
            if gate.matches(bench_id):
                return gate
        return None


GATE_FILE_KEYS = {"repeats", "filter", "bench_args", "sources", "count", "run_budget", "gate"}
GATE_ENTRY_KEYS = {"pattern", "threshold", "note", "whole_pass"}


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
    count = raw.get("count", False)
    if not isinstance(count, bool):
        raise ValueError(f"{path}: count must be true or false, got {count!r}")
    run_budget = raw.get("run_budget")
    if run_budget is not None:
        if isinstance(run_budget, bool) or not isinstance(run_budget, (int, float)):
            raise ValueError(f"{path}: run_budget must be a number of seconds, got {run_budget!r}")
        try:
            seconds = float(run_budget)
        except OverflowError:  # an integer past a float's range
            seconds = math.inf
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"{path}: run_budget must be finite and above zero: {run_budget!r}")
    elif count:
        raise ValueError(
            f"{path}: count = true needs run_budget, the seconds from the run's start by which "
            "the count ends (below the gate step's bound)"
        )
    gates = Gates(
        repeats=repeats,
        filter=filter_,
        bench_args=[str(arg) for arg in bench_args],
        sources=sources,
        count=count,
        run_budget=None if run_budget is None else float(run_budget),
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
        whole_pass = entry.get("whole_pass", False)
        if not isinstance(whole_pass, bool):
            raise ValueError(
                f"{path}: [[gate]] entry {index}: whole_pass must be true or false, "
                f"got {whole_pass!r}"
            )
        if whole_pass and not gates.count:
            raise ValueError(
                f"{path}: [[gate]] entry {index}: whole_pass = true needs count = true, "
                "the mark is read at the count"
            )
        gates.gates.append(Gate(pattern, threshold, str(entry.get("note", "")), whole_pass))
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


def exactly(ids: list[str]) -> str:
    """The filter that runs `ids` and no other id."""
    return "^(" + "|".join(re.escape(bench_id) for bench_id in ids) + ")$"


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


# The fields of an ELF header that name its section table (e_shoff; e_shentsize, e_shnum and
# e_shstrndx): the loader maps them with the first segment, and no run reads them.
ELF_SECTION_FIELDS = ((0x28, 0x30), (0x3A, 0x40))
# The sizes of the file header and of one program header of a 64-bit ELF file.
ELF_EHDR, ELF_PHDR = 64, 56
PT_LOAD, PT_INTERP, PT_NOTE, PT_GNU_PROPERTY, NT_GNU_BUILD_ID = 1, 3, 4, 0x6474E553, 3
# The program headers whose bytes a loader reads from the file itself: the interpreter's path
# and the notes, of which the property note has a header of its own. A run reads what any
# other header names from memory, as the loadable segments map it.
READ_FROM_FILE = (PT_INTERP, PT_NOTE, PT_GNU_PROPERTY)


def image_of(path: Path) -> str | None:
    """The sha256 of what the loader maps of a 64-bit little-endian ELF executable: its
    program headers and the file bytes of every loadable segment, with the build id and the
    header's section-table fields left out. Those follow what no run reads: symbol names and
    debug sections. None for any other file, for one whose headers do not add up, and for one
    whose file header, interpreter's path or notes do not lie whole inside one loadable
    segment: a loader reads those from the file, and the image would not hold them. A header
    of any other kind may name more file bytes than its segment holds, as the range made
    read-only after relocation does in some linked files: no loader reads them from the file."""
    data = bytearray(path.read_bytes())
    if len(data) < ELF_EHDR or data[:6] != b"\x7fELF\x02\x01":
        return None
    (phoff,) = struct.unpack_from("<Q", data, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
    if phentsize != ELF_PHDR or phnum in (0, 0xFFFF) or phoff + phnum * ELF_PHDR > len(data):
        return None
    table = bytes(data[phoff : phoff + phnum * ELF_PHDR])
    for start, end in ELF_SECTION_FIELDS:
        data[start:end] = bytes(end - start)
    loads, spans = [], [(0, ELF_EHDR)]
    for n in range(phnum):
        kind, _, offset, _, _, filesz, _, _ = struct.unpack_from("<IIQQQQQQ", table, n * ELF_PHDR)
        if offset + filesz > len(data):
            return None
        if kind == PT_LOAD:
            loads.append((offset, filesz))
        elif kind in READ_FROM_FILE and filesz:
            spans.append((offset, filesz))
        at, end = offset, offset + filesz
        while kind == PT_NOTE and at + 12 <= end:
            namesz, descsz, note = struct.unpack_from("<III", data, at)
            desc = at + 12 + (namesz + 3) // 4 * 4
            if desc + descsz > end:
                return None
            if note == NT_GNU_BUILD_ID and data[at + 12 : at + 12 + namesz] == b"GNU\x00":
                data[desc : desc + descsz] = bytes(descsz)
            at = desc + (descsz + 3) // 4 * 4
    for offset, filesz in spans:
        if not any(at <= offset and offset + filesz <= at + size for at, size in loads):
            return None
    digest = hashlib.sha256(table)
    for offset, filesz in loads:
        digest.update(data[offset : offset + filesz])
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
    run_started = time.monotonic()
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
    # Two files of other bytes are one program where the loader maps the same image of both.
    images = {side: image_of(binary) for side, binary in binaries.items()}
    same_image = images["base"] is not None and images["base"] == images["pr"]
    results = {
        "gates": str(args.gates),
        "repeats": repeats,
        "filter": gates.filter,
        "bench_args": gates.bench_args + extra_args,
        "sources": sources,
        "binaries": {
            side: {"path": str(binaries[side]), "sha256": hashes[side], "image": images[side]}
            for side in SIDES
        },
        # True of two files that load one image too: the saved key keeps its name.
        "identical": hashes["base"] == hashes["pr"] or same_image,
        "order": [],
        "base": {"binary": str(binaries["base"]), "runs": [], "seconds": [], "load": []},
        "pr": {"binary": str(binaries["pr"]), "runs": [], "seconds": [], "load": []},
    }
    results_path = out / "results.json"

    def save() -> None:
        with results_path.open("w") as handle:
            json.dump(results, handle, indent=1, sort_keys=True)

    if results["identical"]:
        # The same bytes on both sides, or the same loaded image: no code change can move a
        # row, nothing to measure.
        save()
        print(f"results: {results_path}", flush=True)
        return report(results, gates, out / "gate.md", args.summary)
    # Each side from a fresh copy of its binary: a file's cached pages are not a side. The two
    # processes have one shape: paths and criterion homes of one length, the same environment.
    copies = {side: copy_binary(binaries[side], out / SIDE_DIRS[side] / "bench") for side in SIDES}

    # The copies of the second measurement: none until an id asks for it.
    fresh: dict[str, Path] = {}

    def run(i: int, side: str, bench_filter: str | None, label: str, second: bool = False) -> None:
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
        stem = f"second-{i}" if second else f"run-{i}"
        medians = run_once(
            (fresh if second else copies)[side],
            side_dir / stem,
            cwd,
            gates,
            extra_args,
            side_dir / f"{stem}.log",
            bench_filter=bench_filter,
            cpu=args.cpu,
        )
        seconds = time.monotonic() - started
        sink = results["second"] if second else results
        sink[side]["runs"].append(medians)
        sink[side]["seconds"].append(round(seconds, 1))
        sink[side]["load"].append(load)
        sink["order"].append(f"{side}-{i}")
        print(f"[{label}] {side}: {len(medians)} ids in {seconds:.0f} s", flush=True)

    def measure_again(deadline: float) -> None:
        """The second measurement: an id that still fails on a count that reads level is
        measured again before it fails, `repeats` interleaved runs a side from a fresh copy of
        each executable, kept apart from the first (`results["second"]`). The full rounds and
        the re-measured round share one copy a side, and what that copy's pages carry. It
        starts only with room for all of its runs before `deadline`, by the longest run of
        the failing ids so far; without that room the first measurement decides."""
        level = sorted(
            row.bench_id
            for row in compare(results, gates)
            if row.verdict == "regression" and row.reads_level
        )
        if not level:
            return
        second = results["second"] = {"ids": level, "order": []}
        for side in SIDES:
            second[side] = {"runs": [], "seconds": [], "load": []}
        save()
        timed = [results[side]["seconds"] for side in SIDES]
        alone = [took for seconds in timed for took in seconds[repeats:]]
        longest = max(alone or [took for seconds in timed for took in seconds])
        if time.monotonic() + 2 * repeats * longest > deadline:
            second["skipped"] = "budget"
            return
        for side in SIDES:
            fresh[side] = copy_binary(binaries[side], out / SIDE_DIRS[side] / "second")
        exact = exactly(level)
        for i, side in interleaved_order(repeats):
            run(i, side, exact, f"second measurement {i}/{repeats}", second=True)

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
                exact = exactly(failing)
                for extra in range(1, args.recheck + 1):
                    i = repeats + extra
                    results["recheck"]["rounds"] = extra
                    for side in round_sides(i):
                        run(i, side, exact, f"recheck {extra}/{args.recheck}")
            # An id that still fails is counted on both executables: on the same instruction
            # count its time can be a placement in the bench executable, not work. A change
            # beyond twice the threshold is no placement whatever its count reads, and it is
            # counted all the same: a count that reads level asks for a second measurement.
            still = [row for row in compare(results, gates) if row.verdict == "regression"]
            if still and gates.count:
                # The timed rounds are written first: a count that hangs, or a job cancelled
                # in it, still leaves them in the output directory.
                save()
                based = sorted(row.bench_id for row in still if row.change is not None)
                results["count"] = {
                    row.bench_id: {"skipped": "no_base"} for row in still if row.change is None
                }
                deadline = run_started + (gates.run_budget or math.inf)
                results["count"].update(
                    count_ids(based, copies, out, cwd, gates, extra_args, args.cpu, deadline)
                )
                measure_again(deadline)
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
        for copy in (*copies.values(), *fresh.values()):
            copy.unlink(missing_ok=True)


# --------------------------------------------------------------------------- count

# The harness's functions that time a routine, in callgrind's demangled names under either
# symbol mangling. The harness calls one of them once for a batch of iterations.
HARNESS = re.compile(r"criterion::bencher::(?:Async)?Bencher\b.*::iter")
# The same functions as callgrind's own wildcard: it writes one profile after each call.
HARNESS_WILDCARD = "*criterion::bencher::*Bencher*::iter*"
# A count is the harness's one-second warm-up under an emulator: 2 to 3 s where it was read,
# never minutes. A row takes four counts at most.
COUNT_TIMEOUT_S = 120.0
# What a placement can excuse, in thresholds of its id. Code that moved was read to cost
# +10.4 % at most on rows gated at 10 %: a change beyond twice the threshold on the same
# instructions is more than where they sit has explained.
PLACEMENT_CEILING = 2.0
# What part of a count still reads as level. An id that times a whole pass (`whole_pass` in
# its gate entry) is placed on a level count, and any id that fails on one is measured again
# first. A change to the pass moves its count by a few instructions (0.03 and 0.04 % where
# two decide ids read 11.9 and 15.2 % slower by where their code sat), and work inserted into
# the decision that read the same two ids 16.2 and 11.3 % slower moved their counts by 3.1
# and 3.4 %.
LEVEL_COUNT = 0.005


@dataclass
class Batch:
    """What one callgrind profile holds of the harness's timed calls."""

    # The instructions run under the calls into the harness's timing functions.
    count: int = 0
    calls: int = 0
    # The function those calls spent the most instructions in, and where its code starts.
    unit: str = ""
    address: int | None = None


def read_profile(path: Path) -> Batch:
    """The harness's timed calls in one callgrind profile. In the profile a `fn=` line opens a
    function's block; under it a `cfn=` line names a callee, the `calls=` line after it carries
    the number of calls and the callee's address, and the next line their cost, the callee's
    own callees included. A name is spelled once, with a number, and is that number after."""
    names: dict[str, str] = {}
    positions = 1
    function = callee = ""
    edge: tuple[int, int | None] | None = None
    batch = Batch()
    harness: tuple[str, int | None] = ("", None)
    callees: dict[str, tuple[int, int | None]] = {}
    with path.open(encoding="utf-8", errors="replace") as lines:
        for raw in lines:
            line = raw.rstrip("\n")
            if not line:
                continue
            if line[0].isdigit() or line[0] in "+-*":
                if edge is None:
                    continue
                (calls, address), edge = edge, None
                fields = line.split()
                cost = int(fields[positions]) if len(fields) > positions else 0
                if HARNESS.search(function):
                    spent = callees.get(callee, (0, address))[0]
                    callees[callee] = (spent + cost, address)
                elif HARNESS.search(callee):
                    batch.count += cost
                    batch.calls += calls
                    harness = (callee, address)
                continue
            key, sep, value = line.partition("=")
            if not sep:
                if line.startswith("positions:"):
                    positions = len(line.split()) - 1
                continue
            if key in ("fn", "cfn"):
                numbered = re.match(r"\((\d+)\)(?: (.*))?$", value)
                if numbered is None:
                    name = value
                else:
                    if numbered.group(2) is not None:
                        names[numbered.group(1)] = numbered.group(2)
                    name = names.get(numbered.group(1), "")
                if key == "fn":
                    function = name
                else:
                    callee = name
            elif key == "calls":
                fields = value.split()
                address = None
                if positions > 1 and len(fields) > 1 and fields[1].startswith("0x"):
                    address = int(fields[1], 16)
                edge = (int(fields[0]), address)
    # The unit is the callee the timed calls spent most of their instructions in; a routine
    # inlined into the harness's function has none, and the unit is that function.
    batch.unit, batch.address = harness
    if callees:
        name, (spent, address) = max(callees.items(), key=lambda item: item[1][0])
        if 2 * spent >= batch.count:
            batch.unit, batch.address = name, address
    return batch


def count_once(
    binary: Path,
    work: Path,
    cwd: Path,
    gates: Gates,
    extra_args: list[str],
    bench_id: str,
    cpu: int | None,
    bound: float,
) -> dict:
    """The instruction counts of one id on one executable: the bench in the harness's profile
    mode under callgrind, which writes one profile after each batch of iterations (the mode
    is the harness's warm-up: one second of batches of 1, 2, 4, ... iterations, whatever the
    bench configures), ended after `bound` seconds. Returns the batches' counts in order and
    the last batch's unit; a `RuntimeError` says why nothing was read."""
    valgrind = shutil.which("valgrind")
    if valgrind is None:
        raise RuntimeError("valgrind is not on this host")
    work.mkdir(parents=True, exist_ok=True)
    cmd = [
        valgrind,
        "--tool=callgrind",
        "--dump-instr=yes",
        f"--dump-after={HARNESS_WILDCARD}",
        f"--callgrind-out-file={work / 'profile'}.%p",
        str(binary),
        "--bench",
        "--noplot",
        *gates.bench_args,
        *extra_args,
        "--profile-time",
        "1",
        f"^{re.escape(bench_id)}$",
    ]
    if cpu is not None:
        cmd = ["taskset", "-c", str(cpu), *cmd]
    log_path = work / "count.log"
    env = dict(os.environ, CRITERION_HOME=str(work / "home"))
    try:
        with log_path.open("w") as log:
            log.write("+ " + " ".join(cmd) + "\n")
            log.flush()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=cwd,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=bound,
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"the count ran over {bound:g} s; see {log_path}") from None
        if proc.returncode != 0:
            raise RuntimeError(f"the count exited with {proc.returncode}; see {log_path}")
        # callgrind names a profile by its process and numbers the one it writes after each
        # batch; the one without a number is a process's end, the bench's own or that of a
        # child it forked before its first batch.
        numbered: dict[str, list[tuple[int, Path]]] = {}
        for part in work.glob("profile.*.*"):
            pid, _, number = part.name.removeprefix("profile.").partition(".")
            if number.isdigit():
                numbered.setdefault(pid, []).append((int(number), part))
        if len(numbered) > 1:
            raise RuntimeError(f"two processes of the count ran batches; see {log_path}")
        parts = sorted(next(iter(numbered.values()), []))
        try:
            batches = [read_profile(part) for _, part in parts]
        except (ValueError, IndexError) as err:
            raise RuntimeError(
                f"a profile of the count does not parse ({err}); see {log_path}"
            ) from None
    finally:
        for part in work.glob("profile.*"):
            part.unlink()
    if not batches:
        raise RuntimeError(f"the count holds no batch of the harness; see {log_path}")
    if any(batch.calls != 1 for batch in batches):
        raise RuntimeError(f"a profile of the count does not hold one batch; see {log_path}")
    return {
        "batches": [batch.count for batch in batches],
        "unit": batches[-1].unit,
        "address": batches[-1].address,
    }


def count_ids(
    ids: list[str],
    binaries: dict[str, Path],
    out: Path,
    cwd: Path,
    gates: Gates,
    extra_args: list[str],
    cpu: int | None,
    deadline: float,
) -> dict[str, dict]:
    """The ids on both executables, in their order, until the run's budget ends (`deadline`,
    on the monotonic clock); a side that was not read carries the reason. Two counts that
    differ are each taken once more (`again`): a count that differs from itself is the id's
    own (a seed the process draws, the allocator's state) and says nothing of the change. No
    count starts with less than `COUNT_FLOOR_S` left, and one that starts is capped by what
    is left: the id whose turn the budget cut and the ids behind it are not counted."""
    counts: dict[str, dict] = {}
    for n, bench_id in enumerate(ids, start=1):

        def count(side: str, turn: int) -> dict | None:
            """One count; None when the run's budget left no room for it or ended it."""
            left = deadline - time.monotonic()
            if left < COUNT_FLOOR_S:
                return None
            bound = min(COUNT_TIMEOUT_S, left)
            started = time.monotonic()
            try:
                work = out / SIDE_DIRS[side] / f"count-{n}-{turn}"
                read = count_once(
                    binaries[side], work, cwd, gates, extra_args, bench_id, cpu, bound
                )
                said = f"{len(read['batches'])} batches"
            except (RuntimeError, OSError) as err:
                if bound < COUNT_TIMEOUT_S and time.monotonic() - started >= bound:
                    return None
                read = {"error": str(err)}
                said = f"not read, {err}"
            seconds = time.monotonic() - started
            label = f"count {n}/{len(ids)}" + (", once more" if turn == 2 else "")
            print(f"[{label}] {side}: {bench_id}: {said} in {seconds:.0f} s", flush=True)
            return read

        def counted() -> dict | None:
            """The id's two to four counts; None when the run's budget cut its turn."""
            # The two counts of a side run from directories of one length: the length of a
            # path the process holds moves what it allocates after, and a count can follow that.
            sides: dict[str, dict] = {}
            for side in SIDES:
                read = count(side, 1)
                if read is None:
                    return None
                sides[side] = read
            if any("batches" not in read for read in sides.values()):
                return sides
            if differs_at(sides["base"]["batches"], sides["pr"]["batches"]) is None:
                return sides
            for side in SIDES:
                again = count(side, 2)
                if again is None:
                    return None
                if "batches" in again:
                    sides[side]["again"] = again["batches"]
            return sides

        sides = counted()
        if sides is None:
            # This id and the ids behind it keep the time's verdict, and the report is written.
            left_ids = ids[n - 1 :]
            print(
                f"[count] the run's budget of {gates.run_budget:g} s leaves no room: "
                f"{len(left_ids)} of {len(ids)} ids not counted",
                flush=True,
            )
            counts.update({left_id: {"skipped": "budget"} for left_id in left_ids})
            break
        counts[bench_id] = sides
    return counts


def differs_at(one: list[int], other: list[int]) -> int | None:
    """The last batch two counts both ran that differs, the first left out (a first iteration
    pays what the process has not done yet); None when every one is the same."""
    for n in range(min(len(one), len(other)) - 1, 0, -1):
        if one[n] != other[n]:
            return n
    return None


@dataclass
class Counted:
    """What the count of an id read, at the batch that decides it."""

    # "same": every batch both executables ran after the first is equal. "moved": one
    # differs, and each executable counted the same twice. "unsteady": one differs, and so
    # do an executable's own two counts. "once": the last that differs is one a second
    # count did not run, and it reads beyond the level (`Row.beyond_once`).
    state: str
    # The harness's warm-up doubles a batch: 1, 2, 4, ... iterations.
    iterations: int
    # The batch's instructions on the base and on the pull request.
    base: int
    pr: int
    # For an unsteady count: the executable whose two counts differ, and by how much.
    side: str = ""
    own: float = 0.0


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
    # The count of an id that failed by time, per side: the batches' instruction counts and
    # the last batch's unit, or why the side was not read.
    counts: dict | None = None
    # The second measurement of an id that still failed on a count that reads level, per
    # side: its runs from the fresh copies. None for an id not measured again.
    second: dict[str, list[float]] | None = None

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

    @property
    def counted(self) -> Counted | None:
        """What the count read: every batch both executables ran but the first, decided at
        the last one that differs, or at the last one when none does. Two counts that differ
        are decided among the batches all four counts ran: a second count that ended a batch
        early leaves the batches past it counted once. None when a side was not read, when
        the first batch is the only one both ran, or when the counts differ only past what a
        second count ran."""
        sides = [(self.counts or {}).get(side) or {} for side in SIDES]
        if any("batches" not in side for side in sides):
            return None
        base, pr = (side["batches"] for side in sides)
        if min(len(base), len(pr)) < 2:
            return None
        n = differs_at(base, pr)
        if n is None:
            n = min(len(base), len(pr)) - 1
            return Counted("same", 2**n, base[n], pr[n])
        seconds = [side.get("again") for side in sides]
        if all(seconds):
            ran = min(len(base), len(pr), *(len(again) for again in seconds))
            n = differs_at(base[:ran], pr[:ran])
            if n is None:
                return None
        for label, side in zip(("base", "pull request"), sides):
            first, again = side["batches"], side.get("again") or []
            own = differs_at(first, again)
            if own is not None:
                moved = (again[own] - first[own]) / first[own]
                return Counted("unsteady", 2**n, base[n], pr[n], label, moved)
        return Counted("moved", 2**n, base[n], pr[n])

    @property
    def same_count(self) -> bool:
        """Whether the count read the same instructions on the two executables."""
        counted = self.counted
        return counted is not None and counted.state == "same"

    @property
    def moved(self) -> bool | None:
        """Whether the function the count names starts at another address in the pull
        request's executable. None when a side names no function or no address, or when the
        two sides name different functions."""
        sides = [(self.counts or {}).get(side) or {} for side in SIDES]
        unit, places = sides[0].get("unit"), [side.get("address") for side in sides]
        if not unit or unit != sides[1].get("unit") or None in places:
            return None
        return places[0] != places[1]

    @property
    def within_ceiling(self) -> bool:
        """Whether the change is one a placement can excuse: within twice the threshold."""
        change, threshold = self.change, self.gate.threshold if self.gate else 0.0
        return change is not None and change <= PLACEMENT_CEILING * threshold

    @property
    def whole_pass(self) -> bool:
        """Whether the id's gate entry marks it as one that times a whole pass."""
        return self.gate is not None and self.gate.whole_pass

    @property
    def level_count(self) -> bool:
        """Whether the count read level on the two executables: the same, or moved by no more
        than `LEVEL_COUNT` of the base's with each executable's own two counts the same, and
        by no more where the first counts differ last at a batch a second count did not run."""
        counted = self.counted
        if counted is None or counted.state == "unsteady":
            return False
        if counted.state == "moved" and not self.counted_twice:
            return False
        if self.beyond_once is not None:
            return False
        return abs(counted.pr - counted.base) <= LEVEL_COUNT * counted.base

    @property
    def counted_twice(self) -> bool:
        """Whether each executable was counted a second time, which two counts that differ
        ask before either is read."""
        return all(((self.counts or {}).get(side) or {}).get("again") for side in SIDES)

    @property
    def beyond_once(self) -> Counted | None:
        """The last batch of the first counts that differs, where a second count did not run
        it and the two read further apart than `LEVEL_COUNT`; None otherwise. It is the batch
        that would decide the row had every count run it, and no second count says that each
        executable counts it the same twice: it grants no level count, and it refuses one, since
        work that grows with the iterations can read under the level one batch early."""
        if not self.counted_twice:
            return None
        sides = [self.counts[side] for side in SIDES]
        base, pr = (side["batches"] for side in sides)
        n, ran = differs_at(base, pr), min(len(side["again"]) for side in sides)
        if n is None or n < ran or abs(pr[n] - base[n]) <= LEVEL_COUNT * base[n]:
            return None
        return Counted("once", 2**n, base[n], pr[n])

    @property
    def placed(self) -> bool:
        """Whether where the code sits can carry the time, within twice the threshold: for
        an id that times a whole pass a level count, for any other the same count and a
        measured function that starts elsewhere."""
        if not self.within_ceiling:
            return False
        if self.whole_pass:
            return self.level_count
        return bool(self.same_count and self.moved)

    @property
    def reads_level(self) -> bool:
        """Whether no count read more work on the pull request's executable than the level, so
        that the id is measured again before it fails: the same count, or two counts of each
        executable that all lie within `LEVEL_COUNT` of each other at every batch where two of
        the four last differ. Each two are read among the batches all four ran and again over
        the batches the two ran, and a batch is compared on every count that ran it: a count
        that ended a batch early hides no other. A count that moved and one that is not steady
        are one noise in two states (a few instructions an iteration more or fewer from one
        process to the next): inside the level neither carries a failure by the clock."""
        counted = self.counted
        if counted is None or self.beyond_once is not None:
            return False
        if counted.state == "same":
            return True
        if not self.counted_twice:
            return False
        counts = [self.counts[side][take] for side in SIDES for take in ("batches", "again")]
        ran = min(len(count) for count in counts)
        for among in ([count[:ran] for count in counts], counts):
            for one, other in itertools.combinations(among, 2):
                n = differs_at(one, other)
                if n is None:
                    continue
                # The level is a part of the first count that ran the batch: the base's, where
                # it did.
                at = [count[n] for count in among if n < len(count)]
                if max(at) - min(at) > LEVEL_COUNT * at[0]:
                    return False
        return True

    @property
    def measured_again(self) -> Row | None:
        """The second measurement as a row of its own, read within itself. None for an id not
        measured again, and for one a side of which is short of the first measurement's runs:
        the first then decides."""
        second = self.second or {}
        sides = [second.get(side) or [] for side in SIDES]
        if not self.base or any(len(side) < len(self.base) for side in sides):
            return None
        return Row(self.bench_id, *sides, self.gate)

    @property
    def repeated(self) -> bool | None:
        """Whether the second measurement fails the id as the first did, by the judge's own
        two conditions. None for an id not measured again."""
        again = self.measured_again
        return None if again is None else judge(again) == "regression"


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
    otherwise suspect. A regression is a placement, where the code sits in the bench executable
    and not work the change adds, on three facts: the two executables run it in the same
    instruction count, its measured function starts elsewhere, and the change is within twice
    the threshold. An id that times a whole pass is one on two: a level count, and the change
    within twice the threshold. A regression on a count that reads level was measured again:
    it is not repeated where that second measurement does not fail it too."""
    if not row.separated:
        return "suspect (runs overlap)"
    if row.confirmed is False:
        return "suspect (not confirmed by the re-measurement)"
    if row.placed:
        return "placement"
    return "not repeated" if row.repeated is False else "regression"


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
    counts = results.get("count") or {}
    second = results.get("second") or {}
    measured = set() if second.get("skipped") else set(second.get("ids", []))
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
            counts=counts.get(bench_id),
        )
        if bench_id in measured:
            row.second = {
                side: [run[bench_id] for run in second[side]["runs"] if bench_id in run]
                for side in SIDES
            }
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


def fmt_place(address: int) -> str:
    """Where code starts inside its 4 KiB page."""
    return f"0x{address % 4096:03x}"


def stands(row: Row) -> str:
    """How the sentence of a count that places nothing ends: the failure stands, or, for an
    id measured again on that count, the second measurement decides."""
    again = row.measured_again
    return "the failure stands" if again is None else "the second measurement decides"


def fmt_second(row: Row, unit_suffix: str) -> str:
    """The second measurement of a row, read within itself; "" for a row not measured again."""
    again = row.measured_again
    if again is None:
        return ""
    end = "the failure repeats" if row.repeated else "it does not repeat"
    return (
        " Measured again from a fresh copy of each executable: base runs "
        f"{fmt_runs(again.base)}{unit_suffix}, pull request runs {fmt_runs(again.pr)}"
        f"{unit_suffix} ({fmt_pct(again.change)}): {end}."
    )


# Why an id that failed by time was not counted, as its line and the report say it.
NOT_COUNTED = {
    "beyond": (
        f"a change beyond {PLACEMENT_CEILING:g} times the threshold, which no placement excuses"
    ),
    "no_base": "a base median of zero, which leaves no change for a placement to excuse",
    "budget": "the run's budget left no room for its count, and the time alone decides",
}


def fmt_count(row: Row) -> str:
    """What the count of a row read, as sentences for its line; "" for a row never counted."""
    if not row.counts:
        return ""
    skipped = NOT_COUNTED.get(row.counts.get("skipped"))
    if skipped:
        return f" Not counted: {skipped}."
    sides = [row.counts.get(side) or {} for side in SIDES]
    counted = row.counted
    if counted is None:
        errors = [
            f"{label}: {side['error']}"
            for label, side in zip(("base", "pull request"), sides)
            if "error" in side
        ]
        ran = min(len(side.get("batches") or []) for side in sides)
        alone = "the first batch alone ran on both sides"
        short = "a second count ended before the batches that differ"
        why = "; ".join(errors) or (short if ran >= 2 else alone)
        return f" The count was not read ({why}): the time alone decides."
    counts = f"{counted.base:,} instructions on the base and {counted.pr:,} on the pull request"
    batch = f"a batch of {counted.iterations:,} iterations"
    if counted.state != "same":
        change = (counted.pr - counted.base) / counted.base
        each = abs(counted.pr - counted.base) / counted.iterations
        more = "more" if counted.pr > counted.base else "fewer"
        differs = (
            f"{counts} over {batch} ({change:+.3%}, {each:,.1f} instructions an iteration {more})"
        )
        if counted.state == "unsteady":
            return (
                f" The count differs, {differs}, and it is not steady on one executable alone: "
                f"the {counted.side}'s own two counts differ by {abs(counted.own):.3%}. The "
                "time alone decides."
            )
        twice = ", the same in two counts of each" if row.counted_twice else ""
        if row.whole_pass:
            return fmt_level_count(row, counted, differs, twice)
        return f" The count moved too: {differs}{twice}."
    return fmt_same_count(row, counted)


def fmt_level_count(row: Row, counted: Counted, differs: str, twice: str) -> str:
    """The sentences of an id that times a whole pass whose two counts differ: a placement
    on a level count, or what leaves the failure."""
    level = f"the {LEVEL_COUNT:.1%} that reads as level for an id that times a whole pass"
    if abs(counted.pr - counted.base) > LEVEL_COUNT * counted.base:
        return f" The count moved too: {differs}{twice}, beyond {level}."
    if not twice:
        return (
            f" The count moved too: {differs}. A level count is two counts of each executable "
            "that agree, and a second one was not read: the failure stands."
        )
    once = row.beyond_once
    if once is not None:
        return (
            f" The count is level where all four counts ran: {differs}{twice}, within {level}. "
            f"Past it the first counts read {once.base:,} and {once.pr:,} over a batch of "
            f"{once.iterations:,} iterations ({(once.pr - once.base) / once.base:+.3%}), beyond "
            "the level, and a second count did not run that batch: the failure stands."
        )
    if not row.placed:
        return f" The count is level: {differs}{twice}, within {level}.{fmt_ceiling(row)}"
    return (
        f" The count is level: {differs}{twice}, within {level}. The time is where the "
        "pass's code sits in the bench executable, not work the pull request adds."
    )


def fmt_ceiling(row: Row) -> str:
    """The sentence of a count that would place a row whose change is beyond the ceiling."""
    ceiling = PLACEMENT_CEILING * (row.gate.threshold if row.gate else 0.0)
    return (
        f" A placement excuses a change up to {PLACEMENT_CEILING:g} times the threshold, "
        f"{fmt_pct(ceiling, signed=False)}: {stands(row)}."
    )


def fmt_same_count(row: Row, counted: Counted) -> str:
    """The sentences of a row the two executables run in the same instruction count: a
    placement, or which of its two other facts leaves the failure (for an id that times a
    whole pass the one other fact, the ceiling)."""
    same = (
        " The two executables run it in the same instruction count in every batch after the "
        f"first, {counted.base:,} over the {counted.iterations:,} iterations of the last "
        f"({counted.base / counted.iterations:,.1f} an iteration)"
    )
    if row.whole_pass:
        if not row.placed:
            return f"{same}.{fmt_ceiling(row)}"
        return (
            f"{same}: the id times a whole pass, and the time is where the pass's code sits "
            "in the bench executable, not work the pull request adds."
        )
    sides = [(row.counts or {}).get(side) or {} for side in SIDES]
    unit, places = sides[0].get("unit"), [side.get("address") for side in sides]
    moved = row.moved
    if moved is None:
        return (
            f"{same}, but the count names no one function with a place on both sides: "
            f"{stands(row)}."
        )
    if not moved:
        return (
            f"{same}, and `{unit}` starts at 0x{places[0]:x} in both: where the code sits does "
            f"not explain the time, and {stands(row)}."
        )
    if places[0] % 4096 == places[1] % 4096:
        where = (
            f"`{unit}` starts at 0x{places[0]:x} in the base's executable and at "
            f"0x{places[1]:x} in the pull request's, the same place in its 4 KiB page."
        )
    else:
        where = (
            f"`{unit}` starts at {fmt_place(places[0])} of its 4 KiB page in the base's "
            f"executable and at {fmt_place(places[1])} in the pull request's."
        )
    if not row.placed:
        return f"{same}. {where}{fmt_ceiling(row)}"
    return (
        f"{same}: the time is where the code sits in the bench executable, not work the pull "
        f"request adds. {where}"
    )


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
        base, pr = (results["binaries"][side] for side in SIDES)
        if base["sha256"] != pr["sha256"]:
            return (
                "### Benchmark gate: PASS (the same program)\n\n"
                "The base and pull-request bench binaries load the same image (sha256 "
                f"{base['image'][:16]}... of what the loader maps, the build id left out) and "
                "differ outside it, in what no run reads: no code change can move a row, so "
                "nothing was run.\n"
            )
        sha = base["sha256"]
        return (
            "### Benchmark gate: PASS (identical binaries)\n\n"
            f"The base and pull-request bench binaries are the same bytes (sha256 {sha[:16]}...): "
            "no code change can move a row, so nothing was run.\n"
        )
    failed = [row for row in rows if row.fails]
    suspect = [row for row in rows if row.verdict.startswith("suspect")]
    placed = [row for row in rows if row.verdict == "placement"]
    unrepeated = [row for row in rows if row.verdict == "not repeated"]
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
    reads = results.get("count") or {}
    counted = sorted(bench_id for bench_id, read in reads.items() if "skipped" not in read)
    whole = [row.bench_id for row in rows if row.whole_pass and row.bench_id in counted]
    if counted:
        lines += [
            "Counted before the verdict, on both executables under callgrind: "
            f"{', '.join(f'`{bench_id}`' for bench_id in counted)}. An id that "
            "fails by time is a placement (where its code sits in the bench executable) when "
            "it runs the same instruction count on both, its measured function starts at "
            "another address and the change is within twice its threshold: reported, not "
            "failed, and still a row to read."
            + (
                " An id that times a whole pass ("
                + ", ".join(f"`{bench_id}`" for bench_id in whole)
                + f") is a placement on a level count instead: its two counts within "
                f"{LEVEL_COUNT:.1%} of each other, each executable's own two the same, and "
                "the change within twice its threshold."
                if whole
                else ""
            ),
            "",
        ]
    for why, said in NOT_COUNTED.items():
        left = sorted(bench_id for bench_id, read in reads.items() if read.get("skipped") == why)
        if left:
            names = ", ".join(f"`{bench_id}`" for bench_id in left)
            lines += [f"Not counted, {names}: {said}.", ""]
    second = results.get("second") or {}
    if second.get("ids"):
        names = ", ".join(f"`{bench_id}`" for bench_id in second["ids"])
        if second.get("skipped"):
            lines += [
                f"Not measured again, {names}: the run's budget left no room for the second "
                "measurement a count that reads level asks for, and the first decides.",
                "",
            ]
        else:
            lines += [
                f"Measured again before the verdict, {repeats} more run(s) per side from a fresh "
                f"copy of each executable: {names}. An id that still fails by time on a count "
                "that reads level (the same on both executables, or every count within "
                f"{LEVEL_COUNT:.1%}) fails where this second measurement, read within itself, is "
                "beyond its threshold with every pull-request run worse than every base run "
                "too. Otherwise it is reported as not repeated, and still a row to read.",
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
        + (
            f" {len(placed)} placement (beyond the threshold on the same instruction count, "
            "the measured function moved, within twice the threshold: not failed)."
            if placed and not any(row.whole_pass for row in placed)
            else f" {len(placed)} placement (beyond the threshold within twice of it: on the "
            "same instruction count with the measured function moved, or for an id that "
            "times a whole pass on a level count; not failed)."
            if placed
            else ""
        )
        + (
            f" {len(unrepeated)} not repeated (beyond the threshold on a count that reads "
            "level, and not so a second time from fresh copies: not failed)."
            if unrepeated
            else ""
        )
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
                f"{fmt_runs(row.pr)}{unit_suffix}{fmt_rechecks(row, unit_suffix)}.{fmt_count(row)}"
                f"{fmt_second(row, unit_suffix)}"
            )
    # The per-run values of every suspect, placed or re-measured row, so a bimodal set is
    # visible here; a placement carries what its count read.
    for row in rows:
        listed = row in suspect or row in placed or row in unrepeated
        if listed or (row.bench_id in rechecked and not row.fails):
            lines.append(
                f"- `{row.bench_id}`: {row.verdict}, {fmt_pct(row.change)}; base runs "
                f"{fmt_runs(row.base)}{unit_suffix}, pull request runs "
                f"{fmt_runs(row.pr)}{unit_suffix}{fmt_rechecks(row, unit_suffix)}.{fmt_count(row)}"
                f"{fmt_second(row, unit_suffix)}"
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
