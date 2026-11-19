#!/usr/bin/env python3
"""Build the comparative tokenizer harness, check token ids and run its cells.

    run.py prepare --corpus DIR    fetch the pinned tokenizer files, build one binary per column
    run.py ids --corpus DIR        every column's token ids on the parity strings, against `smg`
    run.py run --corpus DIR --cores 2,4,6,8
                                   the cells: one fresh process per column and cell, pinned to a core

README.md beside this file says what a cell measures. Standard library only.
"""

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
MODELS = json.loads((HERE / "models.json").read_text())
REFERENCES = json.loads((HERE / "references.json").read_text())
CORPUS = json.loads((HERE / "corpus.json").read_text())

CLASSES = ("256", "2k", "32k", "128k", "500k", "1m")
MODES = ("cold", "warm")
# column -> (adapter package under harness/, cargo features)
COLUMNS = {
    "smg": ("smg-tok", []),
    "v1": ("smg-tok", ["tokenizers-v1"]),
    "fastokens": ("fastokens-tok", []),
    "tokenizers": ("hf-tok", []),
    "tiktoken-rs": ("tt-tok", []),
}
# The tiktoken adapter reads a `tiktoken.model` ranks file or the crate's own o200k_harmony ranks,
# so it runs on the two families that have one of them.
ONLY = {"tiktoken-rs": ("kimi-k3", "gpt-oss-20b")}
# Columns that take SLOW_REPS instead of REPS at the two largest classes.
SLOW = ("tokenizers",)
# The path dependency each adapter's manifest carries, replaced when a copy is built.
CRATE_PATHS = {"smg-tok": "../../../..", "fastokens-tok": "../reference/rust/src/tokenizer"}
CELL_TIMEOUT = 3600


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_models(root):
    for model in MODELS:
        for name, expected in model["files"].items():
            path = root / model["name"] / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{name}"
                temporary = path.with_name(path.name + ".download")
                urllib.request.urlretrieve(url, temporary)
                if sha256(temporary) != expected:
                    raise ValueError(f"Downloaded file hash mismatch: {path}")
                temporary.replace(path)
            if sha256(path) != expected:
                raise ValueError(f"File hash mismatch: {path}")


def check_corpus(corpus, output, any_corpus):
    """Verify the corpus against corpus.json and derive the 1m class from the 500k class."""
    if not any_corpus:
        for cls, record in CORPUS["classes"].items():
            for name, expected in record["sha256"].items():
                if sha256(corpus / "perf" / cls / name) != expected:
                    raise ValueError(f"Corpus text differs from corpus.json: {cls}/{name}")
        if sha256(corpus / "parity.jsonl") != CORPUS["parity"]["sha256"]:
            raise ValueError("parity.jsonl differs from corpus.json")
    source = corpus / "perf" / "500k"
    names = sorted(path.name for path in source.iterdir())
    target = output / "corpus" / "1m"
    target.mkdir(parents=True, exist_ok=True)
    joint = CORPUS["class_1m"]["joint"].encode()
    for index, name in enumerate(names):
        if not (target / name).exists():
            following = names[(index + 1) % len(names)]
            data = (source / name).read_bytes() + joint + (source / following).read_bytes()
            (target / name).write_bytes(data)


def reference_crate(source, output):
    """A checkout of the fastokens stack at the pinned commit; the path of its tokenizer crate."""
    reference = REFERENCES["fastokens"]
    tree = source or output / "src" / "fastokens-stack"
    if not tree.exists():
        subprocess.run(["git", "clone", "--no-checkout", reference["repository"], tree], check=True)
        subprocess.run(["git", "-C", tree, "checkout", "--detach", reference["commit"]], check=True)
    if (tree / ".git").exists():
        head = subprocess.check_output(["git", "-C", tree, "rev-parse", "HEAD"], text=True).strip()
        if head != reference["commit"]:
            raise ValueError(f"{tree} is at {head}, not at the pinned {reference['commit']}")
    else:
        print(f"{tree} is not a git checkout: its commit is not verified")
    return tree / reference["crate"]


def build(column, adapter, features, crate, output, locked):
    """Build a copy of one adapter whose path dependency names `crate`; the binary's path."""
    stage = output / "build" / column / "harness"
    shutil.copytree(
        HERE / "harness" / adapter,
        stage / adapter,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("target"),
    )
    shutil.copy2(HERE / "harness" / "cells.rs", stage / "cells.rs")
    manifest = stage / adapter / "Cargo.toml"
    text = manifest.read_text()
    if crate is not None:
        old = f'path = "{CRATE_PATHS[adapter]}"'
        if text.count(old) != 1:
            raise ValueError(f"{manifest} does not carry the path dependency {old}")
        manifest.write_text(text.replace(old, f'path = "{crate}"'))
    command = ["cargo", "build", "--release", "--manifest-path", str(manifest)]
    if locked:
        command.append("--locked")
    for feature in features:
        command += ["--features", feature]
    # `CARGO_TARGET_DIR` can point at an existing cache; the default is a target under the output.
    target = Path(os.environ.get("CARGO_TARGET_DIR", output / "target")).resolve()
    environment = dict(os.environ, CARGO_TARGET_DIR=str(target))
    with (output / f"build-{column}.log").open("w") as log:
        # The repository root as the working directory: its toolchain file selects the compiler
        # for every column, so a column of an earlier release differs by the crate alone.
        subprocess.run(
            command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    binary = output / "bin" / column
    binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target / "release" / adapter, binary)
    return binary


def prepare(args):
    if not args.skip_fetch:
        fetch_models(args.models)
    check_corpus(args.corpus, args.output, args.any_corpus)
    wanted = args.columns or list(COLUMNS)
    for column in wanted:
        adapter, features = COLUMNS[column]
        crate = None
        if adapter == "smg-tok":
            crate = REPO / "crates" / "tokenizer"
        elif adapter == "fastokens-tok":
            crate = reference_crate(args.fastokens_src, args.output)
        print(f"built {build(column, adapter, features, crate, args.output, True)}")
    for spec in args.previous:
        column, _, tree = spec.partition("=")
        crate = Path(tree).resolve() / "crates" / "tokenizer"
        # An earlier release may ask for other dependency versions than the recorded lock holds:
        # cargo starts from the lock and changes what that release's manifest requires.
        print(f"built {build(column, 'smg-tok', [], crate, args.output, False)}")


def columns_of(args, family):
    """The built columns that run on `family`, `smg` first."""
    names = sorted(path.name for path in (args.output / "bin").iterdir())
    if args.columns:
        names = [name for name in names if name in args.columns]
    names = [name for name in names if family in ONLY.get(name, (family,))]
    if "smg" not in names:
        raise ValueError("No binary of the column smg under the output directory: run prepare")
    return ["smg"] + [name for name in names if name != "smg"]


def families_of(args):
    names = [model["name"] for model in MODELS]
    return [name for name in names if not args.families or name in args.families]


def one(args, column, family, cores, cell):
    """One fresh process of one column on one cell; the record of its one output line."""
    command = [str(args.output / "bin" / column), *cell]
    if cores:
        command = ["taskset", "-c", cores, *command]
    context = {
        "stamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "column": column,
        "family": family,
        "cores": cores,
        "loadavg1": float(Path("/proc/loadavg").read_text().split()[0]),
        "args": cell,
    }
    try:
        done = subprocess.run(
            command, capture_output=True, encoding="utf-8", errors="replace", timeout=CELL_TIMEOUT
        )
        context["rc"], lines, context["stderr"] = done.returncode, done.stdout, done.stderr[-300:]
    except subprocess.TimeoutExpired:
        context["rc"], lines, context["stderr"] = 124, "", "timeout"
    try:
        out = json.loads(lines.strip().splitlines()[-1]) if lines.strip() else None
    except ValueError:
        out = None
    return {"ctx": context, "out": out}


class Sink:
    """Appends one JSON line per record; shared by the streams of a run."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.lock = path, threading.Lock()

    def write(self, record, **extra):
        record["ctx"].update(extra)
        with self.lock, self.path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def cell(args, sink, family, cores, shift, arguments, **extra):
    """Every column of the family on one cell, in an order that rotates from cell to cell."""
    names = columns_of(args, family)
    for position in range(len(names)):
        column = names[(position + shift) % len(names)]
        sink.write(one(args, column, family, cores, arguments(column)), **extra)


def records(path):
    """The JSON lines of a file, one at a time; nothing when the file is missing."""
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def edge_cells(path, band):
    """The encode cells whose margin against the best reference stands inside +-band.

    A reference is a column of COLUMNS other than smg (an earlier release is none), and the v1
    column only where its build took the v1 encoder (its backend string says `hf:v1`).
    """
    values = {}
    for record in records(path):
        ctx, out = record["ctx"], record["out"]
        if ctx["rc"] != 0 or not out or "ns_per_tok" not in out or ctx["column"] not in COLUMNS:
            continue
        if ctx["column"] == "v1" and "hf:v1" not in out["backend"]:
            continue
        key = (ctx["family"], out["class"], out["mode"])
        values.setdefault(key, {}).setdefault(ctx["column"], []).append(out["ns_per_tok"]["p50"])
    chosen = set()
    for key, by_column in values.items():
        medians = {name: statistics.median(v) for name, v in by_column.items()}
        others = [value for name, value in medians.items() if name != "smg"]
        if "smg" in medians and others and abs(min(others) / medians["smg"] - 1) < band:
            chosen.add(key)
    return chosen


def encode_stream(args, sink, families, core, rounds, chosen):
    texts = {cls: args.corpus / "perf" for cls in CLASSES}
    texts["1m"] = args.output / "corpus"
    for number in rounds:
        for family in families:
            directory = args.models / family
            shift = 0
            for cls in CLASSES:
                for mode in MODES:
                    if chosen is not None and (family, cls, mode) not in chosen:
                        continue

                    def arguments(column, cls=cls, mode=mode, directory=directory):
                        slow = column in SLOW and cls in ("500k", "1m")
                        reps = args.slow_reps if slow else args.reps
                        head = ["enc", str(directory), str(texts[cls]), "--class", cls]
                        return head + [
                            "--mode",
                            mode,
                            "--threads",
                            "1",
                            "--reps",
                            str(reps),
                            "--warm",
                            "3",
                        ]

                    cell(args, sink, family, core, shift, arguments, round=number)
                    shift += 1


def run(args):
    check_corpus(args.corpus, args.output, True)
    cores = args.cores.split(",")
    families = families_of(args)
    perf = args.corpus / "perf"
    if "encode" in args.cells:
        path = args.output / "raw" / "encode.jsonl"
        chosen = edge_cells(path, args.edge) if args.edge else None
        first = 1 + max((r["ctx"]["round"] for r in records(path)), default=0)
        rounds = range(first, first + args.rounds)
        sink = Sink(path)
        streams = [
            threading.Thread(
                target=encode_stream,
                args=(args, sink, families[index :: len(cores)], core, rounds, chosen),
            )
            for index, core in enumerate(cores)
        ]
        for stream in streams:
            stream.start()
        for stream in streams:
            stream.join()
    if "decode" in args.cells:
        path = args.output / "raw" / "decode.jsonl"
        first = 1 + max((r["ctx"]["round"] for r in records(path)), default=0)
        sink = Sink(path)
        long_text, longer_text = str(perf / "32k" / "000.txt"), str(perf / "128k" / "000.txt")
        for number in range(first, first + args.rounds):
            for family in families:
                d = str(args.models / family)
                cells = [
                    ["dec", d, long_text, "--len", "2048", "--reps", "30", "--warm", "3"],
                    ["dec", d, longer_text, "--len", "32768", "--reps", "10", "--warm", "2"],
                    ["stream", d, long_text, "--streams", "64", "--steps", "256", "--reps", "3"],
                    ["batch", d, str(args.corpus / "parity.jsonl"), "--n", "1000"]
                    + ["--reps", "10", "--warm", "2"],
                ]
                for shift, arguments in enumerate(cells):
                    cell(
                        args, sink, family, cores[0], shift, lambda _, a=arguments: a, round=number
                    )
    if "memory" in args.cells:
        sink = Sink(args.output / "raw" / "memory.jsonl")
        block = args.memory_cores or args.cores
        big = str(args.output / "corpus" / "1m")
        for family in families:
            d = str(args.models / family)
            cells = [["info", d], ["mem", d, str(perf), "--c1m", big, "--threads", "64"]]
            for shift, arguments in enumerate(cells):
                cell(args, sink, family, block, shift, lambda _, a=arguments: a)


def ids(args):
    """Every column's token ids on the parity strings; the count that equals the column smg."""
    corpus = args.corpus / "parity.jsonl"
    totals = {}
    lines = []

    def digest(column, family, scratch, digests):
        out = Path(scratch) / f"{column}.jsonl"
        command = [str(args.output / "bin" / column), "parity"]
        command += [str(args.models / family), str(corpus), str(out), "--no-inc"]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        found = {}
        for record in records(out):
            value = record.get("ids")
            text = json.dumps(value).encode()
            found[record["i"]] = hashlib.sha256(text).hexdigest() if value is not None else "error"
        digests[column] = found

    for family in families_of(args):
        columns = columns_of(args, family)
        digests = {}
        # The columns of a family side by side: no time is read here.
        with tempfile.TemporaryDirectory(dir=args.output) as scratch:
            workers = [
                threading.Thread(target=digest, args=(column, family, scratch, digests))
                for column in columns
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
        ours = digests["smg"]
        failed = sum(1 for value in ours.values() if value == "error")
        parts = [f"{family}: {len(ours)} strings, smg failed on {failed}"]
        for column in columns[1:]:
            theirs = digests[column]
            if not theirs:
                # The column has no file of this family to load: no string of it is counted.
                parts.append(f"{column} did not load")
                continue
            same = sum(1 for index, value in ours.items() if theirs.get(index) == value)
            total = totals.setdefault(column, [0, 0])
            total[0] += same
            total[1] += len(ours)
            parts.append(f"{column} {same} of {len(ours)}")
        lines.append("; ".join(parts))
        print(lines[-1], flush=True)
    lines.append("TOTAL SAME: " + "; ".join(f"{c} {t[0]} of {t[1]}" for c, t in totals.items()))
    print(lines[-1])
    (args.output / "ids-summary.txt").write_text("\n".join(lines) + "\n")


def main():
    def names(text):
        return [name for name in text.split(",") if name]

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("action", choices=["prepare", "ids", "run"])
    parser.add_argument(
        "--corpus", type=Path, required=True, help="perf/<class>/*.txt, parity.jsonl"
    )
    parser.add_argument("--models", type=Path, default=Path("/tmp/smg-tokenizer-models"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/smg-compare-results"))
    parser.add_argument("--columns", type=names, default=[], help="a subset, comma separated")
    parser.add_argument("--families", type=names, default=[], help="a subset of models.json names")
    parser.add_argument(
        "--previous",
        action="append",
        default=[],
        metavar="COLUMN=CHECKOUT",
        help="prepare: one more column, this crate in another checkout (an earlier release)",
    )
    parser.add_argument(
        "--fastokens-src", type=Path, help="prepare: an existing checkout at the pinned commit"
    )
    parser.add_argument("--skip-fetch", action="store_true", help="prepare: --models is complete")
    parser.add_argument(
        "--any-corpus", action="store_true", help="prepare: accept texts corpus.json does not list"
    )
    parser.add_argument("--cores", help="run: cores to pin to, comma separated; one stream each")
    parser.add_argument(
        "--memory-cores", help="run: the cores of the memory cell (default --cores)"
    )
    parser.add_argument("--cells", type=names, default=["encode", "decode", "memory"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--edge",
        type=float,
        help="run: more encode rounds, only on the cells within this fraction of the best reference",
    )
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--slow-reps", type=int, default=10)
    args = parser.parse_args()
    if args.action == "run" and not args.cores:
        parser.error("run needs --cores")
    if args.edge:
        args.cells = ["encode"]
    for name in ("corpus", "models", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "ids": ids, "run": run}[args.action](args)


if __name__ == "__main__":
    main()
