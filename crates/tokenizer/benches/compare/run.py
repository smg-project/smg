#!/usr/bin/env python3
"""Build the comparative tokenizer harness, check token ids and run its cells.

    run.py prepare --corpus DIR    fetch the pinned tokenizer files, build one binary per column
    run.py ids --corpus DIR        every column's token ids on the parity strings, against `smg`
    run.py run --corpus DIR --cores 2,4,6,8
                                   the cells: one fresh process per column and cell, pinned to a core
    run.py collect --release 1.16.0 --csv DIR
                                   the three csv files that tables.py reads, from the raw lines

README.md beside this file says what a cell measures. Standard library only.
"""

import argparse
import csv
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
        following = names[(index + 1) % len(names)]
        data = (source / name).read_bytes() + joint + (source / following).read_bytes()
        # A text that a run on another corpus left here is replaced: no hash covers the 1m class.
        if not (target / name).exists() or (target / name).read_bytes() != data:
            (target / name).write_bytes(data)
    # A text under a name this corpus does not have is one that another corpus left, and the cells
    # read every file of the directory: it is removed.
    for path in target.iterdir():
        if path.name not in names:
            path.unlink()


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
    # The copies take the time of this call, not the times of their sources, so cargo compiles the
    # adapter and the cell module again on every call. A cache that another output directory
    # filled would otherwise pass for fresh: cargo knows an adapter by its place inside its own
    # workspace, and such a cache names the cell module of the other copy.
    shutil.copytree(
        HERE / "harness" / adapter,
        stage / adapter,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("target"),
        copy_function=shutil.copy,
    )
    shutil.copy(HERE / "harness" / "cells.rs", stage / "cells.rs")
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
    built = target / "release" / adapter
    if built.stat().st_mtime < (stage / "cells.rs").stat().st_mtime:
        raise ValueError(f"{built} is older than the copy it was built from")
    binary = output / "bin" / column
    binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, binary)
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
    # With this flag a cell that panics says why on stderr, and the record keeps the end of it.
    command = [str(args.output / "bin" / column), *cell, "--show-panics"]
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
    except subprocess.TimeoutExpired as exc:
        # A thread that panicked before a barrier leaves its cell waiting: what the process wrote
        # to stderr until the timeout is kept (the exception holds it as bytes).
        tail = (exc.stderr or b"").decode("utf-8", "replace")[-300:]
        context["rc"], lines = 124, ""
        context["stderr"] = f"timeout; {tail}" if tail else "timeout"
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
    # The parity cell writes one record per string of the corpus: a non-blank line of the file.
    with corpus.open(encoding="utf-8") as handle:
        strings = sum(1 for line in handle if line.strip())
    totals = {}
    lines = []

    def digest(column, family, scratch, digests):
        out = Path(scratch) / f"{column}.jsonl"
        command = [str(args.output / "bin" / column), "parity"]
        command += [str(args.models / family), str(corpus), str(out), "--no-inc"]
        done = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        status[column] = done.returncode
        # The cell makes its file once the column has loaded the family.
        if out.exists():
            written.add(column)
        found = {}
        try:
            for record in records(out):
                value = record.get("ids")
                text = json.dumps(value).encode()
                found[record["i"]] = (
                    hashlib.sha256(text).hexdigest() if value is not None else "error"
                )
        except ValueError:
            # A line that does not parse is where the file was cut off: the strings before it are
            # kept, and the summary names the column, whose file is short of a record per string.
            pass
        digests[column] = found

    for family in families_of(args):
        columns = columns_of(args, family)
        digests, status, written = {}, {}, set()
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
            if column not in written:
                # A column that cannot load the family ends before its cell makes the file: no
                # string of it is counted. A file without a record is compared as any other.
                parts.append(f"{column} did not load")
                continue
            same = sum(1 for index, value in ours.items() if theirs.get(index) == value)
            total = totals.setdefault(column, [0, 0])
            total[0] += same
            total[1] += len(ours)
            parts.append(f"{column} {same} of {len(ours)}")
        # A process that died leaves a short file, and its missing strings read as differences.
        ended = [f"{column} {code}" for column, code in sorted(status.items()) if code]
        if ended:
            parts.append("exit status not 0: " + ", ".join(ended))
        # A file without a whole record for every string is named whatever the exit status: a
        # write that failed leaves it short, cut off inside a line or empty, and the cell still
        # ends at status 0. A column that did not load made no file and is named above.
        short = [
            f"{column} {len(digests[column])} of {strings}"
            for column in columns
            if len(digests[column]) != strings and (column in written or column == "smg")
        ]
        if short:
            parts.append("incomplete output: " + ", ".join(short))
        lines.append("; ".join(parts))
        print(lines[-1], flush=True)
    lines.append("TOTAL SAME: " + "; ".join(f"{c} {t[0]} of {t[1]}" for c, t in totals.items()))
    print(lines[-1])
    (args.output / "ids-summary.txt").write_text("\n".join(lines) + "\n")


ENCODE_HEAD = "family,class,mode,arm,round,rc,ns_per_token_p50,reps,tokens,load_ms,backend"
MISC_HEAD = (
    "family,cell,arm,round,rc,full_ns_per_token_p50,inc_ns_per_token_p50,step_ns_p50,"
    "first_delta_ns_p50,stream_step_ns_p50,stream_step_ns_p99,stream_first_delta_wall_ns_p50,"
    "stream_first_delta_own_ns_p50,batch_tokens,batch_loop_tok_per_s_p50,"
    "batch_api_tok_per_s_p50,backend"
)
MEMORY_HEAD = (
    "family,arm,load_ms,rss_kb_before_load,rss_kb_after_load,hwm_kb_after_500k,hwm_kb_after_1m,"
    "allocs_2k_encode,growth_kb_per_thread_shared,growth_kb_per_thread_own,level_kb_per_thread_own,"
    "backend"
)
# What names a cell of the decode group: the key of its size in the cell's line, its argument.
SIZED = {"dec": ("len", "--len"), "stream": ("streams", "--streams"), "batch": ("n", "--n")}


def p50(out, name):
    value = out.get(name)
    return value.get("p50") if isinstance(value, dict) else None


def shown(value, form):
    return "" if value is None else format(value, form)


def option(arguments, name):
    return arguments[arguments.index(name) + 1] if name in arguments else None


def write_csv(path, head, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(head.split(","))
        writer.writerows(rows)
    print(f"{path}: {len(rows)} lines")


def collect(args):
    """The three csv files that tables.py reads, from the raw lines of `run`: a line per process.

    A process that failed keeps its line with an empty figure. `--release` names the column
    `smg` (`smg-1.16.0`): the name tables.py takes as `--new`.
    """
    raw = args.output / "raw"
    target = args.csv or args.output
    target.mkdir(parents=True, exist_ok=True)
    ours = f"smg-{args.release}" if args.release else "smg"

    def column(ctx):
        return ours if ctx["column"] == "smg" else ctx["column"]

    def backend(out):
        # One csv field without a comma; an error text inside it is cut short.
        return (out.get("backend") or "").replace(",", ";")[:110]

    rows = []
    for record in records(raw / "encode.jsonl"):
        ctx, out = record["ctx"], record["out"] or {}
        figure = p50(out, "ns_per_tok") if ctx["rc"] == 0 else None
        rows.append(
            [
                ctx["family"],
                out.get("class") or option(ctx["args"], "--class"),
                out.get("mode") or option(ctx["args"], "--mode"),
                column(ctx),
                ctx["round"],
                ctx["rc"],
                shown(figure, ".3f"),
                shown(out.get("reps"), ""),
                shown(out.get("total_tokens"), ".0f"),
                shown(out.get("load_ms"), ".1f"),
                backend(out),
            ]
        )
    order = {name: index for index, name in enumerate(CLASSES)}
    rows.sort(key=lambda row: (row[0], order.get(row[1], len(order)), row[2] or "", *row[3:5]))
    write_csv(target / "encode-rounds.csv", ENCODE_HEAD, rows)

    rows = []
    for record in records(raw / "decode.jsonl"):
        ctx, out = record["ctx"], record["out"] or {}
        kind = out.get("cell") or ctx["args"][0]
        if kind not in SIZED:
            continue
        key, flag = SIZED[kind]
        figures = [None] * 11
        # A cell checks its own output: a stream text that differs from the full decode is no result.
        wrong = out.get("inc_eq_full") is False or bool(out.get("inc_ne_full"))
        status = "incorrect" if ctx["rc"] == 0 and wrong else ctx["rc"]
        if status == 0:
            step = out.get("step_ns") or {}
            figures = [
                p50(out, "full_ns_per_tok"),
                p50(out, "inc_ns_per_tok"),
                None if kind == "stream" else step.get("p50"),
                p50(out, "first_delta_ns"),
                step.get("p50") if kind == "stream" else None,
                step.get("p99") if kind == "stream" else None,
                p50(out, "first_delta_wall_ns"),
                p50(out, "first_delta_own_ns"),
                out.get("tokens"),
                p50(out, "loop_tok_per_s"),
                p50(out, "api_tok_per_s") if out.get("api") else None,
            ]
        cell = f"{kind}{out.get(key) or option(ctx['args'], flag)}"
        head = [ctx["family"], cell, column(ctx), ctx["round"], status]
        rows.append(head + [shown(value, ".3f") for value in figures] + [backend(out)])
    rows.sort(key=lambda row: row[:4])
    write_csv(target / "misc-rounds.csv", MISC_HEAD, rows)

    found = {}
    for record in records(raw / "memory.jsonl"):
        ctx, out = record["ctx"], record["out"] or {}
        if ctx["rc"] == 0 and out.get("cell") in ("info", "mem"):
            found.setdefault((ctx["family"], column(ctx)), {})[out["cell"]] = out
    # This release first, then earlier releases, then the references in the order of COLUMNS.
    references = [name for name in COLUMNS if name != "smg"]
    earlier = sorted({name for _, name in found} - {ours} - set(references))
    rows = []
    for model in MODELS:
        for name in [ours, *earlier, *references]:
            cells = found.get((model["name"], name))
            if not cells:
                continue
            info, mem = cells.get("info", {}), cells.get("mem", {})
            rows.append(
                [
                    model["name"],
                    name,
                    shown(info.get("load_ms"), ".1f"),
                    shown(info.get("rss_kb_before"), ".0f"),
                    shown(info.get("rss_kb_after"), ".0f"),
                    shown(mem.get("hwm_kb_after_500k"), ".0f"),
                    shown(mem.get("hwm_kb_after_1m"), ".0f"),
                    shown(mem.get("allocs_2k_encode"), ".0f"),
                    shown(mem.get("growth_kb_per_thread_shared"), ".0f"),
                    shown(mem.get("growth_kb_per_thread_own"), ".0f"),
                    shown(mem.get("level_kb_per_thread_own"), ".0f"),
                    backend(info or mem),
                ]
            )
    write_csv(target / "load-memory.csv", MEMORY_HEAD, rows)


def main():
    def names(text):
        return [name for name in text.split(",") if name]

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("action", choices=["prepare", "ids", "run", "collect"])
    parser.add_argument("--corpus", type=Path, help="perf/<class>/*.txt, parity.jsonl")
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
    parser.add_argument(
        "--csv", type=Path, help="collect: where the csv files go (default --output)"
    )
    parser.add_argument("--release", help="collect: the column smg is written as smg-RELEASE")
    args = parser.parse_args()
    if args.action == "run" and not args.cores:
        parser.error("run needs --cores")
    if args.action != "collect" and not args.corpus:
        parser.error(f"{args.action} needs --corpus")
    if args.edge:
        args.cells = ["encode"]
    for name in ("corpus", "models", "output", "csv", "fastokens_src"):
        if getattr(args, name):
            setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "ids": ids, "run": run, "collect": collect}[args.action](args)


if __name__ == "__main__":
    main()
