#!/usr/bin/env python3
"""The release comparison of the multimodal crate: two builds of one harness, cell by cell.

The harness is the crate's `mm_baseline` example. One source of it is compiled against two
revisions of the library (`build`), and the two executables run the same class lists in
interleaved rounds on pinned cores (`rounds`). `parity` compares what the two write on a
wider class list, on two wires and under two alpha policies, digests only. `reference` runs
the repository's reference arm (the pinned reference processors) on the same class lists.
`summarize` writes `RESULTS.md`, `cells.csv`, `rounds.csv` and `environment.json`.

    python3 run.py build     --base v1.15.0 --head HEAD --out <run dir>
    python3 run.py classes   --files <dir> --configs <configs.json> --out <run dir>/classes
    python3 run.py rounds    --out <run dir> --classes <run dir>/classes --cores 2,4,6,8
    python3 run.py parity    --out <run dir> --classes <run dir>/classes-wide --cores 2,4,6,8
    python3 run.py reference --out <run dir> --classes <run dir>/classes --cores 2,4,6,8
    python3 run.py summarize --out <run dir> --classes <run dir>/classes

Standard library only. README.md beside this file says what a cell measures and how the
recorded results were produced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLE = "crates/multimodal/examples/mm_baseline.rs"
MANIFEST = "crates/multimodal/Cargo.toml"
SIDES = ("base", "head")
# The reference arm the repository already holds (the pinned processors, one compute thread).
REFERENCE = HERE.parents[3] / "scripts" / "bench" / "multimodal" / "reference.py"

# The standard classes, in the order of the tables: the short name a class list uses and
# how a table calls it. `batch-10x1k` is one request of ten images; the others are one image.
CLASSES = {
    "jpeg-256": "JPEG 256x256",
    "jpeg-1k": "JPEG 1024x768",
    "jpeg-1080p": "JPEG 1920x1080",
    "jpeg-4k": "JPEG 4096x2304",
    "jpeg-8k": "JPEG 8192x8192",
    "jpeg-progressive": "JPEG progressive",
    "jpeg-cmyk": "JPEG CMYK",
    "jpeg-gray": "JPEG greyscale",
    "jpeg-exif6": "JPEG EXIF-rotated",
    "png-256": "PNG 256x256",
    "png-1k": "PNG 1024x768",
    "png-1080p": "PNG 1920x1080",
    "png-4k": "PNG 3840x2160",
    "png-rgba": "PNG RGBA (alpha)",
    "webp-lossy": "WebP lossy",
    "webp-lossless": "WebP lossless",
    "gif-static": "GIF",
    "bmp": "BMP",
    "tiff": "TIFF",
    "batch-10x1k": "batch of ten 1024x768 JPEGs",
}
BATCH = "batch-10x1k"
# A cell is faster or slower when its time moves by more than this fraction; else level.
LEVEL = 0.02


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_id(content: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(content)}\0".encode())
    digest.update(content)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


# --- build: one harness source against two revisions of the library ---------------------


def add_allocator_line(tree: Path, source_manifest: str) -> None:
    """A revision whose crate manifest has no jemalloc line gets the harness revision's."""
    manifest = tree / MANIFEST
    text = manifest.read_text()
    if "tikv-jemallocator" in text:
        return
    line = next(
        (item for item in source_manifest.splitlines() if item.startswith("tikv-jemallocator")),
        None,
    )
    if line is None:
        raise SystemExit("the harness revision's crate manifest names no jemalloc line")
    lines = text.split("\n")
    if "[dev-dependencies]" in lines:
        lines.insert(lines.index("[dev-dependencies]") + 1, line)
    else:
        lines += ["[dev-dependencies]", line]
    manifest.write_text("\n".join(lines))


def export(root: Path, commit: str, tree: Path) -> None:
    """The files of one commit in a directory of their own: no checkout is touched."""
    tree.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{tree.name}-", dir=tree.parent) as temporary:
        extracted = Path(temporary) / "tree"
        extracted.mkdir()
        with subprocess.Popen(
            ["git", "-C", str(root), "archive", commit], stdout=subprocess.PIPE
        ) as archive:
            subprocess.run(["tar", "-x", "-C", str(extracted)], stdin=archive.stdout, check=True)
        if archive.returncode != 0:
            raise SystemExit(f"git archive {commit} failed")
        extracted.rename(tree)


def build(args: argparse.Namespace) -> int:
    """Each side from an export of its revision, with the harness source of --harness."""
    root = Path(git(HERE, "rev-parse", "--show-toplevel"))
    out = Path(args.out).resolve()
    (out / "bin").mkdir(parents=True, exist_ok=True)
    harness_rev = git(root, "rev-parse", f"{args.harness}^{{commit}}")
    source = git(root, "show", harness_rev + ":" + EXAMPLE) + "\n"
    source_manifest = git(root, "show", harness_rev + ":" + MANIFEST)
    sides = {}
    for side, rev in (("base", args.base), ("head", args.head)):
        commit = git(root, "rev-parse", f"{rev}^{{commit}}")
        tree = out / "trees" / f"{side}-{commit[:9]}"
        if not tree.exists():
            export(root, commit, tree)
        (tree / EXAMPLE).write_text(source)
        add_allocator_line(tree, source_manifest)
        env = dict(os.environ)
        if args.cargo_target:
            env["CARGO_TARGET_DIR"] = str(Path(args.cargo_target).resolve())
        target = Path(env.get("CARGO_TARGET_DIR", tree / "target"))
        # the toolchain file of each revision picks the compiler: each release builds with its own
        subprocess.run(
            ["cargo", "build", "--release", "-p", "llm-multimodal", "--example", "mm_baseline"],
            cwd=tree,
            env=env,
            check=True,
        )
        binary = out / "bin" / f"mm_baseline-{side}"
        shutil.copy2(target / "release" / "examples" / "mm_baseline", binary)
        sides[side] = {
            "revision": rev,
            "commit": commit,
            "harness_commit": harness_rev,
            "harness_blob": _git_blob_id(source.encode()),
            "cargo": subprocess.run(
                ["cargo", "--version"], cwd=tree, capture_output=True, text=True, check=True
            ).stdout.strip(),
            "binary_sha256": sha256_file(binary),
        }
        print(f"build: {side} = {commit[:9]} -> {binary}", file=sys.stderr)
    (out / "sides.json").write_text(json.dumps(sides, indent=1) + "\n")
    return 0


# --- classes: one list per processor configuration --------------------------------------


def classes(args: argparse.Namespace) -> int:
    """Write `<name>.json` per configuration: the classes of one list over the directory of files."""
    images = Path(args.files).resolve()
    spec = json.loads(Path(args.configs).read_text())
    names = json.loads(Path(args.standard).read_text())["classes"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for config in spec["configurations"]:
        config_dir = Path(config["config_dir"]).resolve()
        if not config_dir.is_dir():
            raise SystemExit(f"{config['name']}: no config directory {config_dir}")
        records = []
        for item in names:
            files = [images / name for name in item["files"]]
            missing = [str(path) for path in files if not path.is_file()]
            if missing:
                raise SystemExit(f"class {item['name']}: missing {missing}")
            records.append(
                {
                    "name": f"{config['name']}/{item['name']}",
                    "title": item.get("title", item["name"]),
                    "kind": item["kind"],
                    "model_id": config["model_id"],
                    "config_dir": str(config_dir),
                    "files": [str(path) for path in files],
                }
            )
        listing = {"label": config.get("label", config["name"]), "classes": records}
        (out / f"{config['name']}.json").write_text(json.dumps(listing, indent=1) + "\n")
    print(f"classes: {len(spec['configurations'])} lists in {out}", file=sys.stderr)
    return 0


# --- rounds: the two executables interleaved on pinned cores ----------------------------


def sibling_cpus(core: int) -> list[int]:
    """The other logical CPUs of the same core (empty where there is none)."""
    path = Path(f"/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list")
    try:
        text = path.read_text().strip()
    except OSError:
        return []
    cpus: set[int] = set()
    for part in text.split(","):
        if "-" in part:
            first, last = part.split("-")
            cpus.update(range(int(first), int(last) + 1))
        elif part:
            cpus.add(int(part))
    cpus.discard(core)
    return sorted(cpus)


def cpu_ticks(cpus: list[int]) -> tuple[int, int]:
    """Busy and total clock ticks of the cpus (busy = every field but idle and iowait)."""
    wanted = {f"cpu{cpu}" for cpu in cpus}
    busy = total = 0
    with open("/proc/stat", encoding="ascii") as handle:
        for line in handle:
            fields = line.split()
            if fields and fields[0] in wanted and len(fields) >= 9:
                values = [int(x) for x in fields[1:9]]
                total += sum(values)
                busy += sum(values) - values[3] - values[4]
    return busy, total


class Log:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def __call__(self, text: str) -> None:
        with self.lock, open(self.path, "a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {text}\n")


def window(
    cmd: list[str], core: int, stem: Path, limit: float, timeout: int, log: Log, tag: str
) -> None:
    """One harness process on one core. A window whose sibling thread was busy is taken again."""
    siblings = sibling_cpus(core)
    env = dict(os.environ, SMG_MM_THREADS="1")
    for attempt in (1, 2, 3):
        partial = Path(f"{stem}.jsonl.part")
        partial.unlink(missing_ok=True)
        before = cpu_ticks(siblings)
        with open(f"{stem}.log", "w", encoding="utf-8") as output:
            try:
                status = subprocess.run(
                    ["taskset", "-c", str(core), *cmd, "--out", f"{stem}.jsonl.part"],
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    env=env,
                    timeout=timeout,
                    check=False,
                ).returncode
            except subprocess.TimeoutExpired:
                status = 124
        after = cpu_ticks(siblings)
        ticks = after[1] - before[1]
        share = 100.0 * (after[0] - before[0]) / ticks if siblings and ticks > 0 else 0.0
        if status != 0:
            log(f"BADEXIT {tag} core {core} status {status} try {attempt}")
            if partial.exists():
                partial.replace(f"{stem}.jsonl.bad{attempt}")
            if attempt == 3:
                raise RuntimeError(f"{tag} failed after 3 attempts (last status {status})")
            continue
        if not partial.exists():
            log(f"NOFILE {tag} core {core}: the harness wrote no file")
            if attempt == 3:
                raise RuntimeError(f"{tag} failed after 3 attempts (the harness wrote no file)")
            continue
        if share <= limit or attempt == 3:
            partial.replace(f"{stem}.jsonl")
            Path(f"{stem}.sib").write_text(f"{share:.2f}\n")
            if share <= limit:
                log(f"WINDOW {tag} core {core} sibling {share:.1f} percent")
            else:
                log(
                    f"LOUD {tag} core {core}: three busy windows one after another, "
                    f"the last one kept ({share:.1f} percent)"
                )
            return
        log(f"VOID {tag} core {core} try {attempt}: sibling {share:.1f} percent")
        os.replace(f"{stem}.jsonl.part", f"{stem}.jsonl.void{attempt}")


def binaries(out: Path, args: argparse.Namespace) -> dict[str, Path]:
    found = {
        "base": Path(args.base_bin) if args.base_bin else out / "bin" / "mm_baseline-base",
        "head": Path(args.head_bin) if args.head_bin else out / "bin" / "mm_baseline-head",
    }
    for side, path in found.items():
        if not os.access(path, os.X_OK):
            raise SystemExit(f"no executable for the {side} side: {path}")
    return found


def lists(directory: Path, only: str | None) -> list[str]:
    names = sorted(path.stem for path in directory.glob("*.json"))
    if only:
        wanted = only.split(",")
        missing = [name for name in wanted if name not in names]
        if missing:
            raise SystemExit(f"no class list for {missing} in {directory}")
        names = wanted
    if not names:
        raise SystemExit(f"no class list in {directory}")
    return names


def pinning_cores(value: str) -> list[int]:
    """Parse the cores and check the pinning program before a command writes output."""
    try:
        cores = [int(core) for core in value.split(",")]
    except ValueError as error:
        raise SystemExit(f"invalid --cores {value!r}: expected comma-separated integers") from error
    if shutil.which("taskset") is None:
        raise SystemExit("this command requires taskset (util-linux)")
    allowed = os.sched_getaffinity(0)
    if unknown := sorted(set(cores) - allowed):
        raise SystemExit(f"--cores {unknown} not in this process's CPUs {sorted(allowed)}")
    seen: set[int] = set()
    duplicates: set[int] = set()
    for core in cores:
        if core in seen:
            duplicates.add(core)
        seen.add(core)
    if duplicates:
        raise SystemExit(f"--cores names {sorted(duplicates)} more than once")
    chosen = set(cores)
    if siblings := sorted(core for core in chosen if chosen.intersection(sibling_cpus(core))):
        raise SystemExit(f"--cores {siblings} are thread siblings")
    return cores


def spread(args: argparse.Namespace, out: Path, cores: list[int], jobs_of) -> None:
    """One thread per core; a core takes every n-th configuration and runs its jobs in order."""
    names = lists(Path(args.classes), args.only)
    stop = out / "STOP"
    failed = threading.Event()
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker(index: int, core: int) -> None:
        try:
            for job in jobs_of([name for i, name in enumerate(names) if i % len(cores) == index]):
                if stop.exists() or failed.is_set():
                    return
                job(core)
        except BaseException as error:  # re-raised by the caller after every thread stops
            with errors_lock:
                errors.append(error)
            failed.set()

    threads = [threading.Thread(target=worker, args=(i, core)) for i, core in enumerate(cores)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]


def input_identities(directory: Path, names: list[str]) -> dict[str, str]:
    """Digest each class list together with the media and configuration files it names."""
    identities = {}
    config_names = {"config.json", "preprocessor_config.json", "processor_config.json"}
    for name in names:
        listing_path = directory / f"{name}.json"
        listing = json.loads(listing_path.read_text())
        digest = hashlib.sha256(listing_path.read_bytes())
        for record in listing["classes"]:
            for path in map(Path, record["files"]):
                digest.update(sha256_file(path).encode())
            config_dir = Path(record["config_dir"])
            for path in sorted(config_dir.glob("*.json")):
                if path.name in config_names:
                    digest.update(path.name.encode())
                    digest.update(sha256_file(path).encode())
        identities[name] = digest.hexdigest()
    return identities


def _old_command_file(out: Path, directory: str, records: tuple[str, ...]) -> Path | None:
    """The first output made by an earlier measurement command, if one is present."""
    if out.is_symlink() or (out.exists() and not out.is_dir()):
        return out
    output_path = out / directory
    if output_path.is_symlink() or (output_path.exists() and not output_path.is_dir()):
        return output_path
    if output_path.is_dir():
        for group_path in output_path.iterdir():
            if group_path.is_symlink() or not group_path.is_dir():
                return group_path
            for path in group_path.iterdir():
                return path
    for name in records:
        path = out / name
        if path.exists() or path.is_symlink():
            return path
    return None


def _old_rounds_file(out: Path) -> Path | None:
    """The first output made by an earlier rounds command, if one is present."""
    return _old_command_file(out, "rounds", ("windows.log", "rounds.json"))


def rounds(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    if old_file := _old_rounds_file(out):
        raise SystemExit(f"{out}: rounds requires a fresh run directory; found {old_file}")
    found = binaries(out, args)
    directory = Path(args.classes).resolve()
    cores = pinning_cores(args.cores)
    settings = {
        "rounds": args.rounds,
        "cores": args.cores,
        "repeats": args.repeats,
        "wire": args.wire,
        "transparency": args.transparency,
        "sibling_limit_percent": args.sibling_limit,
        "timeout_seconds": args.timeout,
        "binary_sha256": {side: sha256_file(path) for side, path in found.items()},
        "inputs": input_identities(directory, lists(directory, args.only)),
    }
    out.mkdir(parents=True, exist_ok=True)
    settings_path = out / "rounds.json"
    settings_path.write_text(json.dumps(settings, indent=1) + "\n")
    (out / "rounds").mkdir(exist_ok=True)
    log = Log(out / "windows.log")

    def jobs_of(share: list[str]):
        # round-major: every configuration once a round, the sides in swapped order each round
        for number in range(1, args.rounds + 1):
            order = SIDES if number % 2 else SIDES[::-1]
            for name in share:
                for side in order:
                    stem = out / "rounds" / f"{side}-r{number}" / name
                    stem.parent.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        str(found[side]),
                        "--manifest",
                        str(directory / f"{name}.json"),
                        "--wire",
                        args.wire,
                        "--transparency",
                        args.transparency,
                        "--threads",
                        "1",
                        "--repeats",
                        str(args.repeats),
                    ]
                    tag = f"{side} r{number} {name}"
                    yield lambda core, cmd=cmd, stem=stem, tag=tag: window(
                        cmd, core, stem, args.sibling_limit, args.timeout, log, tag
                    )

    spread(args, out, cores, jobs_of)
    log("ROUNDS END")
    return 0


def parity(args: argparse.Namespace) -> int:
    """Digests only: every class list on two wires and both alpha policies, one repeat."""
    out = Path(args.out).resolve()
    if old_file := _old_command_file(out, "parity", ("parity.log",)):
        raise SystemExit(f"{out}: parity requires fresh output; found {old_file}")
    found = binaries(out, args)
    directory = Path(args.classes).resolve()
    cores = pinning_cores(args.cores)
    log = Log(out / "parity.log")
    (out / "parity").mkdir(parents=True, exist_ok=True)

    def jobs_of(share: list[str]):
        for name in share:
            for wire in ("bf16", "f32"):
                for policy in ("composite_white", "keep_alpha"):
                    for side in SIDES:
                        stem = out / "parity" / f"{name}-{wire}-{policy}-{side}"
                        cmd = [
                            str(found[side]),
                            "--manifest",
                            str(directory / f"{name}.json"),
                            "--wire",
                            wire,
                            "--transparency",
                            policy,
                            "--threads",
                            "1",
                            "--repeats",
                            "1",
                        ]
                        tag = f"{side} {name} {wire} {policy}"
                        # no time is read here: a busy sibling does not void a window
                        yield lambda core, cmd=cmd, stem=stem, tag=tag: window(
                            cmd, core, stem, 100.0, args.timeout, log, tag
                        )

    spread(args, out, cores, jobs_of)
    log("PARITY END")
    return 0


def reference(args: argparse.Namespace) -> int:
    """The reference arm on the same class lists: the repository's reference.py, one compute thread."""
    out = Path(args.out).resolve()
    if old_file := _old_command_file(out, "reference", ("reference.log",)):
        raise SystemExit(f"{out}: reference requires fresh output; found {old_file}")
    directory = Path(args.classes).resolve()
    script = Path(args.script).resolve() if args.script else REFERENCE
    if not script.is_file():
        raise SystemExit(f"no reference arm at {script}")
    cores = pinning_cores(args.cores)
    log = Log(out / "reference.log")
    tmp = out / "reference" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    # the processors load from the config directories alone: nothing is fetched
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

    def jobs_of(share: list[str]):
        for number in range(1, args.rounds + 1):
            for name in share:
                stem = out / "reference" / f"r{number}" / name
                stem.parent.mkdir(parents=True, exist_ok=True)
                cmd = [
                    args.python,
                    "-I",
                    str(script),
                    "--manifest",
                    str(directory / f"{name}.json"),
                    "--threads",
                    "1",
                    "--repeats",
                    str(args.repeats),
                    "--tmp",
                    str(tmp),
                ]
                tag = f"reference r{number} {name}"
                yield lambda core, cmd=cmd, stem=stem, tag=tag: window(
                    cmd, core, stem, args.sibling_limit, args.timeout, log, tag
                )

    spread(args, out, cores, jobs_of)
    log("REFERENCE END")
    return 0


# --- summarize --------------------------------------------------------------------------


def load(path: Path) -> tuple[dict, dict[str, dict]]:
    meta, records = {}, {}
    for line in path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        if record.get("kind") == "meta":
            meta = record
        elif "class" in record:
            records[record["class"].split("/", 1)[-1]] = record
    return meta, records


def what_is_written(record: dict) -> str:
    """Everything of an output that does not depend on the clock: the digests, the shape, the grid."""
    return json.dumps(
        [record.get("shape"), record.get("grid"), record.get("digests")], sort_keys=True
    )


def label_of(directories: list[Path], name: str) -> str:
    for directory in directories:
        if (directory / f"{name}.json").is_file():
            return json.loads((directory / f"{name}.json").read_text()).get("label", name)
    return name


def class_order(
    directories: list[Path], names: list[str]
) -> tuple[list[str], dict[str, str], set[str], str]:
    """The classes of the first class list found: their order, their titles, the batches, the unit word."""
    for name in names:
        for folder in directories:
            path = folder / f"{name}.json"
            if not path.is_file():
                continue
            records = json.loads(path.read_text())["classes"]
            order = [record["name"].split("/", 1)[-1] for record in records]
            titles = {
                short: record.get("title") or CLASSES.get(short, short)
                for short, record in zip(order, records)
            }
            batches = {short for short, record in zip(order, records) if record["kind"] == "batch"}
            unit = (
                "clip"
                if all(record["kind"] in ("audio", "video") for record in records)
                else "image"
            )
            if set(order) <= set(CLASSES):
                order = [short for short in CLASSES if short in order]
            return order, titles, batches, unit
    return list(CLASSES), dict(CLASSES), set([BATCH]), "image"


def no_reference(error: str) -> str:
    """Why a configuration has no reference column, in a few words."""
    if "custom code" in error:
        return "its processor is checkpoint code"
    if "Unrecognized image processor" in error:
        return "no such class in the library"
    return "the reference arm refused it" if error else "not run"


def fmt_ms(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}" if value >= 10 else f"{value:.2f}"


def fmt_pct(value: float) -> str:
    pct = round(value * 100)
    return f"{pct:+d} %" if pct else "0 %"


def code(label: str) -> str:
    """A label as a table shows it: a class list states its label with the code marks in it."""
    return label


def plain(label: str) -> str:
    """A label as a csv cell: no code marks."""
    return label.replace("`", "")


def read_rounds(roots: list[Path]) -> tuple[dict, dict, list[str], dict]:
    """cells[name][class] = per-side lists by round; the meta of both sides; the problems."""
    cells: dict = {}
    metas: dict = {side: {} for side in SIDES}
    problems: list[str] = []
    numbers = sorted(
        {
            int(path.name.rsplit("-r", 1)[1])
            for out in roots
            for path in (out / "rounds").glob("base-r*")
            if path.is_dir()
        }
    )
    windows = []
    for number in numbers:
        for side in SIDES:
            paths = sorted(
                p for out in roots for p in (out / "rounds" / f"{side}-r{number}").glob("*.jsonl")
            )
            for path in paths:
                name = path.stem
                meta, records = load(path)
                metas[side] = meta or metas[side]
                try:
                    share = float(path.with_suffix(".sib").read_text().split()[0])
                except (OSError, ValueError, IndexError):
                    share = None
                for short, record in records.items():
                    cell = cells.setdefault(name, {}).setdefault(short, {s: {} for s in SIDES})
                    if record.get("error") is not None:
                        cell[side][number] = {"error": str(record["error"])}
                        continue
                    rate = record.get("items_per_s")
                    if not rate:
                        problems.append(f"{side} round {number} {name}/{short}: no rate")
                        continue
                    cell[side][number] = {
                        "ms": 1000.0 / rate,
                        "cpu": record.get("cpu_ms_per_item"),
                        "written": what_is_written(record),
                        "digests": record.get("digests") or {},
                        "shape": record.get("shape"),
                        "grid": record.get("grid"),
                        "stages": {
                            k: v.get("median_ms") for k, v in record.get("stage_ms", {}).items()
                        },
                    }
                    windows.append((side, number, name, short, cell[side][number], share))
    # the rounds that were read: the ones both sides hold a window of
    held = {side: {number for s, number, *_ in windows if s == side} for side in SIDES}
    read = sorted(held["base"] & held["head"])
    return cells, metas, problems, {"numbers": read, "records": windows}


def read_reference(roots: list[Path]) -> tuple[dict, dict]:
    """ref[name][class] = the reference arm by round (time per image, digests, or the error); its meta."""
    ref: dict = {}
    meta: dict = {}
    for out in roots:
        for path in sorted((out / "reference").glob("r*/*.jsonl")):
            number = int(path.parent.name[1:])
            found, records = load(path)
            meta = found or meta
            for short, record in records.items():
                cell = ref.setdefault(path.stem, {}).setdefault(short, {})
                if record.get("error") is not None:
                    cell[number] = {"error": str(record["error"])}
                    continue
                cell[number] = {
                    "ms": record["stage_ms"]["total"]["median_ms"]
                    / max(1, record.get("items_per_request") or 1),
                    "cpu": record.get("cpu_ms_per_item"),
                    "digests": record.get("digests") or {},
                    "shape": record.get("shape"),
                    "grid": record.get("grid"),
                    "processor": (record.get("processor") or {}).get("image"),
                }
    return ref, meta


def read_parity(roots: list[Path]) -> dict[str, dict]:
    """Per configuration: outputs compared, equal, differing, refused on both sides, answered by one."""
    counts: dict[str, dict] = {}
    for path in sorted(p for out in roots for p in (out / "parity").glob("*-base.jsonl")):
        other = path.with_name(path.name[: -len("base.jsonl")] + "head.jsonl")
        name = path.name.split("-bf16-")[0].split("-f32-")[0]
        tally = counts.setdefault(
            name,
            {
                "records": 0,
                "compared": 0,
                "equal": 0,
                "differ": 0,
                "refused_on_both": 0,
                "one_sided": 0,
            },
        )
        _, base = load(path)
        _, head = load(other) if other.is_file() else ({}, {})
        for short in sorted(set(base) | set(head)):
            tally["records"] += 1
            a, b = base.get(short), head.get(short)
            if a is None or b is None or (a.get("error") is None) != (b.get("error") is None):
                tally["one_sided"] += 1
            elif a.get("error") is not None:
                tally["refused_on_both"] += 1
            else:
                tally["compared"] += 1
                same = what_is_written(a) == what_is_written(b)
                tally["equal" if same else "differ"] += 1
    return counts


def read_windows(roots: list[Path]) -> tuple[dict[str, int], list[str]]:
    """The window log of the rounds: the windows by outcome, and its first and last stamp."""
    counts = {"WINDOW": 0, "VOID": 0, "LOUD": 0, "BADEXIT": 0, "NOFILE": 0}
    stamps = []
    for root in roots:
        if (root / "windows.log").is_file():
            for line in (root / "windows.log").read_text().splitlines():
                parts = line.split(" ", 2)
                if len(parts) == 3 and parts[1] in counts:
                    counts[parts[1]] += 1
                    stamps.append(parts[0])
    return counts, ([min(stamps), max(stamps)] if stamps else [])


def summarize(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    directory = [Path(item).resolve() for item in args.classes]
    roots = [out, *(Path(extra).resolve() for extra in args.also)]
    cells, metas, problems, raw = read_rounds(roots)
    parity_counts = read_parity(roots)
    ref, ref_meta = read_reference([Path(item).resolve() for item in args.reference_from] or roots)
    sides = json.loads((out / "sides.json").read_text()) if (out / "sides.json").is_file() else {}
    settings = (
        json.loads((out / "rounds.json").read_text()) if (out / "rounds.json").is_file() else {}
    )
    base_name, head_name = args.base_name, args.head_name
    selections = [
        json.loads((root / "rounds.json").read_text()).get("inputs")
        for root in roots
        if (root / "rounds.json").is_file()
    ]
    selected = (
        set().union(*selections)
        if selections and all(isinstance(item, dict) for item in selections)
        else None
    )

    for folder in directory:
        for path in sorted(folder.glob("*.json")):
            if selected is not None and path.stem not in selected:
                continue
            expected = cells.setdefault(path.stem, {})
            for record in json.loads(path.read_text())["classes"]:
                short = record["name"].split("/", 1)[-1]
                expected.setdefault(short, {side: {} for side in SIDES})

    order, titles, batches, unit = class_order(directory, sorted(cells))
    table: dict = {}
    for name in sorted(cells):
        for short in order:
            cell = cells[name].get(short)
            if not cell:
                continue
            # a round counts when both sides have it: the two medians read the same rounds
            both = sorted(
                {k for k, v in cell["base"].items() if "ms" in v}
                & {k for k, v in cell["head"].items() if "ms" in v}
            )
            base = [cell["base"][k] for k in both]
            head = [cell["head"][k] for k in both]
            if not both:
                failures = []
                for side in SIDES:
                    errors = [v["error"] for v in cell[side].values() if "error" in v]
                    if errors:
                        failures.append(f"{side}: {errors[0]}")
                    elif not any("ms" in value for value in cell[side].values()):
                        failures.append(f"{side}: no output")
                refused = "; ".join(failures) or "no paired output"
                table.setdefault(name, {})[short] = {"refused": refused}
                problems.append(f"{name}/{short}: {refused}")
                continue
            written = {
                side: {v["written"] for v in cell[side].values() if "written" in v}
                for side in SIDES
            }
            stable = len(written["base"]) == 1 and len(written["head"]) == 1
            ms_base = statistics.median(v["ms"] for v in base)
            ms_head = statistics.median(v["ms"] for v in head)
            cpu_base = statistics.median(v["cpu"] for v in base)
            cpu_head = statistics.median(v["cpu"] for v in head)
            table.setdefault(name, {})[short] = {
                "ms_base": ms_base,
                "ms_head": ms_head,
                "ms_change": ms_head / ms_base - 1,
                "cpu_base": cpu_base,
                "cpu_head": cpu_head,
                "cpu_change": cpu_head / cpu_base - 1,
                "rounds": len(both),
                "round_ratios": [
                    round(cell["head"][k]["ms"] / cell["base"][k]["ms"], 4) for k in both
                ],
                "rounds_slower": sum(
                    1 for k in both if cell["head"][k]["ms"] > cell["base"][k]["ms"]
                ),
                "same_bytes": (written["base"] == written["head"]) if stable else None,
            }
            # the reference arm's output of the class, where it has one: its time over the head's,
            # and whether the head's decoded pixels and float32 write are the reference's bytes
            refs = ref.get(name, {}).get(short, {})
            timed = [v for _, v in sorted(refs.items()) if "ms" in v]
            if timed:
                mine, theirs = cell["head"][both[-1]], timed[-1]
                ref_ms = statistics.median(v["ms"] for v in timed)
                table[name][short].update(
                    {
                        "ref_ms": ref_ms,
                        "ref_factor": ref_ms / ms_head,
                        "ref_pixels_same": bool(theirs["digests"].get("rgb_sha256"))
                        and theirs["digests"].get("rgb_sha256")
                        == mine["digests"].get("rgb_sha256"),
                        "ref_f32_same": bool(theirs["digests"].get("f32_sha256"))
                        and theirs["digests"].get("f32_sha256")
                        == mine["digests"].get("f32_sha256"),
                        "ref_shape_same": list(theirs["shape"] or []) == list(mine["shape"] or []),
                        # the same read of the base: did the older release write the reference's float32
                        "ref_base_f32_same": bool(theirs["digests"].get("f32_sha256"))
                        and theirs["digests"].get("f32_sha256")
                        == cell["base"][both[-1]]["digests"].get("f32_sha256"),
                    }
                )
            elif refs:
                table[name][short]["ref_error"] = str(next(iter(refs.values())).get("error", ""))[
                    :200
                ]

    summary: dict = {}
    for name, record in table.items():
        image = [c for short, c in record.items() if short not in batches and "ms_change" in c]
        summary[name] = {
            "label": label_of(directory, name),
            "classes": 0,
            "faster": 0,
            "slower": 0,
            "same_bytes": 0,
            "bytes_compared": 0,
            "refused": sorted(short for short, c in record.items() if "refused" in c),
        }
        if not image:
            continue
        changes = sorted(c["ms_change"] for c in image)
        measured = [c for c in record.values() if "ms_change" in c]
        summary[name].update(
            {
                "classes": len(image),
                "ms_median": statistics.median(changes),
                "ms_least": changes[0],
                "ms_most": changes[-1],
                "cpu_median": statistics.median(c["cpu_change"] for c in image),
                "faster": sum(1 for x in changes if x < -LEVEL),
                "slower": sum(1 for x in changes if x > LEVEL),
                "same_bytes": sum(1 for c in measured if c["same_bytes"]),
                "bytes_compared": len(measured),
            }
        )
        timed = [c for c in record.values() if "ref_factor" in c]
        factors = sorted(
            c["ref_factor"]
            for short, c in record.items()
            if short not in batches and "ref_factor" in c
        )
        if factors:
            summary[name].update(
                {
                    "ref_median": statistics.median(factors),
                    "ref_least": factors[0],
                    "ref_most": factors[-1],
                    "ref_compared": len(timed),
                    "ref_pixels_same": sum(1 for c in timed if c["ref_pixels_same"]),
                    "ref_f32_same": sum(1 for c in timed if c["ref_f32_same"]),
                    "ref_base_f32_same": sum(1 for c in timed if c["ref_base_f32_same"]),
                }
            )
        else:
            errors = [c["ref_error"] for c in record.values() if "ref_error" in c]
            if errors:
                summary[name]["ref_error"] = errors[0]

    names = sorted(summary, key=lambda n: summary[n]["label"])
    image_cells = [
        (n, s, c)
        for n in names
        for s, c in table[n].items()
        if s not in batches and "ms_change" in c
    ]
    faster = [x for x in image_cells if x[2]["ms_change"] < -LEVEL]
    slower = [x for x in image_cells if x[2]["ms_change"] > LEVEL]
    level = len(image_cells) - len(faster) - len(slower)
    rounds_read = max(
        (c["rounds"] for n in names for c in table[n].values() if "ms_change" in c),
        default=0,
    )
    for name in names:
        for short, cell in table[name].items():
            if "ms_change" in cell and cell["rounds"] < rounds_read:
                problems.append(
                    f"{name}/{short}: {cell['rounds']} paired rounds, expected {rounds_read}"
                )

    lines = [
        f"# The multimodal release comparison: {head_name} against {base_name}",
        "",
        f"{len(image_cells)} {unit} cells ({len(names)} processor configurations by "
        f"{len(order) - len(batches)} {unit} classes), "
        f"time per {unit} on one core: {len(faster)} faster by more than 2 %, {level} level (within 2 %), "
        f"{len(slower)} slower by more than 2 %. Medians of {rounds_read} interleaved rounds; "
        "change = the head's time over the base's, minus one, so a negative figure is a faster head.",
        "",
    ]
    counts, stamps = read_windows(roots)
    built = [
        f"{name} is `{sides[side]['commit'][:9]}` built with {sides[side]['cargo']}"
        for side, name in (("head", head_name), ("base", base_name))
        if side in sides
    ]
    if len(built) == 2:
        lines += [
            f"{built[0]}; {built[1]}; both sides run the harness source of "
            f"`{sides['head']['harness_commit'][:9]}`."
            + (f" Measured {stamps[0][:10]}." if stamps else "")
            + " What a cell measures is in `crates/multimodal/benches/release/README.md`; beside this "
            "file, `environment.json` holds the identities, `cells.csv` every cell and `rounds.csv` "
            "every window.",
            "",
        ]
    if slower:
        lines.append(
            "Cells that read slower by more than 2 % (in how many rounds the head was the slower side, "
            "and the head's time over the base's in every round):"
        )
        lines.append("")
        for n, s, c in sorted(slower, key=lambda x: -x[2]["ms_change"]):
            lines.append(
                f"- {code(summary[n]['label'])} {titles[s]}: {fmt_pct(c['ms_change'])} "
                f"({fmt_ms(c['ms_head'])} ms against {fmt_ms(c['ms_base'])} ms; "
                f"slower in {c['rounds_slower']} of {c['rounds']} rounds: "
                f"{' '.join(f'{x:.3f}' for x in c['round_ratios'])})"
            )
        lines.append("")
    lines += [
        f"## Every processor configuration, medians over the {unit} classes",
        "",
        f"| processor | time per {unit} | least .. most | CPU per {unit} "
        + ("| batch of ten " if batches else "")
        + "| faster / level / slower | same bytes, the standard classes |"
        + (
            " same bytes, the wider list on two wires, alpha composited and kept |"
            if parity_counts
            else ""
        )
        + (
            " the reference's time over the head's (least .. most) | the reference's bytes: decoded pixels, float32 "
            "| the base's float32 against the reference |"
            if ref
            else ""
        ),
        "|---|---|---|---|---|---|"
        + ("---|" if batches else "")
        + ("---|" if parity_counts else "")
        + ("---|---|---|" if ref else ""),
    ]
    for n in names:
        s = summary[n]
        batch = next((table[n][b] for b in sorted(batches) if b in table[n]), {})
        record = (
            f"| {code(s['label'])} | "
            + (
                f"{fmt_pct(s['ms_median'])} | {fmt_pct(s['ms_least'])} .. {fmt_pct(s['ms_most'])} | "
                f"{fmt_pct(s['cpu_median'])} | "
                if "ms_median" in s
                else "- | - | - | "
            )
            + (
                f"{fmt_pct(batch['ms_change']) if 'ms_change' in batch else '-'} | "
                if batches
                else ""
            )
            + f"{s['faster']} / {s['classes'] - s['faster'] - s['slower']} / {s['slower']} | "
            f"{s['same_bytes']} of {s['bytes_compared']} |"
        )
        if parity_counts:
            p = parity_counts.get(n)
            if p:
                note = []
                if p["refused_on_both"]:
                    note.append(f"{p['refused_on_both']} refused on both")
                if p["one_sided"]:
                    note.append(f"{p['one_sided']} answered by one side")
                record += (
                    f" {p['equal']} of {p['compared']}"
                    + (f" ({'; '.join(note)})" if note else "")
                    + " |"
                )
            else:
                record += " - |"
        if ref:
            if "ref_median" in s:
                record += (
                    f" {s['ref_median']:.2f}x ({s['ref_least']:.2f}x .. {s['ref_most']:.2f}x) |"
                    f" {s['ref_pixels_same']} of {s['ref_compared']}, {s['ref_f32_same']} of {s['ref_compared']} |"
                    f" {s['ref_base_f32_same']} of {s['ref_compared']} |"
                )
            else:
                record += f" - ({no_reference(s.get('ref_error', ''))}) | - | - |"
        lines.append(record)
    lines.append("")
    for n in names:
        s = summary[n]
        lines += [
            f"## {code(s['label'])}, milliseconds per {unit}",
            "",
            f"| class | {head_name}, one thread | vs {base_name} | CPU ms per {unit} | vs {base_name} "
            f"| same bytes as {base_name} |"
            + (" head over base by round |" if args.round_ratios else "")
            + (
                f" reference ms per {unit} | reference over head | the reference's bytes |"
                if "ref_median" in s
                else ""
            ),
            "|---|---|---|---|---|---|"
            + ("---|" if args.round_ratios else "")
            + ("---|---|---|" if "ref_median" in s else ""),
        ]
        for short in order:
            title = titles[short]
            c = table[n].get(short)
            if c is None:
                continue
            if "refused" in c:
                lines.append(
                    f"| {title} | refused: {c['refused'][:80]} | | | | |"
                    + (" |" if args.round_ratios else "")
                )
                continue
            same = "yes" if c["same_bytes"] else ("no" if c["same_bytes"] is False else "unstable")
            line = (
                f"| {title} | {fmt_ms(c['ms_head'])} | {fmt_pct(c['ms_change'])} | {fmt_ms(c['cpu_head'])} | "
                f"{fmt_pct(c['cpu_change'])} | {same} |"
            )
            if args.round_ratios:
                line += f" {' '.join(f'{x:.3f}' for x in c['round_ratios'])} |"
            if "ref_median" in s:
                if "ref_factor" in c:
                    equal = (
                        "yes"
                        if c["ref_pixels_same"] and c["ref_f32_same"]
                        else ("pixels yes, float32 no" if c["ref_pixels_same"] else "no")
                    )
                    line += f" {fmt_ms(c['ref_ms'])} | {c['ref_factor']:.2f}x | {equal} |"
                else:
                    line += " - | - | - |"
            lines.append(line)
        lines.append("")
    # the files land in the run directory, or in --write-to (a second view of the same rounds)
    out = Path(args.write_to).resolve() if args.write_to else out
    out.mkdir(parents=True, exist_ok=True)
    results = out / "RESULTS.md"
    results.write_text("\n".join(lines))

    with open(out / "cells.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "processor",
                "class",
                "base_ms_per_image",
                "head_ms_per_image",
                "change_percent",
                "base_cpu_ms_per_image",
                "head_cpu_ms_per_image",
                "cpu_change_percent",
                "same_bytes",
                "rounds",
                "rounds_head_slower",
                "head_over_base_by_round",
                "reference_ms_per_image",
                "reference_over_head",
                "reference_pixels_equal",
                "reference_float32_equal",
            ]
        )
        for n in names:
            for short in order:
                c = table[n].get(short)
                if c is None or "refused" in c:
                    continue
                writer.writerow(
                    [
                        plain(summary[n]["label"]),
                        short,
                        f"{c['ms_base']:.4f}",
                        f"{c['ms_head']:.4f}",
                        f"{c['ms_change'] * 100:.2f}",
                        f"{c['cpu_base']:.4f}",
                        f"{c['cpu_head']:.4f}",
                        f"{c['cpu_change'] * 100:.2f}",
                        {True: "yes", False: "no", None: "unstable"}[c["same_bytes"]],
                        c["rounds"],
                        c["rounds_slower"],
                        " ".join(f"{x:.4f}" for x in c["round_ratios"]),
                        *(
                            [
                                f"{c['ref_ms']:.4f}",
                                f"{c['ref_factor']:.3f}",
                                "yes" if c["ref_pixels_same"] else "no",
                                "yes" if c["ref_f32_same"] else "no",
                            ]
                            if "ref_factor" in c
                            else ["", "", "", ""]
                        ),
                    ]
                )
    with open(out / "rounds.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "side",
                "round",
                "processor",
                "class",
                "ms_per_image",
                "cpu_ms_per_image",
                "decode_median_ms",
                "preprocess_median_ms",
                "serialize_median_ms",
                "sibling_busy_percent",
            ]
        )
        for side, number, name, short, value, share in raw["records"]:
            if name not in summary:
                continue
            stages = value["stages"]
            writer.writerow(
                [
                    side,
                    number,
                    plain(summary[name]["label"]),
                    short,
                    f"{value['ms']:.4f}",
                    f"{value['cpu']:.4f}",
                    *(f"{stages.get(k, 0.0):.4f}" for k in ("decode", "preprocess", "serialize")),
                    "" if share is None else f"{share:.2f}",
                ]
            )

    kept = (
        "arch",
        "os",
        "turbojpeg",
        "libwebp",
        "libwebp_version",
        "pool_threads",
        "allocator",
        "allocator_conf",
        "allocator_env",
        "wire",
        "transparency",
        "write_thread",
        "threads",
        "repeats",
        "build",
    )
    # what was measured, by digest: the class files and each configuration's config files
    files: dict = {}
    configurations: dict = {}
    for name in names:
        for folder in directory:
            if not (folder / f"{name}.json").is_file():
                continue
            listing = json.loads((folder / f"{name}.json").read_text())
            for item in listing["classes"]:
                for path in map(Path, item["files"]):
                    if path.name not in files and path.is_file():
                        files[path.name] = {
                            "bytes": path.stat().st_size,
                            "sha256": sha256_file(path),
                        }
            config_dir = Path(listing["classes"][0]["config_dir"])
            configurations[plain(summary[name]["label"])] = {
                path.name: sha256_file(path)
                for path in sorted(config_dir.glob("*.json"))
                if path.name in ("config.json", "preprocessor_config.json", "processor_config.json")
            }
            break
    environment = {
        "base": {"name": base_name, **sides.get("base", {})},
        "head": {"name": head_name, **sides.get("head", {})},
        "harness": {side: {k: metas[side].get(k) for k in kept} for side in SIDES},
        # what the last `rounds` command asked for; `rounds_read` below is what the tables read
        "rounds": {("rounds_asked" if k == "rounds" else k): v for k, v in settings.items()},
        "rounds_read": raw["numbers"],
        "reference": {
            k: (re.match(r"[0-9.]*", v).group(0) if k == "python" else v)
            for k, v in ref_meta.items()
            if k not in ("kind", "arm", "cores", "threads")
        },
        "class_files": files,
        "configurations": configurations,
        "windows": counts,
        "measured": stamps,
        "system": platform.system(),
        "machine": platform.machine(),
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "problems": problems,
    }
    (out / "environment.json").write_text(json.dumps(environment, indent=1) + "\n")
    (out / "tables.json").write_text(
        json.dumps({"cells": table, "summary": summary, "parity": parity_counts}, indent=1) + "\n"
    )
    print(
        f"summarize: {len(names)} configurations, {len(image_cells)} {unit} cells: "
        f"{len(faster)} faster, {level} level, "
        f"{len(slower)} slower; rounds {raw['numbers']}; problems {len(problems)}; {results}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser(
        "build", help="the harness of one revision against the base and the head library"
    )
    p.add_argument("--base", required=True, help="the base revision (a tag or a commit)")
    p.add_argument("--head", default="HEAD", help="the head revision [HEAD]")
    p.add_argument(
        "--harness",
        default="HEAD",
        help="the revision whose harness source both sides compile [HEAD]",
    )
    p.add_argument("--out", required=True, help="the run directory")
    p.add_argument("--cargo-target", default=None, help="one target directory for both sides")
    p.set_defaults(run=build)

    p = commands.add_parser("classes", help="one class list per processor configuration")
    p.add_argument(
        "--files",
        required=True,
        help="the directory holding the class files (written by make_images.py or make_audio.py)",
    )
    p.add_argument(
        "--standard",
        default=str(HERE / "classes.json"),
        help="the class list [classes.json; classes_wide.json and audio_classes.json are the others]",
    )
    p.add_argument("--configs", required=True, help="the configurations file (see README.md)")
    p.add_argument("--out", required=True, help="the directory the lists are written to")
    p.set_defaults(run=classes)

    for name, run, text in (
        ("rounds", rounds, "the timed rounds"),
        ("parity", parity, "the digests on a wider list"),
    ):
        p = commands.add_parser(name, help=text)
        p.add_argument("--out", required=True, help="the run directory")
        p.add_argument("--classes", required=True, help="the directory of class lists")
        p.add_argument(
            "--cores", required=True, help="the cpus to pin to, one harness process each (2,4,6)"
        )
        p.add_argument("--only", default=None, help="these lists only (a,b,c)")
        p.add_argument(
            "--base-bin", default=None, help="the base executable [<out>/bin/mm_baseline-base]"
        )
        p.add_argument(
            "--head-bin", default=None, help="the head executable [<out>/bin/mm_baseline-head]"
        )
        p.add_argument(
            "--timeout", type=int, default=1800, help="seconds a harness process may take"
        )
        if name == "rounds":
            p.add_argument("--rounds", type=int, default=5)
            p.add_argument("--repeats", type=int, default=10)
            p.add_argument("--wire", default="bf16")
            p.add_argument("--transparency", default="composite_white")
            p.add_argument(
                "--sibling-limit",
                type=float,
                default=10.0,
                help="a window is void when the core's other logical CPU was busier than this, in percent",
            )
        p.set_defaults(run=run)

    p = commands.add_parser("summarize", help="RESULTS.md, cells.csv, rounds.csv, environment.json")
    p.add_argument("--out", required=True, help="the run directory")
    p.add_argument(
        "--classes", action="append", default=[], help="a directory of class lists (for the labels)"
    )
    p.add_argument("--base-name", default="base")
    p.add_argument("--head-name", default="head")
    p.add_argument(
        "--also", action="append", default=[], help="another run directory read with this one"
    )
    p.add_argument(
        "--reference-from",
        action="append",
        default=[],
        help="read the reference arm from this run directory instead (may repeat)",
    )
    p.add_argument(
        "--write-to", default=None, help="write the files here instead of the run directory"
    )
    p.add_argument(
        "--round-ratios", action="store_true", help="every round's ratio in the per-class tables"
    )
    p.set_defaults(run=summarize)

    p = commands.add_parser("reference", help="the reference arm on the same class lists")
    p.add_argument("--out", required=True, help="the run directory")
    p.add_argument("--classes", required=True, help="the directory of class lists")
    p.add_argument("--cores", required=True, help="the cpus to pin to, one process each (2,4,6)")
    p.add_argument("--only", default=None, help="these lists only (a,b,c)")
    p.add_argument(
        "--python", default=sys.executable, help="the interpreter with the reference arm's pins"
    )
    p.add_argument(
        "--script", default=None, help="the reference arm [scripts/bench/multimodal/reference.py]"
    )
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--sibling-limit", type=float, default=10.0)
    p.add_argument("--timeout", type=int, default=3600, help="seconds a reference process may take")
    p.set_defaults(run=reference)

    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
