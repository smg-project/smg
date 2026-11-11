#!/usr/bin/env python3
"""The multimodal baseline of record, from one command.

Runs SMG's media pipeline (the ``mm_baseline`` example of the multimodal crate: decode,
the processor's one-pass wire write over the borrowed frames, the move of the wire
bytes) and the serving engine's own media path (``reference.py``: the engine's loader
rules and the pinned ``transformers`` processors) over the same files, class by class,
one item at a time and under load, and writes one table naming the box's architecture,
core count and load and the kernel paths the crate took (TurboJPEG or the fallback
decoder, libwebp, the vector paths of this architecture, the preprocessing pool).

The classes come from ``classes.json`` beside this file: photos, screenshots, formats,
a ten-image batch, three clip lengths and a clip of speech, each a file of the fixture
repository's corpus (``--corpus`` names a checkout or a directory in its layout) or,
where the corpus has no public file of that shape yet, a file derived from one by the
recipe the manifest states. Every derived file is written under the output directory
with its source and recipe recorded in ``resolved.json``.

    python3 scripts/bench/multimodal/mm_baseline.py \\
        --corpus <corpus dir> --config-dir qwen3_vl=<dir> --config-dir qwen3_asr=<dir> \\
        --python <venv python with requirements.txt installed> --out <run dir> --threads 16

Arms: ``crate`` (the example, built in release with ``cargo build --release -p
llm-multimodal --example mm_baseline`` unless ``--example`` names a binary),
``reference`` (``reference.py`` under ``--python``). ``--arms`` picks a subset. The crate
arm runs single-thread twice, with the crate's default preprocessing pool and with a
pool of one thread (``SMG_MM_THREADS=1``, the per-item CPU cost with no fan-out), and
once at ``--threads`` threads; the reference arm runs at one process and at
``--threads`` processes with one compute thread each. The table is ``table.md`` and
``table.json`` in the run directory; the raw rows stay beside it as JSON Lines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_average() -> str:
    try:
        return Path("/proc/loadavg").read_text().split()[0]
    except OSError:
        return "?"


# --- the classes: corpus files and derived files -------------------------------------


def materialize(cls: dict, corpus: Path, out_dir: Path, ffmpeg: str) -> tuple[list[Path], dict]:
    """The class's files on disk: the corpus files, or the derived file and its recipe."""
    if "files" in cls:
        files = [corpus / rel for rel in cls["files"]]
        missing = [str(f) for f in files if not f.is_file()]
        if missing:
            raise FileNotFoundError(f"class {cls['name']}: corpus files missing: {missing}")
        return files, {
            "source": "corpus",
            "files": [{"path": rel, "sha256": sha256_file(corpus / rel)} for rel in cls["files"]],
        }
    recipe = cls["synthetic"]
    source = corpus / recipe["from"]
    if not source.is_file():
        raise FileNotFoundError(f"class {cls['name']}: source file missing: {source}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = recipe["format"]
    # the recipe's digest in the name: a changed recipe derives a new file instead of
    # reusing the old one; the file is written beside its name and moved into place whole,
    # so an interrupted write is never taken for the input
    recipe_id = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:12]
    target = out_dir / f"{cls['name']}-{recipe_id}.{'jpg' if fmt == 'jpeg' else fmt}"
    partial = target.with_name(f"{target.stem}.partial{target.suffix}")
    if not target.exists():
        if cls["kind"] == "video":
            concat = recipe.get("concat", 1)
            listing = out_dir / f"{cls['name']}.concat.txt"
            listing.write_text("".join(f"file '{source.resolve()}'\n" for _ in range(concat)))
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(listing),
                    "-c",
                    "copy",
                    str(partial),
                ],
                check=True,
            )
            listing.unlink()
            partial.replace(target)
        else:
            from PIL import (
                Image,  # noqa: PLC0415  (the materializer runs under the pinned environment)
            )

            with Image.open(source) as image:
                image.load()
                image = image.convert("RGB")
                width, height = recipe["resize"]
                image = image.resize((width, height), Image.LANCZOS)
                if fmt == "jpeg":
                    image.save(
                        partial,
                        "JPEG",
                        quality=recipe.get("quality", 90),
                        subsampling=recipe.get("subsampling", "4:2:0"),
                        optimize=False,
                    )
                elif fmt == "png":
                    image.save(partial, "PNG", compress_level=6)
                else:
                    raise ValueError(f"class {cls['name']}: no recipe for format {fmt!r}")
                partial.replace(target)
    provenance = {
        "source": "derived",
        "from": {"path": recipe["from"], "sha256": sha256_file(source)},
        "recipe": {k: v for k, v in recipe.items() if k != "from"},
        "file": {
            "path": str(target),
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        },
    }
    return [target], provenance


def resolve(
    classes: dict,
    corpus: Path,
    config_dirs: dict[str, Path],
    out_dir: Path,
    ffmpeg: str,
    only: set[str] | None,
) -> dict:
    resolved = []
    for cls in classes["classes"]:
        if only and cls["name"] not in only:
            continue
        config = classes["configs"][cls["config"]]
        config_dir = config_dirs.get(cls["config"])
        if config_dir is None:
            raise SystemExit(
                f"--config-dir {cls['config']}=<dir> is required "
                f"(the checkpoint {config['checkpoint']}'s {config['files']})"
            )
        for name in config["files"]:
            if not (config_dir / name).is_file() and name != "video_preprocessor_config.json":
                raise SystemExit(f"{config_dir / name} is missing ({config['checkpoint']})")
        files, provenance = materialize(cls, corpus, out_dir / "derived", ffmpeg)
        resolved.append(
            {
                "name": cls["name"],
                "kind": cls["kind"],
                "processor": cls["processor"],
                "model_id": config["checkpoint"],
                "config_dir": str(config_dir),
                "files": [str(f) for f in files],
                "note": cls.get("note", ""),
                "repeats": cls.get("repeats"),
                "provenance": provenance,
            }
        )
    return {"classes": resolved}


# --- the arms ---------------------------------------------------------------------------


def build_example(cargo_target: str | None) -> Path:
    env = dict(os.environ)
    if cargo_target:
        env["CARGO_TARGET_DIR"] = cargo_target
    subprocess.run(
        ["cargo", "build", "--release", "-p", "llm-multimodal", "--example", "mm_baseline"],
        cwd=ROOT,
        env=env,
        check=True,
    )
    # cargo resolves a relative target directory against its working directory, ROOT
    target = Path(env.get("CARGO_TARGET_DIR", ROOT / "target"))
    if not target.is_absolute():
        target = ROOT / target
    return target / "release" / "examples" / "mm_baseline"


def run_crate(
    example: Path,
    manifest: Path,
    out: Path,
    threads: int,
    repeats: int,
    pool: int | None,
    cores: str | None,
    extra_env: dict,
) -> None:
    env = dict(os.environ)
    env.update(extra_env)
    if pool is not None:
        env["SMG_MM_THREADS"] = str(pool)
    else:
        env.pop("SMG_MM_THREADS", None)
    cmd = [
        str(example),
        "--manifest",
        str(manifest),
        "--out",
        str(out),
        "--threads",
        str(threads),
        "--repeats",
        str(repeats),
    ]
    if cores:
        cmd = ["taskset", "-c", cores, *cmd]
    subprocess.run(cmd, check=True)


def run_reference(
    python: str, manifest: Path, out: Path, threads: int, repeats: int, cores: str | None, tmp: Path
) -> None:
    cmd = [
        python,
        "-I",
        str(HERE / "reference.py"),
        "--manifest",
        str(manifest),
        "--out",
        str(out),
        "--threads",
        str(threads),
        "--repeats",
        str(repeats),
        "--tmp",
        str(tmp),
    ]
    if cores:
        cmd = ["taskset", "-c", cores, *cmd]
    subprocess.run(cmd, check=True)


# --- the table --------------------------------------------------------------------------


def read_rows(path: Path) -> tuple[dict, dict[str, dict]]:
    meta, rows = {}, {}
    if not path.is_file():
        return meta, rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("kind") == "meta":
            meta = record
        else:
            rows[record["class"]] = record
    return meta, rows


def fmt_ms(row: dict | None, stage: str = "total") -> str:
    if not row or "error" in row:
        return "error" if row else "-"
    return f"{row['stage_ms'][stage]['median_ms']:.2f}"


def fmt_num(value, digits=1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def ratio(a: dict | None, b: dict | None) -> str:
    if not a or not b or "error" in a or "error" in b:
        return "-"
    x, y = a["stage_ms"]["total"]["median_ms"], b["stage_ms"]["total"]["median_ms"]
    return f"{x / y:.2f}x" if y else "-"


def parity(crate: dict | None, ref: dict | None) -> str:
    if not crate or not ref or "error" in crate or "error" in ref:
        return "-"
    marks = []
    c, r = crate["digests"], ref["digests"]
    if c.get("rgb_sha256") and r.get("rgb_sha256"):
        marks.append("rgb=" + ("same" if c["rgb_sha256"] == r["rgb_sha256"] else "DIFF"))
    if c.get("f32_sha256") and r.get("f32_sha256"):
        marks.append("f32=" + ("same" if c["f32_sha256"] == r["f32_sha256"] else "DIFF"))
    shape_same = list(crate.get("shape") or []) == list(ref.get("shape") or [])
    grid_same = list(crate.get("grid") or []) == list(ref.get("grid") or [])
    marks.append(
        "shape=" + ("same" if shape_same else f"DIFF {crate.get('shape')} vs {ref.get('shape')}")
    )
    if crate.get("grid") or ref.get("grid"):
        marks.append("grid=" + ("same" if grid_same else "DIFF"))
    return " ".join(marks)


def write_table(run: Path, manifest: dict, threads: int, arms: set[str], started: float) -> None:
    crate1_meta, crate1 = read_rows(run / "crate-t1.jsonl")
    _, crate1p1 = read_rows(run / "crate-t1-pool1.jsonl")
    _, crateN = read_rows(run / f"crate-t{threads}.jsonl")
    ref1_meta, ref1 = read_rows(run / "reference-t1.jsonl")
    _, refN = read_rows(run / f"reference-t{threads}.jsonl")
    lines = []
    host = {
        "arch": crate1_meta.get("arch") or ref1_meta.get("arch") or platform.machine(),
        "cores": crate1_meta.get("cores") or os.cpu_count(),
        "load_start": manifest.get("load_start"),
        "load_end": load_average(),
        "turbojpeg": crate1_meta.get("turbojpeg"),
        "libwebp": crate1_meta.get("libwebp"),
        "libwebp_version": crate1_meta.get("libwebp_version"),
        "pool_threads": crate1_meta.get("pool_threads"),
        "vector_kernels": crate1_meta.get("vector_kernels"),
        "wire": crate1_meta.get("wire"),
        "reference": {
            k: ref1_meta.get(k)
            for k in (
                "transformers",
                "torch",
                "pillow",
                "numpy",
                "opencv",
                "pillow_libjpeg_turbo",
                "python",
            )
        },
        "threads": threads,
        "wall_s": round(time.time() - started),
    }
    ref = host["reference"]
    turbo = "loaded" if host["turbojpeg"] else "ABSENT (fallback decoder)"
    webp = f"loaded {host['libwebp_version']}" if host["libwebp"] else "ABSENT (fallback decoder)"
    lines.append("# Multimodal baseline\n")
    lines.append(
        f"Box: {host['arch']}, {host['cores']} cores, 1-minute load {host['load_start']} at the start "
        f"and {host['load_end']} at the end. Crate paths: TurboJPEG {turbo}, libwebp {webp}, "
        f"vector kernels {host['vector_kernels']}, preprocessing pool {host['pool_threads']} threads, "
        f"wire {host['wire']}. Reference: transformers {ref['transformers']}, torch {ref['torch']}, "
        f"Pillow {ref['pillow']} (libjpeg-turbo {ref['pillow_libjpeg_turbo']}), numpy {ref['numpy']}, "
        f"OpenCV {ref['opencv']}, one compute thread per process. Load = {threads} threads (crate) "
        f"or {threads} processes (reference). Times are per request (a batch counts as one request), "
        f"medians over the iterations.\n"
    )
    lines.append(
        "| class | items | SMG single ms (decode / preprocess / serialize) | SMG single, pool=1 ms "
        "| SMG CPU ms/item | SMG load req/s | SMG load CPU ms/item "
        "| reference single ms (decode / preprocess) | reference CPU ms/item | reference load req/s "
        "| SMG / reference | parity |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cls in manifest["classes"]:
        name = cls["name"]
        c1, cp1, cn, r1, rn = (
            crate1.get(name),
            crate1p1.get(name),
            crateN.get(name),
            ref1.get(name),
            refN.get(name),
        )

        def stages(row, keys):
            if not row or "error" in row:
                return "error" if row else "-"
            return (
                f"{row['stage_ms']['total']['median_ms']:.2f} ("
                + " / ".join(f"{row['stage_ms'][k]['median_ms']:.2f}" for k in keys)
                + ")"
            )

        lines.append(
            f"| {name} | {cls.get('files') and len(cls['files'])} "
            f"| {stages(c1, ['decode', 'preprocess', 'serialize'])} "
            f"| {fmt_ms(cp1)} "
            f"| {fmt_num(c1 and c1.get('cpu_ms_per_item'), 2) if c1 and 'error' not in c1 else '-'} "
            f"| {fmt_num(cn and cn.get('requests_per_s'), 1) if cn and 'error' not in cn else '-'} "
            f"| {fmt_num(cn and cn.get('cpu_ms_per_item'), 2) if cn and 'error' not in cn else '-'} "
            f"| {stages(r1, ['decode', 'preprocess'])} "
            f"| {fmt_num(r1 and r1.get('cpu_ms_per_item'), 2) if r1 and 'error' not in r1 else '-'} "
            f"| {fmt_num(rn and rn.get('requests_per_s'), 1) if rn and 'error' not in rn else '-'} "
            f"| {ratio(c1, r1)} | {parity(c1, r1)} |"
        )
    errors = [
        (n, r["error"])
        for rows in (crate1, crate1p1, crateN, ref1, refN)
        for n, r in rows.items()
        if "error" in r
    ]
    if errors:
        lines.append("\nErrors:\n")
        for name, error in errors:
            lines.append(f"- {name}: {error}")
    lines.append(
        "\nShapes and bytes (SMG single): class | shape | grid | encoded B | decoded B | wire B | one-pass = two-pass\n"
    )
    for cls in manifest["classes"]:
        row = crate1.get(cls["name"])
        if row and "error" not in row:
            lines.append(
                f"- {cls['name']} | {row['shape']} | {row['grid'][:6]}{'...' if len(row['grid']) > 6 else ''} "
                f"| {row['bytes']['encoded']} | {row['bytes']['decoded']} | {row['bytes']['wire']} "
                f"| {row['one_pass_equals_two_pass']}"
            )
    (run / "table.md").write_text("\n".join(lines) + "\n")
    (run / "table.json").write_text(
        json.dumps(
            {
                "host": host,
                "arms": sorted(arms),
                "crate_single": crate1,
                "crate_single_pool1": crate1p1,
                "crate_load": crateN,
                "reference_single": ref1,
                "reference_load": refN,
            },
            indent=1,
        )
    )
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--classes", default=str(HERE / "classes.json"))
    parser.add_argument(
        "--corpus",
        required=True,
        help="the fixture repository's corpus directory (images/, videos/, audios/)",
    )
    parser.add_argument(
        "--config-dir",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="a checkpoint's config directory per family of classes.json",
    )
    parser.add_argument("--out", required=True, help="the run directory")
    parser.add_argument("--arms", default="crate,reference")
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="threads (crate) and processes (reference) of the load rows",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--only", default=None)
    parser.add_argument(
        "--example", default=None, help="a built mm_baseline example binary (else cargo builds it)"
    )
    parser.add_argument("--cargo-target", default=None)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="the interpreter with requirements.txt installed (the reference arm)",
    )
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    parser.add_argument("--cores", default=None, help="a cpu list every arm is pinned to (taskset)")
    parser.add_argument(
        "--no-pool1",
        action="store_true",
        help="skip the single-thread run with a one-thread preprocessing pool",
    )
    args = parser.parse_args()

    run = Path(args.out)
    run.mkdir(parents=True, exist_ok=True)
    classes = json.loads(Path(args.classes).read_text())
    config_dirs = {k: Path(v) for k, v in (item.split("=", 1) for item in args.config_dir)}
    only = set(args.only.split(",")) if args.only else None
    arms = set(args.arms.split(","))
    started = time.time()
    manifest = resolve(classes, Path(args.corpus), config_dirs, run, args.ffmpeg, only)
    manifest["load_start"] = load_average()
    manifest_path = run / "resolved.json"
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(
        f"mm_baseline: {len(manifest['classes'])} classes resolved into {manifest_path}",
        file=sys.stderr,
    )

    if "crate" in arms:
        example = Path(args.example) if args.example else build_example(args.cargo_target)
        extra = {"PATH": os.environ.get("PATH", "")}
        run_crate(
            example, manifest_path, run / "crate-t1.jsonl", 1, args.repeats, None, args.cores, extra
        )
        if not args.no_pool1:
            run_crate(
                example,
                manifest_path,
                run / "crate-t1-pool1.jsonl",
                1,
                args.repeats,
                1,
                args.cores,
                extra,
            )
        run_crate(
            example,
            manifest_path,
            run / f"crate-t{args.threads}.jsonl",
            args.threads,
            args.repeats,
            None,
            args.cores,
            extra,
        )
    if "reference" in arms:
        tmp = run / "tmp"
        tmp.mkdir(exist_ok=True)
        run_reference(
            args.python, manifest_path, run / "reference-t1.jsonl", 1, args.repeats, args.cores, tmp
        )
        run_reference(
            args.python,
            manifest_path,
            run / f"reference-t{args.threads}.jsonl",
            args.threads,
            args.repeats,
            args.cores,
            tmp,
        )
    write_table(run, manifest, args.threads, arms, started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
