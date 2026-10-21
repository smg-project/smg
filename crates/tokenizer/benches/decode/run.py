#!/usr/bin/env python3
"""Build, validate and compare default/opt-in decode on pinned tokenizer files."""

import argparse
import csv
import hashlib
import json
import platform
import shutil
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
MODELS = json.loads((HERE / "models.json").read_text())
KEYS = ("model", "scenario", "tokens", "skip_special_tokens", "streaming", "callers")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root):
    for model in MODELS:
        path = root / model["name"] / "tokenizer.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            url = (
                f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/tokenizer.json"
            )
            temporary = path.with_suffix(".download")
            urllib.request.urlretrieve(url, temporary)
            if sha256(temporary) != model["sha256"]:
                raise ValueError(f"Downloaded fixture hash mismatch: {path}")
            temporary.replace(path)
        if sha256(path) != model["sha256"]:
            raise ValueError(f"Fixture hash mismatch: {path}")


def run_timed(binary, root, duration, destination):
    def compilers():
        processes = subprocess.check_output(["ps", "-axo", "comm="], text=True)
        return [
            p
            for p in processes.splitlines()
            if Path(p.strip()).name in {"rustc", "clippy-driver", "clang", "cc1"}
        ]

    if active := compilers():
        raise RuntimeError(f"Wait for other compilers before timing: {active}")
    with destination.open("w") as out:
        process = subprocess.Popen([str(binary), str(root), str(duration)], stdout=out)
        try:
            while process.poll() is None:
                if active := compilers():
                    process.terminate()
                    process.wait()
                    destination.rename(destination.with_suffix(".contaminated.jsonl"))
                    raise RuntimeError(f"Discarded timing with concurrent compiler: {active}")
                time.sleep(1)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, process.args)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()


def run(root, output, duration):
    prepare(root)
    metadata = {
        "platform": platform.platform(),
        "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
        "base_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "source_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=REPO)
        ).hexdigest(),
        "profile": "workspace bench: opt-level=3, thin LTO, codegen-units=1",
        "allocator": "system",
        "duration_ms": duration,
        "models": MODELS,
        "binaries": {},
        "harness_sha256": sha256(HERE.parent / "decode_compare.rs"),
        "runner_sha256": sha256(Path(__file__)),
    }
    for name, features in [("default", []), ("opt-in", ["--features", "tokenizers-v1"])]:
        with (output / f"build-{name}.log").open("w") as log:
            result = subprocess.run(
                [
                    "cargo",
                    "bench",
                    "--locked",
                    "--offline",
                    "-p",
                    "llm-tokenizer",
                    "--bench",
                    "decode_compare",
                    "--no-run",
                    "--message-format=json",
                    *features,
                ],
                cwd=REPO,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=log,
            )
        artifacts = [json.loads(line) for line in result.stdout.splitlines()]
        binary = next(
            a["executable"]
            for a in artifacts
            if a.get("reason") == "compiler-artifact"
            and a["target"]["name"] == "decode_compare"
            and a.get("executable")
        )
        shutil.copy2(binary, output / name)
        metadata["binaries"][name] = sha256(output / name)
    metadata["lock_sha256"] = sha256(REPO / "Cargo.lock")
    shutil.copy2(REPO / "Cargo.lock", output / "Cargo.lock")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Complete builds and correctness before starting any timing.
    for name in ["default", "opt-in"]:
        with (output / f"check-{name}.jsonl").open("w") as out:
            subprocess.run(
                [str(output / name), str(root), str(duration), "check"], check=True, stdout=out
            )
    for round_index in range(3):
        order = ["default", "opt-in"] if round_index % 2 == 0 else ["opt-in", "default"]
        for name in order:
            run_timed(output / name, root, duration, output / f"{name}-{round_index}.jsonl")
            print(f"Round {round_index + 1}: {name} completed", flush=True)


def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(output):
    groups, raw, loads = {}, [], []
    expected = None
    for round_index in range(3):
        for name in ["default", "opt-in"]:
            seen = set()
            for line in (output / f"{name}-{round_index}.jsonl").read_text().splitlines():
                row = json.loads(line)
                if row["kind"] == "model":
                    assert row["byte_table_decode"] == (name == "opt-in"), row
                    loads.append({"round": round_index, "build": name, **row})
                    continue
                assert row["kind"] == "timing" and row["samples"] > 0, row
                assert row["feature"] == (name == "opt-in"), row
                key = tuple(row[k] for k in KEYS)
                assert key not in seen, key
                seen.add(key)
                groups.setdefault(key, {}).setdefault(name, []).append(row)
                raw.append({"round": round_index, "build": name, **row})
            if expected is None:
                expected = seen
            assert seen == expected, "Missing or extra benchmark cases"
    assert len(expected) == 340, f"Unexpected matrix size: {len(expected)}"
    results = []
    for key, sides in groups.items():
        a, b = sides["default"], sides["opt-in"]
        assert len(a) == len(b) == 3
        assert len({r["digest"] for r in a + b}) == 1, key
        assert len({r["output_bytes"] for r in a + b}) == 1, key
        ratios = [x["mean_us"] / y["mean_us"] for x, y in zip(a, b)]
        old, new = (statistics.median(r["mean_us"] for r in side) for side in [a, b])
        results.append(
            {
                **dict(zip(KEYS, key)),
                "default_us": old,
                "opt_in_us": new,
                "speedup": old / new,
                "paired_speedup_min": min(ratios),
                "paired_speedup_max": max(ratios),
                "default_rps": statistics.median(r["requests_per_second"] for r in a),
                "opt_in_rps": statistics.median(r["requests_per_second"] for r in b),
            }
        )
    write_csv(output / "rounds.csv", raw)
    write_csv(output / "comparison.csv", results)
    write_csv(output / "loads.csv", loads)
    for row in results:
        if (
            row["scenario"] == "mixed"
            and row["tokens"] in [2048, 80000]
            and row["skip_special_tokens"]
            and row["callers"] == 1
            and not row["streaming"]
        ):
            print(
                f"{row['model']:16} {row['tokens']:5}: "
                f"{row['default_us']:.2f} -> {row['opt_in_us']:.2f} us ({row['speedup']:.2f}x)"
            )
    print(
        f"Validated {len(results)} configurations across six processes; full/stream digests agree."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "summarize"])
    parser.add_argument("--models", type=Path, default=Path("/tmp/smg-tokenizer-models"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/smg-decode-results"))
    parser.add_argument("--duration-ms", type=int, default=200)
    args = parser.parse_args()
    if args.duration_ms <= 0:
        parser.error("--duration-ms must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        prepare(args.models.resolve())
    elif args.action == "run":
        run(args.models.resolve(), args.output.resolve(), args.duration_ms)
    else:
        summarize(args.output.resolve())


if __name__ == "__main__":
    main()
