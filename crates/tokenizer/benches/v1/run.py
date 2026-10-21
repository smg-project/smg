"""Prepare pinned models, benchmark both builds, and summarize their JSONL output."""

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
MODELS = json.loads((HERE / "models.json").read_text())


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
            temporary.replace(path)


def run(root, output):
    # Build everything before measuring: compilation must not compete for CPUs.
    for name, features in [("current", []), ("v1", ["--features", "tokenizers-v1"])]:
        result = subprocess.run(
            [
                "cargo",
                "bench",
                "--locked",
                "-p",
                "llm-tokenizer",
                "--bench",
                "v1_compare",
                "--no-run",
                "--message-format=json",
                *features,
            ],
            cwd=REPO,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        artifacts = [json.loads(line) for line in result.stdout.splitlines()]
        binary = next(
            a["executable"]
            for a in artifacts
            if a.get("reason") == "compiler-artifact"
            and a["target"]["name"] == "v1_compare"
            and a.get("executable")
        )
        shutil.copy2(binary, output / name)
    for name, threads in [("current", 1), ("v1", 1), ("v1", 4)]:
        env = {**os.environ, "SMG_BENCH_MODEL_ROOT": str(root), "SMG_BENCH_THREADS": str(threads)}
        with (output / f"{name}-{threads}.jsonl").open("w") as out:
            subprocess.run([str(output / name)], env=env, check=True, stdout=out)


def write_csv(path, rows):
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(output):
    groups = {}
    auxiliary = []
    for path in sorted(output.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["kind"] != "timing":
                auxiliary.append(row)
                continue
            key = tuple(
                row[k]
                for k in [
                    "model",
                    "tokens",
                    "scenario",
                    "cache",
                    "concurrency",
                    "v1",
                    "internal_threads",
                ]
            )
            groups.setdefault(key, []).append(row)
    summary = []
    for key, rows in groups.items():
        assert len(rows) == 3, (key, len(rows))
        row = dict(
            zip(
                ["model", "tokens", "scenario", "cache", "concurrency", "v1", "internal_threads"],
                key,
            )
        )
        for p in ["p50_ms", "p95_ms", "p99_ms"]:
            row[p] = statistics.median(r[p] for r in rows)
        row["requests"] = sum(r["requests"] for r in rows)
        row["rps"] = row["requests"] / sum(r["wall_s"] for r in rows)
        row["p50_round_min_ms"] = min(r["p50_ms"] for r in rows)
        row["p50_round_max_ms"] = max(r["p50_ms"] for r in rows)
        row["cold_median_ms"] = statistics.median(rows[0]["cold_ms"])
        for i, level in [(0, "l0"), (2, "l1")]:
            hits = sum(r["cache_delta"][i] for r in rows)
            misses = sum(r["cache_delta"][i + 1] for r in rows)
            row[level + "_hit_pct"] = 100 * hits / (hits + misses) if hits + misses else None
        row["l1_budgeted_bytes"] = rows[-1]["l1_budgeted_bytes"]
        for sample in rows:
            h0, m0, h1, m1 = sample["cache_delta"]
            n = sample["requests"]
            if row["cache"] in ["l0", "both"]:
                assert h0 + m0 == n
                assert h0 == (n if row["scenario"] == "repeat" else 0)
            if row["cache"] in ["l1", "both"]:
                assert h1 + m1 == (n - h0)
                if row["scenario"] == "unique":
                    assert h1 == 0
                if row["scenario"] in ["prefix90", "prefix99"]:
                    assert h1 == n
        summary.append(row)
    write_csv(output / "summary.csv", summary)
    (output / "auxiliary.json").write_text(json.dumps(auxiliary, indent=2) + "\n")
    lookup = {
        tuple(
            r[k]
            for k in [
                "model",
                "tokens",
                "scenario",
                "cache",
                "concurrency",
                "v1",
                "internal_threads",
            ]
        ): r
        for r in summary
    }
    comparisons = []
    for row in summary:
        base = lookup[
            row["model"], row["tokens"], row["scenario"], row["cache"], row["concurrency"], False, 1
        ]
        own = lookup[
            row["model"],
            row["tokens"],
            row["scenario"],
            "off",
            row["concurrency"],
            row["v1"],
            row["internal_threads"],
        ]
        comparisons.append(
            {
                **row,
                "saved_ms": base["p50_ms"] - row["p50_ms"],
                "latency_reduction_pct": 100 * (1 - row["p50_ms"] / base["p50_ms"]),
                "speedup": base["p50_ms"] / row["p50_ms"],
                "cache_reduction_pct": 100 * (1 - row["p50_ms"] / own["p50_ms"]),
            }
        )
    write_csv(output / "comparison.csv", comparisons)
    lines = [
        "| Model | Tokens | Current ms | v1 / 1 thread ms | Saved ms | Less time | Speedup | v1 / 4 threads ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for tokens in [128, 2048, 50000, 80000]:
            scenario = "flat" if tokens < 50000 else "unique"
            a = lookup[model["name"], tokens, scenario, "off", 1, False, 1]
            b = lookup[model["name"], tokens, scenario, "off", 1, True, 1]
            c = lookup[model["name"], tokens, scenario, "off", 1, True, 4]
            old, new = a["p50_ms"], b["p50_ms"]
            lines.append(
                f"| {model['name']} | {tokens:,} | {old:.3f} | {new:.3f} | {old - new:.3f} | "
                f"{100 * (1 - new / old):.1f}% | {old / new:.2f}x | {c['p50_ms']:.3f} |"
            )
    (output / "table.md").write_text("\n".join(lines) + "\n")
    print(
        f"Validated {len(summary)} configurations / {sum(r['requests'] for r in summary):,} timed calls"
    )
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "summarize"])
    parser.add_argument("--models", type=Path, default=Path("/tmp/smg-tokenizer-models"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/smg-tokenizer-results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        prepare(args.models.resolve())
    elif args.action == "run":
        run(args.models.resolve(), args.output.resolve())
    else:
        summarize(args.output.resolve())


if __name__ == "__main__":
    main()
