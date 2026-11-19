#!/usr/bin/env python3
"""Qualify both regular PR benchmark protocols independently of advisory thresholds."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Reuse the nightly report's reconciliation of raw request counts/errors.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "e2e_test" / "benchmarks"))
from nightly_summarize import _error_metrics

METRICS = ("ttft", "e2e_latency", "input_throughput", "output_throughput")
PROTOCOLS = ("http", "grpc")


def qualify_file(path: Path) -> dict:
    result = {"path": str(path), "qualified": False}
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or not isinstance(data.get("aggregated_metrics"), dict):
            raise ValueError("missing aggregated_metrics object")
        agg = data["aggregated_metrics"]
        stats = agg.get("stats")
        if not isinstance(stats, dict):
            raise ValueError("missing stats object")
        metrics = {}
        for name in METRICS:
            item = stats.get(name)
            value = item.get("mean") if isinstance(item, dict) else None
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name}.mean must be a finite positive number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name}.mean must be a finite positive number")
            metrics[name] = value
        # Legacy rate-only summaries cannot establish successful request counts.
        rate, note = _error_metrics(data)
        if rate is None or rate != 0 or note:
            raise ValueError(note or "nonzero or unknown request failure rate")
        completed = agg.get("num_completed_requests")
        if completed is None:
            if "num_requests" in agg and "num_error_requests" in agg:
                completed = agg["num_requests"] - agg["num_error_requests"]
            elif isinstance(data.get("individual_request_metrics"), list):
                completed = sum(
                    row.get("error_code") is None and not row.get("error_message")
                    for row in data["individual_request_metrics"]
                )
        if type(completed) is not int or completed <= 0:
            raise ValueError("no positive successful request count")
        # A completed count and a rate without any attempted/error evidence is ambiguous.
        if (
            not any(
                key in agg
                for key in ("num_requests", "num_error_requests", "error_codes_frequency")
            )
            and "individual_request_metrics" not in data
        ):
            raise ValueError("missing attempted/error count evidence")
        result.update(
            qualified=True, completed_requests=completed, error_rate=rate, metrics=metrics
        )
    except (OSError, ValueError, TypeError, AttributeError, OverflowError) as error:
        result["error"] = str(error)
    return result


def qualify(base_dir: Path) -> dict:
    protocols = {}
    for protocol in PROTOCOLS:
        folder_name = f"benchmark_round_robin_regular_{protocol}"
        folders = (
            [base_dir]
            if base_dir.name == folder_name
            else sorted(path for path in base_dir.rglob(folder_name) if path.is_dir())
        )
        files = sorted(
            {
                path
                for folder in folders
                for path in folder.rglob("*.json")
                if "experiment_metadata" not in path.name and "gpu_utilization" not in path.name
            }
        )
        results = [qualify_file(path) for path in files]
        protocols[protocol] = {
            "qualified": bool(results) and all(item["qualified"] for item in results),
            "files": results,
        }
        if not results:
            protocols[protocol]["error"] = "no raw benchmark result JSON"
    return {
        "qualified": all(item["qualified"] for item in protocols.values()),
        "protocols": protocols,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_dir", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = qualify(args.base_dir)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
