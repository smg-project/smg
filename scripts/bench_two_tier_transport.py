#!/usr/bin/env python3
"""Measure one Router/Worker engine-transport benchmark arm.

The load generator runs on the host while service CPU is read from the Docker
cgroup, so client-side JSON/SSE work is not charged to either transport.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


def _container_pid(container: str) -> int:
    output = subprocess.check_output(
        ["docker", "inspect", "-f", "{{.State.Pid}}", container], text=True
    )
    pid = int(output.strip())
    if pid <= 0:
        raise RuntimeError(f"container {container!r} is not running")
    return pid


def _cgroup_dir(pid: int) -> Path:
    for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
        hierarchy, _controllers, relative = line.split(":", 2)
        if hierarchy == "0":
            return Path("/sys/fs/cgroup") / relative.lstrip("/")
    raise RuntimeError(f"cannot find cgroup v2 path for pid {pid}")


def _cpu_usage_usec(cgroup: Path) -> int:
    fields = dict(line.split(maxsplit=1) for line in (cgroup / "cpu.stat").read_text().splitlines())
    return int(fields["usage_usec"])


def _process_cpu(cgroup: Path) -> dict[int, tuple[float, str]]:
    ticks = int(subprocess.check_output(["getconf", "CLK_TCK"], text=True).strip())
    result: dict[int, tuple[float, str]] = {}
    for raw_pid in (cgroup / "cgroup.procs").read_text().split():
        pid = int(raw_pid)
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            close = stat.rfind(")")
            rest = stat[close + 2 :].split()
            cpu_s = (int(rest[11]) + int(rest[12])) / ticks
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            if not cmdline:
                cmdline = stat[stat.find("(") + 1 : close]
            result[pid] = (cpu_s, cmdline.strip())
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    return result


def _classify(command: str) -> str:
    if "smg.worker_sidecar" in command:
        return "worker_sidecar"
    if "smg.cli serve" in command:
        return "router_orchestrator"
    if "smg_grpc_servicer.tokenspeed" in command:
        return "engine_grpc_frontend"
    if "tokenspeed.cli" in command:
        return "engine_zmq_frontend"
    if "tokenspeed::scheduler" in command:
        return "engine_scheduler"
    return "other"


def _process_deltas(
    before: dict[int, tuple[float, str]], after: dict[int, tuple[float, str]]
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for pid, (end_cpu, command) in after.items():
        if pid not in before:
            continue
        delta = max(0.0, end_cpu - before[pid][0])
        category = _classify(command)
        totals[category] = totals.get(category, 0.0) + delta
    return {key: round(value, 3) for key, value in sorted(totals.items())}


def _gpu_sample(gpu: int) -> tuple[float, float]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    util, power = output.strip().split(",", 1)
    return float(util), float(power)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--sim-load", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--rps", type=float, required=True)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--body-words", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    container_pid = _container_pid(args.container)
    cgroup = _cgroup_dir(container_pid)
    cpu_before = _cpu_usage_usec(cgroup)
    process_before = _process_cpu(cgroup)

    stop_sampling = threading.Event()
    gpu_samples: list[tuple[float, float]] = []
    gpu_errors: list[str] = []

    def sample_gpu() -> None:
        while not stop_sampling.wait(0.5):
            try:
                gpu_samples.append(_gpu_sample(args.gpu))
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                # A missing nvidia-smi, a bad --gpu index or a permission error
                # would otherwise just omit the GPU fields, and a summary with
                # no GPU numbers reads the same as one that was never asked for
                # them. Record the first failure so the output says so.
                if not gpu_errors:
                    gpu_errors.append(f"{type(error).__name__}: {error}")

    sampler = threading.Thread(target=sample_gpu, daemon=True)
    sampler.start()
    command = [
        sys.executable,
        args.sim_load,
        "--url",
        args.url,
        "--endpoint",
        "completions",
        "--model",
        args.model,
        "--rps",
        str(args.rps),
        "--duration",
        str(args.duration),
        "--concurrency",
        str(args.concurrency),
        "--output-tokens",
        str(args.output_tokens),
        "--shared-prefix-frac",
        "0",
        "--body-words-min",
        str(args.body_words),
        "--body-words-max",
        str(args.body_words),
        "--seed",
        str(args.seed),
        "--label",
        args.label,
    ]
    started = time.monotonic()
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    wall_s = time.monotonic() - started
    stop_sampling.set()
    sampler.join(timeout=2)

    cpu_after = _cpu_usage_usec(cgroup)
    process_after = _process_cpu(cgroup)
    summary_line = next(
        line.removeprefix("SUMMARY_JSON ")
        for line in completed.stdout.splitlines()
        if line.startswith("SUMMARY_JSON ")
    )
    summary = json.loads(summary_line)
    cpu_s = (cpu_after - cpu_before) / 1_000_000
    summary.update(
        {
            "wall_s": round(wall_s, 3),
            "service_cpu_s": round(cpu_s, 3),
            "service_cpu_cores_avg": round(cpu_s / wall_s, 3),
            "process_cpu_s": _process_deltas(process_before, process_after),
        }
    )
    if gpu_errors:
        summary["gpu_sampling_error"] = gpu_errors[0]
        print(
            f"warning: GPU sampling for --gpu {args.gpu} failed: {gpu_errors[0]}",
            file=sys.stderr,
        )
    if gpu_samples:
        utils = [sample[0] for sample in gpu_samples]
        powers = [sample[1] for sample in gpu_samples]
        summary.update(
            {
                "gpu_samples": len(gpu_samples),
                "gpu_util_avg": round(statistics.fmean(utils), 1),
                "gpu_util_p50": round(_percentile(utils, 0.5), 1),
                "gpu_util_p99": round(_percentile(utils, 0.99), 1),
                "gpu_power_w_avg": round(statistics.fmean(powers), 1),
            }
        )

    print(completed.stdout, end="")
    print("ARM_SUMMARY_JSON " + json.dumps(summary, sort_keys=True))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
