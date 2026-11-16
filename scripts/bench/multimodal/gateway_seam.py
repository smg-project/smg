#!/usr/bin/env python3
"""The gateway seam of the multimodal baseline, on the CPU mock bed.

The crate arm times the pipeline's stages in one process; this arm times the same
classes through the gateway: one release gateway in front of mock gRPC engines (the
bed of ``scripts/bench/grpc_stress/bed.sh``), the model served under the family's
checkpoint directory (``config.json``, the preprocessor configs and the tokenizer), the
images sent as data URLs in chat requests by the bed's load client (``stress_load
--mode image``). The rows read the gateway's own timing lines (``--log-mm-timing``:
fetch and decode, preprocess, token expansion, total per request), the client's
end-to-end latency, and the gateway process's CPU time per request.

Three modes, each on a fresh bed:

- ``classes``: every image class of the resolved manifest (video and audio stay with
  the crate arm; the load client sends chat requests with image parts), one at a time
  (an open-loop rate low enough to keep one request in flight) and under load (the
  rate of ``--load-rate`` requests per second).
- ``burst``: two hundred image requests offered within one second, then the drain:
  the client's p50 and p99 and its failures by status, the gateway's resident-size
  peak, the time until the last request answered.
- ``replay``: one conversation of twenty turns over twenty distinct crops of one page,
  turn k carrying the last ``--window`` of the k screenshots seen so far (the model's
  limit of image items per request bounds a turn, as an agent's context pruning would),
  sent one after another: the gateway's CPU per turn, its media stage times per turn
  and its resident size per turn; once with the router's pixel cache off (the default)
  and once with it on.

    python3 scripts/bench/multimodal/gateway_seam.py --manifest <run>/resolved.json \\
        --config-dir <qwen3_vl dir> --smg <gateway binary> --mock <mock-worker binary> \\
        --load <stress_load binary> --out <run dir> --mode classes --ports 29700
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BED_SH = ROOT / "scripts" / "bench" / "grpc_stress" / "bed.sh"
TIMING = re.compile(
    r"image_count=(?P<images>\d+).*?video_frame_count=(?P<frames>\d+) media_fetch_decode_ms=(?P<fetch>[0-9.]+) "
    r"config_lookup_ms=(?P<config>[0-9.]+) preprocess_ms=(?P<pre>[0-9.]+) token_expand_ms=(?P<expand>[0-9.]+) "
    r"total_ms=(?P<total>[0-9.]+)"
)


def proc_cpu_ms(pid: int) -> float:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    ticks = int(fields[11]) + int(fields[12])
    return ticks * 1000.0 / os.sysconf("SC_CLK_TCK")


def proc_mem_kb(pid: int) -> dict:
    out = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value = line.split(":", 1)
            out[key] = int(value.strip().split()[0])
    return out


def reset_peak(pid: int) -> None:
    """Reset the process's resident high-water mark to its current size (Linux 4.0 and
    later), so the next VmHWM read is this cell's peak and not the process's lifetime one."""
    try:
        Path(f"/proc/{pid}/clear_refs").write_text("5")
    except OSError as error:
        print(f"peak not reset ({error}): the lifetime high-water mark follows", file=sys.stderr)


class Bed:
    """The bed's processes, through bed.sh with this run's ports, cores and flags."""

    def __init__(self, args, run_dir: Path, name: str, gw_args: str = ""):
        self.args = args
        self.dir = run_dir / f"bed-{name}"
        base = args.ports
        self.gw = f"http://127.0.0.1:{base}"
        self.env = dict(
            os.environ,
            GS_DIR=str(self.dir),
            SMG_BIN=args.smg,
            MOCK_BIN=args.mock,
            GW_PORT=str(base),
            PROM_PORT=str(base + 1),
            ADMIN_PORT=str(base + 9),
            MOCK_BASE=str(base + 10),
            WORKERS=str(args.workers),
            MODEL=args.model,
            TOKENIZER=args.config_dir,
            GW_THREADS=str(args.gw_threads),
            GW_LOG_LEVEL="info",
            GW_ARGS=f"--log-mm-timing {gw_args}".strip(),
            ENGINE_ARGS=(
                "--engine realistic --timing linear --prefill-tps 2000000 --decode-base-ms 1 "
                "--decode-per-req-ms 0 --max-running 4096 --max-batched-tokens 1000000 "
                "--context-length 262144 --kv-tokens 2097152"
            ),
        )
        if args.gw_cores:
            self.env["GW_CORES"] = args.gw_cores
        if args.mock_cores:
            self.env["MOCK_CORES"] = args.mock_cores
        self.pid = None

    def __enter__(self):
        already_up = (self.dir / "bed" / "gateway.pid").exists()
        try:
            subprocess.run(["bash", str(BED_SH), "up"], env=self.env, check=True)
            bed = json.loads((self.dir / "bed" / "bed.json").read_text())
        except BaseException:
            # `up` dies after starting the gateway or the fleet on three of its failure
            # paths; left running, they would keep the ports from the next bed. A bed
            # that was up before this call is not ours to stop.
            if not already_up:
                self.__exit__()
            raise
        self.pid = bed["gateway_pid"]
        self.log = self.dir / "bed" / "gateway.log"
        return self

    def __exit__(self, *exc):
        subprocess.run(["bash", str(BED_SH), "down"], env=self.env, check=False)

    def timing_lines(self, since: int) -> tuple[list[dict], int]:
        text = self.log.read_text()
        rows = []
        for line in text.splitlines()[since:]:
            if "smg_mm_timing" not in line:
                continue
            match = TIMING.search(line)
            if match:
                rows.append({k: float(v) for k, v in match.groupdict().items()})
        return rows, len(text.splitlines())


def summarize(values: list[float]) -> dict:
    if not values:
        return {"median_ms": None, "p90_ms": None, "p99_ms": None}
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]  # noqa: E731
    return {"median_ms": pick(0.5), "p90_ms": pick(0.9), "p99_ms": pick(0.99)}


def stage_rows(timing: list[dict]) -> dict:
    return {
        "requests": len(timing),
        "fetch_decode": summarize([t["fetch"] for t in timing]),
        "preprocess": summarize([t["pre"] for t in timing]),
        "token_expand": summarize([t["expand"] for t in timing]),
        "total": summarize([t["total"] for t in timing]),
    }


def drain(bed: Bed, idle_ms: float = 20.0, cap_s: float = 300.0) -> dict:
    """Wait until the gateway has worked through what the previous cell left in its queues.

    An overloaded cell hands its backlog to the next one otherwise: the gateway keeps
    preprocessing the previous cell's frames while the next cell's requests arrive, and
    the next cell's CPU and latency columns read the drain, not the class. Idle means
    less than ``idle_ms`` of process CPU time over one second, twice in a row.
    """
    started = time.time()
    quiet = 0
    cpu0 = proc_cpu_ms(bed.pid)
    while time.time() - started < cap_s:
        time.sleep(1.0)
        cpu1 = proc_cpu_ms(bed.pid)
        quiet = quiet + 1 if cpu1 - cpu0 < idle_ms else 0
        cpu0 = cpu1
        if quiet >= 2:
            break
    return {"drain_s": round(time.time() - started, 1), "rss_kb": proc_mem_kb(bed.pid).get("VmRSS")}


def load_cell(
    args,
    bed: Bed,
    cell_dir: Path,
    files: list[str],
    rate: float,
    duration: int,
    images_per_request: int,
    warmup: int = 2,
) -> dict:
    cell_dir.mkdir(parents=True, exist_ok=True)
    drained = drain(bed)
    cmd = [
        args.load,
        "--gateway",
        bed.gw,
        "--model",
        args.model,
        "--mode",
        "image",
        "--rate",
        str(rate),
        "--duration",
        str(duration),
        "--warmup",
        str(warmup),
        "--drain",
        "120",
        "--max-tokens",
        "1",
        "--images-per-request",
        str(images_per_request),
        "--interim",
        "0",
        "--out",
        str(cell_dir / "summary.json"),
    ]
    for f in files:
        cmd += ["--image", f]
    if args.load_cores:
        cmd = ["taskset", "-c", args.load_cores, *cmd]
    _, line0 = bed.timing_lines(0)
    cpu0 = proc_cpu_ms(bed.pid)
    reset_peak(bed.pid)
    mem0 = proc_mem_kb(bed.pid)
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - started
    cpu1 = proc_cpu_ms(bed.pid)
    mem1 = proc_mem_kb(bed.pid)
    (cell_dir / "gen.err").write_text(proc.stderr)
    summary = (
        json.loads((cell_dir / "summary.json").read_text())
        if (cell_dir / "summary.json").exists()
        else {"error": proc.stderr[-2000:]}
    )
    timing, _ = bed.timing_lines(line0)
    # per request the gateway finished preprocessing inside the CPU window: an overload
    # cell's abandoned requests (the client's `unfinished`) cost CPU too, so the column
    # reads high there and the client keys say how many never finished
    processed = len(timing)
    return {
        "rate": rate,
        "duration": duration,
        "images_per_request": images_per_request,
        "client": {
            k: summary.get(k)
            for k in (
                "sent",
                "sampled",
                "done",
                "ok",
                "failed",
                "unfinished",
                "skipped",
                "achieved_rps",
                "e2e_ms",
                "ttfb_ms",
                "errors",
            )
        },
        "gateway_cpu_ms_per_request": (cpu1 - cpu0) / processed if processed else None,
        "gateway_cpu_ms_total": cpu1 - cpu0,
        "gateway_rss_kb": {
            "before": mem0.get("VmRSS"),
            "after": mem1.get("VmRSS"),
            "peak": mem1.get("VmHWM"),
        },
        "stages": stage_rows(timing),
        "wall_s": wall,
        "drained_before": drained,
        "rc": proc.returncode,
    }


def mode_classes(args, manifest: dict, run_dir: Path) -> dict:
    out = {}
    with Bed(args, run_dir, "classes") as bed:
        for cls in manifest["classes"]:
            if cls["kind"] not in ("image", "batch"):
                continue
            if args.only and cls["name"] not in args.only:
                continue
            print(f"gateway: {cls['name']}", file=sys.stderr)
            files = cls["files"]
            per_request = len(files) if cls["kind"] == "batch" else 1
            single = load_cell(
                args,
                bed,
                run_dir / "cells" / f"{cls['name']}-single",
                files,
                args.single_rate,
                args.single_seconds,
                per_request,
            )
            load = load_cell(
                args,
                bed,
                run_dir / "cells" / f"{cls['name']}-load",
                files,
                args.load_rate,
                args.load_seconds,
                per_request,
            )
            out[cls["name"]] = {"single": single, "load": load}
            (run_dir / "gateway-classes.json").write_text(json.dumps(out, indent=1))
    return out


def mode_burst(args, manifest: dict, run_dir: Path) -> dict:
    cls = next(c for c in manifest["classes"] if c["name"] == args.burst_class)
    with Bed(args, run_dir, "burst") as bed:
        idle = load_cell(
            args, bed, run_dir / "cells" / "burst-idle", cls["files"], 1.0, 3, 1, warmup=1
        )
        burst = load_cell(
            args,
            bed,
            run_dir / "cells" / "burst-200",
            cls["files"],
            float(args.burst_requests),
            1,
            1,
            warmup=0,
        )
        after = load_cell(
            args, bed, run_dir / "cells" / "burst-after", cls["files"], 1.0, 3, 1, warmup=1
        )
    result = {
        "class": cls["name"],
        "requests": args.burst_requests,
        "idle_before": idle,
        "burst": burst,
        "after": after,
    }
    (run_dir / "gateway-burst.json").write_text(json.dumps(result, indent=1))
    return result


def post_chat(url: str, model: str, parts: list[dict], timeout: float = 600.0) -> tuple[int, float]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": parts}],
            "max_tokens": 1,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=body, headers={"content-type": "application/json"}
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            status = resp.status
    except urllib.error.HTTPError as error:
        status = error.code
        print(f"replay: {status} {error.read()[:300]!r}", file=sys.stderr)
    return status, (time.perf_counter() - started) * 1000.0


def scrape(url: str, names: tuple[str, ...]) -> dict:
    """The named counters of the gateway's metrics endpoint, summed over their labels."""
    out = dict.fromkeys(names, 0.0)
    try:
        with urllib.request.urlopen(f"{url}/metrics", timeout=10) as resp:
            text = resp.read().decode()
    except (urllib.error.URLError, OSError) as error:
        # no counter at all, never a zero that reads as "the cache never hit"
        print(f"scrape failed ({error}): the counters are unknown", file=sys.stderr)
        return dict.fromkeys(names)
    for line in text.splitlines():
        for name in names:
            if line.startswith(name) and line[len(name)] in "{ ":
                out[name] += float(line.rsplit(" ", 1)[1])
    return out


def screenshots(source: Path, out_dir: Path, count: int) -> list[Path]:
    """`count` distinct crops of one page: turn k's k-th screenshot."""
    from PIL import Image  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with Image.open(source) as image:
        image.load()
        width, height = image.size
        for k in range(count):
            path = out_dir / f"screenshot-{k + 1:02d}.png"
            if not path.exists():
                crop = image.crop((k, k, width - count + k, height - count + k))
                crop.save(path, "PNG", compress_level=6)
            paths.append(path)
    return paths


def data_url(path: Path) -> str:
    mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
    }.get(path.suffix.lstrip(".").lower(), "image/png")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def mode_replay(args, manifest: dict, run_dir: Path) -> dict:
    source = next(
        Path(c["files"][0]) for c in manifest["classes"] if c["name"] == args.replay_class
    )
    shots = screenshots(source, run_dir / "screenshots", args.turns)
    urls = [data_url(p) for p in shots]
    results = {}
    for label, gw_args in (
        ("cache-off", ""),
        ("cache-on", f"--mm-pixel-cache-mb {args.pixel_cache_mb}"),
    ):
        turns = []
        with Bed(args, run_dir, f"replay-{label}", gw_args) as bed:
            # one warm turn so the first measured turn does not pay the first-request costs;
            # the uncropped page is an image the session never sends, so the cache-on
            # run's first turn is a miss like the cache-off run's
            post_chat(
                bed.gw,
                args.model,
                [
                    {"type": "image_url", "image_url": {"url": data_url(source)}},
                    {"type": "text", "text": "warm"},
                ],
            )
            _, line = bed.timing_lines(0)
            for k in range(1, args.turns + 1):
                # the last `window` screenshots: the model's limit of image items per
                # request bounds what one turn carries, as an agent's context pruning would
                window = urls[max(0, k - args.window) : k]
                parts = [{"type": "image_url", "image_url": {"url": u}} for u in window]
                parts.append(
                    {"type": "text", "text": f"turn {k}: what changed since the last screenshot?"}
                )
                reset_peak(bed.pid)
                cpu0 = proc_cpu_ms(bed.pid)
                status, latency = post_chat(bed.gw, args.model, parts)
                cpu1 = proc_cpu_ms(bed.pid)
                timing, line = bed.timing_lines(line)
                mem = proc_mem_kb(bed.pid)
                turns.append(
                    {
                        "turn": k,
                        "images": len(window),
                        "status": status,
                        "latency_ms": latency,
                        "gateway_cpu_ms": cpu1 - cpu0,
                        "stages": timing[-1] if timing else None,
                        "rss_kb": mem.get("VmRSS"),
                        "hwm_kb": mem.get("VmHWM"),
                    }
                )
                print(
                    f"replay {label} turn {k}: {status} {latency:.0f} ms "
                    f"cpu {cpu1 - cpu0:.0f} ms rss {mem.get('VmRSS')} kB",
                    file=sys.stderr,
                )
            counters = scrape(
                f"http://127.0.0.1:{args.ports + 1}",
                ("smg_mm_pixel_cache_hits_total", "smg_mm_pixel_cache_misses_total"),
            )
        results[label] = {"turns": turns, "pixel_cache": counters}
        (run_dir / "gateway-replay.json").write_text(json.dumps(results, indent=1))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--config-dir", required=True, help="the served model's config and tokenizer directory"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="the served model id (a vision family the registry knows)",
    )
    parser.add_argument("--smg", required=True)
    parser.add_argument("--mock", required=True)
    parser.add_argument("--load", required=True, help="the stress_load binary")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["classes", "burst", "replay"], default="classes")
    parser.add_argument("--only", default=None)
    parser.add_argument(
        "--ports", type=int, default=29700, help="the first of 32 loopback ports the bed binds"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gw-threads", type=int, default=8)
    parser.add_argument("--gw-cores", default=None)
    parser.add_argument("--mock-cores", default=None)
    parser.add_argument("--load-cores", default=None)
    parser.add_argument(
        "--single-rate",
        type=float,
        default=4.0,
        help="requests per second of the one-at-a-time rows",
    )
    parser.add_argument("--single-seconds", type=int, default=10)
    parser.add_argument("--load-rate", type=float, default=40.0)
    parser.add_argument("--load-seconds", type=int, default=10)
    parser.add_argument("--burst-requests", type=int, default=200)
    parser.add_argument("--burst-class", default="png-screenshot-1080p")
    parser.add_argument("--replay-class", default="png-screenshot-1080p")
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--pixel-cache-mb", type=int, default=1024)
    args = parser.parse_args()
    args.only = set(args.only.split(",")) if args.only else None
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    if args.mode == "classes":
        result = mode_classes(args, manifest, run_dir)
    elif args.mode == "burst":
        result = mode_burst(args, manifest, run_dir)
    else:
        result = mode_replay(args, manifest, run_dir)
    print(json.dumps(result, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
