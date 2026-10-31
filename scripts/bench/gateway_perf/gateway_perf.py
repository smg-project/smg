#!/usr/bin/env python3
"""The gateway's cost per request, base against head, on the CPU mock bed.

One release gateway in front of mock gRPC engines that answer from a script (the
answer is served by the wire, so no engine time enters the numbers), an open-loop
client at a fixed rate for a fixed number of requests, and the gateway process's own
counters over the window, read from outside the process:

  cpu    instructions per request (user, and user+kernel) and cycles from `perf stat`
         on the gateway pid when hardware counters are available; CPU time per request
         (utime+stime from /proc) always
  alloc  allocation calls and KiB requested per request, exact, from jemalloc's exit
         statistics (`_RJEM_MALLOC_CONF=stats_print:true`; the gateway prints them at a
         clean shutdown; no special build)
  mem    RSS after each of several windows on one gateway, the growth per 10,000
         requests, the peak (VmHWM) over a burst window, the allocator's live bytes
  live   an open-loop ramp of offered rates, a fresh gateway per step, with the
         capacity rule: the highest offered step at which the SLO holds, and the strict
         reading (the last holding step before the first failing one) beside it

Every cell runs on a FRESH gateway. An idle cell per binary (no requests) gives the
per-second idle rates, which the per-request numbers subtract. Base and head cells are
interleaved (base, head, head, base, ...) so a drift of the host lands on both sides,
and the numbers of each (path, metric) are the medians over the repeats with the
spread (max - min) / median beside them. `results.json` has the shape
`scripts/bench_gate.py compare` reads, so the same noise-aware rule judges a pull
request: a gated id regresses only when its change is worse than the threshold AND
every head repeat is worse than every base repeat. Exit codes: 0, 1 when a gated id
regressed, 2 when the run failed or a cell carries a warning (failed requests, script
misses, missing allocator statistics): a job reads 2 as "no measurement".

Standard library only; the bed is scripts/bench/grpc_stress/bed.sh and the client is
`stress_load` (bin of crates/mock_worker). See README.md next to this file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BED_SH = HERE.parent / "grpc_stress" / "bed.sh"
BENCH_GATE = HERE.parents[1] / "bench_gate.py"
SIDES = ("base", "pr")

# ----------------------------------------------------------------------------- paths

# One cell per path: the client's arguments (a deterministic prompt set per seed) and
# the scripted answer every request of the cell gets. Rates are per second at scale 1.
WORDS = "The quick brown fox jumps over the lazy dog".split()


def words(n: int) -> str:
    return " ".join(WORDS[i % len(WORDS)] for i in range(n))


TOOL_CALL_ANSWER = (
    '{"name": "lookup_1", "parameters": {"key": "k1-0", "fields": ["a", "b"], "limit": 10}}'
)
JSON_ANSWER = '{"answer": "the record is open", "confidence": 42, "status": "ok"}'


@dataclass(frozen=True)
class PathSpec:
    name: str
    rate: float
    client: tuple[str, ...]
    answer: str
    finish: str
    label: str


PATHS = {
    spec.name: spec
    for spec in (
        PathSpec(
            "chat",
            500,
            ("--mode", "chat", "--max-tokens", "8", "--seed", "101"),
            words(8),
            "stop",
            "chat, non-stream, 64-word prompt, 8 output tokens",
        ),
        PathSpec(
            "stream",
            250,
            ("--mode", "chat", "--stream", "--max-tokens", "128", "--seed", "102"),
            words(128),
            "length",
            "chat stream, 128 deltas of one token",
        ),
        PathSpec(
            "tools",
            500,
            (
                "--mode",
                "tools",
                "--turns",
                "2",
                "--tools",
                "4",
                "--max-tokens",
                "64",
                "--seed",
                "103",
            ),
            TOOL_CALL_ANSWER,
            "stop",
            "tool call: 4 schemas, 2 earlier rounds, the answer parsed into tool_calls",
        ),
        PathSpec(
            "json",
            500,
            ("--mode", "chat", "--json-schema", "--max-tokens", "64", "--seed", "104"),
            JSON_ANSWER,
            "stop",
            "json_schema answer",
        ),
        PathSpec(
            "messages",
            500,
            ("--mode", "messages", "--max-tokens", "8", "--seed", "105"),
            words(8),
            "stop",
            "Messages API, non-stream, 8 output tokens",
        ),
    )
}
DEFAULT_PATHS = "chat,stream,tools,json,messages"
DEFAULT_GW_ARGS = "--parser symphony --tool-call-parser json"


def parse_paths(text: str) -> list[PathSpec]:
    names = [n.strip() for n in text.split(",") if n.strip()]
    unknown = [n for n in names if n not in PATHS]
    if unknown:
        raise ValueError(f"unknown path(s) {unknown}; known: {', '.join(PATHS)}")
    if not names:
        raise ValueError("no path given")
    return [PATHS[n] for n in names]


def answers(spec: PathSpec, count: int) -> list[dict]:
    """The scripted answers of one cell: `count` copies, served in order to the next
    requests of every worker (the mock worker's shared queue), the text encoded by the
    worker with the bed's tokenizer; one token per streamed frame, no pause."""
    one = {
        "output_text": spec.answer,
        "finish_reason": spec.finish,
        "chunk_tokens": 1,
        "delay_ms": 0,
    }
    return [dict(one) for _ in range(count)]


def interleaved_order(repeats: int) -> list[tuple[int, str]]:
    """Repeat i alternates which side goes first (as scripts/bench_gate.py does)."""
    order = []
    for i in range(1, repeats + 1):
        sides = SIDES if i % 2 == 1 else tuple(reversed(SIDES))
        order.extend((i, side) for side in sides)
    return order


# ----------------------------------------------------------------------------- bed


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def pick_ports(base: int, workers: int, first: int = 29700, last: int = 29999) -> dict[str, int]:
    """The bed's ports: `base` when given (gateway base, metrics base+1, admin base+9,
    workers from base+10), else the first block of 10 + workers free ports in
    [first, last] (bed.sh refuses a port another process holds, so the choice is checked
    again at start)."""
    span = 10 + workers
    candidates = [base] if base else range(first, last - span + 1, span)
    for b in candidates:
        ports = [b, b + 1, b + 9, *range(b + 10, b + 10 + workers)]
        busy = [p for p in ports if not port_free(p)]
        if not busy:
            return {"gateway": b, "metrics": b + 1, "admin": b + 9, "mock_base": b + 10}
        if base:
            raise RuntimeError(
                f"port(s) {busy} of the block at {base} are in use; pass another --port-base"
            )
    raise RuntimeError(f"no free port block of {span} ports in {first}-{last}; pass --port-base")


@dataclass
class BedConfig:
    mock_bin: Path
    load_bin: Path
    tokenizer: str
    workers: int = 4
    port_base: int = 0
    gw_cores: str = ""
    mock_cores: str = ""
    gen_cores: str = ""
    gw_threads: int = 8
    gw_args: str = DEFAULT_GW_ARGS
    gw_log_level: str = "warn"
    engine_args: str = ""
    ports: dict[str, int] = field(default_factory=dict)

    def choose_ports(self) -> None:
        if not self.ports:
            self.ports = pick_ports(self.port_base, self.workers)


class Bed:
    """One bed (gateway + mock engines) under `run_dir`, through grpc_stress/bed.sh."""

    def __init__(
        self, cfg: BedConfig, gateway_bin: Path, run_dir: Path, jemalloc_stats: bool = True
    ):
        self.cfg = cfg
        self.gateway_bin = gateway_bin
        self.run_dir = run_dir
        self.jemalloc_stats = jemalloc_stats
        self.info: dict = {}

    def env(self) -> dict[str, str]:
        cfg = self.cfg
        env = dict(os.environ)
        env.update(
            GS_DIR=str(self.run_dir),
            SMG_BIN=str(self.gateway_bin),
            MOCK_BIN=str(cfg.mock_bin),
            TOKENIZER=cfg.tokenizer,
            GW_PORT=str(cfg.ports["gateway"]),
            PROM_PORT=str(cfg.ports["metrics"]),
            ADMIN_PORT=str(cfg.ports["admin"]),
            MOCK_BASE=str(cfg.ports["mock_base"]),
            WORKERS=str(cfg.workers),
            GW_CORES=cfg.gw_cores,
            MOCK_CORES=cfg.mock_cores,
            GW_THREADS=str(cfg.gw_threads),
            GW_ARGS=cfg.gw_args,
            GW_LOG_LEVEL=cfg.gw_log_level,
            SCOPE="0",
            GW_ENV="_RJEM_MALLOC_CONF=stats_print:true,stats_print_opts:Jaxe"
            if self.jemalloc_stats
            else "",
        )
        if cfg.engine_args:
            env["ENGINE_ARGS"] = cfg.engine_args
        return env

    def up(self) -> dict:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # The script refuses to start over a running bed (its pid file is there); that
        # bed is not this call's to stop, so the refusal is read here, before anything
        # starts and before the failure path below could take it down.
        if (self.run_dir / "bed" / "gateway.pid").exists():
            raise RuntimeError(
                f"a bed is up already under {self.run_dir} (take it down first: bed down)"
            )
        log = self.run_dir / "bed-up.log"
        with log.open("w") as handle:
            proc = subprocess.run(
                ["bash", str(BED_SH), "up"], env=self.env(), stdout=handle, stderr=subprocess.STDOUT
            )
        if proc.returncode != 0:
            tail = "".join(log.read_text().splitlines(keepends=True)[-15:])
            # The script can fail after the mock fleet or the gateway started (the
            # gateway's health, the workers' registration, the probe): take down what
            # this call started, by the bed's own pid files, before the error goes out,
            # so a caller that has not reached its own `finally` leaves no process and
            # no port behind.
            self.down()
            raise RuntimeError(f"bed up failed (rc {proc.returncode}); {log}:\n{tail}")
        self.info = json.loads((self.run_dir / "bed" / "bed.json").read_text())
        return self.info

    def down(self) -> None:
        log = self.run_dir / "bed-down.log"
        with log.open("w") as handle:
            subprocess.run(
                ["bash", str(BED_SH), "down"],
                env=self.env(),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )

    @property
    def pid(self) -> int:
        return int(self.info["gateway_pid"])

    @property
    def gateway_url(self) -> str:
        return self.info["gateway_url"]

    @property
    def metrics_url(self) -> str:
        return self.info["metrics_url"]

    @property
    def admin_url(self) -> str:
        return self.info["admin_url"]

    def gateway_log(self) -> str:
        path = self.run_dir / "bed" / "gateway.log"
        return path.read_text(errors="replace") if path.exists() else ""


def http(method: str, url: str, body: object | None = None, timeout: float = 30) -> str:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["content-type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode(errors="replace")


ENQUEUE_CHUNK = 500


def enqueue(bed: Bed, items: list[dict]) -> None:
    """Hand the mock engines the cell's answers: the shared queue every worker reads,
    emptied first (an earlier window's leftovers would answer the next requests), filled
    in chunks (ten thousand 128-token answers in one body would exceed the admin API's
    request size)."""
    http("DELETE", f"{bed.admin_url}/admin/script/all", timeout=30)
    for start in range(0, len(items), ENQUEUE_CHUNK):
        http(
            "POST",
            f"{bed.admin_url}/admin/script/all",
            items[start : start + ENQUEUE_CHUNK],
            timeout=120,
        )


def script_counts(bed: Bed) -> dict:
    """Hits and misses over the workers, what is left in the shared queue, and how many
    of those reads failed (`unreadable`): a counter that cannot be read leaves the misses
    unknown, which the callers warn about as they do about misses, never a pass."""
    hits = misses = unreadable = 0
    for i in range(bed.cfg.workers):
        try:
            state = json.loads(
                http("GET", f"{bed.admin_url}/admin/script/grpc:{bed.cfg.ports['mock_base'] + i}")
            )
        except (urllib.error.URLError, ValueError, OSError):
            unreadable += 1
            continue
        hits += int(state.get("hits", 0) or 0)
        misses += int(state.get("misses", 0) or 0)
    try:
        shared = json.loads(http("GET", f"{bed.admin_url}/admin/script/all"))
    except (urllib.error.URLError, ValueError, OSError):
        unreadable += 1
        shared = {}
    return {
        "hits": hits,
        "misses": misses,
        "unreadable": unreadable,
        "queued_left": shared.get("queued", shared.get("queue", None)),
    }


# ----------------------------------------------------------------------------- counters


def proc_cpu_seconds(pid: int) -> float:
    """utime + stime of the whole process (every thread), in seconds."""
    text = Path(f"/proc/{pid}/stat").read_text()
    rest = text[text.rindex(")") + 2 :].split()
    return (int(rest[11]) + int(rest[12])) / os.sysconf("SC_CLK_TCK")


def proc_status(pid: int) -> dict[str, int]:
    out = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        m = re.match(r"(VmRSS|VmHWM|RssAnon|RssFile|Threads):\s+(\d+)", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


GAUGES = (
    "smg_allocator_allocated_bytes",
    "smg_allocator_resident_bytes",
    "smg_allocator_active_bytes",
)


def metrics_gauges(text: str) -> dict[str, float]:
    out = {}
    for line in text.splitlines():
        for name in GAUGES:
            if line.startswith(name + " ") or line.startswith(name + "{"):
                try:
                    out[name] = float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return out


def snapshot(bed: Bed) -> dict:
    try:
        gauges = metrics_gauges(http("GET", bed.metrics_url, timeout=10))
    except (urllib.error.URLError, OSError):
        gauges = {}
    return {
        "t": time.time(),
        "cpu_s": proc_cpu_seconds(bed.pid),
        "status": proc_status(bed.pid),
        "gauges": gauges,
        "loadavg": float(Path("/proc/loadavg").read_text().split()[0]),
    }


PERF_EVENTS = "instructions,instructions:u,cycles,cycles:u,task-clock,context-switches"


def perf_available() -> tuple[bool, str]:
    """Whether `perf stat` can count instructions here (the tool installed, the kernel's
    perf_event_paranoid allowing it, the hardware exposing a PMU: a virtual machine of a
    hosted CI runner usually does not)."""
    if shutil.which("perf") is None:
        return False, "perf not installed"
    try:
        proc = subprocess.run(
            ["perf", "stat", "-x,", "-e", "instructions:u", "--", "true"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as err:
        return False, f"perf stat failed: {err}"
    if proc.returncode != 0:
        return False, f"perf stat exited with {proc.returncode}: {proc.stdout.strip()[:200]}"
    for line in proc.stdout.splitlines():
        if "instructions" in line and not line.startswith("#"):
            value = line.split(",")[0]
            if value in ("<not supported>", "<not counted>"):
                return False, f"instructions {value}"
            return True, "perf stat counts instructions"
    return False, "perf stat printed no instructions line"


class PerfStat:
    """`perf stat` on one pid, without a workload: it counts until SIGINT, then prints."""

    def __init__(self, pid: int, out: Path):
        self.out = out
        self.proc = subprocess.Popen(
            ["perf", "stat", "-e", PERF_EVENTS, "-x,", "-p", str(pid), "-o", str(out)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=dict(os.environ, DEBUGINFOD_URLS=""),
        )

    def stop(self) -> dict[str, float]:
        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        return parse_perf_csv(self.out.read_text() if self.out.exists() else "")


def parse_perf_csv(text: str) -> dict[str, float]:
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split(",")
        if len(fields) >= 3 and fields[0] not in ("<not counted>", "<not supported>"):
            try:
                out[fields[2]] = float(fields[0])
            except ValueError:
                pass
    return out


def parse_jemalloc(log: str) -> dict:
    """jemalloc's exit statistics (the JSON the gateway prints at a clean shutdown with
    `stats_print:true,stats_print_opts:Jaxe`): allocation requests over the process's
    life (tcache hits included) and the bytes requested, summed as size class x
    requests over the merged arena's bins and large extents."""
    i = log.find('"jemalloc"')
    if i < 0:
        return {"error": "no jemalloc statistics in the gateway log"}
    j0 = log.rfind("{", 0, i)
    try:
        obj, _ = json.JSONDecoder().raw_decode(log[j0:])
    except ValueError as err:
        return {"error": f"jemalloc statistics unreadable: {err}"}
    st = obj.get("jemalloc", {})
    merged = st.get("stats.arenas", {}).get("merged", {})
    small, large = merged.get("small", {}), merged.get("large", {})
    bins_cfg = st.get("arenas", {}).get("bin", [])
    lext_cfg = st.get("arenas", {}).get("lextent", [])
    bytes_requested = 0
    for k, b in enumerate(merged.get("bins", [])):
        if k < len(bins_cfg) and bins_cfg[k].get("size"):
            bytes_requested += bins_cfg[k]["size"] * b.get("nrequests", 0)
    for k, b in enumerate(merged.get("lextents", [])):
        if k < len(lext_cfg) and lext_cfg[k].get("size"):
            bytes_requested += lext_cfg[k]["size"] * b.get("nrequests", 0)
    return {
        "nrequests": small.get("nrequests", 0) + large.get("nrequests", 0),
        "small_nrequests": small.get("nrequests", 0),
        "large_nrequests": large.get("nrequests", 0),
        "nmalloc": small.get("nmalloc", 0) + large.get("nmalloc", 0),
        "bytes_requested": bytes_requested,
        "allocated": st.get("stats", {}).get("allocated"),
        "resident": st.get("stats", {}).get("resident"),
    }


# ----------------------------------------------------------------------------- cells


@dataclass
class Settings:
    out: Path
    bed: BedConfig
    binaries: dict[str, Path]
    paths: list[PathSpec]
    requests: int = 10_000
    repeats: int = 3
    rate_scale: float = 1.0
    idle_seconds: int = 25
    counters: bool = True
    counters_note: str = ""
    drain: int = 120
    prompt_tokens: int = 64
    distinct: int = 64


def run_client(
    settings: Settings,
    bed: Bed,
    spec: PathSpec,
    rate: float,
    seconds: int,
    out: Path,
    extra: tuple[str, ...] = (),
    warmup: int = 0,
    threads: int = 4,
) -> dict:
    cmd = []
    if settings.bed.gen_cores:
        cmd += ["taskset", "-c", settings.bed.gen_cores]
    cmd += [
        str(settings.bed.load_bin),
        "--gateway",
        bed.gateway_url,
        "--distinct",
        str(settings.distinct),
        "--prompt-tokens",
        str(settings.prompt_tokens),
        "--rate",
        f"{rate:g}",
        "--duration",
        str(seconds),
        "--warmup",
        str(warmup),
        "--drain",
        str(settings.drain),
        "--threads",
        str(threads),
        "--interim",
        "0",
        *spec.client,
        *extra,
        "--out",
        str(out / "summary.json"),
    ]
    with (out / "client.out").open("w") as o, (out / "client.err").open("w") as e:
        proc = subprocess.run(cmd, stdout=o, stderr=e)
    summary_path = out / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(
            f"the client wrote no summary (rc {proc.returncode}); see {out / 'client.err'}"
        )
    summary = json.loads(summary_path.read_text())
    summary["client_rc"] = proc.returncode
    return summary


def cell(settings: Settings, side: str, spec: PathSpec | None, label: str) -> dict:
    """One fresh gateway of `side`'s binary: a window of requests of `spec` (or an idle
    window), the gateway's counters over it. Returns the cell record (also written as
    cell.json under the cell's directory)."""
    name = spec.name if spec else "idle"
    cdir = settings.out / "cells" / side / name / label
    cdir.mkdir(parents=True, exist_ok=True)
    bed = Bed(settings.bed, settings.binaries[side], cdir / "bed")
    record = {
        "side": side,
        "path": name,
        "label": label,
        "binary": str(settings.binaries[side]),
        "dir": str(cdir),
    }
    t_start = time.time()
    print(
        f"[{time.strftime('%H:%M:%S')}] cell {side}/{name}/{label} load={Path('/proc/loadavg').read_text().split()[0]}",
        flush=True,
    )
    bed.up()
    try:
        rate = spec.rate * settings.rate_scale if spec else 0
        seconds = (
            max(1, -(-settings.requests // max(1, int(rate)))) if spec else settings.idle_seconds
        )
        if spec:
            enqueue(bed, answers(spec, int(rate * seconds) + 256))
        time.sleep(2)
        s0 = snapshot(bed)
        perf = PerfStat(bed.pid, cdir / "perf.csv") if settings.counters else None
        time.sleep(0.5)
        if spec:
            summary = run_client(settings, bed, spec, rate, seconds, cdir)
        else:
            time.sleep(seconds)
            summary = {"sent": 0, "ok": 0, "failed": 0, "skipped": 0, "statuses": {}, "errors": {}}
        time.sleep(1)
        counters = perf.stop() if perf else {}
        s1 = snapshot(bed)
        script = script_counts(bed) if spec else {}
    finally:
        bed.down()
    time.sleep(0.5)
    record.update(
        {
            "sent": summary.get("sent", 0),
            "ok": summary.get("ok", 0),
            "failed": summary.get("failed", 0),
            "skipped": summary.get("skipped", 0),
            "statuses": summary.get("statuses", {}),
            "errors": summary.get("errors", {}),
            "achieved_rps": summary.get("achieved_rps"),
            "ttft_p50_ms": (summary.get("ttft_ms") or {}).get("p50"),
            "ttft_p99_ms": (summary.get("ttft_ms") or {}).get("p99"),
            "e2e_p99_ms": (summary.get("e2e_ms") or {}).get("p99"),
            "frames_per_request": summary.get("frames_per_request"),
            "client_rc": summary.get("client_rc", 0),
            "window_s": s1["t"] - s0["t"],
            "cpu_s": s1["cpu_s"] - s0["cpu_s"],
            "status_0": s0["status"],
            "status_1": s1["status"],
            "gauges_0": s0["gauges"],
            "gauges_1": s1["gauges"],
            "loadavg_0": s0["loadavg"],
            "loadavg_1": s1["loadavg"],
            "perf": counters,
            "script": script,
            "jemalloc": parse_jemalloc(bed.gateway_log()),
            "seconds_total": time.time() - t_start,
        }
    )
    bad = []
    if spec and record["failed"]:
        bad.append(f"{record['failed']} failed requests")
    if spec and record["script"].get("misses"):
        bad.append(
            f"{record['script']['misses']} script misses (answers served by the engine simulator)"
        )
    if spec and record["script"].get("unreadable"):
        bad.append(f"{record['script']['unreadable']} script counters unreadable (misses unknown)")
    if "error" in record["jemalloc"]:
        bad.append(record["jemalloc"]["error"])
    print(
        f"    {record['sent']} sent, {record['ok']} ok, window {record['window_s']:.1f} s, cpu {record['cpu_s']:.2f} s"
        + (f"; WARNING: {'; '.join(bad)}" if bad else ""),
        flush=True,
    )
    record["warnings"] = bad
    (cdir / "cell.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


# ----------------------------------------------------------------------------- arithmetic


def per_request(c: dict, idle: dict | None) -> dict[str, float]:
    """The per-request numbers of one loaded cell, divided by the requests sent. The
    counters read over the window (perf's, CPU time) lose the idle cell's per-second
    rate times the window. The allocator's counters are the process's exit totals, so
    the idle cell's totals are subtracted whole: that removes the start-up share and
    the idle allocations over the idle cell's life; what stays beside the requests'
    own allocations is the idle rate times the difference of the two cells' lives
    (seconds of a rate of a few allocations per second, spread over the requests), the
    same on both sides of an A/B, so the comparison is unbiased and the absolute
    values carry it. Instructions and allocation counts are load-independent; cycles
    and CPU time are not."""
    n = c.get("sent") or 0
    if not n:
        return {}
    w = c.get("window_s") or 0.0
    iw = (idle or {}).get("window_s") or 1.0

    def idle_rate(key: str, table: str) -> float:
        return ((idle or {}).get(table, {}).get(key, 0.0) / iw) if idle else 0.0

    p = c.get("perf", {})
    out = {
        "cpu_us_per_request": (
            (c.get("cpu_s") or 0.0) - ((idle or {}).get("cpu_s") or 0.0) / iw * w
        )
        / n
        * 1e6,
        "rss_mb_after_window": c.get("status_1", {}).get("VmRSS", 0) / 1024,
        "hwm_mb": c.get("status_1", {}).get("VmHWM", 0) / 1024,
    }
    if "instructions:u" in p:
        out["user_instructions_per_request_M"] = (
            (p["instructions:u"] - idle_rate("instructions:u", "perf") * w) / n / 1e6
        )
    if "instructions" in p:
        out["instructions_per_request_M"] = (
            (p["instructions"] - idle_rate("instructions", "perf") * w) / n / 1e6
        )
    if "cycles:u" in p:
        out["user_cycles_per_request_M"] = (
            (p["cycles:u"] - idle_rate("cycles:u", "perf") * w) / n / 1e6
        )
    if "cycles" in p:
        out["cycles_per_request_M"] = (p["cycles"] - idle_rate("cycles", "perf") * w) / n / 1e6
    if "context-switches" in p:
        out["context_switches_per_request"] = (
            p["context-switches"] - idle_rate("context-switches", "perf") * w
        ) / n
    je, ije = c.get("jemalloc", {}), (idle or {}).get("jemalloc", {})
    if "nrequests" in je and "nrequests" in ije:
        out["allocation_calls_per_request"] = (je["nrequests"] - ije["nrequests"]) / n
        out["kib_requested_per_request"] = (
            (je.get("bytes_requested", 0) - ije.get("bytes_requested", 0)) / n / 1024
        )
        out["large_allocations_per_request"] = (
            je.get("large_nrequests", 0) - ije.get("large_nrequests", 0)
        ) / n
    return out


def spread(values: list[float]) -> float | None:
    if not values:
        return None
    med = statistics.median(values)
    return (max(values) - min(values)) / abs(med) if med else None


def results_from_cells(
    cells: list[dict], repeats: int, binaries: dict[str, Path], note: str = ""
) -> dict:
    """results.json in the shape scripts/bench_gate.py reads: per side one run per
    repeat, each run a map of `<path>/<metric>` to its per-request value."""
    idle = {c["side"]: c for c in cells if c["path"] == "idle"}
    runs: dict[str, dict[int, dict[str, float]]] = {side: {} for side in SIDES}
    for c in cells:
        if c["path"] == "idle":
            continue
        m = re.match(r"r(\d+)$", c["label"])
        if not m:
            continue
        values = per_request(c, idle.get(c["side"]))
        run = runs[c["side"]].setdefault(int(m.group(1)), {})
        for metric, value in values.items():
            run[f"{c['path']}/{metric}"] = value
    results = {
        "unit": "",
        "note": note,
        "repeats": repeats,
        "filter": None,
        "bench_args": [],
        "order": [f"{side}-{i}" for i, side in interleaved_order(repeats)],
    }
    for side in SIDES:
        results[side] = {
            "binary": str(binaries.get(side, "")),
            "runs": [runs[side][i] for i in sorted(runs[side])],
            "seconds": [],
        }
    return results


METRIC_ROWS = (
    ("user_instructions_per_request_M", "user instructions per request", "{:.3f} M", True),
    ("allocation_calls_per_request", "allocation calls per request", "{:,.0f}", False),
    ("kib_requested_per_request", "KiB requested per request", "{:.1f}", False),
    ("cpu_us_per_request", "gateway CPU per request", "{:.0f} us", True),
    ("rss_mb_after_window", "RSS after the window", "{:.1f} MB", True),
    ("hwm_mb", "peak RSS (VmHWM)", "{:.1f} MB", True),
    ("instructions_per_request_M", "instructions per request (user + kernel)", "{:.3f} M", True),
    ("rss_mb_after_first_window", "RSS after the first window", "{:.1f} MB", True),
    ("rss_mb_after_last_window", "RSS after the last window", "{:.1f} MB", True),
    ("rss_growth_mb_per_10k_requests", "RSS growth per 10,000 requests", "{:+.2f} MB", False),
    ("peak_rss_mb_after_burst", "peak RSS after the burst window", "{:.1f} MB", True),
    (
        "allocator_live_mb_after_last_window",
        "allocator live bytes after the last window",
        "{:.1f} MB",
        True,
    ),
)


def numbers_table(results: dict, setup: str) -> str:
    """The `## Numbers` table of a pull request: metric, before, after, setup; medians
    over the repeats, the change in the after column, the base's spread in the setup
    column (the floor under which a change is not a change)."""
    per_side: dict[str, dict[str, list[float]]] = {side: {} for side in SIDES}
    for side in SIDES:
        for run in results.get(side, {}).get("runs", []):
            for bench_id, value in run.items():
                per_side[side].setdefault(bench_id, []).append(value)
    ids = sorted(set(per_side["base"]) | set(per_side["pr"]))
    lines = ["| metric | before | after | setup |", "|---|---|---|---|"]
    order = {key: n for n, (key, _, _, _) in enumerate(METRIC_ROWS)}
    rows = []
    for bench_id in ids:
        prefix, _, metric = bench_id.rpartition("/")
        if metric not in order:
            continue
        rows.append((prefix, order[metric], bench_id, metric))
    for prefix, _, bench_id, metric in sorted(rows):
        _, title, fmt, relative = next(r for r in METRIC_ROWS if r[0] == metric)
        b, h = per_side["base"].get(bench_id, []), per_side["pr"].get(bench_id, [])
        if not b or not h:
            continue
        bm, hm = statistics.median(b), statistics.median(h)
        if relative:
            change = f"{(hm - bm) / abs(bm) * 100:+.1f} %" if bm else "-"
        else:
            change = (
                f"{hm - bm:+,.1f}"
                if metric != "allocation_calls_per_request"
                else f"{hm - bm:+,.0f}"
            )
        floor = spread(b)
        floor_text = f"; base spread {floor * 100:.2f} %" if floor is not None else ""
        lines.append(
            f"| {prefix}: {title} | {fmt.format(bm)} | {fmt.format(hm)} ({change}) "
            f"| {setup}; {len(b)}+{len(h)} interleaved cells{floor_text} |"
        )
    return "\n".join(lines)


def load_bench_gate():
    spec = importlib.util.spec_from_file_location("bench_gate", BENCH_GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("bench_gate", module)
    spec.loader.exec_module(module)
    return module


def gate_report(results: dict, gates_path: Path | None) -> tuple[str, int]:
    """The verdict table of scripts/bench_gate.py over results.json (report only without
    a gate file); returns the markdown and the exit code the gate would give."""
    bg = load_bench_gate()
    gates = bg.load_gates(gates_path) if gates_path else bg.Gates()
    rows = bg.compare(results, gates)
    text = bg.markdown(results, rows)
    return text, (1 if any(row.fails for row in rows) else 0)


# ----------------------------------------------------------------------------- commands


def common_settings(args: argparse.Namespace, need_base: bool) -> Settings:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = args.tokenizer or os.environ.get("TOKENIZER") or ""
    if not tokenizer:
        tok_dir = out / "tokenizer"
        subprocess.run(
            [sys.executable, str(HERE / "synthetic_tokenizer.py"), str(tok_dir)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        tokenizer = str(tok_dir)
    bed = BedConfig(
        mock_bin=Path(args.mock).resolve(),
        load_bin=Path(args.load).resolve(),
        tokenizer=tokenizer,
        workers=args.workers,
        port_base=args.port_base,
        gw_cores=args.gw_cores or os.environ.get("GW_CORES", ""),
        mock_cores=args.mock_cores or os.environ.get("MOCK_CORES", ""),
        gen_cores=args.gen_cores or os.environ.get("GEN_CORES", ""),
        gw_threads=args.gw_threads,
        gw_args=args.gw_args,
    )
    bed.choose_ports()
    binaries = {"pr": Path(args.head).resolve()}
    if need_base:
        if not args.base:
            raise ValueError(
                "--base is required: the base branch's gateway binary (see README.md: make perf-base)"
            )
        binaries["base"] = Path(args.base).resolve()
    for side, path in binaries.items():
        if not os.access(path, os.X_OK):
            raise ValueError(f"no executable gateway for {side} at {path}")
    for name in ("mock_bin", "load_bin"):
        if not os.access(getattr(bed, name), os.X_OK):
            raise ValueError(
                f"no executable at {getattr(bed, name)} ({name}); build with: "
                "cargo build --release -p mock-worker --bin mock-worker --bin stress_load"
            )
    counters, note = (
        (False, "counters off by request") if args.counters == "off" else perf_available()
    )
    if args.counters == "on" and not counters:
        raise ValueError(f"--counters on, but {note}")
    settings = Settings(
        out=out,
        bed=bed,
        binaries=binaries,
        paths=parse_paths(args.paths) if hasattr(args, "paths") else [],
        requests=getattr(args, "requests", 10_000),
        repeats=getattr(args, "repeats", 3),
        rate_scale=getattr(args, "rate_scale", 1.0),
        idle_seconds=getattr(args, "idle_seconds", 25),
        counters=counters,
        counters_note=note,
        prompt_tokens=getattr(args, "prompt_tokens", 64),
        distinct=getattr(args, "distinct", 64),
    )
    print(f"out {out}\nports {bed.ports}\ntokenizer {tokenizer}\ncounters: {note}", flush=True)
    return settings


def setup_text(settings: Settings, mode: str) -> str:
    cores = f"; gateway pinned to cores {settings.bed.gw_cores}" if settings.bed.gw_cores else ""
    prompt = f"{settings.prompt_tokens}-word prompts" + (
        f", {settings.distinct} distinct" if settings.distinct != 64 else ""
    )
    return (
        f"{mode}: {settings.requests:,} requests per cell at the path's rate x {settings.rate_scale:g}, {prompt}, "
        f"{settings.bed.workers} scripted mock engines, {settings.bed.gw_threads} runtime threads{cores}"
    )


def write_outputs(
    settings: Settings, cells: list[dict], results: dict, args: argparse.Namespace, setup: str
) -> int:
    (settings.out / "cells.json").write_text(json.dumps(cells, indent=1) + "\n")
    (settings.out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    loads = [
        c[k]
        for c in cells
        for k in ("loadavg_0", "loadavg_1")
        if isinstance(c.get(k), (int, float))
    ]
    load_line = (
        f"1-minute load average of the machine over the cells: {min(loads):.0f}-{max(loads):.0f} "
        "(cycles, CPU time and latencies carry it; instructions and allocation counts do not)."
        if loads
        else ""
    )
    numbers = numbers_table(results, setup) + ("\n\n" + load_line if load_line else "")
    (settings.out / "numbers.md").write_text(numbers + "\n")
    gates = Path(args.gates) if getattr(args, "gates", None) else None
    gate_md, rc = gate_report(results, gates)
    (settings.out / "gate.md").write_text(gate_md)
    print("\n## Numbers\n\n" + numbers + "\n\n" + gate_md, flush=True)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as handle:
            handle.write(f"## Gateway perf: {setup}\n\n{numbers}\n\n{gate_md}\n")
    warnings = [
        f"{c['side']}/{c['path']}/{c['label']}: {w}" for c in cells for w in c.get("warnings", [])
    ]
    print(f"results: {settings.out / 'results.json'}", flush=True)
    if warnings:
        # A cell that did not measure what it should (failed requests, script misses, a
        # lost jemalloc print): the tables stay on disk for a read, the run fails.
        print("WARNINGS:\n  " + "\n  ".join(warnings), flush=True)
        print(
            f"gateway_perf: {len(warnings)} cell warning(s); the run did not measure what it should",
            file=sys.stderr,
        )
        return 2
    return rc


def cmd_cells(args: argparse.Namespace) -> int:
    """cpu and alloc: the interleaved base/head cells over the paths."""
    settings = common_settings(args, need_base=True)
    if args.command == "alloc":
        settings.counters = False
        settings.counters_note = "alloc mode: jemalloc statistics only"
    cells = [cell(settings, side, None, "idle") for side in SIDES]
    for i, side in interleaved_order(settings.repeats):
        for spec in settings.paths:
            cells.append(cell(settings, side, spec, f"r{i}"))
    results = results_from_cells(
        cells, settings.repeats, settings.binaries, note=settings.counters_note
    )
    return write_outputs(settings, cells, results, args, setup_text(settings, args.command))


def mem_soak(
    settings: Settings, side: str, spec: PathSpec, label: str, windows: int, burst: float
) -> dict:
    """One gateway, `windows` windows of requests, RSS/HWM and the allocator gauges after
    every window, then one window at `burst` x the rate for the peak."""
    cdir = settings.out / "cells" / side / f"mem-{spec.name}" / label
    cdir.mkdir(parents=True, exist_ok=True)
    bed = Bed(settings.bed, settings.binaries[side], cdir / "bed", jemalloc_stats=False)
    print(
        f"[{time.strftime('%H:%M:%S')}] mem {side}/{spec.name}/{label}: "
        f"{windows} windows of {settings.requests:,} + a burst at x{burst:g}",
        flush=True,
    )
    bed.up()
    rows = []
    summaries = []
    script = {}
    try:
        rate = spec.rate * settings.rate_scale
        seconds = max(1, -(-settings.requests // max(1, int(rate))))
        time.sleep(2)
        rows.append({"after": "start", **snapshot(bed)})
        for w in range(1, windows + 1):
            wdir = cdir / f"window-{w}"
            wdir.mkdir(exist_ok=True)
            enqueue(bed, answers(spec, int(rate * seconds) + 256))
            summaries.append(run_client(settings, bed, spec, rate, seconds, wdir))
            time.sleep(2)
            rows.append(
                {"after": f"window {w} ({w * settings.requests:,} requests)", **snapshot(bed)}
            )
        bdir = cdir / "burst"
        bdir.mkdir(exist_ok=True)
        enqueue(bed, answers(spec, int(rate * burst * seconds) + 256))
        summaries.append(run_client(settings, bed, spec, rate * burst, seconds, bdir))
        time.sleep(2)
        rows.append({"after": f"burst window (x{burst:g} rate)", **snapshot(bed)})
        time.sleep(10)
        rows.append({"after": "10 s idle", **snapshot(bed)})
        script = script_counts(bed)
    finally:
        bed.down()
    first, last, after_burst = rows[1], rows[windows], rows[windows + 1]
    rss = lambda r: r["status"].get("VmRSS", 0) / 1024  # noqa: E731
    per_10k = 10_000 / settings.requests
    record = {
        "side": side,
        "path": spec.name,
        "label": label,
        "dir": str(cdir),
        "rows": rows,
        "loadavg_0": rows[0]["loadavg"],
        "loadavg_1": rows[-1]["loadavg"],
        "values": {
            f"mem-{spec.name}/rss_mb_after_first_window": rss(first),
            f"mem-{spec.name}/rss_mb_after_last_window": rss(last),
            f"mem-{spec.name}/rss_growth_mb_per_10k_requests": (rss(last) - rss(first))
            / max(1, windows - 1)
            * per_10k,
            f"mem-{spec.name}/peak_rss_mb_after_burst": after_burst["status"].get("VmHWM", 0)
            / 1024,
            f"mem-{spec.name}/allocator_live_mb_after_last_window": last["gauges"].get(
                "smg_allocator_allocated_bytes", 0
            )
            / 1e6,
        },
    }
    failed = sum(int(s.get("failed", 0) or 0) for s in summaries)
    record.update(
        {
            "sent": sum(int(s.get("sent", 0) or 0) for s in summaries),
            "ok": sum(int(s.get("ok", 0) or 0) for s in summaries),
            "failed": failed,
            "client_rc": max((int(s.get("client_rc", 0) or 0) for s in summaries), default=0),
            "script": script,
        }
    )
    # The soak measured what it should only when every window's requests were answered by the
    # script, as a cell's (`cell`): a failed request or a script miss is a warning, and the run
    # fails on it (`write_outputs`), the numbers kept on disk for a read.
    bad = []
    if failed:
        bad.append(f"{failed} failed requests")
    if script.get("misses"):
        bad.append(f"{script['misses']} script misses (answers served by the engine simulator)")
    if script.get("unreadable"):
        bad.append(f"{script['unreadable']} script counters unreadable (misses unknown)")
    if bad:
        print(f"    WARNING: {'; '.join(bad)}", flush=True)
    record["warnings"] = bad
    (cdir / "soak.json").write_text(json.dumps(record, indent=1) + "\n")
    for r in rows:
        print(
            f"    after {r['after']}: RSS {rss(r):.1f} MB, HWM {r['status'].get('VmHWM', 0) / 1024:.1f} MB, "
            f"live {r['gauges'].get('smg_allocator_allocated_bytes', 0) / 1e6:.1f} MB",
            flush=True,
        )
    return record


def cmd_mem(args: argparse.Namespace) -> int:
    settings = common_settings(args, need_base=True)
    records = []
    for i, side in interleaved_order(settings.repeats):
        for spec in settings.paths:
            records.append(mem_soak(settings, side, spec, f"r{i}", args.windows, args.burst))
    results = {
        "unit": "",
        "repeats": settings.repeats,
        "filter": None,
        "bench_args": [],
        "order": [f"{s}-{i}" for i, s in interleaved_order(settings.repeats)],
    }
    for side in SIDES:
        runs: dict[int, dict] = {}
        for r in records:
            if r["side"] == side:
                runs.setdefault(int(r["label"][1:]), {}).update(r["values"])
        results[side] = {
            "binary": str(settings.binaries[side]),
            "runs": [runs[i] for i in sorted(runs)],
            "seconds": [],
        }
    setup = (
        f"mem: {args.windows} windows of {settings.requests:,} requests on one gateway, "
        f"then a burst at x{args.burst:g}; {settings.bed.workers} scripted mock engines"
    )
    return write_outputs(settings, records, results, args, setup)


def capacity(steps: list[dict]) -> dict:
    """The capacity rule over steps sorted by offered rate: HOLD = the highest offered
    step at which the SLO holds; STRICT = the last holding step before the first failing
    one (the first-break reading). Both are reported; HOLD is the capacity."""
    ordered = sorted(steps, key=lambda s: s["offered"])
    holding = [s["offered"] for s in ordered if s["holds"]]
    hold = max(holding) if holding else None
    strict = None
    for s in ordered:
        if s["holds"]:
            strict = s["offered"]
        else:
            break
    failing_below = [
        s["offered"] for s in ordered if not s["holds"] and hold is not None and s["offered"] < hold
    ]
    return {
        "rule": "hold: the highest offered step at which the SLO holds; "
        "strict = the last holding step before the first failing one",
        "hold": hold,
        "strict": strict,
        "failing_steps_below_hold": failing_below,
        # The top step held: the capacity is at least the hold, the ladder was too short.
        "at_ceiling": bool(ordered) and hold == ordered[-1]["offered"],
        "steps": ordered,
    }


def judge_step(summary: dict, cpu_s: float, offered: float, slo: dict) -> dict:
    sent = summary.get("sent", 0) or 0
    failed = (summary.get("failed", 0) or 0) + (summary.get("skipped", 0) or 0)
    ttft = (summary.get("ttft_ms") or {}).get("p99")
    e2e = (summary.get("e2e_ms") or {}).get("p99")
    achieved = summary.get("achieved_rps") or 0.0
    reasons = []
    if ttft is None or ttft > slo["ttft_p99_ms"]:
        reasons.append(f"TTFT p99 {ttft} > {slo['ttft_p99_ms']} ms")
    if e2e is None or e2e > slo["e2e_p99_ms"]:
        reasons.append(f"e2e p99 {e2e} > {slo['e2e_p99_ms']} ms")
    if sent and failed / sent > slo["error_rate"]:
        reasons.append(f"errors {failed}/{sent} > {slo['error_rate']:g}")
    if achieved < offered * slo["achieved_fraction"]:
        reasons.append(
            f"achieved {achieved:.0f} < {slo['achieved_fraction']:.0%} of offered {offered:g} req/s"
        )
    return {
        "offered": offered,
        "achieved": achieved,
        "sent": sent,
        "ok": summary.get("ok", 0),
        "failed": failed,
        "ttft_p50_ms": (summary.get("ttft_ms") or {}).get("p50"),
        "ttft_p99_ms": ttft,
        "e2e_p50_ms": (summary.get("e2e_ms") or {}).get("p50"),
        "e2e_p99_ms": e2e,
        "cpu_ms_per_request": cpu_s * 1000 / sent if sent else None,
        "holds": not reasons,
        "reasons": reasons,
    }


def cmd_live(args: argparse.Namespace) -> int:
    settings = common_settings(args, need_base=bool(args.base))
    spec = parse_paths(args.path)[0]
    rates = [float(r) for r in args.rates.split(",") if r.strip()]
    slo = {
        "ttft_p99_ms": args.slo_ttft_p99_ms,
        "e2e_p99_ms": args.slo_e2e_p99_ms,
        "error_rate": args.slo_error_rate,
        "achieved_fraction": args.slo_achieved_fraction,
    }
    sides = [s for s in SIDES if s in settings.binaries]
    steps: dict[str, list[dict]] = {s: [] for s in sides}
    for offered in rates:
        for side in sides:
            sdir = settings.out / "live" / side / f"rate-{offered:g}"
            sdir.mkdir(parents=True, exist_ok=True)
            bed = Bed(settings.bed, settings.binaries[side], sdir / "bed", jemalloc_stats=False)
            print(
                f"[{time.strftime('%H:%M:%S')}] live {side} {spec.name} at {offered:g} req/s for {args.step_seconds} s",
                flush=True,
            )
            bed.up()
            try:
                enqueue(
                    bed,
                    answers(spec, int(offered * (args.step_seconds + args.step_warmup + 2)) + 256),
                )
                time.sleep(1)
                s0 = snapshot(bed)
                summary = run_client(
                    settings,
                    bed,
                    spec,
                    offered,
                    args.step_seconds,
                    sdir,
                    warmup=args.step_warmup,
                    threads=8,
                )
                s1 = snapshot(bed)
            finally:
                bed.down()
            step = judge_step(summary, s1["cpu_s"] - s0["cpu_s"], offered, slo)
            step["rss_mb_end"] = s1["status"].get("VmRSS", 0) / 1024
            step["loadavg"] = s1["loadavg"]
            steps[side].append(step)
            print(
                f"    achieved {step['achieved']:.0f} req/s, "
                f"TTFT p50/p99 {step['ttft_p50_ms']}/{step['ttft_p99_ms']} ms, e2e p99 {step['e2e_p99_ms']} ms, "
                f"CPU {step['cpu_ms_per_request'] and round(step['cpu_ms_per_request'], 3)} ms/req, "
                f"{'holds' if step['holds'] else 'FAILS: ' + '; '.join(step['reasons'])}",
                flush=True,
            )
    report = {side: capacity(steps[side]) for side in sides}
    (settings.out / "live.json").write_text(
        json.dumps({"path": spec.name, "slo": slo, "sides": report}, indent=1) + "\n"
    )
    lines = [
        f"## Live: open-loop ramp on the {spec.name} path",
        "",
        f"SLO: TTFT p99 <= {slo['ttft_p99_ms']} ms, e2e p99 <= {slo['e2e_p99_ms']} ms, "
        f"error rate <= {slo['error_rate']:g}, achieved >= {slo['achieved_fraction']:.0%} of offered; "
        f"{args.step_seconds} s per step after {args.step_warmup} s warm-up, a fresh gateway per step.",
        "",
    ]
    lines += [
        "| side | offered req/s | achieved | TTFT p50 / p99 ms | e2e p50 / p99 ms | errors "
        "| gateway CPU ms/req | RSS MB | holds |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for side in sides:
        for s in report[side]["steps"]:
            cpu_ms = None if s["cpu_ms_per_request"] is None else round(s["cpu_ms_per_request"], 3)
            lines.append(
                f"| {side} | {s['offered']:g} | {s['achieved']:.0f} "
                f"| {s['ttft_p50_ms']} / {s['ttft_p99_ms']} | {s['e2e_p50_ms']} / {s['e2e_p99_ms']} "
                f"| {s['failed']}/{s['sent']} "
                f"| {cpu_ms} | {s['rss_mb_end']:.0f} | {'yes' if s['holds'] else 'no: ' + '; '.join(s['reasons'])} |"
            )
    lines.append("")
    for side in sides:
        r = report[side]
        ceiling = (
            " (the top step of the ladder held: the capacity is at least this; extend --rates)"
            if r["at_ceiling"]
            else ""
        )
        lines.append(
            f"- {side}: capacity (hold) {r['hold']} req/s{ceiling}; strict first-break reading {r['strict']} req/s"
            + (
                f"; failing steps below the hold: {r['failing_steps_below_hold']}"
                if r["failing_steps_below_hold"]
                else ""
            )
            + f". Rule: {r['rule']}."
        )
    text = "\n".join(lines) + "\n"
    (settings.out / "live.md").write_text(text)
    print("\n" + text, flush=True)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as handle:
            handle.write(text)
    return 0


def cmd_bed(args: argparse.Namespace) -> int:
    """up: start a bed and print its bed.json path (for an interactive session or the
    grpc_stress cells); down: stop the bed under --out; smoke: up, one request per
    path with the first response shown, the script counts, down."""
    if args.action == "down":
        out = Path(args.out).resolve()
        bed = Bed(
            BedConfig(
                Path(args.mock),
                Path(args.load),
                args.tokenizer or "",
                ports={"gateway": 0, "metrics": 0, "admin": 0, "mock_base": 0},
            ),
            Path(args.head),
            out / "bed",
        )
        bed.down()
        print("bed down")
        return 0
    settings = common_settings(args, need_base=False)
    bed = Bed(settings.bed, settings.binaries["pr"], settings.out / "bed")
    info = bed.up()
    print(
        f"bed up: gateway pid {info['gateway_pid']} at {info['gateway_url']}, {settings.bed.workers} workers, "
        f"bed.json {settings.out / 'bed' / 'bed' / 'bed.json'}"
    )
    if args.action == "up":
        print(f"stop it with: {sys.argv[0]} bed down --out {settings.out}")
        return 0
    rc = 0
    try:
        for spec in parse_paths(args.paths):
            sdir = settings.out / "smoke" / spec.name
            sdir.mkdir(parents=True, exist_ok=True)
            enqueue(bed, answers(spec, 16))
            summary = run_client(
                settings, bed, spec, 4, 1, sdir, extra=("--debug-first",), threads=2
            )
            counts = script_counts(bed)
            first = ""
            for line in (sdir / "client.err").read_text(errors="replace").splitlines():
                if "first" in line.lower() and "body" in line.lower():
                    first = line[:300]
            ok = (
                summary.get("ok", 0)
                and not summary.get("failed", 0)
                and not counts.get("misses")
                and not counts.get("unreadable")
            )
            rc |= 0 if ok else 1
            print(
                f"{'ok  ' if ok else 'FAIL'} {spec.name}: sent {summary.get('sent')}, ok {summary.get('ok')}, "
                f"failed {summary.get('failed')}, statuses {summary.get('statuses')}, "
                f"frames/req {summary.get('frames_per_request')}, "
                f"script hits {counts.get('hits')} misses {counts.get('misses')} "
                f"unreadable {counts.get('unreadable')}"
            )
            if first:
                print(f"     {first}")
    finally:
        bed.down()
    print("smoke " + ("passed" if rc == 0 else "FAILED"))
    return rc


def cmd_report(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    results = json.loads((out / "results.json").read_text())
    numbers = numbers_table(results, args.setup or results.get("note") or "see results.json")
    gate_md, rc = gate_report(results, Path(args.gates) if args.gates else None)
    print("## Numbers\n\n" + numbers + "\n\n" + gate_md)
    return rc


# ----------------------------------------------------------------------------- main


def add_bed_args(p: argparse.ArgumentParser, with_base: bool) -> None:
    p.add_argument(
        "--head",
        default=str(ROOT / "target/release/smg"),
        help="the gateway under test (the head) [target/release/smg]",
    )
    if with_base:
        p.add_argument(
            "--base",
            default="",
            help="the base branch's gateway binary (make perf-base builds one)",
        )
    p.add_argument(
        "--mock", default=str(ROOT / "target/release/mock-worker"), help="mock-worker binary"
    )
    p.add_argument(
        "--load", default=str(ROOT / "target/release/stress_load"), help="stress_load binary"
    )
    p.add_argument(
        "--tokenizer",
        default="",
        help="tokenizer directory or hub id [$TOKENIZER, else a synthetic one written under --out]",
    )
    p.add_argument(
        "--out",
        default=str(ROOT / "target/gateway-perf" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())),
        help="output directory [target/gateway-perf/<stamp>]",
    )
    p.add_argument("--workers", type=int, default=4, help="mock engines behind the gateway [4]")
    p.add_argument(
        "--port-base",
        type=int,
        default=0,
        help="first port of the bed's block (gateway; +1 metrics, +9 admin, +10.. workers) [a free block from 29700]",
    )
    p.add_argument("--gw-cores", default="", help="cpu list the gateway is pinned to [$GW_CORES]")
    p.add_argument(
        "--mock-cores", default="", help="cpu list the engines are pinned to [$MOCK_CORES]"
    )
    p.add_argument("--gen-cores", default="", help="cpu list the client is pinned to [$GEN_CORES]")
    p.add_argument("--gw-threads", type=int, default=8, help="gateway runtime worker threads [8]")
    p.add_argument(
        "--gw-args", default=DEFAULT_GW_ARGS, help=f"extra gateway flags [{DEFAULT_GW_ARGS}]"
    )
    p.add_argument(
        "--counters",
        choices=("auto", "on", "off"),
        default="auto",
        help="perf stat hardware counters [auto: when perf can count instructions]",
    )


def add_cell_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--paths", default=DEFAULT_PATHS, help=f"comma-separated paths [{DEFAULT_PATHS}]"
    )
    p.add_argument("--requests", type=int, default=10_000, help="requests per cell [10000]")
    p.add_argument("--repeats", type=int, default=3, help="interleaved repeats per side [3]")
    p.add_argument(
        "--rate-scale", type=float, default=1.0, help="multiply every path's offered rate [1.0]"
    )
    p.add_argument("--idle-seconds", type=int, default=25, help="the idle cell's window [25]")
    p.add_argument(
        "--prompt-tokens",
        type=int,
        default=64,
        help="prompt length in words, about one token each: a request-length class [64]",
    )
    p.add_argument(
        "--distinct", type=int, default=64, help="distinct prompts the client cycles through [64]"
    )
    p.add_argument(
        "--gates",
        default="",
        help="a bench_gate TOML (.github/bench-gates/gateway-perf.toml) for the verdict table [report only]",
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, doc in (
        ("cpu", "instructions, cycles and CPU time per request, base vs head"),
        (
            "alloc",
            "allocation calls and KiB per request, base vs head (jemalloc exit statistics; no perf)",
        ),
    ):
        p = sub.add_parser(name, help=doc)
        add_bed_args(p, with_base=True)
        add_cell_args(p)
        p.set_defaults(func=cmd_cells)
    p = sub.add_parser(
        "mem",
        help="RSS over windows on one gateway, growth per 10k requests, the peak after a burst, base vs head",
    )
    add_bed_args(p, with_base=True)
    add_cell_args(p)
    p.set_defaults(paths="stream", repeats=1, requests=5_000)
    p.add_argument(
        "--windows", type=int, default=5, help="windows of --requests on one gateway [5]"
    )
    p.add_argument(
        "--burst", type=float, default=2.0, help="the burst window's rate multiplier [2.0]"
    )
    p.set_defaults(func=cmd_mem)
    p = sub.add_parser(
        "live", help="an open-loop ramp with the capacity rule (head; base too when given)"
    )
    add_bed_args(p, with_base=True)
    p.add_argument("--path", default="chat", help="the path to ramp [chat]")
    p.add_argument(
        "--rates",
        default="200,400,800,1200,1600,2400,3200",
        help="offered rates, req/s [200,400,800,1200,1600,2400,3200]",
    )
    p.add_argument("--step-seconds", type=int, default=20, help="sampled seconds per step [20]")
    p.add_argument(
        "--step-warmup", type=int, default=3, help="driven, unsampled seconds before each step [3]"
    )
    p.add_argument("--slo-ttft-p99-ms", type=float, default=100.0)
    p.add_argument("--slo-e2e-p99-ms", type=float, default=1000.0)
    p.add_argument("--slo-error-rate", type=float, default=0.0)
    p.add_argument(
        "--slo-achieved-fraction",
        type=float,
        default=0.95,
        help="a step whose achieved rate falls under this share of the offered rate fails [0.95]",
    )
    p.add_argument("--requests", type=int, default=0, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_live)
    p = sub.add_parser("bed", help="bed up | down | smoke")
    p.add_argument("action", choices=("up", "down", "smoke"))
    add_bed_args(p, with_base=False)
    p.add_argument(
        "--paths", default=DEFAULT_PATHS, help=f"smoke: the paths to exercise [{DEFAULT_PATHS}]"
    )
    p.set_defaults(func=cmd_bed)
    p = sub.add_parser(
        "report", help="print the Numbers table and the gate verdicts of a results.json"
    )
    p.add_argument("--out", required=True, help="the run directory holding results.json")
    p.add_argument("--gates", default="", help="a bench_gate TOML for the verdicts [report only]")
    p.add_argument("--setup", default="", help="the setup column's text")
    p.set_defaults(func=cmd_report)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        return args.func(args)
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as err:
        print(f"gateway_perf: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
