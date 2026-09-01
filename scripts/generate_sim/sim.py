#!/usr/bin/env python3
"""Local /generate scale-simulation orchestrator for SMG.

Implements the harness from .claude/generate-scale-sim/01-design.md: build the
gateway, mock fleet, and load generator; launch K SMG replicas against mock
workers in `--engine sim` mode; register every worker with every SMG; drive
the sim-loadgen workload; sample per-SMG resources and /metrics while it runs;
then merge loadgen output, samples, and per-SMG cache-aware branch logs into
report.json / report.md.

Stdlib only. macOS-first (BSD ps/lsof invocations) with Linux fallbacks
(/proc). Profiles are plain JSON (see profiles/); every workload unknown is a
profile or CLI knob, never hardcoded here.

Usage:
  scripts/generate_sim/sim.py run --profile scripts/generate_sim/profiles/local-small.json
  scripts/generate_sim/sim.py run --profile ... --skip-build --smg-bin /path/to/smg
  scripts/generate_sim/sim.py report --run-dir target/generate-sim/<run>
"""

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MOCK_BASE_PORT = 9000
SMG_BASE_PORT = 30000
PROM_BASE_PORT = 39000
INDEX_BASE_PORT = 40000
INDEX_METRICS_BASE = 40100
INDEX_PROXY_BASE = 40300

# Severable TCP proxies for inter-replica links: replica i reaches
# replica j through proxy port INDEX_PROXY_BASE + i*8 + j, so a drill
# can partition the pair (close live conns, refuse new ones) and heal
# it later without touching the processes. Protocol-agnostic byte
# pumps, so gRPC/HTTP2 rides through untouched.
INDEX_SEVERED_LINKS = set()  # {(i, j)}
INDEX_LINK_CONNS = {}  # (i, j) -> [socket, ...]


def _proxy_pump(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for sock in (src, dst):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def start_peer_proxy(i, j, real_port):
    lport = INDEX_PROXY_BASE + i * 8 + j
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", lport))
    srv.listen(16)
    INDEX_LINK_CONNS.setdefault((i, j), [])

    def _accept():
        while True:
            try:
                client, _ = srv.accept()
            except OSError:
                return
            if (i, j) in INDEX_SEVERED_LINKS:
                client.close()
                continue
            try:
                upstream = socket.create_connection(("127.0.0.1", real_port), timeout=5)
            except OSError:
                client.close()
                continue
            INDEX_LINK_CONNS[(i, j)].append(client)
            INDEX_LINK_CONNS[(i, j)].append(upstream)
            threading.Thread(target=_proxy_pump, args=(client, upstream), daemon=True).start()
            threading.Thread(target=_proxy_pump, args=(upstream, client), daemon=True).start()

    threading.Thread(target=_accept, daemon=True).start()
    return lport


def sever_index_links(pairs):
    for pair in pairs:
        INDEX_SEVERED_LINKS.add(tuple(pair))
        for sock in INDEX_LINK_CONNS.get(tuple(pair), []):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        INDEX_LINK_CONNS[tuple(pair)] = []


def heal_index_links(pairs):
    for pair in pairs:
        INDEX_SEVERED_LINKS.discard(tuple(pair))
MESH_BASE_PORT = 41000

# Cache-aware decision branches are DEBUG logs only (no branch metric), scoped
# so the rest of the gateway stays at warn.
SMG_RUST_LOG = "warn,smg::policies::cache_aware=debug"

METRIC_PREFIXES = (
    "smg_admission_queue_depth",
    "smg_http_connections_active",
    "smg_admission_queue_rejected_total",
    "smg_worker_selection_total",
    # With --routing-key-override, follow-up turns route through the sticky
    # pin, which shows up here (occupied_hit/occupied_miss/vacant/...) —
    # cache-aware debug lines then cover only delegated (turn-1) decisions.
    "smg_manual_policy_branch_total",
    "smg_routing_key_source_total",
    # Remote radix-index query outcomes + placement publishes
    # (--kv-indexer-url legs); deadline-miss and fallback rates come
    # straight from these.
    "smg_remote_index_query_total",
    "smg_remote_index_publish_total",
    # Buffered vs streamed body routing, per path/reason — the direct
    # verification of which request-body regime a leg actually ran in.
    "smg_router_request_body_path_total",
)

BRANCH_RE = re.compile(r'branch="?([A-Za-z0-9_.-]+)"?')


# ---- small helpers ----------------------------------------------------------


def log(msg):
    print("==> " + msg, flush=True)


def http_get(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def http_post_json(url, payload, timeout):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()


def raise_nofile_limit():
    # Thousands of mock ports + per-SMG upstream connections need plenty of
    # file descriptors; children inherit the raised limit.
    try:
        import resource
    except ImportError:
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = hard if hard != resource.RLIM_INFINITY else 1048576
    if soft >= target:
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        log("raised RLIMIT_NOFILE %d -> %d" % (soft, target))
    except (ValueError, OSError):
        log("WARN: could not raise RLIMIT_NOFILE beyond %d" % soft)


def flags_from(params):
    """dict -> CLI flags: {"sim_itl_ms": 4.3} -> ["--sim-itl-ms", "4.3"].

    Booleans are value-style ("--stream true"), matching mock-worker's
    --prefix-cache convention; None values are omitted.
    """
    flags = []
    for key, val in params.items():
        if val is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            flags += [flag, "true" if val else "false"]
        else:
            flags += [flag, str(val)]
    return flags


def load_profile(path):
    with open(path) as f:
        return json.load(f)


def apply_override(profile, dotted, value):
    """Set a dotted path ("loadgen.ingress") in the profile dict."""
    node = profile
    keys = dotted.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def parse_override_arg(raw):
    key, _, val = raw.partition("=")
    if not _:
        raise SystemExit("--override expects key=value, got: " + raw)
    try:
        return key, json.loads(val)
    except ValueError:
        return key, val


# ---- process management -----------------------------------------------------


def spawn(name, cmd, log_path, env=None):
    fh = open(log_path, "ab")
    # Children must never inherit our stdout: an unread pipe fills and blocks
    # the process (see scale_test.sh), so everything goes to per-process logs.
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return {"name": name, "proc": proc, "log": fh}


def teardown(children, bins):
    for child in reversed(children):
        if child["proc"].poll() is None:
            child["proc"].terminate()
    deadline = time.time() + 10
    for child in reversed(children):
        remaining = max(0.1, deadline - time.time())
        try:
            child["proc"].wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            child["proc"].kill()
        child["log"].close()
    children.clear()
    # Safety net for anything reparented or leaked from prior runs. Anchored to
    # the start of the command line so it matches only processes exec'd from
    # these binaries — not this orchestrator, whose own argv can mention the
    # smg path (--smg-bin), and not unrelated smg processes.
    for path in bins:
        if path:
            subprocess.run(
                ["pkill", "-9", "-f", "^" + str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )


def wait_health(url, timeout, what):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            http_get(url, timeout=5)
            return
        except OSError:
            time.sleep(1)
    raise RuntimeError("%s never became healthy at %s" % (what, url))


def wait_tcp(port, timeout, what):
    """Liveness for listeners with no HTTP surface (gRPC workers)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=5):
                return
        except OSError:
            time.sleep(1)
    raise RuntimeError("%s never accepted TCP on port %d" % (what, port))


def ensure_local_tokenizer():
    """Generate (once) a WordLevel tokenizer.json covering every token id
    the sim can produce (loadgen ids < 150k, sim outputs < 30k), so the
    gateway's gRPC pipeline can resolve and decode without any network
    fetch. Returned path goes into each gRPC worker's tokenizer_path label.
    """
    path = REPO_ROOT / "target" / "generate-sim" / "wordlevel-tokenizer.json"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    vocab = {"t%d" % i: i for i in range(150_000)}
    vocab["<unk>"] = 150_000
    tok = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
    }
    with open(path, "w") as f:
        json.dump(tok, f)
    log("generated local tokenizer: %s" % path.name)
    return path


# ---- run steps --------------------------------------------------------------


def build_binaries(target_dir, build_gateway):
    packages = ["mock-worker", "sim-loadgen", "radix-index"] + (
        ["smg"] if build_gateway else []
    )
    cmd = ["cargo", "build", "--release"]
    for pkg in packages:
        cmd += ["-p", pkg]
    log("building (release): " + " ".join(packages))
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(target_dir)
    env["RUSTC_WRAPPER"] = ""
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def worker_segments(profile):
    """Normalize `worker_mode` to [(mode, start_port, count)]. Accepts a
    plain "http" / "grpc" (whole fleet) or a list of
    {"mode": ..., "count": ...} segments summing to workers_total, so one
    fleet can mix gRPC-event and HTTP workers (per-worker feed selection
    is only measurable on a mixed fleet)."""
    total = int(profile["workers_total"])
    mode = profile.get("worker_mode", "http")
    if isinstance(mode, str):
        return [(mode, MOCK_BASE_PORT, total)]
    segments = []
    start = MOCK_BASE_PORT
    for seg in mode:
        count = int(seg["count"])
        segments.append((seg["mode"], start, count))
        start += count
    if sum(c for _, _, c in segments) != total:
        raise SystemExit("worker_mode segments must sum to workers_total")
    return segments


def launch_mocks(profile, logs_dir, mock_bin):
    total = int(profile["workers_total"])
    procs = int(profile["mock_processes"])
    segments = worker_segments(profile)
    children = []
    proc_idx = 0
    for mode, seg_start, seg_count in segments:
        # Processes split across segments proportionally, at least one.
        seg_procs = max(1, round(procs * seg_count / max(1, total)))
        per_proc = (seg_count + seg_procs - 1) // seg_procs
        port_flags = (
            ("--grpc-base-port", "--grpc-count")
            if mode == "grpc"
            else ("--http-base-port", "--http-count")
        )
        started = 0
        while started < seg_count:
            count = min(per_proc, seg_count - started)
            base = seg_start + started
            cmd = [
                str(mock_bin),
                "--host",
                "127.0.0.1",
                port_flags[0],
                str(base),
                port_flags[1],
                str(count),
                "--model",
                profile.get("model_id", "mock-model"),
            ] + flags_from(profile.get("mock", {}))
            children.append(
                spawn("mock-%d" % proc_idx, cmd, logs_dir / ("mock-%d.log" % proc_idx))
            )
            proc_idx += 1
            started += count
    log(
        "mock fleet: %d workers (%s) over %d processes (ports %d-%d)"
        % (
            total,
            ", ".join("%d %s" % (c, m) for m, _, c in segments),
            proc_idx,
            MOCK_BASE_PORT,
            MOCK_BASE_PORT + total - 1,
        )
    )
    time.sleep(2)
    for child in children:
        if child["proc"].poll() is not None:
            raise RuntimeError("%s exited early; see its log" % child["name"])
    for mode, seg_start, _ in segments:
        if mode == "grpc":
            wait_tcp(seg_start, 30, "mock fleet (grpc segment)")
        else:
            wait_health("http://127.0.0.1:%d/health" % seg_start, 30, "mock fleet")
    return children


def launch_index_service(profile, logs_dir, index_bin, bridge_bin):
    """Optional radix index service + event bridge, from the profile's
    `index_service` block:
      {"replicas": 2, "inferred_ttl_secs": 18, "default_capacity_blocks": N,
       "sweep_interval_secs": 1, "bridge": true}
    Replicas relay to each other (single-endpoint topology); the bridge
    subscribes to every gRPC worker segment and publishes to replica 0
    (the relay carries updates to the rest)."""
    cfg = profile.get("index_service")
    if not cfg:
        return []
    env = dict(os.environ)
    env["RUST_LOG"] = "info"
    replicas = int(cfg.get("replicas", 2))
    deferred = set(cfg.get("deferred_replicas", []))
    urls = ["http://127.0.0.1:%d" % (INDEX_BASE_PORT + i) for i in range(replicas)]
    partitionable = bool(cfg.get("partitionable"))
    children = []
    for i in range(replicas):
        if i in deferred:
            continue
        cmd = [str(index_bin), "--port", str(INDEX_BASE_PORT + i)]
        cmd += ["--metrics-port", str(INDEX_METRICS_BASE + i)]
        if partitionable:
            peer_urls = [
                "http://127.0.0.1:%d" % start_peer_proxy(i, j, INDEX_BASE_PORT + j)
                for j in range(replicas)
                if j != i
            ]
            peers = ",".join(peer_urls)
        else:
            peers = ",".join(u for j, u in enumerate(urls) if j != i)
        if peers:
            cmd += ["--peers", peers]
        for key in (
            "inferred_ttl_secs",
            "default_capacity_blocks",
            "sweep_interval_secs",
            "apply_delay_stored_ms",
            "apply_delay_removed_ms",
        ):
            if key in cfg:
                cmd += ["--" + key.replace("_", "-"), str(cfg[key])]
        children.append(
            spawn("index-%d" % i, cmd, logs_dir / ("index-%d.log" % i), env=env)
        )
    for i in range(replicas):
        if i in deferred:
            continue
        wait_tcp(INDEX_BASE_PORT + i, 30, "index-%d" % i)
    bridged = 0
    if cfg.get("bridge", True):
        grpc_workers = [
            "grpc://127.0.0.1:%d" % port
            for mode, seg_start, seg_count in worker_segments(profile)
            if mode == "grpc"
            for port in range(seg_start, seg_start + seg_count)
        ]
        if grpc_workers:
            cmd = [
                str(bridge_bin),
                "--workers",
                ",".join(grpc_workers),
                "--index",
                urls[0],
                "--model",
                profile.get("model_id", "mock-model"),
                "--block-size",
                str(profile.get("mock", {}).get("block_size", 128)),
            ]
            children.append(spawn("bridge", cmd, logs_dir / "bridge.log", env=env))
            bridged = len(grpc_workers)
    log(
        "index service: %d replicas on ports %d.. (bridging %d grpc workers)"
        % (replicas, INDEX_BASE_PORT, bridged)
    )
    return children


def launch_smgs(profile, logs_dir, smg_bin):
    count = int(profile["smg_count"])
    env = dict(os.environ)
    env["RUST_LOG"] = SMG_RUST_LOG
    children = []
    for i in range(count):
        cmd = [
            str(smg_bin),
            "--host",
            "127.0.0.1",
            "--port",
            str(SMG_BASE_PORT + i),
            "--prometheus-host",
            "127.0.0.1",
            "--prometheus-port",
            str(PROM_BASE_PORT + i),
        ] + list(profile["smg_flags"])
        if profile.get("mesh_smgs"):
            # Gateway-to-gateway mesh (TreeSync of approximate-tree inserts):
            # per-instance port, full peer list minus self.
            peers = [
                "127.0.0.1:%d" % (MESH_BASE_PORT + j) for j in range(count) if j != i
            ]
            cmd += [
                "--enable-mesh",
                "--mesh-host",
                "127.0.0.1",
                "--mesh-advertise-host",
                "127.0.0.1",
                "--mesh-port",
                str(MESH_BASE_PORT + i),
                "--mesh-server-name",
                "smg-%d" % i,
            ]
            if peers:
                cmd += ["--mesh-peer-urls"] + peers
        children.append(spawn("smg-%d" % i, cmd, logs_dir / ("smg-%d.log" % i), env=env))
    log("gateways: %d on ports %d.. (prometheus %d..)"
        % (count, SMG_BASE_PORT, PROM_BASE_PORT))
    for i in range(count):
        wait_health("http://127.0.0.1:%d/health" % (SMG_BASE_PORT + i), 60, "smg-%d" % i)
    return children


def register_workers(profile):
    """POST every worker URL to every SMG: 64-way per SMG, SMGs in parallel.

    Same WorkerSpec shape scale_test.sh proved at 2k ports; health disabled so
    workers are instantly routable and registration cost stays isolated from
    the health-probe loop.
    """
    total = int(profile["workers_total"])
    model_id = profile.get("model_id", "mock-model")
    segments = worker_segments(profile)
    mode_by_port = {}
    for mode, seg_start, seg_count in segments:
        for port in range(seg_start, seg_start + seg_count):
            mode_by_port[port] = mode
    any_grpc = any(mode == "grpc" for mode, _, _ in segments)
    smg_ports = [SMG_BASE_PORT + i for i in range(int(profile["smg_count"]))]
    tokenizer_path = str(ensure_local_tokenizer()) if any_grpc else None

    def register_one(smg_port, worker_port):
        if mode_by_port.get(worker_port) == "grpc":
            # The URL scheme selects the connection mode; runtime picks the
            # proto dialect the mock implements. weight_version is relayed
            # verbatim in every response's meta_info — it is how the
            # loadgen learns which worker served a request (the gRPC
            # router exposes no other worker identity).
            body = {
                "url": "grpc://127.0.0.1:%d" % worker_port,
                "connection_mode": "grpc",
                "runtime": "tokenspeed",
                "models": [{"id": model_id}],
                "kv_block_size": int(profile.get("mock", {}).get("block_size", 128)),
                "labels": {
                    "tokenizer_path": tokenizer_path,
                    "weight_version": str(worker_port),
                },
                "health": {"disable_health_check": True},
            }
        else:
            body = {
                "url": "http://127.0.0.1:%d" % worker_port,
                "connection_mode": "http",
                "runtime": "sglang",
                "models": [{"id": model_id}],
                "health": {"disable_health_check": True},
            }
        try:
            http_post_json("http://127.0.0.1:%d/workers" % smg_port, body, timeout=20)
            return True
        except OSError:
            return False

    def register_all(smg_port):
        ok = 0
        with ThreadPoolExecutor(max_workers=64) as pool:
            for success in pool.map(
                lambda p: register_one(smg_port, p),
                range(MOCK_BASE_PORT, MOCK_BASE_PORT + total),
            ):
                ok += 1 if success else 0
        return ok

    log("registering %d workers with %d SMGs via POST /workers" % (total, len(smg_ports)))
    registered = {}
    with ThreadPoolExecutor(max_workers=len(smg_ports)) as pool:
        for port, ok in zip(smg_ports, pool.map(register_all, smg_ports)):
            registered[port] = ok
    for port in smg_ports:
        log("    smg :%d accepted %d/%d" % (port, registered[port], total))
    return registered


def wait_ready(profile):
    total = int(profile["workers_total"])
    need = max(1, int(total * float(profile.get("readiness_fraction", 0.99))))
    timeout = float(profile.get("readiness_timeout_secs", 300))
    smg_ports = [SMG_BASE_PORT + i for i in range(int(profile["smg_count"]))]
    log("waiting for >= %d/%d workers per SMG (timeout %ds)" % (need, total, timeout))
    counts = {port: 0 for port in smg_ports}
    deadline = time.time() + timeout
    while time.time() < deadline:
        for port in smg_ports:
            try:
                body = http_get("http://127.0.0.1:%d/workers" % port, timeout=60)
                counts[port] = body.count('"url"')
            except OSError:
                pass
        if all(n >= need for n in counts.values()):
            log("    ready: " + " ".join("%d:%d" % (p, counts[p]) for p in smg_ports))
            return counts
        time.sleep(2)
    log("WARN: readiness timeout; continuing with "
        + " ".join("%d:%d" % (p, counts[p]) for p in smg_ports))
    return counts


# ---- sampling ---------------------------------------------------------------


def ps_stats(pids):
    out = subprocess.run(
        ["ps", "-o", "pid=,rss=,pcpu=", "-p", ",".join(str(p) for p in pids)],
        capture_output=True,
        text=True,
        check=False,
    )
    stats = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            try:
                stats[int(parts[0])] = (int(parts[1]), float(parts[2]))
            except ValueError:
                pass
    return stats


def fd_count(pid):
    proc_fd = "/proc/%d/fd" % pid
    if os.path.isdir(proc_fd):
        try:
            return len(os.listdir(proc_fd))
        except OSError:
            return None
    # Darwin: no /proc; lsof is accurate but slow at very high fd counts,
    # which is why the full profile sets sample_fds=false.
    out = subprocess.run(
        ["lsof", "-p", str(pid)], capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        return None
    return max(0, len(out.stdout.splitlines()) - 1)


def scrape_metrics(prom_port):
    try:
        body = http_get("http://127.0.0.1:%d/metrics" % prom_port, timeout=4)
    except OSError:
        return {}
    values = {}
    for line in body.splitlines():
        if line.startswith("#") or not line.startswith(METRIC_PREFIXES):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        try:
            values[parts[0]] = float(parts[1])
        except ValueError:
            pass
    return values


def sampler_loop(stop, smg_pids, out_path, interval, sample_fds):
    start = time.time()
    with open(out_path, "a") as out:
        while True:
            stats = ps_stats(smg_pids)
            entries = []
            for idx, pid in enumerate(smg_pids):
                rss, cpu = stats.get(pid, (None, None))
                entry = {
                    "idx": idx,
                    "pid": pid,
                    "rss_kib": rss,
                    "cpu_pct": cpu,
                    "fds": fd_count(pid) if sample_fds and rss is not None else None,
                    "metrics": scrape_metrics(PROM_BASE_PORT + idx),
                }
                entries.append(entry)
            record = {
                "ts": round(time.time(), 3),
                "elapsed_s": round(time.time() - start, 1),
                "smg": entries,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            if stop.wait(interval):
                return


# ---- report -----------------------------------------------------------------


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def branch_counts(log_path):
    counts = Counter()
    if not log_path.exists():
        return counts
    with open(log_path, errors="replace") as f:
        for line in f:
            if "Cache-aware selection" not in line:
                continue
            # tracing's fmt layer colors field names, leaving escape codes
            # between `branch` and `=`; strip them before matching.
            m = BRANCH_RE.search(ANSI_RE.sub("", line))
            if m:
                counts[m.group(1)] += 1
    return counts


def imbalance(counter, workers_total):
    if not counter:
        return {"requests": 0, "distinct_workers": 0}
    observed = list(counter.values())
    fleet = observed + [0] * max(0, workers_total - len(observed))
    mean = statistics.mean(fleet)
    return {
        "requests": sum(observed),
        "distinct_workers": len(observed),
        "workers_total": workers_total,
        "cov_fleet": round(statistics.pstdev(fleet) / mean, 4) if mean else None,
        "max_over_mean_fleet": round(max(fleet) / mean, 2) if mean else None,
        "cov_observed": (
            round(statistics.pstdev(observed) / statistics.mean(observed), 4)
            if statistics.mean(observed)
            else None
        ),
    }


def analyze_requests(path, workers_total, window=None):
    """Aggregate requests.jsonl. With `window` = (warmup_secs,
    duration_secs), only requests STARTED inside the steady-state window
    count — mirroring the loadgen summary's finish-time filter, so the
    turn/CoV rows here no longer mix warmup and the drain tail into
    otherwise-windowed compare tables."""
    per_worker = Counter()
    per_turn_worker = {}
    turn_stats = {}
    session_turn_worker = {}
    index_sources = Counter()
    prediction_errors = []
    if not path.exists():
        return {"error": "requests.jsonl missing"}
    records = []
    t0 = None
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            start_ms = rec.get("start_ms")
            if start_ms is not None:
                t0 = start_ms if t0 is None else min(t0, start_ms)
            records.append(rec)
    lo_ms, hi_ms = None, None
    if window and t0 is not None:
        lo_ms = t0 + float(window[0]) * 1000.0
        hi_ms = t0 + float(window[1]) * 1000.0
    dropped = 0
    for rec in records:
        if lo_ms is not None:
            start_ms = rec.get("start_ms")
            if start_ms is None or not (lo_ms <= start_ms < hi_ms):
                dropped += 1
                continue
        turn = int(rec.get("turn", 1))
        ts = turn_stats.setdefault(turn, {"n": 0, "prompt": 0, "cached": 0, "hits": 0})
        ts["n"] += 1
        prompt = rec.get("prompt_tokens") or 0
        cached = rec.get("cached_tokens") or 0
        ts["prompt"] += prompt
        ts["cached"] += cached
        # Request-level "hit" per the design doc: cached/prompt >= 0.3.
        if prompt and cached / prompt >= 0.3:
            ts["hits"] += 1
        source = rec.get("index_source")
        if source:
            index_sources[source] += 1
            predicted = rec.get("index_predicted_tokens")
            cached = rec.get("cached_tokens")
            if predicted is not None and cached is not None:
                prediction_errors.append(predicted - cached)
        port = rec.get("worker_port")
        if port is not None:
            per_worker[port] += 1
            per_turn_worker.setdefault(turn, Counter())[port] += 1
            session = rec.get("session")
            if session is not None:
                session_turn_worker.setdefault(session, {})[turn] = port
    both = [s for s in session_turn_worker.values() if 1 in s and 2 in s]
    same = sum(1 for s in both if s[1] == s[2])
    turns = {}
    for turn, ts in sorted(turn_stats.items()):
        turns["turn%d" % turn] = {
            "requests": ts["n"],
            # Raw sums so every ratio in the tables is verifiable.
            "prompt_tokens_sum": ts["prompt"],
            "cached_tokens_sum": ts["cached"],
            "cached_over_prompt": round(ts["cached"] / ts["prompt"], 4) if ts["prompt"] else None,
            "hit_rate": round(ts["hits"] / ts["n"], 4) if ts["n"] else None,
            "imbalance": imbalance(per_turn_worker.get(turn, Counter()), workers_total),
        }
    return {
        "overall_imbalance": imbalance(per_worker, workers_total),
        "turns": turns,
        "t2_sessions": len(both),
        "t2_same_worker_rate": round(same / len(both), 4) if both else None,
        "windowed": lo_ms is not None,
        "window_dropped_requests": dropped,
        # Remote-index echo (only on --kv-indexer-url legs): what each
        # decision resolved to, and predicted-vs-actual cached tokens —
        # the direct index-accuracy signal, separable from policy spill.
        "index_sources": dict(index_sources),
        "index_prediction_error_tokens": (
            {
                "n": len(prediction_errors),
                "mean": round(statistics.mean(prediction_errors), 1),
                "median": statistics.median(prediction_errors),
                "p95_abs": sorted(abs(e) for e in prediction_errors)[
                    int(len(prediction_errors) * 0.95)
                ],
            }
            if prediction_errors
            else None
        ),
    }


def summarize_samples(path, smg_count, window=None):
    """Aggregate sampler records; with `window` = (start_s, end_s), only
    samples inside the steady-state window count, so resource figures
    exclude warmup and the drain tail exactly like the loadgen stats."""
    series = [
        {"rss_kib": [], "cpu_pct": [], "fds": [], "queue_depth": [], "conns": []}
        for _ in range(smg_count)
    ]
    last_counters = [{"rejected": 0.0, "selections": 0.0} for _ in range(smg_count)]
    sticky_branches = [{} for _ in range(smg_count)]
    body_paths = [{} for _ in range(smg_count)]
    if not path.exists():
        return []
    with open(path, errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if window is not None:
                elapsed = rec.get("elapsed_s")
                if elapsed is None or not (window[0] <= elapsed <= window[1]):
                    continue
            for entry in rec.get("smg", []):
                idx = entry.get("idx")
                if idx is None or idx >= smg_count:
                    continue
                s = series[idx]
                if entry.get("rss_kib") is not None:
                    s["rss_kib"].append(entry["rss_kib"])
                if entry.get("cpu_pct") is not None:
                    s["cpu_pct"].append(entry["cpu_pct"])
                if entry.get("fds") is not None:
                    s["fds"].append(entry["fds"])
                metrics = entry.get("metrics", {})
                depth = conns = rejected = selections = 0.0
                for key, val in metrics.items():
                    if key.startswith("smg_admission_queue_depth"):
                        depth += val
                    elif key.startswith("smg_http_connections_active"):
                        conns += val
                    elif key.startswith("smg_admission_queue_rejected_total"):
                        rejected += val
                    elif key.startswith("smg_worker_selection_total"):
                        selections += val
                    elif key.startswith("smg_manual_policy_branch_total"):
                        m = BRANCH_RE.search(key)
                        if m:
                            sticky_branches[idx][m.group(1)] = val
                    elif key.startswith("smg_router_request_body_path_total"):
                        m = re.search(r'path="?(\w+)"?.*?reason="?(\w+)"?', key)
                        if m:
                            body_paths[idx]["%s:%s" % m.groups()] = val
                if metrics:
                    s["queue_depth"].append(depth)
                    s["conns"].append(conns)
                    last_counters[idx] = {"rejected": rejected, "selections": selections}

    def agg(values, as_int=True):
        if not values:
            return {"peak": None, "mean": None}
        mean = statistics.mean(values)
        return {
            "peak": max(values),
            "mean": int(mean) if as_int else round(mean, 1),
        }

    out = []
    for idx in range(smg_count):
        s = series[idx]
        out.append(
            {
                "idx": idx,
                "rss_kib": agg(s["rss_kib"]),
                "cpu_pct": agg(s["cpu_pct"], as_int=False),
                "fds": agg(s["fds"]),
                "queue_depth": agg(s["queue_depth"]),
                "http_connections_active": agg(s["conns"]),
                "rejected_total": last_counters[idx]["rejected"],
                "worker_selection_total": last_counters[idx]["selections"],
                # Final counter values: sticky-session outcomes under
                # --routing-key-override (occupied_hit = pinned follow-up).
                "sticky_branches": sticky_branches[idx],
                # Final "path:reason" counters — verifies buffered vs
                # streamed routing directly.
                "body_paths": body_paths[idx],
            }
        )
    return out


def build_report(run_dir):
    run_dir = Path(run_dir)
    profile = load_profile(run_dir / "profile.json")
    smg_count = int(profile["smg_count"])
    workers_total = int(profile["workers_total"])

    summary_path = run_dir / "summary.json"
    loadgen_summary = {}
    if summary_path.exists():
        loadgen_summary = load_profile(summary_path)

    per_smg_branches = []
    for i in range(smg_count):
        counts = branch_counts(run_dir / "logs" / ("smg-%d.log" % i))
        per_smg_branches.append({"idx": i, "branches": dict(counts)})

    report = {
        "profile": profile,
        "run_dir": str(run_dir),
        "loadgen_summary": loadgen_summary,
        "requests": analyze_requests(
            run_dir / "requests.jsonl",
            workers_total,
            window=(
                int(profile.get("warmup_secs", 0)),
                int(profile["duration_secs"]),
            ),
        ),
        "samples": summarize_samples(
        run_dir / "samples.jsonl",
        smg_count,
        window=(int(profile.get("warmup_secs", 0)), int(profile["duration_secs"])),
    ),
        "cache_aware_branches": per_smg_branches,
    }
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    write_markdown(report, run_dir / "report.md")
    return report


def _fmt(val):
    if val is None:
        return "n/a"
    if isinstance(val, float):
        return "%.4g" % val
    return str(val)


def write_markdown(report, path):
    profile = report["profile"]
    lines = []
    if int(profile.get("workers_total", 0)) < 4000:
        lines.append(
            "> **Reduced-scale run.** Cache/affinity semantics are "
            "meaningful; CPU/RSS/fd/connection figures are NOT "
            "production-representative (fleet size, body size, concurrency, "
            "sticky-map cardinality, and stream chunking are all scaled "
            "down). Use profiles/full.local.json on a large host for "
            "resource conclusions."
        )
        lines.append("")
    lines.append("# generate-sim report — %s" % profile.get("name", "run"))
    lines.append("")
    loadgen = profile.get("loadgen", {})
    derived_rps = None
    if "session_rps" in loadgen:
        derived_rps = float(loadgen["session_rps"]) * (1.0 + float(loadgen.get("t2_ratio", 0)))
    lines.append("| key | value |")
    lines.append("|---|---|")
    for key, val in [
        ("run_dir", report["run_dir"]),
        ("smg_count", profile.get("smg_count")),
        ("workers_total", profile.get("workers_total")),
        ("mock_processes", profile.get("mock_processes")),
        ("duration_secs", profile.get("duration_secs")),
        ("target aggregate rps", derived_rps),
        ("ingress", loadgen.get("ingress")),
        ("system_prefix_tokens", loadgen.get("system_prefix_tokens")),
        ("t2_ratio", loadgen.get("t2_ratio")),
    ]:
        lines.append("| %s | %s |" % (key, _fmt(val)))
    lines.append("")

    summary = report.get("loadgen_summary", {})
    scalars = {k: v for k, v in summary.items() if isinstance(v, (int, float, str, bool))}
    if scalars:
        lines.append("## Loadgen summary")
        lines.append("")
        lines.append("| key | value |")
        lines.append("|---|---|")
        for key in sorted(scalars):
            lines.append("| %s | %s |" % (key, _fmt(scalars[key])))
        lines.append("")

    req = report.get("requests", {})
    if "overall_imbalance" in req:
        lines.append("## Worker balance (from requests.jsonl)")
        lines.append("")
        lines.append("| slice | requests | distinct | CoV (fleet) | max/mean | cached/prompt | hit rate |")
        lines.append("|---|---|---|---|---|---|---|")
        imb = req["overall_imbalance"]
        lines.append(
            "| overall | %s | %s/%s | %s | %s | | |"
            % (
                _fmt(imb.get("requests")),
                _fmt(imb.get("distinct_workers")),
                _fmt(imb.get("workers_total")),
                _fmt(imb.get("cov_fleet")),
                _fmt(imb.get("max_over_mean_fleet")),
            )
        )
        for name, ts in sorted(req.get("turns", {}).items()):
            timb = ts.get("imbalance", {})
            lines.append(
                "| %s | %s | %s/%s | %s | %s | %s | %s |"
                % (
                    name,
                    _fmt(ts.get("requests")),
                    _fmt(timb.get("distinct_workers")),
                    _fmt(timb.get("workers_total")),
                    _fmt(timb.get("cov_fleet")),
                    _fmt(timb.get("max_over_mean_fleet")),
                    _fmt(ts.get("cached_over_prompt")),
                    _fmt(ts.get("hit_rate")),
                )
            )
        lines.append("")
        lines.append(
            "turn-2 same-worker rate: %s (over %s two-turn sessions)"
            % (_fmt(req.get("t2_same_worker_rate")), _fmt(req.get("t2_sessions")))
        )
        lines.append("")

    branches = report.get("cache_aware_branches", [])
    all_names = sorted({name for e in branches for name in e["branches"]})
    if all_names:
        lines.append("## Cache-aware branches (per SMG, from debug logs)")
        lines.append("")
        lines.append("| smg | " + " | ".join(all_names) + " | total |")
        lines.append("|---|" + "---|" * (len(all_names) + 1))
        for entry in branches:
            row = [str(entry["idx"])]
            row += [str(entry["branches"].get(name, 0)) for name in all_names]
            row.append(str(sum(entry["branches"].values())))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    samples = report.get("samples", [])
    if samples:
        lines.append("## Gateway resources (per SMG, 5 s samples)")
        lines.append("")
        lines.append(
            "| smg | rss peak MiB | rss mean MiB | cpu mean % | cpu peak % | fds peak | queue peak | conns peak | rejected | selections |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for s in samples:
            def mib(kib):
                return _fmt(round(kib / 1024, 1)) if kib is not None else "n/a"

            lines.append(
                "| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    s["idx"],
                    mib(s["rss_kib"]["peak"]),
                    mib(s["rss_kib"]["mean"]),
                    _fmt(s["cpu_pct"]["mean"]),
                    _fmt(s["cpu_pct"]["peak"]),
                    _fmt(s["fds"]["peak"]),
                    _fmt(s["queue_depth"]["peak"]),
                    _fmt(s["http_connections_active"]["peak"]),
                    _fmt(s["rejected_total"]),
                    _fmt(s["worker_selection_total"]),
                )
            )
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---- orchestration ----------------------------------------------------------


def _repo_relative(path):
    """Repo-relative rendering for provenance: committed artifacts must not
    carry absolute local paths."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return path.name


def _git(args):
    try:
        return (
            subprocess.run(
                ["git"] + args,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip()
            or None
        )
    except OSError:
        return None


def _sha256(path):
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def run_profile(profile, run_dir, smg_bin=None, skip_build=False):
    """Full run: build -> mocks -> SMGs -> register -> loadgen -> report.

    Returns the run dir; report.json / report.md are inside it.
    """
    run_dir = Path(run_dir)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Pin the target dir so build and binary resolution always agree
    # (see scale_test.sh); override by exporting CARGO_TARGET_DIR.
    target_dir = Path(os.environ.get("CARGO_TARGET_DIR") or REPO_ROOT / "target")
    if not skip_build:
        build_binaries(target_dir, build_gateway=smg_bin is None)
    smg_bin = Path(smg_bin) if smg_bin else target_dir / "release" / "smg"
    mock_bin = target_dir / "release" / "mock-worker"
    loadgen_bin = target_dir / "release" / "sim-loadgen"
    index_bin = target_dir / "release" / "radix-index-service"
    bridge_bin = target_dir / "release" / "radix-index-bridge"
    required = [smg_bin, mock_bin, loadgen_bin]
    if profile.get("index_service"):
        required += [index_bin, bridge_bin]
    for path in required:
        if not os.access(str(path), os.X_OK):
            raise SystemExit("binary missing: %s (drop --skip-build?)" % path)

    if profile.get("requires_large_linux_host"):
        log("NOTE: " + str(profile["requires_large_linux_host"]))

    with open(run_dir / "profile.json", "w") as f:
        json.dump(profile, f, indent=2)

    raise_nofile_limit()
    all_bins = [smg_bin, mock_bin, loadgen_bin, index_bin, bridge_bin]
    teardown([], all_bins)  # clear leftovers from prior runs so ports are free
    time.sleep(1)

    children = []
    meta = {
        "smg_bin": _repo_relative(smg_bin),
        "started_at": datetime.now().isoformat(),
        # Provenance: enough to reproduce or audit any table built from this
        # run — repo state, exact binaries, profile content, and seed.
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "binary_sha256": {
            "smg": _sha256(smg_bin),
            "mock-worker": _sha256(mock_bin),
            "sim-loadgen": _sha256(loadgen_bin),
            "radix-index-service": _sha256(index_bin) if index_bin.exists() else None,
            "radix-index-bridge": _sha256(bridge_bin) if bridge_bin.exists() else None,
        },
        "profile_sha256": hashlib.sha256(
            json.dumps(profile, sort_keys=True).encode()
        ).hexdigest(),
        "loadgen_seed": profile.get("loadgen", {}).get("seed"),
        "host": platform.platform(),
    }
    stop = threading.Event()
    sampler = None
    try:
        children += launch_mocks(profile, logs_dir, mock_bin)
        children += launch_index_service(profile, logs_dir, index_bin, bridge_bin)
        children += launch_smgs(profile, logs_dir, smg_bin)
        meta["registered"] = register_workers(profile)
        meta["ready"] = wait_ready(profile)

        warmup = float(profile.get("warmup_secs", 10))
        log("warmup sleep %ds" % warmup)
        time.sleep(warmup)

        smg_pids = [c["proc"].pid for c in children if c["name"].startswith("smg-")]
        sampler = threading.Thread(
            target=sampler_loop,
            args=(
                stop,
                smg_pids,
                run_dir / "samples.jsonl",
                float(profile.get("sample_interval_secs", 5)),
                bool(profile.get("sample_fds", True)),
            ),
            daemon=True,
        )
        sampler.start()

        duration = int(profile["duration_secs"])
        smg_urls = ",".join(
            "http://127.0.0.1:%d" % (SMG_BASE_PORT + i)
            for i in range(int(profile["smg_count"]))
        )
        cmd = [
            str(loadgen_bin),
            "--smg-urls",
            smg_urls,
            "--duration-secs",
            str(duration),
            "--out",
            str(run_dir),
        ] + flags_from(profile.get("loadgen", {}))
        log("loadgen: %ds run" % duration)
        loadgen = spawn("loadgen", cmd, logs_dir / "loadgen.log")
        children.append(loadgen)

        # Optional index-replica failover drill: kill replica N at t, and
        # (optionally) relaunch it after a gap; timestamps recorded into
        # meta so analysis can bin around the kill instant. The leg FAILS
        # if the kill was never observed.
        drill = profile.get("kill_index_replica")
        if drill:

            def _kill_index_replica():
                at = float(drill.get("at_secs", 60))
                replica = int(drill.get("replica", 1))
                time.sleep(at)
                name = "index-%d" % replica
                victims = [c for c in children if c["name"] == name and c["proc"].poll() is None]
                if not victims:
                    meta["index_kill_failed"] = name
                    return
                for child in victims:
                    child["proc"].kill()
                meta["index_killed_at_ms"] = int(time.time() * 1000)
                meta["index_killed_replica"] = replica
                relaunch_after = drill.get("relaunch_after_secs")
                if relaunch_after is not None:
                    time.sleep(float(relaunch_after))
                    cfg = profile.get("index_service", {})
                    urls = [
                        "http://127.0.0.1:%d" % (INDEX_BASE_PORT + i)
                        for i in range(int(cfg.get("replicas", 2)))
                    ]
                    cmd = [
                        str(index_bin),
                        "--port",
                        str(INDEX_BASE_PORT + replica),
                        "--bootstrap-from",
                        urls[0 if replica != 0 else 1],
                    ]
                    peers = ",".join(u for j, u in enumerate(urls) if j != replica)
                    if peers:
                        cmd += ["--peers", peers]
                    env = dict(os.environ)
                    env["RUST_LOG"] = "info"
                    children.append(
                        spawn(
                            "index-%d" % replica,
                            cmd,
                            logs_dir / ("index-%d-relaunch.log" % replica),
                            env=env,
                        )
                    )
                    meta["index_relaunched_at_ms"] = int(time.time() * 1000)

            threading.Thread(target=_kill_index_replica, daemon=True).start()

        # Partition drill: sever every inter-replica link both ways at
        # `at_secs`, heal after `heal_after_secs`. Requires
        # index_service.partitionable so peers ride the proxies.
        pdrill = profile.get("partition_drill")
        if pdrill:

            def _partition():
                cfg = profile.get("index_service", {})
                replicas = int(cfg.get("replicas", 2))
                pairs = [
                    (i, j)
                    for i in range(replicas)
                    for j in range(replicas)
                    if i != j
                ]
                time.sleep(float(pdrill.get("at_secs", 60)))
                sever_index_links(pairs)
                meta["index_partitioned_at_ms"] = int(time.time() * 1000)
                heal_after = pdrill.get("heal_after_secs")
                if heal_after is not None:
                    time.sleep(float(heal_after))
                    heal_index_links(pairs)
                    meta["index_healed_at_ms"] = int(time.time() * 1000)

            threading.Thread(target=_partition, daemon=True).start()

        # Hang drill: SIGSTOP a replica (wedged-but-connected peer —
        # TCP stays up, nothing drains) and SIGCONT it later.
        hdrill = profile.get("hang_index_replica")
        if hdrill:

            def _hang():
                import signal as _signal

                replica = int(hdrill.get("replica", 1))
                name = "index-%d" % replica
                time.sleep(float(hdrill.get("at_secs", 60)))
                victims = [
                    c for c in children if c["name"] == name and c["proc"].poll() is None
                ]
                if not victims:
                    meta["index_hang_failed"] = name
                    return
                for child in victims:
                    child["proc"].send_signal(_signal.SIGSTOP)
                meta["index_hung_at_ms"] = int(time.time() * 1000)
                resume = hdrill.get("resume_after_secs")
                if resume is not None:
                    time.sleep(float(resume))
                    for child in victims:
                        child["proc"].send_signal(_signal.SIGCONT)
                    meta["index_resumed_at_ms"] = int(time.time() * 1000)

            threading.Thread(target=_hang, daemon=True).start()

        # F5: start a DEFERRED replica mid-run (the k8s scale-up
        # shape: peers were configured for it from the start, so the
        # running replicas' relay reconnect loops pick it up the
        # moment it binds; it bootstraps from replica 0 first).
        sdrill = profile.get("start_deferred_replica")
        if sdrill:

            def _start_deferred():
                cfg = profile.get("index_service", {})
                replicas = int(cfg.get("replicas", 2))
                replica = int(sdrill.get("replica", replicas - 1))
                time.sleep(float(sdrill.get("at_secs", 60)))
                urls = [
                    "http://127.0.0.1:%d" % (INDEX_BASE_PORT + i)
                    for i in range(replicas)
                ]
                cmd = [
                    str(index_bin),
                    "--port",
                    str(INDEX_BASE_PORT + replica),
                    "--metrics-port",
                    str(INDEX_METRICS_BASE + replica),
                    "--bootstrap-from",
                    urls[0 if replica != 0 else 1],
                ]
                peers = ",".join(u for j, u in enumerate(urls) if j != replica)
                if peers:
                    cmd += ["--peers", peers]
                for key in ("inferred_ttl_secs", "default_capacity_blocks", "sweep_interval_secs"):
                    if key in cfg:
                        cmd += ["--" + key.replace("_", "-"), str(cfg[key])]
                env2 = dict(os.environ)
                env2["RUST_LOG"] = "info"
                children.append(
                    spawn(
                        "index-%d" % replica,
                        cmd,
                        logs_dir / ("index-%d-deferred.log" % replica),
                        env=env2,
                    )
                )
                meta["index_deferred_started_at_ms"] = int(time.time() * 1000)

            threading.Thread(target=_start_deferred, daemon=True).start()

        # F7: flap a replica — kill -> relaunch(bootstrap) repeatedly.
        fdrill = profile.get("flap_index_replica")
        if fdrill:

            def _flap():
                cfg = profile.get("index_service", {})
                replicas = int(cfg.get("replicas", 2))
                replica = int(fdrill.get("replica", 0))
                cycles = int(fdrill.get("cycles", 3))
                period = float(fdrill.get("period_secs", 20))
                time.sleep(float(fdrill.get("at_secs", 45)))
                urls = [
                    "http://127.0.0.1:%d" % (INDEX_BASE_PORT + i)
                    for i in range(replicas)
                ]
                flaps = []
                for cycle in range(cycles):
                    name = "index-%d" % replica
                    for child in children:
                        if child["name"] == name and child["proc"].poll() is None:
                            child["proc"].kill()
                    flaps.append({"killed_ms": int(time.time() * 1000)})
                    time.sleep(period / 2)
                    cmd = [
                        str(index_bin),
                        "--port",
                        str(INDEX_BASE_PORT + replica),
                        "--metrics-port",
                        str(INDEX_METRICS_BASE + replica),
                        "--bootstrap-from",
                        urls[0 if replica != 0 else 1],
                    ]
                    peers = ",".join(u for j, u in enumerate(urls) if j != replica)
                    if peers:
                        cmd += ["--peers", peers]
                    env2 = dict(os.environ)
                    env2["RUST_LOG"] = "info"
                    children.append(
                        spawn(
                            name,
                            cmd,
                            logs_dir / ("index-%d-flap%d.log" % (replica, cycle)),
                            env=env2,
                        )
                    )
                    flaps[-1]["relaunched_ms"] = int(time.time() * 1000)
                    time.sleep(period / 2)
                meta["index_flaps"] = flaps

            threading.Thread(target=_flap, daemon=True).start()

        # Per-replica admin-metrics timeline: applies/blocks/relay
        # drops every 2 s, so divergence during a fault and
        # reconvergence after it are measured, not asserted.
        if profile.get("index_service"):

            def _index_timeline():
                cfg = profile.get("index_service", {})
                replicas = int(cfg.get("replicas", 2))
                timeline = meta.setdefault("index_timeline", [])
                while loadgen["proc"].poll() is None:
                    now_ms = int(time.time() * 1000)
                    for i in range(replicas):
                        try:
                            body = http_get(
                                "http://127.0.0.1:%d/metrics" % (INDEX_METRICS_BASE + i),
                                timeout=2,
                            )
                        except Exception:
                            continue
                        row = {"t_ms": now_ms, "replica": i}
                        for line in body.splitlines():
                            if line.startswith("radix_index_applies_total "):
                                row["applies"] = float(line.split()[1])
                            elif line.startswith("radix_index_blocks "):
                                row["blocks"] = float(line.split()[1])
                            elif line.startswith("radix_index_relay_dropped_total "):
                                row["relay_dropped"] = float(line.split()[1])
                        timeline.append(row)
                    time.sleep(2)

            threading.Thread(target=_index_timeline, daemon=True).start()

        # Optional mid-window gateway restart: sticky pins and hash
        # placements are process state, so affinity must rebuild from
        # scratch; requests during the blackout fail and count as errors.
        restart_at = profile.get("restart_smgs_at_secs")
        if restart_at:

            def _restart_smgs():
                time.sleep(float(restart_at))
                log("restarting all SMGs (sticky pins and placements lost)")
                for child in [c for c in children if c["name"].startswith("smg-")]:
                    if child["proc"].poll() is None:
                        child["proc"].kill()
                new_smgs = launch_smgs(profile, logs_dir, smg_bin)
                children.extend(new_smgs)
                register_workers(profile)
                # The sampler reads this list each tick; swap in the new pids.
                smg_pids[:] = [c["proc"].pid for c in new_smgs]
                meta["restarted_at_secs"] = restart_at

            threading.Thread(target=_restart_smgs, daemon=True).start()
        try:
            meta["loadgen_exit"] = loadgen["proc"].wait(timeout=duration * 3 + 300)
        except subprocess.TimeoutExpired:
            log("WARN: loadgen overran; killing")
            loadgen["proc"].kill()
            meta["loadgen_exit"] = "timeout"
    finally:
        stop.set()
        if sampler is not None:
            sampler.join(timeout=30)
        teardown(children, all_bins)
        meta["finished_at"] = datetime.now().isoformat()
        with open(run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    build_report(run_dir)
    log("report: %s" % (run_dir / "report.md"))
    return run_dir


def default_run_dir(profile, tag=None):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = profile.get("name", "run")
    if tag:
        name = "%s-%s" % (name, tag)
    return REPO_ROOT / "target" / "generate-sim" / ("%s-%s" % (name, stamp))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run one profile end to end")
    run_p.add_argument("--profile", required=True, help="path to a profile JSON")
    run_p.add_argument("--skip-build", action="store_true", help="use existing binaries")
    run_p.add_argument(
        "--smg-bin",
        help="prebuilt gateway binary (e.g. from another checkout for policy A/B); "
        "skips building the smg package",
    )
    run_p.add_argument("--out", help="run directory (default target/generate-sim/<name>-<ts>)")
    run_p.add_argument("--tag", help="suffix for the default run dir name")
    run_p.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted profile override, e.g. loadgen.ingress=random (repeatable)",
    )

    report_p = sub.add_parser("report", help="rebuild report.json/report.md for a run dir")
    report_p.add_argument("--run-dir", required=True)

    args = parser.parse_args()
    if args.cmd == "run":
        profile = load_profile(args.profile)
        for raw in args.override:
            key, val = parse_override_arg(raw)
            apply_override(profile, key, val)
        run_dir = Path(args.out) if args.out else default_run_dir(profile, args.tag)
        run_profile(profile, run_dir, smg_bin=args.smg_bin, skip_build=args.skip_build)
    elif args.cmd == "report":
        build_report(args.run_dir)
        log("report: %s" % (Path(args.run_dir) / "report.md"))


if __name__ == "__main__":
    main()
