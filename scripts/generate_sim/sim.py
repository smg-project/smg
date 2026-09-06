#!/usr/bin/env python3
"""Local /generate scale-simulation orchestrator for SMG.

Implements the harness from .claude/generate-scale-sim/01-design.md: build the
gateway, mock fleet, and load generator; launch K SMG replicas against mock
workers in `--engine realistic` mode; register every worker with every SMG; drive
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
import signal
import socket
import statistics
import subprocess
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
# Peer-facing severable proxies in front of each index replica (partition
# drill): replica j is reached by its peers via INDEX_PROXY_BASE + j.
INDEX_PROXY_BASE = 40200
# Severable proxies between the GATEWAYS and replica 0 (gateway-side partition
# drills): even-numbered gateways dial +0, odd-numbered dial +1, so a drill can
# sever all gateways or only half of them.
INDEX_GW_PROXY_BASE = 40300
# Gateway-to-gateway mesh (TreeSync) ports, one per SMG.
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
    # Buffered vs streamed body routing, per path/reason — the direct
    # verification of which request-body regime a leg actually ran in.
    "smg_router_request_body_path_total",
    # Remote-index lookups as the gateway sees them: outcome counters and
    # the latency histogram the read p50/p90/p99 rows are computed from.
    "smg_remote_index_",
)

BRANCH_RE = re.compile(r'branch="?([A-Za-z0-9_.-]+)"?')

# Every `index_service` key the harness forwards or acts on, and every
# top-level fault-drill key it implements. A profile (or scenario leg) that
# sets anything else in these shapes fails at run start: a silently ignored
# knob would produce a plausible comparison table with no independent
# variable — the worst failure mode a measurement harness can have.
SUPPORTED_INDEX_KEYS = {
    "replicas",
    "bridge",
    "inferred_ttl_secs",
    "default_capacity_blocks",
    "sweep_interval_secs",
    "event_ttl_secs",
    "apply_delay_stored_ms",
    "apply_delay_removed_ms",
    "partitionable",
    "deferred_replicas",
    # Gateways reach replica 0 through severable proxies (gateway_partition).
    "gateway_proxy",
    # Pull every live replica's state at the end of the run and record the
    # per-holder divergence between replicas (consistency audit).
    "dump_on_exit",
    # Forwarded to the service: peer anti-entropy period (0 disables).
    "anti_entropy_secs",
}
SUPPORTED_DRILLS = {
    "restart_smgs_at_secs",
    "kill_index_replica",
    "flap_index_replica",
    "hang_index_replica",
    "start_deferred_replica",
    "partition_drill",
    # Worker churn: deregister N workers from every gateway, then bring N
    # NEW workers (fresh ports) up and register them mid-run.
    "remove_workers_drill",
    "add_workers_drill",
    # One gateway restarted cold while the others keep serving.
    "restart_one_smg_drill",
    # Every replica killed and relaunched in turn (rolling restart).
    "rolling_replica_restart_drill",
    # Gateways lose their link to the index (all of them, or half).
    "gateway_partition_drill",
}
DRILL_KEY_RE = re.compile(r"(_drill|_replica|_at_secs)$")


# ---- small helpers ----------------------------------------------------------


def log(msg):
    print("==> " + msg, flush=True)


def epoch_ms():
    return int(time.time() * 1000)


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
        log(f"raised RLIMIT_NOFILE {soft} -> {target}")
    except (ValueError, OSError):
        log(f"WARN: could not raise RLIMIT_NOFILE beyond {soft}")


def flags_from(params):
    """dict -> CLI flags: {"decode_base_ms": 4.3} -> ["--decode-base-ms", "4.3"].

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


def validate_profile(profile):
    """Reject knobs the harness would silently ignore (see SUPPORTED_*)."""
    unknown_index = set(profile.get("index_service") or {}) - SUPPORTED_INDEX_KEYS
    unknown_drills = {
        key for key in profile if DRILL_KEY_RE.search(key) and key not in SUPPORTED_DRILLS
    }
    problems = []
    if unknown_index:
        problems.append(f"index_service keys {sorted(unknown_index)} are not implemented")
    if unknown_drills:
        problems.append(f"drill keys {sorted(unknown_drills)} are not implemented")
    index_cfg = profile.get("index_service") or {}
    if not index_cfg:
        needs_index = SUPPORTED_DRILLS - {
            "restart_smgs_at_secs",
            "remove_workers_drill",
            "add_workers_drill",
            "restart_one_smg_drill",
        }
        used = sorted(k for k in needs_index if profile.get(k))
        if used:
            problems.append(f"drills {used} need an index_service block")
    if profile.get("partition_drill") and not index_cfg.get("partitionable"):
        problems.append("partition_drill needs index_service.partitionable = true")
    if profile.get("gateway_partition_drill") and not index_cfg.get("gateway_proxy"):
        problems.append("gateway_partition_drill needs index_service.gateway_proxy = true")
    cfg = profile.get("restart_one_smg_drill") or {}
    if cfg and not 0 <= int(cfg.get("smg", 0)) < int(profile["smg_count"]):
        problems.append("restart_one_smg_drill.smg must be < smg_count")
    cfg = profile.get("remove_workers_drill") or {}
    if cfg and not 0 < int(cfg.get("count", 0)) < int(profile["workers_total"]):
        problems.append("remove_workers_drill.count must be in (0, workers_total)")
    if 0 in (index_cfg.get("deferred_replicas") or []):
        problems.append("replica 0 (the publish endpoint) cannot be deferred")
    if index_cfg:
        # A drill naming a replica outside the fleet (or a non-deferred one
        # for the deferred start) would fail only at drill time.
        replicas = int(index_cfg.get("replicas", 1))
        deferred = {int(r) for r in index_cfg.get("deferred_replicas") or []}
        for key in ("kill_index_replica", "flap_index_replica", "hang_index_replica"):
            cfg = profile.get(key) or {}
            if cfg and not 0 <= int(cfg.get("replica", 1)) < replicas:
                problems.append(f"{key}.replica must be < index_service.replicas ({replicas})")
        cfg = profile.get("start_deferred_replica") or {}
        if cfg and int(cfg.get("replica", -1)) not in deferred:
            problems.append(
                "start_deferred_replica.replica must be in index_service.deferred_replicas"
            )
    if problems:
        raise SystemExit("profile invalid; the drill would not fire: " + "; ".join(problems))


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
            # A SIGSTOPped child (hang drill) never handles SIGTERM.
            try:
                child["proc"].send_signal(signal.SIGCONT)
            except OSError:
                pass
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
    raise RuntimeError(f"{what} never became healthy at {url}")


def wait_tcp(port, timeout, what):
    """Liveness for listeners with no HTTP surface (gRPC workers)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=5):
                return
        except OSError:
            time.sleep(1)
    raise RuntimeError(f"{what} never accepted TCP on port {port}")


class TcpProxy:
    """A severable TCP forwarder: `listen_port` -> 127.0.0.1:`target_port`.

    The partition drill puts one in front of each index replica for its
    PEERS (never for gateways or the bridge): severing closes every relayed
    connection and refuses new ones, so the replicas cannot reach each other
    while every client still can — a true inter-replica network partition.
    """

    def __init__(self, listen_port, target_port):
        self.target_port = target_port
        self._open = threading.Event()
        self._open.set()
        self._conns = set()
        self._lock = threading.Lock()
        self._server = socket.create_server(("127.0.0.1", listen_port))
        # 0 = ephemeral (tests); read back the bound port either way.
        self.listen_port = self._server.getsockname()[1]
        self._server.settimeout(0.5)
        self._closed = False
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self):
        while not self._closed:
            try:
                client, _ = self._server.accept()
            except TimeoutError:
                # socket.timeout is a distinct class before Python 3.10.
                continue
            except OSError:
                return
            if not self._open.is_set():
                client.close()
                continue
            try:
                upstream = socket.create_connection(("127.0.0.1", self.target_port), timeout=5)
            except OSError:
                client.close()
                continue
            with self._lock:
                self._conns.update((client, upstream))
            for src, dst in ((client, upstream), (upstream, client)):
                threading.Thread(target=self._pump, args=(src, dst), daemon=True).start()

    def _pump(self, src, dst):
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
        except OSError:
            pass
        finally:
            self._drop(src, dst)

    def _drop(self, *socks):
        with self._lock:
            for s in socks:
                self._conns.discard(s)
        for s in socks:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                s.close()
            except OSError:
                pass

    def sever(self):
        """Refuse new connections and cut every relayed one."""
        self._open.clear()
        with self._lock:
            live = list(self._conns)
        self._drop(*live)

    def heal(self):
        self._open.set()

    def close(self):
        self._closed = True
        self.sever()
        try:
            self._server.close()
        except OSError:
            pass


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
    vocab = {f"t{i}": i for i in range(150_000)}
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
    log(f"generated local tokenizer: {path.name}")
    return path


# ---- run steps --------------------------------------------------------------


def build_binaries(target_dir, build_gateway, build_index):
    """Release build of the harness binaries. The index service package
    (`smg-radix-index`) is built only when the profile asks for an
    `index_service`: it is optional on this branch, and a profile that
    requests it on a tree without the crate fails here, loudly."""
    packages = ["mock-worker", "sim-loadgen"]
    if build_gateway:
        packages.append("smg")
    if build_index:
        packages.append("smg-radix-index")
    cmd = ["cargo", "build", "--release"]
    for pkg in packages:
        cmd += ["-p", pkg]
    log("building (release): " + " ".join(packages))
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(target_dir)
    env["RUSTC_WRAPPER"] = ""
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def launch_mocks(profile, logs_dir, mock_bin):
    total = int(profile["workers_total"])
    procs = int(profile["mock_processes"])
    grpc = profile.get("worker_mode", "http") == "grpc"
    port_flags = (
        ("--grpc-base-port", "--grpc-count") if grpc else ("--http-base-port", "--http-count")
    )
    per_proc = (total + procs - 1) // procs
    children = []
    started = 0
    for j in range(procs):
        count = min(per_proc, total - started)
        base = MOCK_BASE_PORT + started
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
        children.append(spawn(f"mock-{j}", cmd, logs_dir / f"mock-{j}.log"))
        started += count
    mode = "grpc" if grpc else "http"
    log(
        f"mock fleet: {total} {mode} workers over {procs} processes "
        f"(ports {MOCK_BASE_PORT}-{MOCK_BASE_PORT + total - 1})"
    )
    time.sleep(2)
    for child in children:
        if child["proc"].poll() is not None:
            raise RuntimeError("{} exited early; see its log".format(child["name"]))
    if grpc:
        wait_tcp(MOCK_BASE_PORT, 30, "mock fleet")
    else:
        wait_health(f"http://127.0.0.1:{MOCK_BASE_PORT}/health", 30, "mock fleet")
    return children


def index_replica_cmd(profile, index_bin, replica, bootstrap_from=None):
    """argv for index replica `replica`. Peers are every OTHER replica in the
    profile (deferred ones included: they join later and must already be in
    everyone's relay list), reached through the severable proxies when the
    profile is `partitionable`."""
    cfg = profile.get("index_service") or {}
    replicas = int(cfg.get("replicas", 1))
    peer_base = INDEX_PROXY_BASE if cfg.get("partitionable") else INDEX_BASE_PORT
    cmd = [
        str(index_bin),
        "--port",
        str(INDEX_BASE_PORT + replica),
        "--metrics-port",
        str(INDEX_METRICS_BASE + replica),
    ]
    peers = ",".join(f"http://127.0.0.1:{peer_base + j}" for j in range(replicas) if j != replica)
    if peers:
        cmd += ["--peers", peers]
    if bootstrap_from is not None:
        cmd += ["--bootstrap-from", f"http://127.0.0.1:{INDEX_BASE_PORT + bootstrap_from}"]
    for key in (
        "inferred_ttl_secs",
        "default_capacity_blocks",
        "sweep_interval_secs",
        "event_ttl_secs",
        "apply_delay_stored_ms",
        "apply_delay_removed_ms",
        "anti_entropy_secs",
    ):
        if key in cfg:
            cmd += ["--" + key.replace("_", "-"), str(cfg[key])]
    return cmd


def spawn_index_replica(profile, logs_dir, index_bin, replica, bootstrap_from=None, tag=""):
    env = dict(os.environ)
    env["RUST_LOG"] = "info"
    cmd = index_replica_cmd(profile, index_bin, replica, bootstrap_from)
    child = spawn(f"index-{replica}", cmd, logs_dir / f"index-{replica}{tag}.log", env=env)
    wait_tcp(INDEX_BASE_PORT + replica, 30, f"index-{replica}")
    return child


def launch_index_service(profile, logs_dir, index_bin, bridge_bin):
    """Optional radix index service (+ event bridge) from the profile's
    `index_service` block, e.g.
      {"replicas": 2, "bridge": true, "inferred_ttl_secs": 18,
       "sweep_interval_secs": 1, "default_capacity_blocks": N,
       "apply_delay_stored_ms": 0, "partitionable": false,
       "deferred_replicas": []}
    Replicas relay to each other; the bridge (when enabled) subscribes to
    every gRPC worker and publishes to replica 0. `partitionable` routes
    peer traffic through severable proxies (partition drill); replicas in
    `deferred_replicas` are listed as peers but launched later by the
    `start_deferred_replica` drill. Returns (children, proxies)."""
    cfg = profile.get("index_service")
    if not cfg:
        return [], []
    replicas = int(cfg.get("replicas", 1))
    deferred = {int(r) for r in cfg.get("deferred_replicas") or []}
    proxies = []
    if cfg.get("partitionable"):
        proxies = [TcpProxy(INDEX_PROXY_BASE + i, INDEX_BASE_PORT + i) for i in range(replicas)]
    if cfg.get("gateway_proxy"):
        # Two gateway-facing proxies to replica 0 (see INDEX_GW_PROXY_BASE);
        # launch_smgs points each gateway at one of them.
        proxies += [TcpProxy(INDEX_GW_PROXY_BASE + k, INDEX_BASE_PORT) for k in range(2)]
    children = []
    for i in range(replicas):
        if i in deferred:
            continue
        children.append(spawn_index_replica(profile, logs_dir, index_bin, i))
    bridged = 0
    if cfg.get("bridge", True):
        total = int(profile["workers_total"])
        grpc = profile.get("worker_mode", "http") == "grpc"
        grpc_workers = (
            [f"grpc://127.0.0.1:{port}" for port in range(MOCK_BASE_PORT, MOCK_BASE_PORT + total)]
            if grpc
            else []
        )
        if grpc_workers:
            env = dict(os.environ)
            env["RUST_LOG"] = "info"
            cmd = [
                str(bridge_bin),
                "--workers",
                ",".join(grpc_workers),
                "--index",
                f"http://127.0.0.1:{INDEX_BASE_PORT}",
                "--model",
                profile.get("model_id", "mock-model"),
                "--block-size",
                str(profile.get("mock", {}).get("block_size", 128)),
            ]
            children.append(spawn("bridge", cmd, logs_dir / "bridge.log", env=env))
            bridged = len(grpc_workers)
    log(
        f"index service: {replicas - len(deferred)}/{replicas} replicas on ports "
        f"{INDEX_BASE_PORT}.. (bridging {bridged} grpc workers"
        f"{', partitionable' if proxies else ''})"
    )
    return children, proxies


def smg_cmd(profile, smg_bin, i):
    """argv for gateway `i`. With `index_service.gateway_proxy`, the
    `--kv-indexer-url` flag is rewritten to this gateway's severable proxy
    (even gateways +0, odd +1) so a drill can cut all or half of the fleet
    off from the index."""
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
    if (profile.get("index_service") or {}).get("gateway_proxy") and "--kv-indexer-url" in cmd:
        cmd[cmd.index("--kv-indexer-url") + 1] = f"http://127.0.0.1:{INDEX_GW_PROXY_BASE + (i % 2)}"
    if profile.get("mesh_smgs"):
        # Gateway-to-gateway mesh (TreeSync of approximate-tree inserts):
        # per-instance port. The gateway uses only mesh_peer_urls[0], as
        # its ONE-SHOT gossip init peer (round 0 only), so every
        # instance bootstraps from smg-0 — whose port is bound first
        # and gated on below — rather than from a gateway this loop has
        # not spawned yet.
        peers = [f"127.0.0.1:{MESH_BASE_PORT}"] if i else []
        cmd += [
            "--enable-mesh",
            "--mesh-host",
            "127.0.0.1",
            "--mesh-advertise-host",
            "127.0.0.1",
            "--mesh-port",
            str(MESH_BASE_PORT + i),
            "--mesh-server-name",
            f"smg-{i}",
        ]
        if peers:
            cmd += ["--mesh-peer-urls"] + peers
    return cmd


def launch_one_smg(profile, logs_dir, smg_bin, i, tag=""):
    env = dict(os.environ)
    env["RUST_LOG"] = SMG_RUST_LOG
    child = spawn(f"smg-{i}", smg_cmd(profile, smg_bin, i), logs_dir / f"smg-{i}{tag}.log", env=env)
    if profile.get("mesh_smgs") and i == 0:
        # smg-0 must be listening before any peer's single init round.
        wait_tcp(MESH_BASE_PORT, 30, "smg-0 mesh")
    return child


def launch_smgs(profile, logs_dir, smg_bin):
    count = int(profile["smg_count"])
    children = [launch_one_smg(profile, logs_dir, smg_bin, i) for i in range(count)]
    log(f"gateways: {count} on ports {SMG_BASE_PORT}.. (prometheus {PROM_BASE_PORT}..)")
    for i in range(count):
        wait_health(f"http://127.0.0.1:{SMG_BASE_PORT + i}/health", 60, f"smg-{i}")
    return children


def register_workers(profile, worker_ports=None, smg_ports=None):
    """POST every worker URL to every SMG: 64-way per SMG, SMGs in parallel.

    Same WorkerSpec shape scale_test.sh proved at 2k ports; health disabled so
    workers are instantly routable and registration cost stays isolated from
    the health-probe loop. `worker_ports` / `smg_ports` narrow the fan-out
    (the add-workers and single-gateway-restart drills).
    """
    total = int(profile["workers_total"])
    model_id = profile.get("model_id", "mock-model")
    grpc = profile.get("worker_mode", "http") == "grpc"
    if worker_ports is None:
        worker_ports = list(range(MOCK_BASE_PORT, MOCK_BASE_PORT + total))
    if smg_ports is None:
        smg_ports = [SMG_BASE_PORT + i for i in range(int(profile["smg_count"]))]
    tokenizer_path = str(ensure_local_tokenizer()) if grpc else None

    def register_one(smg_port, worker_port):
        if grpc:
            # The URL scheme selects the connection mode; runtime picks the
            # proto dialect the mock implements. weight_version is relayed
            # verbatim in every response's meta_info — it is how the
            # loadgen learns which worker served a request (the gRPC
            # router exposes no other worker identity).
            body = {
                "url": f"grpc://127.0.0.1:{worker_port}",
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
                "url": f"http://127.0.0.1:{worker_port}",
                "connection_mode": "http",
                "runtime": "sglang",
                "models": [{"id": model_id}],
                "health": {"disable_health_check": True},
            }
        try:
            http_post_json(f"http://127.0.0.1:{smg_port}/workers", body, timeout=20)
            return True
        except OSError:
            return False

    def register_all(smg_port):
        ok = 0
        with ThreadPoolExecutor(max_workers=64) as pool:
            for success in pool.map(lambda p: register_one(smg_port, p), worker_ports):
                ok += 1 if success else 0
        return ok

    log(f"registering {len(worker_ports)} workers with {len(smg_ports)} SMGs via POST /workers")
    registered = {}
    with ThreadPoolExecutor(max_workers=len(smg_ports)) as pool:
        for port, ok in zip(smg_ports, pool.map(register_all, smg_ports)):
            registered[port] = ok
    for port in smg_ports:
        log(f"    smg :{port} accepted {registered[port]}/{len(worker_ports)}")
    return registered


def wait_ready(profile):
    """Gate on every SMG reporting >= readiness_fraction of the fleet.

    A timeout FAILS the run: a report over a fleet that was never fully
    routable would describe a topology that did not exist (imbalance
    divides by workers_total, cache hit depends on the reachable set).
    """
    total = int(profile["workers_total"])
    need = max(1, int(total * float(profile.get("readiness_fraction", 0.99))))
    timeout = float(profile.get("readiness_timeout_secs", 300))
    smg_ports = [SMG_BASE_PORT + i for i in range(int(profile["smg_count"]))]
    log(f"waiting for >= {need}/{total} workers per SMG (timeout {timeout:.0f}s)")
    counts = {port: 0 for port in smg_ports}
    deadline = time.time() + timeout
    while time.time() < deadline:
        for port in smg_ports:
            try:
                body = http_get(f"http://127.0.0.1:{port}/workers", timeout=60)
                counts[port] = body.count('"url"')
            except OSError:
                pass
        if all(n >= need for n in counts.values()):
            log("    ready: " + " ".join(f"{p}:{counts[p]}" for p in smg_ports))
            return counts
        time.sleep(2)
    seen = " ".join(f"{p}:{counts[p]}" for p in smg_ports)
    raise RuntimeError(f"readiness timeout: need >= {need}/{total} workers per SMG, got {seen}")


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
    proc_fd = f"/proc/{pid}/fd"
    if os.path.isdir(proc_fd):
        try:
            return len(os.listdir(proc_fd))
        except OSError:
            return None
    # Darwin: no /proc; lsof is accurate but slow at very high fd counts,
    # which is why the full profile sets sample_fds=false.
    out = subprocess.run(["lsof", "-p", str(pid)], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        return None
    return max(0, len(out.stdout.splitlines()) - 1)


def scrape_metrics(prom_port):
    try:
        body = http_get(f"http://127.0.0.1:{prom_port}/metrics", timeout=4)
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


def scrape_index_metrics(metrics_port):
    """radix_index_* lines from a replica's admin port (gauges, counters and
    the apply/query latency histograms)."""
    try:
        body = http_get(f"http://127.0.0.1:{metrics_port}/metrics", timeout=2)
    except OSError:
        return {}
    values = {}
    for line in body.splitlines():
        if line.startswith("#") or not line.startswith("radix_index_"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        try:
            values[parts[0]] = float(parts[1])
        except ValueError:
            pass
    return values


def sampler_loop(stop, smg_pids, out_path, interval, sample_fds, index_pids=None):
    """`index_pids`: callable returning [(replica, pid)] for the live index
    replicas (they change across kill/relaunch drills), or None."""
    start = time.time()
    with open(out_path, "a") as out:
        while True:
            index_live = index_pids() if index_pids else []
            stats = ps_stats(list(smg_pids) + [pid for _, pid in index_live])
            index_entries = []
            for replica, pid in index_live:
                rss, cpu = stats.get(pid, (None, None))
                index_entries.append(
                    {
                        "replica": replica,
                        "pid": pid,
                        "rss_kib": rss,
                        "cpu_pct": cpu,
                        "metrics": scrape_index_metrics(INDEX_METRICS_BASE + replica),
                    }
                )
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
                "index": index_entries,
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


def analyze_requests(path, workers_total):
    per_worker = Counter()
    per_turn_worker = {}
    turn_stats = {}
    session_turn_worker = {}
    index_sources = Counter()
    prediction_errors = []
    # Per-minute timeline of follow-up cache ratio and index outcomes, so a
    # drift over a long run (capacity, staleness, leaks) is visible as a
    # curve rather than averaged away.
    timeline = {}
    t0 = None
    if not path.exists():
        return {"error": "requests.jsonl missing"}
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
            port = rec.get("worker_port")
            if port is not None:
                per_worker[port] += 1
                per_turn_worker.setdefault(turn, Counter())[port] += 1
                session = rec.get("session")
                if session is not None:
                    session_turn_worker.setdefault(session, {})[turn] = port
            src = rec.get("index_source")
            if src:
                index_sources[src] += 1
                pred = rec.get("index_predicted_tokens")
                if pred is not None and cached is not None:
                    prediction_errors.append(int(pred) - int(cached))
            if start_ms is not None and turn >= 2 and rec.get("status") == 200 and prompt:
                minute = int(start_ms // 60_000)
                bucket = timeline.setdefault(
                    minute, {"n": 0, "prompt": 0, "cached": 0, "timeouts": 0, "disconnected": 0}
                )
                bucket["n"] += 1
                bucket["prompt"] += prompt
                bucket["cached"] += cached
                bucket["timeouts"] += src == "remote_timeout"
                bucket["disconnected"] += src == "remote_disconnected"
    both = [s for s in session_turn_worker.values() if 1 in s and 2 in s]
    same = sum(1 for s in both if s[1] == s[2])
    turns = {}
    for turn, ts in sorted(turn_stats.items()):
        turns[f"turn{turn}"] = {
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
        "index_sources": dict(index_sources),
        "index_prediction_error_tokens": prediction_error_summary(prediction_errors),
        # Minute-by-minute follow-up cache ratio (token-weighted) and index
        # outcome shares, keyed by minutes since the first request.
        "followup_timeline": [
            {
                "minute": minute - int(t0 // 60_000),
                "requests": b["n"],
                "cached_over_prompt": round(b["cached"] / b["prompt"], 4),
                "timeout_share": round(b["timeouts"] / b["n"], 4),
                "disconnected_share": round(b["disconnected"] / b["n"], 4),
            }
            for minute, b in sorted(timeline.items())
        ]
        if timeline and t0 is not None
        else [],
    }


def prediction_error_summary(errors):
    """predicted − actual cached tokens over every index-routed request:
    signed mean (bias), absolute p50/p90/p95/max, and the share of exact
    answers. An index that is wrong by a block on one request in twenty is
    a different thing from one wrong by 4k tokens on every request, and the
    p95 alone cannot tell them apart."""
    if not errors:
        return None
    abs_sorted = sorted(abs(e) for e in errors)
    n = len(abs_sorted)
    pick = lambda q: abs_sorted[min(n - 1, int(q * (n - 1)))]  # noqa: E731
    return {
        "requests": n,
        "mean": round(statistics.fmean(errors), 2),
        "exact_share": round(sum(1 for e in errors if e == 0) / n, 4),
        "p50_abs": pick(0.50),
        "p90_abs": pick(0.90),
        "p95_abs": pick(0.95),
        "max_abs": abs_sorted[-1],
    }


HIST_BUCKET_RE = re.compile(r"^(?P<name>[a-zA-Z_:]+)_bucket\{(?P<labels>[^}]*)\}$")


def histogram_deltas(first, last, name):
    """Cumulative bucket counts of Prometheus histogram `name` accumulated
    between two metric snapshots (dicts of metric-line -> value), summed over
    any extra labels: {upper_bound_seconds: count}, plus the sample count."""
    buckets = {}
    for key, val in last.items():
        m = HIST_BUCKET_RE.match(key)
        if not m or m.group("name") != name:
            continue
        le = None
        for label in m.group("labels").split(","):
            label = label.strip()
            if label.startswith("le="):
                le = label[3:].strip('"')
        if le is None:
            continue
        bound = float("inf") if le == "+Inf" else float(le)
        buckets[bound] = buckets.get(bound, 0.0) + val - first.get(key, 0.0)
    return buckets


def histogram_percentiles(buckets, quantiles=(0.5, 0.9, 0.99)):
    """Percentiles (seconds) from cumulative buckets by linear interpolation
    inside the bucket that crosses each quantile; the +Inf bucket reports
    its lower edge (the last finite bound) — a floor, never an estimate."""
    if not buckets:
        return {}
    bounds = sorted(buckets)
    total = buckets[bounds[-1]]
    if total <= 0:
        return {}
    out = {}
    for q in quantiles:
        target = q * total
        prev_bound, prev_count = 0.0, 0.0
        for bound in bounds:
            count = buckets[bound]
            if count >= target:
                if bound == float("inf"):
                    out[f"p{int(q * 100)}"] = prev_bound
                else:
                    span = count - prev_count
                    frac = (target - prev_count) / span if span > 0 else 1.0
                    out[f"p{int(q * 100)}"] = prev_bound + frac * (bound - prev_bound)
                break
            prev_bound, prev_count = bound, count
    return {k: round(v * 1000, 3) for k, v in out.items()}  # milliseconds


def summarize_index_samples(path, window=None):
    """Per-replica peak RSS / mean CPU inside `window`, and the service-side
    apply and query latency percentiles over the window (histogram deltas,
    summed across replicas); plus the gateway-side remote-index query
    latency percentiles (summed across gateways)."""
    per_replica = {}
    first_index, last_index = {}, {}
    first_gw, last_gw = {}, {}
    if not path.exists():
        return {}
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
            for entry in rec.get("index", []):
                r = entry.get("replica")
                s = per_replica.setdefault(r, {"rss_kib": [], "cpu_pct": []})
                if entry.get("rss_kib") is not None:
                    s["rss_kib"].append(entry["rss_kib"])
                if entry.get("cpu_pct") is not None:
                    s["cpu_pct"].append(entry["cpu_pct"])
                metrics = entry.get("metrics") or {}
                if metrics:
                    # Counters are monotonic per replica process; a relaunch
                    # resets them, so keep the first/last per replica and sum
                    # the deltas afterwards.
                    first_index.setdefault(r, metrics)
                    last_index[r] = metrics
            for entry in rec.get("smg", []):
                metrics = entry.get("metrics") or {}
                if any(k.startswith("smg_remote_index_query_duration_seconds") for k in metrics):
                    first_gw.setdefault(entry.get("idx"), metrics)
                    last_gw[entry.get("idx")] = metrics

    def merged(firsts, lasts, name):
        total = {}
        for key in lasts:
            for bound, count in histogram_deltas(firsts.get(key, {}), lasts[key], name).items():
                total[bound] = total.get(bound, 0.0) + count
        return total

    out = {
        "replicas": {
            str(r): {
                "rss_peak_kib": max(s["rss_kib"]) if s["rss_kib"] else None,
                "cpu_mean_pct": round(statistics.mean(s["cpu_pct"]), 1) if s["cpu_pct"] else None,
                "cpu_peak_pct": max(s["cpu_pct"]) if s["cpu_pct"] else None,
            }
            for r, s in sorted(per_replica.items())
        },
        "apply_ms": histogram_percentiles(
            merged(first_index, last_index, "radix_index_apply_duration_seconds")
        ),
        "query_ms": histogram_percentiles(
            merged(first_index, last_index, "radix_index_query_duration_seconds")
        ),
        "gateway_lookup_ms": histogram_percentiles(
            merged(first_gw, last_gw, "smg_remote_index_query_duration_seconds")
        ),
    }
    return out


def summarize_samples(path, smg_count, window=None):
    """Aggregate sampler records; with `window` = (start_s, end_s), only
    samples with `elapsed_s` inside it count. The sampler starts AFTER the
    warmup sleep, so `elapsed_s == 0` is already the start of the load
    period: the steady-state window is (0, duration_secs)."""
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
                            body_paths[idx]["{}:{}".format(*m.groups())] = val
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

    meta_path = run_dir / "meta.json"
    meta = load_profile(meta_path) if meta_path.exists() else {}

    per_smg_branches = []
    for i in range(smg_count):
        counts = branch_counts(run_dir / "logs" / f"smg-{i}.log")
        per_smg_branches.append({"idx": i, "branches": dict(counts)})

    report = {
        "profile": profile,
        "run_dir": str(run_dir),
        "loadgen_summary": loadgen_summary,
        "requests": analyze_requests(run_dir / "requests.jsonl", workers_total),
        "samples": summarize_samples(
            run_dir / "samples.jsonl",
            smg_count,
            window=(0, int(profile["duration_secs"])),
        ),
        "index_samples": summarize_index_samples(
            run_dir / "samples.jsonl",
            window=(0, int(profile["duration_secs"])),
        ),
        "index_divergence": meta.get("index_divergence"),
        "cache_aware_branches": per_smg_branches,
        # Drill outcomes ride into the report so a drill that never fired
        # (or failed half-way) cannot be mistaken for a measured effect.
        "drills": {
            k: v
            for k, v in meta.items()
            if k.startswith("index_")
            or k.startswith("workers_")
            or k.startswith("smg_")
            or k.startswith("gateway_")
            or "restart" in k
            or k == "loadgen_exit"
            or any(k.startswith(d) for d in SUPPORTED_DRILLS)
        },
    }
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    write_markdown(report, run_dir / "report.md")
    return report


def _fmt(val):
    if val is None:
        return "n/a"
    if isinstance(val, float):
        return f"{val:.4g}"
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
    lines.append("# generate-sim report — {}".format(profile.get("name", "run")))
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
        lines.append(f"| {key} | {_fmt(val)} |")
    lines.append("")

    drills = report.get("drills") or {}
    if drills:
        lines.append("## Drills (from meta.json)")
        lines.append("")
        lines.append("| key | value |")
        lines.append("|---|---|")
        for key in sorted(drills):
            lines.append(f"| {key} | {_fmt(drills[key])} |")
        lines.append("")

    summary = report.get("loadgen_summary", {})
    scalars = {k: v for k, v in summary.items() if isinstance(v, (int, float, str, bool))}
    if scalars:
        lines.append("## Loadgen summary")
        lines.append("")
        lines.append("| key | value |")
        lines.append("|---|---|")
        for key in sorted(scalars):
            lines.append(f"| {key} | {_fmt(scalars[key])} |")
        lines.append("")

    req = report.get("requests", {})
    if "overall_imbalance" in req:
        lines.append("## Worker balance (from requests.jsonl)")
        lines.append("")
        lines.append(
            "| slice | requests | distinct | CoV (fleet) | max/mean | cached/prompt | hit rate |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        imb = req["overall_imbalance"]
        lines.append(
            "| overall | {} | {}/{} | {} | {} | | |".format(
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
                "| {} | {} | {}/{} | {} | {} | {} | {} |".format(
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
            "turn-2 same-worker rate: {} (over {} two-turn sessions)".format(
                _fmt(req.get("t2_same_worker_rate")), _fmt(req.get("t2_sessions"))
            )
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
            "| smg | rss peak MiB | rss mean MiB | cpu mean % | cpu peak % | fds peak "
            "| queue peak | conns peak | rejected | selections |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")

        def mib(kib):
            return _fmt(round(kib / 1024, 1)) if kib is not None else "n/a"

        for s in samples:
            cells = [
                str(s["idx"]),
                mib(s["rss_kib"]["peak"]),
                mib(s["rss_kib"]["mean"]),
                _fmt(s["cpu_pct"]["mean"]),
                _fmt(s["cpu_pct"]["peak"]),
                _fmt(s["fds"]["peak"]),
                _fmt(s["queue_depth"]["peak"]),
                _fmt(s["http_connections_active"]["peak"]),
                _fmt(s["rejected_total"]),
                _fmt(s["worker_selection_total"]),
            ]
            lines.append("| " + " | ".join(cells) + " |")
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


class Drills:
    """The mid-run fault drills. Each runs on its own daemon thread, records
    what it did (epoch-ms timestamps) in `meta`, and on ANY failure records
    `<drill>_error` instead of dying silently — a drill that half-happened
    must be distinguishable from a measured blackout. `children` mutations
    are serialized with teardown through `lock`, and every step checks
    `stop` so a drill cannot relaunch anything into a run that is ending.
    """

    def __init__(
        self,
        profile,
        meta,
        children,
        lock,
        stop,
        logs_dir,
        index_bin,
        proxies,
        bins=None,
        smg_pids=None,
    ):
        self.profile = profile
        self.meta = meta
        self.children = children
        self.lock = lock
        self.stop = stop
        self.logs_dir = logs_dir
        self.index_bin = index_bin
        # Inter-replica proxies come first (one per replica when
        # `partitionable`), then the two gateway-facing ones (`gateway_proxy`).
        replicas = int((profile.get("index_service") or {}).get("replicas", 1))
        n_peer = replicas if (profile.get("index_service") or {}).get("partitionable") else 0
        self.proxies = proxies[:n_peer]
        self.gateway_proxies = proxies[n_peer:]
        self.bins = bins or {}
        self.smg_pids = smg_pids
        self._threads = []

    def start_all(self):
        for key, target in (
            ("kill_index_replica", self._kill_index_replica),
            ("flap_index_replica", self._flap_index_replica),
            ("hang_index_replica", self._hang_index_replica),
            ("start_deferred_replica", self._start_deferred_replica),
            ("partition_drill", self._partition),
            ("remove_workers_drill", self._remove_workers),
            ("add_workers_drill", self._add_workers),
            ("restart_one_smg_drill", self._restart_one_smg),
            ("rolling_replica_restart_drill", self._rolling_replica_restart),
            ("gateway_partition_drill", self._gateway_partition),
        ):
            cfg = self.profile.get(key)
            if cfg:
                thread = threading.Thread(
                    target=self._guarded, args=(key, target, cfg), daemon=True
                )
                thread.start()
                self._threads.append(thread)

    def join(self, timeout):
        """Let in-flight drills finish recording before meta is written: a
        drill past its last sleep (e.g. mid-relaunch) would otherwise write
        into a dict that has already been dumped, or make the dump raise."""
        deadline = time.time() + timeout
        for thread in self._threads:
            thread.join(timeout=max(0.0, deadline - time.time()))

    def _guarded(self, key, target, cfg):
        try:
            target(cfg)
        except Exception as e:  # noqa: BLE001 — the outcome IS the result
            log(f"WARN: drill {key} failed: {e!r}")
            self.meta[f"{key}_error"] = repr(e)

    def _sleep(self, secs):
        """Sleep unless the run is ending; returns False when it is."""
        return not self.stop.wait(float(secs))

    def _live_replica(self, replica):
        name = f"index-{replica}"
        with self.lock:
            return [c for c in self.children if c["name"] == name and c["proc"].poll() is None]

    def _kill(self, replica):
        victims = self._live_replica(replica)
        if not victims:
            raise RuntimeError(f"index-{replica} is not running; nothing to kill")
        for child in victims:
            child["proc"].kill()
        return epoch_ms()

    def _relaunch(self, replica, tag):
        # Bootstrap from the lowest live replica that is not this one.
        replicas = int(self.profile["index_service"].get("replicas", 1))
        source = next(
            (r for r in range(replicas) if r != replica and self._live_replica(r)),
            None,
        )
        child = spawn_index_replica(
            self.profile, self.logs_dir, self.index_bin, replica, bootstrap_from=source, tag=tag
        )
        with self.lock:
            if self.stop.is_set():
                teardown([child], [])
                raise RuntimeError("run ended during relaunch; replica torn down")
            self.children.append(child)
        return epoch_ms()

    def _kill_index_replica(self, cfg):
        replica = int(cfg.get("replica", 1))
        if not self._sleep(cfg.get("at_secs", 60)):
            return
        log(f"drill: killing index-{replica}")
        self.meta["index_killed_at_ms"] = self._kill(replica)
        self.meta["index_killed_replica"] = replica
        relaunch_after = cfg.get("relaunch_after_secs")
        if relaunch_after is not None and self._sleep(relaunch_after):
            log(f"drill: relaunching index-{replica}")
            self.meta["index_relaunched_at_ms"] = self._relaunch(replica, "-relaunch")

    def _flap_index_replica(self, cfg):
        replica = int(cfg.get("replica", 1))
        cycles = int(cfg.get("cycles", 3))
        period = float(cfg.get("period_secs", 20))
        if not self._sleep(cfg.get("at_secs", 45)):
            return
        events = self.meta.setdefault("index_flap_events", [])
        for cycle in range(cycles):
            log(f"drill: flap {cycle + 1}/{cycles} of index-{replica}")
            killed = self._kill(replica)
            if not self._sleep(period / 2):
                return
            relaunched = self._relaunch(replica, f"-flap{cycle + 1}")
            events.append({"killed_at_ms": killed, "relaunched_at_ms": relaunched})
            if cycle + 1 < cycles and not self._sleep(period / 2):
                return
        self.meta["index_flap_replica"] = replica

    def _hang_index_replica(self, cfg):
        replica = int(cfg.get("replica", 1))
        if not self._sleep(cfg.get("at_secs", 60)):
            return
        victims = self._live_replica(replica)
        if not victims:
            raise RuntimeError(f"index-{replica} is not running; nothing to hang")
        log(f"drill: SIGSTOP index-{replica} (TCP up, nothing drains)")
        for child in victims:
            child["proc"].send_signal(signal.SIGSTOP)
        self.meta["index_hung_at_ms"] = epoch_ms()
        self.meta["index_hung_replica"] = replica
        resume_after = cfg.get("resume_after_secs")
        if resume_after is not None and self._sleep(resume_after):
            log(f"drill: SIGCONT index-{replica}")
            for child in victims:
                child["proc"].send_signal(signal.SIGCONT)
            self.meta["index_resumed_at_ms"] = epoch_ms()

    def _start_deferred_replica(self, cfg):
        replica = int(cfg["replica"])
        deferred = {int(r) for r in self.profile["index_service"].get("deferred_replicas") or []}
        if replica not in deferred:
            raise RuntimeError(f"replica {replica} is not in index_service.deferred_replicas")
        if not self._sleep(cfg.get("at_secs", 60)):
            return
        log(f"drill: starting deferred index-{replica} (bootstrap under load)")
        self.meta["index_deferred_started_at_ms"] = self._relaunch(replica, "-deferred")
        self.meta["index_deferred_replica"] = replica

    # ---- worker churn --------------------------------------------------------

    def _smg_ports(self):
        return [SMG_BASE_PORT + i for i in range(int(self.profile["smg_count"]))]

    def _remove_workers(self, cfg):
        """Deregister the LAST `count` workers from every gateway (DELETE
        /workers/{id}, resolved from each gateway's own listing). The mock
        listeners stay up: this is the control-plane removal a drain or a
        scale-down performs, and the index must learn it through the
        gateways' lifecycle signals, not through the worker dying."""
        count = int(cfg["count"])
        total = int(self.profile["workers_total"])
        if not self._sleep(cfg.get("at_secs", 60)):
            return
        victims = {MOCK_BASE_PORT + total - 1 - k for k in range(count)}
        log(f"drill: removing {count} workers from every gateway")
        removed = {}
        for smg_port in self._smg_ports():
            listing = json.loads(http_get(f"http://127.0.0.1:{smg_port}/workers", timeout=30))
            workers = listing["workers"] if isinstance(listing, dict) else listing
            done = 0
            for w in workers:
                port = int(str(w.get("url", "")).rsplit(":", 1)[-1] or 0)
                if port in victims:
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{smg_port}/workers/{w['id']}", method="DELETE"
                    )
                    with urllib.request.urlopen(req, timeout=30):
                        done += 1
            removed[str(smg_port)] = done
        self.meta["workers_removed_at_ms"] = epoch_ms()
        self.meta["workers_removed"] = {"ports": sorted(victims), "per_gateway": removed}

    def _add_workers(self, cfg):
        """Bring `count` NEW workers up (a fresh mock process on ports past
        the fleet) and register them with every gateway mid-run."""
        count = int(cfg["count"])
        total = int(self.profile["workers_total"])
        if not self._sleep(cfg.get("at_secs", 60)):
            return
        base = MOCK_BASE_PORT + total
        grpc = self.profile.get("worker_mode", "http") == "grpc"
        port_flags = (
            ("--grpc-base-port", "--grpc-count") if grpc else ("--http-base-port", "--http-count")
        )
        cmd = [
            str(self.bins["mock"]),
            "--host",
            "127.0.0.1",
            port_flags[0],
            str(base),
            port_flags[1],
            str(count),
            "--model",
            self.profile.get("model_id", "mock-model"),
        ] + flags_from(self.profile.get("mock", {}))
        env = dict(os.environ)
        env["RUST_LOG"] = "info"
        log(f"drill: adding {count} new workers on ports {base}-{base + count - 1}")
        child = spawn("mock-added", cmd, self.logs_dir / "mock-added.log", env=env)
        wait_tcp(base, 30, "added workers")
        with self.lock:
            if self.stop.is_set():
                teardown([child], [])
                return
            self.children.append(child)
        registered = register_workers(self.profile, worker_ports=list(range(base, base + count)))
        self.meta["workers_added_at_ms"] = epoch_ms()
        self.meta["workers_added"] = {"ports": [base, base + count - 1], "per_gateway": registered}

    # ---- gateway churn -------------------------------------------------------

    def _restart_one_smg(self, cfg):
        """Kill ONE gateway and relaunch it cold (no local trees, no sticky
        pins); re-register every worker with it. The shared-index thesis in
        one drill: a cold gateway routes on the fleet's knowledge from its
        first request, or it does not."""
        idx = int(cfg.get("smg", 0))
        if not self._sleep(cfg.get("at_secs", 60)):
            return
        log(f"drill: restarting smg-{idx} cold")
        with self.lock:
            old = [
                c for c in self.children if c["name"] == f"smg-{idx}" and c["proc"].poll() is None
            ]
        for child in old:
            child["proc"].kill()
        self.meta["smg_restarted_at_ms"] = epoch_ms()
        child = launch_one_smg(self.profile, self.logs_dir, self.bins["smg"], idx, tag="-restart")
        with self.lock:
            if self.stop.is_set():
                teardown([child], [])
                raise RuntimeError("run ended during gateway restart")
            self.children.append(child)
        wait_health(f"http://127.0.0.1:{SMG_BASE_PORT + idx}/health", 60, f"smg-{idx}")
        register_workers(self.profile, smg_ports=[SMG_BASE_PORT + idx])
        if self.smg_pids is not None and idx < len(self.smg_pids):
            self.smg_pids[idx] = child["proc"].pid
        self.meta["smg_restarted"] = idx
        self.meta["smg_restart_ready_at_ms"] = epoch_ms()

    # ---- replica churn -------------------------------------------------------

    def _rolling_replica_restart(self, cfg):
        """Kill and relaunch every replica in turn, `gap_secs` apart, each
        bootstrapping from a live peer — the rolling-restart a deploy does."""
        replicas = int(self.profile["index_service"].get("replicas", 1))
        gap = float(cfg.get("gap_secs", 20))
        if not self._sleep(cfg.get("at_secs", 60)):
            return
        events = self.meta.setdefault("index_rolling_restart_events", [])
        for replica in range(replicas):
            log(f"drill: rolling restart — killing index-{replica}")
            killed = self._kill(replica)
            if not self._sleep(gap / 2):
                return
            relaunched = self._relaunch(replica, f"-rolling{replica}")
            events.append(
                {"replica": replica, "killed_at_ms": killed, "relaunched_at_ms": relaunched}
            )
            if replica + 1 < replicas and not self._sleep(gap / 2):
                return

    # ---- partitions ----------------------------------------------------------

    def _gateway_partition(self, cfg):
        """Sever the gateways' link to the index: `scope: all` cuts every
        gateway (even- and odd-numbered proxies), `scope: half` only the
        even-numbered ones — an asymmetric partition where half the fleet
        routes on the index and half falls open to load-only."""
        if not self.gateway_proxies:
            raise RuntimeError("gateway_partition_drill needs index_service.gateway_proxy")
        scope = cfg.get("scope", "all")
        targets = self.gateway_proxies if scope == "all" else self.gateway_proxies[:1]
        if not self._sleep(cfg.get("at_secs", 60)):
            return
        log(f"drill: severing gateway→index links ({scope})")
        for proxy in targets:
            proxy.sever()
        self.meta["gateway_partitioned_at_ms"] = epoch_ms()
        self.meta["gateway_partition_scope"] = scope
        heal_after = cfg.get("heal_after_secs")
        if heal_after is not None and self._sleep(heal_after):
            log("drill: healing gateway→index links")
            for proxy in targets:
                proxy.heal()
            self.meta["gateway_partition_healed_at_ms"] = epoch_ms()

    def _partition(self, cfg):
        if not self.proxies:
            raise RuntimeError("partition_drill needs index_service.partitionable")
        if not self._sleep(cfg.get("at_secs", 60)):
            return
        log("drill: severing every inter-replica link")
        for proxy in self.proxies:
            proxy.sever()
        self.meta["index_partitioned_at_ms"] = epoch_ms()
        heal_after = cfg.get("heal_after_secs")
        if heal_after is not None and self._sleep(heal_after):
            log("drill: healing the partition")
            for proxy in self.proxies:
                proxy.heal()
            self.meta["index_healed_at_ms"] = epoch_ms()


def dump_replicas(profile, run_dir, dump_bin, live_replicas):
    """Pull every live replica's state with radix-index-dump and compare
    their per-holder digests. Returns the divergence summary recorded in
    meta: holders whose block SETS differ between replicas, holders present
    on only some replicas, and the per-replica totals."""
    dumps = {}
    for replica in live_replicas:
        out = subprocess.run(
            [str(dump_bin), "--connect", f"http://127.0.0.1:{INDEX_BASE_PORT + replica}"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if out.returncode != 0:
            dumps[replica] = {"error": out.stderr.strip()[-400:]}
            continue
        try:
            dumps[replica] = json.loads(out.stdout)
        except ValueError as e:
            dumps[replica] = {"error": f"unparsable dump: {e}"}
        with open(run_dir / f"index-dump-{replica}.json", "w") as f:
            f.write(out.stdout)
    good = {r: d for r, d in dumps.items() if "holders" in d}
    summary = {
        "replicas": sorted(dumps),
        "errors": {str(r): d["error"] for r, d in dumps.items() if "error" in d},
        "total_blocks": {str(r): d.get("total_blocks") for r, d in good.items()},
    }
    if len(good) >= 2:
        base_r = min(good)
        base = good[base_r]["holders"]
        differing, only_in_one = [], []
        for r, d in good.items():
            if r == base_r:
                continue
            other = d["holders"]
            for key in set(base) | set(other):
                if key not in base or key not in other:
                    only_in_one.append(key)
                elif base[key]["digest"] != other[key]["digest"]:
                    differing.append(key)
        summary.update(
            {
                "holders_compared": len(base),
                "holders_differing": len(set(differing)),
                "holders_only_in_one": len(set(only_in_one)),
                "converged": not differing and not only_in_one,
            }
        )
    return summary


def run_profile(profile, run_dir, smg_bin=None, skip_build=False):
    """Full run: build -> mocks -> SMGs -> register -> loadgen -> report.

    Returns the run dir; report.json / report.md are inside it.
    """
    validate_profile(profile)
    run_dir = Path(run_dir)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Pin the target dir so build and binary resolution always agree
    # (see scale_test.sh); override by exporting CARGO_TARGET_DIR.
    target_dir = Path(os.environ.get("CARGO_TARGET_DIR") or REPO_ROOT / "target")
    wants_index = bool(profile.get("index_service"))
    if not skip_build:
        build_binaries(target_dir, build_gateway=smg_bin is None, build_index=wants_index)
    smg_bin = Path(smg_bin) if smg_bin else target_dir / "release" / "smg"
    mock_bin = target_dir / "release" / "mock-worker"
    loadgen_bin = target_dir / "release" / "sim-loadgen"
    index_bin = target_dir / "release" / "radix-index-service"
    bridge_bin = target_dir / "release" / "radix-index-bridge"
    dump_bin = target_dir / "release" / "radix-index-dump"
    required = [smg_bin, mock_bin, loadgen_bin]
    if wants_index:
        required += [index_bin, bridge_bin]
        if (profile.get("index_service") or {}).get("dump_on_exit"):
            required.append(dump_bin)
    for path in required:
        if not os.access(str(path), os.X_OK):
            raise SystemExit(f"binary missing: {path} (drop --skip-build?)")

    if profile.get("requires_large_linux_host"):
        log("NOTE: " + str(profile["requires_large_linux_host"]))

    with open(run_dir / "profile.json", "w") as f:
        json.dump(profile, f, indent=2)

    raise_nofile_limit()
    # Every binary the run may spawn, so a crashed run cannot leave a stale
    # index replica holding 40000/40100 for the next run to silently reuse.
    all_bins = [smg_bin, mock_bin, loadgen_bin, index_bin, bridge_bin]
    teardown([], all_bins)  # clear leftovers from prior runs so ports are free
    time.sleep(1)

    children = []
    children_lock = threading.Lock()
    proxies = []
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
        "profile_sha256": hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest(),
        "loadgen_seed": profile.get("loadgen", {}).get("seed"),
        "host": platform.platform(),
    }
    stop = threading.Event()
    sampler = None
    drills = None
    try:
        children += launch_mocks(profile, logs_dir, mock_bin)
        index_children, proxies = launch_index_service(profile, logs_dir, index_bin, bridge_bin)
        children += index_children
        children += launch_smgs(profile, logs_dir, smg_bin)
        meta["registered"] = register_workers(profile)
        meta["ready"] = wait_ready(profile)

        warmup = float(profile.get("warmup_secs", 10))
        log(f"warmup sleep {warmup:.0f}s")
        time.sleep(warmup)

        smg_pids = [c["proc"].pid for c in children if c["name"].startswith("smg-")]

        def live_index_pids():
            with children_lock:
                return [
                    (int(c["name"].split("-", 1)[1]), c["proc"].pid)
                    for c in children
                    if c["name"].startswith("index-") and c["proc"].poll() is None
                ]

        sampler = threading.Thread(
            target=sampler_loop,
            args=(
                stop,
                smg_pids,
                run_dir / "samples.jsonl",
                float(profile.get("sample_interval_secs", 5)),
                bool(profile.get("sample_fds", True)),
                live_index_pids if wants_index else None,
            ),
            daemon=True,
        )
        sampler.start()

        duration = int(profile["duration_secs"])
        smg_urls = ",".join(
            f"http://127.0.0.1:{SMG_BASE_PORT + i}" for i in range(int(profile["smg_count"]))
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
        log(f"loadgen: {duration}s run")
        loadgen = spawn("loadgen", cmd, logs_dir / "loadgen.log")
        with children_lock:
            children.append(loadgen)

        drills = Drills(
            profile,
            meta,
            children,
            children_lock,
            stop,
            logs_dir,
            index_bin,
            proxies,
            bins={"smg": smg_bin, "mock": mock_bin},
            smg_pids=smg_pids,
        )
        drills.start_all()

        # Optional mid-window gateway restart: sticky pins and hash
        # placements are process state, so affinity must rebuild from
        # scratch; requests during the blackout fail and count as errors.
        restart_at = profile.get("restart_smgs_at_secs")
        if restart_at:

            def _restart_smgs():
                if stop.wait(float(restart_at)):
                    return
                log("restarting all SMGs (sticky pins and placements lost)")
                meta["restart_attempted_at_secs"] = restart_at
                try:
                    with children_lock:
                        old = [c for c in children if c["name"].startswith("smg-")]
                    for child in old:
                        if child["proc"].poll() is None:
                            child["proc"].kill()
                    new_smgs = launch_smgs(profile, logs_dir, smg_bin)
                    with children_lock:
                        if stop.is_set():
                            teardown(new_smgs, [])
                            return
                        children.extend(new_smgs)
                    register_workers(profile)
                    # Re-gate readiness: registration only waits for the
                    # POSTs, not for readiness_fraction, and metrics over a
                    # partial relaunched fleet would read as measured.
                    meta["restart_ready"] = wait_ready(profile)
                    # The sampler reads this list each tick; swap in the new pids.
                    smg_pids[:] = [c["proc"].pid for c in new_smgs]
                    meta["restarted_at_secs"] = restart_at
                except Exception as e:  # noqa: BLE001 — the outcome IS the result
                    log(f"WARN: SMG restart drill failed: {e!r}")
                    meta["restart_error"] = repr(e)

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
        if drills is not None:
            drills.join(timeout=10)
        if wants_index and (profile.get("index_service") or {}).get("dump_on_exit"):
            # Consistency audit before the replicas go away: pull every live
            # replica and diff their per-holder block sets.
            with children_lock:
                live = sorted(
                    int(c["name"].split("-", 1)[1])
                    for c in children
                    if c["name"].startswith("index-") and c["proc"].poll() is None
                )
            try:
                meta["index_divergence"] = dump_replicas(profile, run_dir, dump_bin, live)
                log(f"index divergence: {meta['index_divergence']}")
            except Exception as e:  # noqa: BLE001 — the outcome IS the result
                meta["index_divergence"] = {"error": repr(e)}
        with children_lock:
            teardown(children, all_bins)
        for proxy in proxies:
            proxy.close()
        meta["finished_at"] = datetime.now().isoformat()
        with open(run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    build_report(run_dir)
    if meta.get("loadgen_exit") not in (0, None):
        # The loadgen exits non-zero only for run-invalidating conditions
        # (>50% errors, truncated requests.jsonl, timeout). The report is
        # still written for diagnosis, but the run must not read as measured.
        raise SystemExit(
            f"loadgen exited {meta['loadgen_exit']}; the run is not usable "
            f"(see {run_dir / 'logs' / 'loadgen.log'})"
        )
    log(f"report: {run_dir / 'report.md'}")
    return run_dir


def default_run_dir(profile, tag=None):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = profile.get("name", "run")
    if tag:
        name = f"{name}-{tag}"
    return REPO_ROOT / "target" / "generate-sim" / (f"{name}-{stamp}")


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
        log(f"report: {Path(args.run_dir) / 'report.md'}")


if __name__ == "__main__":
    main()
