"""Observe the original fixture's native gateway and headless ZMQ handshake."""

import hashlib
import json
import os
import re
import urllib.parse
from pathlib import Path

import pytest
from metrics_activity import ActivityWindow
from rank_processes import rank_processes
from reuse import require

NATIVE_HASHES = {}


def process_identity(process, title=None):
    require(process is not None and process.poll() is None, "owned process not alive")
    root = Path("/proc") / str(process.pid)
    fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    require(int(fields[1]) == os.getpid(), "process is not a child of this pytest harness")
    argv = [x.decode() for x in (root / "cmdline").read_bytes().split(b"\0") if x]
    launch = list(process.args)
    overwritten = title is not None and argv == [title]
    require(overwritten or argv == launch, "actual process argv differs from harness launch")
    marker = ("E2E_JOB_MARKER=" + os.environ["E2E_JOB_MARKER"]).encode()
    visible = marker in (root / "environ").read_bytes().split(b"\0")
    require(visible, "process job marker is missing or differs")
    return {
        "pid": process.pid,
        "parent_pid": int(fields[1]),
        "start_ticks": fields[19],
        "argv": argv,
        "launch_argv": launch,
        "marker_visible_in_proc": visible,
        "ownership": "live fixture Popen child + kernel PPid + start ticks",
        "title_overwrote_initial_argv": overwritten,
    }


def argument(argv, name):
    require(argv.count(name) == 1 and argv.index(name) + 1 < len(argv), "missing/duplicate " + name)
    return argv[argv.index(name) + 1]


def witness(gateway, workers, model, native):
    from smg.serve import _zmq_handshake_port

    result = process_identity(gateway.process, "smg::router")
    argv = result["launch_argv"]
    require(argument(argv, "-m") == "smg.launch_router", "wrong original gateway launcher")
    require(argument(argv, "--backend") == "vllm", "actual gateway backend differs")
    i = argv.index("--worker-urls") + 1 if argv.count("--worker-urls") == 1 else 0
    require(i > 0, "missing/duplicate worker URLs")
    urls = []
    while i < len(argv) and not argv[i].startswith("--"):
        urls.append(argv[i])
        i += 1
    require(urls and all(x.startswith("ipc://") for x in urls), "gateway is not direct IPC ZMQ")
    require(
        set(urls) == {w.base_url for w in workers}, "actual fixture workers differ from gateway"
    )
    engines = []
    for worker in workers:
        require(worker.engine == "vllm" and worker.mode.value == "zmq", "wrong actual worker mode")
        require(worker.model_id == model, "actual worker model differs")
        engine = process_identity(worker.process)
        args = engine["argv"]
        require(
            "vllm.entrypoints.cli.main" in args and args.count("--headless") == 1,
            "not headless vLLM",
        )
        require(
            argument(args, "--data-parallel-size")
            == argument(args, "--data-parallel-size-local")
            == "2",
            "original DP2 changed",
        )
        require(argument(args, "--data-parallel-address") == "127.0.0.1", "nonlocal engine")
        require(
            argument(args, "--data-parallel-rpc-port") == str(_zmq_handshake_port(worker.base_url)),
            "engine handshake differs",
        )
        require(
            args.count("serve") == 1
            and args[args.index("serve") + 1] == argument(argv, "--model-path"),
            "gateway/engine model path differs",
        )
        require(worker.gpu_ids == [0, 1], "original two-VF worker slice changed")
        engines.append({**engine, "ipc_url": worker.base_url})
    require(len(engines) == len(urls) == 1, "original single-engine topology changed")
    mapped = set()
    for line in (Path("/proc") / str(result["pid"]) / "maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and "x" in fields[1] and Path(fields[5]).name.startswith("smg_rs"):
            mapped.add(Path(fields[5]).resolve())
    require(len(mapped) == 1, "missing/ambiguous loaded gateway native")
    path = mapped.pop()
    st = path.stat()
    key = (str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if key not in NATIVE_HASHES:
        with path.open("rb") as stream:
            NATIVE_HASHES[key] = hashlib.file_digest(stream, "sha256").hexdigest()
    require(
        st.st_size == native["bytes"] and NATIVE_HASHES[key] == native["sha256"],
        "loaded native differs from producer",
    )
    require(gateway.log_path is not None, "gateway log was not captured")
    with Path(gateway.log_path).open("rb") as stream:
        raw = stream.read(16 * 1024 * 1024 + 1)
    require(0 < len(raw) <= 16 * 1024 * 1024, "gateway log cap")
    lines = re.sub(r"\x1b\[[0-9;]*m", "", raw.decode(errors="replace")).splitlines()
    bindings = [
        line
        for line in lines
        if "Binding ZMQ client for worker " + urls[0] in line and "engines=2" in line
    ]
    ready = [
        line
        for line in lines
        if "engines connected" in line and re.search(r"engine_count\s*=\s*2(?:\D|$)", line)
    ]
    require(bindings and ready, "real ZMQ DP2 READY/input-registration witness missing")
    rank_lines = {}
    for stage in ["received READY from", "received input registration from engine"]:
        observed = {}
        for line in lines:
            match = re.search(r"engine_id\s*=\s*EngineId\(([0-9a-f]{4})\)", line)
            if stage in line and match:
                observed[int.from_bytes(bytes.fromhex(match[1]), "little")] = line
        require(set(observed) == {0, 1}, "missing actual two-rank " + stage)
        rank_lines[stage] = observed
    rank_pids = rank_processes(workers[0].process.pid)
    metrics = urllib.parse.urlsplit(gateway.metrics_url)
    require(argument(argv, "--prometheus-port") == str(metrics.port), "metrics listener differs")
    stdout = (Path("/proc") / str(result["pid"]) / "fd/1").stat()
    logged = Path(gateway.log_path).stat()
    require(
        (stdout.st_dev, stdout.st_ino) == (logged.st_dev, logged.st_ino),
        "gateway stdout/log differs",
    )
    return {
        **result,
        "engines": engines,
        "native_path": str(path),
        "native_sha256": NATIVE_HASHES[key],
        "log": gateway.log_path,
        "binding_lines": bindings,
        "completed_handshake_lines": ready,
        "rank_handshake_lines": rank_lines,
        "rank_processes": rank_pids,
        "rank_identities": [
            [r[k] for k in ("rank", "pid", "parent_pid", "start_ticks", "argv")] for r in rank_pids
        ],
    }


class ZmqProof:
    def __init__(self, output, recipe):
        self.output, self.recipe = output, recipe
        self.records, self.errors, self.activity = [], [], {}
        self.native = json.loads(Path(__file__).with_name("inputs.json").read_text())["zmq_native"]

    def save(self):
        (self.output / "zmq-proof.json").write_text(
            json.dumps(
                {
                    "records": self.records,
                    "errors": self.errors,
                    "activity": [
                        {**s, "ranks": sorted(s["ranks"])} for s in self.activity.values()
                    ],
                    "activity_scope": (
                        "existing rank running-request gauges during original calls; "
                        "not per-request IDs or token-completion counters"
                    ),
                },
                indent=2,
            )
            + "\n"
        )

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        backend = item.funcargs.get("setup_backend")
        record, window = None, None
        if backend:
            try:
                from fixtures.markers import model_id_for_engine, resolve_class_marker
                from fixtures.mock_answers import _CLASS_WORKERS

                model = model_id_for_engine(resolve_class_marker(item, "model"), "vllm")
                gateway, workers = backend[3], list(_CLASS_WORKERS)
                record = {
                    "nodeid": item.nodeid,
                    "model": model,
                    "before": witness(gateway, workers, model, self.native),
                }
                # Stop extra sampling after one original gateway/model proves both ranks.
                if not any(s["ranks"] == {0, 1} for s in self.activity.values()):
                    key = (record["before"]["pid"], record["before"]["start_ticks"], model)
                    state = self.activity.setdefault(
                        key,
                        {
                            "model": model,
                            "gateway_pid": key[0],
                            "gateway_start_ticks": key[1],
                            "worker": workers[0].base_url,
                            "ranks": set(),
                            "records": [],
                            "errors": [],
                            "error_count": 0,
                        },
                    )
                    window = ActivityWindow(
                        gateway.metrics_url,
                        workers[0].base_url,
                        model,
                        item.nodeid,
                        self.output,
                        state,
                    )
                    window.__enter__()
            except Exception as error:
                self.errors.append({"nodeid": item.nodeid, "stage": "before", "error": str(error)})
        try:
            yield
        finally:
            if window is not None:
                try:
                    window.__exit__()
                except Exception as error:
                    self.errors.append(
                        {"nodeid": item.nodeid, "stage": "activity", "error": str(error)}
                    )
        if record is not None:
            try:
                record["after"] = witness(gateway, workers, model, self.native)
                require(
                    all(
                        record["before"][key] == record["after"][key]
                        for key in (
                            "pid",
                            "parent_pid",
                            "launch_argv",
                            "start_ticks",
                            "argv",
                            "engines",
                            "native_path",
                            "native_sha256",
                            "rank_identities",
                        )
                    ),
                    "ZMQ process identity changed during original case",
                )
                self.records.append(record)
            except Exception as error:
                self.errors.append({"nodeid": item.nodeid, "stage": "after", "error": str(error)})
        self.save()

    def verify(self):
        self.save()
        require(not self.errors, "incomplete ZMQ observations: " + repr(self.errors))
        require(
            {row["model"] for row in self.records} == set(self.recipe["models"]),
            "actual ZMQ witness missing for a model",
        )

        require(
            any(s["ranks"] == {0, 1} for s in self.activity.values()),
            "original cases did not prove actual request activity on both DP ranks",
        )
