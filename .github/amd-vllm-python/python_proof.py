"""Bind the original cases to owned Python servicer workers and native gateways."""

import hashlib
import json
import os
from pathlib import Path

import pytest
from reuse import require

NATIVE_HASHES = {}


def process_identity(process, title=None):
    require(process is not None and process.poll() is None, "owned process not alive")
    root = Path("/proc") / str(process.pid)
    fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    require(int(fields[1]) == os.getpid(), "process is not a child of this pytest harness")
    argv = [x.decode() for x in (root / "cmdline").read_bytes().split(b"\0") if x]
    launch = list(process.args)
    require(argv == launch or (title is not None and argv == [title]), "actual argv/title differs")
    marker = ("E2E_JOB_MARKER=" + os.environ["E2E_JOB_MARKER"]).encode()
    require(marker in (root / "environ").read_bytes().split(b"\0"), "process marker differs")
    return {
        "pid": process.pid,
        "parent_pid": int(fields[1]),
        "start_ticks": fields[19],
        "argv": argv,
        "launch_argv_intent": launch,
        "marker_visible_in_proc": True,
    }


def argument(argv, name):
    require(argv.count(name) == 1 and argv.index(name) + 1 < len(argv), "missing/duplicate " + name)
    return argv[argv.index(name) + 1]


def gateway_identity(gateway, native):
    result = process_identity(gateway.process, "smg::router")
    launch = result["launch_argv_intent"]
    require(argument(launch, "-m") == "smg.launch_router", "wrong gateway launcher")
    require("--backend" not in launch, "original gRPC auto backend changed")
    require(argument(launch, "--worker-urls").startswith("grpc://"), "gateway is not gRPC")
    mapped = set()
    for line in (Path("/proc") / str(result["pid"]) / "maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and "x" in fields[1] and Path(fields[5]).name.startswith("smg_rs"):
            mapped.add(Path(fields[5]).resolve())
    require(len(mapped) == 1, "missing/ambiguous actual gateway native")
    path = mapped.pop()
    info = path.stat()
    key = (str(path), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    if key not in NATIVE_HASHES:
        with path.open("rb") as stream:
            NATIVE_HASHES[key] = hashlib.file_digest(stream, "sha256").hexdigest()
    require(NATIVE_HASHES[key] == native["sha256"], "loaded gateway native differs")
    return {**result, "native": str(path), "native_sha256": NATIVE_HASHES[key]}


def worker_identity(worker, model, gateway):
    require(worker.engine == "vllm" and worker.mode.value == "grpc", "wrong original worker")
    require(worker.model_id == model, "actual worker model differs")
    require(
        argument(gateway["launch_argv_intent"], "--worker-urls") == worker.base_url,
        "gateway and original worker gRPC URL differ",
    )
    identity = process_identity(worker.process)
    argv = identity["argv"]
    require(argument(argv, "-m") == "vllm.entrypoints.grpc_server", "not vLLM gRPC launcher")
    require("--servicer-impl" not in argv, "original default servicer flag changed")
    require(
        argument(argv, "--model") == argument(gateway["launch_argv_intent"], "--model-path"),
        "gateway/worker model path differs",
    )
    path = Path(worker.log_path).resolve()
    require(
        (Path("/proc") / str(identity["pid"]) / "fd/1").resolve() == path,
        "worker stdout is not its original fixture log",
    )
    return {**identity, "log": str(path)}


def python_log(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(16 * 1024 * 1024 + 1)
    require(0 < len(raw) <= 16 * 1024 * 1024, "worker log cap")
    lines = raw.decode(errors="replace").splitlines()
    require(
        not any("Rust vLLM servicer listening on " in line for line in lines),
        "Rust listener observed in Python worker",
    )
    constructor = [
        line for line in lines if "VllmEngineServicer initialized (mm_processor=" in line
    ]
    listener = [line for line in lines if "vLLM gRPC server started on " in line]
    require(constructor and listener, "actual Python constructor/listener witness missing")
    return {
        "bytes_observed": len(raw),
        "sha256_observed": hashlib.sha256(raw).hexdigest(),
        "constructor": constructor,
        "listener": listener,
    }


class PythonProof:
    def __init__(self, output, recipe):
        self.output, self.recipe = output, recipe
        self.records, self.errors = [], []
        self.native = json.loads(Path(__file__).with_name("inputs.json").read_text())[
            "gateway_native"
        ]
        self.binding = json.loads((output.parent / "python-servicer-preflight.json").read_text())[
            "python_binding"
        ]

    def save(self):
        (self.output / "python-proof.json").write_text(
            json.dumps(
                {"binding": self.binding, "records": self.records, "errors": self.errors}, indent=2
            )
            + "\n"
        )

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        record = None
        backend = item.funcargs.get("setup_backend")
        if backend:
            try:
                from fixtures.markers import model_id_for_engine, resolve_class_marker
                from fixtures.mock_answers import _CLASS_WORKERS

                model = model_id_for_engine(resolve_class_marker(item, "model"), "vllm")
                if model not in {row["model"] for row in self.records}:
                    gateway = backend[3]
                    workers = list(_CLASS_WORKERS)
                    require(len(workers) == 1, "original single-worker topology changed")
                    identity = gateway_identity(gateway, self.native)
                    record = {
                        "nodeid": item.nodeid,
                        "model": model,
                        "gateway": identity,
                        "worker": worker_identity(workers[0], model, identity),
                    }
            except Exception as error:
                self.errors.append({"nodeid": item.nodeid, "stage": "before", "error": str(error)})
        yield
        if record is not None:
            try:
                require(
                    gateway_identity(gateway, self.native) == record["gateway"], "gateway changed"
                )
                require(
                    worker_identity(workers[0], model, record["gateway"]) == record["worker"],
                    "worker changed during the original case",
                )
                for source in self.binding["sources"]:
                    require(
                        hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest()
                        == source["sha256"],
                        "installed Python/launcher source changed",
                    )
                record["python_log"] = python_log(record["worker"]["log"])
                self.records.append(record)
            except Exception as error:
                self.errors.append({"nodeid": item.nodeid, "stage": "after", "error": str(error)})
        self.save()

    def verify(self):
        self.save()
        require(not self.errors, "Python worker observations incomplete: " + repr(self.errors))
        require(
            {row["model"] for row in self.records} == set(self.recipe["models"]),
            "actual Python witness missing for a model",
        )
