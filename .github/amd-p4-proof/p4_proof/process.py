"""Bind the original cases to owned Python servicer workers and native gateways."""

import hashlib
import os
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


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
    require(" (deleted)" not in str(path), "deleted native mapping")
    info = path.stat()
    require(info.st_size == native["bytes"], "loaded native size differs")
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
    require(worker.gpu_ids == [0, 1, 2, 3], "original four logical devices changed")
    require(argument(argv, "--tensor-parallel-size") == "4", "actual TP4 flag missing")
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
