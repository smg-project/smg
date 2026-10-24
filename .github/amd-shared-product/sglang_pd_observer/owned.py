"""Observe actual SG fixture-owned PID/native identities; no engine import/canary."""

import hashlib
import os
from pathlib import Path

NATIVE_HASHES = {}

def require(condition, message):
    if not condition:
        raise ValueError(message)


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


def mapped_native(pid, words):
    found = {}
    for line in (Path("/proc") / str(pid) / "maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or "x" not in fields[1] or not any(word in fields[5] for word in words):
            continue
        require(not fields[5].endswith(" (deleted)"), "deleted native mapping")
        path = Path(fields[5]).resolve()
        info = path.stat()
        require(0 < info.st_size <= 256 * 1024**2, "native byte cap")
        key = (str(path), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        if key not in NATIVE_HASHES:
            with path.open("rb") as stream:
                NATIVE_HASHES[key] = hashlib.file_digest(stream, "sha256").hexdigest()
        found[str(path)] = {"bytes": info.st_size, "sha256": NATIVE_HASHES[key]}
    return found


def transport_mappings(process):
    """Read only the launcher's descendants; markers bind accepted mappings."""
    if process is None or process.poll() is not None:
        return []
    parents = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            parents[int(path.parent.name)] = (int(fields[1]), fields[19])
        except (OSError, ValueError):
            continue
    selected = {process.pid}
    while True:
        expanded = selected | {pid for pid, (parent, _) in parents.items() if parent in selected}
        if selected == expanded:
            break
        selected = expanded
    found = []
    marker = ("E2E_JOB_MARKER=" + os.environ["E2E_JOB_MARKER"]).encode()
    for pid in sorted(selected):
        try:
            loaded = mapped_native(
                pid, ["nixl", "mooncake", "libucp", "libucs", "libuct", "libucm"]
            )
            if loaded:
                require(
                    marker in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0"),
                    "transport native process marker differs",
                )
                current = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
                require(
                    (int(current[1]), current[19]) == parents[pid],
                    "transport process identity changed",
                )
                found.append(
                    {
                        "pid": pid,
                        "parent_pid": parents[pid][0],
                        "start_ticks": parents[pid][1],
                        "loaded_native": loaded,
                    }
                )
        except FileNotFoundError:
            continue  # A child ended during this read; absence is not native proof.
    return found


def stdio(process, log_path):
    """Record real inherited streams or the actual captured-file inode."""
    child = Path("/proc") / str(process.pid)

    def identity(path):
        info = path.stat()
        return {"device": info.st_dev, "inode": info.st_ino}

    out, err = identity(child / "fd/1"), identity(child / "fd/2")
    if log_path is not None:
        require(out == identity(Path(log_path)), "stdout/log inode differs")
        return {"mode": "captured_file", "log": str(log_path), "stdout": out, "stderr": err}
    parent = Path("/proc") / str(os.getpid())
    require(out == identity(parent / "fd/1"), "inherited stdout differs")
    require(err == identity(parent / "fd/2"), "inherited stderr differs")
    return {"mode": "inherited_parent", "log": None, "stdout": out, "stderr": err}


def gateway(obj, native):
    row = process_identity(obj.process, "smg::router")
    launch = row["launch_argv_intent"]
    require(argument(launch, "-m") == "smg.launch_router", "gateway launcher")
    for flag, workers in (("--prefill", obj.prefill_workers), ("--decode", obj.decode_workers)):
        if obj.pd_mode:
            require("--pd-disaggregation" in launch, "PD gateway flag")
            actual = [launch[i + 1] for i, value in enumerate(launch) if value == flag]
            require(actual == [worker.base_url for worker in workers], "gateway role URLs")
    if obj.igw_mode:
        require("--enable-igw" in launch, "IGW gateway flag")
    loaded = mapped_native(row["pid"], ["/smg_rs"])
    require(len(loaded) == 1, "missing/ambiguous loaded gateway native")
    require(next(iter(loaded.values())) == {k: native[k] for k in ("bytes", "sha256")},
            "loaded gateway native differs from actual445")
    return {**row, "loaded_native": loaded, "stdio": stdio(obj.process, obj.log_path),
            "pd_mode": obj.pd_mode, "igw_mode": obj.igw_mode,
            "prefill_urls": [w.base_url for w in obj.prefill_workers],
            "decode_urls": [w.base_url for w in obj.decode_workers]}


def worker(obj):
    row = process_identity(obj.process)
    require(obj.engine == "sglang" and obj.mode.value in {"grpc", "http"}, "original SG wire")
    argv = row["launch_argv_intent"]
    require(argument(argv, "-m") == "sglang.launch_server", "actual SG launcher")
    require(("--grpc-mode" in argv) == (obj.mode.value == "grpc"), "SG HTTP/gRPC mode")
    from infra.model_specs import get_model_spec

    model = get_model_spec(obj.model_id)["model"]
    require(argument(argv, "--model-path") == model, "original model resolver differs")
    require(argument(argv, "--tp-size") == "1", "original SG2 TP1 changed")
    env = dict(value.split(b"=", 1) for value in
               (Path("/proc") / str(row["pid"]) / "environ").read_bytes().split(b"\0")
               if b"=" in value)
    mask = ",".join(map(str, obj.gpu_ids))
    require(env.get(b"CUDA_VISIBLE_DEVICES", b"").decode() == mask, "logical device mask")
    require(env.get(b"SPT_NOENV") == b"1", "marker preservation")
    role = obj.worker_type.value
    backend = None
    if role in {"prefill", "decode"}:
        require(argument(argv, "--disaggregation-mode") == role, "SG disaggregation role")
        backend = argument(argv, "--disaggregation-transfer-backend")
        require(backend == obj.effective_kv_backend() == "mooncake", "default Mooncake backend")
    else:
        require(role == "regular" and "--disaggregation-mode" not in argv, "regular SG role")
    return {**row, "role": role, "mode": obj.mode.value, "URL": obj.base_url,
            "model_id": obj.model_id, "model_path": model, "TP": 1,
            "logical_devices": obj.gpu_ids, "CUDA_VISIBLE_DEVICES": mask,
            "backend": backend, "stdio": stdio(obj.process, obj.log_path)}


