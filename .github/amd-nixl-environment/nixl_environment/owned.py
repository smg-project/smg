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


def gateway(gateway, native):
    row = process_identity(gateway.process, "smg::router")
    require(argument(row["launch_argv_intent"], "-m") == "smg.launch_router", "gateway launcher")
    launch = row["launch_argv_intent"]
    if gateway.pd_mode:
        require("--pd-disaggregation" in launch, "PD gateway flag differs")
        for flag, workers in [
            ("--prefill", gateway.prefill_workers),
            ("--decode", gateway.decode_workers),
        ]:
            urls = [launch[i + 1] for i, value in enumerate(launch) if value == flag]
            require(
                urls == [worker.base_url for worker in workers], "gateway role URLs differ: " + flag
            )
    if gateway.igw_mode:
        require("--enable-igw" in launch, "IGW gateway flag differs")
    loaded = mapped_native(row["pid"], ["/smg_rs"])
    require(len(loaded) == 1, "missing/ambiguous gateway native")
    require(
        next(iter(loaded.values())) == {key: native[key] for key in ("bytes", "sha256")},
        "loaded gateway native differs",
    )
    return {
        **row,
        "loaded_native": loaded,
        "prefill_urls": [w.base_url for w in gateway.prefill_workers],
        "decode_urls": [w.base_url for w in gateway.decode_workers],
        "pd_mode": gateway.pd_mode,
        "igw_mode": gateway.igw_mode,
    }


def worker(worker):
    row = process_identity(worker.process)
    require(worker.engine == "vllm" and worker.mode.value == "grpc", "original vLLM gRPC worker")
    argv = row["argv"]
    require(argument(argv, "-m") == "vllm.entrypoints.grpc_server", "worker launcher")
    require("--servicer-impl" not in argv, "default original servicer changed")
    env = dict(
        value.split(b"=", 1)
        for value in (Path("/proc") / str(row["pid"]) / "environ").read_bytes().split(b"\0")
        if b"=" in value
    )
    mask = env.get(b"CUDA_VISIBLE_DEVICES", b"").decode()
    require(mask == ",".join(map(str, worker.gpu_ids)), "actual logical device mask differs")
    require(env.get(b"SPT_NOENV") == b"1", "marker preservation missing")
    from infra.model_specs import get_model_spec

    require(
        argument(argv, "--model") == get_model_spec(worker.model_id)["model"],
        "actual model path differs from original fixture resolver",
    )
    role = worker.worker_type.value
    config = None
    if role in {"prefill", "decode"}:
        import json

        config = json.loads(argument(argv, "--kv-transfer-config"))
        backend = worker.effective_kv_backend()
        require(backend in {"nixl", "mooncake"}, "unexpected original PD backend")
        expected = "NixlConnector" if backend == "nixl" else "MooncakeConnector"
        require(config["kv_connector"] == expected, "role connector differs")
        require(
            config["kv_role"] == ("kv_producer" if role == "prefill" else "kv_consumer"),
            "prefill/decode transfer role differs",
        )
    path = Path(worker.log_path).resolve()
    require((Path("/proc") / str(row["pid"]) / "fd/1").resolve() == path, "worker stdout differs")
    return {
        **row,
        "role": role,
        "model_id": worker.model_id,
        "model_path": argument(argv, "--model"),
        "URL": worker.base_url,
        "logical_devices": worker.gpu_ids,
        "CUDA_VISIBLE_DEVICES": mask,
        "TP": argument(argv, "--tensor-parallel-size"),
        "kv_config": config,
        "log": str(path),
    }


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
