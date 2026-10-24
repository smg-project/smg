"""Bounded local evidence from the original worker log and its owned descendants."""

import hashlib
import json
import os
import re
from pathlib import Path

from .process import require

RANK = re.compile(
    r"\(Worker pid=(\d+)\).*world_size=4 rank=(\d+) local_rank=(\d+) "
    r"distributed_init_method=(\S+) backend=nccl"
)


def worker_log(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(16 * 1024 * 1024 + 1)
    require(len(raw) <= 16 * 1024 * 1024, "worker log exceeds observation cap")
    return raw.decode(errors="replace")


def proc(pid):
    root = Path("/proc") / str(pid)
    fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    require(fields[0] != "Z", "rank or ancestor is a zombie")
    environment = dict(
        item.split(b"=", 1) for item in (root / "environ").read_bytes().split(b"\0") if b"=" in item
    )
    require(
        environment.get(b"E2E_JOB_MARKER") == os.environ["E2E_JOB_MARKER"].encode(),
        "rank marker differs",
    )
    require(environment.get(b"SPT_NOENV") == b"1", "rank marker preservation missing")
    return {
        "pid": pid,
        "parent_pid": int(fields[1]),
        "start_ticks": fields[19],
        "argv": [x.decode() for x in (root / "cmdline").read_bytes().split(b"\0") if x],
        "device_masks": {
            key.decode(): environment[key].decode()
            for key in [b"CUDA_VISIBLE_DEVICES", b"HIP_VISIBLE_DEVICES", b"ROCR_VISIBLE_DEVICES"]
            if key in environment
        },
    }


def ranks(log, owner_pid):
    matches = RANK.findall(log)
    require(len(matches) == 4, "four unique runtime rank registrations required")
    require({int(rank) for _, rank, _, _ in matches} == set(range(4)), "runtime TP ranks differ")
    require(len({pid for pid, _, _, _ in matches}) == 4, "runtime rank PID reused")
    require(len({group for _, _, _, group in matches}) == 1, "rank collective groups differ")
    result = []
    for pid, rank, local, group in sorted(matches, key=lambda item: int(item[1])):
        require(rank == local, "local TP rank differs")
        identity = proc(int(pid))
        require(
            identity["device_masks"].get("CUDA_VISIBLE_DEVICES") == "0,1,2,3",
            "rank device slice differs",
        )
        chain, current = [], identity
        while current["pid"] != owner_pid and len(chain) < 8:
            chain.append(current)
            current = proc(current["parent_pid"])
        require(current["pid"] == owner_pid, "rank is not descended from the owned worker")
        require(
            re.search(rf"\(Worker_TP{rank} pid={pid}\).*Graph capturing finished", log),
            "TP rank graph execution witness missing",
        )
        result.append(
            {
                "rank": int(rank),
                "local_rank": int(local),
                "collective_group": group,
                "process": identity,
                "ancestry": chain,
            }
        )
    return result


def model_cache(model, resolved):
    root = Path(
        os.environ.get("HF_HUB_CACHE", str(Path(os.environ.get("HF_HOME", "/models")) / "hub"))
    ).resolve()
    cache = root / ("models--" + model.replace("/", "--"))
    revision = (cache / "refs/main").read_text().strip()
    require(re.fullmatch(r"[0-9a-f]{40}", revision), "cached revision is not a commit")
    snapshot = (cache / "snapshots" / revision).resolve()
    require(snapshot.is_relative_to(root), "model snapshot escapes cache")
    require(
        resolved == model or Path(resolved).resolve() == snapshot,
        "resolved model differs from cached snapshot",
    )
    files = {}
    for name in ["config.json", "model.safetensors.index.json"]:
        path = snapshot / name
        require(path.resolve().is_relative_to(root), "model metadata escapes cache")
        with path.open("rb") as stream:
            raw = stream.read(4 * 1024 * 1024 + 1)
        require(
            len(raw) <= 4 * 1024 * 1024 and isinstance(json.loads(raw), dict),
            "model metadata invalid",
        )
        files[name] = {
            "path": str(path.resolve()),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return {
        "model": model,
        "resolved": resolved,
        "revision": revision,
        "snapshot": str(snapshot),
        "metadata": files,
        "full_weight_hashes_verified": False,
    }
