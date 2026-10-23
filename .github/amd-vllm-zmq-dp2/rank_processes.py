"""Bind actual DP child identities; retain optional DRM evidence without a new gate."""

import os
import re
from itertools import islice
from pathlib import Path

from reuse import require

TITLE = re.compile(r"VLLM::EngineCore_DP([01])")


def bounded(path, limit):
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, "process evidence cap")
    return raw


def rank_processes(parent_pid, proc=Path("/proc")):
    pending, seen, ranks = [(parent_pid, 0, None)], set(), []
    while pending:
        pid, depth, expected_parent = pending.pop()
        require(pid not in seen and len(seen) < 64 and depth < 8, "process tree cap/cycle")
        seen.add(pid)
        root = proc / str(pid)
        try:
            children = bounded(root / "task" / str(pid) / "children", 4096).decode().split()
        except FileNotFoundError:
            children = []
        pending.extend((int(child), depth + 1, pid) for child in children)
        try:
            argv = [x.decode() for x in bounded(root / "cmdline", 8192).split(b"\0") if x]
        except FileNotFoundError:
            continue
        match = TITLE.fullmatch(argv[0]) if len(argv) == 1 else None
        if not match:
            continue
        fields = bounded(root / "stat", 8192).decode().rsplit(")", 1)[1].split()
        require(int(fields[1]) == expected_parent, "DP child parent changed")
        environment = bounded(root / "environ", 128 * 1024).split(b"\0")
        marker = ("E2E_JOB_MARKER=" + os.environ["E2E_JOB_MARKER"]).encode()
        require(marker in environment, "DP child marker differs")
        drm, errors = [], []
        try:
            files = list(islice((root / "fdinfo").iterdir(), 257))
        except (FileNotFoundError, PermissionError) as error:
            files = []
            errors.append(type(error).__name__)
        for path in files[:256]:
            try:
                raw = bounded(path, 64 * 1024).decode()
                lines = [line for line in raw.splitlines() if line.startswith("drm-")]
                if lines:
                    drm.append({"fd": path.name, "lines": lines})
            except (FileNotFoundError, PermissionError) as error:
                errors.append(type(error).__name__)
        masks = [
            x.decode()
            for x in environment
            if x.startswith(
                (b"CUDA_VISIBLE_DEVICES=", b"HIP_VISIBLE_DEVICES=", b"ROCR_VISIBLE_DEVICES=")
            )
        ]
        ranks.append(
            {
                "rank": int(match[1]),
                "pid": pid,
                "parent_pid": int(fields[1]),
                "start_ticks": fields[19],
                "argv": argv,
                "marker_visible": True,
                "visible_device_environment": masks,
                "drm_fdinfo": drm,
                "fdinfo_read_errors": errors,
                "fdinfo_truncated": len(files) > 256,
                "physical_gpu_exclusivity_or_topology_proven": False,
            }
        )
    require(
        len(ranks) == 2 and {r["rank"] for r in ranks} == {0, 1},
        "two live DP rank children required",
    )
    return sorted(ranks, key=lambda item: item["rank"])
