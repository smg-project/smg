"""Require actual E/P/D process imports; this does not assert transferred buffers."""

import json
import os
from collections import defaultdict
from pathlib import Path

from artifact_io import require


def verify(directory):
    groups = defaultdict(lambda: {"processes": [], "modules": set()})
    for path in sorted((directory / "workers").glob("*.jsonl")):
        with path.open("rb") as stream:
            data = stream.read(1024 * 1024 + 1)
        require(len(data) <= 1024 * 1024, "worker import evidence cap")
        rows = [json.loads(line) for line in data.splitlines()]
        process = [row for row in rows if row["kind"] == "process"]
        require(len(process) == 1, "worker process evidence missing/duplicated")
        identity = process[0]
        require(identity["pid"] == int(path.stem), "worker evidence PID differs")
        require(identity["job_marker"] == os.environ["E2E_JOB_MARKER"], "worker marker differs")
        require(identity["model_id"] == "Qwen/Qwen3.5-9B", "worker model differs")
        require(
            identity["original_launcher"][:2] == ["-m", "sglang.launch_server"],
            "worker module differs",
        )
        group = groups[identity["launcher_pid"]]
        group["processes"].append(identity)
        group["modules"].update(row["module"] for row in rows if row["kind"] == "import")
    counts = defaultdict(int)
    for pid, group in groups.items():
        roots = [row for row in group["processes"] if row["pid"] == pid]
        require(len(roots) == 1, "original worker launcher record missing")
        role = roots[0]["role"]
        require(role in {"encode", "prefill", "decode"}, "unknown worker role")
        require(
            all(row["role"] == role for row in group["processes"]),
            "role differs within worker tree",
        )
        required = (
            {
                "smg_grpc_proto.generated.sglang_encoder_pb2",
                "sglang.srt.disaggregation.encoder.grpc_server",
            }
            if role == "encode"
            else {
                "smg_grpc_servicer.sglang.servicer",
                "smg_grpc_servicer.sglang.encoder_handoff",
                "smg_grpc_proto.generated.sglang_scheduler_pb2",
            }
        )
        require(
            required <= group["modules"],
            "worker actual imports missing: " + repr(required - group["modules"]),
        )
        counts[role] += 1
    require(
        all(counts[role] >= 5 for role in ("encode", "prefill", "decode")),
        "four-topology role evidence incomplete",
    )
    result = {
        "verified": True,
        "launcher_counts": dict(counts),
        "group_count": len(groups),
        "proof": "actual imported module origins/hash checked inside worker processes",
        "not_proven": [
            "buffer ACK/physical KV release",
            "HBM release",
            "checkpoint full tensor hashes",
        ],
    }
    (directory / "worker-import-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    verify(Path(os.environ["E2E_EPD_RPC_PROOF_DIRECTORY"]))
