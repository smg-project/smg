"""Retain the original Mooncake transfer oracles in their owned fixture logs."""

import hashlib
import json
import re
from pathlib import Path

from .owned import require


def verify(recipe, events, directory):
    cases = {node for node in recipe["cases"] if "::TestPDMooncakeKvTransfer::" in node}
    require(len(cases) == 2, "both original Mooncake transfer cases required")
    sources, roles = {}, set()
    for event in events:
        identity = event.get("identity")
        if event["nodeid"] not in cases or not identity:
            continue
        if event["kind"] == "worker":
            config = identity["kv_config"]
            require(
                config and config["kv_connector"] == "MooncakeConnector",
                "transfer fixture connector",
            )
            roles.add(identity["role"])
            sources[identity["log"]] = "worker"
        elif event["kind"] == "gateway":
            for path in event["log_sources"]:
                sources[path] = "gateway"
    require(roles == {"prefill", "decode"}, "Mooncake fixture lacks both owned roles")
    offsets, streams = {}, {}
    for event in events:
        if event["kind"] != "log":
            continue
        name = event["artifact"]
        require(Path(name).name == name, "unsafe log artifact")
        start = offsets.get(name, 0)
        offsets[name] = start + event["bytes"]
        if event["nodeid"] not in cases or event["source"] not in sources:
            continue
        require(0 < event["bytes"] <= 16 * 1024**2, "transfer log increment cap")
        with (directory / name).open("rb") as stream:
            stream.seek(start)
            raw = stream.read(event["bytes"])
        require(
            len(raw) == event["bytes"] and hashlib.sha256(raw).hexdigest() == event["sha256"],
            "transfer log increment differs",
        )
        data = streams.setdefault(event["source"], bytearray())
        data.extend(raw)
        require(len(data) <= 32 * 1024**2, "Mooncake fixture log cap")
    router = b"\n".join(bytes(v) for path, v in streams.items() if sources[path] == "gateway")
    workers = b"\n".join(bytes(v) for path, v in streams.items() if sources[path] == "worker")
    marker = b"vLLM PD (Mooncake): injecting minted kv_transfer_params"
    successes = [int(v) for v in re.findall(rb"Num successful transfers=([1-9]\d*)", workers)]
    proof = {
        "cases": sorted(cases),
        "owned_roles": sorted(roles),
        "source_logs": sources,
        "minted_params_marker": marker in router,
        "positive_transfer_counts": successes,
        "scope": (
            "Original two-case owned fixture: minted decode params and positive KV-transfer metrics; "
            "not per-request transfer IDs or ACK/peer/HBM release"
        ),
        "accepted": None,
    }
    (directory / "mooncake-transfer.json").write_text(json.dumps(proof, indent=2) + "\n")
    require(marker in router, "owned Mooncake router did not mint transfer parameters")
    require(successes, "owned Mooncake workers did not report successful KV transfer")
