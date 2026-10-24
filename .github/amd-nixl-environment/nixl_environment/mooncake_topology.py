"""Read the original nominal 2P2D fixture's owned logs; never issue requests."""

import hashlib
import json
import re
from pathlib import Path

from .owned import require

CASE = "router/test_pd_topologies.py::TestPDTopology::test_long_prompt_context_reaches_decode[2p2d]"


def verify(recipe, events, directory):
    require(CASE in recipe["cases"], "original nominal topology case missing")
    candidates = {}
    for event in events:
        identity = event.get("identity", {})
        if event["nodeid"] != CASE or event["kind"] != "gateway":
            continue
        if len(identity.get("prefill_urls", [])) == len(identity.get("decode_urls", [])) == 2:
            key = (identity["pid"], identity["start_ticks"])
            candidates.setdefault(key, []).append(event)
    require(len(candidates) == 1, "nominal case needs one owned 2P2D gateway")
    gateways = next(iter(candidates.values()))
    identity = gateways[-1]["identity"]
    roles = {
        **dict.fromkeys(identity["prefill_urls"], "prefill"),
        **dict.fromkeys(identity["decode_urls"], "decode"),
    }
    require(len(roles) == 4, "nominal role URLs overlap")
    workers, sources = {}, {}
    for event in gateways:
        for path in event["log_sources"]:
            sources[str(Path(path).resolve())] = "gateway"
    for event in events:
        identity = event.get("identity", {})
        if event["nodeid"] != CASE or event["kind"] != "worker" or identity.get("URL") not in roles:
            continue
        config = identity["kv_config"]
        require(identity["role"] == roles[identity["URL"]], "nominal worker role differs")
        require(
            config and config["kv_connector"] == "MooncakeConnector", "nominal connector differs"
        )
        require(
            identity["TP"] == "1" and len(identity["logical_devices"]) == 1, "nominal TP1 differs"
        )
        workers[identity["URL"]] = identity
        sources[str(Path(identity["log"]).resolve())] = "worker"
    require(set(workers) == set(roles), "nominal case lacks four owned role identities")
    require(
        len({w["logical_devices"][0] for w in workers.values()}) == 4,
        "nominal logical GPU slots overlap",
    )
    offsets, streams = {}, {}
    for event in events:
        if event["kind"] != "log":
            continue
        name = event["artifact"]
        require(Path(name).name == name and name not in {".", ".."}, "unsafe log artifact")
        start = offsets.get(name, 0)
        offsets[name] = start + event["bytes"]
        source = str(Path(event["source"]).resolve())
        if event["nodeid"] != CASE or source not in sources:
            continue
        require(0 < event["bytes"] <= 16 * 1024**2, "nominal log increment cap")
        with (directory / name).open("rb") as stream:
            stream.seek(start)
            raw = stream.read(event["bytes"])
        require(
            len(raw) == event["bytes"] and hashlib.sha256(raw).hexdigest() == event["sha256"],
            "nominal log increment differs",
        )
        data = streams.setdefault(source, bytearray())
        data.extend(raw)
        require(len(data) <= 32 * 1024**2, "nominal owned log cap")
    router = b"\n".join(bytes(v) for path, v in streams.items() if sources[path] == "gateway")
    counters = {
        path: [int(v) for v in re.findall(rb"Num successful transfers=([1-9]\d*)", raw)]
        for path, raw in streams.items()
        if sources[path] == "worker"
    }
    marker = b"vLLM PD (Mooncake): injecting minted kv_transfer_params"
    (directory / "mooncake-topology-transfer.json").write_text(
        json.dumps(
            {
                "case": CASE,
                "gateway_pid_start": list(next(iter(candidates))),
                "role_URLs": roles,
                "owned_worker_logs": {url: w["log"] for url, w in workers.items()},
                "source_logs": sources,
                "minted_params_marker": marker in router,
                "positive_transfer_counts_by_owned_log": counters,
                "accepted": None,
                "scope": (
                    "Case-associated nominal fixture logs and cumulative counters; "
                    "not request-specific KV bytes, ACK, peer release or HBM proof"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    require(marker in router, "owned nominal Mooncake router did not mint transfer parameters")
    require(any(counters.values()), "owned nominal Mooncake roles lack positive transfer metrics")
