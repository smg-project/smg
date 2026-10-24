"""Strict offline child attribution. Launcher imports never establish child imports."""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

STAGES = {
    "receiver_before": ("prefill", "sglang.srt.disaggregation.encoder.receiver"),
    "receiver_after": ("prefill", "sglang.srt.disaggregation.encoder.receiver"),
    "prefill_send": ("prefill", "sglang.srt.disaggregation.prefill"),
    "decode_allocate": ("decode", "sglang.srt.disaggregation.decode"),
    "decode_commit_before": ("decode", "sglang.srt.disaggregation.decode"),
    "decode_commit_after": ("decode", "sglang.srt.disaggregation.decode"),
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def expected_sources(manifest):
    root = Path(manifest["root"])
    sources = {r["module"]: {"module": r["module"], "path": str(root / r["path"]),
                            "sha256": r["patched_sha256"]} for r in manifest["modules"]}
    helper = manifest["helper"]
    sources[helper["module"]] = {"module": helper["module"], "path": str(root / helper["path"]),
                                 "sha256": helper["sha256"]}
    if "delta_helper" in manifest:
        helper = manifest["delta_helper"]
        sources[helper["module"]] = {"module": helper["module"], "path": str(root / helper["path"]),
                                     "sha256": helper["sha256"]}
    return sources


def validate_identity(row, processes, overlays, manifest, manifest_sha, marker):
    require(set(row) == {"kind", "pid", "parent_pid", "start_ticks", "role", "model_id",
                        "job_marker", "launcher", "ancestry", "helper", "first_origin",
                        "wall_time_ns", "monotonic_ns"}, "child identity schema")
    require(row["kind"] == "smg_epd_position_identity_v2", "child identity version")
    anchor = row["launcher"]
    require(set(anchor) == {"pid", "start_ticks", "role", "model_id", "job_marker", "nonce"},
            "launcher anchor schema")
    require(type(anchor["pid"]) is int and anchor["pid"] > 0
            and re.fullmatch(r"[0-9a-f]{32}", anchor["nonce"]), "launcher PID/nonce")
    require(row["job_marker"] == anchor["job_marker"] == marker, "child/launcher marker")
    require(row["model_id"] == anchor["model_id"] == manifest["model_id"], "child model")
    require(row["role"] == anchor["role"] and row["role"] in {"prefill", "decode"}, "child role")
    launcher_rows = processes.get(anchor["pid"], [])
    require(len(launcher_rows) == 1, "unique original launcher record missing")
    launcher = launcher_rows[0]
    require(launcher["pid"] == launcher["launcher_pid"] == anchor["pid"]
            and launcher["position_anchor"] == anchor
            and launcher["start_ticks"] == anchor["start_ticks"]
            and launcher["job_marker"] == marker and launcher["role"] == row["role"]
            and launcher["model_id"] == row["model_id"]
            and launcher["original_launcher"][:2] == ["-m", "sglang.launch_server"],
            "launcher identity/start ticks/anchor differs")
    chain = row["ancestry"]
    require(type(chain) is list and 1 <= len(chain) <= 16, "child ancestry bound")
    require(chain[0] == {k: row[k] for k in ("pid", "parent_pid", "start_ticks")}, "self identity differs")
    require(len({r["pid"] for r in chain}) == len(chain), "ancestry loop")
    for index, item in enumerate(chain):
        require(set(item) == {"pid", "parent_pid", "start_ticks"}
                and type(item["pid"]) is int and item["pid"] > 0
                and type(item["parent_pid"]) is int and item["parent_pid"] > 0
                and type(item["start_ticks"]) is str and item["start_ticks"].isdigit(), "ancestry fields")
        if index + 1 < len(chain):
            parent = chain[index + 1]
            require(item["parent_pid"] == parent["pid"]
                    and int(item["start_ticks"]) >= int(parent["start_ticks"]), "ancestry link/start ticks")
    require(chain[-1]["pid"] == anchor["pid"]
            and chain[-1]["start_ticks"] == anchor["start_ticks"], "launcher ancestry/PID reuse")
    sources = expected_sources(manifest)
    overlay = overlays.get(anchor["pid"])
    require(overlay is not None and overlay["applied"] is True
            and overlay["pid"] == anchor["pid"] and overlay["manifest_sha256"] == manifest_sha
            and overlay["modules"] == {k: {f: r[f] for f in ("path", "sha256")}
                                       for k, r in sources.items()}, "launcher overlay missing/differs")
    require(row["helper"] == sources[manifest["helper"]["module"]], "actual child helper origin")
    first = row["first_origin"]
    allowed = {module for role, module in STAGES.values() if role == row["role"]}
    require(first["module"] in allowed and first == sources[first["module"]], "actual child first origin")
    require(type(row["wall_time_ns"]) is int and type(row["monotonic_ns"]) is int, "identity clocks")
    return digest(row)


def analyze(records, processes, overlays, manifest, manifest_sha, marker):
    identities, events, rejected, drops = {}, [], [], []
    candidates = defaultdict(list)
    for row, evidence in records:
        if row.get("kind") == "smg_epd_position_identity_v2":
            candidates[row.get("pid")].append((row, evidence))
    for pid, rows in candidates.items():
        try:
            require(len(rows) == 1, "multiple identities/PID reuse")
            row, evidence = rows[0]
            key = validate_identity(row, processes, overlays, manifest, manifest_sha, marker)
            identities[pid] = {"record": row, "sha256": key, "evidence": evidence}
        except (ValueError, KeyError, TypeError) as error:
            rejected.append({"pid": pid, "reason": str(error), "scope": "identity"})
    usage, per_request, first_event = defaultdict(int), defaultdict(list), {}
    sources = expected_sources(manifest)
    for row, evidence in records:
        if row.get("kind") == "smg_epd_position_identity_v2":
            usage[row.get("pid")] += evidence["JSON_bytes"]
            continue
        try:
            require(row.get("kind") == "smg_epd_position_v2", "legacy/drop/unknown record cannot establish child attribution")
            require(set(row) == {"kind", "stage", "role", "pid", "start_ticks", "identity_sha256", "rid",
                                "wall_time_ns", "monotonic_ns", "origin", "event", "input_ids", "origin_ids",
                                "unpadded_ids", "mm_present", "model_is_mrope", "mrope_positions", "mrope_delta",
                                "embedding", "bootstrap_room", "processor_class", "lengths"}, "event schema")
            binding = identities[row["pid"]]
            identity = binding["record"]
            require(row["start_ticks"] == identity["start_ticks"]
                    and row["identity_sha256"] == binding["sha256"], "event PID generation/identity digest")
            role, module = STAGES[row["stage"]]
            require(row["role"] == identity["role"] == role and row["origin"] == sources[module], "event role/source/hash")
            require(type(row["rid"]) is str and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", row["rid"]), "event RID")
            require(type(row["event"]) is int and 1 <= row["event"] <= 8, "event sequence cap")
            require(type(row["wall_time_ns"]) is int and type(row["monotonic_ns"]) is int
                    and row["wall_time_ns"] >= identity["wall_time_ns"]
                    and row["monotonic_ns"] >= identity["monotonic_ns"], "event precedes identity")
            if not row["stage"].startswith("receiver_"):
                require(type(row["bootstrap_room"]) is int and row["bootstrap_room"] > 0, "event bootstrap room")
            first_event.setdefault(row["pid"], row)
            key = (row["pid"], row["rid"])
            previous = per_request[key]
            require(len(previous) < 8 and row["stage"] not in {r["stage"] for r in previous}
                    and row["event"] == len(previous) + 1, "duplicate/out-of-order/capped event")
            require(len({rid for pid, rid in per_request if pid == row["pid"]}) <= 8, "request cap")
            usage[row["pid"]] += evidence["JSON_bytes"]
            require(usage[row["pid"]] <= 256 * 1024, "PID log cap")
            previous.append(row)
            events.append(dict(row, evidence=evidence))
        except (ValueError, KeyError, TypeError) as error:
            rejected.append({"evidence": evidence, "pid": row.get("pid"), "reason": str(error)})
            if row.get("kind") in {"smg_epd_position_drop_v2", "smg_epd_position_drop"}:
                drops.append(dict(row, evidence=evidence))
    for pid, first in first_event.items():
        if first["origin"] != identities[pid]["record"]["first_origin"]:
            rejected.append({"pid": pid, "reason": "first emitted module differs from identity"})
    invalid_pids = {r["pid"] for r in rejected if r.get("pid") is not None}
    events = [row for row in events if row["pid"] not in invalid_pids]
    return {"identities": identities, "validated_events": events, "rejected_records": rejected, "drops": drops}


def compare(events):
    rooms, groups, unmatched = defaultdict(set), defaultdict(lambda: defaultdict(list)), []
    for row in events:
        if type(row["bootstrap_room"]) is int and row["bootstrap_room"] > 0:
            rooms[(row["pid"], row["rid"])].add(row["bootstrap_room"])
    for row in events:
        room = row["bootstrap_room"]
        inferred = type(room) is not int or room <= 0
        if inferred:
            choices = rooms[(row["pid"], row["rid"])]
            if len(choices) != 1:
                unmatched.append(row)
                continue
            room = next(iter(choices))
        groups[(row["rid"], room)][row["stage"]].append(dict(row, room_inferred_within_same_PID_RID=inferred))
    tables = []
    for (rid, room), stages in sorted(groups.items()):
        comparisons = []
        for left, right, lf, rf in [("receiver_before", "receiver_after", "input_ids", "input_ids"),
                                    ("receiver_after", "prefill_send", "input_ids", "origin_ids"),
                                    ("prefill_send", "decode_allocate", "origin_ids", "origin_ids"),
                                    ("decode_allocate", "decode_commit_after", "origin_ids", "origin_ids")]:
            a, b = stages.get(left, []), stages.get(right, [])
            result = {"from": left + "." + lf, "to": right + "." + rf, "status": "MISSING_OR_AMBIGUOUS"}
            if len(a) == len(b) == 1:
                x, y = a[0][lf], b[0][rf]
                result.update(left=x, right=y)
                if x.get("status") == y.get("status") == "CPU_FULL":
                    result["status"] = "SAME_CPU_IDS" if (x["length"], x["sha256"]) == (y["length"], y["sha256"]) else "DIFFERENT_CPU_IDS"
                else:
                    result["status"] = "NOT_BOTH_FULL_CPU_IDS"
            comparisons.append(result)
        tables.append({"rid": rid, "bootstrap_room": room, "stages": dict(stages),
                       "CPU_ID_comparisons": comparisons, "missing_sites": sorted(set(STAGES) - set(stages))})
    return tables, unmatched
