"""Consume only identity-validated local allocator records, never peer ACKs."""

import re

from .pd_local_kv_compat import SOURCE_SHA256
from .pd_local_kv_state import require
from .pd_scheduler_trace import nonnegative_int


def positive(value):
    require(nonnegative_int(value) > 0, "missing positive local KV integer")
    return value


def identity(value):
    require(
        isinstance(value, str) and re.fullmatch(r"[1-9a-f][0-9a-f]*", value) is not None,
        "missing local KV object identity",
    )
    return value


def blocks(records):
    require(type(records) is list and 0 < len(records) <= 4096, "missing block observations")
    ids, objects = set(), set()
    for record in records:
        block_id = nonnegative_int(record["block_id"])
        obj = identity(record["block_identity"])
        require(block_id not in ids and obj not in objects, "duplicate local KV block")
        ids.add(block_id)
        objects.add(obj)
        require(positive(record["held_references"]) <= 4096, "unknown held reference count")
        nonnegative_int(record["ref_count"])
        require(
            type(record["in_free_queue"]) is bool and type(record["prefix_cached"]) is bool,
            "unknown queue/cache membership",
        )
    return records


def same_blocks(before, after):
    keys = ("block_id", "block_identity", "held_references", "prefix_cached")
    require(
        len(before) == len(after)
        and all(all(a[k] == b[k] for k in keys) for a, b in zip(before, after, strict=True)),
        "local KV block ownership identity changed",
    )


def transition(record, released):
    positive(record["ticket"])
    identity(record["pool_identity"])
    nonnegative_int(record["processed_step_seq"])
    require(
        record["request_holds_after"] == []
        and record["request_references_released"] is released
        and record["transfer_ack"] is None
        and record["peer_memory_released"] is None,
        "missing local release boundary or unsupported peer claim",
    )
    before, after = blocks(record["before"]), blocks(record["after"])
    same_blocks(before, after)
    require(
        all(b["ref_count"] >= b["held_references"] and not b["in_free_queue"] for b in before),
        "missing held local references",
    )
    if not released:
        require(
            before == after
            and record["held_by"] == "deferred_queue"
            and positive(record["fence"]) > record["processed_step_seq"]
            and record["exclusive_pages_reusable"] == [],
            "invalid deferred ownership",
        )
        return
    exclusive = []
    for previous, current in zip(before, after, strict=True):
        require(
            previous["ref_count"] - current["ref_count"] == previous["held_references"]
            and current["in_free_queue"] is (current["ref_count"] == 0),
            "reference decrement or reusable-page membership unproved",
        )
        if previous["ref_count"] == previous["held_references"]:
            exclusive.append(previous["block_id"])
    require(
        bool(exclusive)
        and all(type(block_id) is int for block_id in record["exclusive_pages_reusable"])
        and record["exclusive_pages_reusable"] == exclusive,
        "no strictly reusable exclusive page",
    )


def validate_local_release(records, internal, checkpoint_seq):
    require(
        records[0]["source_contract"].get("local_kv_sources") == SOURCE_SHA256,
        "missing local KV source contract",
    )
    require(
        not any(row["event"] == "local_kv_unknown" for row in records),
        "unknown local KV observation",
    )
    rows = [
        row
        for row in records
        if row["event"] == "local_kv_return" and row.get("request_id") == internal
    ]
    require(rows and all(r["seq"] > checkpoint_seq for r in rows), "missing post-close local KV")
    released = [r for r in rows if r.get("phase") in ("immediate", "drained")]
    require(len(released) == 1, "missing or duplicate local KV release")
    result = released[0]
    transition(result, True)
    if result["phase"] == "immediate":
        require(len(rows) == 1 and result["fence"] is None, "ambiguous immediate release")
    else:
        require(len(rows) == 2 and rows[0].get("phase") == "deferred", "missing deferred owner")
        pending = rows[0]
        transition(pending, False)
        require(
            pending["seq"] < result["seq"]
            and all(pending[k] == result[k] for k in ("pool_identity", "ticket", "fence"))
            and positive(result["fence"]) <= result["processed_step_seq"],
            "unmatched local KV pool/ticket/fence",
        )
        same_blocks(pending["after"], result["before"])
    return {
        "internal_request_id": internal,
        "pool_identity": result["pool_identity"],
        "ticket": result["ticket"],
        "release_seq": result["seq"],
        "fence": result["fence"],
        "processed_step_seq": result["processed_step_seq"],
        "request_references_released": True,
        "exclusive_pages_reusable": result["exclusive_pages_reusable"],
        "transfer_ack": None,
        "peer_memory_released": None,
        "observation": "partial",
        "accepted": False,
    }
