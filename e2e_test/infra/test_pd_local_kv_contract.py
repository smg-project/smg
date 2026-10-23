"""Native-method CPU records and corrupted evidence; never an engine pass."""

import copy
import json

import pytest
from infra.pd_local_kv_compat import SOURCE_SHA256
from infra.pd_local_kv_contract import validate_local_release
from infra.pd_local_kv_test_support import observed


@pytest.fixture(autouse=True)
def opt_in(monkeypatch):
    monkeypatch.setenv("SMG_PD_TRACE_LOCAL_KV", "1")


def records(*, deferred=False, shared=False):
    scheduler, req, raw = observed(deferred=deferred, shared=shared, cached=True)
    scheduler._free_blocks(req)
    if deferred:
        scheduler.processed_step_seq = 4
        scheduler._drain_deferred_frees()
    rows = [json.loads(line) for line in raw]
    rows = [row for row in rows if row["event"] in ("startup", "local_kv_return")]
    rows[0]["source_contract"] = {"local_kv_sources": dict(SOURCE_SHA256)}
    return rows, req.request_id


@pytest.mark.parametrize("deferred", [False, True])
def test_actual_native_method_records_prove_only_local_reusable_page(deferred):
    rows, internal = records(deferred=deferred)
    result = validate_local_release(rows, internal, 0)
    assert result["exclusive_pages_reusable"] == [1]
    assert result["request_references_released"] is True
    assert result["accepted"] is False and result["observation"] == "partial"
    assert result["transfer_ack"] is result["peer_memory_released"] is None


def test_shared_page_release_does_not_meet_strict_reusable_page_goal():
    rows, internal = records(shared=True)
    with pytest.raises(ValueError, match="exclusive page"):
        validate_local_release(rows, internal, 0)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "pool",
        "object",
        "ticket",
        "fence",
        "bool_ref",
        "held_mapping",
        "ack",
        "unknown",
        "source",
        "queue",
        "before_close",
        "prefix",
        "bool_page",
        "different_request",
        "held_count",
    ],
)
def test_incomplete_or_forged_release_is_rejected(fault):
    rows, internal = records(deferred=True)
    pending, release = rows[1:]
    checkpoint = 0
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(copy.deepcopy(release))
    elif fault in ("pool", "ticket"):
        release["pool_identity" if fault == "pool" else "ticket"] = (
            "bad" if fault == "pool" else 999
        )
    elif fault == "object":
        release["after"][0]["block_identity"] = "bad"
    elif fault == "fence":
        release["processed_step_seq"] = pending["fence"] - 1
    elif fault == "bool_ref":
        release["after"][0]["ref_count"] = False
    elif fault == "held_mapping":
        release["request_holds_after"] = [1]
    elif fault == "ack":
        release["transfer_ack"] = True
    elif fault == "unknown":
        rows.append({"event": "local_kv_unknown"})
    elif fault == "source":
        rows[0]["source_contract"]["local_kv_sources"].pop(next(iter(SOURCE_SHA256)))
    elif fault == "queue":
        release["after"][0]["in_free_queue"] = False
    elif fault == "before_close":
        checkpoint = release["seq"]
    elif fault == "prefix":
        release["after"][0]["prefix_cached"] = False
    elif fault == "bool_page":
        release["exclusive_pages_reusable"] = [True]
    elif fault == "different_request":
        release["request_id"] = "another-request"
    else:
        release["before"][0]["held_references"] = 2
    with pytest.raises(ValueError):
        validate_local_release(rows, internal, checkpoint)
