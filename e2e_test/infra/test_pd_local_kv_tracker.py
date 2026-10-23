"""Native queue/free/fence CPU controls; no engine or transfer is executed."""

import json

import pytest
from infra.pd_local_kv_state import block_snapshot, free_members
from infra.pd_local_kv_test_support import observed


def local(rows):
    return [json.loads(row) for row in rows if json.loads(row)["event"] == "local_kv_return"]


@pytest.fixture(autouse=True)
def opt_in(monkeypatch):
    monkeypatch.setenv("SMG_PD_TRACE_LOCAL_KV", "1")


@pytest.mark.parametrize("cached", [False, True])
def test_exclusive_native_release_is_reusable_even_if_prefix_cached(cached):
    scheduler, req, rows = observed(cached=cached)
    scheduler._free_blocks(req)
    event = local(rows)[0]
    assert event["phase"] == "immediate" and event["request_references_released"] is True
    assert event["before"][0]["ref_count"] == 1 and event["after"][0]["ref_count"] == 0
    assert event["after"][0]["in_free_queue"] is True
    assert event["after"][0]["prefix_cached"] is cached
    assert event["exclusive_pages_reusable"] == [scheduler.target.block_id]
    assert event["transfer_ack"] is event["peer_memory_released"] is None


def test_shared_prefix_reference_release_is_not_a_reusable_page():
    scheduler, req, rows = observed(shared=True, cached=True)
    scheduler._free_blocks(req)
    event = local(rows)[0]
    assert event["request_references_released"] is True
    assert event["before"][0]["ref_count"] == 2 and event["after"][0]["ref_count"] == 1
    assert event["exclusive_pages_reusable"] == []
    assert event["after"][0]["in_free_queue"] is False
    assert scheduler.owners["another-request"] == [scheduler.target]


def test_native_deferred_mapping_removal_needs_actual_satisfied_fence():
    scheduler, req, rows = observed(deferred=True)
    scheduler._free_blocks(req)
    assert req.request_id not in scheduler.requests
    assert scheduler.target.ref_cnt == 1
    assert local(rows)[0]["held_by"] == "deferred_queue"
    scheduler.processed_step_seq = 3
    scheduler._drain_deferred_frees()
    assert len(local(rows)) == 1 and scheduler.target.ref_cnt == 1
    scheduler.processed_step_seq = 4
    scheduler._drain_deferred_frees()
    deferred, released = local(rows)
    assert deferred["ticket"] == released["ticket"]
    assert deferred["pool_identity"] == released["pool_identity"]
    assert released["phase"] == "drained" and released["fence"] == 4
    assert released["exclusive_pages_reusable"] == [scheduler.target.block_id]
    assert scheduler._smg_local_kv.pending == {}


def test_other_request_reoccupying_same_native_page_is_not_accepted():
    scheduler, req, rows = observed()
    original = scheduler.kv_cache_manager.free

    def release_then_allocate(request):
        original(request)
        block = scheduler.kv_cache_manager.block_pool.free_block_queue.popleft()
        assert block is scheduler.target
        block.ref_cnt += 1
        scheduler.owners["another-request"] = [block]

    scheduler.kv_cache_manager.free = release_then_allocate
    scheduler._free_blocks(req)
    assert local(rows) == [] and scheduler._smg_local_kv.unknown is True
    assert any(json.loads(row)["event"] == "free_blocks_return" for row in rows)


def test_overlapping_deferred_owner_is_unknown_even_when_native_drain_frees_it():
    scheduler, req, rows = observed(deferred=True)
    scheduler._free_blocks(req)
    scheduler.target.ref_cnt += 1
    scheduler.deferred_frees.append((4, [scheduler.target]))
    scheduler.processed_step_seq = 4
    scheduler._drain_deferred_frees()
    assert scheduler.target.ref_cnt == 0
    assert scheduler._smg_local_kv.unknown is True
    assert not any(event["request_references_released"] for event in local(rows))


def test_snapshot_does_not_mutate_native_queue_and_rejects_corrupt_links():
    scheduler, _, _ = observed()
    pool = scheduler.kv_cache_manager.block_pool
    before = pool.free_block_queue.get_all_free_blocks()
    block_snapshot(pool, [scheduler.target])
    assert pool.free_block_queue.get_all_free_blocks() == before
    pool.free_block_queue.fake_free_list_head.next_free_block.prev_free_block = None
    with pytest.raises(ValueError, match="links"):
        free_members(pool)


def test_native_exception_propagates_without_manufacturing_release():
    scheduler, req, rows = observed()

    def broken(request):
        raise RuntimeError("native free failed")

    scheduler.kv_cache_manager.free = broken
    with pytest.raises(RuntimeError, match="native free failed"):
        scheduler._free_blocks(req)
    assert local(rows) == [] and scheduler.requests[req.request_id] is req


def test_disabled_path_emits_only_original_events(monkeypatch):
    monkeypatch.delenv("SMG_PD_TRACE_LOCAL_KV")
    scheduler, req, rows = observed()
    scheduler._free_blocks(req)
    assert local(rows) == [] and scheduler._smg_local_kv is None
    assert scheduler.target.ref_cnt == 0


def test_shared_owner_releases_before_native_drain():
    scheduler, req, rows = observed(deferred=True, shared=True, cached=True)
    scheduler._free_blocks(req)
    pool = scheduler.kv_cache_manager.block_pool
    pool.free_blocks(scheduler.owners.pop("another-request"))
    scheduler.processed_step_seq = 4
    scheduler._drain_deferred_frees()
    pending, released = local(rows)
    assert pending["before"][0]["ref_count"] == 2
    assert released["before"][0]["ref_count"] == 1
    assert released["after"][0]["ref_count"] == 0
    assert released["exclusive_pages_reusable"] == [scheduler.target.block_id]


def test_removed_deferred_entry_before_fence_is_unknown():
    scheduler, req, rows = observed(deferred=True)
    scheduler._free_blocks(req)
    scheduler.deferred_frees.clear()
    scheduler._drain_deferred_frees()
    assert scheduler._smg_local_kv.unknown is True
    assert not any(event["request_references_released"] for event in local(rows))


@pytest.mark.parametrize("fault", ["pool", "object", "occupied", "cycle"])
def test_corrupt_native_free_queue_cannot_prove_reuse(fault):
    scheduler, _, _ = observed()
    pool = scheduler.kv_cache_manager.block_pool
    block = pool.free_block_queue.fake_free_list_head.next_free_block
    if fault == "pool":
        block.pool = object()
    elif fault == "object":
        pool.blocks[block.block_id] = object()
    elif fault == "occupied":
        block.ref_cnt = 1
    else:
        block.next_free_block = block
    with pytest.raises(ValueError):
        free_members(pool)
