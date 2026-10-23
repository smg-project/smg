"""Bounded read-only allocator snapshots for the opt-in single-pool PD pilot."""

from collections import Counter

from .pd_scheduler_trace import nonnegative_int


def require(value, message):
    if not value:
        raise ValueError(message)


def held_blocks(scheduler, request_id):
    groups = scheduler.kv_cache_manager.get_blocks(request_id).blocks
    require(
        type(groups) is tuple and all(type(g) in (list, tuple) for g in groups),
        "unknown block groups",
    )
    require(sum(map(len, groups)) <= 4096, "too many request block references")
    return [block for group in groups for block in group]


def free_members(pool):
    limit = nonnegative_int(pool.num_gpu_blocks)
    require(0 < limit <= 131072 and len(pool.blocks) == limit, "unknown or oversized pool")
    queue = pool.free_block_queue
    previous = queue.fake_free_list_head
    current = previous.next_free_block
    members = set()
    for _ in range(limit + 1):
        if current is queue.fake_free_list_tail:
            require(current.prev_free_block is previous, "inconsistent queue tail")
            require(
                len(members) == nonnegative_int(queue.num_free_blocks), "inconsistent queue count"
            )
            return members
        require(current is not None and id(current) not in members, "invalid or cyclic free queue")
        require(current.prev_free_block is previous, "inconsistent free queue links")
        block_id = nonnegative_int(current.block_id)
        require(
            current.pool is pool
            and block_id < limit
            and pool.blocks[block_id] is current
            and current.ref_cnt == 0
            and current.is_null is False,
            "free queue has foreign or occupied pages",
        )
        members.add(id(current))
        previous, current = current, current.next_free_block
    raise ValueError("free queue exceeds pool bound")


def block_snapshot(pool, blocks):
    members = free_members(pool)
    multiplicity = Counter(map(id, blocks))
    unique = {id(block): block for block in blocks}
    require(
        len({block.block_id for block in unique.values()}) == len(unique), "duplicate block IDs"
    )
    records = []
    for identity, block in unique.items():
        require(block.pool is pool and block.is_null is False, "unsupported pool or null block")
        require(pool.blocks[nonnegative_int(block.block_id)] is block, "block object was replaced")
        refs = nonnegative_int(block.ref_cnt)
        records.append(
            {
                "block_id": nonnegative_int(block.block_id),
                "block_identity": f"{identity:x}",
                "held_references": multiplicity[identity],
                "ref_count": refs,
                "in_free_queue": identity in members,
                "prefix_cached": block.block_hash is not None,
            }
        )
    return sorted(records, key=lambda item: item["block_id"])


def capture(scheduler, request):
    blocks = held_blocks(scheduler, request.request_id)
    require(blocks, "no request-owned blocks to observe")
    pool = scheduler.kv_cache_manager.block_pool
    before = block_snapshot(pool, blocks)
    require(
        all(r["ref_count"] >= r["held_references"] and not r["in_free_queue"] for r in before),
        "invalid held references",
    )
    return {
        "request_id": request.request_id,
        "pool": pool,
        "blocks": blocks,
        "before": before,
        "queue_ids": {id(entry) for entry in scheduler.deferred_frees},
    }


def released(scheduler, state, phase, ticket, fence=None):
    require(scheduler.kv_cache_manager.block_pool is state["pool"], "pool identity changed")
    require(not held_blocks(scheduler, state["request_id"]), "request still holds KV blocks")
    after = block_snapshot(state["pool"], state["blocks"])
    exclusive = []
    for before, current in zip(state["before"], after, strict=True):
        require(before["block_identity"] == current["block_identity"], "block identity changed")
        require(
            before["ref_count"] - current["ref_count"] == before["held_references"],
            "reference decrement unproved",
        )
        require(
            current["in_free_queue"] == (current["ref_count"] == 0),
            "reuse membership differs from refcount",
        )
        if before["ref_count"] == before["held_references"]:
            require(current["in_free_queue"], "exclusive page is not reusable")
            exclusive.append(current["block_id"])
    return {
        "request_id": state["request_id"],
        "pool_identity": f"{id(state['pool']):x}",
        "phase": phase,
        "ticket": ticket,
        "fence": fence,
        "processed_step_seq": nonnegative_int(scheduler.processed_step_seq),
        "before": state["before"],
        "after": after,
        "request_holds_after": [],
        "request_references_released": True,
        "exclusive_pages_reusable": exclusive,
        "transfer_ack": None,
        "peer_memory_released": None,
    }
