"""The KV-event relay: parity with the Rust normalizer on the engines' wire
shapes, lenient decoding, and multi-rank streaming with per-rank replay over
real ZMQ and gRPC (no engine needed)."""

from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

import pytest
import pytest_asyncio

pytest.importorskip("smg_grpc_proto")
grpc = pytest.importorskip("grpc")
zmq = pytest.importorskip("zmq")
msgspec = pytest.importorskip("msgspec")
import zmq.asyncio  # noqa: E402, F811
from smg_grpc_proto.generated import common_pb2  # noqa: E402
from smg_grpc_servicer import kv_relay  # noqa: E402

TIERS = {
    "device": common_pb2.KV_CACHE_TIER_DEVICE,
    "host": common_pb2.KV_CACHE_TIER_HOST,
    "disk": common_pb2.KV_CACHE_TIER_DISK,
    "external": common_pb2.KV_CACHE_TIER_EXTERNAL,
}
U64 = (1 << 64) - 1


def _optional(message, name):
    return getattr(message, name) if message.HasField(name) else None


def _key_shape(key):
    which = key.WhichOneof("key")
    if which == "blob":
        return {"blob_len": len(key.blob)}
    if which == "multimodal":
        return {"multimodal": [key.multimodal.identifier, key.multimodal.offset]}
    return {which: getattr(key, which)}


# ---------------------------------------------------------------------------
# Parity with the Rust relay
#
# The scenarios of ``crates/engine_servicer/src/kv_wire_shapes.rs``, encoded
# here with msgspec through mirrors of the engines' own structs so the bytes
# are what the publishers put on the wire:
#
# - vLLM ``vllm/distributed/kv_events.py``: ``EventBatch`` is ``array_like``
#   (``[ts, events, data_parallel_rank]``), events are tagged maps (``type``)
#   with ``omit_defaults``; required fields are present even when nil.
# - SGLang ``python/sglang/srt/disaggregation/kv_events.py``: the same shape,
#   ``attn_dp_rank`` in the batch's third slot (nil when unset, SGLang's batch
#   does not omit defaults), ``medium``, ``cache_salt`` and ``session_id``
#   omitted when None, bigram pages as ``[t, t+1]`` cells.
# - The legacy layout both engines used before the map encoding: the same
#   structs with ``array_like=True``, tag first, every field in declaration
#   order and nil when unset (``omit_defaults`` does not thin an array).
#
# Each scenario is a sequence of batches (one ZMQ message each) with the
# normalizer's expected output: the forwarded events in order and the
# counters, keyed by the relay's ``DropReason`` names. The expectations are
# written by hand next to the events, as the rules in
# ``crates/engine_servicer/src/kv_wire.rs`` say; the Rust test asserts the
# same ones, which keeps the two relays in step.
# ---------------------------------------------------------------------------


def _vllm_structs(array_like):
    """vLLM's event structs; ``array_like`` selects the legacy layout."""

    class EventBatch(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
        ts: float
        events: list
        data_parallel_rank: int | None = None

    class KVCacheEvent(
        msgspec.Struct, array_like=array_like, omit_defaults=True, gc=False, tag=True
    ):
        pass

    class BlockStored(KVCacheEvent):
        block_hashes: list
        parent_block_hash: int | bytes | None
        token_ids: list
        block_size: int
        lora_id: int | None
        medium: str | None
        lora_name: str | None
        extra_keys: list | None = None
        group_idx: int | None = None
        kv_cache_spec_kind: str | None = None
        kv_cache_spec_sliding_window: int | None = None
        locality: str | None = None
        ownership: str | None = None
        session_id: str | None = None

    class BlockRemoved(KVCacheEvent):
        block_hashes: list
        medium: str | None
        group_idx: int | None = None
        locality: str | None = None
        ownership: str | None = None

    class AllBlocksCleared(KVCacheEvent):
        pass

    class BlockMigrated(KVCacheEvent):
        """An event type the relay does not know (stands in for a future one)."""

        block_hashes: list
        destination: str

    return EventBatch, BlockStored, BlockRemoved, AllBlocksCleared, BlockMigrated


def _sglang_structs(array_like):
    """SGLang's event structs; ``array_like`` selects the legacy layout."""

    class EventBatch(msgspec.Struct, array_like=True, gc=False):
        ts: float
        events: list
        attn_dp_rank: int | None = None

    class KVCacheEvent(
        msgspec.Struct, array_like=array_like, omit_defaults=True, gc=False, tag=True
    ):
        pass

    class BlockStored(KVCacheEvent):
        block_hashes: list
        parent_block_hash: int | None
        token_ids: list  # ints, or [t, t+1] pairs under bigram hashing
        block_size: int
        lora_id: int | None
        medium: str | None = None
        cache_salt: str | None = None
        session_id: str | None = None

    class BlockRemoved(KVCacheEvent):
        block_hashes: list
        medium: str | None = None

    class AllBlocksCleared(KVCacheEvent):
        pass

    class BlockMigrated(KVCacheEvent):
        block_hashes: list
        destination: str

    return EventBatch, BlockStored, BlockRemoved, AllBlocksCleared, BlockMigrated


def _digest(label):
    return hashlib.sha256(label.encode()).digest()


def _as_i64(value):
    value &= U64
    return value - (1 << 64) if value >= 1 << 63 else value


def _vllm_int(label):
    """vLLM's integer form: the low 64 bits of the digest, unsigned."""
    return int.from_bytes(_digest(label), "big") & U64


def _vllm_expected(label):
    """What the relay forwards for either form of a vLLM hash."""
    return _as_i64(_vllm_int(label))


def _sglang_int(label):
    """SGLang's integer form: the high 64 bits of the digest, signed."""
    return int.from_bytes(_digest(label)[:8], "big", signed=True)


def _stored_expect(
    hashes,
    tokens,
    *,
    dp_rank=0,
    parent=None,
    tier="device",
    cache_level=None,
    lora_name=None,
    cache_salt=None,
    group_idx=None,
    session_id=None,
    extra_keys=None,
):
    return {
        "kind": "stored",
        "dp_rank": dp_rank,
        "hashes": hashes,
        "parent": parent,
        "tier": tier,
        "cache_level": cache_level,
        "tokens": tokens,
        "lora_name": lora_name,
        "cache_salt": cache_salt,
        "group_idx": group_idx,
        "session_id": session_id,
        "extra_keys": extra_keys,
    }


def _removed_expect(hashes, *, dp_rank=0, tier="device", cache_level=None):
    return {
        "kind": "removed",
        "dp_rank": dp_rank,
        "hashes": hashes,
        "tier": tier,
        "cache_level": cache_level,
    }


def _cleared_expect(*, dp_rank=0):
    return {"kind": "cleared", "dp_rank": dp_rank}


def _vllm_scenario(array_like):
    """Both hash forms, a sliding-window group in both of its shapes, a second
    physical copy with per-copy removals, the offload tiers and every drop
    rule, a pool reset, a LoRA request with a multimodal item, a cache salt
    and a prompt embeddings digest whose child inherits the salt, and a
    second DP rank."""
    EventBatch, Stored, Removed, Cleared, Migrated = _vllm_structs(array_like)
    bs = 4
    h = {name: _vllm_int(f"vllm-{name}") for name in "abcdefghijk"}
    e = {name: _vllm_expected(f"vllm-{name}") for name in "abcdefghijk"}
    d1, d2 = _digest("vllm-digest-1"), _digest("vllm-digest-2")
    embeds = _digest("vllm-prompt-embeds")
    # The low 64 bits of a digest can exceed i64::MAX; make sure one does.
    assert any(v >= 1 << 63 for v in h.values()), "pick labels with a high bit set"

    def gpu_stored(hashes, parent, tokens, **kw):
        kw.setdefault("medium", "GPU")
        kw.setdefault("group_idx", 0)
        kw.setdefault("kv_cache_spec_kind", "full_attention")
        return Stored(
            block_hashes=hashes,
            parent_block_hash=parent,
            token_ids=tokens,
            block_size=kw.pop("block_size", bs),
            lora_id=kw.pop("lora_id", None),
            lora_name=kw.pop("lora_name", None),
            **kw,
        )

    batches = []
    forwarded = []

    # Batch 0: a plain chain in both hash forms; the sliding-window group in
    # both of its shapes.
    batches.append(
        EventBatch(
            ts=1700000000.0,
            data_parallel_rank=0,
            events=[
                gpu_stored([h["a"], h["b"]], None, list(range(1, 9)), session_id="req-1"),
                # Sliding-window group: more tokens than hashes x block size,
                # no hashes at all. Dropped by the group gate.
                gpu_stored(
                    [],
                    None,
                    list(range(1, 9)),
                    group_idx=1,
                    kv_cache_spec_kind="sliding_window",
                    kv_cache_spec_sliding_window=128,
                ),
                # The same group's usual shape: token_ids span the whole
                # computed range and block_hashes name only the window's last
                # blocks. Dropped whole by the group gate, never sliced from
                # the head.
                gpu_stored(
                    [h["k"]],
                    None,
                    list(range(1, 25)),
                    group_idx=1,
                    kv_cache_spec_kind="sliding_window",
                    kv_cache_spec_sliding_window=128,
                ),
                # Raw digests (VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0), the
                # parent given as an int, no extra keys on either block.
                gpu_stored([d1, d2], h["b"], list(range(9, 17)), extra_keys=[None, None]),
            ],
        )
    )
    forwarded += [
        _stored_expect(
            [e["a"], e["b"]],
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            group_idx=0,
            session_id="req-1",
        ),
        _stored_expect(
            [_as_i64(int.from_bytes(d1[-8:], "big")), _as_i64(int.from_bytes(d2[-8:], "big"))],
            [[9, 10, 11, 12], [13, 14, 15, 16]],
            parent=e["b"],
            group_idx=0,
        ),
    ]

    # Batch 1: a second physical copy, per-copy removals, offload tiers and
    # every other drop rule.
    batches.append(
        EventBatch(
            ts=1700000001.0,
            data_parallel_rank=0,
            events=[
                gpu_stored([h["a"], h["b"]], None, list(range(1, 9))),  # duplicate copy
                Removed(block_hashes=[h["a"]], medium="GPU", group_idx=0),
                Removed(block_hashes=[h["a"]], medium="GPU", group_idx=0),  # other copy
                # CPU offload placeholder: a chunk key, no tokens, block_size 0.
                gpu_stored([h["c"]], None, [], medium="CPU", block_size=0, kv_cache_spec_kind=None),
                Removed(block_hashes=[h["c"]], medium="CPU", group_idx=0),
                gpu_stored([h["d"]], None, [1, 2, 3, 4], medium="STORAGE", locality="REMOTE"),
                gpu_stored(
                    [h["d"]],
                    None,
                    [1, 2, 3, 4],
                    medium="STORAGE",
                    locality="LOCAL",
                    ownership="kvcr",
                ),
                gpu_stored([h["d"]], None, [1, 2, 3, 4], medium="STORAGE", locality="LOCAL"),
                gpu_stored([h["f"]], None, [1, 2, 3, 4], medium="MARS"),
                gpu_stored([h["g"]], None, [1, 2, 3, 4, 5, 6]),  # unaligned
                gpu_stored([h["i"]], h["i"], [1, 2, 3, 4]),  # parent is itself
                Migrated(block_hashes=[h["j"]], destination="peer"),
                # Malformed: hashes are not a list. Written out by hand
                # because no struct produces it.
                (
                    ["BlockStored", "nope", None, [1, 2, 3, 4], bs]
                    if array_like
                    else {
                        "type": "BlockStored",
                        "block_hashes": "nope",
                        "parent_block_hash": None,
                        "token_ids": [1, 2, 3, 4],
                        "block_size": bs,
                    }
                ),
            ],
        )
    )
    forwarded += [
        _stored_expect([e["a"], e["b"]], [[1, 2, 3, 4], [5, 6, 7, 8]], group_idx=0),
        _removed_expect([e["a"]]),
        _removed_expect([e["a"]]),
        _removed_expect([e["c"]], tier="host", cache_level=1),
        _stored_expect([e["d"]], [[1, 2, 3, 4]], tier="disk", cache_level=2, group_idx=0),
    ]

    # Batch 2: the pool reset, then the chain again (not a duplicate any more).
    batches.append(
        EventBatch(
            ts=1700000002.0,
            data_parallel_rank=0,
            events=[
                Cleared(),
                gpu_stored([h["a"], h["b"]], None, list(range(1, 9))),
            ],
        )
    )
    forwarded += [
        _cleared_expect(),
        _stored_expect([e["a"], e["b"]], [[1, 2, 3, 4], [5, 6, 7, 8]], group_idx=0),
    ]

    # Batch 3: a LoRA request with a multimodal item, a cache salt and prompt
    # embeddings; the salt rides in block 0's extra keys only and the child
    # inherits it.
    batches.append(
        EventBatch(
            ts=1700000003.0,
            data_parallel_rank=0,
            events=[
                gpu_stored(
                    [h["e"]],
                    None,
                    [1, 2, 3, 4],
                    lora_id=7,
                    lora_name="adapter",
                    extra_keys=[("adapter", ("mm-abc", 0), "salt-1", embeds)],
                    session_id="req-2",
                ),
                gpu_stored(
                    [h["h"]],
                    h["e"],
                    [5, 6, 7, 8],
                    lora_id=7,
                    lora_name="adapter",
                    extra_keys=[("adapter",)],
                    session_id="req-2",
                ),
            ],
        )
    )
    forwarded += [
        _stored_expect(
            [e["e"]],
            [[1, 2, 3, 4]],
            lora_name="adapter",
            cache_salt="salt-1",
            group_idx=0,
            session_id="req-2",
            extra_keys=[
                [
                    {"text": "adapter"},
                    {"multimodal": ["mm-abc", 0]},
                    {"text": "salt-1"},
                    {"blob_len": 32},
                ]
            ],
        ),
        _stored_expect(
            [e["h"]],
            [[5, 6, 7, 8]],
            parent=e["e"],
            lora_name="adapter",
            cache_salt="salt-1",
            group_idx=0,
            session_id="req-2",
            extra_keys=[[{"text": "adapter"}]],
        ),
    ]

    # Batch 4: another DP rank stores the same hashes; seen-sets are per rank.
    batches.append(
        EventBatch(
            ts=1700000004.0,
            data_parallel_rank=1,
            events=[gpu_stored([h["a"], h["b"]], None, list(range(1, 9)))],
        )
    )
    forwarded += [
        _stored_expect([e["a"], e["b"]], [[1, 2, 3, 4], [5, 6, 7, 8]], dp_rank=1, group_idx=0),
    ]

    counts = {
        "forwarded_stored": 8,
        "forwarded_removed": 3,
        "forwarded_cleared": 1,
        "duplicate_stores": 1,
        "bigram_stores": 0,
        "dropped": {
            "non_main_attention_group": 2,
            "placeholder": 1,
            "non_local_locality": 1,
            "unsupported_ownership": 1,
            "unknown_medium": 1,
            "unaligned_blocks": 1,
            "self_referencing_hashes": 1,
            "unknown_type": 1,
            "malformed": 1,
        },
    }
    return batches, {"forwarded": forwarded, "counts": counts}


def _sglang_scenario(array_like):
    """The startup clear, a chain with a coalesced two-page store, HiCache
    write-through, a salted chain, an Eagle bigram page, the DISK and
    EXTERNAL media, an unknown event type and a second attention DP rank."""
    EventBatch, Stored, Removed, Cleared, Migrated = _sglang_structs(array_like)
    bs = 4
    s = {name: _sglang_int(f"sglang-{name}") for name in "abcdefgh"}
    assert any(v < 0 for v in s.values()), "pick labels with a negative i64"

    def stored(hashes, parent, tokens, **kw):
        kw.setdefault("medium", "GPU")
        return Stored(
            block_hashes=hashes,
            parent_block_hash=parent,
            token_ids=tokens,
            block_size=bs,
            lora_id=None,
            **kw,
        )

    batches = []
    forwarded = []

    # Batch 0: the first batch after startup clears.
    batches.append(EventBatch(ts=1700000000.0, attn_dp_rank=0, events=[Cleared()]))
    forwarded += [_cleared_expect()]

    # Batch 1: a chain; the second store is coalesced over two pages.
    batches.append(
        EventBatch(
            ts=1700000001.0,
            attn_dp_rank=0,
            events=[
                stored([s["a"]], None, [1, 2, 3, 4], session_id="req-1"),
                stored([s["b"], s["c"]], s["a"], list(range(5, 13)), session_id="req-1"),
            ],
        )
    )
    forwarded += [
        _stored_expect([s["a"]], [[1, 2, 3, 4]], session_id="req-1"),
        _stored_expect(
            [s["b"], s["c"]], [[5, 6, 7, 8], [9, 10, 11, 12]], parent=s["a"], session_id="req-1"
        ),
    ]

    # Batch 2: HiCache write-through: back up to host, demote (device copy
    # goes, host stays), load back, evict the host copy.
    batches.append(
        EventBatch(
            ts=1700000002.0,
            attn_dp_rank=0,
            events=[
                stored([s["a"]], None, [1, 2, 3, 4], medium="CPU_PINNED"),
                Removed(block_hashes=[s["a"]], medium="GPU"),
                stored([s["a"]], None, [1, 2, 3, 4]),
                Removed(block_hashes=[s["a"]], medium="CPU_PINNED"),
            ],
        )
    )
    forwarded += [
        _stored_expect([s["a"]], [[1, 2, 3, 4]], tier="host", cache_level=1),
        _removed_expect([s["a"]]),
        _stored_expect([s["a"]], [[1, 2, 3, 4]]),
        _removed_expect([s["a"]], tier="host", cache_level=1),
    ]

    # Batch 3: a salted request's chain (the legacy array layout has no
    # readable salt slot, so that variant stores the chain unsalted).
    salt = None if array_like else "tenant-a"
    batches.append(
        EventBatch(
            ts=1700000003.0,
            attn_dp_rank=0,
            events=[
                stored([s["d"]], None, [1, 2, 3, 4], cache_salt=salt),
                stored([s["e"]], s["d"], [5, 6, 7, 8], cache_salt=salt),
            ],
        )
    )
    forwarded += [
        _stored_expect([s["d"]], [[1, 2, 3, 4]], cache_salt=salt),
        _stored_expect([s["e"]], [[5, 6, 7, 8]], parent=s["d"], cache_salt=salt),
    ]

    # Batch 4: an Eagle bigram page, removed again in the same batch. (Its
    # tokens differ from the plain chain's: two engine hashes with the same
    # tokens at the same position share one index membership per worker.)
    batches.append(
        EventBatch(
            ts=1700000004.0,
            attn_dp_rank=0,
            events=[
                stored([s["f"]], None, [[21, 22], [22, 23], [23, 24], [24, 25]]),
                Removed(block_hashes=[s["f"]], medium="GPU"),
            ],
        )
    )
    forwarded += [
        _stored_expect([s["f"]], [[21, 22, 23, 24]]),
        _removed_expect([s["f"]]),
    ]

    # Batch 5: the tiers the default core never emits but defines; an event
    # type the relay does not know.
    batches.append(
        EventBatch(
            ts=1700000005.0,
            attn_dp_rank=0,
            events=[
                stored([s["g"]], None, [1, 2, 3, 4], medium="DISK"),
                stored([s["h"]], None, [1, 2, 3, 4], medium="EXTERNAL"),
                Migrated(block_hashes=[s["h"]], destination="peer"),
            ],
        )
    )
    forwarded += [
        _stored_expect([s["g"]], [[1, 2, 3, 4]], tier="disk", cache_level=2),
        _stored_expect([s["h"]], [[1, 2, 3, 4]], tier="external", cache_level=3),
    ]

    # Batch 6: another attention DP rank; its batch carries its rank.
    batches.append(
        EventBatch(
            ts=1700000006.0,
            attn_dp_rank=1,
            events=[stored([s["a"]], None, [1, 2, 3, 4])],
        )
    )
    forwarded += [_stored_expect([s["a"]], [[1, 2, 3, 4]], dp_rank=1)]

    if array_like:
        # SGLang's legacy arrays put session_id where vLLM has extra_keys; the
        # relay cannot read it there.
        for item in forwarded:
            if "session_id" in item:
                item["session_id"] = None

    counts = {
        "forwarded_stored": 10,
        "forwarded_removed": 3,
        "forwarded_cleared": 1,
        "duplicate_stores": 0,
        "bigram_stores": 1,
        "dropped": {"unknown_type": 1},
    }
    return batches, {"forwarded": forwarded, "counts": counts}


SCENARIOS = {"vllm": _vllm_scenario, "sglang": _sglang_scenario}


@pytest.mark.parametrize(
    "engine,layout",
    [("vllm", "map"), ("vllm", "array"), ("sglang", "map"), ("sglang", "array")],
)
def test_the_engines_wire_shapes_normalize_like_the_rust_relay(engine, layout):
    name = f"{engine}-{layout}"
    batches, expect = SCENARIOS[engine](layout == "array")
    encoder = msgspec.msgpack.Encoder()
    normalizer = kv_relay.Normalizer()
    normalized = [
        normalizer.normalize_batch(kv_relay.decode_batch(encoder.encode(batch)), seq)
        for seq, batch in enumerate(batches)
    ]
    forwarded = [
        (_optional(batch, "dp_rank"), event) for batch in normalized for event in batch.events
    ]
    assert len(forwarded) == len(expect["forwarded"]), name
    for index, ((rank, event), want) in enumerate(zip(forwarded, expect["forwarded"])):
        at = f"{name} forwarded event {index}"
        assert rank == want["dp_rank"], at
        kind = event.WhichOneof("data")
        assert kind == want["kind"], at
        if kind == "stored":
            stored = event.stored
            assert [b.block_hash for b in stored.blocks] == want["hashes"], at
            assert _optional(stored, "parent_block_hash") == want["parent"], at
            assert stored.tier == TIERS[want["tier"]], at
            assert [list(b.token_ids) for b in stored.blocks] == want["tokens"], at
            for block in stored.blocks:
                assert _optional(block, "cache_level") == want["cache_level"], at
                assert block.block_size == len(block.token_ids), at
            assert _optional(stored, "lora_name") == want["lora_name"], at
            assert _optional(stored, "cache_salt") == want["cache_salt"], at
            assert _optional(stored, "group_idx") == want["group_idx"], at
            assert _optional(stored, "session_id") == want.get("session_id"), at
            if want.get("extra_keys") is not None:
                got = [[_key_shape(key) for key in block.extra_keys] for block in stored.blocks]
                assert got == want["extra_keys"], at
        elif kind == "removed":
            removed = event.removed
            assert list(removed.block_hashes) == want["hashes"], at
            assert removed.tier == TIERS[want["tier"]], at
            assert _optional(removed, "cache_level") == want["cache_level"], at
    counts = normalizer.counts
    want = expect["counts"]
    assert counts.forwarded_stored == want["forwarded_stored"], name
    assert counts.forwarded_removed == want["forwarded_removed"], name
    assert counts.forwarded_cleared == want["forwarded_cleared"], name
    assert counts.duplicate_stores == want["duplicate_stores"], name
    assert counts.bigram_stores == want["bigram_stores"], name
    assert counts.dropped == want["dropped"], name


# ---------------------------------------------------------------------------
# Lenient decoding
# ---------------------------------------------------------------------------


def _store(hashes, tokens, **extra):
    event = {
        "type": "BlockStored",
        "block_hashes": hashes,
        "parent_block_hash": None,
        "token_ids": tokens,
        "block_size": 4,
        "lora_id": None,
        "medium": "GPU",
        "lora_name": None,
    }
    event.update(extra)
    return event


def _batch(events, rank=0, ts=1.5):
    return msgspec.msgpack.encode([ts, events, rank])


def _normalize(payloads, rank=None):
    normalizer = kv_relay.Normalizer()
    batches = [
        normalizer.normalize_batch(kv_relay.decode_batch(payload), seq + 1, rank)
        for seq, payload in enumerate(payloads)
    ]
    return batches, normalizer.counts


def test_hashes_fold_like_the_engines_send_them():
    assert kv_relay.fold_hash(7) == 7
    assert kv_relay.fold_hash(2**63) == -(2**63)
    assert kv_relay.fold_hash(-3) == -3
    digest = bytes(range(32))
    assert kv_relay.fold_hash(digest) == int.from_bytes(digest[-8:], "big", signed=True)
    assert kv_relay.fold_hash("nope") is None


def test_unknown_keys_and_bigram_cells_decode_and_a_bad_event_costs_itself():
    payload = _batch(
        [
            _store([1], [[1, 2], [2, 3], [3, 4], [4, 5]], future_field={"nested": True}),
            {"type": "BlockStored", "block_hashes": "nope", "token_ids": [1], "block_size": 1},
            {"type": "BlockMigrated", "block_hashes": [1]},
            ["BlockRemoved", [1], "GPU"],
        ]
    )
    batches, counts = _normalize([payload])
    events = batches[0].events
    assert [event.WhichOneof("data") for event in events] == ["stored", "removed"]
    assert list(events[0].stored.blocks[0].token_ids) == [1, 2, 3, 4]
    assert events[1].event_id == 4, "ids advance for dropped events too"
    assert counts.bigram_stores == 1
    assert counts.dropped == {"malformed": 1, "unknown_type": 1}


def test_a_parent_record_outlives_all_but_its_last_copy():
    """vLLM keeps up to two physical copies of a hash and removes them one at
    a time. The record a child inherits its namespace from, and the digest the
    hash check chains on, stay until the last copy is removed and survive a
    repeat store, as the Rust normalizer keeps them; after the last removal a
    child is an orphan: no inherited salt, unverifiable."""
    seed = kv_relay.sglang_salt_seed("tenant-a")
    root = kv_relay.sglang_chain([1, 2, 3, 4], 4, seed)[0][1]
    child = kv_relay.sglang_chain([1, 2, 3, 4, 5, 6, 7, 8], 4, seed)[1][1]
    orphan = kv_relay.sglang_chain([1, 2, 3, 4, 13, 14, 15, 16], 4, seed)[1][1]
    normalizer = kv_relay.Normalizer(hash_check="sglang")

    def normalize(events, seq):
        return normalizer.normalize_batch(kv_relay.decode_batch(_batch(events)), seq)

    first = normalize(
        [
            _store([root], [1, 2, 3, 4], cache_salt="tenant-a"),
            _store([root], [1, 2, 3, 4], cache_salt="tenant-a"),  # the second physical copy
            {"type": "BlockRemoved", "block_hashes": [root], "medium": "GPU"},  # one copy goes
            _store([child], [5, 6, 7, 8], parent_block_hash=root),  # no salt of its own
        ],
        1,
    )
    stored = [event.stored for event in first.events if event.WhichOneof("data") == "stored"]
    assert stored[2].cache_salt == "tenant-a", "inherited from the copy still cached"
    counts = normalizer.counts
    assert (
        counts.hash_checked,
        counts.hash_mismatch,
        counts.hash_unverifiable,
        counts.duplicate_stores,
    ) == (3, 0, 0, 1)
    second = normalize(
        [
            {"type": "BlockRemoved", "block_hashes": [root], "medium": "GPU"},  # the last copy
            _store([orphan], [13, 14, 15, 16], parent_block_hash=root),
        ],
        2,
    )
    assert not second.events[1].stored.HasField("cache_salt")
    assert (counts.hash_checked, counts.hash_unverifiable) == (3, 1)
    assert counts.forwarded_removed == 2


def test_socket_rank_wins_over_the_payload_rank():
    batches, _ = _normalize([_batch([_store([1], [1, 2, 3, 4])], rank=0)], rank=3)
    assert batches[0].dp_rank == 3
    batches, _ = _normalize([msgspec.msgpack.encode([1.5, [_store([1], [1, 2, 3, 4])]])])
    assert not batches[0].HasField("dp_rank")


def test_a_non_batch_payload_is_an_error():
    with pytest.raises(ValueError):
        kv_relay.decode_batch(msgspec.msgpack.encode({"not": "a batch"}))


# ---------------------------------------------------------------------------
# Streaming: ranks, cursors, replay
# ---------------------------------------------------------------------------


def _bind_consecutive(ctx, kind, count, attempts=20):
    """``count`` sockets of ``kind`` on consecutive ports, as the engines lay out ranks."""
    for _ in range(attempts):
        sockets = []
        try:
            first = ctx.socket(kind)
            if kind == zmq.XPUB:
                first.setsockopt(zmq.XPUB_VERBOSE, 1)
            base = first.bind_to_random_port("tcp://127.0.0.1")
            sockets.append(first)
            for offset in range(1, count):
                sock = ctx.socket(kind)
                if kind == zmq.XPUB:
                    sock.setsockopt(zmq.XPUB_VERBOSE, 1)
                sock.bind(f"tcp://127.0.0.1:{base + offset}")
                sockets.append(sock)
            return base, sockets
        except zmq.ZMQError:
            for sock in sockets:
                sock.close(linger=0)
    raise RuntimeError("no consecutive ports available")


@pytest_asyncio.fixture
async def bridge():
    ctx = zmq.asyncio.Context()
    base, pubs = _bind_consecutive(ctx, zmq.XPUB, 2)
    replay_base, routers = _bind_consecutive(ctx, zmq.ROUTER, 2)
    config = SimpleNamespace(
        endpoint=f"tcp://127.0.0.1:{base}",
        replay_endpoint=f"tcp://127.0.0.1:{replay_base}",
        topic="kv",
    )
    options = {"replay_timeout": 0.3, "recv_timeout": 0.1}

    async def handler(request, context):
        sources = kv_relay.rank_sources(config, range(len(pubs)))
        async for batch in kv_relay.relay(
            sources,
            kv_relay.Engine.SGLANG,
            request.start_sequence_number,
            context,
            topic=config.topic,
            zmq_context=ctx,
            **options,
        ):
            yield batch

    server = grpc.aio.server()
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                "test.KvEvents",
                {
                    "Subscribe": grpc.unary_stream_rpc_method_handler(
                        handler,
                        request_deserializer=common_pb2.SubscribeKvEventsRequest.FromString,
                        response_serializer=common_pb2.KvEventBatch.SerializeToString,
                    )
                },
            ),
        )
    )
    grpc_port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{grpc_port}")
    rpc = channel.unary_stream(
        "/test.KvEvents/Subscribe",
        request_serializer=common_pb2.SubscribeKvEventsRequest.SerializeToString,
        response_deserializer=common_pb2.KvEventBatch.FromString,
    )

    def subscribe(cursor=0):
        return rpc(common_pb2.SubscribeKvEventsRequest(start_sequence_number=cursor))

    async def subscribed():
        # XPUB acknowledges each actual subscription; no timing sleeps needed.
        for pub in pubs:
            while await asyncio.wait_for(pub.recv(), 3) != b"\x01kv":
                pass

    async def publish(rank, seq, payload=None):
        payload = (
            _batch([_store([seq + 1], [1, 2, 3, 4])], rank=None) if payload is None else payload
        )
        await pubs[rank].send_multipart([b"kv", seq.to_bytes(8, "big"), payload])

    async def replay_request(rank):
        frames = await asyncio.wait_for(routers[rank].recv_multipart(), 3)
        assert frames[1] == b""
        return frames[0], int.from_bytes(frames[2], "big")

    async def replay_send(rank, identity, seq, payload=None, framing="sglang"):
        payload = _batch([_store([seq + 1], [1, 2, 3, 4])]) if payload is None else payload
        seq_bytes = kv_relay._END_SEQ if seq == -1 else seq.to_bytes(8, "big")
        frames = [identity, b"", seq_bytes, payload if seq != -1 else b""]
        if framing == "vllm":
            frames.insert(2, b"kv" if seq != -1 else b"")
        await routers[rank].send_multipart(frames)

    try:
        yield SimpleNamespace(
            subscribe=subscribe,
            subscribed=subscribed,
            publish=publish,
            replay_request=replay_request,
            replay_send=replay_send,
            pubs=pubs,
            routers=routers,
            config=config,
            options=options,
        )
    finally:
        await channel.close()
        await server.stop(None)
        for sock in pubs + routers:
            sock.close(linger=0)
        ctx.term()


async def read(call):
    return await asyncio.wait_for(call.read(), 3)


async def read_error(call, after_at_most=3):
    """The status the stream ends with, allowing a few batches before it."""
    with pytest.raises(grpc.aio.AioRpcError) as error:
        for _ in range(after_at_most + 1):
            await read(call)
    return error.value.code()


@pytest.mark.asyncio
async def test_ranks_are_tagged_from_their_socket_and_numbered_contiguously(bridge):
    call = bridge.subscribe()
    await bridge.subscribed()
    await bridge.publish(0, 0)
    await bridge.publish(0, 1)
    await bridge.publish(1, 0)  # rank 1 starts its own count; not stale
    await bridge.publish(1, 1)
    seen = [await read(call) for _ in range(4)]
    assert [batch.sequence_number for batch in seen] == [1, 2, 3, 4]
    # The two sockets are polled together, so ranks interleave; each rank's
    # own order holds and every batch carries its socket's rank.
    per_rank = {0: [], 1: []}
    for batch in seen:
        per_rank[batch.dp_rank].append(batch.events[0].stored.blocks[0].block_hash)
    assert per_rank == {0: [1, 2], 1: [1, 2]}
    call.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize("framing", ["sglang", "vllm"])
async def test_a_gap_is_filled_from_that_ranks_replay_before_later_batches(bridge, framing):
    call = bridge.subscribe()
    await bridge.subscribed()
    await bridge.publish(1, 0)
    assert (await read(call)).dp_rank == 1
    # Rank 1 skips 1 and 2; rank 0 keeps publishing meanwhile.
    await bridge.publish(1, 3)
    identity, start = await bridge.replay_request(1)
    assert start == 1
    await bridge.publish(0, 0)
    for seq in (1, 2, 3):  # the replay overlaps the live batch 3
        await bridge.replay_send(1, identity, seq, framing=framing)
    await bridge.replay_send(1, identity, -1, framing=framing)
    hashes = []
    for _ in range(4):
        batch = await read(call)
        hashes.append((batch.dp_rank, batch.events[0].stored.blocks[0].block_hash))
    # Replayed 1, 2, 3 for rank 1 in order, the live 3 deduplicated, then rank 0.
    assert hashes[:3] == [(1, 2), (1, 3), (1, 4)]
    assert hashes[3] == (0, 1)
    await bridge.publish(1, 4)
    assert (await read(call)).events[0].stored.blocks[0].block_hash == 5
    call.cancel()


@pytest.mark.asyncio
async def test_live_batches_the_replay_already_covered_are_duplicates_not_a_restart(bridge):
    """A gap's replay usually runs past the live batch that exposed it, and the
    live batches the publisher sent meanwhile are queued on the SUB socket:
    they arrive below the cursor the replay left. They are duplicates of what
    the replay forwarded, not a publisher restart, and the stream goes on; a
    sequence below what the replay covered is still a restart."""
    call = bridge.subscribe()
    await bridge.subscribed()
    await bridge.publish(0, 0)
    assert (await read(call)).sequence_number == 1
    await bridge.publish(0, 3)  # 1 and 2 skipped
    identity, start = await bridge.replay_request(0)
    assert start == 1
    await bridge.publish(0, 4)  # published while the replay runs: queued behind it
    await bridge.publish(0, 5)
    for seq in (1, 2, 3, 4, 5):  # the replay fills the gap and runs past the live 3, 4 and 5
        await bridge.replay_send(0, identity, seq)
    await bridge.replay_send(0, identity, -1)
    hashes = [(await read(call)).events[0].stored.blocks[0].block_hash for _ in range(5)]
    assert hashes == [2, 3, 4, 5, 6], "replayed 1..5, each once"
    await bridge.publish(0, 6)  # the queued 4 and 5 were duplicates; live continues
    assert (await read(call)).events[0].stored.blocks[0].block_hash == 7
    await bridge.publish(0, 0)  # below everything the replay covered: a restart
    assert await read_error(call) == grpc.StatusCode.DATA_LOSS


@pytest.mark.asyncio
async def test_the_replay_window_closes_once_the_live_stream_is_past_it(bridge):
    """The duplicates a replay leaves on the SUB queue can only arrive until
    the first live batch past the replay; after that a sequence inside the
    old window can only come from a restarted publisher, and it must end
    the stream like any other restart instead of being swallowed."""
    call = bridge.subscribe()
    await bridge.subscribed()
    await bridge.publish(0, 0)
    assert (await read(call)).sequence_number == 1
    await bridge.publish(0, 3)  # 1 and 2 skipped
    identity, start = await bridge.replay_request(0)
    assert start == 1
    await bridge.publish(0, 4)  # queued behind the replay
    for seq in (1, 2, 3, 4):
        await bridge.replay_send(0, identity, seq)
    await bridge.replay_send(0, identity, -1)
    for _ in range(4):
        await read(call)
    await bridge.publish(0, 5)  # live catches up past the replay: the window closes
    assert (await read(call)).events[0].stored.blocks[0].block_hash == 6
    await bridge.publish(0, 2)  # inside the old window now: a restarted publisher
    assert await read_error(call) == grpc.StatusCode.DATA_LOSS


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["truncated", "empty", "timeout", "short", "malformed"])
async def test_an_unverifiable_replay_ends_the_stream_with_data_loss(bridge, fault):
    call = bridge.subscribe()
    await bridge.subscribed()
    await bridge.publish(0, 0)
    assert (await read(call)).sequence_number == 1
    await bridge.publish(0, 4)
    identity, start = await bridge.replay_request(0)
    assert start == 1
    if fault == "truncated":
        await bridge.replay_send(0, identity, 2)  # history no longer holds 1
    elif fault == "empty":
        await bridge.replay_send(0, identity, -1)
    elif fault == "short":
        await bridge.replay_send(0, identity, 1)
        await bridge.replay_send(0, identity, -1)  # ends before 3
    elif fault == "malformed":
        await bridge.routers[0].send_multipart([identity, b"", b"bad"])
    assert await read_error(call) == grpc.StatusCode.DATA_LOSS


@pytest.mark.asyncio
async def test_a_gap_without_a_replay_endpoint_ends_the_stream(bridge):
    bridge.config.replay_endpoint = None
    call = bridge.subscribe()
    await bridge.subscribed()
    await bridge.publish(0, 0)
    assert (await read(call)).sequence_number == 1
    await bridge.publish(0, 2)
    assert await read_error(call) == grpc.StatusCode.DATA_LOSS


@pytest.mark.asyncio
async def test_a_publisher_restart_ends_the_stream_with_data_loss(bridge):
    call = bridge.subscribe()
    await bridge.subscribed()
    await bridge.publish(0, 5)
    assert (await read(call)).sequence_number == 1
    await bridge.publish(0, 0)
    assert await read_error(call) == grpc.StatusCode.DATA_LOSS


@pytest.mark.asyncio
async def test_a_nonzero_cursor_is_refused_before_subscribing(bridge):
    call = bridge.subscribe(100)
    assert await read_error(call) == grpc.StatusCode.OUT_OF_RANGE
    assert not await bridge.pubs[0].poll(timeout=50)


@pytest.mark.asyncio
async def test_bad_payloads_and_duplicates_are_skipped_without_replay(bridge):
    call = bridge.subscribe()
    await bridge.subscribed()
    await bridge.publish(0, 0)
    await bridge.publish(0, 1, b"not msgpack")
    await bridge.publish(0, 1, b"not msgpack")
    await bridge.pubs[0].send_multipart([b"kv", b"short"])
    await bridge.publish(0, 2)
    await bridge.publish(0, 2)
    first = await read(call)
    second = await read(call)
    assert (first.sequence_number, second.sequence_number) == (1, 2)
    assert second.events[0].stored.blocks[0].block_hash == 3
    assert not await bridge.routers[0].poll(timeout=50), "no replay for a consumed sequence"
    call.cancel()


@pytest.mark.asyncio
async def test_cancellation_releases_every_subscription(bridge):
    call = bridge.subscribe()
    await bridge.subscribed()
    call.cancel()
    for pub in bridge.pubs:
        assert await asyncio.wait_for(pub.recv(), 3) == b"\x00kv"


# ---------------------------------------------------------------------------
# Engine-exact hashes and the opt-in check
# ---------------------------------------------------------------------------


def test_sglang_hashes_reproduce_the_published_vectors():
    def ints(tokens, page, prior=None):
        return [i for _, i in kv_relay.sglang_chain(tokens, page, prior)]

    assert ints([1, 2, 3, 4], 4) == [-3488128144981237669]
    assert ints([10, 20, 30, 40, 50, 60, 70, 80], 2) == [
        978178666101069530,
        -895308556211281782,
        -8033692805846017938,
        835415944263129316,
    ]
    assert kv_relay.sglang_event_int(hashlib.sha256(b"").digest()) == -2039914840885289964
    seed = kv_relay.sglang_salt_seed("tenant-a")
    assert seed.hex() == "f5d0f785efe6042a4e4b0297a4d712917e1763c850835e93df58b373fd47fd2c"
    assert ints([1, 2, 3, 4, 5, 6, 7, 8], 4, seed) == [3718046898735569995, 3308615664605479373]
    assert (
        kv_relay.sglang_event_int(kv_relay.sglang_page(None, [1, 2, 2, 3, 3, 4, 4, 5]))
        == -638950109823820341
    )
    assert ints([1, 2, 3], 4) == [], "only full pages"


def test_vllm_sha256_cbor_hashes_reproduce_the_reference_run():
    # hash_block_tokens with sha256_cbor from vLLM at 0c16eee3f1, PYTHONHASHSEED unset.
    assert (
        kv_relay.vllm_none_hash().hex()
        == "9bd96a485ad84efdafb72ee48a1d7a69bcead0f8f0433173941b276b9581eef0"
    )
    a = kv_relay.vllm_block(None, [1, 2, 3, 4])
    assert a.hex() == "58d0879dff3800f65f8c5fd449d73048c0f7699dd238a03b84b146b151d55111"
    assert kv_relay.vllm_event_int(a) == -8885242862429187823
    b = kv_relay.vllm_block(a, [5, 6, 7, 8])
    assert b.hex() == "e1bc29393a2da9fd798f35ff04d215f846aee6df4b803ce3d43b57df56b58fb1"
    assert [i for _, i in kv_relay.vllm_chain([1, 2, 3, 4, 5, 6, 7, 8], 4)] == [
        -8885242862429187823,
        -3153830497298837583,
    ]
    lora = ("lora", "adapter", "/adapters/adapter")
    c = kv_relay.vllm_block(
        None,
        [1, 2, 3, 4],
        [lora, ("mm", "mm-abc", 0), ("cache_salt", "salt-1"), ("prompt_embeds", bytes(range(32)))],
    )
    assert c.hex() == "280bce663478b44320e68be593605214c94f7648ac34b4b94274fdcd0f2a8a04"
    d = kv_relay.vllm_block(c, [5, 6, 7, 8], [lora, ("mm", "mm-abc", -4)])
    assert d.hex() == "33e7394e25b7be46c7bf0c090da02e903b4e9d9536b1c18eed4d6e57e7a3658a"
    e = kv_relay.vllm_block(
        None,
        [1, 2, 3, 4],
        [("mm", "mm-abc", 0), ("cache_salt", "salt-1"), ("prompt_embeds", bytes(range(32)))],
    )
    assert e.hex() == "0d7b0d8344b79ad5e183b117cacc04aeb415bfdfa968fd28f611bc6971f4abba"
    f = kv_relay.vllm_block(e, [5, 6, 7, 8], [("mm", "mm-abc", -4)])
    assert f.hex() == "b81a4631bec909b72bcd0081cc2d7c87bcce5652c48318f9615e4aa3370c1d69"
    g = kv_relay.vllm_block(f, [9, 10, 11, 12])
    assert g.hex() == "024cf74ecdb93c6049fe59a66321192aab04701a75ac3e86619395104df937c4"


def _checked(normalizer):
    c = normalizer.counts
    return c.hash_checked, c.hash_mismatch, c.hash_unverifiable


def test_hash_check_verifies_sglang_chains_and_counts_mismatches():
    chain = kv_relay.sglang_chain([1, 2, 3, 4, 5, 6, 7, 8], 4)
    first, second = chain[0][1], chain[1][1]
    normalizer = kv_relay.Normalizer(hash_check="sglang")
    batch = normalizer.normalize_batch(
        kv_relay.decode_batch(
            _batch(
                [
                    _store([first], [1, 2, 3, 4]),
                    _store([second], [5, 6, 7, 8], parent_block_hash=first),
                    _store([second + 1], [5, 6, 7, 8], parent_block_hash=first),  # tampered
                    _store([99], [9, 10, 11, 12], parent_block_hash=12345),  # unknown parent
                ]
            )
        ),
        1,
    )
    assert len(batch.events) == 4, "a mismatch never drops"
    assert _checked(normalizer) == (3, 1, 1)

    seed = kv_relay.sglang_salt_seed("tenant-a")
    salted = kv_relay.sglang_chain([1, 2, 3, 4, 5, 6, 7, 8], 4, seed)
    normalizer = kv_relay.Normalizer(hash_check="SGLANG")
    normalizer.normalize_batch(
        kv_relay.decode_batch(
            _batch([_store([salted[0][1], salted[1][1]], list(range(1, 9)), cache_salt="tenant-a")])
        ),
        1,
    )
    assert _checked(normalizer) == (2, 0, 0)

    normalizer = kv_relay.Normalizer(hash_check="sglang")
    normalizer.normalize_batch(
        kv_relay.decode_batch(
            _batch([_store([-638950109823820341], [[1, 2], [2, 3], [3, 4], [4, 5]])])
        ),
        1,
    )
    assert _checked(normalizer) == (1, 0, 0)


def test_hash_check_verifies_vllm_sha256_cbor_chains():
    a, b = -8885242862429187823, -3153830497298837583
    normalizer = kv_relay.Normalizer(hash_check="vllm_sha256_cbor")
    batch = normalizer.normalize_batch(
        kv_relay.decode_batch(
            _batch(
                [
                    _store([a, b], list(range(1, 9))),
                    _store([7], [9, 10, 11, 12], parent_block_hash=b, lora_name="adapter"),
                    _store(
                        [8], [9, 10, 11, 12, 13], parent_block_hash=b
                    ),  # unaligned, dropped first
                ]
            )
        ),
        1,
    )
    assert len(batch.events) == 2
    assert _checked(normalizer) == (2, 0, 1)

    embeds = bytes(range(32))
    e = kv_relay.vllm_block(
        None,
        [1, 2, 3, 4],
        [("mm", "mm-abc", 0), ("cache_salt", "salt-1"), ("prompt_embeds", embeds)],
    )
    f = kv_relay.vllm_block(e, [5, 6, 7, 8], [("mm", "mm-abc", -4)])
    g = kv_relay.vllm_block(f, [9, 10, 11, 12])
    ei, fi, gi = (kv_relay.vllm_event_int(x) for x in (e, f, g))
    normalizer = kv_relay.Normalizer(hash_check="vllm-sha256-cbor")
    batch = normalizer.normalize_batch(
        kv_relay.decode_batch(
            _batch(
                [
                    _store([ei], [1, 2, 3, 4], extra_keys=[[["mm-abc", 0], "salt-1", embeds]]),
                    _store([fi], [5, 6, 7, 8], parent_block_hash=ei, extra_keys=[[["mm-abc", -4]]]),
                    _store([gi], [9, 10, 11, 12], parent_block_hash=fi),
                    _store(
                        [5], [13, 14, 15, 16], parent_block_hash=gi, extra_keys=[[3]]
                    ),  # unknown key shape
                ]
            )
        ),
        1,
    )
    assert len(batch.events) == 4
    assert _checked(normalizer) == (3, 0, 1)
    assert batch.events[0].stored.cache_salt == "salt-1"


def test_hash_check_is_off_unless_asked(caplog):
    assert kv_relay.Normalizer().hash_check is None
    assert kv_relay.Normalizer(hash_check="").hash_check is None
    with caplog.at_level("WARNING"):
        assert kv_relay.Normalizer(hash_check="xxhash").hash_check is None
    assert "names no known engine hash" in caplog.text
    normalizer = kv_relay.Normalizer()
    normalizer.normalize_batch(kv_relay.decode_batch(_batch([_store([1], [1, 2, 3, 4])])), 1)
    assert _checked(normalizer) == (0, 0, 0)
