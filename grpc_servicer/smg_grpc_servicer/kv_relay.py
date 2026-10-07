"""The engines' KV-cache events relayed into the gateway's ``KvEventBatch`` stream.

One relay per ``SubscribeKvEvents`` call: a SUB socket per data-parallel rank
(vLLM and SGLang publish one stream per rank on ``base_port + rank``, each
with its own sequence counter), each rank's sequence followed with gap replay
through the publisher's ROUTER, and every event normalized by the rules the
Rust relay applies (``crates/engine_servicer/src/kv_wire.rs``; the relay tests
on both sides build the same engine wire shapes in code and expect the same
output of them).

What the gateway sees is one stream with its own contiguous sequence numbers
and the rank on every batch. Recovery is the relay's: a gap on a rank is
filled from that rank's replay endpoint before anything later is forwarded;
a replay that cannot be verified (history truncated, timeout, malformed) and
a publisher restart (its counter starts over) end the stream with
``DATA_LOSS``, which the gateway answers by clearing the worker and
resubscribing from zero. A non-zero ``start_sequence_number`` is refused with
``OUT_OF_RANGE`` for the same reason: the relay's numbering is per call. A
zero cursor is live only: nothing is kept between calls, so there is no
history and no state snapshot to serve (the Rust relay in
``crates/engine_servicer`` keeps both for the servicer's lifetime and serves a
``KvSnapshotChunk`` snapshot once its history window has rolled).

Decoding is lenient, as the engines evolve their events by adding optional
keys: batches are ``[ts, events, rank]``, events are tagged maps (``type``) or
the legacy tag-first arrays, unknown keys are ignored, missing or unreadable
fields cost that event and not the batch, and ``token_ids`` cells may be an
int or a ``[token, next_token]`` bigram (Eagle-family speculative decoding),
folded to their tokens.

Hash identity: an integer hash is used as is (vLLM sends the low 64 bits of
the digest unsigned, SGLang the high 64 bits signed; both are 64-bit patterns
the proto carries as ``int64``); a raw digest folds to its last eight bytes
big-endian, the integer vLLM would have sent for it.

Stores and removals are forwarded one for one. vLLM keeps up to two physical
copies of one hash and removes them one at a time; every copy's store and
every removal go through, and the gateway counts copies per worker and tier.
The relay's own record of a hash (the namespace a child inherits, the digest
the hash check chains on) counts the copies too, capped as the gateway caps
them, and goes with the last removal. Of a hybrid model's KV-cache
groups only the main-attention ones are forwarded; this relay drops every
sliding-window and state-space group's events (the Rust relay additionally
forwards a rank that publishes sliding-window groups only, with the hashes
aligned to the tail of the tokens).

``SMG_KV_EVENT_HASH_CHECK=sglang|vllm-sha256-cbor`` (or ``relay(...,
hash_check=...)``) turns on engine-hash verification: every store whose
parent is known is rehashed the worker's way (SGLang's per-page SHA-256
chain; vLLM's ``sha256_cbor`` with the default seed) and mismatches are
counted, never dropped, so a worker with a different algorithm, seed or page
size shows up as a counter. The algorithms and their test vectors mirror
``crates/engine_servicer/src/engine_hash.rs``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import grpc
import msgspec
from smg_grpc_proto.generated import common_pb2

if TYPE_CHECKING:
    import zmq
    import zmq.asyncio

# ``zmq`` is imported where the relay opens sockets (``replay_frames`` and
# ``relay``), not here: the rank discovery and normalization half of this
# module serves engines whose servicers carry no ``pyzmq`` (the vLLM
# servicer's unit tests import it on a bare interpreter).

logger = logging.getLogger(__name__)

__all__ = [
    "HASH_CHECKS",
    "HASH_CHECK_ENV",
    "Counts",
    "Engine",
    "Normalizer",
    "RankSource",
    "RelayFailure",
    "WireBatch",
    "WireEvent",
    "decode_batch",
    "endpoint_for_rank",
    "fold_hash",
    "rank_sources",
    "relay",
    "replay_frames",
    "sglang_chain",
    "sglang_event_int",
    "sglang_page",
    "sglang_salt_seed",
    "vllm_block",
    "vllm_chain",
    "vllm_event_int",
    "vllm_none_hash",
]

_U64_MASK = 0xFFFF_FFFF_FFFF_FFFF
_I64_SIGN = 0x8000_0000_0000_0000
_END_SEQ = _U64_MASK.to_bytes(8, "big")  # (-1) signed and 2**64-1 unsigned are the same bytes

MAIN_ATTENTION_KINDS = ("full_attention", "mla_attention", "sink_full_attention")
# The most physical copies of one block counted per record: the gateway's cap.
COPIES_CAP = 8
_RESIDENCY_AGENT = "kvcr"

_TIER_BY_MEDIUM = {
    "GPU": common_pb2.KV_CACHE_TIER_DEVICE,
    "DEVICE": common_pb2.KV_CACHE_TIER_DEVICE,
    "CPU": common_pb2.KV_CACHE_TIER_HOST,
    "CPU_PINNED": common_pb2.KV_CACHE_TIER_HOST,
    "CPU_TIER1": common_pb2.KV_CACHE_TIER_HOST,
    "CPU_TIER2": common_pb2.KV_CACHE_TIER_DISK,
    "DISK": common_pb2.KV_CACHE_TIER_DISK,
    "NVME": common_pb2.KV_CACHE_TIER_DISK,
    "STORAGE": common_pb2.KV_CACHE_TIER_DISK,
    "EXTERNAL": common_pb2.KV_CACHE_TIER_EXTERNAL,
    "NETWORK": common_pb2.KV_CACHE_TIER_EXTERNAL,
    "REMOTE": common_pb2.KV_CACHE_TIER_EXTERNAL,
    "SHARED": common_pb2.KV_CACHE_TIER_EXTERNAL,
}
_CACHE_LEVEL = {
    common_pb2.KV_CACHE_TIER_DEVICE: None,
    common_pb2.KV_CACHE_TIER_HOST: 1,
    common_pb2.KV_CACHE_TIER_DISK: 2,
    common_pb2.KV_CACHE_TIER_EXTERNAL: 3,
}


class Engine(Enum):
    """Which publisher is on the other end; only the replay reply framing differs."""

    VLLM = "vllm"
    SGLANG = "sglang"


# ---------------------------------------------------------------------------
# Wire decoding
# ---------------------------------------------------------------------------


def fold_hash(value: object) -> int | None:
    """An engine block hash as the proto's signed 64-bit identity; ``None`` if unreadable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        masked = value & _U64_MASK
    elif isinstance(value, (bytes, bytearray)):
        masked = int.from_bytes(bytes(value)[-8:], "big")
    else:
        return None
    return masked - (1 << 64) if masked >= _I64_SIGN else masked


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _unsigned32(value: object) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 2**32
        else None
    )


def _signed(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _hashes(value: object) -> list[int] | None:
    if not isinstance(value, (list, tuple)):
        return None
    folded = [fold_hash(item) for item in value]
    return None if any(item is None for item in folded) else folded  # type: ignore[return-value]


def _tokens(value: object) -> tuple[list[int], list[int] | None] | None:
    """Token ids as plain ints, or bigram ``[token, next]`` cells folded to their
    tokens; the second item is both words of every bigram (for the engine-hash
    check), ``None`` for plain ids."""
    if not isinstance(value, (list, tuple)):
        return None
    ids: list[int] = []
    words: list[int] = []
    pairs = 0
    for cell in value:
        if isinstance(cell, int) and not isinstance(cell, bool):
            if not 0 <= cell < 2**32:
                return None
            ids.append(cell)
        elif (
            isinstance(cell, (list, tuple))
            and len(cell) == 2
            and all(isinstance(t, int) and not isinstance(t, bool) and 0 <= t < 2**32 for t in cell)
        ):
            ids.append(cell[0])
            words.extend(cell)
            pairs += 1
        else:
            return None
    if pairs and pairs != len(value):
        return None
    return ids, (words if pairs else None)


def _extra_key(value: object) -> tuple | None:
    """One item of vLLM's untagged per-block extra keys; ``None`` for shapes not modeled."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        return ("text", value)
    if isinstance(value, int):
        return ("number", value) if -(2**63) <= value < 2**63 else None
    if isinstance(value, (bytes, bytearray)):
        return ("blob", bytes(value))
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    ):
        return ("multimodal", value[0], value[1])
    return None


def _extra_keys(value: object) -> list[list[tuple] | None] | None:
    if not isinstance(value, (list, tuple)):
        return None
    per_block: list[list[tuple] | None] = []
    for keys in value:
        if isinstance(keys, (list, tuple)):
            per_block.append(
                [key for key in (_extra_key(item) for item in keys) if key is not None]
            )
        else:
            per_block.append(None)
    return per_block


_TAIL_KEYS = (
    "medium",
    "group_idx",
    "kv_cache_spec_kind",
    "kv_cache_spec_sliding_window",
    "locality",
    "ownership",
    "session_id",
)

# The legacy array layout, slots in the order vLLM's msgspec structs declare
# their fields; trailing defaults are omitted on the wire.
_STORED_SLOTS = (
    "block_hashes",
    "parent_block_hash",
    "token_ids",
    "block_size",
    "lora_id",
    "medium",
    "lora_name",
    "extra_keys",
    "group_idx",
    "kv_cache_spec_kind",
    "kv_cache_spec_sliding_window",
    "locality",
    "ownership",
    "session_id",
)
_REMOVED_SLOTS = ("block_hashes", "medium", "group_idx", "locality", "ownership")


@dataclass
class WireEvent:
    """One decoded event. ``kind`` is the engine's tag, ``"unknown"`` for a tag
    the relay does not convert, or ``"malformed"`` (``missing`` names the field
    that was absent or unreadable)."""

    kind: str
    missing: str | None = None
    block_hashes: list[int] = field(default_factory=list)
    parent_block_hash: int | None = None
    token_ids: list[int] = field(default_factory=list)
    bigrams: bool = False
    bigram_words: list[int] | None = None
    block_size: int = 0
    lora_id: int | None = None
    lora_name: str | None = None
    cache_salt: str | None = None
    extra_keys: list[list[tuple] | None] | None = None
    medium: str | None = None
    group_idx: int | None = None
    kv_cache_spec_kind: str | None = None
    kv_cache_spec_sliding_window: int | None = None
    locality: str | None = None
    ownership: str | None = None
    session_id: str | None = None


@dataclass
class WireBatch:
    ts: float
    events: list[WireEvent]
    dp_rank: int | None


def _event_from_fields(kind: str, raw: dict[str, Any]) -> WireEvent:
    tail = {key: raw.get(key) for key in _TAIL_KEYS}
    tail_parsed = {
        "medium": _text(tail["medium"]),
        "group_idx": _unsigned32(tail["group_idx"]),
        "kv_cache_spec_kind": _text(tail["kv_cache_spec_kind"]),
        "kv_cache_spec_sliding_window": _unsigned32(tail["kv_cache_spec_sliding_window"]),
        "locality": _text(tail["locality"]),
        "ownership": _text(tail["ownership"]),
        "session_id": _text(tail["session_id"]),
    }
    if kind == "BlockStored":
        hashes = _hashes(raw.get("block_hashes"))
        if hashes is None:
            return WireEvent("malformed", "block_hashes")
        tokens = _tokens(raw.get("token_ids"))
        if tokens is None:
            return WireEvent("malformed", "token_ids")
        block_size = _signed(raw.get("block_size"))
        if block_size is None:
            return WireEvent("malformed", "block_size")
        return WireEvent(
            kind,
            block_hashes=hashes,
            parent_block_hash=fold_hash(raw.get("parent_block_hash")),
            token_ids=tokens[0],
            bigrams=tokens[1] is not None,
            bigram_words=tokens[1],
            block_size=block_size,
            lora_id=_signed(raw.get("lora_id")),
            lora_name=_text(raw.get("lora_name")),
            cache_salt=_text(raw.get("cache_salt")),
            extra_keys=_extra_keys(raw.get("extra_keys")),
            **tail_parsed,
        )
    if kind == "BlockRemoved":
        hashes = _hashes(raw.get("block_hashes"))
        if hashes is None:
            return WireEvent("malformed", "block_hashes")
        return WireEvent(kind, block_hashes=hashes, **tail_parsed)
    if kind == "AllBlocksCleared":
        return WireEvent(kind, ownership=tail_parsed["ownership"])
    return WireEvent("unknown")


def parse_event(raw: object) -> WireEvent:
    """A tagged map or a tag-first array as a :class:`WireEvent`."""
    if isinstance(raw, dict):
        kind = raw.get("type")
        if not isinstance(kind, str):
            return WireEvent("malformed", "type")
        return _event_from_fields(kind, raw)
    if isinstance(raw, (list, tuple)):
        if not raw or not isinstance(raw[0], str):
            return WireEvent("malformed", "type")
        kind, slots = raw[0], raw[1:]
        names = {
            "BlockStored": _STORED_SLOTS,
            "BlockRemoved": _REMOVED_SLOTS,
            "AllBlocksCleared": (),
        }.get(kind)
        if names is None:
            return WireEvent("unknown")
        return _event_from_fields(kind, dict(zip(names, slots)))
    return WireEvent("malformed", "event")


_decoder = msgspec.msgpack.Decoder()


def decode_batch(payload: bytes) -> WireBatch:
    """A publisher payload, ``[ts, events, rank]``, leniently decoded.

    Raises ``ValueError`` when the envelope itself is not a batch; a bad event
    inside a good envelope becomes a ``malformed`` event instead.
    """
    raw = _decoder.decode(payload)
    if not isinstance(raw, (list, tuple)) or len(raw) < 2 or not isinstance(raw[1], (list, tuple)):
        raise ValueError("KV event batch is not [ts, events, rank]")
    ts = raw[0] if isinstance(raw[0], (int, float)) and not isinstance(raw[0], bool) else 0.0
    rank = raw[2] if len(raw) > 2 else None
    rank = rank if isinstance(rank, int) and not isinstance(rank, bool) else None
    return WireBatch(ts=float(ts), events=[parse_event(item) for item in raw[1]], dp_rank=rank)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


@dataclass
class Counts:
    """What one stream has forwarded and dropped, by ``DropReason::as_str`` name."""

    forwarded_stored: int = 0
    forwarded_removed: int = 0
    forwarded_cleared: int = 0
    duplicate_stores: int = 0
    bigram_stores: int = 0
    hash_checked: int = 0
    hash_mismatch: int = 0
    hash_unverifiable: int = 0
    dropped: dict[str, int] = field(default_factory=dict)


def tier_of(medium: str | None) -> int | None:
    """The tier a medium names: the device when absent, ``None`` when unknown."""
    if medium is None:
        return common_pb2.KV_CACHE_TIER_DEVICE
    return _TIER_BY_MEDIUM.get(medium.upper())


def _locality_of(locality: str | None) -> int | None:
    if locality is None or locality.upper() == "LOCAL":
        return common_pb2.KV_CACHE_LOCALITY_LOCAL
    return None


def _is_residency_agent(ownership: str | None) -> bool:
    return ownership is not None and ownership.lower() == _RESIDENCY_AGENT


def _salt_from_extra_keys(extra_keys, lora_name: str | None) -> str | None:
    """vLLM's cache salt: the first text key of block 0 that is not the LoRA name."""
    if not extra_keys or not extra_keys[0]:
        return None
    for key in extra_keys[0]:
        if key[0] == "text" and key[1] and key[1] != lora_name:
            return key[1]
    return None


def _extra_key_proto(key: tuple) -> common_pb2.KvBlockExtraKey:
    if key[0] == "text":
        return common_pb2.KvBlockExtraKey(text=key[1])
    if key[0] == "number":
        return common_pb2.KvBlockExtraKey(number=key[1])
    if key[0] == "blob":
        return common_pb2.KvBlockExtraKey(blob=key[1])
    return common_pb2.KvBlockExtraKey(
        multimodal=common_pb2.KvMultimodalKey(identifier=key[1], offset=key[2])
    )


class _RankState:
    __slots__ = ("tiers", "groups")

    def __init__(self) -> None:
        # tier -> engine hash -> ((lora_name, cache_salt), recomputed digest or
        # None, physical copies stored and not yet removed). A record lives
        # until its last copy is removed, so a child stored after one copy
        # went still finds its parent's namespace and digest.
        self.tiers: dict[
            int, dict[int, tuple[tuple[str | None, str | None], bytes | None, int]]
        ] = {}
        # cache group -> whether it is a main-attention group
        self.groups: dict[int, bool] = {}


class Normalizer:
    """Per-stream normalization: the drop rules, the namespace inheritance and
    the counters, as ``kv_wire::Normalizer`` keeps them. ``hash_check`` names
    the engine hash to recompute per store (one of :data:`HASH_CHECKS`), or
    ``None`` for no verification; an unknown name is logged and ignored."""

    def __init__(self, hash_check: str | None = None) -> None:
        self.ranks: dict[int, _RankState] = {}
        self.counts = Counts()
        self._event_id = 0
        self.hash_check = _hash_check_name(hash_check)

    def _drop(self, reason: str, event_id: int) -> None:
        count = self.counts.dropped.get(reason, 0) + 1
        self.counts.dropped[reason] = count
        if count <= 3:
            logger.debug("KV event %d not forwarded: %s", event_id, reason)

    def normalize_batch(
        self, batch: WireBatch, sequence_number: int, dp_rank: int | None = None
    ) -> common_pb2.KvEventBatch:
        """A whole batch as its proto. ``dp_rank`` (the socket's rank) wins over
        the payload's; event ids advance once per event, forwarded or not."""
        rank = batch.dp_rank if dp_rank is None else dp_rank
        proto = common_pb2.KvEventBatch(sequence_number=sequence_number, timestamp=batch.ts)
        if rank is not None:
            proto.dp_rank = rank
        for event in batch.events:
            self._event_id += 1
            converted = self.normalize(event, rank, self._event_id)
            if converted is not None:
                proto.events.append(converted)
        return proto

    def normalize(
        self, event: WireEvent, dp_rank: int | None, event_id: int
    ) -> common_pb2.KvCacheEvent | None:
        rank = -1 if dp_rank is None else dp_rank
        if event.kind == "unknown":
            self._drop("unknown_type", event_id)
            return None
        if event.kind == "malformed":
            logger.debug("KV event %d field unreadable: %s", event_id, event.missing)
            self._drop("malformed", event_id)
            return None
        if event.kind == "AllBlocksCleared":
            if _is_residency_agent(event.ownership):
                self._drop("unsupported_ownership", event_id)
                return None
            self.ranks.pop(rank, None)
            self.counts.forwarded_cleared += 1
            cleared = common_pb2.KvCacheCleared()
            if event.ownership is not None:
                cleared.ownership = event.ownership
            return common_pb2.KvCacheEvent(event_id=event_id, cleared=cleared)
        if event.kind == "BlockStored":
            data = self._stored(event, rank, event_id)
            return None if data is None else common_pb2.KvCacheEvent(event_id=event_id, stored=data)
        if event.kind == "BlockRemoved":
            data = self._removed(event, rank, event_id)
            return (
                None if data is None else common_pb2.KvCacheEvent(event_id=event_id, removed=data)
            )
        self._drop("unknown_type", event_id)
        return None

    def _admit(self, event: WireEvent, rank: int, event_id: int, learn_group: bool):
        """The shared gates: ownership, locality, medium, cache group."""
        if _is_residency_agent(event.ownership):
            self._drop("unsupported_ownership", event_id)
            return None
        locality = _locality_of(event.locality)
        if locality is None:
            self._drop("non_local_locality", event_id)
            return None
        tier = tier_of(event.medium)
        if tier is None:
            self._drop("unknown_medium", event_id)
            return None
        if event.group_idx is not None:
            state = self.ranks.setdefault(rank, _RankState())
            if event.kv_cache_spec_kind is not None:
                main = event.kv_cache_spec_kind in MAIN_ATTENTION_KINDS
                if learn_group:
                    state.groups[event.group_idx] = main
            else:
                # A kind-less event follows its learned group; an unknown
                # group counts as main, as a single-group publisher's does.
                main = state.groups.get(event.group_idx, True)
            if not main:
                self._drop("non_main_attention_group", event_id)
                return None
        return tier, locality

    def _stored(self, event: WireEvent, rank: int, event_id: int):
        admitted = self._admit(event, rank, event_id, learn_group=True)
        if admitted is None:
            return None
        tier, locality = admitted
        if not event.block_hashes or not event.token_ids:
            self._drop("placeholder", event_id)
            return None
        width = event.block_size
        if width <= 0 or width >= 2**31 or len(event.block_hashes) * width != len(event.token_ids):
            self._drop("unaligned_blocks", event_id)
            return None
        seen = set()
        if event.parent_block_hash is not None:
            seen.add(event.parent_block_hash)
        for block_hash in event.block_hashes:
            if block_hash in seen:
                self._drop("self_referencing_hashes", event_id)
                return None
            seen.add(block_hash)

        lora_name = event.lora_name or None
        cache_salt = (event.cache_salt or None) or _salt_from_extra_keys(
            event.extra_keys, lora_name
        )
        blocks_state = self.ranks.setdefault(rank, _RankState()).tiers.setdefault(tier, {})
        if (lora_name is None or cache_salt is None) and event.parent_block_hash is not None:
            parent = blocks_state.get(event.parent_block_hash)
            if parent is not None:
                lora_name = lora_name if lora_name is not None else parent[0][0]
                cache_salt = cache_salt if cache_salt is not None else parent[0][1]
        namespace = (lora_name, cache_salt)
        all_seen = True
        for block_hash in event.block_hashes:
            record = blocks_state.get(block_hash)
            if record is None:
                all_seen = False
                blocks_state[block_hash] = (namespace, None, 1)
            else:
                # A second physical copy: the digest stays, one more copy counted.
                blocks_state[block_hash] = (namespace, record[1], min(record[2] + 1, COPIES_CAP))
        if all_seen:
            self.counts.duplicate_stores += 1
        if event.bigrams:
            self.counts.bigram_stores += 1
        if self.hash_check is not None:
            self._verify_hashes(blocks_state, event, width, lora_name, cache_salt)

        cache_level = _CACHE_LEVEL[tier]
        extra_keys = event.extra_keys or []
        blocks = []
        for index, block_hash in enumerate(event.block_hashes):
            block = common_pb2.KvBlock(
                block_hash=block_hash,
                token_ids=event.token_ids[index * width : (index + 1) * width],
                block_size=width,
            )
            if event.lora_id is not None:
                block.lora_id = event.lora_id
            if cache_level is not None:
                block.cache_level = cache_level
            keys = extra_keys[index] if index < len(extra_keys) else None
            if keys:
                block.extra_keys.extend(_extra_key_proto(key) for key in keys)
            blocks.append(block)
        stored = common_pb2.KvBlocksStored(blocks=blocks, tier=tier, locality=locality)
        if event.parent_block_hash is not None:
            stored.parent_block_hash = event.parent_block_hash
        for name, value in (
            ("medium", event.medium),
            ("group_idx", event.group_idx),
            ("kv_cache_spec_kind", event.kv_cache_spec_kind),
            ("kv_cache_spec_sliding_window", event.kv_cache_spec_sliding_window),
            ("ownership", event.ownership),
            ("session_id", event.session_id),
            ("lora_name", lora_name),
            ("cache_salt", cache_salt),
        ):
            if value is not None:
                setattr(stored, name, value)
        self.counts.forwarded_stored += 1
        return stored

    def _removed(self, event: WireEvent, rank: int, event_id: int):
        admitted = self._admit(event, rank, event_id, learn_group=False)
        if admitted is None:
            return None
        tier, locality = admitted
        state = self.ranks.get(rank)
        if state is not None:
            blocks_state = state.tiers.get(tier)
            if blocks_state:
                # One physical copy goes; the record goes with the last.
                for block_hash in event.block_hashes:
                    record = blocks_state.get(block_hash)
                    if record is None:
                        continue
                    if record[2] <= 1:
                        del blocks_state[block_hash]
                    else:
                        blocks_state[block_hash] = (record[0], record[1], record[2] - 1)
        removed = common_pb2.KvBlocksRemoved(
            block_hashes=event.block_hashes, tier=tier, locality=locality
        )
        cache_level = _CACHE_LEVEL[tier]
        if cache_level is not None:
            removed.cache_level = cache_level
        for name, value in (
            ("medium", event.medium),
            ("group_idx", event.group_idx),
            ("ownership", event.ownership),
        ):
            if value is not None:
                setattr(removed, name, value)
        self.counts.forwarded_removed += 1
        return removed

    def _verify_hashes(self, blocks_state, event: WireEvent, width: int, lora_name, cache_salt):
        """Rehash the store's blocks the worker's way and count the outcome;
        what is forwarded never changes. Digests stay on the records so
        children can chain on them."""
        check = self.hash_check
        blocks = len(event.block_hashes)
        if event.parent_block_hash is not None:
            record = blocks_state.get(event.parent_block_hash)
            prior = record[1] if record is not None else None
            if prior is None:
                self.counts.hash_unverifiable += blocks
                return
        elif check == "sglang" and cache_salt:
            prior = sglang_salt_seed(cache_salt)
        else:
            prior = None  # vllm_block applies NONE_HASH itself
        if check == "vllm-sha256-cbor" and (
            lora_name is not None or event.lora_id is not None or event.bigram_words is not None
        ):
            self.counts.hash_unverifiable += blocks
            return
        extra_keys = event.extra_keys or []
        for index, block_hash in enumerate(event.block_hashes):
            tokens = event.token_ids[index * width : (index + 1) * width]
            if check == "sglang":
                words = (
                    event.bigram_words[index * 2 * width : (index + 1) * 2 * width]
                    if event.bigram_words is not None
                    else tokens
                )
                digest = sglang_page(prior, words)
                expected = sglang_event_int(digest)
            else:
                keys = _vllm_hash_keys(
                    extra_keys[index] if index < len(extra_keys) else None, index
                )
                if keys is _UNVERIFIABLE:
                    self.counts.hash_unverifiable += blocks - index
                    return
                digest = vllm_block(prior, tokens, keys)
                expected = vllm_event_int(digest)
            self.counts.hash_checked += 1
            if expected != block_hash:
                self.counts.hash_mismatch += 1
            record = blocks_state.get(block_hash)
            if record is not None:
                blocks_state[block_hash] = (record[0], digest, record[2])
            prior = digest


# ---------------------------------------------------------------------------
# Engine-exact hashes (verification only; the index uses SMG content hashes)
# ---------------------------------------------------------------------------

HASH_CHECK_ENV = "SMG_KV_EVENT_HASH_CHECK"
HASH_CHECKS = ("sglang", "vllm-sha256-cbor")
_UNVERIFIABLE = object()


def _hash_check_name(value: str | None) -> str | None:
    if value is None:
        return None
    name = value.strip().lower().replace("_", "-")
    if not name:
        return None
    if name not in HASH_CHECKS:
        logger.warning("%s names no known engine hash (%r); check off", HASH_CHECK_ENV, value)
        return None
    return name


def sglang_salt_seed(cache_salt: str) -> bytes:
    """The chain seed of a salted SGLang request."""
    return hashlib.sha256(b"sglang-cache-salt-v1\0" + cache_salt.encode("utf-8")).digest()


def sglang_page(prior: bytes | None, words) -> bytes:
    """One SGLang page: the prior digest (if any), then each word as four
    little-endian bytes (both words of every bigram under Eagle hashing)."""
    hasher = hashlib.sha256()
    if prior is not None:
        hasher.update(prior)
    for word in words:
        hasher.update(int(word).to_bytes(4, "little"))
    return hasher.digest()


def sglang_event_int(digest: bytes) -> int:
    """The integer SGLang publishes: the first eight digest bytes, big-endian, signed."""
    return int.from_bytes(digest[:8], "big", signed=True)


def sglang_chain(tokens, page_size: int, prior: bytes | None = None) -> list[tuple[bytes, int]]:
    """Every full page of ``tokens`` chained from ``prior``: (digest, published int)."""
    out = []
    if page_size <= 0:
        return out
    for start in range(0, len(tokens) - page_size + 1, page_size):
        prior = sglang_page(prior, tokens[start : start + page_size])
        out.append((prior, sglang_event_int(prior)))
    return out


def _cbor_head(out: bytearray, major: int, value: int) -> None:
    major <<= 5
    if value < 24:
        out.append(major | value)
    elif value <= 0xFF:
        out += bytes((major | 24, value))
    elif value <= 0xFFFF:
        out += bytes((major | 25,)) + value.to_bytes(2, "big")
    elif value <= 0xFFFF_FFFF:
        out += bytes((major | 26,)) + value.to_bytes(4, "big")
    else:
        out += bytes((major | 27,)) + value.to_bytes(8, "big")


def _cbor(out: bytearray, value) -> None:
    """Canonical CBOR for the shapes vLLM's hash input uses (what ``cbor2.dumps(
    value, canonical=True)`` emits for them): ints, bytes, text, tuples/lists, None."""
    if value is None:
        out.append(0xF6)
    elif isinstance(value, bool):
        raise TypeError("booleans are not part of a block hash input")
    elif isinstance(value, int):
        if value >= 0:
            _cbor_head(out, 0, value)
        else:
            _cbor_head(out, 1, -1 - value)
    elif isinstance(value, (bytes, bytearray)):
        _cbor_head(out, 2, len(value))
        out += value
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        _cbor_head(out, 3, len(encoded))
        out += encoded
    elif isinstance(value, (list, tuple)):
        _cbor_head(out, 4, len(value))
        for item in value:
            _cbor(out, item)
    else:
        raise TypeError(f"unsupported hash input {type(value).__name__}")


_VLLM_NONE_HASH_SEED = "vllm-none-hash"


def vllm_none_hash() -> bytes:
    """``NONE_HASH``: sha256 of the CBOR text ``vllm-none-hash`` (PYTHONHASHSEED unset)."""
    out = bytearray()
    _cbor(out, _VLLM_NONE_HASH_SEED)
    return hashlib.sha256(out).digest()


def vllm_block(parent: bytes | None, tokens, extra_keys=None) -> bytes:
    """One vLLM ``sha256_cbor`` block hash: ``sha256(cbor([parent or NONE,
    [tokens...], extra_keys or None]))`` with the keys as the tagged tuples the
    engine folds in (``("lora", name, path)``, ``("mm", identifier, offset)``,
    ``("cache_salt", salt)``, ``("prompt_embeds", digest)``)."""
    out = bytearray()
    _cbor(
        out,
        (
            parent if parent is not None else vllm_none_hash(),
            tuple(int(t) for t in tokens),
            tuple(extra_keys) if extra_keys else None,
        ),
    )
    return hashlib.sha256(out).digest()


def vllm_event_int(digest: bytes) -> int:
    """The integer vLLM publishes: the low 64 bits, carried as the same bits in an int64."""
    return fold_hash(digest)


def vllm_chain(tokens, block_size: int, parent: bytes | None = None) -> list[tuple[bytes, int]]:
    """Every full block of ``tokens`` chained from ``parent`` with no extra keys."""
    out = []
    if block_size <= 0:
        return out
    for start in range(0, len(tokens) - block_size + 1, block_size):
        parent = vllm_block(parent, tokens[start : start + block_size])
        out.append((parent, vllm_event_int(parent)))
    return out


def _vllm_hash_keys(keys, index: int):
    """vLLM's untagged event keys as the tagged keys inside the hash: block 0's
    text is the cache salt (a LoRA request was excluded before), a pair is a
    multimodal item, bytes are a prompt-embeddings digest."""
    if not keys:
        return None
    tagged = []
    for key in keys:
        if key[0] == "text" and index == 0:
            tagged.append(("cache_salt", key[1]))
        elif key[0] == "multimodal":
            tagged.append(("mm", key[1], key[2]))
        elif key[0] == "blob":
            tagged.append(("prompt_embeds", key[1]))
        else:
            return _UNVERIFIABLE
    return tagged


# ---------------------------------------------------------------------------
# Endpoints and ranks
# ---------------------------------------------------------------------------


def endpoint_for_rank(endpoint: str, dp_rank: int) -> str:
    """A connectable SUB address for rank ``dp_rank`` of a publisher at ``endpoint``.

    Bind wildcards (``*``, ``0.0.0.0``) become loopback; tcp ports are offset
    by the rank, as both engines' publishers offset theirs; other transports
    get no port arithmetic.
    """
    resolved = endpoint.replace("*", "127.0.0.1").replace("0.0.0.0", "127.0.0.1")
    if resolved.startswith("tcp://") and dp_rank:
        host, sep, port = resolved.rpartition(":")
        if sep and port.isdigit() and int(port) != 0:
            return f"{host}:{int(port) + dp_rank}"
    return resolved


@dataclass(frozen=True)
class RankSource:
    """One rank's publisher: where to subscribe and where to ask for replay."""

    rank: int
    endpoint: str
    replay_endpoint: str | None = None


def rank_sources(config: object, ranks: Iterable[int]) -> list[RankSource]:
    """Sources for ``ranks`` of a publisher configured by ``config`` (an
    object with ``endpoint`` and optional ``replay_endpoint``)."""
    endpoint = str(getattr(config, "endpoint", "") or "")
    replay = getattr(config, "replay_endpoint", None)
    return [
        RankSource(
            rank=rank,
            endpoint=endpoint_for_rank(endpoint, rank),
            replay_endpoint=endpoint_for_rank(str(replay), rank) if replay else None,
        )
        for rank in ranks
    ]


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class RelayFailure(Exception):
    """The stream can no longer be trusted; the gateway must clear and resubscribe."""


async def replay_frames(
    endpoint: str,
    start: int,
    *,
    timeout: float,
    context: zmq.asyncio.Context | None = None,
) -> AsyncIterator[tuple[int, bytes]]:
    """Ask a publisher's replay ROUTER for every buffered batch from ``start``.

    The request is ``[b"", start as 8-byte big-endian]`` on a DEALER. Replies
    are ``[b"", seq, payload]`` (SGLang) or ``[b"", topic, seq, payload]``
    (vLLM) and end with the all-ones sequence. The first reply must be
    ``start`` itself and the rest contiguous; anything else, or no reply in
    ``timeout`` seconds, raises :class:`RelayFailure`.
    """
    import zmq
    import zmq.asyncio

    ctx = context or zmq.asyncio.Context.instance()
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.LINGER, 0)
    try:
        dealer.connect(endpoint)
        await dealer.send_multipart([b"", start.to_bytes(8, "big")])
        expected = start
        while True:
            if not await dealer.poll(timeout=int(timeout * 1000)):
                raise RelayFailure(f"KV event replay from {endpoint} timed out at {expected}")
            frames = await dealer.recv_multipart()
            if len(frames) == 3:
                seq_bytes, payload = frames[1], frames[2]
            elif len(frames) == 4:
                seq_bytes, payload = frames[2], frames[3]
            else:
                raise RelayFailure(f"malformed KV event replay reply ({len(frames)} frames)")
            if len(seq_bytes) != 8:
                raise RelayFailure("malformed KV event replay sequence")
            if seq_bytes == _END_SEQ:
                if expected == start:
                    raise RelayFailure(
                        f"KV event replay history at {endpoint} no longer holds {start}"
                    )
                return
            seq = int.from_bytes(seq_bytes, "big")
            if seq != expected:
                raise RelayFailure(
                    f"KV event replay from {endpoint} is not contiguous: expected {expected}, got {seq}"
                )
            yield seq, payload
            expected += 1
    except zmq.ZMQError as error:
        raise RelayFailure(f"KV event replay transport failed: {error}") from error
    finally:
        dealer.close(linger=0)


class _RankStream:
    __slots__ = ("source", "socket", "cursor", "replayed")

    def __init__(self, source: RankSource, socket: zmq.asyncio.Socket) -> None:
        self.source = source
        self.socket = socket
        self.cursor: int | None = None
        # The sequence range the last gap replay forwarded, while it is still
        # open. A replay usually runs past the live batch that exposed the gap,
        # and the live batches the publisher sent meanwhile are still queued
        # on the SUB socket: they arrive below the cursor the replay left and
        # are duplicates of what it forwarded, not a restarted publisher. The
        # queue drains in order, so the first live batch past the range closes
        # it; a sequence inside it after that can only be a restart.
        self.replayed: tuple[int, int] | None = None

    def covered_by_replay(self, seq: int) -> bool:
        return self.replayed is not None and self.replayed[0] <= seq <= self.replayed[1]


async def relay(
    sources: list[RankSource],
    engine: Engine,
    start_sequence_number: int,
    context: grpc.aio.ServicerContext,
    *,
    topic: str = "",
    hwm: int | None = None,
    recv_timeout: float = 1.0,
    replay_timeout: float = 5.0,
    zmq_context: zmq.asyncio.Context | None = None,
    normalizer: Normalizer | None = None,
    hash_check: str | None = None,
    on_counts: Callable[[Counts], Awaitable[None] | None] | None = None,
) -> AsyncIterator[common_pb2.KvEventBatch]:
    """Relay every rank in ``sources`` into one ``KvEventBatch`` stream.

    See the module docstring for the contract. ``engine`` is informational
    (both replay reply framings are accepted); ``hash_check`` (default: the
    ``SMG_KV_EVENT_HASH_CHECK`` environment variable) turns engine-hash
    verification on; ``on_counts`` is called with the stream's counters when
    it ends.
    """
    import zmq
    import zmq.asyncio

    if start_sequence_number:
        await context.abort(
            grpc.StatusCode.OUT_OF_RANGE,
            "KV event replay cursors are per subscription; resubscribe with zero and rebuild",
        )
        return
    if not sources:
        await context.abort(
            grpc.StatusCode.UNIMPLEMENTED, "KV cache events have no publisher to relay"
        )
        return

    ctx = zmq_context or zmq.asyncio.Context.instance()
    if normalizer is None:
        normalizer = Normalizer(
            hash_check if hash_check is not None else os.environ.get(HASH_CHECK_ENV)
        )
    streams: dict[Any, _RankStream] = {}
    poller = zmq.asyncio.Poller()
    relay_seq = 0
    sent_headers = False

    def forward(payload: bytes, stream: _RankStream, seq: int, kind: str):
        nonlocal relay_seq
        try:
            batch = decode_batch(payload)
        except Exception as error:  # noqa: BLE001 - one bad payload must not end the stream
            logger.warning(
                "Failed to decode %s KV event batch %d from rank %d: %s",
                kind,
                seq,
                stream.source.rank,
                error,
            )
            return None
        relay_seq += 1
        return normalizer.normalize_batch(batch, relay_seq, stream.source.rank)

    try:
        for source in sources:
            socket = ctx.socket(zmq.SUB)
            if hwm is not None:
                socket.setsockopt(zmq.RCVHWM, int(hwm))
            socket.subscribe(topic.encode("utf-8"))
            socket.connect(source.endpoint)
            streams[socket] = _RankStream(source, socket)
            poller.register(socket, zmq.POLLIN)
        logger.info(
            "SubscribeKvEvents: subscribed to %d %s rank(s): %s",
            len(sources),
            engine.value,
            ", ".join(source.endpoint for source in sources),
        )
        await context.send_initial_metadata(())
        sent_headers = True

        while not context.cancelled():
            ready = await poller.poll(timeout=int(recv_timeout * 1000))
            for socket, _ in ready:
                stream = streams[socket]
                frames = await socket.recv_multipart()
                if len(frames) < 3 or len(frames[1]) != 8:
                    continue
                seq = int.from_bytes(frames[1], "big")
                cursor = stream.cursor
                if cursor is not None:
                    if seq == cursor or stream.covered_by_replay(seq):
                        continue  # a duplicate of what a replay already forwarded
                    if seq < cursor:
                        raise RelayFailure(
                            f"publisher of rank {stream.source.rank} restarted its sequence at "
                            f"{seq} after {cursor}"
                        )
                    if seq > cursor + 1:
                        replay_endpoint = stream.source.replay_endpoint
                        if replay_endpoint is None:
                            raise RelayFailure(
                                f"rank {stream.source.rank} skipped from {cursor} to {seq} and has "
                                "no replay endpoint"
                            )
                        logger.warning(
                            "KV events of rank %d skipped from %d to %d; replaying",
                            stream.source.rank,
                            cursor,
                            seq,
                        )
                        async for replayed_seq, payload in replay_frames(
                            replay_endpoint, cursor + 1, timeout=replay_timeout, context=ctx
                        ):
                            proto = forward(payload, stream, replayed_seq, "replayed")
                            stream.cursor = replayed_seq
                            stream.replayed = (cursor + 1, replayed_seq)
                            if proto is not None:
                                yield proto
                        if stream.cursor < seq - 1:
                            raise RelayFailure(
                                f"KV event replay of rank {stream.source.rank} ended at "
                                f"{stream.cursor}, before {seq - 1}"
                            )
                        if seq <= stream.cursor:
                            continue  # the replay ran past the live message
                stream.cursor = seq
                if stream.replayed is not None and seq > stream.replayed[1]:
                    stream.replayed = None  # the live stream is past the replay's range
                proto = forward(frames[2], stream, seq, "live")
                if proto is not None:
                    yield proto
    except asyncio.CancelledError:
        pass
    except RelayFailure as error:
        logger.warning("SubscribeKvEvents: %s; the gateway will clear and resubscribe", error)
        await context.abort(
            grpc.StatusCode.DATA_LOSS if sent_headers else grpc.StatusCode.OUT_OF_RANGE, str(error)
        )
    finally:
        for socket in streams:
            socket.close(linger=0)
        if on_counts is not None:
            result = on_counts(normalizer.counts)
            if asyncio.iscoroutine(result):
                await result
        logger.info("SubscribeKvEvents: stream closed; %s", normalizer.counts)
