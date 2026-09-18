"""Preserve received vLLM group reports without guessing sparse token offsets."""

from __future__ import annotations

import logging

from smg_grpc_proto.generated import common_pb2

logger = logging.getLogger(__name__)
_GROUP_FIELDS = {"group_idx", "kv_cache_spec_kind", "kv_cache_spec_sliding_window", "extra_keys"}


def _uint(value: object, *, positive: bool = False) -> int:
    if type(value) is not int or not int(positive) <= value < 2**32:
        raise ValueError("Invalid unsigned event field")
    return value


def _hash(value: object) -> bytes:
    if type(value) is bytes and value:
        return value
    if type(value) is int and 0 <= value < 2**64:
        return value.to_bytes(8, "big")
    raise ValueError("Invalid native block hash")


class GroupEventConverter:
    """Convert GPU-local reports; repeated stores remain repeated reports."""

    def _event(self, event: object) -> common_pb2.KvGroupEvent | None:
        name = type(event).__name__
        if name == "AllBlocksCleared":
            return common_pb2.KvGroupEvent(operation=common_pb2.KvGroupEvent.CLEAR)
        if name not in ("BlockStored", "BlockRemoved"):
            raise ValueError(f"Unknown KV event type: {name}")
        if (
            event.medium != "GPU"
            or getattr(event, "locality", None) not in (None, "LOCAL")
            or getattr(event, "ownership", None) is not None
        ):
            return None
        if not isinstance(event.block_hashes, list):
            raise ValueError("Malformed block hashes")
        converted = common_pb2.KvGroupEvent(
            operation=(
                common_pb2.KvGroupEvent.STORE
                if name == "BlockStored"
                else common_pb2.KvGroupEvent.REMOVE
            ),
            group_id=_uint(event.group_idx),
            block_hashes=[_hash(h) for h in event.block_hashes],
        )
        if name == "BlockStored":
            if not isinstance(event.kv_cache_spec_kind, str) or not event.kv_cache_spec_kind:
                raise ValueError("Missing cache-spec kind")
            converted.kind = event.kv_cache_spec_kind
            converted.sliding_window = (
                0
                if event.kv_cache_spec_sliding_window is None
                else _uint(event.kv_cache_spec_sliding_window)
            )
            converted.parent_block_hash = (
                b"" if event.parent_block_hash is None else _hash(event.parent_block_hash)
            )
            converted.block_size = _uint(event.block_size, positive=True)
            converted.token_ids.extend(_uint(token) for token in event.token_ids)
            extras = event.extra_keys
            if extras is not None and (
                not isinstance(extras, list)
                or len(extras) != len(event.block_hashes)
                or any(key is not None and not isinstance(key, tuple) for key in extras)
            ):
                raise ValueError("Malformed extra hash inputs")
            converted.tokens_matchable = (
                event.lora_id is None
                and event.lora_name is None
                and (extras is None or all(key is None for key in extras))
            )
        return converted

    def convert_batch(self, raw_batch: object, seq_num: int, event_id_start: int):
        batch = common_pb2.KvEventBatch(
            sequence_number=seq_num, timestamp=raw_batch.ts, group_events_enabled=True
        )
        if raw_batch.data_parallel_rank is not None:
            batch.dp_rank = raw_batch.data_parallel_rank
        for event in raw_batch.events:
            try:
                converted = self._event(event)
            except (ValueError, AttributeError, TypeError) as error:
                logger.warning("Invalid group KV event: %s", error)
                converted = common_pb2.KvGroupEvent(operation=common_pb2.KvGroupEvent.INVALID)
            if converted is not None:
                batch.group_events.append(converted)
        return batch, event_id_start + len(raw_batch.events)


def resolve_group_event_converter(block_stored_type: type) -> GroupEventConverter | None:
    """Select by installed event schema, without consulting engine configuration."""
    if _GROUP_FIELDS.issubset(getattr(block_stored_type, "__struct_fields__", ())):
        return GroupEventConverter()
    return None
