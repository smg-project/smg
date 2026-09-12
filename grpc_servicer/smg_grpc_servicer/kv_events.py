"""Engine-neutral KV-cache-event → proto conversion and ZMQ streaming.

Shared by every engine bridge (vLLM, TokenSpeed, ...). Imports only stdlib +
the generated proto, and dispatches engine events by class name (BlockStored /
BlockRemoved / AllBlocksCleared), so it needs no engine import and is
unit-testable without any engine installed.

Each engine package keeps its own ``resolve_kv_events_config`` (the only
engine-specific seam); everything here is wire-format-only.
"""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable

from smg_grpc_proto.generated import common_pb2

logger = logging.getLogger(__name__)

_U64_MASK = 0xFFFFFFFFFFFFFFFF
_I64_SIGN_BIT = 0x8000000000000000
_U64_MODULUS = 0x10000000000000000


def to_int64(value: int | bytes) -> int:
    """Reduce an engine block hash to a signed int64 for the proto block_hash field.

    An engine's block hash may be ``int | bytes`` (sha256 bytes when int hashes
    are disabled); bytes are read big-endian. SMG uses the hash only as a node
    identity, so the 64-bit reduction is safe as long as it stays deterministic.
    """
    if isinstance(value, (bytes, bytearray)):
        value = int.from_bytes(value, "big")
    masked = value & _U64_MASK
    if masked >= _I64_SIGN_BIT:
        masked -= _U64_MODULUS
    return masked


def endpoint_for_rank(endpoint: str, dp_rank: int) -> str:
    """Resolve a KV-events PUB endpoint to a connectable SUB address.

    Bind wildcards (``*``, ``0.0.0.0``) are rewritten to ``127.0.0.1`` (the
    latter is not connectable on macOS/Windows). For data-parallel deployments
    each rank publishes on ``base_port + dp_rank``; non-tcp endpoints (ipc://,
    inproc://) get the wildcard substituted but no port arithmetic.
    """
    resolved = endpoint.replace("*", "127.0.0.1").replace("0.0.0.0", "127.0.0.1")
    if resolved.startswith("tcp://") and dp_rank:
        host, sep, port = resolved.rpartition(":")
        if sep and port.isdigit():
            return f"{host}:{int(port) + dp_rank}"
    return resolved


def convert_event(event: object, event_id: int) -> common_pb2.KvCacheEvent | None:
    """Convert a decoded event, skipping unknown types or unaligned block stores."""
    name = type(event).__name__

    if name == "BlockStored":
        block_size = int(event.block_size)
        # Ordinal slicing is valid only for a dense sequence of complete blocks.
        if block_size <= 0 or len(event.block_hashes) * block_size != len(event.token_ids):
            logger.warning(
                "Skipping BlockStored: %d hashes of block size %d cannot map to %d tokens",
                len(event.block_hashes),
                block_size,
                len(event.token_ids),
            )
            return None
        blocks = []
        for i, block_hash in enumerate(event.block_hashes):
            start = i * block_size
            end = start + block_size
            block = common_pb2.KvBlock(
                block_hash=to_int64(block_hash),
                token_ids=list(event.token_ids[start:end]),
                block_size=block_size,
            )
            lora_id = getattr(event, "lora_id", None)
            if lora_id is not None:
                block.lora_id = to_int64(lora_id)
            blocks.append(block)
        stored = common_pb2.KvBlocksStored(blocks=blocks)
        parent = getattr(event, "parent_block_hash", None)
        if parent is not None:
            stored.parent_block_hash = to_int64(parent)
        return common_pb2.KvCacheEvent(event_id=event_id, stored=stored)

    if name == "BlockRemoved":
        return common_pb2.KvCacheEvent(
            event_id=event_id,
            removed=common_pb2.KvBlocksRemoved(
                block_hashes=[to_int64(h) for h in event.block_hashes]
            ),
        )

    if name == "AllBlocksCleared":
        return common_pb2.KvCacheEvent(event_id=event_id, cleared=common_pb2.KvCacheCleared())

    logger.debug("Unknown KV event type %r, skipping", name)
    return None


def convert_batch(
    raw_batch: object, seq_num: int, event_id_start: int
) -> tuple[common_pb2.KvEventBatch, int]:
    """Convert a decoded engine KVEventBatch to a proto KvEventBatch.

    Returns the proto batch and the new event-id counter. The counter advances
    once per input event (even if unconvertible) so ids stay monotonic.

    The DP rank is read from ``data_parallel_rank`` (vLLM) or ``attn_dp_rank``
    (TokenSpeed); engines that carry neither leave the proto field unset.
    """
    proto = common_pb2.KvEventBatch(sequence_number=seq_num, timestamp=raw_batch.ts)
    dp_rank = getattr(raw_batch, "data_parallel_rank", None)
    if dp_rank is None:
        dp_rank = getattr(raw_batch, "attn_dp_rank", None)
    if dp_rank is not None:
        proto.dp_rank = dp_rank

    event_id = event_id_start
    for event in raw_batch.events:
        event_id += 1
        proto_event = convert_event(event, event_id)
        if proto_event is not None:
            proto.events.append(proto_event)
    return proto, event_id


async def stream_kv_events(
    sub_socket: object,
    decode: Callable[[bytes], object],
    send_initial_metadata: Callable[[], Awaitable[None]],
    is_cancelled: Callable[[], bool],
    *,
    recv_timeout: float = 1.0,
) -> AsyncIterator[common_pb2.KvEventBatch]:
    """Core ZMQ→proto streaming loop, decoupled from any engine and gRPC types.

    Args:
        sub_socket: a connected ``zmq.asyncio`` SUB socket (duck-typed; only
            ``poll()`` and ``recv_multipart()`` are used). The caller owns the
            socket lifecycle (this function never closes it).
        decode: bytes → decoded engine batch (e.g. ``msgspec.msgpack.Decoder(KVEventBatch).decode``).
        send_initial_metadata: awaitable called once before the first recv so the
            gRPC client's ``subscribe_kv_events().await`` resolves promptly.
        is_cancelled: returns True when the RPC is cancelled; loop then exits.
        recv_timeout: poll timeout so cancellation is observed even when idle.

    Yields proto KvEventBatch using the ZMQ publisher's native sequence numbers.
    """
    await send_initial_metadata()
    event_id = 0
    while not is_cancelled():
        # poll() before recv: cancelling a zmq.asyncio recv future does not
        # cancel the in-flight ZMQ recv and can drop an already-dequeued message.
        if not await sub_socket.poll(timeout=int(recv_timeout * 1000)):
            continue
        frames = await sub_socket.recv_multipart()

        # ZMQ multipart: [topic, 8-byte big-endian seq, msgpack payload].
        if len(frames) < 3:
            continue
        zmq_seq = int.from_bytes(frames[1], "big")
        try:
            raw_batch = decode(frames[2])
        except Exception as e:  # noqa: BLE001 - one bad frame must not kill the stream
            logger.warning("Failed to decode KV event batch: %s", e)
            continue

        proto_batch, event_id = convert_batch(raw_batch, zmq_seq, event_id)
        yield proto_batch
