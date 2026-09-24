"""SGLang KV-event transport, independent of engine imports."""

import logging
from collections.abc import AsyncIterator, Callable

import grpc
import zmq
import zmq.asyncio
from smg_grpc_proto.generated import common_pb2

from smg_grpc_servicer.kv_events import endpoint_for_rank

logger = logging.getLogger(__name__)


async def subscribe_kv_events(
    config: object,
    start_sequence_number: int,
    context: grpc.aio.ServicerContext,
    decode: Callable[[bytes], object],
    convert: Callable[[object, int], common_pb2.KvEventBatch],
) -> AsyncIterator[common_pb2.KvEventBatch]:
    """Forward live events; reject resume cursors we cannot replay.

    OUT_OF_RANGE tells the gateway to discard this worker's stale mappings
    and resubscribe with zero. A zero cursor starts a fresh live subscription,
    not a complete cache snapshot; knowledge rebuilds as events arrive.
    """
    if start_sequence_number:
        await context.abort(
            grpc.StatusCode.OUT_OF_RANGE,
            "KV event replay is unavailable; resubscribe with zero for live events",
        )
        return

    # Each DP rank has independent sequence numbers. Keep the existing rank-0
    # subscription until the gateway supports per-rank indexes.
    endpoint = endpoint_for_rank(config.endpoint, 0)
    sub = zmq.asyncio.Context.instance().socket(zmq.SUB)
    try:
        sub.subscribe(config.topic.encode("utf-8"))
        sub.connect(endpoint)
        logger.info("SubscribeKvEvents: connected to ZMQ endpoint %s", endpoint)
        await context.send_initial_metadata(())
        while not context.cancelled():
            # Cancelling recv_multipart on an idle timeout can lose a message.
            if not await sub.poll(timeout=1000):
                continue
            frames = await sub.recv_multipart()
            if len(frames) < 3:
                continue
            seq = int.from_bytes(frames[1], "big")
            try:
                batch = decode(frames[2])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to decode KV event batch: %s", exc)
                continue
            yield convert(batch, seq)
    finally:
        sub.close(linger=0)
        logger.info("SubscribeKvEvents: stream closed")
