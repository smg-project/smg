"""TokenSpeed KV-event transport: the config resolution, and every attention
data-parallel rank's publisher relayed through :mod:`smg_grpc_servicer.kv_relay`,
independent of engine imports.

TokenSpeed's ``--kv-events-config`` is a raw JSON string on ``ServerArgs``
(``server_args.kv_events_config``), unlike vLLM's already-parsed config object.
We parse it with stdlib ``json`` (not TokenSpeed's pydantic ``KVEventsConfig``)
so this resolver stays unit-testable without TokenSpeed installed; the defaults
below mirror ``tokenspeed.runtime.pd.kv_events.KVEventsConfig``.

Under attention data parallelism the engine runs one publisher per attention
rank (each rank's event loop creates its own with ``attn_dp_rank=rank``), on
``endpoint_port + rank`` with its own sequence counter and, when configured,
its own replay ROUTER on ``replay_endpoint_port + rank``; the rank is what the
batches carry (``attn_dp_rank``). :func:`dp_rank_count` reads that width
(``mapping.attn.dp_size``) off the server args, the figure the Rust servicer's
launcher reports as its ``data_parallel_size``.
"""

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

import grpc
from smg_grpc_proto.generated import common_pb2

from smg_grpc_servicer.kv_relay import Engine, RankSource, rank_sources, relay
from smg_grpc_servicer.load_record import LoadSource
from smg_grpc_servicer.tokenspeed.config_view import EngineConfigView

__all__ = [
    "ResolvedKvEventsConfig",
    "dp_rank_count",
    "resolve_kv_events_config",
    "sources",
    "subscribe_kv_events",
]

logger = logging.getLogger(__name__)

# Defaults mirror tokenspeed.runtime.pd.kv_events.KVEventsConfig.
_DEFAULT_ENDPOINT = "tcp://*:5557"
_DEFAULT_TOPIC = ""


@dataclass(frozen=True)
class ResolvedKvEventsConfig:
    """The subset of KVEventsConfig the bridge needs to open a SUB socket, and
    the replay ROUTER endpoint (empty when the publisher runs none) the relay
    asks for gaps and for the batches published before its subscription joined."""

    endpoint: str
    topic: str
    replay_endpoint: str = ""


def resolve_kv_events_config(server_args: object) -> ResolvedKvEventsConfig | None:
    """Return the resolved ZMQ endpoint iff TokenSpeed KV events are enabled, else None.

    Reads ``server_args.kv_events_config`` (a JSON string). Returns ``None`` —
    leaving ``SubscribeKvEvents`` to report UNIMPLEMENTED — when events are
    disabled, the publisher is not ``zmq``, or the config is absent/malformed.
    """
    raw = EngineConfigView(server_args).server_args.kv_events_config
    if not raw:
        return None

    try:
        cfg = json.loads(raw)
    except (TypeError, ValueError) as e:
        logger.warning("Could not parse --kv-events-config %r: %s", raw, e)
        return None
    if not isinstance(cfg, dict):
        logger.warning("--kv-events-config did not decode to an object: %r", raw)
        return None

    if not cfg.get("enable_kv_cache_events", False):
        logger.info("TokenSpeed KV cache events not enabled; SubscribeKvEvents disabled")
        return None

    # publisher unset → "zmq" when enabled (matches EventPublisherFactory.create).
    publisher = cfg.get("publisher")
    if publisher is None:
        publisher = "zmq"
    if publisher != "zmq":
        logger.info(
            "TokenSpeed KV events publisher is %r, not 'zmq'; SubscribeKvEvents disabled",
            publisher,
        )
        return None

    endpoint = cfg.get("endpoint") or _DEFAULT_ENDPOINT
    topic = cfg.get("topic")
    if topic is None:
        topic = _DEFAULT_TOPIC
    # Guard against non-string values from arbitrary JSON; otherwise they fail
    # later (e.g. topic.encode()) as opaque INTERNAL errors at subscribe time.
    if not isinstance(endpoint, str) or not isinstance(topic, str):
        logger.warning(
            "TokenSpeed kv-events endpoint/topic must be strings (got endpoint=%r topic=%r); "
            "SubscribeKvEvents disabled",
            endpoint,
            topic,
        )
        return None
    replay_endpoint = cfg.get("replay_endpoint") or ""
    if not isinstance(replay_endpoint, str):
        logger.warning(
            "TokenSpeed kv-events replay_endpoint must be a string (got %r); replay disabled",
            replay_endpoint,
        )
        replay_endpoint = ""
    logger.info(
        "TokenSpeed KV events enabled: endpoint=%s replay_endpoint=%s",
        endpoint,
        replay_endpoint or "(none)",
    )
    return ResolvedKvEventsConfig(endpoint=endpoint, topic=topic, replay_endpoint=replay_endpoint)


def dp_rank_count(server_args: object) -> int:
    """How many KV-event publishers this engine runs: one per attention
    data-parallel rank (``mapping.attn.dp_size``), 1 when the args carry none."""
    return EngineConfigView(server_args).server_args.attention_dp_size or 1


def sources(config: object, server_args: object) -> list[RankSource]:
    """One source per rank of the publisher ``config`` describes."""
    return rank_sources(config, range(dp_rank_count(server_args)))


async def subscribe_kv_events(
    config: object,
    server_args: object,
    start_sequence_number: int,
    context: grpc.aio.ServicerContext,
    *,
    load_source: LoadSource | None = None,
) -> AsyncIterator[common_pb2.KvEventBatch]:
    """Relay every rank's events, with the servicer's load record on each
    when ``load_source`` is given; see :func:`smg_grpc_servicer.kv_relay.relay`."""
    async for batch in relay(
        sources(config, server_args),
        Engine.TOKENSPEED,
        start_sequence_number,
        context,
        topic=str(getattr(config, "topic", "") or ""),
        hwm=getattr(config, "hwm", None),
        load_source=load_source,
    ):
        yield batch
