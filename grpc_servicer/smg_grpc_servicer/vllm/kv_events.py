"""vLLM-specific KV-events config resolution and per-rank publisher discovery.

The relay itself (lenient decoding, normalization, per-rank cursors and
replay) is the engine-neutral ``smg_grpc_servicer.kv_relay``. The older
single-rank helpers in ``smg_grpc_servicer.kv_events`` are re-exported here
for backwards compatibility (existing imports and tests reference them via
this module).
"""

import logging

from smg_grpc_servicer.kv_events import (
    convert_batch,
    convert_event,
    endpoint_for_rank,
    stream_kv_events,
    to_int64,
)
from smg_grpc_servicer.kv_relay import RankSource, rank_sources

__all__ = [
    "convert_batch",
    "convert_event",
    "endpoint_for_rank",
    "rank_sources_for",
    "resolve_kv_events_config",
    "stream_kv_events",
    "to_int64",
]

logger = logging.getLogger(__name__)


def resolve_kv_events_config(engine: object):
    """Return the vLLM KVEventsConfig iff ZMQ event publishing is enabled, else None.

    Reads ``engine.vllm_config.kv_events_config`` via getattr so it tolerates any
    vLLM version and is testable with a fake engine.
    """
    vllm_config = getattr(engine, "vllm_config", None)
    cfg = getattr(vllm_config, "kv_events_config", None)
    if cfg is None:
        return None
    if not getattr(cfg, "enable_kv_cache_events", False):
        logger.info("vLLM KV cache events not enabled; SubscribeKvEvents disabled")
        return None
    if getattr(cfg, "publisher", None) != "zmq":
        logger.info(
            "vLLM KV events publisher is %r, not 'zmq'; SubscribeKvEvents disabled",
            getattr(cfg, "publisher", None),
        )
        return None
    logger.info("vLLM KV events enabled: endpoint=%s", getattr(cfg, "endpoint", "?"))
    return cfg


def rank_sources_for(config: object, engine: object) -> list[RankSource]:
    """One KV-event source per data-parallel rank.

    vLLM's engine client reports each ready engine core's publisher config
    (``get_kv_event_sources``, keyed by DP rank; the endpoints in it are the
    rank's own, already offset). Older engines without it get
    ``data_parallel_size`` ranks offset from the base endpoint, as the
    publishers offset theirs.
    """
    reported: dict = {}
    getter = getattr(engine, "get_kv_event_sources", None)
    if callable(getter):
        try:
            reported = dict(getter() or {})
        except Exception as error:  # noqa: BLE001 - discovery is best effort
            logger.warning("get_kv_event_sources failed; deriving ranks from the config: %s", error)
            reported = {}
    if reported:
        sources = []
        for rank, rank_config in sorted(reported.items()):
            if not isinstance(rank, int):
                continue
            endpoint = str(getattr(rank_config, "endpoint", "") or "")
            replay = getattr(rank_config, "replay_endpoint", None)
            if not endpoint:
                continue
            sources.append(
                RankSource(
                    rank=rank,
                    endpoint=endpoint_for_rank(endpoint, 0),
                    replay_endpoint=endpoint_for_rank(str(replay), 0) if replay else None,
                )
            )
        if sources:
            return sources
    parallel = getattr(getattr(engine, "vllm_config", None), "parallel_config", None)
    dp_size = getattr(parallel, "data_parallel_size", None)
    dp_size = int(dp_size) if isinstance(dp_size, int) and dp_size > 0 else 1
    return rank_sources(config, range(dp_size))
