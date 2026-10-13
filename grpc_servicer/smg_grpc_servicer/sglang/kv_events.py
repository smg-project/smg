"""SGLang KV-event transport: every DP rank's publisher relayed through
:mod:`smg_grpc_servicer.kv_relay`, independent of engine imports.

SGLang runs one KV-event publisher per independent KV cache, one per
scheduler: ``dp_size`` of them on the pinned release (under DP attention the
attention-DP ranks are those), ``dp_size x attn_dp_size`` on the newer
releases, which resolve ``--dp-size N --enable-dp-attention`` into
``attn_dp_size=N, dp_size=1`` (the engine runs one of the two widths at 1:
attention DP does not run inside replicas, so the product is its own
``num_dp_ranks``). Each publishes on ``endpoint_port_base + rank`` with its
own sequence counter and, when configured, its own replay ROUTER on
``replay_endpoint_port_base + rank``; the rank is the one the batches carry
(``attn_dp_rank``). This is what ``/server_info.kv_events`` advertises;
in-process the same figures come from the server args and the KV-events
config.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import grpc
from smg_grpc_proto.generated import common_pb2

from smg_grpc_servicer.kv_relay import (
    Engine,
    RankSource,
    endpoint_for_rank,
    rank_sources,
    relay,
)
from smg_grpc_servicer.load_record import LoadSource
from smg_grpc_servicer.sglang.config_view import EngineConfigView

__all__ = ["dp_rank_count", "endpoint_for_rank", "sources", "subscribe_kv_events"]

logger = logging.getLogger(__name__)


def dp_rank_count(server_args: object) -> int:
    """How many KV-event publishers this SGLang instance runs: one per
    scheduler, on either engine release (see
    `ServerArgsSection.scheduler_count`: ``dp_size`` times the resolved
    ``attn_dp_size``, 1 when absent)."""
    return EngineConfigView(server_args).server_args.scheduler_count


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
        Engine.SGLANG,
        start_sequence_number,
        context,
        topic=str(getattr(config, "topic", "") or ""),
        hwm=getattr(config, "hwm", None),
        load_source=load_source,
    ):
        yield batch
