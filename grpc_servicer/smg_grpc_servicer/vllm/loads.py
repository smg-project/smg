"""Read cached vLLM scheduler loads without dispatching engine work."""

import math
from collections.abc import Mapping
from datetime import datetime, timezone

from smg_grpc_proto import vllm_engine_pb2 as pb

_MAX_INT32 = 2**31 - 1


def _managed_ranks(engine):
    """Return the global ranks managed by this vLLM frontend."""
    manager = getattr(engine, "logger_manager", None)
    ranks = getattr(manager, "engine_indexes", None)
    if ranks is None:
        engine_core = getattr(engine, "engine_core", None)
        ranks = getattr(engine_core, "engine_ranks_managed", None)
    if ranks is None:
        config = getattr(engine, "vllm_config", None)
        parallel = getattr(config, "parallel_config", None)
        size = getattr(parallel, "data_parallel_size", 1)
        ranks = (
            range(size)
            if isinstance(size, int) and not isinstance(size, bool) and 0 < size <= _MAX_INT32
            else ()
        )
    return sorted(
        {
            rank
            for rank in ranks
            if isinstance(rank, int) and not isinstance(rank, bool) and 0 <= rank <= _MAX_INT32
        }
    )


def _scheduler_stats_by_rank(engine, ranks):
    """Collect only rank-addressable snapshots from compatible logger shapes."""
    manager = getattr(engine, "logger_manager", None)
    snapshots = {}
    for logger in getattr(manager, "stat_loggers", None) or []:
        per_rank = getattr(logger, "last_scheduler_stats_dict", None)
        if isinstance(per_rank, Mapping):
            candidates = per_rank
        else:
            per_engine = getattr(logger, "per_engine_stat_loggers", None)
            if isinstance(per_engine, Mapping):
                candidates = {
                    rank: getattr(nested, "last_scheduler_stats", None)
                    for rank, nested in per_engine.items()
                }
            elif len(ranks) == 1 and not getattr(logger, "aggregated", False):
                candidates = {ranks[0]: getattr(logger, "last_scheduler_stats", None)}
            else:
                continue

        for rank in ranks:
            stats = candidates.get(rank)
            if stats is not None:
                snapshots.setdefault(rank, stats)
    return snapshots


def _is_nonnegative_int32(value):
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= _MAX_INT32


def build_loads_response(engine, request, version=""):
    """Build cached per-rank loads; omit missing or invalid telemetry."""
    ranks = _managed_ranks(engine)
    requested_rank = request.dp_rank if request.HasField("dp_rank") else None
    if requested_rank is not None and requested_rank not in ranks:
        raise ValueError(f"DP rank {requested_rank} is not managed by this engine: {ranks}")

    snapshots = _scheduler_stats_by_rank(engine, ranks)
    if requested_rank is not None:
        snapshots = (
            {requested_rank: snapshots[requested_rank]} if requested_rank in snapshots else {}
        )

    config = getattr(engine, "vllm_config", None)
    cache = getattr(config, "cache_config", None)
    # This is profiled per DP engine and accounts for hybrid KV-cache groups.
    # num_gpu_blocks can be DP-summed and block multiplication is not group-aware.
    capacity = getattr(cache, "kv_cache_size_tokens", None)
    if not _is_nonnegative_int32(capacity) or capacity == 0:
        capacity = 0
    scheduler = getattr(config, "scheduler_config", None)
    max_running = getattr(scheduler, "max_num_seqs", 0)
    if not _is_nonnegative_int32(max_running):
        max_running = 0

    loads = []
    for rank, stats in sorted(snapshots.items()):
        running = getattr(stats, "num_running_reqs", None)
        waiting = getattr(stats, "num_waiting_reqs", None)
        skipped = getattr(stats, "num_skipped_waiting_reqs", 0)
        try:
            usage = float(getattr(stats, "kv_cache_usage", None))
        except (TypeError, ValueError, OverflowError):
            continue
        if (
            not all(_is_nonnegative_int32(value) for value in (running, waiting, skipped))
            or not math.isfinite(usage)
            or not 0.0 <= usage <= 1.0
        ):
            continue
        waiting += skipped
        total = running + waiting
        if waiting > _MAX_INT32 or total > _MAX_INT32:
            continue

        loads.append(
            pb.SchedulerLoad(
                dp_rank=rank,
                num_running_reqs=running,
                num_waiting_reqs=waiting,
                num_total_reqs=total,
                token_usage=usage,
                num_used_tokens=math.ceil(usage * capacity),
                max_total_num_tokens=capacity,
                max_running_requests=max_running,
            )
        )

    return pb.GetLoadsResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=version,
        dp_rank_count=len(loads),
        loads=loads,
    )
