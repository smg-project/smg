"""Engine-free regression coverage for vLLM per-DP GetLoads."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from smg_grpc_proto import vllm_engine_pb2 as pb

_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "loads.py"
_SPEC = importlib.util.spec_from_file_location("vllm_loads", _PATH)
loads = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(loads)


def stats(running=0, waiting=0, usage=0.0, skipped=0):
    return NS(
        num_running_reqs=running,
        num_waiting_reqs=waiting,
        num_skipped_waiting_reqs=skipped,
        kv_cache_usage=usage,
    )


def engine(loggers, ranks=(0, 1, 2, 3), capacity=1000):
    cache = NS(kv_cache_size_tokens=capacity, num_gpu_blocks=99999, block_size=128)
    return NS(
        logger_manager=NS(stat_loggers=loggers, engine_indexes=list(ranks)),
        vllm_config=NS(
            parallel_config=NS(data_parallel_size=4),
            cache_config=cache,
            scheduler_config=NS(max_num_seqs=256),
        ),
    )


@pytest.fixture
def dp4():
    per_rank = {rank: stats(rank + 1, rank, (rank + 1) / 8, skipped=rank) for rank in range(4)}
    aggregate = NS(
        last_scheduler_stats_dict=per_rank,
        last_scheduler_stats=stats(999, 999, 0.99),
        aggregated=True,
    )
    return engine([NS(), aggregate])


def test_distinct_dp4_loads_with_per_engine_capacity(dp4):
    result = loads.build_loads_response(dp4, pb.GetLoadsRequest(), "test-version")
    assert result.version == "test-version"
    assert result.timestamp
    assert result.dp_rank_count == 4
    assert [item.dp_rank for item in result.loads] == [0, 1, 2, 3]
    assert [item.num_running_reqs for item in result.loads] == [1, 2, 3, 4]
    assert [item.num_waiting_reqs for item in result.loads] == [0, 2, 4, 6]
    assert [item.num_total_reqs for item in result.loads] == [1, 4, 7, 10]
    assert [item.num_used_tokens for item in result.loads] == [125, 250, 375, 500]
    assert all(item.max_total_num_tokens == 1000 for item in result.loads)
    assert all(item.max_running_requests == 256 for item in result.loads)


@pytest.mark.parametrize("rank", range(4))
def test_rank_filter_including_explicit_zero(dp4, rank):
    result = loads.build_loads_response(dp4, pb.GetLoadsRequest(dp_rank=rank))
    assert [(item.dp_rank, item.num_running_reqs) for item in result.loads] == [(rank, rank + 1)]


@pytest.mark.parametrize("rank", [-1, 4, 100])
def test_unmanaged_rank_is_rejected(dp4, rank):
    with pytest.raises(ValueError, match="not managed"):
        loads.build_loads_response(dp4, pb.GetLoadsRequest(dp_rank=rank))


def test_aggregate_disabled_loggers_preserve_nonzero_global_ranks():
    nested = NS(
        per_engine_stat_loggers={
            4: NS(last_scheduler_stats=stats(7, usage=0.25)),
            5: NS(last_scheduler_stats=None),
        }
    )
    e = engine([nested], (4, 5))
    del e.logger_manager.engine_indexes
    e.engine_core = NS(engine_ranks_managed=[4, 5])

    result = loads.build_loads_response(e, pb.GetLoadsRequest())
    assert [(item.dp_rank, item.num_running_reqs) for item in result.loads] == [(4, 7)]
    assert not loads.build_loads_response(e, pb.GetLoadsRequest(dp_rank=5)).loads


@pytest.mark.parametrize("ranks", [(0,), (0, 1, 2, 3)])
def test_rank_dictionary_never_falls_back_to_aggregate(ranks):
    aggregate = NS(
        last_scheduler_stats_dict={},
        last_scheduler_stats=stats(999, usage=0.99),
        aggregated=True,
    )
    assert not loads.build_loads_response(engine([aggregate], ranks), pb.GetLoadsRequest()).loads


def test_scalar_snapshot_only_for_single_nonaggregate_rank():
    logger = NS(last_scheduler_stats=stats(7, usage=0.25), aggregated=False)
    assert not loads.build_loads_response(engine([logger]), pb.GetLoadsRequest()).loads
    result = loads.build_loads_response(engine([logger], (5,)), pb.GetLoadsRequest())
    assert [(item.dp_rank, item.num_running_reqs) for item in result.loads] == [(5, 7)]
    logger.aggregated = True
    assert not loads.build_loads_response(engine([logger], (5,)), pb.GetLoadsRequest()).loads


def test_missing_logger_is_not_reported_as_idle(dp4):
    dp4.logger_manager = None
    assert not loads.build_loads_response(dp4, pb.GetLoadsRequest()).loads


def test_idle_snapshot_is_reported():
    logger = NS(last_scheduler_stats=stats(), aggregated=False)
    result = loads.build_loads_response(engine([logger], (0,)), pb.GetLoadsRequest())
    assert result.dp_rank_count == 1
    assert result.loads[0].num_used_tokens == 0


@pytest.mark.parametrize("capacity", [None, False, True, 0, -1, 2**31])
def test_unknown_capacity_never_uses_dp_summed_gpu_blocks(dp4, capacity):
    dp4.vllm_config.cache_config.kv_cache_size_tokens = capacity
    result = loads.build_loads_response(dp4, pb.GetLoadsRequest())
    assert all(item.max_total_num_tokens == item.num_used_tokens == 0 for item in result.loads)
    assert result.loads[3].token_usage == 0.5


def test_capacity_int32_boundary_is_representable():
    logger = NS(last_scheduler_stats=stats(usage=1.0), aggregated=False)
    result = loads.build_loads_response(engine([logger], (0,), 2**31 - 1), pb.GetLoadsRequest())
    assert result.loads[0].max_total_num_tokens == 2**31 - 1
    assert result.loads[0].num_used_tokens == 2**31 - 1


@pytest.mark.parametrize("usage", [None, float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_usage_is_omitted(usage):
    logger = NS(last_scheduler_stats=stats(usage=usage), aggregated=False)
    assert not loads.build_loads_response(engine([logger], (0,)), pb.GetLoadsRequest()).loads


@pytest.mark.parametrize(
    ("running", "waiting", "skipped"),
    [
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (True, 0, 0),
        (2**31, 0, 0),
        (2**31 - 1, 1, 0),
        (0, 2**31 - 1, 1),
    ],
)
def test_invalid_or_overflowing_request_counts_are_omitted(running, waiting, skipped):
    logger = NS(
        last_scheduler_stats=stats(running, waiting, usage=0.25, skipped=skipped),
        aggregated=False,
    )
    assert not loads.build_loads_response(engine([logger], (0,)), pb.GetLoadsRequest()).loads


def test_optional_sections_do_not_hide_core_metrics(dp4):
    request = pb.GetLoadsRequest(include=["core", "disagg", "queues", "memory"])
    assert loads.build_loads_response(dp4, request).dp_rank_count == 4
