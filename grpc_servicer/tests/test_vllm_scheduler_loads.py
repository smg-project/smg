"""Engine-free regression coverage for vLLM per-DP GetLoads."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from smg_grpc_proto import vllm_engine_pb2 as pb

_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "scheduler_loads.py"
_SPEC = importlib.util.spec_from_file_location("vllm_loads", _PATH)
loads = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(loads)


def stats(running=0, waiting=0, usage=0.0, skipped=0):
    """Build one scheduler snapshot including skipped waiting requests."""
    return NS(
        num_running_reqs=running,
        num_waiting_reqs=waiting,
        num_skipped_waiting_reqs=skipped,
        kv_cache_usage=usage,
    )


def engine(loggers, ranks=(0, 1, 2, 3), capacity=1000):
    """Build a frontend with global ranks and per-engine profiled capacity."""
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
    """Expose four distinct ranks alongside a deliberately different aggregate."""
    per_rank = {rank: stats(rank + 1, rank, (rank + 1) / 8, skipped=rank) for rank in range(4)}
    aggregate = NS(
        last_scheduler_stats_dict=per_rank,
        last_scheduler_stats=stats(999, 999, 0.99),
        aggregated=True,
    )
    return engine([NS(), aggregate])


def test_distinct_dp4_loads_with_per_engine_capacity(dp4):
    """Use per-engine capacity and include skipped requests for each rank."""
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
    """An explicitly selected rank, including zero, is the only returned entry."""
    result = loads.build_loads_response(dp4, pb.GetLoadsRequest(dp_rank=rank))
    assert [(item.dp_rank, item.num_running_reqs) for item in result.loads] == [(rank, rank + 1)]


@pytest.mark.parametrize("rank", [-1, 4, 100])
def test_unmanaged_rank_is_rejected(dp4, rank):
    """Reject filters outside this frontend rather than returning another rank."""
    with pytest.raises(ValueError, match="not managed"):
        loads.build_loads_response(dp4, pb.GetLoadsRequest(dp_rank=rank))


def test_aggregate_disabled_loggers_preserve_nonzero_global_ranks():
    """Nested loggers retain global rank IDs and omit missing snapshots."""
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
    """An empty per-rank dictionary does not authorize a scalar fallback."""
    aggregate = NS(
        last_scheduler_stats_dict={},
        last_scheduler_stats=stats(999, usage=0.99),
        aggregated=True,
    )
    assert not loads.build_loads_response(engine([aggregate], ranks), pb.GetLoadsRequest()).loads


def test_scalar_snapshot_only_for_single_nonaggregate_rank():
    """Scalar telemetry is usable only when its rank is unambiguous."""
    logger = NS(last_scheduler_stats=stats(7, usage=0.25), aggregated=False)
    assert not loads.build_loads_response(engine([logger]), pb.GetLoadsRequest()).loads
    result = loads.build_loads_response(engine([logger], (5,)), pb.GetLoadsRequest())
    assert [(item.dp_rank, item.num_running_reqs) for item in result.loads] == [(5, 7)]
    logger.aggregated = True
    assert not loads.build_loads_response(engine([logger], (5,)), pb.GetLoadsRequest()).loads


def test_missing_logger_is_not_reported_as_idle(dp4):
    """Disabled stats yield no idle rank."""
    dp4.logger_manager = None
    assert not loads.build_loads_response(dp4, pb.GetLoadsRequest()).loads


def test_idle_snapshot_is_reported():
    """A valid zero snapshot remains distinguishable from absent stats."""
    logger = NS(last_scheduler_stats=stats(), aggregated=False)
    result = loads.build_loads_response(engine([logger], (0,)), pb.GetLoadsRequest())
    assert result.dp_rank_count == 1
    assert result.loads[0].num_used_tokens == 0


@pytest.mark.parametrize("capacity", [None, False, True, 0, -1, 2**31])
def test_unknown_capacity_never_uses_dp_summed_gpu_blocks(dp4, capacity):
    """Unknown capacity never falls back to a DP-summed block count."""
    dp4.vllm_config.cache_config.kv_cache_size_tokens = capacity
    result = loads.build_loads_response(dp4, pb.GetLoadsRequest())
    assert all(item.max_total_num_tokens == item.num_used_tokens == 0 for item in result.loads)
    assert result.loads[3].token_usage == 0.5


def test_capacity_int32_boundary_is_representable():
    """The largest valid capacity fits the protobuf without overflow."""
    logger = NS(last_scheduler_stats=stats(usage=1.0), aggregated=False)
    result = loads.build_loads_response(engine([logger], (0,), 2**31 - 1), pb.GetLoadsRequest())
    assert result.loads[0].max_total_num_tokens == 2**31 - 1
    assert result.loads[0].num_used_tokens == 2**31 - 1


@pytest.mark.parametrize("usage", [None, float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_usage_is_omitted(usage):
    """Nonfinite and out-of-range utilization omit the snapshot."""
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
    """Invalid protobuf counts must omit a snapshot instead of raising."""
    logger = NS(
        last_scheduler_stats=stats(running, waiting, usage=0.25, skipped=skipped),
        aggregated=False,
    )
    assert not loads.build_loads_response(engine([logger], (0,)), pb.GetLoadsRequest()).loads


def test_optional_sections_do_not_hide_core_metrics(dp4):
    """Optional section names leave core load telemetry available."""
    request = pb.GetLoadsRequest(include=["core", "disagg", "queues", "memory"])
    assert loads.build_loads_response(dp4, request).dp_rank_count == 4


def test_rank_estimates_preserve_upstream_metrics_without_cross_rank_leakage(dp4):
    """Read the upstream tracker per rank without mixing request estimates."""
    from smg_grpc_servicer.vllm.loads import LoadTracker

    trackers = LoadTracker(clock=lambda: 1000.0)
    rank1 = trackers
    rank1.submitted("started", 100, rank=1)
    rank1.first_output("started", 100, 50, rank=1)
    rank1.submitted("waiting", 400, rank=1)
    rank1.generated(200, rank=1)
    result = loads.build_loads_response(dp4, pb.GetLoadsRequest(), trackers=trackers)
    assert result.loads[1].num_waiting_uncached_tokens == 200
    assert result.loads[1].gen_throughput == 100.0
    assert result.loads[1].cache_hit_rate == 0.5
    assert result.loads[1].utilization == result.loads[1].token_usage
    assert result.loads[0].gen_throughput == 0.0
    assert result.loads[2].num_waiting_uncached_tokens == 0
    rank1.finished("waiting")
    assert rank1.pending() == 0


def test_single_nonzero_rank_retains_unassigned_request_estimates():
    """Preserve the upstream resolver for an unpinned single-rank request."""
    from smg_grpc_servicer.vllm.loads import LoadTracker

    trackers = LoadTracker(clock=lambda: 1000.0, rank_of=lambda _: 5)
    trackers.submitted("waiting", 300, rank=None)
    e = engine([NS(last_scheduler_stats=stats(waiting=1))], (5,))
    result = loads.build_loads_response(e, pb.GetLoadsRequest(), trackers=trackers)
    assert result.loads[0].dp_rank == 5
    assert result.loads[0].num_waiting_uncached_tokens == 300


def test_cluster_size_does_not_invent_managed_global_ranks():
    """Cluster size alone does not identify a frontend managed-rank slice."""
    e = engine([NS(last_scheduler_stats_dict={0: stats()})])
    del e.logger_manager.engine_indexes
    assert loads.managed_ranks(e) == []
    assert not loads.build_loads_response(e, pb.GetLoadsRequest()).loads
    e.vllm_config.parallel_config.data_parallel_size = 1
    assert loads.managed_ranks(e) == [0]


def test_missing_tracker_omits_estimate_fields(dp4, monkeypatch):
    """Missing estimates stay absent before proto3 scalar defaults erase presence."""
    constructed_fields = []

    def scheduler_load(**fields):
        """Capture field presence at the protobuf construction boundary."""
        constructed_fields.append(fields)
        return pb.SchedulerLoad(**fields)

    monkeypatch.setattr(
        loads, "pb", NS(SchedulerLoad=scheduler_load, GetLoadsResponse=pb.GetLoadsResponse)
    )
    result = loads.build_loads_response(dp4, pb.GetLoadsRequest())

    assert result.dp_rank_count == len(constructed_fields) == 4
    for fields in constructed_fields:
        assert "num_waiting_uncached_tokens" not in fields
        assert "gen_throughput" not in fields
        assert "cache_hit_rate" not in fields


def test_unpinned_request_estimates_follow_upstream_routing(dp4):
    """The engine's eventual route decides which rank gets queued token work."""
    from smg_grpc_servicer.vllm.loads import LoadTracker

    routed = {}
    tracker = LoadTracker(rank_of=routed.get)
    tracker.submitted("pending", 300, rank=None)
    before = loads.build_loads_response(dp4, pb.GetLoadsRequest(), trackers=tracker)
    assert all(load.num_waiting_uncached_tokens == 0 for load in before.loads)
    routed["pending"] = 2
    after = loads.build_loads_response(dp4, pb.GetLoadsRequest(), trackers=tracker)
    assert [load.num_waiting_uncached_tokens for load in after.loads] == [0, 0, 300, 0]


@pytest.mark.parametrize("logger_rank", [-1, 0, 4])
def test_scalar_snapshot_cannot_override_its_explicit_rank(logger_rank):
    """A scalar with a different or aggregate engine index is not rank five."""
    logger = NS(engine_index=logger_rank, last_scheduler_stats=stats(7, usage=0.25))
    assert not loads.build_loads_response(engine([logger], (5,)), pb.GetLoadsRequest()).loads


def test_single_engine_fallback_preserves_configured_global_rank():
    """Without managed-rank lists, a single engine retains its configured rank."""
    e = engine([NS(last_scheduler_stats=stats(7))], (5,))
    del e.logger_manager.engine_indexes
    e.vllm_config.parallel_config = NS(data_parallel_size=1, data_parallel_rank=5)
    assert loads.managed_ranks(e) == [5]
    assert loads.build_loads_response(e, pb.GetLoadsRequest()).loads[0].dp_rank == 5


@pytest.mark.asyncio
async def test_get_loads_rejects_unmanaged_rank_with_invalid_argument(dp4):
    """The real RPC maps invalid rank filters to gRPC INVALID_ARGUMENT."""
    import ast

    import grpc

    source = _PATH.with_name("servicer.py")
    method = next(
        node
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "GetLoads"
    )
    namespace = dict(
        vllm_engine_pb2=pb,
        grpc=grpc,
        VLLM_VERSION="test",
        build_loads_response=loads.build_loads_response,
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)

    async def abort(code, details):
        """Capture the status emitted by the RPC without starting a server."""
        assert code == grpc.StatusCode.INVALID_ARGUMENT
        assert "not managed" in details
        raise RuntimeError("aborted")

    with pytest.raises(RuntimeError, match="aborted"):
        await namespace["GetLoads"](
            NS(engine=dp4, _loads=None), pb.GetLoadsRequest(dp_rank=9), NS(abort=abort)
        )
