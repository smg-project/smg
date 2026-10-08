"""GetLoads must not round-trip to the scheduler on an encode worker.

The EPD encode loop has no control-message dispatch: a GetLoadReqInput sent
over the scheduler channel is submitted to the encode worker as an encode
request and kills the scheduler. The servicer answers encode-mode load polls
itself with an empty, well-formed response.

Run with: pytest grpc_servicer/tests/test_tokenspeed_get_loads.py
"""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("smg_grpc_proto")
servicer_mod = pytest.importorskip("smg_grpc_servicer.tokenspeed.servicer")


def _bare_servicer(disaggregation_mode: str):
    servicer = servicer_mod.TokenSpeedSchedulerServicer.__new__(
        servicer_mod.TokenSpeedSchedulerServicer
    )
    servicer.server_args = SimpleNamespace(disaggregation_mode=disaggregation_mode)

    async def _must_not_be_called():
        raise AssertionError("encode-mode GetLoads must not reach the scheduler")

    servicer.async_llm = SimpleNamespace(get_load=_must_not_be_called)
    servicer.scheduler_info = {}
    return servicer


class TestEncodeModeGetLoads:
    def test_encode_worker_answers_empty_without_touching_the_scheduler(self):
        servicer = _bare_servicer("encode")
        request = servicer_mod.tokenspeed_scheduler_pb2.GetLoadsRequest()

        response = asyncio.run(servicer.GetLoads(request, context=None))

        assert response.dp_rank_count == 0
        assert len(response.loads) == 0
        assert response.aggregate.total_reqs == 0
        assert response.aggregate.avg_token_usage == 0.0
        assert response.version == "tokenspeed"

    def test_dp_rank_filter_is_irrelevant_on_the_empty_answer(self):
        servicer = _bare_servicer("encode")
        request = servicer_mod.tokenspeed_scheduler_pb2.GetLoadsRequest(dp_rank=1)

        response = asyncio.run(servicer.GetLoads(request, context=None))

        assert response.dp_rank_count == 0
        assert len(response.loads) == 0


def test_snapshot_geometry_and_active_pressure_survive_dp_filter():
    servicer = _bare_servicer("null")
    args = SimpleNamespace(prefix_granularity=64, max_num_seqs=16)
    snapshots = [
        SimpleNamespace(
            dp_rank=rank,
            num_running_reqs=1,
            num_waiting_reqs=2,
            num_active_pages=4,
            num_used_pages=8,
            max_total_pages=16,
        )
        for rank in (0, 1)
    ]

    async def get_load():
        return [SimpleNamespace(dp_rank=rank) for rank in (0, 1)]

    servicer.async_llm = SimpleNamespace(
        server_args=args,
        get_load=get_load,
        load_snapshot_store=SimpleNamespace(fresh_snapshots=lambda: snapshots),
    )
    # Native LCM blocks are not prefix pages; raw launch grain is irrelevant.
    servicer.scheduler_info = {"max_total_num_tokens": 16384}
    response = asyncio.run(
        servicer.GetLoads(servicer_mod.tokenspeed_scheduler_pb2.GetLoadsRequest(dp_rank=1), None)
    )
    assert response.dp_rank_count == 1
    load = response.loads[0]
    assert load.dp_rank == 1
    assert load.num_running_reqs == 1
    assert load.num_waiting_reqs == 2
    assert load.num_total_reqs == 3
    assert load.num_used_tokens == 8192
    assert load.max_total_num_tokens == 16384
    assert load.token_usage == 0.5
    assert load.HasField("active_token_usage")
    assert load.active_token_usage == 0.25
    assert load.max_running_requests == 16
    assert response.aggregate.avg_token_usage == 0.5

    # get_load() projects nothing until all DP ranks have fresh snapshots.
    async def incomplete_load():
        return []

    servicer.async_llm.get_load = incomplete_load
    snapshots.pop()
    response = asyncio.run(
        servicer.GetLoads(servicer_mod.tokenspeed_scheduler_pb2.GetLoadsRequest(), None)
    )
    assert not response.loads
