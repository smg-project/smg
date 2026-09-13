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
