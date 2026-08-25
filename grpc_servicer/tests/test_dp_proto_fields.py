"""Unit tests for DP-aware routing proto fields (engine-free, no vLLM required).

Run with: pytest grpc_servicer/tests/test_dp_proto_fields.py
"""

import pytest

pytest.importorskip("smg_grpc_proto")
from smg_grpc_proto import vllm_engine_pb2  # noqa: E402


class TestGenerateRequestDataParallelRank:
    def test_unset_by_default(self):
        request = vllm_engine_pb2.GenerateRequest()
        assert not request.HasField("data_parallel_rank")

    def test_rank_zero_is_distinguishable_from_unset(self):
        request = vllm_engine_pb2.GenerateRequest(data_parallel_rank=0)
        assert request.HasField("data_parallel_rank")
        assert request.data_parallel_rank == 0

    def test_set_rank_roundtrips(self):
        request = vllm_engine_pb2.GenerateRequest(data_parallel_rank=3)
        parsed = vllm_engine_pb2.GenerateRequest.FromString(request.SerializeToString())
        assert parsed.HasField("data_parallel_rank")
        assert parsed.data_parallel_rank == 3


class TestGetServerInfoResponseDataParallelSize:
    def test_defaults_to_zero_when_unreported(self):
        info = vllm_engine_pb2.GetServerInfoResponse()
        assert info.data_parallel_size == 0

    def test_size_roundtrips(self):
        info = vllm_engine_pb2.GetServerInfoResponse(data_parallel_size=4)
        parsed = vllm_engine_pb2.GetServerInfoResponse.FromString(info.SerializeToString())
        assert parsed.data_parallel_size == 4


class TestTokenSpeedGenerateRequestDataParallelRank:
    """TokenSpeed mirror of the vLLM pin-field presence contract (field 12)."""

    def test_unset_by_default(self):
        from smg_grpc_proto.generated import tokenspeed_scheduler_pb2

        request = tokenspeed_scheduler_pb2.GenerateRequest()
        assert not request.HasField("data_parallel_rank")

    def test_rank_zero_is_distinguishable_from_unset(self):
        from smg_grpc_proto.generated import tokenspeed_scheduler_pb2

        request = tokenspeed_scheduler_pb2.GenerateRequest(data_parallel_rank=0)
        assert request.HasField("data_parallel_rank")
        assert request.data_parallel_rank == 0

    def test_set_rank_roundtrips(self):
        from smg_grpc_proto.generated import tokenspeed_scheduler_pb2

        request = tokenspeed_scheduler_pb2.GenerateRequest(data_parallel_rank=3)
        parsed = tokenspeed_scheduler_pb2.GenerateRequest.FromString(request.SerializeToString())
        assert parsed.HasField("data_parallel_rank")
        assert parsed.data_parallel_rank == 3
