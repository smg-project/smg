"""Unit tests for the engine-declared constrained-decoding capability.

Run with: pytest grpc_servicer/tests/test_constrained_decoding_mode.py
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("smg_grpc_proto")
from smg_grpc_proto import vllm_engine_pb2  # noqa: E402

_HAS_FIELD = (
    "constrained_decoding_mode" in vllm_engine_pb2.GetServerInfoResponse.DESCRIPTOR.fields_by_name
)


@pytest.mark.skipif(not _HAS_FIELD, reason="installed smg-grpc-proto predates the field")
class TestGetServerInfoResponseConstrainedDecodingMode:
    def test_defaults_to_empty_when_undeclared(self):
        info = vllm_engine_pb2.GetServerInfoResponse()
        assert info.constrained_decoding_mode == ""

    def test_mode_roundtrips(self):
        info = vllm_engine_pb2.GetServerInfoResponse(constrained_decoding_mode="after_reasoning")
        parsed = vllm_engine_pb2.GetServerInfoResponse.FromString(info.SerializeToString())
        assert parsed.constrained_decoding_mode == "after_reasoning"


class TestVllmDerivation:
    """`_constrained_decoding_mode` derives honestly from the engine config."""

    @staticmethod
    def _servicer_with(structured_outputs_config):
        vllm_servicer = pytest.importorskip("smg_grpc_servicer.vllm.servicer")
        servicer = vllm_servicer.VllmEngineServicer.__new__(vllm_servicer.VllmEngineServicer)
        servicer.engine = SimpleNamespace(
            vllm_config=SimpleNamespace(structured_outputs_config=structured_outputs_config)
        )
        return servicer

    def test_from_first_token_without_reasoning_parser(self):
        servicer = self._servicer_with(SimpleNamespace(reasoning_parser=""))
        assert servicer._constrained_decoding_mode() == "from_first_token"

    def test_from_first_token_without_structured_outputs_config(self):
        servicer = self._servicer_with(None)
        assert servicer._constrained_decoding_mode() == "from_first_token"

    def test_after_reasoning_with_reasoning_parser(self):
        servicer = self._servicer_with(SimpleNamespace(reasoning_parser="qwen3"))
        assert servicer._constrained_decoding_mode() == "after_reasoning"
