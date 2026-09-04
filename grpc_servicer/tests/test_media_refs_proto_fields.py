"""Unit tests for the media_refs proto contract (engine-free, no vLLM required).

Run with: pytest grpc_servicer/tests/test_media_refs_proto_fields.py
"""

import pytest

pytest.importorskip("smg_grpc_proto")
from smg_grpc_proto import vllm_engine_pb2  # noqa: E402
from smg_grpc_proto.generated import common_pb2  # noqa: E402


class TestGenerateRequestMediaRefs:
    def test_field_number(self):
        field = vllm_engine_pb2.GenerateRequest.DESCRIPTOR.fields_by_name["media_refs"]
        assert field.number == 10

    def test_unset_by_default(self):
        request = vllm_engine_pb2.GenerateRequest()
        assert not request.HasField("media_refs")

    def test_empty_message_is_distinguishable_from_unset(self):
        request = vllm_engine_pb2.GenerateRequest(media_refs=vllm_engine_pb2.MediaRefs())
        assert request.HasField("media_refs")
        assert len(request.media_refs.items) == 0

    def test_roundtrip_keeps_order_and_modality(self):
        request = vllm_engine_pb2.GenerateRequest(
            tokenized=vllm_engine_pb2.TokenizedInput(input_ids=[1, 2, 3]),
            media_refs=vllm_engine_pb2.MediaRefs(
                items=[
                    vllm_engine_pb2.MediaRef(modality=common_pb2.IMAGE, url="https://a/1.png"),
                    vllm_engine_pb2.MediaRef(modality=common_pb2.VIDEO, url="https://a/c.mp4"),
                ]
            ),
        )
        parsed = vllm_engine_pb2.GenerateRequest.FromString(request.SerializeToString())
        assert parsed.HasField("media_refs")
        assert not parsed.HasField("mm_inputs")
        assert [(i.modality, i.url) for i in parsed.media_refs.items] == [
            (common_pb2.IMAGE, "https://a/1.png"),
            (common_pb2.VIDEO, "https://a/c.mp4"),
        ]

    def test_media_ref_uses_shared_modality_enum(self):
        field = vllm_engine_pb2.MediaRef.DESCRIPTOR.fields_by_name["modality"]
        assert field.enum_type is common_pb2.Modality.DESCRIPTOR


class TestGetServerInfoResponseMediaFields:
    def test_field_numbers(self):
        fields = vllm_engine_pb2.GetServerInfoResponse.DESCRIPTOR.fields_by_name
        assert fields["mm_processor"].number == 11
        assert fields["mm_media_ref_schemes"].number == 12

    def test_empty_by_default(self):
        info = vllm_engine_pb2.GetServerInfoResponse()
        assert info.mm_processor == ""
        assert info.mm_media_ref_schemes == ""

    def test_roundtrip(self):
        info = vllm_engine_pb2.GetServerInfoResponse(
            mm_processor="inprocess", mm_media_ref_schemes="http,https,data"
        )
        parsed = vllm_engine_pb2.GetServerInfoResponse.FromString(info.SerializeToString())
        assert parsed.mm_processor == "inprocess"
        assert parsed.mm_media_ref_schemes == "http,https,data"
