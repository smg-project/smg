"""Wire compatibility and an explicit negative boundary, without a model."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import grpc
import pytest
import test_sglang_server_info
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from smg_grpc_proto.generated import sglang_scheduler_pb2 as proto

servicer_mod = test_sglang_server_info.servicer_mod


def test_field21_preserves_existing_media_identity_numbers():
    fields = proto.GenerateRequest.DESCRIPTOR.fields_by_name
    assert fields["return_media_identity"].number == 19
    assert fields["media_identity"].number == 20
    assert fields["encode_handoff"].number == 21
    value = proto.GenerateRequest(request_id="wire", return_media_identity=True)
    value.media_identity.version = 1
    value.media_identity.processor_key = "original-identity"
    value.encode_handoff.request_id = "wire"
    value.encode_handoff.encoder_urls.extend(["grpc://encoder-0:5001"])
    value.encode_handoff.num_items_assigned.extend([1])
    assert proto.GenerateRequest.FromString(value.SerializeToString()) == value


def test_python_servicer_refuses_new_handoff_before_conversion_or_dispatch(servicer_mod):
    api = servicer_mod.SGLangSchedulerServicer.__new__(servicer_mod.SGLangSchedulerServicer)
    api._convert_generate_request = Mock(side_effect=AssertionError("must not convert"))
    api._handle_epd_disaggregation_encode_request = Mock(
        side_effect=AssertionError("must not auto-dispatch")
    )
    api.request_manager = SimpleNamespace(generate_request=Mock())
    value = proto.GenerateRequest(request_id="new-handoff")
    value.encode_handoff.SetInParent()
    context = SimpleNamespace(abort=AsyncMock(side_effect=grpc.aio.AbortError()))

    async def invoke():
        with pytest.raises(grpc.aio.AbortError):
            async for _ in api.Generate(value, context):
                pytest.fail("unsupported handoff yielded a completion")

    asyncio.run(invoke())
    assert context.abort.call_args.args[0] == grpc.StatusCode.UNIMPLEMENTED
    api._convert_generate_request.assert_not_called()
    api._handle_epd_disaggregation_encode_request.assert_not_called()
    api.request_manager.generate_request.assert_not_called()


def test_without_handoff_keeps_the_existing_conversion_and_dispatch(servicer_mod):
    api = servicer_mod.SGLangSchedulerServicer.__new__(servicer_mod.SGLangSchedulerServicer)
    tokenized = object()
    api._convert_generate_request = Mock(return_value=tokenized)
    api._handle_epd_disaggregation_encode_request = Mock()

    async def enqueue(**_):
        if False:
            yield

    api.request_manager = SimpleNamespace(generate_request=Mock(side_effect=enqueue))
    value = proto.GenerateRequest(request_id="ordinary")

    async def invoke():
        assert [item async for item in api.Generate(value, None)] == []

    asyncio.run(invoke())
    api._convert_generate_request.assert_called_once_with(value)
    api._handle_epd_disaggregation_encode_request.assert_called_once_with(value, tokenized)
    api.request_manager.generate_request.assert_called_once()


def _legacy_generate_request_type():
    """Build a real protobuf class with the pre-handoff request descriptor."""
    source_file = proto.GenerateRequest.DESCRIPTOR.file
    pool = descriptor_pool.DescriptorPool()
    added = set()

    def add_file(file_descriptor):
        if file_descriptor.name in added:
            return
        for dependency in file_descriptor.dependencies:
            add_file(dependency)
        file_proto = descriptor_pb2.FileDescriptorProto()
        file_descriptor.CopyToProto(file_proto)
        if file_descriptor.name == source_file.name:
            request_proto = next(
                message for message in file_proto.message_type if message.name == "GenerateRequest"
            )
            field_index = next(
                index for index, field in enumerate(request_proto.field) if field.number == 21
            )
            assert request_proto.field[field_index].name == "encode_handoff"
            del request_proto.field[field_index]
        pool.AddSerializedFile(file_proto.SerializeToString())
        added.add(file_descriptor.name)

    add_file(source_file)
    descriptor = pool.FindMessageTypeByName(proto.GenerateRequest.DESCRIPTOR.full_name)
    return message_factory.GetMessageClass(descriptor)


def test_legacy_protobuf_without_field21_keeps_conversion_and_dispatch(servicer_mod):
    legacy_request = _legacy_generate_request_type()
    assert legacy_request is not proto.GenerateRequest
    assert "encode_handoff" not in legacy_request.DESCRIPTOR.fields_by_name
    assert legacy_request.DESCRIPTOR.fields_by_name["return_media_identity"].number == 19
    assert legacy_request.DESCRIPTOR.fields_by_name["media_identity"].number == 20
    value = legacy_request(request_id="ordinary-legacy")
    with pytest.raises(ValueError, match="encode_handoff"):
        value.HasField("encode_handoff")

    api = servicer_mod.SGLangSchedulerServicer.__new__(servicer_mod.SGLangSchedulerServicer)
    tokenized = object()
    api._convert_generate_request = Mock(return_value=tokenized)
    api._handle_epd_disaggregation_encode_request = Mock()

    async def enqueue(**_):
        if False:
            yield

    api.request_manager = SimpleNamespace(generate_request=Mock(side_effect=enqueue))
    context = SimpleNamespace(abort=AsyncMock(side_effect=grpc.aio.AbortError()))

    async def invoke():
        assert [item async for item in api.Generate(value, context)] == []

    asyncio.run(invoke())
    context.abort.assert_not_awaited()
    api._convert_generate_request.assert_called_once_with(value)
    api._handle_epd_disaggregation_encode_request.assert_called_once_with(value, tokenized)
    api.request_manager.generate_request.assert_called_once_with(
        obj=tokenized, request_id="ordinary-legacy", grpc_context=context
    )
