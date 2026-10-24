"""Production Generate/receiver controls with real protobufs and inert engine imports."""

import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import grpc
import pytest
import test_sglang_server_info
from smg_grpc_proto.generated import sglang_scheduler_pb2 as proto

servicer_mod = test_sglang_server_info.servicer_mod


@pytest.fixture
def service(servicer_mod, monkeypatch):
    # Only the unavailable engine's constructor/imports are stand-ins. The whole
    # SMG constructor, converter, Generate handler and protobufs are production code.
    monkeypatch.setenv("SGLANG_ENCODER_MM_RECEIVER_MODE", "grpc")
    monkeypatch.setattr(servicer_mod, "TokenizedGenerateReqInput", SimpleNamespace)
    monkeypatch.setattr(servicer_mod, "Modality", SimpleNamespace(IMAGE="image"))
    args = SimpleNamespace(
        skip_tokenizer_init=True,
        kv_events_config=None,
        language_only=True,
        disaggregation_mode="prefill",
        encoder_transfer_backend="zmq_to_scheduler",
    )
    manager = SimpleNamespace(auto_create_handle_loop=Mock(), generate_request=Mock())
    api = servicer_mod.SGLangSchedulerServicer(manager, args, None, {}, {})
    api._convert_sampling_params = lambda _: SimpleNamespace(n=1, normalize=lambda **kw: None)
    api._parse_mm_inputs = Mock(side_effect=AssertionError("local vision must not run"))
    return api


def request():
    value = proto.GenerateRequest(request_id="epd-17")
    value.tokenized.input_ids.extend([101, 102])
    value.tokenized.original_text = "original image prompt"
    value.mm_inputs.image_urls.append("data:image/png;base64,red")
    value.encode_handoff.request_id = value.request_id
    value.encode_handoff.encoder_urls.extend(["grpc://encoder-0:5001", "grpc://encoder-1:5002"])
    value.encode_handoff.num_items_assigned.extend([1, 0])
    return proto.GenerateRequest.FromString(value.SerializeToString())


def test_real_generate_submits_the_same_id_assignment_and_original_tokens(service):
    value = request()
    sent = []

    async def enqueue(**kwargs):
        sent.append(kwargs)
        if False:
            yield

    service.request_manager.generate_request = enqueue

    async def run():
        assert [response async for response in service.Generate(value, None)] == []

    asyncio.run(run())
    tokenized = sent[0]["obj"]
    assert sent[0]["request_id"] == tokenized.rid == value.encode_handoff.request_id
    assert tokenized.need_wait_for_mm_inputs is True
    assert tokenized.encoder_urls == list(value.encode_handoff.encoder_urls)
    assert tokenized.num_items_assigned == {"image": [1, 0]}
    assert list(tokenized.input_ids) == [101, 102] and tokenized.mm_inputs is None
    assert service.mm_receiver is None  # No legacy receiver import/creation or encode dispatch.
    service._parse_mm_inputs.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        "id",
        "empty",
        "parts",
        "sum",
        "duplicate",
        "url",
        "url-trailing-slash",
        "url-uppercase",
        "pixel",
        "video",
        "decode",
        "backend",
        "language",
        "n",
    ],
)
def test_invalid_handoff_aborts_before_scheduler_or_legacy_dispatch(service, change):
    value = request()
    if change == "id":
        value.encode_handoff.request_id = "other-request"
    elif change == "empty":
        value.encode_handoff.CopyFrom(proto.EncodeHandoff())
    elif change == "parts":
        del value.encode_handoff.num_items_assigned[:]
    elif change == "sum":
        value.encode_handoff.num_items_assigned[0] = 2
    elif change == "duplicate":
        value.encode_handoff.encoder_urls[1] = value.encode_handoff.encoder_urls[0]
    elif change == "url":
        value.encode_handoff.encoder_urls[0] = "http://encoder:5001"
    elif change == "url-trailing-slash":
        value.encode_handoff.encoder_urls[0] = "grpc://encoder:5001/"
    elif change == "url-uppercase":
        value.encode_handoff.encoder_urls[0] = "GRPC://encoder:5001"
    elif change == "pixel":
        value.mm_inputs.pixel_values.SetInParent()
    elif change == "video":
        value.mm_inputs.video_urls.append("video")
    elif change == "decode":
        service.server_args.disaggregation_mode = "decode"
    elif change == "backend":
        service.server_args.encoder_transfer_backend = "mooncake"
    elif change == "language":
        service.server_args.language_only = False
    elif change == "n":
        service._convert_sampling_params = lambda _: SimpleNamespace(
            n=2, normalize=lambda **kw: None
        )
    context = SimpleNamespace(abort=AsyncMock(side_effect=grpc.aio.AbortError()))

    async def run():
        with pytest.raises(grpc.aio.AbortError):
            async for _ in service.Generate(value, context):
                pytest.fail("invalid handoff yielded a completion")

    asyncio.run(run())
    assert context.abort.call_args.args[0] == grpc.StatusCode.INVALID_ARGUMENT
    service.request_manager.generate_request.assert_not_called()
    service._parse_mm_inputs.assert_not_called()
    assert service.mm_receiver is None


def test_scheduler_id_mismatch_and_zero_assignment_are_not_silently_replaced(service):
    value = request()
    tokenized = service._convert_generate_request(value)
    tokenized.rid = "different"
    with pytest.raises(ValueError, match="ID differs"):
        service._handle_epd_disaggregation_encode_request(value, tokenized)
    tokenized.rid = value.request_id
    value.encode_handoff.num_items_assigned[:] = [0, 0]
    with pytest.raises(ValueError, match="every image"):
        service._handle_epd_disaggregation_encode_request(value, tokenized)


@pytest.mark.parametrize("mode", [None, "http"])
def test_external_handoff_requires_explicit_grpc_receiver_mode(service, monkeypatch, mode):
    if mode is None:
        monkeypatch.delenv("SGLANG_ENCODER_MM_RECEIVER_MODE")
    else:
        monkeypatch.setenv("SGLANG_ENCODER_MM_RECEIVER_MODE", mode)
    context = SimpleNamespace(abort=AsyncMock(side_effect=grpc.aio.AbortError()))

    async def run():
        with pytest.raises(grpc.aio.AbortError):
            async for _ in service.Generate(request(), context):
                pytest.fail("wrong-mode handoff yielded a completion")

    asyncio.run(run())
    assert context.abort.call_args.args[0] == grpc.StatusCode.INVALID_ARGUMENT
    service.request_manager.generate_request.assert_not_called()
    assert service.mm_receiver is None


def test_no_envelope_retains_normal_vision_conversion(service, monkeypatch):
    monkeypatch.delenv("SGLANG_ENCODER_MM_RECEIVER_MODE")
    value = request()
    value.ClearField("encode_handoff")
    value.mm_inputs.pixel_values.SetInParent()
    service.server_args.language_only = False
    service._parse_mm_inputs.side_effect = None
    service._parse_mm_inputs.return_value = "ordinary-mm-inputs"
    tokenized = service._convert_generate_request(value)
    service._handle_epd_disaggregation_encode_request(value, tokenized)
    assert tokenized.mm_inputs == "ordinary-mm-inputs" and service.mm_receiver is None


def test_legacy_auto_receiver_is_created_once_under_concurrent_calls(service, monkeypatch):
    value = request()
    value.ClearField("encode_handoff")
    receiver = SimpleNamespace(
        build_and_send_encode_request=Mock(
            return_value=SimpleNamespace(
                need_wait_for_mm_inputs=True, num_items_assigned={"image": [1]}
            )
        )
    )

    def create(_):
        time.sleep(0.01)
        return receiver

    factory = Mock(side_effect=create)
    package = sys.modules["sglang.srt.disaggregation"]
    monkeypatch.setattr(
        package, "encode_receiver", SimpleNamespace(create_mm_receiver=factory), raising=False
    )
    tokens = [service._convert_generate_request(value) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(
            pool.map(lambda t: service._handle_epd_disaggregation_encode_request(value, t), tokens)
        )
    factory.assert_called_once_with(service.server_args)
    assert receiver.build_and_send_encode_request.call_count == 2
    assert all(t.need_wait_for_mm_inputs and t.num_items_assigned == {"image": [1]} for t in tokens)
