"""The additive handoff uses real protobuf serialization, without an engine."""

from smg_grpc_proto.generated import sglang_scheduler_pb2 as proto


def test_encoder_handoff_wire_preserves_order_zero_counts_and_existing_fields():
    value = proto.GenerateRequest(request_id="epd-wire", require_reasoning=True, stream=True)
    value.tokenized.input_ids.extend([101, 102])
    value.mm_inputs.image_urls.extend(["data:image/png;base64,A", "https://example.com/B"])
    value.encode_handoff.request_id = value.request_id
    urls = ["grpc://encoder-a:5001", "grpc://encoder-b:5002", "grpc://encoder-c:5003"]
    value.encode_handoff.encoder_urls.extend(urls)
    value.encode_handoff.num_items_assigned.extend([0, 2, 0])

    copied = proto.GenerateRequest.FromString(value.SerializeToString())
    assert copied.encode_handoff.request_id == copied.request_id == "epd-wire"
    assert list(copied.encode_handoff.encoder_urls) == urls
    assert list(copied.encode_handoff.num_items_assigned) == [0, 2, 0]
    assert list(copied.mm_inputs.image_urls) == list(value.mm_inputs.image_urls)
    assert list(copied.tokenized.input_ids) == [101, 102]
    assert copied.require_reasoning is True and copied.stream is True
    field = proto.GenerateRequest.DESCRIPTOR.fields_by_name["encode_handoff"]
    assert field.number == 19 and field.message_type == proto.EncodeHandoff.DESCRIPTOR


def test_encoder_handoff_absence_and_clear_preserve_the_ordinary_generate_wire():
    ordinary = proto.GenerateRequest(request_id="ordinary", require_reasoning=True)
    assert not ordinary.HasField("encode_handoff")
    with_handoff = proto.GenerateRequest.FromString(ordinary.SerializeToString())
    with_handoff.encode_handoff.request_id = ordinary.request_id
    assert with_handoff.HasField("encode_handoff")
    with_handoff.ClearField("encode_handoff")
    assert with_handoff == ordinary
