"""The SGLang Python servicer's sampling conversion reads an unset sampling
scalar as its neutral value. proto3 cannot tell an unset scalar from its zero,
and the zeros of ``top_p`` (the top token only), ``top_k`` (an empty candidate
set) and ``repetition_penalty`` (a repeated token's positive logit divided by
zero) are never meant; a set value reaches the scheduler as itself."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from test_sglang_server_info import servicer_mod as _servicer_fixture
from test_sglang_server_info import sglang_scheduler_pb2 as pb

servicer_mod = _servicer_fixture


@pytest.fixture
def convert(monkeypatch, servicer_mod):
    # The scheduler's SamplingParams is a stand-in in these engine-free tests:
    # it keeps the keywords the conversion hands it.
    monkeypatch.setattr(servicer_mod, "SGLSamplingParams", lambda **kw: SimpleNamespace(**kw))
    implementation = servicer_mod.SGLangSchedulerServicer
    servicer = implementation.__new__(implementation)
    servicer._supports_thinking_token_budget = True
    return servicer._convert_sampling_params


@pytest.mark.parametrize(
    ("sent", "field", "expected"),
    [
        pytest.param({}, "repetition_penalty", 1.0, id="penalty-unset-is-no-penalty"),
        pytest.param({"repetition_penalty": 1.0}, "repetition_penalty", 1.0, id="penalty-1.0"),
        pytest.param({"repetition_penalty": 1.3}, "repetition_penalty", 1.3, id="penalty-1.3"),
        pytest.param({}, "top_p", 1.0, id="top_p-unset-is-the-whole-distribution"),
        pytest.param({"top_p": 1.0}, "top_p", 1.0, id="top_p-1.0"),
        pytest.param({"top_p": 0.9}, "top_p", 0.9, id="top_p-0.9"),
        pytest.param({}, "top_k", -1, id="top_k-unset-is-every-token"),
        pytest.param({"top_k": -1}, "top_k", -1, id="top_k-minus-one"),
        pytest.param({"top_k": 1}, "top_k", 1, id="top_k-1"),
        pytest.param({"top_k": 40}, "top_k", 40, id="top_k-40"),
    ],
)
def test_an_unset_sampling_scalar_reaches_the_scheduler_as_its_neutral_value(
    convert, sent, field, expected
):
    params = convert(pb.SamplingParams(temperature=0.0, **sent))
    assert getattr(params, field) == pytest.approx(expected)


@pytest.mark.parametrize("field", ["top_p", "top_k", "repetition_penalty"])
def test_a_zero_sampling_scalar_is_not_expressible_on_the_wire(field):
    # proto3 leaves a scalar at its default off the wire: a client that sets
    # the zero sends the same bytes as one that leaves the field unset, so the
    # conversion cannot tell them apart, and the zero is read as unset.
    explicit_zero = pb.SamplingParams(**{field: 0}).SerializeToString()
    assert explicit_zero == pb.SamplingParams().SerializeToString()


def test_the_other_sampling_scalars_pass_through_as_sent(convert):
    # temperature and min_p are plain fields whose zero is a meaning of its
    # own (greedy; no floor): they reach the scheduler as sent, as do the
    # penalties whose neutral value is 0.
    params = convert(
        pb.SamplingParams(temperature=0.0, min_p=0.0, frequency_penalty=0.0, presence_penalty=0.5)
    )
    assert (params.temperature, params.min_p) == (0.0, 0.0)
    assert (params.frequency_penalty, params.presence_penalty) == (0.0, 0.5)


@pytest.mark.parametrize("budget", [0, 13])
def test_thinking_token_budget_reaches_scheduler_as_an_integer(convert, budget):
    params = pb.SamplingParams(thinking_token_budget=budget)
    params.custom_params.update({"another": True})
    result = convert(pb.SamplingParams.FromString(params.SerializeToString())).custom_params
    assert result == {"another": True, "thinking_budget": budget}
    assert type(result["thinking_budget"]) is int
    assert convert(pb.SamplingParams()).custom_params is None


def test_thinking_token_budget_is_refused_on_an_unconfigured_worker(servicer_mod):
    implementation = servicer_mod.SGLangSchedulerServicer
    servicer = implementation.__new__(implementation)
    servicer.server_args = SimpleNamespace()
    with pytest.raises(ValueError, match="thinking_token_budget"):
        servicer._convert_sampling_params(pb.SamplingParams(thinking_token_budget=0))


@pytest.fixture
def legacy_sampling_params():
    """A generated SamplingParams class from before optional field 24."""
    file = descriptor_pb2.FileDescriptorProto.FromString(pb.DESCRIPTOR.serialized_pb)
    params = next(message for message in file.message_type if message.name == "SamplingParams")
    budget = next(field for field in params.field if field.name == "thinking_token_budget")
    oneof = budget.oneof_index
    params.field.remove(budget)
    del params.oneof_decl[oneof]
    for field in params.field:
        if field.HasField("oneof_index") and field.oneof_index > oneof:
            field.oneof_index -= 1
    pool = descriptor_pool.DescriptorPool()
    for dependency in pb.DESCRIPTOR.dependencies:
        pool.AddSerializedFile(dependency.serialized_pb)
    pool.Add(file)
    return message_factory.GetMessageClass(
        pool.FindMessageTypeByName(pb.SamplingParams.DESCRIPTOR.full_name)
    )


@pytest.mark.parametrize("budget", [None, 0, 64])
def test_legacy_schema_refuses_only_an_explicit_budget(convert, legacy_sampling_params, budget):
    sent = pb.SamplingParams(max_new_tokens=80)
    if budget is not None:
        sent.thinking_token_budget = budget
    # An unrelated unknown varint (field 25) must remain forward-compatible.
    received = legacy_sampling_params.FromString(
        sent.SerializeToString() + bytes.fromhex("c8 01 01")
    )
    if budget is None:
        assert convert(received).max_new_tokens == 80
    else:
        with pytest.raises(ValueError, match="thinking_token_budget"):
            convert(received)
