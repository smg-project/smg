"""The SGLang Python servicer's sampling conversion reads an unset sampling
scalar as its neutral value. proto3 cannot tell an unset scalar from its zero,
and the zeros of ``top_p`` (the top token only), ``top_k`` (an empty candidate
set) and ``repetition_penalty`` (a repeated token's positive logit divided by
zero) are never meant; a set value reaches the scheduler as itself."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_sglang_server_info import servicer_mod as _servicer_fixture
from test_sglang_server_info import sglang_scheduler_pb2 as pb

servicer_mod = _servicer_fixture


@pytest.fixture
def convert(monkeypatch, servicer_mod):
    # The scheduler's SamplingParams is a stand-in in these engine-free tests:
    # it keeps the keywords the conversion hands it.
    monkeypatch.setattr(servicer_mod, "SGLSamplingParams", lambda **kw: SimpleNamespace(**kw))
    servicer = servicer_mod.SGLangSchedulerServicer
    return lambda params: servicer._convert_sampling_params(None, params)


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
