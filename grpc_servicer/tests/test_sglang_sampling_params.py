"""The SGLang Python servicer's sampling conversion: an unset
``repetition_penalty`` reaches the scheduler as 1.0 (no penalty), a set one
as itself. proto3 cannot tell an unset float from 0.0, and a zero penalty
factor is never meant; the Rust servicer applies the same rule."""

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
    ("penalty", "expected"),
    [
        pytest.param({}, 1.0, id="unset-is-no-penalty"),
        pytest.param({"repetition_penalty": 1.0}, 1.0, id="explicit-1.0"),
        pytest.param({"repetition_penalty": 1.3}, 1.3, id="explicit-1.3"),
    ],
)
def test_an_unset_repetition_penalty_reaches_the_scheduler_as_one(convert, penalty, expected):
    params = convert(pb.SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, **penalty))
    assert params.repetition_penalty == pytest.approx(expected)


def test_a_zero_repetition_penalty_is_not_expressible_on_the_wire():
    # proto3 leaves a scalar at its default off the wire: a client that sets
    # 0.0 sends the same bytes as one that leaves the field unset, so the
    # conversion cannot tell them apart, and 0.0 is read as unset.
    explicit_zero = pb.SamplingParams(repetition_penalty=0.0).SerializeToString()
    assert explicit_zero == pb.SamplingParams().SerializeToString()


def test_the_other_sampling_scalars_pass_through_as_sent(convert):
    # temperature, top_p, top_k and min_p are plain fields too; the scheduler
    # reads them as sent (0.0 is greedy, -1 is every token), on both servicers.
    params = convert(pb.SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0))
    assert (params.temperature, params.top_p, params.top_k, params.min_p) == (0.0, 1.0, -1, 0.0)
