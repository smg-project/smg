"""The TokenSpeed servicer refuses an ask its gRPC contract cannot answer
(engine-free).

``GenerateRequest.token_ids_logprob`` asks for the selected candidates'
logprobs, but the contract's ``OutputLogProbs`` carries sampled and ranked
rows only, so the servicer refuses the ask as INVALID_ARGUMENT instead of
forwarding it to an engine whose answer nothing could return; the ranked
candidates (``top_logprobs_num``) and the plain sampled logprob are forwarded
as before.

Run with: pytest grpc_servicer/tests/test_tokenspeed_selected_logprobs.py
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("smg_grpc_proto")
pytest.importorskip("zmq")
pytest.importorskip("msgspec")
from smg_grpc_proto.generated import tokenspeed_scheduler_pb2  # noqa: E402

servicer_mod = pytest.importorskip("smg_grpc_servicer.tokenspeed.servicer")
TokenSpeedSchedulerServicer = servicer_mod.TokenSpeedSchedulerServicer


class _CapturingReq:
    """Stands in for GenerateReqInput; records constructor kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__dict__.update(kwargs)


def _make_servicer():
    servicer = TokenSpeedSchedulerServicer.__new__(TokenSpeedSchedulerServicer)
    servicer.server_args = SimpleNamespace(reasoning_parser=None)
    servicer.async_llm = None
    return servicer


def _request(**kwargs):
    return tokenspeed_scheduler_pb2.GenerateRequest(
        request_id="r1",
        tokenized=tokenspeed_scheduler_pb2.TokenizedInput(input_ids=[1, 2, 3]),
        sampling_params=tokenspeed_scheduler_pb2.SamplingParams(),
        **kwargs,
    )


@pytest.fixture
def capturing(monkeypatch):
    monkeypatch.setattr(servicer_mod, "_lazy_generate_req_input", lambda: _CapturingReq)
    monkeypatch.setattr(servicer_mod, "_engine_supports_dp_rank_pin", lambda: False)


def test_selected_candidates_are_refused_as_the_contract_cannot_return_them(capturing):
    with pytest.raises(ValueError, match="token_ids_logprob is not served"):
        _make_servicer()._build_generate_req(_request(token_ids_logprob=[4, 5]))


def test_ranked_and_sampled_logprob_asks_are_forwarded(capturing):
    obj = _make_servicer()._build_generate_req(
        _request(return_logprob=True, top_logprobs_num=3, logprob_start_len=0)
    )
    assert obj.kwargs["return_logprob"] is True
    assert obj.kwargs["top_logprobs_num"] == 3
    assert obj.kwargs["logprob_start_len"] == 0
    assert obj.kwargs["token_ids_logprob"] is None
