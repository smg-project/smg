"""TokenSpeed attention-DP hard-pin plumbing tests (engine-free).

Covers the three servicer-side pieces of the dp-affinity handshake:
the ``GenerateRequest.data_parallel_rank`` proto field, the
capability-gated forwarding in ``_build_generate_req``, and the
capability-gated ``dp_size`` advertisement in ``GetServerInfo``.

Run with: pytest grpc_servicer/tests/test_tokenspeed_dp_rank_pin.py
"""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("smg_grpc_proto")
pytest.importorskip("zmq")
pytest.importorskip("msgspec")
from smg_grpc_proto.generated import tokenspeed_scheduler_pb2  # noqa: E402

if "data_parallel_rank" not in tokenspeed_scheduler_pb2.GenerateRequest.DESCRIPTOR.fields_by_name:
    pytest.skip(
        "smg_grpc_proto stubs predate GenerateRequest.data_parallel_rank; "
        "regenerate from crates/grpc_client/proto",
        allow_module_level=True,
    )

import smg_grpc_servicer.tokenspeed.servicer as servicer_mod  # noqa: E402
from smg_grpc_servicer.tokenspeed.servicer import (  # noqa: E402
    TokenSpeedSchedulerServicer,
)


class _CapturingReq:
    """Stands in for GenerateReqInput; records constructor kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__dict__.update(kwargs)


def _make_servicer():
    servicer = TokenSpeedSchedulerServicer.__new__(TokenSpeedSchedulerServicer)
    servicer.server_args = SimpleNamespace(reasoning_parser=None)
    return servicer


def _request(**kwargs):
    return tokenspeed_scheduler_pb2.GenerateRequest(
        request_id="r1",
        tokenized=tokenspeed_scheduler_pb2.TokenizedInput(input_ids=[1, 2, 3]),
        sampling_params=tokenspeed_scheduler_pb2.SamplingParams(),
        **kwargs,
    )


class TestBuildGenerateReqPinForwarding:
    def test_pin_forwarded_when_engine_supports_it(self, monkeypatch):
        monkeypatch.setattr(servicer_mod, "_lazy_generate_req_input", lambda: _CapturingReq)
        monkeypatch.setattr(servicer_mod, "_engine_supports_dp_rank_pin", lambda: True)
        obj = _make_servicer()._build_generate_req(_request(data_parallel_rank=5))
        assert obj.kwargs["data_parallel_rank"] == 5

    def test_rank_zero_forwarded(self, monkeypatch):
        monkeypatch.setattr(servicer_mod, "_lazy_generate_req_input", lambda: _CapturingReq)
        monkeypatch.setattr(servicer_mod, "_engine_supports_dp_rank_pin", lambda: True)
        obj = _make_servicer()._build_generate_req(_request(data_parallel_rank=0))
        assert obj.kwargs["data_parallel_rank"] == 0

    def test_unset_pin_forwarded_as_none(self, monkeypatch):
        monkeypatch.setattr(servicer_mod, "_lazy_generate_req_input", lambda: _CapturingReq)
        monkeypatch.setattr(servicer_mod, "_engine_supports_dp_rank_pin", lambda: True)
        obj = _make_servicer()._build_generate_req(_request())
        assert obj.kwargs["data_parallel_rank"] is None

    def test_kwarg_omitted_when_engine_lacks_field(self, monkeypatch):
        # Older engines' GenerateReqInput has no such kwarg; passing it would
        # raise TypeError at construction — the probe must gate the kwarg out.
        monkeypatch.setattr(servicer_mod, "_lazy_generate_req_input", lambda: _CapturingReq)
        monkeypatch.setattr(servicer_mod, "_engine_supports_dp_rank_pin", lambda: False)
        obj = _make_servicer()._build_generate_req(_request(data_parallel_rank=5))
        assert "data_parallel_rank" not in obj.kwargs


def _make_info_servicer(dp_size):
    servicer = TokenSpeedSchedulerServicer.__new__(TokenSpeedSchedulerServicer)
    servicer.server_args = SimpleNamespace(
        model="m",
        mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=dp_size)),
    )
    servicer.scheduler_info = {}
    servicer.start_time = 0.0
    servicer.async_llm = SimpleNamespace(rid_to_state={})
    return servicer


class TestGetServerInfoDpSizeAdvertisement:
    def test_advertised_when_supported_and_dp_active(self, monkeypatch):
        monkeypatch.setattr(servicer_mod, "_engine_supports_dp_rank_pin", lambda: True)
        info = asyncio.run(
            _make_info_servicer(8).GetServerInfo(
                tokenspeed_scheduler_pb2.GetServerInfoRequest(), None
            )
        )
        assert info.server_args["dp_size"] == 8

    def test_not_advertised_without_engine_support(self, monkeypatch):
        # The dp_size label doubles as the capability handshake: an engine that
        # cannot honor a pin must not attract dp-aware virtual workers.
        monkeypatch.setattr(servicer_mod, "_engine_supports_dp_rank_pin", lambda: False)
        info = asyncio.run(
            _make_info_servicer(8).GetServerInfo(
                tokenspeed_scheduler_pb2.GetServerInfoRequest(), None
            )
        )
        assert "dp_size" not in info.server_args

    def test_single_rank_width_is_not_advertised(self, monkeypatch):
        # A width-1 pin has no placement to choose, and an explicit pin
        # still changes engine-side scheduling. Dp-aware gateways degrade
        # a missing label to a plain worker, so DP-off engines stay
        # label-free.
        monkeypatch.setattr(servicer_mod, "_engine_supports_dp_rank_pin", lambda: True)
        info = asyncio.run(
            _make_info_servicer(1).GetServerInfo(
                tokenspeed_scheduler_pb2.GetServerInfoRequest(), None
            )
        )
        assert "dp_size" not in info.server_args
