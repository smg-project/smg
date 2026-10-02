"""The TokenSpeed servicer tells the gateway the truth about RL state.

GetServerInfo advertises the in-engine control endpoint and capabilities and
reports the live pause state; GetModelInfo reports the live weight version;
generate responses carry the version that produced them. Secrets never leave
the engine through server info.

Run with: pytest grpc_servicer/tests/test_tokenspeed_rl_provenance.py
"""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("smg_grpc_proto")
servicer_mod = pytest.importorskip("smg_grpc_servicer.tokenspeed.servicer")
pb = servicer_mod.tokenspeed_scheduler_pb2
TokenSpeedSchedulerServicer = servicer_mod.TokenSpeedSchedulerServicer


def _servicer(*, advertise=True, paused=None, paused_sync=None, api_key="s3cret"):
    s = TokenSpeedSchedulerServicer.__new__(TokenSpeedSchedulerServicer)
    s.server_args = SimpleNamespace(
        model="m",
        weight_version="v7",
        rl_control_api_key=api_key,
        hf_token="hf_abc",
        max_total_tokens=8192,
        mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=1)),
    )
    s.scheduler_info = {}
    s.start_time = 0.0
    llm = SimpleNamespace(rid_to_state={})
    if advertise:
        llm.rl_advertisement = lambda: {
            "rl.control_url": "http://10.0.0.5:40100",
            "rl.pause_modes": "wait,abort,keep",
            "rl.abort": "true",
        }
    if paused is not None:

        async def _is_paused():
            return paused

        llm.is_scheduler_paused = _is_paused
    if paused_sync is not None:
        llm.is_scheduler_paused = lambda: paused_sync
    s.async_llm = llm
    return s


def _server_info(s):
    return asyncio.run(s.GetServerInfo(pb.GetServerInfoRequest(), None))


class TestGetServerInfo:
    def test_advertises_control_url_and_capabilities(self, monkeypatch):
        monkeypatch.setattr(servicer_mod, "_engine_supports_dp_rank_pin", lambda: False)
        info = _server_info(_servicer())
        assert info.server_args["rl.control_url"] == "http://10.0.0.5:40100"
        assert info.server_args["rl.pause_modes"] == "wait,abort,keep"
        assert info.server_args["rl.abort"] == "true"

    def test_older_engine_without_advertisement_still_serves(self, monkeypatch):
        monkeypatch.setattr(servicer_mod, "_engine_supports_dp_rank_pin", lambda: False)
        info = _server_info(_servicer(advertise=False))
        assert "rl.control_url" not in info.server_args
        assert info.server_args["model"] == "m"

    def test_reports_live_pause_state_and_defaults_false(self, monkeypatch):
        monkeypatch.setattr(servicer_mod, "_engine_supports_dp_rank_pin", lambda: False)
        assert _server_info(_servicer(paused=True)).is_paused is True
        assert _server_info(_servicer(paused=False)).is_paused is False
        assert _server_info(_servicer()).is_paused is False

    def test_encode_workers_are_never_asked_for_pause_state(self, monkeypatch):
        """The EPD encode loop crashes on the pause query, so it is skipped there."""
        s = _servicer(paused=True)
        s.server_args.disaggregation_mode = "encode"
        calls = []

        async def _is_paused():
            calls.append(1)
            return True

        s.async_llm.is_scheduler_paused = _is_paused
        assert _server_info(s).is_paused is False
        assert calls == [], "encode worker must not receive the scheduler query"

    def test_reports_pause_state_from_a_synchronous_query(self, monkeypatch):
        """``is_scheduler_paused`` is not pinned to be a coroutine function."""
        monkeypatch.setattr(servicer_mod, "_engine_supports_dp_rank_pin", lambda: False)
        assert _server_info(_servicer(paused_sync=True)).is_paused is True

    def test_secrets_never_leave_the_engine(self, monkeypatch):
        monkeypatch.setattr(servicer_mod, "_engine_supports_dp_rank_pin", lambda: False)
        info = _server_info(_servicer())
        assert "rl_control_api_key" not in info.server_args
        assert "hf_token" not in info.server_args
        assert info.server_args["max_total_tokens"] == 8192, "tokens is not a secret"
        assert info.server_args["weight_version"] == "v7"


def test_is_secret_key_targets_credentials_only():
    is_secret = servicer_mod._is_secret_key
    assert is_secret("rl_control_api_key") and is_secret("api_key") and is_secret("hf_token")
    assert is_secret("some_secret") and is_secret("db_password")
    assert not is_secret("max_total_tokens") and not is_secret("tokenizer")
    assert not is_secret("weight_version")


class TestGetModelInfo:
    def test_weight_version_is_the_live_server_args_value(self):
        s = _servicer()
        s.async_llm = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=None, vocab_size=10, dtype=None),
            max_req_input_len=1,
            context_len=2,
            rid_to_state={},
        )
        s.server_args.preferred_sampling_params = None
        s.server_args.served_model_name = None
        info = asyncio.run(s.GetModelInfo(pb.GetModelInfoRequest(), None))
        assert info.weight_version == "v7"
        s.server_args.weight_version = "v8"
        info = asyncio.run(s.GetModelInfo(pb.GetModelInfoRequest(), None))
        assert info.weight_version == "v8"


class TestGenerateStampsVersion:
    def test_complete_and_chunk_carry_meta_info_weight_version(self):
        s = _servicer()
        output = {
            "output_ids": [1, 2],
            "meta_info": {"finish_reason": {"type": "stop"}, "weight_version": "v7"},
        }
        complete = s._complete_response("r", output, {"type": "stop"}, 0)
        assert complete.complete.weight_version == "v7"
        chunk = s._chunk_response("r", output, None, 0)
        assert chunk.chunk.weight_version == "v7"

    def test_missing_version_leaves_the_field_unset(self):
        s = _servicer()
        output = {"output_ids": [1], "meta_info": {"finish_reason": {"type": "stop"}}}
        complete = s._complete_response("r", output, {"type": "stop"}, 0)
        assert not complete.complete.HasField("weight_version")


class TestGenerateStaleStubGuard:
    """A stale smg-grpc-proto wheel predates ``weight_version`` on these two
    messages entirely; passing the kwarg would raise ``ValueError`` on every
    response. The module-level ``_GENERATE_*_HAS_WEIGHT_VERSION`` flags gate
    it out — force them off here (as they'd be on a stale wheel, regardless
    of what the stub installed for this test run actually supports) and
    confirm generation still succeeds with the field simply left unset.
    """

    def test_complete_and_chunk_survive_when_stub_lacks_the_field(self, monkeypatch):
        monkeypatch.setattr(servicer_mod, "_GENERATE_CHUNK_HAS_WEIGHT_VERSION", False)
        monkeypatch.setattr(servicer_mod, "_GENERATE_COMPLETE_HAS_WEIGHT_VERSION", False)
        s = _servicer()
        output = {
            "output_ids": [1, 2],
            "meta_info": {"finish_reason": {"type": "stop"}, "weight_version": "v7"},
        }

        # Must not raise even though meta_info carries a weight_version — the
        # guard must omit the kwarg entirely, not merely pass None for it.
        complete = s._complete_response("r", output, {"type": "stop"}, 0)
        chunk = s._chunk_response("r", output, None, 0)

        complete_fields = complete.complete.DESCRIPTOR.fields_by_name
        if "weight_version" in complete_fields:
            assert not complete.complete.HasField("weight_version")
        chunk_fields = chunk.chunk.DESCRIPTOR.fields_by_name
        if "weight_version" in chunk_fields:
            assert not chunk.chunk.HasField("weight_version")
