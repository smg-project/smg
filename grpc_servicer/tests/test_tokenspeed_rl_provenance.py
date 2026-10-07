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
from smg_grpc_servicer.tokenspeed import redact

pytest.importorskip("smg_grpc_proto")
servicer_mod = pytest.importorskip("smg_grpc_servicer.tokenspeed.servicer")
pb = servicer_mod.tokenspeed_scheduler_pb2
TokenSpeedSchedulerServicer = servicer_mod.TokenSpeedSchedulerServicer


@pytest.fixture(autouse=True)
def _no_dp_rank_pin(monkeypatch):
    """GetServerInfo's dp-rank-pin probe needs a real engine; none of these do."""
    monkeypatch.setattr(servicer_mod, "_engine_supports_dp_rank_pin", lambda: False)


def _servicer(*, advertise=True, paused=None, api_key="s3cret"):
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
    s.async_llm = llm
    return s


def _server_info(s):
    return asyncio.run(s.GetServerInfo(pb.GetServerInfoRequest(), None))


class TestGetServerInfo:
    def test_advertises_control_url_and_capabilities(self):
        info = _server_info(_servicer())
        assert info.server_args["rl.control_url"] == "http://10.0.0.5:40100"
        assert info.server_args["rl.pause_modes"] == "wait,abort,keep"
        assert info.server_args["rl.abort"] == "true"

    def test_older_engine_without_advertisement_still_serves(self):
        info = _server_info(_servicer(advertise=False))
        assert "rl.control_url" not in info.server_args
        assert info.server_args["model"] == "m"

    def test_failing_advertisement_degrades_to_a_label_free_worker(self):
        """A bug in the engine's advertisement must not take discovery down."""
        s = _servicer()

        def _boom():
            raise RuntimeError("advertisement broke")

        s.async_llm.rl_advertisement = _boom
        info = _server_info(s)
        assert "rl.control_url" not in info.server_args
        assert info.server_args["model"] == "m"

    def test_reports_live_pause_state_and_defaults_false(self):
        assert _server_info(_servicer(paused=True)).is_paused is True
        assert _server_info(_servicer(paused=False)).is_paused is False
        assert _server_info(_servicer()).is_paused is False

    def test_encode_workers_are_never_asked_for_pause_state(self):
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

    def test_silent_scheduler_costs_the_field_not_the_probe(self, monkeypatch):
        """The engine's pause query has no timeout of its own; ours bounds it
        without cancelling it. TokenSpeed's queueing communicator resets its
        state only after a normal completion, so a cancelled query would wedge
        every later one. A timed-out probe therefore keeps running, a later
        call waits on that same probe, and once the scheduler answers the true
        state comes through.
        """
        monkeypatch.setattr(servicer_mod, "PAUSE_PROBE_TIMEOUT", 0.05)
        s = _servicer()
        starts: list[int] = []
        cancelled: list[int] = []

        async def scenario():
            gate = asyncio.Event()

            async def _slow():
                starts.append(1)
                try:
                    await gate.wait()
                except asyncio.CancelledError:
                    cancelled.append(1)
                    raise
                return True

            s.async_llm.is_scheduler_paused = _slow
            first = await s.GetServerInfo(pb.GetServerInfoRequest(), None)
            second = await s.GetServerInfo(pb.GetServerInfoRequest(), None)
            gate.set()
            third = await s.GetServerInfo(pb.GetServerInfoRequest(), None)
            return first.is_paused, second.is_paused, third.is_paused

        assert asyncio.run(scenario()) == (False, False, True)
        assert cancelled == [], "the engine's query must never be cancelled"
        assert starts == [1], "later calls wait on the in-flight probe, not a new one"

    def test_pause_probe_deadline_sits_under_the_gateway_metadata_step(self):
        # The gateway gives the whole GetServerInfo 10 s before it registers the
        # worker without labels; the probe must leave room for the rest.
        assert servicer_mod.PAUSE_PROBE_TIMEOUT <= 5

    def test_secrets_never_leave_the_engine(self):
        info = _server_info(_servicer())
        assert "rl_control_api_key" not in info.server_args
        assert "hf_token" not in info.server_args
        assert info.server_args["max_total_tokens"] == 8192, "tokens is not a secret"
        assert info.server_args["weight_version"] == "v7"


def test_is_secret_key_targets_credentials_only():
    is_secret = redact.is_secret_key
    assert is_secret("rl_control_api_key") and is_secret("api_key") and is_secret("hf_token")
    assert is_secret("some_secret") and is_secret("db_password")
    assert not is_secret("max_total_tokens") and not is_secret("tokenizer")
    assert not is_secret("weight_version")


def test_redaction_reaches_nested_configs_and_lists():
    """Server args are redacted in their serialized form: sub-configs are dicts,
    and a credential inside a nested dict or a list of them goes too."""
    redacted = redact.redact_secrets(
        {
            "model": "m",
            "hf_token": "hf_abc",
            "kv_store": {"endpoint": "redis://cache", "password": "p", "ttl": 5},
            "providers": [{"name": "p", "api_key": "k"}, "plain"],
            "nested": {"deeper": {"api_key": "k", "keep": 1}},
        }
    )
    assert redacted == {
        "model": "m",
        "kv_store": {"endpoint": "redis://cache", "ttl": 5},
        "providers": [{"name": "p"}, "plain"],
        "nested": {"deeper": {"keep": 1}},
    }


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

    def test_non_string_version_is_coerced_not_rejected(self):
        """A trainer that stamps an integer step must not break every response."""
        s = _servicer()
        output = {
            "output_ids": [1],
            "meta_info": {"finish_reason": {"type": "stop"}, "weight_version": 42},
        }
        complete = s._complete_response("r", output, {"type": "stop"}, 0)
        assert complete.complete.weight_version == "42"
        chunk = s._chunk_response("r", output, None, 0)
        assert chunk.chunk.weight_version == "42"
