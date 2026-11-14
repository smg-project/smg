"""Independent prefill terminals use the same slim wire without shared state."""

from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
from smg_grpc_servicer.sglang import zmq_msgpack as wire


def _output(*, flat=True, sparse=True):
    """Build either scheduler layout with a self-contained terminal row."""
    out = SimpleNamespace(
        rids=["p"],
        finished_reasons=[{"type": "length", "length": 0}],
        completion_tokens=[1],
        output_token_logprobs_val=[[-0.5]],
        output_token_logprobs_idx=[[42]],
        output_top_logprobs_val=None,
        output_top_logprobs_idx=None,
    )
    if flat:
        out.output_ids_flat = [42]
        out.output_ids_offsets = [0, 1]
    else:
        out.output_ids = [[42]]
    if sparse:
        out.request_stats = [SimpleNamespace(prompt_tokens=100, cached_tokens=95)]
    else:
        out.prompt_tokens = [100]
        out.cached_tokens = [95]
    return out


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_freeze_terminal_preserves_tokens_usage_and_live_adapter_state(flat, sparse):
    """Freeze an owned wire frame without touching live sender bookkeeping."""
    sender = wire.MsgpackSendSocket(engine_index=3)
    sender.socket = Mock()
    sender.traces = Mock()
    sender._load_probe = Mock(side_effect=AssertionError("must not sample live load"))
    sender.output_adapter._counts["p"] = (100, 90)
    sender.hidden_rows._consumed["p"] = 7
    out = _output(flat=flat, sparse=sparse)
    frame = sender.prepare_prefill_completion(out)
    if flat:
        out.output_ids_flat[0] = 0
    else:
        out.output_ids[0][0] = 0
    if sparse:
        out.request_stats[0].cached_tokens = 0
    else:
        out.cached_tokens[0] = 0
    slim = msgspec.msgpack.decode(frame, type=wire.BatchTokenIDSlimOutput)
    assert slim.rids == ["p"]
    assert slim.engine_index == 3
    assert slim.output_ids == [[42]]
    assert slim.prompt_tokens == [100]
    assert slim.cached_tokens == [95]
    assert slim.finished_reasons == ["length"]
    assert slim.output_token_logprobs_val == [[-0.5]]
    assert slim.kv_total_tokens == 0  # Rust codec treats this as NO load sample.
    sender.socket.send.assert_not_called()
    sender.traces.finish.assert_not_called()
    assert sender.output_adapter._counts["p"] == (100, 90)
    assert sender.hidden_rows._consumed["p"] == 7
    sender.finish_prefill_completion("p")
    sender.traces.finish.assert_called_once_with("p")
    assert "p" not in sender.output_adapter._counts
    assert "p" not in sender.hidden_rows._consumed


@pytest.mark.parametrize("invalid", ["unfinished", "usage", "hidden"])
def test_incomplete_snapshot_is_rejected(invalid):
    """Reject terminals that require the ordinary scheduler output path."""
    sender = wire.MsgpackSendSocket()
    out = _output()
    if invalid == "unfinished":
        out.finished_reasons = [None]
    elif invalid == "usage":
        out.request_stats = [None]
    else:
        out.output_hidden_states = [[[1.0]]]
    with pytest.raises(ValueError):
        sender.prepare_prefill_completion(out)


def test_endpoint_is_only_available_after_handshake():
    """Discover the existing destination without sharing its socket."""
    sender = wire.MsgpackSendSocket()
    assert sender.prefill_completion_endpoint() is None
    sender.socket = Mock()
    sender.socket.getsockopt.return_value = b"ipc:///smg-output"
    assert sender.prefill_completion_endpoint() == "ipc:///smg-output"
    sender.socket.getsockopt.assert_called_once_with(wire.zmq.LAST_ENDPOINT)


def test_mixed_sparse_and_dense_usage_is_self_contained():
    """Apply sparse-to-dense usage fallback independently for each row."""
    out = _output(flat=False)
    out.rids = ["p", "q"]
    out.finished_reasons *= 2
    out.output_ids = [[42], [43]]
    out.request_stats.append(None)
    out.prompt_tokens = [0, 200]
    out.cached_tokens = [0, 150]
    out.output_hidden_states = [None, None]
    frame = wire.MsgpackSendSocket().prepare_prefill_completion(out)
    slim = msgspec.msgpack.decode(frame, type=wire.BatchTokenIDSlimOutput)
    assert slim.prompt_tokens == [100, 200]
    assert slim.cached_tokens == [95, 150]


@pytest.mark.parametrize("field", ["prompt_tokens", "cached_tokens"])
@pytest.mark.parametrize("sparse", [False, True])
def test_missing_dense_usage_cannot_be_frozen(field, sparse):
    """Require both usage columns when no sparse row supplies them."""
    out = _output(sparse=False)
    if sparse:
        out.request_stats = [None]
    setattr(out, field, [])
    with pytest.raises(ValueError, match="self-contained usage"):
        wire.MsgpackSendSocket().prepare_prefill_completion(out)


def test_real_sglang_detokenizer_layout_uses_slim_wire():
    """Exercise the paired engine consumer without advancing live request state."""
    pytest.importorskip("sglang.srt.managers.schedule_batch")
    from unittest.mock import patch

    streamer_module = pytest.importorskip(
        "sglang.srt.managers.scheduler_components.output_streamer"
    )
    SchedulerOutputStreamer = streamer_module.SchedulerOutputStreamer
    if not hasattr(SchedulerOutputStreamer, "prepare_prefill_completion"):
        pytest.skip("installed SGLang has no independent prefill completion API")

    from smg_grpc_servicer.sglang.compat import get_disaggregation_types

    DisaggregationMode, _ = get_disaggregation_types()
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.test.test_utils import published_topology

    sender = wire.MsgpackSendSocket(engine_index=0)
    with (
        published_topology(ranks={"dp_rank": 0}),
        patch(
            "sglang.srt.managers.scheduler_components.output_streamer.get_serving",
            return_value=SimpleNamespace(stream_interval=1, weight_version="default"),
        ),
        patch(
            "sglang.srt.managers.scheduler_components.output_streamer.get_observability",
            return_value=SimpleNamespace(
                enable_metrics=False, enable_request_time_stats_logging=False
            ),
        ),
    ):
        streamer = SchedulerOutputStreamer(
            send_to_detokenizer=sender,
            tree_cache=SimpleNamespace(),
            server_args=SimpleNamespace(enable_lmcache=False, enable_hierarchical_cache=True),
            is_generation=True,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            disaggregation_mode=DisaggregationMode.PREFILL,
            skip_tokenizer_init=False,
            enable_hicache_storage=lambda: True,
        )
        req = Req("real-p", "", [1, 2, 3, 4], SamplingParams())
        req.output_ids = [42]
        req.cached_tokens = req.cached_tokens_storage = 3
        frame = streamer.prepare_prefill_completion(req)
    slim = msgspec.msgpack.decode(frame, type=wire.BatchTokenIDSlimOutput)
    assert slim.output_ids == [[42]]
    assert slim.prompt_tokens == [4]
    assert slim.cached_tokens == [3]
    assert slim.finished_reasons == ["length"]
    assert not req.finished_output
    assert req.send_token_offset == 0
    assert not req.detokenize_header_sent
