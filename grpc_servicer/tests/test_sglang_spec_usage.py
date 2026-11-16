"""Cumulative scheduler speculation counters survive the Python gRPC transport.

Engine imports are stand-ins shared with the selected-logprobs tests. The
request manager, serializer and protobuf wire encoding are production code.
"""

import asyncio
from types import SimpleNamespace

import pytest
import test_selected_token_logprobs as logprob_fixtures

# Reuse the fixtures without unused-import shadowing in test arguments.
manager_mod = logprob_fixtures.manager_mod
servicer = logprob_fixtures.servicer


def _manager(module, rids=("request",), stream=False):
    """Build tracked requests with isolated output state for the counter regressions."""
    manager = module.GrpcRequestManager.__new__(module.GrpcRequestManager)
    from smg_grpc_servicer.sglang.compat import SchedulerOutputAdapter

    manager.output_adapter = SchedulerOutputAdapter(allow_empty_rows=True)
    manager.enable_trace = False  # Built without __init__; tracing is disabled.
    manager.metrics_collector = None  # Metrics are not under test.
    manager.rid_to_state = {
        rid: module.GrpcReqState(
            request_id=rid,
            grpc_context=None,
            out_queue=asyncio.Queue(),
            finished=False,
            event=asyncio.Event(),
            obj=SimpleNamespace(return_logprob=False, stream=stream),
            time_stats=SimpleNamespace(
                first_token_time=1.0,
                set_last_time=lambda: None,
                set_finished_time=lambda **kwargs: None,
            ),
        )
        for rid in rids
    }
    return manager


def _batch(rids=("request",), finished=True, **counters):
    return SimpleNamespace(
        rids=list(rids),
        output_ids=[[11] for _ in rids],
        finished_reasons=[{"type": "length"} if finished else None for _ in rids],
        prompt_tokens=[5 for _ in rids],
        completion_tokens=[3 for _ in rids],
        cached_tokens=[1 for _ in rids],
        reasoning_tokens=[2 for _ in rids],
        # Deliberately unrelated: verify counts cannot determine actual proposals.
        spec_verify_ct=[99 for _ in rids],
        **counters,
    )


def _wire(servicer, rid, output):
    response = servicer._create_completion_response(rid, output)
    return type(response).FromString(response.SerializeToString()).complete


def test_scheduler_accepted_two_survives_completion_wire(manager_mod, servicer):
    async def exercise():
        manager = _manager(manager_mod)
        await manager._handle_batch_output(_batch(spec_num_correct_drafts=[2]))
        output = await manager.rid_to_state["request"].out_queue.get()
        complete = _wire(servicer, "request", output)
        assert complete.spec_accepted_tokens == 2
        assert complete.spec_draft_tokens == 0
        assert output["meta_info"] == {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "cached_tokens": 1,
            "reasoning_tokens": 2,
            "finish_reason": {"type": "length"},
            "spec_num_correct_drafts": 2,
        }
        assert (complete.prompt_tokens, complete.completion_tokens) == (5, 3)
        assert (complete.cached_tokens, complete.reasoning_tokens) == (1, 2)
        assert complete.finish_reason == "length"

    asyncio.run(exercise())


def test_actual_proposed_counter_survives_completion_wire(manager_mod, servicer):
    async def exercise():
        manager = _manager(manager_mod)
        await manager._handle_batch_output(
            _batch(spec_num_correct_drafts=[2], spec_num_proposed_drafts=[7])
        )
        output = await manager.rid_to_state["request"].out_queue.get()
        assert output["meta_info"]["spec_num_proposed_drafts"] == 7
        complete = _wire(servicer, "request", output)
        assert (complete.spec_accepted_tokens, complete.spec_draft_tokens) == (2, 7)

    asyncio.run(exercise())


@pytest.mark.parametrize("stream", [False, True])
def test_mixed_batch_keeps_request_alignment(manager_mod, servicer, stream):
    async def exercise():
        rids = ("unknown", "positive", "cancelled", "zero", "other")
        manager = _manager(manager_mod, rids[1:], stream=stream)
        manager.rid_to_state["cancelled"].finished = True
        await manager._handle_batch_output(
            _batch(
                rids,
                spec_num_correct_drafts=[8, 2, 6, 0, 4],
                spec_num_proposed_drafts=[20, 7, 10, 3, 9],
            )
        )
        for rid, accepted, proposed in (("positive", 2, 7), ("zero", 0, 3), ("other", 4, 9)):
            output = await manager.rid_to_state[rid].out_queue.get()
            complete = _wire(servicer, rid, output)
            assert (complete.spec_accepted_tokens, complete.spec_draft_tokens) == (
                accepted,
                proposed,
            )
        assert manager.rid_to_state["cancelled"].out_queue.empty()

    asyncio.run(exercise())


@pytest.mark.parametrize("stream", [False, True])
def test_final_cumulative_snapshot_is_not_summed(manager_mod, servicer, stream):
    async def exercise():
        manager = _manager(manager_mod, stream=stream)
        for finished, accepted, proposed in ((False, 1, 4), (True, 2, 7)):
            await manager._handle_batch_output(
                _batch(
                    finished=finished,
                    spec_num_correct_drafts=[accepted],
                    spec_num_proposed_drafts=[proposed],
                )
            )
            output = await manager.rid_to_state["request"].out_queue.get()
            complete = _wire(servicer, "request", output)
            assert (complete.spec_accepted_tokens, complete.spec_draft_tokens) == (
                accepted,
                proposed,
            )

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "counters",
    [
        {},
        {"spec_num_correct_drafts": None, "spec_num_proposed_drafts": None},
        {"spec_num_correct_drafts": [0]},
        {"spec_num_correct_drafts": [2], "spec_num_proposed_drafts": None},
    ],
)
def test_missing_none_and_zero_counters_do_not_invent_usage(manager_mod, servicer, counters):
    async def exercise():
        manager = _manager(manager_mod)
        # No server_args exist: fixed draft settings must not be consulted.
        await manager._handle_batch_output(_batch(**counters))
        output = await manager.rid_to_state["request"].out_queue.get()
        for key in ("spec_num_correct_drafts", "spec_num_proposed_drafts"):
            values = counters.get(key)
            if values is None:
                assert key not in output["meta_info"]
            else:
                assert output["meta_info"][key] == values[0]
        complete = _wire(servicer, "request", output)
        assert complete.spec_accepted_tokens == (counters.get("spec_num_correct_drafts") or [0])[0]
        assert complete.spec_draft_tokens == (counters.get("spec_num_proposed_drafts") or [0])[0]

    asyncio.run(exercise())


@pytest.mark.parametrize("values", [[], [None]])
def test_unavailable_per_request_counter_is_omitted(manager_mod, servicer, values):
    async def exercise():
        manager = _manager(manager_mod)
        await manager._handle_batch_output(
            _batch(spec_num_correct_drafts=values, spec_num_proposed_drafts=values)
        )
        output = await manager.rid_to_state["request"].out_queue.get()
        assert "spec_num_correct_drafts" not in output["meta_info"]
        assert "spec_num_proposed_drafts" not in output["meta_info"]
        complete = _wire(servicer, "request", output)
        assert complete.spec_accepted_tokens == complete.spec_draft_tokens == 0

    asyncio.run(exercise())


@pytest.mark.parametrize("key", ["spec_num_correct_drafts", "spec_num_proposed_drafts"])
def test_short_counter_column_keeps_present_row_and_omits_missing_row(manager_mod, servicer, key):
    async def exercise():
        manager = _manager(manager_mod, ("first", "second"))
        await manager._handle_batch_output(
            _batch(("first", "second"), **{"spec_num_correct_drafts": [0, 0], key: [2]})
        )
        for rid in ("first", "second"):
            output = await manager.rid_to_state[rid].out_queue.get()
            complete = _wire(servicer, rid, output)
            if rid == "first":
                assert output["meta_info"][key] == 2
            else:
                assert key not in output["meta_info"]
            expected = 2 if rid == "first" else 0
            if key == "spec_num_correct_drafts":
                assert complete.spec_accepted_tokens == expected
                assert complete.spec_draft_tokens == 0
            else:
                assert complete.spec_accepted_tokens == 0
                assert complete.spec_draft_tokens == expected

    asyncio.run(exercise())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("accepted", ["missing", None, [], [None], [0]])
def test_proposed_requires_per_request_accepted(manager_mod, servicer, stream, accepted):
    async def exercise():
        rids = ("first", "second")
        manager = _manager(manager_mod, rids, stream=stream)
        await manager._handle_batch_output(
            _batch(
                rids,
                spec_num_proposed_drafts=[7, 7],
                **({} if accepted == "missing" else {"spec_num_correct_drafts": accepted}),
            )
        )
        for i, rid in enumerate(rids):
            output = await manager.rid_to_state[rid].out_queue.get()
            known = (
                accepted != "missing"
                and accepted is not None
                and i < len(accepted)
                and accepted[i] is not None
            )
            assert ("spec_num_proposed_drafts" in output["meta_info"]) == known
            complete = _wire(servicer, rid, output)
            assert complete.spec_accepted_tokens == 0
            assert complete.spec_draft_tokens == (7 if known else 0)

    asyncio.run(exercise())


@pytest.mark.parametrize("proposed", [None, [], [None], [7]])
def test_missing_proposed_preserves_independent_accepted(manager_mod, servicer, proposed):
    async def exercise():
        rids = ("first", "second")
        manager = _manager(manager_mod, rids)
        await manager._handle_batch_output(
            _batch(rids, spec_num_correct_drafts=[2, 0], spec_num_proposed_drafts=proposed)
        )
        for i, rid in enumerate(rids):
            output = await manager.rid_to_state[rid].out_queue.get()
            known = proposed is not None and i < len(proposed) and proposed[i] is not None
            complete = _wire(servicer, rid, output)
            assert complete.spec_accepted_tokens == (2 if i == 0 else 0)
            assert complete.spec_draft_tokens == (proposed[i] if known else 0)

    asyncio.run(exercise())
