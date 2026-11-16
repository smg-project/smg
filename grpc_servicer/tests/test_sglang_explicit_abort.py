"""Explicit Abort wakes consumers without a scheduler acknowledgement.

Engine imports and scheduler dispatch use the shared CPU exit fixture; queue
consumption, abort dispatch and output handling are production methods.
"""

import asyncio
from types import SimpleNamespace

import pytest
from test_sglang_request_manager_exits import _handle, _settle, manager  # noqa: F401


@pytest.mark.parametrize(("stream", "send_fails"), [(True, False), (False, True)])
def test_explicit_abort_wakes_waiting_consumer(manager, stream, send_fails):  # noqa: F811
    async def scenario():
        got = []

        async def consume():
            async for item in manager.generate_request(
                SimpleNamespace(stream=stream, sampling_params=SimpleNamespace(n=1)), "r", None
            ):
                got.append(item)

        task = asyncio.create_task(consume())
        await _settle()
        state = manager.rid_to_state["r"]
        if send_fails:

            def refuse(obj):
                raise RuntimeError("synthetic abort dispatch failure")

            manager.send_to_scheduler = SimpleNamespace(send_pyobj=refuse)
        assert await manager.abort_request("r") is not send_fails
        await asyncio.wait_for(task, 0.5)
        assert state.event.is_set()
        assert len(got) == 1 and got[0]["error"] == "Request aborted"
        assert got[0]["finished"]
        assert not manager.rid_to_state

    asyncio.run(scenario())
    assert [span.ended for span in manager.exporter.spans] == [1]


def test_repeated_abort_and_late_outputs_queue_one_terminal(manager):  # noqa: F811
    """Repeated aborts and late generation or embedding outputs cannot double-finish."""

    async def scenario():
        task, got = await _handle(manager, "r")
        state = manager.rid_to_state["r"]
        manager.enable_trace = True  # Request setup used the shared tracing stand-in.
        await manager.abort_request("r")
        assert [span.ended for span in manager.exporter.spans] == [1]
        await manager.abort_request("r")
        await manager._handle_abort_req(SimpleNamespace(rid="r", finished_reason=None))
        await manager._handle_batch_output(
            SimpleNamespace(rids=["r"], output_ids=[[1]], prompt_tokens=[1], cached_tokens=[0])
        )
        await manager._handle_embedding_output(
            SimpleNamespace(
                rids=["r"],
                embeddings=[[0.0]],
                prompt_tokens=[1],
                cached_tokens=[0],
                finished_reasons=[None],
            )
        )
        assert state.out_queue.qsize() == 1
        await asyncio.wait_for(task, 0.5)
        assert len(got) == 1
        assert not manager.rid_to_state

    asyncio.run(scenario())
    assert [span.ended for span in manager.exporter.spans] == [1]


def test_scheduler_terminal_wins_a_later_explicit_abort(manager):  # noqa: F811
    async def scenario():
        task, got = await _handle(manager, "r")
        state = manager.rid_to_state["r"]
        reason = {"type": "abort", "message": "queue full", "status_code": 503}
        await manager._handle_abort_req(SimpleNamespace(rid="r", finished_reason=reason))
        await manager.abort_request("r")
        await manager._handle_abort_req(SimpleNamespace(rid="r", finished_reason=None))
        assert state.out_queue.qsize() == 1
        await asyncio.wait_for(task, 0.5)
        assert got[0]["meta_info"]["finish_reason"] == reason

    asyncio.run(scenario())
    assert [span.ended for span in manager.exporter.spans] == [1]
