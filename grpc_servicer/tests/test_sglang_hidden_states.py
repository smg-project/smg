"""Hidden states on the Python SGLang servicer (#1083 parity): the request's
flag is forwarded under the engine's launch-time cap, the scheduler's
cumulative rows (the prefill's block of prompt rows, then one row per decoded
token) become the per-step delta of a stream and the whole list on completion,
and the proto fields carry them the way the Rust path fills the same fields.
Reuses the request-manager stand-ins of the selected-logprob tests: only the
engine imports are replaced, no manager behaviour is mocked."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("msgspec")
pytest.importorskip("zmq")

_SIBLING = Path(__file__).with_name("test_selected_token_logprobs.py")
_spec = importlib.util.spec_from_file_location("_selected_logprob_tests", _SIBLING)
_selected = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_selected)

manager_mod = _selected.manager_mod
servicer = _selected.servicer
_manager_and_state = _selected._manager_and_state
_output = _selected._output

PREFILL_BLOCK = [[0.5, 1.5], [2.5, 3.5]]  # one row per prompt position
FIRST_ROW = [4.5, 5.5]
SECOND_ROW = [6.5, 7.5]


def _hidden_manager_and_state(module, *, stream):
    manager, state = _manager_and_state(module, stream=stream)
    state.obj = SimpleNamespace(
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=stream,
        return_hidden_states=True,
    )
    return manager, state


def _step(*, cumulative, tokens, finished):
    return _output(
        values=[],
        tokens=tokens,
        finished=finished,
        output_token_ids_logprobs_val=None,
        output_token_ids_logprobs_idx=None,
        output_hidden_states=[cumulative],
    )


def test_a_stream_gets_each_steps_new_rows_and_the_whole_list_on_completion(manager_mod, servicer):
    async def exercise():
        manager, state = _hidden_manager_and_state(manager_mod, stream=True)
        # Step 1: the scheduler ships the prefill block and the first token's row.
        await manager._handle_batch_output(
            _step(cumulative=[PREFILL_BLOCK, FIRST_ROW], tokens=[10], finished=False)
        )
        first = await state.out_queue.get()
        # Step 2: the cumulative list grew by one row; the request finished.
        await manager._handle_batch_output(
            _step(cumulative=[PREFILL_BLOCK, FIRST_ROW, SECOND_ROW], tokens=[11], finished=True)
        )
        last = await state.out_queue.get()

        assert first["hidden_states"] == [*PREFILL_BLOCK, FIRST_ROW]
        assert "all_hidden_states" not in first
        assert last["hidden_states"] == [SECOND_ROW]
        assert last["all_hidden_states"] == [*PREFILL_BLOCK, FIRST_ROW, SECOND_ROW]

        chunk = servicer._create_chunk_response("score", first).chunk
        assert list(chunk.hidden_states) == [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        complete = servicer._create_completion_response("score", last).complete
        assert [(h.position, h.layer, list(h.values)) for h in complete.all_hidden_states] == [
            (0, -1, [0.5, 1.5]),
            (1, -1, [2.5, 3.5]),
            (2, -1, [4.5, 5.5]),
            (3, -1, [6.5, 7.5]),
        ]

    asyncio.run(exercise())


def test_a_non_streaming_request_gets_the_whole_list_once_at_the_end(manager_mod, servicer):
    async def exercise():
        manager, state = _hidden_manager_and_state(manager_mod, stream=False)
        await manager._handle_batch_output(
            _step(cumulative=[PREFILL_BLOCK, FIRST_ROW], tokens=[10], finished=False)
        )
        first = await state.out_queue.get()
        await manager._handle_batch_output(
            _step(cumulative=[PREFILL_BLOCK, FIRST_ROW, SECOND_ROW], tokens=[11], finished=True)
        )
        last = await state.out_queue.get()
        assert "hidden_states" not in first and "all_hidden_states" not in first
        assert "hidden_states" not in last
        assert last["all_hidden_states"] == [*PREFILL_BLOCK, FIRST_ROW, SECOND_ROW]
        complete = servicer._create_completion_response("score", last).complete
        assert len(complete.all_hidden_states) == 4
        assert list(servicer._create_chunk_response("score", last).chunk.hidden_states) == []

    asyncio.run(exercise())


def test_a_request_that_did_not_ask_carries_nothing(manager_mod, servicer):
    async def exercise():
        manager, state = _hidden_manager_and_state(manager_mod, stream=True)
        state.obj.return_hidden_states = False
        await manager._handle_batch_output(
            _output(
                values=[],
                tokens=[10],
                finished=True,
                output_token_ids_logprobs_val=None,
                output_token_ids_logprobs_idx=None,
                output_hidden_states=[None],
            )
        )
        output = await state.out_queue.get()
        assert "hidden_states" not in output and "all_hidden_states" not in output
        assert list(servicer._create_chunk_response("score", output).chunk.hidden_states) == []
        assert (
            list(servicer._create_completion_response("score", output).complete.all_hidden_states)
            == []
        )

    asyncio.run(exercise())


def _fake_forward_batch_info(monkeypatch, mode_name):
    from enum import IntEnum

    class CaptureHiddenMode(IntEnum):
        NULL = 0
        LAST = 1
        FULL = 2

        def need_capture(self):
            return self != CaptureHiddenMode.NULL

    module = ModuleType("sglang.srt.model_executor.forward_batch_info")
    module.CaptureHiddenMode = CaptureHiddenMode
    module.get_server_return_hidden_states_mode = lambda: CaptureHiddenMode[mode_name]
    for parent in ("sglang", "sglang.srt", "sglang.srt.model_executor"):
        if parent not in sys.modules:
            pkg = ModuleType(parent)
            pkg.__path__ = []
            monkeypatch.setitem(sys.modules, parent, pkg)
    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor.forward_batch_info", module)


@pytest.mark.parametrize(
    ("mode", "words"),
    [
        ("NULL", "The server is not configured to return hidden states"),
        ("LAST", "exceeds the server maximum `last`"),
    ],
)
def test_the_engines_launch_time_cap_is_kept_with_its_own_words(monkeypatch, mode, words):
    from smg_grpc_servicer.sglang import hidden_states

    _fake_forward_batch_info(monkeypatch, mode)
    with pytest.raises(ValueError, match=words) as refusal:
        hidden_states.refuse_hidden_states_above_server_mode()
    assert "--return-hidden-states-mode full" in str(refusal.value)
    _fake_forward_batch_info(monkeypatch, "FULL")
    hidden_states.refuse_hidden_states_above_server_mode()


def test_rows_flatten_the_engines_raw_shape():
    from smg_grpc_servicer.sglang.hidden_states import (
        all_hidden_states_proto,
        flat_hidden_states,
        hidden_state_rows,
        new_rows,
    )

    class Tensorish:
        def __init__(self, data):
            self.data = data

        def tolist(self):
            return self.data

    assert hidden_state_rows(None) == []
    assert hidden_state_rows([]) == []
    assert hidden_state_rows([PREFILL_BLOCK, FIRST_ROW]) == [*PREFILL_BLOCK, FIRST_ROW]
    assert hidden_state_rows([Tensorish(PREFILL_BLOCK), Tensorish(FIRST_ROW)]) == [
        *PREFILL_BLOCK,
        FIRST_ROW,
    ]
    assert hidden_state_rows(FIRST_ROW) == [FIRST_ROW]  # the engine's "last" mode: one row
    assert new_rows([*PREFILL_BLOCK, FIRST_ROW], 2) == [FIRST_ROW]
    assert flat_hidden_states(None) == [] and flat_hidden_states([FIRST_ROW, SECOND_ROW]) == [
        4.5,
        5.5,
        6.5,
        7.5,
    ]
    pytest.importorskip("smg_grpc_proto")
    assert [(h.position, h.layer) for h in all_hidden_states_proto([FIRST_ROW, SECOND_ROW])] == [
        (0, -1),
        (1, -1),
    ]
