"""Hidden states on the Python SGLang servicer: a request's flag is refused
ahead of the scheduler, first under the engine's launch-time cap in its own
words, then by name (hidden states are served by the Rust servicer), where the
servicer once accepted the flag and returned nothing. The scheduler-side helper
that turns the engine's cumulative list into a step's new rows (the plugin's
column, which the Rust servicer reads) is covered here too. Reuses the servicer
stand-in of the selected-logprob tests: only the engine imports are replaced;
of the servicer, only the sampling-parameter conversion (it needs the engine's
tokenizer) is stood in."""

from __future__ import annotations

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

PREFILL_BLOCK = [[0.5, 1.5], [2.5, 3.5]]  # one row per prompt position
FIRST_ROW = [4.5, 5.5]
SECOND_ROW = [6.5, 7.5]


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


def test_the_python_servicer_refuses_hidden_states_by_name():
    from smg_grpc_servicer.sglang import hidden_states

    with pytest.raises(ValueError, match="served by the Rust servicer") as refusal:
        hidden_states.refuse_hidden_states_on_this_servicer()
    # The message names the switch, both spellings.
    assert "--servicer-impl rust" in str(refusal.value)
    assert "SMG_SGLANG_SERVICER_IMPL=rust" in str(refusal.value)


def _generate_request(*, hidden):
    from smg_grpc_proto import sglang_scheduler_pb2

    return sglang_scheduler_pb2.GenerateRequest(
        request_id="h",
        tokenized=sglang_scheduler_pb2.TokenizedInput(original_text="hi", input_ids=[1, 2]),
        return_hidden_states=hidden,
    )


def test_the_conversion_refuses_under_the_cap_first_then_by_name(monkeypatch, servicer):
    module = sys.modules[type(servicer).__module__]
    calls = []

    def cap():
        calls.append("cap")

    def by_name():
        calls.append("name")
        raise ValueError("Hidden states are served by the Rust servicer")

    monkeypatch.setattr(module, "refuse_hidden_states_above_server_mode", cap)
    monkeypatch.setattr(module, "refuse_hidden_states_on_this_servicer", by_name)
    # The engine's request type is a stand-in here; record what it is built with.
    monkeypatch.setattr(
        module, "TokenizedGenerateReqInput", lambda **fields: SimpleNamespace(**fields)
    )
    # The sampling-parameter conversion needs the engine's tokenizer: stood in.
    servicer._convert_sampling_params = lambda params: SimpleNamespace(
        normalize=lambda tokenizer=None: None
    )
    servicer.tokenizer = None

    with pytest.raises(ValueError, match="served by the Rust servicer"):
        servicer._convert_generate_request(_generate_request(hidden=True))
    assert calls == ["cap", "name"]

    # A request without the flag reaches the scheduler, and the flag with it: never.
    calls.clear()
    built = servicer._convert_generate_request(_generate_request(hidden=False))
    assert calls == []
    assert built.rid == "h" and list(built.input_ids) == [1, 2]
    assert not hasattr(built, "return_hidden_states")


def test_new_rows_follow_the_engines_cumulative_elements():
    from smg_grpc_servicer.sglang.hidden_states import new_hidden_rows

    class Tensorish:
        def __init__(self, data):
            self.data = data

        def tolist(self):
            return self.data

    # The rows past the elements already consumed (the block is one element,
    # a decode step adds one or more), and the count to remember for the next step.
    assert new_hidden_rows(None, 0) == ([], 0)
    assert new_hidden_rows([], 0) == ([], 0)
    assert new_hidden_rows([PREFILL_BLOCK, FIRST_ROW], 0) == ([*PREFILL_BLOCK, FIRST_ROW], 2)
    assert new_hidden_rows([Tensorish(PREFILL_BLOCK), Tensorish(FIRST_ROW)], 0) == (
        [*PREFILL_BLOCK, FIRST_ROW],
        2,
    )
    assert new_hidden_rows([PREFILL_BLOCK, FIRST_ROW], 1) == ([FIRST_ROW], 2)
    assert new_hidden_rows([PREFILL_BLOCK, FIRST_ROW], 2) == ([], 2)
    assert new_hidden_rows([PREFILL_BLOCK, FIRST_ROW, SECOND_ROW], 2) == ([SECOND_ROW], 3)
    # The engine's "last" mode keeps a bare vector: one row, consumed whole.
    assert new_hidden_rows(FIRST_ROW, 0) == ([FIRST_ROW], 2)
    assert new_hidden_rows(FIRST_ROW, 2) == ([], 2)
    assert new_hidden_rows(None, 1) == ([], 1)
