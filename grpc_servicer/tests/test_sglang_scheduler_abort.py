"""The SGLang scheduler's own abort of a request carries a status code (its
refusal of the caller's input, a full queue); the servicer ends the stream with
the matching gRPC status and the scheduler's message instead of a plain
``Complete(finish_reason = "abort")`` that drops both. A plain abort with no
status code (the caller's own cancel) stays a completion."""

from __future__ import annotations

import grpc
from smg_grpc_servicer.sglang import GRAMMAR_REJECTED_ERR_TYPE
from smg_grpc_servicer.sglang.utils import (
    abort_code_from_output,
    grammar_abort,
    scheduler_abort,
)


def _output(finish_reason):
    return {"text": "", "meta_info": {"finish_reason": finish_reason}, "finished": True}


def test_a_scheduler_refusal_is_an_abort_with_its_status():
    output = _output(
        {"type": "abort", "message": "Input length exceeds the limit", "status_code": 400}
    )
    assert scheduler_abort(output) == output["meta_info"]["finish_reason"]
    assert abort_code_from_output(output) == grpc.StatusCode.INVALID_ARGUMENT


def test_a_full_queue_is_an_abort_the_worker_answers_for():
    output = _output({"type": "abort", "message": "The request queue is full", "status_code": 503})
    assert scheduler_abort(output) is not None
    assert abort_code_from_output(output) == grpc.StatusCode.UNAVAILABLE


def test_a_full_queue_refused_with_429_and_a_timeout_carry_their_own_codes():
    # The shared table: the same engine condition is the same status behind every engine.
    busy = _output({"type": "abort", "message": "Too many requests", "status_code": 429})
    assert abort_code_from_output(busy) == grpc.StatusCode.RESOURCE_EXHAUSTED
    late = _output({"type": "abort", "message": "Request timed out", "status_code": 408})
    assert abort_code_from_output(late) == grpc.StatusCode.DEADLINE_EXCEEDED


def test_an_unknown_status_maps_to_internal():
    output = _output({"type": "abort", "message": "scheduler failure", "status_code": 500})
    assert scheduler_abort(output) is not None
    assert abort_code_from_output(output) == grpc.StatusCode.INTERNAL


def test_the_callers_own_abort_stays_a_completion():
    for finish_reason in (
        {"type": "abort"},
        {"type": "abort", "message": "aborted by the client", "status_code": None},
        "abort",
    ):
        assert scheduler_abort(_output(finish_reason)) is None


def test_other_finishes_are_not_aborts():
    for finish_reason in ({"type": "stop", "matched": 2}, {"type": "length"}, None):
        assert scheduler_abort(_output(finish_reason)) is None


def test_only_the_abort_the_scheduler_attributes_to_the_grammar_is_the_constraint_failure():
    # The scheduler's own grammar failure, named by the plugin's hooks: bare
    # (the result processor's paths) or with the 500 a PD decode side adds.
    named = {
        "type": "abort",
        "message": "the grammar rejected a sampled token",
        "status_code": None,
        "err_type": GRAMMAR_REJECTED_ERR_TYPE,
    }
    assert grammar_abort(_output(named)) == named
    decode_side = {
        "type": "abort",
        "message": "Grammar accept_token failed for req r1 with token 7: Tokens not accepted: 7",
        "status_code": 500,
        "err_type": GRAMMAR_REJECTED_ERR_TYPE,
    }
    assert grammar_abort(_output(decode_side)) == decode_side
    # The same bare abort without the name (an abort-all, a session's end, a
    # scheduler without the hooks) is not the grammar's, whatever the request
    # carries: it stays what it was.
    for bare in (
        {"type": "abort", "message": "Aborted", "status_code": None, "err_type": None},
        {"type": "abort", "message": "Aborted", "status_code": None},
        {"type": "abort"},
        "abort",
    ):
        assert grammar_abort(_output(bare)) is None
        assert scheduler_abort(_output(bare)) is None
    # An abort with a status and another name is the scheduler's own.
    with_status = {
        "type": "abort",
        "message": "The request queue is full",
        "status_code": 503,
        "err_type": "ServiceUnavailable",
    }
    assert grammar_abort(_output(with_status)) is None
    assert scheduler_abort(_output(with_status)) == with_status
    for finish_reason in ({"type": "stop", "matched": 2}, {"type": "length"}, None):
        assert grammar_abort(_output(finish_reason)) is None
