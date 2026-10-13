"""The SGLang scheduler's own abort of a request carries a status code (its
refusal of the caller's input, a full queue); the servicer ends the stream with
the matching gRPC status and the scheduler's message instead of a plain
``Complete(finish_reason = "abort")`` that drops both. A plain abort with no
status code (the caller's own cancel) stays a completion."""

from __future__ import annotations

import grpc
from smg_grpc_servicer.sglang.utils import abort_code_from_output, scheduler_abort


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
