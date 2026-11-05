"""The one table from an engine's HTTP status to the gRPC code, behind every engine."""

import asyncio
from http import HTTPStatus

import grpc
import pytest
from smg_grpc_servicer.engine_status import (
    CONSTRAINT_FAILED_METADATA_KEY,
    abort_for_constraint_failure,
    constraint_failed_message,
    constraint_failed_trailer,
    grpc_code_for_http_status,
)
from smg_grpc_servicer.sglang.utils import abort_code_from_output


@pytest.mark.parametrize(
    "status,expected",
    [
        (HTTPStatus.BAD_REQUEST, grpc.StatusCode.INVALID_ARGUMENT),
        (HTTPStatus.NOT_FOUND, grpc.StatusCode.NOT_FOUND),
        (HTTPStatus.REQUEST_TIMEOUT, grpc.StatusCode.DEADLINE_EXCEEDED),
        (HTTPStatus.TOO_MANY_REQUESTS, grpc.StatusCode.RESOURCE_EXHAUSTED),
        (HTTPStatus.INTERNAL_SERVER_ERROR, grpc.StatusCode.INTERNAL),
        (HTTPStatus.SERVICE_UNAVAILABLE, grpc.StatusCode.UNAVAILABLE),
        (HTTPStatus.GATEWAY_TIMEOUT, grpc.StatusCode.DEADLINE_EXCEEDED),
        (HTTPStatus.IM_A_TEAPOT, grpc.StatusCode.INTERNAL),
    ],
)
def test_each_engine_status_has_one_grpc_code(status, expected):
    assert grpc_code_for_http_status(status) is expected
    # the engines hand the status over as an int or as text as often as an HTTPStatus
    assert grpc_code_for_http_status(int(status)) is expected
    assert grpc_code_for_http_status(str(int(status))) is expected


@pytest.mark.parametrize("not_a_status", [None, "", "busy", object()])
def test_anything_that_is_not_a_status_is_internal(not_a_status):
    assert grpc_code_for_http_status(not_a_status) is grpc.StatusCode.INTERNAL


@pytest.mark.parametrize(
    "status",
    [
        HTTPStatus.BAD_REQUEST,
        HTTPStatus.NOT_FOUND,
        HTTPStatus.REQUEST_TIMEOUT,
        HTTPStatus.TOO_MANY_REQUESTS,
        HTTPStatus.INTERNAL_SERVER_ERROR,
        HTTPStatus.SERVICE_UNAVAILABLE,
        HTTPStatus.GATEWAY_TIMEOUT,
        None,
    ],
)
def test_the_same_engine_condition_is_the_same_code_behind_every_engine(
    status, vllm_errors_mod, tokenspeed_servicer_mod
):
    # The three servicers answer an engine's HTTP-style status from the one
    # table: a refused request, a full queue, a timeout and an unavailable
    # engine reach the client as the same status whichever engine is behind
    # the worker; a status-less abort is the engine's own fault behind each.
    expected = grpc_code_for_http_status(status)
    code = None if status is None else int(status)
    vllm_failure = vllm_errors_mod.GracefulHTTPError("refused", status)
    assert vllm_errors_mod.grpc_code_for(vllm_failure) is expected
    sglang_output = {
        "meta_info": {"finish_reason": {"type": "abort", "message": "refused", "status_code": code}}
    }
    assert abort_code_from_output(sglang_output) is expected
    tokenspeed_reason = {"type": "abort", "message": "refused", "status_code": code}
    assert tokenspeed_servicer_mod._abort_status_code(tokenspeed_reason) is expected


class _Aborted(Exception):
    pass


class _Context:
    """A servicer context that records its one abort, as grpc.aio's raises."""

    def __init__(self):
        self.aborts = []

    async def abort(self, code, details="", trailing_metadata=()):
        self.aborts.append((code, details, tuple(trailing_metadata)))
        raise _Aborted()


def test_a_constraint_failure_is_one_status_with_its_trailer():
    context = _Context()
    with pytest.raises(_Aborted):
        asyncio.run(
            abort_for_constraint_failure(context, "the scheduler aborted the request: Aborted")
        )
    assert context.aborts == [
        (
            grpc.StatusCode.FAILED_PRECONDITION,
            "the engine could not continue the request under its constraint: "
            "the scheduler aborted the request: Aborted",
            ((CONSTRAINT_FAILED_METADATA_KEY, "engine"),),
        )
    ]
    assert constraint_failed_trailer() == (("smg-constraint-failed", "engine"),)
    assert constraint_failed_message("x").endswith(": x")
