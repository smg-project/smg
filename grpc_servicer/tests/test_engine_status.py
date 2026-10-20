"""The one table from an engine's HTTP status to the gRPC code, behind every engine."""

from http import HTTPStatus

import grpc
import pytest
from smg_grpc_servicer.engine_status import grpc_code_for_http_status


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
