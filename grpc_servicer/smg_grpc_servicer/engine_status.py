"""The gRPC status for an engine's HTTP-style status, the same behind every engine.

An engine refuses or fails a request with an HTTP status of its own (a bad
request, a full queue, a timeout, a fault); the servicer ends the stream with
the gRPC status the gateway maps back to the client's status in one place.
One table here, so the same engine condition is the same status behind every
engine.
"""

from __future__ import annotations

from http import HTTPStatus

import grpc

_HTTP_TO_GRPC_CODE = {
    HTTPStatus.BAD_REQUEST: grpc.StatusCode.INVALID_ARGUMENT,
    HTTPStatus.NOT_FOUND: grpc.StatusCode.NOT_FOUND,
    HTTPStatus.REQUEST_TIMEOUT: grpc.StatusCode.DEADLINE_EXCEEDED,
    HTTPStatus.TOO_MANY_REQUESTS: grpc.StatusCode.RESOURCE_EXHAUSTED,
    HTTPStatus.SERVICE_UNAVAILABLE: grpc.StatusCode.UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT: grpc.StatusCode.DEADLINE_EXCEEDED,
}


def grpc_code_for_http_status(status: object) -> grpc.StatusCode:
    """The gRPC code for an engine's HTTP status (an ``HTTPStatus``, an int or
    a numeric string); ``INTERNAL`` for any status not in the table, 500
    included, and for anything that is not a status."""
    try:
        code = int(status)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return grpc.StatusCode.INTERNAL
    return _HTTP_TO_GRPC_CODE.get(code, grpc.StatusCode.INTERNAL)
