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


# The trailer that carries a worker's report that the engine could not continue
# the request under the request's own constraint (a JSON schema, a regular
# expression, a grammar, a structural tag), and its value (the engine's own
# report). The same report on every wire, the Rust servicers included: a router
# answers the client for the constraint (a 422 ``constraint_failed``), replays
# nothing (the same tokens fail the same way, each replay a prefill), and
# samples no circuit breaker against a worker that is healthy; a router that
# predates the trailer answers a 400 for the code and retries nothing. Only a
# path that holds the engine's own attribution raises it (the SGLang
# scheduler's grammar failure, named by this package's plugin:
# ``sglang.utils.grammar_abort``); an abort or error the engine does not
# attribute keeps the status it had, whatever the request carries.
CONSTRAINT_FAILED_METADATA_KEY = "smg-constraint-failed"
_REPORTED_BY_ENGINE = "engine"


def constraint_failed_message(detail: str) -> str:
    """The message of the constraint-failure status: the constraint named,
    the engine's own account (``detail``: its abort message, its finish
    reason) kept."""
    return f"the engine could not continue the request under its constraint: {detail}"


def constraint_failed_trailer() -> tuple[tuple[str, str], ...]:
    """The trailing metadata that marks the status as the report."""
    return ((CONSTRAINT_FAILED_METADATA_KEY, _REPORTED_BY_ENGINE),)


async def abort_for_constraint_failure(context: grpc.aio.ServicerContext, detail: str) -> None:
    """End the stream with the constraint-failure status: ``FAILED_PRECONDITION``,
    the message of :func:`constraint_failed_message`, the trailer of
    :func:`constraint_failed_trailer`. Raises the context's abort error, as
    every ``abort`` does."""
    await context.abort(
        grpc.StatusCode.FAILED_PRECONDITION,
        constraint_failed_message(detail),
        trailing_metadata=constraint_failed_trailer(),
    )
