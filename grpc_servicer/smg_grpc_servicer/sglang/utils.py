"""gRPC utility functions."""

from __future__ import annotations

from array import array
from collections.abc import Iterable
from http import HTTPStatus

import grpc


def to_token_id_array(token_ids: Iterable[int] | None) -> array | None:
    """Coerce a token-id sequence to the ``array("q")`` the SGLang scheduler expects.

    SGLang declares ``TokenizedGenerateReqInput.input_ids`` /
    ``TokenizedEmbeddingReqInput.input_ids`` as ``Optional[array[int]]`` and its
    ``Req`` concatenates ``origin_input_ids + output_ids`` where ``output_ids``
    is ``array("q")``; a plain ``list`` (as gRPC repeated fields decode to) makes
    that concatenation raise ``TypeError``. ``array("q", x)`` accepts any
    iterable of ints, so this is safe at every call site. Returns ``None`` for
    ``None`` input.
    """
    if token_ids is None:
        return None
    return array("q", token_ids)


_HTTP_TO_GRPC_CODE = {
    HTTPStatus.BAD_REQUEST: grpc.StatusCode.INVALID_ARGUMENT,
    HTTPStatus.SERVICE_UNAVAILABLE: grpc.StatusCode.UNAVAILABLE,
    HTTPStatus.INTERNAL_SERVER_ERROR: grpc.StatusCode.INTERNAL,
}


def scheduler_abort(output: dict) -> dict | None:
    """The scheduler's own abort of the request, when the output's finish
    reason carries one with a status code (its refusal of the caller's input,
    a full queue, an internal failure): the request did not complete and the
    status says whose fault it is, so the stream ends with a gRPC status (see
    :func:`abort_code_from_output`), not with a completion. A plain abort with
    no status code (the caller's own cancel) is not the scheduler's and stays a
    completion; so does any other finish. Returns the finish-reason dict."""
    finish_reason = output.get("meta_info", {}).get("finish_reason")
    if (
        isinstance(finish_reason, dict)
        and finish_reason.get("type") == "abort"
        and finish_reason.get("status_code") is not None
    ):
        return finish_reason
    return None


def abort_code_from_output(output: dict) -> grpc.StatusCode:
    """Map a scheduler error output to the appropriate gRPC status code."""
    finish_reason = output.get("meta_info", {}).get("finish_reason")
    if isinstance(finish_reason, dict):
        status_code = finish_reason.get("status_code")
        if status_code is not None:
            return _HTTP_TO_GRPC_CODE.get(status_code, grpc.StatusCode.INTERNAL)
    return grpc.StatusCode.INTERNAL
