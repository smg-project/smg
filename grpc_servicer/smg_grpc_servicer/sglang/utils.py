"""gRPC utility functions."""

from __future__ import annotations

from array import array
from collections.abc import Iterable

import grpc

from smg_grpc_servicer.engine_status import grpc_code_for_http_status
from smg_grpc_servicer.sglang import GRAMMAR_REJECTED_ERR_TYPE


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


def grammar_abort(output: dict) -> dict | None:
    """The scheduler's abort of the request that the engine itself attributes
    to the request's grammar: the finish-reason dict whose ``err_type`` is the
    name this package's plugin stamps on the scheduler's own grammar failure
    (:data:`~smg_grpc_servicer.sglang.GRAMMAR_REJECTED_ERR_TYPE`, see
    ``zmq_plugin``), with or without the status a PD decode side adds to it.
    The servicer ends such a request with the constraint-failure status every
    wire shares (:func:`smg_grpc_servicer.engine_status.abort_for_constraint_failure`).
    ``None`` for every other finish: an abort with a status is the scheduler's
    own (see :func:`scheduler_abort`); a bare abort without the name (an
    abort-all, a session's end, a scheduler without the plugin's hooks) stays
    what it was, whatever the request carries."""
    finish_reason = output.get("meta_info", {}).get("finish_reason")
    if (
        isinstance(finish_reason, dict)
        and finish_reason.get("type") == "abort"
        and finish_reason.get("err_type") == GRAMMAR_REJECTED_ERR_TYPE
    ):
        return finish_reason
    return None


def abort_code_from_output(output: dict) -> grpc.StatusCode:
    """Map a scheduler error output to the appropriate gRPC status code."""
    finish_reason = output.get("meta_info", {}).get("finish_reason")
    if isinstance(finish_reason, dict):
        status_code = finish_reason.get("status_code")
        if status_code is not None:
            return grpc_code_for_http_status(status_code)
    return grpc.StatusCode.INTERNAL
