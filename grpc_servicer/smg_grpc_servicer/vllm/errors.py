"""gRPC status codes for engine exceptions, following vLLM's client/server error split."""

from http import HTTPStatus

import grpc
from vllm.exceptions import VLLMNotFoundError
from vllm.v1.engine.exceptions import EngineGenerateError

from smg_grpc_servicer.engine_status import constraint_refused_trailer, grpc_code_for_http_status

try:
    from vllm.exceptions import VLLMClientError
except ImportError:  # vLLM < 0.27 raises ValueError for request validation
    VLLMClientError = ValueError

try:
    from vllm.exceptions import GracefulHTTPError
except ImportError:  # vLLM < 0.29 has no admission-control rejections

    class GracefulHTTPError(Exception):
        http_status = HTTPStatus.INTERNAL_SERVER_ERROR


def grpc_code_for(exc: BaseException) -> grpc.StatusCode:
    """gRPC code for an engine failure: client-caused errors are INVALID_ARGUMENT."""
    if isinstance(exc, EngineGenerateError) and exc.__cause__ is not None:
        exc = exc.__cause__
    if isinstance(exc, GracefulHTTPError):
        return grpc_code_for_http_status(exc.http_status)
    if isinstance(exc, VLLMNotFoundError):
        return grpc.StatusCode.NOT_FOUND
    if isinstance(exc, (VLLMClientError, ValueError)):
        return grpc.StatusCode.INVALID_ARGUMENT
    return grpc.StatusCode.INTERNAL


# How vLLM's scheduler words a grammar it failed to compile (RequestError "validation_error").
_COMPILE_FAILURE_PREFIX = "Invalid structured outputs: "


def grpc_trailer_for(exc: BaseException) -> tuple[tuple[str, str], ...]:
    """Trailing metadata for an engine failure: a refused constraint's for a failed compile."""
    if isinstance(exc, EngineGenerateError) and exc.__cause__ is not None:
        exc = exc.__cause__
    if isinstance(exc, (VLLMClientError, ValueError)) and str(exc).startswith(
        _COMPILE_FAILURE_PREFIX
    ):
        return constraint_refused_trailer()
    return ()


def grpc_refusal_message(message: str) -> str:
    """A failed compile's message, saying the grammar failed to compile."""
    head = _COMPILE_FAILURE_PREFIX.rstrip()
    if not message.startswith(head):
        return message
    reason = message[len(head) :].lstrip()
    return f"{head} the grammar failed to compile: {reason}".rstrip()
