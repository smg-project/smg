"""gRPC status codes for engine exceptions, following vLLM's client/server error split."""

from http import HTTPStatus

import grpc
from vllm.exceptions import VLLMNotFoundError
from vllm.v1.engine.exceptions import EngineGenerateError

try:
    from vllm.exceptions import VLLMClientError
except ImportError:  # vLLM < 0.27 raises ValueError for request validation
    VLLMClientError = ValueError

try:
    from vllm.exceptions import GracefulHTTPError
except ImportError:  # vLLM < 0.29 has no admission-control rejections

    class GracefulHTTPError(Exception):
        http_status = HTTPStatus.INTERNAL_SERVER_ERROR


_HTTP_TO_GRPC_CODE = {
    HTTPStatus.BAD_REQUEST: grpc.StatusCode.INVALID_ARGUMENT,
    HTTPStatus.NOT_FOUND: grpc.StatusCode.NOT_FOUND,
    HTTPStatus.TOO_MANY_REQUESTS: grpc.StatusCode.RESOURCE_EXHAUSTED,
    HTTPStatus.SERVICE_UNAVAILABLE: grpc.StatusCode.UNAVAILABLE,
}


def grpc_code_for(exc: BaseException) -> grpc.StatusCode:
    """gRPC code for an engine failure: client-caused errors are INVALID_ARGUMENT."""
    if isinstance(exc, EngineGenerateError) and exc.__cause__ is not None:
        exc = exc.__cause__
    if isinstance(exc, GracefulHTTPError):
        return _HTTP_TO_GRPC_CODE.get(exc.http_status, grpc.StatusCode.INTERNAL)
    if isinstance(exc, VLLMNotFoundError):
        return grpc.StatusCode.NOT_FOUND
    if isinstance(exc, (VLLMClientError, ValueError)):
        return grpc.StatusCode.INVALID_ARGUMENT
    return grpc.StatusCode.INTERNAL
