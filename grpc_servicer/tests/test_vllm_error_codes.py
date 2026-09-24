"""Engine-free coverage of the vLLM exception -> gRPC status mapping."""

import builtins
import importlib.util
import sys
from http import HTTPStatus
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import grpc
import pytest

_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "errors.py"


def _hierarchy_main():
    """vLLM >= 0.29: client/server split plus admission-control errors."""

    class VLLMError(Exception):
        pass

    class VLLMClientError(VLLMError):
        pass

    class VLLMServerError(VLLMError):
        pass

    class VLLMValidationError(VLLMClientError):
        pass

    class VLLMNotFoundError(VLLMClientError):
        pass

    class VLLMUnprocessableEntityError(VLLMClientError):
        pass

    class GracefulHTTPError(VLLMError):
        def __init__(self, message, http_status):
            super().__init__(message)
            self.http_status = http_status

    class EngineGenerateError(VLLMServerError):
        pass

    return locals()


def _hierarchy_0_27():
    """vLLM 0.27-0.28: client/server split, no GracefulHTTPError."""
    classes = _hierarchy_main()
    del classes["GracefulHTTPError"]
    return classes


def _hierarchy_release():
    """vLLM 0.19-0.26: validation errors are ValueErrors, no client/server split."""

    class VLLMValidationError(ValueError):
        pass

    class VLLMNotFoundError(Exception):
        pass

    class EngineGenerateError(Exception):
        pass

    return locals()


def _load_errors(monkeypatch, classes):
    """Load errors.py against a stub ``vllm`` exposing ``classes``."""
    for name in ("vllm", "vllm.v1", "vllm.v1.engine"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    exceptions = ModuleType("vllm.exceptions")
    exceptions.__dict__.update(classes)
    monkeypatch.setitem(sys.modules, "vllm.exceptions", exceptions)
    engine_exceptions = ModuleType("vllm.v1.engine.exceptions")
    engine_exceptions.EngineGenerateError = classes["EngineGenerateError"]
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.exceptions", engine_exceptions)
    spec = importlib.util.spec_from_file_location("vllm_errors_under_test", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=[_hierarchy_main, _hierarchy_0_27], ids=["main", "0.27"])
def split_hierarchy(request, monkeypatch):
    classes = request.param()
    return _load_errors(monkeypatch, classes), classes


@pytest.mark.parametrize(
    "name", ["VLLMValidationError", "VLLMUnprocessableEntityError", "VLLMClientError"]
)
def test_client_errors_map_to_invalid_argument(split_hierarchy, name):
    errors, classes = split_hierarchy
    assert errors.grpc_code_for(classes[name]("bad")) is grpc.StatusCode.INVALID_ARGUMENT


def test_value_error_still_maps_to_invalid_argument(split_hierarchy):
    errors, _ = split_hierarchy
    assert errors.grpc_code_for(ValueError("bad")) is grpc.StatusCode.INVALID_ARGUMENT


def test_an_oversized_media_result_is_the_callers_error(split_hierarchy):
    # The sidecar's result_too_large answer reaches the servicer as this ValueError.
    errors, _ = split_hierarchy
    too_large = ValueError("media_too_large: encoded media result is 600000000 bytes")
    assert errors.grpc_code_for(too_large) is grpc.StatusCode.INVALID_ARGUMENT


def test_not_found_maps_to_not_found(split_hierarchy):
    errors, classes = split_hierarchy
    assert errors.grpc_code_for(classes["VLLMNotFoundError"]("lora")) is grpc.StatusCode.NOT_FOUND


@pytest.mark.parametrize("name", ["VLLMServerError", "RuntimeError"])
def test_server_errors_map_to_internal(split_hierarchy, name):
    errors, classes = split_hierarchy
    exc_type = classes.get(name) or getattr(builtins, name)
    assert errors.grpc_code_for(exc_type("boom")) is grpc.StatusCode.INTERNAL


@pytest.mark.parametrize(
    "cause,expected",
    [
        ("VLLMValidationError", grpc.StatusCode.INVALID_ARGUMENT),
        ("VLLMNotFoundError", grpc.StatusCode.NOT_FOUND),
        ("VLLMServerError", grpc.StatusCode.INTERNAL),
        (None, grpc.StatusCode.INTERNAL),
    ],
)
def test_engine_generate_error_is_classified_by_its_cause(split_hierarchy, cause, expected):
    errors, classes = split_hierarchy
    exc = classes["EngineGenerateError"]()
    if cause is not None:
        exc.__cause__ = classes[cause]("wrapped")
    assert errors.grpc_code_for(exc) is expected


@pytest.mark.parametrize(
    "http_status,expected",
    [
        (HTTPStatus.BAD_REQUEST, grpc.StatusCode.INVALID_ARGUMENT),
        (HTTPStatus.NOT_FOUND, grpc.StatusCode.NOT_FOUND),
        (HTTPStatus.TOO_MANY_REQUESTS, grpc.StatusCode.RESOURCE_EXHAUSTED),
        (HTTPStatus.SERVICE_UNAVAILABLE, grpc.StatusCode.UNAVAILABLE),
        (HTTPStatus.INTERNAL_SERVER_ERROR, grpc.StatusCode.INTERNAL),
        (HTTPStatus.IM_A_TEAPOT, grpc.StatusCode.INTERNAL),
    ],
)
def test_graceful_http_error_maps_by_http_status(monkeypatch, http_status, expected):
    classes = _hierarchy_main()
    errors = _load_errors(monkeypatch, classes)
    exc = classes["GracefulHTTPError"]("busy", http_status)
    assert errors.grpc_code_for(exc) is expected


def test_release_hierarchy_without_vllm_client_error_still_maps_validation_to_invalid_argument(
    monkeypatch,
):
    classes = _hierarchy_release()
    errors = _load_errors(monkeypatch, classes)
    assert (
        errors.grpc_code_for(classes["VLLMValidationError"]("bad"))
        is grpc.StatusCode.INVALID_ARGUMENT
    )
    assert errors.grpc_code_for(classes["VLLMNotFoundError"]("lora")) is grpc.StatusCode.NOT_FOUND
    assert errors.grpc_code_for(RuntimeError("boom")) is grpc.StatusCode.INTERNAL
    wrapped = classes["EngineGenerateError"]()
    wrapped.__cause__ = classes["VLLMValidationError"]("wrapped")
    assert errors.grpc_code_for(wrapped) is grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_generate_aborts_with_invalid_argument_for_vllm_validation_error():
    pytest.importorskip("vllm")
    from smg_grpc_proto import vllm_engine_pb2
    from smg_grpc_servicer.vllm.servicer import VllmEngineServicer
    from vllm.exceptions import VLLMValidationError

    async def generate(**kwargs):
        raise VLLMValidationError("Invalid grammar specification")
        yield  # pragma: no cover

    engine = SimpleNamespace(
        generate=generate,
        renderer=SimpleNamespace(process_for_engine=lambda prompt, **kwargs: prompt),
    )
    servicer = VllmEngineServicer(engine, start_time=0.0)
    request = vllm_engine_pb2.GenerateRequest(
        request_id="req-1",
        tokenized=vllm_engine_pb2.TokenizedInput(input_ids=[1, 2, 3]),
        sampling_params=vllm_engine_pb2.SamplingParams(),
    )

    class Aborted(Exception):
        pass

    context = SimpleNamespace(abort=AsyncMock(side_effect=Aborted()))
    with pytest.raises(Aborted):
        async for _ in servicer.Generate(request, context):
            pass
    context.abort.assert_awaited_once_with(
        grpc.StatusCode.INVALID_ARGUMENT, "Invalid grammar specification"
    )
