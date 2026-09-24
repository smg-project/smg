"""Unit tests for the vLLM gRPC health service.

Run with: pytest grpc_servicer/tests/test_vllm_health_servicer.py
"""

import importlib.util
import logging
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
from grpc_health.v1 import health_pb2


@contextmanager
def _borrowed_vllm_logger():
    """Stand in for vllm's logger factory, the module's only use of vllm.

    The stand-in is withdrawn once the module is loaded, because other tests in
    this suite assert that importing the package does not pull vllm in.
    """
    if "vllm.logger" in sys.modules:
        yield
        return
    vllm = types.ModuleType("vllm")
    vllm_logger = types.ModuleType("vllm.logger")
    vllm_logger.init_logger = logging.getLogger
    vllm.logger = vllm_logger
    previous = {name: sys.modules.get(name) for name in ("vllm", "vllm.logger")}
    sys.modules["vllm"] = vllm
    sys.modules["vllm.logger"] = vllm_logger
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


_MODULE_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "health_servicer.py"
_spec = importlib.util.spec_from_file_location("health_servicer", _MODULE_PATH)
health_servicer = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = health_servicer
with _borrowed_vllm_logger():
    _spec.loader.exec_module(health_servicer)


class _Engine:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls = 0

    async def check_health(self):
        self.calls += 1
        if self.error is not None:
            raise self.error


class _Context:
    def __init__(self):
        self.code = None
        self.details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


def _request(service: str = ""):
    return health_pb2.HealthCheckRequest(service=service)


def _servicer(error: Exception | None = None):
    engine = _Engine(error)
    return health_servicer.VllmHealthServicer(engine), engine


class TestCheck:
    @pytest.mark.asyncio
    async def test_a_live_engine_is_serving(self):
        servicer, engine = _servicer()
        response = await servicer.Check(_request(), _Context())
        assert response.status == health_pb2.HealthCheckResponse.SERVING
        assert engine.calls == 1

    @pytest.mark.asyncio
    async def test_a_healthy_probe_says_nothing(self, caplog):
        servicer, _ = _servicer()
        with caplog.at_level(logging.DEBUG):
            await servicer.Check(_request(), _Context())
        assert caplog.records == []

    @pytest.mark.asyncio
    async def test_a_refused_probe_is_reported_once(self, caplog):
        servicer, engine = _servicer(RuntimeError("engine core is gone"))
        with caplog.at_level(logging.DEBUG):
            for _ in range(5):
                response = await servicer.Check(_request(), _Context())
                assert response.status == health_pb2.HealthCheckResponse.NOT_SERVING
        assert engine.calls == 5
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.ERROR
        assert caplog.records[0].exc_info is not None

    @pytest.mark.asyncio
    async def test_shutdown_answers_without_asking_the_engine(self):
        servicer, engine = _servicer()
        servicer.set_not_serving()
        response = await servicer.Check(_request(), _Context())
        assert response.status == health_pb2.HealthCheckResponse.NOT_SERVING
        assert engine.calls == 0

    @pytest.mark.asyncio
    async def test_an_unknown_service_is_not_found(self):
        servicer, engine = _servicer()
        context = _Context()
        response = await servicer.Check(_request("some.other.Service"), context)
        assert response.status == health_pb2.HealthCheckResponse.SERVICE_UNKNOWN
        assert context.code is not None
        assert engine.calls == 0


class TestWatch:
    @pytest.mark.asyncio
    async def test_a_live_engine_is_serving(self):
        servicer, _ = _servicer()
        statuses = [message.status async for message in servicer.Watch(_request(), _Context())]
        assert statuses == [health_pb2.HealthCheckResponse.SERVING]

    @pytest.mark.asyncio
    async def test_an_unknown_service_does_not_set_an_rpc_code(self):
        servicer, _ = _servicer()
        context = _Context()
        statuses = [message.status async for message in servicer.Watch(_request("nope"), context)]
        assert statuses == [health_pb2.HealthCheckResponse.SERVICE_UNKNOWN]
        assert context.code is None

    @pytest.mark.asyncio
    async def test_a_refused_probe_shares_the_one_report_with_check(self, caplog):
        servicer, _ = _servicer(RuntimeError("engine core is gone"))
        with caplog.at_level(logging.DEBUG):
            await servicer.Check(_request(), _Context())
            async for _ in servicer.Watch(_request(), _Context()):
                pass
        assert len(caplog.records) == 1
