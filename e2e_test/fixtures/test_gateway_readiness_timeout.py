"""Unit tests for the gateway readiness wait of ``setup_backend``."""

import importlib

from infra import ENV_STARTUP_TIMEOUT, ConnectionMode

# The package's ``setup_backend`` is the fixture; the module holds the helper.
setup = importlib.import_module("fixtures.setup_backend")

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def test_grpc_and_http_wait_the_router_timeout(monkeypatch):
    monkeypatch.delenv(ENV_STARTUP_TIMEOUT, raising=False)
    for mode in (ConnectionMode.GRPC, ConnectionMode.HTTP):
        assert setup._gateway_readiness_timeout(mode, MODEL, 60) == 60


def test_startup_floor_raises_the_grpc_and_http_wait(monkeypatch):
    monkeypatch.setenv(ENV_STARTUP_TIMEOUT, "900")
    for mode in (ConnectionMode.GRPC, ConnectionMode.HTTP):
        assert setup._gateway_readiness_timeout(mode, MODEL, 60) == 900
        assert setup._gateway_readiness_timeout(mode, MODEL, 1200) == 1200
