"""Exercise the real Responses fixtures through pytest without starting a backend."""

from __future__ import annotations

import json

import pytest
from infra import constants

pytest_plugins = ["pytester"]


@pytest.mark.parametrize("engine", ["sglang", "vllm"])
@pytest.mark.parametrize(
    "stage", ["worker", "gateway", "client", "ok", "runtime-skip", "model-skip"]
)
def test_required_responses_backend_errors_are_not_skips(pytester, monkeypatch, engine, stage):
    monkeypatch.setenv("E2E_RUNTIME", "tokenspeed" if stage == "model-skip" else engine)
    monkeypatch.setattr(constants, "_RUNTIME_CACHE", None)
    events = pytester.path / "events.json"
    pytester.makeconftest(
        f"""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import infra
from infra import worker_pool
from fixtures.hooks import pytest_configure, pytest_runtest_setup
from responses import conftest as responses
from responses.conftest import (
    gateway_with_mock_mcp_grpc_sglang, gateway_with_mock_mcp_grpc_vllm,
)
EVENTS = Path({str(events)!r})
STAGE = {stage!r}

def record(event):
    previous = json.loads(EVENTS.read_text()) if EVENTS.exists() else []
    EVENTS.write_text(json.dumps(previous + [event]))

class Pool:
    def acquire(self, **kwargs):
        record("acquire")
        if STAGE == "worker":
            raise RuntimeError("worker startup failed")
        return [SimpleNamespace(base_url="grpc://offline", stop=lambda: record("worker-stop"))]

class Gateway:
    base_url = "http://offline.invalid"
    def start(self, **kwargs):
        record("gateway-start")
        if STAGE == "gateway":
            raise RuntimeError("gateway startup failed")
    def shutdown(self):
        record("gateway-stop")

def client(**kwargs):
    record("client-init")
    if STAGE == "client":
        raise RuntimeError("client initialization failed")
    return object()

@pytest.fixture(scope="session", autouse=True)
def offline_backend():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(worker_pool, "get_pool", lambda: Pool())
        patch.setattr(infra, "Gateway", Gateway)
        patch.setattr(responses.openai, "OpenAI", client)
        yield

@pytest.fixture(scope="session")
def mock_mcp_server():
    return object()

@pytest.fixture(scope="session")
def mock_mcp_config_file():
    return "offline-config"
"""
    )
    markers = ""
    if stage == "runtime-skip":
        markers = f"@pytest.mark.skip_for_runtime({engine!r}, reason='explicit unsupported case')"
    elif stage == "model-skip":
        markers = "@pytest.mark.model('openai/gpt-oss-20b')"
    pytester.makepyfile(
        f"""
import pytest
{markers}
class TestActualResponseFixture:
    def test_selected_gpu_backend(self, gateway_with_mock_mcp_grpc_{engine}):
        assert len(gateway_with_mock_mcp_grpc_{engine}) == 4
"""
    )
    result = pytester.runpytest("-q", "-o", "addopts=")
    observed = json.loads(events.read_text()) if events.exists() else []
    if stage.endswith("skip"):
        result.assert_outcomes(skipped=1, errors=0)
        assert observed == []
    elif stage == "ok":
        result.assert_outcomes(passed=1, errors=0, skipped=0)
        assert observed == ["acquire", "gateway-start", "client-init", "gateway-stop"]
    else:
        result.assert_outcomes(errors=1, skipped=0)
        result.stdout.fnmatch_lines([f"*RuntimeError: {stage} *failed*"])
        expected = ["acquire"]
        if stage != "worker":
            expected += ["gateway-start"] + (["client-init"] if stage == "client" else [])
            expected += ["gateway-stop"]
        assert observed == expected
