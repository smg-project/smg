"""Cloud MCP readiness shares each existing startup attempt's deadline."""

from __future__ import annotations

import ast
import logging
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from infra import gateway as gateway_module


@pytest.fixture(autouse=True)
def forbid_external_work(monkeypatch):
    def forbid(*_, **__):
        pytest.fail("CPU cloud fixture attempted an external process or HTTP request")

    monkeypatch.setattr(subprocess, "Popen", forbid)
    monkeypatch.setattr(httpx.Client, "send", forbid)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbid)


@pytest.fixture
def cloud_parts(monkeypatch):
    state = SimpleNamespace(now=0.0, gateways=[], start_seconds=2, start_errors=[])

    class Gateway:
        def __init__(self):
            self.base_url = "http://127.0.0.1:1"
            self.log_path = None
            self._started = False
            self.registered = False
            self.shutdown_states = []
            self.index = len(state.gateways)
            state.gateways.append(self)

        def start(self, **kwargs):
            self.start_kwargs = kwargs
            state.now += state.start_seconds
            if self.index < len(state.start_errors) and state.start_errors[self.index] is not None:
                raise state.start_errors[self.index]
            self._started = True

        def shutdown(self):
            self.shutdown_states.append(self._started)
            self._started = False

    monkeypatch.setattr(gateway_module, "Gateway", Gateway)
    monkeypatch.setattr(gateway_module, "time", SimpleNamespace(monotonic=lambda: state.now))
    return state


def _fixture():
    # Execute the actual cloud fixture, without importing GPU test classes.
    path = Path(__file__).parents[1] / "responses/test_builtin_tools.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "gateway_with_mcp_config"
    )
    node.decorator_list = []
    namespace = {
        "os": os,
        "logger": logging.getLogger(__name__),
        "pytest": pytest,
        "openai": SimpleNamespace(OpenAI=lambda **_: object()),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["gateway_with_mcp_config"]


def test_default_none_keeps_existing_start_arguments_and_return(cloud_parts):
    state = cloud_parts
    result = gateway_module.launch_cloud_gateway("openai", extra_args=["--existing"])
    assert result is state.gateways[0]
    assert len(state.gateways) == 1 and not result.shutdown_states
    assert result.start_kwargs == {
        "cloud_backend": "openai",
        "history_backend": "memory",
        "timeout": 60,
        "show_output": None,
        "extra_args": ["--existing"],
    }


def test_default_none_keeps_three_startup_timeout_attempts(cloud_parts):
    state = cloud_parts
    state.start_seconds = 60
    state.start_errors = [TimeoutError("private startup timeout") for _ in range(3)]
    with pytest.raises(TimeoutError, match="private startup timeout"):
        gateway_module.launch_cloud_gateway("openai")
    assert state.now == 180 and len(state.gateways) == 3
    assert all(gateway.start_kwargs["timeout"] == 60 for gateway in state.gateways)
    assert all(len(gateway.shutdown_states) == 1 for gateway in state.gateways)


def test_callback_gets_each_distinct_original_attempt_deadline(cloud_parts):
    state = cloud_parts
    state.start_seconds = 59
    deadlines = []

    def register(gateway, deadline):
        deadlines.append(deadline)
        assert deadline - state.now == 1
        if len(deadlines) < 3:
            state.now = deadline
            raise TimeoutError("private registration timeout")
        state.now += 0.5
        gateway.registered = True

    result = gateway_module.launch_cloud_gateway("openai", post_start=register)
    assert deadlines == [60, 120, 180]
    assert state.now == 179.5
    assert result is state.gateways[2] and result.registered
    assert [g.shutdown_states for g in state.gateways] == [[False], [False], []]
    assert all(g.start_kwargs["timeout"] == 60 for g in state.gateways)


def test_callback_timeout_exhausts_only_original_three_attempts(cloud_parts):
    state = cloud_parts
    state.start_seconds = 59.75
    deadlines = []

    def fail(_, deadline):
        deadlines.append(deadline)
        assert deadline - state.now == 0.25
        state.now = deadline
        raise TimeoutError("private MCP timeout")

    with pytest.raises(TimeoutError, match="private MCP timeout"):
        gateway_module.launch_cloud_gateway("openai", post_start=fail)
    assert deadlines == [60, 120, 180]
    assert state.now == 180 and len(state.gateways) == 3
    assert all(g.shutdown_states == [False] for g in state.gateways)


def test_callback_runtime_error_cleans_up_and_propagates_without_retry(cloud_parts):
    state = cloud_parts

    def fail(*_):
        raise RuntimeError("private MCP process exit")

    with pytest.raises(RuntimeError, match="private MCP process exit"):
        gateway_module.launch_cloud_gateway("openai", post_start=fail)
    assert len(state.gateways) == 1
    assert state.gateways[0].shutdown_states == [False]


def test_custom_original_timeout_and_attempt_count_are_preserved(cloud_parts):
    state = cloud_parts
    state.start_seconds = 4
    deadlines = []

    def fail(_, deadline):
        deadlines.append(deadline)
        state.now = deadline
        raise TimeoutError("private custom attempt")

    with pytest.raises(TimeoutError, match="private custom attempt"):
        gateway_module.launch_cloud_gateway("openai", timeout=5, max_attempts=2, post_start=fail)
    assert deadlines == [5, 10] and state.now == 10
    assert [g.start_kwargs["timeout"] for g in state.gateways] == [5, 5]


def test_cloud_fixture_waits_for_mcp_before_yield(monkeypatch, cloud_parts):
    import infra

    state = cloud_parts
    monkeypatch.setenv("OPENAI_API_KEY", "private-cpu-placeholder")
    monkeypatch.setattr(infra, "launch_cloud_gateway", gateway_module.launch_cloud_gateway)
    fixture = _fixture()
    deadlines = []

    def register(gateway, *, deadline):
        deadlines.append(deadline)
        gateway.registered = True

    fixture.__globals__["wait_for_single_mcp_registration"] = register
    generator = fixture(None, "unused.yaml")
    try:
        gateway, _ = next(generator)
        assert gateway.registered, "cloud fixture yielded before static MCP registration"
        assert deadlines == [60]
    finally:
        generator.close()
    assert state.gateways[0].shutdown_states == [True]


def test_cloud_client_constructor_failure_shuts_down(monkeypatch, cloud_parts):
    import infra

    state = cloud_parts
    monkeypatch.setenv("OPENAI_API_KEY", "private-cpu-placeholder")
    monkeypatch.setattr(infra, "launch_cloud_gateway", gateway_module.launch_cloud_gateway)
    fixture = _fixture()
    fixture.__globals__["wait_for_single_mcp_registration"] = lambda *_, **__: None

    def fail(**_):
        raise RuntimeError("private client construction failure")

    fixture.__globals__["openai"].OpenAI = fail
    with pytest.raises(RuntimeError, match="private client construction failure"):
        next(fixture(None, "unused.yaml"))
    assert len(state.gateways) == 1
    assert state.gateways[0].shutdown_states == [True]


def test_callback_failure_preserves_actual_temporary_log_without_dumping_it(monkeypatch):
    monkeypatch.delenv("E2E_LOG_DIR", raising=False)
    gateway = gateway_module.Gateway(port=12345, prometheus_port=12346)
    sentinel = "private cloud diagnostic must stay in its log"
    log = gateway._open_log_file()
    log.write(sentinel)
    log.flush()
    log_path = Path(gateway.log_path)

    def started(**_):
        gateway._started = True

    def fail(*_):
        raise RuntimeError("private cloud readiness failure")

    gateway.start = started
    monkeypatch.setattr(gateway_module, "Gateway", lambda: gateway)
    try:
        with pytest.raises(RuntimeError, match="private cloud readiness failure") as error:
            gateway_module.launch_cloud_gateway("openai", post_start=fail)
        assert gateway._log_file is None and log_path.is_file()
        notes = "\n".join(getattr(error.value, "__notes__", []))
        assert str(log_path) in notes
        assert sentinel not in str(error.value) and sentinel not in notes
    finally:
        gateway.shutdown()
        log_path.unlink(missing_ok=True)


def test_callback_cleanup_error_does_not_start_another_attempt(cloud_parts):
    state = cloud_parts
    cleanup_calls = []

    def fail(gateway, _):
        def cleanup():
            cleanup_calls.append(gateway.index)
            if len(cleanup_calls) == 1:
                raise TimeoutError("private cleanup failure")

        gateway.shutdown = cleanup
        raise RuntimeError("private readiness failure")

    with pytest.raises(TimeoutError, match="private cleanup failure") as error:
        gateway_module.launch_cloud_gateway("openai", post_start=fail)
    assert len(state.gateways) == 1 and cleanup_calls == [0]
    assert isinstance(error.value.__context__, RuntimeError)


def test_retried_registration_log_path_is_reported_even_when_next_attempt_succeeds(
    monkeypatch, caplog
):
    monkeypatch.delenv("E2E_LOG_DIR", raising=False)
    gateway_class = gateway_module.Gateway
    gateways = []
    log_paths = []
    sentinel = "private diagnostic content must not appear in the warning"

    def factory():
        index = len(gateways)
        gateway = gateway_class(port=12345 + index * 2, prometheus_port=12346 + index * 2)
        log = gateway._open_log_file()
        log.write(sentinel)
        log.flush()
        log_paths.append(Path(gateway.log_path))

        def started(**_):
            gateway._started = True

        gateway.start = started
        gateways.append(gateway)
        return gateway

    def register(gateway, _):
        if gateway is gateways[0]:
            raise TimeoutError("private first registration timeout")

    monkeypatch.setattr(gateway_module, "Gateway", factory)
    try:
        with caplog.at_level("WARNING", logger=gateway_module.__name__):
            result = gateway_module.launch_cloud_gateway("openai", post_start=register)
        assert len(gateways) == 2 and result is gateways[1]
        assert log_paths[0].is_file() and gateways[0]._log_file is None
        assert str(log_paths[0]) in caplog.text, "retried registration log has no reported path"
        assert "private first registration timeout" in caplog.text
        assert sentinel not in caplog.text
    finally:
        for gateway in gateways:
            gateway.shutdown()
        for path in log_paths:
            path.unlink(missing_ok=True)
