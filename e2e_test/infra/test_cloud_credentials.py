"""CPU checks through actual credential consumers; no gateway/provider I/O."""

from __future__ import annotations

import ast
import logging
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
from infra import gateway as gateway_module
from infra.cloud_credentials import (
    PROXY_ENV,
    PROXY_PLACEHOLDER_KEY,
    cloud_environment,
    resolve_cloud_credentials,
)
from infra.model_specs import THIRD_PARTY_MODELS

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "XAI_API_KEY")


@pytest.fixture(autouse=True)
def forbid_external_work(monkeypatch):
    def forbid(*_, **__):
        pytest.fail("CPU cloud consumer attempted process/provider I/O")

    monkeypatch.setattr(subprocess, "Popen", forbid)
    monkeypatch.setattr(httpx.Client, "send", forbid)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbid)


class _NoKeyReads(dict):
    def __getitem__(self, key):
        assert key not in KEYS, f"proxy mode read {key}"
        return super().__getitem__(key)

    def get(self, key, default=None):
        assert key not in KEYS, f"proxy mode read {key}"
        return super().get(key, default)


@pytest.mark.parametrize("provider", ["openai", "anthropic", "xai"])
def test_proxy_resolution_never_reads_provider_keys(provider):
    env = _NoKeyReads(
        {PROXY_ENV: "http://127.0.0.1:8787/", **dict.fromkeys(KEYS, "real-key-forbidden")}
    )
    value = resolve_cloud_credentials(provider, environ=env)
    assert value.worker_root == f"http://127.0.0.1:8787/{provider}"
    assert value.sdk_base_url == value.worker_root + ("" if provider == "anthropic" else "/v1")
    assert value.api_key == PROXY_PLACEHOLDER_KEY
    assert value.is_proxy
    child = cloud_environment(env, proxy=True)
    assert not any(key in child for key in KEYS)


@pytest.mark.parametrize(
    "proxy",
    [
        "https://evil.example",
        "http://127.0.0.1:8787/openai",
        "http://127.0.0.1:8787/v1",
        "http://127.0.0.1",
        "http://user:pass@127.0.0.1:8787",
        "http://127.0.0.1:8787?x=1",
        "http://127.0.0.1:8787/#fragment",
        "http://127.0.0.1:8787?",
        "http://127.0.0.1:8787#",
        "\x00http://127.0.0.1:8787",
        "http://127.0.0.1:99999",
        "http://127.0.0.1:8787\n",
    ],
)
def test_bad_proxy_never_falls_back_to_external(proxy):
    with pytest.raises(ValueError):
        resolve_cloud_credentials("openai", environ=_NoKeyReads({PROXY_ENV: proxy}))


@pytest.mark.parametrize(
    ("provider", "key", "url"),
    [
        ("openai", "OPENAI_API_KEY", "https://api.openai.com"),
        ("xai", "XAI_API_KEY", "https://api.x.ai"),
        ("anthropic", "ANTHROPIC_API_KEY", "https://api.anthropic.com"),
    ],
)
def test_direct_mode_retains_existing_keys_and_endpoints(provider, key, url):
    env = {key: "developer-key", "KEEP": "value"}
    value = resolve_cloud_credentials(provider, environ=env)
    assert value.worker_root == url
    assert value.api_key == "developer-key" and not value.is_proxy
    assert value.sdk_base_url is None  # SDK default/env retained in direct mode.
    assert cloud_environment(env, proxy=False) == env
    assert resolve_cloud_credentials(provider, require_key=False, environ={}).api_key == ""
    with pytest.raises(ValueError, match=key):
        resolve_cloud_credentials(provider, environ={})


@pytest.mark.parametrize("provider", ["openai", "anthropic", "xai"])
def test_gateway_actual_arguments_and_child_env_have_only_placeholder(provider, monkeypatch):
    env = _NoKeyReads(
        {PROXY_ENV: "http://127.0.0.1:8787", **dict.fromkeys(KEYS, "forbidden"), "KEEP": "yes"}
    )
    gateway = gateway_module.Gateway()
    gateway._env = env
    args = gateway._build_cloud_args(provider, "memory")
    assert args[args.index("--worker-urls") + 1] == f"http://127.0.0.1:8787/{provider}"
    assert args[args.index("--backend") + 1] == (
        "anthropic" if provider == "anthropic" else "openai"
    )
    selected = resolve_cloud_credentials(provider, environ=env).api_key_env
    assert gateway._env == {
        PROXY_ENV: env[PROXY_ENV],
        "KEEP": "yes",
        selected: PROXY_PLACEHOLDER_KEY,
    }


def _consumer(path, name, namespace):
    """Load the exact fixture body without unrelated GPU/model test imports."""
    file = ROOT / path
    node = next(
        node
        for node in ast.parse(file.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    globals_ = {
        "pytest": pytest,
        "os": os,
        "logger": logging.getLogger(__name__),
        "resolve_cloud_credentials": resolve_cloud_credentials,
        "openai": openai,
        **namespace,
    }
    exec(compile(module, str(file), "exec"), globals_)
    return globals_[name]


@pytest.fixture
def consumer_parts(monkeypatch):
    import infra

    monkeypatch.setenv(PROXY_ENV, "http://127.0.0.1:8787")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://must-not-be-used.example/v1")
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    gateways = []

    def launch(*args, **kwargs):
        gateway = SimpleNamespace(
            base_url="http://127.0.0.1:9000",
            ws_base_url="ws://127.0.0.1:9000",
            closed=False,
            launch_args=args,
            launch_kwargs=kwargs,
        )
        gateway.shutdown = lambda: setattr(gateway, "closed", True)
        gateways.append(gateway)
        return gateway

    monkeypatch.setattr(infra, "launch_cloud_gateway", launch)
    return launch, gateways


@pytest.mark.parametrize("provider", ["openai", "xai", "anthropic"])
def test_setup_backend_keeps_client_to_smg_and_model(provider, consumer_parts):
    launch, gateways = consumer_parts
    anthropic_calls = []

    # Anthropic SDK is an external constructor boundary; no new bootstrap.
    def anthropic_client(**kwargs):
        anthropic_calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    function = _consumer(
        "fixtures/setup_backend.py",
        "_setup_cloud",
        {
            "THIRD_PARTY_MODELS": THIRD_PARTY_MODELS,
            "launch_cloud_gateway": launch,
            "get_marker_value": lambda *_, default, **__: default,
            "anthropic": SimpleNamespace(Anthropic=anthropic_client),
        },
    )
    result = function(provider, object(), {})
    backend, model, client, gateway = next(result)
    assert backend == provider and model == THIRD_PARTY_MODELS[provider]["model"]
    assert client.api_key == PROXY_PLACEHOLDER_KEY
    if provider == "anthropic":
        assert anthropic_calls == [{"base_url": gateway.base_url, "api_key": PROXY_PLACEHOLDER_KEY}]
    else:
        assert str(client.base_url) == f"{gateway.base_url}/v1/"
        client.close()
    result.close()
    assert gateways[-1].closed


@pytest.mark.parametrize(
    ("path", "name", "args"),
    [
        (
            "responses/conftest.py",
            "gateway_with_mock_mcp_cloud",
            (SimpleNamespace(url="http://mock"), "mock-config"),
        ),
        (
            "responses/conftest.py",
            "gateway_with_mock_mcp_regular_cloud",
            (SimpleNamespace(url="http://mock"), "mock-config"),
        ),
        ("responses/test_builtin_tools.py", "gateway_with_mcp_config", (None, "mock-config")),
    ],
)
def test_cloud_tool_fixtures_use_placeholder_without_skip(path, name, args, consumer_parts):
    launch, gateways = consumer_parts
    function = _consumer(path, name, {"launch_cloud_gateway": launch})
    result = function(*args)
    value = next(result)
    gateway, client = value[:2]
    assert gateway is gateways[-1]
    assert client.api_key == PROXY_PLACEHOLDER_KEY
    assert str(client.base_url) == gateway.base_url + "/v1/"
    client.close()
    result.close()
    assert gateway.closed


def test_direct_file_setup_uses_actual_openai_sdk_proxy_v1(consumer_parts):
    function = _consumer("responses/test_annotations.py", "openai_direct_client", {})
    result = function()
    client = next(result)
    try:
        assert client.api_key == PROXY_PLACEHOLDER_KEY
        assert str(client.base_url) == "http://127.0.0.1:8787/openai/v1/"
        assert "/9000/" not in str(client.base_url)
    finally:
        client.close()
        result.close()


def test_realtime_fixture_and_headers_do_not_snapshot_a_real_key(consumer_parts):
    launch, gateways = consumer_parts
    function = _consumer("realtime/test_realtime_ws.py", "gateway", {})
    result = function()
    gateway = next(result)
    assert gateway is gateways[-1]
    headers = _consumer("realtime/test_realtime_ws.py", "ws_headers", {})()
    assert headers == {"Authorization": f"Bearer {PROXY_PLACEHOLDER_KEY}"}
    ws_url = _consumer(
        "realtime/test_realtime_ws.py", "ws_url", {"REALTIME_MODEL": "gpt-realtime"}
    )(gateway)
    assert ws_url == "ws://127.0.0.1:9000/v1/realtime?model=gpt-realtime"
    with pytest.raises(StopIteration):
        next(result)  # pytest resumes yield fixtures for their normal teardown.
    assert gateway.closed


def test_vendor_workflow_opts_in_only_existing_vendor_jobs():
    import yaml

    data = yaml.safe_load((ROOT.parent / ".github/workflows/pr-test-rust.yml").read_text())
    vendor = data["jobs"]["e2e-vendor"]
    assert vendor["env"][PROXY_ENV] == "http://127.0.0.1:8787"
    assert not any(key in vendor["env"] for key in KEYS)
    assert [row["name"] for row in vendor["strategy"]["matrix"]["include"]] == [
        "anthropic-messages",
        "openai-responses",
        "openai-realtime",
        "xai-responses",
    ]
    for name, job in data["jobs"].items():
        if name != "e2e-vendor":
            assert PROXY_ENV not in job.get("env", {})


def test_direct_file_setup_preserves_developer_sdk_environment(monkeypatch):
    monkeypatch.delenv(PROXY_ENV, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "developer-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:4444/custom/v1")
    function = _consumer("responses/test_annotations.py", "openai_direct_client", {})
    result = function()
    client = next(result)
    try:
        assert client.api_key == "developer-key"
        assert str(client.base_url) == "http://127.0.0.1:4444/custom/v1/"
    finally:
        client.close()
        result.close()


def test_loopback_origin_is_canonical_for_rust_realtime_scheme():
    value = resolve_cloud_credentials("openai", environ={PROXY_ENV: "HTTP://LOCALHOST:8787/"})
    assert value.worker_root == "http://localhost:8787/openai"


def test_actual_gateway_start_filters_inherited_and_explicit_proxy_keys(monkeypatch):
    inherited = _NoKeyReads(
        {PROXY_ENV: "http://127.0.0.1:8787", **dict.fromkeys(KEYS, "forbidden"), "KEEP": "yes"}
    )
    explicit = _NoKeyReads({"OPENAI_API_KEY": "forbidden-override", "EXPLICIT": "yes"})
    monkeypatch.setattr(gateway_module.os, "environ", inherited)
    gateway = gateway_module.Gateway()
    launched = []
    monkeypatch.setattr(gateway, "_launch", lambda **kwargs: launched.append(kwargs))
    gateway.start(cloud_backend="openai", env=explicit)
    assert launched[0]["mode_args"][3] == "http://127.0.0.1:8787/openai"
    assert gateway._env == {
        PROXY_ENV: inherited[PROXY_ENV],
        "KEEP": "yes",
        "EXPLICIT": "yes",
        "OPENAI_API_KEY": PROXY_PLACEHOLDER_KEY,
    }
