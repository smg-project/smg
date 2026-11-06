"""Shared fixtures for the Responses API test suite.

Hosts the ``image_generation`` test fixtures for the three engine lanes:

* ``gateway_with_mock_mcp_cloud`` — OpenAI cloud backend (R6.2).
* ``gateway_with_mock_mcp_grpc_sglang`` — local SGLang + gpt-oss-20b via
  harmony (R6.3).
* ``gateway_with_mock_mcp_grpc_vllm`` — local vLLM + Llama-3.1-8B-Instruct
  via regular (R6.4).

Each fixture wires the gateway's MCP client to an in-process
``MockMcpServer`` so the image-generation path is exercised deterministically
and without external-service dependencies. Future R6.x PRs can copy the
gRPC fixture pattern to add ``web_search`` / ``file_search`` /
``code_interpreter`` tests once the corresponding tool stubs in
``infra/mock_mcp_server.py`` are uncommented.

Design notes
------------
* The gateway CLI exposes ``--mcp-config-path`` (see
  ``bindings/python/src/smg/router_args.py``) which takes a path to a YAML
  config deserialized into ``crates/mcp/src/core/config.rs::McpConfig``.
  Setting ``builtin_type: image_generation`` on a server makes the gateway
  route every ``{"type": "image_generation"}`` tool request through it
  instead of passing the tool to the upstream model.
* We connect with ``protocol: streamable`` which matches the ``FastMCP`` app
  produced by ``streamable_http_app()``. ``sse`` would also work but would
  require an extra event-stream hop per call.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterator

import openai
import pytest
import yaml

# Fixture re-export: pytest discovers fixtures by walking the names defined
# in the conftest module. Importing ``mock_mcp_server`` here makes it
# visible to every test collected from this conftest without the
# PytestAssertRewriteWarning we'd get from registering ``infra.mock_mcp``
# as a ``pytest_plugins`` entry after it's already been imported via
# ``infra``. Ruff flags this as F401 (unused import) and then flags the
# fixture argument sites as F811 (redefinition); both are ignored
# intentionally — pytest needs the name to be bound at module scope.
from infra import MockMcpServer, launch_cloud_gateway
from infra.mock_mcp import mock_mcp_server as mock_mcp_server  # noqa: F401

logger = logging.getLogger(__name__)


# =============================================================================
# Mock MCP config
# =============================================================================


def _image_generation_mcp_config(mock_mcp_url: str) -> dict:
    """Build an MCP config that routes image_generation through the mock server.

    The shape mirrors ``crates/mcp/src/core/config.rs::McpConfig`` (YAML
    deserialization). ``builtin_type: image_generation`` is the knob that
    tells the gateway to route ``{"type": "image_generation"}`` tool requests
    through this server; ``builtin_tool_name`` names the tool on our server
    that implements the semantics (``image_generation`` for the mock).
    ``response_format: image_generation_call`` asks the transformer to shape
    the MCP tool result into a Responses API ``image_generation_call`` output
    item.
    """
    return {
        "servers": [
            {
                "name": "mock-image-gen",
                "protocol": "streamable",
                "url": mock_mcp_url,
                "builtin_type": "image_generation",
                "builtin_tool_name": "image_generation",
                "tools": {
                    "image_generation": {
                        "response_format": "image_generation_call",
                    }
                },
            }
        ]
    }


@pytest.fixture(scope="session")
def mock_mcp_config_file(mock_mcp_server: MockMcpServer) -> Iterator[str]:  # noqa: F811
    """Write the MCP config YAML to a tempfile and yield its path.

    Session-scoped so all tests in the module share a single config; the
    gateway re-reads the file at startup so sharing is safe.

    Note: ``# noqa: F811`` silences ruff's "redefinition of unused name"
    warning. The ``mock_mcp_server`` import above exists solely so pytest
    discovers the fixture in this module; using the name as a parameter
    here is exactly the pattern that triggers F811.
    """
    config = _image_generation_mcp_config(mock_mcp_server.url)
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="mock_mcp_image_gen_")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(config, f)
        logger.info("MCP config for mock image_generation at %s", path)
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _regular_mcp_config(mock_mcp_url: str) -> dict:
    """Build an MCP config that registers the mock server as a plain MCP
    server — no ``builtin_type``, no ``builtin_tool_name``, no
    ``response_format`` hint per tool. The tool literally named
    ``image_generation`` is therefore exposed as a generic MCP function
    tool, not as a hosted-tool surface.

    This is the configuration that produced the wire shape an external
    reviewer reported (``output`` field carrying a stringified MCP
    payload + ``arguments`` field carrying the dispatched args).
    Locking the behavior here so future "auto-detect from request
    tools" attempts have to break a real test on purpose.
    """
    return {
        "servers": [
            {
                "name": "mock-regular",
                "protocol": "streamable",
                "url": mock_mcp_url,
            }
        ]
    }


@pytest.fixture(scope="session")
def mock_mcp_config_file_regular(mock_mcp_server: MockMcpServer) -> Iterator[str]:  # noqa: F811
    """Counterpart to ``mock_mcp_config_file`` registering the same mock
    server as a plain MCP server (no ``builtin_type`` tag).
    """
    config = _regular_mcp_config(mock_mcp_server.url)
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="mock_mcp_regular_")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(config, f)
        logger.info("MCP config for plain (no-builtin) mock at %s", path)
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)


# =============================================================================
# Tool argument helpers
# =============================================================================


@pytest.fixture
def image_gen_tool_args() -> dict:
    """Canonical ``image_generation`` tool payload used across the suite.

    Centralised here so tests that want to override (e.g., size) can start
    from a shared baseline. The keys mirror
    ``crates/protocols/src/responses.rs::ImageGenerationTool``.
    """
    return {
        "type": "image_generation",
        "size": "1024x1024",
        "quality": "standard",
    }


# =============================================================================
# Cloud gateway fixture
# =============================================================================


@pytest.fixture(scope="class")
def gateway_with_mock_mcp_cloud(
    mock_mcp_server: MockMcpServer,  # noqa: F811
    mock_mcp_config_file: str,
) -> Iterator[tuple]:
    """Launch an OpenAI cloud gateway wired to the mock MCP server.

    Yields ``(gateway, client, mock_mcp_server, model)``. Skips when
    ``OPENAI_API_KEY`` is absent.
    """
    api_key_env = "OPENAI_API_KEY"
    if not os.environ.get(api_key_env):
        pytest.skip(f"{api_key_env} not set — image_generation cloud lane needs OpenAI")

    logger.info(
        "Launching OpenAI cloud gateway with mock MCP config (url=%s, config=%s)",
        mock_mcp_server.url,
        mock_mcp_config_file,
    )
    gateway = launch_cloud_gateway(
        "openai",
        history_backend="memory",
        extra_args=["--mcp-config-path", mock_mcp_config_file],
    )

    try:
        client = openai.OpenAI(
            base_url=f"{gateway.base_url}/v1",
            api_key=os.environ[api_key_env],
        )
    except Exception:
        gateway.shutdown()
        raise

    # ``gpt-5-nano`` is the model configured for the ``openai`` cloud
    # entry in ``e2e_test/infra/model_specs.py::THIRD_PARTY_MODELS``; keep
    # the literal pinned in the fixture so ``_ImageGenerationAssertions``
    # doesn't have to reach into infra.
    try:
        yield gateway, client, mock_mcp_server, "gpt-5-nano"
    finally:
        gateway.shutdown()


# Backwards-compat alias for test code that uses the single-engine name.
# The rewritten suite prefers the explicit ``_cloud`` / ``_grpc_*`` names
# but this preserves anyone who imported ``gateway_with_mock_mcp`` before.
gateway_with_mock_mcp = gateway_with_mock_mcp_cloud


@pytest.fixture(scope="class")
def gateway_with_mock_mcp_regular_cloud(
    mock_mcp_server: MockMcpServer,  # noqa: F811
    mock_mcp_config_file_regular: str,
) -> Iterator[tuple]:
    """Cloud gateway wired to the mock MCP server registered as a plain
    MCP server (no ``builtin_type`` tag). Yields the same
    ``(gateway, client, mock_mcp_server, model)`` tuple shape as
    ``gateway_with_mock_mcp_cloud`` so existing helpers compose.
    Skips when ``OPENAI_API_KEY`` is absent.
    """
    api_key_env = "OPENAI_API_KEY"
    if not os.environ.get(api_key_env):
        pytest.skip(f"{api_key_env} not set — regular-MCP cloud lane needs OpenAI")

    logger.info(
        "Launching OpenAI cloud gateway with regular (no-builtin) MCP config (url=%s, config=%s)",
        mock_mcp_server.url,
        mock_mcp_config_file_regular,
    )
    gateway = launch_cloud_gateway(
        "openai",
        history_backend="memory",
        extra_args=["--mcp-config-path", mock_mcp_config_file_regular],
    )

    try:
        client = openai.OpenAI(
            base_url=f"{gateway.base_url}/v1",
            api_key=os.environ[api_key_env],
        )
    except Exception:
        gateway.shutdown()
        raise

    try:
        yield gateway, client, mock_mcp_server, "gpt-5-nano"
    finally:
        gateway.shutdown()


# =============================================================================
# Local gRPC gateway fixtures (sglang + vllm)
# =============================================================================


def _start_local_grpc_gateway_with_mcp(
    *,
    engine: str,
    model_id: str,
    mcp_config_file: str,
):
    """Launch a local gRPC worker + gateway wired to the mock MCP server.

    Returns ``(gateway, client, workers, model_path)``. Workers come from
    the session-scoped pool (``infra.worker_pool``), so the caller MUST
    NOT stop them on teardown — only ``gateway.shutdown()`` is required.
    Worker startup errors fail the selected test.
    """
    from infra import ConnectionMode, Gateway
    from infra.model_specs import get_model_spec
    from infra.worker_pool import get_pool

    workers = get_pool().acquire(
        model_id=model_id,
        engine=engine,
        mode=ConnectionMode.GRPC,
        count=1,
    )

    # Worker stays in the pool on any downstream failure; the gateway is
    # ours so we shut it down on any error from ``start()`` or client init.
    worker = workers[0]
    model_path = get_model_spec(model_id)["model"]

    gateway = Gateway()
    try:
        gateway.start(
            worker_urls=[worker.base_url],
            model_path=model_path,
            extra_args=[
                "--mcp-config-path",
                mcp_config_file,
                "--history-backend",
                "memory",
            ],
        )
        client = openai.OpenAI(base_url=f"{gateway.base_url}/v1", api_key="not-used")
    except Exception:
        gateway.shutdown()
        raise

    return gateway, client, workers, model_path


@pytest.fixture(scope="class")
def gateway_with_mock_mcp_grpc_sglang(
    mock_mcp_server: MockMcpServer,  # noqa: F811
    mock_mcp_config_file: str,
) -> Iterator[tuple]:
    """Launch a local SGLang gRPC gateway wired to the mock MCP server.

    Uses the gpt-oss-20b model, which flows through the harmony path
    (R6.3).
    """
    gateway, client, _workers, model_path = _start_local_grpc_gateway_with_mcp(
        engine="sglang",
        model_id="openai/gpt-oss-20b",
        mcp_config_file=mock_mcp_config_file,
    )
    try:
        yield gateway, client, mock_mcp_server, model_path
    finally:
        gateway.shutdown()


@pytest.fixture(scope="class")
def gateway_with_mock_mcp_grpc_vllm(
    mock_mcp_server: MockMcpServer,  # noqa: F811
    mock_mcp_config_file: str,
) -> Iterator[tuple]:
    """Launch a local vLLM gRPC gateway wired to the mock MCP server.

    Uses Llama-3.1-8B-Instruct, which flows through the regular (non-
    harmony) path (R6.4).
    """
    gateway, client, _workers, model_path = _start_local_grpc_gateway_with_mcp(
        engine="vllm",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        mcp_config_file=mock_mcp_config_file,
    )
    try:
        yield gateway, client, mock_mcp_server, model_path
    finally:
        gateway.shutdown()
