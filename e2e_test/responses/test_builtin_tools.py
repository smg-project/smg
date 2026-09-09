"""Built-in tool tests for Response API.

Tests for built-in tool routing (web_search_preview, code_interpreter, file_search)
that are routed through MCP servers with response format transformation.

Prerequisites:
- Brave MCP Server running on port 8080 (set up in CI via pr-test-rust.yml)
- OPENAI_API_KEY environment variable set for cloud backend tests
"""

from __future__ import annotations

import logging
import os
import socket
import tempfile
import time

import openai
import pytest
import yaml
from infra import BRAVE_MCP_HOST, BRAVE_MCP_PORT, BRAVE_MCP_URL

logger = logging.getLogger(__name__)

WEB_SEARCH_PREVIEW_TOOL = {"type": "web_search_preview"}

BRAVE_MCP_TOOL = {
    "type": "mcp",
    "server_label": "brave",
    "server_url": BRAVE_MCP_URL,
    "require_approval": "never",
    "allowed_tools": ["brave_web_search"],
}

WEB_SEARCH_PROMPT = (
    "Search the web for the Rust programming language. "
    "Set count to 1 to get only one result, and give a one sentence summary."
)


def is_brave_server_available() -> bool:
    """Check if Brave MCP server is running on expected port."""
    try:
        with socket.create_connection((BRAVE_MCP_HOST, BRAVE_MCP_PORT), timeout=1):
            return True
    except (TimeoutError, OSError):
        return False


def create_mcp_config() -> dict:
    """Create MCP config that routes web_search_preview to Brave MCP server."""
    return {
        "servers": [
            {
                "name": "brave-builtin",
                "protocol": "streamable",
                "url": BRAVE_MCP_URL,
                "builtin_type": "web_search_preview",
                "builtin_tool_name": "brave_web_search",
                "tools": {
                    "brave_web_search": {
                        "response_format": "web_search_call",
                    }
                },
            }
        ]
    }


@pytest.fixture(scope="module")
def mcp_config_file():
    """Create temporary MCP config file for tests."""
    config = create_mcp_config()
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="mcp_builtin_")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(config, f)
        logger.info("Created MCP config at %s", path)
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)


@pytest.fixture(scope="module")
def require_brave_server():
    """Fail if Brave MCP server is not available."""
    if not is_brave_server_available():
        pytest.fail(
            f"Brave MCP server not available on port {BRAVE_MCP_PORT}. "
            "Ensure the brave-search service is running."
        )


@pytest.fixture(scope="class")
def gateway_with_mcp_config(require_brave_server, mcp_config_file):
    """Launch gateway with MCP config that routes web_search_preview to Brave."""
    from infra import launch_cloud_gateway

    api_key_env = "OPENAI_API_KEY"
    if not os.environ.get(api_key_env):
        pytest.skip(f"{api_key_env} not set")

    logger.info("Launching gateway with MCP config: %s", mcp_config_file)
    gateway = launch_cloud_gateway(
        "openai",
        history_backend="memory",
        extra_args=["--mcp-config-path", mcp_config_file],
    )

    client = openai.OpenAI(
        base_url=f"{gateway.base_url}/v1",
        api_key=os.environ.get(api_key_env),
    )

    yield gateway, client

    gateway.shutdown()


@pytest.fixture(scope="class")
def gateway_with_mcp_config_grpc(require_brave_server, mcp_config_file):
    """Launch gRPC gateway with MCP config that routes web_search_preview to Brave."""
    from infra import ConnectionMode, Gateway
    from infra.model_specs import get_model_spec
    from infra.worker import start_workers, stop_workers

    engine = os.environ.get("E2E_ENGINE", "sglang")
    logger.info("Launching gRPC gateway with MCP config: %s", mcp_config_file)

    # Start a gRPC worker
    try:
        workers = start_workers("openai/gpt-oss-20b", engine, mode=ConnectionMode.GRPC, count=1)
    except Exception as e:
        pytest.skip(f"gRPC worker not available: {e}")

    worker = workers[0]
    model_path = get_model_spec("openai/gpt-oss-20b")["model"]

    gateway = Gateway()
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

    client = openai.OpenAI(
        base_url=f"{gateway.base_url}/v1",
        api_key="not-used",
    )

    yield gateway, client, model_path

    gateway.shutdown()
    stop_workers(workers)


# Note: These tests require manual gateway configuration with MCP config.
# In CI, the gateway is started with --mcp-config-path pointing to a config
# that has builtin_type: web_search_preview configured.
#
# The marker approach doesn't work well for dynamic config paths, so these
# tests serve as documentation and can be run manually with proper setup.


@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestBuiltinVsMcpComparison:
    """Compare built-in tool behavior vs direct MCP tool behavior.

    These tests verify baseline MCP behavior without requiring builtin routing config.
    """

    def test_mcp_tool_produces_mcp_call(self, model, api_client):
        """Verify that direct MCP tool produces mcp_call output."""

        time.sleep(2)

        resp = api_client.responses.create(
            model=model,
            input=(
                "Search the web for Python programming language. "
                "Set count to 1 to get only one result and give a one sentence summary."
            ),
            tools=[BRAVE_MCP_TOOL],
            stream=False,
        )

        assert resp.error is None, f"Response error: {resp.error}"
        assert resp.output is not None

        output_types = [item.type for item in resp.output]
        logger.info("MCP tool output types: %s", output_types)

        # Direct MCP should produce mcp_call
        assert "mcp_call" in output_types, (
            f"Direct MCP tool should produce mcp_call, got: {output_types}"
        )


@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestBuiltinToolsCloudBackend:
    """Built-in tool tests against cloud backend (OpenAI).

    These tests verify that built-in tool types are accepted by the API.
    Full routing tests require MCP config with builtin_type configured.
    """

    def test_web_search_preview_accepted(self, model, api_client):
        """Test that web_search_preview tool type is accepted."""

        time.sleep(2)

        resp = api_client.responses.create(
            model=model,
            input=WEB_SEARCH_PROMPT,
            tools=[WEB_SEARCH_PREVIEW_TOOL],
            stream=False,
        )

        assert resp.id is not None
        assert resp.status in ("completed", "incomplete")

    def test_mixed_builtin_and_function_tools(self, model, api_client):
        """Test mixing web_search_preview with function tools."""

        time.sleep(2)

        get_weather_function = {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
        }

        resp = api_client.responses.create(
            model=model,
            input="What's the weather in Seattle?",
            tools=[WEB_SEARCH_PREVIEW_TOOL, get_weather_function],
            stream=False,
        )

        assert resp.error is None
        assert resp.id is not None


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("openai/gpt-oss-20b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestBuiltinToolsLocalBackend:
    """Built-in tool tests against local gRPC backend.

    These tests verify built-in tool handling with local models.
    """

    def test_web_search_preview_accepted(self, model, api_client):
        """Test that web_search_preview tool type is accepted by local backend."""

        time.sleep(1)

        resp = api_client.responses.create(
            model=model,
            input=WEB_SEARCH_PROMPT,
            tools=[WEB_SEARCH_PREVIEW_TOOL],
            stream=False,
        )

        assert resp.id is not None

    def test_mixed_builtin_and_function_tools(self, model, api_client):
        """Test mixing web_search_preview with function tools on local backend."""

        time.sleep(1)

        get_weather_function = {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
        }

        resp = api_client.responses.create(
            model=model,
            input="What's the weather in Seattle?",
            tools=[WEB_SEARCH_PREVIEW_TOOL, get_weather_function],
            stream=False,
        )

        assert resp.id is not None


# =============================================================================
# Full Integration Tests (require Brave MCP server + proper gateway config)
# =============================================================================
# These tests run in CI where:
# 1. Brave MCP server runs on port 8080
# 2. Gateway is configured with builtin_type: web_search_preview via fixture
#
# To run locally:
# 1. Start Brave MCP: docker run -d -p 8080:8080 -e BRAVE_API_KEY=<key> shoofio/brave-search-mcp-sse:1.0.10
# 2. Set OPENAI_API_KEY environment variable
# 3. Run: pytest e2e_test/responses/test_builtin_tools.py::TestBuiltinToolRouting -v


@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
class TestBuiltinToolRouting:
    """Full integration tests for built-in tool routing.

    These tests verify the complete flow:
    1. Client sends {type: "web_search_preview"}
    2. Gateway routes to configured MCP server (Brave)
    3. Response contains web_search_call (not mcp_call)

    Requires:
    - Brave MCP server on port 8080
    - OPENAI_API_KEY environment variable
    """

    def test_web_search_preview_produces_web_search_call(self, gateway_with_mcp_config):
        """Test that web_search_preview produces web_search_call output.

        This is the core test for Phase 2 built-in tool support.
        Verifies:
        1. Send request with {type: "web_search_preview"}
        2. Response has web_search_call in output types
        3. Response does NOT have mcp_call
        """
        gateway, client = gateway_with_mcp_config

        time.sleep(2)

        resp = client.responses.create(
            model="gpt-4o-mini",
            input=WEB_SEARCH_PROMPT,
            tools=[WEB_SEARCH_PREVIEW_TOOL],
            stream=False,
        )

        assert resp.error is None, f"Response error: {resp.error}"
        assert resp.output is not None

        output_types = [item.type for item in resp.output]
        logger.info("Built-in tool output types: %s", output_types)

        # Built-in tool should produce web_search_call, NOT mcp_call
        assert "web_search_call" in output_types, (
            f"Built-in web_search_preview should produce web_search_call, got: {output_types}"
        )
        assert "mcp_call" not in output_types, (
            f"Built-in tool should NOT produce mcp_call, got: {output_types}"
        )

        # Built-in tool should NOT produce mcp_list_tools output
        assert "mcp_list_tools" not in output_types, (
            f"Built-in tool should NOT produce mcp_list_tools, got: {output_types}"
        )

        web_search_calls = [item for item in resp.output if item.type == "web_search_call"]
        assert web_search_calls, "Expected at least one web_search_call item in response output"

        ws_call = web_search_calls[0]
        assert ws_call.status == "completed"
        assert ws_call.action is not None, "Expected web_search_call.action to be populated"
        assert ws_call.action.query, "Expected web_search_call.action.query to be populated"
        assert "rust" in ws_call.action.query.lower(), (
            f"Expected action.query to reference the Rust prompt, got: {ws_call.action.query}"
        )

    def test_response_tools_shows_original_type(self, gateway_with_mcp_config):
        """Test that response tools field shows web_search_preview, not mcp."""
        gateway, client = gateway_with_mcp_config

        time.sleep(2)

        resp = client.responses.create(
            model="gpt-4o-mini",
            input=WEB_SEARCH_PROMPT,
            tools=[WEB_SEARCH_PREVIEW_TOOL],
            stream=False,
        )

        assert resp.error is None, f"Response error: {resp.error}"

        # The tools field in response should mirror the original request
        # It should show web_search_preview, not mcp
        if resp.tools:
            tool_types = [t.type for t in resp.tools]
            logger.info("Response tools types: %s", tool_types)
            assert "mcp" not in tool_types, (
                f"Response tools should not show 'mcp' type, got: {tool_types}"
            )


@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
@pytest.mark.external
@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestMcpWebSearchStreamingEvents:
    """Test MCP tool SSE streaming events (baseline behavior).

    These tests verify the baseline MCP streaming behavior using direct
    MCP tool configuration (not builtin routing).
    """

    @pytest.fixture(autouse=True)
    def check_brave_server(self, require_brave_server):
        """Ensure Brave server is available for these tests."""
        pass

    def test_mcp_web_search_streaming_events(self, model, api_client):
        """Test that MCP web search produces proper streaming events.

        This verifies the baseline MCP streaming behavior that built-in
        tools should eventually match (with different event types).
        """

        time.sleep(2)

        resp = api_client.responses.create(
            model=model,
            input=(
                "Search the web for Python programming language. "
                "Set count to 1 to get only one result and give a one sentence summary."
            ),
            tools=[BRAVE_MCP_TOOL],
            stream=True,
        )

        events = list(resp)
        assert len(events) > 0

        event_types = [event.type for event in events]
        logger.info("MCP streaming event types: %s", sorted(set(event_types)))

        # Core streaming events
        assert "response.created" in event_types, "Should have response.created event"
        assert "response.completed" in event_types, "Should have response.completed event"

        # MCP-specific events
        assert "response.mcp_list_tools.in_progress" in event_types, (
            "Should have mcp_list_tools.in_progress event"
        )
        assert "response.mcp_list_tools.completed" in event_types, (
            "Should have mcp_list_tools.completed event"
        )
        assert "response.mcp_call.in_progress" in event_types, (
            "Should have mcp_call.in_progress event"
        )
        assert "response.mcp_call.completed" in event_types, "Should have mcp_call.completed event"

        # Verify final response
        completed_events = [e for e in events if e.type == "response.completed"]
        assert len(completed_events) == 1

        final_response = completed_events[0].response
        assert final_response.status == "completed"
        assert final_response.output is not None

        # Verify mcp_call in final output
        final_output_types = [item.type for item in final_response.output]
        assert "mcp_call" in final_output_types, (
            f"MCP tool should produce mcp_call output, got: {final_output_types}"
        )

        # Verify mcp_call item structure
        mcp_calls = [item for item in final_response.output if item.type == "mcp_call"]
        for mcp_call in mcp_calls:
            assert mcp_call.status == "completed"
            assert mcp_call.server_label == "brave"
            assert mcp_call.name is not None
            assert mcp_call.output is not None


@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
class TestWebSearchStreamingEvents:
    """Test web_search_call SSE streaming events with builtin routing.

    These tests verify the streaming event sequence for web search when
    using builtin tool routing (web_search_preview -> MCP server):
    - response.web_search_call.in_progress
    - response.web_search_call.searching
    - response.web_search_call.completed

    Reference: OpenAI API spec for Responses streaming events.
    """

    def test_web_search_preview_streaming_events(self, gateway_with_mcp_config):
        """Test that web_search_preview produces web_search_call streaming events.

        Verifies the SSE event sequence:
        - response.web_search_call.in_progress (search started)
        - response.web_search_call.searching (actively searching)
        - response.web_search_call.completed (search finished)
        """
        gateway, client = gateway_with_mcp_config

        time.sleep(2)

        resp = client.responses.create(
            model="gpt-4o-mini",
            input=WEB_SEARCH_PROMPT,
            tools=[WEB_SEARCH_PREVIEW_TOOL],
            stream=True,
        )

        events = list(resp)
        event_types = [event.type for event in events]
        logger.info("web_search_preview streaming event types: %s", sorted(set(event_types)))

        # Core streaming events
        assert "response.created" in event_types
        assert "response.completed" in event_types

        # web_search_call specific events (per OpenAI API spec)
        assert "response.web_search_call.in_progress" in event_types, (
            "Should have web_search_call.in_progress event"
        )
        assert "response.web_search_call.searching" in event_types, (
            "Should have web_search_call.searching event"
        )
        assert "response.web_search_call.completed" in event_types, (
            "Should have web_search_call.completed event"
        )

        # Verify no MCP events (these should be transformed)
        assert "response.mcp_call.in_progress" not in event_types, (
            "Built-in tool should NOT produce mcp_call events"
        )
        assert "response.mcp_call.completed" not in event_types, (
            "Built-in tool should NOT produce mcp_call events"
        )

        # Verify no mcp_list_tools events (builtin servers should be hidden)
        assert "response.mcp_list_tools.in_progress" not in event_types, (
            "Built-in tool should NOT produce mcp_list_tools events"
        )
        assert "response.mcp_list_tools.completed" not in event_types, (
            "Built-in tool should NOT produce mcp_list_tools events"
        )

        # Verify final response
        completed_events = [e for e in events if e.type == "response.completed"]
        final_response = completed_events[0].response
        assert final_response.status == "completed"

        # Verify web_search_call in final output (not mcp_call)
        final_output_types = [item.type for item in final_response.output]
        assert "web_search_call" in final_output_types, (
            f"Built-in tool should produce web_search_call, got: {final_output_types}"
        )
        assert "mcp_call" not in final_output_types, (
            "Built-in tool should NOT produce mcp_call output"
        )

        # Verify no mcp_list_tools in final output
        assert "mcp_list_tools" not in final_output_types, (
            "Built-in tool should NOT produce mcp_list_tools output"
        )

        # Verify web_search_call item structure
        web_search_calls = [
            item for item in final_response.output if item.type == "web_search_call"
        ]
        for ws_call in web_search_calls:
            assert ws_call.status == "completed"
            assert ws_call.id is not None

    def test_web_search_streaming_event_order(self, gateway_with_mcp_config):
        """Test that web_search streaming events occur in correct order.

        Event order should be:
        1. response.created
        2. response.output_item.added (for web_search_call item)
        3. response.web_search_call.in_progress
        4. response.web_search_call.searching (may occur multiple times)
        5. response.web_search_call.completed
        6. response.output_item.done
        7. ... (message events)
        8. response.completed
        """
        gateway, client = gateway_with_mcp_config

        time.sleep(2)

        resp = client.responses.create(
            model="gpt-4o-mini",
            input=WEB_SEARCH_PROMPT,
            tools=[WEB_SEARCH_PREVIEW_TOOL],
            stream=True,
        )

        events = list(resp)
        event_types = [event.type for event in events]

        # Find positions of key events
        def find_first(event_type: str) -> int:
            try:
                return event_types.index(event_type)
            except ValueError:
                return -1

        created_pos = find_first("response.created")
        in_progress_pos = find_first("response.web_search_call.in_progress")
        searching_pos = find_first("response.web_search_call.searching")
        completed_pos = find_first("response.web_search_call.completed")
        response_completed_pos = find_first("response.completed")

        # Verify event ordering
        assert created_pos == 0, "response.created should be first"
        assert in_progress_pos > created_pos, "in_progress should be after created"
        assert searching_pos > in_progress_pos, "searching should be after in_progress"
        assert completed_pos > searching_pos, "completed should be after searching"
        assert response_completed_pos > completed_pos, (
            "response.completed should be after web_search_call.completed"
        )


# =============================================================================
# gRPC Backend Integration Tests (require Brave MCP server + gRPC worker)
# =============================================================================
# These tests run the same builtin tool routing tests against gRPC backend
# to verify the harmony router produces the same events as the regular router.


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("openai/gpt-oss-20b")
class TestBuiltinToolRoutingGrpc:
    """Full integration tests for built-in tool routing on gRPC backend.

    These tests verify the complete flow on gRPC/harmony router:
    1. Client sends {type: "web_search_preview"}
    2. Gateway routes to configured MCP server (Brave)
    3. Response contains web_search_call (not mcp_call)

    Requires:
    - Brave MCP server on port 8080
    - gRPC worker available via start_workers
    """

    def test_web_search_preview_produces_web_search_call(self, gateway_with_mcp_config_grpc):
        """Test that web_search_preview produces web_search_call output on gRPC."""
        gateway, client, model_path = gateway_with_mcp_config_grpc

        time.sleep(2)

        resp = client.responses.create(
            model=model_path,
            input=WEB_SEARCH_PROMPT,
            tools=[WEB_SEARCH_PREVIEW_TOOL],
            stream=False,
        )

        assert resp.error is None, f"Response error: {resp.error}"
        assert resp.output is not None

        output_types = [item.type for item in resp.output]
        logger.info("Built-in tool output types (gRPC): %s", output_types)

        # Built-in tool should produce web_search_call, NOT mcp_call
        assert "web_search_call" in output_types, (
            f"Built-in web_search_preview should produce web_search_call, got: {output_types}"
        )
        assert "mcp_call" not in output_types, (
            f"Built-in tool should NOT produce mcp_call, got: {output_types}"
        )

        # Built-in tool should NOT produce mcp_list_tools output
        assert "mcp_list_tools" not in output_types, (
            f"Built-in tool should NOT produce mcp_list_tools, got: {output_types}"
        )

    def test_response_tools_shows_original_type(self, gateway_with_mcp_config_grpc):
        """Test that response tools field shows web_search_preview, not mcp (gRPC)."""
        gateway, client, model_path = gateway_with_mcp_config_grpc

        time.sleep(2)

        resp = client.responses.create(
            model=model_path,
            input=WEB_SEARCH_PROMPT,
            tools=[WEB_SEARCH_PREVIEW_TOOL],
            stream=False,
        )

        assert resp.error is None, f"Response error: {resp.error}"

        # The tools field in response should mirror the original request
        if resp.tools:
            tool_types = [t.type for t in resp.tools]
            logger.info("Response tools types (gRPC): %s", tool_types)
            assert "mcp" not in tool_types, (
                f"Response tools should not show 'mcp' type, got: {tool_types}"
            )


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("openai/gpt-oss-20b")
class TestWebSearchStreamingEventsGrpc:
    """Test web_search_call SSE streaming events with builtin routing on gRPC.

    These tests verify the streaming event sequence for web search when
    using builtin tool routing on gRPC backend (harmony router).
    Should produce the same events as the regular router:
    - response.web_search_call.in_progress
    - response.web_search_call.searching
    - response.web_search_call.completed
    """

    def test_web_search_preview_streaming_events(self, gateway_with_mcp_config_grpc):
        """Test that web_search_preview produces web_search_call streaming events (gRPC)."""
        gateway, client, model_path = gateway_with_mcp_config_grpc

        time.sleep(2)

        resp = client.responses.create(
            model=model_path,
            input=WEB_SEARCH_PROMPT,
            tools=[WEB_SEARCH_PREVIEW_TOOL],
            stream=True,
        )

        events = list(resp)
        event_types = [event.type for event in events]
        logger.info("web_search_preview streaming event types (gRPC): %s", sorted(set(event_types)))

        # Core streaming events
        assert "response.created" in event_types
        assert "response.completed" in event_types

        # web_search_call specific events (per OpenAI API spec)
        assert "response.web_search_call.in_progress" in event_types, (
            "Should have web_search_call.in_progress event"
        )
        assert "response.web_search_call.searching" in event_types, (
            "Should have web_search_call.searching event"
        )
        assert "response.web_search_call.completed" in event_types, (
            "Should have web_search_call.completed event"
        )

        # Verify no MCP events (these should be transformed)
        assert "response.mcp_call.in_progress" not in event_types, (
            "Built-in tool should NOT produce mcp_call events"
        )
        assert "response.mcp_call.completed" not in event_types, (
            "Built-in tool should NOT produce mcp_call events"
        )

        # Verify no mcp_list_tools events (builtin servers should be hidden)
        assert "response.mcp_list_tools.in_progress" not in event_types, (
            "Built-in tool should NOT produce mcp_list_tools events"
        )
        assert "response.mcp_list_tools.completed" not in event_types, (
            "Built-in tool should NOT produce mcp_list_tools events"
        )

        # Verify final response
        completed_events = [e for e in events if e.type == "response.completed"]
        final_response = completed_events[0].response
        assert final_response.status == "completed"

        # Verify web_search_call in final output (not mcp_call)
        final_output_types = [item.type for item in final_response.output]
        assert "web_search_call" in final_output_types, (
            f"Built-in tool should produce web_search_call, got: {final_output_types}"
        )
        assert "mcp_call" not in final_output_types, (
            "Built-in tool should NOT produce mcp_call output"
        )

        # Verify no mcp_list_tools in final output
        assert "mcp_list_tools" not in final_output_types, (
            "Built-in tool should NOT produce mcp_list_tools output"
        )

        # Verify web_search_call item structure
        web_search_calls = [
            item for item in final_response.output if item.type == "web_search_call"
        ]
        for ws_call in web_search_calls:
            assert ws_call.status == "completed"
            assert ws_call.id is not None

    def test_web_search_streaming_event_order(self, gateway_with_mcp_config_grpc):
        """Test that web_search streaming events occur in correct order (gRPC)."""
        gateway, client, model_path = gateway_with_mcp_config_grpc

        time.sleep(2)

        resp = client.responses.create(
            model=model_path,
            input=WEB_SEARCH_PROMPT,
            tools=[WEB_SEARCH_PREVIEW_TOOL],
            stream=True,
        )

        events = list(resp)
        event_types = [event.type for event in events]

        # Find positions of key events
        def find_first(event_type: str) -> int:
            try:
                return event_types.index(event_type)
            except ValueError:
                return -1

        created_pos = find_first("response.created")
        in_progress_pos = find_first("response.web_search_call.in_progress")
        searching_pos = find_first("response.web_search_call.searching")
        completed_pos = find_first("response.web_search_call.completed")
        response_completed_pos = find_first("response.completed")

        # Verify event ordering
        assert created_pos == 0, "response.created should be first"
        assert in_progress_pos > created_pos, "in_progress should be after created"
        assert searching_pos > in_progress_pos, "searching should be after in_progress"
        assert completed_pos > searching_pos, "completed should be after searching"
        assert response_completed_pos > completed_pos, (
            "response.completed should be after web_search_call.completed"
        )
