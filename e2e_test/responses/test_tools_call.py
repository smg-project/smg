"""Tool calling tests for Response API.

Tests for function calling functionality, tool choices and MCP calling
functionality across different backends.

Source: Migrated from e2e_response_api/features/test_tools_call.py
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from infra import BRAVE_MCP_URL

logger = logging.getLogger(__name__)


# =============================================================================
# Shared Tool Definitions
# =============================================================================


SYSTEM_DIAGNOSTICS_FUNCTION = {
    "type": "function",
    "name": "get_system_diagnostics",
    "description": "Retrieve real-time diagnostics for a spacecraft system.",
    "parameters": {
        "type": "object",
        "properties": {
            "system_name": {
                "type": "string",
                "description": "Name of the spacecraft system to query. "
                "Example: 'Astra-7 Core Reactor'.",
            }
        },
        "required": ["system_name"],
    },
}

GET_WEATHER_FUNCTION = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city name, e.g., San Francisco",
            }
        },
        "required": ["location"],
    },
}

CALCULATE_FUNCTION = {
    "type": "function",
    "name": "calculate",
    "description": "Perform a mathematical calculation",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate",
            }
        },
        "required": ["expression"],
    },
}

SEARCH_WEB_FUNCTION = {
    "type": "function",
    "name": "search_web",
    "description": "Search the web for information",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}

LOCAL_SEARCH_FUNCTION = {
    "type": "function",
    "name": "local_search",
    "description": "Search local database",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}

GET_HOROSCOPE_FUNCTION = {
    "type": "function",
    "name": "get_horoscope",
    "description": "Get today's horoscope for an astrological sign.",
    "parameters": {
        "type": "object",
        "properties": {
            "sign": {
                "type": "string",
                "description": "An astrological sign like Taurus or Aquarius",
            },
        },
        "required": ["sign"],
    },
}

BRAVE_MCP_TOOL = {
    "type": "mcp",
    "server_label": "brave",
    "server_description": "A Tool to do web search",
    "server_url": BRAVE_MCP_URL,
    "require_approval": "never",
    "allowed_tools": ["brave_web_search"],
}

DEEPWIKI_MCP_TOOL = {
    "type": "mcp",
    "server_label": "deepwiki",
    "server_url": "https://mcp.deepwiki.com/mcp",
    "require_approval": "never",
}

BRAVE_MCP_TOOL_REQUIRE_APPROVAL_ALWAYS = {
    **BRAVE_MCP_TOOL,
    "require_approval": "always",
}

MCP_TEST_PROMPT = (
    "Search the web for 'Python programming language'. Set count to 1 to "
    "get only one result and return one sentence response."
)


# =============================================================================
# Cloud Backend Tests (OpenAI) - Basic Function Calling
# =============================================================================


def assert_all_mcp_call_ids_prefixed(output):
    """Assert every mcp_call id uses the external mcp_ prefix."""
    mcp_calls = [item for item in output if item.type == "mcp_call"]
    assert len(mcp_calls) > 0, "Expected at least one mcp_call item"
    for mcp_call in mcp_calls:
        assert mcp_call.id is not None
        assert mcp_call.id.startswith("mcp_"), (
            f"mcp_call.id should use mcp_ prefix, got {mcp_call.id!r}"
        )


def assert_streaming_mcp_call_ids_match(events):
    """Assert one streaming MCP call keeps the same external id across all events."""
    added_ids = [
        event.item.id
        for event in events
        if event.type == "response.output_item.added"
        and event.item is not None
        and event.item.type == "mcp_call"
    ]
    done_ids = [
        event.item.id
        for event in events
        if event.type == "response.output_item.done"
        and event.item is not None
        and event.item.type == "mcp_call"
    ]
    arg_done_ids = [
        event.item_id for event in events if event.type == "response.mcp_call_arguments.done"
    ]
    completed_ids = [
        event.item_id for event in events if event.type == "response.mcp_call.completed"
    ]

    assert added_ids, "Expected at least one streaming mcp_call added event"
    assert done_ids, "Expected at least one streaming mcp_call done event"
    assert arg_done_ids, "Expected at least one streaming mcp_call_arguments.done event"
    assert completed_ids, "Expected at least one streaming mcp_call.completed event"

    expected_ids = set(added_ids)
    assert all(item_id.startswith("mcp_") for item_id in expected_ids)
    assert set(done_ids) == expected_ids
    assert set(arg_done_ids) == expected_ids
    assert set(completed_ids) == expected_ids

    completed_events = [e for e in events if e.type == "response.completed"]
    assert len(completed_events) == 1
    final_mcp_ids = [
        item.id for item in completed_events[0].response.output if item.type == "mcp_call"
    ]
    assert set(final_mcp_ids) == expected_ids


def mcp_list_tools_labels(output):
    return [item.server_label for item in output if item.type == "mcp_list_tools"]


def streaming_added_mcp_list_tools_labels(events):
    return [
        event.item.server_label
        for event in events
        if event.type == "response.output_item.added"
        and event.item is not None
        and event.item.type == "mcp_list_tools"
    ]


def assert_mcp_approval_interruption_non_streaming(resp):
    assert resp.error is None
    assert resp.id is not None
    assert resp.status == "completed"
    assert resp.output is not None

    output_types = [item.type for item in resp.output]
    assert "mcp_call" in output_types
    assert "mcp_approval_request" in output_types

    mcp_calls = [item for item in resp.output if item.type == "mcp_call"]
    deepwiki_calls = [item for item in mcp_calls if item.server_label == "deepwiki"]
    assert len(deepwiki_calls) > 0, "Expected at least one deepwiki mcp_call item"
    for mcp_call in deepwiki_calls:
        assert mcp_call.id is not None
        assert mcp_call.id.startswith("mcp_")
        assert mcp_call.name is not None
        assert mcp_call.arguments is not None
        assert mcp_call.output is not None

    approval_requests = [item for item in resp.output if item.type == "mcp_approval_request"]
    brave_approval_requests = [item for item in approval_requests if item.server_label == "brave"]
    assert len(brave_approval_requests) > 0, "Expected at least one brave mcp_approval_request"
    for approval_request in brave_approval_requests:
        assert approval_request.id is not None
        assert approval_request.id.startswith("mcpr_")
        assert approval_request.name is not None
        assert approval_request.arguments is not None

    assert all(item.server_label != "brave" for item in mcp_calls), (
        "Expected approval-required brave tools to stop at mcp_approval_request "
        "instead of emitting mcp_call"
    )


def assert_previous_response_id_mcp_binding_behavior_non_streaming(model, api_client):
    time.sleep(2)

    resp1 = api_client.responses.create(
        model=model,
        input=MCP_TEST_PROMPT,
        tools=[BRAVE_MCP_TOOL],
        stream=False,
        reasoning={"effort": "low"},
    )
    assert resp1.error is None
    assert resp1.status == "completed"
    assert mcp_list_tools_labels(resp1.output) == ["brave"]

    resp2 = api_client.responses.create(
        model=model,
        input=(
            "Search the web for 'Rust programming language'. Set count to 1 and return one "
            "sentence response."
        ),
        previous_response_id=resp1.id,
        tools=[BRAVE_MCP_TOOL],
        stream=False,
        reasoning={"effort": "low"},
    )
    assert resp2.error is None
    assert resp2.status == "completed"
    assert mcp_list_tools_labels(resp2.output) == []
    assert any(item.type == "mcp_call" for item in resp2.output)

    resp3 = api_client.responses.create(
        model=model,
        input=(
            "Use deepwiki to tell me which transport protocols the 2025-03-26 MCP spec "
            "supports, and also use brave_web_search to search the web for 'Rust programming "
            "language'. Return exactly two bullet points."
        ),
        previous_response_id=resp2.id,
        tools=[BRAVE_MCP_TOOL, DEEPWIKI_MCP_TOOL],
        stream=False,
        reasoning={"effort": "low"},
    )
    assert resp3.error is None
    assert resp3.status == "completed"
    # OpenAI may serve the newly added server's tool list from cache (other
    # tests in this suite attach deepwiki minutes earlier), in which case no
    # mcp_list_tools item is emitted. Pin only the stable contract: existing
    # bindings are never relisted, and the request still completes with calls.
    assert mcp_list_tools_labels(resp3.output) in ([], ["deepwiki"])
    assert any(item.type == "mcp_call" for item in resp3.output)


def assert_previous_response_id_mcp_binding_behavior_streaming(model, api_client):
    time.sleep(2)

    events1 = list(
        api_client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[BRAVE_MCP_TOOL],
            stream=True,
            reasoning={"effort": "low"},
        )
    )
    assert streaming_added_mcp_list_tools_labels(events1) == ["brave"]

    events2 = list(
        api_client.responses.create(
            model=model,
            input=(
                "Search the web for 'Rust programming language'. Set count to 1 and return one "
                "sentence response."
            ),
            previous_response_id=[e for e in events1 if e.type == "response.completed"][
                0
            ].response.id,
            tools=[BRAVE_MCP_TOOL],
            stream=True,
            reasoning={"effort": "low"},
        )
    )
    assert streaming_added_mcp_list_tools_labels(events2) == []
    assert any(event.type == "response.mcp_call.completed" for event in events2)

    events3 = list(
        api_client.responses.create(
            model=model,
            input=(
                "Use deepwiki to tell me which transport protocols the 2025-03-26 MCP spec "
                "supports, and also use brave_web_search to search the web for 'Rust programming "
                "language'. Return exactly two bullet points."
            ),
            previous_response_id=[e for e in events2 if e.type == "response.completed"][
                0
            ].response.id,
            tools=[BRAVE_MCP_TOOL, DEEPWIKI_MCP_TOOL],
            stream=True,
            reasoning={"effort": "low"},
        )
    )
    # Same cache caveat as the non-streaming variant: the added server's
    # listing may be cache-suppressed, but a listing for an already-bound
    # server must never reappear, and the event stream must stay consistent
    # with whichever happened.
    labels3 = streaming_added_mcp_list_tools_labels(events3)
    assert labels3 in ([], ["deepwiki"])
    expected_listings = len(labels3)
    assert [e.type for e in events3].count(
        "response.mcp_list_tools.in_progress"
    ) == expected_listings
    assert [e.type for e in events3].count("response.mcp_list_tools.completed") == expected_listings
    completed_events = [e for e in events3 if e.type == "response.completed"]
    assert len(completed_events) == 1
    assert mcp_list_tools_labels(completed_events[0].response.output) == labels3


@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestToolCallingCloud:
    """Tool calling tests against cloud APIs."""

    def test_basic_function_call(self, model, api_client):
        """Test basic function calling workflow."""
        tools = [GET_HOROSCOPE_FUNCTION]
        system_prompt = (
            "You are a helpful assistant that can call functions. "
            "When a user asks for horoscope information, call the function. "
            "IMPORTANT: Don't reply directly to the user, only call the function. "
        )

        input_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is my horoscope? I am an Aquarius."},
        ]

        resp = api_client.responses.create(model=model, input=input_list, tools=tools)

        assert resp.error is None
        assert resp.id is not None
        assert resp.status == "completed"
        assert resp.output is not None

        output = resp.output
        assert isinstance(output, list)
        assert len(output) > 0

        # Check for function_call in output
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0, "Response should contain at least one function_call"

        # Verify function_call structure
        function_call = function_calls[0]
        assert function_call.call_id is not None
        assert function_call.name == "get_horoscope"
        assert function_call.arguments is not None

        # Parse arguments
        args = json.loads(function_call.arguments)
        assert "sign" in args
        assert args["sign"].lower() == "aquarius"

        # Provide function call output
        input_list.append(function_call)
        horoscope = f"{args['sign']}: Next Tuesday you will befriend a baby otter."
        input_list.append(
            {
                "type": "function_call_output",
                "call_id": function_call.call_id,
                "output": json.dumps({"horoscope": horoscope}),
            }
        )

        # Second request with function output
        resp2 = api_client.responses.create(
            model=model,
            input=input_list,
            instructions="Respond only with a horoscope generated by a tool.",
            tools=tools,
        )
        assert resp2.error is None
        assert resp2.status == "completed"

        output2 = resp2.output
        assert len(output2) > 0

        messages = [item for item in output2 if item.type == "message"]
        assert len(messages) > 0

        message = messages[0]
        assert message.content is not None
        text_parts = [part.text for part in message.content if part.type == "output_text"]
        full_text = " ".join(text_parts).lower()
        # "baby otter" is the sentinel injected via function_call_output above;
        # its presence proves the tool result was actually consumed. ("aquarius"
        # would be satisfiable from the prompt alone.)
        assert "baby otter" in full_text, (
            f"Tool-output sentinel 'baby otter' missing from final response: {full_text!r}"
        )

    def test_mcp_basic_tool_call(self, model, api_client):
        """Test basic MCP tool call (non-streaming)."""

        time.sleep(2)  # Avoid rate limiting

        resp = api_client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[BRAVE_MCP_TOOL],
            stream=False,
            reasoning={"effort": "low"},
        )

        assert resp.error is None
        assert resp.id is not None
        assert resp.status == "completed"
        assert resp.model is not None
        assert resp.output is not None
        assert len(resp.output_text) > 0

        output_types = [item.type for item in resp.output]
        assert "mcp_list_tools" in output_types

        mcp_calls = [item for item in resp.output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0
        assert_all_mcp_call_ids_prefixed(resp.output)

        for mcp_call in mcp_calls:
            assert mcp_call.error is None
            assert mcp_call.status == "completed"
            assert mcp_call.server_label == "brave"
            assert mcp_call.name is not None
            assert mcp_call.arguments is not None
            assert mcp_call.output is not None

        # Strict validation for cloud backends
        messages = [item for item in resp.output if item.type == "message"]
        assert len(messages) > 0, "Response should contain at least one message"
        for msg in messages:
            assert msg.content is not None
            assert isinstance(msg.content, list)
            for content_item in msg.content:
                if content_item.type == "output_text":
                    assert content_item.text is not None
                    assert isinstance(content_item.text, str)
                    assert len(content_item.text) > 0

    def test_mcp_basic_tool_call_streaming(self, model, api_client):
        """Test basic MCP tool call (streaming)."""

        time.sleep(2)  # Avoid rate limiting

        resp = api_client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[BRAVE_MCP_TOOL],
            stream=True,
            reasoning={"effort": "low"},
        )

        events = list(resp)
        assert len(events) > 0

        event_types = [event.type for event in events]
        assert "response.created" in event_types, "Should have response.created event"
        assert "response.completed" in event_types, "Should have response.completed event"
        assert "response.output_item.added" in event_types, "Should have output_item.added events"
        assert "response.mcp_list_tools.in_progress" in event_types, (
            "Should have mcp_list_tools.in_progress event"
        )
        assert "response.mcp_list_tools.completed" in event_types, (
            "Should have mcp_list_tools.completed event"
        )
        assert "response.mcp_call.in_progress" in event_types, (
            "Should have mcp_call.in_progress event"
        )
        assert "response.mcp_call_arguments.delta" in event_types, (
            "Should have mcp_call_arguments.delta event"
        )
        assert "response.mcp_call_arguments.done" in event_types, (
            "Should have mcp_call_arguments.done event"
        )
        assert "response.mcp_call.completed" in event_types, "Should have mcp_call.completed event"
        assert_streaming_mcp_call_ids_match(events)

        completed_events = [e for e in events if e.type == "response.completed"]
        assert len(completed_events) == 1

        final_response = completed_events[0].response
        assert final_response.id is not None
        assert final_response.status == "completed"
        assert final_response.output is not None

        final_output = final_response.output
        final_output_types = [item.type for item in final_output]
        assert "mcp_list_tools" in final_output_types
        assert "mcp_call" in final_output_types

        # Verify mcp_call items in final output
        mcp_calls = [item for item in final_output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0
        assert_all_mcp_call_ids_prefixed(final_output)

        for mcp_call in mcp_calls:
            assert mcp_call.error is None
            assert mcp_call.status == "completed"
            assert mcp_call.server_label == "brave"
            assert mcp_call.name is not None
            assert mcp_call.arguments is not None
            assert mcp_call.output is not None

        # Strict validation for cloud backends - check for text output events
        assert "response.content_part.added" in event_types, "Should have content_part.added event"
        assert "response.output_text.delta" in event_types, "Should have output_text.delta events"
        assert "response.output_text.done" in event_types, "Should have output_text.done event"
        assert "response.content_part.done" in event_types, "Should have content_part.done event"

        assert "message" in final_output_types

        # Verify text deltas combine to final message
        text_deltas = [e.delta for e in events if e.type == "response.output_text.delta"]
        assert len(text_deltas) > 0, "Should have text deltas"

        # Get final text from output_text.done event
        text_done_events = [e for e in events if e.type == "response.output_text.done"]
        assert len(text_done_events) > 0

        final_text = text_done_events[0].text
        assert len(final_text) > 0, "Final text should not be empty"

    def test_mcp_approval_required_interrupts_and_resumes(self, model, api_client):
        """Test MCP approval-required workflow eventually supports interruption and resume."""

        time.sleep(2)  # Avoid rate limiting

        resp = api_client.responses.create(
            model=model,
            input=(
                "First use deepwiki's ask_question tool with repoName "
                "'modelcontextprotocol/specification' to answer: 'According to the 2025-03-26 "
                "MCP specification, what transport protocols are supported?' in one short "
                "sentence. Then use brave_web_search to search for 'Python programming "
                "language' with count set to 1."
            ),
            tools=[DEEPWIKI_MCP_TOOL, BRAVE_MCP_TOOL_REQUIRE_APPROVAL_ALWAYS],
            stream=False,
            reasoning={"effort": "low"},
        )

        assert_mcp_approval_interruption_non_streaming(resp)

        # TODO: extend this same test in a follow-up PR with the approval continuation
        # request and assert the resumed turn emits mcp_call plus the final assistant output.

    def test_mcp_multi_server_tool_call(self, model, api_client):
        """Test MCP tool call with multiple MCP servers (non-streaming)."""

        time.sleep(2)  # Avoid rate limiting

        resp = api_client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[DEEPWIKI_MCP_TOOL, BRAVE_MCP_TOOL],
            stream=False,
            reasoning={"effort": "low"},
        )

        assert resp.error is None
        assert resp.id is not None
        assert resp.status == "completed"
        assert resp.output is not None

        list_tools_items = [item for item in resp.output if item.type == "mcp_list_tools"]
        assert len(list_tools_items) == 2
        list_tool_labels = {item.server_label for item in list_tools_items}
        assert list_tool_labels == {"brave", "deepwiki"}

        mcp_calls = [item for item in resp.output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0
        for mcp_call in mcp_calls:
            assert mcp_call.server_label == "brave"

    def test_mcp_multi_server_tool_call_streaming(self, model, api_client):
        """Test MCP tool call with multiple MCP servers (streaming)."""

        time.sleep(2)  # Avoid rate limiting

        resp = api_client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[DEEPWIKI_MCP_TOOL, BRAVE_MCP_TOOL],
            stream=True,
            reasoning={"effort": "low"},
        )

        events = list(resp)
        assert len(events) > 0

        list_tools_events = [
            event
            for event in events
            if event.type == "response.output_item.added"
            and event.item is not None
            and event.item.type == "mcp_list_tools"
        ]
        assert len(list_tools_events) == 2
        list_tool_labels = {event.item.server_label for event in list_tools_events}
        assert list_tool_labels == {"brave", "deepwiki"}

        completed_events = [e for e in events if e.type == "response.completed"]
        assert len(completed_events) == 1
        final_response = completed_events[0].response
        assert final_response.output is not None

        final_list_tools = [item for item in final_response.output if item.type == "mcp_list_tools"]
        assert len(final_list_tools) == 2
        final_labels = {item.server_label for item in final_list_tools}
        assert final_labels == {"brave", "deepwiki"}

        mcp_calls = [item for item in final_response.output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0
        for mcp_call in mcp_calls:
            assert mcp_call.server_label == "brave"

    def test_previous_response_id_mcp_binding_behavior(self, model, api_client):
        """Resumed turns should not relist existing MCP bindings."""

        assert_previous_response_id_mcp_binding_behavior_non_streaming(model, api_client)

    def test_previous_response_id_mcp_binding_behavior_streaming(self, model, api_client):
        """Streaming resumed turns should only list newly added MCP bindings."""

        assert_previous_response_id_mcp_binding_behavior_streaming(model, api_client)

    def test_concurrent_mcp_different_servers(self, model, api_client):
        """Concurrent non-streaming requests with different MCP servers don't contaminate each other."""

        def brave_request():
            return api_client.responses.create(
                model=model,
                input=MCP_TEST_PROMPT,
                tools=[BRAVE_MCP_TOOL],
                stream=False,
                reasoning={"effort": "low"},
            )

        def deepwiki_request():
            return api_client.responses.create(
                model=model,
                input=(
                    "What transport protocols does the 2025-03-26 version of the MCP spec "
                    "(modelcontextprotocol/modelcontextprotocol) support?"
                ),
                tools=[DEEPWIKI_MCP_TOOL],
                stream=False,
                reasoning={"effort": "low"},
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            future_brave = pool.submit(brave_request)
            future_deepwiki = pool.submit(deepwiki_request)

            resp_brave = future_brave.result(timeout=120)
            resp_deepwiki = future_deepwiki.result(timeout=120)

        def _check_response(resp, expected_label):
            assert resp.error is None
            calls = [item for item in resp.output if item.type == "mcp_call"]
            assert len(calls) > 0
            for call in calls:
                assert call.server_label == expected_label, (
                    f"{expected_label} request got server_label={call.server_label}"
                )

        _check_response(resp_brave, "brave")
        _check_response(resp_deepwiki, "deepwiki")

    def test_concurrent_mcp_different_servers_streaming(self, model, api_client):
        """Concurrent streaming requests with different MCP servers don't contaminate each other."""

        def brave_stream():
            return list(
                api_client.responses.create(
                    model=model,
                    input=MCP_TEST_PROMPT,
                    tools=[BRAVE_MCP_TOOL],
                    stream=True,
                    reasoning={"effort": "low"},
                )
            )

        def deepwiki_stream():
            return list(
                api_client.responses.create(
                    model=model,
                    input=(
                        "What transport protocols does the 2025-03-26 version of the MCP spec "
                        "(modelcontextprotocol/modelcontextprotocol) support?"
                    ),
                    tools=[DEEPWIKI_MCP_TOOL],
                    stream=True,
                    reasoning={"effort": "low"},
                )
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            future_brave = pool.submit(brave_stream)
            future_deepwiki = pool.submit(deepwiki_stream)

            events_brave = future_brave.result(timeout=120)
            events_deepwiki = future_deepwiki.result(timeout=120)

        def _check_stream(events, expected_label):
            list_tools = [
                e
                for e in events
                if e.type == "response.output_item.added"
                and e.item is not None
                and e.item.type == "mcp_list_tools"
            ]
            assert len(list_tools) > 0
            for e in list_tools:
                assert e.item.server_label == expected_label, (
                    f"{expected_label} stream got mcp_list_tools server_label={e.item.server_label}"
                )

        _check_stream(events_brave, "brave")
        _check_stream(events_deepwiki, "deepwiki")


# =============================================================================
# Local Backend Tests (gRPC with Harmony model) - Tool Choice
# =============================================================================


@pytest.mark.engine("sglang", "vllm", "trtllm", "tokenspeed")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("openai/gpt-oss-20b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
@pytest.mark.parametrize("api_client", ["openai", "smg"], indirect=True)
class TestToolChoiceGptOss:
    """Tool choice tests against local gRPC backend with Harmony model."""

    def test_tool_choice_auto(self, model, api_client):
        """Test tool_choice="auto" allows model to decide whether to use tools."""

        tools = [GET_WEATHER_FUNCTION]

        resp = api_client.responses.create(
            model=model,
            input="What is the weather in Seattle?",
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        output = resp.output
        assert len(output) > 0

        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0, (
            "Model should choose to call function with tool_choice='auto'"
        )

    def test_tool_choice_required(self, model, api_client):
        """Test tool_choice="required" forces the model to call at least one tool."""

        tools = [CALCULATE_FUNCTION]

        resp = api_client.responses.create(
            model=model,
            input="What is 15 * 23?",
            tools=tools,
            tool_choice="required",
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        output = resp.output
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0, (
            "tool_choice='required' must force at least one function call"
        )

    def test_tool_choice_specific_function(self, model, api_client):
        """Test tool_choice with specific function name forces that function to be called."""

        tools = [SEARCH_WEB_FUNCTION, GET_WEATHER_FUNCTION]

        resp = api_client.responses.create(
            model=model,
            input="What's happening in the news today?",
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "search_web"}},
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        output = resp.output
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0, "Must call the specified function"
        assert function_calls[0].name == "search_web", (
            "Must call the function specified in tool_choice"
        )

    def test_tool_choice_streaming(self, model, api_client):
        """Test tool_choice parameter works correctly with streaming."""

        tools = [CALCULATE_FUNCTION]

        resp = api_client.responses.create(
            model=model,
            input="Calculate 42 * 17",
            tools=tools,
            tool_choice="required",
            stream=True,
        )

        events = list(resp)
        assert len(events) > 0

        event_types = [e.type for e in events]
        assert "response.function_call_arguments.delta" in event_types

        completed_events = [e for e in events if e.type == "response.completed"]
        assert len(completed_events) == 1

        output = completed_events[0].response.output
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0

    def test_tool_choice_with_mcp_tools(self, model, api_client):
        """Test tool_choice parameter works with MCP tools."""

        tools = [DEEPWIKI_MCP_TOOL]

        resp = api_client.responses.create(
            model=model,
            input=(
                "What transport protocols does the 2025-03-26 version of the MCP spec "
                "(modelcontextprotocol/modelcontextprotocol) support?"
            ),
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        output = resp.output
        mcp_calls = [item for item in output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0, "tool_choice='auto' should allow MCP tool calls"

    def test_tool_choice_mixed_function_and_mcp(self, model, api_client):
        """Test tool_choice with mixed function and MCP tools."""

        tools = [DEEPWIKI_MCP_TOOL, LOCAL_SEARCH_FUNCTION]

        resp = api_client.responses.create(
            model=model,
            input="Search for information about Python",
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "local_search"}},
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        output = resp.output
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0
        assert function_calls[0].name == "local_search"

        mcp_calls = [item for item in output if item.type == "mcp_call"]
        assert len(mcp_calls) == 0, "Should only call specified function, not MCP tools"

    def test_basic_function_call(self, model, api_client):
        """Test basic function calling workflow."""

        tools = [GET_HOROSCOPE_FUNCTION]
        system_prompt = (
            "You are a helpful assistant that can call functions. "
            "When a user asks for horoscope information, call the function. "
            "IMPORTANT: Don't reply directly to the user, only call the function. "
        )

        input_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is my horoscope? I am an Aquarius."},
        ]

        resp = api_client.responses.create(model=model, input=input_list, tools=tools)

        assert resp.error is None
        assert resp.id is not None
        assert resp.status == "completed"

        output = resp.output
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0

        function_call = function_calls[0]
        assert function_call.name == "get_horoscope"

        args = json.loads(function_call.arguments)
        assert "sign" in args
        assert args["sign"].lower() == "aquarius"

    def test_mcp_basic_tool_call(self, model, api_client):
        """Test basic MCP tool call (non-streaming)."""

        time.sleep(2)

        resp = api_client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[BRAVE_MCP_TOOL],
            stream=False,
            reasoning={"effort": "low"},
        )

        assert resp.error is None
        assert resp.id is not None
        assert resp.status == "completed"
        assert len(resp.output_text) > 0

        output_types = [item.type for item in resp.output]
        assert "mcp_list_tools" in output_types

        mcp_calls = [item for item in resp.output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0

        for mcp_call in mcp_calls:
            assert mcp_call.id is not None
            assert mcp_call.error is None
            assert mcp_call.status == "completed"
            assert mcp_call.server_label == "brave"

    def test_mcp_basic_tool_call_streaming(self, model, api_client):
        """Test basic MCP tool call (streaming)."""

        time.sleep(2)

        resp = api_client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[BRAVE_MCP_TOOL],
            stream=True,
            reasoning={"effort": "low"},
        )

        events = list(resp)
        assert len(events) > 0

        event_types = [event.type for event in events]
        assert "response.created" in event_types
        assert "response.completed" in event_types
        assert "response.mcp_list_tools.completed" in event_types
        assert "response.mcp_call.completed" in event_types

    def test_mixed_mcp_and_function_tools(self, model, api_client):
        """Test mixed MCP and function tools (non-streaming)."""

        resp = api_client.responses.create(
            model=model,
            input="Give me diagnostics for the Astra-7 Core Reactor.",
            tools=[BRAVE_MCP_TOOL, SYSTEM_DIAGNOSTICS_FUNCTION],
            stream=False,
            tool_choice="auto",
        )

        assert resp.error is None
        assert resp.id is not None
        assert resp.output is not None

        output = resp.output
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0

        system_diagnostics_call = function_calls[0]
        assert system_diagnostics_call.name == "get_system_diagnostics"
        assert system_diagnostics_call.call_id is not None

        args = json.loads(system_diagnostics_call.arguments)
        assert "system_name" in args
        assert "astra-7" in args["system_name"].lower()

    def test_mixed_mcp_and_function_tools_streaming(self, model, api_client):
        """Test mixed MCP and function tools (streaming)."""

        resp = api_client.responses.create(
            model=model,
            input="Give me diagnostics for the Astra-7 Core Reactor.",
            tools=[BRAVE_MCP_TOOL, SYSTEM_DIAGNOSTICS_FUNCTION],
            stream=True,
            tool_choice="auto",
        )

        events = list(resp)
        assert len(events) > 0

        event_types = [e.type for e in events]
        assert "response.created" in event_types
        assert "response.mcp_list_tools.completed" in event_types
        assert "response.function_call_arguments.delta" in event_types
        assert "response.function_call_arguments.done" in event_types

        func_arg_deltas = [e for e in events if e.type == "response.function_call_arguments.delta"]
        assert len(func_arg_deltas) > 0

        full_delta_event = "".join(e.delta for e in func_arg_deltas)
        assert "system_name" in full_delta_event.lower() and "astra-7" in full_delta_event.lower()

    def test_mcp_multi_server_tool_call(self, model, api_client):
        """Test MCP tool call with multiple MCP servers (non-streaming)."""

        time.sleep(2)

        resp = api_client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[DEEPWIKI_MCP_TOOL, BRAVE_MCP_TOOL],
            stream=False,
            reasoning={"effort": "low"},
        )

        assert resp.error is None
        assert resp.id is not None
        assert resp.status == "completed"
        assert resp.output is not None

        list_tools_items = [item for item in resp.output if item.type == "mcp_list_tools"]
        assert len(list_tools_items) == 2
        list_tool_labels = {item.server_label for item in list_tools_items}
        assert list_tool_labels == {"brave", "deepwiki"}

        mcp_calls = [item for item in resp.output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0
        for mcp_call in mcp_calls:
            assert mcp_call.server_label == "brave"

    def test_mcp_multi_server_tool_call_streaming(self, model, api_client):
        """Test MCP tool call with multiple MCP servers (streaming)."""

        time.sleep(2)

        resp = api_client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[DEEPWIKI_MCP_TOOL, BRAVE_MCP_TOOL],
            stream=True,
            reasoning={"effort": "low"},
        )

        events = list(resp)
        assert len(events) > 0

        list_tools_events = [
            event
            for event in events
            if event.type == "response.output_item.added"
            and event.item is not None
            and event.item.type == "mcp_list_tools"
        ]
        assert len(list_tools_events) == 2
        list_tool_labels = {event.item.server_label for event in list_tools_events}
        assert list_tool_labels == {"brave", "deepwiki"}

        completed_events = [e for e in events if e.type == "response.completed"]
        assert len(completed_events) == 1
        final_response = completed_events[0].response
        assert final_response.output is not None

        final_list_tools = [item for item in final_response.output if item.type == "mcp_list_tools"]
        assert len(final_list_tools) == 2
        final_labels = {item.server_label for item in final_list_tools}
        assert final_labels == {"brave", "deepwiki"}

        mcp_calls = [item for item in final_response.output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0
        for mcp_call in mcp_calls:
            assert mcp_call.server_label == "brave"


# =============================================================================
# Local Backend Tests (gRPC with Qwen model) - Tool Choice
# =============================================================================


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("Qwen/Qwen2.5-14B-Instruct")
@pytest.mark.gateway(extra_args=["--tool-call-parser", "qwen", "--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
@pytest.mark.parametrize("api_client", ["openai", "smg"], indirect=True)
class TestToolChoiceLocal:
    """Tool choice tests against local gRPC backend with Qwen model."""

    def test_tool_choice_auto(self, model, api_client):
        """Test tool_choice="auto" allows model to decide whether to use tools."""

        tools = [GET_WEATHER_FUNCTION]

        resp = api_client.responses.create(
            model=model,
            input="What is the weather in Seattle?",
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        output = resp.output
        assert len(output) > 0

        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0

    def test_tool_choice_required(self, model, api_client):
        """Test tool_choice="required" forces the model to call at least one tool."""

        tools = [CALCULATE_FUNCTION]

        resp = api_client.responses.create(
            model=model,
            input="What is 15 * 23?",
            tools=tools,
            tool_choice="required",
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        function_calls = [item for item in resp.output if item.type == "function_call"]
        assert len(function_calls) > 0

    def test_tool_choice_specific_function(self, model, api_client):
        """Test tool_choice with specific function name forces that function to be called."""

        tools = [SEARCH_WEB_FUNCTION, GET_WEATHER_FUNCTION]

        resp = api_client.responses.create(
            model=model,
            input="What's happening in the news today?",
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "search_web"}},
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        function_calls = [item for item in resp.output if item.type == "function_call"]
        assert len(function_calls) > 0
        assert function_calls[0].name == "search_web"

    def test_mcp_basic_tool_call(self, model, api_client):
        """Test basic MCP tool call (non-streaming)."""

        time.sleep(2)

        resp = api_client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[BRAVE_MCP_TOOL],
            stream=False,
            reasoning={"effort": "low"},
        )

        assert resp.error is None
        assert resp.id is not None
        assert resp.status == "completed"

        output_types = [item.type for item in resp.output]
        assert "mcp_list_tools" in output_types

        mcp_calls = [item for item in resp.output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0

    def test_mcp_basic_tool_call_streaming(self, model, api_client):
        """Test basic MCP tool call (streaming)."""

        time.sleep(2)

        resp = api_client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[BRAVE_MCP_TOOL],
            stream=True,
            reasoning={"effort": "low"},
        )

        events = list(resp)
        assert len(events) > 0

        event_types = [event.type for event in events]
        assert "response.created" in event_types
        assert "response.completed" in event_types

    def test_tool_choice_with_mcp_tools(self, model, api_client):
        """Test tool_choice parameter works with MCP tools."""

        tools = [DEEPWIKI_MCP_TOOL]

        resp = api_client.responses.create(
            model=model,
            input=(
                "What transport protocols does the 2025-03-26 version of the MCP spec "
                "(modelcontextprotocol/modelcontextprotocol) support?"
            ),
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        output = resp.output
        mcp_calls = [item for item in output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0, "tool_choice='auto' should allow MCP tool calls"

    def test_tool_choice_mixed_function_and_mcp(self, model, api_client):
        """Test tool_choice with mixed function and MCP tools."""

        tools = [DEEPWIKI_MCP_TOOL, LOCAL_SEARCH_FUNCTION]

        resp = api_client.responses.create(
            model=model,
            input="Search for information about Python",
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "local_search"}},
            stream=False,
        )

        assert resp.id is not None
        assert resp.error is None

        output = resp.output
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0
        assert function_calls[0].name == "local_search"

        mcp_calls = [item for item in output if item.type == "mcp_call"]
        assert len(mcp_calls) == 0, "Should only call specified function, not MCP tools"

    def test_mixed_mcp_and_function_tools(self, model, api_client):
        """Test mixed MCP and function tools (non-streaming)."""

        resp = api_client.responses.create(
            model=model,
            input="Give me diagnostics for the Astra-7 Core Reactor.",
            tools=[BRAVE_MCP_TOOL, SYSTEM_DIAGNOSTICS_FUNCTION],
            stream=False,
            tool_choice="auto",
        )

        assert resp.error is None
        assert resp.id is not None
        assert resp.output is not None

        output = resp.output
        function_calls = [item for item in output if item.type == "function_call"]
        assert len(function_calls) > 0

        system_diagnostics_call = function_calls[0]
        assert system_diagnostics_call.name == "get_system_diagnostics"
        assert system_diagnostics_call.call_id is not None

        args = json.loads(system_diagnostics_call.arguments)
        assert "system_name" in args
        assert "astra-7" in args["system_name"].lower()

    def test_mixed_mcp_and_function_tools_streaming(self, model, api_client):
        """Test mixed MCP and function tools (streaming)."""

        resp = api_client.responses.create(
            model=model,
            input="Give me diagnostics for the Astra-7 Core Reactor.",
            tools=[BRAVE_MCP_TOOL, SYSTEM_DIAGNOSTICS_FUNCTION],
            stream=True,
            tool_choice="auto",
        )

        events = list(resp)
        assert len(events) > 0

        event_types = [e.type for e in events]
        assert "response.created" in event_types
        assert "response.mcp_list_tools.completed" in event_types
        assert "response.function_call_arguments.delta" in event_types
        assert "response.function_call_arguments.done" in event_types

        func_arg_deltas = [e for e in events if e.type == "response.function_call_arguments.delta"]
        assert len(func_arg_deltas) > 0

        full_delta_event = "".join(e.delta for e in func_arg_deltas)
        assert "system_name" in full_delta_event.lower() and "astra-7" in full_delta_event.lower()

    def test_mcp_multi_server_tool_call(self, model, api_client):
        """Test MCP tool call with multiple MCP servers (non-streaming)."""

        time.sleep(2)

        resp = api_client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[DEEPWIKI_MCP_TOOL, BRAVE_MCP_TOOL],
            stream=False,
            reasoning={"effort": "low"},
        )

        assert resp.error is None
        assert resp.id is not None
        assert resp.status == "completed"
        assert resp.output is not None

        list_tools_items = [item for item in resp.output if item.type == "mcp_list_tools"]
        assert len(list_tools_items) == 2
        list_tool_labels = {item.server_label for item in list_tools_items}
        assert list_tool_labels == {"brave", "deepwiki"}

        mcp_calls = [item for item in resp.output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0
        for mcp_call in mcp_calls:
            assert mcp_call.server_label == "brave"

    def test_mcp_multi_server_tool_call_streaming(self, model, api_client):
        """Test MCP tool call with multiple MCP servers (streaming)."""

        time.sleep(2)

        resp = api_client.responses.create(
            model=model,
            input=MCP_TEST_PROMPT,
            tools=[DEEPWIKI_MCP_TOOL, BRAVE_MCP_TOOL],
            stream=True,
            reasoning={"effort": "low"},
        )

        events = list(resp)
        assert len(events) > 0

        list_tools_events = [
            event
            for event in events
            if event.type == "response.output_item.added"
            and event.item is not None
            and event.item.type == "mcp_list_tools"
        ]
        assert len(list_tools_events) == 2
        list_tool_labels = {event.item.server_label for event in list_tools_events}
        assert list_tool_labels == {"brave", "deepwiki"}

        completed_events = [e for e in events if e.type == "response.completed"]
        assert len(completed_events) == 1
        final_response = completed_events[0].response
        assert final_response.output is not None

        final_list_tools = [item for item in final_response.output if item.type == "mcp_list_tools"]
        assert len(final_list_tools) == 2
        final_labels = {item.server_label for item in final_list_tools}
        assert final_labels == {"brave", "deepwiki"}

        mcp_calls = [item for item in final_response.output if item.type == "mcp_call"]
        assert len(mcp_calls) > 0
        for mcp_call in mcp_calls:
            assert mcp_call.server_label == "brave"
