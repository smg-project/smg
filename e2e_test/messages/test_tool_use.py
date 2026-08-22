"""Tool use tests for Anthropic Messages API.

Tests for tool calling functionality using the Anthropic tool format,
including single tool calls, round-trip tool results, streaming tool use,
and multiple tools.
"""

from __future__ import annotations

import json
import logging

import pytest

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Definitions (Anthropic format)
# =============================================================================

GET_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather in a given location.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city name, e.g., San Francisco",
            },
        },
        "required": ["location"],
    },
}

CALCULATE_TOOL = {
    "name": "calculate",
    "description": "Perform a mathematical calculation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate, e.g., '2 + 2'",
            },
        },
        "required": ["expression"],
    },
}


# =============================================================================
# Tool Use Tests
# =============================================================================


@pytest.mark.vendor("anthropic")
@pytest.mark.gpu(0)
@pytest.mark.parametrize("setup_backend", ["anthropic"], indirect=True)
class TestToolUseBasic:
    """Tool use tests against the Anthropic Messages API."""

    def test_single_tool_call(self, model, api_client):
        """Test that the model calls a single tool when appropriate."""

        response = api_client.messages.create(
            model=model,
            max_tokens=200,
            tools=[GET_WEATHER_TOOL],
            messages=[
                {"role": "user", "content": "What is the weather in San Francisco?"},
            ],
        )

        assert response.id is not None
        assert response.stop_reason == "tool_use"

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        assert len(tool_use_blocks) > 0, "Response should contain a tool_use block"

        tool_use = tool_use_blocks[0]
        assert tool_use.name == "get_weather"
        assert tool_use.id is not None
        assert isinstance(tool_use.input, dict)
        assert "location" in tool_use.input

    def test_tool_call_and_result_round_trip(self, model, api_client):
        """Test full round-trip: tool call -> tool result -> final text response."""

        # First request: model should request a tool call
        response1 = api_client.messages.create(
            model=model,
            max_tokens=200,
            tools=[GET_WEATHER_TOOL],
            messages=[
                {"role": "user", "content": "What is the weather in Tokyo?"},
            ],
        )

        assert response1.stop_reason == "tool_use"
        tool_use_blocks = [b for b in response1.content if b.type == "tool_use"]
        assert len(tool_use_blocks) > 0
        tool_use = tool_use_blocks[0]

        # Second request: provide tool result, expect final text
        response2 = api_client.messages.create(
            model=model,
            max_tokens=200,
            tools=[GET_WEATHER_TOOL],
            messages=[
                {"role": "user", "content": "What is the weather in Tokyo?"},
                {"role": "assistant", "content": response1.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": json.dumps({"temperature": "22°C", "condition": "sunny"}),
                        }
                    ],
                },
            ],
        )

        assert response2.id is not None
        assert response2.stop_reason == "end_turn"

        text_blocks = [b for b in response2.content if b.type == "text"]
        assert len(text_blocks) > 0, "Final response should contain text"
        assert len(text_blocks[0].text) > 0

    def test_tool_use_streaming(self, model, api_client):
        """Test streaming with tool use returns input_json delta events."""

        with api_client.messages.stream(
            model=model,
            max_tokens=200,
            tools=[GET_WEATHER_TOOL],
            messages=[
                {"role": "user", "content": "What is the weather in London?"},
            ],
        ) as stream:
            event_types = set()
            input_json_deltas: dict[int, list[str]] = {}
            for event in stream:
                event_types.add(event.type)
                if event.type == "content_block_delta" and hasattr(event.delta, "partial_json"):
                    # Key by block index: two tool_use blocks would otherwise
                    # concatenate into "{...}{...}" and fail to parse.
                    input_json_deltas.setdefault(event.index, []).append(event.delta.partial_json)
            final_message = stream.get_final_message()

        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types

        # The weather prompt must produce a tool call (mirrors
        # test_single_tool_call), so input_json deltas must be present:
        # an empty list means the stream lost the tool-input deltas.
        assert final_message.stop_reason == "tool_use"
        assert input_json_deltas, "Expected input_json_delta events for the tool call"

        # Each tool_use block's deltas must concatenate to one valid JSON object
        for index, deltas in input_json_deltas.items():
            parsed = json.loads("".join(deltas))
            assert isinstance(parsed, dict), (
                f"content block {index} input_json did not parse to a dict: {parsed!r}"
            )

    def test_multiple_tools_available(self, model, api_client):
        """Test that model selects the correct tool when multiple are available."""

        response = api_client.messages.create(
            model=model,
            max_tokens=200,
            tools=[GET_WEATHER_TOOL, CALCULATE_TOOL],
            messages=[
                {"role": "user", "content": "What is 15 * 23?"},
            ],
        )

        assert response.id is not None
        assert response.stop_reason == "tool_use"

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        assert len(tool_use_blocks) > 0
        assert tool_use_blocks[0].name == "calculate"
