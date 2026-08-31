"""Muse-Glimmer E2E tests.

Muse-Glimmer frames every assistant message as a channel-scoped segment,
``<|start|>assistant to=<recipient><|message|>…`` closed by ``<|eom|>`` or
``<|eot|>``. ``to=self`` is chain-of-thought, ``to=user`` is the answer, and any
other recipient addresses a tool whose body carries an ATEM call block.

The unit and integration tests for the two parsers assert against transcripts we
wrote from the published format. These tests are the only ones that assert
against what the model actually emits, so they deliberately check the things a
hand-written fixture cannot: that the framing markers survive detokenization at
all, that they never reach the client, and that the reasoning/content/tool-call
split holds on real generations, streaming and not.

Only SGLang is marked: support for this architecture landed in SGLang 0.5.18,
which the repo pins, and is not in the pinned vLLM or TensorRT-LLM.
"""

from __future__ import annotations

import json
import logging

import pytest

logger = logging.getLogger(__name__)

MODEL_ID = "meta-models/Muse-Glimmer-30B"

# Anything from the wire protocol. None of these may reach a client field.
FRAMING_MARKERS = (
    "<|start|>",
    "<|message|>",
    "<|eom|>",
    "<|eot|>",
    "<atem:",
    "to=self",
    "to=user",
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"},
            },
            "required": ["city"],
        },
    },
}


def assert_no_framing(label: str, text: str | None) -> None:
    """A client-visible field must never carry protocol framing."""
    if not text:
        return
    for marker in FRAMING_MARKERS:
        assert marker not in text, f"{label} leaked {marker!r}: {text!r}"


@pytest.mark.engine("sglang")
@pytest.mark.gpu(4)
@pytest.mark.e2e
@pytest.mark.model(MODEL_ID)
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
@pytest.mark.parametrize("api_client", ["openai"], indirect=True)
class TestMuseGlimmerParsing:
    """Reasoning and tool-call parsing against the real checkpoint.

    No parser is passed on the gateway command line on purpose: resolution goes
    through the model-id mapping, so this is also the only place the
    ``muse-glimmer*`` globs are exercised end to end.
    """

    def test_reasoning_and_content_are_separated(self, model, api_client):
        """A plain turn splits into reasoning_content and content, with no framing."""
        response = api_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is 17 * 23? Think it through."}],
            max_tokens=400,
            extra_body={"separate_reasoning": True},
        )

        message = response.choices[0].message
        reasoning = getattr(message, "reasoning_content", None)
        assert_no_framing("content", message.content)
        assert_no_framing("reasoning_content", reasoning)
        assert message.content, "expected a user-facing answer"
        # Without this the test passes if the parser drops every to=self segment:
        # the answer still arrives, framing-free, and nothing else notices.
        assert reasoning, "expected reasoning_content with separate_reasoning enabled"

    def test_streaming_never_leaks_framing(self, model, api_client):
        """Streaming deltas must not surface framing, even mid-marker.

        The parser holds back partial markers across chunk boundaries; if that
        regressed, a `<|eo` fragment would arrive as visible content.
        """
        stream = api_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Name three primary colors."}],
            max_tokens=300,
            stream=True,
            extra_body={"separate_reasoning": True},
        )

        content = ""
        reasoning = ""
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content
            if getattr(delta, "reasoning_content", None):
                reasoning += delta.reasoning_content

        assert_no_framing("streamed content", content)
        assert_no_framing("streamed reasoning", reasoning)
        assert content, "expected streamed answer text"
        # Same gap as the non-streaming case: assert_no_framing returns early on
        # an empty string, so silently dropping every to=self delta would pass.
        assert reasoning, "expected streamed reasoning_content"

    def test_tool_call_is_parsed(self, model, api_client):
        """An ATEM call in a tool channel becomes a structured tool_call."""
        response = api_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=500,
        )

        choice = response.choices[0]
        assert_no_framing("content", choice.message.content)

        tool_calls = choice.message.tool_calls
        assert tool_calls, f"expected a tool call, got content={choice.message.content!r}"
        call = tool_calls[0]
        assert call.function.name == "get_weather"
        # Arguments must be valid JSON — the ATEM body is XML-ish on the wire,
        # so this proves the parameter extraction and typing ran.
        args = json.loads(call.function.arguments)
        assert "city" in args, f"expected a city argument, got {args!r}"
        assert choice.finish_reason == "tool_calls"

    def test_streaming_tool_call_arguments_are_valid_json(self, model, api_client):
        """Accumulated streamed tool-call arguments must parse as JSON.

        ATEM parameters are not incremental JSON, so the parser emits each call
        whole; a regression that streamed partial fragments would produce
        unparsable arguments here.
        """
        stream = api_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is the weather in Berlin?"}],
            tools=[WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=500,
            stream=True,
        )

        names: dict[int, str] = {}
        arguments: dict[int, str] = {}
        content = ""
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content
            for call in delta.tool_calls or []:
                if call.function and call.function.name:
                    names[call.index] = call.function.name
                if call.function and call.function.arguments:
                    arguments[call.index] = arguments.get(call.index, "") + call.function.arguments

        assert_no_framing("streamed content", content)
        assert names, "expected at least one streamed tool call"
        for index, name in names.items():
            assert name == "get_weather"
            parsed = json.loads(arguments.get(index, "{}"))
            assert "city" in parsed, f"call {index} lost its arguments: {parsed!r}"

    def test_default_request_never_leaks_framing(self, model, api_client):
        """The default request path must parse reasoning and strip framing.

        ``separate_reasoning`` defaults to true. Omitting the extension here
        protects that API default as well as the parser's automatic model-id
        selection.
        """
        response = api_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Briefly, why is the sky blue?"}],
            max_tokens=300,
        )

        message = response.choices[0].message
        assert_no_framing("content", message.content)
        assert_no_framing("reasoning_content", getattr(message, "reasoning_content", None))
        assert message.content, "expected an answer"
        assert getattr(message, "reasoning_content", None), "expected parsed reasoning"
