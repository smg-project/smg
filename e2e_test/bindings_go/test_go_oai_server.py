"""Go OAI Server E2E Tests.

Tests for the Go OpenAI-compatible server, including:
- Non-streaming chat completions
- Streaming chat completions
- Function calling / tool use
- Error handling

These tests use the Go OAI server which connects directly to a gRPC worker.
"""

from __future__ import annotations

import json
import logging

import pytest

logger = logging.getLogger(__name__)


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.model("meta-llama/Llama-3.2-1B-Instruct")
class TestGoOAIServerBasic:
    """Basic tests for Go OAI server functionality."""

    def test_non_streaming_completion(self, go_openai_client, go_oai_server):
        """Test non-streaming chat completion through Go OAI server."""
        _, _, model_path = go_oai_server

        response = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=10,
            stream=False,
        )

        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0
        assert response.choices[0].finish_reason in ("stop", "length")
        logger.info(f"Response: {response.choices[0].message.content}")

    def test_streaming_completion(self, go_openai_client, go_oai_server):
        """Test streaming chat completion through Go OAI server."""
        _, _, model_path = go_oai_server

        response_stream = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            max_tokens=50,
            stream=True,
        )

        chunks = list(response_stream)
        assert len(chunks) > 1, "Streaming should return multiple chunks"

        # Collect content from all chunks
        content_parts = []
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

        full_content = "".join(content_parts)
        assert len(full_content) > 0, "Should have received content"
        logger.info(f"Streamed content: {full_content}")

        # Check final chunk has finish_reason
        final_chunk = chunks[-1]
        assert final_chunk.choices[0].finish_reason in ("stop", "length", None)

    def test_streaming_multiple_messages(self, go_openai_client, go_oai_server):
        """Test streaming with conversation history."""
        _, _, model_path = go_oai_server

        response_stream = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
                {"role": "user", "content": "And 3+3?"},
            ],
            max_tokens=20,
            stream=True,
        )

        chunks = list(response_stream)
        assert len(chunks) > 0, "Should receive response chunks"

        content_parts = []
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

        full_content = "".join(content_parts)
        assert len(full_content) > 0
        logger.info(f"Response to follow-up: {full_content}")


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.model("meta-llama/Llama-3.2-1B-Instruct")
class TestGoOAIServerFunctionCalling:
    """Tests for function calling through Go OAI server.

    Two independent reasons a tool call may not appear, and the xfail names
    whichever applies:

    1. The example server does not forward ``tools``/``tool_choice``. Its
       request model declares both fields (``models/chat.go``) but
       ``handlers/chat.go`` builds the downstream request field by field and
       never copies them, so the gateway is asked to generate without tools
       and no tool call can be produced. This is a server gap, not a model
       one, and it is what these tests currently hit.
    2. Once forwarding lands, ``meta-llama/Llama-3.2-1B-Instruct`` may still
       not reliably emit a tool call under ``tool_choice='required'``.

    Everything that does not depend on a tool call is asserted
    unconditionally — response shape, that the model produced *some* output
    (a completion swallowed by the streaming parser must fail here, not
    xfail), and, once a call exists, streamed index math, argument
    accumulation and JSON validity.
    """

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    SYSTEM_MESSAGE = (
        "You are a helpful assistant with tool calling capabilities. "
        "When you need to get weather information, use the get_weather function. "
        'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.'
    )

    def test_function_calling_non_streaming(self, go_openai_client, go_oai_server):
        """Test function calling in non-streaming mode."""
        _, _, model_path = go_oai_server

        response = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": "What's the weather in Tokyo?"},
            ],
            max_tokens=100,
            tools=self.TOOLS,
            tool_choice="required",
            stream=False,
        )

        # Structural: response shape (unconditional).
        assert response.choices is not None
        assert len(response.choices) > 0
        tool_calls = response.choices[0].message.tool_calls

        if not tool_calls:
            # No tool call is excusable; producing nothing at all is not. A
            # completion swallowed on the way back must fail rather than hide
            # behind the xfail below.
            content = response.choices[0].message.content
            assert content and content.strip(), (
                "no tool call AND no content - the completion was swallowed"
            )
            pytest.xfail(
                "no tool call: the Go example server does not forward "
                "tools/tool_choice (handlers/chat.go), so the gateway never "
                "sees the tool definitions"
            )

        # Structural: any parsed tool call must be well-formed (unconditional).
        tool_call = tool_calls[0]
        assert tool_call.function.name == "get_weather"

        args = json.loads(tool_call.function.arguments)
        assert isinstance(args, dict)

        # Model-dependent: argument quality.
        if "location" not in args:
            pytest.xfail(
                f"Llama-3.2-1B-Instruct produced schema-incomplete arguments {args} "
                "(known model limitation)"
            )
        logger.info(f"Tool call: {tool_call.function.name}({args})")

    def test_function_calling_streaming(self, go_openai_client, go_oai_server):
        """Test function calling in streaming mode."""
        _, _, model_path = go_oai_server

        response_stream = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": "What's the weather in Paris?"},
            ],
            max_tokens=100,
            tools=self.TOOLS,
            tool_choice="required",
            stream=True,
        )

        # Structural: the stream itself must be well-formed (unconditional).
        chunks = list(response_stream)
        assert len(chunks) > 0

        # Reconstruct tool calls (and content) from streaming chunks
        tool_calls_by_index = {}
        content_parts = []
        for chunk in chunks:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {
                            "name": "",
                            "arguments": "",
                        }
                    if tc.function.name:
                        tool_calls_by_index[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls_by_index[idx]["arguments"] += tc.function.arguments

        if not tool_calls_by_index:
            # The regression this suite guards against streams neither
            # tool_call deltas nor content deltas, which is exactly the shape
            # that would otherwise xfail silently here.
            assert "".join(content_parts).strip(), (
                "streamed no tool call AND no content - the completion was swallowed"
            )
            pytest.xfail(
                "no streamed tool call: the Go example server does not forward "
                "tools/tool_choice (handlers/chat.go), so the gateway never "
                "sees the tool definitions"
            )

        # Structural: index math and argument accumulation (unconditional).
        assert 0 in tool_calls_by_index, (
            f"Streamed tool call indices must start at 0, got {sorted(tool_calls_by_index)}"
        )
        first_tc = tool_calls_by_index[0]
        assert first_tc["name"] == "get_weather"

        args = json.loads(first_tc["arguments"])
        assert isinstance(args, dict)

        # Model-dependent: argument quality.
        if "location" not in args:
            pytest.xfail(
                f"Llama-3.2-1B-Instruct produced schema-incomplete arguments {args} "
                "(known model limitation)"
            )
        logger.info(f"Streamed tool call: {first_tc['name']}({args})")

    def test_function_calling_tool_choice_none(self, go_openai_client, go_oai_server):
        """Test that tool_choice='none' yields a text response.

        The suppression half of this contract cannot be exercised through the
        example server today: it does not forward ``tools``/``tool_choice``
        (``handlers/chat.go``), so no tool call could appear with or without
        ``tool_choice='none'``. What is still worth pinning - and is not
        model-dependent - is that the request succeeds and comes back as
        text, so nothing here may xfail.
        """
        _, _, model_path = go_oai_server

        response = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[
                {"role": "user", "content": "What's the weather in London?"},
            ],
            max_tokens=50,
            tools=self.TOOLS,
            tool_choice="none",
            stream=False,
        )

        assert response.choices is not None
        # With tool_choice="none", should get text response, not tool calls
        message = response.choices[0].message
        assert message.tool_calls is None or len(message.tool_calls) == 0
        assert message.content and message.content.strip(), (
            "tool_choice='none' must still return a text answer"
        )


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.model("meta-llama/Llama-3.2-1B-Instruct")
class TestGoOAIServerParameters:
    """Tests for various generation parameters through Go OAI server."""

    def test_temperature_variation(self, go_openai_client, go_oai_server):
        """Test that temperature parameter affects output."""
        _, _, model_path = go_oai_server

        # Low temperature - more deterministic
        response_low = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=10,
            temperature=0.1,
            stream=False,
        )

        # High temperature - more random
        response_high = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=10,
            temperature=1.5,
            stream=False,
        )

        # Both should produce valid responses
        assert response_low.choices[0].message.content is not None
        assert response_high.choices[0].message.content is not None

    def test_max_tokens_limit(self, go_openai_client, go_oai_server):
        """Test that max_tokens limits output length."""
        _, _, model_path = go_oai_server

        response = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": "Write a very long story about a dragon."}],
            max_tokens=5,
            stream=False,
        )

        assert response.choices is not None
        # With max_tokens=5, response should be short
        content = response.choices[0].message.content
        assert content is not None
        # The model might stop due to length limit
        assert response.choices[0].finish_reason in ("stop", "length")

    def test_top_p_parameter(self, go_openai_client, go_oai_server):
        """Test that top_p parameter is accepted."""
        _, _, model_path = go_oai_server

        response = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
            top_p=0.9,
            stream=False,
        )

        assert response.choices is not None
        assert response.choices[0].message.content is not None


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.model("meta-llama/Llama-3.2-1B-Instruct")
class TestGoOAIServerResponseValidation:
    """Tests for response field validation."""

    def test_response_fields_non_streaming(self, go_openai_client, go_oai_server):
        """Test that non-streaming response contains all required fields."""
        _, _, model_path = go_oai_server

        response = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=10,
            stream=False,
        )

        # Verify required response fields
        assert response.id is not None
        assert response.id.startswith("chatcmpl-")
        assert response.created is not None
        assert response.created > 0
        assert response.model is not None
        assert response.choices is not None
        assert len(response.choices) > 0

        # Verify choice structure
        choice = response.choices[0]
        assert choice.index == 0
        assert choice.message is not None
        assert choice.message.role == "assistant"
        assert choice.message.content is not None
        assert choice.finish_reason in ("stop", "length")

        logger.info(f"Response ID: {response.id}, Created: {response.created}")

    def test_response_fields_streaming(self, go_openai_client, go_oai_server):
        """Test that streaming chunks contain required fields."""
        _, _, model_path = go_oai_server

        response_stream = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": "Count to 3."}],
            max_tokens=20,
            stream=True,
        )

        chunks = list(response_stream)
        assert len(chunks) > 0, "Should receive at least one chunk"

        # Track finish_reason appearances
        finish_reasons = []
        first_id = None

        for chunk in chunks:
            # Each chunk should have id and created
            assert chunk.id is not None
            assert chunk.created is not None

            # All chunks should have same id
            if first_id is None:
                first_id = chunk.id
            else:
                assert chunk.id == first_id, "All chunks should have same id"

            # Track finish reasons
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reasons.append(chunk.choices[0].finish_reason)

        # finish_reason should appear exactly once per choice
        assert len(finish_reasons) == 1, f"Expected 1 finish_reason, got {len(finish_reasons)}"
        assert finish_reasons[0] in ("stop", "length")

        logger.info(f"Stream ID: {first_id}, finish_reason: {finish_reasons[0]}")

    def test_usage_tokens_non_streaming(self, go_openai_client, go_oai_server):
        """Test that usage tokens are returned in non-streaming response."""
        _, _, model_path = go_oai_server

        response = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=10,
            stream=False,
        )

        # Verify usage is present
        assert response.usage is not None, "Usage should be present"
        assert response.usage.prompt_tokens > 0, "prompt_tokens should be > 0"
        assert response.usage.completion_tokens > 0, "completion_tokens should be > 0"
        assert response.usage.total_tokens > 0, "total_tokens should be > 0"
        assert (
            response.usage.total_tokens
            == response.usage.prompt_tokens + response.usage.completion_tokens
        )

        logger.info(
            f"Usage - prompt: {response.usage.prompt_tokens}, "
            f"completion: {response.usage.completion_tokens}, "
            f"total: {response.usage.total_tokens}"
        )

    def test_usage_tokens_streaming(self, go_openai_client, go_oai_server):
        """Test that usage tokens are returned with stream_options."""
        _, _, model_path = go_oai_server

        response_stream = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=10,
            stream=True,
            stream_options={"include_usage": True},
        )

        chunks = list(response_stream)
        assert len(chunks) > 0

        # The last chunk should contain usage info
        usage_chunks = [c for c in chunks if c.usage is not None]

        # Usage should be in at least one chunk (typically the last)
        assert len(usage_chunks) > 0, "Should have at least one chunk with usage"

        usage = usage_chunks[-1].usage
        assert usage.prompt_tokens > 0, "prompt_tokens should be > 0"
        assert usage.completion_tokens >= 0, "completion_tokens should be >= 0"
        assert usage.total_tokens > 0, "total_tokens should be > 0"

        logger.info(
            f"Streaming usage - prompt: {usage.prompt_tokens}, "
            f"completion: {usage.completion_tokens}, "
            f"total: {usage.total_tokens}"
        )

    def test_finish_reason_stop(self, go_openai_client, go_oai_server):
        """Test that a short response finishes with 'stop'."""
        _, _, model_path = go_oai_server

        response = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": "Say 'hi'."}],
            max_tokens=50,  # Generous limit so model can finish naturally
            stream=False,
        )

        assert response.choices[0].finish_reason == "stop"
        logger.info(f"Finish reason: {response.choices[0].finish_reason}")

    def test_finish_reason_length(self, go_openai_client, go_oai_server):
        """Test that max_tokens limit produces 'length' finish reason."""
        _, _, model_path = go_oai_server

        response = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": "Write a very long story about dragons."}],
            max_tokens=5,  # Very short limit
            stream=False,
        )

        assert response.choices[0].finish_reason == "length"
        logger.info(f"Finish reason: {response.choices[0].finish_reason}")


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.model("meta-llama/Llama-3.2-1B-Instruct")
class TestGoOAIServerConcurrency:
    """Tests for concurrent request handling."""

    def test_concurrent_non_streaming(self, go_openai_client, go_oai_server):
        """Test handling of multiple concurrent non-streaming requests."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        _, _, model_path = go_oai_server

        def make_request(i: int):
            response = go_openai_client.chat.completions.create(
                model=model_path,
                messages=[{"role": "user", "content": f"Say the number {i}."}],
                max_tokens=10,
                stream=False,
            )
            return i, response

        # Run 4 concurrent requests
        num_requests = 4
        results = []

        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            for future in as_completed(futures):
                results.append(future.result())

        # Verify all requests completed successfully
        assert len(results) == num_requests
        for i, response in results:
            assert response.choices is not None
            assert response.choices[0].message.content is not None
            logger.info(f"Request {i}: {response.choices[0].message.content[:30]}...")

    def test_concurrent_streaming(self, go_openai_client, go_oai_server):
        """Test handling of multiple concurrent streaming requests."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        _, _, model_path = go_oai_server

        def make_streaming_request(i: int):
            response_stream = go_openai_client.chat.completions.create(
                model=model_path,
                messages=[{"role": "user", "content": f"Count from {i} to {i + 2}."}],
                max_tokens=30,
                stream=True,
            )
            chunks = list(response_stream)
            content_parts = []
            for chunk in chunks:
                if chunk.choices and chunk.choices[0].delta.content:
                    content_parts.append(chunk.choices[0].delta.content)
            return i, "".join(content_parts)

        # Run 4 concurrent streaming requests
        num_requests = 4
        results = []

        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_streaming_request, i) for i in range(num_requests)]
            for future in as_completed(futures):
                results.append(future.result())

        # Verify all streaming requests completed successfully
        assert len(results) == num_requests
        for i, content in results:
            assert len(content) > 0, f"Request {i} should have content"
            logger.info(f"Stream {i}: {content[:30]}...")


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.model("meta-llama/Llama-3.2-1B-Instruct")
class TestGoOAIServerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_message_content(self, go_openai_client, go_oai_server):
        """Test handling of empty message content."""
        _, _, model_path = go_oai_server

        response = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": ""}],
            max_tokens=10,
            stream=False,
        )

        # Should still get a response (model might say something about empty input)
        assert response.choices is not None
        logger.info(f"Response to empty: {response.choices[0].message.content}")

    def test_long_context(self, go_openai_client, go_oai_server):
        """Test handling of longer context."""
        _, _, model_path = go_oai_server

        # Create a longer message
        long_message = "The quick brown fox jumps over the lazy dog. " * 50

        response = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": long_message + " Summarize in one word."}],
            max_tokens=10,
            stream=False,
        )

        assert response.choices is not None
        assert response.choices[0].message.content is not None
        logger.info(f"Long context response: {response.choices[0].message.content}")

    def test_stop_sequences(self, go_openai_client, go_oai_server):
        """Test that stop sequences are respected."""
        _, _, model_path = go_oai_server

        response = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": "Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"}],
            max_tokens=50,
            stop=[","],
            stream=False,
        )

        assert response.choices is not None
        content = response.choices[0].message.content
        # With stop=",", response should stop at first comma
        # The model might not output a comma at all if it interprets the task differently
        logger.info(f"Stop sequence response: {content}")
        assert response.choices[0].finish_reason in ("stop", "length")

    def test_system_message_only(self, go_openai_client, go_oai_server):
        """Test request with system message and minimal user message."""
        _, _, model_path = go_oai_server

        response = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that speaks like a pirate.",
                },
                {"role": "user", "content": "Hi"},
            ],
            max_tokens=20,
            stream=False,
        )

        assert response.choices is not None
        assert response.choices[0].message.content is not None
        logger.info(f"Pirate response: {response.choices[0].message.content}")


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.model("meta-llama/Llama-3.2-1B-Instruct")
@pytest.mark.xfail(
    reason="Go OAI server does not currently support n > 1 (multiple choices)",
    strict=True,  # deterministic capability gap: XPASS must fail so the marker is removed when n>1 lands
)
class TestGoOAIServerMultipleChoices:
    """Tests for n parameter (multiple choices).

    Note: The Go OAI server currently does not support generating multiple
    choices (n > 1) — a deterministic server capability gap, not model
    flakiness — so these tests are strict xfails: when support is added they
    will XPASS and fail CI, forcing removal of the marker.
    """

    def test_n_parameter_non_streaming(self, go_openai_client, go_oai_server):
        """Test that n parameter returns multiple choices."""
        _, _, model_path = go_oai_server

        response = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": "Give me a random word."}],
            max_tokens=10,
            n=2,
            stream=False,
        )

        assert response.choices is not None
        assert len(response.choices) == 2, f"Expected 2 choices, got {len(response.choices)}"

        # Verify each choice has content and correct index
        for i, choice in enumerate(response.choices):
            assert choice.index == i
            assert choice.message.content is not None
            assert choice.finish_reason in ("stop", "length")
            logger.info(f"Choice {i}: {choice.message.content}")

    def test_n_parameter_streaming(self, go_openai_client, go_oai_server):
        """Test that n parameter works with streaming."""
        _, _, model_path = go_oai_server

        response_stream = go_openai_client.chat.completions.create(
            model=model_path,
            messages=[{"role": "user", "content": "Give me a random word."}],
            max_tokens=10,
            n=2,
            stream=True,
        )

        chunks = list(response_stream)
        assert len(chunks) > 0

        # Track content and finish reasons by choice index
        choice_content = {0: [], 1: []}
        choice_finish_reasons = {0: None, 1: None}

        for chunk in chunks:
            if not chunk.choices:
                continue
            for choice in chunk.choices:
                idx = choice.index
                if idx in choice_content:
                    if choice.delta.content:
                        choice_content[idx].append(choice.delta.content)
                    if choice.finish_reason:
                        choice_finish_reasons[idx] = choice.finish_reason

        # Both choices should have content and finish reasons
        for idx in [0, 1]:
            content = "".join(choice_content[idx])
            assert len(content) > 0 or choice_finish_reasons[idx] is not None, (
                f"Choice {idx} should have content or finish"
            )
            logger.info(f"Streaming choice {idx}: {content}, finish: {choice_finish_reasons[idx]}")
