"""MLX backend E2E tests (Apple Silicon only).

Exercises router → gRPC → MLX worker. Backend-scoped (separate from
`chat_completions/`) because the fixtures need a macOS runner and a
different model than the sglang/vllm/trtllm parameterization.

Run: `E2E_RUNTIME=mlx E2E_ENGINE=mlx pytest e2e_test/mlx/test_mlx_backend.py -v`
"""

from __future__ import annotations

import json
import platform
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="MLX backend requires Apple Silicon (macOS arm64)",
)


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. 'Tokyo' or 'Paris'.",
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


@pytest.mark.engine("mlx")
@pytest.mark.gpu(0)  # MLX uses unified memory, no discrete GPU
@pytest.mark.model("mlx-community/Qwen3-0.6B-4bit")
@pytest.mark.gateway(extra_args=["--tool-call-parser", "qwen", "--reasoning-parser", "qwen3"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestMlxBackend:
    """E2E tests for the MLX gRPC backend.

    Qwen3-0.6B-4bit (~400 MB) is small enough for macos-latest and covers
    both tool calling and thinking mode.
    """

    # Disable thinking on basic chat tests so the 0.6B model has budget for
    # actual content within max_tokens.
    NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}

    def test_basic_chat(self, model, api_client):
        response = api_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word 'OK'."}],
            max_tokens=20,
            temperature=0,
            extra_body=self.NO_THINKING,
        )
        assert response.id
        assert response.choices[0].message.role == "assistant"
        assert isinstance(response.choices[0].message.content, str)
        assert len(response.choices[0].message.content) > 0
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    def test_streaming_chat(self, model, api_client):
        stream = api_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Count: 1, 2, 3."}],
            max_tokens=30,
            temperature=0,
            stream=True,
            extra_body=self.NO_THINKING,
        )
        chunks = list(stream)
        assert len(chunks) >= 2
        text_pieces = [
            c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content
        ]
        assert len(text_pieces) > 0
        assert "".join(text_pieces).strip()

    def test_tool_calling(self, model, api_client):
        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "What's the weather in Tokyo right now?"},
            ],
            tools=[WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=300,
            temperature=0,
        )
        msg = response.choices[0].message
        assert response.choices[0].finish_reason == "tool_calls", (
            f"Expected finish_reason=tool_calls, got {response.choices[0].finish_reason!r} "
            f"with content={msg.content!r}"
        )
        assert msg.tool_calls, "Expected tool_calls to be populated"
        call = msg.tool_calls[0]
        assert call.function.name == "get_weather"
        args = json.loads(call.function.arguments)
        assert "tokyo" in args.get("location", "").lower()

    def test_reasoning_thinking_mode(self, model, api_client):
        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Alice has 5 apples. She gives 2 to Bob. "
                        "How many apples does Alice have? Think step by step."
                    ),
                }
            ],
            max_tokens=600,
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        msg = response.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""
        # Asserting reasoning separately catches a regression that dumps the
        # whole <think>...</think> span into content.
        assert reasoning.strip(), (
            f"Expected non-empty reasoning_content with enable_thinking=True, "
            f"got content={content!r} reasoning={reasoning!r}"
        )
        assert "3" in (content + reasoning), (
            f"Expected '3' in response, got content={content!r} reasoning={reasoning!r}"
        )

    def test_max_tokens_finish_reason(self, model, api_client):
        response = api_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Tell me a story."}],
            max_tokens=10,
            temperature=0,
            extra_body=self.NO_THINKING,
        )
        assert response.choices[0].finish_reason == "length"
        assert response.usage.completion_tokens == 10
