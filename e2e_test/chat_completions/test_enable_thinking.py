"""Enable Thinking E2E Tests.

Tests for chat completions with enable_thinking feature (Qwen3 reasoning).

Source: Migrated from e2e_grpc/features/test_enable_thinking.py
"""

from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)

# API key is not validated by the gateway, but required for OpenAI-compatible headers
API_KEY = "not-used"


# =============================================================================
# Enable Thinking Tests (Qwen3)
# =============================================================================


@pytest.mark.engine("sglang", "vllm", "trtllm")
@pytest.mark.gpu(1)
@pytest.mark.model("Qwen/Qwen3-30B-A3B")
@pytest.mark.gateway(extra_args=["--reasoning-parser", "qwen3", "--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
@pytest.mark.parametrize("api_client", ["openai", "smg"], indirect=True)
class TestEnableThinking:
    """Tests for enable_thinking feature with Qwen3 reasoning parser."""

    def test_chat_completion_with_reasoning(self, model, api_client):
        """Test non-streaming with enable_thinking=True, reasoning_content should not be empty."""

        response = api_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0,
            extra_body={
                "separate_reasoning": True,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )

        assert len(response.choices) > 0
        assert response.choices[0].message.reasoning_content is not None

    def test_chat_completion_without_reasoning(self, model, api_client):
        """Test non-streaming with enable_thinking=False, reasoning_content should be empty."""

        response = api_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0,
            extra_body={
                "separate_reasoning": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        assert len(response.choices) > 0
        # With enable_thinking=False, reasoning_content should be empty
        assert response.choices[0].message.reasoning_content is None

    def test_stream_chat_completion_with_reasoning(self, model, api_client):
        """Test streaming with enable_thinking=True, reasoning_content should not be empty."""

        has_reasoning = False
        has_content = False
        with api_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0,
            stream=True,
            extra_body={
                "separate_reasoning": True,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        ) as stream:
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    if getattr(chunk.choices[0].delta, "reasoning_content", None):
                        has_reasoning = True
                    if chunk.choices[0].delta.content:
                        has_content = True

        assert has_reasoning, "The reasoning content is not included in the stream response"
        assert has_content, "The stream response does not contain normal content"

    def test_stream_chat_completion_without_reasoning(self, model, api_client):
        """Test streaming with enable_thinking=False, reasoning_content should be empty."""

        has_reasoning = False
        has_content = False
        with api_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0,
            stream=True,
            extra_body={
                "separate_reasoning": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ) as stream:
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    if getattr(chunk.choices[0].delta, "reasoning_content", None):
                        has_reasoning = True
                    if chunk.choices[0].delta.content:
                        has_content = True

        assert not has_reasoning, (
            "The reasoning content should not be included in the stream response"
        )
        assert has_content, "The stream response does not contain normal content"
