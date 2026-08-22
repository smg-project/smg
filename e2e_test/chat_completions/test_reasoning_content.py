"""Reasoning Content E2E Tests.

Tests for chat completions with reasoning content (DeepSeek R1 reasoning parser).

Source: Migrated from e2e_grpc/features/test_reasoning_content.py
"""

from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)


# =============================================================================
# Reasoning Content API Tests (DeepSeek 7B)
# =============================================================================


@pytest.mark.engine("sglang", "vllm", "trtllm")
@pytest.mark.gpu(1)
@pytest.mark.model("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
@pytest.mark.gateway(
    extra_args=["--reasoning-parser", "deepseek_r1", "--history-backend", "memory"]
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
@pytest.mark.parametrize("api_client", ["openai", "smg"], indirect=True)
class TestReasoningContentAPI:
    """Tests for reasoning content API with DeepSeek R1 reasoning parser."""

    def test_streaming_separate_reasoning_false(self, model, api_client):
        """Test streaming with separate_reasoning=False, reasoning_content should be empty."""

        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            max_tokens=100,
            stream=True,
            extra_body={"separate_reasoning": False},
        )

        reasoning_content = ""
        content = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            elif chunk.choices[0].delta.reasoning_content:
                reasoning_content += chunk.choices[0].delta.reasoning_content

        assert len(reasoning_content) == 0
        assert len(content) > 0

    def test_streaming_separate_reasoning_true(self, model, api_client):
        """Test streaming with separate_reasoning=True, reasoning_content should not be empty."""

        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            max_tokens=100,
            stream=True,
            extra_body={"separate_reasoning": True},
        )

        reasoning_content = ""
        content = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            elif chunk.choices[0].delta.reasoning_content:
                reasoning_content += chunk.choices[0].delta.reasoning_content

        assert len(reasoning_content) > 0
        assert len(content) > 0

    def test_streaming_separate_reasoning_true_stream_reasoning_false(self, model, api_client):
        """Test streaming with separate_reasoning=True and stream_reasoning=False."""

        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            max_tokens=100,
            stream=True,
            extra_body={"separate_reasoning": True, "stream_reasoning": False},
        )

        reasoning_content = ""
        content = ""
        first_chunk = False
        for chunk in response:
            if chunk.choices[0].delta.reasoning_content:
                reasoning_content = chunk.choices[0].delta.reasoning_content
                first_chunk = True
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
                if not first_chunk:
                    reasoning_content = chunk.choices[0].delta.reasoning_content
                first_chunk = True
            if not first_chunk:
                assert (
                    not chunk.choices[0].delta.reasoning_content
                    or len(chunk.choices[0].delta.reasoning_content) == 0
                )

        assert len(reasoning_content) > 0
        assert len(content) > 0

    @pytest.mark.parametrize("stop", [["wtf"], ["wtfx"], ["0123456789"]])
    def test_streaming_unmatched_stop_word_does_not_change_output(self, model, api_client, stop):
        """A `stop` word that never fires must not change what the client sees.

        The stop decoder withholds text that could still complete a stop
        sequence. When generation is cut short — here by `max_tokens`, before
        the model ever emits the stop word — whatever is still held is released
        at end of stream. That release used to skip the reasoning parser, so the
        held bytes surfaced as assistant `content`: a tail of the reasoning
        text, or a fragment of the model's own structural tokens. The leak was
        exactly as long as the stop word, which is what the parametrize covers.
        """

        def run(stop_words):
            kwargs = {"stop": stop_words} if stop_words else {}
            response = api_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "What is 1+3?"}],
                # Cut generation off early so the stream ends mid-reasoning,
                # with text still held back by the stop decoder.
                max_tokens=24,
                temperature=0,
                stream=True,
                extra_body={"separate_reasoning": True},
                **kwargs,
            )
            reasoning, content = "", ""
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    content += delta.content
                if delta.reasoning_content:
                    reasoning += delta.reasoning_content
            return reasoning, content

        baseline_reasoning, baseline_content = run(None)
        stopped_reasoning, stopped_content = run(stop)

        assert stopped_content == baseline_content, (
            f"stop={stop!r} leaked {stopped_content[len(baseline_content) :]!r} "
            "into content; text held by the stop decoder must still be parsed"
        )
        assert stopped_reasoning == baseline_reasoning, (
            f"stop={stop!r} changed reasoning_content; a stop word that never "
            "matches must not affect the reasoning/content split"
        )

    def test_nonstreaming_separate_reasoning_false(self, model, api_client):
        """Test non-streaming with separate_reasoning=False, reasoning_content should be empty."""

        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            max_tokens=100,
            extra_body={"separate_reasoning": False},
        )

        assert (
            not response.choices[0].message.reasoning_content
            or len(response.choices[0].message.reasoning_content) == 0
        )
        assert len(response.choices[0].message.content) > 0

    def test_nonstreaming_separate_reasoning_true(self, model, api_client):
        """Test non-streaming with separate_reasoning=True, reasoning_content should not be empty."""

        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            max_tokens=100,
            extra_body={"separate_reasoning": True},
        )

        assert len(response.choices[0].message.reasoning_content) > 0
        assert len(response.choices[0].message.content) > 0
