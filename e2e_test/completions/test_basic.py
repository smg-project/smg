"""Basic tests for OpenAI Completions API (/v1/completions).

Tests for non-streaming and streaming text completion, echo, suffix,
stop sequences, and parallel sampling via the OpenAI SDK.
"""

from __future__ import annotations

import pytest


@pytest.mark.engine("sglang", "vllm", "tokenspeed")
@pytest.mark.gpu(1)
@pytest.mark.model("meta-llama/Llama-3.1-8B-Instruct")
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestCompletionBasic:
    """Tests for OpenAI-compatible /v1/completions API (non-streaming)."""

    STOP_SEQUENCE_TRIMMED = True

    def test_non_streaming_basic(self, model, api_client):
        """Test basic non-streaming text completion with response structure."""

        response = api_client.completions.create(
            model=model,
            prompt="The capital of France is",
            max_tokens=20,
            temperature=0,
        )

        assert response.id is not None
        assert response.object == "text_completion"
        assert response.model is not None
        assert response.created is not None
        assert len(response.choices) == 1

        choice = response.choices[0]
        assert choice.index == 0
        assert isinstance(choice.text, str)
        assert len(choice.text) > 0
        assert choice.finish_reason in ("stop", "length")

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )

    def test_non_streaming_max_tokens(self, model, api_client):
        """Test that max_tokens limits output length."""

        response = api_client.completions.create(
            model=model,
            prompt="Count from 1 to 100: 1, 2, 3,",
            max_tokens=5,
            temperature=0,
        )

        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "length"
        assert response.usage.completion_tokens <= 5

    def test_non_streaming_stop_sequence(self, model, api_client):
        """Test that stop sequences cause the model to stop generating."""

        response = api_client.completions.create(
            model=model,
            prompt="Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
            max_tokens=200,
            temperature=0,
            stop=[","],
        )

        assert response.choices[0].finish_reason == "stop"
        text = response.choices[0].text
        if self.STOP_SEQUENCE_TRIMMED:
            assert "," not in text, f"Stop sequence ',' should not appear in output: {text}"
        else:
            assert text.endswith(","), f"Stop sequence ',' should be the suffix of output: {text}"

    def test_non_streaming_echo(self, model, api_client):
        """Test that echo=True prepends the prompt to the output."""

        prompt = "The capital of France is"
        response = api_client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=20,
            temperature=0,
            echo=True,
        )

        assert response.choices[0].text.startswith(prompt)

    def test_non_streaming_suffix(self, model, api_client):
        """Test that suffix is appended to the output."""

        suffix = " -- END"
        response = api_client.completions.create(
            model=model,
            prompt="The capital of France is",
            max_tokens=20,
            temperature=0,
            suffix=suffix,
        )

        assert response.choices[0].text.endswith(suffix)

    @pytest.mark.parametrize("n", [1, 2])
    def test_non_streaming_parallel_sampling(self, model, api_client, n):
        """Test parallel sampling with n > 1."""

        temperature = 0.7 if n > 1 else 0
        response = api_client.completions.create(
            model=model,
            prompt="The meaning of life is",
            max_tokens=30,
            temperature=temperature,
            n=n,
        )

        assert len(response.choices) == n
        for i, choice in enumerate(response.choices):
            assert choice.index == i
            assert isinstance(choice.text, str)
            assert len(choice.text) > 0

    def test_non_streaming_usage(self, model, api_client):
        """Test that usage statistics are returned correctly."""

        response = api_client.completions.create(
            model=model,
            prompt="Hello",
            max_tokens=10,
            temperature=0,
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )

    def test_rid_becomes_response_id(self, model, api_client):
        """A protocol `rid` is used as the backend request id verbatim."""

        response = api_client.completions.create(
            model=model,
            prompt="Hello",
            max_tokens=5,
            temperature=0,
            extra_body={"rid": "cmpl-correlation-1"},
        )

        assert response.id == "cmpl-correlation-1"

    @pytest.mark.skip_for_runtime("vllm", reason="vLLM rejects max_tokens=0")
    def test_non_streaming_echo_max_tokens_zero(self, model, api_client):
        """Test that echo=True with max_tokens=0 returns just the prompt."""

        prompt = "The capital of France is"
        response = api_client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=0,
            temperature=0,
            echo=True,
        )

        assert response.choices[0].text == prompt
        assert response.choices[0].finish_reason in ("stop", "length")
        assert response.usage.completion_tokens == 0


@pytest.mark.engine("sglang", "vllm", "tokenspeed")
@pytest.mark.gpu(1)
@pytest.mark.model("meta-llama/Llama-3.1-8B-Instruct")
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestCompletionStreaming:
    """Tests for streaming /v1/completions API."""

    STOP_SEQUENCE_TRIMMED = True

    @staticmethod
    def _collect_stream(stream):
        """Consume a streaming response, returning (full_text, finish_reasons)."""
        texts = []
        finish_reasons = []
        for chunk in stream:
            assert chunk.object == "text_completion"
            if chunk.choices:
                choice = chunk.choices[0]
                if choice.text:
                    texts.append(choice.text)
                if choice.finish_reason:
                    finish_reasons.append(choice.finish_reason)
        return "".join(texts), finish_reasons

    def test_streaming_basic(self, model, api_client):
        """Test streaming completion returns chunks with text deltas."""

        stream = api_client.completions.create(
            model=model,
            prompt="The capital of France is",
            max_tokens=20,
            temperature=0,
            stream=True,
        )

        full_text, finish_reasons = self._collect_stream(stream)

        assert len(full_text) > 0, "No text chunks received"
        assert len(finish_reasons) == 1
        assert finish_reasons[0] in ("stop", "length")

    def test_streaming_stop_sequence(self, model, api_client):
        """Test that stop sequences work in streaming mode."""

        stream = api_client.completions.create(
            model=model,
            prompt="Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
            max_tokens=200,
            temperature=0,
            stop=[","],
            stream=True,
        )

        full_text, finish_reasons = self._collect_stream(stream)

        assert len(finish_reasons) == 1
        assert finish_reasons[0] == "stop"
        if self.STOP_SEQUENCE_TRIMMED:
            assert "," not in full_text, (
                f"Stop sequence ',' should not appear in output: {full_text}"
            )
        else:
            assert full_text.endswith(","), (
                f"Stop sequence ',' should be the suffix of output: {full_text}"
            )

    def test_streaming_collects_full_text(self, model, api_client):
        """Test that streaming deltas concatenate to a non-empty completion."""

        stream = api_client.completions.create(
            model=model,
            prompt="The capital of France is",
            max_tokens=20,
            temperature=0,
            stream=True,
        )

        full_text, _ = self._collect_stream(stream)

        assert len(full_text) > 0

    @pytest.mark.skip_for_runtime("vllm", reason="vLLM rejects max_tokens=0")
    def test_streaming_echo_max_tokens_zero(self, model, api_client):
        """Test that echo=True with max_tokens=0 streams just the prompt."""

        prompt = "The capital of France is"
        stream = api_client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=0,
            temperature=0,
            echo=True,
            stream=True,
        )

        full_text, finish_reasons = self._collect_stream(stream)

        assert full_text == prompt, f"Expected echoed prompt, got: {full_text!r}"
        assert len(finish_reasons) == 1
        assert finish_reasons[0] in ("stop", "length")


@pytest.mark.engine("sglang", "vllm", "tokenspeed")
@pytest.mark.gpu(1)
@pytest.mark.model("meta-llama/Llama-3.1-8B-Instruct")
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestCompletionBatch:
    """Tests for batched prompt arrays on /v1/completions."""

    PROMPTS = ["The capital of France is", "The capital of Germany is"]

    @staticmethod
    def _collect_stream_by_index(stream):
        """Consume a stream, returning per-index text, finish reasons, usage, and usage-chunk count."""
        texts = {}
        finish_reasons = {}
        usage = None
        usage_chunks = 0
        for chunk in stream:
            assert chunk.object == "text_completion"
            if chunk.usage is not None:
                usage = chunk.usage
                usage_chunks += 1
            for choice in chunk.choices:
                if choice.text:
                    texts.setdefault(choice.index, []).append(choice.text)
                if choice.finish_reason:
                    finish_reasons.setdefault(choice.index, []).append(choice.finish_reason)
        return (
            {index: "".join(parts) for index, parts in texts.items()},
            finish_reasons,
            usage,
            usage_chunks,
        )

    def test_batch_non_streaming(self, model, api_client):
        """Test that a prompt array returns one choice per prompt with global indices."""

        response = api_client.completions.create(
            model=model,
            prompt=self.PROMPTS,
            max_tokens=20,
            temperature=0,
        )

        assert len(response.choices) == len(self.PROMPTS)
        assert sorted(choice.index for choice in response.choices) == [0, 1]
        for choice in response.choices:
            assert isinstance(choice.text, str)
            assert len(choice.text) > 0
            assert choice.finish_reason in ("stop", "length")

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )

    def test_batch_non_streaming_with_n(self, model, api_client):
        """Test prompt-major global indices with n > 1: index = prompt_index * n + i."""

        n = 2
        response = api_client.completions.create(
            model=model,
            prompt=self.PROMPTS,
            max_tokens=20,
            temperature=0.7,
            n=n,
            echo=True,
        )

        assert len(response.choices) == len(self.PROMPTS) * n
        assert [choice.index for choice in response.choices] == [0, 1, 2, 3]
        for choice in response.choices:
            assert choice.text.startswith(self.PROMPTS[choice.index // n])

    def test_batch_echo_maps_prompts_to_choices(self, model, api_client):
        """Test that echo=True prepends each prompt to its own choice."""

        response = api_client.completions.create(
            model=model,
            prompt=self.PROMPTS,
            max_tokens=10,
            temperature=0,
            echo=True,
        )

        choices = {choice.index: choice for choice in response.choices}
        assert len(choices) == len(self.PROMPTS)
        for prompt_index, prompt in enumerate(self.PROMPTS):
            assert choices[prompt_index].text.startswith(prompt)

    def test_batch_rid_is_shared_response_id(self, model, api_client):
        """A protocol `rid` becomes the shared response id for a prompt array."""

        response = api_client.completions.create(
            model=model,
            prompt=self.PROMPTS,
            max_tokens=5,
            temperature=0,
            extra_body={"rid": "batch-correlation-1"},
        )

        assert response.id == "batch-correlation-1"
        assert sorted(choice.index for choice in response.choices) == [0, 1]

    def test_batch_streaming(self, model, api_client):
        """Test streaming with a prompt array and n > 1: global per-choice deltas
        and exactly one aggregated usage chunk."""

        n = 2
        stream = api_client.completions.create(
            model=model,
            prompt=self.PROMPTS,
            max_tokens=20,
            temperature=0.7,
            n=n,
            stream=True,
            stream_options={"include_usage": True},
        )

        texts, finish_reasons, usage, usage_chunks = self._collect_stream_by_index(stream)

        expected_indices = list(range(len(self.PROMPTS) * n))
        assert sorted(texts) == expected_indices, (
            f"Expected deltas for all choices, got {sorted(texts)}"
        )
        assert sorted(finish_reasons) == expected_indices
        for index in expected_indices:
            assert len(texts[index]) > 0
            assert finish_reasons[index] in (["stop"], ["length"])

        assert usage_chunks == 1, f"Expected exactly one usage chunk, got {usage_chunks}"
        assert usage is not None
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
