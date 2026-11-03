"""Sampling parameter pass-through tests for Response API.

Verifies that sampling parameters (temperature, top_p, top_k, min_p,
frequency_penalty, presence_penalty, repetition_penalty) are accepted
by the gateway and produce valid responses through gRPC backends.
"""

from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)


class _SamplingParamsBase:
    """Shared sampling parameter tests — subclass to set model/engine marks.

    On the mock engine each test states the answer first (``mock_answers``, a no-op on a
    real engine): what the tests show there is the gateway accepting the parameters and
    echoing them back; whether they reached the worker is the engine rows' verdict.
    """

    # The budget of the one-word answer. A reasoning model's class raises it: the analysis
    # channel comes before the word, and a turn the budget cuts reads incomplete.
    one_word_budget = 16

    def test_all_sampling_params_accepted(self, model, api_client, mock_answers):
        """Non-default sampling params produce a valid response with correct echo-back."""
        mock_answers.enqueue("2 + 2 = 4.")
        resp = api_client.responses.create(
            model=model,
            input="What is 2+2?",
            temperature=0.5,
            top_p=0.9,
            max_output_tokens=256,
            extra_body={
                "top_k": 40,
                "min_p": 0.05,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.3,
                "repetition_penalty": 1.1,
            },
        )
        assert resp.status == "completed"
        assert resp.error is None
        assert len(resp.output) > 0

        assert resp.temperature == pytest.approx(0.5)
        assert resp.top_p == pytest.approx(0.9, abs=1e-4)
        assert resp.max_output_tokens == 256

        assert resp.usage is not None
        tokens = resp.usage.output_tokens or resp.usage.completion_tokens
        assert tokens is not None
        assert tokens > 0

    def test_temperature_zero_is_deterministic(self, model, api_client, mock_answers):
        """temperature=0: the same prompt yields identical output on an engine; on the
        mock both answers are stated, so the pair shows two requests completing with the
        parameter accepted."""
        for _ in range(2):
            mock_answers.enqueue("Paris")
        kwargs = {
            "model": model,
            "input": "What is the capital of France? Answer in one word.",
            "temperature": 0,
            "max_output_tokens": self.one_word_budget,
        }
        resp1 = api_client.responses.create(**kwargs)
        resp2 = api_client.responses.create(**kwargs)

        assert resp1.status == "completed"
        assert resp2.status == "completed"
        assert resp1.output_text == resp2.output_text

    def test_sampling_params_streaming(self, model, api_client, mock_answers):
        """Sampling params work in streaming mode."""
        mock_answers.enqueue("Hello!")
        stream = api_client.responses.create(
            model=model,
            input="Say hello.",
            stream=True,
            temperature=0.5,
            top_p=0.9,
            max_output_tokens=256,
            extra_body={
                "top_k": 40,
                "min_p": 0.05,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.3,
                "repetition_penalty": 1.1,
            },
        )
        events = list(stream)
        completed = [e for e in events if e.type == "response.completed"]
        assert len(completed) == 1

        resp = completed[0].response
        assert resp.status == "completed"
        assert len(resp.output) > 0
        assert resp.temperature == pytest.approx(0.5)
        assert resp.top_p == pytest.approx(0.9, abs=1e-4)
        assert resp.usage is not None
        tokens = resp.usage.output_tokens or resp.usage.completion_tokens
        assert tokens is not None
        assert tokens > 0


@pytest.mark.engine("sglang", "mock")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("Qwen/Qwen2.5-14B-Instruct")
@pytest.mark.gateway(extra_args=["--tool-call-parser", "qwen", "--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestSamplingParamsLocal(_SamplingParamsBase):
    """Regular model (Qwen via SGLang)."""


@pytest.mark.engine("sglang", "vllm", "trtllm", "tokenspeed", "mock")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("openai/gpt-oss-20b", mock="openai/gpt-oss-20b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestSamplingParamsHarmony(_SamplingParamsBase):
    """Harmony model (gpt-oss-20b via all gRPC backends)."""

    one_word_budget = 256
