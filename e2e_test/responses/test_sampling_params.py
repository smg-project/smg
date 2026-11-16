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
    """Regular model (Qwen via SGLang).

    The two ``stop`` rows read the gateway's own trim of the matched sequence; the Harmony
    class has its own, where the sequence stays as the suffix of the output.
    """

    def test_stop_sequence_ends_the_answer(self, model, api_client, mock_answers):
        """A ``stop`` sequence on a plain Responses request ends the answer and is
        trimmed from the output, as it is on the chat path. On the mock the engine's
        part is stated (the answer ends on the matched stop, as an engine reports it);
        what the row reads is the gateway's own trim of the sequence, which is built
        from the request the gateway forwards: a dropped ``stop`` leaves the sequence
        in the text."""
        mock_answers.enqueue("1,", finish_reason="stop", matched_stop=",")
        resp = api_client.responses.create(
            model=model,
            input="Count from 1 to 10: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
            temperature=0,
            max_output_tokens=200,
            extra_body={"stop": ","},
        )
        assert resp.status == "completed"
        assert resp.output_text, "the text before the stop sequence is the answer"
        assert "," not in resp.output_text, (
            f"the stop sequence ',' should not appear in the output: {resp.output_text!r}"
        )

    def test_stop_sequence_ends_the_streamed_answer(self, model, api_client, mock_answers):
        """The same ``stop`` sequence, in its list form, on a streamed request: neither
        the text deltas nor the terminal event's object carry the sequence."""
        mock_answers.enqueue("1,", finish_reason="stop", matched_stop=",")
        stream = api_client.responses.create(
            model=model,
            input="Count from 1 to 10: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
            temperature=0,
            max_output_tokens=200,
            stream=True,
            extra_body={"stop": [","]},
        )
        events = list(stream)
        completed = [e for e in events if e.type == "response.completed"]
        assert len(completed) == 1

        text = "".join(e.delta for e in events if e.type == "response.output_text.delta")
        assert text, "the text before the stop sequence is the answer"
        assert "," not in text, f"the stop sequence ',' should not appear in the deltas: {text!r}"
        resp = completed[0].response
        assert resp.status == "completed"
        assert "," not in resp.output_text, (
            f"the stop sequence ',' should not appear in the terminal object: {resp.output_text!r}"
        )


@pytest.mark.engine("sglang", "vllm", "trtllm", "tokenspeed", "mock")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("openai/gpt-oss-20b", mock="openai/gpt-oss-20b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestSamplingParamsHarmony(_SamplingParamsBase):
    """Harmony model (gpt-oss-20b via all gRPC backends).

    The two ``stop`` rows read the Harmony shape of a stopped turn: the engine ends the
    output at the sequence and the sequence stays as the suffix (the Harmony
    detokenization is not channel-aware, as the chat rows of this model read it). On the
    mock the engine's part is stated, so the rows read the shape; on an engine the
    forwarded ``stop`` is what ends the answer, and a dropped one runs on to the count's
    end.
    """

    one_word_budget = 256

    @staticmethod
    def _answer_text(resp) -> str:
        """The turn's text: the message's, or the reasoning's when the stop ended the turn
        inside the analysis channel (the chat rows of this model read it the same way)."""
        if resp.output_text:
            return resp.output_text
        for item in resp.output:
            if item.type == "reasoning":
                return "".join(getattr(part, "text", "") or "" for part in item.content or [])
        return ""

    def test_stop_sequence_ends_the_answer(self, model, api_client, mock_answers):
        """A ``stop`` sequence on a Harmony Responses request ends the answer on the
        sequence, as it does on the chat path of this model."""
        mock_answers.enqueue("1,", finish_reason="stop", matched_stop=",")
        resp = api_client.responses.create(
            model=model,
            input="Count from 1 to 10: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
            temperature=0,
            max_output_tokens=200,
            extra_body={"stop": ","},
        )
        assert resp.status == "completed"
        text = self._answer_text(resp)
        assert text, "the text up to the stop sequence is the answer"
        assert text.endswith(","), (
            f"the stop sequence ',' should be the suffix of the output: {text!r}"
        )
        assert "2" not in text, f"the answer ran past the stop sequence: {text!r}"

    def test_stop_sequence_ends_the_streamed_answer(self, model, api_client, mock_answers):
        """The same ``stop`` sequence, in its list form, on a streamed request: the text
        deltas end on the sequence and so does the terminal event's object. A turn the stop
        ended inside the analysis channel streams no text on this pipeline (its reasoning
        item carries none), so the text that did stream is what the row reads."""
        mock_answers.enqueue("1,", finish_reason="stop", matched_stop=",")
        stream = api_client.responses.create(
            model=model,
            input="Count from 1 to 10: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
            temperature=0,
            max_output_tokens=200,
            stream=True,
            extra_body={"stop": [","]},
        )
        events = list(stream)
        completed = [e for e in events if e.type == "response.completed"]
        assert len(completed) == 1

        text = "".join(e.delta for e in events if e.type == "response.output_text.delta")
        resp = completed[0].response
        assert resp.status == "completed"
        assert resp.output_text == text, (
            f"the terminal object's text is the streamed text: {resp.output_text!r} vs {text!r}"
        )
        if text:
            assert text.endswith(","), (
                f"the stop sequence ',' should be the suffix of the streamed text: {text!r}"
            )
        assert "2" not in text, f"the answer ran past the stop sequence: {text!r}"
