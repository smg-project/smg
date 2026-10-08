"""OpenAI Decisions through regular HTTP and gRPC SGLang workers.

Uses raw HTTP because the installed OpenAI SDK can predate Decisions. SGLang
scores candidate labels during prefill, so successful responses have zero
output usage. Run with E2E_ENGINE=sglang E2E_RUNTIME=sglang pytest
e2e_test/router/test_decisions.py -v. Set E2E_SGLANG_SERVICER_IMPL=rust to
exercise the Rust gRPC servicer; the default uses the Python servicer.
"""

from __future__ import annotations

import math

import httpx
import pytest

_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def _post(gateway, body: dict) -> httpx.Response:
    return httpx.post(f"{gateway.base_url}/v1/decisions", json=body, timeout=120.0)


def _assert_distribution(answer: dict) -> None:
    probabilities = [item["probability"] for item in answer["probabilities"]]
    assert probabilities
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-6)
    assert math.isfinite(answer["confidence"])
    assert 0 <= answer["confidence"] <= 1


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model(_MODEL)
@pytest.mark.parametrize("setup_backend", ["http", "grpc"], indirect=True)
class TestDecisions:
    def test_all_question_types_order_names_and_prefill_usage(self, setup_backend):
        _, model, _, gateway = setup_backend
        response = _post(
            gateway,
            {
                "model": model,
                "input": "The sky is blue. Two plus two is four. The review is excellent.",
                "questions": [
                    {"type": "predicate", "instructions": "Is two plus two four?"},
                    {
                        "type": "choice",
                        "name": "repeated",
                        "instructions": "What color is the sky?",
                        "choices": [{"value": "blue"}, {"value": "red"}],
                    },
                    {
                        "type": "score",
                        "name": "repeated",
                        "instructions": "Rate the review quality.",
                        "levels": [
                            {"label": "poor", "description": "Bad quality"},
                            {"label": "excellent", "description": "Very good quality"},
                        ],
                    },
                    {
                        "type": "choice",
                        "instructions": "Is the sky blue?",
                        "choices": [
                            {"value": True, "description": "Yes, the sky is blue"},
                            {"value": False, "description": "No, it is another color"},
                        ],
                    },
                ],
            },
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("application/json")
        result = response.json()
        answers = result["answers"]
        assert [a["type"] for a in answers] == ["predicate", "choice", "score", "choice"]
        assert [a["name"] for a in answers] == [None, "repeated", "repeated", None]
        assert 0.5 < answers[0]["probability"] <= 1
        assert answers[1]["choice"] == "blue"
        assert [p["value"] for p in answers[1]["probabilities"]] == ["blue", "red"]
        assert [p["value"] for p in answers[2]["probabilities"]] == [0, 1]
        assert [p["label"] for p in answers[2]["probabilities"]] == ["poor", "excellent"]
        # Scores are expectations over level indices, not the winning level.
        assert answers[2]["score"] == pytest.approx(
            sum(p["value"] * p["probability"] for p in answers[2]["probabilities"]),
            abs=1e-6,
        )
        assert type(answers[3]["choice"]) is bool
        assert [p["value"] for p in answers[3]["probabilities"]] == [True, False]
        assert all(type(p["value"]) is bool for p in answers[3]["probabilities"])
        for answer in answers[1:]:
            _assert_distribution(answer)
        usage = result["usage"]
        assert usage["input_tokens"] > 0
        assert usage["output_tokens"] == 0
        assert usage["total_tokens"] == usage["input_tokens"]
        assert usage["output_tokens_details"]["reasoning_tokens"] == 0
        assert set(usage["input_tokens_details"]) >= {"cached_tokens", "cache_write_tokens"}

    def test_ordered_user_messages_are_accepted(self, setup_backend):
        _, model, _, gateway = setup_backend
        response = _post(
            gateway,
            {
                "model": model,
                "input": [
                    {"role": "user", "content": "The delivery arrived."},
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "The package is intact."}],
                    },
                ],
                "questions": [
                    {"type": "predicate", "instructions": "Did the delivery arrive intact?"}
                ],
            },
        )
        assert response.status_code == 200, response.text
        answer = response.json()["answers"][0]
        assert answer["type"] == "predicate"
        assert answer["name"] is None
        assert 0.5 < answer["probability"] <= 1

    @pytest.mark.parametrize(
        "invalid", ["missing_model", "unknown_type", "numeric_choice", "image"]
    )
    def test_invalid_or_unsupported_requests_are_client_errors(self, setup_backend, invalid):
        _, model, _, gateway = setup_backend
        body = {
            "model": model,
            "input": "A short text.",
            "questions": [{"type": "predicate", "instructions": "Is this text?"}],
        }
        if invalid == "missing_model":
            del body["model"]
        elif invalid == "unknown_type":
            body["questions"][0]["type"] = "unknown"
        elif invalid == "numeric_choice":
            body["questions"] = [
                {"type": "choice", "instructions": "Choose", "choices": [{"value": 1}]}
            ]
        else:
            # Valid OpenAI image input; SGLang's pinned text-only adapter rejects it.
            body["input"] = [
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}],
                }
            ]
        response = _post(gateway, body)
        assert 400 <= response.status_code < 500, response.text
        if invalid == "image":
            assert response.status_code == 400, response.text
            assert isinstance(response.json()["error"]["message"], str)
