"""Native SystemOne through a regular HTTP gateway and real SGLang inference.

Run with E2E_ENGINE=sglang E2E_RUNTIME=sglang pytest
e2e_test/router/test_systemone.py -v. Uses the SGLang v0.5.21 native schema.
"""

from __future__ import annotations

import math

import httpx
import pytest

_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def _post(gateway, body: dict) -> httpx.Response:
    return httpx.post(f"{gateway.base_url}/v1/systemone", json=body, timeout=120.0)


def _assert_distribution(answer: dict) -> None:
    probabilities = list(answer["probabilities"].values())
    assert probabilities
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-6)
    assert math.isfinite(answer["confidence"])
    assert 0 <= answer["confidence"] <= 1


def _assert_prefill_usage(result: dict) -> None:
    assert result["usage"]["input_tokens"] > 0
    assert result["usage"]["output_tokens"] == 0


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model(_MODEL)
@pytest.mark.parametrize("setup_backend", ["http"], indirect=True)
class TestSystemOne:
    def test_native_question_ids_answers_criteria_and_prefill_usage(self, setup_backend):
        _, model, _, gateway = setup_backend
        levels = [
            {"label": "poor", "description": "Bad quality"},
            "excellent quality",
            ["outstanding quality", {"description": "Best possible"}],
        ]
        response = _post(
            gateway,
            {
                "model": model,
                "state": "The sky is blue. Two plus two is four. The review is excellent.",
                "questions": {
                    "z-arithmetic": {
                        "type": "noul",
                        "criteria": {
                            "true": "Two plus two equals four.",
                            "false": "Two plus two does not equal four.",
                        },
                    },
                    "a-sky-color": {
                        "type": "choice",
                        "instructions": "What color is the sky?",
                        "criteria": {"blue": None, "red": "The sky is red"},
                    },
                    "m-review-quality": {
                        "type": "score",
                        "instructions": "Rate the review quality.",
                        "criteria": levels,
                    },
                },
            },
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("application/json")
        result = response.json()
        assert result["model"] == model
        answers = result["answers"]
        assert set(answers) == {"z-arithmetic", "a-sky-color", "m-review-quality"}
        noul = answers["z-arithmetic"]
        assert noul["type"] == "noul"
        assert math.isfinite(noul["noul"])
        assert 0.5 < noul["noul"] <= 1
        choice = answers["a-sky-color"]
        assert choice["type"] == "choice"
        assert choice["choice"] == "blue"
        assert list(choice["probabilities"]) == ["blue", "red"]
        _assert_distribution(choice)
        score = answers["m-review-quality"]
        assert score["type"] == "score"
        assert list(score["probabilities"]) == ["0", "1", "2"]
        assert score["legend"] == {"0": levels[0], "1": levels[1], "2": levels[2]}
        assert math.isfinite(score["score"])
        assert 0 <= score["score"] <= 2
        assert score["score"] == pytest.approx(
            sum(int(level) * p for level, p in score["probabilities"].items()), abs=1e-6
        )
        _assert_distribution(score)
        for answer in answers.values():
            assert math.isfinite(answer["x_label_mass"])
            # Native mass sums exp(raw logprobs), so rounding can exceed one.
            assert 0 <= answer["x_label_mass"] <= 1 + 1e-6
        _assert_prefill_usage(result)

    @pytest.mark.parametrize(
        "state",
        [
            {"delivery": {"arrived": True, "condition": "intact"}, "missing": None},
            ["The delivery arrived.", {"condition": "The package is intact."}],
        ],
        ids=["object", "array"],
    )
    def test_structured_state_is_accepted(self, setup_backend, state):
        _, model, _, gateway = setup_backend
        response = _post(
            gateway,
            {
                "model": model,
                "state": state,
                "questions": {
                    "delivery-status": {
                        "type": "noul",
                        "instructions": "Did the delivery arrive intact?",
                    }
                },
            },
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert set(result["answers"]) == {"delivery-status"}
        answer = result["answers"]["delivery-status"]
        assert answer["type"] == "noul"
        assert math.isfinite(answer["noul"])
        assert 0.5 < answer["noul"] <= 1
        _assert_prefill_usage(result)

    def test_missing_model_is_a_client_error(self, setup_backend):
        _, _, _, gateway = setup_backend
        response = _post(
            gateway,
            {
                "state": "The sky is blue.",
                "questions": {"sky": {"type": "noul", "instructions": "Is the sky blue?"}},
            },
        )
        assert 400 <= response.status_code < 500, response.text

    def test_backend_rejects_unknown_nested_question_field(self, setup_backend):
        _, model, _, gateway = setup_backend
        response = _post(
            gateway,
            {
                "model": model,
                "state": "The sky is blue.",
                "questions": {
                    "sky": {
                        "type": "noul",
                        "instructions": "Is the sky blue?",
                        "native_extension": False,
                    }
                },
            },
        )
        assert response.status_code == 422, response.text
        # The native error identifies the forwarded extension. SGLang strips
        # invalid input values from validation errors before returning them.
        errors = response.json()["detail"]
        assert any(
            error["type"] == "extra_forbidden" and error["loc"][-1] == "native_extension"
            for error in errors
        ), errors
