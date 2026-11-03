"""Sampling default E2E tests for gRPC workers.

These tests run a real SGLang worker and a real gateway. Qwen's published
generation_config.json includes sampling defaults, which should be surfaced
through worker metadata and usable for regular omitted-sampling requests.

On the mock engine the worker reads the same defaults from the model files of
an ungated Qwen repository with the same generation config, and the chat
request's answer is stated first (``mock_answers``, a no-op on a real engine).
"""

from __future__ import annotations

import json

import httpx
import pytest

DEFAULT_SAMPLING_PARAMS_LABEL = "default_sampling_params_json"


def _worker_sampling_defaults(gateway) -> dict:
    response = httpx.get(f"{gateway.base_url}/workers", timeout=10)
    response.raise_for_status()
    workers = response.json()["workers"]
    for worker in workers:
        labels = worker.get("labels") or {}
        raw_defaults = labels.get(DEFAULT_SAMPLING_PARAMS_LABEL)
        if raw_defaults:
            return json.loads(raw_defaults)
    raise AssertionError(f"missing {DEFAULT_SAMPLING_PARAMS_LABEL} label")


@pytest.mark.engine("sglang", "mock")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("Qwen/Qwen2.5-7B-Instruct", mock="Qwen/Qwen2.5-14B-Instruct")
@pytest.mark.gateway(extra_args=["--tool-call-parser", "qwen", "--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestSamplingDefaultsGrpc:
    """Verify real generation-config sampling defaults flow through the gateway."""

    def test_generation_config_sampling_defaults_are_discovered(self, setup_backend):
        _, _, _, gateway = setup_backend

        defaults = _worker_sampling_defaults(gateway)

        assert defaults["temperature"] == pytest.approx(0.7)
        assert defaults["top_p"] == pytest.approx(0.8)
        assert defaults["top_k"] == 20
        assert defaults["repetition_penalty"] == pytest.approx(1.05)

    def test_omitted_chat_sampling_params_complete_with_discovered_defaults(
        self, setup_backend, api_client, mock_answers
    ):
        _, model, _, gateway = setup_backend
        defaults = _worker_sampling_defaults(gateway)
        assert defaults["top_k"] == 20

        mock_answers.enqueue("Nebula Latte")
        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Invent a short surreal product name for a futuristic cafe. "
                        "Only output the name."
                    ),
                }
            ],
            max_tokens=16,
        )

        assert response.choices
        assert (response.choices[0].message.content or "").strip()
