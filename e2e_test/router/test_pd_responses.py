"""Responses API tests for PD (Prefill-Decode) disaggregated gRPC routing.

/v1/responses rides the same mode-parameterized gRPC pipeline as chat
completions, so every create below exercises prefill/decode pair selection,
bootstrap injection, and dual dispatch.

Backends:
- "pd_http": HTTP mode (SGLang only - vLLM does not support HTTP)
- "pd_grpc": gRPC mode (both SGLang and vLLM)

Requirements:
    - SGLang: sgl_kernel package
    - vLLM: NIXL or Mooncake KV transfer support
    - GPUs: num_prefill + num_decode (default: 2 GPUs for 1+1)

Usage:
    pytest e2e_test/router/test_pd_responses.py -v

    # vLLM
    E2E_RUNTIME=vllm pytest e2e_test/router/test_pd_responses.py -v
"""

from __future__ import annotations

import logging

import openai
import pytest
import smg_client

logger = logging.getLogger(__name__)


@pytest.mark.skip(
    reason="SGLang's /v1/responses does not accept PD-disaggregated requests "
    "yet (bootstrap fields are not carried through to the scheduler). Unskip "
    "when the engine forwards them."
)
@pytest.mark.engine("sglang")
@pytest.mark.gpu(2)
@pytest.mark.model("meta-llama/Llama-3.1-8B-Instruct")
@pytest.mark.e2e
@pytest.mark.skip_for_runtime("vllm", reason="vLLM does not support HTTP mode")
@pytest.mark.parametrize("setup_backend", ["pd_http"], indirect=True)
class TestPDResponsesHttp:
    """Responses API through the HTTP PD router's dual dispatch.

    Creation and streaming only: in HTTP mode the gateway proxies to the
    engine's own /v1/responses, so storage semantics (retrieval, chaining)
    are the engine's and are not asserted here.
    """

    def test_basic_response_creation(self, model, api_client):
        """Test basic response creation."""
        resp = api_client.responses.create(model=model, input="What is 2+2?")

        assert resp.id is not None
        assert resp.error is None
        assert resp.status == "completed"
        assert len(resp.output_text) > 0
        assert resp.usage is not None

    def test_streaming_response(self, model, api_client):
        """Test streaming response."""
        resp = api_client.responses.create(
            model=model, input="Count to 5", stream=True, max_output_tokens=50
        )

        events = list(resp)
        created_events = [e for e in events if e.type == "response.created"]
        assert len(created_events) > 0

        delta_events = [e for e in events if e.type == "response.output_text.delta"]
        assert len(delta_events) > 0
        streamed_text = "".join(e.delta for e in delta_events)
        assert len(streamed_text) > 0

        completed_events = [e for e in events if e.type == "response.completed"]
        assert len(completed_events) == 1
        assert completed_events[0].response.status == "completed"


@pytest.mark.engine("sglang", "vllm")
@pytest.mark.gpu(2)
@pytest.mark.model("meta-llama/Llama-3.1-8B-Instruct")
@pytest.mark.e2e
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["pd_grpc"], indirect=True)
@pytest.mark.parametrize("api_client", ["openai", "smg"], indirect=True)
class TestPDResponsesGrpc:
    """Responses API tests using PD disaggregation (gRPC mode)."""

    def test_basic_response_creation(self, model, api_client):
        """Test basic response creation."""
        resp = api_client.responses.create(model=model, input="What is 2+2?")

        assert resp.id is not None
        assert resp.error is None
        assert resp.status == "completed"
        assert len(resp.output_text) > 0
        assert resp.usage is not None

    def test_streaming_response(self, model, api_client):
        """Test streaming response."""
        resp = api_client.responses.create(
            model=model, input="Count to 5", stream=True, max_output_tokens=50
        )

        events = list(resp)
        created_events = [e for e in events if e.type == "response.created"]
        assert len(created_events) > 0

        delta_events = [e for e in events if e.type == "response.output_text.delta"]
        assert len(delta_events) > 0
        streamed_text = "".join(e.delta for e in delta_events)
        assert len(streamed_text) > 0

        completed_events = [e for e in events if e.type == "response.completed"]
        assert len(completed_events) == 1
        assert completed_events[0].response.status == "completed"

    def test_previous_response_id_chaining(self, model, api_client):
        """Test chaining responses using previous_response_id."""
        # First response
        resp1 = api_client.responses.create(
            model=model, input="My name is Alice and my friend is Bob. Remember it."
        )
        assert resp1.error is None
        assert resp1.status == "completed"

        # Second response referencing first
        resp2 = api_client.responses.create(
            model=model, input="What is my name", previous_response_id=resp1.id
        )
        assert resp2.error is None
        assert resp2.status == "completed"
        assert "Alice" in resp2.output_text

        # Third response referencing second
        resp3 = api_client.responses.create(
            model=model,
            input="What is my friend name?",
            previous_response_id=resp2.id,
        )
        assert resp3.error is None
        assert resp3.status == "completed"
        assert "Bob" in resp3.output_text

    def test_store_false_not_retrievable(self, model, api_client):
        """Test that store=false responses cannot be retrieved."""
        resp = api_client.responses.create(model=model, input="Hello", store=False)
        assert resp.id is not None
        assert resp.status == "completed"

        with pytest.raises((openai.NotFoundError, smg_client.NotFoundError)):
            api_client.responses.retrieve(response_id=resp.id)
