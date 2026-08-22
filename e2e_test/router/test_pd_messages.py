"""Anthropic Messages API tests for PD (Prefill-Decode) disaggregated routing.

/v1/messages enters PD dual dispatch like chat completions: the HTTP PD
router forwards the request to both legs with bootstrap fields injected,
and the gRPC PD router runs the mode-parameterized Messages pipeline. Every
create below exercises prefill/decode pair selection and dual dispatch.

Backends:
- "pd_http": HTTP mode (SGLang only - vLLM does not support HTTP)
- "pd_grpc": gRPC mode (both SGLang and vLLM)

Requirements:
    - SGLang: sgl_kernel package
    - vLLM: NIXL or Mooncake KV transfer support
    - GPUs: num_prefill + num_decode (default: 2 GPUs for 1+1)

Usage:
    # SGLang (runs both HTTP and gRPC)
    pytest e2e_test/router/test_pd_messages.py -v

    # vLLM (runs gRPC only, HTTP skipped)
    E2E_RUNTIME=vllm pytest e2e_test/router/test_pd_messages.py -v
"""

from __future__ import annotations

import logging

import anthropic
import pytest

logger = logging.getLogger(__name__)


@pytest.fixture
def anthropic_client(setup_backend):
    """Anthropic SDK client pointed at the gateway under test."""
    _, _, _, gateway = setup_backend
    client = anthropic.Anthropic(base_url=gateway.base_url, api_key="not-used")
    yield client
    client.close()


class MessagesOverPD:
    """Shared test bodies; subclasses pin the backend mode via markers."""

    def test_non_streaming_message(self, model, anthropic_client):
        """Basic message creation through PD dual dispatch."""
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=64,
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
        )

        assert response.id is not None
        assert response.role == "assistant"
        assert response.content is not None
        assert len(response.content) > 0
        assert response.content[0].type == "text"
        assert len(response.content[0].text) > 0
        assert response.usage is not None
        assert response.usage.output_tokens > 0

    def test_streaming_message(self, model, anthropic_client):
        """Streaming events arrive and deltas concatenate to a full message."""
        expected_event_types = {
            "message_start",
            "content_block_delta",
            "message_stop",
        }

        with anthropic_client.messages.stream(
            model=model,
            max_tokens=64,
            messages=[{"role": "user", "content": "Count from 1 to 3."}],
        ) as stream:
            event_types = set()
            for event in stream:
                event_types.add(event.type)
            full_text = stream.get_final_text()

        missing = expected_event_types - event_types
        assert not missing, f"Missing expected event types: {missing}"
        assert len(full_text) > 0


@pytest.mark.skip(
    reason="SGLang's /v1/messages does not carry PD bootstrap fields through "
    "to the scheduler yet: the decode leg rejects with 'Disaggregated request "
    "received without bootstrap room id'. Unskip when the engine forwards them."
)
@pytest.mark.engine("sglang")
@pytest.mark.gpu(2)
@pytest.mark.model("meta-llama/Llama-3.1-8B-Instruct")
@pytest.mark.e2e
@pytest.mark.skip_for_runtime("vllm", reason="vLLM does not support HTTP mode")
@pytest.mark.parametrize("setup_backend", ["pd_http"], indirect=True)
class TestPDMessagesHttp(MessagesOverPD):
    """Messages API through the HTTP PD router's dual dispatch."""


@pytest.mark.engine("sglang", "vllm")
@pytest.mark.gpu(2)
@pytest.mark.model("meta-llama/Llama-3.1-8B-Instruct")
@pytest.mark.e2e
@pytest.mark.parametrize("setup_backend", ["pd_grpc"], indirect=True)
class TestPDMessagesGrpc(MessagesOverPD):
    """Messages API through the gRPC PD pipeline."""
