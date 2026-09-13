"""State management tests for Response API.

Tests both previous_response_id and conversation-based state management.
These tests work across local (gRPC) and cloud (OpenAI, xAI) backends.

Source: Migrated from e2e_response_api/features/test_state_management.py
"""

from __future__ import annotations

import logging

import openai
import pytest
import smg_client

logger = logging.getLogger(__name__)


# =============================================================================
# Cloud Backend Tests (base mixin — no parametrize, subclasses add their own)
# =============================================================================


class _StateManagementCloudBase:
    """Base test methods for state management against cloud APIs."""

    def test_basic_response_creation(self, model, api_client):
        """Test basic response creation without state."""
        resp = api_client.responses.create(model=model, input="What is 2+2?")

        assert resp.id is not None
        assert resp.error is None
        assert resp.status == "completed"
        assert len(resp.output_text) > 0
        assert resp.usage is not None
        assert resp.usage.total_tokens == resp.usage.input_tokens + resp.usage.output_tokens

    def test_streaming_response(self, model, api_client):
        """Test streaming response."""
        resp = api_client.responses.create(
            model=model, input="Count to 5", stream=True, max_output_tokens=50
        )

        events = list(resp)
        created_events = [e for e in events if e.type == "response.created"]
        assert len(created_events) > 0

        assert any(e.type in ["response.completed", "response.in_progress"] for e in events)

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

    def test_conversation_with_multiple_turns(self, model, api_client):
        """Test state management using conversation ID."""

        # Create conversation
        conv_resp = api_client.conversations.create(metadata={"topic": "math"})
        assert conv_resp.id is not None
        assert conv_resp.created_at is not None
        conversation_id = conv_resp.id

        # First response in conversation
        resp1 = api_client.responses.create(
            model=model, input="I have 5 apples.", conversation=conversation_id
        )
        assert resp1.error is None
        assert resp1.status == "completed"

        # Second response in same conversation
        resp2 = api_client.responses.create(
            model=model,
            input="How many apples do I have?",
            conversation=conversation_id,
        )
        assert resp2.error is None
        assert resp2.status == "completed"
        assert "5" in resp2.output_text or "five" in resp2.output_text.lower()

        # Third response in same conversation
        resp3 = api_client.responses.create(
            model=model,
            input="If I get 3 more, how many total?",
            conversation=conversation_id,
        )
        assert resp3.error is None
        assert resp3.status == "completed"
        assert "8" in resp3.output_text or "eight" in resp3.output_text.lower()

        items = api_client.conversations.items.list(conversation_id)
        assert items.data is not None
        assert len(items.data) >= 6  # 3 inputs + 3 outputs

        # SmgClient: conversations API not supported, skipping comparison

    def test_previous_response_id_invalid(self, model, api_client):
        """Test using invalid previous_response_id."""
        with pytest.raises((openai.BadRequestError, smg_client.BadRequestError)):
            api_client.responses.create(
                model=model,
                input="Test",
                previous_response_id="resp_invalid123",
                max_output_tokens=50,
            )

    def test_store_false_not_retrievable(self, model, api_client):
        """Test that store=false responses cannot be retrieved."""
        resp = api_client.responses.create(model=model, input="Hello", store=False)
        assert resp.id is not None
        assert resp.status == "completed"

        with pytest.raises((openai.NotFoundError, smg_client.NotFoundError)):
            api_client.responses.retrieve(response_id=resp.id)

    def test_store_false_previous_response_id_rejected(self, model, api_client):
        """Test that store=false response id cannot be used as previous_response_id."""
        resp = api_client.responses.create(model=model, input="Hello", store=False)
        assert resp.id is not None

        with pytest.raises((openai.BadRequestError, smg_client.BadRequestError)):
            api_client.responses.create(
                model=model,
                input="Follow up",
                previous_response_id=resp.id,
            )

    def test_store_false_conversation_no_items(self, model, api_client):
        """Test that store=false with conversation does not persist items."""
        conv = api_client.conversations.create(metadata={"test": "store_false"})

        resp = api_client.responses.create(
            model=model, input="Hello", store=False, conversation=conv.id
        )
        assert resp.status == "completed"

        items = api_client.conversations.items.list(conv.id)
        assert len(items.data) == 0

    def test_mutually_exclusive_parameters(self, model, api_client):
        """Test that previous_response_id and conversation are mutually exclusive."""
        conversation_id = "conv_123"
        resp1 = api_client.responses.create(model=model, input="Test")

        with pytest.raises((openai.BadRequestError, smg_client.BadRequestError)):
            api_client.responses.create(
                model=model,
                input="This should fail",
                previous_response_id=resp1.id,
                conversation=conversation_id,
            )


# =============================================================================
# Cloud Backend Tests (OpenAI)
# =============================================================================


@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestStateManagementCloud(_StateManagementCloudBase):
    """State management tests against OpenAI cloud API."""


# =============================================================================
# Cloud Backend Tests (xAI)
# =============================================================================


@pytest.mark.vendor("xai")
@pytest.mark.gpu(0)
@pytest.mark.parametrize("setup_backend", ["xai"], indirect=True)
class TestStateManagementCloudXai(_StateManagementCloudBase):
    """State management tests against xAI cloud API."""


# =============================================================================
# Cloud Backend Tests with Flyway-managed Oracle schema (oracle-custom)
# =============================================================================


@pytest.mark.vendor("openai")
@pytest.mark.gpu(0)
@pytest.mark.storage("oracle-custom")
@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestStateManagementOracleCustom(_StateManagementCloudBase):
    """State management tests against Oracle with Flyway-managed schema (schema-config).

    The storage("oracle-custom") marker causes the gateway to launch with
    --schema-config pointing to the Flyway schema, using ATP_FLYWAY_* env vars.
    """


# =============================================================================
# Local Backend Tests (gRPC with Qwen model)
# =============================================================================


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("Qwen/Qwen2.5-14B-Instruct")
@pytest.mark.gateway(extra_args=["--tool-call-parser", "qwen", "--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
@pytest.mark.parametrize("api_client", ["openai", "smg"], indirect=True)
class TestStateManagementLocal:
    """State management tests against local gRPC backend."""

    def test_previous_response_id_invalid(self, model, api_client):
        """Test using invalid previous_response_id."""
        with pytest.raises((openai.BadRequestError, smg_client.BadRequestError)):
            api_client.responses.create(
                model=model,
                input="Test",
                previous_response_id="resp_invalid123",
                max_output_tokens=50,
            )

    def test_basic_response_creation(self, model, api_client):
        """Test basic response creation without state."""
        resp = api_client.responses.create(model=model, input="What is 2+2?")

        assert resp.id is not None
        assert resp.error is None
        assert resp.status == "completed"
        assert len(resp.output_text) > 0
        assert resp.usage is not None
        assert resp.usage.total_tokens == resp.usage.input_tokens + resp.usage.output_tokens

    def test_streaming_response(self, model, api_client):
        """Test streaming response."""
        resp = api_client.responses.create(
            model=model, input="Count to 5", stream=True, max_output_tokens=50
        )

        events = list(resp)
        created_events = [e for e in events if e.type == "response.created"]
        assert len(created_events) > 0

        assert any(e.type in ["response.completed", "response.in_progress"] for e in events)

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

    def test_store_false_previous_response_id_rejected(self, model, api_client):
        """Test that store=false response id cannot be used as previous_response_id."""
        resp = api_client.responses.create(model=model, input="Hello", store=False)
        assert resp.id is not None

        with pytest.raises((openai.BadRequestError, smg_client.BadRequestError)):
            api_client.responses.create(
                model=model,
                input="Follow up",
                previous_response_id=resp.id,
            )

    def test_mutually_exclusive_parameters(self, model, api_client):
        """Test that previous_response_id and conversation are mutually exclusive."""
        conversation_id = "conv_123"
        resp1 = api_client.responses.create(model=model, input="Test")

        with pytest.raises((openai.BadRequestError, smg_client.BadRequestError)):
            api_client.responses.create(
                model=model,
                input="This should fail",
                previous_response_id=resp1.id,
                conversation=conversation_id,
            )


# =============================================================================
# Local Backend Tests (gRPC with Harmony/Reasoning model)
# =============================================================================


@pytest.mark.engine("sglang", "vllm", "trtllm", "tokenspeed")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("openai/gpt-oss-20b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
@pytest.mark.parametrize("api_client", ["openai", "smg"], indirect=True)
class TestStateManagementGptOss:
    """State management tests against local gRPC backend with Harmony model."""

    def test_previous_response_id_invalid(self, model, api_client):
        """Test using invalid previous_response_id."""
        with pytest.raises((openai.BadRequestError, smg_client.BadRequestError)):
            api_client.responses.create(
                model=model,
                input="Test",
                previous_response_id="resp_invalid123",
                max_output_tokens=50,
            )

    def test_basic_response_creation(self, model, api_client):
        """Test basic response creation without state."""
        resp = api_client.responses.create(model=model, input="What is 2+2?")

        assert resp.id is not None
        assert resp.error is None
        assert resp.status == "completed"
        assert len(resp.output_text) > 0
        assert resp.usage is not None
        assert resp.usage.total_tokens == resp.usage.input_tokens + resp.usage.output_tokens

    def test_streaming_response(self, model, api_client):
        """Test streaming response."""
        resp = api_client.responses.create(
            model=model, input="Count to 5", stream=True, max_output_tokens=50
        )

        events = list(resp)
        created_events = [e for e in events if e.type == "response.created"]
        assert len(created_events) > 0

        assert any(e.type in ["response.completed", "response.in_progress"] for e in events)

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

    def test_store_false_previous_response_id_rejected(self, model, api_client):
        """Test that store=false response id cannot be used as previous_response_id."""
        resp = api_client.responses.create(model=model, input="Hello", store=False)
        assert resp.id is not None

        with pytest.raises((openai.BadRequestError, smg_client.BadRequestError)):
            api_client.responses.create(
                model=model,
                input="Follow up",
                previous_response_id=resp.id,
            )

    def test_mutually_exclusive_parameters(self, model, api_client):
        """Test that previous_response_id and conversation are mutually exclusive."""
        conversation_id = "conv_123"
        resp1 = api_client.responses.create(model=model, input="Test")

        with pytest.raises((openai.BadRequestError, smg_client.BadRequestError)):
            api_client.responses.create(
                model=model,
                input="This should fail",
                previous_response_id=resp1.id,
                conversation=conversation_id,
            )
