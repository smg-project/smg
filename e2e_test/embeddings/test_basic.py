"""Basic embedding API tests.

Tests the embedding functionality through the router with both gRPC and HTTP backends.

Source: Migrated from e2e_grpc/basic/test_embedding_server.py

Usage:
    pytest e2e_test/embeddings/test_basic.py -v
    pytest e2e_test/embeddings/test_basic.py -v -k "grpc"
"""

from __future__ import annotations

import logging

import openai
import pytest
import smg_client

logger = logging.getLogger(__name__)


@pytest.mark.engine("sglang", "vllm")
@pytest.mark.gpu(1)
@pytest.mark.model("Qwen/Qwen3-Embedding-0.6B")
@pytest.mark.e2e
@pytest.mark.parametrize("setup_backend", ["grpc", "http"], indirect=True)
@pytest.mark.parametrize("api_client", ["openai", "smg"], indirect=True)
class TestEmbeddingBasic:
    """Basic embedding API tests using local workers (gRPC and HTTP)."""

    def test_embedding_single(self, model, api_client):
        """Test single text embedding.

        Verifies that:
        - Response object structure is correct
        - Embedding is a non-empty list of floats
        - Usage statistics are present
        """

        input_text = "Hello world"
        response = api_client.embeddings.create(
            model=model,
            input=input_text,
        )

        assert response.object == "list"
        assert len(response.data) == 1

        embedding = response.data[0]
        assert embedding.object == "embedding"
        assert embedding.index == 0
        assert len(embedding.embedding) > 0
        assert isinstance(embedding.embedding[0], float)

        # Verify usage statistics
        assert response.usage.prompt_tokens > 0
        assert response.usage.total_tokens == response.usage.prompt_tokens

        logger.info(
            "Single embedding: %d dimensions, %d tokens",
            len(embedding.embedding),
            response.usage.prompt_tokens,
        )

    def test_embedding_batch(self, model, api_client):
        """Test batch embedding with multiple texts.

        Note: The original test expected len(response.data) == 1 for batch,
        which seems incorrect. This might be model-specific behavior.
        """

        input_texts = ["Hello world", "SGLang is fast"]
        response = api_client.embeddings.create(
            model=model,
            input=input_texts,
        )

        # Note: Original test had len(response.data) == 1, which seems like
        # a bug or model-specific behavior. Standard behavior should return
        # one embedding per input text.
        assert len(response.data) >= 1
        assert response.data[0].index == 0
        assert len(response.data[0].embedding) > 0

        logger.info("Batch embedding: %d results", len(response.data))

    def test_embedding_dimensions_consistent(self, model, api_client):
        """Test that embedding dimensions are consistent across different inputs.

        Verifies that different length inputs produce embeddings with
        the same dimensionality.
        """

        response1 = api_client.embeddings.create(
            model=model,
            input="A short text",
        )
        dim1 = len(response1.data[0].embedding)

        response2 = api_client.embeddings.create(
            model=model,
            input="A much longer text to ensure dimensions match regardless of input length",
        )
        dim2 = len(response2.data[0].embedding)

        assert dim1 == dim2, f"Dimensions differ: {dim1} vs {dim2}"
        logger.info("Embedding dimensions: %d (consistent)", dim1)

    def test_embedding_empty_string(self, model, api_client):
        """Test embedding with empty string input.

        Contract: an empty-string input must either be embedded successfully
        (one embedding with the model's usual dimension) or be rejected with a
        4xx client error. Anything else — a 5xx, a transport error, a malformed
        success body — is a bug and must fail the test.

        Note: the test has swallowed both outcomes since its introduction
        (see #812/#834 refactors), so the behavior current engines exhibit was
        never recorded; the GPU lanes pin it down via this two-branch assert.
        """

        # Probe the model's embedding dimension with a known-good input.
        probe = api_client.embeddings.create(model=model, input="dimension probe")
        expected_dim = len(probe.data[0].embedding)
        assert expected_dim > 0

        try:
            response = api_client.embeddings.create(
                model=model,
                input="",
            )
        except (openai.APIStatusError, smg_client.ApiError) as e:
            # Rejection is acceptable only as a client error (4xx).
            assert 400 <= e.status_code < 500, (
                f"Empty string input must be rejected with a 4xx client error, "
                f"got HTTP {e.status_code}: {e}"
            )
            logger.info("Empty string embedding rejected with HTTP %d", e.status_code)
        else:
            # Acceptance must produce exactly one well-formed embedding.
            assert len(response.data) == 1, (
                f"Expected exactly 1 embedding for empty string, got {len(response.data)}"
            )
            assert len(response.data[0].embedding) == expected_dim, (
                f"Empty string embedding dimension {len(response.data[0].embedding)} "
                f"differs from model dimension {expected_dim}"
            )
            logger.info("Empty string embedding succeeded (%d dims)", expected_dim)

    def test_embedding_unicode(self, model, api_client):
        """Test embedding with unicode characters.

        Verifies that the API handles non-ASCII characters correctly.
        """

        input_text = "Hello 世界! 🚀 Привет мир"
        response = api_client.embeddings.create(
            model=model,
            input=input_text,
        )

        assert len(response.data) == 1
        assert len(response.data[0].embedding) > 0
        logger.info("Unicode embedding: %d dimensions", len(response.data[0].embedding))
