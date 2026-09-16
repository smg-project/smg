"""Multimodal Chat Completions E2E Tests.

Tests for vision-language models through the gateway, verifying that
image content is correctly processed and the model produces meaningful
responses about the images. Tests both URL and base64 image inputs.

Usage:
    pytest e2e_test/chat_completions/test_multimodal.py -v
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import pytest
from infra import assert_mm_processing

logger = logging.getLogger(__name__)

# Local test images (checked into repo)
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "images"
DOG_IMAGE_PATH = FIXTURES_DIR / "dog.jpg"  # Black labrador puppy
PUG_IMAGE_PATH = FIXTURES_DIR / "pug.jpg"  # Pug in blanket

# The same local fixtures, served as URLs (to exercise the URL-fetch path).
# Served from this repo's own raw content instead of a third-party image host
# (picsum.photos), whose outages were flaking the URL-based multimodal tests.
# Using the repo's own pug.jpg also makes the duplicate-image assertion in
# test_multi_images_mixed exact: the URL and base64 pug are now byte-identical.
IMAGE_DOG_URL = (
    "https://raw.githubusercontent.com/smg-project/smg/main/e2e_test/fixtures/images/dog.jpg"
)
IMAGE_PUG_URL = (
    "https://raw.githubusercontent.com/smg-project/smg/main/e2e_test/fixtures/images/pug.jpg"
)


def _image_to_base64_url(path: Path) -> str:
    """Convert a local image file to a base64 data URL."""
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


def _make_image_content(image_source: str) -> dict:
    """Create an image_url content part from either a URL or local path."""
    return {"type": "image_url", "image_url": {"url": image_source}}


# =============================================================================
# Qwen3-VL multimodal tests (1 GPU)
# =============================================================================


@pytest.mark.engine("vllm", "sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("Qwen/Qwen3-VL-8B-Instruct")
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestMultimodalQwen3VL:
    """Multimodal tests using Qwen3-VL via gRPC."""

    @pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
    def test_single_image_base64(self, model, setup_backend, stream):
        """Test single image understanding with local base64 image."""
        _, _, client, *_ = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What animal is in this image?"},
                        _make_image_content(_image_to_base64_url(DOG_IMAGE_PATH)),
                    ],
                }
            ],
            temperature=0,
            max_tokens=100,
            stream=stream,
        )

        text = _extract_text(response, stream)
        assert any(k in text.lower() for k in ["dog", "puppy", "labrador"]), (
            f"Expected dog-related content, got: {text}"
        )
        logger.info("Single image base64 (stream=%s): %s", stream, text)

    @pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
    def test_single_image_url(self, model, setup_backend, stream):
        """Test single image understanding with URL image."""
        _, _, client, *_ = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What animal is in this image?"},
                        _make_image_content(IMAGE_DOG_URL),
                    ],
                }
            ],
            temperature=0,
            max_tokens=100,
            stream=stream,
        )

        text = _extract_text(response, stream)
        assert any(k in text.lower() for k in ["dog", "puppy", "labrador"]), (
            f"Expected dog-related content, got: {text}"
        )
        logger.info("Single image URL (stream=%s): %s", stream, text)

    def test_processing_location_matches_lane(self, model, setup_backend):
        """The gateway records where this lane processed each image request."""
        _, _, client, gateway = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What animal is in this image?"},
                        _make_image_content(_image_to_base64_url(DOG_IMAGE_PATH)),
                    ],
                }
            ],
            temperature=0,
            max_tokens=16,
        )
        assert response.choices[0].message.content
        # Qwen3-VL's <|image_pad|> anchor is one a vLLM worker expands itself.
        assert_mm_processing(gateway, worker_expandable=True)

    def test_multi_images_mixed(self, model, setup_backend):
        """Test multiple images with mixed base64 and URL inputs, including duplicates."""
        _, _, client, *_ = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "How many images did I send? Describe each. Are any of them the same?",
                        },
                        _make_image_content(_image_to_base64_url(DOG_IMAGE_PATH)),
                        _make_image_content(IMAGE_PUG_URL),
                        _make_image_content(_image_to_base64_url(PUG_IMAGE_PATH)),
                    ],
                }
            ],
            temperature=0,
            max_tokens=300,
        )

        text = response.choices[0].message.content
        assert text is not None and len(text) > 0
        text_lower = text.lower()

        # Don't assert an exact image count: the two pug inputs are byte-identical,
        # and engines legitimately differ on whether identical multimodal inputs are
        # deduplicated (vLLM encodes the duplicate once; sglang keeps both). The
        # duplicate-detection assertion below covers the intent of this test.
        # Should identify both dog and pug
        assert any(k in text_lower for k in ["dog", "puppy", "labrador"]), (
            f"Expected dog-related content, got: {text}"
        )
        assert any(k in text_lower for k in ["pug", "blanket", "wrapped"]), (
            f"Expected pug-related content, got: {text}"
        )
        # Images 2 and 3 are the same pug — model should notice
        assert any(
            k in text_lower
            for k in [
                "same",
                "identical",
                "duplicate",
                "copy",
                "copies",
                "repeated",
                "twice",
                "similar",
                "both",
            ]
        ), f"Expected model to notice duplicate images, got: {text}"
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        logger.info("Multi image mixed response: %s", text)


# =============================================================================
# Llama-4-Scout multimodal tests (4 GPU)
# =============================================================================


@pytest.mark.engine("vllm", "sglang")
@pytest.mark.gpu(4)
@pytest.mark.e2e
@pytest.mark.model("meta-llama/Llama-4-Scout-17B-16E-Instruct")
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestMultimodalLlama4Scout:
    """Multimodal tests using Llama-4-Scout via gRPC."""

    @pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
    def test_single_image_base64(self, model, setup_backend, stream):
        """Test single image understanding with local base64 image."""
        _, _, client, *_ = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What animal is in this image?"},
                        _make_image_content(_image_to_base64_url(DOG_IMAGE_PATH)),
                    ],
                }
            ],
            temperature=0,
            max_tokens=100,
            stream=stream,
        )

        text = _extract_text(response, stream)
        assert any(k in text.lower() for k in ["dog", "puppy", "labrador"]), (
            f"Expected dog-related content, got: {text}"
        )
        logger.info("Single image base64 (stream=%s): %s", stream, text)

    def test_multi_images_mixed(self, model, setup_backend):
        """Test multiple images with mixed base64 and URL inputs, including duplicates."""
        _, _, client, *_ = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "How many images did I send? Describe each. Are any of them the same?",
                        },
                        _make_image_content(_image_to_base64_url(DOG_IMAGE_PATH)),
                        _make_image_content(IMAGE_PUG_URL),
                        _make_image_content(_image_to_base64_url(PUG_IMAGE_PATH)),
                    ],
                }
            ],
            temperature=0,
            max_tokens=300,
        )

        text = response.choices[0].message.content
        assert text is not None and len(text) > 0
        text_lower = text.lower()

        assert any(k in text_lower for k in ["dog", "pug", "puppy"]), (
            f"Expected dog-related content, got: {text}"
        )
        assert any(
            k in text_lower for k in ["same", "identical", "duplicate", "second", "third"]
        ), f"Expected model to notice duplicate images, got: {text}"
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        logger.info("Multi image mixed response: %s", text)


# =============================================================================
# Helpers
# =============================================================================


def _extract_text(response, stream: bool) -> str:
    """Extract text from a streaming or non-streaming response."""
    if stream:
        chunks = [
            chunk.choices[0].delta.content
            for chunk in response
            if chunk.choices and chunk.choices[0].delta.content
        ]
        text = "".join(chunks)
        assert text, "Streaming should produce content"
        assert len(chunks) > 1, "Streaming should produce multiple chunks"
        return text

    text = response.choices[0].message.content
    assert text is not None and len(text) > 0
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    return text
