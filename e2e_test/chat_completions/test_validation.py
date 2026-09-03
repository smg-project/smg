"""Validation E2E Tests.

Tests for validation features like ignore_eos, large token handling,
and request parameter conflict detection.

Source: Migrated from e2e_grpc/validation/test_openai_server_ignore_eos.py
        and e2e_grpc/validation/test_large_max_new_tokens.py
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import openai
import pytest
import smg_client

logger = logging.getLogger(__name__)

# Lazy load tokenizer to avoid import errors if transformers not installed
_tokenizer_cache: dict = {}


def get_tokenizer(model_path: str):
    """Get tokenizer for a model, with caching."""
    if model_path not in _tokenizer_cache:
        from transformers import AutoTokenizer

        _tokenizer_cache[model_path] = AutoTokenizer.from_pretrained(model_path)
    return _tokenizer_cache[model_path]


# =============================================================================
# Ignore EOS Tests (Llama 8B)
# =============================================================================


@pytest.mark.engine("sglang", "vllm", "trtllm", "tokenspeed")
@pytest.mark.gpu(1)
@pytest.mark.model("meta-llama/Llama-3.1-8B-Instruct")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
@pytest.mark.parametrize("api_client", ["openai", "smg"], indirect=True)
class TestIgnoreEOS:
    """Tests for ignore_eos feature."""

    def test_ignore_eos(self, model, api_client):
        """Test that ignore_eos=True allows generation to continue beyond EOS token.

        When ignore_eos=True, the model should generate until max_tokens is reached,
        even if it encounters an EOS token.
        """

        tokenizer = get_tokenizer(model)
        max_tokens = 200

        # Request without ignore_eos (default behavior - stops at EOS)
        response_default = api_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count from 1 to 20."},
            ],
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"ignore_eos": False},
        )

        # Request with ignore_eos=True (continues past EOS until max_tokens)
        response_ignore_eos = api_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count from 1 to 20."},
            ],
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"ignore_eos": True},
        )

        default_tokens = len(tokenizer.encode(response_default.choices[0].message.content))
        ignore_eos_tokens = len(tokenizer.encode(response_ignore_eos.choices[0].message.content))

        # Check if ignore_eos resulted in more tokens or exactly max_tokens
        # The ignore_eos response should either:
        # 1. Have more tokens than the default response (if default stopped at EOS before max_tokens)
        # 2. Have exactly max_tokens (if it reached the max_tokens limit)
        assert ignore_eos_tokens > default_tokens or ignore_eos_tokens >= max_tokens, (
            f"ignore_eos did not generate more tokens: {ignore_eos_tokens} vs {default_tokens}"
        )

        assert response_ignore_eos.choices[0].finish_reason == "length", (
            f"Expected finish_reason='length' for ignore_eos=True, "
            f"got {response_ignore_eos.choices[0].finish_reason}"
        )


# =============================================================================
# Large Max New Tokens Tests (Llama 8B)
#
# NOTE: This test verifies concurrent request handling with large token limits.
# The original test monitored server logs to verify concurrency, which is not
# possible with the pool-based infrastructure. This simplified version verifies
# that concurrent requests complete successfully.
# =============================================================================


@pytest.mark.engine("sglang", "vllm", "trtllm", "tokenspeed")
@pytest.mark.gpu(1)
@pytest.mark.model("meta-llama/Llama-3.1-8B-Instruct")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
@pytest.mark.parametrize("api_client", ["openai", "smg"], indirect=True)
class TestLargeMaxNewTokens:
    """Tests for handling large max_new_tokens with concurrent requests."""

    def test_concurrent_chat_completions(self, model, api_client):
        """Test that multiple concurrent requests with large token generation complete.

        This test sends multiple requests that ask for long outputs concurrently
        to verify the server can handle concurrent long-running requests.
        """

        num_requests = 4

        def run_chat_completion():
            response = api_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant"},
                    {
                        "role": "user",
                        "content": "Please repeat the word 'hello' for 100 times.",
                    },
                ],
                temperature=0,
                max_tokens=256,  # Reasonable limit for concurrent test
            )
            return response

        # Send concurrent requests
        start_time = time.time()
        futures = []
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            for _ in range(num_requests):
                futures.append(executor.submit(run_chat_completion))

            # Wait for all to complete and collect results
            responses = [f.result() for f in futures]

        elapsed = time.time() - start_time
        logger.info("Completed %d concurrent requests in %.2fs", num_requests, elapsed)

        # Verify all requests completed successfully
        assert len(responses) == num_requests
        for i, response in enumerate(responses):
            assert response.choices[0].message.content, f"Request {i} returned empty content"
            assert response.choices[0].finish_reason in (
                "stop",
                "length",
            ), f"Request {i} had unexpected finish_reason: {response.choices[0].finish_reason}"


# =============================================================================
# Harmony Validation Tests (GPT-OSS)
# =============================================================================


@pytest.mark.engine("sglang", "vllm", "trtllm", "tokenspeed")
@pytest.mark.gpu(1)
@pytest.mark.model("openai/gpt-oss-20b")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
@pytest.mark.parametrize("api_client", ["openai", "smg"], indirect=True)
class TestGptOssValidation:
    """Validation tests for Harmony models (GPT-OSS)."""

    def test_ignore_eos_rejected(self, model, api_client):
        """Test that ignore_eos is rejected for Harmony models with HTTP 400."""

        with pytest.raises((openai.BadRequestError, smg_client.BadRequestError)) as exc_info:
            api_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "Hello"},
                ],
                extra_body={"ignore_eos": True},
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.code == "ignore_eos_not_supported"

    def test_tool_choice_with_response_format_rejected(self, model, api_client):
        """Test that tool_choice + response_format is rejected with HTTP 400."""

        with pytest.raises((openai.BadRequestError, smg_client.BadRequestError)) as exc_info:
            api_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "List 2 fruits."},
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_fruits",
                            "description": "Get a list of fruits",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "count": {"type": "integer"},
                                },
                                "required": ["count"],
                            },
                        },
                    }
                ],
                tool_choice="required",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "fruits",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "items": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["items"],
                            "additionalProperties": False,
                        },
                    },
                },
            )
        assert exc_info.value.status_code == 400


# =============================================================================
# EOS Token Stripping Tests (Llama 1B)
# Verify that EOS tokens are stripped from output by StopDecoder even when
# skip_special_tokens=false. Regression test for generation_config.json
# EOS loading and StopDecoder EOS stripping.
# =============================================================================


@pytest.mark.engine("sglang", "vllm", "tokenspeed")
@pytest.mark.gpu(1)
@pytest.mark.model("meta-llama/Llama-3.2-1B-Instruct")
@pytest.mark.gateway(extra_args=["--tool-call-parser", "llama", "--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
@pytest.mark.parametrize("api_client", ["openai", "smg"], indirect=True)
class TestEosTokenStripping:
    """Verify EOS tokens are stripped from output by StopDecoder.

    gRPC backends return raw token IDs including EOS. The StopDecoder must
    strip EOS at the token ID level before decoding, matching vllm/sglang
    HTTP behavior. Without this fix, skip_special_tokens=false causes EOS
    tokens like <|eom_id|> to appear in decoded text.
    """

    # Llama 3 EOS tokens that should never appear in output
    EOS_TOKENS = ["<|eom_id|>", "<|eot_id|>", "<|end_of_text|>"]

    def test_no_eos_in_content_default(self, model, api_client):
        """With default settings, EOS tokens should not appear in content."""
        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Say hello in one sentence."},
            ],
            temperature=0,
            max_tokens=50,
            stream=False,
        )
        content = response.choices[0].message.content
        assert content is not None
        for eos in self.EOS_TOKENS:
            assert eos not in content, f"EOS token {eos} leaked into content: {content}"

    def test_no_eos_in_content_skip_special_false(self, model, api_client):
        """With skip_special_tokens=false, EOS should still be stripped by StopDecoder."""
        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Say hello in one sentence."},
            ],
            temperature=0,
            max_tokens=50,
            stream=False,
            extra_body={"skip_special_tokens": False},
        )
        content = response.choices[0].message.content
        assert content is not None
        for eos in self.EOS_TOKENS:
            assert eos not in content, f"EOS token {eos} leaked into content: {content}"

    def test_no_eos_in_tool_call_with_tools(self, model, api_client):
        """With tools and skip_special_tokens=false, tool calls should not contain EOS."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Compute the sum of two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer", "description": "A number"},
                            "b": {"type": "integer", "description": "A number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]
        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Compute (3+5)"},
            ],
            tools=tools,
            tool_choice="required",
            temperature=0,
            max_tokens=200,
            stream=False,
            extra_body={"skip_special_tokens": False},
        )
        choice = response.choices[0]

        # Check tool call arguments for EOS
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                for eos in self.EOS_TOKENS:
                    assert eos not in tc.function.arguments, (
                        f"EOS token {eos} in tool call arguments: {tc.function.arguments}"
                    )

        # Check content for EOS (if model returned content instead of tool calls)
        if choice.message.content:
            for eos in self.EOS_TOKENS:
                assert eos not in choice.message.content, (
                    f"EOS token {eos} in content: {choice.message.content}"
                )

    def test_no_stop_trim_with_skip_special_true(self, model, api_client):
        """With no_stop_trim=true and skip_special_tokens=true (default),
        EOS should be kept in token list but invisible (decoded to empty)."""
        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Say hello in one sentence."},
            ],
            temperature=0,
            max_tokens=50,
            stream=False,
            extra_body={"no_stop_trim": True},
        )
        content = response.choices[0].message.content
        assert content is not None
        for eos in self.EOS_TOKENS:
            assert eos not in content, (
                f"EOS token {eos} should be invisible with skip_special_tokens=true: {content}"
            )

    def test_no_stop_trim_with_skip_special_false(self, model, api_client):
        """With no_stop_trim=true and skip_special_tokens=false,
        EOS should be visible in output (matching sglang behavior)."""
        response = api_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Say hello in one sentence."},
            ],
            temperature=0,
            max_tokens=50,
            stream=False,
            extra_body={"no_stop_trim": True, "skip_special_tokens": False},
        )
        content = response.choices[0].message.content
        assert content is not None
        assert response.choices[0].finish_reason == "stop", (
            "Model hit max_tokens before EOS — test is inconclusive"
        )
        # EOS should be visible when both no_stop_trim=true and skip_special_tokens=false
        has_eos = any(eos in content for eos in self.EOS_TOKENS)
        assert has_eos, (
            f"EOS token should be visible with no_stop_trim=true + skip_special_tokens=false: {content}"
        )


# =============================================================================
# Unknown model rejection (gRPC pipeline)
# =============================================================================


@pytest.mark.engine("sglang", "vllm", "trtllm", "tokenspeed")
@pytest.mark.gpu(1)
@pytest.mark.model("meta-llama/Llama-3.1-8B-Instruct")
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
@pytest.mark.parametrize("api_client", ["openai", "smg"], indirect=True)
class TestUnknownModelRejected:
    """An unknown model must be rejected with 404 before the pipeline runs.

    The gRPC pipeline tokenizes locally, so its preparation stage runs ahead
    of worker selection. Without an up-front check, an unknown model reaches
    tokenizer resolution and surfaces as a 500 ``tokenizer_not_found`` — a
    client error reported as a server fault. The responses endpoint already
    fails fast with 404 ``model_not_found``; every gRPC entry point should.
    """

    UNKNOWN = "nonexistent-model-xyz"

    def test_chat_unknown_model_is_404(self, api_client):
        with pytest.raises((openai.NotFoundError, smg_client.NotFoundError)) as exc_info:
            api_client.chat.completions.create(
                model=self.UNKNOWN,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=8,
            )
        assert exc_info.value.status_code == 404
        assert exc_info.value.code == "model_not_found"

    def test_completions_unknown_model_is_404(self, api_client):
        with pytest.raises((openai.NotFoundError, smg_client.NotFoundError)) as exc_info:
            api_client.completions.create(
                model=self.UNKNOWN,
                prompt="Hello",
                max_tokens=8,
            )
        assert exc_info.value.status_code == 404
        assert exc_info.value.code == "model_not_found"
