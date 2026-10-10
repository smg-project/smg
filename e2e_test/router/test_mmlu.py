"""MMLU canary for the gRPC router path.

One small MMLU run through a gRPC worker: the gateway renders the chat
template, tokenizes, builds the sampling parameters and detokenizes, so a
broken step there drops the score below the floor. At 64 examples the score
moves by about ten points between runs, so this only catches gross breakage.

Usage:
    pytest e2e_test/router/test_mmlu.py -v
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from infra import run_eval

logger = logging.getLogger(__name__)


@pytest.mark.engine("sglang", "vllm", "tokenspeed")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestMMLUGrpc:
    """MMLU evaluation tests using gRPC workers."""

    def test_mmlu_basic(self, setup_backend):
        """Basic MMLU evaluation with score threshold.

        Runs MMLU evaluation with 64 examples and validates that
        accuracy meets minimum threshold (>= 0.65).
        """
        backend, model, client, *_ = setup_backend

        args = SimpleNamespace(
            base_url=str(client.base_url),
            model=model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
            temperature=0.1,
        )
        metrics = run_eval(args)

        assert metrics["score"] >= 0.65, f"MMLU score {metrics['score']:.2f} below threshold 0.65"
        logger.info("MMLU gRPC score: %.2f (threshold: 0.65)", metrics["score"])
