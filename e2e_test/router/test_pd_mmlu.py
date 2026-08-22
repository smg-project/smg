"""MMLU evaluation tests for PD (Prefill-Decode) disaggregated routing.

PD disaggregation separates prefill and decode phases across different
workers for improved throughput and resource utilization.

Backends:
- "pd_http": HTTP mode (SGLang parallel bootstrap dispatch; vLLM sequential
  prefill-then-decode with kv_transfer_params relay)
- "pd_grpc": gRPC mode (both SGLang and vLLM)

Requirements:
    - SGLang: sgl_kernel package
    - vLLM: NIXL support, RDMA/InfiniBand connectivity
    - GPUs: num_prefill + num_decode (default: 2 GPUs for 1+1)

Usage:
    # SGLang (runs both HTTP and gRPC)
    pytest e2e_test/router/test_pd_mmlu.py -v

    # vLLM (runs both HTTP and gRPC)
    E2E_RUNTIME=vllm pytest e2e_test/router/test_pd_mmlu.py -v
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from infra import run_eval

logger = logging.getLogger(__name__)


@pytest.mark.engine("sglang", "vllm")
@pytest.mark.gpu(2)
@pytest.mark.model("meta-llama/Llama-3.1-8B-Instruct")
@pytest.mark.e2e
@pytest.mark.parametrize("setup_backend", ["pd_http"], indirect=True)
class TestPDMMLUHttp:
    """MMLU evaluation tests using PD disaggregation (HTTP mode)."""

    def test_pd_mmlu_basic(self, setup_backend):
        """Basic MMLU evaluation with PD disaggregation."""
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

        assert metrics["score"] >= 0.65, (
            f"PD MMLU score {metrics['score']:.2f} below threshold 0.65"
        )
        logger.info("PD HTTP MMLU score: %.2f (threshold: 0.65)", metrics["score"])


@pytest.mark.engine("sglang", "vllm")
@pytest.mark.gpu(2)
@pytest.mark.model("meta-llama/Llama-3.1-8B-Instruct")
@pytest.mark.e2e
@pytest.mark.parametrize("setup_backend", ["pd_grpc"], indirect=True)
class TestPDMMLUGrpc:
    """MMLU evaluation tests using PD disaggregation (gRPC mode)."""

    def test_pd_mmlu_basic(self, setup_backend):
        """Basic MMLU evaluation with PD disaggregation."""
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

        assert metrics["score"] >= 0.65, (
            f"PD MMLU score {metrics['score']:.2f} below threshold 0.65"
        )
        logger.info("PD gRPC MMLU score: %.2f (threshold: 0.65)", metrics["score"])
