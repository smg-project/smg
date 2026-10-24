"""Mooncake KV-transfer verification for vLLM PD disaggregation (gRPC mode).

The MooncakeConnector is push-based: the prefill engine returns no
kv_transfer_params, so the router mints a transfer_id, tags the prefill
request, and synthesizes the decode params (remote_engine_id discovered via
GetServerInfo, remote_bootstrap_addr from worker metadata). This test asserts:

1. The router injected minted kv_transfer_params into the decode request
   (router debug log).
2. The workers actually transferred KV blocks ("KV Transfer metrics" with
   successful transfers in the worker logs).

Runs only on the mooncake leg of e2e-2gpu-pd (E2E_VLLM_KV_BACKEND=mooncake).

Usage:
    E2E_RUNTIME=vllm E2E_VLLM_KV_BACKEND=mooncake \
        pytest e2e_test/router/test_pd_mooncake.py -v
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path

import pytest
from infra.constants import vllm_kv_backend
from infra.pd_logs import (
    assert_worker_logs_captured,
    unique_prompt,
    wait_for_marker,
    wait_for_pattern_in_files,
)

logger = logging.getLogger(__name__)

# Router logs land here via the gateway marker (rolling files named smg.YYYY-MM-DD)
_LOG_DIR = Path(tempfile.gettempdir()) / f"smg-e2e-pd-mooncake-{os.getpid()}"

MINT_MARKER = "vLLM PD (Mooncake): injecting minted kv_transfer_params"
# Periodic connector metrics line; a positive count is required — the line is
# also printed with "Num successful transfers=0" when every transfer failed
TRANSFER_SUCCESS_RE = re.compile(r"Num successful transfers=([1-9]\d*)")


@pytest.mark.skipif(vllm_kv_backend() != "mooncake", reason="mooncake PD leg only")
@pytest.mark.engine("vllm")
@pytest.mark.gpu(2)
@pytest.mark.model("meta-llama/Llama-3.1-8B-Instruct")
@pytest.mark.e2e
@pytest.mark.gateway(log_level="debug", log_dir=str(_LOG_DIR))
@pytest.mark.parametrize("setup_backend", ["pd_grpc"], indirect=True)
class TestPDMooncakeKvTransfer:
    """Verify KV cache actually moves from prefill to decode over Mooncake."""

    def test_minted_params_injected(self, setup_backend):
        backend, model, client, *_ = setup_backend

        for _ in range(3):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": unique_prompt()}],
                max_tokens=64,
                temperature=0.0,
            )
            assert response.choices[0].message.content, "Empty completion from PD pipeline"

        router_logs = wait_for_marker(_LOG_DIR, "smg*", MINT_MARKER)
        assert MINT_MARKER in router_logs, (
            "Router never injected minted Mooncake kv_transfer_params — "
            f"engine_id discovery or minting is broken; checked logs under {_LOG_DIR}"
        )

    def test_workers_transferred_kv(self, setup_backend):
        backend, model, client, gateway = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": unique_prompt()}],
            max_tokens=32,
            temperature=0.0,
        )
        assert response.choices[0].message.content

        worker_paths = [
            worker.log_path for worker in gateway.prefill_workers + gateway.decode_workers
        ]
        worker_logs = wait_for_pattern_in_files(worker_paths, TRANSFER_SUCCESS_RE)
        assert_worker_logs_captured(worker_logs, "Mooncake transfer")
        assert TRANSFER_SUCCESS_RE.search(worker_logs), (
            "No successful Mooncake KV transfer in worker logs (zero-success "
            "metrics mean decode recomputed locally); checked "
            f"owned worker paths {worker_paths} for {TRANSFER_SUCCESS_RE.pattern}"
        )
