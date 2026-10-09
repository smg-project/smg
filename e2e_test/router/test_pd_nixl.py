"""NIXL KV-transfer verification for vLLM PD disaggregation (gRPC mode).

Output correctness alone cannot catch a broken KV transfer: the decode worker
silently recomputes the prefill and still produces correct answers. This test
asserts the transfer actually happens by checking:

1. The router harvested kv_transfer_params from prefill and relayed them to
   decode (router debug log).
2. The decode worker performed a NIXL handshake with the prefill worker
   ("Transfer plan" log, vLLM >= 0.20; handshake markers on older versions).

Requirements:
    - vLLM with NixlConnector support (workers launched with
      kv_role=kv_producer / kv_consumer by the worker fixture)
    - 2 GPUs (1 prefill + 1 decode)

Usage:
    E2E_RUNTIME=vllm pytest e2e_test/router/test_pd_nixl.py -v
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import pytest
from infra.constants import vllm_kv_backend
from infra.pd_logs import (
    assert_worker_logs_captured,
    unique_prompt,
    wait_for_marker,
    worker_log_dir,
)

logger = logging.getLogger(__name__)

# Router logs land here via the gateway marker (rolling files named smg.YYYY-MM-DD)
_LOG_DIR = Path(tempfile.gettempdir()) / f"smg-e2e-pd-nixl-{os.getpid()}"

RELAY_MARKER = "relaying prefill kv_transfer_params"
NO_PARAMS_MARKER = "prefill returned no kv_transfer_params"
# "Transfer plan" was introduced in vLLM 0.20; v0.19 logs
# "NIXL compatibility check passed" at INFO during the handshake
HANDSHAKE_MARKERS = (
    "Transfer plan",
    "NIXL compatibility check passed",
    "NIXL handshake",
    "Registering remote agent",
)


@pytest.mark.skipif(vllm_kv_backend() != "nixl", reason="nixl PD leg only")
@pytest.mark.engine("vllm")
@pytest.mark.gpu(2)
@pytest.mark.model("meta-llama/Llama-3.1-8B-Instruct")
@pytest.mark.e2e
@pytest.mark.gateway(log_level="debug", log_dir=str(_LOG_DIR))
@pytest.mark.parametrize("setup_backend", ["pd_grpc"], indirect=True)
class TestPDNixlKvTransfer:
    """Verify KV cache actually moves from prefill to decode over NIXL."""

    def test_kv_transfer_params_relayed(self, setup_backend):
        backend, model, client, *_ = setup_backend

        for _ in range(3):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": unique_prompt()}],
                max_tokens=64,
                temperature=0.0,
            )
            choice = response.choices[0]
            assert choice.message.content, "Empty completion from PD pipeline"

        router_logs = wait_for_marker(_LOG_DIR, "smg*", RELAY_MARKER)
        assert NO_PARAMS_MARKER not in router_logs, (
            "Router reported prefill returned no kv_transfer_params — "
            "NIXL handoff is broken (decode recomputed the prefill)"
        )
        assert RELAY_MARKER in router_logs, (
            f"Router never relayed kv_transfer_params to decode; checked logs under {_LOG_DIR}"
        )

    def test_decode_worker_performed_nixl_handshake(self, setup_backend):
        backend, model, client, *_ = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": unique_prompt()}],
            max_tokens=32,
            temperature=0.0,
        )
        assert response.choices[0].message.content

        worker_dir = worker_log_dir(_LOG_DIR)
        worker_logs = wait_for_marker(worker_dir, "worker-*.log", HANDSHAKE_MARKERS)
        assert_worker_logs_captured(worker_logs, "NIXL handshake")
        assert any(marker in worker_logs for marker in HANDSHAKE_MARKERS), (
            "No NIXL handshake found in worker logs — the decode worker never "
            f"pulled KV from prefill; checked {worker_dir}/worker-*.log for "
            f"{HANDSHAKE_MARKERS}"
        )
