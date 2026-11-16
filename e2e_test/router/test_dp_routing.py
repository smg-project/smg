"""DP rank-pinning verification for vLLM gRPC workers (``--dp-aware``), and
its twin on the mock worker.

Output correctness alone cannot catch broken pinning: when no rank reaches the
engine, vLLM load-balances across DP ranks internally and still produces
correct completions. This test asserts the transmitted rank instead, via the
servicer's per-request ``dp_rank=<n>`` log line:

1. Both ``dp_rank=0`` and ``dp_rank=1`` appear in the worker log — the router
   discovered dp_size, expanded the single worker URL into one worker per
   rank, and pinned requests to both.
2. ``dp_rank=None`` never appears — every request carried a pinned rank
   instead of falling back to vLLM-internal balancing.

The mock worker (``crates/mock_worker``) serves ``--grpc-dp-size`` ranks under
its realistic engine and logs the same line per request when it has more than
one rank, so the second class below proves the same two facts on the CPU row of
every pull request, with no engine and no card.

Requirements (the engine class):
    - vLLM gRPC worker launched with --data-parallel-size 2
    - 2 GPUs (one per DP rank)

Usage:
    E2E_RUNTIME=vllm E2E_ENGINE=vllm pytest e2e_test/router/test_dp_routing.py -v
    E2E_RUNTIME=mock E2E_ENGINE=mock pytest e2e_test/router/test_dp_routing.py -v
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import pytest
from infra.pd_logs import (
    assert_worker_logs_captured,
    unique_prompt,
    wait_for_marker,
    worker_log_dir,
)

logger = logging.getLogger(__name__)

# Router logs land here via the gateway marker (rolling files named smg.YYYY-MM-DD)
_LOG_DIR = Path(tempfile.gettempdir()) / f"smg-e2e-dp-routing-{os.getpid()}"

# Scope to this test's worker — PD workers in the same CI job log dp_rank=None
_WORKER_LOG_GLOB = "worker-meta-llama__Llama-3.2-1B-Instruct_vllm_grpc_*.log"
_MOCK_WORKER_LOG_GLOB = "worker-RedHatAI__Llama-3.2-1B-Instruct-FP8_mock_grpc_*.log"

RANK_MARKERS = ("dp_rank=0", "dp_rank=1")
UNPINNED_MARKER = "dp_rank=None"


def _assert_requests_pinned_to_both_ranks(setup_backend, worker_log_glob: str) -> None:
    """Eight requests reach both ranks and none reaches the worker unpinned."""
    backend, model, client, *_ = setup_backend

    for _ in range(8):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": unique_prompt()}],
            max_tokens=16,
            temperature=0.0,
        )
        assert response.choices[0].message.content, "Empty completion from DP backend"

    worker_dir = worker_log_dir(_LOG_DIR)
    wait_for_marker(worker_dir, worker_log_glob, RANK_MARKERS[0])
    worker_logs = wait_for_marker(worker_dir, worker_log_glob, RANK_MARKERS[1])
    assert_worker_logs_captured(worker_logs, "DP rank pinning")

    missing = [m for m in RANK_MARKERS if m not in worker_logs]
    assert not missing, (
        f"Router never pinned {missing} — 8 round-robin requests across "
        f"dp_size=2 must hit both ranks; checked {worker_dir}/{worker_log_glob}"
    )
    assert UNPINNED_MARKER not in worker_logs, (
        "Worker received a request without a pinned DP rank — the router "
        f"fell back to engine-internal balancing; checked {worker_dir}/{worker_log_glob}"
    )


@pytest.mark.engine("vllm")
@pytest.mark.gpu(2)
@pytest.mark.model("meta-llama/Llama-3.2-1B-Instruct")
@pytest.mark.e2e
@pytest.mark.gateway(log_level="debug", log_dir=str(_LOG_DIR), extra_args=["--dp-aware"])
@pytest.mark.workers(count=1, gpus=2, extra_engine_args=["--data-parallel-size", "2"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestDpAwareRankPinning:
    """Verify the router pins every request to a DP rank and covers both ranks."""

    def test_requests_pinned_to_both_ranks(self, setup_backend):
        _assert_requests_pinned_to_both_ranks(setup_backend, _WORKER_LOG_GLOB)


@pytest.mark.engine("mock")
@pytest.mark.gpu(1)
@pytest.mark.model("RedHatAI/Llama-3.2-1B-Instruct-FP8")
@pytest.mark.e2e
@pytest.mark.gateway(log_level="debug", log_dir=str(_LOG_DIR), extra_args=["--dp-aware"])
@pytest.mark.workers(count=1, extra_engine_args=["--engine", "realistic", "--grpc-dp-size", "2"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestDpAwareRankPinningOnTheMockWorker:
    """The engine class's two facts on the mock worker's two ranks, on the CPU row.

    The mock advertises ``dp_size=2`` in its server info, places a pinned
    request on its rank and logs ``dp_rank=<n>`` per request as an engine's
    servicer does; an unpinned request would log ``dp_rank=None``.
    """

    def test_requests_pinned_to_both_ranks(self, setup_backend):
        _assert_requests_pinned_to_both_ranks(setup_backend, _MOCK_WORKER_LOG_GLOB)
