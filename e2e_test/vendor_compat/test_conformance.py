"""Vendor-truth conformance: replay recorded vendor behavior clusters against
the lane's real gateway + engine and assert structural conformance.

Ground truth is the checked-in distillation of full-matrix recordings against
the real OpenAI Responses / Anthropic Messages APIs
(``vendor_probe/baselines/<date>/``). One test per (surface, cluster) so a
failure names the exact vendor behavior that regressed; the comparison logic
(status class, field-path skeletons, SSE event framing, error envelopes,
volatile-value normalization) is shared with ``vendor_probe.compat_diff``.

Divergences already present when the baseline was cut are grandfathered in
``known_divergences.jsonl`` and report as xfail; anything beyond that state —
a new divergent cluster or a known one getting worse — fails, with a
ready-to-paste ``TO-TRIAGE`` allowlist line in the failure message.

Content-dependent behavior (which output items the model generated, delta
arity, incomplete_details) never fails: the local model's content legitimately
differs from the vendor's. Structure is compared for real.
"""

from __future__ import annotations

import logging

import pytest
from vendor_compat import conformance

logger = logging.getLogger(__name__)


def _check_cluster(vendor_replay, cluster_id: str) -> None:
    provider, records, sel = vendor_replay
    cluster = conformance.cluster_by_id(provider)[cluster_id]
    outcome = conformance.judge_cluster(provider, cluster, sel[cluster_id], records)
    disposition = outcome["disposition"]

    if disposition == "conformant":
        logger.info("%s %s: %s", provider, cluster_id, outcome["verdict"])
        return
    if disposition == "no-coverage":
        pytest.skip(
            f"{cluster_id}: no comparable replay (gated/skipped dependencies or "
            f"probe ids missing from the current matrix)"
        )
    if disposition == "expected-divergence":
        allowed = outcome["allowlisted"] or {}
        pytest.xfail(
            f"known divergence {cluster_id} ({outcome['verdict']}"
            + (f", allowlisted as {allowed.get('verdict')}" if allowed else "")
            + f"): {allowed.get('note') or 'grandfathered in known_divergences.jsonl'}"
        )
    # new-divergence / severity-escalation
    pytest.fail(conformance.failure_message(provider, cluster, outcome))


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("Qwen/Qwen2.5-14B-Instruct")
@pytest.mark.gateway(extra_args=["--tool-call-parser", "qwen", "--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
@pytest.mark.parametrize("vendor_replay", ["openai"], indirect=True)
class TestOpenAIResponsesConformance:
    """OpenAI Responses surface vs the recorded vendor behavior clusters."""

    @pytest.mark.parametrize("cluster_id", conformance.cluster_ids("openai"))
    def test_cluster(self, vendor_replay, cluster_id):
        _check_cluster(vendor_replay, cluster_id)


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("Qwen/Qwen2.5-14B-Instruct")
@pytest.mark.gateway(extra_args=["--tool-call-parser", "qwen", "--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
@pytest.mark.parametrize("vendor_replay", ["anthropic"], indirect=True)
class TestAnthropicMessagesConformance:
    """Anthropic Messages surface vs the recorded vendor behavior clusters."""

    @pytest.mark.parametrize("cluster_id", conformance.cluster_ids("anthropic"))
    def test_cluster(self, vendor_replay, cluster_id):
        _check_cluster(vendor_replay, cluster_id)
