"""Every OpenAI ``reasoning.effort`` tier must be accepted on the Responses API.

OpenAI defines seven tiers: ``none``, ``minimal``, ``low``, ``medium``,
``high``, ``xhigh`` and ``max``. The Responses request type used to know four
of them, so ``none``, ``xhigh`` and ``max`` failed deserialization with a 422
before any handler ran while the same values worked on Chat Completions
(#2410). Harmony models clamp the outer tiers inward, so every tier must
produce a normal response on gpt-oss.

Usage:
    E2E_RUNTIME=sglang pytest e2e_test/responses/test_reasoning_effort.py -v
"""

from __future__ import annotations

import logging

import openai
import pytest

logger = logging.getLogger(__name__)

# OpenAI's tiers, lowest to highest. ``none`` is distinct from omitting the field.
REASONING_EFFORT_TIERS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]


@pytest.mark.engine("sglang")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model("openai/gpt-oss-20b")
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestReasoningEffortTiers:
    """Each tier reaches the model and yields output; an unknown tier is rejected."""

    @pytest.mark.parametrize("effort", REASONING_EFFORT_TIERS)
    def test_tier_is_accepted(self, model, api_client, effort):
        response = api_client.responses.create(
            model=model,
            input="What is 12 times 12? Reply with the number only.",
            reasoning={"effort": effort},
            max_output_tokens=512,
        )

        item_types = [item.type for item in response.output]
        logger.info("effort=%s status=%s output=%s", effort, response.status, item_types)
        assert response.output, f"effort={effort!r} produced no output items"
        assert {"reasoning", "message"} & set(item_types), (
            f"effort={effort!r} produced neither reasoning nor a message: {item_types}"
        )

    def test_unknown_tier_is_rejected(self, model, api_client):
        with pytest.raises(openai.APIStatusError) as exc_info:
            api_client.responses.create(
                model=model,
                input="Reply with OK.",
                reasoning={"effort": "extreme"},
                max_output_tokens=32,
            )

        assert exc_info.value.status_code in (400, 422), exc_info.value
