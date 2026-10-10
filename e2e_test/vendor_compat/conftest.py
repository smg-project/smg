"""Fixtures for the vendor-truth conformance suite.

``vendor_replay`` runs the whole selected probe batch once per test class
(dependency chains need a single ordered run — per-test requests would both
break placeholder resolution and multiply wall time), then the parametrized
per-cluster tests assert on the recorded results.

Env knobs:
- ``VENDOR_COMPAT_FULL=1``            replay every baseline member (nightly)
- ``VENDOR_COMPAT_CONCURRENCY`` (16)  runner concurrency vs the lane gateway
- ``VENDOR_COMPAT_TIMEOUT`` (120)     per-request timeout, seconds
- ``VENDOR_COMPAT_STREAM_TIMEOUT`` (240)  per-stream timeout, seconds
"""

from __future__ import annotations

import logging
import os

import pytest
from vendor_compat import conformance

logger = logging.getLogger(__name__)

# Guardrails: a green suite must mean "the gateway conformed", never "nothing
# actually ran". Tunable only here, on purpose.
_MAX_STALE_FRACTION = 0.2  # baseline members missing from the probe matrix
_MIN_RESPONDED_FRACTION = 0.5  # attempted probes that got an HTTP status


@pytest.fixture(scope="class")
def vendor_replay(request, setup_backend):
    """Replay the selected vendor probes against the lane's gateway.

    Parametrized (indirect) with the provider surface: ``openai`` replays the
    Responses matrix via the ``smg-openai`` adapter, ``anthropic`` the
    Messages matrix via ``smg-anthropic`` — both against the same local
    gateway, exactly like the recorded replays in ``vendor_probe/README.md``.

    Yields ``(provider, records_by_probe_id, selection)``.
    """
    provider = request.param
    _, model_path, _client, gateway = setup_backend

    records, sel, missing = conformance.run_replay(
        provider,
        gateway.base_url,
        model_path,
        concurrency=int(os.environ.get("VENDOR_COMPAT_CONCURRENCY", "16")),
        timeout=float(os.environ.get("VENDOR_COMPAT_TIMEOUT", "120")),
        stream_timeout=float(os.environ.get("VENDOR_COMPAT_STREAM_TIMEOUT", "240")),
    )
    health = conformance.replay_health(records)
    selected = sum(len(v) for v in sel.values())
    logger.info(
        "vendor_compat[%s]: %d clusters, %d selected probes (%d replayed incl. deps), "
        "%d stale, health=%s",
        provider,
        len(sel),
        selected,
        health["total"],
        len(missing),
        health,
    )

    if selected and len(missing) / selected > _MAX_STALE_FRACTION:
        pytest.fail(
            f"{len(missing)}/{selected} selected baseline probes no longer exist in the "
            f"probe matrix — the checked-in baseline is stale; refresh it "
            f"(vendor_probe/README.md, 'Refreshing the baselines')"
        )
    if health["attempted"] and health["responded_fraction"] < _MIN_RESPONDED_FRACTION:
        pytest.fail(
            f"replay unhealthy: only {health['responded']}/{health['attempted']} attempted "
            f"probes got an HTTP response from {gateway.base_url} — gateway/worker problem, "
            f"not a conformance signal"
        )
    yield provider, records, sel
