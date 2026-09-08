"""Every worker must deliver load reports to the gateway.

The gateway polls each worker's load endpoint (gRPC GetLoads, or the metrics
scrape for HTTP SGLang) every ``--load-monitor-interval`` seconds and exposes
the fleet view at ``GET /loads``. Load-aware policies and overload protection
act on that view, but a worker whose poll fails silently degrades to in-flight
counting, so a broken load path shows up nowhere else: a renamed field in the
SGLang servicer made every GetLoads on a disaggregated worker raise for weeks
without a red lane (#2458). This module asserts that every registered worker,
including prefill and decode workers, actually reports.

Usage:
    E2E_RUNTIME=sglang pytest e2e_test/router/test_loads.py -v
"""

from __future__ import annotations

import logging
import threading
import time

import httpx
import pytest

logger = logging.getLogger(__name__)

# Short poll interval so a report lands within seconds; the gateway default is 10s.
# One second also keeps a burst observable: a small model on an H100 finishes
# sixteen 512-token generations in about a second, inside a single wider poll.
LOAD_MONITOR_INTERVAL_SECS = 1
_GATEWAY_ARGS = ["--load-monitor-interval", str(LOAD_MONITOR_INTERVAL_SECS)]
_REPORT_TIMEOUT_SECS = 8 * LOAD_MONITOR_INTERVAL_SECS


def _worker_urls(gateway) -> set[str]:
    urls = {w.url for w in gateway.list_workers(strict=True)}
    assert urls, "gateway reports no workers"
    return urls


def _fleet_loads(gateway) -> list[dict]:
    resp = httpx.get(f"{gateway.base_url}/loads", timeout=5.0)
    assert resp.status_code == 200, f"GET /loads returned {resp.status_code}: {resp.text}"
    body = resp.json()
    assert isinstance(body.get("loads"), list), f"unexpected /loads body: {body}"
    return body["loads"]


def _wait_for_reports(gateway, expected: set[str], timeout: float) -> list[dict]:
    """Poll /loads until every expected worker has at least one rank report."""
    deadline = time.monotonic() + timeout
    while True:
        loads = _fleet_loads(gateway)
        reported: set[str] = {
            w for w in (entry.get("worker") for entry in loads) if isinstance(w, str)
        }
        if expected <= reported:
            return loads
        if time.monotonic() >= deadline:
            pytest.fail(
                f"workers never reported load within {timeout:.0f}s: "
                f"missing={sorted(expected - reported)} reported={sorted(reported)}"
            )
        time.sleep(0.5)


def _assert_report_is_sane(entry: dict) -> None:
    """Every rank report carries the counters the policies act on."""
    for field in ("num_running_reqs", "num_waiting_reqs", "num_total_reqs"):
        assert isinstance(entry.get(field), int) and entry[field] >= 0, (
            f"{field} missing or negative in {entry}"
        )
    assert 0.0 <= entry.get("token_usage", 0.0) <= 1.0, f"token_usage out of range: {entry}"
    assert entry.get("dp_rank", 0) >= 0, f"negative dp_rank: {entry}"


@pytest.mark.engine("sglang", "vllm", "tokenspeed")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.gateway(extra_args=_GATEWAY_ARGS)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestWorkerLoadReports:
    """A regular worker reports load, and the report follows real traffic."""

    def test_every_worker_reports_load(self, setup_backend):
        _, _, _, gateway = setup_backend

        loads = _wait_for_reports(gateway, _worker_urls(gateway), _REPORT_TIMEOUT_SECS)

        for entry in loads:
            logger.info("load report: %s", entry)
            assert entry.get("worker_type") == "regular", f"unexpected worker type: {entry}"
            _assert_report_is_sane(entry)

    def test_report_reflects_in_flight_requests(self, setup_backend):
        """While a burst is in flight, at least one poll must see running or waiting work.

        A report that never moves off zero would still pass the presence check
        above; this proves the counters come from the engine, not a default.
        The burst is sized to outlast several poll intervals on a fast engine.
        """
        _, model, client, gateway = setup_backend
        expected = _worker_urls(gateway)
        _wait_for_reports(gateway, expected, _REPORT_TIMEOUT_SECS)

        errors: list[BaseException] = []

        def _long_request() -> None:
            try:
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": "Write a long story about a lighthouse."}
                    ],
                    max_tokens=1536,
                    temperature=0.0,
                    extra_body={"ignore_eos": True},
                )
            except BaseException as exc:  # surfaced below; never swallow silently
                errors.append(exc)

        threads = [threading.Thread(target=_long_request, daemon=True) for _ in range(32)]
        for t in threads:
            t.start()

        observed = False
        deadline = time.monotonic() + 30.0
        try:
            while time.monotonic() < deadline and not observed:
                for entry in _fleet_loads(gateway):
                    if entry["num_running_reqs"] + entry["num_waiting_reqs"] > 0:
                        logger.info("in-flight work visible in report: %s", entry)
                        observed = True
                        break
                time.sleep(0.25)
        finally:
            for t in threads:
                t.join(timeout=300)

        assert not errors, f"burst requests failed: {errors[:3]}"
        assert observed, "no load report showed running or waiting work during a 32-request burst"


@pytest.mark.engine("sglang", "vllm", "tokenspeed")
@pytest.mark.gpu(2)
@pytest.mark.e2e
@pytest.mark.model("meta-llama/Llama-3.2-1B-Instruct")
@pytest.mark.gateway(extra_args=_GATEWAY_ARGS)
@pytest.mark.parametrize("setup_backend", ["pd_grpc"], indirect=True)
class TestDisaggregatedWorkerLoadReports:
    """Prefill and decode workers both report load through the same path."""

    def test_prefill_and_decode_workers_report_load(self, setup_backend):
        _, _, _, gateway = setup_backend

        loads = _wait_for_reports(gateway, _worker_urls(gateway), _REPORT_TIMEOUT_SECS)

        by_type: dict[str, list[dict]] = {}
        for entry in loads:
            logger.info("load report: %s", entry)
            _assert_report_is_sane(entry)
            by_type.setdefault(str(entry.get("worker_type")), []).append(entry)
        assert set(by_type) == {"prefill", "decode"}, (
            f"expected reports from both PD roles, got {sorted(by_type)}"
        )
