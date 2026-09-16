"""Where a multimodal request was processed (router or worker), read back from
the gateway's ``smg_mm_processing_total`` counter."""

from __future__ import annotations

import re

import requests

from .constants import MM_PROCESSING_WORKER, get_mm_processing, is_vllm

MM_PROCESSING_METRIC = "smg_mm_processing_total"

_SAMPLE = re.compile(rf"^{MM_PROCESSING_METRIC}\{{(?P<labels>[^}}]*)\}}\s+(?P<value>\S+)", re.M)
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse_mm_processing_samples(text: str) -> list[tuple[dict[str, str], float]]:
    """(labels, value) for every ``smg_mm_processing_total`` sample in a scrape."""
    return [(dict(_LABEL.findall(m["labels"])), float(m["value"])) for m in _SAMPLE.finditer(text)]


def mm_processing_samples(gateway) -> list[tuple[dict[str, str], float]]:
    response = requests.get(f"{gateway.metrics_url}/metrics", timeout=10)
    response.raise_for_status()
    return parse_mm_processing_samples(response.text)


def expected_mm_processing(*, worker_expandable: bool) -> tuple[str, str | None]:
    """The (mode, reason) the gateway must record for this lane and model.

    Only vLLM gRPC workers advertise worker-side processing, and only models
    whose placeholder anchor a vLLM worker can expand take the worker path;
    every other combination stays on the router path.
    """
    if get_mm_processing() == MM_PROCESSING_WORKER and is_vllm():
        if worker_expandable:
            return "worker", "auto_uniform"
        return "router", "model_not_opted_in"
    return "router", None


def assert_mm_processing(gateway, *, worker_expandable: bool) -> None:
    """Every multimodal request this gateway served took the lane's expected path."""
    mode, reason = expected_mm_processing(worker_expandable=worker_expandable)
    samples = [(labels, value) for labels, value in mm_processing_samples(gateway) if value > 0]
    assert samples, f"gateway recorded no {MM_PROCESSING_METRIC} samples"
    modes = {labels["mode"] for labels, _ in samples}
    assert modes == {mode}, f"expected every multimodal request on the {mode} path, got {samples}"
    if reason is not None:
        reasons = {labels["reason"] for labels, _ in samples}
        assert reasons == {reason}, f"expected resolution reason {reason!r}, got {samples}"
