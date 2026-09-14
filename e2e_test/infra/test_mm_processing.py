"""Unit tests for the E2E_MM_PROCESSING lane switch and the metric readback."""

import pytest
from infra import mm_processing
from infra.constants import ENV_MM_PROCESSING, get_mm_processing


def test_unset_or_blank_is_router_path(monkeypatch):
    monkeypatch.delenv(ENV_MM_PROCESSING, raising=False)
    assert get_mm_processing() is None
    monkeypatch.setenv(ENV_MM_PROCESSING, "  ")
    assert get_mm_processing() is None


def test_worker_is_case_insensitive(monkeypatch):
    monkeypatch.setenv(ENV_MM_PROCESSING, " Worker ")
    assert get_mm_processing() == "worker"


def test_unknown_value_raises(monkeypatch):
    monkeypatch.setenv(ENV_MM_PROCESSING, "sidecar")
    with pytest.raises(ValueError, match="E2E_MM_PROCESSING='sidecar'"):
        get_mm_processing()


@pytest.mark.parametrize(
    ("lane", "vllm", "expandable", "expected"),
    [
        ("worker", True, True, ("worker", "auto_uniform")),
        ("worker", True, False, ("router", "model_not_opted_in")),
        ("worker", False, True, ("router", None)),
        (None, True, True, ("router", None)),
    ],
)
def test_expected_path_matrix(monkeypatch, lane, vllm, expandable, expected):
    monkeypatch.setattr(mm_processing, "get_mm_processing", lambda: lane)
    monkeypatch.setattr(mm_processing, "is_vllm", lambda: vllm)
    assert mm_processing.expected_mm_processing(worker_expandable=expandable) == expected


def test_parse_samples_reads_labels_and_values():
    text = (
        "# HELP smg_mm_processing_total Multimodal requests by processing location\n"
        "# TYPE smg_mm_processing_total counter\n"
        'smg_mm_processing_total{model="Qwen/Qwen3-VL-8B-Instruct",mode="worker",'
        'reason="auto_uniform"} 3\n'
        'smg_mm_processing_total{model="m",mode="router",reason="model_not_opted_in"} 1.5\n'
        'smg_requests_total{model="m"} 9\n'
    )
    samples = mm_processing.parse_mm_processing_samples(text)
    assert samples == [
        ({"model": "Qwen/Qwen3-VL-8B-Instruct", "mode": "worker", "reason": "auto_uniform"}, 3.0),
        ({"model": "m", "mode": "router", "reason": "model_not_opted_in"}, 1.5),
    ]
