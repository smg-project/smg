"""Unit tests for reading per-GPU used memory, which worker restarts wait on."""

from __future__ import annotations

import json
import subprocess

import pytest
from infra import process_utils
from infra.process_utils import gpu_memory_used_mib

# amd-smi on an MI355X node, trimmed to the fields read. amd-smi numbers the
# GPUs by PCI address, HIP in KFD node order: HIP's GPU 0 is amd-smi's GPU 3.
_AMD_SMI_LIST = [
    {"gpu": 0, "bdf": "0000:05:00.0", "node_id": 3},
    {"gpu": 1, "bdf": "0000:15:00.0", "node_id": 5},
    {"gpu": 2, "bdf": "0000:65:00.0", "node_id": 4},
    {"gpu": 3, "bdf": "0000:75:00.0", "node_id": 2},
]
_AMD_SMI_METRIC = {
    "gpu_data": [
        {"gpu": gpu, "mem_usage": {"used_vram": {"value": mib, "unit": "MB"}}}
        for gpu, mib in [(0, 1000), (1, 1001), (2, 1002), (3, 261326)]
    ]
}
_AMD_SMI = {
    "amd-smi list": json.dumps(_AMD_SMI_LIST),
    "amd-smi metric": json.dumps(_AMD_SMI_METRIC),
}


@pytest.fixture(autouse=True)
def _all_gpus_visible(monkeypatch):
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)


def _fake_run(outputs: dict[str, str]):
    def run(cmd, **kwargs):
        key = cmd[0] if cmd[0] == "nvidia-smi" else " ".join(cmd[:2])
        if key not in outputs:
            raise FileNotFoundError(cmd[0])
        return subprocess.CompletedProcess(cmd, 0, stdout=outputs[key], stderr="")

    return run


def test_nvidia_smi_is_read_first(monkeypatch):
    outputs = {"nvidia-smi": "0, 1200\n1, 30\n", **_AMD_SMI}
    monkeypatch.setattr(process_utils.subprocess, "run", _fake_run(outputs))

    assert gpu_memory_used_mib([0, 1]) == {0: 1200, 1: 30}


def test_amd_smi_readings_are_numbered_as_hip_numbers_the_gpus(monkeypatch):
    monkeypatch.setattr(process_utils.subprocess, "run", _fake_run(_AMD_SMI))

    assert gpu_memory_used_mib([0, 1]) == {0: 261326, 1: 1000}


def test_visible_devices_renumber_the_gpus(monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,3")
    monkeypatch.setattr(process_utils.subprocess, "run", _fake_run(_AMD_SMI))

    assert gpu_memory_used_mib([0, 1]) == {0: 1002, 1: 1001}


@pytest.mark.parametrize("amd_smi", [{}, {"amd-smi list": "not json", "amd-smi metric": "{}"}])
def test_no_reading_without_a_working_tool(monkeypatch, amd_smi):
    monkeypatch.setattr(process_utils.subprocess, "run", _fake_run(amd_smi))

    assert gpu_memory_used_mib([0]) is None


def test_total_readback_budget_decreases_across_both_tools(monkeypatch):
    clock = [0.0]
    calls = []
    monkeypatch.setattr(process_utils.time, "monotonic", lambda: clock[0])

    def run(cmd, **kwargs):
        calls.append((cmd[:2], kwargs["timeout"]))
        if cmd[0] == "nvidia-smi":
            clock[0] += 10
            raise FileNotFoundError(cmd[0])
        clock[0] += 15 if cmd[1] == "list" else 3
        return _fake_run(_AMD_SMI)(cmd, **kwargs)

    monkeypatch.setattr(process_utils.subprocess, "run", run)
    assert gpu_memory_used_mib([0, 1], timeout=30) == {0: 261326, 1: 1000}
    assert [timeout for _, timeout in calls] == [15, 15, 5]
    assert [cmd[0] for cmd, _ in calls] == ["nvidia-smi", "amd-smi", "amd-smi"]


def test_expired_total_budget_cannot_return_a_successful_read_or_start_fallback(monkeypatch):
    clock = [0.0]
    calls = []
    monkeypatch.setattr(process_utils.time, "monotonic", lambda: clock[0])

    def run(cmd, **kwargs):
        calls.append(kwargs["timeout"])
        clock[0] += 6
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(process_utils.subprocess, "run", run)
    assert gpu_memory_used_mib([0], timeout=5) is None
    assert calls == [5]


@pytest.mark.parametrize("raw", [True, False, 12.5, -1, "01", "-1", "1.5", " 1", "1e2"])
@pytest.mark.parametrize("field", ["node_id", "listed_gpu", "metric_gpu", "used_vram"])
def test_strict_amd_parser_rejects_invalid_raw_identity_and_memory(monkeypatch, raw, field):
    listed = [{"gpu": 0, "node_id": 2}]
    metric = [{"gpu": 0, "mem_usage": {"used_vram": {"value": 12}}}]
    if field == "node_id":
        listed[0]["node_id"] = raw
    elif field == "listed_gpu":
        listed[0]["gpu"] = raw
    elif field == "metric_gpu":
        metric[0]["gpu"] = raw
    else:
        metric[0]["mem_usage"]["used_vram"]["value"] = raw
    outputs = {"amd-smi list": json.dumps(listed), "amd-smi metric": json.dumps(metric)}
    monkeypatch.setattr(process_utils.subprocess, "run", _fake_run(outputs))
    assert gpu_memory_used_mib([0], timeout=30) is None


@pytest.mark.parametrize("raw", [12, "12", 0, "0"])
def test_strict_amd_parser_accepts_integer_and_canonical_decimal_memory(monkeypatch, raw):
    outputs = {
        "amd-smi list": json.dumps([{"gpu": "0", "node_id": "2"}]),
        "amd-smi metric": json.dumps([{"gpu": "0", "mem_usage": {"used_vram": {"value": raw}}}]),
    }
    monkeypatch.setattr(process_utils.subprocess, "run", _fake_run(outputs))
    assert gpu_memory_used_mib([0], timeout=30) == {0: int(raw)}


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate_node",
        "duplicate_gpu",
        "unknown_metric_gpu",
        "duplicate_metric_gpu",
        "negative_requested_gpu",
        "negative_visibility",
        "duplicate_visibility",
        "visibility_out_of_range",
    ],
)
def test_strict_amd_mapping_cannot_select_an_ambiguous_or_wrong_gpu(monkeypatch, defect):
    listed = [{"gpu": 0, "node_id": 2}, {"gpu": 1, "node_id": 3}]
    metrics = [{"gpu": g, "mem_usage": {"used_vram": 12}} for g in [0, 1]]
    gpu_ids = [0]
    if defect == "duplicate_node":
        listed[1]["node_id"] = 2
    elif defect == "duplicate_gpu":
        listed[1]["gpu"] = 0
    elif defect == "unknown_metric_gpu":
        metrics[0]["gpu"] = 7
    elif defect == "duplicate_metric_gpu":
        metrics[1]["gpu"] = 0
    elif defect == "negative_requested_gpu":
        gpu_ids = [-1]
    else:
        monkeypatch.setenv(
            "HIP_VISIBLE_DEVICES",
            {
                "negative_visibility": "-1",
                "duplicate_visibility": "0,0",
                "visibility_out_of_range": "2",
            }[defect],
        )
    outputs = {"amd-smi list": json.dumps(listed), "amd-smi metric": json.dumps(metrics)}
    monkeypatch.setattr(process_utils.subprocess, "run", _fake_run(outputs))
    assert gpu_memory_used_mib(gpu_ids, timeout=30) is None
