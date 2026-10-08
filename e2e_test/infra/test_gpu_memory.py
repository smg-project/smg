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
