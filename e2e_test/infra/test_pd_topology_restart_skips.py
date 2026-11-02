"""The PD topology sweep's restart skips: which sole-worker restart names an engine-side reason.

A vLLM decode over NIXL on CUDA forgets a dead prefill only while it serves a
request, so the sweep skips the sole prefill's restart on that lane alone;
every other lane, backend, role and vendor keeps the restart.
"""

from __future__ import annotations

import importlib
import shutil

import pytest
from infra import constants


def _sweep(monkeypatch, runtime: str, kv_backend: str | None, vendor: str = "nvidia"):
    monkeypatch.setattr(constants, "_RUNTIME_CACHE", None)
    monkeypatch.setenv(constants.ENV_RUNTIME, runtime)
    monkeypatch.setenv(constants.ENV_GPU_VENDOR, vendor)
    monkeypatch.delenv(constants.ENV_KV_BACKEND, raising=False)
    if kv_backend is None:
        monkeypatch.delenv(constants.ENV_VLLM_KV_BACKEND, raising=False)
    else:
        monkeypatch.setenv(constants.ENV_VLLM_KV_BACKEND, kv_backend)
    return importlib.import_module("router.test_pd_topologies")


@pytest.mark.parametrize(
    ("runtime", "kv_backend", "role", "vendor", "skipped"),
    [
        ("vllm", "nixl", "prefill", "nvidia", True),
        ("vllm", None, "prefill", "nvidia", True),  # NIXL is the vLLM lane's default backend
        ("vllm", "nixl", "decode", "nvidia", False),
        ("vllm", "nixl", "prefill", "amd", False),  # the ROCm lane releases the region
        ("vllm", "mooncake", "prefill", "nvidia", False),
        ("vllm", "moriio", "prefill", "nvidia", False),
        ("sglang", None, "prefill", "nvidia", False),
        ("tokenspeed", None, "prefill", "nvidia", False),
    ],
)
def test_only_the_cuda_vllm_nixl_sole_prefill_restart_is_skipped(
    monkeypatch, runtime, kv_backend, role, vendor, skipped
):
    sweep = _sweep(monkeypatch, runtime, kv_backend, vendor)
    assert sweep._nixl_decodes_keep_a_dead_prefills_memory(role) is skipped


def test_the_lane_wide_backend_setting_counts_too(monkeypatch):
    sweep = _sweep(monkeypatch, "vllm", "nixl")
    monkeypatch.setenv(constants.ENV_KV_BACKEND, "mooncake")
    assert sweep._nixl_decodes_keep_a_dead_prefills_memory("prefill") is False


def test_the_vendor_is_read_from_the_job_or_from_the_smi_on_the_path(monkeypatch):
    monkeypatch.setenv(constants.ENV_GPU_VENDOR, " AMD ")
    assert constants.get_gpu_vendor() == "amd"
    monkeypatch.setenv(constants.ENV_GPU_VENDOR, "tpu")
    with pytest.raises(ValueError, match="E2E_GPU_VENDOR"):
        constants.get_gpu_vendor()
    monkeypatch.delenv(constants.ENV_GPU_VENDOR)
    found = {"amd-smi"}
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}" if name in found else None)
    assert constants.get_gpu_vendor() == "amd"
    found.add("nvidia-smi")
    assert constants.get_gpu_vendor() == "nvidia"
    found.clear()
    assert constants.get_gpu_vendor() == "nvidia"


def test_the_skip_reason_names_the_engine_side_cause_and_when_to_lift_it(monkeypatch):
    sweep = _sweep(monkeypatch, "vllm", "nixl")
    assert "startup memory check" in sweep._NIXL_HELD_MEMORY
    assert "while it serves a request" in sweep._NIXL_HELD_MEMORY
    assert "0.31.0" in sweep._NIXL_HELD_MEMORY
    assert "lift the skip" in sweep._NIXL_HELD_MEMORY
