"""The PD topology sweep's restart skips: which sole-worker restart names an engine-side reason.

A vLLM decode over NIXL forgets a dead prefill only while it serves a request,
so the sweep skips the sole prefill's restart on that lane alone; every other
lane, backend and role keeps the restart.
"""

from __future__ import annotations

import importlib

import pytest
from infra import constants


def _sweep(monkeypatch, runtime: str, kv_backend: str | None):
    monkeypatch.setattr(constants, "_RUNTIME_CACHE", None)
    monkeypatch.setenv(constants.ENV_RUNTIME, runtime)
    monkeypatch.delenv(constants.ENV_KV_BACKEND, raising=False)
    if kv_backend is None:
        monkeypatch.delenv(constants.ENV_VLLM_KV_BACKEND, raising=False)
    else:
        monkeypatch.setenv(constants.ENV_VLLM_KV_BACKEND, kv_backend)
    return importlib.import_module("router.test_pd_topologies")


@pytest.mark.parametrize(
    ("runtime", "kv_backend", "role", "skipped"),
    [
        ("vllm", "nixl", "prefill", True),
        ("vllm", None, "prefill", True),  # NIXL is the vLLM lane's default backend
        ("vllm", "nixl", "decode", False),
        ("vllm", "mooncake", "prefill", False),
        ("vllm", "moriio", "prefill", False),
        ("sglang", None, "prefill", False),
        ("tokenspeed", None, "prefill", False),
    ],
)
def test_only_the_vllm_nixl_sole_prefill_restart_is_skipped(
    monkeypatch, runtime, kv_backend, role, skipped
):
    sweep = _sweep(monkeypatch, runtime, kv_backend)
    assert sweep._nixl_decodes_keep_a_dead_prefills_memory(role) is skipped


def test_the_lane_wide_backend_setting_counts_too(monkeypatch):
    sweep = _sweep(monkeypatch, "vllm", "nixl")
    monkeypatch.setenv(constants.ENV_KV_BACKEND, "mooncake")
    assert sweep._nixl_decodes_keep_a_dead_prefills_memory("prefill") is False


def test_the_skip_reason_names_the_engine_side_cause(monkeypatch):
    sweep = _sweep(monkeypatch, "vllm", "nixl")
    assert "startup memory check" in sweep._NIXL_HELD_MEMORY
    assert "while it serves a request" in sweep._NIXL_HELD_MEMORY
