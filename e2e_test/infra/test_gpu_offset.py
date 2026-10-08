"""Unit tests for the ``E2E_GPU_OFFSET`` lane knob."""

import pytest
from infra.constants import ENV_GPU_OFFSET, get_gpu_offset


def test_gpu_offset_defaults_to_zero_and_reads_the_env(monkeypatch):
    monkeypatch.delenv(ENV_GPU_OFFSET, raising=False)
    assert get_gpu_offset() == 0
    monkeypatch.setenv(ENV_GPU_OFFSET, " 2 ")
    assert get_gpu_offset() == 2
    monkeypatch.setenv(ENV_GPU_OFFSET, "")
    assert get_gpu_offset() == 0


def test_gpu_offset_rejects_negative_values(monkeypatch):
    monkeypatch.setenv(ENV_GPU_OFFSET, "-1")
    with pytest.raises(ValueError, match=ENV_GPU_OFFSET):
        get_gpu_offset()
