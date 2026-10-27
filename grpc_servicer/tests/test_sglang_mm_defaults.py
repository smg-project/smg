"""Automatic worker defaults preserve explicit overrides and engine precision."""

from types import SimpleNamespace

import pytest
from smg_grpc_servicer.sglang import mm_broadcast, mm_defaults, rust


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_dtype_uses_effective_engine_config(dtype):
    """Verify dtype uses effective engine config."""
    assert mm_defaults.encoder_dtype(None, lambda: "torch." + dtype) == dtype


def test_dtype_override_does_not_load_model():
    """Verify dtype override does not load model."""

    def unavailable():
        """Fail if an explicit dtype override attempts to load model configuration."""
        raise AssertionError("explicit dtype must not load model")

    assert mm_defaults.encoder_dtype("float16", unavailable) == "float16"
    with pytest.raises(ValueError):
        mm_defaults.encoder_dtype("fp8", unavailable)


def test_unknown_dtype_retains_float32_but_config_errors_propagate():
    """Verify unknown dtype retains float32 but config errors propagate."""
    assert mm_defaults.encoder_dtype(None, lambda: "float8_e4m3fn") == "float32"

    def invalid_config():
        """Simulate a model configuration error that must propagate."""
        raise RuntimeError("invalid model")

    with pytest.raises(RuntimeError, match="invalid model"):
        mm_defaults.encoder_dtype(None, invalid_config)


def test_worker_thread_defaults_and_explicit_overrides():
    """Verify worker thread defaults and explicit overrides."""
    env = {}
    mm_defaults.configure_threads(env)
    assert env == {
        "SMG_SERVICER_WORKER_THREADS": "8",
        "SMG_MM_THREADS": "32",
        "SMG_MM_IMAGE_CONCURRENCY": "8",
    }
    env["SMG_MM_THREADS"] = "4"
    env["SMG_SERVICER_WORKER_THREADS"] = "1"
    mm_defaults.configure_threads(env)
    assert env["SMG_MM_THREADS"] == "4"
    assert env["SMG_SERVICER_WORKER_THREADS"] == "1"


def test_worker_launch_derives_dtype_and_admission(monkeypatch):
    """Verify worker launch derives dtype and admission."""
    monkeypatch.setenv("SMG_SGLANG_MM_PROCESSOR", "smg")
    for key in [
        "SMG_SGLANG_MM_DTYPE",
        "SMG_SGLANG_MM_MAX_INFLIGHT",
        "SMG_SERVICER_WORKER_THREADS",
        "SMG_MM_THREADS",
        "SMG_MM_IMAGE_CONCURRENCY",
    ]:
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    monkeypatch.setattr(rust, "_model_config", lambda _: SimpleNamespace(dtype="torch.bfloat16"))
    args = SimpleNamespace(model_path="model", limit_mm_data_per_request={})
    options = rust.SglangLauncher().server_extras(args, "tokenizer")
    assert options["smg_media_processor"]["encoder_dtype"] == "bfloat16"
    assert options["smg_media_processor"]["max_inflight"] == 64
    import os

    assert os.environ["SMG_MM_THREADS"] == "32"
    assert os.environ["SMG_SERVICER_WORKER_THREADS"] == "8"


def test_router_mode_leaves_threads_and_model_config_untouched(monkeypatch):
    """Verify router mode leaves threads and model config untouched."""
    monkeypatch.delenv("SMG_SGLANG_MM_PROCESSOR", raising=False)

    def unexpected(*args):
        """Fail if router mode reads worker-only model configuration."""
        raise AssertionError("worker-only defaults reached router mode")

    monkeypatch.setattr(rust, "_model_config", unexpected)
    monkeypatch.setattr(mm_defaults, "configure_threads", unexpected)
    assert rust.SglangLauncher().server_extras(SimpleNamespace(), None) == {}


def test_shm_auto_requires_worker_mode_and_honors_opt_out(monkeypatch):
    """Verify shm auto requires worker mode and honors opt out."""
    monkeypatch.delenv(mm_broadcast.ENV, raising=False)
    monkeypatch.delenv("SMG_SGLANG_MM_PROCESSOR", raising=False)
    assert not mm_broadcast.enabled()
    monkeypatch.setenv("SMG_SGLANG_MM_PROCESSOR", "smg")
    assert mm_broadcast.enabled()
    monkeypatch.setenv(mm_broadcast.ENV, "0")
    assert not mm_broadcast.enabled()
