"""Unit tests for the worker-side media processor configuration (engine-free).

Run with: pytest grpc_servicer/tests/test_vllm_mm_processor.py
"""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

# Import the module directly to avoid pulling vllm via the package __init__
_MODULE_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "mm_processor.py"
_spec = importlib.util.spec_from_file_location("mm_processor", _MODULE_PATH)
mm_processor = importlib.util.module_from_spec(_spec)
# Registered so dataclass decorators can resolve the module by name.
sys.modules[_spec.name] = mm_processor
_spec.loader.exec_module(mm_processor)


@dataclass
class _Item:
    modality: str
    url: str


class TestResolveMode:
    def test_defaults_to_off(self):
        assert mm_processor.resolve_mm_processor_mode({}) == "off"
        assert mm_processor.resolve_mm_processor_mode({"SMG_VLLM_MM_PROCESSOR": ""}) == "off"

    def test_normalizes_case_and_whitespace(self):
        assert mm_processor.resolve_mm_processor_mode({"SMG_VLLM_MM_PROCESSOR": " InProcess "}) == (
            "inprocess"
        )

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="SMG_VLLM_MM_PROCESSOR='sidecar'"):
            mm_processor.resolve_mm_processor_mode({"SMG_VLLM_MM_PROCESSOR": "sidecar"})


class TestEnvInt:
    def test_default_when_unset(self):
        assert mm_processor.env_int({}, "K", 7) == 7
        assert mm_processor.env_int({"K": "  "}, "K", 7) == 7

    def test_parses_positive(self):
        assert mm_processor.env_int({"K": "12"}, "K", 7) == 12

    @pytest.mark.parametrize("raw", ["0", "-1", "abc"])
    def test_rejects_invalid(self, raw):
        with pytest.raises(ValueError, match="K="):
            mm_processor.env_int({"K": raw}, "K", 7)


class TestItemBytes:
    def test_non_data_urls_are_unbounded(self):
        assert mm_processor.data_url_payload_bytes("https://a/1.png") is None
        assert mm_processor.data_url_payload_bytes("file:///a.png") is None

    def test_base64_payload_estimate(self):
        assert mm_processor.data_url_payload_bytes("data:image/png;base64,AAAAAAAA") == 6

    def test_plain_payload_length(self):
        assert mm_processor.data_url_payload_bytes("DATA:text/plain,hello") == 5

    def test_enforce_caps_data_urls_only(self):
        items = [
            _Item("image", "https://a/1.png"),
            _Item("image", "data:image/png;base64,AAAAAAAA"),
        ]
        mm_processor.enforce_item_bytes(items, 6)
        with pytest.raises(ValueError, match="media_refs\\[1\\].*6 bytes, above the 5-byte cap"):
            mm_processor.enforce_item_bytes(items, 5)


class TestBuildProcessor:
    class _ModelConfig:
        def __init__(self, multimodal: bool):
            self.is_multimodal_model = multimodal

    class _Engine:
        def __init__(self, multimodal: bool = True):
            self.model_config = TestBuildProcessor._ModelConfig(multimodal)

    def test_off_returns_none(self):
        assert mm_processor.build_mm_processor(self._Engine(), env={}) is None

    def test_non_multimodal_model_returns_none(self):
        env = {"SMG_VLLM_MM_PROCESSOR": "inprocess"}
        assert mm_processor.build_mm_processor(self._Engine(multimodal=False), env=env) is None

    def test_redis_not_available_yet(self):
        env = {"SMG_VLLM_MM_PROCESSOR": "redis"}
        with pytest.raises(ValueError, match="not available"):
            mm_processor.build_mm_processor(self._Engine(), env=env)

    def test_invalid_item_cap_is_rejected_before_construction(self):
        env = {"SMG_VLLM_MM_PROCESSOR": "inprocess", "SMG_VLLM_MM_MAX_ITEM_BYTES": "0"}
        with pytest.raises(ValueError, match="SMG_VLLM_MM_MAX_ITEM_BYTES"):
            mm_processor.build_mm_processor(self._Engine(), env=env)


class TestServicerWiring:
    """With vLLM installed, the servicer constructor stays two-argument and reads the env."""

    def test_default_off_keeps_processor_unset(self, monkeypatch):
        pytest.importorskip("vllm")
        from smg_grpc_servicer.vllm.servicer import VllmEngineServicer

        monkeypatch.delenv("SMG_VLLM_MM_PROCESSOR", raising=False)

        class _Engine:
            vllm_config = type("VC", (), {"kv_events_config": None})()
            model_config = type("MC", (), {"is_multimodal_model": False})()

        servicer = VllmEngineServicer(_Engine(), start_time=0.0)
        assert servicer._mm_processor is None
