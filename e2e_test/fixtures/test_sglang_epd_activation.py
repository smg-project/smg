"""Import original EPD marks without starting workers or calling test bodies."""

import importlib.util
import tempfile
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "chat_completions" / "test_epd_multimodal.py"
OPT_IN = "E2E_AMD_SGLANG_EPD_EQUIVALENT"


def load_module(monkeypatch, tmp_path, flag):
    spec = importlib.util.spec_from_file_location("epd_activation_cpu", SOURCE)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as patcher:
        patcher.delenv(OPT_IN, raising=False)
        if flag is not None:
            patcher.setenv(OPT_IN, flag)
        patcher.setattr(tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
        spec.loader.exec_module(module)
    return module


def marker(module, name):
    matches = [m for m in module.TestEPDMultimodal.pytestmark if m.name == name]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("flag", [None, "0", "true", "1"])
def test_sglang_activation_is_explicit_and_tokenspeed_remains(monkeypatch, tmp_path, flag):
    module = load_module(monkeypatch, tmp_path, flag)
    expected = ("tokenspeed", "sglang") if flag == "1" else ("tokenspeed",)
    assert marker(module, "engine").args == expected


def test_original_eight_cases_model_topologies_and_gateway_marks_remain(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path, "1")
    assert marker(module, "model").args == ("Qwen/Qwen3.5-9B",)
    assert marker(module, "gpu").args == (4,)
    config = marker(module, "gateway").kwargs
    assert config == {"log_level": "debug", "policy": "cache_aware", "log_dir": str(tmp_path)}
    params = marker(module, "parametrize")
    assert params.args[0] == "setup_backend" and params.kwargs == {"indirect": True}
    topologies = {p.id: p.values[0] for p in params.args[1]}
    assert topologies == {
        "1e1p1d": ("epd_grpc", (1, 1, 1)),
        "1e2p1d": ("epd_grpc", (1, 2, 1)),
        "2e1p1d": ("epd_grpc", (2, 1, 1)),
        "1e1p2d": ("epd_grpc", (1, 1, 2)),
    }
    methods = {name for name in vars(module.TestEPDMultimodal) if name.startswith("test_")}
    assert methods == {"test_color_images", "test_animal_images"}
    assert len(methods) * len(topologies) == 8


def test_opt_in_does_not_add_other_engine_lanes(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path, "1")
    assert set(marker(module, "engine").args).isdisjoint({"vllm", "trtllm", "generic"})
