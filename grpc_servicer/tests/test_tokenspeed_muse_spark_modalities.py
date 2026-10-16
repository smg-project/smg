"""The TokenSpeed servicer advertises Muse Spark's encoders from the checkpoint's
own config: a vision tower (image), a video token (video) and a waveform tower
declared as ``audio_config`` with an ``audio_token_id`` (audio), whether or not
the engine's architecture list names the model an audio model.

Run with: pytest grpc_servicer/tests/test_tokenspeed_muse_spark_modalities.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("msgspec")
pytest.importorskip("zmq")
pytest.importorskip("numpy")
pytest.importorskip("smg_grpc_proto")
from smg_grpc_proto.generated import common_pb2  # noqa: E402

_ROOT = Path(__file__).parents[1] / "smg_grpc_servicer" / "tokenspeed"
MODEL_TYPE = "muse_spark"
_CONFIG = {
    "model_type": MODEL_TYPE,
    "architectures": ["MuseSparkForConditionalGeneration"],
    "image_token_id": 200_092,
    "video_token_id": 200_091,
    "audio_token_id": 200_038,
    "audio_config": {"model_type": "muse_spark_audio"},
}


@pytest.fixture
def servicer_mod(monkeypatch):
    # Importing the engine (and torch) loads the runtime's dependencies; these
    # stand-ins cover the servicer module's imports and nothing of the
    # modality advertisement.
    symbols = {
        "torch": "",
        "tokenspeed.runtime.multimodal.inputs": "Modality MultimodalDataItem MultimodalInputs",
        "tokenspeed.runtime.multimodal.shm_transport": "ShmTensorHandle",
        "tokenspeed.runtime.pd.kv_events": "KVEventBatch",
    }
    for name, attrs in symbols.items():
        parts = name.split(".")
        for end in range(1, len(parts)):
            parent = ".".join(parts[:end])
            if parent not in sys.modules:
                module = ModuleType(parent)
                module.__path__ = []
                monkeypatch.setitem(sys.modules, parent, module)
        module = ModuleType(name)
        for attr in attrs.split():
            setattr(module, attr, object)
        monkeypatch.setitem(sys.modules, name, module)
    name = "smg_grpc_servicer.tokenspeed.servicer"
    spec = importlib.util.spec_from_file_location(name, _ROOT / "servicer.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _modalities(servicer_mod, hf_config, *, is_audio_model: bool):
    # The engine's ModelConfig flags for this architecture: multimodal, and an
    # audio model only when its architecture list says so.
    model_config = SimpleNamespace(
        is_multimodal_active=True, is_multimodal=True, is_audio_model=is_audio_model
    )
    return servicer_mod.TokenSpeedSchedulerServicer._static_supported_modalities(
        model_config, hf_config
    )


def test_the_declared_waveform_tower_advertises_audio_without_the_engine_flag(servicer_mod):
    assert _modalities(servicer_mod, _CONFIG, is_audio_model=False) == [
        common_pb2.IMAGE,
        common_pb2.VIDEO,
        common_pb2.AUDIO,
    ]


def test_the_engine_flag_still_advertises_audio(servicer_mod):
    assert _modalities(servicer_mod, _CONFIG, is_audio_model=True) == [
        common_pb2.IMAGE,
        common_pb2.VIDEO,
        common_pb2.AUDIO,
    ]


def test_without_an_audio_tower_or_token_audio_is_not_advertised(servicer_mod):
    no_tower = {key: value for key, value in _CONFIG.items() if key != "audio_config"}
    assert _modalities(servicer_mod, no_tower, is_audio_model=False) == [
        common_pb2.IMAGE,
        common_pb2.VIDEO,
    ]
    no_token = {**_CONFIG, "audio_token_id": None}
    assert _modalities(servicer_mod, no_token, is_audio_model=False) == [
        common_pb2.IMAGE,
        common_pb2.VIDEO,
    ]
    assert _modalities(servicer_mod, no_token, is_audio_model=True) == [
        common_pb2.IMAGE,
        common_pb2.VIDEO,
    ]
