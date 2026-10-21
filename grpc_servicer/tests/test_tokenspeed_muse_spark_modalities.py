"""The TokenSpeed servicer advertises Muse Spark's encoders from the checkpoint's
own config: a vision tower (image), a video token (video) and a waveform tower
declared as ``audio_config`` with an ``audio_token_id`` (audio), whether or not
the engine's architecture list names the model an audio model.

Run with: pytest grpc_servicer/tests/test_tokenspeed_muse_spark_modalities.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("msgspec")
pytest.importorskip("zmq")
pytest.importorskip("numpy")
pytest.importorskip("smg_grpc_proto")
from smg_grpc_proto.generated import common_pb2  # noqa: E402

MODEL_TYPE = "muse_spark"
_CONFIG = {
    "model_type": MODEL_TYPE,
    "architectures": ["MuseSparkForConditionalGeneration"],
    "image_token_id": 200_092,
    "video_token_id": 200_091,
    "audio_token_id": 200_038,
    "audio_config": {"model_type": "muse_spark_audio"},
}


def _modalities(tokenspeed_servicer_mod, hf_config, *, is_audio_model: bool):
    # The engine's ModelConfig flags for this architecture: multimodal, and an
    # audio model only when its architecture list says so.
    model_config = SimpleNamespace(
        is_multimodal_active=True, is_multimodal=True, is_audio_model=is_audio_model
    )
    return tokenspeed_servicer_mod.TokenSpeedSchedulerServicer._static_supported_modalities(
        model_config, hf_config
    )


def test_the_declared_waveform_tower_advertises_audio_without_the_engine_flag(
    tokenspeed_servicer_mod,
):
    assert _modalities(tokenspeed_servicer_mod, _CONFIG, is_audio_model=False) == [
        common_pb2.IMAGE,
        common_pb2.VIDEO,
        common_pb2.AUDIO,
    ]


def test_the_engine_flag_still_advertises_audio(tokenspeed_servicer_mod):
    assert _modalities(tokenspeed_servicer_mod, _CONFIG, is_audio_model=True) == [
        common_pb2.IMAGE,
        common_pb2.VIDEO,
        common_pb2.AUDIO,
    ]


def test_without_an_audio_tower_or_token_audio_is_not_advertised(tokenspeed_servicer_mod):
    no_tower = {key: value for key, value in _CONFIG.items() if key != "audio_config"}
    assert _modalities(tokenspeed_servicer_mod, no_tower, is_audio_model=False) == [
        common_pb2.IMAGE,
        common_pb2.VIDEO,
    ]
    no_token = {**_CONFIG, "audio_token_id": None}
    assert _modalities(tokenspeed_servicer_mod, no_token, is_audio_model=False) == [
        common_pb2.IMAGE,
        common_pb2.VIDEO,
    ]
    assert _modalities(tokenspeed_servicer_mod, no_token, is_audio_model=True) == [
        common_pb2.IMAGE,
        common_pb2.VIDEO,
    ]
