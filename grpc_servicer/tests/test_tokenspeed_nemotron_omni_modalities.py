"""The TokenSpeed servicer advertises Nemotron-H Omni's encoders from the engine's
generic flags: multimodal (image), a video token (video) and an audio model with an
audio token (audio).

Run with: pytest grpc_servicer/tests/test_tokenspeed_nemotron_omni_modalities.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("msgspec")
pytest.importorskip("zmq")
pytest.importorskip("numpy")
pytest.importorskip("smg_grpc_proto")
from smg_grpc_proto.generated import common_pb2  # noqa: E402

MODEL_TYPE = "NemotronH_Nano_Omni_Reasoning_V3"


def _modalities(tokenspeed_servicer_mod, hf_config):
    # The engine's ModelConfig flags for this architecture.
    model_config = SimpleNamespace(
        is_multimodal_active=True, is_multimodal=True, is_audio_model=True
    )
    return tokenspeed_servicer_mod.TokenSpeedSchedulerServicer._static_supported_modalities(
        model_config, hf_config
    )


def test_vision_and_sound_towers_advertise_image_video_and_audio(tokenspeed_servicer_mod):
    config = {"model_type": MODEL_TYPE, "video_token_id": 18, "audio_token_id": 27}
    assert _modalities(tokenspeed_servicer_mod, config) == [
        common_pb2.IMAGE,
        common_pb2.VIDEO,
        common_pb2.AUDIO,
    ]


def test_without_an_audio_token_audio_is_not_advertised(tokenspeed_servicer_mod):
    config = {"model_type": MODEL_TYPE, "video_token_id": 18, "audio_token_id": None}
    assert _modalities(tokenspeed_servicer_mod, config) == [common_pb2.IMAGE, common_pb2.VIDEO]
