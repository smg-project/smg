"""Unit tests for the multimodal wire-key helpers (engine-free, no vLLM required).

Run with: pytest grpc_servicer/tests/test_vllm_mm_keys.py
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from smg_grpc_proto import vllm_engine_pb2

# Import the module directly to avoid pulling vllm via the package __init__
_MODULE_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "mm_keys.py"
_spec = importlib.util.spec_from_file_location("mm_keys", _MODULE_PATH)
mm_keys = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mm_keys)


def test_default_primary_key_is_pixel_values():
    mm = vllm_engine_pb2.MultimodalInputs(pixel_values=vllm_engine_pb2.TensorData(dtype="float32"))
    assert mm_keys.primary_encoder_key(mm) == "pixel_values"
    assert mm_keys.modality_key(mm_keys.primary_encoder_key(mm), is_video=False) == "pixel_values"
    assert (
        mm_keys.modality_key(mm_keys.primary_encoder_key(mm), is_video=True)
        == "pixel_values_videos"
    )


def test_encoder_input_key_renames_the_primary_tensor():
    # DeepSeek-V4.1: the router names the tensor `patches` and slices it by
    # `patches_per_image`; the servicer must register it under that key.
    mm = vllm_engine_pb2.MultimodalInputs(
        pixel_values=vllm_engine_pb2.TensorData(dtype="float32"),
        encoder_input_key="patches",
    )
    mm.flat_keys["patches"] = "patches_per_image"
    assert mm_keys.primary_encoder_key(mm) == "patches"
    # The router names the flat layout by the same key it names the tensor.
    assert mm.flat_keys[mm_keys.primary_encoder_key(mm)] == "patches_per_image"
    assert mm_keys.modality_key("patches", is_video=False) == "patches"
    # A renamed primary tensor is never the video pixel key.
    assert mm_keys.modality_key("patches", is_video=True) == "patches"


def test_primary_key_tolerates_an_older_proto_stub():
    # A `smg-grpc-proto` stub built before `encoder_input_key` (field 11) has
    # no such attribute; the servicer must fall back to the default, not raise.
    stub = SimpleNamespace(pixel_values=vllm_engine_pb2.TensorData(dtype="float32"))
    assert mm_keys.primary_encoder_key(stub) == "pixel_values"


def test_other_keys_pass_through_unchanged():
    for key in ["image_grid_thw", "vit_grid", "types", "patches_per_image"]:
        assert mm_keys.modality_key(key, is_video=False) == key
        assert mm_keys.modality_key(key, is_video=True) == key
