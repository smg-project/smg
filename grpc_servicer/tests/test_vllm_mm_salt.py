"""Unit tests for the multimodal identity cache salt (engine-free, no vLLM required).

Run with: pytest grpc_servicer/tests/test_vllm_mm_salt.py
"""

import importlib.util
from pathlib import Path

from smg_grpc_proto import vllm_engine_pb2

# Import the module directly to avoid pulling vllm via the package __init__
_MODULE_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "mm_salt.py"
_spec = importlib.util.spec_from_file_location("mm_salt", _MODULE_PATH)
mm_salt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mm_salt)


def test_gate_accepts_pixel_payloads():
    mm = vllm_engine_pb2.MultimodalInputs(pixel_values=vllm_engine_pb2.TensorData(dtype="float32"))
    assert mm_salt.has_preprocessed_mm_payload(mm)


def test_gate_accepts_grid_only_payloads():
    # The PD decode leg's form: grid tensors without pixels.
    mm = vllm_engine_pb2.MultimodalInputs(mm_hashes=["h1"])
    mm.model_specific_tensors["image_grid_thw"].CopyFrom(
        vllm_engine_pb2.TensorData(dtype="int64", shape=[1, 3])
    )
    assert mm_salt.has_preprocessed_mm_payload(mm)


def test_gate_rejects_identity_only_payloads():
    # Hashes alone cannot rebuild mm features; this leg takes the salt path.
    mm = vllm_engine_pb2.MultimodalInputs(mm_hashes=["h1"])
    assert not mm_salt.has_preprocessed_mm_payload(mm)


def test_empty_hashes_produce_no_salt():
    assert mm_salt.mm_identity_cache_salt([]) is None


def test_salt_is_deterministic_per_content():
    salt = mm_salt.mm_identity_cache_salt(["h1", "h2"])
    assert salt == "mm:h1,h2"
    assert salt == mm_salt.mm_identity_cache_salt(["h1", "h2"])


def test_different_images_get_different_salts():
    assert mm_salt.mm_identity_cache_salt(["dog"]) != mm_salt.mm_identity_cache_salt(["passport"])


def test_salt_is_order_sensitive():
    # Same images in a different order occupy different placeholder positions.
    assert mm_salt.mm_identity_cache_salt(["h1", "h2"]) != mm_salt.mm_identity_cache_salt(
        ["h2", "h1"]
    )


class _ModelConfig:
    """Minimal stand-in for vLLM's ModelConfig."""

    def __init__(self, is_multimodal_model, supports_multimodal_inputs=None):
        self.is_multimodal_model = is_multimodal_model
        if supports_multimodal_inputs is not None:
            self.supports_multimodal_inputs = supports_multimodal_inputs


def test_engine_accepts_mm_inputs_full_vision_worker():
    assert mm_salt.engine_accepts_mm_inputs(_ModelConfig(True, True))


def test_engine_accepts_mm_inputs_language_model_only():
    # --language-model-only keeps the multimodal architecture but zeroes
    # every modality limit: the engine accepts no multimodal inputs.
    assert not mm_salt.engine_accepts_mm_inputs(_ModelConfig(True, False))


def test_engine_accepts_mm_inputs_text_model():
    assert not mm_salt.engine_accepts_mm_inputs(_ModelConfig(False, False))


def test_engine_accepts_mm_inputs_older_vllm_falls_back_to_architecture():
    # vLLM builds without supports_multimodal_inputs keep today's behavior.
    assert mm_salt.engine_accepts_mm_inputs(_ModelConfig(True))
    assert not mm_salt.engine_accepts_mm_inputs(_ModelConfig(False))
