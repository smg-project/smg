"""Unit tests for the cache salt and the media identity in it (engine-free, no vLLM required).

Run with: pytest grpc_servicer/tests/test_vllm_mm_salt.py
"""

import importlib.util
from pathlib import Path

import pytest
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
    # The PD decode side's form: grid tensors without pixels.
    mm = vllm_engine_pb2.MultimodalInputs(mm_hashes=["h1"])
    mm.model_specific_tensors["image_grid_thw"].CopyFrom(
        vllm_engine_pb2.TensorData(dtype="int64", shape=[1, 3])
    )
    assert mm_salt.has_preprocessed_mm_payload(mm)


def test_gate_rejects_identity_only_payloads():
    # Hashes alone cannot rebuild mm features; this side takes the salt path.
    mm = vllm_engine_pb2.MultimodalInputs(mm_hashes=["h1"])
    assert not mm_salt.has_preprocessed_mm_payload(mm)


def test_empty_hashes_keep_the_request_salt():
    assert mm_salt.engine_cache_salt(None, []) is None
    assert mm_salt.engine_cache_salt("tenant-a", []) == "tenant-a"


# The same vectors are in `crates/engine_zmq_adapter/src/vllm_cache_salt.rs`.
@pytest.mark.parametrize(
    ("request_salt", "hashes", "digest"),
    [
        (None, ["h1", "h2"], "c67efe1de6d4c5ef289ccdfd6953f085e29da935ac501db9cd8512e6a39f185a"),
        (
            "tenant-a",
            ["h1", "h2"],
            "662beced721917156bf4b76f866d2c307d82dd3bcfa95262ec7104fa2ba0fa3f",
        ),
        (
            "tenant-b",
            ["h1", "h2"],
            "905c55f34a7ed6802bd075bbf00211abf77cab562139443433b6edb664a9ec54",
        ),
        ("tenant-a", ["h1h2"], "54ce43702e479ec4a01a1298d1077d8907e957821dc5a80b97ceea5bbc25f978"),
    ],
)
def test_hashes_give_the_digest_of_the_salt_and_the_hashes(request_salt, hashes, digest):
    assert mm_salt.engine_cache_salt(request_salt, hashes) == f"mm:{digest}"


def test_different_images_get_different_salts():
    assert mm_salt.engine_cache_salt(None, ["dog"]) != mm_salt.engine_cache_salt(None, ["passport"])


def test_salt_is_order_sensitive():
    # Same images in a different order occupy different placeholder positions.
    assert mm_salt.engine_cache_salt(None, ["h1", "h2"]) != mm_salt.engine_cache_salt(
        None, ["h2", "h1"]
    )


@pytest.mark.parametrize("images", range(1, 9))
def test_the_engine_salt_stays_inside_vllms_limits(images):
    salt = mm_salt.engine_cache_salt("s" * 128, ["a" * 64] * images)
    assert len(salt) == 67
    assert not set(salt) & set("@/\\\0")


@pytest.mark.parametrize("salt", [None, "tenant-a", "s" * 128, "é" * 128])
def test_request_salt_is_read_verbatim(salt):
    request = vllm_engine_pb2.GenerateRequest(cache_salt=salt)
    assert mm_salt.request_cache_salt(request) == salt


@pytest.mark.parametrize(
    ("salt", "reason"),
    [
        ("", "empty"),
        ("s" * 129, "at most 128 characters"),
        ("secret@", "not contain"),
        ("secret/", "not contain"),
        ("secret\\", "not contain"),
        ("secret\0", "not contain"),
        ("mm:secret", "reserved prefix 'mm:'"),
    ],
)
def test_a_bad_request_salt_is_refused_without_its_value(salt, reason):
    request = vllm_engine_pb2.GenerateRequest(cache_salt=salt)
    with pytest.raises(ValueError, match=f"^cache_salt must .*{reason}") as refused:
        mm_salt.request_cache_salt(request)
    assert "secret" not in str(refused.value)


def test_an_older_proto_package_has_no_request_salt():
    # EmbedRequest stands in for a request message without the field.
    assert mm_salt.request_cache_salt(vllm_engine_pb2.EmbedRequest()) is None


class _MmConfig:
    """Minimal stand-in for vLLM's MultiModalConfig."""

    def __init__(self, *, language_model_only=False, enable_mm_embeds=False, limit_per_prompt=None):
        self.language_model_only = language_model_only
        self.enable_mm_embeds = enable_mm_embeds
        self.limit_per_prompt = limit_per_prompt or {}

    def get_limit_per_prompt(self, modality):
        # Mirrors MultiModalConfig: the flag zeroes every modality.
        if self.language_model_only:
            return 0
        return self.limit_per_prompt.get(modality, 999)


class _ModelConfig:
    """Minimal stand-in for vLLM's ModelConfig."""

    def __init__(self, is_multimodal_model, supports_multimodal_inputs=None, mm_config=None):
        self.is_multimodal_model = is_multimodal_model
        self.multimodal_config = _MmConfig() if is_multimodal_model else None
        if mm_config is not None:
            self.multimodal_config = mm_config
        if supports_multimodal_inputs is not None:
            self.supports_multimodal_inputs = supports_multimodal_inputs


def _fake_vllm_registry(monkeypatch, probe, modalities=("image",), *, public_api=False):
    """Install a fake ``vllm.multimodal`` whose registry probe is ``probe``."""
    import sys
    import types

    vllm_mod = types.ModuleType("vllm")
    mm_mod = types.ModuleType("vllm.multimodal")
    mm_mod.MULTIMODAL_REGISTRY = types.SimpleNamespace(
        supports_multimodal_inputs=probe,
        _create_processing_info=lambda mc, tokenizer=None: types.SimpleNamespace(
            supported_mm_limits=dict.fromkeys(modalities)
        ),
    )
    if public_api:
        registry = mm_mod.MULTIMODAL_REGISTRY
        registry.get_processing_info = lambda mc: types.SimpleNamespace(
            supported_mm_limits=dict.fromkeys(modalities)
        )
        del registry._create_processing_info
    vllm_mod.multimodal = mm_mod
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.multimodal", mm_mod)


# --- the property path (vLLM main) ---


def test_engine_accepts_mm_inputs_full_vision_worker():
    assert mm_salt.engine_accepts_mm_inputs(_ModelConfig(True, True))


def test_engine_accepts_mm_inputs_language_model_only():
    # --language-model-only keeps the multimodal architecture but zeroes
    # every modality limit: the engine accepts no multimodal inputs.
    assert not mm_salt.engine_accepts_mm_inputs(_ModelConfig(True, False))


def test_engine_accepts_mm_inputs_text_model():
    assert not mm_salt.engine_accepts_mm_inputs(_ModelConfig(False, False))


def test_engine_accepts_mm_inputs_mm_embeds_only(monkeypatch):
    _fake_vllm_registry(monkeypatch, lambda mc: True)
    # enable_mm_embeds with every modality limit at 0 ingests pre-computed
    # embeddings but has no encoder for pixel payloads.
    embeds_only = _ModelConfig(
        True,
        True,  # vLLM counts embeds-only engines as accepting mm inputs
        mm_config=_MmConfig(enable_mm_embeds=True, limit_per_prompt={"image": 0}),
    )
    assert not mm_salt.engine_accepts_mm_inputs(embeds_only)
    flagged = _ModelConfig(
        True,
        True,
        mm_config=_MmConfig(language_model_only=True, enable_mm_embeds=True),
    )
    assert not mm_salt.engine_accepts_mm_inputs(flagged)


@pytest.mark.parametrize("public_api", [False, True], ids=["legacy-api", "public-api"])
@pytest.mark.parametrize("supports", [True, None], ids=["property", "registry"])
@pytest.mark.parametrize(
    "modalities,limits,expected",
    [
        (("image", "video"), {"video": 0}, True),
        (("image", "video"), {"image": 0}, True),
        (("image", "video"), {}, True),
        (("image", "video"), {"image": 1, "video": 0}, True),
        (("image", "video"), {"image": 0, "video": 0}, False),
        (("image",), {"image": 0, "video": 1}, False),
        (("image", "audio"), {"image": 0}, False),
        (("image", "video", "audio"), {"image": 0, "video": 0}, False),
        (("image", "audio"), {"audio": 0}, True),
        (("audio",), {}, False),
    ],
)
def test_engine_accepts_mm_inputs_uses_supported_visual_modalities(
    monkeypatch, supports, modalities, limits, expected, public_api
):
    _fake_vllm_registry(monkeypatch, lambda mc: True, modalities, public_api=public_api)
    model = _ModelConfig(
        True,
        supports,
        mm_config=_MmConfig(enable_mm_embeds=True, limit_per_prompt=limits),
    )
    assert mm_salt.engine_accepts_mm_inputs(model) is expected


@pytest.mark.parametrize("language_model_only", [False, True])
def test_engine_accepts_mm_embeds_without_vllm(monkeypatch, language_model_only):
    import sys

    monkeypatch.setitem(sys.modules, "vllm", None)
    monkeypatch.setitem(sys.modules, "vllm.multimodal", None)
    model = _ModelConfig(
        True,
        True,
        mm_config=_MmConfig(
            enable_mm_embeds=True,
            language_model_only=language_model_only,
            limit_per_prompt={"video": 0},
        ),
    )
    assert mm_salt.engine_accepts_mm_inputs(model) is not language_model_only


def test_engine_accepts_mm_inputs_processing_info_failure(monkeypatch, caplog):
    _fake_vllm_registry(monkeypatch, lambda mc: True)
    from vllm.multimodal import MULTIMODAL_REGISTRY

    def broken_info(mc, tokenizer=None):
        raise RuntimeError("processing info unavailable")

    monkeypatch.setattr(MULTIMODAL_REGISTRY, "_create_processing_info", broken_info)
    model = _ModelConfig(
        True,
        True,
        mm_config=_MmConfig(enable_mm_embeds=True, limit_per_prompt={"video": 0}),
    )
    assert mm_salt.engine_accepts_mm_inputs(model)
    assert "Multimodal limits probe failed" in caplog.text


# --- the registry fallback (vLLM 0.19-0.20, no ModelConfig property) ---


def test_engine_accepts_mm_inputs_registry_fallback(monkeypatch):
    _fake_vllm_registry(monkeypatch, lambda mc: True)
    assert mm_salt.engine_accepts_mm_inputs(_ModelConfig(True))
    _fake_vllm_registry(monkeypatch, lambda mc: False)
    assert not mm_salt.engine_accepts_mm_inputs(_ModelConfig(True))


def test_engine_accepts_mm_inputs_registry_fallback_language_model_only(monkeypatch):
    # On vLLM 0.19-0.20 the registry's check reads the zeroed limits and
    # answers False; is_multimodal_model alone would have said True.
    _fake_vllm_registry(monkeypatch, lambda mc: False)
    lmo = _ModelConfig(True, mm_config=_MmConfig(language_model_only=True))
    assert not mm_salt.engine_accepts_mm_inputs(lmo)


def test_engine_accepts_mm_inputs_text_model_skips_the_probe(monkeypatch):
    def probe(mc):
        raise AssertionError("text models must not reach the registry probe")

    _fake_vllm_registry(monkeypatch, probe)
    assert not mm_salt.engine_accepts_mm_inputs(_ModelConfig(False))


def test_engine_accepts_mm_inputs_registry_without_processor(monkeypatch):
    # A multimodal architecture with no registered processor is text-only;
    # the registry raises ValueError (main's own check treats it the same).
    def probe(mc):
        raise ValueError("no processor")

    _fake_vllm_registry(monkeypatch, probe)
    assert not mm_salt.engine_accepts_mm_inputs(_ModelConfig(True))


def test_engine_accepts_mm_inputs_registry_failure_keeps_architecture_answer(monkeypatch):
    # An unexpected probe failure must not blind a healthy full-vision worker.
    def probe(mc):
        raise RuntimeError("registry broken")

    _fake_vllm_registry(monkeypatch, probe)
    assert mm_salt.engine_accepts_mm_inputs(_ModelConfig(True))


def test_engine_accepts_mm_inputs_without_vllm(monkeypatch):
    # Engine-free context: the architecture is all we have.
    import sys

    monkeypatch.setitem(sys.modules, "vllm", None)
    assert mm_salt.engine_accepts_mm_inputs(_ModelConfig(True))
    assert not mm_salt.engine_accepts_mm_inputs(_ModelConfig(False))
