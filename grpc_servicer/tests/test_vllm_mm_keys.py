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


def test_mm_batches_lists_the_primary_batch_then_the_extras():
    from smg_grpc_proto.generated import common_pb2

    request = vllm_engine_pb2.GenerateRequest(
        mm_inputs=vllm_engine_pb2.MultimodalInputs(modality=common_pb2.IMAGE),
        extra_mm_inputs=[vllm_engine_pb2.MultimodalInputs(modality=common_pb2.VIDEO)],
    )
    batches = mm_keys.mm_batches(request)
    assert [mm_keys.modality_name(batch) for batch in batches] == ["image", "video"]

    assert mm_keys.mm_batches(vllm_engine_pb2.GenerateRequest()) == []
    only_extra = vllm_engine_pb2.GenerateRequest(
        extra_mm_inputs=[vllm_engine_pb2.MultimodalInputs(modality=common_pb2.VIDEO)]
    )
    assert [mm_keys.modality_name(batch) for batch in mm_keys.mm_batches(only_extra)] == ["video"]


def _batch(modality, *, pixels: bool, hashes=()):
    batch = vllm_engine_pb2.MultimodalInputs(modality=modality, mm_hashes=list(hashes))
    if pixels:
        batch.pixel_values.CopyFrom(vllm_engine_pb2.TensorData(dtype="float32"))
    return batch


def test_a_request_may_not_describe_its_media_twice():
    from smg_grpc_proto.generated import common_pb2

    refs = vllm_engine_pb2.MediaRefs(items=[vllm_engine_pb2.MediaRef(url="https://e.test/a.png")])

    # Media references alone, and batches alone, each say what they want.
    assert not mm_keys.describes_media_twice(vllm_engine_pb2.GenerateRequest(media_refs=refs))
    assert not mm_keys.describes_media_twice(
        vllm_engine_pb2.GenerateRequest(mm_inputs=_batch(common_pb2.IMAGE, pixels=True))
    )

    # Both together do not: whichever the worker picks, the other is dropped.
    assert mm_keys.describes_media_twice(
        vllm_engine_pb2.GenerateRequest(
            media_refs=refs, mm_inputs=_batch(common_pb2.IMAGE, pixels=True)
        )
    )
    # A second modality is carried in the extras, and counts the same way.
    assert mm_keys.describes_media_twice(
        vllm_engine_pb2.GenerateRequest(
            media_refs=refs, extra_mm_inputs=[_batch(common_pb2.VIDEO, pixels=True)]
        )
    )


def test_every_batch_needs_its_own_encoder_tensor():
    from smg_grpc_proto.generated import common_pb2

    images = _batch(common_pb2.IMAGE, pixels=True)
    video = _batch(common_pb2.VIDEO, pixels=True)
    assert mm_keys.batches_missing_pixels([images, video]) == []

    # One batch's tensor does not stand in for the other's: the modality left
    # without one would have its encoder run over nothing.
    bare_video = _batch(common_pb2.VIDEO, pixels=False)
    assert mm_keys.batches_missing_pixels([images, bare_video]) == [bare_video]

    bare_images = _batch(common_pb2.IMAGE, pixels=False)
    assert mm_keys.batches_missing_pixels([bare_images, bare_video]) == [bare_images, bare_video]
    assert mm_keys.batches_missing_pixels([]) == []


def test_media_identity_covers_every_batch():
    from smg_grpc_proto.generated import common_pb2

    request = vllm_engine_pb2.GenerateRequest(
        mm_inputs=_batch(common_pb2.IMAGE, pixels=False, hashes=["img-a"]),
        extra_mm_inputs=[_batch(common_pb2.VIDEO, pixels=False, hashes=["vid-a", "vid-b"])],
    )
    assert mm_keys.mm_identity_hashes(mm_keys.mm_batches(request)) == ["img-a", "vid-a", "vid-b"]
    assert mm_keys.mm_identity_hashes([]) == []

    # Two requests that differ only in the second batch's media must not read
    # as the same one.
    other = vllm_engine_pb2.GenerateRequest(
        mm_inputs=_batch(common_pb2.IMAGE, pixels=False, hashes=["img-a"]),
        extra_mm_inputs=[_batch(common_pb2.VIDEO, pixels=False, hashes=["vid-c"])],
    )
    assert mm_keys.mm_identity_hashes(mm_keys.mm_batches(request)) != mm_keys.mm_identity_hashes(
        mm_keys.mm_batches(other)
    )


def test_mm_batches_tolerates_an_older_proto_stub():
    # A GenerateRequest stub built before `extra_mm_inputs` (field 11) has no
    # such attribute; the servicer must still see the primary batch.
    primary = vllm_engine_pb2.MultimodalInputs()
    stub = SimpleNamespace(mm_inputs=primary, HasField=lambda name: name == "mm_inputs")
    assert mm_keys.mm_batches(stub) == [primary]
