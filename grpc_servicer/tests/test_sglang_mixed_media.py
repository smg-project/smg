"""Mixed tensor batches keep their per-modality metadata in the Python adapter."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "_mixed_media_servicer_fixture",
    Path(__file__).with_name("test_selected_token_logprobs.py"),
)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
manager_mod = _fixture.manager_mod
servicer = _fixture.servicer


@pytest.mark.parametrize("video_first", [False, True])
@pytest.mark.parametrize("image_token_id", [7, None])
def test_mixed_media_preserves_batches(monkeypatch, servicer, video_first, image_token_id):
    """Keep tensors and offsets attached to their modality in either batch order."""
    from smg_grpc_proto import sglang_scheduler_pb2 as pb

    module = sys.modules[type(servicer).__module__]
    monkeypatch.setattr(module, "Modality", SimpleNamespace(IMAGE="image", VIDEO="video"))
    monkeypatch.setattr(module, "MultimodalDataItem", SimpleNamespace)
    monkeypatch.setattr(module, "MultimodalProcessorOutput", SimpleNamespace)
    monkeypatch.setattr(servicer, "_decode_tensor_data", lambda t: (t.dtype, list(t.shape), t.data))
    image = pb.MultimodalInputs(
        modalities=["image"],
        im_token_id=image_token_id,
        pixel_values=pb.TensorData(dtype="float32", shape=[1], data=b"image"),
        model_specific_tensors={"image_grid_thw": pb.TensorData(dtype="int64", shape=[1, 3])},
        mm_placeholders=[pb.PlaceholderRange(offset=1, length=2)],
    )
    video = pb.MultimodalInputs(
        modalities=["video"],
        im_token_id=8,
        pixel_values=pb.TensorData(dtype="bfloat16", shape=[2], data=b"video"),
        model_specific_tensors={"video_grid_thw": pb.TensorData(dtype="int64", shape=[2, 3])},
        mm_placeholders=[pb.PlaceholderRange(offset=4, length=3)],
    )
    first, extra = (video, image) if video_first else (image, video)
    request = pb.GenerateRequest(mm_inputs=first, extra_mm_inputs=[extra])
    result = servicer._parse_mm_inputs(request.mm_inputs, request.extra_mm_inputs)
    assert result.im_token_id == image_token_id
    assert result.video_token_id == 8
    items = {item.modality: item for item in result.mm_items}
    assert items["image"].offsets == [(1, 2)]
    assert items["video"].offsets == [(4, 6)]
    assert items["image"].feature == ("float32", [1], b"image")
    assert items["video"].feature == ("bfloat16", [2], b"video")
    assert "image_grid_thw" in items["image"].model_specific_data
    assert "video_grid_thw" in items["video"].model_specific_data

    request.extra_mm_inputs.add().CopyFrom(extra)
    with pytest.raises(ValueError, match="one batch per modality"):
        servicer._parse_mm_inputs(request.mm_inputs, request.extra_mm_inputs)


@pytest.mark.parametrize("references", [False, True])
def test_extra_batches_require_primary_tensors(servicer, references):
    """Reject orphaned extras instead of silently dropping their media."""
    from smg_grpc_proto import sglang_scheduler_pb2 as pb

    request = pb.GenerateRequest(extra_mm_inputs=[pb.MultimodalInputs(modalities=["video"])])
    if references:
        request.mm_inputs.image_urls.append("https://example.test/image.png")
    with pytest.raises(ValueError, match="requires primary preprocessed inputs"):
        servicer._convert_generate_request(request)
