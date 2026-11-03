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


@pytest.fixture
def media_converter(monkeypatch, servicer):
    """Provide the real request converter with lightweight engine and tensor stand-ins."""
    from smg_grpc_proto import sglang_scheduler_pb2 as pb

    module = sys.modules[type(servicer).__module__]
    monkeypatch.setattr(module, "Modality", SimpleNamespace(IMAGE="image", VIDEO="video"))
    monkeypatch.setattr(module, "MultimodalDataItem", SimpleNamespace)
    monkeypatch.setattr(module, "MultimodalProcessorOutput", SimpleNamespace)
    monkeypatch.setattr(servicer, "_decode_tensor_data", lambda t: (t.dtype, list(t.shape), t.data))
    monkeypatch.setattr(
        module, "TokenizedGenerateReqInput", lambda **fields: SimpleNamespace(**fields)
    )
    servicer._convert_sampling_params = lambda params: SimpleNamespace(
        normalize=lambda tokenizer=None: None
    )
    servicer.tokenizer = None
    image = pb.MultimodalInputs(
        modalities=["image"],
        im_token_id=7,
        pixel_values=pb.TensorData(dtype="float32", shape=[1], data=b"image"),
        model_specific_tensors={
            "image_grid_thw": pb.TensorData(dtype="int64", shape=[1, 3], data=b"image-grid")
        },
        mm_placeholders=[pb.PlaceholderRange(offset=1, length=2)],
    )
    video = pb.MultimodalInputs(
        modalities=["video"],
        im_token_id=8,
        pixel_values=pb.TensorData(dtype="bfloat16", shape=[2], data=b"video"),
        model_specific_tensors={
            "video_grid_thw": pb.TensorData(dtype="int64", shape=[2, 3], data=b"video-grid")
        },
        mm_placeholders=[pb.PlaceholderRange(offset=4, length=3)],
    )
    return servicer, pb, image, video


@pytest.mark.parametrize("video_first", [False, True])
@pytest.mark.parametrize("image_token_id", [7, None])
def test_mixed_media_preserves_batches(media_converter, video_first, image_token_id):
    """Keep both batches through request conversion, including their exact grid tensors."""
    servicer, pb, image, video = media_converter
    if image_token_id is None:
        image.ClearField("im_token_id")
    first, extra = (video, image) if video_first else (image, video)
    request = pb.GenerateRequest(
        tokenized=pb.TokenizedInput(input_ids=list(range(8))),
        mm_inputs=first,
        extra_mm_inputs=[extra],
    )
    result = servicer._convert_generate_request(request).mm_inputs
    assert result.im_token_id == image_token_id
    assert result.video_token_id == 8
    items = {item.modality: item for item in result.mm_items}
    assert items["image"].offsets == [(1, 2)]
    assert items["video"].offsets == [(4, 6)]
    assert items["image"].feature == ("float32", [1], b"image")
    assert items["video"].feature == ("bfloat16", [2], b"video")
    assert set(items) == {"image", "video"}
    assert items["image"].model_specific_data == {
        "image_grid_thw": ("int64", [1, 3], b"image-grid")
    }
    assert items["video"].model_specific_data == {
        "video_grid_thw": ("int64", [2, 3], b"video-grid")
    }

    request.extra_mm_inputs.add().CopyFrom(extra)
    with pytest.raises(ValueError, match="one batch per modality"):
        servicer._convert_generate_request(request)


@pytest.mark.parametrize("references", [False, True])
def test_extra_batches_require_primary_tensors(servicer, references):
    """Reject orphaned extras instead of silently dropping their media."""
    from smg_grpc_proto import sglang_scheduler_pb2 as pb

    request = pb.GenerateRequest(extra_mm_inputs=[pb.MultimodalInputs(modalities=["video"])])
    if references:
        request.mm_inputs.image_urls.append("https://example.test/image.png")
    with pytest.raises(ValueError, match="requires primary preprocessed inputs"):
        servicer._convert_generate_request(request)


@pytest.mark.parametrize("extra", [False, True])
@pytest.mark.parametrize("offset,length", [(1, 0), (8, 1), (7, 2)])
def test_invalid_placeholder_ranges(media_converter, extra, offset, length):
    """Reject empty and out-of-prompt ranges in either the primary or extra batch."""
    servicer, pb, image, video = media_converter
    placeholder = (video if extra else image).mm_placeholders[0]
    placeholder.offset, placeholder.length = offset, length
    request = pb.GenerateRequest(
        tokenized=pb.TokenizedInput(input_ids=list(range(8))),
        mm_inputs=image,
        extra_mm_inputs=[video],
    )
    with pytest.raises(ValueError, match="zero length|outside the prompt"):
        servicer._convert_generate_request(request)


def test_video_only_token_fallback(media_converter):
    """A video-only batch retains the legacy image-token fallback."""
    servicer, pb, _, video = media_converter
    request = pb.GenerateRequest(
        tokenized=pb.TokenizedInput(input_ids=list(range(7))),
        mm_inputs=video,
    )
    result = servicer._convert_generate_request(request).mm_inputs
    assert result.im_token_id == result.video_token_id == 8
    assert [item.modality for item in result.mm_items] == ["video"]


@pytest.mark.parametrize("with_image", [False, True])
def test_legacy_proto_without_extra_field(media_converter, with_image):
    """An older request schema still supports text and single-image conversion."""
    servicer, pb, image, _ = media_converter
    request = pb.GenerateRequest(tokenized=pb.TokenizedInput(input_ids=list(range(8))))
    if with_image:
        request.mm_inputs.CopyFrom(image)

    class LegacyRequest:
        """Expose the old schema without the newly added field."""

        def __getattr__(self, name):
            """Forward unchanged fields and reject the unavailable field."""
            if name == "extra_mm_inputs":
                raise AttributeError(name)
            return getattr(request, name)

    result = servicer._convert_generate_request(LegacyRequest())
    if with_image:
        assert [item.modality for item in result.mm_inputs.mm_items] == ["image"]
    else:
        assert result.mm_inputs is None
