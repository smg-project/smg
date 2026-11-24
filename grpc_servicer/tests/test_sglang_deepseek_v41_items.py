"""DeepSeek-V4.1 images reach SGLang as one item per image, the shape its own processor builds.

The Router sends a V4.1 image batch as one tensor of every image's patches with
the per-image grids alongside (``vit_grid``, ``patches_per_image``). SGLang's
model reads ``item.n_vit_h`` / ``item.n_vit_w`` from each item and encodes one
image per item, so the servicer cuts the batch into items before the scheduler
sees it, and refuses a batch whose counts disagree.
"""

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_spec = importlib.util.spec_from_file_location(
    "_v41_items_servicer_fixture",
    Path(__file__).with_name("test_selected_token_logprobs.py"),
)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
manager_mod = _fixture.manager_mod
servicer = _fixture.servicer

_TENSOR_WIRE = Path(__file__).parents[1] / "smg_grpc_servicer" / "tensor_wire.py"

# Two images: a 2x3 and a 1x2 patch grid, 6 + 2 patches of 3x2x2, filling
# token spans of 5 and 4 positions in a 12-token prompt. The served model takes
# 2x2 patches and merges 2x2 of them per token: the grids make 1*(2+1)+2 = 5
# and 1*(1+1)+2 = 4 tokens.
GRIDS = [(2, 3), (1, 2)]
PATCHES = [6, 2]
SPANS = [(1, 5), (7, 4)]
PROMPT_LEN = 12
V41_CONFIG = SimpleNamespace(
    model_type="deepseek_v41", vision_patch_size=2, vision_downsample_ratio=2
)


@pytest.fixture
def v41_converter(monkeypatch, servicer):
    """The real request converter and tensor decoder, with SGLang's types as plain records."""
    from smg_grpc_proto import sglang_scheduler_pb2 as pb

    spec = importlib.util.spec_from_file_location("_v41_tensor_wire", _TENSOR_WIRE)
    tensor_wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tensor_wire)

    module = sys.modules[type(servicer).__module__]
    monkeypatch.setattr(module, "Modality", SimpleNamespace(IMAGE="image", VIDEO="video"))
    monkeypatch.setattr(module, "MultimodalDataItem", SimpleNamespace)
    monkeypatch.setattr(module, "MultimodalProcessorOutput", SimpleNamespace)
    monkeypatch.setattr(module, "tensor_from_parts", tensor_wire.tensor_from_parts)
    monkeypatch.setattr(
        module, "TokenizedGenerateReqInput", lambda **fields: SimpleNamespace(**fields)
    )
    servicer._convert_sampling_params = lambda params: SimpleNamespace(
        normalize=lambda tokenizer=None: None
    )
    servicer.tokenizer = None
    servicer.model_config = SimpleNamespace(hf_config=V41_CONFIG)
    return servicer, pb


def _tensor(pb, values, shape, dtype):
    data = torch.tensor(values, dtype=dtype).numpy().tobytes()
    name = {torch.float32: "float32", torch.int64: "int64"}[dtype]
    return pb.TensorData(dtype=name, shape=list(shape), data=data)


def _v41_batch(pb, grids=GRIDS, patches=PATCHES, spans=SPANS, types_per_image=True):
    """A V4.1 image batch as the Router assembles it: patch row ``i`` holds the value ``i``."""
    rows = sum(patches)
    pixels = [[[[float(row)] * 2] * 2] * 3 for row in range(rows)]
    tensors = {
        "vit_grid": _tensor(pb, [list(g) for g in grids], (len(grids), 2), torch.int64),
        "llm_grid": _tensor(pb, [[1, 1]] * len(grids), (len(grids), 2), torch.int64),
        "patches_per_image": _tensor(pb, patches, (len(patches),), torch.int64),
        "types": _tensor(
            pb, [0] * sum(n for _, n in spans), (sum(n for _, n in spans),), torch.int64
        ),
    }
    if types_per_image:
        counts = [n for _, n in spans]
        tensors["types_per_image"] = _tensor(pb, counts, (len(counts),), torch.int64)
    return pb.MultimodalInputs(
        modalities=["image"],
        im_token_id=9,
        pixel_values=_tensor(pb, pixels, (rows, 3, 2, 2), torch.float32),
        model_specific_tensors=tensors,
        mm_placeholders=[pb.PlaceholderRange(offset=o, length=n) for o, n in spans],
    )


def _convert(servicer, pb, batch, extras=()):
    request = pb.GenerateRequest(
        tokenized=pb.TokenizedInput(input_ids=list(range(PROMPT_LEN))),
        mm_inputs=batch,
        extra_mm_inputs=extras,
    )
    return servicer._convert_generate_request(request).mm_inputs


@pytest.mark.parametrize("types_per_image", [True, False])
def test_each_image_is_its_own_item(v41_converter, types_per_image):
    """Each image's patches, span and grid make one item SGLang's V4.1 model can encode."""
    servicer, pb = v41_converter
    result = _convert(servicer, pb, _v41_batch(pb, types_per_image=types_per_image))

    assert result.im_token_id == 9
    assert [item.modality for item in result.mm_items] == ["image", "image"]
    first, second = result.mm_items
    assert first.offsets == [(1, 5)]
    assert second.offsets == [(7, 10)]
    assert first.model_specific_data == {"n_vit_h": 2, "n_vit_w": 3}
    assert second.model_specific_data == {"n_vit_h": 1, "n_vit_w": 2}
    assert all(
        type(value) is int
        for item in result.mm_items
        for value in item.model_specific_data.values()
    )
    assert first.feature.shape == (6, 3, 2, 2)
    assert second.feature.shape == (2, 3, 2, 2)
    assert first.feature[:, 0, 0, 0].tolist() == [0, 1, 2, 3, 4, 5]
    assert second.feature[:, 0, 0, 0].tolist() == [6, 7]
    # The scheduler receives each item pickled: a view would carry every
    # image's patches with it, so each item owns only its own.
    for item in result.mm_items:
        own_bytes = item.feature.numel() * item.feature.element_size()
        assert item.feature.untyped_storage().nbytes() == own_bytes


def test_one_image_is_one_item(v41_converter):
    """A single image keeps one item, now carrying its grid as SGLang reads it."""
    servicer, pb = v41_converter
    batch = _v41_batch(pb, grids=[(2, 3)], patches=[6], spans=[(2, 5)])
    (item,) = _convert(servicer, pb, batch).mm_items
    assert item.offsets == [(2, 6)]
    assert item.model_specific_data == {"n_vit_h": 2, "n_vit_w": 3}
    assert item.feature.shape == (6, 3, 2, 2)


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"spans": SPANS[:1]}, "2 images but 1 placeholders"),
        ({"patches": [6]}, "2 images but 1 patch counts"),
        ({"patches": [5, 2]}, "7 patches but the pixel tensor has 8 rows"),
        ({"patches": [6, 3]}, "9 patches but the pixel tensor has 8 rows"),
        ({"grids": [(2, 3), (1, 3)]}, "image 1 has a 1x3 grid but 2 patches"),
        ({"grids": [(2, 3), (-1, -2)]}, "image 1 has a -1x-2 grid but 2 patches"),
        (
            {
                "grids": [(2, 3), (1, 2), (0, 5)],
                "patches": [6, 2, 0],
                "spans": [(1, 5), (7, 4), (11, 1)],
            },
            "image 2 has a 0x5 grid but 0 patches",
        ),
    ],
)
def test_counts_that_disagree_are_refused(v41_converter, fields, message):
    """A batch whose grids, patch counts and placeholders disagree never reaches the scheduler."""
    servicer, pb = v41_converter
    batch = _v41_batch(pb, **fields)
    if "patches" in fields and sum(fields["patches"]) != sum(PATCHES):
        # Keep the pixel tensor at the original 8 rows: only the counts lie.
        batch.pixel_values.CopyFrom(_v41_batch(pb).pixel_values)
    with pytest.raises(ValueError, match=message):
        _convert(servicer, pb, batch)


def test_span_length_must_match_the_image_tokens(v41_converter):
    """A placeholder whose length is not the image's token count is refused."""
    servicer, pb = v41_converter
    batch = _v41_batch(pb)
    batch.model_specific_tensors["types_per_image"].CopyFrom(_tensor(pb, [5, 3], (2,), torch.int64))
    with pytest.raises(ValueError, match="image 1 fills 4 prompt positions but has 3 tokens"):
        _convert(servicer, pb, batch)


@pytest.mark.parametrize(
    "spans, message",
    [
        ([(1, 5), (1, 5)], "image 1's placeholder 1..=5 does not start after image 0's, "),
        ([(1, 5), (5, 4)], "image 1's placeholder 5..=8 does not start after image 0's, "),
        ([(7, 5), (1, 4)], "image 1's placeholder 1..=4 does not start after image 0's, "),
    ],
    ids=["same-range", "overlap", "out-of-order"],
)
def test_placeholders_out_of_prompt_order_are_refused(v41_converter, spans, message):
    """SGLang writes the items' embeddings over the placeholders in prompt order:
    a shared position, or a range before the one of the image ahead of it, would
    put one image's embeddings on another's positions."""
    servicer, pb = v41_converter
    with pytest.raises(ValueError, match=message):
        _convert(servicer, pb, _v41_batch(pb, spans=spans))


def test_placeholders_that_touch_are_two_items(v41_converter):
    """A range may start right after the one before it ends."""
    servicer, pb = v41_converter
    result = _convert(servicer, pb, _v41_batch(pb, spans=[(1, 5), (6, 4)]))
    assert [item.offsets for item in result.mm_items] == [[(1, 5)], [(6, 9)]]


def _with_pixel_rows(pb, batch, row_shape):
    """``batch`` with each of its patch rows holding a ``row_shape`` of zeros."""
    shape = (batch.pixel_values.shape[0], *row_shape)
    batch.pixel_values.CopyFrom(_tensor(pb, [0.0] * math.prod(shape), shape, torch.float32))
    return batch


@pytest.mark.parametrize(
    "row_shape, values", [((3, 2, 3), 18), ((1,), 1)], ids=["3x2x3", "one-value"]
)
def test_patch_rows_must_fit_the_models_patches(v41_converter, row_shape, values):
    """SGLang's V4.1 patch embedding takes rows of 3 x patch_size x patch_size values
    and fails on any other width, in the scheduler."""
    servicer, pb = v41_converter
    batch = _with_pixel_rows(pb, _v41_batch(pb), row_shape)
    message = f"each patch row holds {values} values, but the model's 2x2 patches take 3x2x2 = 12"
    with pytest.raises(ValueError, match=message):
        _convert(servicer, pb, batch)


@pytest.mark.parametrize("types_per_image", [True, False])
def test_span_length_must_match_the_grid(v41_converter, types_per_image):
    """SGLang sizes an image's embedding from its grid alone. A span of another length
    is refused, also when the token counts agree with the spans or are absent."""
    servicer, pb = v41_converter
    batch = _v41_batch(pb, grids=[(2, 3), (2, 3)], patches=[6, 6], types_per_image=types_per_image)
    with pytest.raises(
        ValueError, match="image 1 fills 4 prompt positions but its 2x3 grid makes 5 tokens"
    ):
        _convert(servicer, pb, batch)


def test_a_one_patch_image_fills_four_positions(v41_converter):
    """A 1x1 grid makes a start, one token, a newline and an end."""
    servicer, pb = v41_converter
    batch = _v41_batch(pb, grids=[(1, 1)], patches=[1], spans=[(2, 4)])
    (item,) = _convert(servicer, pb, batch).mm_items
    assert item.offsets == [(2, 5)]


@pytest.mark.parametrize(
    "hf_config",
    [
        SimpleNamespace(model_type="deepseek_v41"),
        SimpleNamespace(model_type="other", vision_patch_size=2, vision_downsample_ratio=2),
    ],
    ids=["no-vision-fields", "another-model"],
)
def test_without_the_models_vision_config_its_checks_are_skipped(v41_converter, hf_config):
    """The width and grid checks need the served V4.1 model's patch size and
    downsample ratio; without them the batch converts as before."""
    servicer, pb = v41_converter
    servicer.model_config = SimpleNamespace(hf_config=hf_config)
    batch = _v41_batch(pb, grids=[(2, 3), (2, 3)], patches=[6, 6])
    batch = _with_pixel_rows(pb, batch, (3, 2, 3))
    assert len(_convert(servicer, pb, batch).mm_items) == 2


@pytest.mark.parametrize(
    "name, values, shape, dtype, message",
    [
        ("vit_grid", [2, 3, 1], (3,), torch.int64, "vit_grid holds 3 values, not"),
        (
            "vit_grid",
            [[2.0, 3.0], [1.0, 2.0]],
            (2, 2),
            torch.float32,
            "vit_grid is torch.float32, not an integer tensor",
        ),
        (
            "patches_per_image",
            [6.0, 2.0],
            (2,),
            torch.float32,
            "patches_per_image is torch.float32, not an integer tensor",
        ),
        (
            "pixel_values",
            [0.0] * 8,
            (8,),
            torch.float32,
            r"pixel_values has shape \[8\], not rows of patches",
        ),
    ],
)
def test_malformed_tensors_are_refused(v41_converter, name, values, shape, dtype, message):
    """Unpaired grids, counts that are not integers and pixels that are not patch rows are
    refused as the caller's error, not failed on inside the servicer."""
    servicer, pb = v41_converter
    batch = _v41_batch(pb)
    replacement = _tensor(pb, values, shape, dtype)
    target = batch.pixel_values if name == "pixel_values" else batch.model_specific_tensors[name]
    target.CopyFrom(replacement)
    with pytest.raises(ValueError, match=message):
        _convert(servicer, pb, batch)


def test_a_batch_without_images_is_refused(v41_converter):
    """Empty grids make no items; the batch is refused rather than sent with none."""
    servicer, pb = v41_converter
    batch = _v41_batch(pb, grids=[], patches=[], spans=[])
    with pytest.raises(ValueError, match="DeepSeek-V4.1 image batch: no images"):
        _convert(servicer, pb, batch)


@pytest.mark.parametrize("missing", [(), ("patches_per_image",)])
def test_grids_without_pixels_are_refused(v41_converter, missing):
    """V4.1 grids with no pixel_values are refused, not dropped with the images,
    also when the patch counts are missing as well."""
    servicer, pb = v41_converter
    batch = _v41_batch(pb)
    batch.ClearField("pixel_values")
    for name in missing:
        del batch.model_specific_tensors[name]
    with pytest.raises(ValueError, match="no pixel_values to cut its per-image items from"):
        _convert(servicer, pb, batch)


def test_a_batch_without_patch_counts_is_refused(v41_converter):
    """``vit_grid`` marks a V4.1 batch: without its patch counts it is refused, not sent
    to the scheduler as one item."""
    servicer, pb = v41_converter
    batch = _v41_batch(pb)
    del batch.model_specific_tensors["patches_per_image"]
    with pytest.raises(ValueError, match="DeepSeek-V4.1 image batch: no patches_per_image"):
        _convert(servicer, pb, batch)


def test_a_batch_without_its_image_token_id_is_refused(v41_converter):
    """SGLang pads the image positions only when the request names the image token
    id. Without it the model finds no position for the items' embeddings and fails
    in the scheduler, so the batch is refused here."""
    servicer, pb = v41_converter
    batch = _v41_batch(pb)
    batch.ClearField("im_token_id")
    with pytest.raises(ValueError, match="DeepSeek-V4.1 image batch: no im_token_id"):
        _convert(servicer, pb, batch)
    # Also as the extra batch of a mixed request, after a video batch.
    video = pb.MultimodalInputs(
        modalities=["video"],
        im_token_id=8,
        pixel_values=_tensor(pb, [0.0, 0.0], (2,), torch.float32),
        mm_placeholders=[pb.PlaceholderRange(offset=0, length=1)],
    )
    with pytest.raises(ValueError, match="DeepSeek-V4.1 image batch: no im_token_id"):
        _convert(servicer, pb, video, extras=[batch])


@pytest.mark.parametrize("row_shape", [(0,), (3, 0, 2)], ids=["no-values", "a-zero-dimension"])
def test_patch_rows_without_values_are_refused(v41_converter, row_shape):
    """A patch row holds at least one value. A pixel tensor with a row dimension of
    zero is refused when the model's geometry is not known, where an empty tensor
    would otherwise reach the scheduler."""
    servicer, pb = v41_converter
    servicer.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="other"))
    batch = _with_pixel_rows(pb, _v41_batch(pb), row_shape)
    with pytest.raises(ValueError, match="not rows of patches"):
        _convert(servicer, pb, batch)


def test_a_grid_on_a_video_batch_is_refused(v41_converter):
    """Only an image batch is cut into V4.1 items. A batch that carries ``vit_grid``
    under the video label would go on as a video item, which SGLang's V4.1 model has
    no embedder for, so it is refused: alone, and as the extra batch after an image
    batch."""
    servicer, pb = v41_converter
    video = _v41_batch(pb)
    video.modalities[:] = ["video"]
    message = "DeepSeek-V4.1 image batch: vit_grid on a video batch"
    with pytest.raises(ValueError, match=message):
        _convert(servicer, pb, video)
    with pytest.raises(ValueError, match=message):
        _convert(servicer, pb, _v41_batch(pb), extras=[video])


def test_media_references_without_tensors_still_convert(v41_converter):
    """A request carrying only media references (the encode dispatch's) is not refused."""
    servicer, pb = v41_converter
    batch = pb.MultimodalInputs(modalities=["image"], image_urls=["https://example.test/a.png"])
    assert _convert(servicer, pb, batch) is None


@pytest.mark.parametrize("patch_counts", [False, True], ids=["qwen", "glm53-flash"])
def test_other_families_keep_one_item_per_batch(v41_converter, patch_counts):
    """A batch without ``vit_grid`` is not cut: a Qwen-style ``image_grid_thw``, or
    GLM-5.3 Flash's, which carries ``patches_per_image`` as well."""
    servicer, pb = v41_converter
    tensors = {"image_grid_thw": _tensor(pb, [[1, 2, 2], [1, 2, 2]], (2, 3), torch.int64)}
    if patch_counts:
        tensors["patches_per_image"] = _tensor(pb, [4, 4], (2,), torch.int64)
    batch = pb.MultimodalInputs(
        modalities=["image"],
        im_token_id=9,
        pixel_values=_tensor(pb, [0.0] * 8, (8,), torch.float32),
        model_specific_tensors=tensors,
        mm_placeholders=[
            pb.PlaceholderRange(offset=1, length=2),
            pb.PlaceholderRange(offset=5, length=2),
        ],
    )
    (item,) = _convert(servicer, pb, batch).mm_items
    assert item.offsets == [(1, 2), (5, 6)]
    assert set(item.model_specific_data) == set(tensors)
    assert item.feature.shape == (8,)
    # Only a V4.1 batch must name its image token id: this one converts
    # without it, as before.
    batch.ClearField("im_token_id")
    result = _convert(servicer, pb, batch)
    assert result.im_token_id is None and len(result.mm_items) == 1
