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
from types import ModuleType, SimpleNamespace

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


class _MultimodalItem(SimpleNamespace):
    """The item protocol used here, including the SDK's unpadded defaults."""

    pad_value = None
    precomputed_embeddings = None
    hash = None

    def is_image(self):
        return self.modality == "image"


class _MultimodalOutput(SimpleNamespace):
    padded_input_ids = None


@pytest.fixture
def legacy_multimodal_sdk(monkeypatch):
    """Keep the legacy no-splitter boundary independent of installed engines.

    The real SMG normalizer still runs. Modern SDK delegation is exercised
    separately below, with the same converter and original batch inputs.
    """
    module = ModuleType("sglang.srt.managers.mm_utils")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


@pytest.fixture
def v41_converter(monkeypatch, servicer, legacy_multimodal_sdk):
    """Use the real converter/decoder with explicit media and legacy SDK protocols."""
    from smg_grpc_proto import sglang_scheduler_pb2 as pb

    spec = importlib.util.spec_from_file_location("_v41_tensor_wire", _TENSOR_WIRE)
    tensor_wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tensor_wire)

    module = sys.modules[type(servicer).__module__]
    monkeypatch.setattr(module, "Modality", SimpleNamespace(IMAGE="image", VIDEO="video"))
    monkeypatch.setattr(module, "MultimodalDataItem", _MultimodalItem)
    monkeypatch.setattr(module, "MultimodalProcessorOutput", _MultimodalOutput)
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


@pytest.mark.parametrize("patch_counts", [False, True], ids=["qwen", "glm53-flash"])
def test_other_families_are_delegated_to_the_sdk_without_v41_splitting(
    v41_converter, legacy_multimodal_sdk, patch_counts
):
    """The SDK may split an intact non-V4.1 batch into two per-image items.

    This controlled SDK response checks the real SMG normalizer's delegation
    boundary; it is not an execution of SGLang's implementation or GPU code.
    Distinct pixel values expose lost, reordered or duplicated image data.
    """
    servicer, pb = v41_converter
    seen = []
    returned = []

    def split(items):
        (item,) = items
        seen.append(item)
        assert item.is_image()
        assert item.offsets == [(1, 2), (5, 6)]
        assert item.feature.tolist() == list(range(8))
        assert item.model_specific_data["image_grid_thw"].tolist() == [[1, 2, 2]] * 2
        if patch_counts:
            assert item.model_specific_data["patches_per_image"].tolist() == [4, 4]
        for index, (start, stop) in enumerate([(0, 4), (4, 8)]):
            part = _MultimodalItem(
                modality=item.modality,
                feature=item.feature[start:stop].clone(),
                model_specific_data={
                    key: value[index : index + 1].clone()
                    for key, value in item.model_specific_data.items()
                },
                offsets=[item.offsets[index]],
            )
            returned.append(part)
        return returned

    legacy_multimodal_sdk.get_new_expanded_mm_items = split
    tensors = {"image_grid_thw": _tensor(pb, [[1, 2, 2], [1, 2, 2]], (2, 3), torch.int64)}
    if patch_counts:
        tensors["patches_per_image"] = _tensor(pb, [4, 4], (2,), torch.int64)
    batch = pb.MultimodalInputs(
        modalities=["image"],
        im_token_id=9,
        pixel_values=_tensor(pb, list(range(8)), (8,), torch.float32),
        model_specific_tensors=tensors,
        mm_placeholders=[
            pb.PlaceholderRange(offset=1, length=2),
            pb.PlaceholderRange(offset=5, length=2),
        ],
    )
    wire_before = batch.SerializeToString()
    result = _convert(servicer, pb, batch)

    assert len(seen) == 1
    assert len(result.mm_items) == 2
    assert all(actual is expected for actual, expected in zip(result.mm_items, returned))
    assert [item.offsets for item in result.mm_items] == [[(1, 2)], [(5, 6)]]
    assert [item.feature.tolist() for item in result.mm_items] == [
        list(range(4)),
        list(range(4, 8)),
    ]
    assert [item.model_specific_data["image_grid_thw"].tolist() for item in result.mm_items] == [
        [[1, 2, 2]],
        [[1, 2, 2]],
    ]
    if patch_counts:
        assert [
            item.model_specific_data["patches_per_image"].tolist() for item in result.mm_items
        ] == [
            [4],
            [4],
        ]
    assert result.im_token_id == 9
    assert seen[0].feature.tolist() == list(range(8))
    assert seen[0].offsets == [(1, 2), (5, 6)]
    assert seen[0].pad_value is None and seen[0].hash is None
    assert batch.SerializeToString() == wire_before


_INSTALLED_SGLANG_MEDIA_BRIDGE = r'''"""Run by the proposed test in a fresh interpreter, never by preparation."""

import ast
import base64
import hashlib
import importlib
import importlib.metadata
import json
import marshal
import os
from pathlib import Path
import resource
import sys
import traceback
import urllib.parse


def file_pin(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def distribution_pin(name):
    dist = importlib.metadata.distribution(name)
    files = list(dist.files or [])
    result = {"name": dist.metadata["Name"], "version": dist.version, "metadata": {}}
    for leaf in ("METADATA", "WHEEL", "RECORD"):
        matches = [f for f in files if f.name == leaf and ".dist-info" in str(f.parent)]
        assert len(matches) == 1, (name, leaf, "installed wheel metadata unavailable")
        result["metadata"][leaf] = file_pin(dist.locate_file(matches[0]))
    # An installed wheel's WHEEL/RECORD is observable; its original archive need
    # not remain cached. Do not manufacture an archive digest from these files.
    result["original_wheel_archive_sha256"] = None
    return dist, files, result


def record_file_pin(dist, row):
    pin = file_pin(dist.locate_file(row))
    assert row.hash and row.hash.mode == "sha256", (str(row), "missing SHA256 RECORD anchor")
    encoded = base64.urlsafe_b64encode(bytes.fromhex(pin["sha256"])).decode().rstrip("=")
    assert encoded == row.hash.value and row.size == pin["bytes"], (str(row), "RECORD mismatch")
    return dict(pin, RECORD_path=str(row), RECORD_sha256=row.hash.value)


def editable_module_pin(module, dist, files, pin):
    # A RECORD miss alone never authorizes arbitrary source/module fallback.
    assert dist.metadata["Name"].lower().replace("_", "-") == "sglang"
    assert module.__name__ == "sglang" or module.__name__.startswith("sglang.")
    direct_rows = [f for f in files if f.name == "direct_url.json" and ".dist-info" in str(f.parent)]
    assert len(direct_rows) == 1, "editable source requires distribution-owned direct_url.json"
    direct_pin = record_file_pin(dist, direct_rows[0])
    direct = json.loads(Path(direct_pin["path"]).read_text())
    assert set(direct) <= {"url", "dir_info", "subdirectory"}
    assert direct.get("dir_info", {}).get("editable") is True, "unproven source install is not editable"
    url = urllib.parse.urlsplit(direct["url"])
    assert url.scheme == "file" and url.netloc in ("", "localhost")
    assert not url.query and not url.fragment and not url.username and not url.password
    project = Path(urllib.parse.unquote(url.path, errors="strict"))
    assert project.is_absolute()
    if "subdirectory" in direct:
        relative = Path(direct["subdirectory"])
        assert not relative.is_absolute() and ".." not in relative.parts
        project = project / relative
    project = project.resolve(strict=True)
    package = sys.modules.get("sglang")
    assert package is not None, "the real parent package must already have been imported"
    package_init = file_pin(package.__file__)
    package_root = Path(package_init["path"]).parent
    candidates = [project / "sglang", project / "src/sglang", project / "python/sglang"]
    assert package_root in [p.resolve() for p in candidates]
    assert Path(package_init["path"]).name == "__init__.py"
    assert Path(package.__spec__.origin).resolve() == Path(package_init["path"])
    assert [Path(p).resolve() for p in package.__spec__.submodule_search_locations] == [package_root]
    source_version = vars(package).get("__version__")
    assert isinstance(source_version, str) and source_version == dist.version
    project_metadata = [file_pin(project / name) for name in ("pyproject.toml", "setup.py")
                        if (project / name).is_file()]
    assert project_metadata, "editable project metadata is not observable"
    expected = package_root.joinpath(*module.__name__.split(".")[1:])
    expected = expected / "__init__.py" if vars(module).get("__path__") is not None else expected.with_suffix(".py")
    path = Path(pin["path"])
    assert path.is_relative_to(package_root) and path == expected.resolve(strict=True)
    assert Path(module.__spec__.origin).resolve() == path and path.suffix == ".py"
    bindings = []
    for row in files:
        if row.suffix != ".pth":
            continue
        pth = record_file_pin(dist, row)
        lines = [line.strip() for line in Path(pth["path"]).read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        if len(lines) != 1:
            continue
        line = lines[0]
        if not line.startswith(("import ", "import\t")):
            entry = Path(line)
            if not entry.is_absolute():
                entry = Path(pth["path"]).parent / entry
            if entry.resolve() == package_root.parent and any(
                Path(p).resolve() == entry.resolve() for p in sys.path if p
            ):
                bindings.append({"kind": "PEP610-editable-path-pth", "files": [pth]})
            continue
        # Inspect, never execute, the finite setuptools editable-finder form.
        statements = ast.parse(line).body
        if len(statements) != 2 or not isinstance(statements[0], ast.Import):
            continue
        aliases = statements[0].names
        if len(aliases) != 1 or aliases[0].asname is not None:
            continue
        finder_name = aliases[0].name
        if not (finder_name.isidentifier() and finder_name.startswith("__editable__")):
            continue
        call = statements[1].value if isinstance(statements[1], ast.Expr) else None
        if not (isinstance(call, ast.Call) and not call.args and not call.keywords
                and isinstance(call.func, ast.Attribute) and call.func.attr == "install"
                and isinstance(call.func.value, ast.Name) and call.func.value.id == finder_name):
            continue
        finder_rows = [f for f in files if f.name == finder_name + ".py"]
        assert len(finder_rows) == 1, "editable finder must be distribution RECORD-owned"
        finder_pin = record_file_pin(dist, finder_rows[0])
        parsed = ast.parse(Path(finder_pin["path"]).read_text())
        values = []
        for statement in parsed.body:
            if isinstance(statement, ast.Assign):
                targets = statement.targets
            elif isinstance(statement, ast.AnnAssign):
                targets = [statement.target]
            else:
                continue
            if any(isinstance(target, ast.Name) and target.id == "MAPPING" for target in targets):
                values.append(ast.literal_eval(statement.value))
        assert len(values) == 1 and isinstance(values[0], dict)
        mapping = values[0]
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items())
        finder = sys.modules.get(finder_name)
        assert finder is not None and Path(finder.__file__).resolve() == Path(finder_pin["path"])
        assert Path(finder.__spec__.origin).resolve() == Path(finder_pin["path"])
        assert vars(finder).get("MAPPING") == mapping
        live_finder = vars(finder).get("_EditableFinder")
        assert live_finder is not None and live_finder.__module__ == finder_name
        assert live_finder in sys.meta_path
        resolver = live_finder.find_spec
        resolver_code = getattr(resolver, "__func__", resolver).__code__
        assert Path(resolver_code.co_filename).resolve() == Path(finder_pin["path"])
        assert "sglang" in mapping and Path(mapping["sglang"]).resolve() == package_root
        bindings.append({"kind": "PEP610-editable-record-owned-finder", "files": [pth, finder_pin]})
    assert len(bindings) == 1, "missing or ambiguous active distribution-owned editable import mapping"
    evidence = [direct_pin, package_init, *project_metadata, *bindings[0]["files"]]
    version_module = sys.modules.get("sglang.version")
    if version_module is not None:
        version_pin = file_pin(version_module.__file__)
        assert Path(version_pin["path"]).is_relative_to(package_root)
        evidence.append(version_pin)
    pin.update(
        module=module.__name__, RECORD_path=None, RECORD_sha256=None,
        installation_kind=bindings[0]["kind"], installation_evidence=evidence,
        editable_project_root=str(project), editable_package_root=str(package_root),
        actual_source_package_version=source_version, installed_distribution_version=dist.version,
        scope="installed metadata and active editable mapping to actual source bytes; no archive/Git identity inferred",
    )
    return pin


def installed_module_pin(module, dist, files):
    pin = file_pin(module.__file__)
    path = Path(pin["path"])
    rows = [f for f in files if Path(dist.locate_file(f)).resolve() == path]
    if not rows:
        return editable_module_pin(module, dist, files, pin)
    assert len(rows) == 1 and rows[0].hash, (module.__name__, "module not RECORD-bound")
    row = rows[0]
    assert row.hash.mode == "sha256", (module.__name__, row.hash.mode)
    encoded = base64.urlsafe_b64encode(bytes.fromhex(pin["sha256"])).decode().rstrip("=")
    assert encoded == row.hash.value, (module.__name__, "installed module RECORD mismatch")
    pin.update(module=module.__name__, RECORD_path=str(row), RECORD_sha256=row.hash.value)
    return pin


def main():
    root, proto_root, output = map(Path, sys.argv[1:])
    root = root.resolve(strict=True)
    proto_root = proto_root.resolve(strict=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (262144, 262144))
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    # -I gives a new module table and ignores PYTHONPATH/user-site. These are
    # actual checkout/proto roots, not modules copied from pytest's fixtures.
    sys.path[:0] = [str(root / "grpc_servicer"), str(proto_root)]
    result = {
        "status": "FAIL",
        "scope": "real installed SDK + SMG normalizer/tensor wire/protobuf/media parser",
        "real_servicer_module_imported": False,
        "real_media_parser_called": False,
        "full_generate_request_or_model_E2E_executed": False,
        "python": {
            "executable": sys.executable, "version": sys.version,
            "real_executable": file_pin(sys.executable),
            "prefix": sys.prefix, "base_prefix": sys.base_prefix,
        },
        "bridge_script": file_pin(__file__),
        "stage": "installed distributions",
    }
    process_stat = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    result["native_self_identity"] = {
        "PID": os.getpid(), "start": int(process_stat[19]),
        "PGID": os.getpgrp(), "SID": os.getsid(0),
    }
    try:
        sdk_dist, sdk_files, sdk_metadata = distribution_pin("sglang")
        torch_dist, torch_files, torch_metadata = distribution_pin("torch")
        result["distributions"] = {"sglang": sdk_metadata, "torch": torch_metadata}
        result["stage"] = "real imports"
        import torch
        import google.protobuf
        from smg_grpc_proto import sglang_scheduler_pb2 as pb
        from smg_grpc_servicer import tensor_wire
        from smg_grpc_servicer.sglang import compat

        torch.set_num_threads(2)
        Modality, Item, Output = compat.get_multimodal_types()
        assert Item.__module__.startswith("sglang.") and Output.__module__.startswith("sglang.")
        result["real_SDK_types"] = {
            cls.__name__: cls.__module__ + "." + cls.__qualname__
            for cls in (Modality, Item, Output)
        }
        types_module = importlib.import_module(Item.__module__)
        output_module = importlib.import_module(Output.__module__)
        mm_name = "sglang.srt.managers.mm_utils"
        try:
            mm_utils = importlib.import_module(mm_name)
        except ModuleNotFoundError as exc:
            if exc.name != mm_name:
                raise
            mm_utils = None
        splitter = None
        if mm_utils is not None and "get_new_expanded_mm_items" in vars(mm_utils):
            splitter = mm_utils.get_new_expanded_mm_items
            assert callable(splitter)
            assert splitter.__module__ == mm_utils.__name__
            assert Path(splitter.__code__.co_filename).resolve() == Path(mm_utils.__file__).resolve()
        modern = splitter is not None
        result["SDK_mode"] = "modern_splitter" if modern else "legacy_no_splitter"
        result["original_0_5_21_modern_qualification"] = False
        if sdk_dist.version.split("+")[0] == "0.5.21":
            assert modern, "the original installed0.5.21 requires the real modern splitter"
        expected_count = 2 if modern else 1
        result["SDK_modules"] = {
            m.__name__: installed_module_pin(m, sdk_dist, sdk_files)
            for m in (types_module, output_module, mm_utils) if m is not None
        }
        result["torch"] = installed_module_pin(torch, torch_dist, torch_files)
        result["torch_version"] = torch.__version__
        proto_dist, proto_files, proto_metadata = distribution_pin("protobuf")
        result["distributions"]["protobuf"] = proto_metadata
        result["protobuf_runtime"] = installed_module_pin(
            google.protobuf, proto_dist, proto_files
        )
        try:
            result["SMG_proto_distribution_version"] = importlib.metadata.version("smg-grpc-proto")
        except importlib.metadata.PackageNotFoundError:
            result["SMG_proto_distribution_version"] = None
        result["real_callable"] = {
            "module": splitter.__module__, "qualname": splitter.__qualname__,
            "co_filename": splitter.__code__.co_filename,
            "live_code_sha256": hashlib.sha256(marshal.dumps(splitter.__code__)).hexdigest(),
        } if modern else None
        if "torch._C" in sys.modules:
            result["torch_native_module"] = installed_module_pin(
                sys.modules["torch._C"], torch_dist, torch_files
            )
        assert Path(compat.__file__).resolve() == root / "grpc_servicer/smg_grpc_servicer/sglang/compat.py"
        assert Path(tensor_wire.__file__).resolve() == root / "grpc_servicer/smg_grpc_servicer/tensor_wire.py"
        result["SMG_modules"] = {m.__name__: file_pin(m.__file__) for m in (compat, tensor_wire)}
        result["Source_proto"] = file_pin(root / "crates/grpc_client/proto/sglang_scheduler.proto")
        result["proto_descriptor_sha256"] = hashlib.sha256(pb.DESCRIPTOR.serialized_pb).hexdigest()
        result["proto_modules"] = {
            name: file_pin(module.__file__)
            for name, module in list(sys.modules.items())
            if name.startswith("smg_grpc_proto") and getattr(module, "__file__", None)
        }
        result["stage"] = "actual SDK and normalizer calls"

        def wire_tensor(values, shape, dtype):
            tensor = torch.tensor(values, dtype=dtype, device="cpu")
            label = "float32" if dtype == torch.float32 else "int64"
            return pb.TensorData(dtype=label, shape=shape, data=tensor.numpy().tobytes())

        def decode(tensor):
            return tensor_wire.tensor_from_parts(tensor.data, tensor.shape, tensor.dtype)

        result["cases"] = []
        wire_cases = []
        for patch_counts in (False, True):
            tensors = {"image_grid_thw": wire_tensor([[1, 2, 2], [1, 2, 2]], [2, 3], torch.int64)}
            if patch_counts:
                tensors["patches_per_image"] = wire_tensor([4, 4], [2], torch.int64)
            batch = pb.MultimodalInputs(
                modalities=["image"], im_token_id=9,
                pixel_values=wire_tensor(list(range(8)), [8], torch.float32),
                model_specific_tensors=tensors,
                mm_placeholders=[pb.PlaceholderRange(offset=1, length=2), pb.PlaceholderRange(offset=5, length=2)],
            )
            wire_before = batch.SerializeToString(deterministic=True)
            wire_cases.append((patch_counts, wire_before))

            def item():
                return Item(
                    modality=Modality.IMAGE, feature=decode(batch.pixel_values),
                    model_specific_data={k: decode(v) for k, v in batch.model_specific_tensors.items()},
                    offsets=[(p.offset, p.offset + p.length - 1) for p in batch.mm_placeholders],
                )

            direct_input = item()
            sdk_parts = splitter([direct_input]) if modern else None
            original = item()
            original_features = original.feature.tolist()
            output_before = Output(mm_items=[original], im_token_id=9, video_token_id=None)
            actual = compat.normalize_multimodal_processor_output(output_before)
            if modern:
                assert mm_utils.get_new_expanded_mm_items is splitter
            for parts in ((sdk_parts, actual.mm_items) if modern else ()):
                assert len(parts) == 2
                assert [p.offsets for p in parts] == [[(1, 2)], [(5, 6)]]
                assert [p.feature.tolist() for p in parts] == [list(range(4)), list(range(4, 8))]
                assert all(tuple(p.feature.shape) == (4,) and p.feature.device.type == "cpu" for p in parts)
                assert all(p.feature.dtype == torch.float32 and p.is_image() for p in parts)
                assert [p.model_specific_data["image_grid_thw"].tolist() for p in parts] == [[[1, 2, 2]], [[1, 2, 2]]]
                assert all(set(p.model_specific_data) == set(tensors) for p in parts)
                if patch_counts:
                    assert [p.model_specific_data["patches_per_image"].tolist() for p in parts] == [[4], [4]]
            if not modern:
                assert actual is output_before and actual.mm_items == [original]
                assert actual.mm_items[0].model_specific_data["image_grid_thw"].tolist() == [[1, 2, 2]] * 2
                if patch_counts:
                    assert actual.mm_items[0].model_specific_data["patches_per_image"].tolist() == [4, 4]
            assert actual.im_token_id == 9
            if modern:
                assert actual.padded_input_ids is None
            assert output_before.mm_items == [original]
            assert original.feature.tolist() == original_features == list(range(8))
            assert original.offsets == [(1, 2), (5, 6)]
            if modern:
                assert original.pad_value is None and original.hash is None
            assert original.model_specific_data["image_grid_thw"].tolist() == [[1, 2, 2]] * 2
            assert direct_input.offsets == [(1, 2), (5, 6)]
            assert direct_input.feature.tolist() == list(range(8))
            assert batch.SerializeToString(deterministic=True) == wire_before
            # Preserve the original test's second conversion without im_token_id.
            batch.ClearField("im_token_id")
            wire_without_token = batch.SerializeToString(deterministic=True)
            without_token = compat.normalize_multimodal_processor_output(
                Output(mm_items=[item()], im_token_id=None, video_token_id=None)
            )
            assert without_token.im_token_id is None and len(without_token.mm_items) == expected_count
            assert [p.feature.tolist() for p in without_token.mm_items] == (
                [list(range(4)), list(range(4, 8))] if modern else [list(range(8))]
            )
            assert [p.offsets for p in without_token.mm_items] == (
                [[(1, 2)], [(5, 6)]] if modern else [[(1, 2), (5, 6)]]
            )
            assert batch.SerializeToString(deterministic=True) == wire_without_token
            result["cases"].append({
                "family": "glm53-flash" if patch_counts else "qwen", "status": "PASS",
                "input_items": 1, "input_images": 2, "actual_output_items": len(actual.mm_items),
                "feature_rows": [len(p.feature) for p in actual.mm_items],
                "pixels": [p.feature.tolist() for p in actual.mm_items],
                "offsets": [p.offsets for p in actual.mm_items],
                "grid": [p.model_specific_data["image_grid_thw"].tolist() for p in actual.mm_items],
                "patch_counts": (
                    [p.model_specific_data["patches_per_image"].tolist() for p in actual.mm_items]
                    if patch_counts else None
                ),
                "im_token_id": actual.im_token_id, "wire_sha256": hashlib.sha256(wire_before).hexdigest(),
                "token_variants": [9, None],
                "wire_without_token_sha256": hashlib.sha256(wire_without_token).hexdigest(),
                "input_and_wire_unchanged": True,
            })
        result["normalizer_stage_status"] = "PASS"
        if modern:
            # Import the actual module, without any of pytest's manager_mod imports.
            # __new__ avoids creating a backend, scheduler or model. This specific
            # non-V4.1 parser path needs no model configuration or sampling stand-in.
            result["stage"] = "real SMG media-parser import and calls"
            servicer_module = importlib.import_module("smg_grpc_servicer.sglang.servicer")
            assert Path(servicer_module.__file__).resolve() == (
                root / "grpc_servicer/smg_grpc_servicer/sglang/servicer.py"
            )
            result["SMG_modules"][servicer_module.__name__] = file_pin(servicer_module.__file__)
            result["real_servicer_module_imported"] = True
            cls = servicer_module.SGLangSchedulerServicer
            parser = cls.__new__(cls)
            result["real_media_parser"] = {
                "co_filename": cls._parse_mm_inputs.__code__.co_filename,
                "live_code_sha256": hashlib.sha256(marshal.dumps(cls._parse_mm_inputs.__code__)).hexdigest(),
            }
            result["media_parser_cases"] = []
            for patch_counts, original_wire in wire_cases:
                for token in (9, None):
                    batch = pb.MultimodalInputs()
                    batch.ParseFromString(original_wire)
                    if token is None:
                        batch.ClearField("im_token_id")
                    wire = batch.SerializeToString(deterministic=True)
                    parsed = parser._parse_mm_inputs(batch, [], 12)
                    assert len(parsed.mm_items) == expected_count and parsed.im_token_id == token
                    assert [p.feature.tolist() for p in parsed.mm_items] == (
                        [list(range(4)), list(range(4, 8))] if modern else [list(range(8))]
                    )
                    assert [p.offsets for p in parsed.mm_items] == (
                        [[(1, 2)], [(5, 6)]] if modern else [[(1, 2), (5, 6)]]
                    )
                    assert [p.model_specific_data["image_grid_thw"].tolist() for p in parsed.mm_items] == (
                        [[[1, 2, 2]], [[1, 2, 2]]] if modern else [[[1, 2, 2], [1, 2, 2]]]
                    )
                    if patch_counts:
                        assert [p.model_specific_data["patches_per_image"].tolist() for p in parsed.mm_items] == (
                            [[4], [4]] if modern else [[4, 4]]
                        )
                    assert all(p.feature.device.type == "cpu" and p.is_image() for p in parsed.mm_items)
                    assert batch.SerializeToString(deterministic=True) == wire
                    result["media_parser_cases"].append({
                        "family": "glm53-flash" if patch_counts else "qwen", "im_token_id": token,
                        "status": "PASS", "actual_output_items": len(parsed.mm_items),
                        "wire_sha256": hashlib.sha256(wire).hexdigest(), "wire_unchanged": True,
                    })
            result["real_media_parser_called"] = True
        else:
            result["media_parser_stage_status"] = "NOT_APPLICABLE_LEGACY_NO_SPLITTER"
            result["media_parser_cases"] = []
            result["scope"] = "real legacy SDK types + SMG normalizer/tensor wire/protobuf"
        for name, module in list(sys.modules.items()):
            if name.startswith("smg_grpc_servicer") and getattr(module, "__file__", None):
                source_path = Path(module.__file__).resolve()
                assert source_path.is_relative_to(root / "grpc_servicer"), (name, source_path)
                result["SMG_modules"][name] = file_pin(source_path)
        for pins in (result["SDK_modules"], result["SMG_modules"], result["proto_modules"]):
            assert all(file_pin(pin["path"])["sha256"] == pin["sha256"] for pin in pins.values())
            for pin in pins.values():
                for evidence in pin.get("installation_evidence", []):
                    assert file_pin(evidence["path"])["sha256"] == evidence["sha256"]
        result.update(
            status="PASS", stage="complete", actual_SDK_function_called=modern,
            original_0_5_21_modern_qualification=sdk_dist.version.split("+")[0] == "0.5.21" and modern,
        )
    except BaseException as exc:
        result.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    output.write_text(json.dumps(result, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def test_installed_sglang_sdk_media_bridge_in_fresh_process(tmp_path, capsys):
    """Record a real installed-SDK bridge, independently of all engine fixtures.

    This test is not applicable where SGLang is not installed. Such a skip is
    never a successful SDK qualification: the SGLang CI lane must observe PASS
    and the provenance/cleanup receipt below, including both image profiles.
    """
    import base64
    import hashlib
    import importlib.metadata
    import json
    import os
    import signal
    import subprocess
    import time

    try:
        installed_sdk_version = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("real SGLang SDK absent: SDK bridge qualification NOT ESTABLISHED")
    proto = importlib.util.find_spec("smg_grpc_proto")
    assert proto is not None and proto.submodule_search_locations
    roots = list(proto.submodule_search_locations)
    assert len(roots) == 1, "bind the actual protobuf package instead of guessing a root"
    proto_root = str(Path(roots[0]).resolve().parent)
    root = Path(__file__).resolve().parents[2]
    work = tmp_path / "installed-sdk-bridge"
    work.mkdir(mode=0o700)
    script = work / "bridge.py"
    output = work / "report.json"
    stdout_path = work / "stdout"
    stderr_path = work / "stderr"
    script.write_text(_INSTALLED_SGLANG_MEDIA_BRIDGE)
    argv = [sys.executable, "-I", "-B", str(script), str(root), proto_root, str(output)]
    keep = {
        "HOME",
        "PATH",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "VIRTUAL_ENV",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "TMPDIR",
        "E2E_JOB_MARKER",
    }
    env = {key: value for key, value in os.environ.items() if key in keep}
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
    )
    receipt = {
        "test": "test_installed_sglang_sdk_media_bridge_in_fresh_process",
        "SDK_distribution_version_in_parent": installed_sdk_version,
        "argv": argv,
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "test_Source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inherited_environment_keys": sorted(env),
        "vendor_credentials_exported": False,
        "full_servicer_converter_or_model_E2E_qualified": False,
    }
    child = None
    problem = None
    interrupted = None
    actual = None
    raw = {}

    def group_exists(pid):
        try:
            os.killpg(pid, 0)
            return True
        except ProcessLookupError:
            return False

    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            child = subprocess.Popen(
                argv,
                cwd=root,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                fields = Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()
                receipt["native_handle"] = {
                    "PID": child.pid,
                    "start": int(fields[19]),
                    "PGID": int(fields[2]),
                    "SID": int(fields[3]),
                }
            except FileNotFoundError:
                receipt["native_handle"] = {"PID": child.pid, "already_exited": True}
            until = time.monotonic() + 120
            while child.poll() is None:
                assert time.monotonic() < until, "installed SDK bridge wall timeout"
                assert stdout_path.stat().st_size + stderr_path.stat().st_size <= 524288
                time.sleep(0.05)
            receipt["native_exit"] = child.wait()
        if output.exists():
            assert output.stat().st_size <= 262144
            actual = json.loads(output.read_text())
    except BaseException as exc:
        problem = repr(exc)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            interrupted = exc
    finally:
        cleanup_errors = []
        if child is not None:
            try:
                # The child owns a new session/group. Keep E2E_JOB_MARKER for
                # the existing runner cleanup if the entire test is cancelled.
                for sig, grace in [(signal.SIGTERM, 2), (signal.SIGKILL, 3)]:
                    if not group_exists(child.pid):
                        break
                    try:
                        os.killpg(child.pid, sig)
                    except ProcessLookupError:
                        pass
                    end = time.monotonic() + grace
                    while group_exists(child.pid) and time.monotonic() < end:
                        child.poll()
                        time.sleep(0.05)
                receipt["native_exit"] = child.wait(timeout=2)
            except BaseException as exc:
                cleanup_errors.append(repr(exc))
            receipt["reaped"] = child.poll() is not None
            try:
                receipt["owned_group_gone"] = not group_exists(child.pid)
            except OSError as exc:
                receipt["owned_group_gone"] = False
                cleanup_errors.append(repr(exc))
        else:
            receipt.update(reaped=True, owned_group_gone=True, native_exit=None)
        for name, path in [("stdout", stdout_path), ("stderr", stderr_path), ("report", output)]:
            size = path.stat().st_size if path.exists() else 0
            if path.exists():
                with path.open("rb") as stream:
                    data = stream.read(262144)
            else:
                data = b""
            raw[name] = {
                "bytes_total": size,
                "bytes_captured": len(data),
                "truncated": size != len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "base64": base64.b64encode(data).decode(),
            }
        if receipt["reaped"] and receipt["owned_group_gone"]:
            try:
                for path in (script, output, stdout_path, stderr_path):
                    path.unlink(missing_ok=True)
                work.rmdir()
            except OSError as exc:
                cleanup_errors.append(repr(exc))
        receipt.update(
            actual=actual,
            error=problem,
            cleanup_errors=cleanup_errors,
            raw=raw,
            owned_tmp_absent=not work.exists(),
            retained_private_tmp=None if not work.exists() else str(work),
        )
        # -q normally hides successful captured output. Preserve the real
        # provenance, raw bytes and ACK in the original normal CI job log.
        with capsys.disabled():
            print(
                "\nSMG_REAL_SGLANG_SDK_BRIDGE_RECEIPT=" + json.dumps(receipt, sort_keys=True),
                flush=True,
            )

    if interrupted is not None:
        raise interrupted
    assert not receipt["cleanup_errors"], receipt
    assert all(not entry["truncated"] for entry in receipt["raw"].values()), receipt
    assert problem is None, problem
    assert receipt["native_exit"] == 0, receipt
    assert receipt["reaped"] and receipt["owned_group_gone"] and receipt["owned_tmp_absent"]
    assert actual is not None and actual["status"] == "PASS", actual
    mode = actual["SDK_mode"]
    assert mode in {"modern_splitter", "legacy_no_splitter"}
    modern = mode == "modern_splitter"
    assert actual["actual_SDK_function_called"] is modern
    assert actual["real_media_parser_called"] is modern
    if installed_sdk_version.split("+")[0] == "0.5.21":
        assert modern and actual["original_0_5_21_modern_qualification"]
    native = actual["native_self_identity"]
    assert native["PID"] == receipt["native_handle"]["PID"] == native["PGID"] == native["SID"]
    if "start" in receipt["native_handle"]:
        assert native["start"] == receipt["native_handle"]["start"]
    assert actual["distributions"]["sglang"]["version"] == installed_sdk_version
    assert [case["family"] for case in actual["cases"]] == ["qwen", "glm53-flash"]
    assert all(
        case["status"] == "PASS"
        and case["actual_output_items"] == (2 if modern else 1)
        and case["token_variants"] == [9, None]
        for case in actual["cases"]
    )
    if modern:
        assert {(case["family"], case["im_token_id"]) for case in actual["media_parser_cases"]} == {
            ("qwen", 9),
            ("qwen", None),
            ("glm53-flash", 9),
            ("glm53-flash", None),
        }
        assert all(
            case["status"] == "PASS" and case["actual_output_items"] == 2 and case["wire_unchanged"]
            for case in actual["media_parser_cases"]
        )
    else:
        assert actual["media_parser_stage_status"] == "NOT_APPLICABLE_LEGACY_NO_SPLITTER"
    with capsys.disabled():
        print(
            "\nSMG_REAL_SGLANG_SDK_BRIDGE_TEST_PASS="
            + json.dumps(
                {
                    "test": receipt["test"],
                    "SDK_version": actual["distributions"]["sglang"]["version"],
                    "SDK_mode": actual["SDK_mode"],
                    "original_0_5_21_modern_qualification": actual[
                        "original_0_5_21_modern_qualification"
                    ],
                    "real_media_parser_called": actual["real_media_parser_called"],
                    "receipt_sha256": hashlib.sha256(
                        json.dumps(receipt, sort_keys=True).encode()
                    ).hexdigest(),
                    "test_Source_sha256": receipt["test_Source_sha256"],
                    "scope": actual["scope"],
                    "full_servicer_converter_or_model_E2E_qualified": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
