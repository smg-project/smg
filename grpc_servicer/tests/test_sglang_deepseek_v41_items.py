"""DeepSeek-V4.1 images reach SGLang as one item per image, the shape its own processor builds.

The Router sends a V4.1 image batch as one tensor of every image's patches with
the per-image grids alongside (``vit_grid``, ``patches_per_image``). SGLang's
model reads ``item.n_vit_h`` / ``item.n_vit_w`` from each item and encodes one
image per item, so the servicer cuts the batch into items before the scheduler
sees it, and refuses a batch whose counts disagree.
"""

import importlib.util
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
# token spans of 5 and 4 positions in a 12-token prompt.
GRIDS = [(2, 3), (1, 2)]
PATCHES = [6, 2]
SPANS = [(1, 5), (7, 4)]
PROMPT_LEN = 12


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


def _convert(servicer, pb, batch):
    request = pb.GenerateRequest(
        tokenized=pb.TokenizedInput(input_ids=list(range(PROMPT_LEN))),
        mm_inputs=batch,
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


def test_other_families_keep_one_item_per_batch(v41_converter):
    """A batch without the V4.1 grids (a Qwen-style ``image_grid_thw``) is not cut."""
    servicer, pb = v41_converter
    batch = pb.MultimodalInputs(
        modalities=["image"],
        im_token_id=9,
        pixel_values=_tensor(pb, [0.0] * 8, (8,), torch.float32),
        model_specific_tensors={
            "image_grid_thw": _tensor(pb, [[1, 2, 2], [1, 2, 2]], (2, 3), torch.int64),
        },
        mm_placeholders=[
            pb.PlaceholderRange(offset=1, length=2),
            pb.PlaceholderRange(offset=5, length=2),
        ],
    )
    (item,) = _convert(servicer, pb, batch).mm_items
    assert item.offsets == [(1, 2), (5, 6)]
    assert set(item.model_specific_data) == {"image_grid_thw"}
    assert item.feature.shape == (8,)
