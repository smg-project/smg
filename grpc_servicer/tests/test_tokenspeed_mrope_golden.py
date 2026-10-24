"""The engine's MRoPE position rule, pinned on the shapes the Rust servicer's
port is tested on.

The Rust TokenSpeed servicer computes the Qwen3.5 lineage's MRoPE position
table itself (the msgpack wire bypasses the engine's input processor, which
computes it on the Python path). Its unit tests pin these exact tables; this
test pins the same tables on the engine's own ``compute_mrope_positions``, so
an engine release that changes the rule turns red here, where the engine is
installed, and names the port as the thing to update.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
mrope = pytest.importorskip("tokenspeed.runtime.multimodal.mrope")
inputs = pytest.importorskip("tokenspeed.runtime.multimodal.inputs")

IMAGE_TOKEN = 248056
VIDEO_TOKEN = 248057
VISION_START = 248053
VISION_END = 248054

HF_CONFIG = SimpleNamespace(
    architectures=["Qwen3_5ForConditionalGeneration"],
    model_type="qwen3_5",
    image_token_id=IMAGE_TOKEN,
    video_token_id=VIDEO_TOKEN,
    vision_start_token_id=VISION_START,
    vision_config=SimpleNamespace(spatial_merge_size=2),
)


def _item(modality, offsets, grid):
    key = f"{modality.name.lower()}_grid_thw"
    return inputs.MultimodalDataItem(
        modality=modality,
        offsets=offsets,
        model_specific_data={key: torch.tensor(grid, dtype=torch.long)},
    )


def _positions(input_ids, items):
    positions, delta = mrope.compute_mrope_positions(HF_CONFIG, input_ids, items)
    return positions.tolist(), int(delta.flatten()[0].item())


def test_an_image_grid_lays_out_over_time_height_and_width():
    # Mirrors the Rust test of the same name (engine_zmq_adapter tokenspeed::mrope).
    ids = [10, VISION_START, *([IMAGE_TOKEN] * 6), 11]
    positions, delta = _positions(ids, [_item(inputs.Modality.IMAGE, [(2, 7)], [[1, 4, 6]])])
    assert positions == [
        [0, 1, 2, 2, 2, 2, 2, 2, 5],
        [0, 1, 2, 2, 2, 3, 3, 3, 5],
        [0, 1, 2, 3, 4, 2, 3, 4, 5],
    ]
    assert delta == -3


def test_a_video_row_is_one_frame_per_temporal_step():
    ids = [
        VISION_START,
        42,
        *([VIDEO_TOKEN] * 4),
        VISION_END,
        42,
        VISION_START,
        *([VIDEO_TOKEN] * 4),
        VISION_END,
    ]
    positions, delta = _positions(
        ids, [_item(inputs.Modality.VIDEO, [(2, 5), (9, 12)], [[2, 4, 4]])]
    )
    assert positions == [
        [0, 1, 2, 2, 2, 2, 4, 5, 6, 7, 7, 7, 7, 9],
        [0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7, 8, 8, 9],
        [0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 7, 8, 9],
    ]
    assert delta == -4


def test_mixed_items_advance_by_each_grids_largest_side():
    ids = [
        10,
        VISION_START,
        *([IMAGE_TOKEN] * 4),
        11,
        VISION_START,
        *([VIDEO_TOKEN] * 4),
        12,
        VISION_START,
        *([VIDEO_TOKEN] * 4),
        13,
    ]
    positions, delta = _positions(
        ids,
        [
            _item(inputs.Modality.IMAGE, [(2, 5)], [[1, 4, 4]]),
            _item(inputs.Modality.VIDEO, [(8, 11), (14, 17)], [[2, 4, 4]]),
        ],
    )
    assert positions[0][:8] == [0, 1, 2, 2, 2, 2, 4, 5]
    assert positions[2][:8] == [0, 1, 2, 3, 2, 3, 4, 5]
    assert positions[0][8:] == [6, 6, 6, 6, 8, 9, 10, 10, 10, 10, 12]
    assert positions[1][8:] == [6, 6, 7, 7, 8, 9, 10, 10, 11, 11, 12]
    assert delta == 13 - 19


def test_a_prompt_without_grids_gets_linear_positions_and_a_zero_delta():
    item = inputs.MultimodalDataItem(
        modality=inputs.Modality.IMAGE, offsets=[(1, 1)], model_specific_data={}
    )
    positions, delta = _positions([5, 6, 7], [item])
    assert positions == [[0, 1, 2]] * 3
    assert delta == 0


def test_the_fast_path_is_abandoned_on_the_shapes_the_rust_port_refuses():
    # Where the engine falls back to rescanning the prompt, the Rust port
    # refuses the item by name instead; both sides agree on which shapes.
    ids = [10, VISION_START, *([IMAGE_TOKEN] * 6), 11]
    fast = mrope._compute_qwen35_mrope_positions_from_offsets
    assert fast(HF_CONFIG, ids, [_item(inputs.Modality.IMAGE, [(2, 7)], [[1, 4, 6]])]) is not None
    assert fast(HF_CONFIG, ids, [_item(inputs.Modality.IMAGE, [], [[1, 4, 6]])]) is None
    assert (
        fast(HF_CONFIG, ids, [_item(inputs.Modality.IMAGE, [(2, 7)], [[1, 4, 6], [1, 4, 6]])])
        is None
    )
    assert fast(HF_CONFIG, ids, [_item(inputs.Modality.IMAGE, [(1, 7)], [[1, 4, 6]])]) is None
    assert fast(HF_CONFIG, ids, [_item(inputs.Modality.IMAGE, [(2, 6)], [[1, 4, 6]])]) is None
    assert fast(HF_CONFIG, ids, [_item(inputs.Modality.IMAGE, [(2, 7)], [[1, 4, 4]])]) is None
