"""The engine's MRoPE position rules, pinned on the shapes the Rust servicer's
ports are tested on.

The Rust TokenSpeed servicer computes the Qwen3.5 lineage's and Qwen3-Omni's
MRoPE position tables itself (the msgpack wire bypasses the engine's input
processor, which computes them on the Python path). Its unit tests pin these
exact tables; this test pins the same tables on the engine's own
``compute_mrope_positions``, so an engine release that changes a rule turns
red here, where the engine is installed, and names the port as the thing to
update.
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


OMNI_CONFIG = SimpleNamespace(
    architectures=["Qwen3OmniMoeForConditionalGeneration"],
    model_type="qwen3_omni_moe",
    thinker_config=SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        text_config=SimpleNamespace(hidden_size=2),
        position_id_per_seconds=13,
        seconds_per_chunk=2,
    ),
)


def _omni_item(modality, offsets, grid=None, seconds=None):
    data = {}
    if grid is not None:
        data[f"{modality.name.lower()}_grid_thw"] = torch.tensor(grid, dtype=torch.long)
    if seconds is not None:
        data["video_second_per_grid"] = torch.tensor(seconds, dtype=torch.float32)
    return inputs.MultimodalDataItem(modality=modality, offsets=offsets, model_specific_data=data)


def _omni_positions(input_len, items):
    positions, delta = mrope.compute_mrope_positions(OMNI_CONFIG, list(range(input_len)), items)
    return positions.tolist(), int(delta.flatten()[0].item())


def test_omni_mixed_modalities_follow_the_authored_offsets():
    # Mirrors the Rust test of the same name: an audio run, an image and a
    # one-second-per-grid video over 25 tokens, the items out of order.
    positions, delta = _omni_positions(
        25,
        [
            _omni_item(inputs.Modality.VIDEO, [(15, 22)], [[2, 4, 4]], [1.0]),
            _omni_item(inputs.Modality.AUDIO, [(2, 4)]),
            _omni_item(inputs.Modality.IMAGE, [(8, 11)], [[1, 4, 4]]),
        ],
    )
    for axis in positions:
        assert axis[:8] == list(range(8))
        assert axis[12:15] == [10, 11, 12]
        assert axis[23:] == [27, 28]
    assert positions[0][8:12] == [8, 8, 8, 8]
    assert positions[1][8:12] == [8, 8, 9, 9]
    assert positions[2][8:12] == [8, 9, 8, 9]
    assert positions[0][15:23] == [13, 13, 13, 13, 26, 26, 26, 26]
    assert positions[1][15:23] == [13, 13, 14, 14, 13, 13, 14, 14]
    assert positions[2][15:23] == [13, 14, 13, 14, 13, 14, 13, 14]
    assert delta == 4


def test_omni_videos_tick_their_own_seconds_per_grid():
    positions, delta = _omni_positions(
        22,
        [
            _omni_item(inputs.Modality.VIDEO, [(12, 19)], [[2, 4, 4]], [1.5]),
            _omni_item(inputs.Modality.VIDEO, [(2, 9)], [[2, 4, 4]], [0.5]),
        ],
    )
    assert positions[0][2:10] == [2, 2, 2, 2, 8, 8, 8, 8]
    assert positions[0][12:20] == [11, 11, 11, 11, 30, 30, 30, 30]
    assert delta == 11
    # A video without its seconds ticks the checkpoint's seconds per chunk.
    positions, delta = _omni_positions(
        10, [_omni_item(inputs.Modality.VIDEO, [(1, 8)], [[2, 4, 4]])]
    )
    assert positions[0][1:9] == [1, 1, 1, 1, 27, 27, 27, 27]
    assert delta == 28 + 1 - 10


def test_the_omni_rule_refuses_the_shapes_the_rust_port_refuses():
    video = _omni_item(inputs.Modality.VIDEO, [(1, 4)], [[1, 4, 4]])
    video.model_specific_data["use_audio_in_video"] = torch.tensor([True])
    with pytest.raises(ValueError, match="use_audio_in_video"):
        mrope.compute_mrope_positions(OMNI_CONFIG, list(range(6)), [video])
    with pytest.raises(ValueError, match="grid requires 4"):
        mrope.compute_mrope_positions(
            OMNI_CONFIG, list(range(5)), [_omni_item(inputs.Modality.IMAGE, [(1, 3)], [[1, 4, 4]])]
        )
    with pytest.raises(ValueError, match="missing image_grid_thw"):
        mrope.compute_mrope_positions(
            OMNI_CONFIG, list(range(6)), [_omni_item(inputs.Modality.IMAGE, [(1, 4)])]
        )
    with pytest.raises(ValueError, match="overlap"):
        mrope.compute_mrope_positions(
            OMNI_CONFIG,
            list(range(9)),
            [
                _omni_item(inputs.Modality.IMAGE, [(1, 4)], [[1, 4, 4]]),
                _omni_item(inputs.Modality.IMAGE, [(4, 7)], [[1, 4, 4]]),
            ],
        )
