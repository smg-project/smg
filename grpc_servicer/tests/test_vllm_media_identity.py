"""Unit tests for the prefill leg's media identity (engine-free, needs torch).

Run with: pytest grpc_servicer/tests/test_vllm_media_identity.py
"""

import importlib.util
import struct
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("smg_grpc_proto")
from smg_grpc_proto.generated import common_pb2  # noqa: E402
from smg_grpc_servicer.vllm.mm_tensors import tensor_from_proto  # noqa: E402

# Import the module directly to avoid pulling vllm via the package __init__
_MODULE_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "media_identity.py"
_spec = importlib.util.spec_from_file_location("media_identity", _MODULE_PATH)
media_identity = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = media_identity
_spec.loader.exec_module(media_identity)


def _range(offset, length, is_embed=None):
    return types.SimpleNamespace(offset=offset, length=length, is_embed=is_embed)


def _elem(data, keep_on_cpu=False):
    return types.SimpleNamespace(data=data, field=types.SimpleNamespace(keep_on_cpu=keep_on_cpu))


def _image_prompt():
    """Two images with M-RoPE grids, as a processed engine input."""
    grids = [torch.tensor([1, 4, 6]), torch.tensor([1, 2, 2])]
    return {
        "prompt_token_ids": [7, 8, 100, 100, 100, 9, 100, 5],
        "mm_hashes": {"image": ["h1", "h2"]},
        "mm_placeholders": {
            "image": [
                # <start> token, three patches, <end> token: embeds land on 100.
                _range(1, 5, torch.tensor([False, True, True, True, False])),
                _range(6, 1),
            ]
        },
        "mm_kwargs": {
            "image": [
                {
                    "pixel_values": _elem(torch.zeros(3, 8, 8)),
                    "image_grid_thw": _elem(grids[0], True),
                },
                {
                    "pixel_values": _elem(torch.zeros(3, 8, 8)),
                    "image_grid_thw": _elem(grids[1], True),
                },
            ]
        },
    }


class TestBuildMediaIdentity:
    def test_carries_ids_hashes_placeholders_and_grids_but_no_pixels(self):
        identity = media_identity.build_media_identity(_image_prompt())
        assert list(identity.prompt_token_ids) == [7, 8, 100, 100, 100, 9, 100, 5]
        mm = identity.mm_inputs
        assert mm.modality == common_pb2.IMAGE
        assert list(mm.mm_hashes) == ["h1", "h2"]
        assert [(p.offset, p.length) for p in mm.mm_placeholders] == [(1, 5), (6, 1)]
        assert mm.im_token_id == 100, "read off the first embed position of the masked range"
        assert not mm.HasField("pixel_values")
        assert set(mm.model_specific_tensors) == {"image_grid_thw"}
        assert list(mm.batched_keys) == ["image_grid_thw"]
        assert list(mm.keep_on_cpu_keys) == ["image_grid_thw"]
        grid = tensor_from_proto(mm.model_specific_tensors["image_grid_thw"])
        assert grid.dtype == torch.int64
        assert grid.tolist() == [[1, 4, 6], [1, 2, 2]]
        assert list(identity.extra_mm_inputs) == []

    # Qwen-VL processors spell the video timing `second_per_grid_ts`; the
    # Qwen-Omni family and the router's own processors `video_second_per_grid`.
    @pytest.mark.parametrize("timing_key", ["second_per_grid_ts", "video_second_per_grid"])
    def test_a_second_modality_goes_to_extra_mm_inputs(self, timing_key):
        prompt = _image_prompt()
        prompt["mm_hashes"]["video"] = ["v1"]
        prompt["mm_placeholders"]["video"] = [_range(7, 1)]
        prompt["mm_kwargs"]["video"] = [
            {
                "video_grid_thw": _elem(torch.tensor([2, 2, 2])),
                timing_key: _elem(torch.tensor(0.5)),
            }
        ]
        identity = media_identity.build_media_identity(prompt)
        assert identity.mm_inputs.modality == common_pb2.IMAGE
        (video,) = identity.extra_mm_inputs
        assert video.modality == common_pb2.VIDEO
        assert list(video.mm_hashes) == ["v1"]
        assert set(video.model_specific_tensors) == {"video_grid_thw", timing_key}
        seconds = tensor_from_proto(video.model_specific_tensors[timing_key])
        assert seconds.dtype == torch.float32 and seconds.tolist() == [0.5]
        assert not video.HasField("im_token_id")

    def test_no_media_means_no_identity(self):
        assert media_identity.build_media_identity({"prompt_token_ids": [1, 2]}) is None
        assert (
            media_identity.build_media_identity({"prompt_token_ids": [1], "mm_hashes": {}}) is None
        )

    def test_a_grid_that_is_not_one_row_per_item_yields_no_identity(self, caplog):
        prompt = _image_prompt()
        prompt["mm_kwargs"]["image"][1]["image_grid_thw"] = _elem(torch.tensor([[1, 2, 2]]))
        with caplog.at_level("WARNING", logger="media_identity"):
            assert media_identity.build_media_identity(prompt) is None
        assert any("image_grid_thw" in r.getMessage() for r in caplog.records)

    def test_older_field_names_are_read_too(self):
        prompt = _image_prompt()
        prompt["multi_modal_hashes"] = prompt.pop("mm_hashes")
        prompt["multi_modal_placeholders"] = prompt.pop("mm_placeholders")
        prompt["multi_modal_kwargs"] = prompt.pop("mm_kwargs")
        assert media_identity.build_media_identity(prompt).mm_inputs.mm_hashes == ["h1", "h2"]


class TestSupport:
    def test_the_installed_proto_carries_the_identity(self):
        # The proto package built from this tree (as CI does) has the field.
        assert media_identity.media_identity_supported()


class TestTensorToProto:
    def test_round_trips_and_widens_unnamed_dtypes(self):
        for tensor in (
            torch.tensor([[1, 2, 3]], dtype=torch.int32),
            torch.tensor([1.5, -2.0], dtype=torch.float64),
            torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
        ):
            proto = media_identity.tensor_to_proto(tensor)
            back = tensor_from_proto(proto)
            assert back.tolist() == tensor.tolist()
            assert list(proto.shape) == list(tensor.shape)
        assert media_identity.tensor_to_proto(torch.tensor([1], dtype=torch.int32)).dtype == "int64"
        assert (
            media_identity.tensor_to_proto(torch.tensor([1.0], dtype=torch.float64)).dtype
            == "float32"
        )

    def test_matches_the_routers_wire_form(self):
        # The router writes little-endian element bytes; a decode leg reads
        # either sender the same way.
        proto = media_identity.tensor_to_proto(torch.tensor([1, 4, 6], dtype=torch.int64))
        assert proto.inline == struct.pack("<3q", 1, 4, 6)
        assert proto.dtype == "int64" and list(proto.shape) == [3]
