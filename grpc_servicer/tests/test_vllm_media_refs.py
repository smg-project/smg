"""Unit tests for media_refs parsing (engine-free, no vLLM required).

Run with: pytest grpc_servicer/tests/test_vllm_media_refs.py
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("smg_grpc_proto")
from smg_grpc_proto import vllm_engine_pb2  # noqa: E402
from smg_grpc_proto.generated import common_pb2  # noqa: E402

# Import the module directly to avoid pulling vllm via the package __init__
_MODULE_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "media_refs.py"
_spec = importlib.util.spec_from_file_location("media_refs", _MODULE_PATH)
media_refs = importlib.util.module_from_spec(_spec)
# Registered so dataclass decorators can resolve the module by name.
sys.modules[_spec.name] = media_refs
_spec.loader.exec_module(media_refs)


def _refs(*pairs):
    return vllm_engine_pb2.MediaRefs(
        items=[vllm_engine_pb2.MediaRef(modality=m, url=u) for m, u in pairs]
    )


class TestParseMediaRefs:
    def test_preserves_prompt_order_and_maps_modalities(self):
        items = media_refs.parse_media_refs(
            _refs(
                (common_pb2.IMAGE, "https://a/1.png"),
                (common_pb2.VIDEO, "https://a/clip.mp4"),
                (common_pb2.IMAGE, "data:image/png;base64,AAAA"),
            )
        )
        assert [(i.modality, i.url) for i in items] == [
            ("image", "https://a/1.png"),
            ("video", "https://a/clip.mp4"),
            ("image", "data:image/png;base64,AAAA"),
        ]

    def test_rejects_audio(self):
        with pytest.raises(ValueError, match="media_refs\\[0\\].*AUDIO"):
            media_refs.parse_media_refs(_refs((common_pb2.AUDIO, "https://a/x.wav")))

    def test_rejects_unspecified_modality(self):
        with pytest.raises(ValueError, match="MODALITY_UNSPECIFIED"):
            media_refs.parse_media_refs(_refs((common_pb2.MODALITY_UNSPECIFIED, "https://a/1")))

    def test_rejects_empty_url(self):
        with pytest.raises(ValueError, match="media_refs\\[1\\]: empty url"):
            media_refs.parse_media_refs(
                _refs((common_pb2.IMAGE, "https://a/1.png"), (common_pb2.IMAGE, ""))
            )

    def test_rejects_empty_list(self):
        with pytest.raises(ValueError, match="no items"):
            media_refs.parse_media_refs(vllm_engine_pb2.MediaRefs())


class TestSchemes:
    @pytest.mark.parametrize(
        ("url", "scheme"),
        [
            ("https://host/a.png", "https"),
            ("HTTP://host/a.png", "http"),
            ("data:image/png;base64,AAAA", "data"),
            ("file:///models/a.png", "file"),
            ("/models/a.png", ""),
        ],
    )
    def test_url_scheme(self, url, scheme):
        assert media_refs.url_scheme(url) == scheme

    def test_advertised_schemes_without_local_path(self):
        assert media_refs.advertised_schemes("") == "http,https,data"
        assert media_refs.advertised_schemes(None) == "http,https,data"

    def test_advertised_schemes_with_local_path(self):
        assert media_refs.advertised_schemes("/media") == "http,https,data,file"

    def test_parse_scheme_list_roundtrips(self):
        assert media_refs.parse_scheme_list("http,https,data,file") == {
            "http",
            "https",
            "data",
            "file",
        }

    def test_validate_schemes_accepts_listed(self):
        items = [
            media_refs.MediaRefItem("image", "https://a/1.png"),
            media_refs.MediaRefItem("image", "data:image/png;base64,AAAA"),
        ]
        media_refs.validate_schemes(items, {"http", "https", "data"})

    def test_validate_schemes_rejects_file_without_local_path(self):
        items = [media_refs.MediaRefItem("image", "file:///models/a.png")]
        with pytest.raises(ValueError, match="allowed-local-media-path"):
            media_refs.validate_schemes(items, {"http", "https", "data"})


def test_group_urls_by_modality_keeps_order():
    items = [
        media_refs.MediaRefItem("image", "u1"),
        media_refs.MediaRefItem("video", "v1"),
        media_refs.MediaRefItem("image", "u2"),
    ]
    assert media_refs.group_urls_by_modality(items) == {"image": ["u1", "u2"], "video": ["v1"]}
