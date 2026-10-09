"""Unit tests for the `mm_item_limits` label helpers (engine-free)."""

from __future__ import annotations

from types import SimpleNamespace

from smg_grpc_servicer import mm_item_limits


def test_format_keeps_positive_integral_counts_in_modality_order():
    assert mm_item_limits.format_item_limits({"video": 2, "image": 8}) == "image=8,video=2"
    assert mm_item_limits.format_item_limits({"image": 4.0, "audio": 1}) == "audio=1,image=4"
    # Zero, negative, fractional, boolean and non-numeric counts, and modalities
    # the Router does not count, are left out; nothing to say is an empty label.
    assert (
        mm_item_limits.format_item_limits(
            {"image": 0, "video": -1, "audio": 1.5, "pdf": 3, "point_cloud": True, "x": "2"}
        )
        == ""
    )
    assert mm_item_limits.format_item_limits({}) == ""


def test_sglang_limits_come_from_limit_mm_data_per_request():
    key = mm_item_limits.MM_ITEM_LIMITS_KEY
    assert key == "mm_item_limits"
    # Unset (SGLang's default): no limit, no label.
    assert mm_item_limits.sglang_item_limits(SimpleNamespace()) == ""
    assert mm_item_limits.sglang_item_limits(SimpleNamespace(limit_mm_data_per_request=None)) == ""
    # The parsed object, or the JSON text an unparsed argument still carries.
    parsed = SimpleNamespace(limit_mm_data_per_request={"image": 1, "video": 1, "audio": 1})
    assert mm_item_limits.sglang_item_limits(parsed) == "audio=1,image=1,video=1"
    text = SimpleNamespace(limit_mm_data_per_request='{"image": 2}')
    assert mm_item_limits.sglang_item_limits(text) == "image=2"
    # Text that is not a JSON object advertises nothing rather than failing discovery.
    assert mm_item_limits.sglang_item_limits(SimpleNamespace(limit_mm_data_per_request="2")) == ""
    assert mm_item_limits.sglang_item_limits(SimpleNamespace(limit_mm_data_per_request="{")) == ""
