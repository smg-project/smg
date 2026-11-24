"""Diagnostic guard/byte/identity contracts, not an SDK or model qualification."""

import gc
import hashlib
import json
import logging
import weakref

import pytest
from smg_grpc_servicer.sglang import qwenvl_numerical_diagnostic as diagnostic


class Item:
    pass


@pytest.fixture
def state(monkeypatch):
    monkeypatch.setenv("SMG_QWENVL_REQUEST_DIAGNOSTIC", "1")
    monkeypatch.setattr(diagnostic, "_installed", True)
    monkeypatch.setattr(diagnostic, "_traces", {})
    monkeypatch.setattr(diagnostic, "_items", {})
    monkeypatch.setattr(diagnostic, "_process_bytes", 0)
    monkeypatch.setattr(diagnostic, "_events", 0)
    monkeypatch.setattr(diagnostic, "_unknown_boundaries", set())
    rid = "smg-qwenvl-" + "a" * 32
    diagnostic._traces[rid] = dict(bytes=0, active=True, visual_calls=0, events=0)
    return rid


def test_disabled_observer_does_not_touch_request_or_frames(monkeypatch):
    class Forbidden:
        def __getattribute__(self, name):
            raise AssertionError("disabled diagnostic inspected an object")

    monkeypatch.delenv("SMG_QWENVL_REQUEST_DIAGNOSTIC", raising=False)
    diagnostic.observe_wire(Forbidden(), Forbidden())
    diagnostic.bind_scheduler(Forbidden())
    diagnostic.install(Forbidden())


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32", "int32"])
def test_full_hash_covers_logical_bytes_and_preserves_tensor(state, dtype):
    torch = pytest.importorskip("torch")
    value = torch.arange(24).to(getattr(torch, dtype)).reshape(4, 6).transpose(0, 1)
    before = value.clone()
    expected = hashlib.sha256(
        value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    row = diagnostic._tensor(state, value)
    assert row["status"] == "CAPTURED_FULL_LOGICAL_BYTES"
    assert (
        row["full_byte_sha256"] == expected and row["bytes"] == value.numel() * value.element_size()
    )
    assert row["stride"] == list(value.stride()) and torch.equal(value, before)


def test_inference_tensor_hash_and_nonfinite_counts_are_explicit(state):
    torch = pytest.importorskip("torch")
    with torch.inference_mode():
        value = torch.tensor([1.0, float("nan"), float("inf")])
    row = diagnostic._tensor(state, value)
    assert row["full_byte_sha256"] is not None
    assert row["version_guard"] == "UNKNOWN_NO_VERSION_COUNTER"
    assert row["finite_count"] == 1 and row["nonfinite_count"] == 2
    assert "mean" not in row


@pytest.mark.parametrize("kind", ["tensor", "trace", "process"])
def test_over_budget_never_becomes_sample_hash(state, monkeypatch, kind):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        diagnostic,
        {"tensor": "_TENSOR_BYTES", "trace": "_TRACE_BYTES", "process": "_PROCESS_BYTES"}[kind],
        1,
    )
    row = diagnostic._tensor(state, torch.arange(8))
    assert row["status"] == "UNKNOWN_" + kind.upper() + "_BYTE_CAP"
    assert row["full_byte_sha256"] is None


def test_wire_auxiliary_hashes_never_emit_payload_or_feature_values(state, caplog):
    torch = pytest.importorskip("torch")
    del diagnostic._traces[state]
    items = [Item() for _ in range(3)]
    for item in items:
        item.feature = torch.arange(4, dtype=torch.bfloat16)
        item.modality = Item()
        item.modality.name = "IMAGE"
    request, mm = Item(), Item()
    request.rid, request.mm_inputs, mm.mm_items = state, mm, items
    frames = [b"\x00", b"PRIVATE_REQUEST_PAYLOAD", b"auxiliary", b"PRIVATE_METADATA"]
    with caplog.at_level(logging.WARNING):
        diagnostic.observe_wire(request, frames)
    text = "\n".join(record.message for record in caplog.records)
    row = json.loads(text.split("SMG_QWENVL_REQUEST_DIAGNOSTIC ", 1)[1])
    assert (
        row["auxiliary_frames"][0]["full_byte_sha256"] == hashlib.sha256(b"auxiliary").hexdigest()
    )
    assert "PRIVATE" not in text and 'auxiliary"' not in text
    assert frames[1] == b"PRIVATE_REQUEST_PAYLOAD" and request.mm_inputs.mm_items is items


@pytest.mark.parametrize("case", ["copied", "mixed", "duplicate"])
def test_uncertain_identity_never_gets_a_trace(state, case):
    items = [Item() for _ in range(3)]
    for index, item in enumerate(items):
        diagnostic._items[id(item)] = (weakref.ref(item), {state: index})
    if case == "copied":
        candidate = [Item(), *items[1:]]
    elif case == "mixed":
        diagnostic._items[id(items[1])][1][state + "-child"] = 1
        candidate = items
    else:
        candidate = [items[0], items[0], items[2]]
    assert diagnostic._matched(candidate) is None
    assert diagnostic._traces[state]["visual_calls"] == 0


def test_identity_uses_no_strong_item_reference_and_finish_releases_it(state):
    items = [Item() for _ in range(3)]
    for index, item in enumerate(items):
        diagnostic._items[id(item)] = (weakref.ref(item), {state: index})
    del item
    reference = weakref.ref(items[0])
    assert diagnostic._matched(items) == state
    assert diagnostic._matched(items) is None  # Re-prefill is not a new comparable observation.
    del items
    gc.collect()
    assert reference() is None
    diagnostic.finish(state)
    assert diagnostic._items == {} and diagnostic._traces[state]["active"] is False
