"""The child-process plugin supplies original image anchors to the SDK's M-RoPE fallback."""

import sys
from array import array
from types import SimpleNamespace

import pytest
from smg_grpc_servicer.sglang import mm_broadcast, zmq_plugin

TARGET = "sglang.srt.managers.scheduler.Scheduler._maybe_compute_mrope_positions"


def registered_hook(monkeypatch, *, active=True, available=True):
    registered = []
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.plugins.hook_registry",
        SimpleNamespace(
            HookRegistry=SimpleNamespace(register=lambda *args: registered.append(args)),
            HookType=SimpleNamespace(AROUND=1, AFTER=2),
        ),
    )
    monkeypatch.setattr(zmq_plugin, "attributes_grammar_aborts", lambda: False)
    monkeypatch.setattr(zmq_plugin, "active", lambda: active)
    monkeypatch.setattr(zmq_plugin, "_resolves", lambda target: available)
    monkeypatch.setattr(mm_broadcast, "enabled", lambda: False)
    zmq_plugin.register()
    return next((hook for target, hook, _ in registered if target == TARGET), None)


def invoke(hook, original, scheduler, req, *args, **kwargs):
    return (
        original(scheduler, req, *args, **kwargs)
        if hook is None
        else hook(original, scheduler, req, *args, **kwargs)
    )


@pytest.mark.parametrize("images", [1, 2, 3])
def test_registered_fallback_reads_original_anchors_without_rewriting_cache_tokens(
    monkeypatch, images
):
    hook = registered_hook(monkeypatch)
    original_ids = array("q", [10, 11, 11, 11, 11, 13] * images)
    padded_ids = array("q", [10, 100_000, 100_000, 100_000, 100_000, 13] * images)
    mm = SimpleNamespace(mrope_positions=None)
    req = SimpleNamespace(
        origin_input_ids=padded_ids, origin_input_ids_unpadded=original_ids, multimodal_inputs=mm
    )
    scheduler = SimpleNamespace(_mm_processor=object())
    observed = []

    def sdk_fallback(owner, sdk_req, marker=None):
        assert owner is scheduler and sdk_req.multimodal_inputs is mm
        observed.append(list(sdk_req.origin_input_ids))
        mm.mrope_positions = "SDK-produced positions"
        return marker

    marker = object()
    assert invoke(hook, sdk_fallback, scheduler, req, marker=marker) is marker
    assert observed == [list(original_ids)]
    assert req.origin_input_ids is padded_ids and list(padded_ids).count(100_000) == images * 4
    assert req.origin_input_ids_unpadded is original_ids
    assert mm.mrope_positions == "SDK-produced positions"


@pytest.mark.parametrize(
    "condition", ["precomputed", "legacy", "length_mismatch", "no_media", "no_processor"]
)
def test_existing_or_unsupported_contract_is_delegated_unchanged(monkeypatch, condition):
    hook = registered_hook(monkeypatch)
    req = SimpleNamespace(
        origin_input_ids=array("q", [1, 100_000, 2]),
        origin_input_ids_unpadded=array("q", [1, 11, 2]),
        multimodal_inputs=SimpleNamespace(mrope_positions=None),
    )
    scheduler = SimpleNamespace(_mm_processor=object())
    if condition == "precomputed":
        req.multimodal_inputs.mrope_positions = object()
    elif condition == "legacy":
        del req.origin_input_ids_unpadded
    elif condition == "length_mismatch":
        req.origin_input_ids_unpadded.append(3)
    elif condition == "no_media":
        req.multimodal_inputs = None
    elif condition == "no_processor":
        scheduler._mm_processor = None
    observed = []

    def sdk_fallback(owner, sdk_req):
        observed.append(sdk_req)

    invoke(hook, sdk_fallback, scheduler, req)
    assert observed == [req] and observed[0] is req


def test_sdk_failure_propagates_without_mutating_original_buffer(monkeypatch):
    hook = registered_hook(monkeypatch)
    padded = array("q", [10, 100_000, 13])
    req = SimpleNamespace(
        origin_input_ids=padded,
        origin_input_ids_unpadded=array("q", [10, 11, 13]),
        multimodal_inputs=SimpleNamespace(mrope_positions=None),
    )

    def fails(owner, view):
        raise ValueError("SDK failure")

    with pytest.raises(ValueError, match="SDK failure"):
        invoke(hook, fails, SimpleNamespace(_mm_processor=object()), req)
    assert req.origin_input_ids is padded and list(padded) == [10, 100_000, 13]


@pytest.mark.parametrize("active,available", [(False, True), (True, False)])
def test_no_hook_without_headless_or_supported_sdk(monkeypatch, active, available):
    assert registered_hook(monkeypatch, active=active, available=available) is None


@pytest.mark.parametrize(
    "invalid",
    ["tokens", b"tokens", array("i", [1, 11, 2]), [1, "11", 2], array("q", [1, -1, 2]), []],
)
def test_invalid_saved_buffer_does_not_replace_existing_sdk_input(monkeypatch, invalid):
    hook = registered_hook(monkeypatch)
    req = SimpleNamespace(
        origin_input_ids=array("q", [1, 100_000, 2]),
        origin_input_ids_unpadded=invalid,
        multimodal_inputs=SimpleNamespace(mrope_positions=None),
    )
    observed = []
    invoke(
        hook,
        lambda scheduler, actual: observed.append(actual),
        SimpleNamespace(_mm_processor=object()),
        req,
    )
    assert observed == [req] and observed[0] is req


def test_saved_list_buffer_is_supported_without_changing_caller(monkeypatch):
    hook = registered_hook(monkeypatch)
    saved = [10, 11, 13]
    padded = [10, 100_000, 13]
    req = SimpleNamespace(
        origin_input_ids=padded,
        origin_input_ids_unpadded=saved,
        multimodal_inputs=SimpleNamespace(mrope_positions=None),
    )
    observed = []
    invoke(
        hook,
        lambda scheduler, view: observed.append(view.origin_input_ids),
        SimpleNamespace(_mm_processor=object()),
        req,
    )
    assert observed[0] is saved and req.origin_input_ids is padded
