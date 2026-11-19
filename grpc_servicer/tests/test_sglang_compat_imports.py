"""Engine module moves preserve old installs and dependency failures."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
from smg_grpc_servicer.sglang import compat


@pytest.mark.parametrize("modern", [False, True])
def test_disaggregation_module_move(monkeypatch, modern):
    """Resolve the mode and fake bootstrap host on both engine module layouts."""
    prefix = "sglang.srt.disaggregation."
    old = ModuleType(prefix + "utils")
    old.DisaggregationMode = object()
    old.FAKE_BOOTSTRAP_HOST = "old"
    new = ModuleType(prefix + "mode")
    new.DisaggregationMode = object()
    new.FAKE_BOOTSTRAP_HOST = "new"
    monkeypatch.setitem(sys.modules, old.__name__, old)
    monkeypatch.setitem(sys.modules, new.__name__, new if modern else None)
    mode, host = compat.get_disaggregation_types()
    expected = new if modern else old
    assert mode is expected.DisaggregationMode
    assert host == expected.FAKE_BOOTSTRAP_HOST


@pytest.mark.parametrize("modern", [False, True])
def test_multimodal_module_move(monkeypatch, modern):
    """Resolve all media types from the installed engine owner module."""
    prefix = "sglang.srt.managers."
    old = ModuleType(prefix + "schedule_batch")
    new = ModuleType(prefix + "req_types")
    for module in [old, new]:
        for name in ["Modality", "MultimodalDataItem", "MultimodalProcessorOutput"]:
            setattr(module, name, object())
    monkeypatch.setitem(sys.modules, old.__name__, old)
    monkeypatch.setitem(sys.modules, new.__name__, new if modern else None)
    result = compat.get_multimodal_types()
    expected = new if modern else old
    for actual, name in zip(
        result, ["Modality", "MultimodalDataItem", "MultimodalProcessorOutput"]
    ):
        assert actual is getattr(expected, name)


@pytest.mark.parametrize("modern", [False, True])
def test_tokenizer_factory_module_move(monkeypatch, modern):
    """Forward tokenizer arguments unchanged through either import path."""
    calls = []
    result = object()

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    def load(name):
        calls.append(name)
        if name == "sglang.srt.hf_transformers.tokenizer" and not modern:
            raise ModuleNotFoundError(name=name)
        return SimpleNamespace(get_tokenizer=factory)

    monkeypatch.setattr(compat.importlib, "import_module", load)
    assert compat.get_tokenizer("model", trust_remote_code=True) is result
    assert calls[-1] == (("model",), {"trust_remote_code": True})
    assert calls[:-1] == ["sglang.srt.hf_transformers.tokenizer"] + (
        [] if modern else ["sglang.srt.utils.hf_transformers_utils"]
    )


@pytest.mark.parametrize(
    "helper", [compat.get_tokenizer, compat.get_disaggregation_types, compat.get_multimodal_types]
)
def test_compat_import_dependency_failure_is_not_hidden(monkeypatch, helper):
    """Only a missing owner module permits fallback; missing dependencies propagate."""
    failure = ModuleNotFoundError("dependency missing", name="missing_dependency")

    def load(_name):
        raise failure

    monkeypatch.setattr(compat.importlib, "import_module", load)
    with pytest.raises(ModuleNotFoundError) as raised:
        helper("model") if helper is compat.get_tokenizer else helper()
    assert raised.value is failure


@pytest.mark.parametrize("kind", ["none", "empty", "single", "video"])
def test_media_normalization_leaves_nonbundled_images_and_video_untouched(monkeypatch, kind):
    item = SimpleNamespace(
        is_image=lambda: kind != "video",
        offsets=[(1, 2)] if kind == "single" else [(1, 2), (4, 5)],
        hash=123,
        pad_value=456,
    )
    output = (
        None
        if kind == "none"
        else SimpleNamespace(mm_items=[] if kind == "empty" else [item], padded_input_ids=None)
    )

    def unexpected_import(_name):
        pytest.fail("unchanged media must not import or call the SDK splitter")

    monkeypatch.setattr(compat.importlib, "import_module", unexpected_import)
    assert compat.normalize_multimodal_processor_output(output) is output
    assert (item.hash, item.pad_value) == (123, 456)


@pytest.mark.parametrize("missing", ["owner", "function", "dependency"])
def test_media_normalization_only_falls_back_for_exact_legacy_absence(monkeypatch, missing):
    output = SimpleNamespace(
        mm_items=[SimpleNamespace(is_image=lambda: True, offsets=[(1, 2), (4, 5)])],
        padded_input_ids=None,
    )
    name = "sglang.srt.managers.mm_utils"
    failure = ModuleNotFoundError(name=name if missing == "owner" else "dependency")

    def load(module_name):
        assert module_name == name
        if missing == "function":
            return ModuleType(name)
        raise failure

    monkeypatch.setattr(compat.importlib, "import_module", load)
    if missing == "dependency":
        with pytest.raises(ModuleNotFoundError) as raised:
            compat.normalize_multimodal_processor_output(output)
        assert raised.value is failure
    else:
        assert compat.normalize_multimodal_processor_output(output) is output


def test_media_normalization_does_not_hide_a_real_splitter_failure(monkeypatch):
    failure = ValueError("invalid processor grid")

    def split(_items):
        raise failure

    module = ModuleType("sglang.srt.managers.mm_utils")
    module.get_new_expanded_mm_items = split
    monkeypatch.setattr(compat.importlib, "import_module", lambda _: module)
    output = SimpleNamespace(
        mm_items=[SimpleNamespace(is_image=lambda: True, offsets=[(1, 2), (4, 5)], pad_value=None)],
        padded_input_ids=None,
    )
    with pytest.raises(ValueError) as raised:
        compat.normalize_multimodal_processor_output(output)
    assert raised.value is failure
