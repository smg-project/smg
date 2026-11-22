"""The child-process plugin supplies original image anchors to the SDK's M-RoPE fallback."""

import json
import sys
from array import array
from types import ModuleType, SimpleNamespace

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


class _DiagnosticTensor:
    dtype = "torch.bfloat16"
    device = SimpleNamespace(type="cuda")

    def __init__(self, storage_offset=0):
        self.layout_reads = []
        self.data_reads = []
        self._storage_offset = storage_offset

    @property
    def shape(self):
        self.layout_reads.append("shape")
        return (280, 1536)

    def stride(self):
        self.layout_reads.append("stride")
        return (1536, 1)

    def storage_offset(self):
        self.layout_reads.append("storage_offset")
        return self._storage_offset

    def __getattr__(self, name):
        self.data_reads.append(name)
        raise AssertionError(f"Unexpected feature operation: {name}")


def _diagnostic_request(monkeypatch):
    monkeypatch.setenv("SMG_QWENVL_REQUEST_DIAGNOSTIC", "1")
    monkeypatch.setattr(zmq_plugin, "_QWENVL_TRACE_SLOTS", iter(range(40)))
    tensors = [_DiagnosticTensor(i * 430080) for i in range(3)]
    items = [
        SimpleNamespace(
            modality=SimpleNamespace(name="IMAGE"),
            model_specific_data={"image_grid_thw": None},
            feature=tensor,
            hash=(2**64 - 1) if i == 0 else 23,
            pad_value=1000023,
            offsets=[(start, start + 69)],
        )
        for i, (tensor, start) in enumerate(zip(tensors, [2, 75, 148]))
    ]
    padded = array("q", [42] * 220)
    unpadded = padded[:]
    for item in items:
        start, end = item.offsets[0]
        padded[start : end + 1] = array("q", [item.pad_value]) * (end - start + 1)
        unpadded[start : end + 1] = array("q", [11]) * (end - start + 1)
    req = SimpleNamespace(
        rid="smg-qwenvl-" + "a" * 32,
        origin_input_ids=padded,
        origin_input_ids_unpadded=unpadded,
        multimodal_inputs=SimpleNamespace(mm_items=items),
    )
    scheduler = SimpleNamespace(server_args=SimpleNamespace(model_path="Qwen/Qwen3-VL-8B-Instruct"))
    return req, scheduler, tensors


@pytest.mark.parametrize("gate", ["disabled", "exhausted"])
def test_qwenvl_diagnostic_gate_precedes_identity_and_layout_reads(monkeypatch, caplog, gate):
    req, scheduler, tensors = _diagnostic_request(monkeypatch)
    if gate == "disabled":
        monkeypatch.delenv("SMG_QWENVL_REQUEST_DIAGNOSTIC")
    else:
        monkeypatch.setattr(zmq_plugin, "_QWENVL_TRACE_SLOTS", iter(()))
    zmq_plugin._trace_qwenvl("probe", scheduler, req)
    assert not any(t.layout_reads or t.data_reads for t in tensors)
    assert not caplog.records


def test_qwenvl_diagnostic_records_existing_identity_layout_and_loaded_settings(
    monkeypatch, caplog
):
    req, scheduler, tensors = _diagnostic_request(monkeypatch)
    reads = []

    def setting(name, value):
        return SimpleNamespace(get=lambda: (reads.append(name), value)[1])

    loaded = ModuleType("sglang.srt.environ")
    loaded.envs = SimpleNamespace(
        SGLANG_MM_BUFFER_SIZE_MB=setting("buffer", 64),
        SGLANG_MM_SKIP_COMPUTE_HASH=setting("skip", False),
    )
    monkeypatch.setitem(sys.modules, loaded.__name__, loaded)
    before = [
        (item.hash, item.pad_value, list(item.offsets), item.feature)
        for item in req.multimodal_inputs.mm_items
    ]
    saved = req.origin_input_ids[:]
    zmq_plugin._trace_qwenvl("probe", scheduler, req)
    record = caplog.records[-1].getMessage().split("SMG_QWENVL_REQUEST_DIAGNOSTIC ", 1)[1]
    facts = json.loads(record)
    assert len(record) <= 8192 and facts["slot"] == 0
    assert facts["sdk_effective_settings"] == {
        "SGLANG_MM_BUFFER_SIZE_MB": {"status": "CAPTURED", "value": 64},
        "SGLANG_MM_SKIP_COMPUTE_HASH": {"status": "CAPTURED", "value": False},
    }
    assert reads == ["buffer", "skip"]
    for i, item in enumerate(facts["mm_items"]):
        assert item["hash"] == before[i][0] and item["pad_value"] == before[i][1]
        assert item["offsets"] == [list(before[i][2][0])]
        assert item["offsets_count"] == 1 and item["offsets_truncated"] is False
        assert item["observed_origin_span_matches_pad"] == [True]
        assert item["feature"]["stride"] == [1536, 1]
        assert item["feature"]["storage_offset"] == i * 430080
    assert req.origin_input_ids == saved
    assert before == [
        (item.hash, item.pad_value, list(item.offsets), item.feature)
        for item in req.multimodal_inputs.mm_items
    ]
    assert not any(t.data_reads for t in tensors)


@pytest.mark.parametrize("kind", ["not_loaded", "wrong_types", "read_error"])
def test_qwenvl_diagnostic_unknown_settings_and_bounded_metadata(monkeypatch, caplog, kind):
    req, scheduler, tensors = _diagnostic_request(monkeypatch)
    monkeypatch.delitem(sys.modules, "sglang.srt.environ", raising=False)
    if kind != "not_loaded":

        def read_error():
            raise RuntimeError("Unavailable SDK setting")

        loaded = ModuleType("sglang.srt.environ")
        loaded.envs = SimpleNamespace(
            SGLANG_MM_BUFFER_SIZE_MB=SimpleNamespace(
                get=read_error if kind == "read_error" else lambda: True
            ),
            SGLANG_MM_SKIP_COMPUTE_HASH=SimpleNamespace(
                get=read_error if kind == "read_error" else lambda: "false"
            ),
        )
        monkeypatch.setitem(sys.modules, loaded.__name__, loaded)
    first = req.multimodal_inputs.mm_items[0]
    first.hash = 2**1024
    first.pad_value = True
    first.offsets = [(i, i + 1) for i in range(6)]
    tensors[0]._storage_offset = 2**1024
    modules_before = {name for name in sys.modules if name.startswith("sglang")}
    zmq_plugin._trace_qwenvl("probe", scheduler, req)
    facts = json.loads(
        caplog.records[-1].getMessage().split("SMG_QWENVL_REQUEST_DIAGNOSTIC ", 1)[1]
    )
    assert all(
        value == {"status": "UNKNOWN", "value": None}
        for value in facts["sdk_effective_settings"].values()
    )
    item = facts["mm_items"][0]
    assert item["hash"] is None and item["pad_value"] is None
    assert item["offsets"] == [[i, i + 1] for i in range(4)]
    assert item["offsets_count"] == 6 and item["offsets_truncated"] is True
    assert item["feature"]["storage_offset"] is None
    assert modules_before == {name for name in sys.modules if name.startswith("sglang")}
    assert not any(t.data_reads for t in tensors)


def test_qwenvl_span_observation_is_event_local_and_scan_bounded(monkeypatch, caplog):
    req, scheduler, tensors = _diagnostic_request(monkeypatch)
    req.origin_input_ids = req.origin_input_ids_unpadded
    zmq_plugin._trace_qwenvl("mrope_before", scheduler, req)
    facts = json.loads(
        caplog.records[-1].getMessage().split("SMG_QWENVL_REQUEST_DIAGNOSTIC ", 1)[1]
    )
    assert all(item["observed_origin_span_matches_pad"] == [False] for item in facts["mm_items"])
    assert "unpadded MRoPE view" in facts["observed_origin_span_scope"]
    first = req.multimodal_inputs.mm_items[0]
    first.pad_value = 42
    first.offsets = [(0, 32767)] * 4
    req.origin_input_ids = [42] * 32768
    saved = list(req.origin_input_ids)
    zmq_plugin._trace_qwenvl("probe", scheduler, req)
    facts = json.loads(
        caplog.records[-1].getMessage().split("SMG_QWENVL_REQUEST_DIAGNOSTIC ", 1)[1]
    )
    assert facts["mm_items"][0]["observed_origin_span_matches_pad"] == [True, None, None, None]
    assert all(item["observed_origin_span_matches_pad"] == [None] for item in facts["mm_items"][1:])
    assert req.origin_input_ids == saved and not any(t.data_reads for t in tensors)


@pytest.mark.parametrize("gate", ["disabled", "exhausted"])
def test_qwenvl_loaded_model_metadata_gate_precedes_source_reads(monkeypatch, caplog, gate):
    monkeypatch.setattr(zmq_plugin, "_QWENVL_MODEL_METADATA_ONCE", iter((True,)))
    monkeypatch.setattr(
        zmq_plugin, "_QWENVL_TRACE_SLOTS", iter(()) if gate == "exhausted" else iter(range(64))
    )
    if gate == "disabled":
        monkeypatch.delenv("SMG_QWENVL_REQUEST_DIAGNOSTIC", raising=False)
    else:
        monkeypatch.setenv("SMG_QWENVL_REQUEST_DIAGNOSTIC", "1")

    def forbidden(*args, **kwargs):
        raise AssertionError("metadata source I/O before diagnostic gate")

    monkeypatch.setattr(zmq_plugin.os, "open", forbidden)
    zmq_plugin._trace_qwenvl_model_once(object(), "smg-qwenvl-" + "a" * 32)
    assert not caplog.records


def test_qwenvl_loaded_model_metadata_is_once_bounded_and_read_only(monkeypatch, caplog, tmp_path):
    monkeypatch.setenv("SMG_QWENVL_REQUEST_DIAGNOSTIC", "1")
    monkeypatch.setattr(zmq_plugin, "_QWENVL_MODEL_METADATA_ONCE", iter((True,)))
    monkeypatch.setattr(zmq_plugin, "_QWENVL_TRACE_SLOTS", iter(range(64)))
    name = "sglang.srt.test_loaded_metadata_fixture"
    module = ModuleType(name)
    source = tmp_path / "loaded_fixture.py"
    content = b"# public SDK source fixture; not executed\n"
    source.write_bytes(content)
    module.__file__ = str(source)
    package = ModuleType("sglang")
    package.__file__ = str(tmp_path / "__init__.py")
    monkeypatch.setitem(sys.modules, "sglang", package)
    monkeypatch.setitem(sys.modules, name, module)

    class Loaded:
        def forward(self, *args, **kwargs):
            raise AssertionError("metadata must not invoke a model")

    Loaded.__module__ = name
    vision, model, backend = Loaded(), Loaded(), Loaded()
    model._modules = {"visual": vision}
    config = SimpleNamespace(
        model_type="qwen3_vl",
        _commit_hash="abc",
        vision_config=SimpleNamespace(
            patch_size=16, temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=4096
        ),
    )
    runner = SimpleNamespace(
        model=model, model_config=SimpleNamespace(hf_config=config), attn_backend=backend
    )
    worker = SimpleNamespace(model_runner=runner)
    scheduler = SimpleNamespace(tp_worker=worker)
    modules_before = set(sys.modules)
    trace = "smg-qwenvl-" + "a" * 32
    zmq_plugin._trace_qwenvl_model_once(scheduler, trace)
    zmq_plugin._trace_qwenvl_model_once(scheduler, trace)
    records = [r.getMessage().split("SMG_QWENVL_REQUEST_DIAGNOSTIC ", 1)[1] for r in caplog.records]
    assert len(records) == 1 and len(records[0]) <= 8192
    facts = json.loads(records[0])["metadata"]
    assert facts["runner_lookup"] == {"path": "tp_worker.model_runner", "status": "found"}
    assert facts["vision_lookup"] == {"path": "model.visual", "status": "found"}
    assert facts["hf_vision_config"]["patch_size"] == 16
    assert facts["hf_vision_config"]["out_hidden_size"] == 4096
    assert facts["hf_vision_config"]["num_heads"] == "unavailable"
    assert facts["resolved_decode_backend"] == {"status": "unavailable"}
    assert facts["source_files"][name]["sha256"] == zmq_plugin.hashlib.sha256(content).hexdigest()
    assert facts["source_bytes_read"] == len(content)
    assert (
        scheduler.tp_worker is worker
        and runner.model is model
        and model._modules["visual"] is vision
    )
    assert set(sys.modules) == modules_before and source.read_bytes() == content


def test_qwenvl_loaded_model_metadata_does_not_invoke_properties(monkeypatch, caplog):
    monkeypatch.setenv("SMG_QWENVL_REQUEST_DIAGNOSTIC", "1")
    monkeypatch.setattr(zmq_plugin, "_QWENVL_MODEL_METADATA_ONCE", iter((True,)))
    monkeypatch.setattr(zmq_plugin, "_QWENVL_TRACE_SLOTS", iter(range(64)))

    class Scheduler:
        @property
        def tp_worker(self):
            raise AssertionError("do not invoke SDK property fallback")

    zmq_plugin._trace_qwenvl_model_once(Scheduler(), "smg-qwenvl-" + "a" * 32)
    facts = json.loads(
        caplog.records[-1].getMessage().split("SMG_QWENVL_REQUEST_DIAGNOSTIC ", 1)[1]
    )["metadata"]
    assert facts["tp_worker"] == {"status": "unavailable"}
    assert facts["model_runner"] == {"status": "unavailable"}
    assert facts["source_bytes_read"] == 0


def test_qwenvl_loaded_model_metadata_rejects_source_outside_loaded_package(
    monkeypatch, caplog, tmp_path
):
    monkeypatch.setenv("SMG_QWENVL_REQUEST_DIAGNOSTIC", "1")
    monkeypatch.setattr(zmq_plugin, "_QWENVL_MODEL_METADATA_ONCE", iter((True,)))
    monkeypatch.setattr(zmq_plugin, "_QWENVL_TRACE_SLOTS", iter(range(64)))
    name = "sglang.srt.test_loaded_source_privacy"
    package = ModuleType("sglang")
    package.__file__ = str(tmp_path / "sdk" / "__init__.py")
    module = ModuleType(name)
    module.__file__ = str(tmp_path / "credential.py")
    monkeypatch.setitem(sys.modules, "sglang", package)
    monkeypatch.setitem(sys.modules, name, module)

    class Loaded:
        pass

    Loaded.__module__ = name
    scheduler = SimpleNamespace(
        tp_worker=SimpleNamespace(model_runner=SimpleNamespace(model=Loaded()))
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("must not read outside public loaded SDK package")

    monkeypatch.setattr(zmq_plugin.os, "open", forbidden)
    zmq_plugin._trace_qwenvl_model_once(scheduler, "smg-qwenvl-" + "a" * 32)
    raw = caplog.records[-1].getMessage().split("SMG_QWENVL_REQUEST_DIAGNOSTIC ", 1)[1]
    source = json.loads(raw)["metadata"]["source_files"][name]
    assert source == {"status": "outside_loaded_package"}
    assert "credential.py" not in raw and "path" not in source
