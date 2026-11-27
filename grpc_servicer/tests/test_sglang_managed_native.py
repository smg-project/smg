"""Native layout parity and managed plugin checks."""

import asyncio
import os
import signal
import sys
from types import SimpleNamespace

import pytest
from smg_grpc_servicer.sglang import epd_plugin as plugin


@pytest.mark.parametrize(
    ("model_type", "has_vision"),
    [("qwen3_vl", True), ("vision_vlm", True), ("other_vision", True), ("qwen3", False)],
)
def test_verify_contract_checks_vision_capability(monkeypatch, model_type, has_vision):
    native = plugin.native
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.req_types",
        SimpleNamespace(
            MultimodalDataItem=SimpleNamespace(__struct_fields__=native._ITEM_FIELDS),
            MultimodalProcessorOutput=SimpleNamespace(__struct_fields__=native._OUTPUT_FIELDS),
        ),
    )

    class Scheduler:
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type=model_type, vision_config=SimpleNamespace() if has_vision else None
            )
        )

        def _add_request_to_queue(self, req):
            pass

        def handle_generate_request(self, recv_req, *, mm_input_error=None):
            pass

        _get_multimodal_inputs = _maybe_compute_mrope_positions = _add_request_to_queue

    if has_vision:
        native.verify_contract(Scheduler())

        class MissingMmInputError(Scheduler):
            def handle_generate_request(self, recv_req):
                pass

        with pytest.raises(ValueError, match="mm_input_error"):
            native.verify_contract(MissingMmInputError())
    else:
        with pytest.raises(ValueError, match="requires a vision model"):
            native.verify_contract(Scheduler())


@pytest.mark.parametrize(
    ("image_token_id", "image_start_id", "image_end_id", "require_mrope"),
    [(151655, None, None, True), (154854, 154830, 154831, True), (200025, None, None, False)],
    ids=["qwen3-vl", "mrope-with-delimiters", "without-mrope"],
)
def test_native_epd_metadata_roundtrip_without_media_or_embeddings(
    image_token_id, image_start_id, image_end_id, require_mrope
):
    """Preserve native absence as well as identity, sentinel sequences and positions."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sglang.srt.managers.req_types")
    from sglang.srt.managers.req_types import (
        MultimodalDataItem,
        MultimodalInputs,
        MultimodalProcessorOutput,
    )
    from sglang.srt.multimodal.modality import Modality
    from smg_grpc_servicer.sglang import prepared_input as native

    m = native._proto()
    b = m.InputBinding(
        prefill_rid="p",
        decode_rid="d",
        layout_contract=native.LAYOUT,
        model_revision="weights",
        processor_contract="smg-image-grid-v1",
        bootstrap=m.Bootstrap(host="p", port=8998, room=1),
    )
    prefix = [1] if image_start_id is None else [1, image_start_id]
    suffix = [2] if image_end_id is None else [image_end_id, 2]
    unpadded_ids = [*prefix, *([image_token_id] * 4), *suffix]
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=[(len(prefix), len(prefix) + 3)],
        precomputed_embeddings=torch.zeros((4, 8)),
    )
    item.set_pad_value()
    mm = MultimodalInputs.from_processor_output(
        MultimodalProcessorOutput(
            mm_items=[item],
            input_ids=unpadded_ids,
            im_token_id=image_token_id,
            im_start_id=image_start_id,
            im_end_id=image_end_id,
            mrope_positions=torch.zeros((3, len(unpadded_ids)), dtype=torch.int64)
            if require_mrope
            else None,
            mrope_position_delta=torch.tensor([0]) if require_mrope else None,
        )
    )
    req = SimpleNamespace(
        multimodal_inputs=mm,
        to_finish=None,
        rid="p",
        origin_input_ids_unpadded=unpadded_ids,
        origin_input_ids=[*prefix, *item.padding_sequence(4), *suffix],
    )
    export = native.export_layout(req, b, require_mrope=require_mrope)
    restored = native.decode_input(export, require_mrope=require_mrope)
    assert restored.im_token_id == image_token_id
    assert restored.im_start_id == image_start_id
    assert restored.im_end_id == image_end_id
    assert restored.mm_items[0].model_specific_data == {}
    assert restored.mm_items[0].identity == item.identity
    assert restored.mm_items[0].padding_sequence(4) == item.padding_sequence(4)
    assert (
        restored.mm_items[0].feature is None and restored.mm_items[0].precomputed_embeddings is None
    )
    if require_mrope:
        assert torch.equal(restored.mrope_positions, mm.mrope_positions)
    else:
        assert restored.mrope_positions is None and restored.mrope_position_delta is None
        with pytest.raises(ValueError, match="requires native M-RoPE"):
            native.decode_input(export)

    export.images[0].routing_label = "media-1"
    if native.supports_routing_labels():
        assert (
            native.decode_input(export, require_mrope=require_mrope).mm_items[0].routing_label
            == "media-1"
        )
    else:
        with pytest.raises(ValueError, match="does not support media routing labels"):
            native.decode_input(export, require_mrope=require_mrope)


def test_plugin_intake_queue_and_backpressure_errors(monkeypatch):
    zmq = pytest.importorskip("zmq")

    request = SimpleNamespace(rid="bad", sampling_params=SimpleNamespace(custom_params={}))
    seen = []
    plugin.prepare_request(
        lambda s, r, **kw: seen.append((r, kw["mm_input_error"])), object(), request
    )
    assert seen[0][0] is request and "unregistered legacy Generate" in seen[0][1]

    req = SimpleNamespace(rid="bad", to_finish=None)
    req.set_finish_with_abort = lambda reason: setattr(req, "to_finish", reason)
    monkeypatch.setattr(plugin, "_envelope", lambda r: ({}, object()))
    monkeypatch.setattr(
        plugin.native, "export_layout", lambda *a, **kw: (_ for _ in ()).throw(ValueError("bad"))
    )
    queued = []
    plugin.before_queue(lambda s, r: queued.append(r), SimpleNamespace(model_config=object()), req)
    assert queued == [req] and "bad" in req.to_finish

    req2 = SimpleNamespace(rid="blocked", to_finish=None)
    req2.set_finish_with_abort = lambda reason: setattr(req2, "to_finish", reason)
    monkeypatch.setattr(
        plugin.native,
        "export_layout",
        lambda *a, **kw: SimpleNamespace(SerializeToString=lambda **kw: b"layout", images=[]),
    )
    monkeypatch.setattr(
        plugin,
        "_sender",
        SimpleNamespace(send_multipart=lambda *a, **kw: (_ for _ in ()).throw(zmq.Again())),
    )
    queued.clear()
    plugin.before_queue(
        lambda s, r: queued.append(r),
        SimpleNamespace(model_config=object(), server_args=SimpleNamespace(tp_size=1)),
        req2,
    )
    assert queued == [req2] and "metadata publication failed" in req2.to_finish
    assert not getattr(req2, "_smg_managed_exported", False)

    # At TP > 1 (or when TP size cannot be determined), rank 0 must NOT unilaterally abort
    # on metadata publish failure so PrefillBootstrapQueue stays symmetric across ranks.
    req3 = SimpleNamespace(rid="blocked-tp4", to_finish=None)
    req3.set_finish_with_abort = lambda reason: setattr(req3, "to_finish", reason)
    queued.clear()
    plugin.before_queue(
        lambda s, r: queued.append(r),
        SimpleNamespace(model_config=object(), server_args=SimpleNamespace(tp_size=4)),
        req3,
    )
    assert queued == [req3] and req3.to_finish is None
    assert not getattr(req3, "_smg_managed_exported", False)


@pytest.mark.parametrize(
    "managed,terminal,expected", [(False, False, 1), (True, False, 0), (True, True, 1)]
)
def test_conversion_error_is_not_a_managed_native_receipt(monkeypatch, managed, terminal, expected):
    from smg_grpc_servicer.sglang.zmq_msgpack import MsgpackSendSocket

    sender = MsgpackSendSocket()
    sender.managed_native_receipts = managed
    sent = []
    monkeypatch.setattr(sender, "_relay", lambda *a: (_ for _ in ()).throw(ValueError("bad")))
    monkeypatch.setattr(sender, "send_terminal_abort", lambda *a, **kw: sent.append(a))
    sender.send_output(
        SimpleNamespace(rids=["r"], finished_reasons=[{"type": "stop"} if terminal else None])
    )
    assert len(sent) == expected and list(sender.pending_aborts) == ["r"]


def test_normalized_prefill_input_retains_native_receiver_pool_owner():
    """Normalization must not free a pool slot while the request still holds its tensor views."""
    import gc
    import weakref

    torch = pytest.importorskip("torch")
    pytest.importorskip("sglang.srt.managers.req_types")
    from sglang.srt.managers.req_types import (
        MultimodalDataItem,
        MultimodalInputs,
        MultimodalProcessorOutput,
    )
    from sglang.srt.multimodal.modality import Modality

    released = []
    item = MultimodalDataItem(
        modality=Modality.IMAGE, offsets=[(1, 4)], precomputed_embeddings=torch.zeros((4, 8))
    )
    item.set_pad_value()
    received = MultimodalProcessorOutput(
        mm_items=[item], input_ids=[1, 151655, 151655, 151655, 151655, 2], im_token_id=151655
    )
    owner = weakref.ref(received)
    finalizer = weakref.finalize(received, released.append, "pool slot")
    prepared = plugin.retain_receiver_owner(
        lambda scheduler, raw: MultimodalInputs.from_processor_output(raw), object(), received
    )
    del received
    gc.collect()
    assert owner() is not None and finalizer.alive and not released
    del prepared
    gc.collect()
    assert owner() is None and not finalizer.alive and released == ["pool slot"]


@pytest.mark.parametrize(
    ("attrs", "expected"),
    [
        ({"mm_inputs": None, "need_wait_for_mm_inputs": False}, True),
        ({"mm_inputs": object(), "need_wait_for_mm_inputs": False}, False),
        ({"mm_inputs": None, "need_wait_for_mm_inputs": True}, False),
        ({"mm_inputs": None, "input_embeds": [1.0]}, False),
        ({"sampling_params": SimpleNamespace(custom_params={plugin.ENVELOPE: {}})}, False),
    ],
)
def test_ordinary_text_predicate(attrs, expected):
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(custom_params={}),
        mm_inputs=None,
        need_wait_for_mm_inputs=False,
        input_embeds=None,
    )
    for k, v in attrs.items():
        setattr(req, k, v)
    assert plugin._ordinary_text(req) is expected


@pytest.mark.asyncio
async def test_encoder_shutdown_drains_and_exits_on_stuck_active(monkeypatch):
    from smg_grpc_servicer.sglang import epd_encoder, epd_launcher, model_artifacts

    events = []
    active_count = [1]

    class FakeServer:
        def __init__(self, *args):
            self.drained = False

        def drain(self):
            self.drained = True
            events.append("drain")

        def active(self):
            return active_count[0]

        def running(self):
            return True

        def last_error(self):
            return None

        def stop(self, timeout):
            events.append("stop")

    class FakeProc:
        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    backend = SimpleNamespace(
        encoder=SimpleNamespace(
            preprocessor=SimpleNamespace(max_artifact_bytes=1024),
            model_config=SimpleNamespace(
                hidden_size=64, hf_config=SimpleNamespace(model_type="qwen3_vl", image_token_id=1)
            ),
            model=SimpleNamespace(visual=SimpleNamespace(dtype="bfloat16")),
        ),
        scheduler=SimpleNamespace(max_batch_size=4),
        tp_processes=[FakeProc()],
        start=lambda: events.append("backend.start"),
        stop=lambda: asyncio.sleep(0),
    )
    monkeypatch.setitem(sys.modules, "smg.smg_rs", SimpleNamespace(ManagedEncoderServer=FakeServer))
    monkeypatch.setattr(model_artifacts, "encoder_artifact_dirs", lambda args: ("/m", "/t"))
    monkeypatch.setattr(epd_encoder, "launch_runtime", lambda args, max_active: backend)

    async def fake_serve_native(cfg, b):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        await asyncio.Future()

    monkeypatch.setattr(epd_encoder, "serve_native", fake_serve_native)
    monkeypatch.setattr(os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

    args = SimpleNamespace(model_path="m", served_model_name=None, host="127.0.0.1", port=30000)
    config = dict(
        max_active=4,
        max_control_bytes=4096,
        max_artifact_bytes=4096,
        max_receivers=2,
        embedding_dtype="bfloat16",
        embedding_backend="mooncake",
        processor_options={"model_id": "m"},
        shutdown_timeout_seconds=0.05,
    )
    with pytest.raises(SystemExit) as exc:
        await epd_launcher.serve(config, args)
    assert exc.value.code == 1
    assert "drain" in events
    assert "stop" not in events


def test_prepared_input_video_metadata_roundtrip_and_limits_without_torch(monkeypatch):
    from smg_grpc_servicer.sglang import prepared_input as native

    assert native.MAX_MANIFEST_BYTES == 32 * 1024 * 1024
    assert native.MAX_TOKENS == 1_048_576
    assert native.MAX_IMAGES == 1024

    class FakeTensor:
        def __init__(self, values, shape=None, dtype="int64"):
            self._values = list(values) if isinstance(values, (list, tuple)) else [values]
            self.shape = tuple(shape) if shape is not None else (len(self._values),)
            self.dtype = dtype

        @property
        def ndim(self):
            return len(self.shape)

        def numel(self):
            return len(self._values)

        def detach(self):
            return self

        def cpu(self):
            return self

        def reshape(self, *shape):
            if len(shape) == 1 and isinstance(shape[0], tuple):
                shape = shape[0]
            return FakeTensor(self._values, shape, self.dtype)

        def tolist(self):
            if self.shape == (1, 3):
                return [self._values]
            return list(self._values)

    fake_torch = SimpleNamespace(
        int32="int32",
        int64="int64",
        float32="float32",
        Tensor=FakeTensor,
        tensor=lambda vals, dtype=None: FakeTensor(vals, dtype=dtype),
    )

    class ProtoMsg:
        def __init__(self, **kw):
            self._fields = set(kw)
            for k, v in kw.items():
                setattr(self, k, v)

        def HasField(self, name):
            val = getattr(self, name, None)
            return val is not None

        def CopyFrom(self, other):
            for k, v in other.__dict__.items():
                if not k.startswith("_"):
                    setattr(self, k, v)
                    self._fields.add(k)

    class RepeatedList(list):
        def __init__(self, factory):
            super().__init__()
            self._factory = factory

        def add(self, **kw):
            obj = self._factory(**kw)
            self.append(obj)
            return obj

    class PreparedDecodeInputStub(ProtoMsg):
        def __init__(self, **kw):
            kw.setdefault("images", RepeatedList(ProtoMsg))
            kw.setdefault("mrope_positions", ProtoMsg())
            kw.setdefault("mrope_delta", ProtoMsg())
            kw.setdefault("token_types", ProtoMsg())
            super().__init__(**kw)

    class ModalityEnum:
        def __init__(self, name, value):
            self.name = name
            self.value = value

    ModalityStub = SimpleNamespace(IMAGE=ModalityEnum("IMAGE", 1), VIDEO=ModalityEnum("VIDEO", 2))
    ModalityStub.AUDIO = ModalityEnum("AUDIO", 3)

    class InputFormatStub:
        def __init__(self, val):
            self.value = val

    class DataItemStub:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class ProcessorOutputStub:
        __struct_fields__ = {
            "mm_items",
            "input_ids",
            "padded_input_ids",
            "im_token_id",
            "im_start_id",
            "im_end_id",
            "video_token_id",
            "mrope_positions",
            "mrope_position_delta",
            "token_type_ids",
        }

        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    pb_stub = SimpleNamespace(
        PreparedDecodeInput=PreparedDecodeInputStub,
        IntegerArray=ProtoMsg,
        Offset=ProtoMsg,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(native, "_proto", lambda: pb_stub)
    monkeypatch.setattr(
        native,
        "supports_routing_labels",
        lambda: True,
    )
    for mod_name, mod_obj in [
        ("sglang", SimpleNamespace()),
        ("sglang.srt", SimpleNamespace()),
        ("sglang.srt.managers", SimpleNamespace()),
        (
            "sglang.srt.managers.req_types",
            SimpleNamespace(
                MultimodalDataItem=DataItemStub, MultimodalProcessorOutput=ProcessorOutputStub
            ),
        ),
        ("sglang.srt.multimodal", SimpleNamespace()),
        (
            "sglang.srt.multimodal.modality",
            SimpleNamespace(Modality=ModalityStub, MultimodalInputFormat=InputFormatStub),
        ),
    ]:
        monkeypatch.setitem(sys.modules, mod_name, mod_obj)

    binding = ProtoMsg(prefill_rid="p")
    video_item = SimpleNamespace(
        modality=ModalityStub.VIDEO,
        identity=b"\x01" * 32,
        pad_values=[1_000_002],
        processor_fingerprint=None,
        offsets=[(1, 2)],
        format=InputFormatStub(1),
        hash=99,
        model_specific_data={
            "video_grid_thw": FakeTensor([1, 2, 2], (1, 3)),
            "second_per_grid_ts": FakeTensor([0.5], (1,), dtype="float32"),
        },
    )
    req = SimpleNamespace(
        rid="p",
        to_finish=None,
        origin_input_ids_unpadded=[1, 154854, 154854, 2],
        origin_input_ids=[1, 1_000_002, 1_000_002, 2],
        multimodal_inputs=SimpleNamespace(
            mm_items=[video_item],
            im_token_id=154854,
            im_start_id=154830,
            im_end_id=154831,
            video_token_id=154855,
            mrope_positions=FakeTensor([0] * 12, (3, 4)),
            mrope_position_delta=FakeTensor([0], (1,)),
            token_type_ids=None,
        ),
    )
    manifest = native.export_layout(req, binding)
    assert manifest.video_token_id == 154855
    assert manifest.images[0].modality == 2
    assert manifest.images[0].grid_thw == [1, 2, 2]
    assert manifest.images[0].second_per_grid_ts == [0.5]
    manifest.mrope_positions._has = True
    manifest.HasField = lambda name: name == "mrope_positions"
    manifest.images[0].HasField = lambda name: False
    restored = native.decode_input(manifest)
    assert restored.video_token_id == 154855
    assert restored.mm_items[0].modality is ModalityStub.VIDEO
    assert restored.mm_items[0].model_specific_data["video_grid_thw"].tolist() == [[1, 2, 2]]
    assert restored.mm_items[0].model_specific_data["second_per_grid_ts"].tolist() == [0.5]

    audio_item = SimpleNamespace(modality=ModalityStub.AUDIO, model_specific_data={})
    req.multimodal_inputs.mm_items = [audio_item]
    with pytest.raises(ValueError, match="image and video"):
        native.export_layout(req, binding)


def test_split_work_and_control_reqs_preserves_dp_targeted_aborts(monkeypatch):
    from smg_grpc_servicer.sglang import zmq_msgpack, zmq_plugin

    class AbortReq:
        def __init__(self, rid):
            self.rid = rid

    class EncoderDispatchErrorReq:
        def __init__(self, rid):
            self.rid = rid

    class FlushCacheReqInput:
        pass

    io_mod = SimpleNamespace(
        AbortReq=AbortReq,
        EncoderDispatchErrorReq=EncoderDispatchErrorReq,
        FlushCacheReqInput=FlushCacheReqInput,
    )
    managers_mod = SimpleNamespace(io_struct=io_mod)
    srt_mod = SimpleNamespace(managers=managers_mod)
    sglang_mod = SimpleNamespace(srt=srt_mod)
    for name, mod in (
        ("sglang", sglang_mod),
        ("sglang.srt", srt_mod),
        ("sglang.srt.managers", managers_mod),
        ("sglang.srt.managers.io_struct", io_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    abort_req = AbortReq("r-dp1")
    err_req = EncoderDispatchErrorReq("r-dp2")
    flush_req = FlushCacheReqInput()
    work_item = SimpleNamespace(rid="r-work")

    def fake_original(receiver, recv_reqs):
        return [work_item], [abort_req, err_req, flush_req]

    msgpack_receiver = SimpleNamespace(
        recv_from_tokenizer=object.__new__(zmq_msgpack.MsgpackRecvSocket)
    )
    work, control = zmq_plugin.split_work_and_control_reqs(
        fake_original, msgpack_receiver, [work_item, abort_req, err_req, flush_req]
    )
    assert work == [work_item, abort_req, err_req]
    assert control == [flush_req]

    plain_receiver = SimpleNamespace(recv_from_tokenizer=object())
    work2, control2 = zmq_plugin.split_work_and_control_reqs(
        fake_original, plain_receiver, [work_item, abort_req, err_req, flush_req]
    )
    assert work2 == [work_item]
    assert control2 == [abort_req, err_req, flush_req]


def test_prepared_input_v0521_schedule_batch_schema_compat(monkeypatch):
    """Verify v0.5.21 schedule_batch module path and MultimodalDataItem schema without identity/pad_values."""
    import enum

    torch = pytest.importorskip("torch")
    from smg_grpc_servicer.sglang import prepared_input as native

    class Modality(enum.Enum):
        IMAGE = 1
        VIDEO = 2

    class MultimodalInputFormat(enum.Enum):
        EMBEDDING = 1

    v0521_item_fields = (
        "modality",
        "hash",
        "pad_value",
        "offsets",
        "format",
        "feature",
        "precomputed_embeddings",
        "keep_device_embedding",
        "model_specific_data",
    )

    class V0521Item:
        __struct_fields__ = v0521_item_fields

        def __init__(self, **kwargs):
            unexpected = set(kwargs) - set(self.__struct_fields__)
            assert not unexpected, f"unexpected v0.5.21 fields: {unexpected}"
            for k, v in kwargs.items():
                setattr(self, k, v)

    class V0521Output:
        __struct_fields__ = tuple(native._OUTPUT_FIELDS)

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    sb_mod = SimpleNamespace(
        MultimodalDataItem=V0521Item,
        MultimodalProcessorOutput=V0521Output,
        Modality=Modality,
        MultimodalInputFormat=MultimodalInputFormat,
    )
    # Hide req_types & multimodal.modality so compat fallback resolves schedule_batch.
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.req_types", None)
    monkeypatch.setitem(sys.modules, "sglang.srt.multimodal.modality", None)
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.schedule_batch", sb_mod)

    class Scheduler:
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_vl", vision_config=SimpleNamespace())
        )

        def _add_request_to_queue(self, req):
            pass

        def handle_generate_request(self, recv_req, *, mm_input_error=None):
            pass

        _get_multimodal_inputs = _maybe_compute_mrope_positions = _add_request_to_queue

    native.verify_contract(Scheduler())
    assert native.supports_routing_labels() is False

    pb = native._proto()
    binding = pb.InputBinding(
        prefill_rid="p-521",
        decode_rid="d-521",
        layout_contract=native.LAYOUT,
        model_revision="rev1",
        processor_contract="smg-image-grid-v1",
        bootstrap=pb.Bootstrap(host="127.0.0.1", port=8998, room=9),
    )
    item = V0521Item(
        modality=Modality.IMAGE,
        hash=123456789,
        pad_value=987654,
        offsets=[(1, 2)],
        format=MultimodalInputFormat.EMBEDDING,
        feature=None,
        precomputed_embeddings=None,
        keep_device_embedding=False,
        model_specific_data={"image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int64)},
    )
    req = SimpleNamespace(
        rid="p-521",
        to_finish=None,
        origin_input_ids_unpadded=[1, 151655, 151655, 2],
        origin_input_ids=[1, 987654, 987654, 2],
        multimodal_inputs=SimpleNamespace(
            mm_items=[item],
            im_token_id=151655,
            im_start_id=None,
            im_end_id=None,
            mrope_positions=torch.zeros((3, 4), dtype=torch.int64),
            mrope_position_delta=torch.tensor([0], dtype=torch.int64),
            token_type_ids=None,
        ),
    )
    manifest = native.export_layout(req, binding)
    assert len(manifest.images[0].full_identity) == 32
    assert list(manifest.images[0].pad_values) == [987654]
    assert manifest.images[0].native_hash == 123456789
    restored = native.decode_input(manifest)
    assert len(restored.mm_items) == 1
    assert restored.mm_items[0].hash == 123456789
    assert restored.mm_items[0].pad_value == 987654
    assert not hasattr(restored.mm_items[0], "identity")
    assert not hasattr(restored.mm_items[0], "pad_values")


def test_prepare_request_preserves_array_input_ids(monkeypatch):
    """Preserve array.array input_ids on sglang v0.5.21 while keeping list compatibility."""
    from array import array

    proto = plugin.native._proto()
    manifest = proto.PreparedDecodeInput(version=1, unpadded_ids=[10, 151655, 151655, 20])
    binding = SimpleNamespace(bootstrap=SimpleNamespace(room=77))
    envelope = {"role": "decode", "manifest": manifest.SerializeToString().hex()}

    monkeypatch.setattr(plugin, "_envelope", lambda _req: (envelope, binding))
    monkeypatch.setattr(plugin.native, "decode_input", lambda _m, require_mrope=True: "bound-mm")
    scheduler = SimpleNamespace(model_config=SimpleNamespace(model_is_mrope=True))

    # sglang v0.5.21 passes array('i', ...) for TokenizedGenerateReqInput.input_ids
    req_arr = SimpleNamespace(
        rid="d-array",
        mm_inputs=None,
        need_wait_for_mm_inputs=False,
        input_ids=array("i", [1, 2]),
        input_text="ignored",
        sampling_params=SimpleNamespace(custom_params={plugin.ENVELOPE: envelope}),
    )
    seen = []
    plugin.prepare_request(lambda _s, r, **kw: seen.append((r, kw)), scheduler, req_arr)
    assert len(seen) == 1 and "mm_input_error" not in seen[0][1]
    assert isinstance(req_arr.input_ids, array)
    assert req_arr.input_ids.typecode == "i"
    assert list(req_arr.input_ids) == [10, 151655, 151655, 20]
    assert list(req_arr.input_ids + array("i", [99])) == [10, 151655, 151655, 20, 99]
    assert req_arr.mm_inputs == "bound-mm" and req_arr.input_text is None

    # sglang v0.5.23+ passes list[int]
    req_list = SimpleNamespace(
        rid="d-list",
        mm_inputs=None,
        need_wait_for_mm_inputs=False,
        input_ids=[1, 2],
        input_text="ignored",
        sampling_params=SimpleNamespace(custom_params={plugin.ENVELOPE: envelope}),
    )
    seen.clear()
    plugin.prepare_request(lambda _s, r, **kw: seen.append((r, kw)), scheduler, req_list)
    assert len(seen) == 1 and "mm_input_error" not in seen[0][1]
    assert isinstance(req_list.input_ids, list)
    assert req_list.input_ids == [10, 151655, 151655, 20]


def test_prefill_receiver_owner_deterministic_release_and_chunked_safety():
    """Release EmbeddingPool slots immediately on done/abort without waiting for cyclic GC."""
    import weakref

    class _DummyOwner:
        pass

    def _make_req(label):
        received = _DummyOwner()
        released = []
        finalizer = weakref.finalize(received, released.append, label)
        prepared = plugin.retain_receiver_owner(
            lambda _sched, _raw: SimpleNamespace(mm_items=[]), object(), received
        )
        req = SimpleNamespace(rid=label, multimodal_inputs=prepared)
        # Create a reference cycle on req (matching SGLang Req <-> batch/output graphs)
        # so the slot would leak until Gen-2 GC without deterministic owner clearing.
        req._cycle = req
        del received
        return req, released, finalizer

    # 1. Chunked prefill safety + normal completion: an in-flight chunked req retains its slot
    # while another completed req is released immediately without gc.collect().
    chunked_req, chunked_released, chunked_fin = _make_req("chunked")
    done_req, done_released, done_fin = _make_req("done")
    assert plugin.release_inflight_owners([done_req], object()) is None
    assert done_released == ["done"] and not done_fin.alive
    assert not chunked_released and chunked_fin.alive

    # Once the chunked req finishes all chunks and KV transfer, its slot frees immediately.
    assert plugin.release_inflight_owners([chunked_req], object()) is None
    assert chunked_released == ["chunked"] and not chunked_fin.alive

    # 2. release_features hook frees the slot immediately even while Req stays alive.
    rf_req, rf_released, rf_fin = _make_req("release_features")
    assert plugin.release_mm_features_owner(None, rf_req.multimodal_inputs) is None
    assert rf_released == ["release_features"] and not rf_fin.alive

    # 3. _retire_aborted_prefill_result only releases when retirement actually occurred (True).
    abort_req, abort_released, abort_fin = _make_req("aborted")
    assert plugin.release_aborted_prefill_owner(False, object(), abort_req) is None
    assert not abort_released and abort_fin.alive
    assert plugin.release_aborted_prefill_owner(True, object(), abort_req) is None
    assert abort_released == ["aborted"] and not abort_fin.alive

    # 4. handle_bootstrap_failure frees the slot immediately.
    boot_req, boot_released, boot_fin = _make_req("bootstrap_fail")
    assert plugin.release_bootstrap_failure_owner(None, object(), boot_req) is None
    assert boot_released == ["bootstrap_fail"] and not boot_fin.alive


def test_managed_launcher_launch_wires_bootstrap_and_scheduler_metrics(monkeypatch):
    """ManagedLauncher.launch delegates to SglangLauncher.launch so bootstrap + metrics both run."""
    import smg_grpc_servicer.sglang.epd_launcher as epd_launcher
    import smg_grpc_servicer.sglang.rust as sglang_rust

    calls = []
    metrics_sentinel = object()
    monkeypatch.setattr(
        sglang_rust,
        "launch_scheduler_ranks",
        lambda args, *, handshake_port, encode_bridge=None: (
            calls.append(("ranks", handshake_port, encode_bridge)) or "engine"
        ),
    )
    monkeypatch.setattr(
        sglang_rust,
        "serve_scheduler_metrics",
        lambda args: calls.append(("metrics", args)) or metrics_sentinel,
    )
    args = SimpleNamespace(host="127.0.0.1", encoder_bootstrap_port=9100)

    prefill_launcher = epd_launcher.launcher({"role": "prefill"})
    assert not hasattr(prefill_launcher, "launch_ranks")
    assert prefill_launcher.launch(args, 5555, {}) == "engine"
    assert prefill_launcher.scheduler_metrics is metrics_sentinel
    assert calls[0][0] == "ranks" and calls[0][1] == 5555
    assert isinstance(calls[0][2], epd_launcher.ManagedBootstrap)
    assert calls[1] == ("metrics", args)

    calls.clear()
    decode_launcher = epd_launcher.launcher({"role": "decode"})
    assert decode_launcher.launch(args, 5556, {}) == "engine"
    assert decode_launcher.scheduler_metrics is metrics_sentinel
    assert calls[0] == ("ranks", 5556, None)
    assert calls[1] == ("metrics", args)


def test_encoder_embedding_width_accounts_for_deepstack():
    """DeepStack multi-layer vision outputs widen per-token staging bytes beyond hidden_size."""
    import smg_grpc_servicer.sglang.epd_launcher as epd_launcher

    encoder = SimpleNamespace(model_config=SimpleNamespace(hidden_size=4096))
    plain_cfg = SimpleNamespace(vision_config=SimpleNamespace())
    deepstack_cfg = SimpleNamespace(
        vision_config=SimpleNamespace(deepstack_visual_indexes=[8, 16, 24])
    )
    assert epd_launcher._encoder_embedding_width(encoder, plain_cfg) == 4096
    assert epd_launcher._encoder_embedding_width(encoder, deepstack_cfg) == 16384
