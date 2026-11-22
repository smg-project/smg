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
    [("qwen3_vl", True), ("glm5_next", True), ("other_vision", True), ("qwen3", False)],
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
    ids=["qwen3-vl", "glm53-flash", "without-mrope"],
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
    plugin.before_queue(lambda s, r: queued.append(r), SimpleNamespace(model_config=object()), req2)
    assert queued == [req2] and "metadata publication failed" in req2.to_finish
    assert not getattr(req2, "_smg_managed_exported", False)


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
