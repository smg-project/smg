"""Native layout parity checks; requires the supported community SGLang installation."""

from types import SimpleNamespace

import pytest
from smg_grpc_servicer.sglang.managed import plugin


def test_native_epd_metadata_roundtrip_without_media_or_embeddings():
    """Preserve native absence as well as identity, sentinel sequences and positions."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sglang.srt.managers.req_types")
    from sglang.srt.managers.req_types import (
        MultimodalDataItem,
        MultimodalInputs,
        MultimodalProcessorOutput,
    )
    from sglang.srt.multimodal.modality import Modality
    from smg_grpc_servicer.sglang.managed import native

    m = native._proto()
    b = m.InputBinding(prefill_rid="p", layout_contract=native.LAYOUT)
    item = MultimodalDataItem(
        modality=Modality.IMAGE, offsets=[(1, 4)], precomputed_embeddings=torch.zeros((4, 8))
    )
    item.set_pad_value()
    mm = MultimodalInputs.from_processor_output(
        MultimodalProcessorOutput(
            mm_items=[item],
            input_ids=[1, 151655, 151655, 151655, 151655, 2],
            im_token_id=151655,
            mrope_positions=torch.zeros((3, 6), dtype=torch.int64),
            mrope_position_delta=torch.tensor([0]),
        )
    )
    req = SimpleNamespace(
        multimodal_inputs=mm,
        to_finish=None,
        rid="p",
        origin_input_ids_unpadded=[1, 151655, 151655, 151655, 151655, 2],
        origin_input_ids=[1, *item.padding_sequence(4), 2],
    )
    export = native.export_layout(req, b)
    restored = native.decode_input(export, b)
    assert restored.mm_items[0].model_specific_data == {}
    assert restored.mm_items[0].processor_fingerprint is None
    assert restored.mm_items[0].identity == item.identity
    assert restored.mm_items[0].padding_sequence(4) == item.padding_sequence(4)
    assert (
        restored.mm_items[0].feature is None and restored.mm_items[0].precomputed_embeddings is None
    )
    assert torch.equal(restored.mrope_positions, mm.mrope_positions)

    export.padded_ids[0] += 1
    export.integrity_digest = native.digest(export)
    with pytest.raises(ValueError, match="non-image token"):
        native.validate_layout(export, b)


def test_invalid_envelope_uses_native_intake_rejection(monkeypatch):
    """The native handler owns terminal output and cleanup even for malformed metadata."""
    request = SimpleNamespace(rid="bad", sampling_params=SimpleNamespace(custom_params={}))
    seen = []

    def original(scheduler, req, **kwargs):
        seen.append((req, kwargs["mm_input_error"]))

    plugin.prepare_request(original, object(), request)
    assert seen[0][0] is request
    assert "unregistered legacy Generate" in seen[0][1]


def test_layout_refusal_marks_only_the_request_before_native_queue(monkeypatch):
    """All ranks follow the same deterministic rejection through the native retire path."""
    request = SimpleNamespace(rid="bad", to_finish=None)
    request.set_finish_with_abort = lambda reason: setattr(request, "to_finish", reason)
    monkeypatch.setattr(plugin, "_envelope", lambda req: ({}, object()))

    def invalid(req, binding):
        raise ValueError("bad native layout")

    monkeypatch.setattr(plugin.native, "export_layout", invalid)
    queued = []
    plugin.before_queue(lambda scheduler, req: queued.append(req), object(), request)
    assert queued == [request]
    assert "bad native layout" in request.to_finish


def test_full_metadata_channel_leaves_deadline_owner_in_control(monkeypatch):
    """An I/O-rank-only send failure never crashes or locally retires a TP request."""
    zmq = pytest.importorskip("zmq")
    request = SimpleNamespace(rid="blocked", to_finish=None)
    monkeypatch.setattr(plugin, "_envelope", lambda req: ({}, object()))
    monkeypatch.setattr(
        plugin.native,
        "export_layout",
        lambda req, binding: SimpleNamespace(SerializeToString=lambda **kwargs: b"layout"),
    )

    def blocked(*args, **kwargs):
        raise zmq.Again()

    monkeypatch.setattr(plugin, "_sender", SimpleNamespace(send_multipart=blocked))
    queued = []
    plugin.before_queue(lambda scheduler, req: queued.append(req), object(), request)
    assert queued == [request]
    assert request.to_finish is None


@pytest.mark.parametrize(
    "managed,terminal,expected", [(False, False, 1), (True, False, 0), (True, True, 1)]
)
def test_conversion_error_is_not_a_managed_native_receipt(monkeypatch, managed, terminal, expected):
    """Only a real terminal scheduler output can become a managed release receipt."""
    from smg_grpc_servicer.sglang.zmq_msgpack import MsgpackSendSocket

    sender = MsgpackSendSocket()
    sender.managed_native_receipts = managed
    sent = []

    def fail(*args):
        raise ValueError("conversion failed")

    monkeypatch.setattr(sender, "_relay", fail)
    monkeypatch.setattr(sender, "send_terminal_abort", lambda *args, **kwargs: sent.append(args))
    sender.send_output(
        SimpleNamespace(rids=["r"], finished_reasons=[{"type": "stop"} if terminal else None])
    )
    assert len(sent) == expected
    assert list(sender.pending_aborts) == ["r"]


def test_shutdown_stops_followers_even_before_service_construction():
    """A processor construction failure cannot leave this launcher's TP children alive."""
    import asyncio

    from smg_grpc_servicer.sglang.managed.encoder_launch import shutdown

    events = []
    process = SimpleNamespace(
        is_alive=lambda: "terminate" not in events,
        terminate=lambda: events.append("terminate"),
        join=lambda timeout: events.append("join"),
    )

    async def stop():
        events.append("stop")

    asyncio.run(shutdown(None, None, SimpleNamespace(stop=stop, tp_processes=[process]), 1))
    assert events == ["stop", "terminate", "join"]


def test_shutdown_timeout_does_not_unwind_live_native_owners(monkeypatch):
    """Forced process exit follows follower termination without declaring live work quiesced."""
    import asyncio

    from smg_grpc_servicer.sglang.managed import encoder_launch

    class ForcedExit(BaseException):
        pass

    events = []

    async def followers(backend):
        events.append("followers stopped")

    def exit_process(status):
        events.append(("exit", status))
        raise ForcedExit()

    monkeypatch.setattr(encoder_launch, "stop_followers", followers)
    monkeypatch.setattr(encoder_launch.os, "_exit", exit_process)
    entry = SimpleNamespace(terminal=False)
    service = SimpleNamespace(
        entries={"r": entry},
        _cancel=lambda item: events.append("cancel requested"),
        dispatcher=None,
    )
    with pytest.raises(ForcedExit):
        asyncio.run(encoder_launch.shutdown(service, None, object(), 0.001))
    assert events == ["cancel requested", "followers stopped", ("exit", 1)]
    assert not entry.terminal


def test_sigterm_returns_to_owner_cleanup_before_process_exit():
    """A real SIGTERM in a subprocess must leave the managed drain reachable."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, signal; from types import SimpleNamespace; "
            "from smg_grpc_servicer.sglang.managed.encoder_launch import shutdown_signals; "
            "server = SimpleNamespace(handle_exit=lambda *_: print('shutdown requested')); "
            "scope = shutdown_signals(server); scope.__enter__(); "
            "os.kill(os.getpid(), signal.SIGTERM); scope.__exit__(None, None, None); "
            "print('owner cleanup reachable')",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert result.stdout.splitlines() == ["shutdown requested", "owner cleanup reachable"]


def test_shutdown_keeps_reconciliation_open_until_native_work_is_terminal():
    """Draining closes admission, but not the RPC that confirms safe native release."""
    import asyncio

    from smg_grpc_servicer.sglang.managed.encoder_launch import shutdown

    async def scenario():
        entry = SimpleNamespace(terminal=False)
        calls = []
        service = SimpleNamespace(
            draining=False,
            entries={"op": entry},
            dispatcher=None,
            _cancel=lambda _: calls.append("cancel"),
        )

        async def stop_server(grace):
            assert entry.terminal
            assert grace > 0
            calls.append("stop_rpc")

        async def stop_backend():
            calls.append("stop_backend")

        task = asyncio.create_task(
            shutdown(
                service,
                SimpleNamespace(stop=stop_server),
                SimpleNamespace(stop=stop_backend, tp_processes=[]),
                1,
            )
        )
        await asyncio.sleep(0)
        assert service.draining and calls == ["cancel"]
        entry.terminal = True
        await task
        assert calls == ["cancel", "stop_rpc", "stop_backend"]

    asyncio.run(scenario())


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
    assert prepared.mm_items[0] is item
    del received
    gc.collect()
    assert owner() is not None and finalizer.alive and not released
    assert plugin.retain_receiver_owner(lambda scheduler, raw: raw, object(), prepared) is prepared
    del prepared
    gc.collect()
    assert owner() is None and not finalizer.alive and released == ["pool slot"]
