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
    b = m.InputBinding(
        prefill_rid="p",
        layout_contract=native.LAYOUT,
        model_revision="weights",
        processor_contract="qwen3",
        prefill=m.Instance(worker_id="p", generation="p-generation"),
        bootstrap=m.Bootstrap(host="p", port=8998, room=1),
        decode=m.OperationRef(
            identity=m.Identity(logical_request_id="r", attempt_id="a"),
            target=m.Instance(worker_id="d", generation="d-generation"),
            operation_id="d",
            request_digest=b"r" * 32,
            plan_digest=b"p" * 32,
        ),
    )
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


def _managed_encoder_lifecycle(
    monkeypatch,
    *,
    active=0,
    processor_fails=False,
    backend_start_fails=False,
    missing_bridge=False,
    timeout=1,
):
    """Exercise the real launcher with CPU-only native/extension boundaries."""
    import asyncio
    import sys

    from smg_grpc_servicer.sglang.managed import config, encoder_bridge, encoder_native

    state = SimpleNamespace(active=active, events=[], started=asyncio.Event(), alive=True)

    def record(event):
        state.events.append(event)
        print(event, flush=True)

    def processor(options, tokenizer_dir, _limit):
        assert options["model_dir"] == "model"
        assert tokenizer_dir == "tokenizer"
        if processor_fails:
            raise RuntimeError("injected processor construction failure")
        return object()

    def stop_server(_timeout):
        assert state.active == 0
        record("rpc stopped")

    server = SimpleNamespace(
        drain=lambda: record("draining"),
        active=lambda: state.active,
        running=lambda: True,
        last_error=lambda: None,
        stop=stop_server,
    )

    async def native(*_args):
        state.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            record("native observer stopped")

    async def stop_backend():
        record("backend stopped")

    def terminate():
        state.alive = False
        record("follower terminated")

    process = SimpleNamespace(
        is_alive=lambda: state.alive,
        terminate=terminate,
        join=lambda _timeout: record("follower joined"),
        kill=lambda: record("follower killed"),
    )

    def start_backend():
        record("backend started")
        if backend_start_fails:
            raise RuntimeError("injected native start failure")

    backend = SimpleNamespace(
        scheduler=SimpleNamespace(max_batch_size=1),
        encoder=SimpleNamespace(
            preprocessor=None if missing_bridge else SimpleNamespace(max_artifact_bytes=128),
            model_config=SimpleNamespace(
                hidden_size=4, hf_config=SimpleNamespace(model_type="qwen3_vl")
            ),
            model=SimpleNamespace(visual=SimpleNamespace(dtype="torch.bfloat16")),
        ),
        start=start_backend,
        stop=stop_backend,
        tp_processes=[process],
    )
    monkeypatch.setitem(
        sys.modules,
        "smg.smg_rs",
        SimpleNamespace(
            ManagedEncoderProcessor=processor,
            ManagedEncoderServer=lambda *_args: server,
        ),
    )
    from smg_grpc_servicer.sglang import model_artifacts

    monkeypatch.setattr(
        model_artifacts, "encoder_artifact_dirs", lambda _args: ("model", "tokenizer")
    )
    monkeypatch.setattr(encoder_bridge, "launch_runtime", lambda *_args: backend)
    monkeypatch.setattr(encoder_native, "serve_native", native)
    monkeypatch.setattr(config, "encoder_cpu_limit", lambda *_args: 1)
    monkeypatch.setattr(config, "encoder_staging_bytes", lambda *_args: 128)
    monkeypatch.setattr(config, "encoder_window", lambda *_args: (1, 1))
    settings = {
        "max_active": 1,
        "embedding_dtype": "bfloat16",
        "processor_options": {},
        "max_artifact_bytes": 128,
        "shutdown_timeout_seconds": timeout,
    }
    args = SimpleNamespace(
        model_path="model", tokenizer_path=None, served_model_name=None, host="127.0.0.1", port=1234
    )
    return state, settings, args


@pytest.mark.parametrize(
    "fault,message,events",
    [
        ("processor_fails", "processor construction", []),
        ("missing_bridge", "did not install", []),
        (
            "backend_start_fails",
            "native start failure",
            ["backend started", "draining", "rpc stopped"],
        ),
    ],
)
@pytest.mark.asyncio
async def test_startup_failure_releases_native_owners(monkeypatch, fault, message, events):
    from smg_grpc_servicer.sglang.managed.launch import serve

    state, config, args = _managed_encoder_lifecycle(monkeypatch, **{fault: True})
    with pytest.raises(RuntimeError, match=message):
        await serve(config, args)
    assert state.events == events + ["backend stopped", "follower terminated", "follower joined"]


def _encoder_lifecycle_subprocess(force_timeout):
    import subprocess
    import sys
    from pathlib import Path

    source = f"""
import asyncio, importlib.util, os, signal, threading
import pytest
spec = importlib.util.spec_from_file_location("lifecycle", {str(Path(__file__).resolve())!r})
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
from smg_grpc_servicer.sglang.managed import launch
async def main():
    with pytest.MonkeyPatch.context() as patch:
        state, config, args = module._managed_encoder_lifecycle(patch, active={int(force_timeout)}, timeout=0.01)
        previous = signal.getsignal(signal.SIGTERM)
        # Keep the selector idle until a real signal wakes it.
        patch.setattr(launch, "wait_encoder_shutdown", lambda server, stopped: stopped.wait())
        task = asyncio.create_task(launch.serve(config, args))
        await state.started.wait()
        threading.Timer(0.05, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
        await task
        assert signal.getsignal(signal.SIGTERM) == previous
        print("owner cleanup complete", flush=True)
asyncio.run(main())
"""
    return subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, timeout=10
    )


def test_shutdown_timeout_does_not_unwind_live_native_owners():
    result = _encoder_lifecycle_subprocess(True)
    assert result.returncode == 1, result.stderr
    assert "follower terminated" in result.stdout and "follower joined" in result.stdout
    for forbidden in [
        "rpc stopped",
        "native observer stopped",
        "backend stopped",
        "owner cleanup complete",
    ]:
        assert forbidden not in result.stdout


def test_sigterm_returns_to_owner_cleanup_before_process_exit():
    result = _encoder_lifecycle_subprocess(False)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert (
        lines.index("rpc stopped")
        < lines.index("native observer stopped")
        < lines.index("backend stopped")
    )
    assert lines[-1] == "owner cleanup complete"


@pytest.mark.asyncio
async def test_shutdown_keeps_reconciliation_open_until_native_work_is_terminal(monkeypatch):
    import asyncio
    import signal

    from smg_grpc_servicer.sglang.managed.launch import serve

    state, config, args = _managed_encoder_lifecycle(monkeypatch, active=1)
    task = asyncio.create_task(serve(config, args))
    await state.started.wait()
    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    await asyncio.sleep(0.06)
    assert "draining" in state.events and "rpc stopped" not in state.events
    assert not task.done()
    state.active = 0  # The independent native owner finally proves quiescence.
    await task
    assert state.events.index("rpc stopped") < state.events.index("native observer stopped")
    assert state.events.index("native observer stopped") < state.events.index("backend stopped")
    assert state.events[-2:] == ["follower terminated", "follower joined"]


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


def test_ordinary_text_uses_native_intake_and_queue_without_export(monkeypatch):
    request = SimpleNamespace(
        rid="ordinary-text",
        sampling_params=SimpleNamespace(custom_params={}),
        mm_inputs=None,
        need_wait_for_mm_inputs=False,
        to_finish=None,
    )
    monkeypatch.setattr(plugin, "_envelope", lambda req: pytest.fail("text needs no envelope"))
    seen = []

    def original(scheduler, req, **kwargs):
        seen.append((req, kwargs))

    plugin.prepare_request(original, object(), request)
    queued = SimpleNamespace(
        rid=request.rid,
        sampling_params=request.sampling_params,
        multimodal_inputs=None,
        input_embeds=None,
        to_finish=None,
    )
    plugin.before_queue(original, object(), queued)
    assert seen == [(request, {}), (queued, {})]


@pytest.mark.parametrize(
    "changes",
    [
        {"mm_inputs": object()},
        {"need_wait_for_mm_inputs": True},
        {"input_embeds": [1.0]},
        {"sampling_params": SimpleNamespace(custom_params={plugin.ENVELOPE: None})},
    ],
)
def test_ordinary_text_bypass_never_accepts_media_or_a_malformed_envelope(changes):
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(custom_params={}),
        mm_inputs=None,
        need_wait_for_mm_inputs=False,
    )
    for key, value in changes.items():
        setattr(request, key, value)
    assert not plugin._ordinary_text(request)


@pytest.mark.parametrize("media", [object(), []])
def test_scheduler_text_bypass_refuses_materialized_media(media):
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(custom_params={}),
        multimodal_inputs=media,
        input_embeds=None,
    )
    assert not plugin._ordinary_text(request, "multimodal_inputs")


def test_text_bypass_refuses_unknown_request_shape():
    request = SimpleNamespace(sampling_params=SimpleNamespace(custom_params={}))
    assert not plugin._ordinary_text(request)
    assert not plugin._ordinary_text(request, "multimodal_inputs")


def test_encoder_metadata_failure_precedes_native_runtime(monkeypatch):
    import asyncio

    from smg_grpc_servicer.sglang import model_artifacts
    from smg_grpc_servicer.sglang.managed import encoder_bridge, launch

    _, settings, args = _managed_encoder_lifecycle(monkeypatch)

    def reject(_args):
        raise ValueError("metadata unavailable")

    def unexpected(*_args):
        pytest.fail("native runtime started before metadata resolution")

    monkeypatch.setattr(model_artifacts, "encoder_artifact_dirs", reject)
    monkeypatch.setattr(encoder_bridge, "launch_runtime", unexpected)
    with pytest.raises(ValueError, match="metadata unavailable"):
        asyncio.run(launch.serve(settings, args))
