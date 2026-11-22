"""Tests for managed SGLang encoder ownership, zero-copy bridge, and launcher config."""

import asyncio
import gc
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from smg_grpc_servicer.sglang import epd_encoder, epd_launcher, epd_plugin


@pytest.mark.parametrize("images_per_request", [[1], [2, 1]])
def test_materialized_pixels_remain_owned_after_artifact_release(monkeypatch, images_per_request):
    """Single and fused inputs retain writable storage without mutating wire artifacts."""
    torch = pytest.importorskip("torch")
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.disaggregation.encoder.preprocessor",
        SimpleNamespace(EncoderPreprocessResult=SimpleNamespace),
    )
    from smg_grpc_proto.generated import sglang_scheduler_pb2 as sg

    request = sg.GenerateRequest()
    mm = request.mm_inputs
    mm.pixel_values.dtype = "float32"
    mm.pixel_values.shape.extend([4, 1])
    mm.pixel_values.data = struct.pack("4f", 1, 2, 3, 4)
    grid = mm.model_specific_tensors["image_grid_thw"]
    grid.dtype = "int64"
    grid.shape.extend([1, 3])
    grid.data = struct.pack("3q", 1, 2, 2)
    mm.mm_placeholders.add(length=1)

    requests, expected = [], []
    for count in images_per_request:
        artifact = sg.GenerateRequest(mm_inputs=mm)
        for index in range(count):
            batch = artifact.extra_mm_inputs.add() if index else artifact.mm_inputs
            batch.CopyFrom(mm)
            values = list(range(len(expected), len(expected) + 4))
            batch.pixel_values.data = struct.pack("4f", *values)
            expected.extend(values)
        requests.append(
            {epd_encoder.ARTIFACT_KEY: artifact.SerializeToString(), "mm_items": [None] * count}
        )
    first_wire = requests[0][epd_encoder.ARTIFACT_KEY]
    result, counts = epd_encoder.RustPreprocessorBridge(None, 4096)._materialize_batch(
        requests, SimpleNamespace(name="IMAGE")
    )
    del requests, request
    gc.collect()
    pixels = result.mm_inputs["pixel_values"]
    assert torch.equal(pixels.flatten(), torch.tensor(expected))
    pixels[0, 0] = 9
    unchanged = sg.GenerateRequest.FromString(first_wire)
    assert struct.unpack("4f", unchanged.mm_inputs.pixel_values.data) == (0, 1, 2, 3)
    assert counts == images_per_request
    assert result.token_counts == [1] * sum(images_per_request)
    assert result.grid_thw.tolist() == [[1, 2, 2]] * sum(images_per_request)


@pytest.mark.asyncio
async def test_run_uninterruptible_retains_owner_across_repeated_cancel():
    entered, release = threading.Event(), threading.Event()
    finished = []

    def work():
        entered.set()
        release.wait(2)
        finished.append(True)
        return "ok"

    with ThreadPoolExecutor(max_workers=1) as pool:
        task = asyncio.create_task(epd_encoder.run_uninterruptible(pool, work))
        while not entered.is_set():
            await asyncio.sleep(0.005)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished == [True]


@pytest.mark.asyncio
async def test_native_encoder_early_facade_and_release_barrier(monkeypatch):
    """Early facade waits for execute, enforces host fence, and release waits for active sends."""
    release_send, send_started = asyncio.Event(), asyncio.Event()
    calls, fail_release = [], [True]

    class FakeEncoder:
        transfer_backend = "mooncake"

        async def abandon_request(self, rid):
            calls.append(("abandon", rid))

        async def release_request(self, rid):
            calls.append(("release", rid))
            if fail_release[0]:
                raise RuntimeError("release failed")

        def clear_abandoned_request(self, rid):
            calls.append(("clear", rid))

    async def fake_send(encoder, req, *, release_without_count):
        send_started.set()
        await release_send.wait()
        return True

    server_mod = SimpleNamespace(
        meta_registry=SimpleNamespace(publish=lambda *a, **kw: asyncio.sleep(0))
    )
    runtime_mod = SimpleNamespace(send_staged_embedding=fake_send)
    encoder_pkg = SimpleNamespace(server=server_mod, runtime=runtime_mod)
    for name, mod in [
        ("sglang", SimpleNamespace()),
        ("sglang.srt", SimpleNamespace()),
        ("sglang.srt.disaggregation", SimpleNamespace()),
        ("sglang.srt.disaggregation.encoder", encoder_pkg),
        ("sglang.srt.disaggregation.encoder.server", server_mod),
        ("sglang.srt.disaggregation.encoder.runtime", runtime_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    backend = SimpleNamespace(encoder=FakeEncoder(), scheduler=SimpleNamespace())
    owner = epd_encoder.NativeEncoder({"max_receivers": 4}, backend)

    # Early facade waits for entry registration.
    early = asyncio.create_task(
        owner.facade(
            "send",
            {
                "req_id": "r1",
                "receive_count": 1,
                "prefill_host": "10.0.0.1",
                "embedding_port": 5000,
            },
        )
    )
    await asyncio.sleep(0.02)
    assert not early.done()
    entry = epd_encoder.NativeEntry(time.monotonic() + 5.0, "r1", prefill_host="10.0.0.1")
    owner.entries["r1"] = entry
    for waiter in owner.waiters.pop("r1", ()):
        waiter.set()
    await send_started.wait()

    # Wrong host is rejected.
    with pytest.raises(epd_encoder.Rejected, match="frozen P host"):
        await owner.facade(
            "send",
            {
                "req_id": "r1",
                "receive_count": 1,
                "prefill_host": "10.0.0.2",
                "embedding_port": 5000,
            },
        )

    finish_task = asyncio.create_task(owner._finish(entry, "cancelled"))
    await asyncio.sleep(0.02)
    assert not finish_task.done()
    assert ("release", "r1") not in calls

    release_send.set()
    await early
    receipt = await finish_task
    assert receipt["released"] is False
    assert "r1" in owner.entries

    fail_release[0] = False
    entry2 = epd_encoder.NativeEntry(time.monotonic() + 5.0, "r2", prefill_host="10.0.0.1")
    owner.entries["r2"] = entry2
    assert await owner._finish(entry2, None) == {"released": True, "error": None}
    assert "r2" not in owner.entries


@pytest.mark.asyncio
async def test_serve_native_ipc_roundtrip_and_clean_shutdown(tmp_path):
    zmq = pytest.importorskip("zmq")
    zmq_asyncio = pytest.importorskip("zmq.asyncio")

    cmd_ep = f"ipc://{tmp_path}/cmd.sock"
    evt_ep = f"ipc://{tmp_path}/evt.sock"
    ctx = zmq_asyncio.Context()
    push, pull = ctx.socket(zmq.PUSH), ctx.socket(zmq.PULL)
    push.setsockopt(zmq.LINGER, 0)
    pull.setsockopt(zmq.LINGER, 0)
    push.bind(cmd_ep)
    pull.bind(evt_ep)

    class FakeEncoder:
        transfer_backend = "mooncake"

        async def abandon_request(self, rid):
            pass

        async def release_request(self, rid):
            pass

        def clear_abandoned_request(self, rid):
            pass

    backend = SimpleNamespace(encoder=FakeEncoder(), scheduler=SimpleNamespace())
    cfg = {
        "native_commands": cmd_ep,
        "native_events": evt_ep,
        "max_control_bytes": 4096,
        "max_artifact_bytes": 4096,
        "max_receivers": 2,
    }
    task = asyncio.create_task(epd_encoder.serve_native(cfg, backend))
    ready = await asyncio.wait_for(pull.recv_json(), timeout=2)
    assert ready == {"ready": True}
    await push.send_json({"sequence": 1, "operation": "cancel", "rid": "missing"})
    reply = await asyncio.wait_for(pull.recv_json(), timeout=2)
    assert reply == {"sequence": 1, "result": None}
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    push.close()
    pull.close()
    ctx.term()


def test_launcher_complete_and_window_validation():
    args = SimpleNamespace(
        nnodes=1,
        pp_size=1,
        dp_size=1,
        attn_dp_size=1,
        tp_size=1,
        enable_unified_memory=False,
        revision="rev1",
        host="10.0.0.1",
        port=30000,
        dtype="bfloat16",
        disaggregation_transfer_backend="mooncake",
        encoder_transfer_backend="mooncake",
        disaggregation_bootstrap_port=8998,
        model_path="model",
    )
    cfg = epd_launcher.complete(args, "encoder")
    assert cfg["control_listen"] == "10.0.0.1:30010"
    assert cfg["receive_endpoint"] == "http://10.0.0.1:30000"
    cap, _ = epd_launcher.encoder_window(cfg)
    assert 0 < cap <= cfg["max_active"]
    assert epd_launcher.encoder_staging_bytes(4096, "bfloat16") == 4096 * 4096 * 2

    args.host = "::1"
    args.dtype = "half"
    v6_cfg = epd_launcher.complete(args, "encoder")
    assert v6_cfg["control_listen"] == "[::1]:30010"
    assert v6_cfg["receive_endpoint"] == "http://[::1]:30000"
    assert v6_cfg["embedding_dtype"] == "float16"

    args.dtype = "float"
    assert epd_launcher.complete(args, "encoder")["embedding_dtype"] == "float32"

    args.revision = None
    with pytest.raises(ValueError, match="--revision"):
        epd_launcher.complete(args, "prefill")


def test_plugin_register_hooks_by_role(monkeypatch):
    registered = []
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.plugins.hook_registry",
        SimpleNamespace(
            HookRegistry=SimpleNamespace(register=lambda *a: registered.append(a)),
            HookType=SimpleNamespace(AROUND="around", AFTER="after"),
        ),
    )
    monkeypatch.setattr(epd_plugin, "_registered_role", None)
    monkeypatch.setenv(epd_plugin.ROLE_ENV, "prefill")
    epd_plugin.register()
    targets = {t for t, _, _ in registered}
    assert any(t.endswith("._add_request_to_queue") for t in targets)
    assert any(t.endswith("._get_multimodal_inputs") for t in targets)
