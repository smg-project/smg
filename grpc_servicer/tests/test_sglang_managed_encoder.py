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
    assert epd_launcher.encoder_staging_bytes(4096, "bfloat16") == 32768 * 4096 * 2
    qwen3_cfg = SimpleNamespace(
        vision_config={
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "in_chans": 3,
        }
    )
    assert epd_launcher.encoder_artifact_bytes(qwen3_cfg, "bfloat16") == 388 * 1024**2
    assert epd_launcher.encoder_artifact_bytes(qwen3_cfg, "float32") == 772 * 1024**2
    qwen25_cfg = SimpleNamespace(
        vision_config=SimpleNamespace(
            patch_size={"height": 14, "width": 14},
            temporal_patch_size=2,
            merge_size=2,
            num_channels=3,
        )
    )
    assert epd_launcher.encoder_artifact_bytes(qwen25_cfg, "bfloat16") == 298 * 1024**2
    assert epd_launcher.encoder_artifact_bytes(SimpleNamespace(), "bfloat16") == 388 * 1024**2
    for backend_name in ("mooncake", "zmq_to_scheduler"):
        win_cfg = {
            **cfg,
            "embedding_backend": backend_name,
            "max_artifact_bytes": epd_launcher.encoder_artifact_bytes(qwen3_cfg, "bfloat16"),
            "max_staging_bytes_per_request": epd_launcher.encoder_staging_bytes(4096, "bfloat16"),
        }
        win_cap, _ = epd_launcher.encoder_window(win_cfg)
        assert 0 < win_cap <= win_cfg["max_active"]
    with pytest.raises(ValueError, match="visual dtype"):
        epd_launcher.encoder_artifact_bytes(qwen3_cfg, "int8")
    with pytest.raises(ValueError, match="vision geometry"):
        epd_launcher.encoder_artifact_bytes(
            SimpleNamespace(vision_config={"patch_size": True}), "bfloat16"
        )

    args.host = "::1"
    args.dtype = "half"
    v6_cfg = epd_launcher.complete(args, "encoder")
    assert v6_cfg["control_listen"] == "[::1]:30010"
    assert v6_cfg["receive_endpoint"] == "http://[::1]:30000"
    assert v6_cfg["embedding_dtype"] == "float16"

    args.dtype = "float"
    assert epd_launcher.complete(args, "encoder")["embedding_dtype"] == "float32"

    args.port = 65530
    with pytest.raises(ValueError, match="--port"):
        epd_launcher.complete(args, "encoder")
    args.port = 30000

    for field in ("dp_size", "attn_dp_size"):
        setattr(args, field, 2)
        for role in ("encoder", "prefill", "decode"):
            with pytest.raises(ValueError, match=field):
                epd_launcher.complete(args, role)
        setattr(args, field, 1)

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


@pytest.mark.asyncio
async def test_native_encoder_forget_and_unknown_cancel_wake_waiters():
    backend = SimpleNamespace(
        encoder=SimpleNamespace(transfer_backend="mooncake"),
        scheduler=SimpleNamespace(),
    )
    owner = epd_encoder.NativeEncoder({"max_receivers": 4}, backend)

    waiter_forget = asyncio.create_task(owner._wait_entry("r-forget", timeout=120.0))
    await asyncio.sleep(0.01)
    owner.forget("r-forget")
    with pytest.raises(epd_encoder.Rejected, match="aborted before execution"):
        await asyncio.wait_for(waiter_forget, timeout=0.5)

    with pytest.raises(epd_encoder.Rejected, match="aborted before execution"):
        await asyncio.wait_for(owner._wait_entry("r-forget", timeout=120.0), timeout=0.5)

    waiter_cancel = asyncio.create_task(owner._wait_entry("r-cancel", timeout=120.0))
    await asyncio.sleep(0.01)
    await owner.cancel("r-cancel")
    with pytest.raises(epd_encoder.Rejected, match="aborted before execution"):
        await asyncio.wait_for(waiter_cancel, timeout=0.5)

    assert "r-forget" in owner.abandoned

    # Re-executing a previously forgotten rid must clear the abandoned tombstone.
    class FakeEncoder:
        transfer_backend = "mooncake"

        async def abandon_request(self, rid):
            pass

        async def release_request(self, rid):
            pass

        def clear_abandoned_request(self, rid):
            pass

    monkeypatch = pytest.MonkeyPatch()
    try:
        server_mod = SimpleNamespace(
            meta_registry=SimpleNamespace(publish=lambda *a, **kw: asyncio.sleep(0))
        )
        runtime_mod = SimpleNamespace(execute_encode_pipeline=lambda *a, **kw: asyncio.sleep(0))
        for name, mod in [
            ("sglang", SimpleNamespace()),
            ("sglang.srt", SimpleNamespace()),
            ("sglang.srt.disaggregation", SimpleNamespace()),
            ("sglang.srt.disaggregation.encoder", SimpleNamespace()),
            ("sglang.srt.disaggregation.encoder.server", server_mod),
            ("sglang.srt.disaggregation.encoder.runtime", runtime_mod),
            ("sglang.srt.managers", SimpleNamespace()),
            (
                "sglang.srt.managers.req_types",
                SimpleNamespace(Modality=SimpleNamespace(IMAGE="img")),
            ),
        ]:
            monkeypatch.setitem(sys.modules, name, mod)
        owner.backend.encoder = FakeEncoder()
        receipt = await owner.execute(
            {
                "rid": "r-forget",
                "prefill_host": "10.0.0.1",
                "remaining_ms": 5000,
                "embedding_tokens": 0,
            },
            b"",
        )
        assert receipt["released"] is True
        assert "r-forget" not in owner.abandoned
    finally:
        monkeypatch.undo()


@pytest.mark.asyncio
async def test_http_facade_body_limit_independent_of_max_control_bytes(monkeypatch):
    backend = SimpleNamespace(
        encoder=SimpleNamespace(transfer_backend="mooncake"),
        scheduler=SimpleNamespace(),
    )
    owner = epd_encoder.NativeEncoder({"max_receivers": 4}, backend)
    captured = {}

    class StubRouter:
        def add_post(self, path, handler):
            captured["path"] = path
            captured["handler"] = handler

    class StubApp:
        def __init__(self, *, client_max_size):
            captured["client_max_size"] = client_max_size
            self.router = StubRouter()

    class StubRunner:
        def __init__(self, app, *, access_log=None):
            captured["app"] = app

        async def setup(self):
            pass

        async def cleanup(self):
            pass

    class StubSite:
        def __init__(self, runner, host, port):
            captured["bind"] = (host, port)

        async def start(self):
            pass

    class StubResp:
        def __init__(self, payload, *, status=200):
            self.payload = payload
            self.status = status

    stub_web = SimpleNamespace(
        Application=StubApp,
        AppRunner=StubRunner,
        TCPSite=StubSite,
        json_response=lambda payload, status=200: StubResp(payload, status=status),
    )
    monkeypatch.setitem(sys.modules, "aiohttp", SimpleNamespace(web=stub_web))
    await epd_encoder._start_http_facade(
        owner,
        {
            "http_listen": "127.0.0.1:0",
            "max_control_bytes": 64 * 1024**2,
            "max_http_connections": 16,
        },
    )
    expected_cap = epd_encoder.MAX_RECEIVER_HTTP_BODY_BYTES
    assert captured["client_max_size"] == expected_cap + 1024

    class OversizedRequest:
        match_info = {"route": "send"}

        async def read(self):
            return b'{"pad":"' + b"a" * expected_cap + b'"}'

    resp = await captured["handler"](OversizedRequest())
    assert resp.status == 413


@pytest.mark.asyncio
async def test_serve_native_bounded_shutdown_on_hung_task(tmp_path, monkeypatch):
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

    async def hung_execute(request, owner_cancel=None):
        await asyncio.sleep(10)

    class FakeEncoder:
        transfer_backend = "mooncake"
        _embedding_dims = {}
        _element_size = 2

        async def abandon_request(self, rid):
            await asyncio.sleep(10)

        async def release_request(self, rid):
            pass

        def clear_abandoned_request(self, rid):
            pass

    monkeypatch.setattr(
        epd_encoder.NativeEncoder,
        "execute",
        lambda self, cmd, art: hung_execute(cmd),
    )
    backend = SimpleNamespace(encoder=FakeEncoder(), scheduler=SimpleNamespace())
    cfg = {
        "native_commands": cmd_ep,
        "native_events": evt_ep,
        "max_control_bytes": 4096,
        "max_artifact_bytes": 4096,
        "max_receivers": 2,
        "shutdown_timeout_seconds": 0.15,
    }
    task = asyncio.create_task(epd_encoder.serve_native(cfg, backend))
    try:
        assert await asyncio.wait_for(pull.recv_json(), timeout=2) == {"ready": True}
        await push.send_json({"sequence": 1, "operation": "execute", "rid": "hung"})
        await asyncio.sleep(0.05)
        start = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2.0)
        assert time.monotonic() - start < 1.0
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        push.close()
        pull.close()
        ctx.term()
