"""Tests for managed SGLang encoder ownership, zero-copy bridge, and launcher config."""

import asyncio
import gc
import json
import os
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
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


class _FakeEncoder:
    transfer_backend = "mooncake"
    _embedding_dims = {}
    _element_size = 2

    def __init__(self, calls=None, fail_release=None, hang_abandon=False):
        self.calls = calls if calls is not None else []
        self.fail_release = fail_release or [False]
        self.hang_abandon = hang_abandon

    async def abandon_request(self, rid):
        self.calls.append(("abandon", rid))
        if self.hang_abandon:
            await asyncio.sleep(10)

    async def release_request(self, rid):
        self.calls.append(("release", rid))
        if self.fail_release[0]:
            raise RuntimeError("release failed")

    def clear_abandoned_request(self, rid):
        self.calls.append(("clear", rid))


def _ipc_sockets(tmp_path):
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
    return ctx, push, pull, cmd_ep, evt_ep


@pytest.mark.asyncio
async def test_native_encoder_early_facade_and_release_barrier(monkeypatch):
    """Early facade waits for execute, enforces host fence, and release waits for active sends."""
    release_send, send_started = asyncio.Event(), asyncio.Event()
    calls, fail_release = [], [True]

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

    backend = SimpleNamespace(
        encoder=_FakeEncoder(calls, fail_release), scheduler=SimpleNamespace()
    )
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
    owner._wake_waiters("r1", "registered")
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
    ctx, push, pull, cmd_ep, evt_ep = _ipc_sockets(tmp_path)
    backend = SimpleNamespace(encoder=_FakeEncoder(), scheduler=SimpleNamespace())
    cfg = {
        "native_commands": cmd_ep,
        "native_events": evt_ep,
        "max_control_bytes": 4096,
        "max_artifact_bytes": 4096,
        "max_receivers": 2,
    }
    task = asyncio.create_task(epd_encoder.serve_native(cfg, backend))
    try:
        assert await asyncio.wait_for(pull.recv_json(), timeout=2) == {"ready": True}
        await push.send_json({"sequence": 1, "operation": "cancel", "rid": "missing"})
        assert await asyncio.wait_for(pull.recv_json(), timeout=2) == {
            "sequence": 1,
            "result": None,
        }
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
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
        win_cap, win_exec_bytes = epd_launcher.encoder_window(win_cfg)
        assert 0 < win_cap <= win_cfg["max_active"]
        expected_copies = 1 if backend_name == "mooncake" else win_cfg["max_receivers"]
        assert win_exec_bytes == 2 * win_cfg["max_artifact_bytes"] + win_cfg[
            "max_staging_bytes_per_request"
        ] * (1 + expected_copies)
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

    # P/D support native DP (e.g. Decode DP8); only encoder restricts dp_size/attn_dp_size to 1.
    args.dp_size = 8
    args.attn_dp_size = 8
    assert epd_launcher.complete(args, "decode")["role"] == "decode"
    assert epd_launcher.complete(args, "prefill")["role"] == "prefill"
    with pytest.raises(ValueError, match="dp_size"):
        epd_launcher.complete(args, "encoder")
    args.dp_size = 1
    with pytest.raises(ValueError, match="attn_dp_size"):
        epd_launcher.complete(args, "encoder")
    args.attn_dp_size = 1

    args.revision = None
    with pytest.raises(ValueError, match="--revision"):
        epd_launcher.complete(args, "prefill")
    args.revision = "rev1"

    args.port = 65530
    with pytest.raises(ValueError, match="--port"):
        epd_launcher.complete(args, "encoder")
    args.port = 30000


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

    waiter_race = asyncio.create_task(owner._wait_entry("r-race", timeout=120.0))
    await asyncio.sleep(0.01)
    owner.forget("r-race")
    owner.abandoned.pop("r-race", None)
    owner.entries["r-race"] = epd_encoder.NativeEntry(
        time.monotonic() + 5.0, "r-race", prefill_host="10.0.0.1"
    )
    with pytest.raises(epd_encoder.Rejected, match="aborted before execution"):
        await asyncio.wait_for(waiter_race, timeout=0.5)
    owner.entries.pop("r-race", None)

    assert "r-forget" in owner.abandoned

    # Re-executing a previously forgotten rid must clear the abandoned tombstone.
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
        owner.backend.encoder = _FakeEncoder()
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

    assert epd_encoder._HTTP_STATUS
    for code in epd_encoder._HTTP_STATUS.values():
        assert isinstance(code, HTTPStatus)
        assert int(code) != 499
        assert HTTPStatus(int(code)) == code

    for mod in list(sys.modules):
        if mod == "sglang" or mod.startswith("sglang."):
            monkeypatch.setitem(sys.modules, mod, None)
    monkeypatch.setitem(sys.modules, "sglang", None)
    monkeypatch.setitem(sys.modules, "sglang.srt.disaggregation.encoder.server", None)
    monkeypatch.setitem(sys.modules, "sglang.srt.disaggregation.encoder.runtime", None)
    owner.forget("cancelled-rid")

    class CancelledRequest:
        match_info = {"route": "metadata"}

        async def read(self):
            return json.dumps({"req_id": "cancelled-rid", "part_idx": 0}).encode()

    cancelled_resp = await captured["handler"](CancelledRequest())
    assert cancelled_resp.status == 404


@pytest.mark.asyncio
async def test_serve_native_bounded_shutdown_on_hung_task(tmp_path, monkeypatch):
    ctx, push, pull, cmd_ep, evt_ep = _ipc_sockets(tmp_path)
    started, calls = asyncio.Event(), []

    async def hung_execute_encode_pipeline(encoder, submitter, native):
        started.set()
        await asyncio.sleep(10)

    server_mod = SimpleNamespace(
        meta_registry=SimpleNamespace(publish=lambda *a, **kw: asyncio.sleep(0))
    )
    runtime_mod = SimpleNamespace(execute_encode_pipeline=hung_execute_encode_pipeline)
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

    encoder = _FakeEncoder(calls, hang_abandon=True)
    encoder._embedding_dims = {"img": 16}
    backend = SimpleNamespace(encoder=encoder, scheduler=SimpleNamespace())
    cfg = {
        "native_commands": cmd_ep,
        "native_events": evt_ep,
        "max_control_bytes": 4096,
        "max_artifact_bytes": 4096,
        "max_staging_bytes_per_request": 4096,
        "max_receivers": 2,
        "shutdown_timeout_seconds": 0.15,
    }
    task = asyncio.create_task(epd_encoder.serve_native(cfg, backend))
    try:
        assert await asyncio.wait_for(pull.recv_json(), timeout=2) == {"ready": True}
        await push.send_json(
            {
                "sequence": 1,
                "operation": "execute",
                "rid": "hung",
                "prefill_host": "10.0.0.1",
                "remaining_ms": 5000,
                "embedding_tokens": 4,
                "image_count": 1,
            }
        )
        await asyncio.wait_for(started.wait(), timeout=2.0)
        start = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2.0)
        assert ("abandon", "hung") in calls
        assert time.monotonic() - start < 1.0
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        push.close()
        pull.close()
        ctx.term()


def test_launcher_pd_item_limits_and_high_token_windows(monkeypatch):
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
        limit_mm_data_per_request='{"image": 64, "video": 2}',
    )
    cfg = epd_launcher.complete(args, "encoder")
    assert cfg["max_images"] == 1024
    assert cfg["max_control_bytes"] == 256 * 1024**2
    assert cfg["mm_item_limits"] == "image=64,video=2"
    assert cfg["processor_options"]["engine_item_limits"] == {"image": 64, "video": 2}

    qwen3_cfg = SimpleNamespace(
        vision_config={
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "in_chans": 3,
        }
    )
    for backend_name in ("mooncake", "zmq_to_scheduler"):
        win_cfg = {
            **cfg,
            "embedding_backend": backend_name,
            "max_artifact_bytes": epd_launcher.encoder_artifact_bytes(
                qwen3_cfg, "bfloat16", max_tokens=131072
            ),
            "max_staging_bytes_per_request": epd_launcher.encoder_staging_bytes(
                4096, "bfloat16", max_tokens=131072
            ),
        }
        win_cap, _ = epd_launcher.encoder_window(win_cfg)
        assert win_cap >= 1

    large_vlm_cfg = SimpleNamespace(
        vision_config={
            "patch_size": 14,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "in_channels": 3,
        }
    )
    large_vlm_win_cfg = {
        **cfg,
        "embedding_backend": "mooncake",
        "max_artifact_bytes": epd_launcher.encoder_artifact_bytes(
            large_vlm_cfg, "bfloat16", max_tokens=131072
        ),
        "max_staging_bytes_per_request": epd_launcher.encoder_staging_bytes(
            6144, "bfloat16", max_tokens=131072
        ),
    }
    large_vlm_cap, _ = epd_launcher.encoder_window(large_vlm_win_cfg)
    assert large_vlm_cap >= 7


@pytest.mark.asyncio
async def test_native_encoder_video_and_multipart_metadata(monkeypatch):
    published_calls = []
    fake_modality = SimpleNamespace(IMAGE="img", VIDEO="vid")
    execute_started = asyncio.Event()
    allow_deliver = asyncio.Event()

    class FakeMetaRegistry:
        async def publish(self, *args, **kwargs):
            published_calls.append((args, kwargs))

        async def wait(self, rid):
            return {"embedding_size": 128}

    server_mod = SimpleNamespace(meta_registry=FakeMetaRegistry())

    async def fake_execute(encoder, submitter, native, send_sockets=None):
        published_calls.append(("execute", native, send_sockets))
        execute_started.set()
        await allow_deliver.wait()
        owner.entries[native["req_id"]].delivered.set()

    runtime_mod = SimpleNamespace(
        execute_encode_pipeline=fake_execute,
        send_staged_embedding=lambda *a, **kw: asyncio.sleep(0, result=True),
    )
    for name, mod in [
        ("sglang", SimpleNamespace()),
        ("sglang.srt", SimpleNamespace()),
        ("sglang.srt.disaggregation", SimpleNamespace()),
        (
            "sglang.srt.disaggregation.encoder",
            SimpleNamespace(server=server_mod, runtime=runtime_mod),
        ),
        ("sglang.srt.disaggregation.encoder.server", server_mod),
        ("sglang.srt.disaggregation.encoder.runtime", runtime_mod),
        ("sglang.srt.managers", SimpleNamespace()),
        ("sglang.srt.managers.req_types", SimpleNamespace(Modality=fake_modality)),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    encoder = _FakeEncoder()
    encoder._embedding_dims = {"img": 16, "vid": 16}
    backend = SimpleNamespace(
        encoder=encoder, scheduler=SimpleNamespace(), send_sockets=["tp-sock-1"]
    )
    owner = epd_encoder.NativeEncoder(
        {"max_receivers": 2, "max_staging_bytes_per_request": 4096}, backend
    )

    bridge = epd_encoder.RustPreprocessorBridge(None, 4096)
    assert bridge.supports_modality(SimpleNamespace(name="IMAGE"))
    assert bridge.supports_modality(SimpleNamespace(name="VIDEO"))
    assert not bridge.supports_modality(SimpleNamespace(name="AUDIO"))

    exec_task = asyncio.create_task(
        owner.execute(
            {
                "rid": "vid-rid",
                "prefill_host": "10.0.0.1",
                "remaining_ms": 5000,
                "embedding_tokens": 8,
                "item_count": 1,
                "modality": "video",
                "num_parts": 2,
                "part_idx": 1,
            },
            b"artifact",
        )
    )
    await asyncio.wait_for(execute_started.wait(), timeout=1.0)
    expected_native = {
        "req_id": "vid-rid",
        "num_parts": 2,
        "part_idx": 1,
        "modality": "video",
        "mm_items": [{epd_encoder.ARTIFACT_KEY: b"artifact"}],
    }
    assert ("execute", expected_native, ["tp-sock-1"]) in published_calls
    meta = await asyncio.wait_for(
        owner.facade("metadata", {"req_id": "vid-rid", "part_idx": 1}), timeout=1.0
    )
    assert meta == {"req_id": "vid-rid", "part_idx": 1, "embedding_size": 128}
    with pytest.raises(epd_encoder.Rejected, match="invalid part index"):
        await asyncio.wait_for(
            owner.facade("metadata", {"req_id": "vid-rid", "part_idx": 0}), timeout=1.0
        )
    allow_deliver.set()
    assert await asyncio.wait_for(exec_task, timeout=1.0) == {"released": True, "error": None}


def test_materialized_video_and_timing_metadata(monkeypatch):
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
    mm.pixel_values.shape.extend([8, 1])
    mm.pixel_values.data = struct.pack("8f", *range(8))
    grid = mm.model_specific_tensors["video_grid_thw"]
    grid.dtype = "int64"
    grid.shape.extend([1, 3])
    grid.data = struct.pack("3q", 2, 2, 2)
    spg = mm.model_specific_tensors["video_second_per_grid"]
    spg.dtype = "float32"
    spg.shape.extend([1])
    spg.data = struct.pack("1f", 1.0)
    mm.mm_placeholders.add(length=1)
    mm.mm_placeholders.add(length=1)

    result, counts = epd_encoder.RustPreprocessorBridge(None, 4096)._materialize_batch(
        [{"mm_items": [{epd_encoder.ARTIFACT_KEY: request.SerializeToString()}]}],
        SimpleNamespace(name="VIDEO"),
    )
    assert counts == [1]
    assert result.token_counts == [2]
    assert torch.equal(result.mm_inputs["second_per_grid_ts"], torch.tensor([1.0]))
    assert torch.equal(result.mm_inputs["video_grid_thw"], torch.tensor([[2, 2, 2]]))


def test_encoder_main_sets_mooncake_send_timeout_default(monkeypatch):
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
        encoder_only=True,
        language_only=False,
    )
    monkeypatch.setitem(
        sys.modules,
        "smg_grpc_servicer.sglang.mm_defaults",
        SimpleNamespace(configure_threads=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "smg_grpc_servicer.sglang.epd_plugin",
        SimpleNamespace(EVENTS_ENV="E", ROLE_ENV="R", load_role_hooks=lambda: None),
    )
    monkeypatch.setattr(epd_launcher, "ROLE_ENV", "SMG_TEST_ROLE")
    seen_env = []
    monkeypatch.setattr(
        epd_launcher,
        "serve",
        lambda cfg, a: asyncio.sleep(
            0, result=seen_env.append(os.environ.get("SGLANG_ENCODER_SEND_TIMEOUT"))
        ),
    )
    monkeypatch.delenv("SGLANG_ENCODER_SEND_TIMEOUT", raising=False)
    epd_launcher.encoder_main(args)
    assert seen_env == ["15"]

    monkeypatch.setenv("SGLANG_ENCODER_SEND_TIMEOUT", "42")
    epd_launcher.encoder_main(args)
    assert seen_env == ["15", "42"]


@pytest.mark.asyncio
async def test_native_encoder_cancel_wakes_embedding_waiter(monkeypatch):
    req_states = {}
    abandoned = []

    class FakeMetaRegistry:
        async def publish(self, *args, **kwargs):
            pass

    server_mod = SimpleNamespace(meta_registry=FakeMetaRegistry())
    encoder_mod = SimpleNamespace(server=server_mod)
    disagg_mod = SimpleNamespace(encoder=encoder_mod)
    srt_mod = SimpleNamespace(disaggregation=disagg_mod)
    sglang_mod = SimpleNamespace(srt=srt_mod)
    for name, mod in (
        ("sglang", sglang_mod),
        ("sglang.srt", srt_mod),
        ("sglang.srt.disaggregation", disagg_mod),
        ("sglang.srt.disaggregation.encoder", encoder_mod),
        ("sglang.srt.disaggregation.encoder.server", server_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    class FakeBackend:
        scheduler = SimpleNamespace()

        class _Encoder:
            transfer_backend = "mooncake"
            _embedding_dims = {}
            _element_size = 2

            def __init__(self):
                self.req_states = req_states

            async def abandon_request(self, rid):
                abandoned.append(rid)

            async def release_request(self, rid):
                req_states.pop(rid, None)

            def clear_abandoned_request(self, rid):
                pass

        encoder = _Encoder()

    owner = epd_encoder.NativeEncoder({"max_receivers": 1, "max_active": 8}, FakeBackend())
    entry = epd_encoder.NativeEntry(time.monotonic() + 30.0, "req-stuck", prefill_host="10.0.0.1")
    owner.entries["req-stuck"] = entry
    ready_event = asyncio.Event()
    req_states["req-stuck"] = SimpleNamespace(active_encodes=1, embedding_ready=ready_event)

    # Active encode prevents premature waking.
    owner._wake_abandoned_sends("req-stuck")
    assert not ready_event.is_set()

    # Once active_encodes reaches 0, cancel breaks an orphaned _wait_for_embedding waiter.
    req_states["req-stuck"].active_encodes = 0
    await owner.cancel("req-stuck")
    assert ready_event.is_set()
    assert abandoned == ["req-stuck"]
    assert await asyncio.wait_for(owner._finish(entry, "cancelled"), timeout=1.0) == {
        "released": True,
        "error": "cancelled",
    }


def _make_single_image_artifact(sg, values):
    req = sg.GenerateRequest()
    mm = req.mm_inputs
    mm.pixel_values.dtype = "float32"
    mm.pixel_values.shape.extend([4, 1])
    mm.pixel_values.data = struct.pack("4f", *values)
    grid = mm.model_specific_tensors["image_grid_thw"]
    grid.dtype = "int64"
    grid.shape.extend([1, 3])
    grid.data = struct.pack("3q", 1, 2, 2)
    mm.mm_placeholders.add(length=1)
    return req.SerializeToString()


def test_stable_identity_namespace_ignores_model_path_differences():
    def fake_enc(model_path, *, revision="rev-1", hidden_size=1152):
        return SimpleNamespace(
            model_type="qwen3_vl",
            _embedding_dtype="bfloat16",
            server_args=SimpleNamespace(revision=revision),
            model_config=SimpleNamespace(
                revision=revision,
                hf_config=SimpleNamespace(
                    model_type="qwen3_vl",
                    vision_config={
                        "_name_or_path": model_path,
                        "hidden_size": hidden_size,
                        "patch_size": 14,
                    },
                ),
            ),
        )

    ns0 = epd_encoder._build_stable_identity_namespace(fake_enc("/scratch/vision-0"), "c-v1")
    ns1 = epd_encoder._build_stable_identity_namespace(fake_enc("/scratch/vision-1"), "c-v1")
    ns_rev = epd_encoder._build_stable_identity_namespace(
        fake_enc("/scratch/vision-0", revision="rev-2"), "c-v1"
    )
    ns_cfg = epd_encoder._build_stable_identity_namespace(
        fake_enc("/scratch/vision-0", hidden_size=1024), "c-v1"
    )
    assert len(ns0) == 32
    assert ns0 == ns1
    assert ns0 != ns_rev
    assert ns0 != ns_cfg


@pytest.mark.asyncio
async def test_materialize_batch_produces_stable_item_identities_across_batch_sizes(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.disaggregation.encoder.preprocessor",
        SimpleNamespace(EncoderPreprocessResult=SimpleNamespace),
    )
    from smg_grpc_proto.generated import sglang_scheduler_pb2 as sg

    art_a = _make_single_image_artifact(sg, (1.0, 2.0, 3.0, 4.0))
    art_b = _make_single_image_artifact(sg, (5.0, 6.0, 7.0, 8.0))
    sink_solo = {}
    sink_fused = {}
    ns = epd_encoder._build_stable_identity_namespace(
        SimpleNamespace(model_type="qwen3_vl", _embedding_dtype="bfloat16"), "smg-image-grid-v1"
    )
    bridge_solo = epd_encoder.RustPreprocessorBridge(
        None, 4096, stable_identity_ns=ns, identity_sink=sink_solo
    )
    bridge_fused = epd_encoder.RustPreprocessorBridge(
        None, 4096, stable_identity_ns=ns, identity_sink=sink_fused
    )
    res_solo, _ = bridge_solo._materialize_batch(
        [{"req_id": "r1", "mm_items": [{epd_encoder.ARTIFACT_KEY: art_a}]}],
        SimpleNamespace(name="IMAGE"),
    )
    res_fused, _ = bridge_fused._materialize_batch(
        [
            {"req_id": "r0", "mm_items": [{epd_encoder.ARTIFACT_KEY: art_b}]},
            {"req_id": "r1-fused", "mm_items": [{epd_encoder.ARTIFACT_KEY: art_a}]},
        ],
        SimpleNamespace(name="IMAGE"),
    )
    assert sink_solo["r1"] == sink_fused["r1-fused"]
    assert len(sink_solo["r1"]) == 1 and len(sink_solo["r1"][0]) == 64
    assert res_solo._smg_item_keys == [res_fused._smg_item_keys[1]]

    # NativeEncoder.facade("metadata", ...) exposes item_identities and _finish cleans up.
    meta_wait = asyncio.Future()
    meta_wait.set_result({"embedding_size": 16, "token_count": 1, "embedding_dim": 8})
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.disaggregation.encoder.server",
        SimpleNamespace(meta_registry=SimpleNamespace(wait=lambda rid: meta_wait)),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.disaggregation.encoder.runtime",
        SimpleNamespace(send_staged_embedding=None),
    )
    fake_enc = _FakeEncoder()
    fake_enc._smg_req_item_identities = dict(sink_solo)
    owner = epd_encoder.NativeEncoder(
        {"max_receivers": 4},
        SimpleNamespace(encoder=fake_enc, scheduler=SimpleNamespace()),
    )
    entry = epd_encoder.NativeEntry(time.monotonic() + 10.0, "r1", "127.0.0.1", part_idx=0)
    owner.entries["r1"] = entry
    resp = await owner.facade("metadata", {"req_id": "r1", "part_idx": 0})
    assert resp["item_identities"] == sink_solo["r1"]
    await owner._finish(entry, None)
    assert "r1" not in fake_enc._smg_req_item_identities


def test_install_bridge_preserves_sha256_cache_namespace_format():
    from smg_grpc_servicer.sglang import epd_encoder

    encoder = SimpleNamespace(
        rank=0,
        cache_namespace="sha256:" + "a" * 64,
        preprocessor=SimpleNamespace(
            process_batch_mm_items=lambda *a, **k: None,
            get_num_patches=lambda *a, **k: 1,
            supports_modality=lambda *a, **k: True,
            preproc_executor=SimpleNamespace(submit=lambda *a, **k: None),
        ),
    )
    epd_encoder.install_bridge(encoder, "smg-image-grid-v1", 1024)
    assert encoder.cache_namespace.startswith("sha256:")
    digest = encoder.cache_namespace.removeprefix("sha256:")
    assert len(digest) == 64
    bytes.fromhex(digest)

    # v0.5.21 MMEncoder does not define cache_namespace; install_bridge must stay compatible.
    v0521_encoder = SimpleNamespace(
        rank=0,
        preprocessor=SimpleNamespace(
            process_batch_mm_items=lambda *a, **k: None,
            get_num_patches=lambda *a, **k: 1,
            supports_modality=lambda *a, **k: True,
            preproc_executor=SimpleNamespace(submit=lambda *a, **k: None),
        ),
    )
    epd_encoder.install_bridge(v0521_encoder, "smg-image-grid-v1", 1024)
    assert not hasattr(v0521_encoder, "cache_namespace")
    assert isinstance(v0521_encoder.preprocessor, epd_encoder.RustPreprocessorBridge)


@pytest.mark.parametrize(
    ("transfer_backend", "initial_send_timeout", "initial_sweep_timeout", "expected_sweep_timeout"),
    [
        ("mooncake", 15.0, 15.0, epd_encoder.ABANDONED_TTL_SECONDS),
        ("mooncake", 15.0, 180.0, 180.0),
        ("zmq_to_scheduler", 180.0, 180.0, 180.0),
        ("zmq_to_scheduler", 15.0, 15.0, epd_encoder.ABANDONED_TTL_SECONDS),
    ],
)
def test_launch_runtime_decouples_meta_registry_sweep_timeout_from_send_timeout(
    monkeypatch,
    transfer_backend,
    initial_send_timeout,
    initial_sweep_timeout,
    expected_sweep_timeout,
):
    """launch_runtime raises meta_registry.sweep_timeout >= 125s without altering send_timeout."""
    meta_registry = SimpleNamespace(sweep_timeout=initial_sweep_timeout)
    server_mod = SimpleNamespace(meta_registry=meta_registry)
    published = []
    seen_targets = []
    original_launch_encoder = object()

    def fake_launch_local_runtime(server_args):
        seen_targets.append(runtime_mod.launch_encoder)
        meta_registry.sweep_timeout = initial_sweep_timeout
        return SimpleNamespace(
            encoder=SimpleNamespace(
                transfer_backend=transfer_backend,
                send_timeout=initial_send_timeout,
            ),
            scheduler=SimpleNamespace(),
        )

    runtime_mod = SimpleNamespace(
        launch_encoder=original_launch_encoder,
        launch_local_runtime=fake_launch_local_runtime,
    )
    encoder_pkg = SimpleNamespace(runtime=runtime_mod, server=server_mod)
    runtime_context_mod = SimpleNamespace(
        publish=lambda args, role=None: published.append((args, role))
    )
    for name, mod in (
        ("sglang", SimpleNamespace()),
        ("sglang.srt", SimpleNamespace()),
        ("sglang.srt.runtime_context", runtime_context_mod),
        ("sglang.srt.disaggregation", SimpleNamespace(encoder=encoder_pkg)),
        ("sglang.srt.disaggregation.encoder", encoder_pkg),
        ("sglang.srt.disaggregation.encoder.runtime", runtime_mod),
        ("sglang.srt.disaggregation.encoder.server", server_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    server_args = SimpleNamespace(dp_size=1, nnodes=1)
    backend = epd_encoder.launch_runtime(server_args, max_active=8)

    assert published == [(server_args, "encoder")]
    assert seen_targets == [epd_encoder.launch_managed_rank]
    assert runtime_mod.launch_encoder is original_launch_encoder
    assert backend.scheduler.pending_queue.maxsize == 8
    assert backend.encoder.send_timeout == initial_send_timeout
    assert meta_registry.sweep_timeout == expected_sweep_timeout


@pytest.mark.parametrize("bad_value", [None, "15", True, False, float("nan"), -5.0, float("-inf")])
def test_configure_meta_registry_sweep_timeout_normalizes_invalid_values(monkeypatch, bad_value):
    meta_registry = SimpleNamespace(sweep_timeout=bad_value)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.disaggregation.encoder.server",
        SimpleNamespace(meta_registry=meta_registry),
    )
    assert epd_encoder._configure_meta_registry_sweep_timeout() == epd_encoder.ABANDONED_TTL_SECONDS
    assert meta_registry.sweep_timeout == epd_encoder.ABANDONED_TTL_SECONDS


def test_configure_meta_registry_sweep_timeout_edge_cases(monkeypatch):
    inf_registry = SimpleNamespace(sweep_timeout=float("inf"))
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.disaggregation.encoder.server",
        SimpleNamespace(meta_registry=inf_registry),
    )
    assert epd_encoder._configure_meta_registry_sweep_timeout() == float("inf")
    assert inf_registry.sweep_timeout == float("inf")

    no_attr_registry = SimpleNamespace()
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.disaggregation.encoder.server",
        SimpleNamespace(meta_registry=no_attr_registry),
    )
    assert epd_encoder._configure_meta_registry_sweep_timeout() is None
    assert not hasattr(no_attr_registry, "sweep_timeout")

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.disaggregation.encoder.server",
        SimpleNamespace(meta_registry=None),
    )
    assert epd_encoder._configure_meta_registry_sweep_timeout() is None

    monkeypatch.setitem(sys.modules, "sglang.srt.disaggregation.encoder.server", None)
    assert epd_encoder._configure_meta_registry_sweep_timeout() is None
