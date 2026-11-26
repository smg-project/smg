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
    try:
        await asyncio.sleep(0.02)
        assert not early.done()
        entry = epd_encoder.NativeEntry(time.monotonic() + 5.0, "r1", prefill_host="10.0.0.1")
        owner.entries["r1"] = entry
        owner._wake_waiters("r1", "registered")
        await asyncio.wait_for(send_started.wait(), timeout=2.0)

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
        await asyncio.wait_for(early, timeout=2.0)
        receipt = await asyncio.wait_for(finish_task, timeout=2.0)
        assert receipt["released"] is False
        assert "r1" in owner.entries
    finally:
        release_send.set()
        if not early.done():
            early.cancel()
            await asyncio.gather(early, return_exceptions=True)

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
    assert cfg["max_active"] == 256
    assert cfg["max_http_connections"] == 256 * (8 + 1) + 16
    assert cfg["max_images"] == 1024
    assert cfg["max_control_bytes"] == 256 * 1024**2
    assert cfg["mm_item_limits"] == "image=64,video=2"
    assert cfg["processor_options"]["engine_item_limits"] == {"image": 64, "video": 2}
    assert cfg["shutdown_timeout_seconds"] == 30.0

    # Native --max-running-requests controls max_active; invalid/non-positive values are rejected.
    args.max_running_requests = 128
    custom_active_cfg = epd_launcher.complete(args, "encoder")
    assert custom_active_cfg["max_active"] == 128
    assert custom_active_cfg["max_http_connections"] == 128 * (8 + 1) + 16
    for invalid_max_running in (0, -4, True, 1.5, "not-an-int"):
        args.max_running_requests = invalid_max_running
        with pytest.raises(ValueError, match="--max-running-requests"):
            epd_launcher.complete(args, "encoder")
    args.max_running_requests = None

    # SMG_SGLANG_SERVICER_DRAIN_SECS controls shutdown_timeout_seconds via rust_lifecycle.env_float.
    monkeypatch.setenv("SMG_SGLANG_SERVICER_DRAIN_SECS", "45.5")
    assert epd_launcher.complete(args, "encoder")["shutdown_timeout_seconds"] == 45.5
    for invalid_drain in ("0", "-1", "nan", "inf", "abc"):
        monkeypatch.setenv("SMG_SGLANG_SERVICER_DRAIN_SECS", invalid_drain)
        with pytest.raises(ValueError, match="SMG_SGLANG_SERVICER_DRAIN_SECS"):
            epd_launcher.complete(args, "encoder")
    monkeypatch.delenv("SMG_SGLANG_SERVICER_DRAIN_SECS")

    # Baseline media token floor and auto-derived ceiling constants.
    assert epd_launcher.MAX_IMAGE_TOKENS_PER_REQUEST == 32768
    assert epd_launcher.MAX_MEDIA_TOKENS_CEILING == 262144

    qwen3_cfg = SimpleNamespace(
        vision_config={
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "in_chans": 3,
        }
    )
    # Verify realistic execution_capacity across 8B, 32B, and 122B MoE models under the 48 GiB
    # fallback budget: default cap stays tightly aligned with EncoderScheduler (8) so bursts queue
    # in Rust, while memory budget headroom remains >= 30..37 when max_batch_size is raised.
    for hidden_size, min_budget_cap in (
        (4096, 33),  # Qwen3-VL-8B
        (5120, 30),  # Qwen3-VL-32B
        (3072, 37),  # Qwen3.5-122B-A10B
    ):
        model_win_cfg = {
            **cfg,
            "max_context_bytes": epd_launcher.MAX_CONTEXT_BYTES,
            "embedding_backend": "mooncake",
            "max_artifact_bytes": epd_launcher.encoder_artifact_bytes(qwen3_cfg, "bfloat16"),
            "max_staging_bytes_per_request": epd_launcher.encoder_staging_bytes(
                hidden_size, "bfloat16"
            ),
        }
        default_cap, _ = epd_launcher.encoder_window(model_win_cfg, max_batch_size=8)
        assert default_cap == 8
        assert default_cap <= cfg["max_active"]
        raised_cap, _ = epd_launcher.encoder_window(model_win_cfg, max_batch_size=64)
        assert raised_cap >= min_budget_cap

    # Verify custom max_batch_size alignment and fallback on unset/non-positive max_batch_size.
    custom_bs_cfg = {
        **cfg,
        "max_context_bytes": epd_launcher.MAX_CONTEXT_BYTES,
        "embedding_backend": "mooncake",
        "max_artifact_bytes": epd_launcher.encoder_artifact_bytes(
            qwen3_cfg, "bfloat16", max_tokens=32768
        ),
        "max_staging_bytes_per_request": epd_launcher.encoder_staging_bytes(
            3072, "bfloat16", max_tokens=32768
        ),
    }
    assert epd_launcher.encoder_window(custom_bs_cfg, max_batch_size=16)[0] == 16
    assert epd_launcher.max_execution_capacity(None) == epd_launcher.DEFAULT_MAX_EXECUTION_CAPACITY
    assert epd_launcher.max_execution_capacity(0) == epd_launcher.DEFAULT_MAX_EXECUTION_CAPACITY
    assert epd_launcher.max_execution_capacity(-2) == epd_launcher.DEFAULT_MAX_EXECUTION_CAPACITY

    # Verify resolve_max_cpu default and SMG_SGLANG_MM_MAX_INFLIGHT reuse/validation.
    monkeypatch.delenv("SMG_SGLANG_MM_MAX_INFLIGHT", raising=False)
    monkeypatch.setattr(epd_launcher, "available_cpus", lambda: 16)
    assert epd_launcher.resolve_max_cpu(8, max_batch_size=8) == 8
    assert epd_launcher.resolve_max_cpu(20, max_batch_size=4) == 8
    monkeypatch.setattr(epd_launcher, "available_cpus", lambda: 2)
    assert epd_launcher.resolve_max_cpu(8, max_batch_size=8) == 2
    monkeypatch.setenv("SMG_SGLANG_MM_MAX_INFLIGHT", "12")
    assert epd_launcher.resolve_max_cpu(8, max_batch_size=8) == 12
    for invalid_inflight in ("", "0", "-3", "sixteen"):
        monkeypatch.setenv("SMG_SGLANG_MM_MAX_INFLIGHT", invalid_inflight)
        with pytest.raises(ValueError, match="SMG_SGLANG_MM_MAX_INFLIGHT must be positive"):
            epd_launcher.resolve_max_cpu(8, max_batch_size=8)
    monkeypatch.delenv("SMG_SGLANG_MM_MAX_INFLIGHT")

    # Safety math still holds for large media token budgets and tight memory contexts.
    for backend_name in ("mooncake", "zmq_to_scheduler"):
        win_cfg = {
            **cfg,
            "max_context_bytes": epd_launcher.MAX_CONTEXT_BYTES,
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

    tight_cfg = {
        **cfg,
        "embedding_backend": "mooncake",
        "max_context_bytes": 6 * 1024**3,
        "max_artifact_bytes": epd_launcher.encoder_artifact_bytes(
            qwen3_cfg, "bfloat16", max_tokens=32768
        ),
        "max_staging_bytes_per_request": epd_launcher.encoder_staging_bytes(
            3072, "bfloat16", max_tokens=32768
        ),
    }
    tight_cap, tight_exec = epd_launcher.encoder_window(tight_cfg)
    assert tight_cap < 8
    assert tight_cap * tight_exec <= 6 * 1024**3

    glm55_cfg = SimpleNamespace(
        vision_config={
            "patch_size": 14,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "in_channels": 3,
        }
    )
    glm55_win_cfg = {
        **cfg,
        "max_context_bytes": epd_launcher.MAX_CONTEXT_BYTES,
        "embedding_backend": "mooncake",
        "max_artifact_bytes": epd_launcher.encoder_artifact_bytes(
            glm55_cfg, "bfloat16", max_tokens=131072
        ),
        "max_staging_bytes_per_request": epd_launcher.encoder_staging_bytes(
            6144, "bfloat16", max_tokens=131072
        ),
    }
    glm55_cap, _ = epd_launcher.encoder_window(glm55_win_cfg)
    assert glm55_cap >= 7


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

    # Multi-encoder fan-out (num_parts > 2, e.g. 4 encoders) is accepted and validated.
    execute_started.clear()
    allow_deliver.clear()
    fanout_task = asyncio.create_task(
        owner.execute(
            {
                "rid": "fanout-rid_local_part_3",
                "prefill_host": "10.0.0.1",
                "remaining_ms": 5000,
                "embedding_tokens": 8,
                "item_count": 1,
                "modality": "image",
                "num_parts": 4,
                "part_idx": 3,
            },
            b"artifact",
        )
    )
    await asyncio.wait_for(execute_started.wait(), timeout=1.0)
    fanout_meta = await asyncio.wait_for(
        owner.facade("metadata", {"req_id": "fanout-rid_local_part_3", "part_idx": 3}),
        timeout=1.0,
    )
    assert fanout_meta == {
        "req_id": "fanout-rid_local_part_3",
        "part_idx": 3,
        "embedding_size": 128,
    }
    allow_deliver.set()
    assert await asyncio.wait_for(fanout_task, timeout=1.0) == {"released": True, "error": None}

    with pytest.raises(epd_encoder.Rejected, match="invalid native part count"):
        await owner.execute(
            {
                "rid": "bad-parts",
                "prefill_host": "10.0.0.1",
                "remaining_ms": 5000,
                "embedding_tokens": 8,
                "item_count": 1,
                "modality": "image",
                "num_parts": 1025,
                "part_idx": 0,
            },
            b"artifact",
        )


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
    before_keys = set(os.environ)
    epd_launcher.encoder_main(args)
    assert seen_env == ["15"]
    assert set(os.environ) - before_keys <= {epd_launcher.ROLE_ENV, "SGLANG_ENCODER_SEND_TIMEOUT"}

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


@pytest.mark.asyncio
async def test_per_item_embedding_cache_hits_eviction_and_bypass(monkeypatch):
    """Per-item LRU cache reuses hit items across turns/batches and evicts oldest entries."""
    torch = pytest.importorskip("torch")
    from smg_grpc_servicer.sglang import epd_encoder

    monkeypatch.setenv("SMG_SGLANG_ENCODER_ITEM_CACHE_MB", "1")
    forward_calls = []

    class FakePreprocessor:
        @staticmethod
        def get_num_patches(row, modality):
            return int(row[0]) * int(row[1]) * int(row[2])

    class FakeEncoder:
        tp_size = 1
        cache_namespace = "ns-v1"
        preprocessor = FakePreprocessor()
        _build_calls = []

        def _build_model_mm_items(self, mm_feature, preprocess_result, indices, modality):
            self._build_calls.append(list(indices))
            # Return dummy slice indices so get_feature_fn produces deterministic outputs per item.
            grid = preprocess_result.grid_thw
            pix = preprocess_result.mm_inputs["pixel_values"]
            offsets = [0]
            cur = 0
            for r in grid.tolist():
                cur += self.preprocessor.get_num_patches(r, modality)
                offsets.append(cur)
            return [(idx, pix[offsets[idx] : offsets[idx + 1]]) for idx in indices]

    def get_feature_fn(mm_items):
        forward_calls.append([idx for idx, _ in mm_items])
        chunks = []
        for _, feat in mm_items:
            # 1 token per item with 4-d embedding derived from patch sum so ordering is verifiable.
            chunks.append(feat.sum(dim=0, keepdim=True).expand(1, 4).clone())
        return torch.cat(chunks, dim=0)

    encoder = FakeEncoder()

    def make_ctx(imgs):
        pix = torch.cat(imgs, dim=0)
        grid = torch.tensor([[1, 2, 2] for _ in imgs], dtype=torch.int64)
        return SimpleNamespace(
            is_health_check=False,
            modality=SimpleNamespace(name="IMAGE"),
            num_items=len(imgs),
            mm_feature=pix,
            preprocess_result=SimpleNamespace(
                mm_inputs={"pixel_values": pix, "image_grid_thw": grid},
                grid_thw=grid,
                token_counts=[1] * len(imgs),
            ),
            get_feature_fn=get_feature_fn,
        )

    img1 = torch.full((4, 1), 10.0)
    img2 = torch.full((4, 1), 20.0)
    img3 = torch.full((4, 1), 30.0)

    async def fallback(*args, **kwargs):
        raise AssertionError("original should not be called when item cache applies")

    # 1. Turn 1: full miss [img1]
    out1 = await epd_encoder.cached_compute_direct_embedding(
        fallback, encoder, make_ctx([img1]), keep_on_gpu=False
    )
    assert forward_calls == [[0]]
    assert torch.equal(out1, torch.full((1, 4), 40.0))

    # 1b. Two consecutive single-item hits must not alias cached tensor storage
    # (MooncakeDelivery deregisters mm_data.embedding.data_ptr() per request).
    hit_a = await epd_encoder.cached_compute_direct_embedding(
        fallback, encoder, make_ctx([img1]), keep_on_gpu=False
    )
    hit_b = await epd_encoder.cached_compute_direct_embedding(
        fallback, encoder, make_ctx([img1]), keep_on_gpu=False
    )
    assert forward_calls == [[0]]
    assert torch.equal(hit_a, out1) and torch.equal(hit_b, out1)
    assert hit_a.data_ptr() != hit_b.data_ptr()

    # 2. Turn 2: [img1, img2] -> img1 hits cache, only idx 1 (img2) runs forward!
    out2 = await epd_encoder.cached_compute_direct_embedding(
        fallback, encoder, make_ctx([img1, img2]), keep_on_gpu=False
    )
    assert forward_calls == [[0], [1]]
    assert torch.equal(out2, torch.cat([torch.full((1, 4), 40.0), torch.full((1, 4), 80.0)]))

    # 3. Multi-request batch [img2, img1, img3] -> only idx 2 (img3) runs forward, order preserved.
    out3 = await epd_encoder.cached_compute_direct_embedding(
        fallback, encoder, make_ctx([img2, img1, img3]), keep_on_gpu=False
    )
    assert forward_calls == [[0], [1], [2]]
    assert torch.equal(
        out3,
        torch.cat([torch.full((1, 4), 80.0), torch.full((1, 4), 40.0), torch.full((1, 4), 120.0)]),
    )

    # 4. Repeat [img1, img2, img3] -> 100% cache hit, zero forward calls!
    out4 = await epd_encoder.cached_compute_direct_embedding(
        fallback, encoder, make_ctx([img1, img2, img3]), keep_on_gpu=False
    )
    assert forward_calls == [[0], [1], [2]]
    assert torch.equal(
        out4,
        torch.cat([torch.full((1, 4), 40.0), torch.full((1, 4), 80.0), torch.full((1, 4), 120.0)]),
    )

    # 5. LRU eviction test on PerItemEmbeddingCache directly.
    lru = epd_encoder.PerItemEmbeddingCache(max_bytes=32)  # holds at most two 16-byte tensors
    t_a = torch.ones(4, dtype=torch.float32)  # 16 bytes
    t_b = torch.ones(4, dtype=torch.float32) * 2
    t_c = torch.ones(4, dtype=torch.float32) * 3
    lru.put(b"a", t_a)
    lru.put(b"b", t_b)
    assert lru.get(b"a") is not None  # touch a so b becomes LRU
    lru.put(b"c", t_c)
    assert lru.get(b"b") is None
    assert lru.get(b"a") is not None
    assert lru.get(b"c") is not None

    # 6. Bypass when SMG_SGLANG_ENCODER_ITEM_CACHE_MB=0 or _has_native_item_cache=True.
    bypass_calls = []

    async def fallback_ok(enc, ctx, *, keep_on_gpu=False):
        bypass_calls.append((enc, keep_on_gpu))
        return "bypassed"

    monkeypatch.setenv("SMG_SGLANG_ENCODER_ITEM_CACHE_MB", "0")
    assert (
        await epd_encoder.cached_compute_direct_embedding(
            fallback_ok, encoder, make_ctx([img1]), keep_on_gpu=True
        )
        == "bypassed"
    )
    monkeypatch.setenv("SMG_SGLANG_ENCODER_ITEM_CACHE_MB", "1")
    encoder._has_native_item_cache = True
    assert (
        await epd_encoder.cached_compute_direct_embedding(
            fallback_ok, encoder, make_ctx([img1]), keep_on_gpu=True
        )
        == "bypassed"
    )
    assert len(bypass_calls) == 2


def test_plugin_registers_item_cache_hook_on_encoder(monkeypatch):
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
    monkeypatch.setenv(epd_plugin.ROLE_ENV, "encoder")
    epd_plugin.register()
    targets = {t for t, _, _ in registered}
    assert any(t.endswith(".MMEncoder._compute_direct_embedding") for t in targets)


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
            model_type="glm5_next",
            _embedding_dtype="bfloat16",
            server_args=SimpleNamespace(revision=revision),
            model_config=SimpleNamespace(
                revision=revision,
                hf_config=SimpleNamespace(
                    model_type="glm5_next",
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
        SimpleNamespace(model_type="glm5_next", _embedding_dtype="bfloat16"), "smg-image-grid-v1"
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
    server_mod = SimpleNamespace(meta_registry=SimpleNamespace(wait=lambda rid: meta_wait))
    runtime_mod = SimpleNamespace(send_staged_embedding=None)
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
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
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


@pytest.mark.asyncio
async def test_cached_compute_direct_embedding_reuses_materialized_item_keys(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("SMG_SGLANG_ENCODER_ITEM_CACHE_MB", "4")

    def boom(*args, **kwargs):
        raise AssertionError(
            "_compute_image_item_keys should not be called when _smg_item_keys set"
        )

    monkeypatch.setattr(epd_encoder, "_compute_image_item_keys", boom)
    pix = torch.full((4, 1), 3.0)
    grid = torch.tensor([[1, 2, 2]], dtype=torch.int64)
    encoder = SimpleNamespace(
        tp_size=1,
        _build_model_mm_items=lambda feat, res, indices, mod: [(0, feat)],
    )
    ctx = SimpleNamespace(
        is_health_check=False,
        modality=SimpleNamespace(name="IMAGE"),
        num_items=1,
        mm_feature=pix,
        preprocess_result=SimpleNamespace(
            mm_inputs={"pixel_values": pix, "image_grid_thw": grid},
            grid_thw=grid,
            token_counts=[1],
            _smg_item_keys=[b"k" * 32],
        ),
        get_feature_fn=lambda items: torch.full((1, 4), 12.0),
    )

    async def fallback(*args, **kwargs):
        raise AssertionError("original should not run")

    out1 = await epd_encoder.cached_compute_direct_embedding(fallback, encoder, ctx)
    out2 = await epd_encoder.cached_compute_direct_embedding(fallback, encoder, ctx)
    assert torch.equal(out1, torch.full((1, 4), 12.0))
    assert torch.equal(out2, out1)
    assert encoder._smg_item_cache.hits == 1
    assert encoder._smg_item_cache.misses == 1


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


def test_materialize_compact_v2_inline_and_shm_numerical_equivalence(monkeypatch):
    """Compact v2 (inline + /dev/shm) matches legacy proto numerically without Python pixel SHA-256."""
    torch = pytest.importorskip("torch")
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.disaggregation.encoder.preprocessor",
        SimpleNamespace(EncoderPreprocessResult=SimpleNamespace),
    )
    from smg_grpc_proto.generated import sglang_scheduler_pb2 as sg

    def forbid_full_tensor_hash(*args, **kwargs):
        raise AssertionError("_hash_materialized_item must not run on compact v2 artifacts")

    monkeypatch.setattr(epd_encoder, "_hash_materialized_item", forbid_full_tensor_hash)

    vals_a = (1.0, 2.0, 3.0, 4.0)
    vals_b = (5.0, 6.0, 7.0, 8.0)
    bytes_a = struct.pack("4f", *vals_a)
    bytes_b = struct.pack("4f", *vals_b)
    hash_a = "11" * 32
    hash_b = "22" * 32
    ns = epd_encoder._build_stable_identity_namespace(
        SimpleNamespace(model_type="qwen3_vl", _embedding_dtype="bfloat16"), "smg-image-grid-v1"
    )

    # Legacy reference tensor values.
    legacy_art = _make_single_image_artifact(sg, vals_a)
    monkeypatch.undo()
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.disaggregation.encoder.preprocessor",
        SimpleNamespace(EncoderPreprocessResult=SimpleNamespace),
    )
    legacy_res, _ = epd_encoder.RustPreprocessorBridge(
        None, 4096, stable_identity_ns=ns
    )._materialize_batch(
        [{"req_id": "leg", "mm_items": [{epd_encoder.ARTIFACT_KEY: legacy_art}]}],
        SimpleNamespace(name="IMAGE"),
    )
    monkeypatch.setattr(epd_encoder, "_hash_materialized_item", forbid_full_tensor_hash)

    # Compact v2 inline single-item.
    sink_inline = {}
    bridge_inline = epd_encoder.RustPreprocessorBridge(
        None, 4096, stable_identity_ns=ns, identity_sink=sink_inline, tp_size=1
    )
    inline_art = {
        "kind": "compact",
        "shm_name": None,
        "shm_nbytes": 0,
        "inline_pixels": bytes_a,
        "batches": [
            {
                "pixel_dtype": "float32",
                "pixel_shape": [4, 1],
                "pixel_offset": 0,
                "pixel_nbytes": len(bytes_a),
                "grid_thw": [[1, 2, 2]],
                "placeholder_lengths": [1],
                "item_hashes": [hash_a],
            }
        ],
    }
    res_inline, counts_inline = bridge_inline._materialize_batch(
        [{"req_id": "inl-1", "mm_items": [{epd_encoder.ARTIFACT_V2_KEY: inline_art}]}],
        SimpleNamespace(name="IMAGE"),
    )
    assert counts_inline == [1]
    assert torch.equal(res_inline.mm_inputs["pixel_values"], legacy_res.mm_inputs["pixel_values"])
    assert torch.equal(res_inline.grid_thw, legacy_res.grid_thw)
    assert res_inline.token_counts == [1]
    assert len(sink_inline["inl-1"]) == 1 and len(sink_inline["inl-1"][0]) == 64

    # Compact v2 /dev/shm single-item: zero-copy COW writable + auto-unlink at tp_size=1.
    shm_name = f"smg-epd-pytest-{os.getpid()}-single"
    shm_path = os.path.join("/dev/shm", shm_name)
    with open(shm_path, "wb") as f:
        f.write(bytes_a)
    try:
        sink_shm = {}
        bridge_shm = epd_encoder.RustPreprocessorBridge(
            None, 4096, stable_identity_ns=ns, identity_sink=sink_shm, tp_size=1
        )
        shm_art = {
            "kind": "compact",
            "shm_name": shm_name,
            "shm_nbytes": len(bytes_a),
            "batches": [
                {
                    "pixel_dtype": "float32",
                    "pixel_shape": [4, 1],
                    "pixel_offset": 0,
                    "pixel_nbytes": len(bytes_a),
                    "grid_thw": [[1, 2, 2]],
                    "placeholder_lengths": [1],
                    "item_hashes": [hash_a],
                }
            ],
        }
        res_shm, _ = bridge_shm._materialize_batch(
            [{"req_id": "shm-1", "mm_items": [{epd_encoder.ARTIFACT_V2_KEY: shm_art}]}],
            SimpleNamespace(name="IMAGE"),
        )
        assert not os.path.exists(shm_path), "tp_size=1 must unlink SHM immediately after mmap"
        assert torch.equal(res_shm.mm_inputs["pixel_values"], legacy_res.mm_inputs["pixel_values"])
        assert sink_shm["shm-1"] == sink_inline["inl-1"]
        assert res_shm._smg_item_keys == res_inline._smg_item_keys
        # Writable COW storage survives GC without error.
        gc.collect()
        res_shm.mm_inputs["pixel_values"][0, 0] = 99.0
        assert float(res_shm.mm_inputs["pixel_values"][0, 0]) == 99.0
    finally:
        if os.path.exists(shm_path):
            os.unlink(shm_path)

    # Multi-item batch with different Rust hashes produces distinct keys; same hash matches solo.
    sink_multi = {}
    bridge_multi = epd_encoder.RustPreprocessorBridge(
        None, 4096, stable_identity_ns=ns, identity_sink=sink_multi, tp_size=1
    )
    multi_art = {
        "kind": "compact",
        "shm_name": None,
        "shm_nbytes": 0,
        "inline_pixels": bytes_b + bytes_a,
        "batches": [
            {
                "pixel_dtype": "float32",
                "pixel_shape": [8, 1],
                "pixel_offset": 0,
                "pixel_nbytes": len(bytes_b) + len(bytes_a),
                "grid_thw": [[1, 2, 2], [1, 2, 2]],
                "placeholder_lengths": [1, 1],
                "item_hashes": [hash_b, hash_a],
            }
        ],
    }
    res_multi, counts_multi = bridge_multi._materialize_batch(
        [{"req_id": "mul-1", "mm_items": [{epd_encoder.ARTIFACT_V2_KEY: multi_art}, None]}],
        SimpleNamespace(name="IMAGE"),
    )
    assert counts_multi == [2]
    assert sink_multi["mul-1"][1] == sink_inline["inl-1"][0]
    assert sink_multi["mul-1"][0] != sink_multi["mul-1"][1]
    assert torch.equal(
        res_multi.mm_inputs["pixel_values"].flatten(),
        torch.tensor([*vals_b, *vals_a], dtype=torch.float32),
    )


@pytest.mark.asyncio
async def test_early_cache_hit_mixed_batch_and_eviction_race(monkeypatch):
    """Early cache_hit pins embeddings, fuses with compact misses, and reports evictions cleanly."""
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("SMG_SGLANG_ENCODER_ITEM_CACHE_MB", "4")
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.disaggregation.encoder.preprocessor",
        SimpleNamespace(EncoderPreprocessResult=SimpleNamespace),
    )

    ns = epd_encoder._build_stable_identity_namespace(
        SimpleNamespace(model_type="qwen3_vl", _embedding_dtype="bfloat16"), "smg-image-grid-v1"
    )
    forward_calls = []

    class FakeEncoder(_FakeEncoder):
        tp_size = 1
        _has_native_item_cache = True
        _embedding_dims = {"img": 4}

        def __init__(self):
            super().__init__()
            self._smg_item_cache = epd_encoder.PerItemEmbeddingCache(4 * 1024 * 1024)
            self._smg_req_item_identities = {}
            self.preprocessor = SimpleNamespace(
                get_num_patches=lambda row, _mod: int(row[0]) * int(row[1]) * int(row[2])
            )

        def _build_model_mm_items(self, mm_feature, preprocess_result, indices, modality):
            grid = preprocess_result.grid_thw
            offsets = [0]
            cur = 0
            for r in grid.tolist():
                cur += self.preprocessor.get_num_patches(r, modality)
                offsets.append(cur)
            return [(idx, mm_feature[offsets[idx] : offsets[idx + 1]]) for idx in indices]

    enc = FakeEncoder()
    key_a = epd_encoder._derive_item_key_from_rust_hash(ns, "IMAGE", "aa" * 32, [1, 2, 2])
    key_a_hex = key_a.hex()
    emb_a = torch.full((1, 4), 42.0)
    enc._smg_item_cache.put(key_a, emb_a)

    # 1. Mixed batch: [Req0 (cache_hit, no pixels), Req1 (compact miss, has pixels)].
    sink = {}
    bridge = epd_encoder.RustPreprocessorBridge(
        enc.preprocessor, 4096, stable_identity_ns=ns, identity_sink=sink, tp_size=1
    )
    bytes_b = struct.pack("4f", 1.0, 2.0, 3.0, 4.0)
    req_hit = {
        "req_id": "r-hit",
        "mm_items": [
            {
                epd_encoder.ARTIFACT_V2_KEY: {
                    "kind": "cache_hit",
                    "item_keys": [key_a_hex],
                    "grid_thw": [[1, 2, 2]],
                    "token_counts": [1],
                    "pinned_embeddings": [emb_a],
                }
            }
        ],
    }
    req_miss = {
        "req_id": "r-miss",
        "mm_items": [
            {
                epd_encoder.ARTIFACT_V2_KEY: {
                    "kind": "compact",
                    "shm_name": None,
                    "shm_nbytes": 0,
                    "inline_pixels": bytes_b,
                    "batches": [
                        {
                            "pixel_dtype": "float32",
                            "pixel_shape": [4, 1],
                            "pixel_offset": 0,
                            "pixel_nbytes": len(bytes_b),
                            "grid_thw": [[1, 2, 2]],
                            "placeholder_lengths": [1],
                            "item_hashes": ["bb" * 32],
                        }
                    ],
                }
            }
        ],
    }
    preproc, items_per_req = bridge._materialize_batch(
        [req_hit, req_miss], SimpleNamespace(name="IMAGE")
    )
    assert items_per_req == [1, 1]
    assert sink["r-hit"] == [key_a_hex]
    assert getattr(preproc, "_smg_pixel_item_indices") == [1]

    def get_feature_fn(mm_items):
        forward_calls.append(len(mm_items))
        return torch.cat(
            [feat.sum(dim=0, keepdim=True).expand(1, 4).clone() for _, feat in mm_items]
        )

    ctx = SimpleNamespace(
        is_health_check=False,
        modality=SimpleNamespace(name="IMAGE"),
        num_items=2,
        mm_feature=preproc.mm_inputs["pixel_values"],
        preprocess_result=preproc,
        get_feature_fn=get_feature_fn,
    )

    async def fallback(*args, **kwargs):
        raise AssertionError("original must not run when _smg_item_keys is set")

    out = await epd_encoder.cached_compute_direct_embedding(fallback, enc, ctx, keep_on_gpu=False)
    assert forward_calls == [1], "only the miss item should execute ViT forward"
    assert torch.equal(out[0:1], torch.full((1, 4), 42.0))
    assert torch.equal(out[1:2], torch.full((1, 4), 10.0))

    # 2. All-hit batch touches zero pixels and zero forward calls.
    preproc_all_hit, _ = bridge._materialize_batch([req_hit], SimpleNamespace(name="IMAGE"))
    ctx_hit = SimpleNamespace(
        is_health_check=False,
        modality=SimpleNamespace(name="IMAGE"),
        num_items=1,
        mm_feature=preproc_all_hit.mm_inputs["pixel_values"],
        preprocess_result=preproc_all_hit,
        get_feature_fn=lambda _: (_ for _ in ()).throw(AssertionError("no forward on all-hit")),
    )
    out_hit = await epd_encoder.cached_compute_direct_embedding(
        fallback, enc, ctx_hit, keep_on_gpu=False
    )
    assert torch.equal(out_hit, torch.full((1, 4), 42.0))
    assert out_hit.data_ptr() != emb_a.data_ptr()

    # 3. reuse_materialized_feature_hashes hook returns _smg_item_keys without calling original.
    reused = epd_plugin.reuse_materialized_feature_hashes(
        lambda *a: (_ for _ in ()).throw(AssertionError("must not call original")),
        enc,
        preproc.mm_inputs["pixel_values"],
        preproc,
        SimpleNamespace(name="IMAGE"),
    )
    assert reused == preproc._smg_item_keys

    # 4. NativeEncoder.execute returns cache_miss=True without registering entry when evicted.
    server_mod = SimpleNamespace(
        meta_registry=SimpleNamespace(publish=lambda *a, **kw: asyncio.sleep(0))
    )
    runtime_mod = SimpleNamespace(
        execute_encode_pipeline=lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("pipeline must not run on cache_miss")
        )
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
        ("sglang.srt.managers.req_types", SimpleNamespace(Modality=SimpleNamespace(IMAGE="img"))),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    owner = epd_encoder.NativeEncoder(
        {"max_receivers": 2, "max_staging_bytes_per_request": 4096},
        SimpleNamespace(encoder=enc, scheduler=SimpleNamespace()),
    )
    missing_hex = "ff" * 32
    miss_receipt = await owner.execute(
        {
            "rid": "r-evicted",
            "prefill_host": "10.0.0.1",
            "remaining_ms": 5000,
            "embedding_tokens": 1,
            "item_count": 1,
            "modality": "image",
            "artifact_v2": {
                "kind": "cache_hit",
                "item_keys": [missing_hex],
                "grid_thw": [[1, 2, 2]],
                "token_counts": [1],
            },
        },
        b"",
    )
    assert miss_receipt["released"] is True
    assert miss_receipt["cache_miss"] is True
    assert missing_hex in miss_receipt["cache_evicted"]
    assert "r-evicted" not in owner.entries

    # 5. PerItemEmbeddingCache drain_evictions tracks evicted hex keys accurately.
    small = epd_encoder.PerItemEmbeddingCache(max_bytes=16)
    k1, k2 = b"\x01" * 32, b"\x02" * 32
    small.put(k1, torch.ones(4, dtype=torch.float32))
    assert small.drain_evictions() == []
    small.put(k2, torch.ones(4, dtype=torch.float32) * 2)
    assert small.drain_evictions() == [k1.hex()]
    assert small.drain_evictions() == []


def test_plugin_registers_feature_hash_reuse_hook(monkeypatch):
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
    monkeypatch.setenv(epd_plugin.ROLE_ENV, "encoder")
    epd_plugin.register()
    targets = {t for t, _, _ in registered}
    assert any(t.endswith(".MMEncoder._calculate_hashes_from_features") for t in targets)


def test_item_cache_budget_env_defaults_disable_and_strict_validation(monkeypatch):
    monkeypatch.delenv("SMG_SGLANG_ENCODER_ITEM_CACHE_MB", raising=False)
    monkeypatch.delenv("SGLANG_VLM_CACHE_SIZE_MB", raising=False)
    monkeypatch.setattr(epd_launcher, "available_memory_bytes", lambda: 256 * 1024**3)
    assert epd_encoder._item_cache_budget_bytes() == 16384 * 1024 * 1024

    # Container-safe scaling on smaller cgroup RAM limits (1/16 of available RAM = 1/8 of context).
    monkeypatch.setattr(epd_launcher, "available_memory_bytes", lambda: 8 * 1024**3)
    assert epd_encoder._item_cache_budget_bytes() == 512 * 1024**2

    # Falls back to SGLANG_VLM_CACHE_SIZE_MB when SMG_SGLANG_ENCODER_ITEM_CACHE_MB is unset.
    monkeypatch.setenv("SGLANG_VLM_CACHE_SIZE_MB", "0")
    assert epd_encoder._item_cache_budget_bytes() == 0
    monkeypatch.setenv("SGLANG_VLM_CACHE_SIZE_MB", "256")
    assert epd_encoder._item_cache_budget_bytes() == 256 * 1024 * 1024
    monkeypatch.delenv("SGLANG_VLM_CACHE_SIZE_MB", raising=False)

    # Explicit SMG_SGLANG_ENCODER_ITEM_CACHE_MB overrides and validates strictly.
    monkeypatch.setenv("SMG_SGLANG_ENCODER_ITEM_CACHE_MB", "0")
    assert epd_encoder._item_cache_budget_bytes() == 0
    monkeypatch.setenv("SMG_SGLANG_ENCODER_ITEM_CACHE_MB", "512")
    assert epd_encoder._item_cache_budget_bytes() == 512 * 1024 * 1024
    for bad in ("-32", "not-a-number"):
        monkeypatch.setenv("SMG_SGLANG_ENCODER_ITEM_CACHE_MB", bad)
        with pytest.raises(ValueError, match="SMG_SGLANG_ENCODER_ITEM_CACHE_MB"):
            epd_encoder._item_cache_budget_bytes()


@pytest.mark.asyncio
async def test_partial_pixel_cache_hit_with_invalid_keys_raises_instead_of_fallback(monkeypatch):
    """Never fall through to original() when pinned_embs is present on a partial-pixel batch."""
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    preproc = SimpleNamespace(
        token_counts=[4, 4],
        _smg_item_keys=[b"only-one"],
        _smg_pinned_embeddings=[object(), None],
    )
    ctx = SimpleNamespace(
        is_health_check=False,
        modality=SimpleNamespace(name="IMAGE"),
        num_items=2,
        preprocess_result=preproc,
    )
    encoder = SimpleNamespace(tp_size=1)
    called = []

    async def original(*_a, **_kw):
        called.append(True)
        return "should-not-run"

    with pytest.raises(ValueError, match="missing valid per-item keys"):
        await epd_encoder.cached_compute_direct_embedding(original, encoder, ctx)
    assert not called


def test_explicit_large_context_length_still_fits_within_memory_budget():
    """fit_auto_media_tokens bounds explicit --context-length and reserves item_cache_bytes."""
    qwen3_cfg = SimpleNamespace(
        vision_config={
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "in_chans": 3,
        }
    )
    cfg = {
        "max_active": 64,
        "max_control_bytes": 256 * 1024**2,
        "max_receivers": 8,
        "embedding_backend": "mooncake",
        "max_context_bytes": 24 * 1024**3,
        "item_cache_bytes": 4 * 1024**3,
    }
    target = epd_launcher.resolve_max_media_tokens(SimpleNamespace(context_length=262144))
    fitted = epd_launcher.fit_auto_media_tokens(
        target, qwen3_cfg, "bfloat16", 4096, "bfloat16", cfg, max_batch_size=8
    )
    cfg["max_artifact_bytes"] = epd_launcher.encoder_artifact_bytes(
        qwen3_cfg, "bfloat16", max_tokens=fitted
    )
    cfg["max_staging_bytes_per_request"] = epd_launcher.encoder_staging_bytes(
        4096, "bfloat16", max_tokens=fitted
    )
    cap, _ = epd_launcher.encoder_window(cfg, max_batch_size=8)
    assert fitted <= target and cap >= 8


def test_encoder_initialized_uses_derived_context_bytes_without_handoff_envs(monkeypatch):
    monkeypatch.setenv("SMG_SGLANG_ENCODER_ITEM_CACHE_MB", "0")
    monkeypatch.setattr(epd_launcher, "resolve_max_context_bytes", lambda: 96 * 1024**3)
    encoder = SimpleNamespace(
        rank=0,
        tp_size=2,
        cache_namespace="sha256:" + "b" * 64,
        preprocessor=SimpleNamespace(
            process_batch_mm_items=lambda *a, **k: None,
            get_num_patches=lambda *a, **k: 1,
            supports_modality=lambda *a, **k: True,
            preproc_executor=SimpleNamespace(submit=lambda *a, **k: None),
        ),
    )
    epd_plugin.encoder_initialized(None, encoder)
    assert isinstance(encoder.preprocessor, epd_encoder.RustPreprocessorBridge)
    assert encoder.preprocessor.max_artifact_bytes == 96 * 1024**3
    assert encoder.preprocessor.tp_size == 2


def test_stable_identity_namespace_uses_model_dtype_not_mm_dtype(monkeypatch):
    monkeypatch.setenv("SMG_SGLANG_MM_DTYPE", "float32")
    enc_a = SimpleNamespace(
        model_type="qwen3_vl",
        model_config=SimpleNamespace(dtype="bfloat16", revision="r1", hf_config=None),
    )
    ns_a = epd_encoder._build_stable_identity_namespace(enc_a, "smg-image-grid-v1")
    monkeypatch.setenv("SMG_SGLANG_MM_DTYPE", "float16")
    ns_b = epd_encoder._build_stable_identity_namespace(enc_a, "smg-image-grid-v1")
    assert ns_a == ns_b


@pytest.mark.asyncio
async def test_serve_uses_mm_dtype_mm_max_inflight_and_zmq_socket_dir(monkeypatch, tmp_path):
    from smg_grpc_servicer.rust_lifecycle import ServicerConfigError
    from smg_grpc_servicer.sglang import model_artifacts

    captured = {}

    class FakeServer:
        def __init__(self, cfg_json, proc_opts, tok_dir, max_art_bytes):
            captured["config"] = json.loads(cfg_json)
            captured["processor_options"] = dict(proc_opts)
            captured["max_artifact_bytes"] = max_art_bytes

        def drain(self):
            pass

        def active(self):
            return 0

        def running(self):
            return True

        def last_error(self):
            return None

        def stop(self, timeout):
            pass

    backend = SimpleNamespace(
        encoder=SimpleNamespace(
            preprocessor=SimpleNamespace(max_artifact_bytes=epd_launcher.MAX_CONTEXT_BYTES),
            model_config=SimpleNamespace(
                hidden_size=4096,
                hf_config=SimpleNamespace(
                    model_type="qwen3_vl",
                    vision_config={
                        "patch_size": 16,
                        "temporal_patch_size": 2,
                        "spatial_merge_size": 2,
                        "in_chans": 3,
                    },
                ),
            ),
            model=SimpleNamespace(visual=SimpleNamespace(dtype="bfloat16")),
        ),
        scheduler=SimpleNamespace(max_batch_size=8),
        tp_processes=[],
        start=lambda: None,
        stop=lambda: asyncio.sleep(0),
    )

    monkeypatch.setitem(sys.modules, "smg.smg_rs", SimpleNamespace(ManagedEncoderServer=FakeServer))
    monkeypatch.setattr(model_artifacts, "encoder_artifact_dirs", lambda args: ("/m", "/t"))
    monkeypatch.setattr(epd_encoder, "launch_runtime", lambda args, max_active: backend)
    monkeypatch.setattr(epd_encoder, "serve_native", lambda cfg, b: asyncio.sleep(0))

    socket_root = tmp_path / "epd-sockets"
    monkeypatch.setenv("SMG_ZMQ_SOCKET_DIR", str(socket_root))
    monkeypatch.setenv("SMG_SGLANG_MM_DTYPE", "float32")
    monkeypatch.setenv("SMG_SGLANG_MM_MAX_INFLIGHT", "12")

    args = SimpleNamespace(model_path="m", served_model_name=None, host="127.0.0.1", port=30000)
    config = dict(
        max_active=256,
        max_control_bytes=256 * 1024**2,
        max_artifact_bytes=388 * 1024**2,
        max_receivers=8,
        embedding_dtype="bfloat16",
        embedding_backend="mooncake",
        processor_options={"model_id": "m"},
        shutdown_timeout_seconds=30.0,
    )
    await epd_launcher.serve(config, args)

    assert captured["processor_options"]["encoder_dtype"] == "float32"
    assert captured["processor_options"]["max_inflight"] == 12
    assert captured["config"]["max_cpu"] == 12
    assert captured["max_artifact_bytes"] == 772 * 1024**2
    assert captured["config"]["native_commands"].startswith(f"ipc://{socket_root}/smg-encoder-")
    assert captured["config"]["native_events"].startswith(f"ipc://{socket_root}/smg-encoder-")
    # TemporaryDirectory cleans up the per-run subdir while socket_root stays created.
    assert socket_root.is_dir()
    assert list(socket_root.iterdir()) == []

    # Invalid SMG_SGLANG_MM_DTYPE is rejected before server bind.
    monkeypatch.setenv("SMG_SGLANG_MM_DTYPE", "fp8")
    with pytest.raises(ValueError, match="SMG_SGLANG_MM_DTYPE"):
        await epd_launcher.serve(dict(config, processor_options={"model_id": "m"}), args)

    # Regular file as SMG_ZMQ_SOCKET_DIR raises ServicerConfigError.
    monkeypatch.delenv("SMG_SGLANG_MM_DTYPE")
    bad_socket_path = tmp_path / "not-a-dir"
    bad_socket_path.write_text("x")
    monkeypatch.setenv("SMG_ZMQ_SOCKET_DIR", str(bad_socket_path))
    with pytest.raises(ServicerConfigError, match="SMG_ZMQ_SOCKET_DIR"):
        await epd_launcher.serve(dict(config, processor_options={"model_id": "m"}), args)


def test_run_server_uses_zmq_socket_dir_and_clears_stale_events_for_decode(monkeypatch, tmp_path):
    from smg_grpc_servicer.sglang import rust

    seen = []

    class FakeLauncher:
        def __init__(self, cfg):
            self.cfg = cfg

        async def serve(self, args):
            seen.append(
                (
                    self.cfg["role"],
                    self.cfg["metadata_endpoint"],
                    os.environ.get(epd_plugin.EVENTS_ENV),
                )
            )
            return 0

    monkeypatch.setattr(rust, "resolve_servicer_impl", lambda args: "rust")
    monkeypatch.setattr(epd_launcher, "launcher", FakeLauncher)
    monkeypatch.setattr(epd_plugin, "load_role_hooks", lambda: None)

    socket_root = tmp_path / "pd-sockets"
    monkeypatch.setenv("SMG_ZMQ_SOCKET_DIR", str(socket_root))

    args = SimpleNamespace(
        encoder_only=False,
        disaggregation_mode="prefill",
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
    with pytest.raises(SystemExit) as exc:
        epd_launcher.run_server(args)
    assert exc.value.code == 0
    role, endpoint, env_ep = seen[-1]
    assert role == "prefill"
    assert endpoint.startswith(f"ipc://{socket_root}/smg-managed-")
    assert env_ep == endpoint
    assert list(socket_root.iterdir()) == []

    # Decode clears any stale EVENTS_ENV left in the environment.
    args.disaggregation_mode = "decode"
    with pytest.raises(SystemExit) as exc:
        epd_launcher.run_server(args)
    assert exc.value.code == 0
    assert seen[-1] == ("decode", None, None)


def test_available_memory_and_context_bytes_cgroup_v2_v1_and_host_fallback(monkeypatch, tmp_path):
    gib = 1024**3
    v2_path = tmp_path / "memory.max"
    v1_path = tmp_path / "memory.limit_in_bytes"
    meminfo_path = tmp_path / "meminfo"

    monkeypatch.setattr(epd_launcher, "_CGROUP_V2_MEMORY_MAX", v2_path)
    monkeypatch.setattr(epd_launcher, "_CGROUP_V1_MEMORY_LIMIT", v1_path)
    monkeypatch.setattr(epd_launcher, "_PROC_MEMINFO", meminfo_path)
    monkeypatch.setattr(epd_launcher, "_host_memory_bytes", lambda: 1024 * gib)

    # 1. cgroup v2 explicit limit (e.g. GLM-5.4 8e4p2d 928 GiB pod).
    v2_path.write_text(f"{928 * gib}\n")
    assert epd_launcher.available_memory_bytes() == 928 * gib
    assert epd_launcher.resolve_max_context_bytes() == 464 * gib

    # Small container limit never exceeds container RAM.
    v2_path.write_text(f"{16 * gib}\n")
    assert epd_launcher.available_memory_bytes() == 16 * gib
    assert epd_launcher.resolve_max_context_bytes() == 8 * gib

    # cgroup v2 limit larger than physical host RAM is capped at host RAM.
    v2_path.write_text(f"{2048 * gib}\n")
    assert epd_launcher.available_memory_bytes() == 1024 * gib
    assert epd_launcher.resolve_max_context_bytes() == 512 * gib

    # 2. cgroup v2 "max" (e.g. GLM-5.4 8e1p1d) falls back to host RAM even if a v1 file exists.
    v2_path.write_text("max\n")
    v1_path.write_text(f"{64 * gib}\n")
    assert epd_launcher.available_memory_bytes() == 1024 * gib
    assert epd_launcher.resolve_max_context_bytes() == 512 * gib

    # Malformed or non-positive cgroup v2 value falls back to host RAM.
    for bad_v2 in ("", "0\n", "-100\n", "not-a-number\n"):
        v2_path.write_text(bad_v2)
        assert epd_launcher.available_memory_bytes() == 1024 * gib

    # 3. cgroup v1 fallback when cgroup v2 file is absent.
    v2_path.unlink()
    v1_path.write_text(f"{128 * gib}\n")
    assert epd_launcher.available_memory_bytes() == 128 * gib
    assert epd_launcher.resolve_max_context_bytes() == 64 * gib

    # cgroup v1 unlimited sentinel (~2^63-4096) falls back to host RAM.
    v1_path.write_text(f"{(1 << 63) - 4096}\n")
    assert epd_launcher.available_memory_bytes() == 1024 * gib

    # cgroup v1 limit larger than host RAM is capped at host RAM.
    v1_path.write_text(f"{2048 * gib}\n")
    assert epd_launcher.available_memory_bytes() == 1024 * gib

    # 4. Neither cgroup file present -> falls back to host RAM via /proc/meminfo when sysconf fails.
    v1_path.unlink()
    monkeypatch.undo()
    monkeypatch.setattr(epd_launcher, "_CGROUP_V2_MEMORY_MAX", v2_path)
    monkeypatch.setattr(epd_launcher, "_CGROUP_V1_MEMORY_LIMIT", v1_path)
    monkeypatch.setattr(epd_launcher, "_PROC_MEMINFO", meminfo_path)

    def broken_sysconf(_name):
        raise OSError("unavailable")

    monkeypatch.setattr(os, "sysconf", broken_sysconf)
    meminfo_path.write_text("MemTotal:       67108864 kB\nMemFree:        1024 kB\n")
    assert epd_launcher.available_memory_bytes() == 64 * gib
    assert epd_launcher.resolve_max_context_bytes() == 32 * gib

    # 5. Ultimate fallback to MAX_CONTEXT_BYTES (48 GiB) when all detection fails.
    meminfo_path.unlink()
    assert epd_launcher.available_memory_bytes() is None
    assert epd_launcher.resolve_max_context_bytes() == epd_launcher.MAX_CONTEXT_BYTES


def test_resolve_max_media_tokens_cli_model_config_and_clamping():
    # Default fallback when neither CLI nor model_config provides a context length.
    assert epd_launcher.resolve_max_media_tokens() == 32768
    assert (
        epd_launcher.resolve_max_media_tokens(SimpleNamespace(context_length=None), None) == 32768
    )

    # Model config context_len is clamped to [32768, 262144].
    assert epd_launcher.resolve_max_media_tokens(None, SimpleNamespace(context_len=8192)) == 32768
    assert (
        epd_launcher.resolve_max_media_tokens(None, SimpleNamespace(context_len=202752)) == 202752
    )
    assert (
        epd_launcher.resolve_max_media_tokens(None, SimpleNamespace(context_len=262144)) == 262144
    )
    assert (
        epd_launcher.resolve_max_media_tokens(None, SimpleNamespace(context_len=1013760)) == 262144
    )
    # Non-positive or non-int model_config.context_len falls back to floor.
    for bad_model_ctx in (0, -1, True, 65536.0, "262144"):
        assert (
            epd_launcher.resolve_max_media_tokens(None, SimpleNamespace(context_len=bad_model_ctx))
            == 32768
        )

    # Explicit --context-length overrides model_config.context_len and clamps to [32768, 262144].
    assert (
        epd_launcher.resolve_max_media_tokens(
            SimpleNamespace(context_length=262144), SimpleNamespace(context_len=202752)
        )
        == 262144
    )
    assert (
        epd_launcher.resolve_max_media_tokens(
            SimpleNamespace(context_length=1048576), SimpleNamespace(context_len=202752)
        )
        == 262144
    )
    assert (
        epd_launcher.resolve_max_media_tokens(
            SimpleNamespace(context_length=65536), SimpleNamespace(context_len=262144)
        )
        == 65536
    )
    assert (
        epd_launcher.resolve_max_media_tokens(
            SimpleNamespace(context_length=16384), SimpleNamespace(context_len=262144)
        )
        == 32768
    )

    # Invalid CLI --context-length fails fast in resolve_max_media_tokens and complete().
    base_args = SimpleNamespace(
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
    for invalid_ctx in (0, -8, True, 1.5, "not-an-int"):
        base_args.context_length = invalid_ctx
        with pytest.raises(ValueError, match="--context-length"):
            epd_launcher.resolve_max_media_tokens(base_args)
        with pytest.raises(ValueError, match="--context-length"):
            epd_launcher.complete(base_args, "encoder")


def test_auto_budget_glm54_and_qwen3vl_window_stability():
    gib = 1024**3
    video_1080p_60s_bytes = int(1.415 * gib)
    glm54_hf = SimpleNamespace(
        vision_config={
            "patch_size": 14,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "in_channels": 3,
        }
    )
    qwen3_hf = SimpleNamespace(
        vision_config={
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "in_chans": 3,
        }
    )

    def make_cfg(ram_gib, backend="mooncake"):
        return {
            "max_active": 256,
            "max_control_bytes": 256 * 1024**2,
            "max_receivers": 8,
            "max_context_bytes": (ram_gib * gib) // 2,
            "embedding_backend": backend,
            "embedding_dtype": "bfloat16",
        }

    # 1. GLM-5.4 8e4p2d (928 GiB cgroup) & 8e1p1d (1024 GiB host RAM):
    # both 202752 native context and 262144 ceiling admit 1080p/60s video (~1.415 GiB) with 8 slots.
    for ram_gib in (928, 1024):
        for ctx_len, expected_tokens in ((1013760, 262144), (202752, 202752)):
            cfg = make_cfg(ram_gib, "mooncake")
            target = epd_launcher.resolve_max_media_tokens(
                None, SimpleNamespace(context_len=ctx_len)
            )
            tokens = epd_launcher.fit_auto_media_tokens(
                target, glm54_hf, "bfloat16", 6144, "bfloat16", cfg, max_batch_size=8
            )
            assert tokens == expected_tokens
            cfg["max_artifact_bytes"] = epd_launcher.encoder_artifact_bytes(
                glm54_hf, "bfloat16", max_tokens=tokens
            )
            cfg["max_staging_bytes_per_request"] = epd_launcher.encoder_staging_bytes(
                6144, "bfloat16", max_tokens=tokens
            )
            assert cfg["max_artifact_bytes"] > video_1080p_60s_bytes
            if expected_tokens == 262144:
                assert cfg["max_artifact_bytes"] == 262144 * 9408 + 4 * 1024**2
            cap, exec_bytes = epd_launcher.encoder_window(cfg, max_batch_size=8)
            assert cap == 8
            assert cap * exec_bytes <= cfg["max_context_bytes"]

    # 2. Qwen3-VL-8B & Qwen3-VL-32B: full 262144 tokens on H200 / 256 GiB+ containers,
    # stable 8-slot execution_capacity on 32..128 GiB containers, and safe non-zero cap on 16 GiB.
    for hidden_size in (4096, 5120):
        for ram_gib, backend, expect_full_tokens, expected_cap in (
            (2048, "mooncake", True, 8),
            (1024, "mooncake", True, 8),
            (1024, "zmq_to_scheduler", True, 8),
            (256, "mooncake", True, 8),
            (128, "mooncake", False, 8),
            (64, "mooncake", False, 8),
            (32, "mooncake", False, 8),
            (16, "mooncake", False, 5),
        ):
            cfg = make_cfg(ram_gib, backend)
            target = epd_launcher.resolve_max_media_tokens(
                None, SimpleNamespace(context_len=262144)
            )
            tokens = epd_launcher.fit_auto_media_tokens(
                target, qwen3_hf, "bfloat16", hidden_size, "bfloat16", cfg, max_batch_size=8
            )
            if expect_full_tokens:
                assert tokens == 262144
            elif ram_gib >= 32:
                assert 32768 < tokens < 262144
            else:
                assert tokens == 32768
            cfg["max_artifact_bytes"] = epd_launcher.encoder_artifact_bytes(
                qwen3_hf, "bfloat16", max_tokens=tokens
            )
            cfg["max_staging_bytes_per_request"] = epd_launcher.encoder_staging_bytes(
                hidden_size, "bfloat16", max_tokens=tokens
            )
            cap, exec_bytes = epd_launcher.encoder_window(cfg, max_batch_size=8)
            assert cap == expected_cap
            assert cap * exec_bytes <= cfg["max_context_bytes"]


@pytest.mark.asyncio
async def test_serve_auto_derives_budgets_from_cgroup_and_context_length(monkeypatch, tmp_path):
    from smg_grpc_servicer.sglang import model_artifacts

    gib = 1024**3
    captured = {}

    class FakeServer:
        def __init__(self, cfg_json, proc_opts, tok_dir, max_art_bytes):
            captured["config"] = json.loads(cfg_json)
            captured["max_artifact_bytes"] = max_art_bytes

        def drain(self):
            pass

        def active(self):
            return 0

        def running(self):
            return True

        def last_error(self):
            return None

        def stop(self, timeout):
            pass

    monkeypatch.setitem(sys.modules, "smg.smg_rs", SimpleNamespace(ManagedEncoderServer=FakeServer))
    monkeypatch.setattr(model_artifacts, "encoder_artifact_dirs", lambda args: ("/m", "/t"))
    monkeypatch.setattr(epd_encoder, "serve_native", lambda cfg, b: asyncio.sleep(0))
    monkeypatch.delenv("SMG_SGLANG_MM_DTYPE", raising=False)
    monkeypatch.delenv("SMG_SGLANG_MM_MAX_INFLIGHT", raising=False)
    monkeypatch.setenv("SMG_ZMQ_SOCKET_DIR", str(tmp_path / "sockets"))

    async def run_case(ram_gib, *, hidden_size, vision_cfg, model_ctx, cli_ctx=None):
        monkeypatch.setattr(epd_launcher, "available_memory_bytes", lambda: ram_gib * gib)
        backend = SimpleNamespace(
            encoder=SimpleNamespace(
                preprocessor=SimpleNamespace(
                    max_artifact_bytes=epd_launcher.resolve_max_context_bytes()
                ),
                model_config=SimpleNamespace(
                    hidden_size=hidden_size,
                    context_len=model_ctx,
                    hf_config=SimpleNamespace(model_type="glm5_next", vision_config=vision_cfg),
                ),
                model=SimpleNamespace(visual=SimpleNamespace(dtype="bfloat16")),
            ),
            scheduler=SimpleNamespace(max_batch_size=8),
            tp_processes=[],
            start=lambda: None,
            stop=lambda: asyncio.sleep(0),
        )
        monkeypatch.setattr(epd_encoder, "launch_runtime", lambda args, max_active: backend)
        args = SimpleNamespace(
            nnodes=1,
            pp_size=1,
            dp_size=1,
            attn_dp_size=1,
            tp_size=1,
            enable_unified_memory=False,
            revision="glm54-rev",
            host="127.0.0.1",
            port=30000,
            dtype="bfloat16",
            disaggregation_transfer_backend="mooncake",
            encoder_transfer_backend="mooncake",
            disaggregation_bootstrap_port=8998,
            model_path="m",
            served_model_name=None,
            context_length=cli_ctx,
        )
        config = epd_launcher.complete(args, "encoder")
        await epd_launcher.serve(config, args)
        return dict(captured["config"]), captured["max_artifact_bytes"]

    glm54_vision = {
        "patch_size": 14,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "in_channels": 3,
    }
    qwen3_vision = {
        "patch_size": 16,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "in_chans": 3,
    }

    # Case 1: GLM-5.4 8e4p2d (928 GiB cgroup, model context_len=1013760) -> 262144 tokens, 8 slots.
    cfg_glm, art_glm = await run_case(
        928, hidden_size=6144, vision_cfg=glm54_vision, model_ctx=1013760
    )
    assert cfg_glm["max_context_bytes"] == 464 * gib
    assert art_glm == 262144 * 9408 + 4 * 1024**2
    assert cfg_glm["max_staging_bytes_per_request"] == 262144 * 6144 * 2
    assert cfg_glm["execution_capacity"] == 8

    # Case 2: Small 32 GiB container with Qwen3-VL-32B (auto-derived) keeps 8 execution slots.
    cfg_qwen, art_qwen = await run_case(
        32, hidden_size=5120, vision_cfg=qwen3_vision, model_ctx=262144
    )
    assert cfg_qwen["max_context_bytes"] == 16 * gib
    assert cfg_qwen["execution_capacity"] == 8
    assert art_qwen > epd_launcher.encoder_artifact_bytes(
        SimpleNamespace(vision_config=qwen3_vision), "bfloat16", max_tokens=32768
    )

    # Case 3: Explicit --context-length 262144 on 128 GiB container (64 GiB budget) uses 262144 as
    # upper cap and fits media tokens to preserve full target execution_capacity (8 slots).
    cfg_explicit, art_explicit = await run_case(
        128, hidden_size=6144, vision_cfg=glm54_vision, model_ctx=202752, cli_ctx=262144
    )
    assert cfg_explicit["max_context_bytes"] == 64 * gib
    assert art_explicit <= 262144 * 9408 + 4 * 1024**2
    assert art_explicit > epd_launcher.encoder_artifact_bytes(
        SimpleNamespace(vision_config=glm54_vision), "bfloat16", max_tokens=32768
    )
    assert cfg_explicit["execution_capacity"] == 8


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
        # Mimic MMEncoder.__init__ writing sweep_timeout before launch_local_runtime returns.
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
    # Positive infinity stays untouched (never shrunk).
    inf_registry = SimpleNamespace(sweep_timeout=float("inf"))
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.disaggregation.encoder.server",
        SimpleNamespace(meta_registry=inf_registry),
    )
    assert epd_encoder._configure_meta_registry_sweep_timeout() == float("inf")
    assert inf_registry.sweep_timeout == float("inf")

    # Missing sweep_timeout attribute is a clean no-op without adding a bogus attribute.
    no_attr_registry = SimpleNamespace()
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.disaggregation.encoder.server",
        SimpleNamespace(meta_registry=no_attr_registry),
    )
    assert epd_encoder._configure_meta_registry_sweep_timeout() is None
    assert not hasattr(no_attr_registry, "sweep_timeout")

    # None meta_registry is a clean no-op.
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.disaggregation.encoder.server",
        SimpleNamespace(meta_registry=None),
    )
    assert epd_encoder._configure_meta_registry_sweep_timeout() is None

    # Missing server module is a clean no-op.
    monkeypatch.setitem(sys.modules, "sglang.srt.disaggregation.encoder.server", None)
    assert epd_encoder._configure_meta_registry_sweep_timeout() is None


def test_reuse_materialized_feature_hashes_v0521_and_0523_compat():
    """Verify reuse_materialized_feature_hashes supports both sglang v0.5.21 and 0.5.23+ signatures."""
    from smg_grpc_servicer.sglang import epd_plugin

    calls = []

    def original_0523(encoder, mm_feature, preprocess_result, modality):
        calls.append(("0523", mm_feature, preprocess_result, modality))
        return ["orig-0523"]

    def original_v0521(encoder, mm_feature, grid_thw, modality, mm_inputs=None):
        calls.append(("v0521", mm_feature, grid_thw, modality, mm_inputs))
        return ["orig-v0521"]

    # sglang 0.5.23+ 3-arg signature with _smg_item_keys -> fast path reuses Rust keys.
    preproc = SimpleNamespace(_smg_item_keys=[b"k1", b"k2"], grid_thw=[[1, 2, 2], [1, 2, 2]])
    assert epd_plugin.reuse_materialized_feature_hashes(
        original_0523, object(), "feat", preproc, "IMAGE"
    ) == [b"k1", b"k2"]
    assert calls == []

    # sglang v0.5.21 4-arg signature (positional mm_inputs=None) -> clean fallback, no TypeError.
    grid = [[1, 2, 2]]
    assert epd_plugin.reuse_materialized_feature_hashes(
        original_v0521, object(), "feat", grid, "IMAGE", None
    ) == ["orig-v0521"]
    assert len(calls) == 1 and calls[0][0] == "v0521"

    # sglang v0.5.21 4-arg signature (keyword mm_inputs with _smg_item_keys) -> fast path.
    assert epd_plugin.reuse_materialized_feature_hashes(
        original_v0521, object(), "feat", grid, "IMAGE", mm_inputs={"_smg_item_keys": [b"kv21"]}
    ) == [b"kv21"]

    # install_bridge on v0.5.21 MMEncoder lacking cache_namespace succeeds cleanly.
    encoder = SimpleNamespace(
        preprocessor=SimpleNamespace(
            process_batch_mm_items=lambda *a, **kw: None,
            get_num_patches=lambda *a, **kw: 1,
            supports_modality=lambda *a, **kw: True,
            preproc_executor=SimpleNamespace(submit=lambda *a, **kw: None),
        ),
        tp_size=1,
        rank=0,
    )
    epd_encoder.install_bridge(encoder, "smg-image-grid-v1", 1024 * 1024)
    assert not hasattr(encoder, "cache_namespace")
    assert isinstance(encoder.preprocessor, epd_encoder.RustPreprocessorBridge)
