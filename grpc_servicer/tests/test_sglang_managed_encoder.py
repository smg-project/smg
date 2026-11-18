"""CPU ownership regressions; these are not GPU/TP or transport acceptance tests."""

import asyncio
import inspect
import sys
from types import SimpleNamespace

import pytest
from smg_grpc_servicer.sglang.managed.config import complete
from smg_grpc_servicer.sglang.managed.encoder_bridge import NativeReferences


@pytest.fixture
def native_owner(monkeypatch):
    """Construct the real lifecycle owner with only native release/metadata boundaries mocked."""
    from unittest.mock import AsyncMock, Mock

    from smg_grpc_servicer.sglang.managed.encoder_native import NativeEncoder

    def create(*, remaining_ms=1000, host="p", rid="epoch.req", publish=None, **operations):
        metadata = SimpleNamespace(publish=publish if publish is not None else AsyncMock())
        server = SimpleNamespace(meta_registry=metadata)
        monkeypatch.setitem(sys.modules, "sglang.srt.disaggregation.encoder.server", server)
        if parent := sys.modules.get("sglang.srt.disaggregation.encoder"):
            monkeypatch.setattr(parent, "server", server, raising=False)
        backend = SimpleNamespace(
            scheduler=None,
            encoder=SimpleNamespace(
                abandon_request=AsyncMock(),
                release_request=AsyncMock(),
                clear_abandoned_request=Mock(),
            ),
        )
        for name, operation in operations.items():
            setattr(backend.encoder, name, operation)
        owner = NativeEncoder({"generation": "epoch", "retention_ms": 1000}, backend)
        owner.register({"rid": rid, "remaining_ms": remaining_ms, "prefill_host": host})
        return owner, backend, owner.entries[rid]

    return create


def test_native_send_reference_outlives_cancel_acknowledgement():
    """Cancellation closes new work but only the send owner can establish quiescence."""

    async def scenario():
        refs = NativeReferences()
        sending, finish = asyncio.Event(), asyncio.Event()

        async def native_send():
            async with refs.retain():
                sending.set()
                await finish.wait()

        sender = asyncio.create_task(native_send())
        await sending.wait()
        refs.cancel()
        quiesced = asyncio.create_task(refs.wait_quiesced())
        await asyncio.sleep(0)
        assert not quiesced.done()
        with pytest.raises(asyncio.CancelledError):
            async with refs.retain():
                pytest.fail("cancelled part admitted new native work")
        finish.set()
        await sender
        await quiesced
        assert refs.count == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("release_fails", [False, True])
def test_request_payload_released_only_after_native_quiescence(
    monkeypatch, native_owner, release_fails
):
    """Replay fences cannot retain uncharged media or failed-send traceback buffers."""
    import gc
    import weakref
    from unittest.mock import AsyncMock

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.disaggregation.encoder.runtime",
        SimpleNamespace(execute_encode_pipeline=AsyncMock()),
    )

    async def scenario():
        releasing, finish = asyncio.Event(), asyncio.Event()

        async def release(_rid):
            releasing.set()
            await finish.wait()
            if release_fails:
                raise RuntimeError("native release has not established quiescence")

        service, backend, entry = native_owner(
            rid="epoch.rid", remaining_ms=10000, release_request=release
        )
        payload = SimpleNamespace(mm_items=["x" * 1048576])
        entry.request = payload

        class SendBuffer:
            """Stand in for an embedding held by a failed native send's frame."""

        references = []

        async def failed_send():
            """A completed task keeps its exception traceback and native frame locals."""
            buffer = SendBuffer()
            references.append(weakref.ref(buffer))
            raise RuntimeError("native send failed")

        sender = asyncio.create_task(failed_send())
        try:
            await sender
        except RuntimeError:
            pass
        entry.sends["destination"] = (b"digest", sender)
        del sender
        task = asyncio.create_task(service.finish(entry.rid))
        await asyncio.wait_for(releasing.wait(), 2)
        assert entry.request is payload
        assert not entry.quiesced
        finish.set()
        receipt = await task
        assert receipt["released"] is not release_fails
        gc.collect()
        if release_fails:
            assert entry.request is payload
            assert not entry.quiesced
            assert references[0]() is not None
        else:
            assert entry.request is None
            assert entry.quiesced
            assert references[0]() is None

        entry.lease.cancel()
        await asyncio.gather(entry.lease, return_exceptions=True)

    asyncio.run(scenario())


def native_args():
    """Use explicit native values without importing or launching an engine."""
    return SimpleNamespace(
        host="10.0.0.1",
        port=43000,
        dtype="bfloat16",
        tp_size=2,
        disaggregation_transfer_backend="mooncake",
        encoder_transfer_backend="zmq_to_scheduler",
        disaggregation_bootstrap_port=43002,
        model_path="/models/qwen3",
        revision="weights",
    )


@pytest.mark.parametrize("role", ["encoder", "prefill", "decode"])
def test_minimal_config_derives_contract_and_finite_bounds(role):
    """All roles share the same model contract without copying TP/dtype/transport settings."""
    config = complete(native_args(), role)
    contract = config if role == "encoder" else config["contract"]
    assert contract["model_revision"] == "weights"
    assert contract["tp_size"] == 2
    assert contract["embedding_backend"] == "zmq_to_scheduler"
    assert contract["embedding_dtype"] == "bfloat16"
    assert config["max_records"] > 0
    if role == "encoder":
        assert config["max_records"] >= config["max_active"] > 0
    else:
        assert "max_active" not in config  # P/D admission belongs to the shared gateway path.
    assert config["max_duration_ms"] >= 120000 + config["max_clock_error_ms"]
    if role == "encoder":
        assert config["control_listen"] == "0.0.0.0:43010"
        assert config["receive_endpoint"] == "http://10.0.0.1:43000"


@pytest.mark.parametrize(
    "field,value",
    [
        ("nnodes", 2),
        ("pp_size", 2),
        ("dp_size", 2),
        ("attn_dp_size", 2),
        ("enable_unified_memory", True),
    ],
)
def test_native_layout_cannot_advertise_an_unsupported_topology(field, value):
    """Reject layouts the adapter cannot represent before advertising a single TP group."""
    args = native_args()
    setattr(args, field, value)
    with pytest.raises(ValueError, match="does not support"):
        complete(args, "prefill")


@pytest.mark.parametrize("backend", ["mooncake", "zmq_to_scheduler"])
def test_receiver_fanout_preserves_memory_bound_and_advertised_capacity(backend):
    """Eight P ranks are allowed without overcommitting the E owner's context budget."""
    from smg_grpc_servicer.sglang.managed.config import encoder_window

    args = native_args()
    args.encoder_transfer_backend = backend
    config = complete(args, "encoder")
    config["generation"] = "generation"
    execution_capacity, reserved = encoder_window(config)
    capacity = config["max_active"]
    assert config["max_receivers"] == 8
    # Physical storage is shared only for native Mooncake GPU-direct sends.
    assert execution_capacity == (14 if backend == "mooncake" else 5)
    controls = capacity * config["max_control_bytes"]
    assert capacity == config["max_active"]
    assert execution_capacity < capacity
    assert controls + reserved * execution_capacity <= config["max_context_bytes"]
    assert controls + reserved * (execution_capacity + 1) > config["max_context_bytes"]
    assert config["max_early_waiters"] >= capacity * config["max_receivers"]
    assert config["max_http_connections"] > config["max_early_waiters"]
    config["max_context_bytes"] = controls + reserved - 1
    with pytest.raises(ValueError, match="cannot admit"):
        encoder_window(config)


def test_native_mooncake_staging_owns_per_request_storage():
    """Optional native staging checks do not skip dependency-free capacity coverage."""
    # Exercise native staging on CPU tensors: this checks the ownership
    # contract, not GPU/RDMA completion (covered by real transport tests).
    torch = pytest.importorskip("torch")
    server = pytest.importorskip("sglang.srt.disaggregation.encoder.server")
    Modality = pytest.importorskip("sglang.srt.managers.req_types").Modality
    MMEncoder = server.MMEncoder

    staged = []
    native = SimpleNamespace(
        rank=0,
        _register_shared_mr=lambda *args: None,
        _deregister_shared_mr=lambda *args: None,
        _stage_embedding_batch=staged.extend,
    )
    batch = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    ctx = SimpleNamespace(
        items_per_req=[1, 1],
        preprocess_result=SimpleNamespace(
            token_counts=[2, 4], grid_thw=torch.ones((2, 3), dtype=torch.int64)
        ),
        aux_data={},
        modality=Modality.IMAGE,
        use_global_cache=False,
    )
    requests = [{"req_id": str(i), "num_parts": 1, "part_idx": 0} for i in range(2)]
    result = MMEncoder._stage_embeddings(native, ctx, requests, batch, keep_on_gpu=True)
    if inspect.isawaitable(result):
        asyncio.run(result)
    assert len(staged) == len(requests)
    for item, expected in zip(staged, (batch[:2], batch[2:])):
        assert torch.equal(item.embedding, expected)
        assert item.embedding.data_ptr() != expected.data_ptr()
        assert item.embedding.untyped_storage().nbytes() == item.embedding.nbytes


@pytest.mark.parametrize("kind", ["metadata", "register"])
def test_cancelled_queued_encode_retires_early_receiver_control(monkeypatch, kind):
    """Metadata or registration waits cannot pin a cancelled admission until its full deadline."""
    from unittest.mock import AsyncMock

    from smg_grpc_servicer.sglang.managed.encoder_native import NativeEncoder, NativeEntry, Rejected

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.disaggregation.encoder.runtime",
        SimpleNamespace(send_staged_embedding=AsyncMock()),
    )

    async def scenario():
        started, retired = asyncio.Event(), asyncio.Event()

        async def register(*_args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                retired.set()

        server = SimpleNamespace(meta_registry=SimpleNamespace(wait=register))
        monkeypatch.setitem(sys.modules, "sglang.srt.disaggregation.encoder.server", server)
        if parent := sys.modules.get("sglang.srt.disaggregation.encoder"):
            monkeypatch.setattr(parent, "server", server, raising=False)
        else:
            monkeypatch.setitem(
                sys.modules, "sglang.srt.disaggregation.encoder", SimpleNamespace(server=server)
            )
        service = object.__new__(NativeEncoder)
        service.config = {"max_receivers": 2}
        service.backend = SimpleNamespace(
            encoder=SimpleNamespace(
                transfer_backend="zmq_to_scheduler", register_embedding_destinations=register
            )
        )
        entry = NativeEntry(
            float("inf"),
            "rid",
            request=SimpleNamespace(prefill_host="10.0.0.2"),
        )
        service.part = AsyncMock(return_value=entry)
        task = asyncio.create_task(
            service.facade(
                kind,
                {
                    "req_id": "rid",
                    "part_idx": 0,
                    "receive_count": 2,
                    "receive_url": "10.0.0.2:1234",
                },
            )
        )
        await asyncio.wait_for(started.wait(), 1)
        assert entry.refs.count == 1
        entry.cancelled = True
        entry.refs.cancel()
        entry.cancellation.set()
        with pytest.raises(Rejected, match="control cancelled"):
            await asyncio.wait_for(task, 1)
        assert retired.is_set() and entry.refs.count == 0
        assert not entry.quiesced  # Only the actual Encode owner can remove its admission.

    asyncio.run(scenario())


@pytest.mark.parametrize("expire", [False, True])
@pytest.mark.parametrize("abandon_fails", [False, True])
def test_zmq_abandonment_wakes_native_wait_without_releasing_active_send(
    monkeypatch, expire, abandon_fails
):
    """Actual native completion, not cancel acknowledgement, owns buffer release."""
    import time
    from unittest.mock import AsyncMock, Mock

    from smg_grpc_servicer.sglang.managed.encoder_native import NativeEncoder

    async def scenario():
        started, abandoned, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def pipeline(*_args):
            started.set()
            await abandoned.wait()
            await finish.wait()

        async def abandon(_rid):
            abandoned.set()
            if abandon_fails:
                raise RuntimeError("injected partial abandonment failure")

        monkeypatch.setitem(
            sys.modules,
            "sglang.srt.disaggregation.encoder.runtime",
            SimpleNamespace(execute_encode_pipeline=pipeline),
        )
        monkeypatch.setitem(
            sys.modules,
            "sglang.srt.disaggregation.encoder.server",
            SimpleNamespace(meta_registry=SimpleNamespace(publish=AsyncMock())),
        )
        monkeypatch.setitem(
            sys.modules,
            "sglang.srt.managers.req_types",
            SimpleNamespace(Modality=SimpleNamespace(IMAGE="image")),
        )
        encoder = SimpleNamespace(
            _embedding_dims={"image": 1},
            _element_size=2,
            transfer_backend="zmq_to_scheduler",
            abandon_request=abandon,
            release_request=AsyncMock(),
            clear_abandoned_request=Mock(),
        )
        service = NativeEncoder(
            {"generation": "epoch", "retention_ms": 1000, "max_staging_bytes_per_request": 1024},
            SimpleNamespace(scheduler=None, encoder=encoder),
        )
        service.register({"rid": "epoch.rid", "remaining_ms": 10000, "prefill_host": "p"})
        entry = service.entries["epoch.rid"]
        execution = asyncio.create_task(
            service.execute(
                {"rid": entry.rid, "embedding_tokens": 1, "image_count": 1}, b"artifact"
            )
        )
        await asyncio.wait_for(started.wait(), 1)
        if expire:
            entry.lease.cancel()
            await asyncio.gather(entry.lease, return_exceptions=True)
            entry.deadline = time.monotonic()
            entry.lease = asyncio.create_task(service._lease(entry))
            cancellation = entry.lease
        else:
            cancellation = asyncio.create_task(service.cancel(entry.rid))
        await asyncio.wait_for(abandoned.wait(), 1)
        assert entry.cancelled and entry.request is not None and not entry.quiesced
        assert entry.refs.count == 1  # Native batch/send still holds the input.
        encoder.release_request.assert_not_awaited()
        assert not execution.done()
        finish.set()
        receipt = await asyncio.wait_for(execution, 1)
        await asyncio.gather(cancellation, return_exceptions=True)
        if abandon_fails:
            assert receipt["released"] is False and receipt["error"]
            assert not entry.quiesced and entry.rid in service.entries
            encoder.release_request.assert_not_awaited()
            encoder.clear_abandoned_request.assert_not_called()
        else:
            assert receipt["released"] is True and receipt["error"]
            assert entry.quiesced and entry.request is None
            encoder.release_request.assert_awaited_once_with(entry.rid)
            encoder.clear_abandoned_request.assert_called_once_with(entry.rid)
        entry.lease.cancel()
        await asyncio.gather(entry.lease, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_count", [0, 1, 3])
@pytest.mark.parametrize("cpu_fails", [False, True])
def test_artifact_materialization_keeps_control_loop_live_and_owns_cpu_work(
    cancel_count, cpu_fails
):
    """Control callbacks run during CPU work, and cancellation waits for its actual exit."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from smg_grpc_servicer.sglang.managed.encoder_bridge import RustPreprocessorBridge

    async def scenario():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        finish = threading.Event()
        executor_work = []
        unhandled = []
        submit = loop.run_in_executor

        def capture_work(*args):
            work = submit(*args)
            executor_work.append(work)
            return work

        loop.run_in_executor = capture_work
        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
        bridge = RustPreprocessorBridge(SimpleNamespace(preproc_executor=executor), 1024)
        requests = [object()]
        result = object()

        def materialize(actual_requests, modality):
            assert actual_requests is requests
            assert threading.get_ident() != control_thread
            loop.call_soon_threadsafe(started.set)
            assert finish.wait(2), "CPU owner was not released by the control loop"
            if cpu_fails:
                raise ValueError("CPU materialization failed")
            return result

        control_thread = threading.get_ident()
        bridge._materialize_batch = materialize
        task = asyncio.create_task(bridge.process_batch_mm_items(requests, None))
        try:
            await asyncio.wait_for(started.wait(), 1)
            for cancellation in range(cancel_count):
                task.cancel(f"cancel {cancellation}")
                await asyncio.sleep(0)
                assert not task.done()
                assert len(executor_work) == 1
                assert not executor_work[0].done() and not executor_work[0].cancelled()
            assert not task.done()
        finally:
            loop.run_in_executor = submit
            finish.set()
        if cancel_count:
            with pytest.raises(asyncio.CancelledError, match="cancel 0"):
                await task
        elif cpu_fails:
            with pytest.raises(ValueError, match="CPU materialization failed"):
                await task
        else:
            assert await task is result
        assert executor_work[0].done() and not executor_work[0].cancelled()
        executor_work.clear()
        await asyncio.sleep(0)
        assert not unhandled

    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(scenario())


def test_cpu_capacity_respects_affinity_and_container_quota(monkeypatch):
    """Host core counts must not override a smaller container CPU allocation."""
    from smg_grpc_servicer.sglang.managed import config

    monkeypatch.setattr(config.os, "sched_getaffinity", lambda _: set(range(16)))
    monkeypatch.setattr(config.Path, "read_text", lambda _: "250000 100000")
    assert config.available_cpus() == 2
    monkeypatch.setattr(config.Path, "read_text", lambda _: "max 100000")
    assert config.available_cpus() == 16


@pytest.mark.parametrize("processor_limit", [0, -1, None, True, 1.5])
def test_adaptive_cpu_limit_rejects_invalid_processor_bound(processor_limit):
    """A zero or nonintegral hard ceiling must fail instead of stalling dispatch."""
    from smg_grpc_servicer.sglang.managed.config import encoder_cpu_limit

    config = complete(native_args(), "encoder")
    config["processor_options"]["max_inflight"] = processor_limit
    with pytest.raises(ValueError, match="positive integer"):
        encoder_cpu_limit(config, 8)


def test_adaptive_cpu_limit_intersects_native_memory_and_host_bounds(monkeypatch):
    """An explicit ceiling cannot overrun actual CPU or retained tensor capacity."""
    from smg_grpc_servicer.sglang.managed import config as settings

    config = complete(native_args(), "encoder")
    config["max_cpu"] = config["processor_options"]["max_inflight"] = 32
    monkeypatch.setattr(settings, "available_cpus", lambda: 16)
    assert settings.encoder_cpu_limit(config, 1) == 2
    assert settings.encoder_cpu_limit(config, 32) == settings.encoder_window(config)[0]
    monkeypatch.setattr(settings, "available_cpus", lambda: 1)
    assert settings.encoder_cpu_limit(config, 8) == 1


@pytest.mark.parametrize("role", ["encoder", "prefill", "decode"])
def test_native_revision_needs_no_managed_config_file(role):
    """Existing native identity and role arguments are sufficient for default deployment."""
    args = native_args()
    args.revision = "pinned-weights"
    config = complete(args, role)
    contract = config if role == "encoder" else config["contract"]
    assert contract["model_revision"] == args.revision
    assert contract["tp_size"] == args.tp_size


def test_staging_budget_tracks_model_width_and_native_dtype():
    """Larger Qwen3-VL embeddings need a larger reservation, not a deployment override."""
    from smg_grpc_servicer.sglang.managed.config import encoder_staging_bytes

    assert encoder_staging_bytes(4096, "bfloat16") == 32 * 1024**2
    assert encoder_staging_bytes(5120, "bfloat16") == 40 * 1024**2
    assert encoder_staging_bytes(5120, "float32") == 80 * 1024**2
    with pytest.raises(ValueError, match="dimensions or dtype"):
        encoder_staging_bytes(0, "bfloat16")


@pytest.mark.parametrize("bind_host", ["0.0.0.0", "::"])
def test_wildcard_bind_uses_native_address_discovery(monkeypatch, bind_host):
    """Containers without resolvable hostnames reuse the native engine's route/NIC discovery."""

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.utils.network",
        SimpleNamespace(get_local_ip_auto=lambda: "10.0.0.2"),
    )
    args = native_args()
    args.host = bind_host
    args.revision = "weights"
    config = complete(args, "encoder")
    assert config["receive_endpoint"] == "http://10.0.0.2:43000"
    assert config["worker_id"] == "encoder-10.0.0.2:43000"


@pytest.mark.parametrize("batch_size", [1, 2])
def test_materialized_pixels_remain_owned_after_artifact_release(batch_size):
    """Single and fused inputs retain writable storage without mutating wire artifacts."""
    import gc
    import struct

    torch = pytest.importorskip("torch")
    pytest.importorskip("sglang.srt.disaggregation.encoder.preprocessor")
    from smg_grpc_proto.generated import sglang_scheduler_pb2 as sg
    from smg_grpc_servicer.sglang.managed.encoder_bridge import ARTIFACT_KEY, RustPreprocessorBridge

    request = sg.GenerateRequest()
    mm = request.mm_inputs
    mm.pixel_values.dtype = "float32"
    mm.pixel_values.shape.extend([4, 1])
    mm.pixel_values.data = struct.pack("4f", 1, 2, 3, 4)
    for name, shape, values in (
        ("image_grid_thw", [1, 3], [1, 2, 2]),
        ("patches_per_image", [1], [4]),
    ):
        tensor = mm.model_specific_tensors[name]
        tensor.dtype = "int64"
        tensor.shape.extend(shape)
        tensor.data = struct.pack(f"{len(values)}q", *values)
    mm.mm_placeholders.add(length=1)
    wire = request.SerializeToString()
    requests = [{ARTIFACT_KEY: wire, "mm_items": [None]} for _ in range(batch_size)]
    result, counts = RustPreprocessorBridge(None, 4096)._materialize_batch(
        requests, SimpleNamespace(name="IMAGE")
    )
    del requests, request
    gc.collect()
    pixels = result.mm_inputs["pixel_values"]
    assert torch.equal(pixels.flatten(), torch.tensor([1, 2, 3, 4] * batch_size))
    pixels[0, 0] = 9
    unchanged = sg.GenerateRequest.FromString(wire)
    assert struct.unpack("4f", unchanged.mm_inputs.pixel_values.data) == (1, 2, 3, 4)
    assert counts == [1] * batch_size
    assert result.token_counts == [1] * batch_size
    assert result.grid_thw.tolist() == [[1, 2, 2]] * batch_size


def test_encoder_forward_wait_yields_and_retains_owner_through_repeated_cancel(monkeypatch):
    """Cancellation cannot outlive the GPU completion owner; other loop tasks still run."""
    import threading

    from smg_grpc_servicer.sglang.managed.plugin import await_encoder_forward

    async def scenario():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        finish = threading.Event()
        event_loop_thread = threading.get_ident()
        device = object()
        stream = object()
        result = SimpleNamespace(is_cuda=True, device=device)

        class Event:
            def record(self, selected):
                assert selected is stream
                assert threading.get_ident() == event_loop_thread

            def synchronize(self):
                assert threading.get_ident() != event_loop_thread
                loop.call_soon_threadsafe(started.set)
                assert finish.wait(5)

        def current_stream(selected):
            assert selected is device
            return stream

        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(cuda=SimpleNamespace(Event=Event, current_stream=current_stream)),
        )

        async def original(_encoder):
            return result

        task = asyncio.create_task(await_encoder_forward(original, object()))
        try:
            await asyncio.wait_for(started.wait(), 2)
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
            # asyncio.run shutdown cancels every Task, including a to_thread Task.
            # The private executor Future must remain owned until GPU completion.
            for pending in asyncio.all_tasks():
                if pending is not asyncio.current_task():
                    pending.cancel()
            for _ in range(4):
                await asyncio.sleep(0)
            assert not task.done()
        finally:
            finish.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert await await_encoder_forward(original, object()) is result

    asyncio.run(scenario())


@pytest.mark.parametrize("enabled", [None, "", "0", "1", "invalid"])
def test_native_launch_switch_preserves_default_and_parsed_arguments(monkeypatch, enabled):
    """Only explicit opt-in changes dispatch; invalid values cannot silently use legacy HTTP."""
    from smg_grpc_servicer.sglang.managed import launch, plugin

    calls = []
    args = SimpleNamespace(encoder_only=True, tp_size=2)
    monkeypatch.delenv(plugin.ENABLE_ENV, raising=False)
    if enabled is not None:
        monkeypatch.setenv(plugin.ENABLE_ENV, enabled)
    monkeypatch.setattr(launch, "run_server", lambda value: calls.append(("managed", value)))
    if enabled == "invalid":
        with pytest.raises(ValueError, match="must be 0 or 1"):
            plugin.native_launch(lambda value: calls.append(("native", value)), args)
        assert not calls
    else:
        plugin.native_launch(lambda value: calls.append(("native", value)), args)
        assert calls == [("managed" if enabled == "1" else "native", args)]


@pytest.mark.parametrize("module", ["sglang.launch_server", "sglang.cli.main"])
def test_native_entry_hook_targets_running_module(monkeypatch, module):
    """The -m entry is a distinct module; the CLI imports the canonical run_server."""

    from smg_grpc_servicer.sglang.managed import plugin

    registrations = []
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.plugins.hook_registry",
        SimpleNamespace(
            HookRegistry=SimpleNamespace(register=lambda *args: registrations.append(args)),
            HookType=SimpleNamespace(AROUND="around"),
        ),
    )
    monkeypatch.setattr(sys.modules["__main__"], "__spec__", SimpleNamespace(name=module))
    monkeypatch.delenv(plugin.ROLE_ENV, raising=False)
    monkeypatch.delenv(plugin.ENABLE_ENV, raising=False)
    plugin.register()
    assert not registrations
    monkeypatch.setenv(plugin.ENABLE_ENV, "1")
    plugin.register()
    target = "__main__" if module == "sglang.launch_server" else "sglang.launch_server"
    assert registrations == [(target + ".run_server", plugin.native_launch, "around")]


def test_role_hooks_install_after_native_plugins_loaded_once(monkeypatch):
    """An idempotent native loader must not hide the role discovered after CLI parsing."""

    from smg_grpc_servicer.sglang.managed import plugin

    registrations, applied = [], []
    monkeypatch.setattr(plugin, "_registered_role", None)
    monkeypatch.setenv(plugin.ROLE_ENV, "encoder")
    monkeypatch.setitem(
        sys.modules, "sglang.srt.plugins", SimpleNamespace(load_plugins=lambda: None)
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.plugins.hook_registry",
        SimpleNamespace(
            HookRegistry=SimpleNamespace(
                register=lambda *args: registrations.append(args),
                apply_hooks=lambda: applied.append(True),
            ),
            HookType=SimpleNamespace(AROUND="around", AFTER="after"),
        ),
    )
    plugin.load_role_hooks()
    plugin.load_role_hooks()
    assert len(registrations) == 2
    assert registrations[1][1] is plugin.await_encoder_forward
    assert applied == [True, True]


@pytest.mark.parametrize("override", [None, "4"])
def test_encoder_defaults_precede_config_and_runtime(monkeypatch, override):
    """Native startup inherits media defaults without replacing deployment overrides."""
    import os

    from smg_grpc_servicer.sglang.managed import config, launch

    for name in ("SMG_MM_THREADS", "SMG_MM_IMAGE_CONCURRENCY", "SMG_SERVICER_WORKER_THREADS"):
        monkeypatch.delenv(name, raising=False)
    if override is not None:
        monkeypatch.setenv("SMG_MM_THREADS", override)

    class StopBeforeRuntime(Exception):
        pass

    def inspect_defaults(*args):
        assert os.environ["SMG_MM_THREADS"] == (override or "32")
        assert os.environ["SMG_MM_IMAGE_CONCURRENCY"] == "8"
        assert os.environ["SMG_SERVICER_WORKER_THREADS"] == "8"
        raise StopBeforeRuntime

    monkeypatch.setattr(config, "complete", inspect_defaults)
    with pytest.raises(StopBeforeRuntime):
        launch.encoder_main(SimpleNamespace())


@pytest.mark.parametrize("publication_fails", [False, True])
def test_rust_native_cleanup_is_single_owner_and_receipt_is_replayable(
    native_owner, publication_fails
):
    """Wire retries cannot call native release twice or turn failed cleanup into success."""
    from unittest.mock import AsyncMock

    publish = AsyncMock(
        side_effect=RuntimeError("metadata publication failed") if publication_fails else None
    )

    async def scenario():
        release_started, allow_release = asyncio.Event(), asyncio.Event()

        async def release(rid):
            release_started.set()
            await allow_release.wait()

        release_mock = AsyncMock(side_effect=release)
        owner, backend, entry = native_owner(
            host="127.0.0.1", publish=publish, release_request=release_mock
        )
        first = asyncio.create_task(owner.finish("epoch.req", "CPU failed"))
        second = asyncio.create_task(owner.finish("epoch.req", "duplicate CPU failure"))
        await release_started.wait()
        assert not first.done() and not second.done()
        await owner.cancel("epoch.req")
        assert "epoch.req" in owner.entries
        allow_release.set()
        receipt, duplicate = await asyncio.gather(first, second)
        assert receipt == duplicate == await owner.finish("epoch.req", "retry")
        assert receipt["released"] is True
        assert release_mock.await_count == 1
        assert "epoch.req" not in owner.entries
        if publication_fails:
            assert receipt["error"] == "metadata publication failed"
        entry.lease.cancel()
        await asyncio.gather(entry.lease, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_failures", [0, 1, 2])
def test_native_orphan_registration_is_released_after_lease(native_owner, cancel_failures):
    """A Rust control loop lost between registration and execute does not orphan P waits."""

    async def scenario():
        owner, backend, entry = native_owner(remaining_ms=1, host="127.0.0.1")
        assert (
            entry.task is None
            and entry.refs.count == 0
            and entry.request.prefill_host == "127.0.0.1"
        )
        backend.encoder.abandon_request.assert_not_awaited()
        backend.encoder.release_request.assert_not_awaited()
        if cancel_failures:
            backend.encoder.abandon_request.side_effect = [
                RuntimeError("cancel failed"),
                RuntimeError("cleanup cancel failed") if cancel_failures == 2 else None,
            ]
        lease = entry.lease
        if cancel_failures:
            with pytest.raises(RuntimeError, match="^cancel failed$"):
                await asyncio.wait_for(lease, 1)
        else:
            await asyncio.wait_for(lease, 1)
        receipt = await owner.finish("epoch.req")
        if cancel_failures == 2:
            assert owner.entries[entry.rid] is entry
            assert not entry.quiesced and entry.refs.count == 0
            assert receipt == {"released": False, "error": "cleanup cancel failed"}
            backend.encoder.release_request.assert_not_awaited()
            assert await owner.finish(entry.rid) == receipt
        else:
            assert not owner.entries
            assert receipt["released"] is True
            assert backend.encoder.release_request.await_count == 1
        assert backend.encoder.abandon_request.await_count == 2

    asyncio.run(scenario())


def test_native_failed_release_is_retained_and_never_retried(native_owner):
    from unittest.mock import AsyncMock

    async def scenario():
        owner, backend, entry = native_owner(
            release_request=AsyncMock(side_effect=RuntimeError("release uncertain"))
        )
        first = await owner.finish("epoch.req")
        assert (
            first
            == await owner.finish("epoch.req")
            == {"released": False, "error": "release uncertain"}
        )
        assert owner.entries["epoch.req"] is entry and not entry.quiesced
        assert backend.encoder.release_request.await_count == 1
        backend.encoder.clear_abandoned_request.assert_not_called()
        entry.lease.cancel()
        await asyncio.gather(entry.lease, return_exceptions=True)

    asyncio.run(scenario())


def test_encoder_signal_wakes_idle_selector():
    """A real signal must wake the loop rather than wait for an unrelated native timer."""
    import json
    import signal
    import subprocess
    import time

    script = """
import asyncio, json, signal, time
from types import SimpleNamespace
from smg_grpc_servicer.sglang.managed.launch import encoder_stop_handler
async def main():
    loop = asyncio.get_running_loop()
    stopped = asyncio.Event()
    calls = []
    server = SimpleNamespace(drain=lambda: calls.append(time.monotonic()))
    previous = signal.signal(signal.SIGTERM, encoder_stop_handler(loop, server, stopped))
    loop.call_later(3, lambda: None)
    print('READY', flush=True)
    try:
        await stopped.wait()
        print(json.dumps({'drained': calls, 'resumed': time.monotonic()}), flush=True)
    finally:
        signal.signal(signal.SIGTERM, previous)
asyncio.run(main())
"""
    process = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "READY"
        time.sleep(0.1)  # Let the selector enter its three-second sleep.
        sent = time.monotonic()
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=5)
        assert process.returncode == 0
        marks = json.loads(output)
        assert len(marks["drained"]) == 1
        assert sent <= marks["drained"][0] <= marks["resumed"]
        assert marks["resumed"] - sent < 1, marks
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_native_replay_receipt_expiry_preserves_live_duplicate_fence(monkeypatch):
    from smg_grpc_servicer.sglang.managed import encoder_native

    monkeypatch.setattr(encoder_native, "time", SimpleNamespace(monotonic=lambda: 100.0))

    async def scenario():
        owner = encoder_native.NativeEncoder(
            {"generation": "epoch", "retention_ms": 1000}, SimpleNamespace(scheduler=None)
        )
        receipt = {"released": True, "error": None}
        owner.completed.update(
            {
                "epoch.expired": (99.0, receipt),
                "epoch.boundary": (100.0, receipt),
                "epoch.live": (101.0, receipt),
            }
        )
        try:
            for rid in ("epoch.expired", "epoch.boundary"):
                owner.register({"rid": rid, "remaining_ms": 1000, "prefill_host": "p"})
            assert owner.completed == {"epoch.live": (101.0, receipt)}
            with pytest.raises(encoder_native.Rejected, match="completed native operation"):
                owner.register({"rid": "epoch.live", "remaining_ms": 1000, "prefill_host": "p"})
            assert await owner.finish("epoch.live") is receipt
        finally:
            for entry in owner.entries.values():
                entry.lease.cancel()
            await asyncio.gather(
                *(entry.lease for entry in owner.entries.values()), return_exceptions=True
            )

    asyncio.run(scenario())


def test_native_release_waits_for_inflight_cancel_and_rejects_late_cleanup(native_owner):
    """The released receipt includes native cancellation, even if its wire ACK is lost."""

    async def scenario():
        cancel_started, allow_cancel = asyncio.Event(), asyncio.Event()
        release_started, allow_release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def abandon(_rid):
            nonlocal calls
            calls += 1
            if calls == 1:
                cancel_started.set()
                await allow_cancel.wait()

        async def release(_rid):
            release_started.set()
            await allow_release.wait()

        owner, backend, entry = native_owner(
            remaining_ms=10000, abandon_request=abandon, release_request=release
        )
        cancel = asyncio.create_task(owner.cancel(entry.rid))
        await cancel_started.wait()
        finish = asyncio.create_task(owner.finish(entry.rid))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not release_started.is_set() and not finish.done()
        assert entry.refs.count == 1 and not entry.quiesced
        allow_cancel.set()
        await cancel
        await release_started.wait()
        assert calls == 2  # cancel plus the single final cleanup owner
        await owner.cancel(entry.rid)
        assert calls == 2  # late cancel cannot cross the final quiescence barrier
        allow_release.set()
        assert (await finish)["released"] is True
        assert entry.refs.count == 0 and entry.quiesced
        await owner.cancel(entry.rid)
        assert calls == 2
        entry.lease.cancel()
        await asyncio.gather(entry.lease, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("terminal_error", [None, "injected Rust failure"])
def test_encoder_watchdog_observes_rust_terminal_state(terminal_error):
    from smg_grpc_servicer.sglang.managed.launch import wait_encoder_shutdown

    async def scenario():
        stopped = asyncio.Event()
        server = SimpleNamespace(last_error=lambda: terminal_error, running=lambda: False)
        with pytest.raises(RuntimeError, match="Rust encoder control server stopped"):
            await wait_encoder_shutdown(server, stopped)
        stopped.set()
        await wait_encoder_shutdown(server, stopped)

    asyncio.run(scenario())


@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize("processor", [None, "smg"])
def test_managed_launcher_extras_ignore_unused_legacy_processor_env(monkeypatch, role, processor):
    import json

    from smg_grpc_servicer.sglang.managed.launch import launcher

    class LegacyLauncher:
        def server_extras(self, *_args):
            pytest.fail("managed startup must not build the legacy media processor")

    monkeypatch.setitem(
        sys.modules,
        "smg_grpc_servicer.sglang.rust",
        SimpleNamespace(
            SglangLauncher=LegacyLauncher, launch_scheduler_ranks=lambda *_args, **_kwargs: None
        ),
    )
    if processor is None:
        monkeypatch.delenv("SMG_SGLANG_MM_PROCESSOR", raising=False)
    else:
        monkeypatch.setenv("SMG_SGLANG_MM_PROCESSOR", processor)
    config = {"role": role}
    args = SimpleNamespace(disaggregation_mode=role, language_only=True)
    managed = launcher(config)
    assert managed.server_extras(args, None) == {"managed_config": json.dumps(config)}
    args.language_only = False
    with pytest.raises(ValueError, match="language-only"):
        managed.server_extras(args, None)
    args.language_only = True
    args.disaggregation_mode = "prefill" if role == "decode" else "decode"
    with pytest.raises(ValueError, match="role must equal"):
        managed.server_extras(args, None)


def test_native_ipc_shutdown_drops_only_completed_reply_observers(monkeypatch, tmp_path):
    """Real IPC backpressure cannot pin shutdown after native work truly completes."""
    import json

    import zmq
    import zmq.asyncio
    from smg_grpc_servicer.sglang.managed import encoder_native

    async def scenario():
        state = SimpleNamespace(completed=0, active=False, cancelled=False)
        release, started = asyncio.Event(), asyncio.Event()
        deferred, allow_deferred = asyncio.Event(), asyncio.Event()
        future_counts = {"fast": 0, "blocked": 0}
        original_send = zmq.asyncio.Socket.send_json
        original_create = asyncio.create_task

        def observe_send(socket, *args, **kwargs):
            future = original_send(socket, *args, **kwargs)
            if socket.getsockopt(zmq.LAST_ENDPOINT).endswith(b"/events"):
                assert asyncio.isfuture(future)
                future_counts["fast" if future.done() else "blocked"] += 1
            return future

        def defer_one_handle(coroutine, **kwargs):
            if coroutine.__qualname__.endswith("serve_native.<locals>.handle"):
                command = json.loads(coroutine.cr_frame.f_locals["frames"][0])
                if command.get("defer"):
                    deferred.set()

                    async def resume():
                        await allow_deferred.wait()
                        return await coroutine

                    return original_create(resume(), **kwargs)
            return original_create(coroutine, **kwargs)

        class LegacyAsyncTimeout(Exception):
            pass

        monkeypatch.setattr(asyncio, "TimeoutError", LegacyAsyncTimeout)

        class NativeOwner:
            def __init__(self, *_args):
                self.entries = {"failed_cancel": object(), "active": object()}

            async def dispatch(self, command, artifact):
                if command.get("timeout"):
                    raise LegacyAsyncTimeout("receiver expired")
                assert not command.get("defer"), "shutdown must not start a queued native operation"
                if command.get("hold"):
                    state.active = True
                    started.set()
                    await release.wait()
                    state.active = False
                state.completed += 1
                return {"released": True, "error": "x" * 1024}

            async def cancel(self, rid):
                if rid == "failed_cancel":
                    raise RuntimeError("first native abandonment failed")
                state.cancelled = True

        monkeypatch.setattr(encoder_native, "NativeEncoder", NativeOwner)
        monkeypatch.setattr(zmq.asyncio.Socket, "send_json", observe_send)
        monkeypatch.setattr(asyncio, "create_task", defer_one_handle)
        context = zmq.asyncio.Context()
        command, replies = context.socket(zmq.PUSH), context.socket(zmq.PULL)
        for socket in (command, replies):
            socket.setsockopt(zmq.LINGER, 0)
        commands = f"ipc://{tmp_path}/commands"
        events = f"ipc://{tmp_path}/events"
        command.bind(commands)
        replies.bind(events)  # Deliberately do not consume replies.
        config = {"generation": "test", "max_control_bytes": 4096, "max_artifact_bytes": 4096}
        serving = original_create(encoder_native.serve_native(config, None, commands, events))
        try:
            await command.send_json({"generation": "test", "sequence": -1, "timeout": True})
            timeout_reply = await replies.recv_json()
            assert timeout_reply["error_code"] == 4 and "result" not in timeout_reply
            for index in range(10002):
                await command.send_json(
                    {
                        "generation": "test",
                        "sequence": index,
                        "hold": index == 10000,
                        "defer": index == 10001,
                    }
                )
            await started.wait()
            await deferred.wait()
            assert state.completed == 10000
            assert future_counts["fast"] > 0 and future_counts["blocked"] > 0
            serving.cancel()
            while not state.cancelled:
                await asyncio.sleep(0)
            allow_deferred.set()
            await asyncio.sleep(0)
            assert state.active and not serving.done()
            release.set()
            await asyncio.gather(serving, return_exceptions=True)
            assert not state.active and state.completed == 10001
            assert serving.cancelled()
        finally:
            release.set()
            allow_deferred.set()
            if not serving.done():
                serving.cancel()
                await asyncio.gather(serving, return_exceptions=True)
            command.close()
            replies.close()
            context.term()

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_native_invalid_receiver_does_not_commit_receive_count(monkeypatch):
    """A malformed first destination must not poison a subsequent valid receiver."""
    import time

    from smg_grpc_servicer.sglang.managed.encoder_native import NativeEncoder, NativeEntry, Rejected

    native_server = SimpleNamespace()
    monkeypatch.setitem(
        sys.modules, "sglang.srt.disaggregation.encoder", SimpleNamespace(server=native_server)
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.disaggregation.encoder.server", native_server)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.disaggregation.encoder.runtime",
        SimpleNamespace(send_staged_embedding=None),
    )

    async def scenario():
        calls = []

        async def register(*args):
            calls.append(args)

        backend = SimpleNamespace(
            scheduler=None,
            encoder=SimpleNamespace(
                transfer_backend="zmq_to_scheduler", register_embedding_destinations=register
            ),
        )
        owner = NativeEncoder({"max_receivers": 8}, backend)
        entry = NativeEntry(
            time.monotonic() + 10, "epoch.req", request=SimpleNamespace(prefill_host="127.0.0.1")
        )
        owner.entries[entry.rid] = entry
        with pytest.raises(Rejected):
            await owner.facade(
                "register",
                {"req_id": entry.rid, "receive_count": 2, "receive_url": "wrong-host:1234"},
            )
        assert entry.receivers is None and not entry.sends and not calls
        await owner.facade(
            "register", {"req_id": entry.rid, "receive_count": 1, "receive_url": "127.0.0.1:1234"}
        )
        assert entry.receivers == 1 and len(calls) == 1

    asyncio.run(scenario())
