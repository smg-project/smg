"""CPU ownership regressions; these are not GPU/TP or transport acceptance tests."""

import asyncio
from types import SimpleNamespace

import pytest
from smg_grpc_servicer.sglang.managed.config import complete
from smg_grpc_servicer.sglang.managed.encoder_bridge import NativeReferences
from smg_grpc_servicer.sglang.managed.encoder_service import FairQueue


def test_drr_aging_admits_large_work_and_reclaims_tenant_state(monkeypatch):
    """Small requests cannot indefinitely delay an aged large request."""
    clock = [0.0]
    monkeypatch.setattr(
        "smg_grpc_servicer.sglang.managed.encoder_service.time.monotonic", lambda: clock[0]
    )
    queue = FairQueue(10, 2.0, {"small": 2})
    large = SimpleNamespace(work=10000)
    queue.put("large", large)
    queue.put("small", SimpleNamespace(work=1))
    assert queue.pop().work == 1
    clock[0] = 3.0
    queue.put("small", SimpleNamespace(work=1))
    assert queue.pop() is large
    queue.pop()
    assert not queue.queues and not queue.deficits


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


def test_pull_before_encode_shares_bounded_waiter_for_native_tp_receivers():
    """Two native TP receivers can wait for the same part without duplicating admission."""
    from smg_grpc_servicer.sglang.managed.encoder_service import EncoderService

    async def scenario():
        service = object.__new__(EncoderService)
        service.instance = SimpleNamespace(generation="epoch")
        service.config = {"max_early_waiters": 2, "early_wait_seconds": 1}
        service.parts, service.early, service.early_count = {}, {}, 0
        first = asyncio.create_task(service.part("epoch.part"))
        second = asyncio.create_task(service.part("epoch.part"))
        await asyncio.sleep(0)
        assert service.early_count == 2 and len(service.early) == 1
        entry = SimpleNamespace(cancelled=False, quiesced=False, deadline=float("inf"))
        service.parts["epoch.part"] = entry
        service.early["epoch.part"][0].set()
        assert await first is entry and await second is entry
        assert not service.early and service.early_count == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("release_fails", [False, True])
def test_request_payload_released_only_after_native_quiescence(monkeypatch, release_fails):
    """Replay fences cannot retain uncharged media or failed-send traceback buffers."""
    import gc
    import sys
    import weakref
    from unittest.mock import AsyncMock

    from smg_grpc_servicer.sglang.managed.encoder_service import EncoderService, Entry

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.disaggregation.encoder.runtime",
        SimpleNamespace(execute_encode_pipeline=AsyncMock()),
    )
    metadata = SimpleNamespace(publish=AsyncMock())
    server = SimpleNamespace(meta_registry=metadata)
    monkeypatch.setitem(sys.modules, "sglang.srt.disaggregation.encoder.server", server)
    if parent := sys.modules.get("sglang.srt.disaggregation.encoder"):
        monkeypatch.setattr(parent, "server", server, raising=False)

    async def scenario():
        releasing, finish = asyncio.Event(), asyncio.Event()

        async def release(_rid):
            releasing.set()
            await finish.wait()
            if release_fails:
                raise RuntimeError("native release has not established quiescence")

        service = object.__new__(EncoderService)
        service.backend = SimpleNamespace(encoder=SimpleNamespace(release_request=release))
        service.cpu = asyncio.Semaphore(0)
        service.config = {"max_cpu": 1, "image_token_id": 1}
        service.cpu_target, service.cpu_active = 1, 1
        service.cpu_changed = asyncio.Event()
        service.execution = asyncio.BoundedSemaphore(1)
        await service.execution.acquire()
        service._event = AsyncMock()
        payload = SimpleNamespace(image_references=["x" * 1048576])
        entry = Entry(None, 1, float("inf"), "rid", request=payload, cancelled=True)
        entry.context_bytes = 1048576

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
        task = asyncio.create_task(service._run(entry, execution_held=True))
        await asyncio.wait_for(releasing.wait(), 2)
        assert entry.request is payload and entry.context_bytes == 1048576
        assert not entry.quiesced
        finish.set()
        await task
        assert entry.terminal
        gc.collect()
        assert service.execution._value == (0 if release_fails else 1)
        if release_fails:
            assert entry.request is payload and entry.context_bytes == 1048576
            assert not entry.quiesced
            assert references[0]() is not None
        else:
            assert entry.request is None and entry.context_bytes == 0
            assert entry.quiesced
            assert references[0]() is None

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
        revision=None,
    )


@pytest.mark.parametrize("role", ["encoder", "prefill", "decode"])
def test_minimal_config_derives_contract_and_finite_bounds(role):
    """All roles share the same model contract without copying TP/dtype/transport settings."""
    config = complete({"model_revision": "weights"}, native_args(), role)
    contract = config if role == "encoder" else config["contract"]
    assert contract["model_revision"] == "weights"
    assert contract["tp_size"] == 2
    assert contract["embedding_backend"] == "zmq_to_scheduler"
    assert contract["embedding_dtype"] == "bfloat16"
    assert config["max_records"] >= config["max_active"] > 0
    assert config["max_duration_ms"] >= 120000 + config["max_clock_error_ms"]
    if role == "encoder":
        assert config["control_listen"] == "0.0.0.0:43010"
        assert config["receive_endpoint"] == "http://10.0.0.1:43000"


@pytest.mark.parametrize("role", ["encoder", "prefill", "decode"])
@pytest.mark.parametrize("field,value", [("tp_size", 8), ("topology_contract", "single-host-tp8")])
def test_explicit_contract_cannot_misadvertise_native_settings(role, field, value):
    """A legacy explicit file must agree with the actual runtime."""
    config = {"model_revision": "weights"}
    if role == "encoder":
        config[field] = value
    else:
        config["contract"] = {field: value}
    with pytest.raises(ValueError, match="differs from native"):
        complete(config, native_args(), role)


def test_bounds_override_keeps_processor_concurrency_consistent():
    """One CPU limit controls both admission and the Rust preprocessing pool."""
    config = complete({"model_revision": "weights", "max_cpu": 4}, native_args(), "encoder")
    assert config["processor_options"]["max_inflight"] == 4
    with pytest.raises(ValueError, match="unknown managed"):
        complete({"model_revision": "weights", "max_cup": 4}, native_args(), "encoder")
    with pytest.raises(ValueError, match="pinned"):
        complete({}, native_args(), "prefill")


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
        complete({"model_revision": "weights"}, args, "prefill")


def test_encoder_nested_contract_cannot_override_native_tp():
    """Legacy nested contracts obey the same runtime fence as encoder top-level fields."""
    with pytest.raises(ValueError, match="differs from native"):
        complete(
            {"model_revision": "weights", "contract": {"tp_size": 8}}, native_args(), "encoder"
        )


@pytest.mark.parametrize("backend", ["mooncake", "zmq_to_scheduler"])
def test_receiver_fanout_preserves_memory_bound_and_advertised_capacity(backend):
    """Eight P ranks are allowed without overcommitting the E owner's context budget."""
    from smg_grpc_servicer.sglang.managed.encoder_service import EncoderService

    args = native_args()
    args.encoder_transfer_backend = backend
    config = complete({"model_revision": "weights"}, args, "encoder")
    config["generation"] = "generation"
    service = EncoderService(config, SimpleNamespace(scheduler=None), None)
    assert config["max_receivers"] == 8
    # Physical storage is shared only for native Mooncake GPU-direct sends.
    assert service.execution_capacity == (14 if backend == "mooncake" else 5)
    controls = service.capacity * config["max_control_bytes"]
    reserved = service.execution_bytes
    assert service.capacity == config["max_active"]
    assert service.execution_capacity < service.capacity
    assert controls + reserved * service.execution_capacity <= config["max_context_bytes"]
    assert controls + reserved * (service.execution_capacity + 1) > config["max_context_bytes"]
    assert config["max_early_waiters"] >= service.capacity * config["max_receivers"]
    assert config["max_http_connections"] > config["max_early_waiters"]
    config["max_context_bytes"] = controls + reserved - 1
    with pytest.raises(ValueError, match="cannot admit"):
        EncoderService(config, SimpleNamespace(scheduler=None), None)


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
    MMEncoder._stage_embeddings(native, ctx, requests, batch, keep_on_gpu=True)
    for item, expected in zip(staged, (batch[:2], batch[2:])):
        assert torch.equal(item.embedding, expected)
        assert item.embedding.data_ptr() != expected.data_ptr()
        assert item.embedding.untyped_storage().nbytes() == item.embedding.nbytes


def test_receiver_errors_retain_native_timeout_and_admission_semantics():
    """SGLang reads message and maps HTTP 504/429 to deadline/capacity failures."""
    import grpc
    from smg_grpc_servicer.sglang.managed.encoder_launch import receiver_error
    from smg_grpc_servicer.sglang.managed.encoder_service import Rejected

    for error, status in [
        (Rejected(grpc.StatusCode.DEADLINE_EXCEEDED, "expired"), 504),
        (Rejected(grpc.StatusCode.RESOURCE_EXHAUSTED, "full"), 429),
        (TimeoutError(), 504),
        (ValueError("invalid receiver"), 400),
    ]:
        body, actual_status = receiver_error(error)
        assert actual_status == status
        assert body["message"]


def test_queued_cancel_does_not_wait_for_or_release_a_native_execution_slot(monkeypatch):
    """Queued controls can retire while all tensor windows belong to other requests."""
    import sys
    from unittest.mock import AsyncMock

    from smg_grpc_servicer.sglang.managed.encoder_service import EncoderService, Entry

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.disaggregation.encoder.runtime",
        SimpleNamespace(execute_encode_pipeline=AsyncMock()),
    )
    metadata = SimpleNamespace(publish=AsyncMock())
    server = SimpleNamespace(meta_registry=metadata)
    monkeypatch.setitem(sys.modules, "sglang.srt.disaggregation.encoder.server", server)
    if parent := sys.modules.get("sglang.srt.disaggregation.encoder"):
        monkeypatch.setattr(parent, "server", server, raising=False)

    async def scenario():
        config = complete({"model_revision": "weights"}, native_args(), "encoder")
        config["generation"] = "generation"
        backend = SimpleNamespace(
            scheduler=None, encoder=SimpleNamespace(release_request=AsyncMock())
        )
        service = EncoderService(config, backend, None)
        service._event = AsyncMock()
        for _ in range(service.execution_capacity):
            await service.execution.acquire()
        entry = Entry(None, 1, float("inf"), "cancelled", request=object(), context_bytes=64)
        sibling = Entry(None, 1, float("inf"), "sibling")
        service.queue.put("tenant", entry)
        service.queue.put("tenant", sibling)
        waiting, failure = asyncio.Event(), asyncio.Event()
        metadata.publish.side_effect = lambda *_args, **_kwargs: failure.set()

        async def early_metadata_reader():
            async with entry.refs.retain():
                waiting.set()
                await failure.wait()

        reader = asyncio.create_task(early_metadata_reader())
        await waiting.wait()
        service._cancel(entry)
        await asyncio.wait_for(entry.task, 1)
        await reader
        metadata.publish.assert_awaited_once()
        assert entry.quiesced and entry.request is None and entry.context_bytes == 0
        assert service.execution._value == 0
        assert service.cpu._value == config["max_cpu"]
        assert service.queue.pop() is sibling
        assert not service.queue.queues and not service.queue.deficits
        backend.encoder.release_request.assert_awaited_once_with("cancelled")

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["metadata", "register"])
def test_cancelled_queued_encode_retires_early_receiver_control(monkeypatch, kind):
    """Metadata or registration waits cannot pin a cancelled admission until its full deadline."""
    import sys
    from unittest.mock import AsyncMock

    from smg_grpc_servicer.sglang.managed.encoder_service import EncoderService, Entry, Rejected

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
        service = object.__new__(EncoderService)
        service.config = {"max_receivers": 2}
        service.backend = SimpleNamespace(
            encoder=SimpleNamespace(
                transfer_backend="zmq_to_scheduler", register_embedding_destinations=register
            )
        )
        entry = Entry(
            None,
            1,
            float("inf"),
            "rid",
            request=SimpleNamespace(
                receiver_binding=SimpleNamespace(bootstrap=SimpleNamespace(host="10.0.0.2"))
            ),
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


def test_zmq_abandonment_wakes_native_wait_without_releasing_active_send(monkeypatch):
    """Cancellation requests native abandonment, but charges survive the actual send owner."""
    import sys
    from unittest.mock import AsyncMock, Mock

    from smg_grpc_proto.generated import sglang_managed_pb2 as pb
    from smg_grpc_proto.generated import sglang_scheduler_pb2 as sg
    from smg_grpc_servicer.sglang.managed.encoder_service import EncoderService, Entry

    async def scenario():
        started, abandoned, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def pipeline(*_args):
            started.set()
            await abandoned.wait()
            await finish.wait()  # Represents a send that already owns the native buffer.

        async def abandon(_rid):
            abandoned.set()

        monkeypatch.setitem(
            sys.modules,
            "sglang.srt.disaggregation.encoder.runtime",
            SimpleNamespace(execute_encode_pipeline=pipeline),
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
        service = object.__new__(EncoderService)
        service.backend = SimpleNamespace(encoder=encoder)
        service.config = {"image_token_id": 1}
        service.processor = SimpleNamespace(
            prepare=lambda *_args: sg.GenerateRequest().SerializeToString()
        )
        service.submitter = object()
        service.cpu = asyncio.Semaphore(0)
        service.config = {"max_cpu": 1, "image_token_id": 1}
        service.cpu_target, service.cpu_active = 1, 1
        service.cpu_changed = asyncio.Event()
        service.execution = asyncio.BoundedSemaphore(1)
        await service.execution.acquire()
        service._event = AsyncMock()
        entry = Entry(
            None, 1, float("inf"), "rid", request=SimpleNamespace(image_references=["image"])
        )
        entry.context_bytes = 1024
        entry.task = asyncio.create_task(service._run(entry, execution_held=True))
        await asyncio.wait_for(started.wait(), 1)
        service._cancel(entry)
        await asyncio.wait_for(abandoned.wait(), 1)
        assert entry.context_bytes == 1024 and not entry.quiesced
        encoder.release_request.assert_not_awaited()
        assert service.execution._value == 0
        finish.set()
        await asyncio.wait_for(entry.task, 1)
        encoder.release_request.assert_awaited_once_with("rid")
        encoder.clear_abandoned_request.assert_called_once_with("rid")
        assert not any(
            call.kwargs.get("stage") == pb.DELIVERED for call in service._event.call_args_list
        )
        assert entry.quiesced and entry.context_bytes == 0 and service.execution._value == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_artifact_materialization_keeps_control_loop_live_and_owns_cpu_work(cancel):
    """Control callbacks run during CPU work, and cancellation waits for its actual exit."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from smg_grpc_servicer.sglang.managed.encoder_bridge import RustPreprocessorBridge

    async def scenario():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        finish = threading.Event()
        bridge = RustPreprocessorBridge(SimpleNamespace(preproc_executor=executor), 1024)
        requests = [object()]
        result = object()

        def materialize(actual_requests, modality):
            assert actual_requests is requests
            assert threading.get_ident() != control_thread
            loop.call_soon_threadsafe(started.set)
            assert finish.wait(2), "CPU owner was not released by the control loop"
            return result

        control_thread = threading.get_ident()
        bridge._materialize_batch = materialize
        task = asyncio.create_task(bridge.process_batch_mm_items(requests, None))
        try:
            await asyncio.wait_for(started.wait(), 1)
            if cancel:
                task.cancel()
                await asyncio.sleep(0)
            assert not task.done()
        finally:
            finish.set()
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert await task is result

    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(scenario())


@pytest.mark.parametrize("batch_size", [2, 8])
@pytest.mark.parametrize("busy,pending,expected", [(False, 0, 3), (True, 0, 2), (False, 1, 2)])
def test_cpu_supply_grows_only_when_native_encoder_needs_input(busy, pending, expected, batch_size):
    """Backlogged CPU producers grow only at an idle native completion boundary."""
    import asyncio
    from unittest.mock import AsyncMock

    from smg_grpc_servicer.sglang.managed.encoder_service import EncoderService

    async def scenario():
        config = complete({"model_revision": "weights", "max_cpu": 4}, native_args(), "encoder")
        config["generation"] = "generation"
        queue, lock = asyncio.Queue(), asyncio.Lock()
        for _ in range(pending):
            queue.put_nowait(object())
        if busy:
            await lock.acquire()
        backend = SimpleNamespace(
            scheduler=SimpleNamespace(pending_queue=queue, max_batch_size=batch_size),
            encoder=SimpleNamespace(encode_dispatch_lock=lock),
        )
        service = EncoderService(config, backend, None)
        service._event = AsyncMock()
        service.parts["rid"] = SimpleNamespace(gpu_work=1)
        service.queue.put("tenant", object())
        for _ in range(2):
            await service.cpu.acquire()
            service.cpu_active += 1
        await service._encoded("rid")
        assert service.cpu_target == expected
        # Multiple request callbacks from the same native batch cannot multiply growth.
        await service._encoded("rid")
        assert service.cpu_target == expected
        assert service.cpu_active == 2
        # Backpressure changes future grants; the running owners remain charged.
        queue.put_nowait(object())
        queue.put_nowait(object())
        service._release_cpu()
        assert service.cpu_target == 2
        assert service.cpu_active == 1 and service.cpu._value == 3
        service._release_cpu()
        assert service.cpu_active == 0 and service.cpu._value == 4

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

    config = complete({"model_revision": "weights"}, native_args(), "encoder")
    config["processor_options"]["max_inflight"] = processor_limit
    with pytest.raises(ValueError, match="positive integer"):
        encoder_cpu_limit(config, 8)


def test_adaptive_cpu_limit_intersects_native_memory_and_host_bounds(monkeypatch):
    """An explicit ceiling cannot overrun actual CPU or retained tensor capacity."""
    from smg_grpc_servicer.sglang.managed import config as settings

    config = complete({"model_revision": "weights", "max_cpu": 32}, native_args(), "encoder")
    monkeypatch.setattr(settings, "available_cpus", lambda: 16)
    assert settings.encoder_cpu_limit(config, 1) == 2
    assert settings.encoder_cpu_limit(config, 32) == settings.encoder_window(config)[0]
    monkeypatch.setattr(settings, "available_cpus", lambda: 1)
    assert settings.encoder_cpu_limit(config, 8) == 1
