"""CPU ownership regressions; these are not GPU/TP or transport acceptance tests."""

import asyncio
from types import SimpleNamespace

import pytest
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
