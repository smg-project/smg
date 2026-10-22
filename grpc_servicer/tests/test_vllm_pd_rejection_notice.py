"""The PD decode side's pre-admission notice stops once the engine owns the request.

A decode request that dies before the engine received it owes its prefill side a
notice (``notify_kv_transfer_request_rejected``: a second, pre-aborted engine
request under the same id, so the connector's ``request_finished`` frees the
prefill's blocks). A caller that gives up after the request was handed to the
engine is answered by the engine's own abort; sending the notice then kills the
engine either way: an aborted request whose remote KV is still being pulled
stays tracked until the pull completes and the duplicate id is refused, and once
the pull is done the notice's own request is freed as it is admitted, so the
connector's report of its empty pull names an id the scheduler no longer tracks.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("vllm")
torch = pytest.importorskip("torch")
from smg_grpc_proto import vllm_engine_pb2  # noqa: E402
from smg_grpc_proto.generated import common_pb2  # noqa: E402
from smg_grpc_servicer.vllm.servicer import VllmEngineServicer  # noqa: E402

DECODE_KV = json.dumps(
    {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "remote_engine_id": "prefill-engine",
        "remote_bootstrap_addr": "127.0.0.1:1",
        "transfer_id": "xfer-1",
    }
)


class _Engine:
    """An engine whose generate() admits the request and then produces nothing."""

    def __init__(self):
        self.entered = asyncio.Event()
        self.notify_kv_transfer_request_rejected = AsyncMock()
        self.renderer = SimpleNamespace(process_for_engine=lambda prompt, **kwargs: prompt)
        self.model_config = SimpleNamespace(
            is_multimodal_model=False, dtype=torch.float32, uses_mrope=False
        )
        self.vllm_config = SimpleNamespace(kv_events_config=None)

    async def generate(self, *, prompt, sampling_params, request_id, **kwargs):
        self.entered.set()
        await asyncio.Event().wait()  # the KV pull never completes
        yield  # pragma: no cover - never reached; makes this an async generator


class _HangingProcessor:
    """Media processing that never finishes: the request dies before the engine."""

    def __init__(self):
        self.entered = asyncio.Event()
        self.accepted_schemes = {"https"}

    async def probe(self):
        return None

    async def process(self, *args, **kwargs):
        self.entered.set()
        await asyncio.Event().wait()


def _servicer(monkeypatch, engine, processor=None):
    monkeypatch.delenv("SMG_VLLM_MM_PROCESSOR", raising=False)
    servicer = VllmEngineServicer(engine, start_time=0.0)
    if processor is not None:
        servicer._mm_processor = processor
    return servicer


def _decode_request(*, refs=False):
    request = vllm_engine_pb2.GenerateRequest(
        request_id="req-decode",
        tokenized=vllm_engine_pb2.TokenizedInput(input_ids=[7, 8, 9]),
        sampling_params=vllm_engine_pb2.SamplingParams(max_tokens=16),
        kv_transfer_params_json=DECODE_KV,
    )
    if refs:
        request.media_refs.items.append(
            vllm_engine_pb2.MediaRef(modality=common_pb2.IMAGE, url="https://a/1.png")
        )
    return request


async def _cancel_after(servicer, request, entered: asyncio.Event):
    context = SimpleNamespace(abort=AsyncMock(), invocation_metadata=lambda: ())

    async def drive():
        async for _ in servicer.Generate(request, context):
            pass

    task = asyncio.create_task(drive())
    await asyncio.wait_for(entered.wait(), timeout=5.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_a_caller_that_gives_up_after_admission_sends_no_notice(monkeypatch):
    engine = _Engine()
    servicer = _servicer(monkeypatch, engine)
    await _cancel_after(servicer, _decode_request(), engine.entered)
    engine.notify_kv_transfer_request_rejected.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_caller_that_gives_up_before_the_engine_still_sends_the_notice(monkeypatch):
    engine = _Engine()
    processor = _HangingProcessor()
    servicer = _servicer(monkeypatch, engine, processor)
    await _cancel_after(servicer, _decode_request(refs=True), processor.entered)
    assert not engine.entered.is_set()
    engine.notify_kv_transfer_request_rejected.assert_awaited_once()
    (request_id, params), _ = engine.notify_kv_transfer_request_rejected.await_args
    assert request_id == "req-decode"
    assert params["transfer_id"] == "xfer-1"
