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

A refusal ``generate()`` raises before the engine core holds the request (vLLM's
``add_request`` raises a ``VLLMClientError`` for the request's inputs, a
``ValueError`` before 0.27, and ``generate()`` re-raises it without an abort)
leaves no live id behind and still gets the notice, as the Rust wire sends it
when its generate call returns an error.

Where a Mooncake connector runs, alone or as any child of a wrapper, the notice
is withheld altogether, a legitimate pre-admission one included (that connector
reports the notice's empty pull as a finished transfer and the decode engine
dies on it); NIXL keeps it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import grpc
import pytest

pytest.importorskip("vllm")
torch = pytest.importorskip("torch")
from smg_grpc_proto import vllm_engine_pb2  # noqa: E402
from smg_grpc_proto.generated import common_pb2  # noqa: E402
from smg_grpc_servicer.vllm.servicer import VllmEngineServicer  # noqa: E402
from vllm.v1.engine.exceptions import EngineGenerateError  # noqa: E402

try:
    from vllm.exceptions import VLLMValidationError as RefusalError
except ImportError:  # vLLM < 0.27 raises ValueError for request validation
    RefusalError = ValueError

DECODE_KV = json.dumps(
    {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "remote_engine_id": "prefill-engine",
        "remote_bootstrap_addr": "127.0.0.1:1",
        "transfer_id": "xfer-1",
    }
)


def _kv_transfer_config(connector: str, **extra):
    """A decode side's ``KVTransferConfig`` as the servicer reads it."""
    return SimpleNamespace(
        kv_connector=connector,
        kv_role="kv_consumer",
        engine_id="decode-engine",
        kv_connector_extra_config=extra or None,
    )


class _Engine:
    """An engine whose generate() admits the request and then produces nothing."""

    def __init__(self, kv_transfer_config=None):
        self.entered = asyncio.Event()
        self.notify_kv_transfer_request_rejected = AsyncMock()
        self.renderer = SimpleNamespace(process_for_engine=lambda prompt, **kwargs: prompt)
        self.model_config = SimpleNamespace(
            is_multimodal_model=False, dtype=torch.float32, uses_mrope=False
        )
        self.vllm_config = SimpleNamespace(
            kv_events_config=None, kv_transfer_config=kv_transfer_config
        )

    async def generate(self, *, prompt, sampling_params, request_id, **kwargs):
        self.entered.set()
        await asyncio.Event().wait()  # the KV pull never completes
        yield  # pragma: no cover - never reached; makes this an async generator


class _RefusingEngine(_Engine):
    """An engine whose generate() raises before its first output."""

    def __init__(self, error: Exception):
        super().__init__()
        self.error = error

    async def generate(self, *, prompt, sampling_params, request_id, **kwargs):
        self.entered.set()
        raise self.error
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


async def _drive_to_the_end(servicer, request):
    context = SimpleNamespace(abort=AsyncMock(), invocation_metadata=lambda: ())
    async for _ in servicer.Generate(request, context):
        pass
    return context


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


@pytest.mark.asyncio
async def test_a_request_the_engine_refuses_before_its_first_output_still_sends_the_notice(
    monkeypatch,
):
    # vLLM's add_request raises the validation error (VLLMValidationError, a
    # VLLMClientError; the ValueError of releases before 0.27) before the
    # engine core holds the request and generate() re-raises it without an
    # abort: no live id is left behind, so the notice is owed as for any
    # pre-admission failure.
    engine = _RefusingEngine(RefusalError("prompt plus max_tokens over the model length"))
    servicer = _servicer(monkeypatch, engine)
    context = await _drive_to_the_end(servicer, _decode_request())
    assert engine.entered.is_set()
    engine.notify_kv_transfer_request_rejected.assert_awaited_once()
    (request_id, params), _ = engine.notify_kv_transfer_request_rejected.await_args
    assert request_id == "req-decode"
    assert params["transfer_id"] == "xfer-1"
    context.abort.assert_awaited_once()
    assert context.abort.await_args.args[0] is grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_an_engine_failure_after_admission_sends_no_notice(monkeypatch):
    # Any other failure of generate() comes after the engine's own abort of
    # the request (vLLM raises it as EngineGenerateError), and the id may
    # still be tracked while its KV pull completes: the notice stays off.
    engine = _RefusingEngine(EngineGenerateError())
    servicer = _servicer(monkeypatch, engine)
    context = await _drive_to_the_end(servicer, _decode_request())
    assert engine.entered.is_set()
    engine.notify_kv_transfer_request_rejected.assert_not_awaited()
    context.abort.assert_awaited_once()
    assert context.abort.await_args.args[0] is grpc.StatusCode.INTERNAL


@pytest.mark.asyncio
async def test_a_nixl_decode_side_still_sends_the_pre_admission_notice(monkeypatch):
    engine = _Engine(_kv_transfer_config("NixlConnector"))
    processor = _HangingProcessor()
    servicer = _servicer(monkeypatch, engine, processor)
    await _cancel_after(servicer, _decode_request(refs=True), processor.entered)
    engine.notify_kv_transfer_request_rejected.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_mooncake_decode_side_withholds_the_pre_admission_notice(monkeypatch, caplog):
    engine = _Engine(_kv_transfer_config("MooncakeConnector"))
    processor = _HangingProcessor()
    servicer = _servicer(monkeypatch, engine, processor)
    with caplog.at_level(logging.WARNING):
        await _cancel_after(servicer, _decode_request(refs=True), processor.entered)
    assert not engine.entered.is_set()
    engine.notify_kv_transfer_request_rejected.assert_not_awaited()
    withheld = [r for r in caplog.records if "withheld on the Mooncake connector" in r.getMessage()]
    assert len(withheld) == 1, caplog.text
    assert "req-decode" in withheld[0].getMessage()


@pytest.mark.asyncio
async def test_a_mooncake_decode_side_withholds_the_notice_for_a_refused_request(monkeypatch):
    # The refusal generate() raises before holding the request keeps the
    # notice on every other connector; on Mooncake that notice is fatal too.
    engine = _RefusingEngine(RefusalError("prompt plus max_tokens over the model length"))
    engine.vllm_config.kv_transfer_config = _kv_transfer_config("MooncakeConnector")
    servicer = _servicer(monkeypatch, engine)
    context = await _drive_to_the_end(servicer, _decode_request())
    assert engine.entered.is_set()
    engine.notify_kv_transfer_request_rejected.assert_not_awaited()
    context.abort.assert_awaited_once()
    assert context.abort.await_args.args[0] is grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "connectors",
    [
        # the projection's mover
        [
            {"kv_connector": "MooncakeConnector", "engine_id": "decode-engine"},
            {"kv_connector": "SharedStorageConnector"},
        ],
        # a second P2P child beside it: the projection stays at the wrapper
        [
            {"kv_connector": "NixlConnector", "engine_id": "decode-engine"},
            {"kv_connector": "MooncakeConnector", "engine_id": "decode-engine"},
        ],
        # an empty child engine id: the projection stays at the wrapper
        [{"kv_connector": "MooncakeConnector", "engine_id": ""}],
        # a malformed sibling: the projection stays at the wrapper
        [{"kv_connector": "MooncakeConnector", "engine_id": "decode-engine"}, "not a child"],
    ],
    ids=["mover", "second-p2p-child", "empty-engine-id", "malformed-sibling"],
)
async def test_a_wrapper_with_a_mooncake_child_withholds_the_notice_under_any_projection(
    monkeypatch, connectors
):
    engine = _Engine(_kv_transfer_config("MultiConnector", connectors=connectors))
    processor = _HangingProcessor()
    servicer = _servicer(monkeypatch, engine, processor)
    await _cancel_after(servicer, _decode_request(refs=True), processor.entered)
    engine.notify_kv_transfer_request_rejected.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_wrapper_without_a_mooncake_child_still_sends_the_notice(monkeypatch):
    wrapper = _kv_transfer_config(
        "MultiConnector",
        connectors=[
            {"kv_connector": "NixlConnector", "engine_id": "decode-engine"},
            {"kv_connector": "SharedStorageConnector"},
        ],
    )
    engine = _Engine(wrapper)
    processor = _HangingProcessor()
    servicer = _servicer(monkeypatch, engine, processor)
    await _cancel_after(servicer, _decode_request(refs=True), processor.entered)
    engine.notify_kv_transfer_request_rejected.assert_awaited_once()
