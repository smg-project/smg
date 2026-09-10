"""Received-event conversion and actual subscription wiring, without vLLM."""

import ast
import asyncio
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import grpc
import pytest
from smg_grpc_proto import vllm_engine_pb2
from smg_grpc_proto.generated import common_pb2
from smg_grpc_servicer.kv_events import endpoint_for_rank, stream_kv_events

_ROOT = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm"
_SPEC = importlib.util.spec_from_file_location("kv_group_events", _ROOT / "kv_group_events.py")
module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(module)


def event(name="BlockStored", **overrides):
    fields = dict(
        block_hashes=[bytes(range(32))],
        parent_block_hash=None,
        group_idx=0,
        block_size=4,
        token_ids=[1, 2, 3, 4],
        medium="GPU",
        lora_id=None,
        lora_name=None,
        locality=None,
        ownership=None,
        extra_keys=[None],
        kv_cache_spec_kind="full_attention",
        kv_cache_spec_sliding_window=None,
    )
    fields.update(overrides)
    return type(name, (), fields)()


def batch(events, **overrides):
    raw = NS(ts=3.5, events=events, data_parallel_rank=None)
    raw.__dict__.update(overrides)
    return module.GroupEventConverter().convert_batch(raw, 7, 10)


def test_schema_selects_group_conversion_without_engine_config():
    supported = type("BlockStored", (), {"__struct_fields__": tuple(module._GROUP_FIELDS)})
    assert isinstance(module.resolve_group_event_converter(supported), module.GroupEventConverter)
    assert module.resolve_group_event_converter(type("BlockStored", (), {})) is None
    for missing in module._GROUP_FIELDS:
        older = type(
            "BlockStored", (), {"__struct_fields__": tuple(module._GROUP_FIELDS - {missing})}
        )
        assert module.resolve_group_event_converter(older) is None


@pytest.mark.parametrize("rank", [None, 0, 2])
def test_batch_preserves_sequence_timestamp_rank_and_native_keys(rank):
    hashes = [0, 2**63, 2**64 - 1, bytes(range(32)), b"x" * 16]
    converted, event_id = batch(
        [event(block_hashes=hashes, extra_keys=None)], data_parallel_rank=rank
    )
    assert converted.sequence_number == 7
    assert converted.timestamp == 3.5
    assert event_id == 11
    assert converted.group_events_enabled
    assert not converted.events
    assert converted.HasField("dp_rank") == (rank is not None)
    if rank is not None:
        assert converted.dp_rank == rank
    assert list(converted.group_events[0].block_hashes) == [
        (0).to_bytes(8, "big"),
        (2**63).to_bytes(8, "big"),
        (2**64 - 1).to_bytes(8, "big"),
        bytes(range(32)),
        b"x" * 16,
    ]
    assert common_pb2.KvEventBatch.FromString(converted.SerializeToString()) == converted


def test_sparse_span_unknown_parent_and_partial_store_are_not_sliced():
    hashes = [b"a" * 32, b"b" * 32]
    tokens = list(range(20))
    converted, _ = batch(
        [
            event(
                block_hashes=hashes, token_ids=tokens, extra_keys=[None, None], parent_block_hash=9
            ),
            event(group_idx=8, kv_cache_spec_kind="mamba", block_size=2, token_ids=[20, 21]),
        ]
    )
    sparse, partial = converted.group_events
    assert list(sparse.block_hashes) == hashes
    assert list(sparse.token_ids) == tokens
    assert sparse.parent_block_hash == (9).to_bytes(8, "big")
    assert sparse.tokens_matchable
    assert partial.group_id == 8
    assert partial.kind == "mamba"
    assert partial.block_size == 2
    assert list(partial.token_ids) == [20, 21]


@pytest.mark.parametrize(
    "kind,window", [("full_attention", None), ("sliding_window", 1024), ("future", 17)]
)
def test_received_group_metadata_is_preserved(kind, window):
    converted, _ = batch(
        [event(group_idx=73, kv_cache_spec_kind=kind, kv_cache_spec_sliding_window=window)]
    )
    stored = converted.group_events[0]
    assert stored.operation == common_pb2.KvGroupEvent.STORE
    assert (stored.group_id, stored.kind, stored.sliding_window) == (73, kind, window or 0)


def test_empty_sparse_store_still_preserves_group_metadata():
    converted, _ = batch([event(block_hashes=[], extra_keys=[])])
    stored = converted.group_events[0]
    assert stored.operation == common_pb2.KvGroupEvent.STORE
    assert not stored.block_hashes
    assert list(stored.token_ids) == [1, 2, 3, 4]
    assert stored.kind == "full_attention"


def test_repeated_reports_and_remove_clear_preserve_order():
    events = [event(), event(), event("BlockRemoved"), event("AllBlocksCleared")]
    converted, event_id = batch(events)
    assert [e.operation for e in converted.group_events] == [1, 1, 2, 3]
    assert converted.group_events[0] == converted.group_events[1]
    assert batch(events)[0] == converted
    assert event_id == 14


@pytest.mark.parametrize(
    "extra",
    [
        dict(medium="CPU"),
        dict(medium="STORAGE"),
        dict(medium=None),
        dict(locality="REMOTE"),
        dict(ownership="offload"),
    ],
)
def test_nonlocal_events_are_filtered_symmetrically(extra):
    converted, event_id = batch([event(), event(**extra), event("BlockRemoved", **extra)])
    assert [e.operation for e in converted.group_events] == [common_pb2.KvGroupEvent.STORE]
    assert event_id == 13
    filtered, _ = batch([event(**extra)])
    assert filtered.group_events_enabled and not filtered.group_events


@pytest.mark.parametrize(
    "extra",
    [
        dict(lora_id=0),
        dict(lora_name="adapter"),
        dict(extra_keys=[("image-hash",)]),
        dict(extra_keys=[("cache-salt",)]),
        dict(extra_keys=[("embedding-hash",)]),
    ],
)
def test_non_text_identity_is_retained_without_token_matching(extra):
    converted, _ = batch([event(**extra)])
    stored = converted.group_events[0]
    assert stored.operation == common_pb2.KvGroupEvent.STORE
    assert not stored.tokens_matchable
    assert list(stored.block_hashes) == [bytes(range(32))]
    assert list(stored.token_ids) == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "extra",
    [
        dict(group_idx=None),
        dict(group_idx=-1),
        dict(group_idx=2**32),
        dict(block_hashes=[-1]),
        dict(block_hashes=[2**64]),
        dict(block_hashes=[True]),
        dict(block_hashes=[b""]),
        dict(block_hashes=b"bad"),
        dict(parent_block_hash=b""),
        dict(block_size=0),
        dict(token_ids=[-1]),
        dict(token_ids=[True]),
        dict(extra_keys=[]),
        dict(extra_keys=[1]),
        dict(kv_cache_spec_kind=None),
        dict(kv_cache_spec_sliding_window=-1),
    ],
)
def test_malformed_event_invalidates_without_legacy_fallback(extra):
    converted, _ = batch([event(**extra)])
    assert converted.group_events[0].operation == common_pb2.KvGroupEvent.INVALID
    assert not converted.events


def test_unknown_event_and_missing_required_metadata_invalidate():
    for raw in (event("FutureEvent"), type("BlockStored", (), {})()):
        converted, _ = batch([raw])
        assert converted.group_events[0].operation == common_pb2.KvGroupEvent.INVALID


def _servicer_method(name, **namespace):
    """Execute the actual method with engine imports replaced by boundary fakes."""
    source = ast.parse((_ROOT / "servicer.py").read_text())
    cls = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "VllmEngineServicer"
    )
    method = next(node for node in cls.body if getattr(node, "name", None) == name)
    parsed = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            method,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(parsed)
    namespace.update(
        asyncio=asyncio,
        grpc=grpc,
        logger=logging.getLogger(__name__),
        common_pb2=common_pb2,
        vllm_engine_pb2=vllm_engine_pb2,
    )
    exec(compile(parsed, str(_ROOT / "servicer.py"), "exec"), namespace)
    return namespace[name]


def test_servicer_initialization_does_not_need_group_metadata():
    config = object()
    method = _servicer_method(
        "__init__",
        resolve_kv_events_config=lambda _: config,
        resolve_group_event_converter=Mock(side_effect=AssertionError("eager schema lookup")),
    )
    servicer, engine = NS(), object()
    method(servicer, engine, 0.0)
    assert servicer.engine is engine
    assert servicer._kv_events_config is config


@pytest.mark.parametrize("group_schema,bad_final", [(False, False), (True, False), (True, True)])
@pytest.mark.asyncio
async def test_actual_subscription_selects_schema_and_closes_socket(group_schema, bad_final):
    fields = tuple(module._GROUP_FIELDS) if group_schema else ()
    stored_type = type("BlockStored", (), {"__struct_fields__": fields})
    socket = NS(
        subscribe=Mock(),
        connect=Mock(),
        close=Mock(),
        poll=AsyncMock(return_value=True),
        recv_multipart=AsyncMock(side_effect=[[b"", (4).to_bytes(8, "big"), b"payload"], []]),
    )
    context = NS(
        send_initial_metadata=AsyncMock(),
        cancelled=lambda: False,
        abort=AsyncMock(side_effect=RuntimeError("stream aborted")),
    )
    raw = NS(ts=1.0, events=[event()], data_parallel_rank=2)
    method = _servicer_method(
        "SubscribeKvEvents",
        BlockStored=stored_type,
        KVEventBatch=object,
        endpoint_for_rank=endpoint_for_rank,
        resolve_group_event_converter=module.resolve_group_event_converter,
        stream_kv_events=stream_kv_events,
        msgspec=NS(msgpack=NS(Decoder=lambda _: NS(decode=lambda _: raw))),
        zmq=NS(SUB=1, asyncio=NS(Context=NS(instance=lambda: NS(socket=lambda _: socket)))),
    )

    stream = method(
        NS(_kv_events_config=NS(endpoint="tcp://*:5557", topic="kv")),
        common_pb2.SubscribeKvEventsRequest(),
        context,
    )
    converted = await anext(stream)
    if bad_final:
        with pytest.raises(RuntimeError, match="stream aborted"):
            await anext(stream)
    await stream.aclose()
    assert converted.sequence_number == 4
    assert converted.dp_rank == 2
    assert converted.group_events_enabled == group_schema
    assert bool(converted.group_events) == group_schema
    assert bool(converted.events) != group_schema
    socket.close.assert_called_once_with(linger=0)
    context.send_initial_metadata.assert_awaited_once_with(())
    if bad_final:
        context.abort.assert_awaited_once_with(
            grpc.StatusCode.INTERNAL, "Malformed KV event multipart frame"
        )
    else:
        context.abort.assert_not_awaited()


@pytest.mark.parametrize(
    "bad",
    [[], [b"", b"short", b"bad"], [b"", bytes(8), b"bad"], [b"", bytes(8), b"good", b"extra"]],
)
@pytest.mark.asyncio
async def test_group_stream_aborts_on_final_corrupt_frame(bad):
    frames = iter([[b"", bytes(8), b"good"], bad])
    socket = NS(
        poll=AsyncMock(return_value=True),
        recv_multipart=AsyncMock(side_effect=lambda: next(frames)),
    )

    def decode(payload):
        if payload != b"good":
            raise ValueError("corrupt payload")
        return NS(ts=0, events=[event()], data_parallel_rank=None)

    stream = stream_kv_events(
        socket,
        decode,
        AsyncMock(),
        lambda: False,
        convert=module.GroupEventConverter().convert_batch,
        strict=True,
    )
    assert (await anext(stream)).group_events[0].operation == common_pb2.KvGroupEvent.STORE
    with pytest.raises(ValueError):
        await anext(stream)
    await stream.aclose()


@pytest.mark.asyncio
async def test_generate_does_not_reject_or_modify_inference_for_group_events():
    sampling = NS(
        logprobs=None,
        prompt_logprobs=0,
        skip_reading_prefix_cache=True,
        extra_args={"kv_cache_report_mode": "full"},
    )
    received = []

    async def generate(**kwargs):
        received.append(kwargs)
        if False:
            yield

    request = vllm_engine_pb2.GenerateRequest(request_id="text-with-mm-identity", stream=False)
    request.tokenized.input_ids.extend([1, 2])
    request.mm_inputs.mm_hashes.append("image")
    transfer = {"do_remote_prefill": True}
    method = _servicer_method(
        "Generate",
        time=NS(time=lambda: 1),
        params_from_request=lambda _: transfer,
        has_preprocessed_mm_payload=lambda _: False,
        mm_identity_cache_salt=lambda _: "mm:image",
    )
    servicer = NS(
        engine=NS(generate=generate, renderer=NS(process_for_engine=lambda prompt, **_: prompt)),
        _sampling_params_from_proto=lambda *_, **__: sampling,
        _tokenization_kwargs_from_proto=lambda _: None,
        _notify_kv_transfer_rejected=AsyncMock(),
    )

    results = [
        item
        async for item in method(
            servicer,
            request,
            NS(abort=AsyncMock(side_effect=AssertionError("inference rejected"))),
        )
    ]
    assert results == []
    assert received[0]["sampling_params"] is sampling
    assert sampling.extra_args == {"kv_cache_report_mode": "full"}
    assert received[0]["prompt"]["cache_salt"] == "mm:image"


@pytest.mark.asyncio
async def test_embed_remains_available_without_group_contract():
    async def encode(**kwargs):
        yield NS(
            finished=True, outputs=NS(data=NS(tolist=lambda: [0.1, 0.2])), prompt_token_ids=[1, 2]
        )

    request = vllm_engine_pb2.EmbedRequest(request_id="embed")
    request.tokenized.input_ids.extend([1, 2])
    method = _servicer_method(
        "Embed", tokens_input=lambda **kwargs: kwargs, PoolingParams=lambda **kwargs: NS(**kwargs)
    )
    result = await method(
        NS(engine=NS(encode=encode)),
        request,
        NS(abort=AsyncMock(side_effect=AssertionError("embedding rejected"))),
    )
    assert result.embedding_dim == 2 and result.prompt_tokens == 2
