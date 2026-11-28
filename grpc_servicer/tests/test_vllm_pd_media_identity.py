"""PD sides and worker-side media (needs vLLM: runs on the vLLM e2e lane).

The prefill side processes the references once and answers with their
identity; a regular request answers with none; the decode side, given that
identity, never touches the media processor.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("vllm")
torch = pytest.importorskip("torch")
from smg_grpc_proto import vllm_engine_pb2  # noqa: E402
from smg_grpc_proto.generated import common_pb2  # noqa: E402
from smg_grpc_servicer.vllm.servicer import VllmEngineServicer  # noqa: E402
from vllm.outputs import CompletionOutput, RequestOutput  # noqa: E402

# Every case builds the Python servicer directly: the explicit choice its
# constructor requires (conftest).
pytestmark = pytest.mark.usefixtures("the_python_vllm_servicer_chosen")

PREFILL_KV = json.dumps({"do_remote_decode": True, "do_remote_prefill": False})


class _Processor:
    """Stands in for the worker-side media processor; records its calls."""

    name = "inprocess"
    max_inflight = 4
    schemes = "http,https,data"
    accepted_schemes = {"http", "https", "data"}

    def __init__(self):
        self.calls = []

    async def probe(self):
        return True

    async def process(self, prompt_token_ids, prompt_text, items, arrival_time, *, request_id=""):
        self.calls.append(request_id)
        grid = SimpleNamespace(
            data=torch.tensor([1, 4, 6]), field=SimpleNamespace(keep_on_cpu=True)
        )
        return {
            "prompt_token_ids": [7, 100, 100, 100, 9],
            "mm_hashes": {"image": ["h1"]},
            "mm_placeholders": {"image": [SimpleNamespace(offset=1, length=3, is_embed=None)]},
            "mm_kwargs": {
                "image": [
                    {
                        "pixel_values": SimpleNamespace(data=torch.zeros(3, 2, 2)),
                        "image_grid_thw": grid,
                    }
                ]
            },
            "arrival_time": arrival_time,
        }


def _engine(seen_prompts):
    async def generate(*, prompt, sampling_params, request_id, **kwargs):
        seen_prompts.append(prompt)
        yield RequestOutput(
            request_id=request_id,
            prompt=None,
            prompt_token_ids=[7, 100, 100, 100, 9],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="",
                    token_ids=[42],
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason="length",
                )
            ],
            finished=True,
            kv_transfer_params={"remote_block_ids": [1]},
        )

    return SimpleNamespace(
        generate=generate,
        renderer=SimpleNamespace(process_for_engine=lambda prompt, **kwargs: prompt),
        model_config=SimpleNamespace(
            is_multimodal_model=True, dtype=torch.float32, uses_mrope=True
        ),
        vllm_config=SimpleNamespace(kv_events_config=None),
    )


def _servicer(monkeypatch, processor):
    monkeypatch.delenv("SMG_VLLM_MM_PROCESSOR", raising=False)
    servicer = VllmEngineServicer(_engine([]), start_time=0.0)
    servicer._mm_processor = processor
    return servicer


def _request(*, refs=False, kv=False, identity=None):
    request = vllm_engine_pb2.GenerateRequest(
        request_id="req-pd",
        tokenized=vllm_engine_pb2.TokenizedInput(input_ids=[7, 8, 9]),
        sampling_params=vllm_engine_pb2.SamplingParams(max_tokens=1),
    )
    if refs:
        request.media_refs.items.append(
            vllm_engine_pb2.MediaRef(modality=common_pb2.IMAGE, url="https://a/1.png")
        )
    if kv:
        request.kv_transfer_params_json = PREFILL_KV
    if identity is not None:
        request.tokenized.input_ids[:] = identity.prompt_token_ids
        request.mm_inputs.CopyFrom(identity.mm_inputs)
    return request


async def _complete(servicer, request):
    context = SimpleNamespace(abort=AsyncMock(), invocation_metadata=lambda: ())
    responses = [r async for r in servicer.Generate(request, context)]
    context.abort.assert_not_awaited()
    (final,) = [r for r in responses if r.HasField("complete")]
    return final.complete


@pytest.mark.asyncio
async def test_prefill_side_answers_with_the_media_identity(monkeypatch):
    processor = _Processor()
    servicer = _servicer(monkeypatch, processor)
    complete = await _complete(servicer, _request(refs=True, kv=True))
    assert processor.calls == ["req-pd"], "the media was processed once"
    assert complete.HasField("media_identity")
    identity = complete.media_identity
    assert list(identity.prompt_token_ids) == [7, 100, 100, 100, 9]
    assert list(identity.mm_inputs.mm_hashes) == ["h1"]
    assert [(p.offset, p.length) for p in identity.mm_inputs.mm_placeholders] == [(1, 3)]
    assert "image_grid_thw" in identity.mm_inputs.model_specific_tensors
    assert not identity.mm_inputs.HasField("pixel_values")


@pytest.mark.asyncio
async def test_a_regular_request_answers_with_no_identity(monkeypatch):
    processor = _Processor()
    servicer = _servicer(monkeypatch, processor)
    complete = await _complete(servicer, _request(refs=True, kv=False))
    assert processor.calls == ["req-pd"]
    assert not complete.HasField("media_identity")


@pytest.mark.asyncio
async def test_decode_side_from_identity_never_touches_the_processor(monkeypatch):
    processor = _Processor()
    servicer = _servicer(monkeypatch, processor)
    prefill = await _complete(servicer, _request(refs=True, kv=True))
    processor.calls.clear()
    decode = await _complete(servicer, _request(kv=True, identity=prefill.media_identity))
    assert processor.calls == [], "the decode side carried identity, not references"
    assert not decode.HasField("media_identity")


@pytest.mark.asyncio
@pytest.mark.parametrize("kv", [False, True])
@pytest.mark.parametrize("producer", [False, True])
@pytest.mark.parametrize("connector", ["ECCPUConnector", "ECMooncakeConnector"])
async def test_ec_consumer_accepts_metadata_without_a_kv_handoff(
    monkeypatch, kv, producer, connector
):
    """An ec_both prefill consumes E's handles without having to publish new ones."""
    processor = _Processor()
    servicer = _servicer(monkeypatch, processor)
    prefill = await _complete(servicer, _request(refs=True, kv=True))
    processor.calls.clear()
    servicer.engine.vllm_config.ec_transfer_config = SimpleNamespace(
        is_ec_consumer=True, is_ec_producer=producer, ec_connector=connector
    )
    request = _request(identity=prefill.media_identity, kv=kv)
    request.ec_transfer_params_json = json.dumps(
        {"ec_items": [{"mm_hash": "h1", "transfer_id": "transfer-1"}]}
        if connector == "ECMooncakeConnector"
        else {"h1": {"peer_host": "encoder", "peer_port": 9000}}
    )
    generate = servicer.engine.generate

    async def capture(**kwargs):
        assert kwargs["sampling_params"].extra_args["ec_transfer_params"] == json.loads(
            request.ec_transfer_params_json
        )
        async for output in generate(**kwargs):
            yield output

    servicer.engine.generate = capture
    complete = await _complete(servicer, request)
    assert list(complete.output_ids) == [42]
    assert processor.calls == []


@pytest.mark.asyncio
async def test_encoder_only_returns_handles_without_output_tokens(monkeypatch):
    servicer = _servicer(monkeypatch, _Processor())
    handles = {"h1": {"metadata": {"image_grid_thw": [1, 4, 6]}, "transfer_id": "t1"}}

    async def generate(**kwargs):
        yield RequestOutput(
            request_id=kwargs["request_id"],
            prompt=None,
            prompt_token_ids=[7, 100, 100, 100, 9],
            prompt_logprobs=None,
            outputs=[],
            finished=True,
            ec_transfer_params=handles,
        )

    servicer.engine.generate = generate
    request = _request(refs=True)
    request.ec_transfer_params_json = "{}"
    complete = await _complete(servicer, request)
    assert complete.finish_reason == "stop"
    assert not complete.output_ids
    assert complete.prompt_tokens == 5
    assert json.loads(complete.ec_transfer_params_json) == handles
    assert list(complete.media_identity.prompt_token_ids) == [7, 100, 100, 100, 9]
    assert list(complete.media_identity.mm_inputs.mm_hashes) == ["h1"]


@pytest.mark.asyncio
async def test_an_identity_that_cannot_be_built_does_not_fail_the_request(monkeypatch, caplog):
    # The identity is an optimisation with a fallback (decode reprocesses the
    # media), so a shape it did not anticipate must not turn a served request
    # into an error.
    from smg_grpc_servicer.vllm import servicer as servicer_module

    def explode(prompt):
        raise RuntimeError("stack expects each tensor to be equal size")

    monkeypatch.setattr(servicer_module, "build_media_identity", explode)
    processor = _Processor()
    servicer = _servicer(monkeypatch, processor)
    with caplog.at_level("WARNING", logger="smg_grpc_servicer.vllm.servicer"):
        complete = await _complete(servicer, _request(refs=True, kv=True))
    assert processor.calls == ["req-pd"]
    assert list(complete.output_ids) == [42]
    assert not complete.HasField("media_identity")
    assert any(
        "media identity not built" in r.getMessage() and "equal size" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ec", [None, "{}", '{"ec_items":[]}', '{"ec_items":[{"mm_hash":"h1"}]}', "{", "[]", "null"]
)
async def test_ec_consumer_rejects_metadata_without_a_usable_handoff(monkeypatch, ec):
    """The real handler refuses malformed/missing EC params before engine submission."""
    import grpc

    servicer = _servicer(monkeypatch, _Processor())
    prefill = await _complete(servicer, _request(refs=True, kv=True))
    servicer.engine.vllm_config.ec_transfer_config = SimpleNamespace(
        is_ec_consumer=True, is_ec_producer=False, ec_connector="ECMooncakeConnector"
    )
    request = _request(identity=prefill.media_identity)
    if ec is not None:
        request.ec_transfer_params_json = ec
    seen = []
    servicer.engine.generate = _engine(seen).generate
    context = SimpleNamespace(abort=AsyncMock(), invocation_metadata=lambda: ())
    assert [r async for r in servicer.Generate(request, context)] == []
    context.abort.assert_awaited_once()
    assert context.abort.call_args.args[0] is grpc.StatusCode.INVALID_ARGUMENT
    assert seen == []


@pytest.mark.asyncio
async def test_router_processed_encoder_returns_a_pixel_less_identity(monkeypatch):
    """Real vLLM tensor assembly supplies E's metadata without running the media processor."""
    from smg_grpc_servicer.vllm.media_identity import tensor_to_proto

    processor = _Processor()
    servicer = _servicer(monkeypatch, processor)
    prefill = await _complete(servicer, _request(refs=True, kv=True))
    processor.calls.clear()
    request = _request(identity=prefill.media_identity)
    request.mm_inputs.pixel_values.CopyFrom(tensor_to_proto(torch.zeros(1, 3, 2, 2)))
    request.mm_inputs.batched_keys.append("pixel_values")
    request.ec_transfer_params_json = "{}"
    handles = {"h1": {"peer_host": "encoder", "peer_port": 9000, "size_bytes": 128}}
    servicer.engine.vllm_config.ec_transfer_config = SimpleNamespace(
        is_ec_consumer=False, is_ec_producer=True, ec_connector="ECCPUConnector"
    )

    async def generate(**kwargs):
        """Return an encoder-only output after exercising the real tensor assembler."""
        assert kwargs["prompt"]["mm_hashes"] == {"image": ["h1"]}
        yield RequestOutput(
            request_id=kwargs["request_id"],
            prompt=None,
            prompt_token_ids=[7, 100, 100, 100, 9],
            prompt_logprobs=None,
            outputs=[],
            finished=True,
            ec_transfer_params=handles,
        )

    servicer.engine.generate = generate
    complete = await _complete(servicer, request)
    assert processor.calls == []
    assert list(complete.media_identity.mm_inputs.mm_hashes) == ["h1"]
    assert not complete.media_identity.mm_inputs.HasField("pixel_values")
    assert json.loads(complete.ec_transfer_params_json) == handles


@pytest.mark.asyncio
@pytest.mark.parametrize("kv", [False, True])
@pytest.mark.parametrize(
    "handles",
    [
        {"other": {"peer_host": "e", "peer_port": 1}},
        {"h1": {}},
        {"h1": {"peer_host": "e"}},
        {"h1": {"peer_host": "e", "peer_port": 0}},
        {"h1": {"peer_host": "e", "peer_port": "9000"}},
    ],
)
async def test_cpu_consumer_rejects_missing_or_invalid_hash_handles(monkeypatch, handles, kv):
    """A grid-only ECCPU request must never fall back to encoding missing pixels."""
    import grpc

    servicer = _servicer(monkeypatch, _Processor())
    prefill = await _complete(servicer, _request(refs=True, kv=True))
    servicer.engine.vllm_config.ec_transfer_config = SimpleNamespace(
        is_ec_consumer=True, is_ec_producer=False, ec_connector="ECCPUConnector"
    )
    request = _request(identity=prefill.media_identity, kv=kv)
    request.ec_transfer_params_json = json.dumps(handles)
    seen = []
    servicer.engine.generate = _engine(seen).generate
    context = SimpleNamespace(abort=AsyncMock(), invocation_metadata=lambda: ())
    assert [r async for r in servicer.Generate(request, context)] == []
    context.abort.assert_awaited_once()
    assert context.abort.call_args.args[0] is grpc.StatusCode.INVALID_ARGUMENT
    assert "ec_transfer_params_json is not usable" in context.abort.call_args.args[1]
    assert seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kv", [False, True])
@pytest.mark.parametrize("connector", ["ECCPUConnector", "ECMooncakeConnector"])
async def test_ec_consumer_rejects_a_pixel_less_batch_without_hashes(monkeypatch, kv, connector):
    """Missing hashes cannot make either connector's coverage check vacuously pass."""
    from unittest.mock import Mock

    import grpc

    servicer = _servicer(monkeypatch, _Processor())
    prefill = await _complete(servicer, _request(refs=True, kv=True))
    servicer.engine.vllm_config.ec_transfer_config = SimpleNamespace(
        is_ec_consumer=True, is_ec_producer=False, ec_connector=connector
    )
    request = _request(identity=prefill.media_identity, kv=kv)
    request.mm_inputs.ClearField("mm_hashes")
    request.ec_transfer_params_json = json.dumps(
        {
            "h1": {"peer_host": "e", "peer_port": 9000},
            "ec_items": [{"mm_hash": "h1", "transfer_id": "t"}],
        }
    )
    build = Mock(wraps=servicer._build_preprocessed_mm_inputs)
    servicer._build_preprocessed_mm_inputs = build
    context = SimpleNamespace(abort=AsyncMock(), invocation_metadata=lambda: ())
    assert [r async for r in servicer.Generate(request, context)] == []
    assert context.abort.call_args.args[0] is grpc.StatusCode.INVALID_ARGUMENT
    build.assert_not_called()


@pytest.mark.asyncio
async def test_ec_both_normal_pd_does_not_require_an_ec_completion(monkeypatch):
    """KV-only PD may return a media identity without soliciting any EC handles."""
    servicer = _servicer(monkeypatch, _Processor())
    servicer.engine.vllm_config.ec_transfer_config = SimpleNamespace(
        is_ec_consumer=True, is_ec_producer=True, ec_connector="ECCPUConnector"
    )
    complete = await _complete(servicer, _request(refs=True, kv=True))
    assert list(complete.output_ids) == [42]
    assert complete.HasField("media_identity")
