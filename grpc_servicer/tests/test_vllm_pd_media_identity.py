"""PD legs and worker-side media (needs vLLM: runs on the vLLM e2e lane).

The prefill leg processes the references once and answers with their
identity; a regular request answers with none; the decode leg, given that
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
    context = SimpleNamespace(abort=AsyncMock())
    responses = [r async for r in servicer.Generate(request, context)]
    context.abort.assert_not_awaited()
    (final,) = [r for r in responses if r.HasField("complete")]
    return final.complete


@pytest.mark.asyncio
async def test_prefill_leg_answers_with_the_media_identity(monkeypatch):
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
async def test_regular_leg_answers_with_no_identity(monkeypatch):
    processor = _Processor()
    servicer = _servicer(monkeypatch, processor)
    complete = await _complete(servicer, _request(refs=True, kv=False))
    assert processor.calls == ["req-pd"]
    assert not complete.HasField("media_identity")


@pytest.mark.asyncio
async def test_decode_leg_from_identity_never_touches_the_processor(monkeypatch):
    processor = _Processor()
    servicer = _servicer(monkeypatch, processor)
    prefill = await _complete(servicer, _request(refs=True, kv=True))
    processor.calls.clear()
    decode = await _complete(servicer, _request(kv=True, identity=prefill.media_identity))
    assert processor.calls == [], "the decode leg carried identity, not references"
    assert not decode.HasField("media_identity")
