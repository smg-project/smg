"""The request salt on each prompt shape, through the real Generate handler (no vLLM required)."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import grpc
import pytest
from smg_grpc_proto import vllm_engine_pb2 as pb
from smg_grpc_proto.generated import common_pb2
from smg_grpc_servicer.vllm.media_refs import parse_media_refs, validate_schemes
from smg_grpc_servicer.vllm.mm_keys import describes_media_twice, mm_identity_hashes
from smg_grpc_servicer.vllm.mm_salt import engine_cache_salt
from test_vllm_ec_handoff import _Aborted, _generate

DECODE_KV = {"do_remote_prefill": True}


def _request(shape, salt):
    request = pb.GenerateRequest(
        request_id="salted", kv_transfer_params_json=json.dumps(DECODE_KV), cache_salt=salt
    )
    if shape == "text":
        request.text = "hello"
        return request
    request.tokenized.input_ids.extend([1, 2])
    if shape == "media_refs":
        request.media_refs.items.add(modality=common_pb2.IMAGE, url="https://a/1.png")
    if shape in ("identity", "grid", "pixels"):
        request.mm_inputs.mm_hashes.extend(["h1", "h2"])
    if shape == "grid":
        request.mm_inputs.model_specific_tensors["image_grid_thw"].shape.extend([1, 3])
    if shape == "pixels":
        request.mm_inputs.pixel_values.shape.extend([1, 2])
    return request


async def _engine_prompts(request):
    """The prompts that reach the engine. A refusal owes the prefill side its notice."""
    prompts = []

    async def generate(*, prompt, **kwargs):
        prompts.append(prompt)
        if False:
            yield

    async def abort(code, message):
        raise _Aborted(code, message)

    async def processed(*args, **kwargs):
        return {"type": "multimodal", "prompt_token_ids": [1, 2]}

    servicer = SimpleNamespace(
        engine=SimpleNamespace(
            generate=generate,
            renderer=SimpleNamespace(
                process_for_engine=lambda prompt, **kwargs: {"type": "token", **prompt}
            ),
            vllm_config=SimpleNamespace(ec_transfer_config=None),
        ),
        _mm_processor=SimpleNamespace(accepted_schemes={"https"}, process=processed),
        _acquire_mm_slot=AsyncMock(),
        _mm_inflight=Mock(),
        _build_preprocessed_mm_inputs=Mock(),
        _off_the_event_loop=processed,
        _sampling_params_from_proto=lambda *args, **kwargs: SimpleNamespace(
            extra_args=None, logprobs=None, prompt_logprobs=None
        ),
        _tokenization_kwargs_from_proto=lambda _: {},
        _loads=Mock(),
        _notify_kv_transfer_rejected=AsyncMock(),
    )
    handler = _generate(
        describes_media_twice=describes_media_twice,
        engine_cache_salt=engine_cache_salt,
        media_identity_supported=lambda: True,
        mm_identity_hashes=mm_identity_hashes,
        parse_media_refs=parse_media_refs,
        uses_mrope=lambda _: False,
        validate_schemes=validate_schemes,
        _prompt_length=lambda _: 2,
    )
    context = SimpleNamespace(abort=abort, invocation_metadata=lambda: ())
    try:
        assert [reply async for reply in handler(servicer, request, context)] == []
    except _Aborted:
        servicer._notify_kv_transfer_rejected.assert_awaited_once_with("salted", DECODE_KV, False)
        assert not prompts
        raise
    return prompts


@pytest.mark.asyncio
@pytest.mark.parametrize("salt", [None, "tenant-a"])
async def test_each_prompt_shape_carries_the_salt(salt):
    for shape in ("text", "tokenized", "media_refs", "identity", "grid", "pixels"):
        (prompt,) = await _engine_prompts(_request(shape, salt))
        if shape == "text":
            assert prompt == ({"prompt": "hello", "cache_salt": salt} if salt else "hello")
            continue
        expected = engine_cache_salt(salt, ["h1", "h2"]) if shape == "identity" else salt
        assert prompt.get("cache_salt") == expected, shape
        assert ("cache_salt" in prompt) == (expected is not None), shape


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["text", "tokenized", "media_refs", "identity", "pixels"])
async def test_a_bad_salt_is_refused_before_the_engine(shape):
    with pytest.raises(_Aborted) as refused:
        await _engine_prompts(_request(shape, "mm:secret"))
    code, message = refused.value.args
    assert code is grpc.StatusCode.INVALID_ARGUMENT
    assert "cache_salt" in message
    assert "secret" not in message
