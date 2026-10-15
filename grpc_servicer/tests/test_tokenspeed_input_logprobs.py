"""The TokenSpeed servicer ships the engine's prompt logprobs to the gateway.

A request with ``logprob_start_len >= 0`` gets ``input_token_logprobs`` (and,
with ``top_logprobs_num``, ``input_top_logprobs``) in one frame's meta_info.
The servicer puts them on that frame's chunk and on the choice's Complete, the
first position's value unset (no context predicts it).

Run with: pytest grpc_servicer/tests/test_tokenspeed_input_logprobs.py
"""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("smg_grpc_proto")
servicer_mod = pytest.importorskip("smg_grpc_servicer.tokenspeed.servicer")
pb = servicer_mod.tokenspeed_scheduler_pb2
TokenSpeedSchedulerServicer = servicer_mod.TokenSpeedSchedulerServicer

# The engine's rendering of a scored three-token prompt.
PROMPT = [(None, 5, "a"), (-0.5, 6, "b"), (-1.25, 7, "c")]
PROMPT_TOP = [None, [(-0.5, 6, "b"), (-1.5, 8, "d")], [(-0.25, 9, "e"), (-1.25, 7, "c")]]


def _servicer(frames=()):
    s = TokenSpeedSchedulerServicer.__new__(TokenSpeedSchedulerServicer)
    s.server_args = SimpleNamespace()

    async def generate_request(_request):
        for frame in frames:
            yield frame

    s.async_llm = SimpleNamespace(
        rid_to_state={}, generate_request=generate_request, abort_request=lambda _rid: None
    )
    s._build_generate_req = lambda request: SimpleNamespace(rid=request.request_id)
    return s


def _meta(**fields):
    return {"finish_reason": None, "prompt_tokens": 3, "completion_tokens": 1, **fields}


def _check(logprobs):
    assert list(logprobs.token_ids) == [5, 6, 7]
    assert not logprobs.token_logprobs[0].HasField("value")
    assert [t.value for t in logprobs.token_logprobs[1:]] == [-0.5, -1.25]
    assert [list(t.token_ids) for t in logprobs.top_logprobs] == [[], [6, 8], [9, 7]]
    assert list(logprobs.top_logprobs[2].values) == [-0.25, -1.25]


def test_complete_and_chunk_carry_the_prompt_logprobs():
    s = _servicer()
    output = {
        "output_ids": [4],
        "meta_info": _meta(
            input_token_logprobs=PROMPT,
            input_top_logprobs=PROMPT_TOP,
            finish_reason={"type": "length"},
        ),
    }
    _check(s._complete_response("r", output, {"type": "length"}, 0).complete.input_logprobs)
    _check(s._chunk_response("r", output, None, 0).chunk.input_logprobs)


def test_prompt_logprobs_without_alternatives_keep_an_entry_per_position():
    s = _servicer()
    output = {"output_ids": [4], "meta_info": _meta(input_token_logprobs=PROMPT)}
    logprobs = s._chunk_response("r", output, None, 0).chunk.input_logprobs
    assert list(logprobs.token_ids) == [5, 6, 7]
    assert [list(t.token_ids) for t in logprobs.top_logprobs] == [[], [], []]


def test_a_frame_without_prompt_logprobs_leaves_the_field_unset():
    s = _servicer()
    for meta in (_meta(), _meta(input_token_logprobs=[])):
        output = {"output_ids": [4], "meta_info": meta}
        complete = s._complete_response("r", output, {"type": "stop"}, 0)
        assert not complete.complete.HasField("input_logprobs")
        assert not s._chunk_response("r", output, None, 0).chunk.HasField("input_logprobs")


def test_streamed_prompt_logprobs_ride_one_chunk_and_the_complete():
    # The engine ships them on the frame after the prompt's last chunk; later
    # stream frames carry empty lists.
    frames = [
        {
            "output_ids": [4],
            "meta_info": _meta(input_token_logprobs=PROMPT, input_top_logprobs=PROMPT_TOP),
        },
        {
            "output_ids": [5],
            "meta_info": _meta(
                input_token_logprobs=[],
                input_top_logprobs=[],
                completion_tokens=2,
                finish_reason={"type": "length"},
            ),
        },
    ]
    s = _servicer(frames)
    request = pb.GenerateRequest(
        request_id="r", stream=True, return_logprob=True, logprob_start_len=0, top_logprobs_num=2
    )

    async def collect():
        return [response async for response in s.Generate(request, SimpleNamespace())]

    responses = asyncio.run(collect())
    assert [r.WhichOneof("response") for r in responses] == ["chunk", "chunk", "complete"]
    assert [r.chunk.HasField("input_logprobs") for r in responses[:2]] == [True, False]
    _check(responses[0].chunk.input_logprobs)
    _check(responses[2].complete.input_logprobs)
