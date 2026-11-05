"""The TokenSpeed servicer's Generate closes at semantic completion.

The engine's iterator may stay open briefly after it yields its final output;
the gRPC stream must not make a client wait for transport EOF once every
requested choice has its Complete message, and Generate closes that iterator
before it reads the engine's request table. A final engine delta that is
nothing but the matched stop token leaves no Chunk behind once its choice has
sent one; a choice's first output goes out even when empty, since the gateway
opens a choice on its first Chunk.

Run with: pytest grpc_servicer/tests/test_tokenspeed_generate_completion.py
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import grpc
import pytest

pytest.importorskip("msgspec")
pytest.importorskip("zmq")
pytest.importorskip("numpy")
pytest.importorskip("smg_grpc_proto")
from smg_grpc_proto.generated import tokenspeed_scheduler_pb2  # noqa: E402

STOP_TOKEN_ID = 200_008


class _AsyncLlm:
    """An engine whose iterator stays open after its last output.

    ``open_rids`` sit in ``rid_to_state`` as unfinished requests until the
    iterator is closed: a stand-in for what an engine releases at its
    generator's exit, which Generate must run before it reads the table.
    TokenSpeed's exit releases the model-update reader lock it holds across
    yields, not request state: a finished request's state goes before that
    request's final yield, and an abandoned request is still Generate's own
    check to abort.
    """

    def __init__(self, outputs, open_rids=()):
        self.outputs = outputs
        self.open_rids = list(open_rids)
        self.rid_to_state = {}

    async def generate_request(self, request):
        del request
        for rid in self.open_rids:
            self.rid_to_state[rid] = SimpleNamespace(finished=False)
        try:
            for output in self.outputs:
                yield output
            raise AssertionError("Generate consumed the engine iterator after completion")
        finally:
            for rid in self.open_rids:
                self.rid_to_state.pop(rid, None)

    def abort_request(self, rid):
        raise AssertionError(f"finished request {rid} was aborted")


def _request(stream: bool):
    return tokenspeed_scheduler_pb2.GenerateRequest(
        request_id="request-1",
        stream=stream,
        sampling_params=tokenspeed_scheduler_pb2.SamplingParams(),
    )


def _output(index: int):
    return {
        "index": index,
        "output_ids": [42],
        "meta_info": {
            "finish_reason": {"type": "stop", "matched": STOP_TOKEN_ID},
            "completion_tokens": 1,
        },
    }


def _partial_output(index: int):
    return {
        "index": index,
        "output_ids": [42],
        "meta_info": {"finish_reason": None, "completion_tokens": 1},
    }


def _stop_only_output(index: int):
    return {
        "index": index,
        "output_ids": [STOP_TOKEN_ID],
        "meta_info": {
            "finish_reason": {"type": "stop", "matched": STOP_TOKEN_ID},
            "completion_tokens": 1,
        },
    }


def _servicer(tokenspeed_servicer_mod, outputs, expanded_rid, open_rids=()):
    servicer = tokenspeed_servicer_mod.TokenSpeedSchedulerServicer.__new__(
        tokenspeed_servicer_mod.TokenSpeedSchedulerServicer
    )
    servicer.async_llm = _AsyncLlm(outputs, open_rids)
    servicer._build_generate_req = lambda request: SimpleNamespace(rid=expanded_rid)
    return servicer


async def _collect(servicer, request):
    return [response async for response in servicer.Generate(request, context=None)]


def _kinds(responses):
    return [response.WhichOneof("response") for response in responses]


def test_non_streaming_single_choice_closes_after_complete(tokenspeed_servicer_mod):
    servicer = _servicer(tokenspeed_servicer_mod, [_output(0)], "engine-rid")
    responses = asyncio.run(_collect(servicer, _request(False)))

    assert len(responses) == 1
    assert responses[0].HasField("complete")


def test_streaming_waits_for_every_choice_then_closes(tokenspeed_servicer_mod):
    servicer = _servicer(tokenspeed_servicer_mod, [_output(0), _output(1)], ["rid-n0", "rid-n1"])
    responses = asyncio.run(_collect(servicer, _request(True)))

    assert _kinds(responses) == ["chunk", "complete", "chunk", "complete"]


def test_streaming_skips_an_empty_stop_only_terminal_chunk(tokenspeed_servicer_mod):
    servicer = _servicer(
        tokenspeed_servicer_mod, [_partial_output(0), _stop_only_output(0)], "engine-rid"
    )
    responses = asyncio.run(_collect(servicer, _request(True)))

    assert _kinds(responses) == ["chunk", "complete"]
    assert list(responses[0].chunk.token_ids) == [42]
    assert list(responses[1].complete.output_ids) == []
    assert responses[1].complete.matched_token_id == STOP_TOKEN_ID


def test_streaming_opens_a_choice_whose_only_output_is_the_stop_token(tokenspeed_servicer_mod):
    servicer = _servicer(tokenspeed_servicer_mod, [_stop_only_output(0)], "engine-rid")
    responses = asyncio.run(_collect(servicer, _request(True)))

    assert _kinds(responses) == ["chunk", "complete"]
    assert list(responses[0].chunk.token_ids) == []
    assert list(responses[1].complete.output_ids) == []
    assert responses[1].complete.matched_token_id == STOP_TOKEN_ID


def test_streaming_opens_each_choice_even_one_that_is_only_the_stop_token(tokenspeed_servicer_mod):
    servicer = _servicer(
        tokenspeed_servicer_mod, [_output(0), _stop_only_output(1)], ["rid-n0", "rid-n1"]
    )
    responses = asyncio.run(_collect(servicer, _request(True)))

    assert _kinds(responses) == ["chunk", "complete", "chunk", "complete"]
    assert responses[2].chunk.index == 1
    assert list(responses[2].chunk.token_ids) == []


def test_streaming_keeps_the_stop_token_when_the_client_asked_for_it(tokenspeed_servicer_mod):
    servicer = _servicer(tokenspeed_servicer_mod, [_stop_only_output(0)], "engine-rid")
    request = tokenspeed_scheduler_pb2.GenerateRequest(
        request_id="request-1",
        stream=True,
        sampling_params=tokenspeed_scheduler_pb2.SamplingParams(no_stop_trim=True),
    )
    responses = asyncio.run(_collect(servicer, request))

    assert _kinds(responses) == ["chunk", "complete"]
    assert list(responses[0].chunk.token_ids) == [STOP_TOKEN_ID]


def test_non_streaming_choice_list_closes_after_the_batch(tokenspeed_servicer_mod):
    servicer = _servicer(tokenspeed_servicer_mod, [[_output(0), _output(1)]], ["rid-n0", "rid-n1"])
    responses = asyncio.run(_collect(servicer, _request(False)))

    assert len(responses) == 2
    assert all(response.HasField("complete") for response in responses)


@pytest.mark.parametrize(
    ("outputs", "expanded_rid", "stream", "expected"),
    [
        pytest.param([_output(0)], "engine-rid", True, ["chunk", "complete"], id="streamed"),
        pytest.param(
            [[_output(0), _output(1)]],
            ["rid-n0", "rid-n1"],
            False,
            ["complete", "complete"],
            id="choice-list",
        ),
    ],
)
def test_the_engine_iterator_is_closed_before_its_request_table_is_checked(
    tokenspeed_servicer_mod, outputs, expanded_rid, stream, expected
):
    open_rids = expanded_rid if isinstance(expanded_rid, list) else [expanded_rid]
    servicer = _servicer(tokenspeed_servicer_mod, outputs, expanded_rid, open_rids=open_rids)
    responses = asyncio.run(_collect(servicer, _request(stream)))

    assert _kinds(responses) == expected
    assert servicer.async_llm.rid_to_state == {}


class _Aborted(grpc.aio.AbortError):
    """What grpc.aio's abort raises: the servicer lets it through untouched."""


class _Context:
    """A servicer context that records its one abort, as grpc.aio's raises."""

    def __init__(self):
        self.aborts = []

    async def abort(self, code, details="", trailing_metadata=()):
        self.aborts.append((code, details, tuple(trailing_metadata)))
        raise _Aborted()


def _abort_output(index: int, **reason):
    return {
        "index": index,
        "output_ids": [],
        "meta_info": {"finish_reason": {"type": "abort", **reason}, "completion_tokens": 0},
    }


def _constrained_request(stream: bool):
    return tokenspeed_scheduler_pb2.GenerateRequest(
        request_id="request-1",
        stream=stream,
        sampling_params=tokenspeed_scheduler_pb2.SamplingParams(json_schema='{"type":"object"}'),
    )


async def _aborted_with(servicer, request):
    context = _Context()
    try:
        async for _ in servicer.Generate(request, context=context):
            pass
    except _Aborted:
        pass
    assert len(context.aborts) == 1, context.aborts
    return context.aborts[0]


@pytest.mark.parametrize("stream", [False, True])
def test_an_abort_of_a_constrained_request_is_the_engines_own_abort(
    tokenspeed_servicer_mod, stream
):
    # The engine has no abort code for its grammar: an abort of a constrained
    # request with no status is a pause aborting every request in flight (the
    # finish an RL driver resubmits on) or a capacity abort as often as
    # anything else, so it keeps the status every request's abort has, and
    # never the constraint-failure status (which only an engine's own
    # attribution raises); a full queue keeps its own code.
    servicer = _servicer(
        tokenspeed_servicer_mod,
        [_abort_output(0, message="Request aborted by the scheduler: paused", err_type=524)],
        "rid",
    )
    code, details, trailer = asyncio.run(_aborted_with(servicer, _constrained_request(stream)))
    assert code is grpc.StatusCode.INTERNAL
    assert details == "Request aborted by the scheduler: paused"
    assert trailer == ()

    servicer = _servicer(
        tokenspeed_servicer_mod, [_abort_output(0, message="busy", status_code=429)], "rid"
    )
    code, _, trailer = asyncio.run(_aborted_with(servicer, _constrained_request(stream)))
    assert code is grpc.StatusCode.RESOURCE_EXHAUSTED
    assert trailer == ()


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (400, grpc.StatusCode.INVALID_ARGUMENT),
        (404, grpc.StatusCode.NOT_FOUND),
        (408, grpc.StatusCode.DEADLINE_EXCEEDED),
        (429, grpc.StatusCode.RESOURCE_EXHAUSTED),
        (500, grpc.StatusCode.INTERNAL),
        (503, grpc.StatusCode.UNAVAILABLE),
        (504, grpc.StatusCode.DEADLINE_EXCEEDED),
        (None, grpc.StatusCode.INTERNAL),
    ],
)
@pytest.mark.parametrize("choice_list", [False, True], ids=["streamed", "choice-list"])
def test_an_engine_abort_carries_the_status_its_http_code_means(
    tokenspeed_servicer_mod, status_code, expected, choice_list
):
    # The engine's HTTP-style status goes through the table every engine
    # servicer shares (the vLLM and SGLang servicers answer the same code for
    # the same status), on the streamed path and on the choice-list path alike;
    # the engine's message is the status message.
    output = _abort_output(0, message="the engine refused the request", status_code=status_code)
    outputs = [[output]] if choice_list else [output]
    servicer = _servicer(tokenspeed_servicer_mod, outputs, "rid")
    code, details, trailer = asyncio.run(_aborted_with(servicer, _request(not choice_list)))
    assert code is expected
    assert details == "the engine refused the request"
    assert trailer == ()
