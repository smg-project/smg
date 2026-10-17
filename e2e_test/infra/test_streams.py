"""The stream collectors, on recorded streams: the shapes the suites assert on."""

from __future__ import annotations

import pytest
from infra.streams import (
    StreamedToolCall,
    collect_streamed_chat,
    collect_streamed_response,
)
from openai.types.chat import ChatCompletionChunk
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseTextDeltaEvent,
)


def _chunk(*, index: int = 0, delta: dict, finish_reason: str | None = None) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [{"index": index, "delta": delta, "finish_reason": finish_reason}],
        }
    )


def _tool_delta(
    index: int, *, id: str | None = None, name: str | None = None, arguments: str | None = None
) -> dict:
    return {
        "index": index,
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


# A tool call streamed the way the gateway streams it: the name (with the id)
# on the first delta, the arguments in fragments after it, text deltas around
# it, a second call interleaved, and the finish reason on the last chunk.
RECORDED_TOOL_CALL_STREAM = [
    _chunk(delta={"role": "assistant", "content": ""}),
    _chunk(delta={"content": "Let me check."}),
    _chunk(delta={"tool_calls": [_tool_delta(0, id="call_a", name="get_weather", arguments="")]}),
    _chunk(delta={"tool_calls": [_tool_delta(0, arguments='{"city": ')]}),
    _chunk(delta={"tool_calls": [_tool_delta(1, id="call_b", name="get_time", arguments="{}")]}),
    _chunk(delta={"tool_calls": [_tool_delta(0, arguments='"Paris"}')]}),
    _chunk(delta={}, finish_reason="tool_calls"),
]


def test_tool_calls_are_assembled_by_index_in_order():
    streamed = collect_streamed_chat(iter(RECORDED_TOOL_CALL_STREAM))

    assert streamed.tool_calls == [
        StreamedToolCall(index=0, id="call_a", name="get_weather", arguments='{"city": "Paris"}'),
        StreamedToolCall(index=1, id="call_b", name="get_time", arguments="{}"),
    ]
    assert streamed.content == "Let me check."
    assert streamed.finish_reason == "tool_calls"
    assert len(streamed.chunks) == len(RECORDED_TOOL_CALL_STREAM)
    assert len(streamed.tool_call_deltas) == 4


def test_a_stream_without_tool_calls_keeps_its_text_and_finish_reason():
    streamed = collect_streamed_chat(
        [
            _chunk(delta={"role": "assistant", "content": "Hello"}),
            _chunk(delta={"content": " there"}),
            _chunk(delta={}, finish_reason="stop"),
        ]
    )

    assert streamed.tool_calls == []
    assert streamed.tool_call_deltas == []
    assert streamed.content == "Hello there"
    assert streamed.finish_reason == "stop"


def test_a_chunk_without_choices_is_kept_and_contributes_nothing():
    usage_chunk = ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    streamed = collect_streamed_chat(
        [_chunk(delta={"content": "x"}, finish_reason="stop"), usage_chunk]
    )

    assert streamed.chunks[-1] is usage_chunk
    assert streamed.content == "x"
    assert streamed.finish_reason == "stop"


def test_another_choice_of_a_multi_choice_stream_contributes_nothing():
    streamed = collect_streamed_chat(
        [
            _chunk(index=1, delta={"content": "second"}),
            _chunk(delta={"content": "first"}),
            _chunk(index=1, delta={"tool_calls": [_tool_delta(0, name="other", arguments="{}")]}),
            _chunk(index=1, delta={}, finish_reason="tool_calls"),
            _chunk(delta={}, finish_reason="stop"),
        ]
    )

    assert streamed.content == "first"
    assert streamed.tool_calls == []
    assert streamed.finish_reason == "stop"
    assert len(streamed.chunks) == 5


def _response(status: str = "completed") -> dict:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "m",
        "status": status,
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _recorded_response_stream() -> list:
    return [
        ResponseCreatedEvent.model_validate(
            {"type": "response.created", "sequence_number": 0, "response": _response("in_progress")}
        ),
        ResponseTextDeltaEvent.model_validate(
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hi",
                "logprobs": [],
            }
        ),
        ResponseCompletedEvent.model_validate(
            {"type": "response.completed", "sequence_number": 2, "response": _response()}
        ),
    ]


def test_a_response_stream_keeps_its_events_and_names_the_final_response():
    streamed = collect_streamed_response(iter(_recorded_response_stream()))

    assert streamed.event_types == [
        "response.created",
        "response.output_text.delta",
        "response.completed",
    ]
    assert len(streamed.completed_events) == 1
    assert streamed.final_response.status == "completed"
    assert [item.type for item in streamed.final_response.output] == ["message"]


def test_a_response_stream_that_never_completed_has_no_final_response():
    streamed = collect_streamed_response(_recorded_response_stream()[:2])

    assert streamed.completed_events == []
    with pytest.raises(AssertionError, match="no response.completed event"):
        streamed.final_response
