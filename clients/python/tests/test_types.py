"""Tests for type serialization/deserialization."""

import json

from smg_client.types import (
    ChatCompletionMessage,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatMessageDelta,
    CompletionResponse,
    EmbeddingResponse,
    Message,
    RerankResponse,
)


def test_both_reasoning_fields_are_readable_from_generated_types():
    fields = {"reasoning": "thinking", "reasoning_content": "thinking"}
    message = ChatCompletionMessage.model_validate({"role": "assistant", "content": None, **fields})
    delta = ChatMessageDelta.model_validate(fields)
    for parsed in (message, delta):
        assert parsed.reasoning == parsed.reasoning_content == "thinking"
    assert message.content is None


def test_chat_completion_response_roundtrip():
    raw = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "llama-3.1-8b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    }
    resp = ChatCompletionResponse.model_validate(raw)
    assert resp.id == "chatcmpl-abc123"
    assert resp.choices[0].message.content == "Hello! How can I help you today?"
    assert resp.usage.total_tokens == 18

    # Roundtrip
    dumped = json.loads(resp.model_dump_json(exclude_none=True))
    assert dumped["id"] == raw["id"]
    assert dumped["choices"][0]["message"]["content"] == raw["choices"][0]["message"]["content"]


def test_chat_stream_response():
    raw = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "llama-3.1-8b",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "Hello"},
                "finish_reason": None,
            }
        ],
    }
    resp = ChatCompletionStreamResponse.model_validate(raw)
    assert resp.choices[0].delta.content == "Hello"


def test_embedding_response():
    raw = {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
        "model": "bge-large",
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 0,
            "total_tokens": 5,
        },
    }
    resp = EmbeddingResponse.model_validate(raw)
    assert len(resp.data) == 1
    assert resp.data[0].embedding == [0.1, 0.2, 0.3]


def test_completion_response():
    raw = {
        "id": "cmpl-abc123",
        "object": "text_completion",
        "created": 1700000000,
        "model": "llama-3.1-8b",
        "choices": [
            {
                "text": "world",
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }
    resp = CompletionResponse.model_validate(raw)
    assert resp.choices[0].text == "world"


def test_rerank_response():
    raw = {
        "results": [
            {"score": 0.95, "document": "relevant doc", "index": 0},
            {"score": 0.2, "document": "irrelevant doc", "index": 1},
        ],
        "model": "bge-reranker",
        "object": "rerank",
        "created": 1700000000,
    }
    resp = RerankResponse.model_validate(raw)
    assert len(resp.results) == 2
    assert resp.results[0].score == 0.95


def test_anthropic_message_response():
    raw = {
        "id": "msg_abc123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "model": "claude-3-5-sonnet",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    msg = Message.model_validate(raw)
    assert msg.id == "msg_abc123"
    assert msg.content[0].type == "text"
    assert msg.content[0].text == "Hello!"
    assert msg.usage.input_tokens == 10


def test_chat_completion_with_tool_calls():
    raw = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "llama-3.1-8b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "London"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    resp = ChatCompletionResponse.model_validate(raw)
    assert resp.choices[0].message.tool_calls[0].function.name == "get_weather"
    assert resp.choices[0].finish_reason == "tool_calls"
