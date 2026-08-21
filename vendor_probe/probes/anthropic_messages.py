"""Anthropic Messages API ground-truth probe matrix (data only).

Every probe is a plain dict. The runner injects the model name (sentinel
@MODEL) and resolves {{probe_id#json.path}} placeholders from prior
recordings. anthropic-version is pinned by the adapter (2023-06-01); the
only beta header used here is interleaved-thinking-2025-05-14.

Fields: id, category, endpoint, method, body, stream, depends_on, expect
(ok|error), headers (extra), note.
"""

# --- fixtures ---------------------------------------------------------------
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
PDF_B64 = (
    "JVBERi0xLjEKMSAwIG9iajw8L1R5cGUvQ2F0YWxvZy9QYWdlcyAyIDAgUj4+ZW5kb2JqCjIgMCBv"
    "Ymo8PC9UeXBlL1BhZ2VzL0tpZHNbMyAwIFJdL0NvdW50IDE+PmVuZG9iagozIDAgb2JqPDwvVHlw"
    "ZS9QYWdlL1BhcmVudCAyIDAgUi9NZWRpYUJveFswIDAgMjAwIDIwMF0vQ29udGVudHMgNCAwIFIv"
    "UmVzb3VyY2VzPDwvRm9udDw8L0YxIDUgMCBSPj4+Pj4+ZW5kb2JqCjQgMCBvYmo8PC9MZW5ndGgg"
    "NDQ+PnN0cmVhbQpCVCAvRjEgMTggVGYgMjAgMTAwIFRkIChQcm9iZSBQREYpIFRqIEVUCmVuZHN0"
    "cmVhbSBlbmRvYmoKNSAwIG9iajw8L1R5cGUvRm9udC9TdWJ0eXBlL1R5cGUxL0Jhc2VGb250L0hl"
    "bHZldGljYT4+ZW5kb2JqCnRyYWlsZXI8PC9Sb290IDEgMCBSPj4KJSVFT0Y="
)
IMG_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
    "PNG_transparency_demonstration_1.png/240px-PNG_transparency_demonstration_1.png"
)
# a long-ish string used to cross the 4096-token minimum cache boundary on haiku
CACHE_FILLER = "The quick brown fox jumps over the lazy dog. " * 700

PROBES: list[dict] = []


def _p(
    pid,
    category,
    body=None,
    *,
    endpoint="/v1/messages",
    method="POST",
    stream=False,
    depends_on=None,
    expect="ok",
    headers=None,
    note=None,
):
    PROBES.append(
        {
            "id": pid,
            "category": category,
            "endpoint": endpoint,
            "method": method,
            "body": body,
            "stream": stream,
            "depends_on": depends_on,
            "expect": expect,
            "headers": headers,
            "note": note,
        }
    )


def _u(text):
    return {"role": "user", "content": text}


# =============================================================================
# baseline-and-request-fields (non-streaming)
# =============================================================================
B = "baseline-and-request-fields"
_p("anth.base.minimal", B, {"model": "@MODEL", "max_tokens": 64, "messages": [_u("hi")]})
_p(
    "anth.base.system-string",
    B,
    {"model": "@MODEL", "max_tokens": 64, "system": "You are terse.", "messages": [_u("hi")]},
)
_p(
    "anth.base.system-blocks-1",
    B,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are terse."}],
        "messages": [_u("hi")],
    },
)
_p(
    "anth.base.system-blocks-multi",
    B,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "system": [
            {"type": "text", "text": "You are terse."},
            {"type": "text", "text": "Answer in English."},
        ],
        "messages": [_u("hi")],
    },
)
_p(
    "anth.base.system-blocks-cache-control",
    B,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "system": [{"type": "text", "text": CACHE_FILLER, "cache_control": {"type": "ephemeral"}}],
        "messages": [_u("hi")],
    },
)
_p(
    "anth.base.multi-turn",
    B,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [
            _u("My name is Sam."),
            {"role": "assistant", "content": "Nice to meet you, Sam."},
            _u("What is my name?"),
        ],
    },
)
_p(
    "anth.base.consecutive-user-user",
    B,
    {"model": "@MODEL", "max_tokens": 64, "messages": [_u("first thing"), _u("second thing")]},
    note="docs conflict: combined vs must-alternate; record truth",
)
_p(
    "anth.base.assistant-prefill-last",
    B,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [
            _u("Complete: the capital of France is"),
            {"role": "assistant", "content": "The capital of France is"},
        ],
    },
    note="record continuation semantics, whether prefill echoed",
)
_p(
    "anth.base.metadata-user-id",
    B,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [_u("hi")],
        "metadata": {"user_id": "probe-user-1"},
    },
)
_p(
    "anth.base.metadata-empty-object",
    B,
    {"model": "@MODEL", "max_tokens": 64, "messages": [_u("hi")], "metadata": {}},
)
_p(
    "anth.base.service-tier-auto",
    B,
    {"model": "@MODEL", "max_tokens": 64, "messages": [_u("hi")], "service_tier": "auto"},
)
_p(
    "anth.base.service-tier-standard-only",
    B,
    {"model": "@MODEL", "max_tokens": 64, "messages": [_u("hi")], "service_tier": "standard_only"},
)
_p(
    "anth.base.temperature-0",
    B,
    {"model": "@MODEL", "max_tokens": 64, "messages": [_u("hi")], "temperature": 0},
)
_p(
    "anth.base.temperature-1",
    B,
    {"model": "@MODEL", "max_tokens": 64, "messages": [_u("hi")], "temperature": 1},
)
_p(
    "anth.base.top-p-only",
    B,
    {"model": "@MODEL", "max_tokens": 64, "messages": [_u("hi")], "top_p": 0.5},
)
_p(
    "anth.base.top-k-only",
    B,
    {"model": "@MODEL", "max_tokens": 64, "messages": [_u("hi")], "top_k": 10},
)

# =============================================================================
# content-blocks
# =============================================================================
CB = "content-blocks"
_p(
    "anth.content.multi-text-blocks",
    CB,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part one."},
                    {"type": "text", "text": "part two."},
                ],
            }
        ],
    },
)
_p(
    "anth.content.image-base64-png",
    CB,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64},
                    },
                ],
            }
        ],
    },
)
_p(
    "anth.content.image-url",
    CB,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image", "source": {"type": "url", "url": IMG_URL}},
                ],
            }
        ],
    },
)
_p(
    "anth.content.image-plus-text-order",
    CB,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64},
                    },
                    {"type": "text", "text": "describe the image above"},
                ],
            }
        ],
    },
)
_p(
    "anth.content.two-images",
    CB,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "compare"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64},
                    },
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64},
                    },
                ],
            }
        ],
    },
)
_p(
    "anth.content.document-text-source",
    CB,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": "The mitochondria is the powerhouse of the cell.",
                        },
                    },
                    {"type": "text", "text": "summarize the document"},
                ],
            }
        ],
    },
)
_p(
    "anth.content.document-base64-pdf",
    CB,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": PDF_B64,
                        },
                    },
                    {"type": "text", "text": "what does the pdf say?"},
                ],
            }
        ],
    },
)
_p(
    "anth.content.document-content-source",
    CB,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "content",
                            "content": [
                                {"type": "text", "text": "Custom block one."},
                                {"type": "text", "text": "Custom block two."},
                            ],
                        },
                    },
                    {"type": "text", "text": "summarize"},
                ],
            }
        ],
    },
)
_p(
    "anth.content.document-title-context",
    CB,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": "Section A. Section B.",
                        },
                        "title": "Probe Doc",
                        "context": "A test document.",
                    },
                    {"type": "text", "text": "list the sections"},
                ],
            }
        ],
    },
)
_p(
    "anth.content.document-citations-enabled",
    CB,
    {
        "model": "@MODEL",
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": "The capital of France is Paris. The capital of Japan is Tokyo.",
                        },
                        "citations": {"enabled": True},
                    },
                    {"type": "text", "text": "What is the capital of France? Cite."},
                ],
            }
        ],
    },
)
_p(
    "anth.content.pdf-citations",
    CB,
    {
        "model": "@MODEL",
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": PDF_B64,
                        },
                        "citations": {"enabled": True},
                    },
                    {"type": "text", "text": "Quote the pdf and cite the page."},
                ],
            }
        ],
    },
)
_p(
    "anth.content.search-result-block",
    CB,
    {
        "model": "@MODEL",
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "search_result",
                        "source": "https://example.com/doc",
                        "title": "Probe Result",
                        "content": [{"type": "text", "text": "Paris is the capital of France."}],
                        "citations": {"enabled": True},
                    },
                    {"type": "text", "text": "What is the capital of France? Cite."},
                ],
            }
        ],
    },
)
_p(
    "anth.content.cache-write",
    CB,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "system": [{"type": "text", "text": CACHE_FILLER, "cache_control": {"type": "ephemeral"}}],
        "messages": [_u("hi")],
    },
)
_p(
    "anth.content.cache-read",
    CB,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "system": [{"type": "text", "text": CACHE_FILLER, "cache_control": {"type": "ephemeral"}}],
        "messages": [_u("hi")],
    },
    depends_on="anth.content.cache-write",
    note="identical to cache-write; record cache_read_input_tokens",
)
_p(
    "anth.content.cache-ttl-1h",
    CB,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "system": [
            {
                "type": "text",
                "text": CACHE_FILLER,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
        "messages": [_u("hi")],
    },
    note="SMG CacheControl has no ttl; record vendor acceptance + ephemeral_1h",
)
_p(
    "anth.content.cache-control-on-tool-def",
    CB,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "tools": [
            {
                "name": "get_weather",
                "description": "weather " + CACHE_FILLER[:200],
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [_u("hi")],
    },
)

# =============================================================================
# streaming SSE transcripts
# =============================================================================
STR = "streaming"
_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
_p(
    "anth.stream.basic-text",
    STR,
    {"model": "@MODEL", "max_tokens": 64, "stream": True, "messages": [_u("count to three")]},
    stream=True,
)
_p(
    "anth.stream.max-tokens-1",
    STR,
    {"model": "@MODEL", "max_tokens": 1, "stream": True, "messages": [_u("hi")]},
    stream=True,
)
_p(
    "anth.stream.stop-sequence-fires",
    STR,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "stream": True,
        "stop_sequences": ["BODY"],
        "messages": [_u("Output START then BODY then END")],
    },
    stream=True,
)
_p(
    "anth.stream.tool-use",
    STR,
    {
        "model": "@MODEL",
        "max_tokens": 256,
        "stream": True,
        "tools": [_WEATHER_TOOL],
        "messages": [_u("Weather in Paris? Use the tool.")],
    },
    stream=True,
)
_p(
    "anth.stream.parallel-tools",
    STR,
    {
        "model": "@MODEL",
        "max_tokens": 512,
        "stream": True,
        "tools": [_WEATHER_TOOL],
        "messages": [_u("Weather in Paris and Tokyo at the same time? Use the tool.")],
    },
    stream=True,
)
_p(
    "anth.stream.tool-choice-forced",
    STR,
    {
        "model": "@MODEL",
        "max_tokens": 256,
        "stream": True,
        "tools": [_WEATHER_TOOL],
        "tool_choice": {"type": "tool", "name": "get_weather"},
        "messages": [_u("Weather in Paris?")],
    },
    stream=True,
)
_p(
    "anth.stream.thinking",
    STR,
    {
        "model": "@MODEL",
        "max_tokens": 2048,
        "stream": True,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [_u("What is 17*23? Think.")],
    },
    stream=True,
)
_p(
    "anth.stream.thinking-tools-interleaved",
    STR,
    {
        "model": "@MODEL",
        "max_tokens": 2048,
        "stream": True,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "tools": [_WEATHER_TOOL],
        "messages": [_u("Think, then get the weather in Paris with the tool.")],
    },
    stream=True,
    headers={"anthropic-beta": "interleaved-thinking-2025-05-14"},
)
_p(
    "anth.stream.citations",
    STR,
    {
        "model": "@MODEL",
        "max_tokens": 256,
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": "The capital of France is Paris.",
                        },
                        "citations": {"enabled": True},
                    },
                    {"type": "text", "text": "Capital of France? Cite."},
                ],
            }
        ],
    },
    stream=True,
)
_p(
    "anth.stream.system-multiturn",
    STR,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "stream": True,
        "system": "You are terse.",
        "messages": [_u("hi"), {"role": "assistant", "content": "hello"}, _u("bye")],
    },
    stream=True,
)
_p(
    "anth.stream.cache-hit",
    STR,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "stream": True,
        "system": [{"type": "text", "text": CACHE_FILLER, "cache_control": {"type": "ephemeral"}}],
        "messages": [_u("hi")],
    },
    stream=True,
    depends_on="anth.content.cache-write",
    note="usage fields in message_start vs message_delta",
)
_p(
    "anth.stream.service-tier",
    STR,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "stream": True,
        "service_tier": "standard_only",
        "messages": [_u("hi")],
    },
    stream=True,
)
_p(
    "anth.stream.error-pre-stream",
    STR,
    {"model": "@MODEL", "max_tokens": -1, "stream": True, "messages": [_u("hi")]},
    stream=True,
    expect="error",
    note="confirm plain HTTP 4xx JSON, not SSE error event",
)

# =============================================================================
# tools: definitions and tool_choice
# =============================================================================
TD = "tools-definitions"
_MIN_TOOL = {"name": "noop", "input_schema": {"type": "object", "properties": {}}}
_DESC_TOOL = {
    "name": "get_weather",
    "description": "Get weather",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
}
_p(
    "anth.tools.def-minimal",
    TD,
    {"model": "@MODEL", "max_tokens": 128, "tools": [_MIN_TOOL], "messages": [_u("hi")]},
)
_p(
    "anth.tools.def-with-description",
    TD,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "messages": [_u("Weather in Paris?")],
    },
)
_p(
    "anth.tools.def-explicit-type-custom",
    TD,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [dict(_DESC_TOOL, type="custom")],
        "messages": [_u("Weather in Paris?")],
    },
)
_p(
    "anth.tools.def-schema-enum-nested",
    TD,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "messages": [_u("book a flight")],
        "tools": [
            {
                "name": "book",
                "description": "book a trip",
                "input_schema": {
                    "type": "object",
                    "required": ["dest"],
                    "properties": {
                        "dest": {"type": "string"},
                        "cabin": {"type": "string", "enum": ["economy", "business"]},
                        "legs": {
                            "type": "array",
                            "items": {"type": "object", "properties": {"from": {"type": "string"}}},
                        },
                    },
                },
            }
        ],
    },
)
_p(
    "anth.tools.def-empty-properties",
    TD,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "messages": [_u("ping")],
        "tools": [{"name": "ping", "input_schema": {"type": "object"}}],
    },
)
_p(
    "anth.tools.def-additional-schema-keys",
    TD,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "messages": [_u("Weather in Paris?")],
        "tools": [
            {
                "name": "get_weather",
                "input_schema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"city": {"type": "string"}},
                },
            }
        ],
    },
)
_p(
    "anth.tools.tc-omitted",
    TD,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "messages": [_u("Weather in Paris?")],
    },
)
_p(
    "anth.tools.tc-auto-explicit",
    TD,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "tool_choice": {"type": "auto"},
        "messages": [_u("Weather in Paris?")],
    },
)
_p(
    "anth.tools.tc-auto-disable-parallel",
    TD,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        "messages": [_u("Weather in Paris and Tokyo?")],
    },
)
_p(
    "anth.tools.tc-any",
    TD,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "tool_choice": {"type": "any"},
        "messages": [_u("Weather in Paris?")],
    },
)
_p(
    "anth.tools.tc-any-disable-parallel",
    TD,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
        "messages": [_u("Weather in Paris and Tokyo?")],
    },
)
_p(
    "anth.tools.tc-tool-name",
    TD,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "tool_choice": {"type": "tool", "name": "get_weather"},
        "messages": [_u("Weather in Paris?")],
    },
)
_p(
    "anth.tools.tc-tool-disable-parallel",
    TD,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "tool_choice": {"type": "tool", "name": "get_weather", "disable_parallel_tool_use": True},
        "messages": [_u("Weather in Paris?")],
    },
)
_p(
    "anth.tools.tc-none-with-tools",
    TD,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "tool_choice": {"type": "none"},
        "messages": [_u("Weather in Paris?")],
    },
    note="model answers in text, no tool_use",
)
_p(
    "anth.tools.parallel-natural",
    TD,
    {
        "model": "@MODEL",
        "max_tokens": 512,
        "tools": [_DESC_TOOL],
        "messages": [_u("Weather in SF and NYC at the same time? Use the tool.")],
    },
)
_p(
    "anth.tools.parallel-suppressed",
    TD,
    {
        "model": "@MODEL",
        "max_tokens": 512,
        "tools": [_DESC_TOOL],
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        "messages": [_u("Weather in SF and NYC at the same time? Use the tool.")],
    },
)
_p(
    "anth.tools.forced-tool-irrelevant-prompt",
    TD,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "tool_choice": {"type": "tool", "name": "get_weather"},
        "messages": [_u("Tell me a joke.")],
    },
)
_p(
    "anth.tools.text-before-tool-use",
    TD,
    {
        "model": "@MODEL",
        "max_tokens": 256,
        "tools": [_DESC_TOOL],
        "messages": [_u("Explain what you'll do, then get the weather in Paris.")],
    },
)

# =============================================================================
# tool loop: tool_result round trips
# =============================================================================
TL = "tool-loop"


def _assistant_tool_use(tid="toolu_probe", name="get_weather", inp=None):
    return {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": tid, "name": name, "input": inp or {"city": "Paris"}}
        ],
    }


_p(
    "anth.toolloop.round-trip-string",
    TL,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "messages": [
            _u("Weather in Paris?"),
            _assistant_tool_use(),
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_probe",
                        "content": "20C and sunny",
                    }
                ],
            },
        ],
    },
)
_p(
    "anth.toolloop.result-blocks-text",
    TL,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "messages": [
            _u("Weather in Paris?"),
            _assistant_tool_use(),
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_probe",
                        "content": [{"type": "text", "text": "20C and sunny"}],
                    }
                ],
            },
        ],
    },
)
_p(
    "anth.toolloop.result-blocks-image",
    TL,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "messages": [
            _u("Weather in Paris?"),
            _assistant_tool_use(),
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_probe",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": PNG_B64,
                                },
                            }
                        ],
                    }
                ],
            },
        ],
    },
)
_p(
    "anth.toolloop.result-is-error",
    TL,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "messages": [
            _u("Weather in Paris?"),
            _assistant_tool_use(),
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_probe",
                        "is_error": True,
                        "content": "API timeout",
                    }
                ],
            },
        ],
    },
)
_p(
    "anth.toolloop.result-content-omitted",
    TL,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "messages": [
            _u("Weather in Paris?"),
            _assistant_tool_use(),
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_probe"}]},
        ],
    },
)
_p(
    "anth.toolloop.result-empty-string",
    TL,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "messages": [
            _u("Weather in Paris?"),
            _assistant_tool_use(),
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_probe", "content": ""}],
            },
        ],
    },
)
_p(
    "anth.toolloop.parallel-both-results",
    TL,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "messages": [
            _u("Weather in Paris and Tokyo?"),
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_a",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_b",
                        "name": "get_weather",
                        "input": {"city": "Tokyo"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_a", "content": "20C"},
                    {"type": "tool_result", "tool_use_id": "toolu_b", "content": "18C"},
                ],
            },
        ],
    },
)
_p(
    "anth.toolloop.parallel-results-swapped",
    TL,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "messages": [
            _u("Weather in Paris and Tokyo?"),
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_a",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_b",
                        "name": "get_weather",
                        "input": {"city": "Tokyo"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_b", "content": "18C"},
                    {"type": "tool_result", "tool_use_id": "toolu_a", "content": "20C"},
                ],
            },
        ],
    },
)
_p(
    "anth.toolloop.missing-one-result",
    TL,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "messages": [
            _u("Weather in Paris and Tokyo?"),
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_a",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_b",
                        "name": "get_weather",
                        "input": {"city": "Tokyo"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_a", "content": "20C"}],
            },
        ],
    },
    expect="error",
)
_p(
    "anth.toolloop.unknown-tool-use-id",
    TL,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "messages": [
            _u("Weather in Paris?"),
            _assistant_tool_use(),
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_wrong", "content": "20C"}
                ],
            },
        ],
    },
    expect="error",
)
_p(
    "anth.toolloop.result-without-prior-tool-use",
    TL,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_probe", "content": "20C"}
                ],
            }
        ],
    },
    expect="error",
)
_p(
    "anth.toolloop.followup-without-tools-param",
    TL,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "messages": [
            _u("Weather in Paris?"),
            _assistant_tool_use(),
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_probe", "content": "20C"}
                ],
            },
        ],
    },
    note="history has tool_use/tool_result but tools omitted; accept/reject?",
)

# =============================================================================
# extended thinking
# =============================================================================
TH = "thinking"
_p(
    "anth.think.enabled-min-budget",
    TH,
    {
        "model": "@MODEL",
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [_u("What is 17*23? Think.")],
    },
)
_p(
    "anth.think.enabled-display-summarized",
    TH,
    {
        "model": "@MODEL",
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024, "display": "summarized"},
        "messages": [_u("Why is the sky blue?")],
    },
    note="does haiku accept the display field? record",
)
_p(
    "anth.think.display-omitted",
    TH,
    {
        "model": "@MODEL",
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [_u("Why is the sky blue?")],
    },
)
_p(
    "anth.think.tools-auto-t1",
    TH,
    {
        "model": "@MODEL",
        "max_tokens": 2048,
        "tools": [_DESC_TOOL],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [_u("Think, then get the weather in Paris with the tool.")],
    },
)
_p(
    "anth.think.tools-roundtrip",
    TH,
    {
        "model": "@MODEL",
        "max_tokens": 2048,
        "tools": [_DESC_TOOL],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [
            _u("Think, then get the weather in Paris with the tool."),
            {"role": "assistant", "content": "{{anth.think.tools-auto-t1#body.content}}"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "{{anth.think.tools-auto-t1#body.content[?tool_use].id}}",
                        "content": "20C",
                    }
                ],
            },
        ],
    },
    depends_on="anth.think.tools-auto-t1",
    note="pass thinking block back unchanged; accepted?",
)
_p(
    "anth.think.loop-thinking-stripped",
    TH,
    {
        "model": "@MODEL",
        "max_tokens": 2048,
        "tools": [_DESC_TOOL],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [
            _u("Think, then get the weather in Paris with the tool."),
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_probe",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_probe", "content": "20C"}
                ],
            },
        ],
    },
    note="replay without thinking blocks; error-or-accept + message",
)
_p(
    "anth.think.tampered-signature",
    TH,
    {
        "model": "@MODEL",
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [
            _u("What is 17*23? Think."),
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Let me compute 17*23.",
                        "signature": "deadbeefTAMPERED",
                    }
                ],
            },
            _u("continue"),
        ],
    },
    expect="error",
    note="mutate signature; record 400",
)
_p(
    "anth.think.redacted-trigger",
    TH,
    {
        "model": "@MODEL",
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [_u("ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING")],
    },
    note="record redacted_thinking {data} block",
)
_p(
    "anth.think.redacted-trigger-stream",
    TH,
    {
        "model": "@MODEL",
        "max_tokens": 2048,
        "stream": True,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [_u("ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING")],
    },
    stream=True,
)
_p(
    "anth.think.disabled-explicit",
    TH,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "thinking": {"type": "disabled"},
        "messages": [_u("What is 17*23?")],
    },
)
# thinking errors
_p(
    "anth.think.budget-1023",
    TH,
    {
        "model": "@MODEL",
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1023},
        "messages": [_u("hi")],
    },
    expect="error",
    note="below min 1024",
)
_p(
    "anth.think.budget-gte-max-tokens",
    TH,
    {
        "model": "@MODEL",
        "max_tokens": 1024,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [_u("hi")],
    },
    expect="error",
    note="budget >= max_tokens",
)
_p(
    "anth.think.adaptive-on-haiku",
    TH,
    {
        "model": "@MODEL",
        "max_tokens": 2048,
        "thinking": {"type": "adaptive"},
        "messages": [_u("hi")],
    },
    expect="error",
    note="SMG models Adaptive; haiku likely rejects",
)
_p(
    "anth.think.with-tc-any",
    TH,
    {
        "model": "@MODEL",
        "max_tokens": 2048,
        "tools": [_DESC_TOOL],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "tool_choice": {"type": "any"},
        "messages": [_u("Weather in Paris?")],
    },
    expect="error",
    note="thinking incompatible with forced tool_choice",
)
_p(
    "anth.think.with-tc-tool",
    TH,
    {
        "model": "@MODEL",
        "max_tokens": 2048,
        "tools": [_DESC_TOOL],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "tool_choice": {"type": "tool", "name": "get_weather"},
        "messages": [_u("Weather in Paris?")],
    },
    expect="error",
)
_p(
    "anth.think.with-temperature",
    TH,
    {
        "model": "@MODEL",
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "temperature": 0.5,
        "messages": [_u("hi")],
    },
    expect="error",
    note="only temp=1 allowed with thinking",
)

# =============================================================================
# stop_sequences and max_tokens edges
# =============================================================================
SM = "stop-and-max-tokens"
_p(
    "anth.stop.single-fires",
    SM,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "stop_sequences": ["BODY"],
        "messages": [_u("Output the word START then BODY then END")],
    },
)
_p(
    "anth.stop.multi-sequences",
    SM,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "stop_sequences": ["FOO", "BODY"],
        "messages": [_u("Output START then BODY then END")],
    },
)
_p(
    "anth.stop.not-triggered",
    SM,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "stop_sequences": ["ZZZZZ"],
        "messages": [_u("say hello")],
    },
)
_p(
    "anth.stop.during-tool-json",
    SM,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "stop_sequences": ["Paris"],
        "messages": [_u("Weather in Paris? Use the tool.")],
    },
    note="does tool_use generation honor stop sequences?",
)
_p(
    "anth.stop.max-tokens-1",
    SM,
    {"model": "@MODEL", "max_tokens": 1, "messages": [_u("count to ten")]},
)
_p(
    "anth.stop.max-tokens-exact-boundary",
    SM,
    {"model": "@MODEL", "max_tokens": 3, "messages": [_u("Answer in one word: sky color?")]},
)
_p(
    "anth.stop.max-tokens-0",
    SM,
    {"model": "@MODEL", "max_tokens": 0, "messages": [_u("hi")]},
    note="record 400 vs cache-prewarm content:[] semantics",
)
_p(
    "anth.stop.max-tokens-above-cap",
    SM,
    {"model": "@MODEL", "max_tokens": 100000, "messages": [_u("hi")]},
    expect="error",
    note="above haiku 64K cap; record message naming the cap",
)
_p(
    "anth.stop.context-window-overflow",
    SM,
    {"model": "@MODEL", "max_tokens": 4096, "messages": [_u("word " * 200000)]},
    note="~200K filler; record stop_reason model_context_window_exceeded "
    "(MISSING from SMG StopReason); gated by ANTH_PROBE_EXPENSIVE",
)
_p(
    "anth.stop.whitespace-only-sequence",
    SM,
    {"model": "@MODEL", "max_tokens": 64, "stop_sequences": ["   "], "messages": [_u("hi")]},
    expect="error",
)

# =============================================================================
# count_tokens parity
# =============================================================================
CT = "count-tokens"
_CTE = "/v1/messages/count_tokens"
_p("anth.count.minimal", CT, {"model": "@MODEL", "messages": [_u("hi there")]}, endpoint=_CTE)
_p(
    "anth.count.with-system-string",
    CT,
    {"model": "@MODEL", "system": "You are terse.", "messages": [_u("hi there")]},
    endpoint=_CTE,
)
_p(
    "anth.count.with-system-blocks",
    CT,
    {
        "model": "@MODEL",
        "system": [{"type": "text", "text": "You are terse."}],
        "messages": [_u("hi there")],
    },
    endpoint=_CTE,
)
_p(
    "anth.count.with-tools",
    CT,
    {"model": "@MODEL", "tools": [_DESC_TOOL], "messages": [_u("Weather in Paris?")]},
    endpoint=_CTE,
)
_p(
    "anth.count.with-tool-choice",
    CT,
    {
        "model": "@MODEL",
        "tools": [_DESC_TOOL],
        "tool_choice": {"type": "tool", "name": "get_weather"},
        "messages": [_u("Weather in Paris?")],
    },
    endpoint=_CTE,
)
_p(
    "anth.count.with-thinking",
    CT,
    {
        "model": "@MODEL",
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [_u("Think about 2+2.")],
    },
    endpoint=_CTE,
)
_p(
    "anth.count.with-image",
    CT,
    {
        "model": "@MODEL",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64},
                    },
                ],
            }
        ],
    },
    endpoint=_CTE,
)
_p(
    "anth.count.empty-messages",
    CT,
    {"model": "@MODEL", "messages": []},
    endpoint=_CTE,
    expect="error",
)
_p(
    "anth.count.invalid-model",
    CT,
    {"model": "claude-nope", "messages": [_u("hi")]},
    endpoint=_CTE,
    expect="error",
)

# =============================================================================
# models endpoint
# =============================================================================
MD = "models"
_p("anth.models.list-default", MD, None, method="GET", endpoint="/v1/models")
_p("anth.models.list-paginated", MD, None, method="GET", endpoint="/v1/models?limit=1")
_p("anth.models.retrieve-haiku", MD, None, method="GET", endpoint="/v1/models/@MODEL")
_p(
    "anth.models.retrieve-bogus",
    MD,
    None,
    method="GET",
    endpoint="/v1/models/claude-nope",
    expect="error",
)

# =============================================================================
# error probes: request validation
# =============================================================================
ER = "error-probes"


def _e(slug, body, **kw):
    _p(f"anth.err.{slug}", ER, body, expect="error", **kw)


_e("invalid-model", {"model": "claude-nope", "max_tokens": 64, "messages": [_u("hi")]})
_e("missing-max-tokens", {"model": "@MODEL", "messages": [_u("hi")]})
_e("missing-model", {"max_tokens": 64, "messages": [_u("hi")]})
_e("missing-messages", {"model": "@MODEL", "max_tokens": 64})
_e("empty-messages", {"model": "@MODEL", "max_tokens": 64, "messages": []})
_e("max-tokens-negative", {"model": "@MODEL", "max_tokens": -1, "messages": [_u("hi")]})
_e("max-tokens-string", {"model": "@MODEL", "max_tokens": "many", "messages": [_u("hi")]})
_e(
    "assistant-first",
    {"model": "@MODEL", "max_tokens": 64, "messages": [{"role": "assistant", "content": "hi"}]},
)
_e(
    "assistant-assistant-consecutive",
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [
            _u("hi"),
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b"},
        ],
    },
)
_e(
    "system-role-in-messages-first",
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [{"role": "system", "content": "be terse"}, _u("hi")],
    },
    note="SMG Role::System accepts; haiku likely 400s — KEY divergence",
)
_e(
    "system-role-in-messages-mid",
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [_u("hi"), {"role": "system", "content": "be terse"}, _u("bye")],
    },
    note="KEY divergence probe",
)
_e(
    "empty-content-string",
    {"model": "@MODEL", "max_tokens": 64, "messages": [{"role": "user", "content": ""}]},
)
_e(
    "empty-content-blocks",
    {"model": "@MODEL", "max_tokens": 64, "messages": [{"role": "user", "content": []}]},
)
_e(
    "text-block-empty-text",
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": [{"type": "text", "text": ""}]}],
    },
)
_e(
    "prefill-trailing-whitespace",
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [_u("hi"), {"role": "assistant", "content": "hello "}],
    },
)
_e(
    "unknown-top-level-field",
    {"model": "@MODEL", "max_tokens": 64, "messages": [_u("hi")], "frobnicate": 1},
)
_e(
    "rid-field",
    {"model": "@MODEL", "max_tokens": 64, "messages": [_u("hi")], "rid": "probe-rid"},
    note="SGLang extension; expect unexpected-field 400",
)
_e(
    "unknown-field-in-message",
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi", "extra": 1}],
    },
)
_e(
    "unknown-content-block-type",
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": [{"type": "banana", "x": 1}]}],
    },
)
_e(
    "unknown-field-in-tool-def",
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [_u("hi")],
        "tools": [{"name": "t", "banana": 1, "input_schema": {"type": "object"}}],
    },
)
_e(
    "invalid-input-schema-type",
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [_u("hi")],
        "tools": [{"name": "t", "input_schema": {"type": "banana"}}],
    },
)
_e(
    "input-schema-missing-type",
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [_u("hi")],
        "tools": [{"name": "t", "input_schema": {"properties": {}}}],
    },
)
_e(
    "tool-name-invalid-chars",
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [_u("hi")],
        "tools": [{"name": "my tool!", "input_schema": {"type": "object"}}],
    },
)
_e(
    "tool-name-too-long",
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [_u("hi")],
        "tools": [{"name": "t" * 200, "input_schema": {"type": "object"}}],
    },
)
_e(
    "duplicate-tool-names",
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [_u("hi")],
        "tools": [
            {"name": "dup", "input_schema": {"type": "object"}},
            {"name": "dup", "input_schema": {"type": "object"}},
        ],
    },
)
_e(
    "tc-tool-name-not-in-tools",
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [_u("hi")],
        "tools": [_DESC_TOOL],
        "tool_choice": {"type": "tool", "name": "nope"},
    },
)
_e(
    "tc-with-no-tools",
    {"model": "@MODEL", "max_tokens": 64, "messages": [_u("hi")], "tool_choice": {"type": "any"}},
)
_e(
    "temperature-2.0",
    {"model": "@MODEL", "max_tokens": 64, "messages": [_u("hi")], "temperature": 2.0},
)
_e(
    "temperature-negative",
    {"model": "@MODEL", "max_tokens": 64, "messages": [_u("hi")], "temperature": -1},
)
_e("top-p-1.5", {"model": "@MODEL", "max_tokens": 64, "messages": [_u("hi")], "top_p": 1.5})
_e(
    "temp-plus-top-p",
    {"model": "@MODEL", "max_tokens": 64, "messages": [_u("hi")], "temperature": 0.5, "top_p": 0.5},
    note="Claude4+ reject-both rule — record",
)
_e(
    "oversized-system",
    {"model": "@MODEL", "max_tokens": 64, "system": "word " * 210000, "messages": [_u("hi")]},
    note="~210K tokens > 200K window; prompt-too-long, not billed",
)
_e(
    "metadata-user-id-too-long",
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [_u("hi")],
        "metadata": {"user_id": "u" * 300},
    },
)
_e(
    "image-bad-media-type",
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/tiff", "data": PNG_B64},
                    }
                ],
            }
        ],
    },
)
_e(
    "image-corrupt-base64",
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "not!!base64!!",
                        },
                    }
                ],
            }
        ],
    },
)
_p(
    "anth.err.bad-api-key",
    ER,
    {"model": "@MODEL", "max_tokens": 64, "messages": [_u("hi")]},
    expect="error",
    headers={"x-api-key": "sk-ant-invalid-probe"},
    note="force 401",
)
_p(
    "anth.err.missing-anthropic-version",
    ER,
    {"model": "@MODEL", "max_tokens": 64, "messages": [_u("hi")]},
    expect="error",
    headers={"anthropic-version": None},
    note="drop version header",
)
_p(
    "anth.err.bogus-anthropic-version",
    ER,
    {"model": "@MODEL", "max_tokens": 64, "messages": [_u("hi")]},
    expect="error",
    headers={"anthropic-version": "1999-01-01"},
    note="bogus version header",
)

# =============================================================================
# pairwise interactions
# =============================================================================
PW = "pairwise"
_p(
    "anth.pair.tools-system-cache-write",
    PW,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "system": [{"type": "text", "text": CACHE_FILLER, "cache_control": {"type": "ephemeral"}}],
        "messages": [_u("Weather in Paris?")],
    },
)
_p(
    "anth.pair.tools-system-cache-read",
    PW,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "system": [{"type": "text", "text": CACHE_FILLER, "cache_control": {"type": "ephemeral"}}],
        "messages": [_u("Weather in Paris?")],
    },
    depends_on="anth.pair.tools-system-cache-write",
    note="cache spans tools+system; record read",
)
_p(
    "anth.pair.temp-top-k",
    PW,
    {"model": "@MODEL", "max_tokens": 64, "messages": [_u("hi")], "temperature": 0.5, "top_k": 10},
)
_p(
    "anth.pair.top-p-top-k",
    PW,
    {"model": "@MODEL", "max_tokens": 64, "messages": [_u("hi")], "top_p": 0.5, "top_k": 10},
)
_p(
    "anth.pair.temp-top-p-top-k",
    PW,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": [_u("hi")],
        "temperature": 0.5,
        "top_p": 0.5,
        "top_k": 10,
    },
)
_p(
    "anth.pair.prefill-tools",
    PW,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "messages": [_u("Weather in Paris?"), {"role": "assistant", "content": "I will check:"}],
    },
)
_p(
    "anth.pair.stop-seq-thinking",
    PW,
    {
        "model": "@MODEL",
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "stop_sequences": ["compute"],
        "messages": [_u("Think then compute 2+2.")],
    },
    note="does sequence match inside thinking terminate?",
)
_p(
    "anth.pair.metadata-stream",
    PW,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "stream": True,
        "messages": [_u("hi")],
        "metadata": {"user_id": "probe-user-2"},
    },
    stream=True,
    note="confirm user_id echoed nowhere",
)
_p(
    "anth.pair.service-tier-standard-stream",
    PW,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "stream": True,
        "service_tier": "standard_only",
        "messages": [_u("hi")],
    },
    stream=True,
)
_p(
    "anth.pair.tc-none-tool-history",
    PW,
    {
        "model": "@MODEL",
        "max_tokens": 128,
        "tools": [_DESC_TOOL],
        "tool_choice": {"type": "none"},
        "messages": [
            _u("Weather in Paris?"),
            _assistant_tool_use(),
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_probe", "content": "20C"}
                ],
            },
        ],
    },
    note="Claude-Code-replay shape",
)
_p(
    "anth.pair.long-conversation-20-turns",
    PW,
    {
        "model": "@MODEL",
        "max_tokens": 64,
        "messages": (
            [
                {"role": ("user" if i % 2 == 0 else "assistant"), "content": f"turn {i}"}
                for i in range(19)
            ]
            + [_u("final turn?")]
        ),
    },
    note="alternation + usage growth",
)
