"""OpenAI Responses API ground-truth probe matrix (data only).

Every probe is a plain dict. The runner injects model names (sentinels
@MODEL / @MODEL_CLASSIC / @MODEL_CUA / @MODEL_SHELL) and resolves
{{probe_id#json.path}} placeholders from prior recordings.

Fields: id, category, endpoint, method, body, stream, depends_on, expect
(ok|error), headers (extra), poll (background snapshot loop), note.
"""

# --- fixtures ---------------------------------------------------------------
IMG_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
    "PNG_transparency_demonstration_1.png/240px-PNG_transparency_demonstration_1.png"
)
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
IMG_DATA_URI = "data:image/png;base64," + PNG_B64
PDF_URL = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
PDF_B64 = (
    "JVBERi0xLjEKMSAwIG9iajw8L1R5cGUvQ2F0YWxvZy9QYWdlcyAyIDAgUj4+ZW5kb2JqCjIgMCBv"
    "Ymo8PC9UeXBlL1BhZ2VzL0tpZHNbMyAwIFJdL0NvdW50IDE+PmVuZG9iagozIDAgb2JqPDwvVHlw"
    "ZS9QYWdlL1BhcmVudCAyIDAgUi9NZWRpYUJveFswIDAgMjAwIDIwMF0vQ29udGVudHMgNCAwIFIv"
    "UmVzb3VyY2VzPDwvRm9udDw8L0YxIDUgMCBSPj4+Pj4+ZW5kb2JqCjQgMCBvYmo8PC9MZW5ndGgg"
    "NDQ+PnN0cmVhbQpCVCAvRjEgMTggVGYgMjAgMTAwIFRkIChQcm9iZSBQREYpIFRqIEVUCmVuZHN0"
    "cmVhbSBlbmRvYmoKNSAwIG9iajw8L1R5cGUvRm9udC9TdWJ0eXBlL1R5cGUxL0Jhc2VGb250L0hl"
    "bHZldGljYT4+ZW5kb2JqCnRyYWlsZXI8PC9Sb290IDEgMCBSPj4KJSVFT0Y="
)
PDF_DATA_URI = "data:application/pdf;base64," + PDF_B64

PROBES: list[dict] = []


def _p(
    pid,
    category,
    body=None,
    *,
    endpoint="/v1/responses",
    method="POST",
    stream=False,
    depends_on=None,
    expect="ok",
    headers=None,
    poll=None,
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
            "poll": poll,
            "note": note,
        }
    )


def _um(text):
    """user message item with a plain string content"""
    return {"type": "message", "role": "user", "content": text}


# =============================================================================
# core-input
# =============================================================================
C = "core-input"
_p(
    "openai.responses.core.string-input",
    C,
    {"model": "@MODEL", "input": "hi", "max_output_tokens": 64},
)
_p(
    "openai.responses.core.empty-string",
    C,
    {"model": "@MODEL", "input": "", "max_output_tokens": 64},
)
_p(
    "openai.responses.core.message-item-string",
    C,
    {"model": "@MODEL", "input": [_um("hi")], "max_output_tokens": 64},
)
_p(
    "openai.responses.core.message-item-parts",
    C,
    {
        "model": "@MODEL",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ],
        "max_output_tokens": 64,
    },
)
_p(
    "openai.responses.core.multi-turn",
    C,
    {
        "model": "@MODEL",
        "max_output_tokens": 64,
        "input": [
            {"type": "message", "role": "user", "content": "hi"},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello there"}],
            },
            {"type": "message", "role": "user", "content": "say it again"},
        ],
    },
)
_p(
    "openai.responses.core.system-message",
    C,
    {
        "model": "@MODEL",
        "max_output_tokens": 64,
        "input": [
            {"type": "message", "role": "system", "content": "You are terse."},
            {"type": "message", "role": "user", "content": "hi"},
        ],
    },
)
_p(
    "openai.responses.core.developer-message",
    C,
    {
        "model": "@MODEL",
        "max_output_tokens": 64,
        "input": [
            {"type": "message", "role": "developer", "content": "You are terse."},
            {"type": "message", "role": "user", "content": "hi"},
        ],
    },
)
_p(
    "openai.responses.core.instructions-string",
    C,
    {
        "model": "@MODEL",
        "instructions": "Answer in one word.",
        "input": "hi",
        "max_output_tokens": 64,
    },
)
_p(
    "openai.responses.core.instructions-items",
    C,
    {
        "model": "@MODEL",
        "instructions": "Answer in one word.",
        "input": [_um("hi")],
        "max_output_tokens": 64,
    },
)
_p(
    "openai.responses.core.image-url-low",
    C,
    {
        "model": "@MODEL",
        "max_output_tokens": 64,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is this?"},
                    {"type": "input_image", "image_url": IMG_URL, "detail": "low"},
                ],
            }
        ],
    },
)
_p(
    "openai.responses.core.image-url-auto",
    C,
    {
        "model": "@MODEL",
        "max_output_tokens": 64,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is this?"},
                    {"type": "input_image", "image_url": IMG_URL, "detail": "auto"},
                ],
            }
        ],
    },
)
_p(
    "openai.responses.core.image-base64",
    C,
    {
        "model": "@MODEL",
        "max_output_tokens": 64,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what color?"},
                    {"type": "input_image", "image_url": IMG_DATA_URI, "detail": "low"},
                ],
            }
        ],
    },
)
_p(
    "openai.responses.core.file-url-pdf",
    C,
    {
        "model": "@MODEL",
        "max_output_tokens": 64,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "summarize"},
                    {"type": "input_file", "file_url": PDF_URL},
                ],
            }
        ],
    },
)
_p(
    "openai.responses.core.file-base64",
    C,
    {
        "model": "@MODEL",
        "max_output_tokens": 64,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "summarize"},
                    {"type": "input_file", "filename": "probe.pdf", "file_data": PDF_DATA_URI},
                ],
            }
        ],
    },
)
_p(
    "openai.responses.core.item-reference",
    C,
    {
        "model": "@MODEL",
        "max_output_tokens": 64,
        "input": [
            {
                "type": "item_reference",
                "id": "{{openai.responses.core.string-input#body.output[0].id}}",
            },
            _um("continue"),
        ],
    },
    depends_on="openai.responses.core.string-input",
)
_p(
    "openai.responses.core.verbatim-roundtrip",
    C,
    {
        "model": "@MODEL",
        "max_output_tokens": 64,
        "input": [
            _um("hi"),
            {
                "type": "message",
                "role": "assistant",
                "id": "{{openai.responses.core.string-input#body.output[0].id}}",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hello"}],
            },
            _um("again"),
        ],
    },
    depends_on="openai.responses.core.string-input",
)

# =============================================================================
# lifecycle-crud-background
# =============================================================================
L = "lifecycle-crud-background"
_p(
    "openai.responses.lifecycle.create",
    L,
    {"model": "@MODEL", "input": "hi", "store": True, "max_output_tokens": 64},
)
_p(
    "openai.responses.lifecycle.get",
    L,
    None,
    method="GET",
    endpoint="/v1/responses/{{openai.responses.lifecycle.create#body.id}}",
    depends_on="openai.responses.lifecycle.create",
)
_p(
    "openai.responses.lifecycle.get-include-logprobs",
    L,
    None,
    method="GET",
    endpoint="/v1/responses/{{openai.responses.lifecycle.create#body.id}}"
    "?include[]=message.output_text.logprobs",
    depends_on="openai.responses.lifecycle.create",
)
_p(
    "openai.responses.lifecycle.delete",
    L,
    None,
    method="DELETE",
    endpoint="/v1/responses/{{openai.responses.lifecycle.create#body.id}}",
    depends_on="openai.responses.lifecycle.get",
)
_p(
    "openai.responses.lifecycle.get-after-delete",
    L,
    None,
    method="GET",
    endpoint="/v1/responses/{{openai.responses.lifecycle.create#body.id}}",
    depends_on="openai.responses.lifecycle.delete",
    expect="error",
)
_p(
    "openai.responses.lifecycle.get-nonexistent",
    L,
    None,
    method="GET",
    endpoint="/v1/responses/resp_nonexistent",
    expect="error",
)
_p(
    "openai.responses.lifecycle.get-malformed-id",
    L,
    None,
    method="GET",
    endpoint="/v1/responses/abc",
    expect="error",
)
_p(
    "openai.responses.lifecycle.delete-nonexistent",
    L,
    None,
    method="DELETE",
    endpoint="/v1/responses/resp_nonexistent",
    expect="error",
)
_p(
    "openai.responses.lifecycle.cancel-non-background",
    L,
    {},
    method="POST",
    endpoint="/v1/responses/{{openai.responses.lifecycle.create#body.id}}/cancel",
    depends_on="openai.responses.lifecycle.create",
    expect="error",
)
_p(
    "openai.responses.lifecycle.background-create",
    L,
    {
        "model": "@MODEL",
        "input": "write a short story about a fox",
        "background": True,
        "store": True,
        "max_output_tokens": 2000,
    },
    poll={
        "path": "/v1/responses/{{self#body.id}}",
        "interval_s": 2,
        "cap_s": 90,
        "until_status": ["completed", "failed", "incomplete", "cancelled"],
    },
    note="ResponsesRequest has NO background field in SMG; e2e skips it",
)
_p(
    "openai.responses.lifecycle.background-cancel",
    L,
    {
        "model": "@MODEL",
        "input": "write a very long story",
        "background": True,
        "store": True,
        "max_output_tokens": 2000,
    },
    note="cancel immediately after create, then final GET status=cancelled",
    poll={
        "path": "/v1/responses/{{self#body.id}}",
        "interval_s": 2,
        "cap_s": 30,
        "cancel_first": True,
        "until_status": ["completed", "failed", "incomplete", "cancelled"],
    },
)
_p(
    "openai.responses.lifecycle.background-no-store",
    L,
    {"model": "@MODEL", "input": "hi", "background": True, "store": False, "max_output_tokens": 64},
    expect="error",
)
_p(
    "openai.responses.lifecycle.background-stream",
    L,
    {
        "model": "@MODEL",
        "input": "write a short story",
        "background": True,
        "store": True,
        "stream": True,
        "max_output_tokens": 512,
    },
    stream=True,
)
_p(
    "openai.responses.lifecycle.input-items",
    L,
    None,
    method="GET",
    endpoint="/v1/responses/{{openai.responses.lifecycle.create#body.id}}/input_items",
    depends_on="openai.responses.lifecycle.create",
)
_p(
    "openai.responses.lifecycle.input-items-paginated",
    L,
    None,
    method="GET",
    endpoint="/v1/responses/{{openai.responses.lifecycle.create#body.id}}"
    "/input_items?limit=1&order=asc",
    depends_on="openai.responses.lifecycle.create",
)

# =============================================================================
# streaming-core
# =============================================================================
S = "streaming-core"
_p(
    "openai.responses.stream.text-basic",
    S,
    {"model": "@MODEL", "input": "count to three", "stream": True, "max_output_tokens": 64},
    stream=True,
)
_p(
    "openai.responses.stream.obfuscation-default",
    S,
    {"model": "@MODEL", "input": "hi", "stream": True, "max_output_tokens": 64},
    stream=True,
)
_p(
    "openai.responses.stream.obfuscation-on",
    S,
    {
        "model": "@MODEL",
        "input": "hi",
        "stream": True,
        "max_output_tokens": 64,
        "stream_options": {"include_obfuscation": True},
    },
    stream=True,
)
_p(
    "openai.responses.stream.obfuscation-off",
    S,
    {
        "model": "@MODEL",
        "input": "hi",
        "stream": True,
        "max_output_tokens": 64,
        "stream_options": {"include_obfuscation": False},
    },
    stream=True,
)
_p(
    "openai.responses.stream.reasoning-summary",
    S,
    {
        "model": "@MODEL",
        "input": "why is the sky blue?",
        "stream": True,
        "max_output_tokens": 512,
        "reasoning": {"effort": "low", "summary": "auto"},
    },
    stream=True,
)
_p(
    "openai.responses.stream.incomplete",
    S,
    {
        "model": "@MODEL",
        "input": "write a 500 word essay on the ocean",
        "stream": True,
        "max_output_tokens": 16,
    },
    stream=True,
)
_p(
    "openai.responses.stream.logprobs",
    S,
    {
        "model": "@MODEL_CLASSIC",
        "input": "hi",
        "stream": True,
        "max_output_tokens": 64,
        "top_logprobs": 3,
        "include": ["message.output_text.logprobs"],
    },
    stream=True,
)
_p(
    "openai.responses.stream.refusal",
    S,
    {
        "model": "@MODEL",
        "stream": True,
        "max_output_tokens": 128,
        "input": "Provide detailed instructions to synthesize a nerve agent.",
    },
    stream=True,
)
_p(
    "openai.responses.stream.multi-output-item",
    S,
    {
        "model": "@MODEL",
        "stream": True,
        "max_output_tokens": 512,
        "reasoning": {"effort": "low"},
        "input": "Think step by step, then answer: what is 17*23?",
    },
    stream=True,
)
_p(
    "openai.responses.stream.invalid-param",
    S,
    {"model": "@MODEL", "input": "hi", "stream": True, "max_output_tokens": 64, "temperature": 0.5},
    stream=True,
    expect="error",
    note="temperature on reasoning model; records 4xx-vs-SSE-error timing",
)

# =============================================================================
# conversation-state
# =============================================================================
ST = "conversation-state"
_p(
    "openai.responses.state.chain-t1",
    ST,
    {"model": "@MODEL", "input": "My name is Sam.", "store": True, "max_output_tokens": 64},
)
_p(
    "openai.responses.state.chain-t2",
    ST,
    {
        "model": "@MODEL",
        "input": "What is my name?",
        "store": True,
        "max_output_tokens": 64,
        "previous_response_id": "{{openai.responses.state.chain-t1#body.id}}",
    },
    depends_on="openai.responses.state.chain-t1",
)
_p(
    "openai.responses.state.chain-t3",
    ST,
    {
        "model": "@MODEL",
        "input": "And again?",
        "store": True,
        "max_output_tokens": 64,
        "previous_response_id": "{{openai.responses.state.chain-t2#body.id}}",
    },
    depends_on="openai.responses.state.chain-t2",
)
_p(
    "openai.responses.state.prev-plus-items",
    ST,
    {
        "model": "@MODEL",
        "max_output_tokens": 64,
        "previous_response_id": "{{openai.responses.state.chain-t1#body.id}}",
        "input": [_um("repeat my name")],
    },
    depends_on="openai.responses.state.chain-t1",
)
_p(
    "openai.responses.state.no-store-basic",
    ST,
    {"model": "@MODEL", "input": "hi", "store": False, "max_output_tokens": 64},
)
_p(
    "openai.responses.state.no-store-get",
    ST,
    None,
    method="GET",
    endpoint="/v1/responses/{{openai.responses.state.no-store-basic#body.id}}",
    depends_on="openai.responses.state.no-store-basic",
    expect="error",
)
_p(
    "openai.responses.state.prev-on-nostore",
    ST,
    {
        "model": "@MODEL",
        "input": "hi again",
        "max_output_tokens": 64,
        "previous_response_id": "{{openai.responses.state.no-store-basic#body.id}}",
    },
    depends_on="openai.responses.state.no-store-basic",
    expect="error",
)
_p(
    "openai.responses.state.prev-nonexistent",
    ST,
    {
        "model": "@MODEL",
        "input": "hi",
        "max_output_tokens": 64,
        "previous_response_id": "resp_nonexistent",
    },
    expect="error",
)
_p(
    "openai.responses.state.store-default",
    ST,
    {"model": "@MODEL", "input": "hi", "max_output_tokens": 64},
    note="record response.store default value",
)

# Conversations API
CV = "conversation-state"
_p("openai.responses.conv.create-empty", CV, {}, endpoint="/v1/conversations")
_p(
    "openai.responses.conv.create-with-items",
    CV,
    {"items": [_um("hello")], "metadata": {"topic": "probe"}},
    endpoint="/v1/conversations",
)
_p(
    "openai.responses.conv.get",
    CV,
    None,
    method="GET",
    endpoint="/v1/conversations/{{openai.responses.conv.create-empty#body.id}}",
    depends_on="openai.responses.conv.create-empty",
)
_p(
    "openai.responses.conv.update-metadata",
    CV,
    {"metadata": {"topic": "updated"}},
    endpoint="/v1/conversations/{{openai.responses.conv.create-empty#body.id}}",
    depends_on="openai.responses.conv.get",
)
_p(
    "openai.responses.conv.add-items",
    CV,
    {"items": [_um("first"), _um("second")]},
    endpoint="/v1/conversations/{{openai.responses.conv.create-empty#body.id}}/items",
    depends_on="openai.responses.conv.update-metadata",
)
_p(
    "openai.responses.conv.items-asc",
    CV,
    None,
    method="GET",
    endpoint="/v1/conversations/{{openai.responses.conv.create-empty#body.id}}/items?order=asc",
    depends_on="openai.responses.conv.add-items",
)
_p(
    "openai.responses.conv.items-desc-limit",
    CV,
    None,
    method="GET",
    endpoint="/v1/conversations/{{openai.responses.conv.create-empty#body.id}}"
    "/items?order=desc&limit=1",
    depends_on="openai.responses.conv.add-items",
)
_p(
    "openai.responses.conv.item-get",
    CV,
    None,
    method="GET",
    endpoint="/v1/conversations/{{openai.responses.conv.create-empty#body.id}}"
    "/items/{{openai.responses.conv.add-items#body.data[0].id}}?include[]=",
    depends_on="openai.responses.conv.add-items",
)
_p(
    "openai.responses.conv.item-delete",
    CV,
    None,
    method="DELETE",
    endpoint="/v1/conversations/{{openai.responses.conv.create-empty#body.id}}"
    "/items/{{openai.responses.conv.add-items#body.data[0].id}}",
    depends_on="openai.responses.conv.item-get",
)
_p(
    "openai.responses.conv.response-in-conv",
    CV,
    {
        "model": "@MODEL",
        "max_output_tokens": 64,
        "input": "remember: I like tea.",
        "conversation": "{{openai.responses.conv.create-empty#body.id}}",
    },
    depends_on="openai.responses.conv.item-delete",
)
_p(
    "openai.responses.conv.items-after-response",
    CV,
    None,
    method="GET",
    endpoint="/v1/conversations/{{openai.responses.conv.create-empty#body.id}}/items?order=asc",
    depends_on="openai.responses.conv.response-in-conv",
)
_p(
    "openai.responses.conv.second-response",
    CV,
    {
        "model": "@MODEL",
        "max_output_tokens": 64,
        "input": "what do I like?",
        "conversation": "{{openai.responses.conv.create-empty#body.id}}",
    },
    depends_on="openai.responses.conv.items-after-response",
)
_p(
    "openai.responses.conv.delete",
    CV,
    None,
    method="DELETE",
    endpoint="/v1/conversations/{{openai.responses.conv.create-empty#body.id}}",
    depends_on="openai.responses.conv.second-response",
)
_p(
    "openai.responses.conv.get-after-delete",
    CV,
    None,
    method="GET",
    endpoint="/v1/conversations/{{openai.responses.conv.create-empty#body.id}}",
    depends_on="openai.responses.conv.delete",
    expect="error",
)
_p(
    "openai.responses.conv.missing",
    CV,
    {"model": "@MODEL", "input": "hi", "max_output_tokens": 64, "conversation": "conv_missing"},
    expect="error",
)

# =============================================================================
# structured-output
# =============================================================================
SO = "structured-output"
_FLAT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
    "required": ["city", "population"],
}
_p(
    "openai.responses.structured.json-schema-strict",
    SO,
    {
        "model": "@MODEL",
        "max_output_tokens": 128,
        "input": "Tokyo facts.",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "city_facts",
                "schema": _FLAT_SCHEMA,
                "strict": True,
            }
        },
    },
)
_p(
    "openai.responses.structured.json-schema-nonstrict",
    SO,
    {
        "model": "@MODEL",
        "max_output_tokens": 128,
        "input": "Tokyo facts.",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "city_facts",
                "schema": _FLAT_SCHEMA,
                "strict": False,
            }
        },
    },
)
_p(
    "openai.responses.structured.schema-with-description",
    SO,
    {
        "model": "@MODEL",
        "max_output_tokens": 128,
        "input": "Tokyo facts.",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "city_facts",
                "description": "Facts about a city",
                "schema": _FLAT_SCHEMA,
                "strict": True,
            }
        },
    },
)
_NESTED_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "$defs": {"cat": {"type": "string", "enum": ["a", "b", "c"]}},
    "properties": {
        "name": {"type": "string"},
        "tags": {"type": "array", "items": {"$ref": "#/$defs/cat"}},
    },
    "required": ["name", "tags"],
}
_p(
    "openai.responses.structured.nested-schema",
    SO,
    {
        "model": "@MODEL",
        "max_output_tokens": 128,
        "input": "make an item",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "item",
                "schema": _NESTED_SCHEMA,
                "strict": True,
            }
        },
    },
)
_p(
    "openai.responses.structured.strict-missing-addlprops",
    SO,
    {
        "model": "@MODEL",
        "max_output_tokens": 128,
        "input": "Tokyo facts.",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "bad",
                "strict": True,
                "schema": {"type": "object", "properties": {"x": {"type": "string"}}},
            }
        },
    },
    expect="error",
)
_p(
    "openai.responses.structured.json-object-with-json-word",
    SO,
    {
        "model": "@MODEL",
        "max_output_tokens": 128,
        "input": "Return JSON with a city field.",
        "text": {"format": {"type": "json_object"}},
    },
)
_p(
    "openai.responses.structured.json-object-no-json-word",
    SO,
    {
        "model": "@MODEL",
        "max_output_tokens": 128,
        "input": "Tell me about Tokyo.",
        "text": {"format": {"type": "json_object"}},
    },
    expect="error",
)
_p(
    "openai.responses.structured.explicit-text",
    SO,
    {
        "model": "@MODEL",
        "max_output_tokens": 64,
        "input": "hi",
        "text": {"format": {"type": "text"}},
    },
)
_p(
    "openai.responses.structured.verbosity-low",
    SO,
    {
        "model": "@MODEL",
        "max_output_tokens": 64,
        "input": "explain gravity",
        "text": {"verbosity": "low"},
    },
    note="SMG TextConfig has only `format` — modeled-surface gap",
)
_p(
    "openai.responses.structured.json-schema-classic",
    SO,
    {
        "model": "@MODEL_CLASSIC",
        "max_output_tokens": 128,
        "input": "Tokyo facts.",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "city_facts",
                "schema": _FLAT_SCHEMA,
                "strict": True,
            }
        },
    },
)

# =============================================================================
# function-tools
# =============================================================================
FT = "function-tools"
_WEATHER = {
    "type": "function",
    "name": "get_weather",
    "description": "Get weather for a city",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    "strict": True,
}
_WEATHER_LOOSE = dict(_WEATHER, strict=False)
_ZERO = {
    "type": "function",
    "name": "ping",
    "description": "ping",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}
_TIME = {
    "type": "function",
    "name": "get_time",
    "description": "current time",
    "parameters": {
        "type": "object",
        "properties": {"tz": {"type": "string"}},
        "required": ["tz"],
        "additionalProperties": False,
    },
}
_JOKE = {
    "type": "function",
    "name": "tell_joke",
    "description": "a joke",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}
_p(
    "openai.responses.fntool.single-forced",
    FT,
    {
        "model": "@MODEL",
        "max_output_tokens": 128,
        "tools": [_WEATHER],
        "input": "What is the weather in Paris? Use the tool.",
    },
)
_p(
    "openai.responses.fntool.strict-true",
    FT,
    {
        "model": "@MODEL",
        "max_output_tokens": 128,
        "tools": [_WEATHER],
        "input": "Weather in Paris?",
    },
)
_p(
    "openai.responses.fntool.strict-false",
    FT,
    {
        "model": "@MODEL",
        "max_output_tokens": 128,
        "tools": [_WEATHER_LOOSE],
        "input": "Weather in Paris?",
    },
)
_p(
    "openai.responses.fntool.zero-param",
    FT,
    {"model": "@MODEL", "max_output_tokens": 128, "tools": [_ZERO], "input": "ping the server"},
)
_p(
    "openai.responses.fntool.three-tools",
    FT,
    {
        "model": "@MODEL",
        "max_output_tokens": 128,
        "tools": [_WEATHER, _TIME, _JOKE],
        "input": "What time is it in UTC?",
    },
)
_p(
    "openai.responses.fntool.parallel-true",
    FT,
    {
        "model": "@MODEL",
        "max_output_tokens": 256,
        "tools": [_WEATHER],
        "parallel_tool_calls": True,
        "input": "Weather in Paris and Tokyo? Call the tool for each.",
    },
)
_p(
    "openai.responses.fntool.parallel-false",
    FT,
    {
        "model": "@MODEL",
        "max_output_tokens": 256,
        "tools": [_WEATHER],
        "parallel_tool_calls": False,
        "input": "Weather in Paris and Tokyo? Call the tool for each.",
    },
)
_p(
    "openai.responses.fntool.max-tool-calls-1",
    FT,
    {
        "model": "@MODEL",
        "max_output_tokens": 256,
        "tools": [_WEATHER],
        "max_tool_calls": 1,
        "input": "Weather in Paris and Tokyo? Call the tool for each.",
    },
)
# tool loop A (previous_response_id)
_p(
    "openai.responses.fntool.loopA-t1",
    FT,
    {
        "model": "@MODEL",
        "max_output_tokens": 256,
        "tools": [_WEATHER],
        "store": True,
        "input": "Weather in Paris? Use the tool.",
    },
)
_p(
    "openai.responses.fntool.loopA-t2",
    FT,
    {
        "model": "@MODEL",
        "max_output_tokens": 256,
        "tools": [_WEATHER],
        "store": True,
        "previous_response_id": "{{openai.responses.fntool.loopA-t1#body.id}}",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "{{openai.responses.fntool.loopA-t1#body.output[0].call_id}}",
                "output": '{"temp":20}',
            }
        ],
    },
    depends_on="openai.responses.fntool.loopA-t1",
)
# tool loop B (conversation)
_p("openai.responses.fntool.loopB-conv", FT, {}, endpoint="/v1/conversations")
_p(
    "openai.responses.fntool.loopB-t1",
    FT,
    {
        "model": "@MODEL",
        "max_output_tokens": 256,
        "tools": [_WEATHER],
        "conversation": "{{openai.responses.fntool.loopB-conv#body.id}}",
        "input": "Weather in Paris? Use the tool.",
    },
    depends_on="openai.responses.fntool.loopB-conv",
)
_p(
    "openai.responses.fntool.loopB-t2",
    FT,
    {
        "model": "@MODEL",
        "max_output_tokens": 256,
        "tools": [_WEATHER],
        "conversation": "{{openai.responses.fntool.loopB-conv#body.id}}",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "{{openai.responses.fntool.loopB-t1#body.output[0].call_id}}",
                "output": '{"temp":20}',
            }
        ],
    },
    depends_on="openai.responses.fntool.loopB-t1",
)
# tool loop C (stateless, replay reasoning + call + output)
_p(
    "openai.responses.fntool.loopC-t1",
    FT,
    {
        "model": "@MODEL",
        "max_output_tokens": 256,
        "tools": [_WEATHER],
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "input": "Weather in Paris? Use the tool.",
    },
)
_p(
    "openai.responses.fntool.loopC-t2",
    FT,
    {
        "model": "@MODEL",
        "max_output_tokens": 256,
        "tools": [_WEATHER],
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "input": [
            {"type": "message", "role": "user", "content": "Weather in Paris? Use the tool."},
            "{{openai.responses.fntool.loopC-t1#body.output[0]}}",
            "{{openai.responses.fntool.loopC-t1#body.output[1]}}",
            {
                "type": "function_call_output",
                "call_id": "{{openai.responses.fntool.loopC-t1#body.output[?function_call].call_id}}",
                "output": '{"temp":20}',
            },
        ],
    },
    depends_on="openai.responses.fntool.loopC-t1",
    note="stateless replay incl reasoning item (gpt-5 requires it)",
)
# tool_choice sweep
for slug, tc in [
    ("auto", "auto"),
    ("none", "none"),
    ("required", "required"),
    ("named", {"type": "function", "name": "get_weather"}),
    (
        "allowed-auto",
        {
            "type": "allowed_tools",
            "mode": "auto",
            "tools": [{"type": "function", "name": "get_weather"}],
        },
    ),
    (
        "allowed-required",
        {
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "function", "name": "get_weather"}],
        },
    ),
]:
    _p(
        f"openai.responses.fntool.tc-{slug}",
        FT,
        {
            "model": "@MODEL",
            "max_output_tokens": 128,
            "tools": [_WEATHER],
            "tool_choice": tc,
            "input": "Weather in Paris?",
        },
    )

# =============================================================================
# builtin-tools
# =============================================================================
BT = "builtin-tools"
_p(
    "openai.responses.builtin.web-search-basic",
    BT,
    {
        "model": "@MODEL",
        "max_output_tokens": 512,
        "tools": [{"type": "web_search"}],
        "input": "Latest news about SpaceX?",
    },
)
_p(
    "openai.responses.builtin.web-search-context-low",
    BT,
    {
        "model": "@MODEL",
        "max_output_tokens": 512,
        "tools": [{"type": "web_search", "search_context_size": "low"}],
        "input": "Weather in Tokyo today?",
    },
)
_p(
    "openai.responses.builtin.web-search-domains",
    BT,
    {
        "model": "@MODEL",
        "max_output_tokens": 512,
        "tools": [{"type": "web_search", "filters": {"allowed_domains": ["wikipedia.org"]}}],
        "input": "Who was Ada Lovelace?",
    },
)
_p(
    "openai.responses.builtin.web-search-location",
    BT,
    {
        "model": "@MODEL",
        "max_output_tokens": 512,
        "tools": [
            {
                "type": "web_search",
                "user_location": {"type": "approximate", "country": "US", "city": "San Francisco"},
            }
        ],
        "input": "coffee shops near me",
    },
)
_p(
    "openai.responses.builtin.web-search-preview",
    BT,
    {
        "model": "@MODEL",
        "max_output_tokens": 512,
        "tools": [{"type": "web_search_preview"}],
        "input": "Who won the 2022 World Cup?",
    },
)
_p(
    "openai.responses.builtin.web-search-sources",
    BT,
    {
        "model": "@MODEL",
        "max_output_tokens": 512,
        "tools": [{"type": "web_search"}],
        "include": ["web_search_call.action.sources"],
        "input": "Recent AI news?",
    },
)
# file_search (vector store id supplied via env placeholder at run time)
_FS_STORE = "@VECTOR_STORE_ID"
_p(
    "openai.responses.builtin.file-search-basic",
    BT,
    {
        "model": "@MODEL",
        "max_output_tokens": 256,
        "tools": [{"type": "file_search", "vector_store_ids": [_FS_STORE]}],
        "input": "What does the document say?",
    },
    note="requires OPENAI_VECTOR_STORE_ID env; skipped if absent",
)
_p(
    "openai.responses.builtin.file-search-max-results",
    BT,
    {
        "model": "@MODEL",
        "max_output_tokens": 256,
        "tools": [{"type": "file_search", "vector_store_ids": [_FS_STORE], "max_num_results": 1}],
        "input": "What does the document say?",
    },
    note="requires OPENAI_VECTOR_STORE_ID",
)
_p(
    "openai.responses.builtin.file-search-filter",
    BT,
    {
        "model": "@MODEL",
        "max_output_tokens": 256,
        "tools": [
            {
                "type": "file_search",
                "vector_store_ids": [_FS_STORE],
                "filters": {"key": "kind", "type": "eq", "value": "probe"},
            }
        ],
        "input": "What does the document say?",
    },
    note="requires OPENAI_VECTOR_STORE_ID",
)
_p(
    "openai.responses.builtin.file-search-ranking",
    BT,
    {
        "model": "@MODEL",
        "max_output_tokens": 256,
        "tools": [
            {
                "type": "file_search",
                "vector_store_ids": [_FS_STORE],
                "ranking_options": {"score_threshold": 0.5},
            }
        ],
        "input": "What does the document say?",
    },
    note="requires OPENAI_VECTOR_STORE_ID",
)
_p(
    "openai.responses.builtin.file-search-results",
    BT,
    {
        "model": "@MODEL",
        "max_output_tokens": 256,
        "tools": [{"type": "file_search", "vector_store_ids": [_FS_STORE]}],
        "include": ["file_search_call.results"],
        "input": "What does the document say?",
    },
    note="requires OPENAI_VECTOR_STORE_ID",
)
# code_interpreter
_p(
    "openai.responses.builtin.code-interpreter-basic",
    BT,
    {
        "model": "@MODEL",
        "max_output_tokens": 512,
        "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
        "input": "Compute 2**20 using python.",
    },
)
_p(
    "openai.responses.builtin.code-interpreter-outputs",
    BT,
    {
        "model": "@MODEL",
        "max_output_tokens": 512,
        "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
        "include": ["code_interpreter_call.outputs"],
        "input": "Compute the first 10 primes with python.",
    },
)
_p(
    "openai.responses.builtin.code-interpreter-csv",
    BT,
    {
        "model": "@MODEL",
        "max_output_tokens": 512,
        "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
        "input": "Write a csv file with two rows and let me download it.",
    },
)
# image_generation (cost-flagged)
_p(
    "openai.responses.builtin.image-gen-minimal",
    BT,
    {
        "model": "@MODEL",
        "max_output_tokens": 512,
        "tools": [{"type": "image_generation", "quality": "low", "size": "1024x1024"}],
        "input": "Draw a red circle.",
    },
    note="COST-FLAGGED; gated by OPENAI_PROBE_IMAGES",
)
_p(
    "openai.responses.builtin.image-gen-stream",
    BT,
    {
        "model": "@MODEL",
        "stream": True,
        "max_output_tokens": 512,
        "tools": [
            {"type": "image_generation", "quality": "low", "size": "1024x1024", "partial_images": 1}
        ],
        "input": "Draw a blue square.",
    },
    stream=True,
    note="COST-FLAGGED; gated by OPENAI_PROBE_IMAGES",
)
_p(
    "openai.responses.builtin.image-gen-reference",
    BT,
    {
        "model": "@MODEL",
        "max_output_tokens": 512,
        "tools": [{"type": "image_generation", "quality": "low", "size": "1024x1024"}],
        "input": [
            {
                "type": "image_generation_call",
                "id": "{{openai.responses.builtin.image-gen-minimal#body.output[0].id}}",
            },
            _um("now make it green"),
        ],
    },
    depends_on="openai.responses.builtin.image-gen-minimal",
    note="COST-FLAGGED; gated by OPENAI_PROBE_IMAGES",
)
# computer_use_preview
_p(
    "openai.responses.builtin.computer-use",
    BT,
    {
        "model": "@MODEL_CUA",
        "max_output_tokens": 256,
        "truncation": "auto",
        "tools": [
            {
                "type": "computer_use_preview",
                "display_width": 1024,
                "display_height": 768,
                "environment": "browser",
            }
        ],
        "input": "Open a browser and go to example.com",
    },
    expect="error",
    note="records computer_call OR graceful 4xx; never fails run",
)
# custom tools
_p(
    "openai.responses.builtin.custom-tool",
    BT,
    {
        "model": "@MODEL",
        "max_output_tokens": 128,
        "tools": [{"type": "custom", "name": "run_cmd", "description": "run a shell command"}],
        "input": "Use run_cmd to list files.",
    },
)
_p(
    "openai.responses.builtin.custom-tool-grammar-regex",
    BT,
    {
        "model": "@MODEL",
        "max_output_tokens": 128,
        "tools": [
            {
                "type": "custom",
                "name": "digits",
                "description": "emit digits",
                "format": {"type": "grammar", "syntax": "regex", "definition": "[0-9]+"},
            }
        ],
        "input": "Emit some digits with the tool.",
    },
)
_p(
    "openai.responses.builtin.custom-tool-grammar-lark",
    BT,
    {
        "model": "@MODEL",
        "max_output_tokens": 128,
        "tools": [
            {
                "type": "custom",
                "name": "expr",
                "description": "emit an expr",
                "format": {
                    "type": "grammar",
                    "syntax": "lark",
                    "definition": "start: NUMBER\nNUMBER: /[0-9]+/",
                },
            }
        ],
        "input": "Emit an expression with the tool.",
    },
)
# local_shell rejection
_p(
    "openai.responses.builtin.local-shell-reject",
    BT,
    {
        "model": "@MODEL",
        "max_output_tokens": 128,
        "tools": [{"type": "local_shell"}],
        "input": "list files",
    },
    expect="error",
    note="local_shell requires codex model; record rejection",
)
# mcp unreachable
_p(
    "openai.responses.builtin.mcp-unreachable",
    BT,
    {
        "model": "@MODEL",
        "max_output_tokens": 128,
        "tools": [
            {
                "type": "mcp",
                "server_label": "dead",
                "server_url": "https://127.0.0.1:9/mcp",
                "require_approval": "never",
            }
        ],
        "input": "list tools",
    },
    expect="error",
    note="record mcp_list_tools failure shape, no external dep",
)
# hosted-tool forcing via tool_choice
_p(
    "openai.responses.builtin.tc-web-search",
    BT,
    {
        "model": "@MODEL",
        "max_output_tokens": 512,
        "tools": [{"type": "web_search"}],
        "tool_choice": {"type": "web_search"},
        "input": "SpaceX news?",
    },
)

# =============================================================================
# include-options
# =============================================================================
IN = "include-options"
_p(
    "openai.responses.include.encrypted-content",
    IN,
    {
        "model": "@MODEL",
        "max_output_tokens": 128,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "input": "Think then say hi.",
    },
)
_p(
    "openai.responses.include.encrypted-roundtrip",
    IN,
    {
        "model": "@MODEL",
        "max_output_tokens": 128,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "input": [
            {"type": "message", "role": "user", "content": "Think then say hi."},
            "{{openai.responses.include.encrypted-content#body.output[?reasoning]}}",
            _um("continue"),
        ],
    },
    depends_on="openai.responses.include.encrypted-content",
    note="resubmit reasoning item verbatim in stateless follow-up",
)
_p(
    "openai.responses.include.logprobs-with-top",
    IN,
    {
        "model": "@MODEL_CLASSIC",
        "max_output_tokens": 64,
        "input": "hi",
        "top_logprobs": 5,
        "include": ["message.output_text.logprobs"],
    },
)
_p(
    "openai.responses.include.top-without-include",
    IN,
    {"model": "@MODEL_CLASSIC", "max_output_tokens": 64, "input": "hi", "top_logprobs": 5},
)
_p(
    "openai.responses.include.include-without-top",
    IN,
    {
        "model": "@MODEL_CLASSIC",
        "max_output_tokens": 64,
        "input": "hi",
        "include": ["message.output_text.logprobs"],
    },
)
_p(
    "openai.responses.include.image-url-include",
    IN,
    {
        "model": "@MODEL",
        "max_output_tokens": 64,
        "include": ["message.input_image.image_url"],
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is this"},
                    {"type": "input_image", "image_url": IMG_URL, "detail": "low"},
                ],
            }
        ],
    },
)
_p(
    "openai.responses.include.inapplicable",
    IN,
    {
        "model": "@MODEL",
        "max_output_tokens": 64,
        "input": "hi",
        "include": ["file_search_call.results"],
    },
    note="record silent acceptance vs error (no tools present)",
)

# =============================================================================
# reasoning-params
# =============================================================================
RP = "reasoning-params"
for eff in ["minimal", "low", "medium", "high"]:
    _p(
        f"openai.responses.reasoning.effort-{eff}",
        RP,
        {
            "model": "@MODEL",
            "max_output_tokens": 512,
            "reasoning": {"effort": eff},
            "input": "What is 12*13? Think.",
        },
    )
_p(
    "openai.responses.reasoning.summary-auto",
    RP,
    {
        "model": "@MODEL",
        "max_output_tokens": 512,
        "reasoning": {"effort": "low", "summary": "auto"},
        "input": "Why is the sky blue?",
    },
)
_p(
    "openai.responses.reasoning.summary-detailed",
    RP,
    {
        "model": "@MODEL",
        "max_output_tokens": 512,
        "reasoning": {"effort": "low", "summary": "detailed"},
        "input": "Why is the sky blue?",
    },
)
_p(
    "openai.responses.reasoning.minimal-plus-summary",
    RP,
    {
        "model": "@MODEL",
        "max_output_tokens": 512,
        "reasoning": {"effort": "minimal", "summary": "auto"},
        "input": "Why is the sky blue?",
    },
)

# =============================================================================
# sampling-and-misc-params
# =============================================================================
PM = "sampling-and-misc-params"
_p(
    "openai.responses.params.temp-0",
    PM,
    {"model": "@MODEL_CLASSIC", "input": "hi", "max_output_tokens": 64, "temperature": 0},
)
_p(
    "openai.responses.params.temp-2",
    PM,
    {"model": "@MODEL_CLASSIC", "input": "hi", "max_output_tokens": 64, "temperature": 2.0},
)
_p(
    "openai.responses.params.top-p",
    PM,
    {"model": "@MODEL_CLASSIC", "input": "hi", "max_output_tokens": 64, "top_p": 0.1},
)
_p(
    "openai.responses.params.temp-and-top-p",
    PM,
    {
        "model": "@MODEL_CLASSIC",
        "input": "hi",
        "max_output_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.9,
    },
)
_p(
    "openai.responses.params.temp-on-reasoning",
    PM,
    {"model": "@MODEL", "input": "hi", "max_output_tokens": 64, "temperature": 0.5},
    expect="error",
    note="compat-critical: SMG must reproduce rejection",
)
_p(
    "openai.responses.params.max-tokens-16",
    PM,
    {"model": "@MODEL", "input": "write an essay", "max_output_tokens": 16},
)
_p(
    "openai.responses.params.max-tokens-128000",
    PM,
    {"model": "@MODEL", "input": "hi", "max_output_tokens": 128000},
)
_p(
    "openai.responses.params.truncation-auto",
    PM,
    {"model": "@MODEL", "input": "hi", "max_output_tokens": 64, "truncation": "auto"},
)
_p(
    "openai.responses.params.truncation-disabled",
    PM,
    {"model": "@MODEL", "input": "hi", "max_output_tokens": 64, "truncation": "disabled"},
)
_p(
    "openai.responses.params.metadata",
    PM,
    {
        "model": "@MODEL",
        "input": "hi",
        "max_output_tokens": 64,
        "store": True,
        "metadata": {f"k{i}": ("v" * 512) for i in range(16)},
    },
)
for tier in ["auto", "default", "flex"]:
    _p(
        f"openai.responses.params.service-tier-{tier}",
        PM,
        {"model": "@MODEL", "input": "hi", "max_output_tokens": 64, "service_tier": tier},
        note="flex may 4xx per account; record",
    )
_p(
    "openai.responses.params.user-legacy",
    PM,
    {"model": "@MODEL", "input": "hi", "max_output_tokens": 64, "user": "probe-user"},
)
_p(
    "openai.responses.params.safety-identifier",
    PM,
    {
        "model": "@MODEL",
        "input": "hi",
        "max_output_tokens": 64,
        "safety_identifier": "probe-safety",
    },
)
_p(
    "openai.responses.params.cache-key-t1",
    PM,
    {
        "model": "@MODEL",
        "max_output_tokens": 64,
        "prompt_cache_key": "probe-cache-1",
        "input": "This is a long shared prefix. " * 40 + "First question?",
    },
)
_p(
    "openai.responses.params.cache-key-t2",
    PM,
    {
        "model": "@MODEL",
        "max_output_tokens": 64,
        "prompt_cache_key": "probe-cache-1",
        "input": "This is a long shared prefix. " * 40 + "Second question?",
    },
    depends_on="openai.responses.params.cache-key-t1",
    note="record usage.input_tokens_details.cached_tokens delta",
)
_p(
    "openai.responses.params.stop",
    PM,
    {"model": "@MODEL_CLASSIC", "input": "count to ten", "max_output_tokens": 64, "stop": ["END"]},
    note="SMG models StringOrArray stop; OpenAI may reject",
)
_p(
    "openai.responses.params.frequency-penalty",
    PM,
    {"model": "@MODEL_CLASSIC", "input": "hi", "max_output_tokens": 64, "frequency_penalty": 0.5},
    note="SMG has field; OpenAI Responses does not",
)
_p(
    "openai.responses.params.presence-penalty",
    PM,
    {"model": "@MODEL_CLASSIC", "input": "hi", "max_output_tokens": 64, "presence_penalty": 0.5},
    note="SMG has field; OpenAI Responses does not",
)
_p(
    "openai.responses.params.context-management",
    PM,
    {
        "model": "@MODEL",
        "input": "hi",
        "max_output_tokens": 64,
        "context_management": {"strategy": "compaction"},
    },
    note="exploratory; SMG models ContextManagementEntry + compaction item",
)

# =============================================================================
# pairwise-interactions
# =============================================================================
PW = "pairwise-interactions"
_p(
    "openai.responses.pair.stream-fntool",
    PW,
    {
        "model": "@MODEL",
        "stream": True,
        "max_output_tokens": 128,
        "tools": [_WEATHER],
        "input": "Weather in Paris? Use the tool.",
    },
    stream=True,
)
_p(
    "openai.responses.pair.stream-parallel",
    PW,
    {
        "model": "@MODEL",
        "stream": True,
        "max_output_tokens": 256,
        "tools": [_WEATHER],
        "parallel_tool_calls": True,
        "input": "Weather in Paris and Tokyo? Call for each.",
    },
    stream=True,
)
_p(
    "openai.responses.pair.stream-loop-t1",
    PW,
    {
        "model": "@MODEL",
        "max_output_tokens": 256,
        "tools": [_WEATHER],
        "store": True,
        "input": "Weather in Paris? Use the tool.",
    },
)
_p(
    "openai.responses.pair.stream-loop-t2",
    PW,
    {
        "model": "@MODEL",
        "stream": True,
        "max_output_tokens": 256,
        "tools": [_WEATHER],
        "store": True,
        "previous_response_id": "{{openai.responses.pair.stream-loop-t1#body.id}}",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "{{openai.responses.pair.stream-loop-t1#body.output[0].call_id}}",
                "output": '{"temp":20}',
            }
        ],
    },
    stream=True,
    depends_on="openai.responses.pair.stream-loop-t1",
)
_p(
    "openai.responses.pair.stream-json-schema",
    PW,
    {
        "model": "@MODEL",
        "stream": True,
        "max_output_tokens": 128,
        "input": "Tokyo facts.",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "city_facts",
                "schema": _FLAT_SCHEMA,
                "strict": True,
            }
        },
    },
    stream=True,
)
_p(
    "openai.responses.pair.stream-json-object",
    PW,
    {
        "model": "@MODEL",
        "stream": True,
        "max_output_tokens": 128,
        "input": "Return JSON with a city field.",
        "text": {"format": {"type": "json_object"}},
    },
    stream=True,
)
_p(
    "openai.responses.pair.stream-web-search",
    PW,
    {
        "model": "@MODEL",
        "stream": True,
        "max_output_tokens": 512,
        "tools": [{"type": "web_search"}],
        "input": "SpaceX news?",
    },
    stream=True,
)
_p(
    "openai.responses.pair.stream-file-search",
    PW,
    {
        "model": "@MODEL",
        "stream": True,
        "max_output_tokens": 256,
        "tools": [{"type": "file_search", "vector_store_ids": [_FS_STORE]}],
        "input": "What does the document say?",
    },
    stream=True,
    note="requires OPENAI_VECTOR_STORE_ID",
)
_p(
    "openai.responses.pair.stream-code-interpreter",
    PW,
    {
        "model": "@MODEL",
        "stream": True,
        "max_output_tokens": 512,
        "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
        "input": "Compute 3**7 with python.",
    },
    stream=True,
)
_p(
    "openai.responses.pair.stream-image-gen",
    PW,
    {
        "model": "@MODEL",
        "stream": True,
        "max_output_tokens": 512,
        "tools": [
            {"type": "image_generation", "quality": "low", "size": "1024x1024", "partial_images": 1}
        ],
        "input": "Draw a triangle.",
    },
    stream=True,
    note="COST-FLAGGED; gated by OPENAI_PROBE_IMAGES",
)
_p(
    "openai.responses.pair.stream-custom-tool",
    PW,
    {
        "model": "@MODEL",
        "stream": True,
        "max_output_tokens": 128,
        "tools": [{"type": "custom", "name": "run_cmd", "description": "run a command"}],
        "input": "Use run_cmd to list files.",
    },
    stream=True,
)
_p(
    "openai.responses.pair.stream-reasoning-fntool",
    PW,
    {
        "model": "@MODEL",
        "stream": True,
        "max_output_tokens": 512,
        "tools": [_WEATHER],
        "reasoning": {"effort": "low", "summary": "auto"},
        "input": "Think, then get weather in Paris with the tool.",
    },
    stream=True,
)
_p("openai.responses.pair.conv-fntool-conv", PW, {}, endpoint="/v1/conversations")
_p(
    "openai.responses.pair.conv-fntool-t1",
    PW,
    {
        "model": "@MODEL",
        "max_output_tokens": 256,
        "tools": [_WEATHER],
        "conversation": "{{openai.responses.pair.conv-fntool-conv#body.id}}",
        "input": "Weather in Paris? Use the tool.",
    },
    depends_on="openai.responses.pair.conv-fntool-conv",
)
_p(
    "openai.responses.pair.conv-fntool-t2",
    PW,
    {
        "model": "@MODEL",
        "max_output_tokens": 256,
        "tools": [_WEATHER],
        "conversation": "{{openai.responses.pair.conv-fntool-conv#body.id}}",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "{{openai.responses.pair.conv-fntool-t1#body.output[0].call_id}}",
                "output": '{"temp":20}',
            }
        ],
    },
    depends_on="openai.responses.pair.conv-fntool-t1",
)
_p(
    "openai.responses.pair.conv-fntool-items",
    PW,
    None,
    method="GET",
    endpoint="/v1/conversations/{{openai.responses.pair.conv-fntool-conv#body.id}}/items?order=asc",
    depends_on="openai.responses.pair.conv-fntool-t2",
)
_p(
    "openai.responses.pair.prev-json-schema-t1",
    PW,
    {"model": "@MODEL", "input": "hi", "store": True, "max_output_tokens": 64},
)
_p(
    "openai.responses.pair.prev-json-schema-t2",
    PW,
    {
        "model": "@MODEL",
        "max_output_tokens": 128,
        "input": "Tokyo facts.",
        "previous_response_id": "{{openai.responses.pair.prev-json-schema-t1#body.id}}",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "city_facts",
                "schema": _FLAT_SCHEMA,
                "strict": True,
            }
        },
    },
    depends_on="openai.responses.pair.prev-json-schema-t1",
)
_p(
    "openai.responses.pair.instructions-inherit-t1",
    PW,
    {
        "model": "@MODEL",
        "instructions": "Always answer in French.",
        "input": "hello",
        "store": True,
        "max_output_tokens": 64,
    },
)
_p(
    "openai.responses.pair.instructions-inherit-new",
    PW,
    {
        "model": "@MODEL",
        "instructions": "Always answer in German.",
        "input": "hello again",
        "max_output_tokens": 64,
        "previous_response_id": "{{openai.responses.pair.instructions-inherit-t1#body.id}}",
    },
    depends_on="openai.responses.pair.instructions-inherit-t1",
)
_p(
    "openai.responses.pair.instructions-inherit-none",
    PW,
    {
        "model": "@MODEL",
        "input": "hello again",
        "max_output_tokens": 64,
        "previous_response_id": "{{openai.responses.pair.instructions-inherit-t1#body.id}}",
    },
    depends_on="openai.responses.pair.instructions-inherit-t1",
)
_p(
    "openai.responses.pair.background-fntool",
    PW,
    {
        "model": "@MODEL",
        "max_output_tokens": 256,
        "tools": [_WEATHER],
        "background": True,
        "store": True,
        "input": "Weather in Paris? Use the tool.",
    },
    poll={
        "path": "/v1/responses/{{self#body.id}}",
        "interval_s": 2,
        "cap_s": 60,
        "until_status": ["completed", "failed", "incomplete", "cancelled"],
    },
)
_p(
    "openai.responses.pair.background-json-schema",
    PW,
    {
        "model": "@MODEL",
        "max_output_tokens": 128,
        "background": True,
        "store": True,
        "input": "Tokyo facts.",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "city_facts",
                "schema": _FLAT_SCHEMA,
                "strict": True,
            }
        },
    },
    poll={
        "path": "/v1/responses/{{self#body.id}}",
        "interval_s": 2,
        "cap_s": 60,
        "until_status": ["completed", "failed", "incomplete", "cancelled"],
    },
)
_p(
    "openai.responses.pair.nostore-fntool-t1",
    PW,
    {
        "model": "@MODEL",
        "max_output_tokens": 256,
        "tools": [_WEATHER],
        "store": False,
        "input": "Weather in Paris? Use the tool.",
    },
)
_p(
    "openai.responses.pair.nostore-fntool-t2",
    PW,
    {
        "model": "@MODEL",
        "max_output_tokens": 256,
        "tools": [_WEATHER],
        "store": False,
        "input": [
            {"type": "message", "role": "user", "content": "Weather in Paris? Use the tool."},
            "{{openai.responses.pair.nostore-fntool-t1#body.output[?function_call]}}",
            {
                "type": "function_call_output",
                "call_id": "{{openai.responses.pair.nostore-fntool-t1#body.output[?function_call].call_id}}",
                "output": '{"temp":20}',
            },
        ],
    },
    depends_on="openai.responses.pair.nostore-fntool-t1",
)
_p(
    "openai.responses.pair.codex-stateless-t1",
    PW,
    {
        "model": "@MODEL",
        "max_output_tokens": 256,
        "tools": [_WEATHER],
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "input": "Weather in Paris? Use the tool.",
    },
)
_p(
    "openai.responses.pair.codex-stateless-t2",
    PW,
    {
        "model": "@MODEL",
        "max_output_tokens": 256,
        "tools": [_WEATHER],
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "input": [
            {"type": "message", "role": "user", "content": "Weather in Paris? Use the tool."},
            "{{openai.responses.pair.codex-stateless-t1#body.output[?reasoning]}}",
            "{{openai.responses.pair.codex-stateless-t1#body.output[?function_call]}}",
            {
                "type": "function_call_output",
                "call_id": "{{openai.responses.pair.codex-stateless-t1#body.output[?function_call].call_id}}",
                "output": '{"temp":20}',
            },
        ],
    },
    depends_on="openai.responses.pair.codex-stateless-t1",
    note="codex-style stateless agent pattern — highest-value single recording",
)

# =============================================================================
# error-probes  (all zero token cost)
# =============================================================================
E = "error-probes"


def _e(slug, body, **kw):
    _p(f"openai.responses.err.{slug}", E, body, expect="error", **kw)


# model
_e("model-nonexistent", {"model": "gpt-nope", "input": "hi", "max_output_tokens": 64})
_e("model-empty", {"model": "", "input": "hi", "max_output_tokens": 64})
_e("model-missing", {"input": "hi", "max_output_tokens": 64})
_e("model-wrong-type", {"model": 42, "input": "hi", "max_output_tokens": 64})
# input
_e("input-missing", {"model": "@MODEL", "max_output_tokens": 64})
_e("input-wrong-type-int", {"model": "@MODEL", "input": 42, "max_output_tokens": 64})
_e("input-wrong-type-obj", {"model": "@MODEL", "input": {}, "max_output_tokens": 64})
_e(
    "input-bad-item-type",
    {"model": "@MODEL", "max_output_tokens": 64, "input": [{"type": "bogus"}]},
)
_e(
    "input-bad-content-part",
    {
        "model": "@MODEL",
        "max_output_tokens": 64,
        "input": [{"type": "message", "role": "user", "content": [{"type": "bogus", "text": "x"}]}],
    },
)
_e(
    "input-bad-role",
    {
        "model": "@MODEL",
        "max_output_tokens": 64,
        "input": [{"type": "message", "role": "robot", "content": "hi"}],
    },
)
_e(
    "input-content-wrong-type",
    {
        "model": "@MODEL",
        "max_output_tokens": 64,
        "input": [{"type": "message", "role": "user", "content": 123}],
    },
)
# sampling
_e(
    "temp-3",
    {"model": "@MODEL_CLASSIC", "input": "hi", "max_output_tokens": 64, "temperature": 3.0},
)
_e(
    "temp-string",
    {"model": "@MODEL_CLASSIC", "input": "hi", "max_output_tokens": 64, "temperature": "hot"},
)
_e("top-p-1.5", {"model": "@MODEL_CLASSIC", "input": "hi", "max_output_tokens": 64, "top_p": 1.5})
_e(
    "top-logprobs-25",
    {"model": "@MODEL_CLASSIC", "input": "hi", "max_output_tokens": 64, "top_logprobs": 25},
)
_e("max-tokens-1", {"model": "@MODEL", "input": "hi", "max_output_tokens": 1})
_e("max-tokens-neg", {"model": "@MODEL", "input": "hi", "max_output_tokens": -1})
_e("max-tokens-string", {"model": "@MODEL", "input": "hi", "max_output_tokens": "many"})
# tools
_e(
    "tools-bogus-type",
    {"model": "@MODEL", "input": "hi", "max_output_tokens": 64, "tools": [{"type": "bogus"}]},
)
_e(
    "tools-missing-name",
    {
        "model": "@MODEL",
        "input": "hi",
        "max_output_tokens": 64,
        "tools": [{"type": "function", "parameters": {"type": "object"}}],
    },
)
_e(
    "tools-bad-params",
    {
        "model": "@MODEL",
        "input": "hi",
        "max_output_tokens": 64,
        "tools": [{"type": "function", "name": "f", "parameters": "not-an-object"}],
    },
)
_e(
    "strict-missing-addlprops",
    {
        "model": "@MODEL",
        "input": "hi",
        "max_output_tokens": 64,
        "tools": [
            {
                "type": "function",
                "name": "f",
                "strict": True,
                "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
            }
        ],
    },
)
_e(
    "tc-always",
    {
        "model": "@MODEL",
        "input": "hi",
        "max_output_tokens": 64,
        "tools": [_WEATHER],
        "tool_choice": "always",
    },
)
_e(
    "tc-named-missing",
    {
        "model": "@MODEL",
        "input": "hi",
        "max_output_tokens": 64,
        "tools": [_WEATHER],
        "tool_choice": {"type": "function", "name": "nope"},
    },
)
_e(
    "tc-required-no-tools",
    {"model": "@MODEL", "input": "hi", "max_output_tokens": 64, "tool_choice": "required"},
)
# enums/fields
_e(
    "include-bogus",
    {"model": "@MODEL", "input": "hi", "max_output_tokens": 64, "include": ["bogus.field"]},
)
_e(
    "reasoning-effort-max",
    {"model": "@MODEL", "input": "hi", "max_output_tokens": 64, "reasoning": {"effort": "max"}},
)
_e(
    "reasoning-on-classic",
    {
        "model": "@MODEL_CLASSIC",
        "input": "hi",
        "max_output_tokens": 64,
        "reasoning": {"effort": "low"},
    },
)
_e(
    "truncation-sometimes",
    {"model": "@MODEL", "input": "hi", "max_output_tokens": 64, "truncation": "sometimes"},
)
_e(
    "service-tier-vip",
    {"model": "@MODEL", "input": "hi", "max_output_tokens": 64, "service_tier": "vip"},
)
_e(
    "metadata-17-keys",
    {
        "model": "@MODEL",
        "input": "hi",
        "max_output_tokens": 64,
        "metadata": {f"k{i}": "v" for i in range(17)},
    },
)
_e(
    "metadata-oversized",
    {"model": "@MODEL", "input": "hi", "max_output_tokens": 64, "metadata": {"k": "v" * 600}},
)
_e(
    "metadata-wrong-type",
    {"model": "@MODEL", "input": "hi", "max_output_tokens": 64, "metadata": "hi"},
)
# state
_p(
    "openai.responses.err.prev-and-conversation",
    E,
    {
        "model": "@MODEL",
        "input": "hi",
        "max_output_tokens": 64,
        "previous_response_id": "resp_abc",
        "conversation": "conv_abc",
    },
    expect="error",
)
_e(
    "conversation-bad-id",
    {"model": "@MODEL", "input": "hi", "max_output_tokens": 64, "conversation": "not-a-conv"},
)
# size/shape
_e(
    "context-length-exceeded",
    {"model": "@MODEL", "max_output_tokens": 64, "input": "word " * 60000},
)
_e(
    "unknown-top-level-field",
    {"model": "@MODEL", "input": "hi", "max_output_tokens": 64, "frobnicate": True},
)
_e("stream-yes", {"model": "@MODEL", "input": "hi", "max_output_tokens": 64, "stream": "yes"})
_e("empty-body", {})
_p(
    "openai.responses.err.malformed-json",
    E,
    "{",
    expect="error",
    note="raw malformed JSON body, not a dict",
)
# auth/transport
_p(
    "openai.responses.err.bad-api-key",
    E,
    {"model": "@MODEL", "input": "hi", "max_output_tokens": 64},
    expect="error",
    headers={"Authorization": "Bearer sk-invalid-probe-key"},
    note="override auth to force 401",
)
_p(
    "openai.responses.err.missing-auth",
    E,
    {"model": "@MODEL", "input": "hi", "max_output_tokens": 64},
    expect="error",
    headers={"Authorization": None},
    note="drop Authorization header",
)
_p(
    "openai.responses.err.route-mismatch",
    E,
    None,
    method="GET",
    endpoint="/v1/responses",
    expect="error",
    note="GET list route not supported",
)
