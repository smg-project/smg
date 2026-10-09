"""OpenAI Chat Completions API ground-truth probe matrix (data only).

Same conventions as ``openai_responses.py``: every probe is a plain dict, the
runner injects the model sentinels (``@MODEL`` = the reasoning-capable default,
``@MODEL_CLASSIC`` = the classic model for temperature / top_p / logprobs / n)
and nothing here depends on a prior recording (tool-loop probes carry their
own scripted ``tool_calls`` so they resolve without the model having called a
tool).

Fields: id, category, endpoint, method, body, stream, depends_on, expect
(ok|error), headers (extra), poll, note.
"""

# --- fixtures ---------------------------------------------------------------
# a valid 1x1 RGBA PNG (both vendors decode it)
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA"
    "60e6kgAAAABJRU5ErkJggg=="
)
IMG_DATA_URI = "data:image/png;base64," + PNG_B64
IMG_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
    "PNG_transparency_demonstration_1.png/240px-PNG_transparency_demonstration_1.png"
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current time in a timezone",
        "parameters": {
            "type": "object",
            "properties": {"tz": {"type": "string"}},
            "required": ["tz"],
        },
    },
}
STRICT_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather_strict",
        "description": "Get the current weather in a city",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}
PERSON_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "additionalProperties": False,
}

PROBES: list[dict] = []


def _p(
    pid,
    category,
    body=None,
    *,
    endpoint="/v1/chat/completions",
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


def _u(text):
    return {"role": "user", "content": text}


def _base(model="@MODEL", **extra):
    body = {"model": model, "messages": [_u("Say hi in one word.")], "max_completion_tokens": 64}
    body.update(extra)
    return body


def _classic(**extra):
    return _base(model="@MODEL_CLASSIC", **extra)


WEATHER_Q = _u("What is the weather in Paris right now? Use the tool.")

# =============================================================================
# core
# =============================================================================
C = "core"
_p("openai.chat.core.plain", C, _base())
_p("openai.chat.core.plain-classic", C, _classic())
_p(
    "openai.chat.core.system-message",
    C,
    _base(messages=[{"role": "system", "content": "You are terse."}, _u("hi")]),
)
_p(
    "openai.chat.core.developer-message",
    C,
    _base(messages=[{"role": "developer", "content": "You are terse."}, _u("hi")]),
)
_p(
    "openai.chat.core.multi-turn",
    C,
    _base(
        messages=[
            _u("hi"),
            {"role": "assistant", "content": "hello there"},
            _u("say it again"),
        ]
    ),
)
_p(
    "openai.chat.core.content-parts",
    C,
    _base(messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}]),
)
_p(
    "openai.chat.core.system-content-parts",
    C,
    _base(
        messages=[
            {"role": "system", "content": [{"type": "text", "text": "You are terse."}]},
            _u("hi"),
        ]
    ),
)
_p(
    "openai.chat.core.name-field",
    C,
    _base(messages=[{"role": "user", "content": "hi", "name": "alice"}]),
)
_p(
    "openai.chat.core.assistant-prefill-last",
    C,
    _classic(
        messages=[
            _u("Complete: the capital of France is"),
            {"role": "assistant", "content": "The capital"},
        ]
    ),
    note="trailing assistant turn: the classic API continues it",
)
_p(
    "openai.chat.core.max-tokens-legacy-classic",
    C,
    _classic(max_tokens=64),
    note="max_tokens accepted on classic models",
)
_p(
    "openai.chat.core.max-completion-tokens-16",
    C,
    _classic(
        messages=[_u("Write a long story about a lighthouse keeper.")], max_completion_tokens=16
    ),
    note="finish_reason length expected",
)
_p("openai.chat.core.store-metadata", C, _base(store=True, metadata={"probe": "chat"}))
_p("openai.chat.core.user-field", C, _base(user="probe-user"))
_p("openai.chat.core.service-tier-auto", C, _base(service_tier="auto"))
_p("openai.chat.core.prompt-cache-key", C, _base(prompt_cache_key="probe-chat-cache"))
_p("openai.chat.core.safety-identifier", C, _base(safety_identifier="probe-safety"))
_p("openai.chat.core.reasoning-effort-low", C, _base(reasoning_effort="low"))
_p("openai.chat.core.reasoning-effort-minimal", C, _base(reasoning_effort="minimal"))
_p("openai.chat.core.verbosity-low", C, _base(verbosity="low"))

# =============================================================================
# streaming
# =============================================================================
S = "streaming"
_p("openai.chat.stream.basic", S, _classic(stream=True), stream=True)
_p("openai.chat.stream.basic-reasoning-model", S, _base(stream=True), stream=True)
_p(
    "openai.chat.stream.usage",
    S,
    _classic(stream=True, stream_options={"include_usage": True}),
    stream=True,
)
_p(
    "openai.chat.stream.usage-reasoning-model",
    S,
    _base(stream=True, stream_options={"include_usage": True}),
    stream=True,
)
_p(
    "openai.chat.stream.max-tokens-cut",
    S,
    _classic(
        messages=[_u("Write a long story about a lighthouse keeper.")],
        max_completion_tokens=16,
        stream=True,
        stream_options={"include_usage": True},
    ),
    stream=True,
)
_p(
    "openai.chat.stream.stop-sequence",
    S,
    _classic(messages=[_u("Count from 1 to 20 separated by commas.")], stop=["5"], stream=True),
    stream=True,
)
_p(
    "openai.chat.stream.n-2",
    S,
    _classic(n=2, stream=True, stream_options={"include_usage": True}),
    stream=True,
)
_p(
    "openai.chat.stream.tools-auto",
    S,
    _classic(messages=[WEATHER_Q], tools=[WEATHER_TOOL], tool_choice="auto", stream=True),
    stream=True,
)
_p(
    "openai.chat.stream.tools-required",
    S,
    _classic(
        messages=[WEATHER_Q],
        tools=[WEATHER_TOOL],
        tool_choice="required",
        stream=True,
        stream_options={"include_usage": True},
    ),
    stream=True,
)
_p(
    "openai.chat.stream.tools-parallel",
    S,
    _classic(
        messages=[_u("What is the weather in Paris and the time in Europe/Paris? Use both tools.")],
        tools=[WEATHER_TOOL, TIME_TOOL],
        tool_choice="required",
        parallel_tool_calls=True,
        stream=True,
    ),
    stream=True,
)
_p(
    "openai.chat.stream.logprobs",
    S,
    _classic(logprobs=True, top_logprobs=2, stream=True),
    stream=True,
)
_p(
    "openai.chat.stream.json-schema",
    S,
    _classic(
        messages=[_u("Give me a person named Ada aged 36 as JSON.")],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "person", "strict": True, "schema": PERSON_SCHEMA},
        },
        stream=True,
    ),
    stream=True,
)
_p(
    "openai.chat.stream.invalid-param",
    S,
    _base(stream=True, temperature=0.5),
    stream=True,
    expect="error",
    note="temperature on the reasoning model; records 4xx-vs-SSE timing",
)
_p(
    "openai.chat.stream.unknown-model",
    S,
    {"model": "gpt-nope", "messages": [_u("hi")], "stream": True},
    stream=True,
    expect="error",
)

# =============================================================================
# sampling-params (classic model)
# =============================================================================
P = "sampling-params"
_p("openai.chat.params.temp-0", P, _classic(temperature=0))
_p("openai.chat.params.temp-2", P, _classic(temperature=2.0))
_p("openai.chat.params.top-p", P, _classic(top_p=0.5))
_p("openai.chat.params.temp-and-top-p", P, _classic(temperature=0.7, top_p=0.9))
_p("openai.chat.params.seed", P, _classic(seed=7, temperature=0))
_p("openai.chat.params.n-2", P, _classic(n=2))
_p(
    "openai.chat.params.stop-string",
    P,
    _classic(messages=[_u("Count from 1 to 20 separated by commas.")], stop="5"),
)
_p(
    "openai.chat.params.stop-array-4",
    P,
    _classic(messages=[_u("Count from 1 to 20 separated by commas.")], stop=["5", "6", "7", "8"]),
)
_p("openai.chat.params.frequency-penalty", P, _classic(frequency_penalty=1.5))
_p("openai.chat.params.presence-penalty", P, _classic(presence_penalty=1.5))
_p("openai.chat.params.logit-bias", P, _classic(logit_bias={"1820": -100}))
_p("openai.chat.params.logprobs-top-5", P, _classic(logprobs=True, top_logprobs=5))
_p("openai.chat.params.logprobs-no-top", P, _classic(logprobs=True))
_p(
    "openai.chat.params.max-tokens-and-max-completion-tokens",
    P,
    _classic(max_tokens=64, max_completion_tokens=64),
    note="both caps given",
)
_p("openai.chat.params.temp-on-reasoning", P, _base(temperature=0.5), expect="error")
_p("openai.chat.params.max-tokens-on-reasoning", P, _base(max_tokens=64), expect="error")
_p("openai.chat.params.logprobs-on-reasoning", P, _base(logprobs=True), expect="error")

# =============================================================================
# tools
# =============================================================================
T = "tools"
_p(
    "openai.chat.tools.auto",
    T,
    _classic(messages=[WEATHER_Q], tools=[WEATHER_TOOL], tool_choice="auto"),
)
_p("openai.chat.tools.auto-reasoning-model", T, _base(messages=[WEATHER_Q], tools=[WEATHER_TOOL]))
_p(
    "openai.chat.tools.required",
    T,
    _classic(messages=[WEATHER_Q], tools=[WEATHER_TOOL], tool_choice="required"),
)
_p(
    "openai.chat.tools.named",
    T,
    _classic(
        messages=[WEATHER_Q],
        tools=[WEATHER_TOOL, TIME_TOOL],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    ),
)
_p(
    "openai.chat.tools.none",
    T,
    _classic(messages=[WEATHER_Q], tools=[WEATHER_TOOL], tool_choice="none"),
)
_p(
    "openai.chat.tools.parallel-true",
    T,
    _classic(
        messages=[_u("What is the weather in Paris and the time in Europe/Paris? Use both tools.")],
        tools=[WEATHER_TOOL, TIME_TOOL],
        tool_choice="required",
        parallel_tool_calls=True,
    ),
)
_p(
    "openai.chat.tools.parallel-false",
    T,
    _classic(
        messages=[_u("What is the weather in Paris and the time in Europe/Paris? Use both tools.")],
        tools=[WEATHER_TOOL, TIME_TOOL],
        tool_choice="required",
        parallel_tool_calls=False,
    ),
)
_p(
    "openai.chat.tools.strict",
    T,
    _classic(messages=[WEATHER_Q], tools=[STRICT_TOOL], tool_choice="required"),
)
_p(
    "openai.chat.tools.no-parameters",
    T,
    _classic(
        messages=[_u("What time is it? Use the tool.")],
        tools=[{"type": "function", "function": {"name": "now", "description": "Current time"}}],
        tool_choice="required",
    ),
)
_p(
    "openai.chat.tools.schema-with-defs",
    T,
    _classic(
        messages=[WEATHER_Q],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather_defs",
                    "parameters": {
                        "type": "object",
                        "$defs": {"city": {"type": "string"}},
                        "properties": {"city": {"$ref": "#/$defs/city"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice="required",
    ),
)
_p(
    "openai.chat.tools.roundtrip",
    T,
    _classic(
        messages=[
            WEATHER_Q,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_probe0001",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_probe0001",
                "content": '{"temp_c": 21, "sky": "clear"}',
            },
        ],
        tools=[WEATHER_TOOL],
    ),
    note="scripted tool call + result, no dependency on the model calling",
)
_p(
    "openai.chat.tools.roundtrip-parallel",
    T,
    _classic(
        messages=[
            _u("What is the weather in Paris and the time in Europe/Paris? Use both tools."),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_probe0001",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    },
                    {
                        "id": "call_probe0002",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": '{"tz":"Europe/Paris"}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_probe0001", "content": '{"temp_c": 21}'},
            {"role": "tool", "tool_call_id": "call_probe0002", "content": '{"time": "14:02"}'},
        ],
        tools=[WEATHER_TOOL, TIME_TOOL],
    ),
)
_p(
    "openai.chat.tools.roundtrip-reasoning-model",
    T,
    _base(
        messages=[
            WEATHER_Q,
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_probe0001",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_probe0001", "content": '{"temp_c": 21}'},
        ],
        tools=[WEATHER_TOOL],
    ),
)
_p(
    "openai.chat.tools.legacy-functions",
    T,
    _classic(
        messages=[WEATHER_Q],
        functions=[WEATHER_TOOL["function"]],
        function_call="auto",
    ),
    note="deprecated functions/function_call surface",
)

# =============================================================================
# structured-output
# =============================================================================
J = "structured-output"
_p(
    "openai.chat.structured.json-object",
    J,
    _classic(
        messages=[_u("Return a json object with a key greeting.")],
        response_format={"type": "json_object"},
    ),
)
_p(
    "openai.chat.structured.json-schema-strict",
    J,
    _classic(
        messages=[_u("Give me a person named Ada aged 36.")],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "person", "strict": True, "schema": PERSON_SCHEMA},
        },
    ),
)
_p(
    "openai.chat.structured.json-schema-nonstrict",
    J,
    _classic(
        messages=[_u("Give me a person named Ada aged 36.")],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "person", "schema": PERSON_SCHEMA},
        },
    ),
)
_p(
    "openai.chat.structured.json-schema-reasoning-model",
    J,
    _base(
        messages=[_u("Give me a person named Ada aged 36.")],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "person", "strict": True, "schema": PERSON_SCHEMA},
        },
    ),
)
_p(
    "openai.chat.structured.json-schema-nested",
    J,
    _classic(
        messages=[_u("Give me a team of two people.")],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "team",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"members": {"type": "array", "items": PERSON_SCHEMA}},
                    "required": ["members"],
                    "additionalProperties": False,
                },
            },
        },
    ),
)
_p("openai.chat.structured.text-explicit", J, _classic(response_format={"type": "text"}))

# =============================================================================
# multimodal
# =============================================================================
M = "multimodal"
_p(
    "openai.chat.multimodal.image-data-url",
    M,
    _base(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What colour is this pixel?"},
                    {"type": "image_url", "image_url": {"url": IMG_DATA_URI}},
                ],
            }
        ]
    ),
)
_p(
    "openai.chat.multimodal.image-data-url-detail-low",
    M,
    _base(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What colour is this pixel?"},
                    {"type": "image_url", "image_url": {"url": IMG_DATA_URI, "detail": "low"}},
                ],
            }
        ]
    ),
)
_p(
    "openai.chat.multimodal.image-url",
    M,
    _base(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": IMG_URL}},
                ],
            }
        ]
    ),
)
_p(
    "openai.chat.multimodal.two-images",
    M,
    _base(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": IMG_DATA_URI}},
                    {"type": "image_url", "image_url": {"url": IMG_DATA_URI}},
                    {"type": "text", "text": "Are these the same?"},
                ],
            }
        ]
    ),
)

# =============================================================================
# pairwise-interactions
# =============================================================================
PW = "pairwise-interactions"
_p(
    "openai.chat.pair.stream-tools-usage-multiturn",
    PW,
    _classic(
        messages=[
            WEATHER_Q,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_probe0001",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_probe0001", "content": '{"temp_c": 21}'},
        ],
        tools=[WEATHER_TOOL],
        stream=True,
        stream_options={"include_usage": True},
    ),
    stream=True,
)
_p(
    "openai.chat.pair.json-schema-with-tools",
    PW,
    _classic(
        messages=[_u("Give me a person named Ada aged 36.")],
        tools=[WEATHER_TOOL],
        tool_choice="none",
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "person", "strict": True, "schema": PERSON_SCHEMA},
        },
    ),
)
_p(
    "openai.chat.pair.stop-and-max-tokens",
    PW,
    _classic(
        messages=[_u("Count from 1 to 20 separated by commas.")],
        stop=["5"],
        max_completion_tokens=8,
    ),
)
_p("openai.chat.pair.n-2-seed-temp-0", PW, _classic(n=2, seed=7, temperature=0))
_p(
    "openai.chat.pair.logprobs-stream-usage",
    PW,
    _classic(logprobs=True, top_logprobs=1, stream=True, stream_options={"include_usage": True}),
    stream=True,
)

# =============================================================================
# error-probes  (zero token cost)
# =============================================================================
E = "error-probes"


def _e(slug, body, **kw):
    _p(f"openai.chat.err.{slug}", E, body, expect="error", **kw)


# body / model
_e("bad-json", '{"model": "@MODEL", "messages": [')
_e("empty-body", {})
_e("body-array", [_u("hi")])
_e("model-missing", {"messages": [_u("hi")], "max_completion_tokens": 64})
_e("model-nonexistent", {"model": "gpt-nope", "messages": [_u("hi")], "max_completion_tokens": 64})
_e("model-empty", {"model": "", "messages": [_u("hi")], "max_completion_tokens": 64})
_e("model-wrong-type", {"model": 42, "messages": [_u("hi")], "max_completion_tokens": 64})
# messages
_e("messages-missing", {"model": "@MODEL", "max_completion_tokens": 64})
_e("messages-empty", _base(messages=[]))
_e("messages-wrong-type", _base(messages="hi"))
_e("message-bad-role", _base(messages=[{"role": "robot", "content": "hi"}]))
_e("message-content-wrong-type", _base(messages=[{"role": "user", "content": 123}]))
_e("message-missing-content", _base(messages=[{"role": "user"}]))
_e(
    "message-bad-part-type",
    _base(messages=[{"role": "user", "content": [{"type": "bogus", "text": "x"}]}]),
)
_e("unknown-top-level-field", _base(frobnicate=1))
_e("unknown-field-in-message", _base(messages=[{"role": "user", "content": "hi", "extra": 1}]))
# sampling
_e("temp-3", _classic(temperature=3.0))
_e("temp-string", _classic(temperature="hot"))
_e("top-p-1.5", _classic(top_p=1.5))
_e("n-0", _classic(n=0))
_e("n-string", _classic(n="two"))
_e("max-completion-tokens-0", _classic(max_completion_tokens=0))
_e("max-completion-tokens-neg", _classic(max_completion_tokens=-1))
_e("max-completion-tokens-string", _classic(max_completion_tokens="many"))
_e("max-completion-tokens-huge", _classic(max_completion_tokens=10_000_000))
_e("stop-5-sequences", _classic(stop=["a", "b", "c", "d", "e"]))
_e("stop-wrong-type", _classic(stop=5))
_e("logprobs-top-25", _classic(logprobs=True, top_logprobs=25))
_e("top-logprobs-without-logprobs", _classic(top_logprobs=2))
_e("seed-string", _classic(seed="seven"))
_e("presence-penalty-3", _classic(presence_penalty=3.0))
_e("frequency-penalty-neg-3", _classic(frequency_penalty=-3.0))
_e("logit-bias-bad-key", _classic(logit_bias={"not-a-token": 1}))
_e("stream-options-without-stream", _classic(stream_options={"include_usage": True}))
_e("reasoning-effort-bogus", _base(reasoning_effort="extreme"))
# tools
_e("tools-bad-type", _classic(messages=[WEATHER_Q], tools=[{"type": "bogus"}]))
_e(
    "tools-missing-name",
    _classic(messages=[WEATHER_Q], tools=[{"type": "function", "function": {}}]),
)
_e(
    "tools-bad-params",
    _classic(
        messages=[WEATHER_Q],
        tools=[{"type": "function", "function": {"name": "f", "parameters": "oops"}}],
    ),
)
_e(
    "strict-missing-addlprops",
    _classic(
        messages=[WEATHER_Q],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice="required",
    ),
)
_e(
    "tool-choice-bad-string",
    _classic(messages=[WEATHER_Q], tools=[WEATHER_TOOL], tool_choice="always"),
)
_e(
    "tool-choice-named-missing",
    _classic(
        messages=[WEATHER_Q],
        tools=[WEATHER_TOOL],
        tool_choice={"type": "function", "function": {"name": "not_a_tool"}},
    ),
)
_e("tool-choice-without-tools", _classic(messages=[WEATHER_Q], tool_choice="required"))
_e("duplicate-tool-names", _classic(messages=[WEATHER_Q], tools=[WEATHER_TOOL, WEATHER_TOOL]))
_e(
    "tool-result-without-call",
    _classic(
        messages=[WEATHER_Q, {"role": "tool", "tool_call_id": "call_nope", "content": "{}"}],
        tools=[WEATHER_TOOL],
    ),
)
_e(
    "tool-result-id-mismatch",
    _classic(
        messages=[
            WEATHER_Q,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_probe0001",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_other", "content": "{}"},
        ],
        tools=[WEATHER_TOOL],
    ),
)
_e(
    "tool-call-missing-result",
    _classic(
        messages=[
            WEATHER_Q,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_probe0001",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            _u("thanks"),
        ],
        tools=[WEATHER_TOOL],
    ),
)
# structured output
_e("response-format-bad-type", _classic(response_format={"type": "yaml"}))
_e(
    "json-schema-missing-name",
    _classic(response_format={"type": "json_schema", "json_schema": {"schema": PERSON_SCHEMA}}),
)
_e(
    "json-schema-strict-missing-addlprops",
    _classic(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "p",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        }
    ),
)
_e(
    "json-object-no-json-word",
    _classic(messages=[_u("Tell me about Paris.")], response_format={"type": "json_object"}),
)
# multimodal
_e(
    "image-bad-data-url",
    _base(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,not-base64!!"},
                    },
                ],
            }
        ]
    ),
)
_e(
    "image-url-missing",
    _base(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]),
)
_e(
    "audio-modality-on-text-model",
    _classic(modalities=["text", "audio"], audio={"voice": "alloy", "format": "wav"}),
)
# headers
_e("bad-api-key", _base(), headers={"Authorization": "Bearer sk-invalid-probe"})
_e("no-api-key", _base(), headers={"Authorization": None})
