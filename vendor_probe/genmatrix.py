"""Generated probe tier: deterministic axis-spec expansion (stdlib only).

Axis specs are grounded in crates/protocols/src/responses.rs / messages.rs
(field names, types, ranges, enums) and the recorded vendor-truth runs.
Strategies, in budget-priority order:

  gen-mutation  — per-field wrong-type / null / empty / unknown-enum /
                  unknown-sibling corruption of a minimal valid request;
                  expect 4xx, records the exact error envelope per field path.
  gen-pairwise  — greedy pairwise covering array over ALL interacting axes
                  (seeded RNG, fixed seed); every row is a valid request.
  gen-3wise     — full cartesian product over the most-interacting axis
                  family (superset of full 3-wise coverage over any 5 of
                  them); constraint-pruned so every combo is valid.
  gen-boundary  — numeric range and length-limit sweeps (billed but tiny:
                  max output tokens pinned to 16).
  gen-content   — system-form x turn-count x content-block-combo sweeps.

Probes are the same plain dicts the curated tier uses. IDs are deterministic
and readable; two invocations produce identical output. `--budget N`
truncates by the priority order above.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import random
import re
import sys

SEED = 20250817

# =============================================================================
# fixtures (mirrors of curated-tier constants; kept local so the module is
# standalone data)
# =============================================================================
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
IMG_DATA_URI = "data:image/png;base64," + PNG_B64
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

FLAT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
    "required": ["city", "population"],
}
NESTED_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "$defs": {"cat": {"type": "string", "enum": ["a", "b", "c"]}},
    "properties": {
        "name": {"type": "string"},
        "tags": {"type": "array", "items": {"$ref": "#/$defs/cat"}},
    },
    "required": ["name", "tags"],
}
O_WEATHER = {
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
O_TIME = {
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
O_JOKE = {
    "type": "function",
    "name": "tell_joke",
    "description": "a joke",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}
O_ZERO = {
    "type": "function",
    "name": "ping",
    "description": "ping",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}
O_CUSTOM = {"type": "custom", "name": "run_cmd", "description": "run a shell command"}
A_WEATHER = {
    "name": "get_weather",
    "description": "Get weather",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
}
A_TIME = {
    "name": "get_time",
    "description": "current time",
    "input_schema": {"type": "object", "properties": {"tz": {"type": "string"}}},
}


def _u(text):
    return {"role": "user", "content": text}


def _om(text):
    return {"type": "message", "role": "user", "content": text}


# =============================================================================
# probe emission helpers
# =============================================================================
def _clone(x):
    return copy.deepcopy(x)


def _probe(pid, category, endpoint, body, *, stream=False, expect="ok", note=None):
    return {
        "id": pid,
        "category": category,
        "endpoint": endpoint,
        "method": "POST",
        "body": body,
        "stream": stream,
        "depends_on": None,
        "expect": expect,
        "headers": None,
        "poll": None,
        "note": note,
    }


_PATH_TOK = re.compile(r"^([A-Za-z_]\w*)?(?:\[(\d+)\])?$")


def _set_path(obj, path, value):
    """Set a dotted path like tools[0].name; parents must already exist."""
    cur = obj
    toks = path.split(".")
    for n, tok in enumerate(toks):
        m = _PATH_TOK.match(tok)
        name, idx = m.group(1), m.group(2)
        last = n == len(toks) - 1
        if name and idx is None:
            if last:
                cur[name] = value
                return
            cur = cur[name]
        else:
            if name:
                cur = cur[name]
            if last:
                cur[int(idx)] = value
                return
            cur = cur[int(idx)]


def _get_parent(obj, path):
    """Return the object at `path` ('' = root); used for sibling injection."""
    if path == "":
        return obj
    cur = obj
    for tok in path.split("."):
        m = _PATH_TOK.match(tok)
        name, idx = m.group(1), m.group(2)
        if name:
            cur = cur[name]
        if idx is not None:
            cur = cur[int(idx)]
    return cur


def _slug(path):
    return path.replace("[", "").replace("]", "").replace(".", "-") or "root"


def _merge(base, *overlays):
    body = _clone(base)
    for ov in overlays:
        for k, v in ov.items():
            body[k] = _clone(v)
    return body


# =============================================================================
# field mutations
# =============================================================================
_WRONG = {
    "wrong-type-string": "not-valid",
    "wrong-type-number": 12345,
    "wrong-type-bool": True,
    "wrong-type-array": [1, 2, 3],
    "wrong-type-object": {"bogus": True},
}
_EMPTY = {"str": "", "arr": [], "obj": {}}
_KINDS = {
    "str": [
        "wrong-type-number",
        "wrong-type-bool",
        "wrong-type-array",
        "wrong-type-object",
        "null",
        "empty",
    ],
    "enum": [
        "wrong-type-number",
        "wrong-type-bool",
        "wrong-type-array",
        "wrong-type-object",
        "null",
        "unknown-enum",
    ],
    "int": [
        "wrong-type-string",
        "wrong-type-bool",
        "wrong-type-array",
        "wrong-type-object",
        "null",
    ],
    "num": [
        "wrong-type-string",
        "wrong-type-bool",
        "wrong-type-array",
        "wrong-type-object",
        "null",
    ],
    "bool": ["wrong-type-string", "wrong-type-number", "null"],
    "arr": [
        "wrong-type-string",
        "wrong-type-number",
        "wrong-type-bool",
        "wrong-type-object",
        "null",
        "empty",
    ],
    "obj": [
        "wrong-type-string",
        "wrong-type-number",
        "wrong-type-bool",
        "wrong-type-array",
        "null",
        "empty",
    ],
    # poly fields (string-or-array unions): only unambiguously-wrong shapes
    "poly": ["wrong-type-number", "wrong-type-bool", "null"],
}


def _mutation_value(ftype, kind):
    if kind == "null":
        return None
    if kind == "empty":
        return _clone(_EMPTY[ftype])
    if kind == "unknown-enum":
        return "__bogus_enum__"
    return _clone(_WRONG[kind])


def _gen_mutations(prefix, category, endpoint, base, ctx_table, fields):
    out = []
    for path, ftype, ctxs in fields:
        body_base = _merge(base, *(ctx_table[c] for c in ctxs))
        if ftype == "sibling":
            body = _clone(body_base)
            _get_parent(body, path)["probe_unknown_field"] = 1
            out.append(
                _probe(
                    f"{prefix}.mut.{_slug(path)}.unknown-sibling",
                    category,
                    endpoint,
                    body,
                    expect="error",
                )
            )
            continue
        for kind in _KINDS[ftype]:
            body = _clone(body_base)
            _set_path(body, path, _mutation_value(ftype, kind))
            out.append(
                _probe(
                    f"{prefix}.mut.{_slug(path)}.{kind}",
                    category,
                    endpoint,
                    body,
                    expect="error",
                )
            )
    return out


# --- OpenAI field spec (ResponsesRequest + nested shapes) --------------------
O_BASE = {"model": "@MODEL", "input": "hi", "max_output_tokens": 16}
O_EP = "/v1/responses"
O_CTX = {
    "classic": {"model": "@MODEL_CLASSIC"},
    "msg": {"input": [_om("hi")]},
    "parts": {
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ]
    },
    "img": {
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "image_url": IMG_DATA_URI, "detail": "low"}],
            }
        ]
    },
    "file": {
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_file", "filename": "probe.pdf", "file_data": PDF_DATA_URI}
                ],
            }
        ]
    },
    "tools": {"tools": [O_WEATHER]},
    "tc_obj": {"tools": [O_WEATHER], "tool_choice": {"type": "function", "name": "get_weather"}},
    "tc_str": {"tools": [O_WEATHER], "tool_choice": "auto"},
    "reasoning": {"reasoning": {"effort": "low"}},
    "summary": {"reasoning": {"effort": "low", "summary": "auto"}},
    "text_fmt": {
        "text": {
            "format": {
                "type": "json_schema",
                "name": "city_facts",
                "schema": FLAT_SCHEMA,
                "strict": True,
            }
        }
    },
    "verbosity": {"text": {"verbosity": "low"}},
    "stream": {"stream": True, "stream_options": {"include_obfuscation": False}},
    "include": {"include": ["reasoning.encrypted_content"], "store": False},
    "meta": {"metadata": {"k": "v"}},
    "stop": {"stop": ["END"], "model": "@MODEL_CLASSIC"},
    "bg": {"background": False},
}
O_FIELDS = [
    ("model", "str", ()),
    ("input", "poly", ()),
    ("instructions", "str", ()),
    ("max_output_tokens", "int", ()),
    ("max_tool_calls", "int", ("tools",)),
    ("metadata", "obj", ()),
    ("metadata.k", "str", ("meta",)),
    ("conversation", "str", ()),
    ("parallel_tool_calls", "bool", ("tools",)),
    ("previous_response_id", "str", ()),
    ("reasoning", "obj", ()),
    ("reasoning.effort", "enum", ("reasoning",)),
    ("reasoning.summary", "enum", ("summary",)),
    ("service_tier", "enum", ()),
    ("store", "bool", ()),
    ("stream", "bool", ()),
    ("stream_options", "obj", ("stream",)),
    ("stream_options.include_obfuscation", "bool", ("stream",)),
    ("temperature", "num", ("classic",)),
    ("top_p", "num", ("classic",)),
    ("top_logprobs", "int", ("classic",)),
    ("truncation", "enum", ()),
    ("text", "obj", ()),
    ("text.format", "obj", ("text_fmt",)),
    ("text.format.type", "enum", ("text_fmt",)),
    ("text.format.name", "str", ("text_fmt",)),
    ("text.format.schema", "obj", ("text_fmt",)),
    ("text.format.strict", "bool", ("text_fmt",)),
    ("text.verbosity", "enum", ("verbosity",)),
    ("include", "arr", ("include",)),
    ("include[0]", "enum", ("include",)),
    ("user", "str", ()),
    ("safety_identifier", "str", ()),
    ("prompt_cache_key", "str", ()),
    ("background", "bool", ("bg",)),
    ("stop", "arr", ("stop",)),
    ("frequency_penalty", "num", ("classic",)),
    ("presence_penalty", "num", ("classic",)),
    ("tools", "arr", ("tools",)),
    ("tools[0]", "obj", ("tools",)),
    ("tools[0].type", "enum", ("tools",)),
    ("tools[0].name", "str", ("tools",)),
    ("tools[0].description", "str", ("tools",)),
    ("tools[0].parameters", "obj", ("tools",)),
    ("tools[0].strict", "bool", ("tools",)),
    ("tool_choice", "enum", ("tc_str",)),
    ("tool_choice.type", "enum", ("tc_obj",)),
    ("tool_choice.name", "str", ("tc_obj",)),
    ("input[0]", "obj", ("msg",)),
    ("input[0].type", "enum", ("msg",)),
    ("input[0].role", "enum", ("msg",)),
    ("input[0].content", "poly", ("msg",)),
    ("input[0].content[0]", "obj", ("parts",)),
    ("input[0].content[0].type", "enum", ("parts",)),
    ("input[0].content[0].text", "str", ("parts",)),
    ("input[0].content[0].image_url", "str", ("img",)),
    ("input[0].content[0].detail", "enum", ("img",)),
    ("input[0].content[0].file_data", "str", ("file",)),
    ("input[0].content[0].filename", "str", ("file",)),
    ("", "sibling", ()),
    ("reasoning", "sibling", ("reasoning",)),
    ("text", "sibling", ("verbosity",)),
    ("text.format", "sibling", ("text_fmt",)),
    ("stream_options", "sibling", ("stream",)),
    ("tools[0]", "sibling", ("tools",)),
    ("tool_choice", "sibling", ("tc_obj",)),
    ("input[0]", "sibling", ("msg",)),
    ("input[0].content[0]", "sibling", ("parts",)),
]

# --- Anthropic field spec (CreateMessageRequest + nested shapes) -------------
A_BASE = {"model": "@MODEL", "max_tokens": 16, "messages": [_u("hi")]}
A_EP = "/v1/messages"
A_CTX = {
    "tools": {"tools": [A_WEATHER]},
    "tc": {"tools": [A_WEATHER], "tool_choice": {"type": "tool", "name": "get_weather"}},
    "think": {"max_tokens": 2048, "thinking": {"type": "enabled", "budget_tokens": 1024}},
    "think4k": {"max_tokens": 4096, "thinking": {"type": "enabled", "budget_tokens": 1024}},
    "sysstr": {"system": "You are terse."},
    "sysblocks": {"system": [{"type": "text", "text": "You are terse."}]},
    "syscache": {
        "system": [
            {"type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral"}}
        ]
    },
    "meta": {"metadata": {"user_id": "probe-user"}},
    "stop": {"stop_sequences": ["END"]},
    "blocks": {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
    "img": {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64},
                    }
                ],
            }
        ]
    },
    "toolres": {
        "tools": [A_WEATHER],
        "messages": [
            _u("Weather in Paris?"),
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
    "cachetool": {"tools": [dict(A_WEATHER, cache_control={"type": "ephemeral"})]},
}
A_FIELDS = [
    ("model", "str", ()),
    ("messages", "arr", ()),
    ("messages[0]", "obj", ()),
    ("messages[0].role", "enum", ()),
    ("messages[0].content", "poly", ()),
    ("max_tokens", "int", ()),
    ("metadata", "obj", ()),
    ("metadata.user_id", "str", ("meta",)),
    ("service_tier", "enum", ()),
    ("stop_sequences", "arr", ("stop",)),
    ("stop_sequences[0]", "str", ("stop",)),
    ("stream", "bool", ()),
    ("system", "poly", ("sysstr",)),
    ("system[0]", "obj", ("sysblocks",)),
    ("system[0].type", "enum", ("sysblocks",)),
    ("system[0].text", "str", ("sysblocks",)),
    ("system[0].cache_control", "obj", ("syscache",)),
    ("system[0].cache_control.type", "enum", ("syscache",)),
    ("temperature", "num", ()),
    ("thinking", "obj", ("think",)),
    ("thinking.type", "enum", ("think",)),
    ("thinking.budget_tokens", "int", ("think",)),
    ("tool_choice", "obj", ("tc",)),
    ("tool_choice.type", "enum", ("tc",)),
    ("tool_choice.name", "str", ("tc",)),
    ("tool_choice.disable_parallel_tool_use", "bool", ("tc",)),
    ("tools", "arr", ("tools",)),
    ("tools[0]", "obj", ("tools",)),
    ("tools[0].name", "str", ("tools",)),
    ("tools[0].description", "str", ("tools",)),
    ("tools[0].input_schema", "obj", ("tools",)),
    ("tools[0].input_schema.type", "enum", ("tools",)),
    ("tools[0].cache_control", "obj", ("cachetool",)),
    ("tools[0].cache_control.type", "enum", ("cachetool",)),
    ("top_k", "int", ()),
    ("top_p", "num", ()),
    ("messages[0].content[0]", "obj", ("blocks",)),
    ("messages[0].content[0].type", "enum", ("blocks",)),
    ("messages[0].content[0].text", "str", ("blocks",)),
    ("messages[0].content[0].source", "obj", ("img",)),
    ("messages[0].content[0].source.type", "enum", ("img",)),
    ("messages[0].content[0].source.media_type", "enum", ("img",)),
    ("messages[0].content[0].source.data", "str", ("img",)),
    ("messages[2].content[0].tool_use_id", "str", ("toolres",)),
    ("messages[2].content[0].is_error", "bool", ("toolres",)),
    ("messages[2].content[0].content", "poly", ("toolres",)),
    ("", "sibling", ()),
    ("messages[0]", "sibling", ()),
    ("messages[0].content[0]", "sibling", ("blocks",)),
    ("system[0]", "sibling", ("sysblocks",)),
    ("thinking", "sibling", ("think",)),
    ("tool_choice", "sibling", ("tc",)),
    ("tools[0]", "sibling", ("tools",)),
    ("messages[0].content[0].source", "sibling", ("img",)),
    ("metadata", "sibling", ("meta",)),
]

# =============================================================================
# boundary sweeps  (values grounded in validator ranges + vendor docs;
# unknown limits get a "record whichever behavior" case)
# =============================================================================
O_BOUNDS = [
    (
        "max_output_tokens",
        (),
        [
            ("below-min", 15, "error"),
            ("min", 16, "ok"),
            ("min-plus", 17, "ok"),
            ("typical", 64, "ok"),
            ("max", 128000, "ok"),
            ("above-max", 300000, "error"),
        ],
    ),
    (
        "temperature",
        ("classic",),
        [
            ("below-min", -0.5, "error"),
            ("min", 0, "ok"),
            ("min-plus", 0.01, "ok"),
            ("typical", 1.0, "ok"),
            ("max", 2.0, "ok"),
            ("above-max", 2.5, "error"),
        ],
    ),
    (
        "top_p",
        ("classic",),
        [
            ("below-min", -0.5, "error"),
            ("min", 0, "ok"),
            ("min-plus", 0.01, "ok"),
            ("typical", 0.5, "ok"),
            ("max", 1.0, "ok"),
            ("above-max", 1.5, "error"),
        ],
    ),
    (
        "top_logprobs",
        ("classic",),
        [
            ("below-min", -1, "error"),
            ("min", 0, "ok"),
            ("min-plus", 1, "ok"),
            ("typical", 5, "ok"),
            ("max", 20, "ok"),
            ("above-max", 21, "error"),
        ],
    ),
    (
        "max_tool_calls",
        ("tools",),
        [
            ("below-min", 0, "error"),
            ("min", 1, "ok"),
            ("min-plus", 2, "ok"),
            ("typical", 16, "ok"),
            ("huge", 10000, "ok"),
        ],
    ),
    (
        "frequency_penalty",
        ("classic",),
        [("min", -2.0, "ok"), ("zero", 0.0, "ok"), ("max", 2.0, "ok"), ("above-max", 2.5, "error")],
    ),
    ("presence_penalty", ("classic",), [("min", -2.0, "ok"), ("max", 2.0, "ok")]),
    (
        "metadata",
        (),
        [
            ("empty", {}, "ok"),
            ("one-key", {"k": "v"}, "ok"),
            ("at-limit-16-keys", {f"k{i}": "v" for i in range(16)}, "ok"),
            ("above-limit-17-keys", {f"k{i}": "v" for i in range(17)}, "error"),
            ("value-at-512", {"k": "v" * 512}, "ok"),
            ("value-above-512", {"k": "v" * 513}, "error"),
        ],
    ),
    (
        "instructions",
        (),
        [("empty", "", "ok"), ("one-char", "x", "ok"), ("long-4k", "Be brief. " * 400, "ok")],
    ),
    ("user", (), [("empty", "", "ok"), ("one-char", "u", "ok")]),
    ("safety_identifier", (), [("empty", "", "ok"), ("one-char", "s", "ok")]),
    ("prompt_cache_key", (), [("empty", "", "ok"), ("one-char", "c", "ok")]),
    (
        "tools[0].name",
        ("tools",),
        [
            ("one-char", "f", "ok"),
            ("at-limit-64", "f" * 64, "ok"),
            ("above-limit-65", "f" * 65, "error"),
        ],
    ),
    ("input", (), [("one-char", "x", "ok")]),
]
A_BOUNDS = [
    (
        "max_tokens",
        (),
        [
            ("zero", 0, "error"),
            ("min", 1, "ok"),
            ("min-plus", 2, "ok"),
            ("typical", 16, "ok"),
            ("model-cap", 64000, "ok"),
            ("above-cap", 100000, "error"),
        ],
    ),
    (
        "temperature",
        (),
        [
            ("below-min", -0.5, "error"),
            ("min", 0, "ok"),
            ("min-plus", 0.01, "ok"),
            ("mid", 0.5, "ok"),
            ("max", 1.0, "ok"),
            ("above-max", 1.5, "error"),
        ],
    ),
    (
        "top_p",
        (),
        [
            ("below-min", -0.5, "error"),
            ("tiny", 0.01, "ok"),
            ("mid", 0.5, "ok"),
            ("near-max", 0.99, "ok"),
            ("max", 1.0, "ok"),
            ("above-max", 1.5, "error"),
        ],
    ),
    (
        "top_k",
        (),
        [
            ("negative", -1, "error"),
            ("zero", 0, "ok"),
            ("min", 1, "ok"),
            ("typical", 40, "ok"),
            ("huge", 100000, "ok"),
        ],
    ),
    (
        "thinking.budget_tokens",
        ("think4k",),
        [
            ("below-min", 1023, "error"),
            ("min", 1024, "ok"),
            ("min-plus", 1025, "ok"),
            ("typical", 2048, "ok"),
            ("eq-max-tokens", 4096, "error"),
        ],
    ),
    (
        "metadata.user_id",
        ("meta",),
        [
            ("empty", "", "ok"),
            ("one-char", "u", "ok"),
            ("at-256", "u" * 256, "ok"),
            ("at-300", "u" * 300, "error"),
        ],
    ),
    (
        "tools[0].name",
        ("tools",),
        [
            ("one-char", "f", "ok"),
            ("at-64", "f" * 64, "ok"),
            ("len-128", "f" * 128, "ok"),
            ("len-200", "f" * 200, "error"),
        ],
    ),
    (
        "stop_sequences",
        (),
        [
            ("one", ["END"], "ok"),
            ("four", ["A1", "B2", "C3", "D4"], "ok"),
            ("eight", [f"S{i}E" for i in range(8)], "ok"),
            ("long-element", ["X" * 500], "ok"),
        ],
    ),
    ("system", (), [("empty", "", "ok"), ("one-char", "x", "ok")]),
]


def _gen_bounds(prefix, category, endpoint, base, ctx_table, bounds):
    out = []
    for path, ctxs, cases in bounds:
        body_base = _merge(base, *(ctx_table[c] for c in ctxs))
        for slug, value, expect in cases:
            body = _clone(body_base)
            _set_path(body, path, _clone(value))
            out.append(
                _probe(
                    f"{prefix}.bound.{_slug(path)}.{slug}",
                    category,
                    endpoint,
                    body,
                    expect=expect,
                )
            )
    return out


# =============================================================================
# combination axes (every value = a top-level overlay; each axis owns its
# own top-level keys so overlays compose by plain update)
# =============================================================================
def _o_turns(n, final="Say hi."):
    items = []
    for i in range(n - 1):
        items.append(_om(f"turn {i}"))
        items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": f"ack {i}"}],
            }
        )
    items.append(_om(final))
    return items


O_AX_STREAM = ("stream", "st", [("off", {}), ("on", {"stream": True})])
O_AX_TOOLS = (
    "tools",
    "tl",
    [
        ("abs", {}),
        ("fn1", {"tools": [O_WEATHER]}),
        ("fn3", {"tools": [O_WEATHER, O_TIME, O_JOKE]}),
        ("cus", {"tools": [O_CUSTOM]}),
        ("nostrict", {"tools": [dict(O_WEATHER, strict=False)]}),
        ("zero", {"tools": [O_ZERO]}),
    ],
)
O_AX_TC = (
    "tool_choice",
    "tc",
    [
        ("omit", {}),
        ("auto", {"tool_choice": "auto"}),
        ("none", {"tool_choice": "none"}),
        ("req", {"tool_choice": "required"}),
        ("named", {"tool_choice": {"type": "function", "name": "get_weather"}}),
        (
            "alw-a",
            {
                "tool_choice": {
                    "type": "allowed_tools",
                    "mode": "auto",
                    "tools": [{"type": "function", "name": "get_weather"}],
                }
            },
        ),
        (
            "alw-r",
            {
                "tool_choice": {
                    "type": "allowed_tools",
                    "mode": "required",
                    "tools": [{"type": "function", "name": "get_weather"}],
                }
            },
        ),
    ],
)
O_AX_FMT = (
    "format",
    "fmt",
    [
        ("omit", {}),
        ("text", {"text": {"format": {"type": "text"}}}),
        ("jobj", {"text": {"format": {"type": "json_object"}}}),
        (
            "jschema",
            {
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "city_facts",
                        "schema": FLAT_SCHEMA,
                        "strict": True,
                    }
                }
            },
        ),
        (
            "jnest",
            {
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "item",
                        "schema": NESTED_SCHEMA,
                        "strict": True,
                    }
                }
            },
        ),
    ],
)
O_AX_INPUT = (
    "input",
    "in",
    [
        ("str", {"input": "Say hi."}),
        ("msg", {"input": [_om("Say hi.")]}),
        (
            "parts",
            {
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Say hi."}],
                    }
                ]
            },
        ),
        ("2turn", {"input": _o_turns(2)}),
        ("8turn", {"input": _o_turns(8)}),
        (
            "sys",
            {
                "input": [
                    {"type": "message", "role": "system", "content": "Be terse."},
                    _om("Say hi."),
                ]
            },
        ),
        (
            "dev",
            {
                "input": [
                    {"type": "message", "role": "developer", "content": "Be terse."},
                    _om("Say hi."),
                ]
            },
        ),
        ("instr", {"instructions": "Be terse.", "input": "Say hi."}),
        (
            "instr-sys",
            {
                "instructions": "Be terse.",
                "input": [
                    {"type": "message", "role": "system", "content": "Answer briefly."},
                    _om("Say hi."),
                ],
            },
        ),
        (
            "2text",
            {
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Say"},
                            {"type": "input_text", "text": "hi."},
                        ],
                    }
                ]
            },
        ),
    ],
)
O_AX_STORE = ("store", "store", [("omit", {}), ("off", {"store": False})])
O_AX_REASON = (
    "reasoning",
    "rs",
    [
        ("omit", {}),
        ("minimal", {"reasoning": {"effort": "minimal"}}),
        ("low", {"reasoning": {"effort": "low"}}),
        ("high", {"reasoning": {"effort": "high"}}),
    ],
)
O_AX_TRUNC = (
    "truncation",
    "tr",
    [("omit", {}), ("auto", {"truncation": "auto"}), ("disabled", {"truncation": "disabled"})],
)
O_AX_PAR = (
    "parallel",
    "par",
    [("omit", {}), ("on", {"parallel_tool_calls": True}), ("off", {"parallel_tool_calls": False})],
)
O_AX_INC = (
    "include",
    "inc",
    [
        ("omit", {}),
        ("encrypted", {"include": ["reasoning.encrypted_content"]}),
        ("inapplicable", {"include": ["file_search_call.results"]}),
    ],
)
O_AX_STORE3 = ("store", "store", [("omit", {}), ("on", {"store": True}), ("off", {"store": False})])
O_AX_SVC = (
    "service_tier",
    "svc",
    [("omit", {}), ("auto", {"service_tier": "auto"}), ("default", {"service_tier": "default"})],
)

# the 6-axis full-cartesian family: superset of full 3-wise coverage over the
# stream x tools x tool_choice x format x input 5-axis core
O_CARTESIAN = [O_AX_STREAM, O_AX_TOOLS, O_AX_TC, O_AX_FMT, O_AX_INPUT, O_AX_STORE]
O_PAIRWISE = [
    O_AX_STREAM,
    O_AX_TOOLS,
    O_AX_TC,
    O_AX_FMT,
    O_AX_REASON,
    O_AX_STORE3,
    O_AX_TRUNC,
    O_AX_PAR,
    O_AX_INC,
    O_AX_INPUT,
    O_AX_SVC,
]

_O_TOOLS_WITH_WEATHER = {"fn1", "fn3", "nostrict"}


def _o_forbidden(ax1, v1, ax2, v2):
    """Symmetric pair-level constraint check (all cross-field rules are 2-ary)."""
    d = {ax1: v1, ax2: v2}
    tc, tl = d.get("tool_choice"), d.get("tools")
    if tc is not None and tl is not None:
        if tc != "omit" and tl == "abs":
            return True
        if tc in ("named", "alw-a", "alw-r") and tl not in _O_TOOLS_WITH_WEATHER and tl != "abs":
            return True
    if d.get("parallel") not in (None, "omit") and tl == "abs":
        return True
    if d.get("include") == "encrypted" and d.get("store") in ("omit", "on"):
        return True
    return False


def _a_turns(n, final_content):
    msgs = []
    for i in range(n - 1):
        msgs.append(_u(f"turn {i}"))
        msgs.append({"role": "assistant", "content": f"ack {i}"})
    msgs.append({"role": "user", "content": final_content})
    return msgs


A_DOC_BLOCK = {
    "type": "document",
    "source": {
        "type": "text",
        "media_type": "text/plain",
        "data": "The capital of France is Paris.",
    },
}
A_AX_STREAM = ("stream", "st", [("off", {}), ("on", {"stream": True})])
A_AX_TOOLS = (
    "tools",
    "tl",
    [
        ("abs", {}),
        ("one", {"tools": [A_WEATHER]}),
        ("two", {"tools": [A_WEATHER, A_TIME]}),
        ("cached", {"tools": [dict(A_WEATHER, cache_control={"type": "ephemeral"})]}),
    ],
)
A_AX_TC = (
    "tool_choice",
    "tc",
    [
        ("omit", {}),
        ("auto", {"tool_choice": {"type": "auto"}}),
        ("any", {"tool_choice": {"type": "any"}}),
        ("named", {"tool_choice": {"type": "tool", "name": "get_weather"}}),
        ("none", {"tool_choice": {"type": "none"}}),
        ("auto-nopar", {"tool_choice": {"type": "auto", "disable_parallel_tool_use": True}}),
    ],
)
A_AX_THINK = (
    "thinking",
    "th",
    [
        ("off", {}),
        ("on", {"thinking": {"type": "enabled", "budget_tokens": 1024}, "max_tokens": 2048}),
    ],
)
A_AX_SYS = (
    "system",
    "sys",
    [
        ("omit", {}),
        ("str", {"system": "You are terse."}),
        ("b1", {"system": [{"type": "text", "text": "You are terse."}]}),
        (
            "b2",
            {
                "system": [
                    {"type": "text", "text": "You are terse."},
                    {"type": "text", "text": "Answer in English."},
                ]
            },
        ),
    ],
)
A_AX_CONTENT = (
    "content",
    "ct",
    [
        ("str", {"messages": [_u("Say hi.")]}),
        (
            "blocks",
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "Say hi."}]}]},
        ),
        ("2turn", {"messages": _a_turns(2, "Say hi.")}),
        ("8turn", {"messages": _a_turns(8, "Say hi.")}),
        (
            "prefill",
            {
                "messages": [
                    _u("Complete: the sky is"),
                    {"role": "assistant", "content": "The sky is"},
                ]
            },
        ),
        (
            "doc",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [A_DOC_BLOCK, {"type": "text", "text": "Capital of France?"}],
                    }
                ]
            },
        ),
    ],
)
A_AX_TEMP = (
    "temp",
    "tmp",
    [
        ("omit", {}),
        ("zero", {"temperature": 0.0}),
        ("mid", {"temperature": 0.5}),
        ("one", {"temperature": 1.0}),
    ],
)
A_AX_STOP = (
    "stop",
    "sp",
    [
        ("omit", {}),
        ("one", {"stop_sequences": ["END"]}),
        ("two", {"stop_sequences": ["END", "STOP"]}),
    ],
)

A_CARTESIAN = [A_AX_STREAM, A_AX_TOOLS, A_AX_TC, A_AX_THINK, A_AX_SYS, A_AX_CONTENT, A_AX_TEMP]
A_PAIRWISE = [
    A_AX_STREAM,
    A_AX_TC,
    A_AX_TOOLS,
    A_AX_THINK,
    A_AX_SYS,
    A_AX_STOP,
    A_AX_TEMP,
    A_AX_CONTENT,
]


def _a_forbidden(ax1, v1, ax2, v2):
    d = {ax1: v1, ax2: v2}
    tc, tl = d.get("tool_choice"), d.get("tools")
    if tc is not None and tl is not None and tc != "omit" and tl == "abs":
        return True
    if d.get("thinking") == "on":
        if d.get("tool_choice") in ("any", "named"):
            return True  # thinking incompatible with forced tool use
        if d.get("temp") in ("zero", "mid"):
            return True  # only temperature=1 allowed with thinking
        if d.get("content") == "prefill":
            return True  # assistant prefill incompatible with thinking
    return False


def _combo_valid(axes, forbidden, combo):
    names = [n for n, _, _ in axes]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if forbidden(names[i], combo[i], names[j], combo[j]):
                return False
    return True


def _mention_json(inp):
    """json_object format requires the word JSON somewhere in the input."""
    if isinstance(inp, str):
        return inp + " Respond in JSON."
    out = _clone(inp)
    for item in reversed(out):
        if isinstance(item, dict) and item.get("role") == "user":
            c = item.get("content")
            if isinstance(c, str):
                item["content"] = c + " Respond in JSON."
            elif isinstance(c, list):
                for part in c:
                    if part.get("type") in ("input_text", "text"):
                        part["text"] += " Respond in JSON."
                        break
                else:
                    c.append({"type": "input_text", "text": "Respond in JSON."})
            break
    return out


def _o_combo_body(axes, combo):
    body = {"model": "@MODEL", "max_output_tokens": 256}
    for (name, _short, values), slug in zip(axes, combo):
        overlay = dict(values)[slug]
        body = _merge(body, overlay)
    fmt = (body.get("text") or {}).get("format") or {}
    if fmt.get("type") == "json_object":
        body["input"] = _mention_json(body["input"])
    return body


def _a_combo_body(axes, combo):
    body = {"model": "@MODEL", "max_tokens": 64}
    for (name, _short, values), slug in zip(axes, combo):
        overlay = dict(values)[slug]
        body = _merge(body, overlay)
    return body


def _combo_probe(prefix, category, endpoint, axes, combo, body, kind):
    slugs = ".".join(f"{short}-{slug}" for (_n, short, _v), slug in zip(axes, combo))
    return _probe(
        f"{prefix}.{kind}.{slugs}",
        category,
        endpoint,
        body,
        stream=body.get("stream") is True,
    )


def _gen_cartesian(prefix, endpoint, axes, forbidden, body_fn):
    out = []
    domains = [[s for s, _ in values] for _n, _s, values in axes]
    for combo in itertools.product(*domains):
        if not _combo_valid(axes, forbidden, combo):
            continue
        body = body_fn(axes, combo)
        out.append(_combo_probe(prefix, "gen-3wise", endpoint, axes, combo, body, "3w"))
    return out


# --- greedy pairwise covering array ------------------------------------------
def _row_pairs(row):
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            yield (i, row[i], j, row[j])


def _complete_row(seed_pair, names, domains, forbidden, rng):
    n = len(names)
    row = [None] * n
    i, vi, j, vj = seed_pair
    row[i], row[j] = vi, vj
    order = list(range(n))
    rng.shuffle(order)
    for k in order:
        if row[k] is not None:
            continue
        choices = list(domains[k])
        rng.shuffle(choices)
        for c in choices:
            ok = all(
                row[m] is None or not forbidden(names[k], c, names[m], row[m])
                for m in range(n)
                if m != k
            )
            if ok:
                row[k] = c
                break
        else:
            return None
    return row


def _greedy_pairwise(axes, forbidden, rng, candidates=40):
    """Standard greedy pairwise: each row is the best of N random valid rows."""
    names = [n for n, _, _ in axes]
    domains = [[s for s, _ in values] for _n, _s, values in axes]
    n = len(names)
    uncovered = {}
    for i in range(n):
        for j in range(i + 1, n):
            for vi in domains[i]:
                for vj in domains[j]:
                    if not forbidden(names[i], vi, names[j], vj):
                        uncovered[(i, vi, j, vj)] = True
    rows = []
    while uncovered:
        ulist = list(uncovered)
        best, best_new = None, -1
        for _ in range(candidates):
            seed = ulist[rng.randrange(len(ulist))]
            row = _complete_row(seed, names, domains, forbidden, rng)
            if row is None:
                continue
            new = sum(1 for p in _row_pairs(row) if p in uncovered)
            if new > best_new:
                best, best_new = row, new
        if best is None:
            del uncovered[ulist[0]]  # pair not completable; drop (should not happen)
            continue
        rows.append(best)
        for p in _row_pairs(best):
            uncovered.pop(p, None)
    return rows


def _gen_pairwise(prefix, endpoint, axes, forbidden, body_fn):
    rng = random.Random(SEED)
    rows = _greedy_pairwise(axes, forbidden, rng)
    out = []
    names = [n for n, _, _ in axes]
    for idx, row in enumerate(rows):
        body = body_fn(axes, tuple(row))
        note = " ".join(f"{n}={s}" for n, s in zip(names, row))
        p = _combo_probe(prefix, "gen-pairwise", endpoint, axes, tuple(row), body, "pw")
        p["id"] = f"{prefix}.pw.{idx:03d}"
        p["note"] = note
        out.append(p)
    return out


# =============================================================================
# content-shape sweeps: system-form x turn-count x block-combo
# =============================================================================
O_SYS_FORMS = [
    ("none", {}),
    ("instr", {"instructions": "Be concise."}),
    ("sysrole", None),  # handled in message list
    ("devrole", None),
]
O_BLOCKS = {
    "text": {"type": "input_text", "text": "Describe this."},
    "image": {"type": "input_image", "image_url": IMG_DATA_URI, "detail": "low"},
    "file": {"type": "input_file", "filename": "probe.pdf", "file_data": PDF_DATA_URI},
}
A_BLOCKS = {
    "text": {"type": "text", "text": "Describe this."},
    "image": {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64},
    },
    "doc": A_DOC_BLOCK,
    "search": {
        "type": "search_result",
        "source": "https://example.com/doc",
        "title": "Probe Result",
        "content": [{"type": "text", "text": "Paris is the capital of France."}],
        "citations": {"enabled": True},
    },
}
_O_BLOCK_COMBOS = [
    "text",
    "image",
    "file",
    "text-image",
    "text-file",
    "image-file",
    "text-image-file",
]
_A_BLOCK_COMBOS = ["text", "image", "doc", "text-image", "text-doc", "search", "toolres"]
_TURNS = [1, 2, 8]


def _gen_content_openai():
    out = []
    for sys_slug, sys_overlay in O_SYS_FORMS:
        for n in _TURNS:
            for combo in _O_BLOCK_COMBOS:
                parts = [_clone(O_BLOCKS[b]) for b in combo.split("-")]
                items = _o_turns(n, final="ignored")
                items[-1] = {"type": "message", "role": "user", "content": parts}
                if sys_slug == "sysrole":
                    items.insert(0, {"type": "message", "role": "system", "content": "Be concise."})
                elif sys_slug == "devrole":
                    items.insert(
                        0, {"type": "message", "role": "developer", "content": "Be concise."}
                    )
                body = {"model": "@MODEL", "max_output_tokens": 64, "input": items}
                if sys_overlay:
                    body.update(_clone(sys_overlay))
                out.append(
                    _probe(
                        f"openai.responses.gen.content.sys-{sys_slug}.{n}t.{combo}",
                        "gen-content",
                        O_EP,
                        body,
                    )
                )
    return out


def _gen_content_anthropic():
    out = []
    sys_forms = [(s, dict(A_AX_SYS[2])[s]) for s in ("omit", "str", "b1", "b2")]
    for sys_slug, sys_overlay in sys_forms:
        for n in _TURNS:
            for combo in _A_BLOCK_COMBOS:
                body = {"model": "@MODEL", "max_tokens": 64}
                if combo == "toolres":
                    msgs = _a_turns(n, "ignored")[:-1]
                    msgs += _clone(A_CTX["toolres"]["messages"])
                    body["tools"] = [_clone(A_WEATHER)]
                else:
                    blocks = [_clone(A_BLOCKS[b]) for b in combo.split("-")]
                    msgs = _a_turns(n, blocks)
                body["messages"] = msgs
                body.update(_clone(sys_overlay))
                out.append(
                    _probe(
                        f"anth.gen.content.sys-{sys_slug}.{n}t.{combo}",
                        "gen-content",
                        A_EP,
                        body,
                    )
                )
    return out


# =============================================================================
# public API
# =============================================================================
def _strategies(vendor):
    if vendor == "openai":
        pre = "openai.responses.gen"
        return [
            ("gen-mutation", _gen_mutations(pre, "gen-mutation", O_EP, O_BASE, O_CTX, O_FIELDS)),
            ("gen-pairwise", _gen_pairwise(pre, O_EP, O_PAIRWISE, _o_forbidden, _o_combo_body)),
            ("gen-3wise", _gen_cartesian(pre, O_EP, O_CARTESIAN, _o_forbidden, _o_combo_body)),
            ("gen-boundary", _gen_bounds(pre, "gen-boundary", O_EP, O_BASE, O_CTX, O_BOUNDS)),
            ("gen-content", _gen_content_openai()),
        ]
    if vendor == "anthropic":
        pre = "anth.gen"
        return [
            ("gen-mutation", _gen_mutations(pre, "gen-mutation", A_EP, A_BASE, A_CTX, A_FIELDS)),
            ("gen-pairwise", _gen_pairwise(pre, A_EP, A_PAIRWISE, _a_forbidden, _a_combo_body)),
            ("gen-3wise", _gen_cartesian(pre, A_EP, A_CARTESIAN, _a_forbidden, _a_combo_body)),
            ("gen-boundary", _gen_bounds(pre, "gen-boundary", A_EP, A_BASE, A_CTX, A_BOUNDS)),
            ("gen-content", _gen_content_anthropic()),
        ]
    raise ValueError(f"unknown vendor {vendor!r}")


def generate(vendor, budget=0):
    """All generated probes for a vendor, in budget-priority order."""
    probes = []
    for _name, chunk in _strategies(vendor):
        probes.extend(chunk)
    if budget and budget > 0:
        probes = probes[:budget]
    return probes


def strategy_counts(vendor, budget=0):
    counts = {}
    for p in generate(vendor, budget):
        counts[p["category"]] = counts.get(p["category"], 0) + 1
    return counts


# =============================================================================
# constraint checker: independent body-level invariants for the valid tiers
# =============================================================================
def _body_violations(vendor, probe):
    body = probe["body"]
    bad = []
    if not isinstance(body, dict):
        return [f"{probe['id']}: body not a dict"]
    if "stream_options" in body and body.get("stream") is not True:
        bad.append(f"{probe['id']}: stream_options without stream")
    tc = body.get("tool_choice")
    tools = body.get("tools")
    names = set()
    if isinstance(tools, list):
        for t in tools:
            if isinstance(t, dict) and "name" in t:
                names.add(t["name"])
    if tc is not None and not tools:
        bad.append(f"{probe['id']}: tool_choice without tools")
    if vendor == "openai":
        if isinstance(tc, dict):
            if tc.get("type") == "function" and tc.get("name") not in names:
                bad.append(f"{probe['id']}: named tool_choice not in tools")
            if tc.get("type") == "allowed_tools":
                for t in tc.get("tools", []):
                    if t.get("name") not in names:
                        bad.append(f"{probe['id']}: allowed_tools name not in tools")
        fmt = (body.get("text") or {}).get("format") or {}
        if fmt.get("type") == "json_object" and "JSON" not in json.dumps(body.get("input")):
            bad.append(f"{probe['id']}: json_object without JSON in input")
        inc = body.get("include") or []
        if "reasoning.encrypted_content" in inc and body.get("store") is not False:
            bad.append(f"{probe['id']}: encrypted_content include requires store=false")
    else:
        if isinstance(tc, dict) and tc.get("type") == "tool" and tc.get("name") not in names:
            bad.append(f"{probe['id']}: named tool_choice not in tools")
        think = body.get("thinking") or {}
        if think.get("type") == "enabled":
            if body["max_tokens"] <= think["budget_tokens"]:
                bad.append(f"{probe['id']}: max_tokens <= thinking budget")
            if body.get("temperature") not in (None, 1.0):
                bad.append(f"{probe['id']}: thinking with temperature != 1")
            if body["messages"][-1]["role"] != "user":
                bad.append(f"{probe['id']}: thinking with assistant prefill")
            if isinstance(tc, dict) and tc.get("type") in ("any", "tool"):
                bad.append(f"{probe['id']}: thinking with forced tool_choice")
    return bad


def check(vendor, probes):
    """Return problems: duplicate ids, unserializable bodies, invalid combos."""
    problems = []
    seen = set()
    for p in probes:
        if p["id"] in seen:
            problems.append(f"duplicate id: {p['id']}")
        seen.add(p["id"])
        try:
            json.dumps(p["body"])
        except (TypeError, ValueError) as e:
            problems.append(f"{p['id']}: unserializable body: {e}")
    for p in probes:
        if p["category"] in ("gen-pairwise", "gen-3wise", "gen-content"):
            problems.extend(_body_violations(vendor, p))
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generated probe tier inspector")
    ap.add_argument("--vendor", choices=["openai", "anthropic", "all"], default="all")
    ap.add_argument("--budget", type=int, default=0)
    args = ap.parse_args(argv)
    vendors = ["openai", "anthropic"] if args.vendor == "all" else [args.vendor]
    rc = 0
    for v in vendors:
        probes = generate(v, args.budget)
        counts = strategy_counts(v, args.budget)
        problems = check(v, probes)
        print(
            f"{v}: total={len(probes)} "
            + " ".join(f"{k.removeprefix('gen-')}={n}" for k, n in counts.items())
            + f" problems={len(problems)}"
        )
        for prob in problems[:20]:
            print(f"  PROBLEM: {prob}", file=sys.stderr)
        if problems:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
