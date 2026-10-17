# crates/symphony/tests/fixtures/turns/render.py
"""Render the turns a checkpoint's chat template writes for the shapes a tool-calling
conversation takes, as fixtures for `tests/rendered_turns.rs`.

For each table listed in TABLES the script renders, with jinja2 and the checkpoint's own chat
template, an assistant message of every shape (a plain answer, one call, several calls, a call
with no arguments, a call after content, string arguments that hold the closing marker, a bracket,
a newline, non-ASCII text, a call whose arguments are nested), takes the bytes the template writes
for the assistant turn, and writes one JSON line per turn: the table, the tail of the generation
prompt, the output, the request's tools, and the message the template rendered (content, each
call's name and its arguments as the request gave them, written as `json.dumps` writes them). The
test replays every turn through the table, whole and at every cut, and requires the message back.

The templates come from the Hugging Face hub, from a public mirror of the checkpoint where the
checkpoint itself is gated; each entry pins the file's SHA-256, so a rendering is reproducible, and
names the SHA-256 bellwether recorded for the checkpoint's own file where it has one, so the
mirror's template can be checked against the recorded one (the recorded hand-written set of the
checkpoint renders the same bytes for the same shapes, which the test's cases also hold).

Usage: render.py --cache DIR [--check]
  Downloads each template into DIR (kept across runs), renders the turns and writes the fixture
  files next to this script. With --check nothing is written: the fixtures are rendered again and
  compared with the files, and the script exits 1 on a difference. Run with Python 3.12 and
  jinja2 3.1.6; `python -I` keeps the downloaded files from shadowing a module.
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request

from jinja2 import Environment
from jinja2.ext import loopcontrols

HERE = os.path.dirname(os.path.abspath(__file__))

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "The weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string"},
                    "days": {"type": "integer"},
                    "detailed": {"type": "boolean"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "A quote",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "A note",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "meta": {"type": "object"},
                },
            },
        },
    },
]

# The assistant messages rendered: (name, content, [(function name, arguments)]). The names are
# bellwether's where its common set has the shape.
SHAPES = [
    ("content-plain", "Hello!", []),
    ("call-single", "", [("get_weather", {"city": "Paris"})]),
    ("call-empty-arguments", "", [("get_stock_price", {})]),
    (
        "calls-parallel-three",
        "",
        [
            ("get_weather", {"city": "Paris"}),
            ("get_weather", {"city": "Rome"}),
            ("get_stock_price", {"ticker": "MSFT"}),
        ],
    ),
    ("call-with-content", "Let me check.", [("get_weather", {"city": "Paris", "unit": "celsius"})]),
    ("call-typed-arguments", "", [("get_weather", {"city": "Paris", "days": 3, "detailed": True})]),
    (
        "call-unicode-arguments",
        "",
        [("save_note", {"title": "計画", "body": "今日は晴れ 🌤️ — très bien", "tags": ["日本語"]})],
    ),
    (
        "call-nested-arguments",
        "",
        [
            (
                "save_note",
                {
                    "title": "Plan",
                    "tags": ["a", "b"],
                    "meta": {"nested": {"deep": [1, 2.5, True, None]}},
                },
            )
        ],
    ),
    (
        "call-string-with-bracket",
        "",
        [("save_note", {"title": "Brackets", "body": "a [list] and a (tuple)"})],
    ),
    (
        "call-string-with-newline",
        "",
        [("save_note", {"title": "Lines", "body": "Line one\nLine two"})],
    ),
    (
        "call-string-with-closing-marker",
        "",
        [("save_note", {"title": "Marker", "body": "ends with )] here"})],
    ),
]

TABLES = [
    {
        "table": "llama4",
        # meta-llama/Llama-4-Scout-17B-16E-Instruct is gated; the mirror carries the same
        # chat_template.jinja but for one line that spells the tool definitions' concatenation.
        # bellwether recorded the checkpoint's own file (revision 92f3b1597a195b523d8d9e5700e57e4fbb8f20d3)
        # as fb76136a36a8422930f25fbe3326e9ae0d67e82259c7917b1b71d18ae6934cef; the mirror's renders
        # the same assistant turns, which the recorded common set shows.
        "url": "https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct/resolve/main/chat_template.jinja",
        "sha256": "93c6a232af67e2b3bc9218179ec3fea5176ec88b2a4a4fb66e51feb2ef63c0bd",
        "file": "llama-4-scout-17b-16e-instruct.chat_template.jinja",
        "tail": "<|header_start|>assistant<|header_end|>\n\n",
        "opener": "<|header_start|>assistant<|header_end|>",
        "eot": "<|eot|>",
        "bos": "<|begin_of_text|>",
        "skip": [],
        # The template quotes every value without escaping a quote inside it; a string argument
        # that holds one cannot be read back by any reader (docs/tables.md), so none is rendered.
    },
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(entry, cache):
    path = os.path.join(cache, entry["file"])
    if not os.path.exists(path):
        os.makedirs(cache, exist_ok=True)
        with urllib.request.urlopen(entry["url"], timeout=120) as r, open(path, "wb") as f:
            f.write(r.read())
    digest = sha256(path)
    if entry["sha256"] and digest != entry["sha256"]:
        sys.exit(f"{path}: sha256 {digest}, expected {entry['sha256']}")
    text = open(path, encoding="utf-8").read()
    if entry["file"].endswith(".json"):
        loaded = json.loads(text)
        text = loaded["chat_template"] if "chat_template" in loaded else loaded
    return text


def environment():
    # transformers renders with trim_blocks, lstrip_blocks and jinja2's default Undefined, which
    # a template may read as empty text (Llama 4's reads `tool_definition` before setting it).
    env = Environment(
        extensions=[loopcontrols], trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
    )
    env.filters["tojson"] = lambda value, indent=None: json.dumps(
        value, ensure_ascii=False, indent=indent
    )
    env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(ValueError(message))
    env.globals["strftime_now"] = lambda fmt: "26 Jul 2024"
    return env


def render(entry, template_text, cache):
    env = environment()
    template = env.from_string(template_text)
    lines = []
    for name, content, calls in SHAPES:
        if name in entry["skip"]:
            continue
        messages = [{"role": "user", "content": "Hi"}]
        assistant = {"role": "assistant", "content": content}
        if calls:
            assistant["tool_calls"] = [
                {
                    "id": f"call{index:05d}",
                    "type": "function",
                    "function": {"name": function, "arguments": arguments},
                }
                for index, (function, arguments) in enumerate(calls)
            ]
        prompt = template.render(
            messages=messages,
            tools=TOOLS,
            add_generation_prompt=True,
            bos_token=entry["bos"],
            eos_token=entry["eot"],
        )
        turn = template.render(
            messages=messages + [assistant],
            tools=TOOLS,
            add_generation_prompt=False,
            bos_token=entry["bos"],
            eos_token=entry["eot"],
        )
        if not turn.startswith(prompt):
            sys.exit(f"{entry['table']}/{name}: the turn does not continue the prompt")
        output = turn[len(prompt) :]
        if output.endswith(entry["eot"]):
            output = output[: -len(entry["eot"])]
        if not prompt.endswith(entry["tail"]):
            sys.exit(f"{entry['table']}/{name}: the prompt does not end with the tail")
        lines.append(
            json.dumps(
                {
                    "table": entry["table"],
                    "case": name,
                    "tail": entry["tail"],
                    "output": output,
                    "tools": TOOLS,
                    "content": content,
                    "calls": [
                        [function, json.dumps(arguments, ensure_ascii=False)]
                        for function, arguments in calls
                    ],
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    bad = 0
    for entry in TABLES:
        text = render(entry, fetch(entry, args.cache), args.cache)
        path = os.path.join(HERE, f"{entry['table']}.jsonl")
        if args.check:
            if not os.path.exists(path) or open(path, encoding="utf-8").read() != text:
                print(f"{path}: differs from the rendering", file=sys.stderr)
                bad += 1
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"wrote {path}: {text.count(chr(10))} turns")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
