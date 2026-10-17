# crates/symphony/scripts/render_kimi_k2_turns.py
"""Render the Kimi K2 line's chat templates for the turn shapes a tool-calling conversation takes,
and write each assistant turn as the model writes it, with the message it stands for, to the
fixture `tests/fixtures/rendered/kimi_k2.json`, which `tests/rendered_fixtures.rs` replays through
the `kimi_k2` table at every cut.

Six templates, the `chat_template.jinja` of each checkpoint the corpus has a manifest for, at the
manifest's revision: Kimi-K2-Instruct and Kimi-K2-Instruct-0905 (no thought; the two files differ
in a block's indentation alone and render the same turns), Kimi-K2-Thinking (the model writes its
own `<think>`), Kimi-K2.5 and Kimi-K2.6 (the generation prompt opens the thought, `<think>`, or
closes it at once, `<think></think>`, when `thinking` is false) and Kimi-K2.7-Code (the prompt
opens it, with no switch). Each is rendered as transformers renders a chat template: a Jinja
environment with `trim_blocks`, `lstrip_blocks` and the `loopcontrols` extension, `tojson` as
`json.dumps(..., ensure_ascii=False)` with the separators the template asks for, and
`raise_exception`. A call's arguments are given as an object, which is how a serving layer hands
them to the template (the API carries them as a string, which the template writes as it is); its
id is the line's own, `functions.NAME:N`, the call's place in the turn after the colon.

A turn is what the model writes after the generation prompt: the history rendered with the turn's
message, minus the same history rendered without it and with the generation prompt, minus the
`<|im_end|>` that ends the turn, which the stop decoder hides from the parser. The message a turn
stands for is the one given to the template: its `reasoning_content` is the thought, its
`content` the answer, its `tool_calls` the calls, read back as names and arguments. The templates
that keep a thought only on a message that calls (K2 Thinking, K2.5, K2.6 without
`preserve_thinking`) render the thought-and-answer shape as an empty thought and the answer, so
that shape is rendered where the template keeps it: K2.7-Code, and K2.6 with `preserve_thinking`.
A turn two templates render alike is written once, with both named.

Usage: render_kimi_k2_turns.py <directory> [--check]
  The directory holds `<slug>.jinja` for each of the six slugs below. With --check nothing is
  written: the fixture is rendered again and compared byte for byte, and the script exits 1 on a
  difference. Run with Python 3.12 and Jinja2 3.1.
"""

import hashlib
import json
import sys
from pathlib import Path

from jinja2 import Environment
from jinja2.ext import loopcontrols

OUT = Path(__file__).parents[1] / "tests" / "fixtures" / "rendered" / "kimi_k2.json"
TURN_OPENER = "<|im_assistant|>assistant<|im_middle|>"
STOP = "<|im_end|>"
USER_OPENER = "<|im_user|>"
TOOL_OPENER = "<|im_system|>tool<|im_middle|>"

# The six templates: slug, repository, revision, the file's sha256, and the shapes each renders
# (see SHAPES), with the keyword arguments the rendering passes.
TEMPLATES = [
    {
        "slug": "kimi-k2-instruct",
        "repository": "moonshotai/Kimi-K2-Instruct",
        "revision": "fd1984e2b7a3350dbf7305fe73a4ede25c14de50",
        "sha256": "39e8c195b474c1a4148046f41fedd177b2ff8f420fa376e52fcdddb829e8f0c5",
        "renders": [("plain", {})],
    },
    {
        "slug": "kimi-k2-instruct-0905",
        "repository": "moonshotai/Kimi-K2-Instruct-0905",
        "revision": "ac6c49f04883bd0a0598b790693a72061c676629",
        "sha256": "00938c355ca7195fd4f9c923e93af063846d4d5005e06999274f8db57b65bb0b",
        "renders": [("plain", {})],
    },
    {
        "slug": "kimi-k2-thinking",
        "repository": "moonshotai/Kimi-K2-Thinking",
        "revision": "a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55",
        "sha256": "07a791b8563c5159054394745049c8e487986ecb34aaab77727e1b158bbe6a5c",
        "renders": [("plain", {}), ("thinking", {})],
    },
    {
        "slug": "kimi-k2.5",
        "repository": "moonshotai/Kimi-K2.5",
        "revision": "4d01dfe0332d63057c186e0b262165819efb6611",
        "sha256": "7c91e4b3578b518bfb388d2e3dc645daf0fbf843656e887db75d12891e0030ff",
        "renders": [("plain", {}), ("thinking", {}), ("thinking-off", {"thinking": False})],
    },
    {
        "slug": "kimi-k2.6",
        "repository": "moonshotai/Kimi-K2.6",
        "revision": "7eb5002f6aadc958aed6a9177b7ed26bb94011bb",
        "sha256": "8bf859698fd4781c0e1e1c63ce74422aab27e53ccc5f47116317c64cda06132f",
        "renders": [
            ("plain", {}),
            ("thinking", {}),
            ("thinking-off", {"thinking": False}),
            ("thought-kept", {"preserve_thinking": True}),
        ],
    },
    {
        "slug": "kimi-k2.7-code",
        "repository": "moonshotai/Kimi-K2.7-Code",
        "revision": "74797c9c62378b951a1f6fcf5c4631024e9b8bef",
        "sha256": "8d21200dff7e85beaf51c8b110952e039ffd0779f615d64d63293d74773360f9",
        "renders": [("plain", {}), ("thinking", {}), ("thought-kept", {})],
    },
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "The weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "A stock's price.",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a note.",
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


def call(name, arguments, index=0):
    return {
        "id": f"functions.{name}:{index}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def assistant(content=None, thought=None, tool_calls=None):
    message = {"role": "assistant"}
    if content is not None:
        message["content"] = content
    if thought is not None:
        message["reasoning_content"] = thought
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return message


USER = {"role": "user", "content": "Weather in Paris?"}
PARIS = call("get_weather", {"city": "Paris"})
ROME = call("get_weather", {"city": "Rome", "days": 3}, 1)
TOOL_RESULT = {"role": "tool", "tool_call_id": PARIS["id"], "content": '{"temperature": 18}'}

# The shapes, by the kind of rendering that writes them: `plain` for every template; `thinking`
# where the family thinks (the thought stands on a message that calls); `thinking-off` for a
# template with a switch; `thought-kept` where the template keeps the thought of an answer.
SHAPES = {
    "plain": [
        ("answer", [USER, assistant(content="It is sunny in Paris.")]),
        ("call", [USER, assistant(tool_calls=[PARIS])]),
        ("calls-two", [USER, assistant(tool_calls=[PARIS, ROME])]),
        ("content-call", [USER, assistant(content="Let me check.", tool_calls=[PARIS])]),
        (
            "call-empty-arguments",
            [USER, assistant(tool_calls=[call("get_stock_price", {})])],
        ),
        (
            "call-nested-unicode",
            [
                USER,
                assistant(
                    tool_calls=[
                        call(
                            "save_note",
                            {
                                "title": "計画 🌍",
                                "body": 'a "q" \\ b\nc',
                                "tags": ["a", "b"],
                                "meta": {"nested": {"deep": [1, 2.5, True, None]}},
                            },
                        )
                    ]
                ),
            ],
        ),
        (
            "call-brace-in-string",
            [
                USER,
                assistant(tool_calls=[call("save_note", {"title": "Braces", "body": "a } b { c"})]),
            ],
        ),
        (
            "answer-after-tool-result",
            [
                USER,
                assistant(tool_calls=[PARIS]),
                TOOL_RESULT,
                assistant(content="It is 18 degrees in Paris."),
            ],
        ),
        (
            "call-after-tool-result",
            [
                USER,
                assistant(tool_calls=[PARIS]),
                TOOL_RESULT,
                assistant(tool_calls=[call("get_weather", {"city": "Rome"})]),
            ],
        ),
    ],
    "thinking": [
        ("thought-call", [USER, assistant(thought="Need the weather.", tool_calls=[PARIS])]),
        (
            "thought-content-calls",
            [
                USER,
                assistant(
                    thought="Paris and Rome.", content="Checking both.", tool_calls=[PARIS, ROME]
                ),
            ],
        ),
        (
            "thought-call-after-tool-result",
            [
                USER,
                assistant(tool_calls=[PARIS]),
                TOOL_RESULT,
                assistant(thought="Rome too.", tool_calls=[call("get_weather", {"city": "Rome"})]),
            ],
        ),
    ],
    "thinking-off": [
        ("answer-thinking-off", [USER, assistant(content="It is sunny in Paris.")]),
        ("call-thinking-off", [USER, assistant(tool_calls=[PARIS])]),
    ],
    "thought-kept": [
        (
            "thought-answer",
            [
                USER,
                assistant(
                    thought="The user asks about Paris; no tool is needed.",
                    content="It is sunny in Paris.",
                ),
            ],
        ),
    ],
}


def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    """transformers' `tojson`: `json.dumps` with these defaults, no HTML escaping."""
    return json.dumps(
        value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
    )


def raise_exception(message):
    raise ValueError(message)


def environment(template_text):
    env = Environment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template_text)


def render_case(template, slug, kind, kwargs, name, messages):
    start = len(messages) - 1
    assert messages[start]["role"] == "assistant", f"{name}: the turn is one assistant message"
    prompt = template.render(
        messages=messages[:start], tools=TOOLS, add_generation_prompt=True, **kwargs
    )
    full = template.render(messages=messages, tools=TOOLS, add_generation_prompt=False, **kwargs)
    assert TURN_OPENER in prompt, f"{slug}/{name}: no turn opener in the prompt"
    assert full.startswith(prompt), f"{slug}/{name}: the history does not start with the prompt"
    output = full[len(prompt) :]
    assert output.endswith(STOP), f"{slug}/{name}: the turn ends with no stop: {output[-40:]!r}"
    output = output[: -len(STOP)]
    turn = messages[start]
    reasoning = turn.get("reasoning_content", "")
    calls = [
        {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
        for c in turn.get("tool_calls", [])
    ]
    tail = prompt[max(prompt.rfind(USER_OPENER), prompt.rfind(TOOL_OPENER)) :]
    return {
        "name": f"{slug}/{name}",
        "rendered_by": [slug],
        "kind": kind,
        "kwargs": kwargs,
        "messages": [turn],
        "tail": tail,
        "output": output,
        "reasoning": reasoning,
        "content": turn.get("content", ""),
        "calls": calls,
    }


def render_all(directory):
    turns = []
    by_text = {}
    for entry in TEMPLATES:
        path = directory / f"{entry['slug']}.jinja"
        text = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert digest == entry["sha256"], f"{path}: sha256 {digest}, not the recorded template's"
        template = environment(text)
        for kind, kwargs in entry["renders"]:
            for name, messages in SHAPES[kind]:
                turn = render_case(template, entry["slug"], kind, kwargs, name, messages)
                key = (turn["tail"], turn["output"], turn["reasoning"], turn["content"])
                key += (json.dumps(turn["calls"], ensure_ascii=False, sort_keys=True),)
                if key in by_text:
                    by_text[key]["rendered_by"].append(entry["slug"])
                    continue
                by_text[key] = turn
                turns.append(turn)
    return turns


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    check = "--check" in argv
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    turns = render_all(Path(args[0]))
    fixture = {
        "table": "kimi_k2",
        "sources": [
            {k: entry[k] for k in ("slug", "repository", "revision", "sha256")}
            | {"file": "chat_template.jinja"}
            for entry in TEMPLATES
        ],
        "rendered_as": "transformers renders a chat template: trim_blocks, lstrip_blocks, "
        "loopcontrols, tojson = json.dumps(ensure_ascii=False) with the template's separators",
        "script": "scripts/render_kimi_k2_turns.py",
        "notes": [
            "A turn is the history rendered with the turn's message minus the generation prompt, "
            "minus the <|im_end|> that ends it, which the stop decoder hides from the parser.",
            "K2 Instruct and Instruct-0905 write no thought. K2 Thinking's prompt ends at the turn "
            "opener and the model writes its own <think>. K2.5's and K2.6's prompt ends with "
            "<think>, or with <think></think> when thinking is false; K2.7-Code's with <think>, "
            "with no switch. The tail carries the prompt's ending, so the replay starts where the "
            "template leaves the model.",
            "The templates that render <think></think> for an answer in the history (K2 Thinking, "
            "K2.5, K2.6 without preserve_thinking) keep a thought on a message that calls alone; "
            "the thought-and-answer shape is rendered where the template keeps it (K2.7-Code, "
            "K2.6 with preserve_thinking).",
            "A call's id is the line's own, functions.NAME:N; the name is read from it. The "
            "arguments are given to the template as an object and written with tojson.",
            "A turn two templates render alike is written once, with every template that renders "
            "it under rendered_by; the tail is the prompt from its last user or tool message.",
        ],
        "tools": TOOLS,
        "turns": turns,
    }
    rendered = json.dumps(fixture, ensure_ascii=False, indent=1) + "\n"
    if check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else None
        if current != rendered:
            print(f"{OUT}: differs from what the templates render", file=sys.stderr)
            return 1
        print(f"{OUT}: as rendered ({len(turns)} turns)")
        return 0
    OUT.write_text(rendered, encoding="utf-8")
    print(f"{OUT}: {len(turns)} turns written")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
