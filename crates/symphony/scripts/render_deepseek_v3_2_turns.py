# crates/symphony/scripts/render_deepseek_v3_2_turns.py
"""Render the DeepSeek V3.2 reference encoder for the turn shapes a tool-calling conversation
takes, and write each assistant turn as the model writes it, with the message the encoder's own
parser reads back from it, to the fixture `tests/fixtures/rendered/deepseek_v3_2.json`, which
`tests/rendered_fixtures.rs` replays through the `deepseek_v3.2` table at every cut.

The checkpoint ships no chat template: `encoding/encoding_dsv32.py` of `deepseek-ai/DeepSeek-V3.2`
renders the prompt (`encode_messages`) and reads a completion (`parse_message_from_completion_text`).
DeepSeek-V3.2-Speciale ships the identical file, which `--also` checks byte for byte. The module is
imported from the path given, so the fixture is what the vendor's own code renders.

A turn is what the model writes after the generation prompt: the history rendered with the turn's
message minus the same history rendered without it (the encoder ends the prompt with `<think>` in
thinking mode and `</think>` in chat mode, after the last user message or after a tool result),
minus the end-of-sentence token, which the stop decoder hides from the parser. The message a turn
stands for is what the encoder's parser reads from the completion: its `reasoning_content`, its
`content`, and its `tool_calls`, with one difference the fixture spells out: the two newlines the
encoder writes before a calls block follow the content where there is some, and the table keeps
them as the content's (the parser looks for the block after them and drops them), so a turn with
content and calls has them at the end of its `content` here, and so has a turn after a tool
result in chat mode, whose content stay began in the prompt; alone at a stay's start, after the
thought or at the turn's start, they are the encoder's spacing, which the table drops as the
parser does. `reference_content` is the parser's own.

Usage: render_deepseek_v3_2_turns.py <encoding_dsv32.py> [--also <encoding_dsv32.py> ...] [--check]
  With --check nothing is written: the fixture is rendered again and compared byte for byte, and
  the script exits 1 on a difference. Run with Python 3.12; the encoder needs the standard library
  alone.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

OUT = Path(__file__).parents[1] / "tests" / "fixtures" / "rendered" / "deepseek_v3_2.json"
REPOSITORY = "deepseek-ai/DeepSeek-V3.2"
REVISION = "a7e62ac04ecb2c0a54d736dc46601c5606cf10a6"
ENCODER_SHA256 = "5e068c2ba2a6e5ebe37a49bb005650c507e7935d77a32f3f7c11ee071498b370"
ALSO = ["deepseek-ai/DeepSeek-V3.2-Speciale"]
EOS = "<｜end▁of▁sentence｜>"
BLOCK_LEAD = "\n\n"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "The weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "days": {"type": "integer"},
                    "metric": {"type": "boolean"},
                    "hours": {"type": "array", "items": {"type": "integer"}},
                    "options": {"type": "object"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a note.",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "noop",
            "description": "Nothing.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def call(name, arguments):
    """A call in the API's form, its arguments the JSON text the encoder loads."""
    return {
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def assistant(reasoning, content="", tool_calls=None):
    message = {"role": "assistant", "reasoning_content": reasoning, "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return message


SYSTEM = {"role": "system", "content": "You are a helpful assistant.", "tools": TOOLS}
USER = {"role": "user", "content": "Weather in Paris?"}
TOOL_RESULT = {"role": "tool", "content": '{"temperature": 18}'}
PARIS = call("get_weather", {"location": "Paris"})

# Each case: its name and the messages; the turn is the last message. In chat mode the encoder
# renders no reasoning, so the same messages give both modes' turns.
CASES = [
    (
        "answer",
        [
            SYSTEM,
            USER,
            assistant("The user asks about Paris; no tool is needed.", "It is sunny in Paris."),
        ],
    ),
    (
        "call",
        [
            SYSTEM,
            USER,
            assistant(
                "Need the weather.",
                tool_calls=[call("get_weather", {"location": "Paris", "days": 3})],
            ),
        ],
    ),
    (
        "calls-two",
        [
            SYSTEM,
            USER,
            assistant(
                "Two calls at once.", tool_calls=[PARIS, call("search", {"query": "Paris weather"})]
            ),
        ],
    ),
    (
        "call-after-content",
        [SYSTEM, USER, assistant("Say something first.", "Let me check.", [PARIS])],
    ),
    (
        "call-empty-arguments",
        [SYSTEM, USER, assistant("A call with nothing in it.", tool_calls=[call("noop", {})])],
    ),
    (
        "call-every-value-kind",
        [
            SYSTEM,
            USER,
            assistant(
                "Every kind of value.",
                tool_calls=[
                    call(
                        "get_weather",
                        {
                            "location": "Paris",
                            "days": 3,
                            "metric": True,
                            "hours": [9, 18],
                            "options": {"lang": "fr", "n": 2},
                        },
                    )
                ],
            ),
        ],
    ),
    (
        "call-unicode",
        [
            SYSTEM,
            USER,
            assistant(
                "A note.", tool_calls=[call("save_note", {"title": "計画 🌍", "body": "café"})]
            ),
        ],
    ),
    (
        "answer-after-tool-result",
        [
            SYSTEM,
            USER,
            assistant("Need the weather.", tool_calls=[PARIS]),
            TOOL_RESULT,
            assistant("Got it.", "It is 18 degrees in Paris."),
        ],
    ),
    (
        "call-after-tool-result",
        [
            SYSTEM,
            USER,
            assistant("Need the weather.", tool_calls=[PARIS]),
            TOOL_RESULT,
            assistant("Rome too.", tool_calls=[call("get_weather", {"location": "Rome"})]),
        ],
    ),
]


def load(path):
    text = Path(path).read_bytes()
    digest = hashlib.sha256(text).hexdigest()
    spec = importlib.util.spec_from_file_location("encoding_dsv32", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, digest


def render_case(encoder, name, messages, mode):
    prompt = encoder.encode_messages(messages[:-1], mode)
    full = encoder.encode_messages(messages, mode)
    assert full.startswith(prompt), f"{name}/{mode}: the history does not start with the prompt"
    completion = full[len(prompt) :]
    assert completion.endswith(EOS), (
        f"{name}/{mode}: the turn does not end with the end of sentence"
    )
    read = encoder.parse_message_from_completion_text(completion, mode)
    output = completion[: -len(EOS)]
    calls = [
        {"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])}
        for c in read["tool_calls"]
    ]
    # The two newlines before a block are the content's where the content stay already holds
    # text: the turn's own content, or, after a tool result in chat mode, the prompt's text since
    # the previous block closed (the replay walks it). Alone at the start of a stay they are the
    # encoder's spacing, which the table drops as the parser does.
    after_result = any(m["role"] == "tool" for m in messages)
    stay_has_text = bool(read["content"]) or (mode == "chat" and after_result)
    content = read["content"] + (BLOCK_LEAD if calls and stay_has_text else "")
    assert output.startswith(read["reasoning_content"]) or not read["reasoning_content"]
    tail = prompt[prompt.rfind("<｜User｜>") :]
    return {
        "name": f"{name}/{mode}",
        "thinking_mode": mode,
        "message": messages[-1],
        "tail": tail,
        "output": output,
        "reasoning": read["reasoning_content"],
        "content": content,
        "reference_content": read["content"],
        "calls": calls,
    }


def render_all(encoder):
    return [
        render_case(encoder, name, messages, mode)
        for name, messages in CASES
        for mode in ("thinking", "chat")
    ]


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    check = "--check" in argv
    also = []
    if "--also" in argv:
        also = args[1:]
        args = args[:1]
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    encoder, digest = load(args[0])
    assert digest == ENCODER_SHA256, f"{args[0]}: sha256 {digest}, not the recorded encoder's"
    turns = render_all(encoder)
    for other in also:
        _, other_digest = load(other)
        assert other_digest == ENCODER_SHA256, f"{other}: another encoder (sha256 {other_digest})"
    fixture = {
        "table": "deepseek_v3.2",
        "source": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "file": "encoding/encoding_dsv32.py",
            "sha256": ENCODER_SHA256,
            "also": ALSO,
            "rendered_as": "the encoder's encode_messages for the prompt and the history, its "
            "parse_message_from_completion_text for the message read back",
            "script": "scripts/render_deepseek_v3_2_turns.py",
        },
        "notes": [
            "A turn is the history rendered with the turn's message minus the generation prompt, "
            "minus the end-of-sentence token, which the stop decoder hides from the parser.",
            "The encoder ends the prompt with <think> in thinking mode and </think> in chat mode, "
            "after the last user message and again after a tool result (inside the same "
            "assistant turn, after </function_results>).",
            "The two newlines the encoder writes before a calls block are the content's where the "
            "content stay already holds text (the turn's own content, or the prompt's text after "
            "a tool result in chat mode, which the replay walks); alone at a stay's start they are "
            "the encoder's spacing, dropped. The encoder's parser looks for the block after them "
            "either way. `content` is the table's reading and `reference_content` the parser's.",
            "The tail is the prompt from its last user message; the replay walks it from the last "
            "<｜Assistant｜>.",
        ],
        "tools": TOOLS,
        "turns": turns,
    }
    rendered = json.dumps(fixture, ensure_ascii=False, indent=1) + "\n"
    if check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else None
        if current != rendered:
            print(f"{OUT}: differs from what the encoder renders", file=sys.stderr)
            return 1
        print(f"{OUT}: as rendered ({len(turns)} turns)")
        return 0
    OUT.write_text(rendered, encoding="utf-8")
    print(f"{OUT}: {len(turns)} turns written")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
