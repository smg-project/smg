# crates/tokenizer/scripts/generate_deepseek_v41_fixtures.py
"""Record DeepSeek-V4.1 render fixtures from the reference encoders.

Oracle `hf` (the default): oracle/deepseek_v41_encoding.py (HF encoding.py @
517ef625df, the revision the serving engines ported). The effort tiers are
overridden to the engine table (spec D1). Token ids come from the real
tokenizer.json.

Oracle `hf` at the current revision: cases that pass `encoder_revision` render
with the checkpoint's own `encoding/encoding.py` beside the tests directory,
at the revision the third argument names; the fixture records that revision
and its effort table (50/75/100 since the refresh of 2026-09, the table the
chat template the checkpoint added later carries too) next to the recorded
ones. Without the third argument those cases keep their recorded text.

Oracle `vllm_python`: the serving engine's port of that encoder, imported from the clone at
VLLM_REPO_DIR (never vendored) and named in the fixture with the clone's HEAD
commit. It records the shapes the HF encoder cannot render, today a developer
message it keeps (the serving engines render it as a user turn; the HF encoder has
no developer branch). Without VLLM_REPO_DIR those cases keep their recorded
text and a warning says so. Its cases take string content only: the engine flattens
content parts in its tokenizer wrapper, not in the encoder.

Usage: generate_deepseek_v41_fixtures.py <checkpoint>/encoding/tests <checkpoint>/tokenizer.json [<checkpoint revision>]
"""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "oracle"))
import deepseek_v41_encoding as enc  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

ENGINE_TIERS = {"low": 25, "high": 50, "xhigh": 75, "max": 100}
enc.REASONING_EFFORT_MAPPINGS = ENGINE_TIERS
enc.DEFAULT_REASONING_EFFORT = "high"

HF_TESTS = Path(sys.argv[1])  # .../DeepSeek-V4.1-Flash/encoding/tests
TOKENIZER = Path(sys.argv[2])  # .../DeepSeek-V4.1-Flash/tokenizer.json
# The checkpoint directory's revision, for the cases its own encoder records.
CURRENT_REVISION = sys.argv[3] if len(sys.argv) > 3 else None
CURRENT_SOURCE = HF_TESTS.parent / "encoding.py"
OUT = Path(__file__).parents[1] / "tests" / "fixtures" / "deepseek_v41"
RENDER_FIXTURES = OUT / "render_fixtures.json"

# Repo-relative path of the engine's encoder inside the clone at VLLM_REPO_DIR.
VLLM_SOURCE = "vllm/tokenizers/deepseek_v41_encoding.py"

# The content-part spellings the engine's `_normalize_messages` accepts. The reference
# encoder knows only `text` and `image_url`/`image`; a part in any other spelling
# would silently vanish from its prompt.
TEXT_PART_TYPES = ("text", "input_text", "output_text")
IMAGE_PART_TYPES = ("image_url", "input_image", "image_pil")


def read_head_commit(repo):
    """The clone's HEAD commit, read from its .git files rather than a git
    subprocess: a plain clone keeps HEAD and refs under .git/, a linked
    worktree points at them through `gitdir:` and `commondir`."""
    git_dir = repo / ".git"
    if git_dir.is_file():
        git_dir = (repo / git_dir.read_text().split(":", 1)[1].strip()).resolve()
    head = (git_dir / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    common = git_dir / "commondir"
    ref_dir = (git_dir / common.read_text().strip()).resolve() if common.exists() else git_dir
    loose = ref_dir / ref
    if loose.exists():
        return loose.read_text().strip()
    for line in (ref_dir / "packed-refs").read_text().splitlines():
        if line.endswith(" " + ref):
            return line.split()[0]
    raise ValueError(f"cannot resolve {ref} under {ref_dir}")


def load_vllm_encoder():
    """The engine's encoder module and the clone's HEAD commit, or (None, None) with
    a warning when VLLM_REPO_DIR is unset. Its effort table must still be the
    first published one, which its cases are recorded under: the Rust renderer
    reads the table from the checkpoint revision it serves, so a change in the
    engine's port is a new revision to record, not something to record quietly."""
    repo = os.environ.get("VLLM_REPO_DIR")
    if not repo:
        print(
            "warning: VLLM_REPO_DIR unset; vllm_python cases keep their recorded text",
            file=sys.stderr,
        )
        return None, None
    repo = Path(repo)
    spec = importlib.util.spec_from_file_location("vllm_deepseek_v41_encoding", repo / VLLM_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (
        module.REASONING_EFFORT_MAPPINGS != ENGINE_TIERS
        or module.DEFAULT_REASONING_EFFORT != "high"
    ):
        raise ValueError(
            f"{VLLM_SOURCE} no longer uses the effort table {ENGINE_TIERS} with default 'high'; "
            "record its cases under the revision it follows now (see EffortTable in the Rust "
            "renderer) before regenerating"
        )
    return module, read_head_commit(repo)


def load_current_encoder():
    """The checkpoint's own encoder (`encoding/encoding.py` beside the tests
    directory), or None with a warning when no revision was given for it. Its
    effort table is recorded as it is: the Rust renderer reads the table from
    the checkpoint, so a change there is a new revision to record, not a
    decision."""
    if CURRENT_REVISION is None:
        print(
            "warning: no checkpoint revision given; current-revision cases keep their recorded text",
            file=sys.stderr,
        )
        return None
    spec = importlib.util.spec_from_file_location(
        "hf_deepseek_v41_encoding_current", CURRENT_SOURCE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VLLM, VLLM_REVISION = load_vllm_encoder()
CURRENT = load_current_encoder()
CURRENT_TIERS = None if CURRENT is None else CURRENT.REASONING_EFFORT_MAPPINGS
RECORDED = {}
RECORDED_CURRENT = {}
if (VLLM is None or CURRENT is None) and RENDER_FIXTURES.exists():
    previous = json.loads(RENDER_FIXTURES.read_text())
    if VLLM is None:
        RECORDED = {c["name"]: c["text"] for c in previous["cases"] if c["oracle"] == "vllm_python"}
        VLLM_REVISION = previous["vllm_revision"]
    if CURRENT is None:
        RECORDED_CURRENT = {
            c["name"]: c["text"] for c in previous["cases"] if "encoder_revision" in c
        }
        CURRENT_REVISION = previous["current_revision"]
        CURRENT_TIERS = previous["current_effort_tiers"]


def load_hf_case(path):
    """HF fixture inputs are either a dict with thinking_mode/tools/
    reasoning_effort/messages keys, or a bare JSON list that IS the messages
    (with any tools already embedded on the relevant message)."""
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return {"messages": data}
    return data


def oracle_part(part):
    """One content part in the spelling the reference encoder understands.

    `input_text`/`output_text` become `text`; `input_image`/`image_pil` become
    `image_url`. No payload reaches the prompt, but the oracle insists on an
    image source: `input_image` carries its URL as a string, and `image_pil`
    holds a PIL object in the engine, which JSON cannot hold, so the oracle gets the
    spelling itself as a stand-in. The oracle then substitutes the image
    placeholder and joins the parts, so it stays the byte authority for both.
    The fixture records the request as sent, spellings included: the renderer
    normalises them itself."""
    kind = part.get("type")
    if kind in TEXT_PART_TYPES:
        return {**part, "type": "text"}
    if kind in IMAGE_PART_TYPES and kind != "image_url":
        return {"type": "image_url", "image_url": part.get("image_url", kind)}
    return part


def oracle_messages(messages):
    msgs = json.loads(json.dumps(messages))
    for msg in msgs:
        if isinstance(msg.get("content"), list):
            msg["content"] = [oracle_part(part) for part in msg["content"]]
    return msgs


def attach_tools(msgs, tools):
    """Rule D4, as `inject_tools_into_first_system_message` (huggingface.rs)
    and the engine's `apply_chat_template` do it: the request's tools land on the
    FIRST system message wherever it sits; without one, an empty system
    message is inserted at index 0 and carries them."""
    system = next((msg for msg in msgs if msg["role"] == "system"), None)
    if system is None:
        system = {"role": "system", "content": ""}
        msgs.insert(0, system)
    system["tools"] = tools


def encode(oracle, name, msgs, thinking_mode, drop_thinking, reasoning_effort, current=False):
    if oracle == "hf" and current:
        if CURRENT is None:
            if name not in RECORDED_CURRENT:
                raise SystemExit(f"{name}: never recorded and no checkpoint revision was given")
            return RECORDED_CURRENT[name]
        module = CURRENT
    elif oracle == "hf":
        module = enc
    elif oracle == "vllm_python":
        if VLLM is None:
            if name not in RECORDED:
                raise SystemExit(f"{name}: never recorded and VLLM_REPO_DIR is unset")
            return RECORDED[name]
        module = VLLM
    else:
        raise ValueError(f"{name}: unknown oracle {oracle!r}")
    return module.encode_messages(
        msgs,
        thinking_mode=thinking_mode,
        drop_thinking=drop_thinking,
        reasoning_effort=reasoning_effort,
    )


def case(
    name,
    messages,
    tools=None,
    thinking_mode="chat",
    reasoning_effort=None,
    drop_thinking=True,
    continue_final_message=False,
    oracle="hf",
    current=False,
):
    """One case; `current` records it with the checkpoint's own encoder at the
    current revision (oracle `hf` only) and names that revision on the case."""
    msgs = oracle_messages(messages)
    if tools:
        attach_tools(msgs, tools)
    if continue_final_message:
        msgs[-1]["wo_eos"] = True
    text = encode(oracle, name, msgs, thinking_mode, drop_thinking, reasoning_effort, current)
    revision = {"encoder_revision": CURRENT_REVISION} if current else {}
    return {
        "name": name,
        "oracle": oracle,
        **revision,
        "messages": messages,
        "tools": tools,
        "thinking_mode": thinking_mode,
        "reasoning_effort": reasoning_effort,
        "drop_thinking": drop_thinking,
        "continue_final_message": continue_final_message,
        "text": text,
    }


def tool_call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def tool_result(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


LOOKUP_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]
F_TOOL = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
IMAGE_URL_PART = {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
DEVELOPER_THEN_USER = [
    {"role": "developer", "content": "Follow the policy."},
    {"role": "user", "content": "Question?"},
]

cases = []
for n in range(1, 6):
    inp = load_hf_case(HF_TESTS / f"test_input_{n}.json")
    cases.append(
        case(
            f"hf_{n}",
            inp["messages"],
            inp.get("tools"),
            inp.get("thinking_mode", "chat"),
            inp.get("reasoning_effort"),
        )
    )
cases += [
    case("chat_no_system", [{"role": "user", "content": "q"}]),
    case("thinking_default_effort", [{"role": "user", "content": "q"}], thinking_mode="thinking"),
    case(
        "thinking_low",
        [{"role": "user", "content": "q"}],
        thinking_mode="thinking",
        reasoning_effort="low",
    ),
    case(
        "thinking_int_42",
        [{"role": "user", "content": "q"}],
        thinking_mode="thinking",
        reasoning_effort=42,
    ),
    case(
        "thinking_history_dropped",
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "reasoning_content": "r1", "content": "a1"},
            {"role": "user", "content": "q2"},
        ],
        thinking_mode="thinking",
    ),
    case(
        "thinking_history_kept",
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "reasoning_content": "r1", "content": "a1"},
            {"role": "user", "content": "q2"},
        ],
        thinking_mode="thinking",
        drop_thinking=False,
    ),
    case(
        "mid_system",
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "system", "content": "MID"},
            {"role": "user", "content": "q2"},
        ],
        thinking_mode="thinking",
    ),
    case(
        "two_images",
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                    {"type": "text", "text": "second"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                    {"type": "text", "text": "compare"},
                ],
            }
        ],
    ),
    case(
        "tool_results_sorted",
        [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "reasoning_content": "reason",
                "content": "summary",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"query": "first"}'},
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"query": "second"}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_b", "content": "second result"},
            {"role": "tool", "tool_call_id": "call_a", "content": "first result"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ],
        thinking_mode="thinking",
    ),
    case(
        "continue_final_message",
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "reasoning_content": "r", "content": "Sure,"},
        ],
        thinking_mode="thinking",
        continue_final_message=True,
    ),
    case(
        "assistant_tool_call_false_string_args",
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "f",
                            "arguments": '{"n": 3, "ok": true, "obj": {"temp": "C"}, "s": "raw <x>"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "done"},
        ],
        tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        thinking_mode="thinking",
    ),
    # A developer message before the last user turn is dropped by drop_thinking,
    # which is the only developer shape the HF encoder can render; the kept
    # shapes are recorded from the engine's port at the end of this list.
    case("developer_then_user_thinking", DEVELOPER_THEN_USER, thinking_mode="thinking"),
    case(
        "two_user_turns",
        [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}],
        thinking_mode="thinking",
    ),
    case("image_only_part", [{"role": "user", "content": [IMAGE_URL_PART]}]),
    # The "\n\n" join keeps the empty text part.
    case(
        "empty_text_and_image_parts",
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ""},
                    IMAGE_URL_PART,
                    {"type": "text", "text": "describe"},
                ],
            }
        ],
    ),
    # Engine-only spellings (see oracle_part). `image_pil` carries a PIL object in
    # the engine, which JSON cannot hold; no image payload reaches the prompt anyway.
    case(
        "input_image_and_image_pil_spellings",
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "https://example.com/a.png"},
                    {"type": "text", "text": "which is brighter?"},
                    {"type": "image_pil"},
                ],
            }
        ],
    ),
    # A result whose id matches no call sorts with key 0, stably.
    case(
        "tool_results_unknown_id",
        [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "reasoning_content": "reason",
                "content": "summary",
                "tool_calls": [
                    tool_call("call_a", "lookup", '{"query": "first"}'),
                    tool_call("call_b", "lookup", '{"query": "second"}'),
                ],
            },
            tool_result("call_b", "second result"),
            tool_result("call_x", "unknown result"),
            tool_result("call_a", "first result"),
        ],
        tools=LOOKUP_TOOL,
        thinking_mode="thinking",
    ),
    # A trailing assistant turn that is NOT continued: EOS, no generation header.
    case(
        "thinking_trailing_assistant",
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "reasoning_content": "r", "content": "Sure,"},
        ],
        thinking_mode="thinking",
    ),
    # Non-object arguments: the raw string is wrapped as one string parameter.
    case(
        "assistant_tool_call_array_string_args",
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [tool_call("c1", "f", "[1, 2]")]},
            tool_result("c1", "done"),
        ],
        tools=F_TOOL,
        thinking_mode="thinking",
    ),
    # A double-encoded object is parsed twice and renders its keys.
    case(
        "assistant_tool_call_double_encoded_args",
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call("c1", "f", json.dumps(json.dumps({"a": 1})))],
            },
            tool_result("c1", "done"),
        ],
        tools=F_TOOL,
        thinking_mode="thinking",
    ),
    # D4: the tools attach to the mid-conversation system message, which the
    # header rule also counts as the last user turn.
    case(
        "mid_system_with_tools",
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "system", "content": "Mid-conversation update."},
            {"role": "user", "content": "q2"},
        ],
        tools=LOOKUP_TOOL,
        thinking_mode="thinking",
    ),
    case(
        "response_format_on_system",
        [
            {
                "role": "system",
                "content": "Answer in JSON.",
                "response_format": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
            {"role": "user", "content": "q"},
        ],
        thinking_mode="thinking",
    ),
    # A kept developer message (chat mode; thinking with tools, where
    # drop_thinking is off) renders as a <｜User｜> turn in the serving engines;
    # the HF encoder raises on it, so the engine's port records these two. With
    # tools, D4 puts them on a synthesised leading system message.
    case("developer_then_user_chat", DEVELOPER_THEN_USER, oracle="vllm_python"),
    case(
        "developer_with_tools",
        DEVELOPER_THEN_USER,
        tools=LOOKUP_TOOL,
        thinking_mode="thinking",
        oracle="vllm_python",
    ),
    # The same requests as thinking_default_effort, thinking_low and
    # continue_final_message under the current revision's effort table (the
    # gateway's pipeline test renders the continued message through it); last,
    # so the recorded cases keep their positions in both files.
    case(
        "thinking_default_effort_current",
        [{"role": "user", "content": "q"}],
        thinking_mode="thinking",
        current=True,
    ),
    case(
        "thinking_low_current",
        [{"role": "user", "content": "q"}],
        thinking_mode="thinking",
        reasoning_effort="low",
        current=True,
    ),
    case(
        "continue_final_message_current",
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "reasoning_content": "r", "content": "Sure,"},
        ],
        thinking_mode="thinking",
        continue_final_message=True,
        current=True,
    ),
]

tok = Tokenizer.from_file(str(TOKENIZER))
sha = hashlib.sha256(TOKENIZER.read_bytes()).hexdigest()
OUT.mkdir(parents=True, exist_ok=True)
RENDER_FIXTURES.write_text(
    json.dumps(
        {
            "source": "deepseek-ai/DeepSeek-V4.1-Flash encoding/encoding.py",
            "revision": "517ef625df",
            "effort_tiers": ENGINE_TIERS,
            "current_revision": CURRENT_REVISION,
            "current_effort_tiers": CURRENT_TIERS,
            "vllm_source": VLLM_SOURCE,
            "vllm_revision": VLLM_REVISION,
            "cases": cases,
        },
        ensure_ascii=False,
        indent=1,
    )
    + "\n"
)
(OUT / "render_ids_fixtures.json").write_text(
    json.dumps(
        {
            "tokenizer_sha256": sha,
            "cases": [
                {"name": c["name"], "ids": tok.encode(c["text"], add_special_tokens=False).ids}
                for c in cases
            ],
        },
        indent=1,
    )
    + "\n"
)
print(f"{len(cases)} cases")
