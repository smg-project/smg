# crates/tokenizer/scripts/generate_deepseek_v41_fixtures.py
"""Record DeepSeek-V4.1 render fixtures from the reference encoder.

Oracle: oracle/deepseek_v41_encoding.py (HF encoding.py @ 517ef625df, the
revision vLLM and SGLang ported). The effort tiers are overridden to the
engine table (spec D1). Token ids come from the real tokenizer.json.
"""

import hashlib
import json
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
OUT = Path(__file__).parents[1] / "tests" / "fixtures" / "deepseek_v41"


def load_hf_case(path):
    """HF fixture inputs are either a dict with thinking_mode/tools/
    reasoning_effort/messages keys, or a bare JSON list that IS the messages
    (with any tools already embedded on the relevant message)."""
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return {"messages": data}
    return data


def case(
    name,
    messages,
    tools=None,
    thinking_mode="chat",
    reasoning_effort=None,
    drop_thinking=True,
    continue_final_message=False,
):
    msgs = json.loads(json.dumps(messages))
    if tools:
        first = msgs[0]
        if first["role"] != "system":
            msgs.insert(0, {"role": "system", "content": ""})
        msgs[0]["tools"] = tools
    if continue_final_message:
        msgs[-1]["wo_eos"] = True
    text = enc.encode_messages(
        msgs,
        thinking_mode=thinking_mode,
        drop_thinking=drop_thinking,
        reasoning_effort=reasoning_effort,
    )
    return {
        "name": name,
        "messages": messages,
        "tools": tools,
        "thinking_mode": thinking_mode,
        "reasoning_effort": reasoning_effort,
        "drop_thinking": drop_thinking,
        "continue_final_message": continue_final_message,
        "text": text,
    }


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
]

tok = Tokenizer.from_file(str(TOKENIZER))
sha = hashlib.sha256(TOKENIZER.read_bytes()).hexdigest()
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "render_fixtures.json").write_text(
    json.dumps(
        {
            "source": "deepseek-ai/DeepSeek-V4.1-Flash encoding/encoding.py",
            "revision": "517ef625df",
            "effort_tiers": ENGINE_TIERS,
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
