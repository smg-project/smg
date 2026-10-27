"""The synthetic tokenizer of the gateway perf bed: its files, its vocabulary, its template."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "bench" / "gateway_perf" / "synthetic_tokenizer.py"


@pytest.fixture(scope="module")
def st():
    spec = importlib.util.spec_from_file_location("synthetic_tokenizer", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["synthetic_tokenizer"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def written(st, tmp_path_factory):
    out = tmp_path_factory.mktemp("tok")
    tokens = st.write(out, ["extra_word"])
    return out, tokens


def test_the_directory_holds_the_files_a_tokenizer_loader_reads(written):
    out, _ = written
    assert {p.name for p in out.iterdir()} == {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json",
    }
    tok = json.loads((out / "tokenizer.json").read_text())
    assert tok["model"]["type"] == "BPE" and tok["model"]["unk_token"] is None
    assert tok["normalizer"] == {"type": "NFC"}
    assert [p["type"] for p in tok["pre_tokenizer"]["pretokenizers"]] == ["Split", "ByteLevel"]
    assert tok["pre_tokenizer"]["pretokenizers"][1]["add_prefix_space"] is False
    assert tok["decoder"]["type"] == "ByteLevel" and tok["post_processor"]["type"] == "ByteLevel"
    assert tok["truncation"] is None and tok["padding"] is None
    assert {t["content"] for t in tok["added_tokens"]} == {"<|start|>", "<|end|>", "<|pad|>"}
    assert all(
        t["special"] and not t["single_word"] and not t["lstrip"] for t in tok["added_tokens"]
    )
    cfg = json.loads((out / "tokenizer_config.json").read_text())
    assert (
        cfg["eos_token"] == "<|end|>" and cfg["add_bos_token"] is False and "chat_template" in cfg
    )
    model_cfg = json.loads((out / "config.json").read_text())
    end_id = next(t["id"] for t in tok["added_tokens"] if t["content"] == "<|end|>")
    assert model_cfg["eos_token_id"] == end_id


def test_the_byte_alphabet_is_the_byte_level_one(st):
    alphabet = st.byte_alphabet()
    assert len(alphabet) == 256 and len(set(alphabet)) == 256
    assert alphabet[ord("a")] == "a" and alphabet[ord("!")] == "!"
    assert alphabet[ord(" ")] == st.SPACE == "Ġ"
    assert alphabet[ord("\n")] == "Ċ"


def test_every_bed_word_is_one_token_after_a_space_and_bare(st, written):
    _, tokens = written
    vocab = set(tokens)
    assert len(vocab) == len(tokens), "duplicate ids"
    for word in st.PROMPT_WORDS + st.STRUCTURE_WORDS + ["extra_word", "Tools", "Lookup"]:
        assert word in vocab and st.SPACE + word in vocab, word
    for ch in '{}[]":,.-_0123456789':
        assert ch in vocab
    assert "Ċ" in vocab  # the newline byte
    assert tokens[-3:] == ["<|start|>", "<|end|>", "<|pad|>"]


def test_merges_are_ranked_spaced_first_and_longest_first(st):
    alphabet = st.byte_alphabet()
    vocab, merges = st.merges_for(["own", "brown", "the", "then"], alphabet)
    assert len(set(merges)) == len(merges)
    first_bare = next(i for i, (left, _) in enumerate(merges) if not left.startswith(st.SPACE))
    assert all(left.startswith(st.SPACE) for left, _ in merges[:first_bare])
    # brown's chain (longer) ranks before own's, so "brown" never breaks into "br" + "own"
    assert merges.index((st.SPACE + "b", "r")) < merges.index((st.SPACE + "o", "w"))
    assert merges.index(("b", "r")) < merges.index(("o", "w"))
    assert st.SPACE + "brown" in vocab and "then" in vocab and st.SPACE + "the" in vocab


def test_the_vocabulary_is_deterministic(st, tmp_path):
    a = st.write(tmp_path / "a", [])
    b = st.write(tmp_path / "b", [])
    assert a == b
    assert (tmp_path / "a" / "tokenizer.json").read_bytes() == (
        tmp_path / "b" / "tokenizer.json"
    ).read_bytes()


def test_encode_gives_one_token_per_word_and_decodes_back(written):
    tokenizers = pytest.importorskip("tokenizers")
    out, _ = written
    tok = tokenizers.Tokenizer.from_file(str(out / "tokenizer.json"))
    words = "The quick brown fox jumps over the lazy dog".split()
    text = " ".join(words[i % len(words)] for i in range(128))
    encoding = tok.encode(text, add_special_tokens=False)
    assert len(encoding.ids) == 128
    assert tok.decode(encoding.ids, skip_special_tokens=False) == text
    rendered = "<|start|>user\nthe of and, k12-3: lookup_1\n<|end|>\n<|start|>assistant\n"
    encoding = tok.encode(rendered, add_special_tokens=False)
    assert tok.decode(encoding.ids, skip_special_tokens=False) == rendered
    assert encoding.tokens[0] == "<|start|>"


def test_the_chat_template_renders_roles_tools_and_the_generation_prompt(st):
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.Environment()
    env.filters["tojson"] = lambda v: json.dumps(v, separators=(",", ":"))
    template = env.from_string(st.CHAT_TEMPLATE)
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "the of and"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1_0",
                    "type": "function",
                    "function": {"name": "lookup_0", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1_0", "content": '{"b":0}'},
        {"role": "user", "content": [{"type": "text", "text": "parts"}]},
    ]
    tools = [{"type": "function", "function": {"name": "lookup_0"}}]
    text = template.render(messages=messages, tools=tools, add_generation_prompt=True)
    assert text.startswith('<|start|>system\nbe brief\nTools:\n{"type":"function"')
    assert '<|start|>tool\n{"b":0}<|end|>' in text
    assert '{"name":"lookup_0","arguments":"{}"}' in text
    assert text.endswith("<|start|>user\nparts<|end|>\n<|start|>assistant\n")
    assert text.count("<|start|>system") == 1
    plain = template.render(messages=messages[:2], add_generation_prompt=False)
    assert plain == "<|start|>system\nbe brief<|end|>\n<|start|>user\nthe of and<|end|>\n"


def test_main_writes_and_reports(st, tmp_path, capsys):
    words = tmp_path / "words.txt"
    words.write_text("alpha beta\ngamma\n")
    assert st.main([str(tmp_path / "out"), "--words", str(words)]) == 0
    assert "tokens" in capsys.readouterr().out
    vocab = json.loads((tmp_path / "out" / "tokenizer.json").read_text())["model"]["vocab"]
    assert "gamma" in vocab and st.SPACE + "alpha" in vocab
