"""smg.tokenizer.Tokenizer: SMG's tokenizer, encoding and rendering chats as serving does.

The checkpoints are written by the tests (a byte-level BPE vocabulary of the
256 byte symbols and a few merges, ChatML control tokens, a ChatML template),
so the suite needs no network. What it pins:

- encode and decode follow the checkpoint (merges, special-token text, the
  post-processor's BOS for ``add_special_tokens``), and batches and long texts
  (which run on the binding's thread pool) give what single calls give;
- a chat renders through the gateway's own request path: the template sees
  what the gateway hands it (tool-call ``arguments`` strings parsed into
  objects, ``tool_choice`` / ``reasoning_effort`` / ``chat_template_kwargs``,
  ``strftime_now`` at the render's local time or ``SOURCE_DATE_EPOCH``'s
  instant),
  the ids are the flat encode of the text for a template renderer and the
  renderer's own encode for a segment-aware one (DeepSeek V3.2), and a
  request the gateway refuses raises ``ValueError`` with its reason;
- a request renders under the tokenizer's model id, which selects the
  provider profile (a Hub cache snapshot names its own), and a local path
  without one renders no chat;
- the tokenizer encodes from several threads at once, and works in a child
  forked after the parent used it (rayon's global pool would leave such a
  child hanging); its pool takes its size from ``SMG_TOKENIZER_THREADS`` and
  ``TOKENIZERS_PARALLELISM``.
"""

from __future__ import annotations

import http.server
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import threading
import time
import types
from collections import UserDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from packaging.version import Version

pytest.importorskip("smg.smg_rs")

from smg.tokenizer import Tokenizer

MODEL = "test-org/test-model"

MERGES = [
    ("h", "e"),
    ("l", "l"),
    ("he", "ll"),
    ("hell", "o"),
    ("Ġ", "w"),
    ("o", "r"),
    ("Ġw", "or"),
    ("l", "d"),
    ("Ġwor", "ld"),
]
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
CHATML = (
    "{%- if tools is not none %}<|im_start|>system\n# Tools\n{%- for tool in tools %}\n"
    "{{ tool | tojson }}\n{%- endfor %}<|im_end|>\n{% endif %}"
    "{%- for message in messages %}<|im_start|>{{ message.role }}\n"
    "{% if message.content %}{{ message.content }}{% endif %}"
    "{%- for tool_call in message.tool_calls or [] %}\n"
    "<tool_call>{{ tool_call.function | tojson }}</tool_call>{%- endfor %}"
    "<|im_end|>\n{% endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
# Prints what the gateway hands the template besides the messages.
PROBE = (
    "tools_none={{ tools is none }} tool_choice={{ tool_choice }} "
    "reasoning_effort={{ reasoning_effort }} thinking={{ thinking }} greeting={{ greeting }} "
    "messages={{ messages | length }} generation={{ add_generation_prompt }}"
)
# A short message followed by 96 KB of template text (8000 " hello world").
LONG_TEMPLATE = (
    "{% for message in messages %}{{ message.content }}{% endfor %}"
    "{% for _ in range(8000) %} hello world{% endfor %}"
)
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def byte_symbols() -> list[str]:
    """GPT-2's printable stand-ins for the 256 byte values, in byte order."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1)]
    printable += range(ord("®"), ord("ÿ") + 1)
    symbols = {byte: chr(byte) for byte in printable}
    extra = 0
    for byte in range(256):
        if byte not in symbols:
            symbols[byte] = chr(256 + extra)
            extra += 1
    return [symbols[byte] for byte in range(256)]


VOCAB = {symbol: index for index, symbol in enumerate(byte_symbols())}
for left, right in MERGES:
    VOCAB[left + right] = len(VOCAB)
SPECIAL_IDS = {token: len(VOCAB) + index for index, token in enumerate(SPECIAL_TOKENS)}


def write_checkpoint(
    path: Path,
    *,
    chat_template: str | None = CHATML,
    post_processor: dict | None = None,
    config: dict | None = None,
) -> Path:
    """A byte-level BPE checkpoint: tokenizer.json, tokenizer_config.json and,
    when given, config.json."""
    path.mkdir(parents=True, exist_ok=True)
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": token_id,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for token, token_id in SPECIAL_IDS.items()
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "post_processor": post_processor
        or {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": False,
            "use_regex": True,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": VOCAB,
            "merges": [list(merge) for merge in MERGES],
        },
    }
    (path / "tokenizer.json").write_text(json.dumps(tokenizer))
    tokenizer_config = {
        "bos_token": None,
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "unk_token": None,
    }
    if chat_template is not None:
        tokenizer_config["chat_template"] = chat_template
    (path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    if config is not None:
        (path / "config.json").write_text(json.dumps(config))
    return path


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_checkpoint(tmp_path_factory.mktemp("chatml"))


@pytest.fixture(scope="module")
def tokenizer(checkpoint: Path) -> Tokenizer:
    return Tokenizer(checkpoint, model=MODEL)


@pytest.fixture(scope="module")
def probe_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_checkpoint(tmp_path_factory.mktemp("probe"), chat_template=PROBE)


@pytest.fixture(scope="module")
def probe(probe_checkpoint: Path) -> Tokenizer:
    return Tokenizer(probe_checkpoint, model=MODEL)


HELLO = VOCAB["hello"]
WORLD = VOCAB["Ġworld"]
IM_START = SPECIAL_IDS["<|im_start|>"]
IM_END = SPECIAL_IDS["<|im_end|>"]
ENDOFTEXT = SPECIAL_IDS["<|endoftext|>"]


def symbols(text: str) -> list[int]:
    """The ids of text no merge applies to: one byte symbol per byte."""
    return [VOCAB[byte_symbols()[byte]] for byte in text.encode()]


def chat_text(tokenizer: Tokenizer, conversation: list[Any], **fields: Any) -> str:
    """The prompt of a request for one conversation, with the request fields
    ``fields``; without the generation header unless they ask for it."""
    request = {"messages": conversation, "add_generation_prompt": False, **fields}
    return tokenizer.render_chat(request)["text"]


# --- encode and decode ---------------------------------------------------------


def test_encode_applies_the_checkpoint_merges(tokenizer: Tokenizer) -> None:
    assert tokenizer.encode("hello world") == [HELLO, WORLD]
    assert tokenizer.decode([HELLO, WORLD]) == "hello world"


def test_special_token_text_encodes_to_the_special_token(tokenizer: Tokenizer) -> None:
    ids = tokenizer.encode("hello<|im_end|>")
    assert ids == [HELLO, IM_END]
    assert tokenizer.decode(ids) == "hello<|im_end|>"
    assert tokenizer.decode(ids, skip_special_tokens=True) == "hello"


def test_add_special_tokens_adds_what_the_post_processor_adds(tmp_path: Path) -> None:
    with_bos = {
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
        ],
        "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
        "special_tokens": {
            "<|endoftext|>": {
                "id": "<|endoftext|>",
                "ids": [ENDOFTEXT],
                "tokens": ["<|endoftext|>"],
            }
        },
    }
    tokenizer = Tokenizer(write_checkpoint(tmp_path, post_processor=with_bos))
    assert tokenizer.encode("hello") == [ENDOFTEXT, HELLO]
    assert tokenizer.encode("hello", add_special_tokens=True) == [ENDOFTEXT, HELLO]
    assert tokenizer.encode("hello", add_special_tokens=False) == [HELLO]
    assert tokenizer.encode_batch(["hello"], add_special_tokens=False) == [[HELLO]]


def test_batches_and_long_texts_give_what_single_calls_give(tokenizer: Tokenizer) -> None:
    # 15 KiB encodes on the calling thread, 120 KiB on the pool.
    texts = [
        "hello world",
        "hello<|im_end|>",
        "héllo wörld 👋",
        "hello " * 2500,
        "hello world " * 10000,
    ]
    singles = [tokenizer.encode(text) for text in texts]
    assert tokenizer.encode_batch(texts) == singles
    assert tokenizer.encode_batch(tuple(texts)) == singles
    assert tokenizer.decode_batch(singles) == [tokenizer.decode(ids) for ids in singles]
    assert tokenizer.decode_batch(singles) == texts
    assert tokenizer.encode_batch([]) == []
    with pytest.raises(TypeError, match="not a single text"):
        tokenizer.encode_batch("hello world")


class _Array:
    """Stands for a numpy array or a torch tensor: values through ``tolist()``."""

    def __init__(self, values: list[Any]) -> None:
        self.values = values

    def tolist(self) -> list[Any]:
        return self.values


def test_ids_from_any_integer_sequence_or_array_decode(tokenizer: Tokenizer) -> None:
    assert tokenizer.decode((HELLO, WORLD)) == "hello world"
    assert tokenizer.decode_batch([(HELLO,), [WORLD]]) == ["hello", " world"]
    assert tokenizer.decode(_Array([HELLO, WORLD])) == "hello world"
    assert tokenizer.decode_batch(_Array([[HELLO], [WORLD]])) == ["hello", " world"]
    assert tokenizer.decode_batch([_Array([HELLO]), [WORLD]]) == ["hello", " world"]


def test_metadata(tokenizer: Tokenizer, checkpoint: Path) -> None:
    assert tokenizer.vocab_size == 256 + len(MERGES)
    assert len(tokenizer) == 256 + len(MERGES) + len(SPECIAL_TOKENS)
    assert tokenizer.special_tokens == {
        "bos_token": None,
        "eos_token": "<|im_end|>",
        "unk_token": None,
        "sep_token": None,
        "pad_token": "<|endoftext|>",
        "cls_token": None,
        "mask_token": None,
        "additional_special_tokens": SPECIAL_TOKENS,
    }
    assert (tokenizer.bos_token, tokenizer.bos_token_id) == (None, None)
    assert (tokenizer.eos_token, tokenizer.eos_token_id) == ("<|im_end|>", IM_END)
    assert (tokenizer.pad_token, tokenizer.pad_token_id) == ("<|endoftext|>", ENDOFTEXT)
    assert (tokenizer.unk_token, tokenizer.unk_token_id) == (None, None)
    assert tokenizer.eos_token_ids == [IM_END]
    assert tokenizer.token_to_id("Ġworld") == WORLD
    assert tokenizer.token_to_id("no such token") is None
    assert tokenizer.id_to_token(HELLO) == "hello"
    assert tokenizer.id_to_token(10**6) is None
    assert tokenizer.source == str(checkpoint)
    assert tokenizer.model == MODEL
    assert tokenizer.provider_profile == "generic"
    assert tokenizer.chat_template is None
    assert repr(tokenizer) == (
        f"Tokenizer(source={str(checkpoint)!r}, model={MODEL!r}, provider_profile='generic', "
        "vocab_size=265)"
    )
    assert Tokenizer.smg_build == tokenizer.smg_build
    # The wheel's version (PEP 440) and the crate's (semver) name one release.
    version = Tokenizer.smg_build.split(" (")[0]
    assert Version(version) == Version(__import__("smg").__version__)


def test_eos_token_ids_merge_the_generation_config(tmp_path: Path) -> None:
    write_checkpoint(tmp_path, config={"eos_token_id": ENDOFTEXT})
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [IM_START]}))
    assert Tokenizer(tmp_path).eos_token_ids == sorted([ENDOFTEXT, IM_START, IM_END])


# --- loading -------------------------------------------------------------------


def test_a_directory_named_mock_loads_as_the_checkpoint_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The Rust factory reads the literal paths "mock" and "test" as its toy
    # tokenizer; the binding makes an existing path absolute first.
    write_checkpoint(tmp_path / "mock")
    monkeypatch.chdir(tmp_path)
    tokenizer = Tokenizer("mock")
    assert tokenizer.source == str(tmp_path / "mock")
    assert tokenizer.encode("hello world") == [HELLO, WORLD]


def test_from_pretrained_and_path_objects(checkpoint: Path, tmp_path: Path) -> None:
    template = tmp_path / "template.jinja"
    template.write_text("{% for message in messages %}[{{ message.content }}]{% endfor %}")
    tokenizer = Tokenizer.from_pretrained(checkpoint, chat_template=template, model="my-model")
    assert tokenizer.chat_template == str(template)
    assert tokenizer.model == "my-model"
    messages = [{"role": "user", "content": "hello"}]
    assert chat_text(tokenizer, messages) == "[hello]"


def test_a_name_tiktoken_does_not_know_downloads_as_a_hub_id(
    checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "gpt-2-..." reads as an OpenAI model name to the factory's guess, but
    # tiktoken has no such model: the name is a Hub id, downloaded and loaded
    # from its snapshot as any other. A stand-in Hub lists the repository,
    # whose files are in the cache already.
    repository = tmp_path / "hub" / "models--org--gpt-2-tiny"
    snapshot = repository / "snapshots" / "abc123"
    shutil.copytree(checkpoint, snapshot)
    (repository / "refs").mkdir()
    (repository / "refs" / "main").write_text("abc123")
    files = [{"rfilename": path.name} for path in snapshot.iterdir()]
    listing = json.dumps({"sha": "abc123", "siblings": files}).encode()

    class Hub(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/api/models/org/gpt-2-tiny/revision/main":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(listing)))
            self.end_headers()
            self.wfile.write(listing)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    hub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Hub)
    threading.Thread(target=hub.serve_forever, daemon=True).start()
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{hub.server_port}")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    try:
        tokenizer = Tokenizer("org/gpt-2-tiny")
    finally:
        hub.shutdown()
        hub.server_close()
    assert tokenizer.source == str(snapshot)
    assert tokenizer.model == "org/gpt-2-tiny"
    assert tokenizer.encode("hello world") == [HELLO, WORLD]


@pytest.mark.parametrize(
    ("name", "vocab_size", "ids"),
    [("gpt-4o", 200019, [24912, 2375]), ("gpt-4", 100256, [15339, 1917])],
)
def test_an_openai_model_name_loads_with_tiktoken(
    name: str, vocab_size: int, ids: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    # tiktoken's encodings ship with it: nothing is downloaded, and a name sent
    # to the Hub instead would fail on this closed endpoint.
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    tokenizer = Tokenizer(name)
    assert tokenizer.source == name
    assert tokenizer.vocab_size == vocab_size
    assert tokenizer.encode("hello world") == ids
    assert tokenizer.decode(ids) == "hello world"
    assert tokenizer.model == name
    assert tokenizer.provider_profile == "openai"


def test_load_errors(checkpoint: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chat template file not found: /no/such.jinja"):
        Tokenizer(checkpoint, chat_template="/no/such.jinja")
    with pytest.raises(RuntimeError, match="does not contain a valid tokenizer file"):
        Tokenizer(tmp_path)
    # A path that is not there is not read as a model name, even one that
    # looks like an OpenAI model's (which would load tiktoken's encoding).
    for missing in ("/no/such/gpt-4o-distill", "./gpt-4o", "a/b/c"):
        with pytest.raises(FileNotFoundError, match=f"no such file or directory: {missing}"):
            Tokenizer(missing)


# --- the model id --------------------------------------------------------------


def test_a_local_path_has_no_model_id_and_renders_no_chat_without_one(checkpoint: Path) -> None:
    tokenizer = Tokenizer(checkpoint)
    assert (tokenizer.model, tokenizer.provider_profile) == (None, None)
    assert tokenizer.encode("hello world") == [HELLO, WORLD]
    with pytest.raises(ValueError, match="the request names no model, and this tokenizer has no"):
        tokenizer.render_chat({"messages": CONVERSATION})
    # A request that names its model renders under it.
    rendered = tokenizer.render_chat({"model": "any", "messages": CONVERSATION})
    assert rendered["text"].startswith(CONVERSATION_TEXT)


def test_a_hub_cache_snapshot_names_its_model_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "hub" / "models--deepseek-ai--DeepSeek-V4-Flash" / "snapshots" / "0a1b"
    tokenizer = Tokenizer(write_checkpoint(snapshot))
    assert tokenizer.model == "deepseek-ai/DeepSeek-V4-Flash"
    assert tokenizer.provider_profile == "deepseek"
    assert Tokenizer(snapshot / "tokenizer.json").model == "deepseek-ai/DeepSeek-V4-Flash"
    assert Tokenizer(snapshot, model="served").model == "served"
    monkeypatch.chdir(snapshot)
    assert Tokenizer(".").model == "deepseek-ai/DeepSeek-V4-Flash"
    # A relative path out of the snapshot is not in it.
    write_checkpoint(tmp_path / "models" / "my-ckpt")
    assert Tokenizer("../../../../models/my-ckpt").model is None


@pytest.mark.parametrize(
    ("model", "profile"),
    [
        ("moonshotai/Kimi-K3", "kimi"),
        ("MiniMaxAI/MiniMax-M2", "minimax"),
        ("zai-org/GLM-5.3", "zai"),
        ("deepseek-ai/DeepSeek-V4-Pro", "deepseek"),
        ("deepseek-ai/DeepSeek-V3", "generic"),
        ("gpt-4o", "openai"),
        ("Qwen/Qwen3-8B", "generic"),
    ],
)
def test_the_model_id_selects_the_provider_profile(
    checkpoint: Path, model: str, profile: str
) -> None:
    assert Tokenizer(checkpoint, model=model).provider_profile == profile


def test_the_profile_rewrites_the_request_and_the_request_model_is_replaced(
    probe_checkpoint: Path,
) -> None:
    # Under DeepSeek V4's profile, `medium` effort is the vendor's alias of
    # `high` and the thinking toggle is written into the template kwargs; a
    # request naming another model (a client's alias) renders the same, as
    # the router renders it under the served model's id.
    messages = [{"role": "user", "content": "hi"}]
    deepseek = Tokenizer(probe_checkpoint, model="deepseek-ai/DeepSeek-V4-Flash")
    generic = Tokenizer(probe_checkpoint, model="Qwen/Qwen3-8B")
    request = {"model": "client-alias", "messages": messages, "reasoning_effort": "medium"}
    assert "reasoning_effort=high thinking=True " in deepseek.render_chat(request)["text"]
    assert "reasoning_effort=medium thinking= " in generic.render_chat(request)["text"]


# --- chats ---------------------------------------------------------------------

CONVERSATION = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "world"},
]
CONVERSATION_TEXT = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\nworld<|im_end|>\n"


def test_a_conversation_renders_to_the_template_text_and_its_flat_encode(
    tokenizer: Tokenizer,
) -> None:
    assert chat_text(tokenizer, CONVERSATION) == CONVERSATION_TEXT
    rendered = tokenizer.render_chat({"messages": CONVERSATION, "add_generation_prompt": False})
    ids = rendered["token_ids"]
    assert ids == tokenizer.encode(CONVERSATION_TEXT, add_special_tokens=False)
    assert ids[:2] == [IM_START, *symbols("user")[:1]]
    assert chat_text(tokenizer, CONVERSATION[:1], add_generation_prompt=True) == (
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    )


def test_continue_final_message_leaves_the_last_message_open(tokenizer: Tokenizer) -> None:
    text = chat_text(tokenizer, CONVERSATION, continue_final_message=True)
    assert text == "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\nworld"
    # SMG continues a final assistant message only: asked of a user message,
    # serving renders it closed.
    rendered = tokenizer.render_chat(
        {
            "messages": CONVERSATION[:1],
            "continue_final_message": True,
            "add_generation_prompt": False,
        }
    )
    assert rendered["continued_final_message"] is False


def test_a_renderer_that_cannot_continue_appends_the_text_after_the_header(
    tmp_path: Path,
) -> None:
    # DeepSeek V3.2's encoder does not continue a message itself: SMG renders
    # the generation header and appends the final message's text after it,
    # which is SMG's continuation for that checkpoint.
    tokenizer = Tokenizer(
        write_checkpoint(
            tmp_path, chat_template=None, config={"architectures": ["DeepseekV32ForCausalLM"]}
        ),
        model=MODEL,
    )
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "The answer"},
    ]
    rendered = tokenizer.render_chat(
        {"messages": messages, "continue_final_message": True, "add_generation_prompt": False}
    )
    assert rendered["text"].endswith("The answer")
    assert (rendered["generation_prompt"], rendered["continued_final_message"]) == (True, False)


def test_tool_call_arguments_reach_the_template_as_the_gateway_hands_them(
    tokenizer: Tokenizer,
) -> None:
    # The gateway parses a JSON-string `arguments` into an object for a
    # template that serializes it, so the prompt carries the object, not an
    # escaped string; key order is kept.
    call = {"name": "get_weather", "arguments": '{"city": "Paris", "days": 2}'}
    messages: list[Any] = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_0", "type": "function", "function": call}],
        },
    ]
    text = chat_text(tokenizer, messages, tools=[WEATHER_TOOL])
    assert text == (
        "<|im_start|>system\n# Tools"
        '{"type": "function", "function": {"name": "get_weather", "description": "Weather", '
        '"parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}'
        "<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris", "days": 2}}</tool_call>'
        "<|im_end|>\n"
    )


def test_request_fields_and_template_kwargs_reach_the_template(probe: Tokenizer) -> None:
    messages = [{"role": "user", "content": "hi"}]
    assert chat_text(probe, messages) == (
        "tools_none=True tool_choice= reasoning_effort= thinking= greeting= messages=1 "
        "generation=False"
    )
    # tools=[] is a declared empty list, which normalization gives the
    # tool_choice "none"; a non-empty list gets "auto".
    assert chat_text(probe, messages, tools=[]).startswith("tools_none=False tool_choice=none ")
    # An explicit tool_choice is the request's, not normalization's default.
    text = chat_text(probe, messages, tools=[WEATHER_TOOL], tool_choice="none")
    assert text.startswith("tools_none=False tool_choice=none ")
    text = chat_text(
        probe,
        messages,
        tools=[WEATHER_TOOL],
        add_generation_prompt=True,
        reasoning_effort="low",
        chat_template_kwargs={"greeting": "hey"},
    )
    assert text == (
        "tools_none=False tool_choice=auto reasoning_effort=low thinking= greeting=hey "
        "messages=1 generation=True"
    )


def test_template_kwargs_cannot_replace_what_the_render_sets(probe: Tokenizer) -> None:
    # Serving's render context wins over chat_template_kwargs.
    rendered = probe.render_chat(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "add_generation_prompt": False,
            "chat_template_kwargs": {"add_generation_prompt": True, "messages": []},
        }
    )
    assert rendered["text"].endswith("messages=1 generation=False")


RENDER_A_CHAT = """
import sys
from smg.tokenizer import Tokenizer
tokenizer = Tokenizer(sys.argv[1], model=sys.argv[2])
print(tokenizer.render_chat({"messages": [{"role": "user", "content": "hi"}]})["text"])
"""
# UTC+05:30, spelled out so that it needs no time zone database.
ZONE = "<+0530>-05:30"
ZONE_OFFSET = 5 * 3600 + 30 * 60


@pytest.mark.skipif(sys.platform == "win32", reason="chrono reads the Windows time zone, not TZ")
@pytest.mark.parametrize("pinned", [True, False])
def test_strftime_now_writes_the_local_time_or_source_date_epoch(
    tmp_path: Path, pinned: bool
) -> None:
    # Serving's strftime_now: the local time at the render, or the instant
    # SOURCE_DATE_EPOCH names, in local time too. A process reads the
    # variable once, as it loads its first chat template: hence a process
    # for each case, started with it set or unset.
    template = "{{ strftime_now('%Y-%m-%d %H:%M') }}"
    checkpoint = write_checkpoint(tmp_path, chat_template=template)
    environment = {name: value for name, value in os.environ.items() if name != "SOURCE_DATE_EPOCH"}
    environment["TZ"] = ZONE
    if pinned:
        environment["SOURCE_DATE_EPOCH"] = "1791979200"  # 2026-10-14 12:00 UTC
    clock = "%Y-%m-%d %H:%M"
    before = time.strftime(clock, time.gmtime(time.time() + ZONE_OFFSET))
    child = subprocess.run(
        [sys.executable, "-c", RENDER_A_CHAT, str(checkpoint), MODEL],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    after = time.strftime(clock, time.gmtime(time.time() + ZONE_OFFSET))
    if pinned:
        assert child.stdout.strip() == "2026-10-14 17:30"
    else:
        assert before <= child.stdout.strip() <= after


def test_render_chat_takes_the_request_body(tokenizer: Tokenizer) -> None:
    request = {"messages": CONVERSATION[:1]}
    rendered = tokenizer.render_chat(request)
    assert rendered == {
        "text": "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
        "token_ids": tokenizer.encode(rendered["text"], add_special_tokens=False),
        "from_text": True,
        # The gateway renders the generation header when the request does not
        # say otherwise.
        "generation_prompt": True,
        "continued_final_message": False,
        "warnings": [],
    }
    assert tokenizer.render_chat(json.dumps(request)) == rendered
    assert tokenizer.render_chat(json.dumps(request).encode()) == rendered
    assert tokenizer.render_chat_batch([request, json.dumps(request)]) == [rendered, rendered]
    with pytest.raises(TypeError, match="a chat request is a mapping or its JSON text"):
        tokenizer.render_chat(42)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="not a single request"):
        tokenizer.render_chat_batch(request)  # type: ignore[arg-type]
    # JSON text is a sequence too, of one-character texts, and one the type
    # checker lets through.
    with pytest.raises(TypeError, match="not a single request"):
        tokenizer.render_chat_batch(json.dumps(request))
    with pytest.raises(TypeError, match="not a single request"):
        tokenizer.render_chat_batch(json.dumps(request).encode())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("request_body", "message"),
    [
        ({"messages": []}, "SMG rejects this request: messages: Invalid 'messages': empty array"),
        (
            {
                "messages": CONVERSATION,
                "continue_final_message": True,
                "add_generation_prompt": True,
            },
            "Cannot set both `continue_final_message` and `add_generation_prompt` to True",
        ),
        (
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}],
                    }
                ]
            },
            "SMG does not accept this request: messages[0]: missing field `id`",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}
                ]
            },
            "messages[0] has image_url content",
        ),
        (
            {"messages": CONVERSATION, "return_audio": True},
            "SMG rejects this request: 'return_audio' must be false",
        ),
        (
            {"messages": CONVERSATION, "modalities": ["text", "audio"]},
            "SMG rejects this request: unsupported output modality 'audio'",
        ),
        ("[]", "SMG does not accept this request: invalid type: sequence"),
        ("{", "SMG does not accept this request: EOF while parsing an object"),
        (
            '{"messages": [], "messages": []}',
            "SMG does not accept this request: duplicate field `messages`",
        ),
    ],
    ids=[
        "empty",
        "continue-and-generate",
        "tool-call-without-id",
        "media",
        "audio-output",
        "audio-modality",
        "array",
        "json",
        "duplicate-field",
    ],
)
def test_a_request_the_gateway_refuses_raises_its_reason(
    tokenizer: Tokenizer, request_body: dict[str, Any] | str, message: str
) -> None:
    with pytest.raises(ValueError) as refused:
        tokenizer.render_chat(request_body)
    assert message in str(refused.value)


def test_a_batch_names_the_request_it_refuses(tokenizer: Tokenizer) -> None:
    with pytest.raises(ValueError, match=r"^requests\[1\]: SMG rejects this request"):
        tokenizer.render_chat_batch([{"messages": CONVERSATION}, {"messages": []}])


def test_a_gpt_oss_checkpoint_is_refused_whatever_the_model_id(tmp_path: Path) -> None:
    tokenizer = Tokenizer(
        write_checkpoint(tmp_path, config={"architectures": ["GptOssForCausalLM"]}),
        model="my-model",
    )
    with pytest.raises(ValueError, match="SMG renders gpt-oss models with Harmony"):
        tokenizer.render_chat({"messages": CONVERSATION})
    assert tokenizer.encode("hello") == [HELLO]


def test_a_gpt_oss_model_id_is_refused(checkpoint: Path) -> None:
    with pytest.raises(ValueError, match=r"Harmony.*\(model openai/gpt-oss-20b\)"):
        Tokenizer(checkpoint, model="openai/gpt-oss-20b").render_chat({"messages": CONVERSATION})
    with pytest.raises(ValueError, match=r"Harmony.*\(model GPT-OSS-120B\)"):
        Tokenizer(checkpoint).render_chat({"model": "GPT-OSS-120B", "messages": CONVERSATION})


def test_rewrites_smg_logs_come_back_as_warnings(tokenizer: Tokenizer) -> None:
    # Tool-call arguments that are not JSON render as {} in serving, with a
    # line in its log; here they come back with the rendering.
    call = {"name": "get_weather", "arguments": "{not json"}
    messages: list[Any] = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_0", "type": "function", "function": call}],
        },
    ]
    rendered = tokenizer.render_chat({"messages": messages, "add_generation_prompt": False})
    assert '"arguments": {}' in rendered["text"]
    [warning] = rendered["warnings"]
    assert warning.startswith(
        "tool call arguments in history are not valid JSON; rendered as an empty object "
        "(tool=get_weather, bytes=9, error="
    )
    assert tokenizer.render_chat({"messages": CONVERSATION})["warnings"] == []


class _Message:
    """Stands for a pydantic message, such as the OpenAI SDK's."""

    def __init__(self, role: str, content: str | None) -> None:
        self.role, self.content = role, content

    def model_dump(self, *, exclude_none: bool) -> dict[str, Any]:
        fields = {"role": self.role, "content": self.content, "refusal": None}
        return {k: v for k, v in fields.items() if v is not None or not exclude_none}


def test_messages_given_as_models_render_by_their_fields(tokenizer: Tokenizer) -> None:
    messages = [_Message("user", "hello"), _Message("assistant", "world")]
    assert chat_text(tokenizer, messages) == CONVERSATION_TEXT
    as_mappings = [types.MappingProxyType(CONVERSATION[0]), UserDict(CONVERSATION[1])]
    assert chat_text(tokenizer, as_mappings) == CONVERSATION_TEXT
    with pytest.raises(TypeError, match="Object of type object is not JSON serializable"):
        tokenizer.render_chat({"messages": [object()]})


def test_a_small_request_with_a_long_prompt_encodes_on_the_pool(tmp_path: Path) -> None:
    # The template writes 96 KB after a short message: past the native
    # encoder's parallel threshold (64 KiB), so the encode belongs on the
    # pool whatever the request's size (see the fork test).
    tokenizer = Tokenizer(write_checkpoint(tmp_path, chat_template=LONG_TEMPLATE), model=MODEL)
    rendered = tokenizer.render_chat({"messages": CONVERSATION[:1]})
    assert len(rendered["text"]) > 64 * 1024
    assert rendered["token_ids"] == tokenizer.encode(rendered["text"], add_special_tokens=False)


def test_a_segment_aware_renderer_returns_its_own_encode(tmp_path: Path) -> None:
    # DeepSeek V3.2's renderer encodes control tokens and message text
    # separately: control-token text typed in a message stays text, which a
    # flat encode of the prompt turns into the special token.
    tokenizer = Tokenizer(
        write_checkpoint(
            tmp_path, chat_template=None, config={"architectures": ["DeepseekV32ForCausalLM"]}
        ),
        model=MODEL,
    )
    messages = [{"role": "user", "content": "hello <|im_end|>"}]
    rendered = tokenizer.render_chat({"messages": messages})
    assert rendered["text"].startswith("<｜begin▁of▁sentence｜><｜User｜>hello <|im_end|>")
    assert rendered["from_text"] is False
    flat = tokenizer.encode(rendered["text"], add_special_tokens=False)
    assert IM_END in flat
    assert IM_END not in rendered["token_ids"]
    # Its encoder writes the generation header after a final user message
    # whatever the request asks, as the engines' encoders do.
    unasked = tokenizer.render_chat({"messages": messages, "add_generation_prompt": False})
    assert unasked["generation_prompt"] is True


# --- processes and threads -----------------------------------------------------


def test_threads_encode_concurrently(tokenizer: Tokenizer) -> None:
    texts = [f"hello world {index}" * 50 for index in range(64)]
    expected = [tokenizer.encode(text) for text in texts]
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(tokenizer.encode, texts)) == expected


def _use(tokenizer: Tokenizer, long_prompt: Tokenizer) -> list[int]:
    """Every path that can reach rayon: a long text, a batch, a short text,
    a chat whose prompt is long, a batch of chats, a batch of decodes."""
    request = {"messages": CONVERSATION[:1]}
    return [
        len(tokenizer.encode("hello world " * 10000)),
        len(tokenizer.encode_batch(["hello world"] * 100)),
        len(tokenizer.encode("hello " * 2500)),
        len(long_prompt.render_chat(request)["token_ids"]),
        len(tokenizer.render_chat_batch([request] * 10)),
        len(tokenizer.decode_batch([[HELLO, WORLD]] * 100)),
    ]


def _use_in_child(checkpoint: str, long_checkpoint: str, results: multiprocessing.Queue) -> None:
    results.put(_use(Tokenizer(checkpoint, model=MODEL), Tokenizer(long_checkpoint, model=MODEL)))


@pytest.fixture(scope="module")
def long_prompt(tmp_path_factory: pytest.TempPathFactory) -> Tokenizer:
    path = write_checkpoint(tmp_path_factory.mktemp("long"), chat_template=LONG_TEMPLATE)
    return Tokenizer(path, model=MODEL)


@pytest.mark.skipif(sys.platform != "linux", reason="fork start method")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_child_forked_after_use_does_not_hang(
    tokenizer: Tokenizer, long_prompt: Tokenizer
) -> None:
    # Each path, taken in the parent and again in the child: had one created
    # rayon's global pool, the forked child's first use of it would wait
    # forever for threads it did not inherit.
    expected = _use(tokenizer, long_prompt)
    context = multiprocessing.get_context("fork")
    results = context.Queue()
    child = context.Process(
        target=_use_in_child, args=(tokenizer.source, long_prompt.source, results)
    )
    child.start()
    child.join(timeout=120)
    if child.is_alive():
        child.kill()
        pytest.fail("the forked child hung")
    assert child.exitcode == 0
    assert results.get(timeout=10) == expected


POOL_THREADS = """
import os, sys
from smg.tokenizer import Tokenizer
Tokenizer(sys.argv[1])
names = [open(f"/proc/self/task/{task}/comm").read().strip() for task in os.listdir("/proc/self/task")]
print(sum(name.startswith("smg-tokenizer") for name in names))
"""


@pytest.mark.skipif(sys.platform != "linux", reason="thread names under /proc")
@pytest.mark.parametrize(
    ("environment", "threads"),
    [
        ({"SMG_TOKENIZER_THREADS": "3"}, "3"),
        ({"SMG_TOKENIZER_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}, "2"),
        ({"TOKENIZERS_PARALLELISM": "false"}, "1"),
        ({"TOKENIZERS_PARALLELISM": "", "RAYON_NUM_THREADS": "4"}, "1"),
        ({"TOKENIZERS_PARALLELISM": "true", "RAYON_NUM_THREADS": "4"}, "4"),
    ],
)
def test_the_pool_takes_its_size_from_the_environment(
    checkpoint: Path, environment: dict[str, str], threads: str
) -> None:
    clean = {
        name: value
        for name, value in os.environ.items()
        if name not in ("SMG_TOKENIZER_THREADS", "TOKENIZERS_PARALLELISM", "RAYON_NUM_THREADS")
    }
    child = subprocess.run(
        [sys.executable, "-c", POOL_THREADS, str(checkpoint)],
        env={**clean, **environment},
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert child.stdout.strip() == threads


def test_a_pool_size_that_is_no_number_is_refused(checkpoint: Path) -> None:
    child = subprocess.run(
        [sys.executable, "-c", POOL_THREADS, str(checkpoint)],
        env={**os.environ, "SMG_TOKENIZER_THREADS": "many"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode != 0
    assert 'SMG_TOKENIZER_THREADS must be a positive number of threads, not "many"' in child.stderr
