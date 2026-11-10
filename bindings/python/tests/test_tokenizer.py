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
- ``apply_chat_template`` refuses what SMG would not do (a template variable
  the render fills itself, a generation prompt or continuation the renderer
  does not honour) and issues the rewrites SMG logs as ``RenderWarning``;
- the tokenizer pickles with its smg build, encodes from several threads at
  once, and works in a child forked after the parent used it, or while
  another thread is in a call (rayon's global pool, or a lock held at the
  fork, would leave such a child hanging); its pool takes its size from
  ``SMG_TOKENIZER_THREADS`` and ``TOKENIZERS_PARALLELISM``.
"""

from __future__ import annotations

import copy
import http.server
import inspect
import json
import multiprocessing
import os
import pickle
import shutil
import subprocess
import sys
import threading
import time
import types
import warnings
from collections import UserDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from packaging.version import Version

pytest.importorskip("smg.smg_rs")

from smg.tokenizer import RenderWarning, Tokenizer

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


def chat_text(tokenizer: Tokenizer, conversation: list[Any], **kwargs: Any) -> str:
    """``apply_chat_template(..., tokenize=False)`` of one conversation."""
    text = tokenizer.apply_chat_template(conversation, tokenize=False, **kwargs)
    assert isinstance(text, str)
    return text


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
    assert tokenizer.apply_chat_template(messages, tokenize=False) == "[hello]"


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
    restored = pickle.loads(pickle.dumps(tokenizer))
    assert restored.source == name
    assert restored.fingerprint == tokenizer.fingerprint
    assert restored.encode("hello world") == ids


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
        tokenizer.apply_chat_template(CONVERSATION)
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
    ids = tokenizer.apply_chat_template(CONVERSATION)
    assert ids == tokenizer.encode(CONVERSATION_TEXT, add_special_tokens=False)
    assert ids[:2] == [IM_START, *symbols("user")[:1]]
    assert chat_text(tokenizer, CONVERSATION[:1], add_generation_prompt=True) == (
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    )


def test_continue_final_message_leaves_the_last_message_open(tokenizer: Tokenizer) -> None:
    text = chat_text(tokenizer, CONVERSATION, continue_final_message=True)
    assert text == "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\nworld"
    # SMG continues a final assistant message only: asked of a user message,
    # serving renders it closed, which apply_chat_template refuses to pass
    # off as a continuation.
    with pytest.raises(ValueError, match="SMG did not continue the final message"):
        tokenizer.apply_chat_template(CONVERSATION[:1], continue_final_message=True)
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
    assert chat_text(tokenizer, messages, continue_final_message=True) == rendered["text"]


def test_a_list_of_conversations_renders_each(tokenizer: Tokenizer) -> None:
    batch = [CONVERSATION, CONVERSATION[:1]]
    assert tokenizer.apply_chat_template(batch, tokenize=False) == [
        CONVERSATION_TEXT,
        "<|im_start|>user\nhello<|im_end|>\n",
    ]
    assert tokenizer.apply_chat_template(batch) == [
        tokenizer.apply_chat_template(CONVERSATION),
        tokenizer.apply_chat_template(CONVERSATION[:1]),
    ]


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
        greeting="hey",
    )
    assert text == (
        "tools_none=False tool_choice=auto reasoning_effort=low thinking= greeting=hey "
        "messages=1 generation=True"
    )


def test_template_kwargs_cannot_replace_what_the_render_sets(probe: Tokenizer) -> None:
    # Serving's render context wins over chat_template_kwargs: render_chat
    # renders the request as serving does ...
    rendered = probe.render_chat(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "add_generation_prompt": False,
            "chat_template_kwargs": {"add_generation_prompt": True, "messages": []},
        }
    )
    assert rendered["text"].endswith("messages=1 generation=False")
    # ... and apply_chat_template refuses those SMG would not pass
    # (transformers lets strftime_now be replaced; given messages, its call
    # to Jinja's render() fails).
    messages = [{"role": "user", "content": "hi"}]
    with pytest.raises(TypeError, match="does not pass 'messages' to the template"):
        probe.apply_chat_template(messages, messages=[])
    with pytest.raises(TypeError, match="does not pass 'strftime_now' to the template"):
        probe.apply_chat_template(messages, strftime_now=lambda _: "2026-01-01")


def test_a_special_token_the_checkpoint_defines_is_not_replaced(tmp_path: Path) -> None:
    tokenizer = Tokenizer(
        write_checkpoint(tmp_path, chat_template="bos={{ bos_token }} eos={{ eos_token }}"),
        model=MODEL,
    )
    messages = [{"role": "user", "content": "hi"}]
    assert chat_text(tokenizer, messages) == "bos= eos=<|im_end|>"
    with pytest.raises(TypeError, match=r"does not pass 'eos_token' .* \('<\|im_end\|>'\)"):
        tokenizer.apply_chat_template(messages, eos_token="</s>")
    # One it does not define, the keyword argument fills, as in serving.
    assert chat_text(tokenizer, messages, bos_token="<s>") == "bos=<s> eos=<|im_end|>"


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
        tokenizer.apply_chat_template(CONVERSATION)
    assert tokenizer.encode("hello") == [HELLO]


def test_a_gpt_oss_model_id_is_refused(checkpoint: Path) -> None:
    with pytest.raises(ValueError, match=r"Harmony.*\(model openai/gpt-oss-20b\)"):
        Tokenizer(checkpoint, model="openai/gpt-oss-20b").apply_chat_template(CONVERSATION)
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
    with pytest.warns(RenderWarning, match="rendered as an empty object") as issued:
        tokenizer.apply_chat_template(messages)
        tokenizer.apply_chat_template([messages, messages])
    assert [warning.filename for warning in issued] == [__file__] * 3
    with warnings.catch_warnings():
        warnings.simplefilter("error", RenderWarning)
        with pytest.raises(RenderWarning):
            tokenizer.apply_chat_template([messages, messages])
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


def test_transformers_arguments_without_an_smg_equivalent_are_refused(
    tokenizer: Tokenizer,
) -> None:
    with pytest.raises(TypeError, match="does not take 'documents'"):
        tokenizer.apply_chat_template(CONVERSATION, documents=[{"text": "x"}])
    with pytest.raises(TypeError, match="does not take 'return_assistant_tokens_mask'"):
        tokenizer.apply_chat_template(CONVERSATION, return_assistant_tokens_mask=True)
    with pytest.raises(TypeError, match="does not take 'sanitize_control_tokens'"):
        tokenizer.apply_chat_template(CONVERSATION, sanitize_control_tokens=True)
    with pytest.raises(TypeError, match="continue_final_message='reasoning_content'"):
        tokenizer.apply_chat_template(CONVERSATION, continue_final_message="reasoning_content")
    # Their no-op values are accepted, and transformers 5's name for the content.
    text = tokenizer.apply_chat_template(
        CONVERSATION,
        tokenize=False,
        documents=None,
        padding=False,
        return_dict=False,
        sanitize_control_tokens=False,
    )
    assert text == CONVERSATION_TEXT
    continued = chat_text(tokenizer, CONVERSATION, continue_final_message="content")
    assert continued == chat_text(tokenizer, CONVERSATION, continue_final_message=True)


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
    assert (
        tokenizer.apply_chat_template(messages, add_generation_prompt=True) == rendered["token_ids"]
    )
    # Its encoder writes the generation header after a final user message
    # whatever the request asks, as the engines' encoders do: render_chat
    # reports it, apply_chat_template refuses to call it no header.
    unasked = tokenizer.render_chat({"messages": messages, "add_generation_prompt": False})
    assert unasked["generation_prompt"] is True
    with pytest.raises(ValueError, match="SMG wrote the generation prompt although"):
        tokenizer.apply_chat_template(messages)


# --- processes and threads -----------------------------------------------------


def test_a_tokenizer_pickles_and_reloads(tokenizer: Tokenizer) -> None:
    restored = pickle.loads(pickle.dumps(tokenizer))
    assert restored is not tokenizer
    assert (restored.source, restored.model) == (tokenizer.source, tokenizer.model)
    assert restored.encode("hello world") == tokenizer.encode("hello world")
    assert copy.copy(tokenizer) is tokenizer
    assert copy.deepcopy(tokenizer) is tokenizer


def test_a_pickle_of_another_smg_build_is_refused(tokenizer: Tokenizer) -> None:
    restore, arguments = tokenizer.__reduce__()
    assert arguments[-2:] == (Tokenizer.smg_build, tokenizer.fingerprint)
    with pytest.raises(RuntimeError, match=r"pickled by smg 0\.0\.0 \(old\) and is unpickled by"):
        restore(*arguments[:-2], "0.0.0 (old)", arguments[-1])


def test_the_fingerprint_follows_the_files_and_a_pickle_refuses_changed_ones(
    tmp_path: Path,
) -> None:
    path = write_checkpoint(tmp_path / "ckpt")
    template = tmp_path / "template.jinja"
    template.write_text("{% for message in messages %}[{{ message.content }}]{% endfor %}")
    tokenizer = Tokenizer(path, chat_template=template, model=MODEL)
    same = Tokenizer(path, chat_template=template, model=MODEL)
    assert same.fingerprint == tokenizer.fingerprint
    other = Tokenizer(path, chat_template=template, model="other")
    assert other.fingerprint != tokenizer.fingerprint
    pickled = pickle.dumps(tokenizer)
    # A template edited in place: another fingerprint, so another pickle (the
    # cache key datasets.map derives from it), and the old pickle refused.
    template.write_text("{% for message in messages %}<{{ message.content }}>{% endfor %}")
    edited = Tokenizer(path, chat_template=template, model=MODEL)
    assert edited.fingerprint != tokenizer.fingerprint
    assert pickle.dumps(edited) != pickled
    with pytest.raises(RuntimeError, match="changed since this tokenizer was pickled"):
        pickle.loads(pickled)
    # A pickled path that is not there is refused, not read as a model name.
    moved = pickle.dumps(Tokenizer(write_checkpoint(tmp_path / "gpt-4o"), model=MODEL))
    (tmp_path / "gpt-4o").rename(tmp_path / "elsewhere")
    with pytest.raises(FileNotFoundError, match="no such file or directory"):
        pickle.loads(moved)


def test_a_hub_snapshot_pickles_with_its_model_id(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--org--name" / "snapshots" / "0a1b"
    tokenizer = Tokenizer(write_checkpoint(snapshot))
    restore, arguments = tokenizer.__reduce__()
    assert arguments[1:5] == (str(snapshot), None, "org/name", ("org/name", ""))
    restored = restore(*arguments)
    assert (restored.source, restored.model) == (str(snapshot), "org/name")
    file_arguments = Tokenizer(snapshot / "tokenizer.json").__reduce__()[1]
    assert file_arguments[4] == ("org/name", "tokenizer.json")
    (snapshot / "tok_v2.json").write_text((snapshot / "tokenizer.json").read_text())
    other_file_arguments = Tokenizer(snapshot / "tok_v2.json").__reduce__()[1]
    # The snapshot here, with files that differ from the pickled ones (edited,
    # or only partly downloaded into this cache), is refused.
    (snapshot / "generation_config.json").write_text("{}")
    with pytest.raises(RuntimeError, match=r"Tokenizer\('org/name'\) downloads the missing"):
        restore(*arguments)
    (snapshot / "generation_config.json").unlink()
    # Where the snapshot is missing, its Hub id downloads it, and a file
    # source loads its file in the download: here a stand-in for the
    # download, which loads a copy of the files from elsewhere.
    snapshot.rename(tmp_path / "gone")
    loaded: list[str] = []

    def downloading(copy: Path) -> type[Tokenizer]:
        class Downloaded(Tokenizer):
            def __init__(self, source: str, **kwargs: Any) -> None:
                loaded.append(source)
                super().__init__(copy if source == "org/name" else source, **kwargs)

        return Downloaded

    hub = write_checkpoint(tmp_path / "hub")
    restored = restore(downloading(hub), *arguments[1:])
    assert (restored.source, restored.model) == (str(hub), "org/name")
    restored = restore(downloading(hub), *file_arguments[1:])
    assert restored.source == str(hub / "tokenizer.json")
    assert loaded == ["org/name", "org/name", str(hub / "tokenizer.json")]
    # A file the download has not, or downloaded files that differ (the Hub's
    # main moved on), are refused.
    with pytest.raises(RuntimeError, match="without tok_v2.json"):
        restore(downloading(hub), *other_file_arguments[1:])
    # Downloaded files that differ (the Hub's main moved on) are refused.
    moved = write_checkpoint(tmp_path / "moved", chat_template=PROBE)
    with pytest.raises(RuntimeError, match="which is not here; org/name downloaded from the Hub"):
        restore(downloading(moved), *arguments[1:])


def _encode_in_worker(tokenizer: Tokenizer) -> list[int]:
    return tokenizer.encode("hello world")


def test_a_spawned_worker_unpickles_the_tokenizer(tokenizer: Tokenizer) -> None:
    context = multiprocessing.get_context("spawn")
    with context.Pool(1) as pool:
        assert pool.apply(_encode_in_worker, (tokenizer,)) == [HELLO, WORLD]


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


def _tool(name: str) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


@pytest.mark.skipif(sys.platform != "linux", reason="os.fork")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_child_forked_during_a_call_renders_with_the_inherited_tokenizer(
    tokenizer: Tokenizer,
) -> None:
    # Another thread keeps rendering tool sets never sent before, each a
    # write to the gateway's prepared-tools table. With a long switch
    # interval this thread runs only while that one is in a call, so every
    # fork lands in one; os.fork() waits for it to end, and the child, which
    # renders tools with the tokenizer it inherited, finds no lock held.
    # Forked mid-call instead, about half such children wait forever.
    stop = threading.Event()
    names = iter(range(10**9))

    def request() -> dict[str, Any]:
        return {"messages": CONVERSATION[:1], "tools": [_tool(f"tool_{next(names)}")]}

    def keep_calling() -> None:
        while not stop.is_set():
            tokenizer.render_chat_batch([request() for _ in range(64)])

    caller = threading.Thread(target=keep_calling)
    interval = sys.getswitchinterval()
    caller.start()
    sys.setswitchinterval(100.0)
    try:
        for _ in range(10):
            time.sleep(0.002)
            pid = os.fork()
            if pid == 0:
                code = 2
                try:
                    tokenizer.render_chat_batch([request() for _ in range(8)])
                    code = 0
                finally:
                    os._exit(code)
            deadline = time.monotonic() + 60
            while True:
                done, status = os.waitpid(pid, os.WNOHANG)
                if done:
                    assert os.waitstatus_to_exitcode(status) == 0
                    break
                if time.monotonic() > deadline:
                    os.kill(pid, 9)
                    os.waitpid(pid, 0)
                    pytest.fail("a child forked during a call hung")
                time.sleep(0.01)
    finally:
        sys.setswitchinterval(interval)
        stop.set()
        caller.join()


@pytest.mark.skipif(sys.platform != "linux", reason="os.fork")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_os_fork_waits_until_the_call_in_flight_ends(tokenizer: Tokenizer) -> None:
    # A thread decodes with the GIL released while this one forks: the fork
    # starts only once the decode has ended. The fork waits for the detached
    # part of the call alone; the id list's extraction runs before it with
    # the GIL held. The wait is held to the call's window from the moment
    # this thread first runs: woken by the caller, it takes the GIL no
    # earlier than the decode releases it (the switch interval is set long,
    # so the caller is never made to hand it over), and the extraction is
    # over by then. The window's tail, the text's conversion with the GIL
    # taken back, is a small share of it. Held to a quarter of the call as
    # a whole, the test failed on a runner whose extraction took the larger
    # share, with the guard intact (a wait of 0.081 s for a 0.329 s call).
    # The ids double until a window alone takes a quarter of a second,
    # within a bound on their count; each size's figure is the minimum of
    # three, since one sample inflated by a descheduling on a loaded runner
    # would otherwise stop the doubling at a size whose window is far
    # shorter and overstate the reference. The fork starts 10 ms into the
    # window and its wait must cover more than half of the window measured
    # alone (the forked run's own window would carry a stall of its tail,
    # which runs after the fork, into the reference); a fork that did not
    # wait returns within milliseconds.
    def window(ids: list[int], fork: bool) -> tuple[float, float]:
        calling = threading.Event()
        ended = 0.0
        failure: BaseException | None = None

        def call() -> None:
            nonlocal ended, failure
            calling.set()
            try:
                tokenizer.decode(ids)
            except BaseException as error:
                failure = error
            ended = time.perf_counter()

        caller = threading.Thread(target=call)
        caller.start()
        calling.wait()
        start = time.perf_counter()
        waited = 0.0
        if fork:
            time.sleep(0.01)
            forking = time.perf_counter()
            pid = os.fork()
            if pid == 0:
                os._exit(0)
            waited = time.perf_counter() - forking
            os.waitpid(pid, 0)
        caller.join()
        if failure is not None:
            raise failure
        return ended - start, waited

    interval = sys.getswitchinterval()
    sys.setswitchinterval(100.0)
    try:
        ids = [HELLO, WORLD] * 1_000_000
        while True:
            alone = min(window(ids, fork=False)[0] for _ in range(3))
            if alone >= 0.25 or len(ids) >= 16_000_000:
                break
            ids = ids * 2
        remaining, waited = window(ids, fork=True)
    finally:
        sys.setswitchinterval(interval)
    assert waited > alone / 2, (
        f"the fork waited {waited:.3f}s for a call whose window takes {alone:.3f}s"
        f" alone ({remaining:.3f}s in this run)"
    )


FORK_AFTER_AN_EARLIER_HOOK = """
import os, sys, threading, time
# Registered before smg's hooks, so run after them, as logging's and
# concurrent.futures' are: its sleep releases the GIL between smg's hook and
# the fork, and the thread below calls in.
os.register_at_fork(before=lambda: time.sleep(0.005))
from smg.tokenizer import Tokenizer
tokenizer = Tokenizer(sys.argv[1], model=sys.argv[2])
names = iter(range(10**9))
def request():
    function = {"name": f"tool_{next(names)}", "parameters": {"type": "object"}}
    tool = {"type": "function", "function": function}
    return {"messages": [{"role": "user", "content": "hello"}], "tools": [tool]}
stop = threading.Event()
def keep_calling():
    while not stop.is_set():
        tokenizer.render_chat_batch([request() for _ in range(64)])
caller = threading.Thread(target=keep_calling)
caller.start()
try:
    for _ in range(20):
        pid = os.fork()
        if pid == 0:
            code = 2
            try:
                tokenizer.render_chat_batch([request() for _ in range(8)])
                code = 0
            finally:
                os._exit(code)
        deadline = time.monotonic() + 20
        while True:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                if os.waitstatus_to_exitcode(status) != 0:
                    sys.exit("a child forked after another before hook failed")
                break
            if time.monotonic() > deadline:
                os.kill(pid, 9)
                os.waitpid(pid, 0)
                sys.exit("a child forked after another before hook released the GIL hung")
            time.sleep(0.01)
finally:
    stop.set()
    caller.join()
"""


@pytest.mark.skipif(sys.platform != "linux", reason="os.fork")
def test_a_call_started_while_os_fork_runs_its_hooks_ends_before_the_fork(
    checkpoint: Path,
) -> None:
    # Python runs the before hooks registered ahead of smg's after it, and one
    # that releases the GIL lets another thread start a call between smg's
    # hook and the fork: that call keeps the GIL, so it ends before the fork.
    # Run without the GIL instead, it can still be running at the fork, and
    # the child then waits forever: most runs hang one of the 20 children or
    # more. A fresh interpreter, so that smg's hook is registered after the
    # script's.
    child = subprocess.run(
        [sys.executable, "-c", FORK_AFTER_AN_EARLIER_HOOK, str(checkpoint), MODEL],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert child.returncode == 0, child.stderr


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
    # The child reads the variables when its tokenizer builds the pool. The
    # pool is handed out once every worker has started and named its thread,
    # so the count below does not race the workers' first run.
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


def test_the_api_signature() -> None:
    parameters = inspect.signature(Tokenizer.apply_chat_template).parameters
    assert list(parameters) == [
        "self",
        "conversation",
        "tools",
        "add_generation_prompt",
        "continue_final_message",
        "tokenize",
        "tool_choice",
        "reasoning_effort",
        "template_kwargs",
    ]
    assert parameters["add_generation_prompt"].default is False
    assert parameters["tokenize"].default is True
    assert inspect.signature(Tokenizer.encode).parameters["add_special_tokens"].default is True
    assert inspect.signature(Tokenizer.decode).parameters["skip_special_tokens"].default is False
