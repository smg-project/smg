"""smg.tokenizer.Tokenizer: SMG's tokenizer, encoding and decoding as serving does.

The checkpoints are written by the tests (a byte-level BPE vocabulary of the
256 byte symbols and a few merges, ChatML control tokens, a ChatML template),
so the suite needs no network. What it pins:

- encode and decode follow the checkpoint (merges, special-token text, the
  post-processor's BOS for ``add_special_tokens``), and batches and long texts
  (which run on the binding's thread pool) give what single calls give;
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from packaging.version import Version

pytest.importorskip("smg.smg_rs")

from smg.tokenizer import Tokenizer

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
    return Tokenizer(checkpoint)


HELLO = VOCAB["hello"]
WORLD = VOCAB["Ġworld"]
IM_START = SPECIAL_IDS["<|im_start|>"]
IM_END = SPECIAL_IDS["<|im_end|>"]
ENDOFTEXT = SPECIAL_IDS["<|endoftext|>"]


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
    assert repr(tokenizer) == f"Tokenizer(source={str(checkpoint)!r}, vocab_size=265)"
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


def test_from_pretrained_and_path_objects(checkpoint: Path) -> None:
    tokenizer = Tokenizer.from_pretrained(checkpoint)
    assert tokenizer.source == str(checkpoint)
    assert tokenizer.encode("hello world") == [HELLO, WORLD]


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


def test_load_errors(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="does not contain a valid tokenizer file"):
        Tokenizer(tmp_path)
    # A path that is not there is not read as a model name, even one that
    # looks like an OpenAI model's (which would load tiktoken's encoding).
    for missing in ("/no/such/gpt-4o-distill", "./gpt-4o", "a/b/c"):
        with pytest.raises(FileNotFoundError, match=f"no such file or directory: {missing}"):
            Tokenizer(missing)


# --- processes and threads -----------------------------------------------------


def test_threads_encode_concurrently(tokenizer: Tokenizer) -> None:
    texts = [f"hello world {index}" * 50 for index in range(64)]
    expected = [tokenizer.encode(text) for text in texts]
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(tokenizer.encode, texts)) == expected


def _use(tokenizer: Tokenizer) -> list[int]:
    """Every path that can reach rayon: a long text, a batch, a short text,
    a batch of decodes."""
    return [
        len(tokenizer.encode("hello world " * 10000)),
        len(tokenizer.encode_batch(["hello world"] * 100)),
        len(tokenizer.encode("hello " * 2500)),
        len(tokenizer.decode_batch([[HELLO, WORLD]] * 100)),
    ]


def _use_in_child(checkpoint: str, results: multiprocessing.Queue) -> None:
    results.put(_use(Tokenizer(checkpoint)))


@pytest.mark.skipif(sys.platform != "linux", reason="fork start method")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_child_forked_after_use_does_not_hang(tokenizer: Tokenizer) -> None:
    # Each path, taken in the parent and again in the child: had one created
    # rayon's global pool, the forked child's first use of it would wait
    # forever for threads it did not inherit.
    expected = _use(tokenizer)
    context = multiprocessing.get_context("fork")
    results = context.Queue()
    child = context.Process(target=_use_in_child, args=(tokenizer.source, results))
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
