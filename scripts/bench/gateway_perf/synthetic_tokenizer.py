#!/usr/bin/env python3
"""Write a small, self-contained tokenizer directory for the gateway perf bed.

The bed needs a tokenizer with a chat template so the gateway renders and encodes
every request the way it does in production; fetching one from a model hub makes a
CI job depend on the network and on a third party's rate limits. This writes one
from nothing, in the shape the gateway's own encoder takes (the shape of the common
byte-level BPE tokenizers: a BPE model over the 256 byte symbols, the ByteLevel
pre-tokenizer with its word-splitting pattern, an NFC normalizer, a ByteLevel decoder),
with merges that make every word of the load client's deterministic prompts, tool
schemas and scripted answers one token, in both spellings (after a space and bare)
and in title case. Anything else encodes to its bytes: no unknown token, every text
decodes back to itself. Role markers are added special tokens.

The numbers measured on it are relative (base against head on the same tokenizer).
A real tokenizer directory can be passed instead (`TOKENIZER=<dir or hub id>`) when
absolute per-request costs with a production template are the question.

    synthetic_tokenizer.py <out dir> [--words FILE ...]

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The load client's prompt vocabulary (crates/mock_worker/src/bin/stress_load.rs,
# VOCAB_TEXT), kept in one place here: the bed's prompts encode without an unknown.
PROMPT_WORDS = (
    "the of and to in is that for it as with on be by this are from or an at which have not were but "
    "one all their there when who will more if out so what up its about into than them can only other "
    "new some could time these two may then do first any my now such like our over man me even most "
    "made after also did many before must through back years where much your way well down should "
    "because each just those people how too little state good very make world still own see men work "
    "long get here between both life being under never day same another know while last might us "
    "great old year off come since against go came right used take three states himself few house use "
    "during without again place around however home small found thought went say part once general "
    "high upon school every does got united left number course war until always away something fact "
    "though water less public put think almost hand enough far took head yet government system better "
    "set told nothing night end why called eyes find going look asked later knew point next program "
    "city business give group toward young days let room president side social given present several "
    "order national second possible rather per face among form important often things looked early "
    "white case become large need big four within felt along children saw best church ever least "
    "power development light thing seemed family interest want members mind country area others "
    "turned although open service certain kind problem began different door thus help means sense "
    "whole matter perhaps itself times human law line above name example action company local show "
    "whether five history gave today either act feet across taken past quite anything seen having "
    "death week body word field experience really class already money free information table value "
    "cost market growth design test data model result process report rate level change read write "
    "file code error check build run"
).split()

# Words of the request bodies around the prompts: the roles, the tool schemas and the
# tool results the load client builds, the JSON-schema answer format, the scripted
# answers of answers.py, and the template's own words.
STRUCTURE_WORDS = """
system user assistant tool tools Tools developer content role function type object string integer
array items properties required additionalProperties minimum maximum enum format description
parameters arguments name key fields limit filter since status open closed any ok unsure answer
confidence strict schema json_schema date time record returns looks asked lookup Tool call id
call_id tool_call_id tool_calls a b c k true false null quick brown fox jumps over lazy dog
The An A is was ready done fine lookup_0 lookup_1 lookup_2 lookup_3 lookup_4 lookup_5 lookup_6
lookup_7 lookup_8 lookup_9 lookup_10 lookup_11 lookup_12 lookup_13 lookup_14 lookup_15 mock
model m1 text image_url input_text output_text
""".split()

SPECIAL = ["<|start|>", "<|end|>", "<|pad|>"]
SPACE = "Ġ"  # the byte-level symbol of a space (the GPT-2 byte alphabet)
# The word-splitting pattern of the common byte-level tokenizers: contractions, a
# letter run with one leading non-letter, single digits, punctuation runs, newlines,
# then spaces (the space before a word stays with the word).
SPLIT_PATTERN = (
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
    "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)

CHAT_TEMPLATE = (
    "{%- if tools %}"
    "{{ '<|start|>system\\n' }}"
    "{%- if messages and messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}{{ '\\n' }}{%- endif %}"
    "{{ 'Tools:\\n' }}"
    "{%- for tool in tools %}{{ tool | tojson }}{{ '\\n' }}{%- endfor %}"
    "{{ '<|end|>\\n' }}"
    "{%- endif %}"
    "{%- for message in messages %}"
    "{%- if not (message['role'] == 'system' and tools) %}"
    "{{ '<|start|>' + message['role'] + '\\n' }}"
    "{%- if message['content'] is string %}{{ message['content'] }}"
    "{%- elif message['content'] is iterable %}"
    "{%- for part in message['content'] %}"
    "{%- if part['type'] == 'text' %}{{ part['text'] }}{%- endif %}"
    "{%- endfor %}"
    "{%- endif %}"
    "{%- if message['tool_calls'] %}"
    "{%- for call in message['tool_calls'] %}{{ '\\n' }}{{ call['function'] | tojson }}{%- endfor %}"
    "{%- endif %}"
    "{{ '<|end|>\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|start|>assistant\\n' }}{%- endif %}"
)


def byte_alphabet() -> list[str]:
    """The 256 byte symbols of byte-level BPE (GPT-2's `bytes_to_unicode`): printable
    ASCII and Latin-1 letters stand for themselves, the other bytes get code points
    from 256 up, in byte order."""
    keep = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(0xA1, 0xAC + 1))
        + list(range(0xAE, 0xFF + 1))
    )
    symbols = {b: chr(b) for b in keep}
    n = 0
    for b in range(256):
        if b not in symbols:
            symbols[b] = chr(256 + n)
            n += 1
    return [symbols[b] for b in range(256)]


def to_symbols(word: str, alphabet: list[str]) -> list[str]:
    return [alphabet[b] for b in word.encode("utf-8")]


def words_of(extra_words: list[str]) -> list[str]:
    """Every word that gets a merge chain: the prompt vocabulary, the structure words,
    the extras, each also in title case; sorted so the output is deterministic."""
    words = set(PROMPT_WORDS) | set(STRUCTURE_WORDS) | set(extra_words)
    words |= {word.capitalize() for word in words}
    return sorted(words)


def merges_for(words: list[str], alphabet: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """The vocabulary and the ordered merges: for every word, after a space (the
    common spelling inside a text) and bare (the first word of a text), the chain
    that joins its byte symbols left to right. A BPE applies the lowest-ranked pair
    present, so the order of the chains decides what wins where words overlap: every
    spaced spelling's chain before every bare one (a chain of a spaced word only
    ever meets its own prefixes), and longer words before shorter ones within each
    group, so a word's own chain outranks the chains of the shorter words inside it.
    Every spaced spelling is then one token; a bare word that contains a prefix of
    a longer listed word may take two, at the start of a text only."""
    by_length = sorted(words, key=lambda w: (-len(w), w))
    chains = [[SPACE, *to_symbols(w, alphabet)] for w in by_length] + [
        to_symbols(w, alphabet) for w in by_length
    ]
    vocab: list[str] = list(alphabet)
    seen = set(vocab)
    merges: list[tuple[str, str]] = []
    seen_merges: set[tuple[str, str]] = set()
    for chain in chains:
        for position in range(len(chain) - 1):
            left = "".join(chain[: position + 1])
            right = chain[position + 1]
            pair = (left, right)
            if pair in seen_merges:
                continue
            seen_merges.add(pair)
            merges.append(pair)
            joined = left + right
            if joined not in seen:
                seen.add(joined)
                vocab.append(joined)
    return vocab, merges


def tokenizer_json(vocab: list[str], merges: list[tuple[str, str]]) -> dict:
    ids = {token: i for i, token in enumerate(vocab)}
    added = [
        {
            "id": len(vocab) + n,
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for n, token in enumerate(SPECIAL)
    ]
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": SPLIT_PATTERN},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": False,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": ids,
            "merges": [f"{a} {b}" for a, b in merges],
        },
    }


def tokenizer_config() -> dict:
    return {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 262144,
        "add_bos_token": False,
        "add_prefix_space": False,
        "bos_token": None,
        "eos_token": "<|end|>",
        "pad_token": "<|pad|>",
        "unk_token": None,
        "additional_special_tokens": ["<|start|>", "<|end|>"],
        "clean_up_tokenization_spaces": False,
        "split_special_tokens": False,
        "chat_template": CHAT_TEMPLATE,
    }


def write(out: Path, extra_words: list[str]) -> list[str]:
    """Write the directory; returns the vocabulary (the specials last)."""
    out.mkdir(parents=True, exist_ok=True)
    alphabet = byte_alphabet()
    vocab, merges = merges_for(words_of(extra_words), alphabet)
    (out / "tokenizer.json").write_text(
        json.dumps(tokenizer_json(vocab, merges), indent=1, ensure_ascii=False) + "\n"
    )
    (out / "tokenizer_config.json").write_text(json.dumps(tokenizer_config(), indent=1) + "\n")
    (out / "special_tokens_map.json").write_text(
        json.dumps({"eos_token": "<|end|>", "pad_token": "<|pad|>"}, indent=1) + "\n"
    )
    end_id = len(vocab) + SPECIAL.index("<|end|>")
    (out / "config.json").write_text(
        json.dumps(
            {
                "model_type": "synthetic",
                "eos_token_id": end_id,
                "vocab_size": len(vocab) + len(SPECIAL),
            },
            indent=1,
        )
        + "\n"
    )
    return vocab + SPECIAL


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", help="the tokenizer directory to write")
    parser.add_argument(
        "--words",
        action="append",
        default=[],
        metavar="FILE",
        help="a text file whose whitespace-separated words join the vocabulary (repeatable)",
    )
    args = parser.parse_args(argv)
    extra: list[str] = []
    for path in args.words:
        extra.extend(Path(path).read_text().split())
    tokens = write(Path(args.out), extra)
    print(
        f"{args.out}: {len(tokens)} tokens (byte-level BPE), eos <|end|>, chat template with tools"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
