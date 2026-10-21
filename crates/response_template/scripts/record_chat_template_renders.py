#!/usr/bin/env python3
# crates/response_template/scripts/record_chat_template_renders.py
"""Record what public chat templates render for a probe conversation.

A response template says how a model's output is read. It does not say which
bytes the model's chat template writes around a tool call: the whitespace a
pattern allows, the wrapper text a template drops, or how argument values are
spelled. Those bytes come from rendering the chat template on a probe
conversation and reading the render back with the compiled response template.
This script records such renders with transformers, so tests replay them and no
Jinja engine runs at test time.

Recorded into `tests/fixtures/chat_templates/`:
  - `probe.json`: the probe tools, the cases (conversations, with or without the
    generation prompt), the template settings and the instant `strftime_now`
    writes;
  - `templates.jsonl`: one row per chat template: its source, the special tokens
    transformers passes to it, the digest of the response template its
    repository declares, and its text;
  - `renders-*.jsonl`: one row per render: chat template, setting, case, how the
    tool call arguments were passed, and the rendered text or the error;
  - `manifest.json`: pinned versions and sources, the settings recorded for each
    chat template, the response templates each one is paired with, and counts.

The chat templates are read from the Hugging Face Hub at pinned revisions, small
files only (`tokenizer_config.json`, `chat_template.jinja` and, where a
repository has one, `special_tokens_map.json`), and every file is checked
against its digest. Each render is made twice: with `render_jinja_template` and
the special tokens transformers passes, and with `apply_chat_template` of a
tokenizer that transformers loads from the same files. The two must agree.

Arguments are passed as objects; a render that raises is recorded with its
error and made again with each call's arguments as a JSON string. A setting
other than the default is recorded for a chat template only when it changes at
least one of its renders.

Pinned: transformers 5.18.0, jinja2 3.1.6 and Python 3.12. The digest of a file
is the first 8 bytes of its SHA-256, one letter per nibble, as in
`generate_hf_fixtures.py`.

Usage (Python 3.12, `pip install transformers==5.18.0 jinja2==3.1.6`):
  record_chat_template_renders.py --hub DIR --fetch   # download the pinned files into DIR, then write
  record_chat_template_renders.py --hub DIR           # write from the files already in DIR
  record_chat_template_renders.py --hub DIR --check   # fail if a fixture is stale
  record_chat_template_renders.py --check             # offline: re-render every copied template
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

PINNED_TRANSFORMERS = "5.18.0"
PINNED_JINJA2 = "3.1.6"
PINNED_PYTHON = (3, 12)

CRATE = Path(__file__).resolve().parents[1]
FIXTURES = CRATE / "tests" / "fixtures" / "chat_templates"
HUB = "https://huggingface.co"
# Every pinned file is small; a larger download is refused.
MAX_FILE_BYTES = 1 << 20

# Chat templates, each pinned to a revision of its Hub repository, with the digest
# of every file read from it and the license its model card declares. `copied`:
# the template text is stored in templates.jsonl, so its renders can be checked
# without the Hub. `settings`: the settings recorded, where they are not chosen
# by comparing renders.
SOURCES: list[dict[str, Any]] = [
    {
        "id": "Qwen/Qwen3.5-9B",
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "license": "apache-2.0",
        "files": {
            "tokenizer_config.json": "khpjkgxptrgsqghz",
            "chat_template.jinja": "tmtyyrtzwzjygqhh",
        },
    },
    {
        "id": "meta-models/Muse-Glimmer-30B",
        "revision": "a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
        "license": "apache-2.0",
        "files": {
            "tokenizer_config.json": "qrhypwqmznqhpmjw",
            "chat_template.jinja": "wzwpqynzkmszkqps",
        },
    },
]

# The response templates read with each chat template's renders, and whether a
# call layout is expected from them. Keys name a response template by its
# source: hub/ the tokenizer_config.json of the repository, serve/ and hf-tests/
# transformers v5.18.0 (src/transformers/cli/serving/utils.py and
# tests/utils/test_chat_parsing.py), trl/ huggingface/trl
# trl/chat_template_utils.py.
PAIRS: list[dict[str, Any]] = [
    {"response_template": "serve/qwen3_5", "chat_template": "Qwen/Qwen3.5-9B", "layout": True},
    {"response_template": "trl/qwen3_5", "chat_template": "Qwen/Qwen3.5-9B", "layout": True},
    {
        "response_template": "hub/muse-glimmer",
        "chat_template": "meta-models/Muse-Glimmer-30B",
        "layout": True,
    },
]

# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------

# The instant `strftime_now` writes, in seconds since the Unix epoch (UTC).
NOW = 0

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fn_alpha",
            "description": "A probe tool with one parameter of each JSON type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "s": {"type": "string"},
                    "i": {"type": "integer"},
                    "n": {"type": "number"},
                    "b": {"type": "boolean"},
                    "a": {"type": "array", "items": {"type": "integer"}},
                    "o": {"type": "object"},
                    "z": {"type": "string", "nullable": True},
                },
                "required": ["s", "i", "n", "b", "a", "o", "z"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fn_beta",
            "description": "A probe tool with one string parameter.",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        },
    },
]

ALPHA = {
    "id": "call_alpha",
    "type": "function",
    "function": {
        "name": "fn_alpha",
        "arguments": {
            "s": "sigma",
            "i": 42,
            "n": 2.5,
            "b": True,
            "a": [1, 2],
            "o": {"m": "mu"},
            "z": None,
        },
    },
}
BETA = {
    "id": "call_beta",
    "type": "function",
    "function": {"name": "fn_beta", "arguments": {"q": "kappa"}},
}
USER = {"role": "user", "content": "Call the probe tools."}
TOOL_RESULT = {
    "role": "tool",
    "tool_call_id": "call_alpha",
    "name": "fn_alpha",
    "content": '{"ok": true}',
}
REASONING = "Probe thought."
ANSWER = '{"probe": 1}'
# The message fields chat templates read a turn's reasoning from.
REASONING_FIELDS = ("reasoning_content", "reasoning", "thinking")

# Template keyword arguments, beside the special tokens.
SETTINGS: dict[str, dict[str, Any]] = {
    "default": {},
    "enable_thinking": {"enable_thinking": True},
    "disable_thinking": {"enable_thinking": False},
}


def cases() -> dict[str, dict[str, Any]]:
    """The conversations rendered for every chat template, by name."""

    def turn(**fields: Any) -> list[dict[str, Any]]:
        return [USER, {"role": "assistant", **fields}]

    def case(messages: list[dict[str, Any]], generation_prompt: bool) -> dict[str, Any]:
        return {"messages": messages, "add_generation_prompt": generation_prompt}

    out = {
        "prompt": case([USER], True),
        "prompt_after_tool": case(turn(tool_calls=[ALPHA]) + [TOOL_RESULT], True),
        "calls": case(turn(tool_calls=[ALPHA, BETA]), False),
        "one_call": case(turn(tool_calls=[ALPHA]), False),
        "answer": case(turn(content=ANSWER), False),
    }
    for field in REASONING_FIELDS:
        out[f"calls+{field}"] = case(turn(**{field: REASONING}, tool_calls=[ALPHA, BETA]), False)
    for field in REASONING_FIELDS:
        out[f"one_call+{field}"] = case(turn(**{field: REASONING}, tool_calls=[ALPHA]), False)
    for field in REASONING_FIELDS:
        out[f"answer+{field}"] = case(turn(**{field: REASONING}, content=ANSWER), False)
    return copy.deepcopy(out)


CASES = cases()


def has_calls(case: dict[str, Any]) -> bool:
    return any(m.get("tool_calls") for m in case["messages"])


def with_string_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The messages with every tool call's arguments written as compact JSON text."""
    out = copy.deepcopy(messages)
    for message in out:
        for call in message.get("tool_calls", []):
            arguments = call["function"]["arguments"]
            call["function"]["arguments"] = json.dumps(
                arguments, ensure_ascii=False, separators=(",", ":")
            )
    return out


def probe() -> dict[str, Any]:
    return {
        "tools": TOOLS,
        "cases": CASES,
        "settings": SETTINGS,
        "arguments": {
            "object": "each tool call's arguments as a JSON object",
            "string": "each tool call's arguments as compact JSON text "
            "(json.dumps with separators (',', ':')), tried when the object form raises",
        },
        "now": NOW,
        "notes": [
            "strftime_now writes the instant `now`, in seconds since the Unix epoch, as UTC.",
            "A setting missing from a chat template's settings in manifest.json renders "
            "every case as the default setting does.",
            "Every render passes the tools; the chat template's special tokens come from "
            "templates.jsonl and the setting's keyword arguments are added to them.",
        ],
    }


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

# Digests are written with 16 letters for the 16 nibbles, so a digest never reads
# as a number, a hex string or a word.
DIGEST_LETTERS = "ghjkmnpqrstvwxyz"


def digest_bytes(data: bytes) -> str:
    return "".join(
        DIGEST_LETTERS[b >> 4] + DIGEST_LETTERS[b & 15] for b in hashlib.sha256(data).digest()[:8]
    )


def digest_json(value: Any) -> str:
    """The digest of a JSON value's canonical text: sorted keys, no spaces."""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return digest_bytes(text.encode("utf-8"))


def line(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"


def pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=1) + "\n"


def shards(name: str, rows: list[dict[str, Any]], max_bytes: int = 450_000) -> dict[str, str]:
    """Split rows into files below the repository's large-file limit."""
    out: dict[str, str] = {}
    chunk, size, k = [], 0, 0
    for row in rows:
        text = line(row)
        if chunk and size + len(text.encode()) > max_bytes:
            out[f"{name}-{k}.jsonl"] = "".join(chunk)
            chunk, size, k = [], 0, k + 1
        chunk.append(text)
        size += len(text.encode())
    out[f"{name}-{k}.jsonl"] = "".join(chunk)
    return out


def slug(repo: str) -> str:
    return repo.replace("/", "__")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def check_versions() -> None:
    import jinja2
    import transformers

    if transformers.__version__ != PINNED_TRANSFORMERS:
        raise SystemExit(f"transformers {transformers.__version__} != {PINNED_TRANSFORMERS}")
    if jinja2.__version__ != PINNED_JINJA2:
        raise SystemExit(f"jinja2 {jinja2.__version__} != {PINNED_JINJA2}")
    if sys.version_info[:2] != PINNED_PYTHON:
        raise SystemExit(f"Python {sys.version_info[:2]} != {PINNED_PYTHON}")


def strftime_now(fmt: str) -> str:
    """transformers' `strftime_now` at the fixed instant `NOW`."""
    now = datetime.datetime.fromtimestamp(NOW, tz=datetime.UTC).replace(tzinfo=None)
    return now.strftime(fmt)


def case_messages(case: dict[str, Any], arguments: str | None) -> list[dict[str, Any]]:
    if arguments == "string":
        return with_string_arguments(case["messages"])
    return copy.deepcopy(case["messages"])


def template_kwargs(setting: str) -> dict[str, Any]:
    return {**SETTINGS[setting], "strftime_now": strftime_now}


def render(
    template: str,
    special_tokens: dict[str, str],
    case: dict[str, Any],
    setting: str,
    arguments: str | None,
) -> str:
    """One render, with the keyword arguments `apply_chat_template` passes: the
    special tokens, then the setting's."""
    from transformers.utils.chat_template_utils import render_jinja_template

    rendered, _ = render_jinja_template(
        conversations=[case_messages(case, arguments)],
        tools=copy.deepcopy(TOOLS),
        chat_template=template,
        add_generation_prompt=case["add_generation_prompt"],
        **{**special_tokens, **template_kwargs(setting)},
    )
    return rendered[0]


def apply(tokenizer: Any, case: dict[str, Any], setting: str, arguments: str | None) -> str:
    """The same render through the tokenizer's `apply_chat_template`."""
    return tokenizer.apply_chat_template(
        case_messages(case, arguments),
        tools=copy.deepcopy(TOOLS),
        add_generation_prompt=case["add_generation_prompt"],
        tokenize=False,
        **template_kwargs(setting),
    )


def attempt(fn: Any, *args: Any) -> tuple[str | None, dict[str, str] | None]:
    """The text a render returns, or the error it raises."""
    try:
        return fn(*args), None
    except Exception as e:  # noqa: BLE001
        return None, {"type": type(e).__name__, "message": str(e)}


def render_rows(
    source_id: str,
    template: str,
    special_tokens: dict[str, str],
    setting: str,
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    """Every case of one setting: the object-argument render, and after an error
    the string-argument render. With a tokenizer, `apply_chat_template` must give
    the same text or raise the same error."""
    rows = []
    for name, case in CASES.items():
        for arguments in ("object", "string") if has_calls(case) else (None,):
            text, error = attempt(render, template, special_tokens, case, setting, arguments)
            if tokenizer is not None:
                again = attempt(apply, tokenizer, case, setting, arguments)
                if again != (text, error):
                    raise SystemExit(f"{source_id} {setting} {name}: apply_chat_template differs")
            row = {"template": source_id, "setting": setting, "case": name, "arguments": arguments}
            if error is None:
                rows.append({**row, "text": text})
                break
            rows.append({**row, "error": error})
    return rows


def record_settings(source: dict[str, Any], rows_by_setting: dict[str, list]) -> list[str]:
    """The settings recorded: the default, and every other that changes a render."""
    if "settings" in source:
        return list(source["settings"])

    def key(rows: list[dict[str, Any]]) -> list:
        return [{k: v for k, v in r.items() if k != "setting"} for r in rows]

    default = key(rows_by_setting["default"])
    return ["default"] + [
        name for name in SETTINGS if name != "default" and key(rows_by_setting[name]) != default
    ]


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def fetch(hub: Path) -> None:
    """Download every pinned file into `hub`, one directory per repository."""
    import urllib.request

    for source in SOURCES:
        directory = hub / slug(source["id"])
        directory.mkdir(parents=True, exist_ok=True)
        for name, digest in source["files"].items():
            path = directory / name
            if path.exists() and digest_bytes(path.read_bytes()) == digest:
                continue
            url = f"{HUB}/{source['id']}/resolve/{source['revision']}/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise SystemExit(f"{url}: larger than {MAX_FILE_BYTES} bytes")
            if digest_bytes(data) != digest:
                raise SystemExit(f"{url}: digest {digest_bytes(data)} != {digest}")
            path.write_bytes(data)
            print(f"fetched {source['id']} {name}")


def config_only_tokenizer_class() -> type:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    class ConfigOnlyTokenizer(PreTrainedTokenizerBase):
        """transformers' tokenizer base, loaded from tokenizer_config.json without a
        vocabulary: enough for `apply_chat_template(tokenize=False)`, which reads only
        the chat template and the special tokens."""

        added_tokens_decoder: dict = {}

    return ConfigOnlyTokenizer


def load_hub(source: dict[str, Any], hub: Path) -> tuple[Any, str, dict[str, str], Any]:
    """The tokenizer, chat template, special tokens and declared response template
    that transformers loads from the pinned files of one repository."""
    directory = hub / slug(source["id"])
    present = sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []
    if present != sorted(source["files"]):
        raise SystemExit(f"{directory}: holds {present}, expected {sorted(source['files'])}")
    for name, digest in source["files"].items():
        found = digest_bytes((directory / name).read_bytes())
        if found != digest:
            raise SystemExit(f"{directory / name}: digest {found} != {digest}")
    tokenizer = config_only_tokenizer_class().from_pretrained(str(directory), local_files_only=True)
    template = tokenizer.chat_template
    if template != (directory / "chat_template.jinja").read_bytes().decode("utf-8"):
        raise SystemExit(f"{directory}: transformers reads another chat template")
    config = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
    return tokenizer, template, dict(tokenizer.special_tokens_map), config.get("response_template")


def source_line(source: dict[str, Any]) -> str:
    return (
        f"{HUB}/{source['id']} at revision {source['revision'][:12]}, "
        f"chat_template.jinja ({source['license']})"
    )


def generate(hub: Path | None) -> dict[str, str]:
    """Every fixture file by name. Without `hub`, the copied templates are read from
    the committed templates.jsonl and the renders of the others are kept as
    committed."""
    check_versions()
    from transformers.utils import logging

    logging.set_verbosity_error()
    committed_templates: dict[str, dict[str, Any]] = {}
    committed_renders: list[dict[str, Any]] = []
    if hub is None:
        committed_templates = {
            r["id"]: r for r in read_jsonl(FIXTURES / "templates.jsonl") if r["template"]
        }
        committed_renders = [
            r for p in sorted(FIXTURES.glob("renders-*.jsonl")) for r in read_jsonl(p)
        ]

    templates, renders, recorded = [], [], {}
    for source in SOURCES:
        copied = source.get("copied", True)
        if hub is not None:
            tokenizer, template, special_tokens, declared = load_hub(source, hub)
            response_template = None if declared is None else digest_json(declared)
        elif copied:
            row = committed_templates[source["id"]]
            tokenizer, template, special_tokens = None, row["template"], row["special_tokens"]
            response_template = row["response_template"]
            if digest_bytes(template.encode("utf-8")) != source["files"]["chat_template.jinja"]:
                raise SystemExit(f"templates.jsonl: {source['id']} is not the pinned template")
        else:
            row = next(
                r for r in read_jsonl(FIXTURES / "templates.jsonl") if r["id"] == source["id"]
            )
            templates.append(row)
            rows = [r for r in committed_renders if r["template"] == source["id"]]
            recorded[source["id"]] = list(dict.fromkeys(r["setting"] for r in rows))
            renders += rows
            continue
        templates.append(
            {
                "id": source["id"],
                "source": source_line(source),
                "special_tokens": special_tokens,
                "response_template": response_template,
                "template": template if copied else None,
            }
        )
        wanted = source.get("settings", list(SETTINGS))
        by_setting = {
            setting: render_rows(source["id"], template, special_tokens, setting, tokenizer)
            for setting in wanted
        }
        recorded[source["id"]] = record_settings(source, by_setting)
        for setting in recorded[source["id"]]:
            renders += by_setting[setting]

    files = {"probe.json": pretty(probe())}
    files["templates.jsonl"] = "".join(line(t) for t in templates)
    files.update(shards("renders", renders))
    files["manifest.json"] = pretty(manifest(templates, renders, recorded))
    return files


def manifest(
    templates: list[dict[str, Any]], renders: list[dict[str, Any]], recorded: dict[str, list[str]]
) -> dict[str, Any]:
    return {
        "transformers": PINNED_TRANSFORMERS,
        "jinja2": PINNED_JINJA2,
        "python": ".".join(map(str, PINNED_PYTHON)),
        "generator": digest_bytes(Path(__file__).read_bytes()),
        "now": NOW,
        "sources": [
            {
                "id": s["id"],
                "revision": s["revision"][:12],
                "license": s["license"],
                "copied": s.get("copied", True),
                "settings": recorded[s["id"]],
                "files": s["files"],
            }
            for s in SOURCES
        ],
        "pairs": PAIRS,
        "counts": {
            "templates": len(templates),
            "copied": sum(1 for t in templates if t["template"] is not None),
            "renders": len(renders),
            "render_errors": sum(1 for r in renders if "error" in r),
        },
        "digest": "sha256 of the file bytes (of the canonical JSON for a response template), "
        "first 8 bytes, nibbles as ghjkmnpqrstvwxyz",
        "license": "The chat templates and their renders come from the Hugging Face repositories "
        "named in sources, under the licenses their model cards declare.",
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def self_check(files: dict[str, str]) -> None:
    """The fixtures are public: no long hex runs, and no date that a copied
    template does not itself write."""
    templates = [json.loads(x)["template"] or "" for x in files["templates.jsonl"].splitlines()]
    hex_run = re.compile(r"\b(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\b")
    date = re.compile(r"\b20[0-9]{2}-[01][0-9]-[0-3][0-9]\b")
    for name, text in files.items():
        m = hex_run.search(text)
        if m:
            raise SystemExit(f"{name}: output contains {m.group(0)!r}")
        for m in date.finditer(text):
            if not any(m.group(0) in t for t in templates):
                raise SystemExit(f"{name}: output contains {m.group(0)!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hub", type=Path, help="directory with the pinned Hub files")
    parser.add_argument("--fetch", action="store_true", help="download the pinned files first")
    parser.add_argument("--check", action="store_true", help="fail if a fixture is stale")
    args = parser.parse_args()
    if args.fetch:
        if not args.hub:
            parser.error("--fetch needs --hub")
        fetch(args.hub)
    if not args.hub and not args.check:
        parser.error("writing needs --hub")
    files = generate(args.hub)
    self_check(files)
    if args.check:
        existing = {p.name for p in FIXTURES.glob("*")} if FIXTURES.exists() else set()
        stale = [
            n
            for n, text in files.items()
            if n not in existing or (FIXTURES / n).read_text(encoding="utf-8") != text
        ]
        stale += sorted(existing - set(files))
        if stale:
            print(f"stale: {', '.join(stale)}; rerun {Path(__file__).name}", file=sys.stderr)
            return 1
        print(f"{FIXTURES}: up to date ({'with' if args.hub else 'without'} the Hub files)")
        return 0
    FIXTURES.mkdir(parents=True, exist_ok=True)
    leftover = sorted(p.name for p in FIXTURES.glob("*") if p.name not in files)
    if leftover:
        print(f"remove by hand, no longer generated: {', '.join(leftover)}", file=sys.stderr)
    for name, text in files.items():
        (FIXTURES / name).write_text(text, encoding="utf-8")
    counts = json.loads(files["manifest.json"])["counts"]
    print(f"wrote {FIXTURES}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
