#!/usr/bin/env python3
"""Compile every structural tag Symphony derives with xgrammar, the engines' grammar library.

The README test writes one payload per table and request shape when ``SYMPHONY_WRITE_TAGS`` names
a directory::

    SYMPHONY_WRITE_TAGS=/tmp/tags cargo test -p smg-symphony --test readme structural_tag
    python3 crates/symphony/tools/compile_structural_tags.py /tmp/tags

Each payload is what a request carries to the engine (``{"format": ...}``); xgrammar reads it as a
``StructuralTag`` (its ``type`` defaults) and builds the grammar, which is where a tag an engine
could not take fails: an unknown node, a field of the wrong shape, a regex or a JSON schema it does
not accept. With ``--tokenizer`` (a Hugging Face id or a local path, ``transformers`` installed),
each grammar is compiled against that tokenizer too, which exercises the token-level matcher.

Exit status 0 when every payload compiles, 1 otherwise; one line per payload.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=pathlib.Path, help="the payloads the README test wrote")
    parser.add_argument("--tokenizer", help="compile against this tokenizer as well")
    args = parser.parse_args()

    import xgrammar as xgr  # the engines' grammar library; pip install xgrammar

    compiler = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer))

    # The files the README test writes, by their endings, and no other JSON the directory holds.
    endings = (
        "-required-closed.json",
        "-optional-closed.json",
        "-required-open.json",
        "-optional-open.json",
    )
    paths = sorted(path for path in args.directory.glob("*.json") if path.name.endswith(endings))
    if not paths:
        print(f"no payloads in {args.directory}", file=sys.stderr)
        return 1
    failures = 0
    for path in paths:
        started = time.perf_counter()
        try:
            text = path.read_text(encoding="utf-8")
            grammar = xgr.Grammar.from_structural_tag(json.loads(text))
            if compiler is not None:
                compiler.compile_grammar(grammar)
            verdict = "ok"
        except Exception as error:  # noqa: BLE001 - every failure is reported, whatever its kind
            failures += 1
            first_line = (str(error).splitlines() or [""])[0]
            verdict = f"FAILED {type(error).__name__}: {first_line[:200]}"
        elapsed = (time.perf_counter() - started) * 1000
        print(f"{path.name}: {verdict} ({elapsed:.0f} ms)")
    print(f"{len(paths)} payloads, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
