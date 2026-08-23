#!/usr/bin/env python3
"""Register a model that ``bfcl-eval`` doesn't ship a handler for yet.

The BFCL leaderboard package pins a fixed ``MODEL_CONFIG_MAPPING``; brand-new
models (e.g. ``Qwen/Qwen3.6-27B``, released after the package was cut) aren't in
it, so ``bfcl generate --model <id>-FC`` fails with "Unknown model_name". For an
A/B against a self-hosted OpenAI-compatible endpoint we register the new id with
``OpenAICompletionsHandler`` — the generic OpenAI Chat Completions FC handler: it
sends native ``tools`` and reads the server's parsed ``tool_calls`` (keeping the
server parser on the critical path), and across multi-turn it re-inserts tool
calls as proper OpenAI messages. The model-family *local* handlers (e.g.
``QwenFCHandler``) instead re-serialize multi-turn tool calls into a text prompt
assuming a ``{"name","arguments"}`` shape, which ``KeyError``s on SKUs whose
tool-call JSON differs (DeepSeek-V4, MiniMax-M2). bfcl reads the endpoint from
``OPENAI_BASE_URL`` (set per arm by run_ab.py).

This edits the *installed* ``bfcl_eval/constants/model_config.py`` in place
(idempotent). Re-running is safe. Intended for nightly "test the latest models"
flows where the bfcl release lags new releases.

    python register_bfcl_model.py --model-id Qwen/Qwen3.6-27B
    # registers "Qwen/Qwen3.6-27B-FC" with OpenAICompletionsHandler
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

DEFAULT_ANCHOR = '    "Qwen/Qwen3-32B-FC": ModelConfig('


def find_model_config() -> Path:
    spec = importlib.util.find_spec("bfcl_eval.constants.model_config")
    if spec is None or spec.origin is None:
        raise SystemExit("bfcl_eval not importable in this interpreter")
    return Path(spec.origin)


# Handlers whose tool payload rewrites "." to "_" in function names, per
# bfcl_eval.model_handler.utils.convert_to_tool. For these the checker must
# apply the same rewrite to the expected name, which is what `underscore_to_dot`
# switches on — its own docstring says it "only matters for checker". Setting it
# False alongside one of these handlers is internally inconsistent: the model is
# asked for `math_factorial` and then graded against `math.factorial`, so every
# dotted-name case is scored wrong no matter what the model or the frontend did.
DOT_SANITIZING_HANDLERS = {
    "OpenAICompletionsHandler",
    "OpenAIResponsesHandler",
    "MistralHandler",
    "GoogleHandler",
    "OSSHandler",
    "AnthropicHandler",
    "CohereHandler",
    "AmazonHandler",
    "NovitaHandler",
}


def build_entry(model_id: str, handler: str) -> str:
    return (
        f'    "{model_id}-FC": ModelConfig(\n'
        f'        model_name="{model_id}",\n'
        f'        display_name="{model_id.split("/")[-1]} (FC)",\n'
        f'        url="https://huggingface.co/{model_id}",\n'
        f'        org="{model_id.split("/")[0]}",\n'
        f'        license="apache-2.0",\n'
        f"        model_handler={handler},\n"
        f"        input_price=None,\n"
        f"        output_price=None,\n"
        f"        is_fc_model=True,\n"
        f"        underscore_to_dot={handler in DOT_SANITIZING_HANDLERS},\n"
        f"    ),\n"
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-id", required=True, help="HF id, e.g. Qwen/Qwen3.6-27B")
    p.add_argument(
        "--handler",
        default="OpenAICompletionsHandler",
        help="bfcl handler class already imported in model_config.py",
    )
    p.add_argument("--anchor", default=DEFAULT_ANCHOR, help="existing entry line to insert before")
    args = p.parse_args()

    path = find_model_config()
    src = path.read_text(encoding="utf-8")
    key = f'"{args.model_id}-FC":'
    if key in src:
        print(f"already registered: {args.model_id}-FC")
        return 0
    if args.anchor not in src:
        raise SystemExit(
            f"anchor not found in {path}; pass a valid --anchor (an existing entry line)"
        )
    if args.handler not in src:
        raise SystemExit(
            f"handler {args.handler} is not referenced in {path}; pick one that is imported there"
        )

    entry = build_entry(args.model_id, args.handler)
    src = src.replace(args.anchor, entry + args.anchor, 1)
    path.write_text(src, encoding="utf-8")
    print(f"registered {args.model_id}-FC (handler={args.handler}) in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
