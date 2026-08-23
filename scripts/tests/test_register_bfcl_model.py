"""Tests for scripts/bfcl/register_bfcl_model.py.

The entry this script writes decides how BFCL grades a model. One field in it,
``underscore_to_dot``, is easy to get wrong and fails silently: nothing errors,
the run completes, and the score is simply too low.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "bfcl" / "register_bfcl_model.py"


def _load():
    spec = importlib.util.spec_from_file_location("register_bfcl_model", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["register_bfcl_model"] = module
    spec.loader.exec_module(module)
    return module


def _field(entry: str, name: str) -> str:
    for line in entry.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{name}="):
            return stripped.rstrip(",").split("=", 1)[1]
    raise AssertionError(f"{name} missing from entry:\n{entry}")


@pytest.mark.parametrize(
    "handler",
    ["OpenAICompletionsHandler", "OpenAIResponsesHandler", "AnthropicHandler"],
)
def test_dot_sanitizing_handlers_enable_underscore_to_dot(handler: str) -> None:
    """A handler that rewrites dots must grade against rewritten names.

    ``convert_to_tool`` replaces "." with "_" in function names for these
    styles, so the model is only ever offered ``math_factorial``. The checker
    applies the same rewrite to the expected name only when this flag is set;
    without it every dotted-name case is marked wrong_func_name regardless of
    what the model or the frontend did.
    """
    entry = _load().build_entry("org/Some-Model", handler)
    assert _field(entry, "underscore_to_dot") == "True"


def test_non_sanitizing_handler_leaves_names_alone() -> None:
    """A handler that passes dotted names through must not rewrite expectations."""
    entry = _load().build_entry("org/Some-Model", "SomeLocalFCHandler")
    assert _field(entry, "underscore_to_dot") == "False"


def test_entry_shape_is_registerable() -> None:
    """The generated entry must be a syntactically valid mapping member."""
    module = _load()
    entry = module.build_entry("meta-models/Muse-Glimmer-30B", "OpenAICompletionsHandler")

    assert entry.lstrip().startswith('"meta-models/Muse-Glimmer-30B-FC": ModelConfig(')
    assert _field(entry, "model_name") == '"meta-models/Muse-Glimmer-30B"'
    assert _field(entry, "is_fc_model") == "True"
    # Parses as Python once the ModelConfig call is stubbed out.
    compile(f"ModelConfig = dict\nx = {{\n{entry}}}\n", "<entry>", "exec")
