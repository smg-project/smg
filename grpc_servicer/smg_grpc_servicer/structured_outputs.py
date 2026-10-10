"""The ``structured_outputs_start`` and ``engine_reasoning_parser`` labels: where
an engine starts applying a request's grammar on a prompt that ends inside the
model's thinking block, and the reasoning parser it runs, advertised through
``GetServerInfo`` for the Router to shape a forced tool call's grammar by. An
engine that holds a grammar back until its reasoning parser has seen the
thought's end needs the bare calls, so the call is forced right there, as under
the engine's own server; an engine that applies the grammar from the first
generated token needs the thought and then the calls, so the call is not forced
into the thought. The Router cannot tell the two apart from the outside, so the
servicer that knows its engine says which.

Engine-free at import: the readers take the parsed server arguments. The vLLM
servicers read the same facts off their engine's structured-outputs config
(``smg_grpc_servicer.vllm.model_info``) into two fields of their own proto; the
SGLang and TokenSpeed servicers carry them as ``server_args`` keys, where the
Router reads their other labels.
"""

from __future__ import annotations

from typing import Any

# The `server_args` keys (SGLang, TokenSpeed) and the `GetServerInfo` fields
# (vLLM) the Router reads the labels from. The parser rides under a name of its
# own: `reasoning_parser` is the per-model parser override the Router reads off
# a worker's labels and validates against its own registry at registration, and
# the engines' parser names are the engines'.
ENGINE_REASONING_PARSER_KEY = "engine_reasoning_parser"
STRUCTURED_OUTPUTS_START_KEY = "structured_outputs_start"

# The label's values.
STRUCTURED_OUTPUTS_AFTER_REASONING = "after_reasoning"
STRUCTURED_OUTPUTS_FIRST_TOKEN = "first_token"

# A server-args attribute that is not there at all, as opposed to one set to
# None or "".
_UNKNOWN = object()


def engine_reasoning_parser(server_args: Any) -> str:
    """The engine's reasoning parser (``--reasoning-parser``) as its server
    arguments carry it, for the ``engine_reasoning_parser`` label; "" when it
    runs none."""
    return str(getattr(server_args, "reasoning_parser", None) or "")


def sglang_structured_outputs_start(server_args: Any) -> str:
    """Where SGLang starts a request's grammar on a thinking prompt. With
    ``--reasoning-parser`` the scheduler wraps its grammar backend in a reasoner
    that defers the grammar until the parser's thought-end sequence has been
    generated when the request asks for reasoning (``require_reasoning``, which
    the Router sets on a thinking prompt): ``after_reasoning``. Without a parser
    the grammar runs from the first generated token: ``first_token``. "" when
    the server arguments predate the reasoning parser, which leaves the Router
    to the shape that fits either kind of engine."""
    parser = getattr(server_args, "reasoning_parser", _UNKNOWN)
    if parser is _UNKNOWN:
        return ""
    if parser:
        return STRUCTURED_OUTPUTS_AFTER_REASONING
    return STRUCTURED_OUTPUTS_FIRST_TOKEN


def grammar_start_labels(server_args: Any, start: str) -> dict[str, str]:
    """The two labels as ``server_args`` entries, for the servicers that carry
    their labels there: the parser when the engine runs one, the start when the
    rule says (an empty value is left out, as the Router would drop it)."""
    labels = {}
    parser = engine_reasoning_parser(server_args)
    if parser:
        labels[ENGINE_REASONING_PARSER_KEY] = parser
    if start:
        labels[STRUCTURED_OUTPUTS_START_KEY] = start
    return labels
