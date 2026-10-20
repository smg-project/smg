"""The ``constrained_decoding_start`` and ``engine_reasoning_parser`` labels: where
an engine starts applying a request's grammar on a prompt that ends inside the
model's thinking block, and the reasoning parser it runs, advertised through
``GetServerInfo`` for the Router to shape a forced tool call's grammar by. An
engine that holds a grammar back until its reasoning parser has seen the
thought's end needs the bare calls, so the call is forced right there, as under
the engine's own server; an engine that applies the grammar from the first
generated token needs the thought and then the calls, so the call is not forced
into the thought. The Router cannot tell the two apart from the outside, so the
servicer that knows its engine says which.

The ``grammar_backend`` key beside them: the grammar backend the engine runs,
as its server arguments resolve it, so the Rust servicer refuses a constraint
that backend cannot enforce before the engine (every kind under ``none``, a kind
the backend does not compile, a JSON-schema feature xgrammar drops), the same
refusal the vLLM path gives.

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
CONSTRAINED_DECODING_START_KEY = "constrained_decoding_start"

# The label's values.
CONSTRAINED_DECODING_AFTER_REASONING = "after_reasoning"
CONSTRAINED_DECODING_FIRST_TOKEN = "first_token"

# The `server_args` key (SGLang, TokenSpeed) the Rust servicer reads the
# engine's grammar backend from: the engine's own argument name, carrying the
# value the engine resolved rather than the raw argument.
GRAMMAR_BACKEND_KEY = "grammar_backend"
# The engines' spelling of no grammar backend at all.
GRAMMAR_BACKEND_NONE = "none"
# What SGLang runs when `--grammar-backend` is not given.
SGLANG_DEFAULT_GRAMMAR_BACKEND = "xgrammar"

# A server-args attribute that is not there at all, as opposed to one set to
# None or "".
_UNKNOWN = object()


def _resolved_argument(server_args: Any, name: str) -> Any:
    """The value the engine runs for argument ``name``: the last value its
    argument resolution declared for it (SGLang keeps an argument as typed and
    records what its resolution hooks decided, ``--reasoning-parser auto`` to a
    name or None by template detection, an unset ``--grammar-backend`` to the
    default, in ``_resolved_overrides``, last writer wins, which is how its own
    serving view reads them), else the raw argument; ``_UNKNOWN`` when the
    server arguments have no such field."""
    for _source, declared in reversed(getattr(server_args, "_resolved_overrides", None) or ()):
        if name in declared:
            return declared[name]
    return getattr(server_args, name, _UNKNOWN)


def _resolved_reasoning_parser(server_args: Any) -> Any:
    """The reasoning parser the engine runs (:func:`_resolved_argument`)."""
    return _resolved_argument(server_args, "reasoning_parser")


def engine_reasoning_parser(server_args: Any) -> str:
    """The engine's reasoning parser (``--reasoning-parser``) as its server
    arguments carry it, for the ``engine_reasoning_parser`` label; "" when it
    runs none, and for an ``auto`` the engine has not resolved to a name yet
    (the label names a parser, never the request to pick one)."""
    parser = _resolved_reasoning_parser(server_args)
    if parser is _UNKNOWN or parser == "auto":
        return ""
    return str(parser or "")


def sglang_constrained_decoding_start(server_args: Any) -> str:
    """Where SGLang starts a request's grammar on a thinking prompt. With
    ``--reasoning-parser`` the scheduler wraps its grammar backend in a reasoner
    that defers the grammar until the parser's thought-end sequence has been
    generated when the request asks for reasoning (``require_reasoning``, which
    the Router sets on a thinking prompt): ``after_reasoning``. Without a parser
    the grammar runs from the first generated token: ``first_token``. "" when
    the server arguments predate the reasoning parser, and for an ``auto`` the
    engine has not resolved (its detection may find no parser, and then the
    grammar runs from the first token), which leaves the Router to the shape
    that fits either kind of engine."""
    parser = _resolved_reasoning_parser(server_args)
    if parser is _UNKNOWN or parser == "auto":
        return ""
    if parser:
        return CONSTRAINED_DECODING_AFTER_REASONING
    return CONSTRAINED_DECODING_FIRST_TOKEN


def tokenspeed_constrained_decoding_start() -> str:
    """Where TokenSpeed starts a request's grammar on a thinking prompt: from
    the first generated token, whatever the reasoning parser. Its grammar
    manager takes no reasoning flag and its backends start every compiled
    grammar at once (the token-gated reasoner wrapper of earlier versions is
    gone); ``--reasoning-parser`` only has a JSON-schema constraint rewritten
    into a structural tag that lets the thought precede the JSON, which is
    still a grammar from the first token, and a structural tag from the Router
    is taken as it is. So: ``first_token``."""
    return CONSTRAINED_DECODING_FIRST_TOKEN


def sglang_grammar_backend(server_args: Any) -> str:
    """The grammar backend SGLang's scheduler runs, as its server arguments
    resolve it: ``none`` without the scheduler's tokenizer
    (``--skip-tokenizer-init``), since its grammar manager builds no backend
    then whatever the flag says; else ``--grammar-backend`` as the engine
    resolved it (``xgrammar`` when it was not given, the engine's default,
    which its resolution hook declares and older releases set in place); ""
    when the server arguments have no such field at all (say nothing: the
    servicer then takes the engine's default)."""
    if getattr(server_args, "skip_tokenizer_init", False):
        return GRAMMAR_BACKEND_NONE
    backend = _resolved_argument(server_args, "grammar_backend")
    if backend is _UNKNOWN:
        return ""
    return str(backend or SGLANG_DEFAULT_GRAMMAR_BACKEND)


def tokenspeed_grammar_backend(server_args: Any) -> str:
    """The grammar backend TokenSpeed's scheduler runs, as its server arguments
    resolve it: ``--grammar-backend`` when given (``xgrammar`` or ``none``);
    when it was not given the engine starts from ``none`` and flips the
    checkpoints it defaults otherwise to ``xgrammar`` once it has read the
    checkpoint, so an unset flag reads as the resolved default when the engine
    has read it, and as "" (unknown: say nothing, the servicer then judges
    nothing ahead of the engine) when it has not, or when its own post-init
    has not run yet (the raw argument is then None); "" too when the server
    arguments have no such field."""
    backend = getattr(server_args, "grammar_backend", _UNKNOWN)
    if backend is _UNKNOWN or backend is None:
        return ""
    backend = str(backend)
    if backend == GRAMMAR_BACKEND_NONE and getattr(server_args, "_grammar_backend_unset", False):
        beam_checkpoint = getattr(server_args, "_beam_checkpoint", None)
        if beam_checkpoint is None:
            return ""
        return "xgrammar" if beam_checkpoint else GRAMMAR_BACKEND_NONE
    return backend


def grammar_backend_entry(args_dict: dict[str, Any], backend: str) -> None:
    """Write the resolved ``backend`` under :data:`GRAMMAR_BACKEND_KEY` in the
    ``server_args`` the servicer reports, in place of the raw argument (None
    before the engine resolves it, or a default the engine flips later); an
    empty ``backend`` (nothing known) leaves no key, so the servicer judges
    nothing ahead of the engine."""
    args_dict.pop(GRAMMAR_BACKEND_KEY, None)
    if backend:
        args_dict[GRAMMAR_BACKEND_KEY] = backend


def grammar_start_labels(server_args: Any, start: str) -> dict[str, str]:
    """The two labels as ``server_args`` entries, for the servicers that carry
    their labels there: the parser when the engine runs one, the start when the
    rule says (an empty value is left out, as the Router would drop it)."""
    labels = {}
    parser = engine_reasoning_parser(server_args)
    if parser:
        labels[ENGINE_REASONING_PARSER_KEY] = parser
    if start:
        labels[CONSTRAINED_DECODING_START_KEY] = start
    return labels
