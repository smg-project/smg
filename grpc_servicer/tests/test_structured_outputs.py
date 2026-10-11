"""The ``structured_outputs_start`` and ``engine_reasoning_parser`` labels the
servicers carry as ``server_args`` keys: the rule per engine, under each
server configuration, and the entries written for the Router."""

from __future__ import annotations

from types import SimpleNamespace

from smg_grpc_servicer import structured_outputs as so


def test_sglang_starts_the_grammar_after_the_thought_with_a_reasoning_parser():
    # With --reasoning-parser the scheduler wraps its grammar backend in a
    # reasoner that defers the grammar past the thought's end for a request
    # that asks for reasoning, which the Router does on a thinking prompt.
    assert so.sglang_structured_outputs_start(SimpleNamespace(reasoning_parser="qwen3")) == (
        so.STRUCTURED_OUTPUTS_AFTER_REASONING
    )


def test_sglang_starts_the_grammar_at_the_first_token_without_a_parser():
    assert so.sglang_structured_outputs_start(SimpleNamespace(reasoning_parser=None)) == (
        so.STRUCTURED_OUTPUTS_FIRST_TOKEN
    )
    assert so.sglang_structured_outputs_start(SimpleNamespace(reasoning_parser="")) == (
        so.STRUCTURED_OUTPUTS_FIRST_TOKEN
    )


def test_sglang_says_nothing_when_the_server_args_predate_the_parser():
    assert so.sglang_structured_outputs_start(SimpleNamespace(model_path="org/model")) == ""


def test_sglang_reads_an_auto_parser_as_the_engine_resolved_it():
    # `--reasoning-parser auto` stays "auto" on the argument; the engine's
    # template detection declares what it resolved to (a name, or None when no
    # rule matched) in _resolved_overrides, last writer wins.
    named = SimpleNamespace(
        reasoning_parser="auto",
        _resolved_overrides=[("template-detection", {"reasoning_parser": "qwen3"})],
    )
    assert so.sglang_structured_outputs_start(named) == so.STRUCTURED_OUTPUTS_AFTER_REASONING
    assert so.engine_reasoning_parser(named) == "qwen3"

    none_found = SimpleNamespace(
        reasoning_parser="auto",
        _resolved_overrides=[
            ("template-detection", {"reasoning_parser": "qwen3"}),
            ("later", {"reasoning_parser": None}),
        ],
    )
    assert so.sglang_structured_outputs_start(none_found) == so.STRUCTURED_OUTPUTS_FIRST_TOKEN
    assert so.engine_reasoning_parser(none_found) == ""

    # Not resolved yet (or never, in a process that does not run the
    # detection): the engine may or may not gate its grammar, so say nothing,
    # and never advertise "auto" as a parser's name.
    unresolved = SimpleNamespace(reasoning_parser="auto")
    assert so.sglang_structured_outputs_start(unresolved) == ""
    assert so.engine_reasoning_parser(unresolved) == ""
    assert so.grammar_start_labels(unresolved, so.sglang_structured_outputs_start(unresolved)) == {}


def test_engine_reasoning_parser_is_the_servers_argument_or_empty():
    assert so.engine_reasoning_parser(SimpleNamespace(reasoning_parser="deepseek-r1")) == (
        "deepseek-r1"
    )
    assert so.engine_reasoning_parser(SimpleNamespace(reasoning_parser=None)) == ""
    assert so.engine_reasoning_parser(SimpleNamespace()) == ""


def test_grammar_start_labels_leave_out_what_is_empty():
    args = SimpleNamespace(reasoning_parser="qwen3")
    assert so.grammar_start_labels(args, so.STRUCTURED_OUTPUTS_AFTER_REASONING) == {
        so.ENGINE_REASONING_PARSER_KEY: "qwen3",
        so.STRUCTURED_OUTPUTS_START_KEY: "after_reasoning",
    }
    # No parser: the start alone; the Router would drop an empty value anyway.
    assert so.grammar_start_labels(
        SimpleNamespace(reasoning_parser=None), so.STRUCTURED_OUTPUTS_FIRST_TOKEN
    ) == {so.STRUCTURED_OUTPUTS_START_KEY: "first_token"}
    # Nothing known: nothing written, so an older label set is unchanged.
    assert so.grammar_start_labels(SimpleNamespace(), "") == {}
    # The keys are the labels the Router reads, never its parser override.
    assert so.ENGINE_REASONING_PARSER_KEY != "reasoning_parser"


def test_tokenspeed_starts_the_grammar_at_the_first_token_whatever_the_parser():
    # This engine's grammar manager has no reasoning gate: a grammar runs from
    # the first generated token; its --reasoning-parser only has a JSON schema
    # rewritten into a structural tag, which is still a grammar from the first
    # token. The parser still rides along under its own name.
    assert so.tokenspeed_structured_outputs_start() == so.STRUCTURED_OUTPUTS_FIRST_TOKEN
    assert so.grammar_start_labels(
        SimpleNamespace(reasoning_parser="kimi_k25"), so.tokenspeed_structured_outputs_start()
    ) == {
        so.ENGINE_REASONING_PARSER_KEY: "kimi_k25",
        so.STRUCTURED_OUTPUTS_START_KEY: "first_token",
    }
    assert so.grammar_start_labels(
        SimpleNamespace(reasoning_parser=None), so.tokenspeed_structured_outputs_start()
    ) == {so.STRUCTURED_OUTPUTS_START_KEY: "first_token"}
