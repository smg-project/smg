"""The ``constrained_decoding_start`` and ``engine_reasoning_parser`` labels the
servicers carry as ``server_args`` keys, and the ``grammar_backend`` and
``structural_tag_vocabulary`` keys beside them: the rule per engine, under each
server configuration, and the entries written for the Router and the Rust
servicer."""

from __future__ import annotations

import importlib.metadata
from types import SimpleNamespace

from smg_grpc_servicer import constrained_decoding as cd


def test_sglang_starts_the_grammar_after_the_thought_with_a_reasoning_parser():
    # With --reasoning-parser the scheduler wraps its grammar backend in a
    # reasoner that defers the grammar past the thought's end for a request
    # that asks for reasoning, which the Router does on a thinking prompt.
    assert cd.sglang_constrained_decoding_start(SimpleNamespace(reasoning_parser="qwen3")) == (
        cd.CONSTRAINED_DECODING_AFTER_REASONING
    )


def test_sglang_starts_the_grammar_at_the_first_token_without_a_parser():
    assert cd.sglang_constrained_decoding_start(SimpleNamespace(reasoning_parser=None)) == (
        cd.CONSTRAINED_DECODING_FIRST_TOKEN
    )
    assert cd.sglang_constrained_decoding_start(SimpleNamespace(reasoning_parser="")) == (
        cd.CONSTRAINED_DECODING_FIRST_TOKEN
    )


def test_sglang_says_nothing_when_the_server_args_predate_the_parser():
    assert cd.sglang_constrained_decoding_start(SimpleNamespace(model_path="org/model")) == ""


def test_sglang_reads_an_auto_parser_as_the_engine_resolved_it():
    # `--reasoning-parser auto` stays "auto" on the argument; the engine's
    # template detection declares what it resolved to (a name, or None when no
    # rule matched) in _resolved_overrides, last writer wins.
    named = SimpleNamespace(
        reasoning_parser="auto",
        _resolved_overrides=[("template-detection", {"reasoning_parser": "qwen3"})],
    )
    assert cd.sglang_constrained_decoding_start(named) == cd.CONSTRAINED_DECODING_AFTER_REASONING
    assert cd.engine_reasoning_parser(named) == "qwen3"

    none_found = SimpleNamespace(
        reasoning_parser="auto",
        _resolved_overrides=[
            ("template-detection", {"reasoning_parser": "qwen3"}),
            ("later", {"reasoning_parser": None}),
        ],
    )
    assert cd.sglang_constrained_decoding_start(none_found) == cd.CONSTRAINED_DECODING_FIRST_TOKEN
    assert cd.engine_reasoning_parser(none_found) == ""

    # Not resolved yet (or never, in a process that does not run the
    # detection): the engine may or may not gate its grammar, so say nothing,
    # and never advertise "auto" as a parser's name.
    unresolved = SimpleNamespace(reasoning_parser="auto")
    assert cd.sglang_constrained_decoding_start(unresolved) == ""
    assert cd.engine_reasoning_parser(unresolved) == ""
    assert (
        cd.grammar_start_labels(unresolved, cd.sglang_constrained_decoding_start(unresolved)) == {}
    )


def test_engine_reasoning_parser_is_the_servers_argument_or_empty():
    assert cd.engine_reasoning_parser(SimpleNamespace(reasoning_parser="deepseek-r1")) == (
        "deepseek-r1"
    )
    assert cd.engine_reasoning_parser(SimpleNamespace(reasoning_parser=None)) == ""
    assert cd.engine_reasoning_parser(SimpleNamespace()) == ""


def test_grammar_start_labels_leave_out_what_is_empty():
    args = SimpleNamespace(reasoning_parser="qwen3")
    assert cd.grammar_start_labels(args, cd.CONSTRAINED_DECODING_AFTER_REASONING) == {
        cd.ENGINE_REASONING_PARSER_KEY: "qwen3",
        cd.CONSTRAINED_DECODING_START_KEY: "after_reasoning",
    }
    # No parser: the start alone; the Router would drop an empty value anyway.
    assert cd.grammar_start_labels(
        SimpleNamespace(reasoning_parser=None), cd.CONSTRAINED_DECODING_FIRST_TOKEN
    ) == {cd.CONSTRAINED_DECODING_START_KEY: "first_token"}
    # Nothing known: nothing written, so an older label set is unchanged.
    assert cd.grammar_start_labels(SimpleNamespace(), "") == {}
    # The keys are the labels the Router reads, never its parser override.
    assert cd.ENGINE_REASONING_PARSER_KEY != "reasoning_parser"


def test_tokenspeed_starts_the_grammar_at_the_first_token_whatever_the_parser():
    # This engine's grammar manager has no reasoning gate: a grammar runs from
    # the first generated token; its --reasoning-parser only has a JSON schema
    # rewritten into a structural tag, which is still a grammar from the first
    # token. The parser still rides along under its own name.
    assert cd.tokenspeed_constrained_decoding_start() == cd.CONSTRAINED_DECODING_FIRST_TOKEN
    assert cd.grammar_start_labels(
        SimpleNamespace(reasoning_parser="kimi_k25"), cd.tokenspeed_constrained_decoding_start()
    ) == {
        cd.ENGINE_REASONING_PARSER_KEY: "kimi_k25",
        cd.CONSTRAINED_DECODING_START_KEY: "first_token",
    }
    assert cd.grammar_start_labels(
        SimpleNamespace(reasoning_parser=None), cd.tokenspeed_constrained_decoding_start()
    ) == {cd.CONSTRAINED_DECODING_START_KEY: "first_token"}


def test_sglang_grammar_backend_is_the_resolved_argument_or_the_default():
    # Given as typed.
    assert cd.sglang_grammar_backend(SimpleNamespace(grammar_backend="outlines")) == "outlines"
    assert cd.sglang_grammar_backend(SimpleNamespace(grammar_backend="none")) == "none"
    # Not given: the engine's default, which its resolution hook declares in
    # _resolved_overrides (last writer wins) and older releases set in place.
    assert cd.sglang_grammar_backend(SimpleNamespace(grammar_backend=None)) == "xgrammar"
    declared = SimpleNamespace(
        grammar_backend=None,
        _resolved_overrides=[("_handle_grammar_backend", {"grammar_backend": "xgrammar"})],
    )
    assert cd.sglang_grammar_backend(declared) == "xgrammar"
    # Without the scheduler's tokenizer its grammar manager builds no backend,
    # whatever the flag says.
    assert (
        cd.sglang_grammar_backend(
            SimpleNamespace(grammar_backend="xgrammar", skip_tokenizer_init=True)
        )
        == "none"
    )
    # No such field at all: say nothing.
    assert cd.sglang_grammar_backend(SimpleNamespace(model_path="org/model")) == ""


def test_tokenspeed_grammar_backend_is_the_flag_or_the_resolved_checkpoint_default():
    # Given as typed.
    assert cd.tokenspeed_grammar_backend(SimpleNamespace(grammar_backend="xgrammar")) == "xgrammar"
    explicit_none = SimpleNamespace(grammar_backend="none", _grammar_backend_unset=False)
    assert cd.tokenspeed_grammar_backend(explicit_none) == "none"
    # Not given: the engine starts from none and flips the checkpoints it
    # defaults otherwise once it has read the checkpoint.
    assert (
        cd.tokenspeed_grammar_backend(
            SimpleNamespace(
                grammar_backend="none", _grammar_backend_unset=True, _beam_checkpoint=True
            )
        )
        == "xgrammar"
    )
    assert (
        cd.tokenspeed_grammar_backend(
            SimpleNamespace(
                grammar_backend="none", _grammar_backend_unset=True, _beam_checkpoint=False
            )
        )
        == "none"
    )
    # Not read yet: unknown, say nothing.
    assert (
        cd.tokenspeed_grammar_backend(
            SimpleNamespace(
                grammar_backend="none", _grammar_backend_unset=True, _beam_checkpoint=None
            )
        )
        == ""
    )
    # Not given and the engine's own post-init not run yet (the raw argument
    # is None): unknown as well, say nothing.
    assert cd.tokenspeed_grammar_backend(SimpleNamespace(grammar_backend=None)) == ""
    # No such field at all: say nothing.
    assert cd.tokenspeed_grammar_backend(SimpleNamespace(model="org/model")) == ""


def test_grammar_backend_entry_replaces_the_raw_argument():
    args = {"grammar_backend": None, "tp_size": 2}
    cd.grammar_backend_entry(args, "xgrammar")
    assert args == {"grammar_backend": "xgrammar", "tp_size": 2}
    # Nothing known: no key, so the servicer judges nothing ahead of the engine.
    args = {"grammar_backend": "none", "tp_size": 2}
    cd.grammar_backend_entry(args, "")
    assert args == {"tp_size": 2}
    assert cd.GRAMMAR_BACKEND_KEY == "grammar_backend"


def test_structural_tag_vocabulary_is_older_below_the_first_release_with_the_nodes():
    # xgrammar 0.1.33 is the first release whose structural tags take the
    # optional, plus, star and token nodes; 0.1.32 refuses them at compile
    # time, so the label says `older` below 0.1.33 and nothing from it on.
    assert cd.XGRAMMAR_CURRENT_VOCABULARY_SINCE == (0, 1, 33)
    assert cd.structural_tag_vocabulary("xgrammar", "0.1.32") == cd.STRUCTURAL_TAG_VOCABULARY_OLDER
    assert cd.structural_tag_vocabulary("xgrammar", "0.1.25") == "older"
    assert cd.structural_tag_vocabulary("xgrammar", "0.1.32.post1") == "older"
    assert cd.structural_tag_vocabulary("xgrammar", "0.1.33") == ""
    assert cd.structural_tag_vocabulary("xgrammar", "0.1.34") == ""
    assert cd.structural_tag_vocabulary("xgrammar", "0.2.7") == ""
    assert cd.structural_tag_vocabulary("xgrammar", "1.0.0") == ""
    # A pre-release or a local build of the first release carries the nodes.
    assert cd.structural_tag_vocabulary("xgrammar", "0.1.33rc1") == ""
    assert cd.structural_tag_vocabulary("xgrammar", "0.1.33.dev2+g1234567") == ""
    # Nothing known: no xgrammar installed, or a version that does not parse.
    assert cd.structural_tag_vocabulary("xgrammar", "") == ""
    assert cd.structural_tag_vocabulary("xgrammar", "unknown") == ""
    assert cd.STRUCTURAL_TAG_VOCABULARY_KEY == "structural_tag_vocabulary"


def test_structural_tag_vocabulary_applies_to_the_backends_that_run_xgrammar():
    # xgrammar by name, vLLM's auto (xgrammar for a structural tag) and a
    # backend the launcher could not resolve (the engines default to xgrammar
    # where one runs); another backend has a vocabulary of its own, and
    # `none` enforces nothing: nothing is said for them.
    for backend in ("xgrammar", " XGrammar ", "auto", ""):
        assert cd.structural_tag_vocabulary(backend, "0.1.32") == "older", backend
    for backend in ("none", "llguidance", "guidance", "outlines", "lm-format-enforcer"):
        assert cd.structural_tag_vocabulary(backend, "0.1.32") == "", backend


def test_structural_tag_vocabulary_reads_the_installed_xgrammar_from_its_metadata(monkeypatch):
    # The distribution's metadata, never an import of xgrammar (which loads
    # the engine's runtime); no distribution at all says nothing.
    asked = []

    def version(name):
        asked.append(name)
        if name == "xgrammar":
            return "0.1.32"
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    assert cd.installed_xgrammar_version() == "0.1.32"
    assert cd.structural_tag_vocabulary("xgrammar") == "older"
    assert asked == ["xgrammar", "xgrammar"]

    def absent(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", absent)
    assert cd.installed_xgrammar_version() == ""
    assert cd.structural_tag_vocabulary("xgrammar") == ""


def test_structural_tag_vocabulary_entry_writes_the_label_or_nothing(monkeypatch):
    monkeypatch.setattr(cd, "installed_xgrammar_version", lambda: "0.1.32")
    args = {"grammar_backend": "xgrammar", "tp_size": 2}
    cd.structural_tag_vocabulary_entry(args, "xgrammar")
    assert args == {
        "grammar_backend": "xgrammar",
        "tp_size": 2,
        "structural_tag_vocabulary": "older",
    }
    # A current xgrammar: no key, and a stale one is cleared.
    monkeypatch.setattr(cd, "installed_xgrammar_version", lambda: "0.2.7")
    cd.structural_tag_vocabulary_entry(args, "xgrammar")
    assert args == {"grammar_backend": "xgrammar", "tp_size": 2}
    # Another backend on an older xgrammar: nothing to say.
    monkeypatch.setattr(cd, "installed_xgrammar_version", lambda: "0.1.32")
    cd.structural_tag_vocabulary_entry(args, "llguidance")
    assert "structural_tag_vocabulary" not in args
