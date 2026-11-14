use std::sync::Arc;

use openai_protocol::{
    common::{Function, Tool},
    worker::TagVocabulary,
};
use serde_json::json;

use super::*;
use crate::{
    formats::Table,
    grammar::{GrammarStart, Options},
};

fn tool(name: &str, parameters: serde_json::Value) -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: name.to_string(),
            description: None,
            parameters: Some(parameters),
            strict: None,
            extra: Default::default(),
        },
    }
}

fn tools() -> Vec<Tool> {
    vec![
        tool(
            "get_weather",
            json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
        ),
        tool("ping", json!({"type": "object", "properties": {}})),
    ]
}

/// Every registry table's forced tag, `auto` tag and JSON answer, for both bounds on the calls
/// and every start: the grammars the gateway sends.
fn every_grammar() -> Vec<(String, Grammar)> {
    let tools = tools();
    let schema = Arc::new(json!({"type": "object", "properties": {"answer": {"type": "string"}}}));
    let mut grammars = Vec::new();
    for table in Table::ALL {
        let format = table.shared();
        for start in [
            GrammarStart::Unknown,
            GrammarStart::AfterReasoning,
            GrammarStart::FirstToken,
        ] {
            for (at_least_one, one_call) in
                [(true, false), (true, true), (false, false), (false, true)]
            {
                for (reasoning_open, thought_closed) in
                    [(false, false), (true, false), (false, true)]
                {
                    let options = Options {
                        at_least_one,
                        reasoning_open,
                        thought_closed,
                        one_call,
                        start,
                    };
                    if let Some(grammar) = format.grammar_for(&tools, options) {
                        grammars.push((format!("{} calls {options:?}", table.name()), grammar));
                    }
                }
            }
            for reasoning_open in [false, true] {
                if let Some(grammar) = format.answer_grammar_for(&schema, reasoning_open, start) {
                    grammars.push((
                        format!("{} answer open={reasoning_open} {start:?}", table.name()),
                        grammar,
                    ));
                }
            }
        }
    }
    grammars
}

#[test]
fn xgrammar_spelling_is_the_structural_tag_byte_for_byte() {
    let grammars = every_grammar();
    assert!(
        grammars.len() > Table::ALL.len() * 10,
        "{} grammars",
        grammars.len()
    );
    for (name, grammar) in grammars {
        let current = grammar.payload_text();
        let older = grammar.clone().in_older_vocabulary().payload_text();
        assert_eq!(
            grammar
                .clone()
                .spelled(GrammarBackend::Xgrammar, TagVocabulary::Current),
            Ok(Spelling::StructuralTag(current)),
            "{name}"
        );
        assert_eq!(
            grammar.spelled(GrammarBackend::Xgrammar, TagVocabulary::Older),
            Ok(Spelling::StructuralTag(older)),
            "{name}"
        );
    }
}

#[test]
fn a_backend_without_a_spelling_refuses_by_name() {
    let grammar = Grammar::ConstString("x".to_string());
    for backend in [GrammarBackend::Llguidance, GrammarBackend::Outlines] {
        for vocabulary in [TagVocabulary::Current, TagVocabulary::Older] {
            let refused = grammar.clone().spelled(backend, vocabulary);
            assert_eq!(refused, Err(Unspelled { backend }));
            let words = refused.unwrap_err().to_string();
            assert!(words.contains(backend.name()), "{words}");
        }
    }
}

#[test]
fn the_backends_are_named_as_the_engines_spell_them() {
    assert_eq!(GrammarBackend::default(), GrammarBackend::Xgrammar);
    assert_eq!(GrammarBackend::Xgrammar.to_string(), "xgrammar");
    assert_eq!(GrammarBackend::Llguidance.to_string(), "llguidance");
    assert_eq!(GrammarBackend::Outlines.to_string(), "outlines");
}
