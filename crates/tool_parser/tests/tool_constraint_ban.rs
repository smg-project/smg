#![expect(
    clippy::expect_used,
    reason = "test-only helpers outside #[test] fns; failures are test failures"
)]

//! Suppression-constraint coverage: the `tool_choice: "none"` ban tag.
//!
//! The ban is a structural tag whose format is free text excluding the
//! strings that open a parser's native tool-call syntax. Only parsers with
//! model-native framing (and a curated opener inventory) produce one.

use serde_json::Value;
use tool_parser::ParserFactory;

fn ban_tag_json(factory: &ParserFactory, parser: &str) -> Value {
    let constraint = factory
        .registry()
        .tool_call_ban_constraint(Some(parser))
        .expect("parser should produce a ban constraint");
    let (kind, json) = constraint.to_tuple();
    assert_eq!(kind, "structural_tag", "ban constraint for {parser}");
    serde_json::from_str(&json).expect("ban tag must be valid JSON")
}

fn excludes(tag: &Value) -> Vec<&str> {
    tag["format"]["excludes"]
        .as_array()
        .expect("excludes must be an array")
        .iter()
        .map(|v| v.as_str().expect("excludes entries must be strings"))
        .collect()
}

#[test]
fn ban_tag_shape_is_any_text_with_excludes() {
    let factory = ParserFactory::new();
    let tag = ban_tag_json(&factory, "mistral");
    assert_eq!(tag["format"]["type"], "any_text");
    assert!(tag["format"]["excludes"].is_array());
    // Top-level envelope matches the positive structural tags: one "format" key.
    assert!(tag.get("format").is_some());
    assert_eq!(tag.as_object().map(serde_json::Map::len), Some(1));
}

#[test]
fn curated_ban_inventories_per_parser() {
    let factory = ParserFactory::new();
    assert_eq!(
        excludes(&ban_tag_json(&factory, "mistral")),
        vec!["[TOOL_CALLS]"]
    );
    assert_eq!(
        excludes(&ban_tag_json(&factory, "kimik2")),
        vec!["<|tool_calls_section_begin|>", "<|tool_call_begin|>"]
    );
    // K3 bans ONLY the tools-section opener: its structural-tag triggers also
    // include think/response section closers that ordinary generation must
    // emit, and excluding those would corrupt normal output.
    assert_eq!(
        excludes(&ban_tag_json(&factory, "kimi_k3")),
        vec!["<|open|>tools<|sep|>"]
    );
    // Both Inkling invocation modes: JSON-arguments and TML text-mode.
    assert_eq!(
        excludes(&ban_tag_json(&factory, "inkling")),
        vec![
            "<|content_invoke_tool_json|>",
            "<|content_invoke_tool_text|>"
        ]
    );
}

#[test]
fn parsers_without_native_framing_produce_no_ban() {
    let factory = ParserFactory::new();
    for parser in ["json", "qwen", "qwen_xml", "pythonic", "llama", "deepseek"] {
        assert!(
            factory
                .registry()
                .tool_call_ban_constraint(Some(parser))
                .is_none(),
            "{parser} has no curated ban inventory"
        );
    }
}

#[test]
fn unknown_or_absent_parser_produces_no_ban() {
    let factory = ParserFactory::new();
    assert!(factory
        .registry()
        .tool_call_ban_constraint(Some("no-such-parser"))
        .is_none());
    assert!(factory.registry().tool_call_ban_constraint(None).is_none());
}

#[test]
fn generate_tool_constraint_still_skips_none_choice() {
    use openai_protocol::common::{Function, Tool, ToolChoice, ToolChoiceValue};

    let factory = ParserFactory::new();
    let tools = vec![Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: "get_weather".to_string(),
            description: None,
            parameters: serde_json::json!({"type": "object"}),
            strict: None,
        },
    }];
    // The ban is a separate opt-in surface; the standard constraint generator
    // keeps returning no constraint for tool_choice "none" and "auto".
    for choice in [
        ToolChoice::Value(ToolChoiceValue::None),
        ToolChoice::Value(ToolChoiceValue::Auto),
    ] {
        let constraint = factory
            .registry()
            .generate_tool_constraint(Some("mistral"), &tools, &choice)
            .expect("constraint generation must not error");
        assert!(
            constraint.is_none(),
            "no constraint expected for {choice:?}"
        );
    }
}
