//! Integration tests for the Muse-Glimmer segment-channel reasoning parser.
//!
//! The in-file unit tests cover the format rules. These cover the two
//! properties that only appear across the public API: streaming must agree with
//! a one-shot parse under adversarial chunking, and the tool-channel segments
//! this parser passes through must survive byte-exactly, because the tool
//! parser downstream is the only consumer of them.

use reasoning_parser::{traits::ReasoningParser, ParserFactory, ParserResult};

const START: &str = "<|start|>";
const MESSAGE: &str = "<|message|>";
const EOM: &str = "<|eom|>";
const EOT: &str = "<|eot|>";

fn tool_segment() -> String {
    format!(
        "{START}assistant to=search{MESSAGE}<atem:function_calls>\
         <atem:invoke name=\"search\">\
         <atem:parameter name=\"query\">rust ownership</atem:parameter>\
         </atem:invoke></atem:function_calls>{EOM}"
    )
}

/// A full turn whose first segment is headerless, because the generation prompt
/// already emitted `<|start|>assistant`.
fn full_turn() -> String {
    format!(
        " to=self{MESSAGE}I should look that up.{EOM}{}{START}assistant to=user{MESSAGE}Here is what I found.{EOT}",
        tool_segment()
    )
}

fn parser_for(model: &str) -> Box<dyn ReasoningParser> {
    ParserFactory::new().create(model)
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper: a parse error here should fail the test loudly"
)]
fn stream_in_chunks(model: &str, chunks: &[String]) -> ParserResult {
    let mut parser = parser_for(model);
    let mut merged = ParserResult::default();
    for chunk in chunks {
        let result = parser.parse_reasoning_streaming_incremental(chunk).unwrap();
        merged.reasoning_text.push_str(&result.reasoning_text);
        merged.normal_text.push_str(&result.normal_text);
    }
    merged
}

fn realistic_chunks(input: &str) -> Vec<String> {
    let chars: Vec<char> = input.chars().collect();
    let mut chunks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let size = if chars[i].is_ascii_alphanumeric() {
            3
        } else {
            2
        };
        let end = (i + size).min(chars.len());
        chunks.push(chars[i..end].iter().collect());
        i = end;
    }
    chunks
}

/// Chunk boundaries placed immediately inside every control marker — the split
/// points most likely to tear a marker in half.
fn marker_tearing_chunks(input: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut rest = input;
    while let Some(index) = rest.find("<|") {
        let cut = (index + 3).min(rest.len());
        let cut = (cut..=rest.len())
            .find(|&c| rest.is_char_boundary(c))
            .unwrap_or(rest.len());
        chunks.push(rest[..cut].to_string());
        rest = &rest[cut..];
    }
    if !rest.is_empty() {
        chunks.push(rest.to_string());
    }
    chunks
}

#[test]
fn streaming_agrees_with_one_shot_under_every_chunking() {
    let model = "meta-models/Muse-Glimmer-30B";
    let input = full_turn();
    let expected = parser_for(model)
        .detect_and_parse_reasoning(&input)
        .unwrap();

    let one_char: Vec<String> = input.chars().map(|c| c.to_string()).collect();
    let chunkings: Vec<(&str, Vec<String>)> = vec![
        ("whole", vec![input.clone()]),
        ("one_char", one_char),
        ("realistic", realistic_chunks(&input)),
        ("marker_tearing", marker_tearing_chunks(&input)),
    ];

    for (label, chunks) in chunkings {
        let actual = stream_in_chunks(model, &chunks);
        assert_eq!(
            actual.reasoning_text, expected.reasoning_text,
            "reasoning diverged under {label} chunking"
        );
        assert_eq!(
            actual.normal_text, expected.normal_text,
            "normal text diverged under {label} chunking"
        );
    }
}

/// The contract with the tool-parser crate: a tool segment must come out of the
/// reasoning stage byte-exact, framing included, because that is the only thing
/// downstream has to work from. The leading segment gets its `<|start|>` header
/// synthesized so both look identical.
#[test]
fn tool_segments_pass_through_byte_exactly() {
    let model = "meta-models/Muse-Glimmer-30B";
    let result = parser_for(model)
        .detect_and_parse_reasoning(&full_turn())
        .unwrap();

    assert_eq!(
        result.normal_text,
        format!("{}Here is what I found.", tool_segment())
    );
    assert_eq!(result.reasoning_text, "I should look that up.");
}

/// A leading tool segment arrives without its `<|start|>assistant` prefix; the
/// parser must synthesize one so the downstream grammar is uniform.
#[test]
fn leading_tool_segment_is_canonicalized() {
    let model = "meta-models/Muse-Glimmer-30B";
    let leading = format!(
        " to=search{MESSAGE}<atem:function_calls>\
         <atem:invoke name=\"search\">\
         <atem:parameter name=\"query\">x</atem:parameter>\
         </atem:invoke></atem:function_calls>{EOM}"
    );

    let result = parser_for(model)
        .detect_and_parse_reasoning(&leading)
        .unwrap();

    assert!(
        result
            .normal_text
            .starts_with(&format!("{START}assistant to=search{MESSAGE}")),
        "leading tool segment must gain a canonical header: {:?}",
        result.normal_text
    );
    assert!(result.reasoning_text.is_empty());
}

#[test]
fn factory_resolves_published_model_ids_and_requires_special_tokens() {
    for model in [
        "meta-models/Muse-Glimmer-30B",
        "Muse-Glimmer-30B",
        "muse-glimmer-30b",
        "RedHatAI/Muse-Glimmer-30B-FP8-block",
    ] {
        let parser = parser_for(model);
        assert_eq!(
            parser.model_type(),
            "muse_glimmer",
            "{model} should resolve to muse_glimmer"
        );
        // This is what pins skip_special_tokens=false on the gRPC path; without
        // it the framing is stripped before the parser ever sees it.
        assert!(
            parser.requires_special_tokens(),
            "{model} parser must require special tokens"
        );
    }
}

#[test]
fn unrelated_models_do_not_resolve_to_muse_glimmer() {
    for model in ["meta-llama/Llama-3-70B", "Qwen/Qwen3-Coder-480B"] {
        assert_ne!(
            parser_for(model).model_type(),
            "muse_glimmer",
            "{model} must not resolve to muse_glimmer"
        );
    }
}
