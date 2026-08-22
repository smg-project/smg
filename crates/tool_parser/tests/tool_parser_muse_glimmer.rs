//! Integration tests for the Muse-Glimmer ATEM tool-call parser.
//!
//! The in-file unit tests cover the format rules. These cover the properties
//! that only show up across the public API and under adversarial chunking:
//! streaming must agree with a one-shot parse no matter where the transport
//! splits the bytes, framing must never reach the client, and the registry must
//! resolve the real published model ids.

mod common;

use common::{create_test_tools, streaming_helpers};
use serde_json::Value;
use tool_parser::{parsers::MuseGlimmerParser, traits::ToolParser, ParserFactory};

const MESSAGE: &str = "<|message|>";
const EOM: &str = "<|eom|>";
const EOT: &str = "<|eot|>";

/// A full turn: reasoning, a call, then the answer. `search` is one of the
/// canned tools in `create_test_tools()`, so name validation has something to
/// resolve against.
fn reason_call_answer() -> String {
    format!(
        "<|start|>assistant to=self{MESSAGE}I should look that up.{EOM}\
         <|start|>assistant to=search{MESSAGE}<atem:function_calls>\
         <atem:invoke name=\"search\">\
         <atem:parameter name=\"query\">rust ownership</atem:parameter>\
         </atem:invoke></atem:function_calls>{EOM}\
         <|start|>assistant to=user{MESSAGE}Here is what I found.{EOT}"
    )
}

/// Two calls in one block plus a second tool segment, to pin index continuity.
fn parallel_and_sequential_calls() -> String {
    format!(
        "<|start|>assistant to=search{MESSAGE}<atem:function_calls>\
         <atem:invoke name=\"search\">\
         <atem:parameter name=\"query\">first</atem:parameter></atem:invoke>\
         <atem:invoke name=\"search\">\
         <atem:parameter name=\"query\">second</atem:parameter></atem:invoke>\
         </atem:function_calls>{EOM}\
         <|start|>assistant to=get_weather{MESSAGE}<atem:function_calls>\
         <atem:invoke name=\"get_weather\">\
         <atem:parameter name=\"city\">Paris</atem:parameter></atem:invoke>\
         </atem:function_calls>{EOT}"
    )
}

#[expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test helper: a parse error here should fail the test loudly"
)]
async fn stream_in_chunks(input: &str, chunks: Vec<String>) -> (String, Vec<(String, Value)>) {
    let tools = create_test_tools();
    let mut parser = MuseGlimmerParser::new();
    let mut normal = String::new();
    let mut calls = Vec::new();

    for chunk in chunks {
        let result = parser.parse_incremental(&chunk, &tools).await.unwrap();
        normal.push_str(&result.normal_text);
        for item in result.calls {
            if let Some(name) = item.name {
                let args: Value = serde_json::from_str(&item.parameters).unwrap_or_else(|e| {
                    panic!("streamed arguments must be valid JSON for {name}: {e} ({input})")
                });
                calls.push((name, args));
            }
        }
    }

    (normal, calls)
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper: a parse error here should fail the test loudly"
)]
async fn parse_whole(input: &str) -> (String, Vec<(String, Value)>) {
    let tools = create_test_tools();
    let (normal, calls) = MuseGlimmerParser::new()
        .parse_complete_with_tools(input, &tools)
        .await
        .unwrap();
    let calls = calls
        .into_iter()
        .map(|call| {
            let args: Value = serde_json::from_str(&call.function.arguments).unwrap();
            (call.function.name, args)
        })
        .collect();
    (normal, calls)
}

/// The core property: however the transport chops the stream, the streamed
/// result must equal the one-shot result. Four chunkings, including the two
/// adversarial ones the crate's other parsers are held to.
#[tokio::test]
async fn streaming_agrees_with_one_shot_under_every_chunking() {
    for input in [reason_call_answer(), parallel_and_sequential_calls()] {
        let expected = parse_whole(&input).await;

        let one_char: Vec<String> = input.chars().map(|c| c.to_string()).collect();
        let whole = vec![input.clone()];
        let chunkings: Vec<(&str, Vec<String>)> = vec![
            ("whole", whole),
            ("one_char", one_char),
            (
                "realistic",
                streaming_helpers::create_realistic_chunks(&input),
            ),
            (
                "strategic",
                streaming_helpers::create_strategic_chunks(&input),
            ),
        ];

        for (label, chunks) in chunkings {
            let actual = stream_in_chunks(&input, chunks).await;
            assert_eq!(
                actual.1, expected.1,
                "tool calls diverged under {label} chunking"
            );
            assert_eq!(
                actual.0, expected.0,
                "normal text diverged under {label} chunking"
            );
        }
    }
}

/// Protocol framing must never reach the client, in either mode. This is the
/// failure that shape-only assertions miss: a parser can return the right call
/// and still leak `<|start|>` into the visible answer.
#[tokio::test]
async fn framing_never_reaches_normal_text() {
    let input = reason_call_answer();

    let (whole_normal, _) = parse_whole(&input).await;
    let (streamed_normal, _) =
        stream_in_chunks(&input, streaming_helpers::create_realistic_chunks(&input)).await;

    for (label, normal) in [("one-shot", &whole_normal), ("streamed", &streamed_normal)] {
        for marker in ["<|start|>", MESSAGE, EOM, EOT, "<atem:"] {
            assert!(
                !normal.contains(marker),
                "{label} normal text leaked {marker}: {normal:?}"
            );
        }
    }
    assert_eq!(whole_normal, "Here is what I found.");
}

/// Chain-of-thought must not surface as the answer. With `separate_reasoning`
/// off and tools present the raw stream still reaches this parser, so the
/// `to=self` body has to be dropped here rather than relied on upstream.
#[tokio::test]
async fn reasoning_bodies_are_not_content_and_not_calls() {
    let quoted = format!(
        "<|start|>assistant to=self{MESSAGE}\
         I could call <atem:function_calls><atem:invoke name=\"search\">\
         <atem:parameter name=\"query\">x</atem:parameter></atem:invoke>\
         </atem:function_calls> here.{EOM}\
         <|start|>assistant to=user{MESSAGE}Which topic?{EOT}"
    );

    let (normal, calls) = parse_whole(&quoted).await;

    assert!(
        calls.is_empty(),
        "markup quoted inside reasoning must never become a call: {calls:?}"
    );
    assert_eq!(normal, "Which topic?");
}

#[tokio::test]
async fn call_indices_continue_across_tool_segments() {
    let input = parallel_and_sequential_calls();
    let tools = create_test_tools();
    let mut parser = MuseGlimmerParser::new();

    let result = parser.parse_incremental(&input, &tools).await.unwrap();
    let indices: Vec<usize> = result.calls.iter().map(|item| item.tool_index).collect();

    assert_eq!(indices, vec![0, 1, 2]);
}

/// The published ids must reach this parser through the registry, since the
/// gateway resolves by model name when no parser is configured explicitly.
#[tokio::test]
async fn registry_resolves_the_published_model_ids() {
    let factory = ParserFactory::new();
    let registry = factory.registry();

    for model in [
        "meta-models/Muse-Glimmer-30B",
        "Muse-Glimmer-30B",
        "muse-glimmer-30b",
        "RedHatAI/Muse-Glimmer-30B-FP8-block",
        "unsloth/Muse-Glimmer-30B-GGUF",
    ] {
        assert_eq!(
            registry.resolve_model_to_parser(model),
            Some("muse_glimmer".to_string()),
            "{model} should resolve to muse_glimmer"
        );
        assert!(
            registry.has_parser_for_model(model),
            "{model} should have a constructible parser"
        );
    }

    assert!(factory.has_parser("muse_glimmer"));
}

/// A neighbouring family must not be captured by the new globs.
#[tokio::test]
async fn unrelated_models_do_not_resolve_to_muse_glimmer() {
    let factory = ParserFactory::new();
    let registry = factory.registry();

    for model in [
        "meta-llama/Llama-4-Scout",
        "Qwen/Qwen3-Coder-480B",
        "moonshotai/Kimi-K3",
    ] {
        assert_ne!(
            registry.resolve_model_to_parser(model).unwrap_or_default(),
            "muse_glimmer",
            "{model} must not resolve to muse_glimmer"
        );
    }
}

/// End to end through the pooled factory path the gateway actually uses.
#[tokio::test]
async fn pooled_parser_parses_a_published_model_id() {
    let factory = ParserFactory::new();
    let pooled = factory.get_pooled("meta-models/Muse-Glimmer-30B");
    let input = reason_call_answer();

    let (normal, calls) = pooled.lock().await.parse_complete(&input).await.unwrap();

    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "search");
    assert_eq!(normal, "Here is what I found.");
}
