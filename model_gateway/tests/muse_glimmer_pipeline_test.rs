//! Cross-crate pipeline test for the Muse-Glimmer format.
//!
//! The two parser crates cannot depend on each other, so nothing inside either
//! one proves they compose. The gateway runs them in a fixed order — reasoning
//! separation first over the whole generation, then tool parsing over whatever
//! normal text that produced — and the Muse-Glimmer format makes that handoff
//! load-bearing: the reasoning stage is what decides which segments survive for
//! the tool stage to read. These tests pin that contract end to end, on CPU.

use reasoning_parser::ParserFactory as ReasoningParserFactory;
use serde_json::Value;
use tool_parser::ParserFactory as ToolParserFactory;

const MODEL: &str = "meta-models/Muse-Glimmer-30B";
const START: &str = "<|start|>";
const MESSAGE: &str = "<|message|>";
const EOM: &str = "<|eom|>";
const EOT: &str = "<|eot|>";

struct Parsed {
    reasoning: String,
    content: String,
    calls: Vec<(String, Value)>,
}

/// Run a complete generation through the gateway's parser order.
#[expect(
    clippy::expect_used,
    reason = "test helper: a parse error here should fail the test loudly"
)]
async fn run_pipeline(generation: &str) -> Parsed {
    let reasoning = ReasoningParserFactory::new()
        .create(MODEL)
        .detect_and_parse_reasoning(generation)
        .expect("reasoning parse");

    let pooled = ToolParserFactory::new().get_pooled(MODEL);
    let (content, calls) = pooled
        .lock()
        .await
        .parse_complete(&reasoning.normal_text)
        .await
        .expect("tool parse");

    Parsed {
        reasoning: reasoning.reasoning_text,
        content,
        calls: calls
            .into_iter()
            .map(|call| {
                let args = serde_json::from_str(&call.function.arguments)
                    .expect("tool arguments must be valid JSON");
                (call.function.name, args)
            })
            .collect(),
    }
}

/// The canonical turn, with the first segment headerless because the generation
/// prompt already supplied `<|start|>assistant`.
fn reason_call_answer() -> String {
    format!(
        " to=self{MESSAGE}The user wants current weather.{EOM}\
         {START}assistant to=get_weather{MESSAGE}<atem:function_calls>\
         <atem:invoke name=\"get_weather\">\
         <atem:parameter name=\"city\">Paris</atem:parameter>\
         <atem:parameter name=\"metric\">true</atem:parameter>\
         </atem:invoke></atem:function_calls>{EOM}\
         {START}assistant to=user{MESSAGE}It is 18C in Paris.{EOT}"
    )
}

#[tokio::test]
async fn full_turn_splits_into_reasoning_content_and_calls() {
    let parsed = run_pipeline(&reason_call_answer()).await;

    assert_eq!(parsed.reasoning, "The user wants current weather.");
    assert_eq!(parsed.content, "It is 18C in Paris.");
    assert_eq!(parsed.calls.len(), 1);
    assert_eq!(parsed.calls[0].0, "get_weather");
    assert_eq!(
        parsed.calls[0].1,
        serde_json::json!({"city": "Paris", "metric": true})
    );
}

/// No framing may reach the client on either visible field. A parser can return
/// the right call and still leak protocol markup into the answer.
#[tokio::test]
async fn no_protocol_framing_reaches_the_client() {
    let parsed = run_pipeline(&reason_call_answer()).await;

    for (field, text) in [
        ("reasoning", &parsed.reasoning),
        ("content", &parsed.content),
    ] {
        for marker in [START, MESSAGE, EOM, EOT, "<atem:", "to=self", "to=user"] {
            assert!(!text.contains(marker), "{field} leaked {marker}: {text:?}");
        }
    }
}

/// A turn that is only reasoning plus an answer must yield no calls, and the
/// reasoning must not bleed into the answer.
#[tokio::test]
async fn reasoning_only_turn_produces_no_calls() {
    let generation = format!(
        " to=self{MESSAGE}No tool needed here.{EOM}\
         {START}assistant to=user{MESSAGE}Paris is the capital of France.{EOT}"
    );

    let parsed = run_pipeline(&generation).await;

    assert_eq!(parsed.reasoning, "No tool needed here.");
    assert_eq!(parsed.content, "Paris is the capital of France.");
    assert!(parsed.calls.is_empty());
}

/// Markup the model quotes inside its own chain of thought must not survive the
/// handoff as a call. This is the failure the channel scoping exists to
/// prevent, and it can only be observed once both stages run together.
#[tokio::test]
async fn markup_quoted_in_reasoning_never_becomes_a_call() {
    let generation = format!(
        " to=self{MESSAGE}I could write <atem:function_calls>\
         <atem:invoke name=\"get_weather\">\
         <atem:parameter name=\"city\">Paris</atem:parameter>\
         </atem:invoke></atem:function_calls> but I need the city first.{EOM}\
         {START}assistant to=user{MESSAGE}Which city?{EOT}"
    );

    let parsed = run_pipeline(&generation).await;

    assert!(
        parsed.calls.is_empty(),
        "quoted markup became a call: {:?}",
        parsed.calls
    );
    assert_eq!(parsed.content, "Which city?");
    assert!(parsed.reasoning.contains("but I need the city first."));
}

/// A turn with no final answer — the model calls a tool and stops — must still
/// yield the call, with empty content rather than leaked framing.
#[tokio::test]
async fn tool_call_without_a_final_answer() {
    let generation = format!(
        " to=self{MESSAGE}Looking it up.{EOM}\
         {START}assistant to=get_weather{MESSAGE}<atem:function_calls>\
         <atem:invoke name=\"get_weather\">\
         <atem:parameter name=\"city\">Berlin</atem:parameter>\
         </atem:invoke></atem:function_calls>{EOT}"
    );

    let parsed = run_pipeline(&generation).await;

    assert_eq!(parsed.calls.len(), 1);
    assert_eq!(parsed.calls[0].0, "get_weather");
    assert_eq!(parsed.content, "");
    assert_eq!(parsed.reasoning, "Looking it up.");
}

/// Both stages must agree that this model needs its control tokens preserved.
/// If this regresses, detokenization strips the framing upstream and the whole
/// pipeline degrades to "everything is content" while still passing shape-only
/// assertions.
#[test]
fn reasoning_stage_requires_special_tokens() {
    let parser = ReasoningParserFactory::new().create(MODEL);
    assert_eq!(parser.model_type(), "muse_glimmer");
    assert!(parser.requires_special_tokens());
}
