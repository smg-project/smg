//! The legacy completion types against the public API's shape.

use openai_protocol::completion::{CompletionResponse, CompletionStreamResponse};
use serde_json::{json, Value};

/// A choice without log probabilities writes `"logprobs": null`, as the
/// public API sends it and a chat choice writes it, in a body and in a
/// chunk; a choice with them writes them as before.
#[test]
fn a_completion_choice_writes_logprobs_as_null_when_none_were_asked_for() {
    let body: CompletionResponse = serde_json::from_value(json!({
        "id": "cmpl-1", "object": "text_completion", "created": 1, "model": "m",
        "choices": [{"text": "Hello", "index": 0, "finish_reason": "stop"}]
    }))
    .unwrap();
    let written = serde_json::to_value(&body).unwrap();
    assert_eq!(
        written["choices"][0].get("logprobs"),
        Some(&Value::Null),
        "a body's choice writes the key as null: {written}"
    );

    let chunk: CompletionStreamResponse = serde_json::from_value(json!({
        "id": "cmpl-1", "object": "text_completion", "created": 1, "model": "m",
        "choices": [{"text": "Hel", "index": 0, "finish_reason": null}]
    }))
    .unwrap();
    let written = serde_json::to_value(&chunk).unwrap();
    assert_eq!(
        written["choices"][0].get("logprobs"),
        Some(&Value::Null),
        "a chunk's choice writes the key as null: {written}"
    );

    let logprobs = json!({
        "tokens": ["Hello"], "token_logprobs": [-0.5],
        "top_logprobs": [{"Hello": -0.5}], "text_offset": [0]
    });
    let with: CompletionResponse = serde_json::from_value(json!({
        "id": "cmpl-1", "object": "text_completion", "created": 1, "model": "m",
        "choices": [{"text": "Hello", "index": 0, "logprobs": logprobs, "finish_reason": "stop"}]
    }))
    .unwrap();
    let written = serde_json::to_value(&with).unwrap();
    assert_eq!(
        written["choices"][0].get("logprobs"),
        Some(&logprobs),
        "a choice with log probabilities writes them as they were"
    );
}
