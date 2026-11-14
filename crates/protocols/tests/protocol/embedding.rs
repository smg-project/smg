//! The embeddings types against the public API's shape.

use openai_protocol::embedding::EmbeddingResponse;
use serde_json::json;

/// The public API's embeddings usage has `prompt_tokens` and `total_tokens`
/// only. An answer written that way parses, with no completion tokens, and its
/// usage is written back with `completion_tokens: 0`, as the gateway writes
/// its own.
#[test]
fn an_embeddings_usage_without_completion_tokens_parses() {
    let answer: EmbeddingResponse = serde_json::from_value(json!({
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.5, -0.25], "index": 0}],
        "model": "m",
        "usage": {"prompt_tokens": 3, "total_tokens": 3}
    }))
    .expect("the public API's embeddings answer parses");
    assert_eq!(answer.usage.completion_tokens, 0);
    assert_eq!(
        serde_json::to_value(&answer.usage).unwrap(),
        json!({"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3}),
        "a usage is written as before"
    );
}
