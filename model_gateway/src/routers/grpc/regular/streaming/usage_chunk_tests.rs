//! The usage chunk of a streamed chat completion per provider profile: a
//! stream the client sent without `stream_options` ends with one under the
//! MiniMax profile (the vendor's API ends every stream with a usage chunk,
//! which its clients read the token counts from) and under none of the
//! profiles without that default, where the client asks for it.

use axum::{body::Body, http::StatusCode};
use http_body_util::BodyExt;
use openai_protocol::{chat::ChatCompletionRequest, validated::Normalizable};

use super::{
    eof_tests::{chunk, complete, processor, scripted_stream, CharacterTokenizer},
    *,
};

/// The spec of a streamed chat request for `model` with `fields` on top, as
/// the request reaches the pipeline: normalized by its provider profile.
fn spec_for(model: &str, fields: Value) -> ChatResponseSpec {
    let mut request = serde_json::json!({ "model": model, "messages": [], "stream": true });
    request
        .as_object_mut()
        .unwrap()
        .extend(fields.as_object().cloned().unwrap_or_default());
    let mut request: ChatCompletionRequest = serde_json::from_value(request).expect("chat request");
    request.normalize();
    ChatResponseSpec::from(&request)
}

/// The `data:` payloads of one streamed chat completion for `spec`, in order
/// (`[DONE]` included).
async fn streamed_payloads(spec: ChatResponseSpec) -> Vec<String> {
    let (stream, server) = scripted_stream(vec![chunk(0, "hi"), complete(0, "stop")], "0").await;
    let dispatch = context::DispatchMetadata {
        request_id: "chatcmpl-usage".to_string(),
        model: "m".to_string(),
        created: 1_700_000_000,
        weight_version: None,
        tenant_key: None,
        prompt_tokens: 0,
    };
    let source = utils::StreamSource::new(None, &dispatch, metrics_labels::ENDPOINT_CHAT);
    let response = Arc::new(processor(false))
        .process_streaming_response(
            context::ExecutionResult::Single { stream },
            spec,
            dispatch,
            Arc::new(CharacterTokenizer::default()),
            false,
            None,
            None,
            source,
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let mut body: Body = response.into_body();
    let mut bytes = Vec::new();
    while let Some(frame) = body.frame().await {
        if let Some(data) = frame.expect("an SSE frame").data_ref() {
            bytes.extend_from_slice(data);
        }
    }
    server.abort();
    String::from_utf8(bytes)
        .expect("utf-8 SSE")
        .split("\n\n")
        .filter_map(|event| event.strip_prefix("data: ").map(str::to_string))
        .collect()
}

/// The usage object of the event before `[DONE]`, if that event carries one.
fn closing_usage(payloads: &[String]) -> Option<Value> {
    assert_eq!(
        payloads.last().map(String::as_str),
        Some("[DONE]"),
        "{payloads:?}"
    );
    let closing: Value = serde_json::from_str(&payloads[payloads.len() - 2]).expect("a chunk");
    closing
        .get("usage")
        .filter(|usage| !usage.is_null())
        .cloned()
}

#[tokio::test]
async fn a_minimax_stream_sent_without_stream_options_ends_with_a_usage_chunk() {
    let payloads = streamed_payloads(spec_for("MiniMax-M3", serde_json::json!({}))).await;
    let usage = closing_usage(&payloads).expect("the usage chunk before [DONE]");
    assert!(usage["completion_tokens"].is_u64(), "{usage}");
    assert!(usage["prompt_tokens"].is_u64(), "{usage}");
}

#[tokio::test]
async fn a_minimax_stream_that_opts_out_of_usage_ends_without_one() {
    let payloads = streamed_payloads(spec_for(
        "MiniMax-M3",
        serde_json::json!({"stream_options": {"include_usage": false}}),
    ))
    .await;
    assert_eq!(closing_usage(&payloads), None, "{payloads:?}");
}

#[tokio::test]
async fn a_plain_served_models_stream_carries_usage_only_when_asked() {
    let payloads = streamed_payloads(spec_for("self-hosted-model", serde_json::json!({}))).await;
    assert_eq!(closing_usage(&payloads), None, "{payloads:?}");
    assert!(
        payloads
            .iter()
            .all(|payload| !payload.contains("\"usage\":{")),
        "{payloads:?}"
    );

    let payloads = streamed_payloads(spec_for(
        "self-hosted-model",
        serde_json::json!({"stream_options": {"include_usage": true}}),
    ))
    .await;
    assert!(closing_usage(&payloads).is_some(), "{payloads:?}");
}
