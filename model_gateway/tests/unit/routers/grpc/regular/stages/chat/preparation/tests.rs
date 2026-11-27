use openai_protocol::validated::Normalizable;
use serde_json::{json, Value};

use super::*;
use crate::routers::grpc::testing::tool;

/// A system message declaring `tools`, as the profiles that read such
/// declarations take it.
fn declaring(tools: &[Value]) -> Value {
    json!({"role": "system", "content": "", "tools": tools})
}

fn user() -> Value {
    json!({"role": "user", "content": "weather?"})
}

/// A request as the HTTP layer hands it on: normalized, so an empty `tools`
/// list has become `tool_choice: none`.
fn request(value: Value) -> ChatCompletionRequest {
    let mut request: ChatCompletionRequest =
        serde_json::from_value(value).expect("request deserializes");
    request.normalize();
    request
}

/// [`skips_special_tokens`] under the request's own provider profile.
fn skips(request: &ChatCompletionRequest, json_schema_constraint: bool, kept: bool) -> bool {
    skips_special_tokens(
        request,
        ProviderProfile::for_model(&request.model),
        json_schema_constraint,
        kept,
    )
}

fn message_level_only() -> ChatCompletionRequest {
    request(json!({
        "model": "kimi-k3",
        "messages": [declaring(&[tool("get_weather")]), user()]
    }))
}

#[test]
fn tools_keep_the_special_tokens_wherever_they_are_declared() {
    let request_level_only = request(json!({
        "model": "kimi-k3",
        "messages": [user()],
        "tools": [tool("get_time")]
    }));
    assert!(!skips(&request_level_only, false, false));

    assert!(!skips(&message_level_only(), false, false));

    let both = request(json!({
        "model": "kimi-k3",
        "messages": [declaring(&[tool("get_weather")]), user()],
        "tools": [tool("get_time")]
    }));
    assert!(!skips(&both, false, false));

    // No tools anywhere: the request's own setting, whichever way it goes.
    let none = request(json!({"model": "kimi-k3", "messages": [user()]}));
    assert!(skips(&none, false, false));
    let kept = request(json!({
        "model": "kimi-k3",
        "messages": [user()],
        "skip_special_tokens": false
    }));
    assert!(!skips(&kept, false, false));
}

#[test]
fn conflicting_declarations_resolve_as_the_other_gates_resolve_them() {
    // An empty request-level list beside a message's declaration: the
    // normalizer's `tool_choice: none` switches the calls off, as it does
    // without the message, so the parser is off too and nothing is kept.
    let empty_list = request(json!({
        "model": "kimi-k3",
        "messages": [declaring(&[tool("get_weather")]), user()],
        "tools": []
    }));
    assert!(matches!(
        empty_list.tool_choice,
        Some(ToolChoice::Value(ToolChoiceValue::None))
    ));
    assert!(skips(&empty_list, false, false));

    // `tool_choice: none` written by the client, over a message's declaration.
    let switched_off = request(json!({
        "model": "kimi-k3",
        "messages": [declaring(&[tool("get_weather")]), user()],
        "tool_choice": "none"
    }));
    assert!(skips(&switched_off, false, false));

    // A message's declaration under a profile that reads none widens nothing.
    let unread = request(json!({
        "model": "self-hosted-model",
        "messages": [declaring(&[tool("get_weather")]), user()]
    }));
    assert!(skips(&unread, false, false));
}

#[test]
fn the_constraint_and_the_reasoning_markers_decide_before_the_tools() {
    // A JSON-schema constraint forces the call's JSON: nothing to keep, the
    // request's setting stands even with a message-declared tool.
    assert!(skips(&message_level_only(), true, false));

    // Reasoning markers the decoder would drop are kept whatever the request says.
    let none = request(json!({"model": "kimi-k3", "messages": [user()]}));
    assert!(!skips(&none, false, true));
    assert!(!skips(&none, true, true));
}

/// Media of a model without a media spec is judged by the template the
/// request renders through, the one for requests with tools when the
/// request declares tools and the default one otherwise: refused where
/// that template takes one string per message, served where it takes
/// content parts.
#[tokio::test]
async fn media_without_a_spec_is_judged_by_the_template_the_request_renders_through() {
    use std::sync::Arc;

    use llm_tokenizer::{chat_template::ChatTemplateContentFormat, mock::MockTokenizer};

    use crate::routers::grpc::regular::stages::testing::components_forwarding_media_of;

    for (default, with_tools) in [
        (
            ChatTemplateContentFormat::String,
            ChatTemplateContentFormat::OpenAI,
        ),
        (
            ChatTemplateContentFormat::OpenAI,
            ChatTemplateContentFormat::String,
        ),
    ] {
        let tokenizer = MockTokenizer::new()
            .with_deferred_chat_ids(vec![1, 2, 3])
            .with_content_format(default)
            .with_tool_content_format(with_tools);
        let (_dir, components) = components_forwarding_media_of("m", Arc::new(tokenizer)).await;
        for (declares_tools, format) in [(true, with_tools), (false, default)] {
            let mut body = json!({
                "model": "m",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "https://a/1.png"}}
                ]}]
            });
            if declares_tools {
                body["tools"] = json!([{"type": "function", "function": {
                    "name": "lookup", "parameters": {"type": "object", "properties": {}}
                }}]);
            }
            let mut ctx = RequestContext::for_chat(
                Arc::new(request(body)),
                None,
                "m".into(),
                components.clone(),
            );
            let refused_with = ChatPreparationStage::new(&[])
                .prepare(&mut ctx)
                .await
                .err()
                .map(|response| {
                    response
                        .headers()
                        .get("x-smg-error-code")
                        .and_then(|code| code.to_str().ok())
                        .unwrap_or_default()
                        .to_owned()
                });
            assert_eq!(
                refused_with.as_deref(),
                (format == ChatTemplateContentFormat::String).then_some("multimodal_not_supported"),
                "declares tools: {declares_tools}"
            );
        }
    }
}
