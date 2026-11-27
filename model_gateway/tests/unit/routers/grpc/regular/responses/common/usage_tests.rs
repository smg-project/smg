use serde_json::json;

use super::*;

#[test]
fn dispatch_arguments_keep_an_object_and_make_an_empty_one_of_anything_else() {
    let request = ResponsesRequest::default();
    assert_eq!(
        dispatch_arguments(ResponseFormat::Passthrough, &request, r#"{"q": "x"}"#),
        json!({"q": "x"})
    );
    for payload in ["\"text\"", "[1, 2]", "null", "not json"] {
        assert_eq!(
            dispatch_arguments(ResponseFormat::Passthrough, &request, payload),
            json!({}),
            "{payload}"
        );
    }
}

#[test]
fn dispatch_arguments_merge_the_requests_hosted_tool_config_and_user_into_a_coerced_payload() {
    // The reason for the coercion: a scalar payload under a hosted format
    // still takes the caller-declared config and the request's user.
    let request = ResponsesRequest {
        tools: Some(vec![responses::ResponseTool::WebSearchPreview(
            responses::WebSearchPreviewTool {
                search_context_size: Some("low".to_string()),
                ..Default::default()
            },
        )]),
        user: Some("u-1".to_string()),
        ..Default::default()
    };
    let merged = dispatch_arguments(ResponseFormat::WebSearchCall, &request, "\"text\"");
    assert_eq!(merged["search_context_size"], json!("low"));
    assert_eq!(merged["user"], json!("u-1"));
}

#[test]
fn mcp_usage_accumulates_reasoning_and_cache_and_survives_missing_usage() {
    let mut state = ToolLoopState::default();
    state.record_usage(None);
    assert!(state.usage.is_none());
    state.record_usage(Some(
        &Usage::from_counts(12, 7)
            .with_cached_tokens(3)
            .with_reasoning_tokens(2),
    ));
    state.record_usage(Some(
        &Usage::from_counts(5, 4)
            .with_cached_tokens(0)
            .with_reasoning_tokens(1),
    ));
    state.record_usage(None);
    assert_eq!(
        serde_json::to_value(state.usage).unwrap(),
        json!({
            "input_tokens":17, "output_tokens":11, "total_tokens":28,
            "input_tokens_details":{"cached_tokens":3},
            "output_tokens_details":{"reasoning_tokens":3},
        })
    );
}
