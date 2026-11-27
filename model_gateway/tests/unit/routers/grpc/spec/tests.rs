use serde_json::{json, Value};

use super::*;
use crate::routers::grpc::testing::tool;

fn chat_request(value: Value) -> ChatCompletionRequest {
    serde_json::from_value(value).expect("request deserializes")
}

fn tool_names(spec: &ChatResponseSpec) -> Vec<String> {
    spec.tools
        .iter()
        .flat_map(|tools| tools.iter())
        .map(|tool| tool.function.name.clone())
        .collect()
}

#[test]
fn chat_spec_tools_include_dynamic_tools() {
    let request = chat_request(json!({
        "model": "kimi-k3",
        "messages": [
            {"role": "system", "content": "", "tools": [tool("get_weather")]},
            {"role": "user", "content": "what is the weather in beijing?"}
        ],
        "tool_choice": "required"
    }));

    let spec = ChatResponseSpec::from(&request);

    assert_eq!(tool_names(&spec), ["get_weather"]);
}

/// The point of sharing: with request tools and no message-declared tool,
/// the spec holds the request's own list, not a copy of it.
#[test]
fn chat_spec_shares_the_request_s_tool_list() {
    let request = chat_request(json!({
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [tool("get_weather")]
    }));

    let spec = ChatResponseSpec::from(&request);

    assert!(Arc::ptr_eq(
        request.tools.as_ref().expect("the request's tools"),
        spec.tools.as_ref().expect("the spec's tools")
    ));
}

#[test]
fn chat_spec_tools_keep_request_tools_first() {
    let request = chat_request(json!({
        "model": "kimi-k3",
        "messages": [
            {"role": "system", "content": "", "tools": [tool("dynamic_a")]},
            {"role": "user", "content": "hi"},
            {"role": "developer", "content": "", "tools": [tool("dynamic_b")]}
        ],
        "tools": [tool("global")]
    }));

    let spec = ChatResponseSpec::from(&request);

    assert_eq!(tool_names(&spec), ["global", "dynamic_a", "dynamic_b"]);
}

#[test]
fn chat_spec_tools_follow_an_empty_request_list_beside_a_message_declaration() {
    // The empty request-level list is a declaration (the normalizer answers
    // it with `tool_choice: none`, which switches the calls off); the
    // message's tool is the set.
    let request = chat_request(json!({
        "model": "kimi-k3",
        "messages": [
            {"role": "system", "content": "", "tools": [tool("get_weather")]},
            {"role": "user", "content": "hi"}
        ],
        "tools": []
    }));

    assert_eq!(
        tool_names(&ChatResponseSpec::from(&request)),
        ["get_weather"]
    );
}

#[test]
fn chat_spec_tools_none_for_a_message_declaration_the_profile_does_not_read() {
    let request = chat_request(json!({
        "model": "self-hosted-model",
        "messages": [
            {"role": "system", "content": "", "tools": [tool("get_weather")]},
            {"role": "user", "content": "hi"}
        ]
    }));

    assert!(ChatResponseSpec::from(&request).tools.is_none());
}

#[test]
fn chat_spec_tools_none_without_any_declaration() {
    let request = chat_request(json!({
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "hi"}]
    }));

    assert!(ChatResponseSpec::from(&request).tools.is_none());
}

/// The wire follows the profile of the requested model, which the
/// stream's shape then reads off the spec.
#[test]
fn chat_spec_wire_follows_the_requested_model() {
    for (model, wire) in [
        ("MiniMax-M3", Wire::Minimax),
        ("deepseek-flash", Wire::DeepSeek),
        ("gpt-4o", Wire::OpenAi),
        ("self-hosted-model", Wire::OpenAi),
    ] {
        let request = chat_request(json!({
            "model": model,
            "messages": [{"role": "user", "content": "hi"}]
        }));

        assert_eq!(ChatResponseSpec::from(&request).wire, wire, "{model}");
    }
}

#[test]
fn chat_spec_scans_minimax_responses_for_tool_calls_without_tools() {
    let request = chat_request(json!({
        "model": "MiniMax-M3",
        "messages": [{"role": "user", "content": "try again"}]
    }));

    assert_eq!(
        ChatResponseSpec::from(&request).tools.as_deref(),
        Some(&[][..])
    );
}

/// The Messages spec carries the request's budget; the window waits for
/// request building.
#[test]
fn messages_spec_takes_the_budget_from_the_request() {
    let request: CreateMessageRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .expect("request deserializes");

    let spec = MessagesResponseSpec::from(&request);

    assert_eq!(spec.max_tokens, 64);
    assert_eq!(spec.context_window, None);
}

/// An engine `length` finish is `max_tokens` when the budget was used
/// whole and `model_context_window_exceeded` when the generation stopped
/// short of it or the prompt and the output together filled the window.
#[test]
fn a_length_finish_reads_the_budget_against_the_window() {
    use messages::StopReason::{MaxTokens, ModelContextWindowExceeded};

    let mut spec = MessagesResponseSpec::from(
        &serde_json::from_value::<CreateMessageRequest>(json!({
            "model": "m", "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .expect("request deserializes"),
    );
    spec.prompt_tokens = 100;

    // No window known: the budget alone decides.
    assert_eq!(spec.length_stop_reason(100, 64), MaxTokens);
    assert_eq!(spec.length_stop_reason(100, 3), ModelContextWindowExceeded);
    assert_eq!(spec.length_stop_reason(100, 0), ModelContextWindowExceeded);

    // A window the whole budget stays inside: still the budget.
    spec.context_window = Some(165);
    assert_eq!(spec.length_stop_reason(100, 64), MaxTokens);
    // The prompt and the output together reach the window.
    spec.context_window = Some(164);
    assert_eq!(spec.length_stop_reason(100, 64), ModelContextWindowExceeded);
    // Short of the budget, whatever the window.
    assert_eq!(spec.length_stop_reason(100, 3), ModelContextWindowExceeded);
    // The engine's count decides, not the rendered one the spec carries:
    // media expanded on the worker fills the window the spec would miss.
    spec.context_window = Some(170);
    assert_eq!(spec.length_stop_reason(100, 64), MaxTokens);
    assert_eq!(spec.length_stop_reason(110, 64), ModelContextWindowExceeded);
    // A synthetic count never overflows the sum.
    assert_eq!(
        spec.length_stop_reason(u32::MAX, 64),
        ModelContextWindowExceeded
    );
}

#[test]
fn chat_spec_selects_json_mode_only_for_structured_json() {
    for (format, expected) in [
        (Value::Null, false),
        (json!({"type": "text"}), false),
        (json!({"type": "json_object"}), true),
        (
            json!({"type": "json_schema", "json_schema": {
                "name": "answer", "schema": {"type": "object"}
            }}),
            true,
        ),
    ] {
        let request = chat_request(json!({
            "model": "test", "messages": [], "response_format": format
        }));
        assert_eq!(ChatResponseSpec::from(&request).json_mode, expected);
    }
}
