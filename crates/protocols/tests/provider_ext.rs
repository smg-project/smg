//! Provider extension fields must survive the HTTP router's
//! deserialize→serialize round trip, and tool_choice validation must
//! accept "auto"/"none" without tools.

use openai_protocol::{
    chat::{ChatCompletionRequest, ChatMessage},
    common::{ImageUrl, ToolChoice, ToolChoiceValue, VideoUrl},
};
use serde_json::{json, Value};
use validator::Validate;

#[expect(clippy::expect_used, reason = "test helper")]
fn roundtrip(value: Value) -> Value {
    let req: ChatCompletionRequest = serde_json::from_value(value).expect("request deserializes");
    serde_json::to_value(&req).expect("request serializes")
}

#[test]
fn image_url_preserves_max_long_side_pixel() {
    let image: ImageUrl = serde_json::from_value(json!({
        "url": "data:image/png;base64,AAAA",
        "detail": "high",
        "max_long_side_pixel": 448
    }))
    .unwrap();
    assert_eq!(image.ext.max_long_side_pixel, Some(448));

    let out = serde_json::to_value(&image).unwrap();
    assert_eq!(out["max_long_side_pixel"], json!(448));
    assert_eq!(out["detail"], json!("high"));
}

#[test]
fn video_url_preserves_sizing_and_fps() {
    let video: VideoUrl = serde_json::from_value(json!({
        "url": "https://example.com/clip.mp4",
        "max_long_side_pixel": 896,
        "fps": 2.5
    }))
    .unwrap();
    assert_eq!(video.ext.max_long_side_pixel, Some(896));
    assert_eq!(video.ext.fps, Some(2.5));

    let out = serde_json::to_value(&video).unwrap();
    assert_eq!(out["max_long_side_pixel"], json!(896));
    assert_eq!(out["fps"], json!(2.5));
}

#[test]
fn media_parts_without_ext_stay_wire_identical() {
    let image: ImageUrl = serde_json::from_value(json!({"url": "u"})).unwrap();
    let out = serde_json::to_value(&image).unwrap();
    assert_eq!(out, json!({"url": "u"}));

    let video: VideoUrl = serde_json::from_value(json!({"url": "v"})).unwrap();
    let out = serde_json::to_value(&video).unwrap();
    assert_eq!(out, json!({"url": "v"}));
}

#[test]
fn system_message_preserves_dynamic_tools() {
    let request = json!({
        "model": "kimi-k3",
        "messages": [
            {"role": "system", "content": "", "tools": [
                {"type": "function", "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }}
            ]},
            {"role": "user", "content": "what is the weather in beijing?"}
        ],
        "tool_choice": "required"
    });
    let out = roundtrip(request);

    let system = &out["messages"][0];
    assert_eq!(system["role"], json!("system"));
    assert_eq!(
        system["tools"][0]["function"]["name"],
        json!("get_weather"),
        "dynamic tools must survive the round trip: {system}"
    );
}

#[test]
fn system_message_without_tools_has_no_tools_key() {
    let out = roundtrip(json!({
        "model": "kimi-k3",
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"}
        ]
    }));
    let system = out["messages"][0].as_object().unwrap();
    assert!(!system.contains_key("tools"));
}

#[test]
fn parsed_system_message_exposes_dynamic_tools() {
    let msg: ChatMessage = serde_json::from_value(json!({
        "role": "system",
        "content": "",
        "tools": [{"type": "function", "function": {"name": "get_time"}}]
    }))
    .unwrap();
    match msg {
        ChatMessage::System { ext, .. } => {
            let tools = ext.tools.expect("tools parsed");
            assert_eq!(tools.len(), 1);
            assert_eq!(tools[0].function.name, "get_time");
        }
        other => panic!("expected system message, got {other:?}"),
    }
}

#[expect(clippy::expect_used, reason = "test helper")]
fn request_with_tool_choice(tool_choice: ToolChoice) -> ChatCompletionRequest {
    let mut req: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .expect("request deserializes");
    req.tool_choice = Some(tool_choice);
    req
}

#[test]
fn tool_choice_auto_and_none_valid_without_tools() {
    for value in [ToolChoiceValue::Auto, ToolChoiceValue::None] {
        let req = request_with_tool_choice(ToolChoice::Value(value));
        assert!(
            req.validate().is_ok(),
            "{:?} must not require tools",
            req.tool_choice
        );
    }
}

#[test]
fn tool_choice_required_and_function_still_require_tools() {
    let required = request_with_tool_choice(ToolChoice::Value(ToolChoiceValue::Required));
    assert!(required.validate().is_err());

    let named: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
    }))
    .expect("request deserializes");
    assert!(named.validate().is_err());
}

#[test]
fn tool_choice_required_valid_with_only_dynamic_tools() {
    let req: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "kimi-k3",
        "messages": [
            {"role": "system", "content": "", "tools": [
                {"type": "function", "function": {"name": "get_weather"}}
            ]},
            {"role": "user", "content": "weather in beijing?"}
        ],
        "tool_choice": "required"
    }))
    .expect("request deserializes");
    assert!(
        req.validate().is_ok(),
        "dynamic tools must satisfy tool_choice=required"
    );
}

#[test]
fn system_message_without_content_defaults_to_empty() {
    let msg: ChatMessage = serde_json::from_value(json!({
        "role": "system",
        "tools": [{"type": "function", "function": {"name": "get_weather"}}]
    }))
    .expect("tools-only system message deserializes");
    match msg {
        ChatMessage::System { content, ext, .. } => {
            assert_eq!(content.to_simple_string(), "");
            assert!(ext.tools.is_some());
        }
        other => panic!("expected system message, got {other:?}"),
    }
}

#[expect(clippy::expect_used, reason = "test helper")]
fn request_with_tools_on_role(model: &str, role: &str) -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": model,
        "messages": [
            {"role": role, "content": "hi", "tools": [
                {"type": "function", "function": {"name": "get_weather"}}
            ]},
            {"role": "user", "content": "hello"}
        ]
    }))
    .expect("request deserializes")
}

#[test]
fn kimi_profile_rejects_tools_on_user_and_assistant() {
    for role in ["user", "assistant"] {
        let req = request_with_tools_on_role("kimi-k3", role);
        assert!(
            req.validate().is_err(),
            "kimi profile must reject tools on role {role}"
        );
    }
}

#[test]
fn kimi_profile_allows_tools_on_system() {
    let req = request_with_tools_on_role("kimi-k3", "system");
    assert!(req.validate().is_ok());
}

#[test]
fn non_kimi_models_tolerate_tools_on_any_role() {
    for model in ["gpt-4o-mini", "MiniMax-M3"] {
        let req = request_with_tools_on_role(model, "user");
        assert!(
            req.validate().is_ok(),
            "{model} must not enforce the kimi role restriction"
        );
    }
}

#[expect(clippy::expect_used, reason = "test helper")]
fn tool_history_request(model: &str, tool_call_id: &str, arguments: &str) -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": model,
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": arguments}}
            ]},
            {"role": "tool", "tool_call_id": tool_call_id, "content": "sunny"}
        ]
    }))
    .expect("request deserializes")
}

#[test]
fn minimax_profile_enforces_tool_history_strictness() {
    // MPV 16_08: unknown tool_call_id
    assert!(tool_history_request("MiniMax-M3", "call_999", "{}")
        .validate()
        .is_err());
    // MPV 16_12: invalid JSON arguments
    assert!(
        tool_history_request("MiniMax-M3", "call_1", "{invalid json}")
            .validate()
            .is_err()
    );
    // valid history passes
    assert!(
        tool_history_request("MiniMax-M3", "call_1", "{\"city\":\"Beijing\"}")
            .validate()
            .is_ok()
    );
}

#[test]
fn minimax_profile_rejects_unanswered_tool_calls() {
    let req: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "MiniMax-M3",
        "messages": [
            {"role": "user", "content": "weather in two cities?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}},
                {"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
        ]
    }))
    .expect("request deserializes");
    assert!(req.validate().is_err());
}

#[test]
fn kimi_and_openai_tolerate_loose_tool_history() {
    // KVV requires invalid-JSON history arguments to be ACCEPTED for Kimi
    for model in ["kimi-k3", "gpt-4o-mini"] {
        assert!(
            tool_history_request(model, "call_1", "{invalid json}")
                .validate()
                .is_ok(),
            "{model} must tolerate loose tool history"
        );
        assert!(
            tool_history_request(model, "call_999", "{}")
                .validate()
                .is_ok(),
            "{model} must tolerate id mismatch"
        );
    }
}

#[test]
fn minimax_normalizes_root_to_leading_system() {
    use openai_protocol::validated::Normalizable;
    let mut req: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "MiniMax-M3",
        "messages": [
            {"role": "root", "content": "top priority"},
            {"role": "user", "content": "hi"}
        ]
    }))
    .expect("root role deserializes");
    req.normalize();
    assert!(req.validate().is_ok());

    let out = serde_json::to_value(&req).expect("serializes");
    assert_eq!(out["messages"][0]["role"], json!("system"));
    assert_eq!(out["messages"][0]["content"], json!("top priority"));
}

#[test]
fn kimi_and_openai_reject_root_role() {
    for model in ["kimi-k3", "gpt-4o-mini"] {
        let req: ChatCompletionRequest = serde_json::from_value(json!({
            "model": model,
            "messages": [
                {"role": "root", "content": "x"},
                {"role": "user", "content": "hi"}
            ]
        }))
        .expect("root role deserializes");
        assert!(req.validate().is_err(), "{model} must reject role root");
    }
}
