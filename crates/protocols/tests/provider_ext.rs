//! Provider extension fields must survive the HTTP router's
//! deserialize→serialize round trip, and tool_choice validation must
//! accept "auto"/"none" without tools.

use openai_protocol::{
    chat::{ChatCompletionRequest, ChatMessage},
    common::{ImageUrl, ToolChoice, ToolChoiceValue, VideoUrl},
    validated::Normalizable,
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
    assert_eq!(image.max_long_side_pixel, Some(448));

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
    assert_eq!(video.max_long_side_pixel, Some(896));
    assert_eq!(video.fps, Some(2.5));

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
            let tools = tools.typed().expect("declaration parsed as tools");
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
    for role in ["system", "developer"] {
        let req: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "kimi-k3",
            "messages": [
                {"role": role, "content": "", "tools": [
                    {"type": "function", "function": {"name": "get_weather"}}
                ]},
                {"role": "user", "content": "weather in beijing?"}
            ],
            "tool_choice": "required"
        }))
        .expect("request deserializes");
        assert!(
            req.validate().is_ok(),
            "dynamic tools on {role} must satisfy tool_choice=required"
        );
    }
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

/// Every validation error code the request produced, schema-level ones included.
fn error_codes(req: &ChatCompletionRequest) -> Vec<String> {
    match req.validate() {
        Ok(()) => Vec::new(),
        Err(errors) => errors
            .field_errors()
            .values()
            .flat_map(|errs| errs.iter().map(|e| e.code.to_string()))
            .collect(),
    }
}

#[test]
fn kimi_profile_rejects_tools_on_user_and_assistant() {
    for role in ["user", "assistant"] {
        let mut req = request_with_tools_on_role("kimi-k3", role);
        req.normalize();
        assert!(
            error_codes(&req).contains(&"tools_role_restricted".to_string()),
            "kimi profile must reject tools on role {role} with its own code, got {:?}",
            error_codes(&req)
        );
    }
}

#[test]
fn non_kimi_models_ignore_message_tools_of_any_shape() {
    // The capture is raw JSON, so a malformed value on a role that only the
    // Kimi profile inspects is dropped as before rather than failing parsing.
    for model in ["gpt-4o-mini", "MiniMax-M3"] {
        for role in ["user", "assistant", "system", "developer"] {
            for tools in [json!({"name": "x"}), json!([{}]), json!("x"), json!(null)] {
                let mut req: ChatCompletionRequest = serde_json::from_value(json!({
                    "model": model,
                    "messages": [{"role": role, "content": "hi", "tools": tools}]
                }))
                .unwrap_or_else(|e| panic!("{model}/{role}/{tools}: {e}"));
                req.normalize();
                assert!(req.validate().is_ok(), "{model}/{role}/{tools}");
                let out = serde_json::to_value(&req).expect("serializes");
                assert!(
                    out["messages"][0].get("tools").is_none(),
                    "{model}/{role}/{tools}"
                );
            }
        }
    }
}

#[test]
fn kimi_profile_treats_null_tools_as_absent() {
    for role in ["user", "assistant", "system", "developer"] {
        let mut req: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "kimi-k3",
            "messages": [{"role": role, "content": "hi", "tools": null}]
        }))
        .expect("request deserializes");
        req.normalize();
        assert!(req.validate().is_ok(), "{role}: {:?}", error_codes(&req));
    }
}

#[test]
fn kimi_profile_rejects_malformed_tools_on_system_and_developer() {
    for role in ["system", "developer"] {
        for tools in [json!({"name": "x"}), json!([{}]), json!("x")] {
            let mut req: ChatCompletionRequest = serde_json::from_value(json!({
                "model": "kimi-k3",
                "messages": [{"role": role, "content": "", "tools": tools}]
            }))
            .expect("request deserializes");
            req.normalize();
            assert!(
                error_codes(&req).contains(&"tools_malformed".to_string()),
                "{role}/{tools}: {:?}",
                error_codes(&req)
            );
        }
    }
}

#[test]
fn kimi_profile_rejects_tools_of_any_shape_on_user_and_assistant() {
    for role in ["user", "assistant"] {
        for tools in [json!({"name": "x"}), json!([{}]), json!("x"), json!([])] {
            let mut req: ChatCompletionRequest = serde_json::from_value(json!({
                "model": "kimi-k3",
                "messages": [{"role": role, "content": "hi", "tools": tools}]
            }))
            .expect("request deserializes");
            req.normalize();
            assert!(
                error_codes(&req).contains(&"tools_role_restricted".to_string()),
                "{role}/{tools}: {:?}",
                error_codes(&req)
            );
        }
    }
}

#[test]
fn kimi_profile_allows_tools_on_developer_like_system() {
    // `developer` supersedes `system` in the OpenAI spec, and the verifier
    // has no case for it, so it follows the system rule.
    let mut req = request_with_tools_on_role("kimi-k3", "developer");
    req.normalize();
    assert!(req.validate().is_ok(), "{:?}", error_codes(&req));
}

#[test]
fn kimi_profile_rejects_an_empty_tools_list_on_user() {
    // The contract keys on the key being declared, not on its contents.
    let mut req: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "hi", "tools": []}]
    }))
    .expect("request deserializes");
    req.normalize();
    assert!(error_codes(&req).contains(&"tools_role_restricted".to_string()));
}

#[test]
fn kimi_profile_allows_tools_on_system() {
    let req = request_with_tools_on_role("kimi-k3", "system");
    assert!(req.validate().is_ok());
}

#[test]
fn non_kimi_models_tolerate_tools_on_any_role() {
    for model in ["gpt-4o-mini", "MiniMax-M3"] {
        for role in ["user", "assistant", "developer"] {
            let mut req = request_with_tools_on_role(model, role);
            req.normalize();
            assert!(
                req.validate().is_ok(),
                "{model} must not enforce the kimi role restriction on {role}"
            );
        }
    }
}

#[expect(clippy::expect_used, reason = "test helper")]
fn normalized(value: Value) -> Value {
    let mut req: ChatCompletionRequest =
        serde_json::from_value(value).expect("request deserializes");
    req.normalize();
    serde_json::to_value(&req).expect("request serializes")
}

fn kimi_ext_request(model: &str) -> Value {
    json!({
        "model": model,
        "messages": [
            {"role": "system", "content": "", "tools": [{"type": "function", "function": {"name": "f"}}]},
            {"role": "developer", "content": "", "tools": [{"type": "function", "function": {"name": "d"}}]},
            {"role": "user", "content": "hi", "tools": [{"type": "function", "function": {"name": "g"}}]},
            {"role": "assistant", "content": "ok", "tools": [{"type": "function", "function": {"name": "h"}}]}
        ]
    })
}

#[test]
fn openai_profile_drops_kimi_extensions_on_normalize() {
    // Typed so Kimi can reject them, they must not reach an OpenAI backend:
    // the same outcome as when serde dropped the unknown key.
    let out = normalized(kimi_ext_request("gpt-4o"));
    for message in out["messages"].as_array().expect("messages") {
        assert!(message.get("tools").is_none(), "{message}");
    }
}

#[test]
fn minimax_profile_drops_kimi_extensions_on_normalize() {
    let out = normalized(kimi_ext_request("MiniMax-M3"));
    for message in out["messages"].as_array().expect("messages") {
        assert!(message.get("tools").is_none(), "{message}");
    }
}

#[test]
fn kimi_profile_keeps_its_extensions_on_normalize() {
    // Kept on every role: the system tools are the feature, and the user and
    // assistant ones stay for the profile's rules to reject with a 400.
    let out = normalized(kimi_ext_request("kimi-k3"));
    for message in out["messages"].as_array().expect("messages") {
        assert!(message.get("tools").is_some(), "{message}");
    }
}

#[test]
fn vendor_model_ids_in_paths_and_aggregator_prefixes_keep_kimi_extensions() {
    // Profile selection must agree with the parser factories, or a working
    // feature disappears silently on a mis-detected id.
    for model in [
        "/models/Kimi-K3",
        "moonshotai/kimi-k2",
        "openrouter/moonshotai/kimi-k2",
        "MoonshotAI/Kimi-K2-Instruct",
    ] {
        let out = normalized(kimi_ext_request(model));
        assert!(
            out["messages"][0].get("tools").is_some(),
            "{model}: system tools must survive normalization: {out}"
        );
    }
}

#[test]
fn stripping_runs_before_validation_so_tool_choice_required_needs_request_tools() {
    // For a non-Kimi model the system-message tools are gone by the time
    // rule 7 runs, so nothing can satisfy tool_choice=required: a 400, not a
    // 200 that the backend then cannot honour.
    let mut req: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "", "tools": [{"type": "function", "function": {"name": "f"}}]},
            {"role": "user", "content": "hi"}
        ],
        "tool_choice": "required"
    }))
    .expect("request deserializes");
    req.normalize();
    assert!(error_codes(&req).contains(&"tool_choice_requires_tools".to_string()));

    // The Kimi profile keeps them, so the same request validates there.
    let mut req: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "kimi-k3",
        "messages": [
            {"role": "system", "content": "", "tools": [{"type": "function", "function": {"name": "f"}}]},
            {"role": "user", "content": "hi"}
        ],
        "tool_choice": "required"
    }))
    .expect("request deserializes");
    req.normalize();
    assert!(req.validate().is_ok(), "{:?}", error_codes(&req));
}

#[expect(clippy::expect_used, reason = "test helper")]
fn dynamic_tools_request(
    role: &str,
    request_tools: Value,
    tool_choice: Value,
) -> ChatCompletionRequest {
    let mut req: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "kimi-k3",
        "messages": [
            {"role": role, "content": "", "tools": [
                {"type": "function", "function": {"name": "get_weather"}}
            ]},
            {"role": "user", "content": "weather in beijing?"}
        ],
        "tool_choice": tool_choice
    }))
    .expect("request deserializes");
    if !request_tools.is_null() {
        req.tools = Some(serde_json::from_value(request_tools).expect("tools deserialize"));
    }
    req.normalize();
    req
}

fn named(name: &str) -> Value {
    json!({"type": "function", "function": {"name": name}})
}

fn allowed(name: &str) -> Value {
    json!({"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": name}]})
}

#[test]
fn named_tool_choice_resolves_against_dynamic_tools() {
    // Only dynamic tools, on either role that may declare them: a declared
    // name is accepted and an unknown one rejected, for both choice shapes.
    for role in ["system", "developer"] {
        let known = dynamic_tools_request(role, Value::Null, named("get_weather"));
        assert!(
            known.validate().is_ok(),
            "{role}: {:?}",
            error_codes(&known)
        );
        let known = dynamic_tools_request(role, Value::Null, allowed("get_weather"));
        assert!(
            known.validate().is_ok(),
            "{role}: {:?}",
            error_codes(&known)
        );

        let unknown = dynamic_tools_request(role, Value::Null, named("get_time"));
        assert!(
            error_codes(&unknown).contains(&"tool_choice_function_not_found".to_string()),
            "{role}: {:?}",
            error_codes(&unknown)
        );
        let unknown = dynamic_tools_request(role, Value::Null, allowed("get_time"));
        assert!(
            error_codes(&unknown).contains(&"tool_choice_tool_not_found".to_string()),
            "{role}: {:?}",
            error_codes(&unknown)
        );
    }
}

#[test]
fn named_tool_choice_sees_dynamic_tools_beside_request_tools() {
    // Unrelated request-level tools must not hide a dynamic tool's name.
    for role in ["system", "developer"] {
        let req = dynamic_tools_request(
            role,
            json!([{"type": "function", "function": {"name": "unrelated"}}]),
            named("get_weather"),
        );
        assert!(req.validate().is_ok(), "{role}: {:?}", error_codes(&req));
    }
}
