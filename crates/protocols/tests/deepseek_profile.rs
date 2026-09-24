//! DeepSeek V4 rules calibrated against the official Chat API.
use openai_protocol::{
    chat::ChatCompletionRequest, profile::ProviderProfile, validated::Normalizable,
};
use serde_json::{json, Value};
use validator::Validate;

#[expect(clippy::expect_used, reason = "test helper")]
fn request(extra: Value) -> ChatCompletionRequest {
    let mut body =
        json!({"model": "deepseek-flash", "messages": [{"role": "user", "content": "hello"}]});
    body.as_object_mut()
        .expect("object")
        .extend(extra.as_object().expect("object").clone());
    let mut req: ChatCompletionRequest = serde_json::from_value(body).expect("request");
    req.normalize();
    req
}

#[test]
fn deepseek_profile_is_limited_to_the_verified_v4_family() {
    for model in [
        "deepseek-flash",
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "/models/DeepSeek-V4.1-Flash",
    ] {
        assert_ne!(
            ProviderProfile::for_model(model),
            ProviderProfile::OpenAi,
            "{model}"
        );
    }
    for model in [
        "deepseek-chat",
        "deepseek-reasoner",
        "deepseek-ai/DeepSeek-V3.2",
        "deepseek-ai/DeepSeek-R1",
        "deepseek-v40-pro",
        "deepseek-v4.10-flash",
        "deepseek-v4.2-flash",
        "deepseek-flashlight",
        "my-deepseek-v4-pro",
        "opaque-alias",
    ] {
        assert_eq!(
            ProviderProfile::for_model(model),
            ProviderProfile::OpenAi,
            "{model}"
        );
    }
}

#[test]
fn deepseek_effort_aliases_preserve_thinking_and_normalize_levels() {
    for (input, canonical) in [
        ("minimal", "low"),
        ("medium", "high"),
        ("xhigh", "high"),
        ("none", "none"),
        ("low", "low"),
        ("high", "high"),
        ("max", "max"),
    ] {
        let req = request(json!({"reasoning_effort": input}));
        assert!(req.validate().is_ok());
        assert_eq!(req.effective_reasoning_effort(), Some(canonical));
        let req = request(json!({"thinking": {"effort": input}}));
        assert!(req.validate().is_ok());
        assert_eq!(req.effective_reasoning_effort(), Some(canonical));
    }
    let req = request(json!({"model": "gpt-4o", "reasoning_effort": "minimal"}));
    assert_eq!(req.effective_reasoning_effort(), Some("minimal"));
}

#[test]
fn deepseek_rejects_unsupported_thinking_and_effort() {
    for extra in [
        json!({"thinking": {"type": "adaptive"}}),
        json!({"reasoning_effort": "turbo"}),
        json!({"thinking": {"effort": "turbo"}, "reasoning_effort": "high"}),
    ] {
        assert!(request(extra.clone()).validate().is_err(), "{extra}");
    }
}

#[test]
fn deepseek_forced_tools_require_non_thinking_mode() {
    for choice in [
        json!("required"),
        json!({"type": "function", "function": {"name": "weather"}}),
    ] {
        for (mode, valid) in [
            (json!({}), false),
            (json!({"thinking": {"type": "enabled"}}), false),
            (json!({"thinking": {"type": "disabled"}}), true),
            (json!({"reasoning_effort": "none"}), true),
            (json!({"reasoning_effort": "minimal"}), false),
            (json!({"chat_template_kwargs": {"thinking": false}}), true),
            (
                json!({"chat_template_kwargs": {"enable_thinking": false}}),
                true,
            ),
            (
                json!({"chat_template_kwargs": {"reasoning_effort": "none"}}),
                true,
            ),
        ] {
            let mut extra = json!({"tools": [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object", "properties": {}}}}], "tool_choice": choice});
            extra
                .as_object_mut()
                .unwrap()
                .extend(mode.as_object().unwrap().clone());
            assert_eq!(request(extra.clone()).validate().is_ok(), valid, "{extra}");
        }
    }
    for choice in ["auto", "none"] {
        assert!(request(json!({"tool_choice": choice})).validate().is_ok());
    }
}

#[test]
fn deepseek_projects_one_thinking_decision_for_validation_and_rendering() {
    for model in ["deepseek-v4-pro", "deepseek-ai/DeepSeek-V4.1-Flash"] {
        for (extra, enabled) in [
            (json!({}), true),
            (json!({"reasoning_effort":"minimal"}), true),
            (json!({"reasoning_effort":"low"}), true),
            (json!({"reasoning_effort":"none"}), false),
            (
                json!({"thinking":{"type":"disabled"},"reasoning_effort":"high"}),
                false,
            ),
            (
                json!({"thinking":{"type":"enabled"},"reasoning_effort":"none"}),
                true,
            ),
            (
                json!({"chat_template_kwargs":{"enable_thinking":false},"reasoning_effort":"high"}),
                false,
            ),
            (
                json!({"chat_template_kwargs":{"thinking":true},"thinking":{"type":"disabled"}}),
                true,
            ),
            (
                json!({"chat_template_kwargs":{"reasoning_effort":"none"},"thinking":{"type":"enabled"}}),
                false,
            ),
        ] {
            let mut extra = extra;
            extra["model"] = json!(model);
            let mut req = request(extra.clone());
            assert_eq!(
                req.chat_template_kwargs
                    .as_ref()
                    .and_then(|k| k.get("thinking")),
                Some(&json!(enabled)),
                "{extra}"
            );
            let once = serde_json::to_value(&req).unwrap();
            req.normalize();
            assert_eq!(
                serde_json::to_value(&req).unwrap(),
                once,
                "normalization must be idempotent"
            );
        }
    }
}

#[test]
fn deepseek_required_allowed_tools_obeys_the_same_thinking_guard() {
    for enabled in [true, false] {
        let req = request(json!({
            "thinking":{"type": if enabled {"enabled"} else {"disabled"}},
            "tools":[{"type":"function","function":{"name":"weather"}}],
            "tool_choice":{"type":"allowed_tools","mode":"required","tools":[{"type":"function","name":"weather"}]}
        }));
        assert_eq!(req.validate().is_ok(), !enabled);
    }
}

#[test]
fn deepseek_rejects_ambiguous_template_thinking_overrides() {
    for kwargs in [
        json!({"thinking":"bad","enable_thinking":false}),
        json!({"enable_thinking":0}),
        json!({"thinking":true,"enable_thinking":false}),
    ] {
        let req = request(
            json!({"model":"deepseek-v4-pro","reasoning_effort":"high","chat_template_kwargs":kwargs}),
        );
        assert!(req.validate().is_err());
    }
    let req = request(
        json!({"thinking":{"type":"disabled"},"chat_template_kwargs":{"thinking":null,"enable_thinking":null,"reasoning_effort":42}}),
    );
    assert!(req.validate().is_ok());
    assert_eq!(
        req.chat_template_kwargs.as_ref().unwrap()["thinking"],
        false
    );
}
