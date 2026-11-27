use serde_json::json;

use super::*;

/// Match the rendered tri-state mode (the typed toggle, else the profile's
/// reading of `reasoning_effort`) and leave other template switches untouched.
#[test]
fn minimax_thinking_mode_matches_template_precedence() {
    for (mut body, expected) in [
        (json!({}), None),
        (json!({"thinking": {"type": "adaptive"}}), None),
        (
            json!({"thinking": {"type": "enabled"}}),
            Some(json!("enabled")),
        ),
        (
            json!({"thinking": {"type": "disabled"}}),
            Some(json!("disabled")),
        ),
        (
            json!({"thinking": {"type": "enabled"}, "chat_template_kwargs": {"thinking_mode": "adaptive"}}),
            Some(json!("adaptive")),
        ),
        (
            json!({"thinking": {"type": "disabled"}, "chat_template_kwargs": {"thinking_mode": "enabled"}}),
            Some(json!("enabled")),
        ),
        (
            json!({"chat_template_kwargs": {"thinking_mode": null}}),
            Some(json!(null)),
        ),
        // The profile's reading of the effort: `none` is thinking off under
        // MiniMax; `minimal` is its lowest level, thinking on, so the
        // template's default mode stands.
        (json!({"reasoning_effort": "none"}), Some(json!("disabled"))),
        (json!({"reasoning_effort": "minimal"}), None),
        (json!({"reasoning_effort": "high"}), None),
    ] {
        body["model"] = json!("minimax-m3");
        body["messages"] = json!([]);
        let request = RequestType::Chat(Arc::new(serde_json::from_value(body).unwrap()));
        let actual = thinking_mode_kwargs(
            &request,
            ProviderProfile::Minimax,
            Some(ThinkingKeyName::ThinkingMode),
        )
        .map(|s| serde_json::from_str::<serde_json::Value>(&s).unwrap());
        assert_eq!(actual, expected.map(|mode| json!({"thinking_mode": mode})));
        for key in [
            None,
            Some(ThinkingKeyName::EnableThinking),
            Some(ThinkingKeyName::Thinking),
            Some(ThinkingKeyName::ReasoningEffort),
        ] {
            assert_eq!(
                thinking_mode_kwargs(&request, ProviderProfile::Minimax, key),
                None
            );
        }
    }
}
