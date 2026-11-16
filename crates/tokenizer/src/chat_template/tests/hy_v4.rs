//! The HY-4 template: its reasoning effort controls the prefill, and an explicit kwarg wins.

use super::*;
#[test]
fn hy4_effort_controls_prefill_and_explicit_kwarg_wins() {
    let template = "{% set think_begin_token = '<think:6124c78e>' %}{% if reasoning_effort is not defined %}{% set reasoning_effort = 'high' %}{% endif %}{{ think_begin_token }}{% if reasoning_effort == 'no_think' %}</think:6124c78e>{% endif %}";
    let state = ChatTemplateState::new(Some(template.to_string())).unwrap();
    assert_eq!(state.thinking_toggle(), ThinkingToggle::DefaultOn);
    assert_eq!(
        state.thinking_key_name(),
        Some(ThinkingKeyName::ReasoningEffort)
    );
    assert_eq!(state.native_reasoning_effort_values(), &["high"]);
    assert_eq!(state.native_reasoning_effort_off_values(), &["no_think"]);
    for (thinking, expected) in [
        (None, "<think:6124c78e>"),
        (Some(true), "<think:6124c78e>"),
        (Some(false), "<think:6124c78e></think:6124c78e>"),
    ] {
        assert_eq!(
            state
                .apply(
                    &[],
                    ChatTemplateParams {
                        thinking,
                        ..Default::default()
                    }
                )
                .unwrap(),
            expected
        );
    }
    let kwargs = HashMap::from([(
        "reasoning_effort".to_string(),
        serde_json::json!("no_think"),
    )]);
    assert_eq!(
        state
            .apply(
                &[],
                ChatTemplateParams {
                    thinking: Some(true),
                    template_kwargs: Some(&kwargs),
                    ..Default::default()
                }
            )
            .unwrap(),
        "<think:6124c78e></think:6124c78e>"
    );
}
