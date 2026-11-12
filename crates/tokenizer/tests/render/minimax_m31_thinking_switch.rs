//! The thinking switch of the two MiniMax M3-family templates, from the
//! fixtures: the M3 template branches on `thinking_mode` (the tri-state
//! switch, off by default); the M3.1-Flash template has no branch and prints
//! the word `thinking` from its system prose only ("When thinking is
//! enabled, wrap your reasoning in ..."), so it has no switch, and a prompt
//! it renders ends outside a thought.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test code: panics are intentional"
)]

use llm_tokenizer::chat_template::{
    detect_thinking_toggle, ChatTemplateState, ThinkingKeyName, ThinkingToggle,
};

const M3_TEMPLATE: &str = include_str!("../fixtures/minimax_m3/chat_template.jinja");
const M31_TEMPLATE: &str = include_str!("../fixtures/minimax_m31/chat_template.jinja");

#[test]
fn the_m3_template_has_the_tristate_switch_and_the_m3_1_flash_template_none() {
    assert_eq!(
        detect_thinking_toggle(M3_TEMPLATE),
        (
            ThinkingToggle::DefaultOff,
            Some(ThinkingKeyName::ThinkingMode)
        )
    );
    assert_eq!(
        detect_thinking_toggle(M31_TEMPLATE),
        (ThinkingToggle::None, None)
    );
    assert!(
        M31_TEMPLATE.contains("When thinking is enabled"),
        "the prose the detector must not read as a variable"
    );

    let m31 = ChatTemplateState::new(Some(M31_TEMPLATE.to_string())).expect("the M3.1 template");
    assert_eq!(m31.thinking_toggle(), ThinkingToggle::None);
    assert_eq!(m31.thinking_key_name(), None);
    assert!(!m31.prefill_opens_think_block());

    let m3 = ChatTemplateState::new(Some(M3_TEMPLATE.to_string())).expect("the M3 template");
    assert_eq!(m3.thinking_toggle(), ThinkingToggle::DefaultOff);
    assert_eq!(m3.thinking_key_name(), Some(ThinkingKeyName::ThinkingMode));
}
