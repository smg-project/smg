use super::*;
#[test]
fn hy4_effort_arms_parser_like_template() {
    let tok = llm_tokenizer::MockTokenizer::new()
        .with_thinking_toggle(ThinkingToggle::DefaultOn)
        .with_thinking_key_name(ThinkingKeyName::ReasoningEffort)
        .with_native_reasoning_effort_values(&["high"])
        .with_native_reasoning_effort_off_values(&["no_think"]);
    assert!(should_mark_reasoning_started(None, &tok));
    assert_eq!(
        extract_template_effort_thinking(None, Some("no_think"), &tok),
        Some(false)
    );
    let kwargs = std::collections::HashMap::from([(
        "reasoning_effort".to_string(),
        serde_json::json!("high"),
    )]);
    assert_eq!(
        extract_thinking_from_kwargs(Some(&kwargs), None, &tok),
        Some(true)
    );
    assert_eq!(
        extract_template_effort_thinking(Some(&kwargs), None, &tok),
        Some(true)
    );
    // The request-level off word is written over the kwargs entry.
    assert_eq!(
        extract_template_effort_thinking(Some(&kwargs), Some("no_think"), &tok),
        Some(false)
    );
    // The parser follows the value the template sees: the request-level
    // off word over the kwargs `high` disarms it, as the prompt renders
    // thinking off; the kwargs entry alone arms it.
    assert_eq!(
        extract_thinking_from_kwargs(Some(&kwargs), Some("no_think"), &tok),
        Some(false)
    );
    assert_eq!(
        resolve_user_thinking(
            Some(&kwargs),
            Some("no_think"),
            None,
            &tok,
            ProviderProfile::Generic
        ),
        Some(false)
    );
    assert_eq!(
        resolve_user_thinking(Some(&kwargs), None, None, &tok, ProviderProfile::Generic),
        Some(true)
    );
    // The template renders its default for a value it does not know, so
    // the protocol's `none` must not disarm the parser here.
    assert_eq!(
        resolve_user_thinking(None, Some("none"), None, &tok, ProviderProfile::Generic),
        Some(true)
    );
    let unknown = std::collections::HashMap::from([(
        "reasoning_effort".to_string(),
        serde_json::json!("none"),
    )]);
    assert_eq!(
        resolve_user_thinking(Some(&unknown), None, None, &tok, ProviderProfile::Generic),
        Some(true)
    );
}
