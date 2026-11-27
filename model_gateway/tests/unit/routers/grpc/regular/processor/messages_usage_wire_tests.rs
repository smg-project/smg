use super::messages_usage_from_counts;

/// The contract is about the serialized JSON, not the Rust struct: cache
/// counters must be present as integer zeros (a struct-level Some(0) that
/// a serde attr silently skipped would still fail here).
#[test]
fn cache_counters_serialize_as_integer_zeros() {
    let v = serde_json::to_value(messages_usage_from_counts(25, 150, 0, None)).unwrap();
    assert_eq!(v["input_tokens"], 25);
    assert_eq!(v["output_tokens"], 150);
    assert_eq!(v["cache_creation_input_tokens"], 0);
    assert_eq!(v["cache_read_input_tokens"], 0);
}

/// The engine's whole prompt and its hit become the wire's split: the
/// input outside the cache beside the hit as the cache-read counter, their
/// sum the prompt; a miss leaves the whole prompt as input beside an
/// integer zero; a hit an engine reports above its prompt count leaves no
/// negative input; the creation counter stays zero throughout.
#[test]
fn cache_read_input_tokens_is_the_engines_prefix_cache_hit() {
    let v = serde_json::to_value(messages_usage_from_counts(945, 12, 896, None)).unwrap();
    assert_eq!(v["input_tokens"], 49);
    assert_eq!(v["cache_read_input_tokens"], 896);
    assert_eq!(v["cache_creation_input_tokens"], 0);
    let v = serde_json::to_value(messages_usage_from_counts(945, 12, 0, None)).unwrap();
    assert_eq!(v["input_tokens"], 945);
    assert_eq!(v["cache_read_input_tokens"], 0);
    let v = serde_json::to_value(messages_usage_from_counts(10, 1, 16, None)).unwrap();
    assert_eq!(v["input_tokens"], 0);
    assert_eq!(v["cache_read_input_tokens"], 16);
}

/// The public API writes these keys on every message; a typed client
/// reads a missing `cache_creation` as a decode error, not as zero.
#[test]
fn always_present_usage_keys_are_written_without_thinking() {
    let v = serde_json::to_value(messages_usage_from_counts(25, 150, 0, None)).unwrap();
    assert_eq!(
        v["cache_creation"],
        serde_json::json!({"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0})
    );
    assert_eq!(v["service_tier"], "standard");
    assert_eq!(v["inference_geo"], "not_available");
    let keys: Vec<_> = v.as_object().unwrap().keys().cloned().collect();
    assert!(
        !keys.contains(&"output_tokens_details".to_string()),
        "{keys:?}"
    );
    assert!(!keys.contains(&"server_tool_use".to_string()), "{keys:?}");
}

/// A turn that ran with thinking reports its thinking tokens, zero
/// included: the key says thinking was on, the count says how much.
#[test]
fn thinking_tokens_are_written_when_the_turn_ran_with_thinking() {
    let v = serde_json::to_value(messages_usage_from_counts(25, 150, 0, Some(40))).unwrap();
    assert_eq!(v["output_tokens_details"]["thinking_tokens"], 40);
    let v = serde_json::to_value(messages_usage_from_counts(25, 150, 0, Some(0))).unwrap();
    assert_eq!(v["output_tokens_details"]["thinking_tokens"], 0);
}
