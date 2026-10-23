//! SGLang-specific request fields that the gateway strips before forwarding
//! a request to a backend that does not understand them.
//!
//! Shared by the HTTP proxy, which drops a field only when it carries the
//! value the engine applies without it, and by the external-provider
//! transformers, which drop the fields unconditionally because no
//! third-party API accepts them.

use serde_json::Value;

pub const SGLANG_FIELDS: &[&str] = &[
    "request_id",
    "priority",
    "top_k",
    "min_p",
    "min_tokens",
    "regex",
    "ebnf",
    "json_schema",
    "stop_token_ids",
    "no_stop_trim",
    "ignore_eos",
    "continue_final_message",
    "skip_special_tokens",
    "lora_path",
    "session_params",
    "separate_reasoning",
    "stream_reasoning",
    "chat_template",
    "chat_template_kwargs",
    "return_hidden_states",
    "repetition_penalty",
    "sampling_seed",
    "backend_url",
];

/// What SGLang applies when a boolean field is absent (its
/// `ChatCompletionRequest` defaults, which the typed requests mirror).
///
/// A field carrying exactly this value means the same thing as no field at
/// all, so the proxy may drop it and spare a backend that does not know the
/// field. Any other value is the client's explicit choice: three of these
/// default to `true`, so an explicit `false` is the one that must reach the
/// worker, which would otherwise silently apply its default.
fn sglang_default(field: &str) -> Option<bool> {
    match field {
        "separate_reasoning" | "stream_reasoning" | "skip_special_tokens" => Some(true),
        "no_stop_trim" | "ignore_eos" | "continue_final_message" | "return_hidden_states" => {
            Some(false)
        }
        _ => None,
    }
}

pub fn strip_sglang_fields(payload: &mut Value) {
    if let Some(obj) = payload.as_object_mut() {
        for field in SGLANG_FIELDS {
            obj.remove(*field);
        }
    }
}

/// Drop every [`SGLANG_FIELDS`] entry that is `null` or a boolean equal to
/// the engine default for it ([`sglang_default`]). Everything else goes out
/// as the client sent it.
pub fn strip_default_sglang_fields(payload: &mut Value) {
    if let Some(obj) = payload.as_object_mut() {
        for field in SGLANG_FIELDS {
            if obj
                .get(*field)
                .is_some_and(|value| is_sglang_default(field, value))
            {
                obj.remove(*field);
            }
        }
    }
}

fn is_sglang_default(field: &str, value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Bool(flag) => sglang_default(field) == Some(*flag),
        _ => false,
    }
}

/// Raw-slice twin of [`strip_default_sglang_fields`]: decides whether a
/// [`SGLANG_FIELDS`] entry would be stripped, given the compact serde_json
/// rendering of its value. Must mirror the `Value` version above.
pub fn is_stripped_sglang_default(field: &str, raw_json: &str) -> bool {
    match raw_json {
        "null" => true,
        "true" => sglang_default(field) == Some(true),
        "false" => sglang_default(field) == Some(false),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn strip_default_sglang_fields_removes_null_and_engine_default_values() {
        let mut payload = json!({
            "continue_final_message": false,
            "messages": [],
            "model": "test-model",
            "no_stop_trim": true,
            "return_hidden_states": null,
            "separate_reasoning": true,
            "skip_special_tokens": true,
            "stream_reasoning": true,
            "top_k": null
        });

        strip_default_sglang_fields(&mut payload);

        assert_eq!(payload.get("continue_final_message"), None);
        assert_eq!(payload.get("return_hidden_states"), None);
        assert_eq!(payload.get("separate_reasoning"), None);
        assert_eq!(payload.get("skip_special_tokens"), None);
        assert_eq!(payload.get("stream_reasoning"), None);
        assert_eq!(payload.get("top_k"), None);
        assert_eq!(payload.get("no_stop_trim"), Some(&json!(true)));
        assert_eq!(payload.get("model"), Some(&json!("test-model")));
    }

    /// SGLang defaults these three to `true`, so `false` is the value a
    /// client sends on purpose; stripping it made the worker apply the
    /// opposite of what was asked (inline reasoning came back separated).
    #[test]
    fn strip_default_sglang_fields_keeps_explicit_false_on_true_default_fields() {
        let mut payload = json!({
            "model": "test-model",
            "separate_reasoning": false,
            "skip_special_tokens": false,
            "stream_reasoning": false
        });

        strip_default_sglang_fields(&mut payload);

        assert_eq!(payload.get("separate_reasoning"), Some(&json!(false)));
        assert_eq!(payload.get("skip_special_tokens"), Some(&json!(false)));
        assert_eq!(payload.get("stream_reasoning"), Some(&json!(false)));
    }

    #[test]
    fn strip_default_sglang_fields_keeps_explicit_true_on_false_default_fields() {
        let mut payload = json!({
            "continue_final_message": true,
            "ignore_eos": true,
            "no_stop_trim": true,
            "return_hidden_states": true
        });

        strip_default_sglang_fields(&mut payload);

        for field in [
            "continue_final_message",
            "ignore_eos",
            "no_stop_trim",
            "return_hidden_states",
        ] {
            assert_eq!(payload.get(field), Some(&json!(true)), "{field}");
        }
    }

    /// Fields without a boolean default are stripped only when `null`; a
    /// concrete value, even one equal to the engine default, is forwarded.
    #[test]
    fn strip_default_sglang_fields_keeps_concrete_non_boolean_values() {
        let mut payload = json!({
            "lora_path": null,
            "min_p": 0.0,
            "priority": 0,
            "top_k": -1
        });

        strip_default_sglang_fields(&mut payload);

        assert_eq!(payload.get("lora_path"), None);
        assert_eq!(payload.get("min_p"), Some(&json!(0.0)));
        assert_eq!(payload.get("priority"), Some(&json!(0)));
        assert_eq!(payload.get("top_k"), Some(&json!(-1)));
    }

    #[test]
    fn raw_predicate_agrees_with_value_strip_for_every_field() {
        for field in SGLANG_FIELDS {
            for raw in ["null", "false", "true", "0", "1.5", "\"false\"", "[false]"] {
                let value: Value = serde_json::from_str(raw).expect("literal parses");
                let mut fields = serde_json::Map::new();
                fields.insert((*field).to_string(), value);
                let mut payload = Value::Object(fields);
                strip_default_sglang_fields(&mut payload);

                let value_stripped = payload.get(*field).is_none();
                assert_eq!(
                    is_stripped_sglang_default(field, raw),
                    value_stripped,
                    "field={field} raw={raw}"
                );
            }
        }
    }
}
