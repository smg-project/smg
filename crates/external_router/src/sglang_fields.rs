//! SGLang-specific request fields that the gateway strips before forwarding
//! a request to a backend that does not understand them.
//!
//! Shared by the HTTP proxy, which drops a field only when it carries its
//! default value, and by the external-provider transformers, which drop the
//! fields unconditionally because no third-party API accepts them.

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

pub fn strip_sglang_fields(payload: &mut Value) {
    if let Some(obj) = payload.as_object_mut() {
        for field in SGLANG_FIELDS {
            obj.remove(*field);
        }
    }
}

pub fn strip_default_sglang_fields(payload: &mut Value) {
    if let Some(obj) = payload.as_object_mut() {
        for field in SGLANG_FIELDS {
            if obj.get(*field).is_some_and(|value| {
                value.is_null()
                    || value == false
                    || (matches!(*field, "separate_reasoning" | "stream_reasoning")
                        && value == true)
            }) {
                obj.remove(*field);
            }
        }
    }
}

/// Raw-slice twin of [`strip_default_sglang_fields`]: decides whether a
/// [`SGLANG_FIELDS`] entry would be stripped, given the compact serde_json
/// rendering of its value. Must mirror the `Value` version above.
pub fn is_stripped_sglang_default(field: &str, raw_json: &str) -> bool {
    matches!(raw_json, "null" | "false")
        || (matches!(field, "separate_reasoning" | "stream_reasoning") && raw_json == "true")
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn strip_default_sglang_fields_removes_false_and_null_values() {
        let mut payload = json!({
            "continue_final_message": false,
            "messages": [],
            "model": "test-model",
            "no_stop_trim": true,
            "return_hidden_states": null,
            "separate_reasoning": true,
            "stream_reasoning": true
        });

        strip_default_sglang_fields(&mut payload);

        assert_eq!(payload.get("continue_final_message"), None);
        assert_eq!(payload.get("return_hidden_states"), None);
        assert_eq!(payload.get("separate_reasoning"), None);
        assert_eq!(payload.get("stream_reasoning"), None);
        assert_eq!(payload.get("no_stop_trim"), Some(&json!(true)));
        assert_eq!(payload.get("model"), Some(&json!("test-model")));
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
