use serde_json::Value;
use thiserror::Error;

use crate::worker::Endpoint;

pub(crate) const SGLANG_FIELDS: &[&str] = &[
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

pub(crate) fn strip_sglang_fields(payload: &mut Value) {
    if let Some(obj) = payload.as_object_mut() {
        for field in SGLANG_FIELDS {
            obj.remove(*field);
        }
    }
}

pub(crate) fn strip_default_sglang_fields(payload: &mut Value) {
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
pub(crate) fn is_stripped_sglang_default(field: &str, raw_json: &str) -> bool {
    matches!(raw_json, "null" | "false")
        || (matches!(field, "separate_reasoning" | "stream_reasoning") && raw_json == "true")
}

#[derive(Error, Debug)]
pub enum ProviderError {
    #[error("Unsupported endpoint: {0:?}")]
    UnsupportedEndpoint(Endpoint),

    #[error("Transform error: {0}")]
    TransformError(String),
}
