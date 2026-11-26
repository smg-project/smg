//! SGLang-specific request fields, and the gateway's own `rid`, that the
//! gateway strips before forwarding a request to a backend that does not
//! understand them.
//!
//! Shared by the HTTP proxy, which drops a field only when it carries the
//! value the engine applies without it, and by the external-provider
//! transformers, which drop the fields unconditionally because no
//! third-party API accepts them.

use serde_json::Value;

pub const SGLANG_FIELDS: &[&str] = &[
    "request_id",
    // the gateway's own request id: an engine reads it for log correlation
    // (a string is no default, so the proxy keeps it); a third-party API
    // refuses it as an unknown argument
    "rid",
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
    // the chat template's generation-header switch; no third-party API has it
    "add_generation_prompt",
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
        "separate_reasoning"
        | "stream_reasoning"
        | "skip_special_tokens"
        | "add_generation_prompt" => Some(true),
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
/// the engine default for it (`sglang_default`). Everything else goes out
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

/// Whether the strip could change `compact_json`, an object in serde_json's
/// compact rendering: true when a listed name is followed by a literal that
/// [`is_stripped_sglang_default`] drops (a literal is known by its first
/// byte). A nested or quoted match errs towards true, never towards false.
pub fn may_carry_stripped_sglang_default(compact_json: &[u8]) -> bool {
    memchr::memchr_iter(b':', compact_json).any(|colon| {
        let raw = match compact_json.get(colon + 1) {
            Some(b'n') => "null",
            Some(b't') => "true",
            Some(b'f') => "false",
            _ => return false,
        };
        compact_json[..colon]
            .strip_suffix(b"\"")
            .is_some_and(|key| {
                let name = &key[memchr::memrchr(b'"', key).map_or(0, |quote| quote + 1)..];
                SGLANG_FIELDS
                    .iter()
                    .any(|field| field.as_bytes() == name && is_stripped_sglang_default(field, raw))
            })
    })
}

#[cfg(test)]
#[path = "../tests/unit/sglang_fields/tests.rs"]
mod tests;
