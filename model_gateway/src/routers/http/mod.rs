//! HTTP router implementations

pub(crate) mod deepseek_compat;
pub mod pd_router;
pub mod pd_types;
pub mod router;

use serde_json::Value;

/// Rewrite the `model` field of an outbound request body.
///
/// A worker is registered under its canonical model ID, so a request that
/// arrived under an alias must reach the backend under the canonical name —
/// the backend has never heard of the alias. Both HTTP routers resolve the
/// alias for their own routing decisions; this puts the same answer into the
/// body they forward.
///
/// Leaves a body without a `model` field alone. Inserting the key would change
/// the request the client wrote.
pub(crate) fn set_request_model(json: &mut Value, canonical_model_id: &str) {
    if let Some(model) = json.get_mut("model") {
        *model = Value::String(canonical_model_id.to_owned());
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn set_request_model_replaces_an_existing_model_field() {
        let mut body = json!({"model": "GLM-5.2-Coding", "stream": false});
        set_request_model(&mut body, "GLM-5.2");
        assert_eq!(body, json!({"model": "GLM-5.2", "stream": false}));
    }

    #[test]
    fn set_request_model_leaves_a_body_without_the_field_untouched() {
        let mut body = json!({"text": "Hello"});
        set_request_model(&mut body, "GLM-5.2");
        assert_eq!(body, json!({"text": "Hello"}));
    }
}
