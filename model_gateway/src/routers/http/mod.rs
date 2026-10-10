//! HTTP router implementations

pub mod pd_router;
pub mod pd_types;
pub(crate) mod request_body;
pub(crate) mod request_stream;
pub mod router;

use axum::response::Response;
use openai_protocol::messages::CreateMessageRequest;
use serde_json::Value;

use crate::routers::error;

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

/// The relay routers' answer to `max_tokens: 0`. The relay forwards a Messages
/// request to the worker as sent and renders no prompt, so it cannot give the
/// shape the public API's answer (an empty message carrying the prompt's token
/// count, which the gateway gives where it renders the prompt itself); it
/// keeps the 400 the request validation gave before, naming the field, rather
/// than handing the engine a request that generates nothing.
pub(crate) fn refuse_max_tokens_zero(body: &CreateMessageRequest) -> Option<Response> {
    (body.max_tokens == 0).then(|| {
        error::bad_request(
            "max_tokens_zero_not_relayed",
            "max_tokens: must be greater than 0 for this model, which is served through \
             the HTTP relay: the relay forwards the request as sent and does not count \
             the prompt",
        )
    })
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
