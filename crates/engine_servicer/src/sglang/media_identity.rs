//! Early media identity, before either scheduler waits for its KV peer.
use smg_grpc_client::{sglang_media_identity as contract, sglang_proto as sg};

use super::State;

/// Advertise reuse only when both metadata validation and compatible CPU hashing are enabled.
pub(super) fn key(state: &State) -> Option<&str> {
    (state.media_identity_enabled && state.mm_hasher.is_some())
        .then_some(state.media_identity_key.as_deref())
        .flatten()
}

/// Read the resolved engine role; missing or malformed configuration cannot enable reuse.
pub(super) fn mode(state: &State, expected: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(&state.model.server_args_json)
        .ok()
        .and_then(|v| {
            v.get("disaggregation_mode")
                .and_then(|s| s.as_str())
                .map(str::to_owned)
        })
        .as_deref()
        == Some(expected)
}

/// Consume an optional identity only after validation, retaining original inputs on failure.
pub(super) fn apply(state: &State, req: &mut sg::GenerateRequest) -> bool {
    let identity = req.media_identity.take();
    let Some(identity) = identity else {
        return false;
    };
    let valid = mode(state, "decode")
        && key(state).is_some_and(|key| contract::validate(&identity, req, key));
    if valid {
        if let Some(tokens) = &mut req.tokenized {
            tokens.input_ids = identity.input_ids;
        }
        req.mm_inputs = identity.mm_inputs;
        tracing::debug!(target: "smg_request_timing", request_id = %req.request_id, "sglang_media_identity_reused");
        state.shared.metrics.media_identity("reused");
    } else {
        state.shared.metrics.media_identity("fallback");
        tracing::debug!(target: "smg_request_timing", request_id = %req.request_id, "sglang_media_identity_fallback");
    }
    valid
}
