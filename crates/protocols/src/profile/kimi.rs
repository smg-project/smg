//! Kimi/Moonshot contract rules (Kimi-Vendor-Verifier).
//!
//! The sampling rules are K3's alone; other Kimi and Moonshot models keep
//! OpenAI's ranges. Only requests entering through `ValidatedJson` reach
//! these rules; the Responses conversion builds its chat request without them.

use crate::{
    chat::{ChatCompletionRequest, ChatMessage},
    ext::kimi::DeclaredTools,
};

/// Sampling defaults the K3 serving requirements fix to the official values;
/// applied when the client omits the field so a self-hosted engine does not
/// substitute its own. `max_tokens` is left to the engine: the manual's
/// documented default is inconsistent (262144 vs 32768).
const DEFAULT_TEMPERATURE: f32 = TEMPERATURES[2];
const DEFAULT_TOP_P: f32 = TOP_P;

/// Sampling values the verifier requires accepted (KVV tests/params
/// IMMUTABLE_PARAMS, thinking and non-thinking sets combined); anything else
/// must be rejected. The official API pins temperature to 1.0 alone, but the
/// verifier asserts 0.0 and 0.6 accepted too.
const TEMPERATURES: [f32; 3] = [0.0, 0.6, 1.0];
const TOP_P: f32 = 0.95;

pub(super) fn normalize_chat(req: &mut ChatCompletionRequest) {
    if !is_k3(&req.model) {
        return;
    }
    req.temperature.get_or_insert(DEFAULT_TEMPERATURE);
    req.top_p.get_or_insert(DEFAULT_TOP_P);
    req.presence_penalty.get_or_insert(0.0);
    req.frequency_penalty.get_or_insert(0.0);
    req.n.get_or_insert(1);
}

/// K3 dynamic tools may be declared on system messages, and on developer
/// messages, which the OpenAI spec defines as the successor of `system` and
/// this crate reads the same way; a declaration that is not a list of tools is
/// rejected. A `tools` field set on a user or assistant message is rejected,
/// an empty list included, while null counts as absent: the contract keys on
/// the field being set, not on its contents (KVV test_dynamic_tools; the
/// verifier has no developer case, so that role follows `system`). Tool and
/// function messages capture no such key, so serde drops it there as it
/// always did.
pub(super) fn validate_chat(req: &ChatCompletionRequest) -> Result<(), validator::ValidationError> {
    if is_k3(&req.model) {
        validate_sampling(req)?;
    }
    for msg in &req.messages {
        let (code, role, message) = match msg {
            ChatMessage::User { ext, .. } if ext.tools.is_some() => {
                ("tools_role_restricted", "user", "is not allowed")
            }
            ChatMessage::Assistant { ext, .. } if ext.tools.is_some() => {
                ("tools_role_restricted", "assistant", "is not allowed")
            }
            ChatMessage::System { ext, .. } if is_malformed(ext.tools.as_ref()) => (
                "tools_malformed",
                "system",
                "must be a list of tool declarations",
            ),
            ChatMessage::Developer { ext, .. } if is_malformed(ext.tools.as_ref()) => (
                "tools_malformed",
                "developer",
                "must be a list of tool declarations",
            ),
            _ => continue,
        };
        let mut e = validator::ValidationError::new(code);
        e.message = Some(format!("'tools' on a message with role '{role}' {message}").into());
        return Err(e);
    }
    Ok(())
}

fn is_malformed(tools: Option<&DeclaredTools>) -> bool {
    matches!(tools, Some(DeclaredTools::Malformed(_)))
}

/// Whether a model id names Kimi K3, the only Kimi model with pinned sampling.
fn is_k3(model: &str) -> bool {
    model.split('/').any(|segment| {
        super::starts_with_ignore_ascii_case(segment, "kimi-k3")
            || super::starts_with_ignore_ascii_case(segment, "kimi_k3")
    })
}

/// Immutable sampling parameters: the contract fixes them, so any other value
/// is a 400 rather than a silently different sampling regime.
fn validate_sampling(req: &ChatCompletionRequest) -> Result<(), validator::ValidationError> {
    if let Some(t) = req.temperature {
        if !TEMPERATURES.iter().any(|allowed| approx_eq(t, *allowed)) {
            return Err(pinned(
                "temperature_not_allowed",
                "temperature",
                "0, 0.6 or 1",
            ));
        }
    }
    if req.top_p.is_some_and(|p| !approx_eq(p, TOP_P)) {
        return Err(pinned("top_p_not_allowed", "top_p", "0.95"));
    }
    if req.presence_penalty.is_some_and(|p| !approx_eq(p, 0.0)) {
        return Err(pinned(
            "presence_penalty_not_allowed",
            "presence_penalty",
            "0",
        ));
    }
    if req.frequency_penalty.is_some_and(|p| !approx_eq(p, 0.0)) {
        return Err(pinned(
            "frequency_penalty_not_allowed",
            "frequency_penalty",
            "0",
        ));
    }
    if req.n.is_some_and(|n| n != 1) {
        return Err(pinned("n_not_allowed", "n", "1"));
    }
    Ok(())
}

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-6
}

fn pinned(code: &'static str, field: &str, allowed: &str) -> validator::ValidationError {
    let mut e = validator::ValidationError::new(code);
    e.message = Some(format!("invalid {field}: only {allowed} is allowed for this model").into());
    e
}
