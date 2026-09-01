//! Per-provider protocol profiles.
//!
//! A profile owns the request rules a provider's vendor-acceptance contract
//! enforces beyond (or instead of) the OpenAI baseline. Profiles are selected
//! from the request's model id and applied during request validation, so every
//! entry point using `ValidatedJson` gets them for free.
//!
//! Precedence for what a profile encodes: provider verifier > vendor manual >
//! live API behavior.

mod kimi;
mod minimax;

use crate::chat::ChatCompletionRequest;

/// Provider dialect for a request, selected from the model id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderProfile {
    /// OpenAI baseline: no extra rules beyond core validation.
    OpenAi,
    /// Kimi/Moonshot contract (Kimi-Vendor-Verifier).
    Kimi,
    /// MiniMax contract (MiniMax-Provider-Verifier).
    Minimax,
}

impl ProviderProfile {
    pub fn for_model(model: &str) -> Self {
        let m = model.to_ascii_lowercase();
        if m.starts_with("kimi") || m.starts_with("moonshot") {
            ProviderProfile::Kimi
        } else if m.starts_with("minimax") || m.starts_with("abab") {
            ProviderProfile::Minimax
        } else {
            ProviderProfile::OpenAi
        }
    }

    /// Contract rules applied on top of core validation.
    pub fn validate_chat(
        self,
        req: &ChatCompletionRequest,
    ) -> Result<(), validator::ValidationError> {
        match self {
            ProviderProfile::Kimi => {
                reject_root(req)?;
                kimi::validate_chat(req)
            }
            ProviderProfile::Minimax => minimax::validate_chat(req),
            ProviderProfile::OpenAi => reject_root(req),
        }
    }

    /// Profile-specific request rewrites, applied before validation.
    pub fn normalize_chat(self, req: &mut ChatCompletionRequest) {
        if self == ProviderProfile::Minimax {
            minimax::normalize_chat(req);
        }
    }
}

/// The `root` role is a MiniMax-only extension; other dialects reject it the
/// way their reference APIs do.
fn reject_root(req: &ChatCompletionRequest) -> Result<(), validator::ValidationError> {
    if req
        .messages
        .iter()
        .any(|m| matches!(m, crate::chat::ChatMessage::Root { .. }))
    {
        let mut e = validator::ValidationError::new("invalid_role");
        e.message = Some("invalid role: root".into());
        return Err(e);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_id_selects_profile() {
        assert_eq!(ProviderProfile::for_model("kimi-k3"), ProviderProfile::Kimi);
        assert_eq!(
            ProviderProfile::for_model("Kimi-K2.6"),
            ProviderProfile::Kimi
        );
        assert_eq!(
            ProviderProfile::for_model("MiniMax-M3"),
            ProviderProfile::Minimax
        );
        assert_eq!(
            ProviderProfile::for_model("gpt-4o-mini"),
            ProviderProfile::OpenAi
        );
        assert_eq!(ProviderProfile::for_model(""), ProviderProfile::OpenAi);
    }
}
