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

use crate::{
    chat::{ChatCompletionRequest, ChatMessage},
    ext::retain_if,
};

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

    /// Shape the request for dispatch under this profile: every message drops
    /// the extensions that belong to another provider, so a foreign field
    /// never reaches a backend or a chat template. Runs before validation
    /// and template rendering on every entry point.
    pub fn normalize_chat(self, req: &mut ChatCompletionRequest) {
        for message in &mut req.messages {
            match message {
                ChatMessage::System { ext, .. } => retain_if(ext, self),
                ChatMessage::User { ext, .. } => retain_if(ext, self),
                ChatMessage::Assistant { ext, .. } => retain_if(ext, self),
                _ => {}
            }
        }
    }

    /// Contract rules applied on top of core validation.
    pub fn validate_chat(
        self,
        req: &ChatCompletionRequest,
    ) -> Result<(), validator::ValidationError> {
        match self {
            ProviderProfile::Kimi => kimi::validate_chat(req),
            ProviderProfile::OpenAi | ProviderProfile::Minimax => Ok(()),
        }
    }
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
