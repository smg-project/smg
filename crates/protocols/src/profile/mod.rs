//! Per-provider protocol profiles.
//!
//! A profile owns the request rules a provider's vendor-acceptance contract
//! enforces beyond (or instead of) the OpenAI baseline. Profiles are selected
//! from the request's model id and applied during request validation, so every
//! entry point using `ValidatedJson` gets them for free.
//!
//! Precedence for what a profile encodes: provider verifier > vendor manual >
//! live API behavior.
//!
//! A profile also shapes the request before validation: message-level
//! extension structs that belong to another provider are dropped (see
//! [`crate::ext::ProviderExt`]). Provider fields typed directly onto content
//! parts, such as MiniMax's `max_long_side_pixel` and `fps`, are not covered
//! by that pass and are forwarded as sent.

mod kimi;
mod minimax;

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
    /// Select the profile from a model id.
    ///
    /// Matches the way the tool and reasoning parser factories do: any
    /// `/`-separated segment that starts with a vendor marker selects the
    /// profile, so `kimi-k3`, `/models/Kimi-K3`, `moonshotai/kimi-k2` and
    /// `openrouter/moonshotai/kimi-k2` all resolve to Kimi. Aliases are not
    /// visible here, because normalization runs before alias resolution: an
    /// aliased vendor model falls back to the OpenAI baseline, and any
    /// extension it carried is dropped with a warning.
    pub fn for_model(model: &str) -> Self {
        for segment in model.split('/') {
            if starts_with_ignore_ascii_case(segment, "kimi")
                || starts_with_ignore_ascii_case(segment, "moonshot")
            {
                return ProviderProfile::Kimi;
            }
            if starts_with_ignore_ascii_case(segment, "minimax")
                || starts_with_ignore_ascii_case(segment, "abab")
            {
                return ProviderProfile::Minimax;
            }
        }
        ProviderProfile::OpenAi
    }

    /// Shape the request for dispatch under this profile: the provider's own
    /// normalization first (MiniMax folds a root message into a leading
    /// system message), then every message drops the extension struct that
    /// belongs to another provider, so a foreign field never reaches a
    /// backend or a chat template. Runs before validation and template
    /// rendering on every entry point. Only message-level extension structs
    /// are covered; see the module docs. Dropped extensions are logged once
    /// per request.
    pub fn normalize_chat(self, req: &mut ChatCompletionRequest) {
        if self == ProviderProfile::Minimax {
            minimax::normalize_chat(req);
        }
        let mut dropped: Vec<&'static str> = Vec::new();
        for message in &mut req.messages {
            let role = match message {
                ChatMessage::System { ext, .. } => retain_if(ext, self).then_some("system"),
                ChatMessage::User { ext, .. } => retain_if(ext, self).then_some("user"),
                ChatMessage::Assistant { ext, .. } => retain_if(ext, self).then_some("assistant"),
                ChatMessage::Developer { ext, .. } => retain_if(ext, self).then_some("developer"),
                ChatMessage::Tool { .. }
                | ChatMessage::Function { .. }
                | ChatMessage::Root { .. } => None,
            };
            dropped.extend(role);
        }
        if !dropped.is_empty() {
            // One line per request rather than per message, and the distinct
            // roles rather than one entry per message: the path is client
            // controlled, so both the line count and the line size must be
            // bounded. The model id is what makes a miss diagnosable.
            let count = dropped.len();
            dropped.sort_unstable();
            dropped.dedup();
            tracing::warn!(
                model = %req.model,
                active = ?self,
                dropped = count,
                roles = %dropped.join(","),
                "dropped message extensions that belong to another provider's profile"
            );
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
}

/// Case-insensitive ASCII prefix test that does not allocate.
fn starts_with_ignore_ascii_case(s: &str, prefix: &str) -> bool {
    s.get(..prefix.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
}

/// The `root` role is a MiniMax-only extension; other dialects reject it the
/// way their reference APIs do.
fn reject_root(req: &ChatCompletionRequest) -> Result<(), validator::ValidationError> {
    if req
        .messages
        .iter()
        .any(|m| matches!(m, ChatMessage::Root { .. }))
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
        for model in [
            "kimi-k3",
            "Kimi-K2.6",
            "/models/Kimi-K3",
            "moonshotai/kimi-k2",
            "openrouter/moonshotai/kimi-k2",
            "MoonshotAI/Kimi-K2-Instruct",
        ] {
            assert_eq!(
                ProviderProfile::for_model(model),
                ProviderProfile::Kimi,
                "{model}"
            );
        }
        for model in [
            "MiniMax-M3",
            "/models/MiniMax-M2",
            "MiniMaxAI/MiniMax-M2",
            "abab6.5s-chat",
        ] {
            assert_eq!(
                ProviderProfile::for_model(model),
                ProviderProfile::Minimax,
                "{model}"
            );
        }
        for model in [
            "gpt-4o-mini",
            "",
            "/models/llama-3",
            "my-kimi-alias",
            "openai/gpt-4o",
        ] {
            assert_eq!(
                ProviderProfile::for_model(model),
                ProviderProfile::OpenAi,
                "{model}"
            );
        }
    }
}
