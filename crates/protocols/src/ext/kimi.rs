//! Kimi/Moonshot protocol extensions (K3 serving requirements; Kimi-Vendor-Verifier).

use serde::{Deserialize, Serialize};

use crate::{common::Tool, ext::ProviderExt, profile::ProviderProfile};

/// A declared tool list, kept raw when it does not parse so the Kimi profile
/// can reject it and every other profile can drop it without failing parsing.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum DeclaredTools {
    Tools(Vec<Tool>),
    Malformed(serde_json::Value),
}

impl DeclaredTools {
    /// The declared tools, when the declaration parsed.
    pub fn typed(&self) -> Option<&[Tool]> {
        match self {
            Self::Tools(tools) => Some(tools),
            Self::Malformed(_) => None,
        }
    }
}

/// Dynamic-tool declaration on system messages (K3): tools may be declared on
/// a system message with empty content, at any position in the conversation,
/// with the same status as request-level tools.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
pub struct KimiSystemExt {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<DeclaredTools>,
}

/// Captured so the Kimi profile can reject tools on non-system roles with a
/// 400 instead of dropping them silently (KVV test_dynamic_tools). Capture
/// only, as raw JSON: the rule keys on the field being set (null counts as
/// absent), and a malformed value must not fail parsing elsewhere.
/// The use site keeps it out of the published schema.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
pub struct KimiUserExt {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<serde_json::Value>,
}

/// See [`KimiUserExt`].
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
pub struct KimiAssistantExt {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<serde_json::Value>,
}

/// Dynamic-tool declaration on developer messages, handled like
/// [`KimiSystemExt`]: the OpenAI spec defines `developer` as the successor of
/// `system`, and this crate reads the two roles the same way.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
pub struct KimiDeveloperExt {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<DeclaredTools>,
}

impl ProviderExt for KimiSystemExt {
    const PROFILE: ProviderProfile = ProviderProfile::Kimi;
}

impl ProviderExt for KimiUserExt {
    const PROFILE: ProviderProfile = ProviderProfile::Kimi;
}

impl ProviderExt for KimiAssistantExt {
    const PROFILE: ProviderProfile = ProviderProfile::Kimi;
}

impl ProviderExt for KimiDeveloperExt {
    const PROFILE: ProviderProfile = ProviderProfile::Kimi;
}
