//! Kimi/Moonshot protocol extensions (K3 serving requirements; Kimi-Vendor-Verifier).

use serde::{Deserialize, Serialize};

use crate::{common::Tool, ext::ProviderExt, profile::ProviderProfile};

/// Dynamic-tool declaration on system messages (K3): tools may be declared on
/// a system message with empty content, at any position in the conversation,
/// with the same status as request-level tools.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
pub struct KimiSystemExt {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

/// Captured so the Kimi profile can reject tools on non-system roles with a
/// 400 instead of dropping them silently (KVV test_dynamic_tools). Capture
/// only: the use site keeps it out of the published schema.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
pub struct KimiUserExt {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

/// See [`KimiUserExt`].
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
pub struct KimiAssistantExt {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

/// See [`KimiUserExt`].
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
pub struct KimiDeveloperExt {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
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
