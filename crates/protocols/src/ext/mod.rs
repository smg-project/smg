//! Provider-owned protocol extensions.
//!
//! Each provider contributes one module of all-`Option` extension structs that
//! are `#[serde(flatten)]`-ed into the core request types. Fields are promoted
//! here only when a provider's vendor-acceptance contract enforces behavior on
//! them; cosmetic extras stay out. Absent fields serialize to nothing, so
//! OpenAI-only traffic is wire-identical.

pub mod kimi;

use crate::profile::ProviderProfile;

/// A provider's extension struct knows which profile it belongs to, so a
/// request resolved to another profile drops it during normalization: the
/// field then never reaches a backend or a chat template, which is what serde
/// did before the field was typed. The owning profile keeps it, so its rules
/// can reject misuse with a 400 instead of hiding it.
pub trait ProviderExt {
    /// The profile whose contract defines these fields.
    const PROFILE: ProviderProfile;
    /// Reset every field to "absent".
    fn clear(&mut self);
}

/// Keep `ext` only when `active` is the profile it belongs to.
pub fn retain_if<E: ProviderExt>(ext: &mut E, active: ProviderProfile) {
    if E::PROFILE != active {
        ext.clear();
    }
}
