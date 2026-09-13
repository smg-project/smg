//! Provider-owned protocol extensions.
//!
//! Each provider contributes one module of all-`Option` extension structs that
//! are `#[serde(flatten)]`-ed into the core request types. Fields are promoted
//! here only when a provider's vendor-acceptance contract enforces behavior on
//! them; cosmetic extras stay out. Absent fields serialize to nothing, so
//! OpenAI-only traffic is wire-identical. A struct that exists only so a
//! profile can reject the field is kept out of the published schema with
//! `#[schemars(skip)]` at its use site.

pub mod kimi;

use crate::profile::ProviderProfile;

/// A provider's extension struct knows which profile it belongs to, so a
/// request resolved to another profile drops it during normalization: the
/// field then never reaches a backend or a chat template, which is what serde
/// did before the field was typed. The owning profile keeps it, so its rules
/// can reject misuse with a 400 instead of hiding it.
pub trait ProviderExt: Default + PartialEq {
    /// The profile whose contract defines these fields.
    const PROFILE: ProviderProfile;

    /// Reset every field to "absent". Defaulted so that a field added to an
    /// extension cannot leak to another provider by omission.
    fn clear(&mut self) {
        *self = Self::default();
    }

    /// Whether no field is set.
    fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// Keep `ext` only when `active` is the profile it belongs to. Returns
/// whether a populated extension was dropped, so the caller can report it
/// once per request: a model id that profile selection did not recognise is
/// the usual cause, and a field that vanishes silently is hard to diagnose.
pub fn retain_if<E: ProviderExt>(ext: &mut E, active: ProviderProfile) -> bool {
    if E::PROFILE == active || ext.is_empty() {
        return false;
    }
    ext.clear();
    true
}
