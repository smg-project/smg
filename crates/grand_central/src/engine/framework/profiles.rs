//! The engine's profiles, as the engine answers for them. It keeps their
//! list in the document's order: a binding and an alone place name a profile
//! by its position there, and so does a trace made under one.

use std::sync::Arc;

use super::{Binding, Engine, Plan, Profile};

impl Engine {
    /// The document's profiles, in its order: a profile's place here is its
    /// position.
    pub fn profiles(&self) -> &[Arc<Profile>] {
        &self.profiles
    }

    /// The profiles' names, in the document's order: a name's place here is
    /// its profile's position.
    #[cfg(feature = "in-process")]
    pub(crate) fn profile_names(&self) -> impl Iterator<Item = &str> {
        self.profiles.iter().map(|profile| profile.name.as_str())
    }

    /// The profile at `at` of the document, where `binding` names it for
    /// one of its sides: as its profile, as its decode profile, or as the
    /// encode profile of its plan. `None` for a profile the binding does
    /// not name, and for a position past the list.
    pub(crate) fn profile_under(&self, binding: &Binding, at: u16) -> Option<&Profile> {
        let bound = at == binding.profile
            || binding.decode_profile == Some(at)
            || matches!(binding.plan, Plan::Epd { encode } if encode == at);
        self.profiles
            .get(usize::from(at))
            .filter(|_| bound)
            .map(Arc::as_ref)
    }
}
