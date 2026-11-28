//! The profiles' labels: what a host counts a decision under, each a
//! string that lives as long as the process.
//!
//! A labelled model's profile bears its policy's name, and any other
//! profile its own name ([`of`]). The engine makes the list once, at its
//! start. A placement's label is one read of it ([`Loaded::label_at`]).

use super::Loaded;
use crate::{
    document::{
        spec::{AppliedSpec, PRESETS},
        RoutingSettings,
    },
    engine::view::Phase,
    names::intern,
};

/// The labels of `spec`'s profiles, each at its profile's position in the
/// document.
///
/// A profile is a labelled model's where two things hold. It is named for
/// a preset and a model's id, by the rule of
/// [`RoutingSettings::label_profile`]. And the document places that
/// model's single requests by it. Its label is the preset's name, the
/// preset's own string: no name is kept for a model's id, and the labelled
/// models of one policy bear one label. A supplied document that writes
/// such a profile is labelled the same way.
///
/// Any other profile is labelled by its name: one allocation a distinct
/// name, for the life of the process. So is a profile named that way
/// which the document binds another way, and one whose name begins with
/// no preset's.
///
/// Each such name asks the engine's own lookup of a single request's
/// binding once ([`Engine::profile_for`](crate::engine::Engine::profile_for)),
/// and only where the name begins with a preset's.
pub(super) fn of(spec: &AppliedSpec) -> Box<[&'static str]> {
    let engine = spec.engine();
    let labels = engine.profiles().iter().enumerate().map(|(at, profile)| {
        let named = RoutingSettings::label_policy(&profile.name);
        let policy = named.and_then(|(policy, model)| {
            let preset = PRESETS.into_iter().find(|preset| *preset == policy)?;
            let (placed, _) = engine.profile_for(model, Phase::Single)?;
            (usize::from(placed) == at).then_some(preset)
        });
        policy.unwrap_or_else(|| intern(&profile.name))
    });
    labels.collect()
}

impl Loaded<'_, '_> {
    /// The label of the profile at `position` of the document: a labelled
    /// model's policy, and for any other profile its name as the document
    /// writes it ([`of`]). It is a string that lives as long as the
    /// process. A placement holds it
    /// ([`Placement::profile`](crate::tracking::Placement::profile)).
    ///
    /// The position is the number an answer names its profile by
    /// ([`AlonePick::profile`](crate::engine::decision_engine::AlonePick::profile),
    /// a trace's `profile`), and it is a position in the document that
    /// decided. So the call that made the answer reads the label, under the
    /// document it decided with: a number kept past that call names nothing
    /// once another document is in force.
    ///
    /// One read of a list: no lock, no hash, no allocation. The empty label
    /// where the document holds no profile at the position, as for a
    /// decision made ahead of every profile.
    #[must_use]
    pub(crate) fn label_at(self, position: u16) -> &'static str {
        let label = self.document.labels.get(usize::from(position));
        label.copied().unwrap_or("")
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/engine/in_process/labels/tests.rs"]
mod tests;
