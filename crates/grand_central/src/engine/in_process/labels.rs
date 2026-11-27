//! The profiles' labels: each profile's name as a string that lives as long
//! as the process, which a host counts a decision under.
//!
//! The engine makes the list once, at its start ([`of`]). A placement's
//! label is one read of it ([`Loaded::label_at`]).

use super::Loaded;
use crate::{document::spec::AppliedSpec, names::intern};

/// The labels of `spec`'s profiles, each at its profile's position in the
/// document. One allocation a distinct name, for the life of the process.
pub(super) fn of(spec: &AppliedSpec) -> Box<[&'static str]> {
    spec.engine().profile_names().map(intern).collect()
}

impl Loaded<'_, '_> {
    /// The label of the profile at `position` of the document: the
    /// profile's name as the document writes it, as a string that lives as
    /// long as the process. A placement holds it
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
