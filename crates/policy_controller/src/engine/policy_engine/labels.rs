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
    let profiles = spec.engine().profiles().iter();
    profiles.map(|profile| intern(&profile.name)).collect()
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
mod tests {
    use std::ptr;

    use super::{super::PolicyEngine, *};
    use crate::{document::spec::PolicySpec, engine::decision::Trace};

    /// A profile's label is read by its position in the document, the
    /// number an answer names its profile by: each position reads its
    /// profile's name as the document writes it, one string a name in every
    /// engine of the process. A position the document holds no profile at
    /// reads the empty label, and so does the number of a decision made
    /// ahead of every profile.
    #[test]
    fn a_profiles_label_is_read_by_its_position_in_the_document() {
        let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
        let spare = document.profiles["round_robin"].clone();
        document.profiles.insert("a_spare".to_owned(), spare);
        let engine_of = |document: &PolicySpec| {
            PolicyEngine::new(AppliedSpec::validate(document).expect("the document validates"))
        };
        let (engine, another) = (engine_of(&document), engine_of(&document));
        let labels = (engine.loaded().label_at(0), engine.loaded().label_at(1));
        assert_eq!(labels, ("a_spare", "round_robin"));
        let same = ptr::eq(another.loaded().label_at(1), engine.loaded().label_at(1));
        assert!(same, "one string a name");
        let none = (
            engine.loaded().label_at(2),
            engine.loaded().label_at(Trace::NO_PROFILE),
        );
        assert_eq!(none, ("", ""));
    }
}
