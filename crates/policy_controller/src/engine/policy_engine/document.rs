//! What the engine holds of one policy document.
//!
//! [`InForce`] is one value: the document, the number a list's value is
//! made under, and what the engine reads of the document once. The engine
//! keeps it in one field, so each read of the document names that field.

use std::sync::OnceLock;

use super::{labels, next_applied};
use crate::{
    document::spec::AppliedSpec,
    engine::{framework::SHADOWS, shadow::ShadowCounts},
    stages::Lifecycle,
    tracking::lease::EventSet,
};

/// What the engine holds of one policy document: the document, its number,
/// and what is read of the document once, where the value is made
/// ([`Self::of`]).
#[derive(Debug)]
pub(super) struct InForce {
    /// The validated document: its profiles, its bindings, and its stages
    /// with their state.
    pub(super) spec: AppliedSpec,
    /// The number of the document, which no other applied document of the
    /// process has ([`PolicyEngine::applied`](super::PolicyEngine::applied)).
    pub(super) applied: u64,
    /// The kinds of event the document takes ([`Lifecycle::hears`]), read
    /// once, here. A document that takes no word on a lease
    /// ([`EventSet::LEASE`]) costs a request no lease, no id and no word,
    /// and its plain requests are answered a place
    /// ([`Decided::Alone`](crate::engine::Decided::Alone)).
    pub(super) hears: EventSet,
    /// Each profile's name as a label that lives as long as the process,
    /// by the profile's position in the document
    /// ([`PolicyEngine::label_at`](super::PolicyEngine::label_at)). Made
    /// once, here: naming the profile of a placement takes no lock.
    pub(super) labels: Box<[&'static str]>,
    /// Whether a stage of the document reads a named signal
    /// ([`Needs::signal_cells`](crate::document::spec::Needs::signal_cells)),
    /// read once, here. Under such a document a single request whose
    /// profile reads one is decided with its list's signal cells in its
    /// facts ([`super::signals::decide`]), and any other document pays one
    /// test of this word.
    pub(super) signals: bool,
    /// Whether a binding of the document names a shadow profile
    /// ([`SHADOWS`]), read once, here. A whole decision of a single request
    /// pays one test of it, and a place picked alone pays none.
    pub(super) any_shadow: bool,
    /// The shadow decisions counted so far
    /// ([`DecisionEngine::shadowed`](crate::engine::DecisionEngine::shadowed)),
    /// by binding and kind. The table is allocated at the first of them: a
    /// document that names no shadow keeps none.
    pub(super) shadow_counts: OnceLock<Box<ShadowCounts>>,
}

impl InForce {
    /// What the engine holds of `spec`, under the next number of the
    /// process.
    pub(super) fn of(spec: AppliedSpec) -> Self {
        let hears = spec.engine().hears();
        let signals = spec.needs().signal_cells;
        let any_shadow = spec.engine().seat_flags() & SHADOWS != 0;
        Self {
            hears,
            labels: labels::of(&spec),
            signals,
            any_shadow,
            spec,
            applied: next_applied(),
            shadow_counts: OnceLock::new(),
        }
    }
}
