//! What the engine holds of one policy document, and the engine as one
//! decision reads it.
//!
//! [`InForce`] is one value: the document, the number a list's value is
//! made under, what the engine reads of the document once, and the shadow
//! decisions counted under it. The engine keeps it in one field, so each
//! read of the document names that field.
//!
//! [`Loaded`] is the engine with that value, for one decision. An entry of
//! the engine takes it once ([`PolicyEngine::loaded`]) and hands it to
//! every function of its decision, so a decision reads one document at
//! each of its steps.

use std::{ops::Deref, sync::OnceLock};

use super::{labels, next_applied, PolicyEngine};
use crate::{
    document::spec::AppliedSpec,
    engine::{framework::SHADOWS, shadow::ShadowCounts},
    stages::Lifecycle,
    tracking::lease::{EventKind, EventSet},
};

/// What the engine holds of one policy document: the document, its number,
/// what is read of the document once, where the value is made
/// ([`Self::of`]), and the shadow decisions counted under it. Those counts
/// are written, by each decision a shadow answers beside.
#[derive(Debug)]
pub(in crate::engine) struct InForce {
    /// The validated document: its profiles, its bindings, and its stages
    /// with their state.
    pub(super) spec: AppliedSpec,
    /// The number of the document, which no other applied document of the
    /// process has ([`PolicyEngine::applied`]). A list's value is made
    /// under it and read under it alone ([`Loaded::members_of`]).
    pub(in crate::engine) applied: u64,
    /// The kinds of event the document takes ([`Lifecycle::hears`]), read
    /// once, here. A document that takes no word on a lease
    /// ([`EventSet::LEASE`]) costs a request no lease, no id and no word,
    /// and its plain requests are answered a place
    /// ([`Decided::Alone`](crate::engine::Decided::Alone)).
    pub(super) hears: EventSet,
    /// Each profile's name as a label that lives as long as the process,
    /// by the profile's position in the document
    /// ([`Loaded::label_at`]). Made once, here: naming the profile of a
    /// placement takes no lock.
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

/// The engine as one decision reads it: the engine object, and what it
/// holds of the document the decision runs under.
///
/// An entry of the engine takes the value once ([`PolicyEngine::loaded`])
/// and hands it to every function of its decision. It is two references,
/// copied where it is handed on. It reads as the engine for what outlives
/// a document: the clock, the lease counter, the record of prompts. What
/// is one document's is read in [`Self::document`], and a function of a
/// decision reads the document nowhere else.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Loaded<'e, 'd> {
    /// The engine object, for a caller that keeps it past the decision: a
    /// request's guard holds it.
    pub(crate) object: &'e PolicyEngine,
    /// What the engine holds of the document this decision runs under.
    pub(in crate::engine) document: &'d InForce,
}

impl Deref for Loaded<'_, '_> {
    type Target = PolicyEngine;

    #[inline(always)]
    fn deref(&self) -> &PolicyEngine {
        self.object
    }
}

impl Loaded<'_, '_> {
    /// Whether a stage of the document takes events of `kind`
    /// ([`InForce::hears`]): what a host's value for a request asks before
    /// it keeps anything for a word of that kind.
    #[inline]
    pub(crate) fn takes(self, kind: EventKind) -> bool {
        self.document.hears.contains(kind)
    }
}

impl PolicyEngine {
    /// The engine with its document, as one decision reads it: taken once,
    /// at the decision's entry.
    #[inline(always)]
    pub(crate) fn loaded(&self) -> Loaded<'_, '_> {
        Loaded {
            object: self,
            document: &self.document,
        }
    }
}
