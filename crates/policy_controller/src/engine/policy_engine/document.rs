//! What the engine holds of one policy document, and the engine as one
//! decision reads it.
//!
//! [`InForce`] is one value: the document, the number a list's value is
//! made under, what the engine reads of the document once, the shadow
//! decisions counted under it and the decisions kept under it. The engine
//! keeps the value of the document in force in one field and replaces it
//! whole where it takes a new document ([`PolicyEngine::apply`]).
//!
//! [`Loaded`] is the engine with that value, for one decision. An entry of
//! the engine loads the value once ([`PolicyEngine::in_force`]) and hands
//! it to every function of its decision, so a decision reads one document
//! at each of its steps, whatever document is put in force meanwhile.
//!
//! [`SpecHandle`] is the same value in a caller's hand
//! ([`PolicyEngine::spec`]): it reads as the document.

use std::{
    ops::Deref,
    sync::{Arc, OnceLock},
};

use arc_swap::Guard;

use super::{labels, next_applied, PolicyEngine};
use crate::{
    document::spec::AppliedSpec,
    engine::{framework::SHADOWS, shadow::ShadowCounts},
    stages::Lifecycle,
    store::StaleGeneration,
    tracking::lease::{EventKind, EventSet, Membership},
};

/// What the engine holds of one policy document: the document, its number,
/// what is read of the document once, where the value is made
/// ([`Self::of`]), the shadow decisions counted under it and the decisions
/// kept under it. Those counts are written, by each decision a shadow
/// answers beside, and that ring by each decision kept.
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
    /// Whether a place one pick decided is not the whole of what the
    /// document asks of its request: a stage takes a word on a lease
    /// ([`Self::hears`]), or the document keeps a sample of its decisions
    /// ([`Self::samples`]). Read once, here: a place tests this word once,
    /// and under a document that asks neither it is the whole answer behind
    /// that test.
    pub(super) leases_or_samples: bool,
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
    /// Whether a decision under the document reads the load snapshot the
    /// engine's task keeps at hand: a stage of it reads a load report
    /// ([`Needs::loads`](crate::document::spec::Needs::loads)) or a named
    /// signal, whose values stand in the same reports. Read once, here. A
    /// decision under any other document reads no snapshot of the engine's.
    pub(super) reads_loads: bool,
    /// Whether a binding of the document names a shadow profile
    /// ([`SHADOWS`]), read once, here. A whole decision of a single request
    /// pays one test of it, and a place picked alone pays none.
    pub(super) any_shadow: bool,
    /// Whether the document keeps a sample of its decisions: its
    /// `explain.sample` is not zero
    /// ([`Engine::explain_sample`](crate::engine::Engine::explain_sample)).
    /// Read once, here. A whole decision tests it once
    /// ([`super::sample::whole`]).
    pub(super) samples: bool,
    /// The decisions kept under the document, newest last
    /// ([`super::sample`]): one ring of them, behind one lock that a kept
    /// decision takes for its push and a reader for one scan
    /// ([`PolicyEngine::explained`]). The ring is allocated at its first
    /// push: a document that keeps no sample holds none.
    pub(super) kept: super::sample::Kept,
    /// The shadow decisions counted so far
    /// ([`PolicyEngine::shadow_counts`]), by binding, side and kind. The
    /// tables are allocated at the first of them: a document that names no
    /// shadow keeps none.
    pub(super) shadow_counts: OnceLock<Box<ShadowCounts>>,
}

impl InForce {
    /// What the engine holds of `spec`, under the next number of the
    /// process.
    pub(super) fn of(spec: AppliedSpec) -> Self {
        let samples = spec.engine().explain_sample() != 0;
        let hears = spec.engine().hears();
        let needs = spec.needs();
        let any_shadow = spec.engine().seat_flags() & SHADOWS != 0;
        Self {
            hears,
            leases_or_samples: hears.intersects(EventSet::LEASE) || samples,
            labels: labels::of(&spec),
            signals: needs.signal_cells,
            reads_loads: needs.loads || needs.signal_cells,
            any_shadow,
            samples,
            kept: super::sample::Kept::default(),
            spec,
            applied: next_applied(),
            shadow_counts: OnceLock::new(),
        }
    }
}

/// The policy document an engine decides under, as it stood when it was
/// asked for ([`PolicyEngine::spec`]). It reads as the document.
///
/// An engine takes a new document while it runs ([`PolicyEngine::apply`]).
/// A handle keeps showing the document it was asked under: what a caller
/// reads of it, the place it asks of it and the pick it makes under it
/// ([`PolicyEngine::pick`]) are one document's.
///
/// A handle is one counted reference of its document, and so is each copy
/// of it. A document lives, with its stages and what they keep, while a
/// handle of it is held or a decision under it runs: a caller lets a
/// handle go with the work it asked it for. A caller that reads one word
/// of the document at each request asks
/// [`DecisionEngine::with_spec`](crate::engine::DecisionEngine::with_spec),
/// which borrows the document for the call and counts no reference.
#[derive(Clone, Debug)]
pub struct SpecHandle(pub(super) Arc<InForce>);

impl Deref for SpecHandle {
    type Target = AppliedSpec;

    fn deref(&self) -> &AppliedSpec {
        &self.0.spec
    }
}

/// The engine as one decision reads it: the engine object, and what it
/// holds of the document the decision runs under.
///
/// An entry of the engine loads the document in force once
/// ([`PolicyEngine::in_force`]) and hands this value to every function of
/// its decision. It is two references, copied where it is handed on. It
/// reads as the engine for what outlives a document: the clock, the lease
/// counter, the record of prompts. What is one document's is read in
/// [`Self::document`], and a function of a decision reads the document
/// nowhere else: the engine may hold another by then.
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

impl PolicyEngine {
    /// Whether a stage of the document in force takes events of `kind`
    /// ([`InForce::hears`]): what a host's value for a request asks before
    /// it keeps anything for a word of that kind. One load of the
    /// document.
    #[inline]
    pub(crate) fn takes(&self, kind: EventKind) -> bool {
        self.document.load().hears.contains(kind)
    }
}

impl PolicyEngine {
    /// Puts `spec` in force on the running engine: the next decision is
    /// made under it. A decision under way ends under the document it
    /// began under, and so does a call that places a request, whose
    /// decisions are all one document's ([`Self::place`]). That document
    /// lives until the last such decision and the last handle of it
    /// ([`Self::spec`]) are gone.
    ///
    /// What stands across the call: the engine's clock, its session table
    /// and its record of sent prompts. The table takes the new document's
    /// two times. A count of sends takes over a standing count: which one
    /// is the count's own rule
    /// ([`Engine::succeed`](crate::engine::Engine::succeed)). Every
    /// other stage of the new document starts empty, a registered one too.
    ///
    /// What a caller may rely on when the call returns: the new document's
    /// stages have heard every worker that stands in the registry the
    /// program handed ([`Self::set_registry`]), each as joined and then
    /// the list of them, on the calling thread. So the first decision
    /// under the new document finds its stages told. A count of sends
    /// takes over behind those joins and ahead of the list: a join this
    /// call tells drops no send the count takes over, and the list drops
    /// what it took over under an id that stands no more. A program that
    /// hands no registry says each join again behind the call
    /// ([`Self::joined`]). The engine's task reads the registry whole once
    /// more at its next upkeep and takes the load channel again there
    /// ([`Self::run`]). A stage that keeps load reports is told each
    /// worker's report then, as behind a start. Until then a decision
    /// under a document that reads load reports reads the snapshot the
    /// task kept last, and none behind a document that read none. The
    /// task fills the cells of the program's lists under the new document
    /// at that upkeep too ([`Self::set_lists`]): a request that asks a
    /// list ahead of it makes the list's value itself.
    ///
    /// A word of a lease issued under the standing document reaches no
    /// stage from here on ([`DecisionEngine::say`]), and the shadow counts
    /// start at zero ([`Self::shadow_counts`]). The session table
    /// hears no dispatch of such a lease either: a claim that decision
    /// made is neither confirmed nor taken back, and a pin its dispatch
    /// would move stays. A list keeps the
    /// value the engine made of it under each document until the list
    /// falls ([`LentList`]): a list that stands across many documents is
    /// read past the values of the earlier ones. The document replaced
    /// is freed, with its stages, where its last holder lets it go: this
    /// call, or the last decision under it.
    ///
    /// [`DecisionEngine::say`]: crate::engine::DecisionEngine::say
    /// [`LentList`]: crate::engine::LentList
    ///
    /// The call holds the lock a change of membership is told under, and
    /// asks the program of its workers under it
    /// ([`Members::standing`](super::membership::Members::standing),
    /// [`Members::facts_of`](super::membership::Members::facts_of)). No decision
    /// takes that lock: decisions run on beside the call. The call reaches
    /// no store: a store's word of membership takes the store's lock of its
    /// workers first and this one behind it.
    ///
    /// The engine's records of sent prefixes hear the list told here and
    /// forget every worker that does not stand. A join told here writes
    /// nothing to them.
    ///
    /// # Errors
    ///
    /// [`StaleGeneration`] where `spec`'s number is not above the standing
    /// document's. Nothing changes then.
    pub fn apply(&self, mut spec: AppliedSpec) -> Result<(), StaleGeneration> {
        let _held = self.telling.lock();
        let standing = self.document.load_full();
        let (stood, offered) = (standing.spec.generation(), spec.generation());
        if offered <= stood {
            return Err(StaleGeneration {
                standing: stood,
                offered,
            });
        }
        // The program is asked of its workers ahead of the first change: a
        // panic of its code leaves the engine as the call found it.
        let ids = self.members.standing_ids();
        spec.share_sessions(Arc::clone(&self.sessions));
        let next = Arc::new(InForce::of(spec));
        // The new document's stages hear the workers that stand before a
        // decision reads them: each as joined, then the list of them, as
        // the task's whole read of the registry says it. A count of sends
        // takes over between the two: a join told here finds the count
        // empty and drops no send the standing count holds, and the list
        // drops what stands under an id that is gone.
        for id in ids.as_deref().unwrap_or_default() {
            self.told(&next, &Membership::Added(*id));
        }
        next.spec.engine().succeed(standing.spec.engine());
        if let Some(ids) = &ids {
            self.told(&next, &Membership::Snapshot(ids));
        }
        self.document.store(next);
        self.members.owe_read();
        self.loads.handed_again();
        Ok(())
    }

    /// What the engine holds of the document in force, loaded for one
    /// decision or one read: a decision loads it once, at its entry, and no
    /// lock is taken.
    #[inline(always)]
    pub(in crate::engine) fn in_force(&self) -> Guard<Arc<InForce>> {
        self.document.load()
    }

    /// The engine with `document`, as one decision reads it.
    #[inline(always)]
    pub(in crate::engine) fn under<'d>(&self, document: &'d InForce) -> Loaded<'_, 'd> {
        Loaded {
            object: self,
            document,
        }
    }
}

#[cfg(test)]
mod tests;
