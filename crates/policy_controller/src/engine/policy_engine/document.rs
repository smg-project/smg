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
//! the engine reads the value once ([`PolicyEngine::in_force`]) and hands
//! it to every function of its decision, so a decision reads one document
//! at each of its steps, whatever document is put in force meanwhile.
//!
//! [`AtEntry`] is that one read. A thread keeps a counted reference of the
//! document it read last ([`KEPT`]), and an entry that finds the engine's
//! number on it takes a plain count on the thread and makes no atomic
//! exchange.
//!
//! [`SpecHandle`] is the same value in a caller's hand
//! ([`PolicyEngine::spec`]): it reads as the document.

use std::{
    cell::Cell,
    ops::Deref,
    rc::Rc,
    sync::{atomic::Ordering, Arc, OnceLock},
};

use super::{labels, next_applied, PolicyEngine};
use crate::{
    document::spec::AppliedSpec,
    engine::{
        framework::{Engine, SHADOWS},
        shadow::ShadowCounts,
    },
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
    /// What an older request's word is told to ([`Self::stood_at`]): for
    /// each generation below the document's own under which a profile it
    /// kept was built, in their order, the profiles and counts of sends
    /// that stood by then ([`Engine::hearing`]). Empty where every profile
    /// was built new.
    older: Box<[(u64, Engine)]>,
    /// No stage and no count, over the engine's session table: what hears
    /// a word older than every stage the document kept.
    table: Engine,
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
        let engine = spec.engine();
        let kept = |since| (since, engine.hearing(|made| made <= since));
        Self {
            older: engine.kept_since().into_iter().map(kept).collect(),
            table: engine.hearing(|_| false),
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

    /// What hears a word of a lease issued under `generation`, an earlier
    /// one than the document's: the stages the document kept that stood
    /// under that generation, and the session table. A stage built since
    /// hears none of it: it never heard that lease's issue. The lease's
    /// dispatch is counted apart, by every count of sends in force
    /// ([`lease_events::of_host`](crate::engine::lease_events::of_host)).
    pub(super) fn stood_at(&self, generation: u64) -> &Engine {
        let stood = |kept: &&(u64, Engine)| kept.0 <= generation;
        let kept = self.older.iter().rev().find(stood);
        kept.map_or(&self.table, |kept| &kept.1)
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
/// handle go with the work it asked it for. A thread that read it last
/// keeps it too, until it asks an engine again or ends
/// ([`PolicyEngine::apply`]). A caller that reads one word
/// of the document at each request asks
/// [`DecisionEngine::with_spec`](crate::engine::DecisionEngine::with_spec),
/// which borrows the document for the call: it reads through the
/// reference its thread keeps.
#[derive(Clone, Debug)]
pub struct SpecHandle(pub(super) Arc<InForce>);

impl Deref for SpecHandle {
    type Target = AppliedSpec;

    fn deref(&self) -> &AppliedSpec {
        &self.0.spec
    }
}

thread_local! {
    /// The document this thread read last, of whichever engine it asked
    /// last ([`PolicyEngine::in_force`]): one counted reference of it,
    /// which every read the thread makes of that document counts on. It is
    /// let go where the thread reads another document, and at the thread's
    /// end.
    static KEPT: Cell<Option<Rc<Kept>>> = const { Cell::new(None) };
}

/// One counted reference of a document, as a thread keeps it ([`KEPT`]).
#[derive(Debug)]
struct Kept(Arc<InForce>);

/// What an engine holds of the document in force, as one entry read it
/// ([`PolicyEngine::in_force`]). It reads as that value, and the entry
/// holds the document by it: a decision ends under the document it began
/// under.
///
/// It is one count on its thread's own reference of the document
/// ([`Kept`]): taking it and letting it go are plain steps, and neither
/// touches the document's own count. So it stays on its thread: it is
/// neither `Send` nor `Sync`, and a future that held it across an await
/// could be spawned on no runtime that moves its tasks. No entry holds it
/// so: each is a plain function, and the engine's own task reads the
/// document in plain functions too ([`PolicyEngine::run`]).
#[derive(Debug)]
pub(in crate::engine) struct AtEntry(Rc<Kept>);

impl Deref for AtEntry {
    type Target = InForce;

    #[inline(always)]
    fn deref(&self) -> &InForce {
        let Kept(document) = &*self.0;
        document
    }
}

/// The engine as one decision reads it: the engine object, and what it
/// holds of the document the decision runs under.
///
/// An entry of the engine reads the document in force once
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
    /// it keeps anything for a word of that kind. One read of the
    /// document ([`Self::in_force`]).
    #[inline]
    pub(crate) fn takes(&self, kind: EventKind) -> bool {
        self.in_force().hears.contains(kind)
    }
}

impl PolicyEngine {
    /// Puts `spec` in force on the running engine: the next decision is
    /// made under it. A decision under way ends under the document it
    /// began under, and so does a call that places a request, whose
    /// decisions are all one document's ([`Self::place`]). That document
    /// lives until the last such decision and the last handle of it
    /// ([`Self::spec`]) are gone, and until each thread that read it last
    /// asks an engine again or ends.
    ///
    /// What stands across the call: the engine's clock, its session table
    /// and its record of sent prompts. The table takes the new document's
    /// two times. The stages of a profile whose name and text did not
    /// change stand too, with what they hold: the new document keeps their
    /// objects. A profile's stages are built from its own block, the
    /// document's `view` and its `signals`, so a new document that changes
    /// `view` or `signals` builds every profile anew. A profile is built
    /// anew too where one of its stages, built again, says otherwise of
    /// itself than the standing one: another name, other words of a
    /// request's life to hear, or anything else the engine reads of a
    /// stage when it builds a profile. A factory of the host's own may
    /// build such a stage. A count of sends stands where a profile that
    /// reads it is kept, whichever of its other readers changed, left or
    /// came: a profile built new beside it reads the same count. A count
    /// none of whose readers is kept is built new and takes over a
    /// standing count: which one is the count's own rule
    /// ([`Engine::succeed`](crate::engine::Engine::succeed)). Every other
    /// stage starts empty, a registered one too. The keep stands between
    /// two documents validated over the stock table, or over one table of
    /// the host's own as it stood at both validations
    /// ([`StageRegistry`](crate::stages::StageRegistry)): a host that
    /// builds its table anew for each document keeps nothing.
    ///
    /// What a caller may rely on when the call returns: the stages built
    /// new have heard every worker that stands in the registry the
    /// program handed ([`Self::set_registry`]), each as joined and then
    /// the list of them, on the calling thread. A stage kept is told none
    /// of them again: it heard each as it came. So the first decision
    /// under the new document finds its stages told. A count built new
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
    /// A word of a lease issued under the standing document reaches the
    /// stages kept, the counts kept and the session table from here on,
    /// and no stage built since ([`DecisionEngine::say`]): such a stage
    /// never heard the lease's issue. The shadow counts start at zero
    /// ([`Self::shadow_counts`]). A list keeps the
    /// value the engine made of it under each document until the list
    /// falls ([`LentList`]): a list that stands across many documents is
    /// read past the values of the earlier ones. The document replaced
    /// is freed, with the stages the new one did not keep, where its last
    /// holder lets it go: this call, the last decision under it, or a
    /// thread that read it last. Each thread that asks an engine keeps the
    /// one document it read last, until it asks again or ends: so at most
    /// one replaced document stands for each thread, and an idle thread
    /// holds one for as long as it idles. As a rule the last holder is
    /// such a thread, and those stages are dropped at its next entry of
    /// an engine, which is a decision, a word of a lease or a read of the
    /// document, or at its end. A stage whose drop blocks holds that
    /// entry meanwhile, once for each document replaced. A drop that
    /// panics at an entry unwinds out of it. The process aborts where
    /// the entry is made while the thread already unwinds, as a request's
    /// guard dropped with a report owed speaks for its lease, and where
    /// the drop runs at a thread's end, as any value does that panics
    /// while a thread's own storage is torn down.
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
        spec.keep_from(&standing.spec);
        let next = Arc::new(InForce::of(spec));
        // The stages built new hear the workers that stand before a
        // decision reads them: each as joined, then the list of them, as
        // the task's whole read of the registry says it. A stage kept
        // heard them as they came. A count built new takes over between
        // the two: a join told here finds the count empty and drops no
        // send the standing count holds, and the list drops what stands
        // under an id that is gone.
        let new = next.spec.engine().hearing(|made| made == offered);
        for id in ids.as_deref().unwrap_or_default() {
            self.told(&new, &Membership::Added(*id));
        }
        next.spec.engine().succeed(standing.spec.engine());
        if let Some(ids) = &ids {
            self.told(&new, &Membership::Snapshot(ids));
        }
        let number = next.applied;
        self.document.store(next);
        // The document first, its number second: an entry that reads the
        // new number finds the new document in the cell. Until the number
        // stands, a thread that kept the standing document decides under
        // it once more, as a decision that began ahead of this call does.
        // Stored the other way round, an entry between the two would read
        // the standing document under the new number, keep it, and read
        // the cell again at its next entry.
        self.applied.store(number, Ordering::Release);
        self.members.owe_read();
        self.loads.handed_again();
        Ok(())
    }

    /// What the engine holds of the document in force, read for one
    /// decision or one read: a decision reads it once, at its entry, and no
    /// lock is taken.
    ///
    /// The calling thread keeps the document it read last ([`KEPT`]). Where
    /// that document's number is the engine's, the entry takes a count on
    /// the thread's reference of it: one atomic load of the engine's
    /// number, no atomic exchange, and nothing allocated. The numbers are
    /// compared and no address: no two applied documents of the process
    /// share a number. Otherwise the entry reads the engine's cell and the
    /// thread keeps what it read ([`Self::read_anew`]).
    ///
    /// An entry made inside another on one thread, as by a stage or a
    /// request's maker that asks the engine, takes its count like any
    /// other. A thread whose own storage is being torn down keeps nothing:
    /// an entry made there, as by a word of a lease said from a
    /// destructor, reads the cell each time.
    #[inline(always)]
    pub(in crate::engine) fn in_force(&self) -> AtEntry {
        let applied = self.applied.load(Ordering::Acquire);
        let kept = KEPT.try_with(|slot| {
            let kept = match slot.take() {
                Some(kept) if kept.0.applied == applied => kept,
                _ => self.read_anew(),
            };
            slot.set(Some(Rc::clone(&kept)));
            kept
        });
        AtEntry(kept.unwrap_or_else(|_| self.read_anew()))
    }

    /// The document in force read of the engine's cell, as a reference a
    /// thread keeps: one counted reference of the document, and one
    /// allocation. What an entry pays where its thread kept another
    /// document or none: at a thread's first entry, at its first entry
    /// behind a new document, and at each entry of a thread that asks two
    /// engines by turns.
    #[cold]
    #[inline(never)]
    fn read_anew(&self) -> Rc<Kept> {
        Rc::new(Kept(self.document.load_full()))
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
