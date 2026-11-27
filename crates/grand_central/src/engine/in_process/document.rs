//! What the engine holds of one policy document, and the engine as one
//! decision reads it.
//!
//! [`InForce`] is one value: the document, the number a list's value is
//! made under, what the engine reads of the document once, the shadow
//! decisions counted under it and the decisions kept under it. The engine
//! keeps the value of the document in force in one field and replaces it
//! whole where it takes a new document ([`InProcess::apply`]).
//!
//! [`Loaded`] is the engine with that value, for one decision. An entry of
//! the engine reads the value once ([`InProcess::in_force`]) and hands
//! it to every function of its decision, so a decision reads one document
//! at each of its steps, whatever document is put in force meanwhile.
//!
//! [`AtEntry`] is that one read. A thread keeps a counted reference of the
//! document it read last ([`KEPT`]), and an entry that finds the engine's
//! number on it takes a plain count on the thread and makes no atomic
//! exchange.
//!
//! [`SpecHandle`] is the same value in a caller's hand
//! ([`InProcess::spec`]): it reads as the document.

use std::{
    cell::Cell,
    ops::Deref,
    rc::Rc,
    sync::{atomic::Ordering, Arc, OnceLock},
    time::Instant,
};

use parking_lot::Mutex;

use super::{labels, next_applied, InProcess};
use crate::{
    document::spec::AppliedSpec,
    engine::{
        decision::{Decision, PairSide, ShadowKind},
        explain::{DecisionRecord, Samples},
        framework::{ShadowRows, Stood, SHADOWS},
        shadow::{ShadowCount, ShadowCounts, ShadowSide, Shadowed, SHADOW_ROWS},
        view::{fingerprint, Phase},
    },
    stages::Lifecycle,
    store::{GatewayId, StaleGeneration},
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
    /// process has ([`InProcess::applied`]). A list's value is made
    /// under it and read under it alone ([`Loaded::members_of`]).
    pub(in crate::engine) applied: u64,
    /// The kinds of event the document takes ([`Lifecycle::hears`]), read
    /// once, here. A document that takes no word on a lease
    /// ([`EventSet::LEASE`]) costs a request no lease, no id and no word,
    /// and its plain requests are answered a place
    /// ([`Decided::Alone`](crate::engine::Decided::Alone)).
    pub(super) hears: EventSet,
    /// The kinds the document takes of a pick its session table has no
    /// word for: [`Self::hears`] without the table's own, read once, here.
    /// Under a sessions rule beside which no stage and no count of sends
    /// takes a word on a lease, such a pick of a whole pass costs its
    /// request no lease either ([`super::single::issued`]), and a place one
    /// pick decided is its request's whole answer where its worker reads
    /// ready a second time ([`super::single::still_ready`]). Where it does
    /// not, the pick is noted and takes its lease.
    pub(super) hears_beside_table: EventSet,
    /// Whether a place one pick decided may not be the whole of what the
    /// document asks of its request: a stage, a count of sends or the
    /// session table takes a word on a lease ([`Self::hears`]), or the
    /// document keeps a sample of its decisions ([`Self::samples`]). Read
    /// once, here: a place tests this word once, and under a document that
    /// asks neither it is the whole answer behind that test. Under one whose
    /// table alone takes a word, a place is the whole answer all the same
    /// where its worker reads ready a second time
    /// ([`Self::hears_beside_table`]).
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
    /// Whether a stage of the document that keeps load reports keeps a
    /// signal's column too ([`Engine::keepers_read_signals`]), read once,
    /// here. Under such a document a report the task tells states its
    /// named values ([`InProcess::tell_loads`]). Under any other none is
    /// converted for a telling: a column held by a stage that keeps no
    /// report is read at decisions alone.
    ///
    /// [`Engine::keepers_read_signals`]: crate::engine::Engine::keepers_read_signals
    pub(super) keepers_read_signals: bool,
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
    /// ([`super::sample::whole`]), and a single request's kept behind a
    /// placement once more, for the best rows of its trace
    /// ([`super::single::decide`]).
    pub(super) samples: bool,
    /// The decisions kept under the document, newest last
    /// ([`super::sample`]): one ring of them behind one lock. A kept
    /// decision takes the lock for its push ([`Self::keep_sample`]), and a
    /// reader for one scan ([`Self::kept_sample`]). The ring is allocated
    /// at its first push: a document that keeps no sample holds none.
    kept: Mutex<Samples>,
    /// The shadow decisions counted so far ([`Self::shadow_counts`]), by
    /// binding, side and kind. The tables are allocated at the first of
    /// them ([`Self::count_shadow`]): a document that names no shadow keeps
    /// none.
    shadow_tables: OnceLock<Box<ShadowCounts>>,
}

impl InForce {
    /// What the engine holds of `spec`, under the next number of the
    /// process.
    pub(super) fn of(spec: AppliedSpec) -> Self {
        let samples = spec.engine().explain_sample() != 0;
        let hears = spec.engine().hears();
        let hears_beside_table = spec.engine().hears_beside_table();
        let needs = spec.needs();
        let any_shadow = spec.engine().seat_flags() & SHADOWS != 0;
        let engine = spec.engine();
        Self {
            hears,
            hears_beside_table,
            leases_or_samples: hears.intersects(EventSet::LEASE) || samples,
            labels: labels::of(&spec),
            signals: needs.signal_cells,
            keepers_read_signals: engine.keepers_read_signals(),
            reads_loads: needs.loads || needs.signal_cells,
            any_shadow,
            samples,
            kept: Mutex::default(),
            spec,
            applied: next_applied(),
            shadow_tables: OnceLock::new(),
        }
    }

    /// What hears a word of a lease issued under `generation`, an earlier
    /// one than the document's: the stages the document kept that stood
    /// under that generation, and the session table. A stage built since
    /// hears none of it: it never heard that lease's issue. The lease's
    /// dispatch is counted apart, by every count of sends in force
    /// ([`lease_events::of_host`](crate::engine::lease_events::of_host)).
    ///
    /// The document holds nothing for it: the answer reads the document's
    /// own profiles by the generations they were made under
    /// ([`Engine::stood_by`]). So a document whose profiles were made under
    /// many generations holds each of them once.
    ///
    /// It is written for an earlier generation alone. Asked the document's
    /// own generation or one above it, it answers every stage the document
    /// kept and no stage it built: a caller that may hold such a lease
    /// tests the number first (`InForce::hearing`).
    ///
    /// [`Engine::stood_by`]: crate::engine::Engine::stood_by
    #[inline]
    pub(super) fn stood_at(&self, generation: u64) -> Stood<'_> {
        self.spec.engine().stood_by(generation)
    }

    /// What hears a word of a lease issued under `generation`, for a host
    /// that tells a lease's words off a file of its own and may find a
    /// lease of any generation there. A lease of the document's own
    /// generation is heard by the document's whole engine. One of an
    /// earlier generation is heard by what the document kept
    /// ([`Self::stood_at`]). One of a generation above the document's is
    /// heard by no stage and no table: no stage of this engine booked it.
    #[cfg(feature = "controller")]
    fn hearing(&self, generation: u64) -> Option<Stood<'_>> {
        match generation.cmp(&self.spec.generation()) {
            std::cmp::Ordering::Equal => Some(self.spec.engine().stands()),
            std::cmp::Ordering::Less => Some(self.stood_at(generation)),
            std::cmp::Ordering::Greater => None,
        }
    }

    /// `decision` pushed onto the document's ring as the decision of the
    /// request `id` names, taken at `at`. It is kept under the binding a
    /// request of `asked`'s model at its phase matches, and with what a
    /// single's shadow said. The document states its own number for the
    /// push, so no push of its own starts the ring over, and it holds the
    /// lock for that one push.
    pub(super) fn keep_sample(
        &self,
        id: &[u8],
        asked: (&str, Phase),
        at: Instant,
        decision: &Decision,
        shadow: Option<Shadowed>,
    ) {
        let binding = self.spec.engine().binding_position(asked.0, asked.1);
        let taken = (at, GatewayId::LOCAL, id);
        let mut kept = self.kept.lock();
        kept.push(self.applied, binding, taken, decision, shadow, [None; 2]);
    }

    /// The newest decision the document kept of the request `request_id`
    /// names, where it kept one ([`InProcess::explained`]): one scan of
    /// the ring under its lock, by the document's own number.
    pub(super) fn kept_sample(&self, request_id: &[u8]) -> Option<DecisionRecord> {
        let kept = self.kept.lock();
        kept.find(self.applied, GatewayId::LOCAL, fingerprint(request_id))
    }

    /// Counts what a shadow said, for its binding, in the table of the
    /// place it was asked: beside a single request, or beside a side of a
    /// pair. The tables are allocated at the first count.
    pub(super) fn count_shadow(&self, said: Shadowed, asked: ShadowSide) {
        let counts = self.shadow_tables.get_or_init(Box::<ShadowCounts>::default);
        let count = counts.of(said.binding, asked, said.kind);
        count.fetch_add(1, Ordering::Relaxed);
    }

    /// Tells `each` every count of the shadow decisions made under the
    /// document, and answers the document's number
    /// ([`InProcess::shadow_counts`]). The counts are of each binding that
    /// names a shadow, by the sides its plan asks one at and by each kind,
    /// a count of zero too. It takes no lock and allocates nothing: a count
    /// is one read of its cell, and a document no shadow was counted under
    /// yet reads as zeros.
    pub(super) fn shadow_counts(&self, mut each: impl FnMut(ShadowCount)) -> u64 {
        let (engine, counts) = (self.spec.engine(), self.shadow_tables.get());
        // Which sides each row counts by, a side at its place.
        let mut due = [[false; ShadowSide::COUNT]; SHADOW_ROWS];
        for at in (0..).take_while(|&at| engine.binding_at(at).is_some()) {
            let word = engine.shadow_at(at);
            let single = word.is_some_and(|(_, rows)| rows != ShadowRows::Sides);
            let [prefill, decode] = PairSide::BOTH.map(|side| engine.shadows_side(at, side));
            let row = &mut due[(at as usize).min(SHADOW_ROWS - 1)];
            for side in ShadowSide::ALL {
                row[side.index()] |= match side {
                    ShadowSide::Single => single,
                    ShadowSide::Prefill => prefill,
                    ShadowSide::Decode | ShadowSide::DecodeAgain => decode,
                };
            }
        }
        for (binding, due) in (0..).zip(due) {
            for side in ShadowSide::ALL.into_iter().filter(|side| due[side.index()]) {
                for kind in ShadowKind::ALL {
                    let cell = counts.map(|counts| counts.of(binding, side, kind));
                    let count = cell.map_or(0, |cell| cell.load(Ordering::Relaxed));
                    each(ShadowCount {
                        binding,
                        side,
                        kind,
                        count,
                    });
                }
            }
        }
        self.applied
    }
}

/// The policy document an engine decides under, as it stood when it was
/// asked for ([`InProcess::spec`]). It reads as the document.
///
/// An engine takes a new document while it runs ([`InProcess::apply`]).
/// A handle keeps showing the document it was asked under: what a caller
/// reads of it, the place it asks of it and the pick it makes under it
/// ([`InProcess::pick`]) are one document's.
///
/// A handle is one counted reference of its document, and so is each copy
/// of it. A document lives, with its stages and what they keep, while a
/// handle of it is held or a decision under it runs: a caller lets a
/// handle go with the work it asked it for. A thread that read it last
/// keeps it too, until it asks an engine again or ends
/// ([`InProcess::apply`]). A caller that reads one word
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

impl SpecHandle {
    /// `engine` with this handle's document, as one decision reads it
    /// ([`Loaded`]). For a host that keeps a handle beside each state it
    /// publishes, and decides a request under the document of the state
    /// the request read, whatever document is in force by then: a store's
    /// world is such a state. `engine` is the engine the handle was asked
    /// of. The value borrows the handle: taking it counts nothing.
    #[cfg_attr(
        not(any(test, feature = "controller")),
        expect(dead_code, reason = "a store's world asks it for a deciding thread")
    )]
    pub(crate) fn under<'a>(&'a self, engine: &'a InProcess) -> Loaded<'a, 'a> {
        engine.under(&self.0)
    }
}

thread_local! {
    /// The document this thread read last, of whichever engine it asked
    /// last ([`InProcess::in_force`]): one counted reference of it,
    /// which every read the thread makes of that document counts on. It is
    /// let go where the thread reads another document, and at the thread's
    /// end.
    static KEPT: Cell<Option<Rc<Kept>>> = const { Cell::new(None) };
}

/// One counted reference of a document, as a thread keeps it ([`KEPT`]).
#[derive(Debug)]
struct Kept(Arc<InForce>);

/// What an engine holds of the document in force, as one entry read it
/// ([`InProcess::in_force`]). It reads as that value, and the entry
/// holds the document by it: a decision ends under the document it began
/// under.
///
/// It is one count on its thread's own reference of the document
/// ([`Kept`]): taking it and letting it go are plain steps, and neither
/// touches the document's own count. So it stays on its thread: it is
/// neither `Send` nor `Sync`, and a future that held it across an await
/// could be spawned on no runtime that moves its tasks. No entry holds it
/// so: each is a plain function, and the engine's own task reads the
/// document in plain functions too ([`InProcess::run`]).
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
/// ([`InProcess::in_force`]) and hands this value to every function of
/// its decision. It is two references, copied where it is handed on. It
/// reads as the engine for what outlives a document: the clock, the lease
/// counter, the record of prompts. What is one document's is read in
/// [`Self::document`], and a function of a decision reads the document
/// nowhere else: the engine may hold another by then.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Loaded<'e, 'd> {
    /// The engine object, for a caller that keeps it past the decision: a
    /// request's guard holds it.
    pub(crate) object: &'e InProcess,
    /// What the engine holds of the document this decision runs under.
    pub(in crate::engine) document: &'d InForce,
}

impl Deref for Loaded<'_, '_> {
    type Target = InProcess;

    #[inline(always)]
    fn deref(&self) -> &InProcess {
        self.object
    }
}

impl<'d> Loaded<'_, 'd> {
    /// Whether a stage of the document this decision runs under takes
    /// events of `kind` ([`InForce::hears`]): what the call that places a
    /// request states of its placement, for the split that keeps something
    /// for a word of that kind
    /// ([`Placement::sides`](crate::tracking::Placement::sides)).
    #[inline]
    pub(crate) fn takes(self, kind: EventKind) -> bool {
        self.document.hears.contains(kind)
    }

    /// What hears a word of a lease issued under `generation`, by the
    /// document this value holds ([`InForce::hearing`]): `None` for a
    /// lease of a generation above that document's.
    #[cfg(feature = "controller")]
    #[inline]
    pub(crate) fn hearing(self, generation: u64) -> Option<Stood<'d>> {
        self.document.hearing(generation)
    }

    /// Whether the document this value holds is the one in force on its
    /// engine: one atomic load of the engine's number, compared with the
    /// document's own, and no read of the engine's cell. The test an entry
    /// makes of the document its thread kept ([`InProcess::in_force`]).
    /// A host that holds a document beside a state it published asks it,
    /// where it tells a word that goes by the document in force.
    #[cfg(feature = "controller")]
    #[inline]
    pub(crate) fn still_in_force(self) -> bool {
        self.object.applied.load(Ordering::Acquire) == self.document.applied
    }
}

impl InProcess {
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
    /// objects.
    ///
    /// A profile's stages are built from its own block, the document's
    /// `view` and its `signals`. So a new document that changes `view` or
    /// `signals` builds every profile anew. One change of `view` keeps
    /// them: another header list, where nothing else of the view changed.
    /// No stage is built from that list. A stage that reads a header by a
    /// place its host wrote is kept as it stands, and reads that place of
    /// the new list.
    ///
    /// A profile is built anew too where one of its stages, built again,
    /// says otherwise of itself than the standing one. That is another
    /// name, other words of a request's life to hear, or anything else the
    /// engine reads of a stage when it builds a profile. A factory of the
    /// host's own may build such a stage.
    ///
    /// A count of sends stands where a profile that reads it is kept,
    /// whichever of its other readers changed, left or came. A profile
    /// built new beside it reads the same count. A count none of whose
    /// readers is kept is built new and takes over what the standing
    /// counts hold: how is the count's own rule
    /// ([`Engine::succeed`](crate::engine::Engine::succeed)). Every other
    /// stage starts empty, a registered one too.
    ///
    /// The keep stands between two documents validated over the stock
    /// table, or over one table of the host's own as it stood at both
    /// validations ([`StageRegistry`](crate::stages::StageRegistry)). A
    /// host that builds its table anew for each document keeps nothing.
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
#[path = "../../../tests/unit/engine/in_process/document/tests.rs"]
mod tests;
