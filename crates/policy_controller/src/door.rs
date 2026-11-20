//! The decision door: the order of one decision's calls, written once.
//!
//! A decision is one load of the published rows
//! ([`MemoryStore::deciding`]), the engine's decision over them, the
//! lease filed at the issue and the engine's hearing of it, in that
//! order. Every host runs this one order, behind a transport of its own.
//! The served controller runs it on its deciders' threads, behind the
//! fence and the budget that a wire and an election bring. A host that
//! embeds the engine runs it on the caller's thread ([`Door`]): no frame
//! is built and no thread is hopped. A single's issued lease records the
//! prefix for its first target when the naming read a record of
//! sent prefixes and the store has heard no cache event of the target.
//!
//! A request reaches the order in the form it was asked in (the trait
//! `Asked`): a frame as the served controller checked it, or the facts a
//! host in process holds ([`Request`]).
//!
//! With sampling on, a seat counts picks and sheds under its decision
//! lock. Only a decision that keeps a record also takes the ring lock.
//! [`Door::explain`] reads by gateway and request id under the ring lock.
//! [`Door::explained`] also reads the names when their generation stands.
//!
//! A host with encode items can place them through [`Door::decide_encode_pair`].
//! The pair's prefill and decode are the sides that hold leases.

use std::{
    cell::Cell,
    mem,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering},
        Arc, OnceLock,
    },
    time::{Duration, Instant},
};

use parking_lot::{Mutex, MutexGuard};

pub(crate) use crate::store::world::narrow;
use crate::{
    counts::Slot,
    decider::Decider,
    decision::{Decision, Pair, Pick, ShadowKind, Shed, ShedReason, Target, Targets, Trace},
    demands::{DemandFacts, Demands, Unmet},
    framework::{
        release, EncodeItems, EncodePlaced, Engine, Plan, Scratch, ShadowRows, KEEPS_DECIDERS,
        SAMPLES_DECISIONS, SHADOWS,
    },
    lease::{Lease, LeaseId, LeaseKind, Lifecycle, Outcome, Side},
    sessions::{Forgets, SessionVerdict},
    spec::AppliedSpec,
    store::{
        leases::DISPATCH_TTL, Ask, AtWork, Built, DecodeRefused, DecodeSide, Dispatch,
        EncodePairBuffers, GatewayId, Hand, HealthSink, LeaseRecord, LeaseRefused, LeaseState,
        LeasesStore, MemoryStore, Named, Naming, PairBuffers, Picked, Pin, Said, SessionsStore,
        Sides, ViewBuffers, World,
    },
    view::{
        fingerprint, AffinityFacts, Candidate, Constraints, HashKind, KeyedRow, NamedAffinity,
        Phase, PinSource, PoolFacts, RequestFacts, Role, Session, WireModes, WorkerId,
    },
};

mod explain;

pub use explain::{DecisionRecord, Explained, Names};
pub(crate) use explain::{SampleCount, Samples};

/// A request as a door reads it: what the rows' narrowing, the engine's
/// request and the lease take of it, in whatever form it was asked.
pub(crate) trait Asked {
    /// The buffers a read of the request's lists is written on, kept
    /// across requests: `()` for a request that holds its own lists.
    type Lists: Default;
    /// The model the request asks for.
    fn model(&self) -> &str;
    /// The phase of a plan the request is for.
    fn phase(&self) -> Phase;
    /// The request's id as its asker stated it: a lease keeps its
    /// fingerprint.
    fn id(&self) -> &[u8];
    /// The wires the asker dials, where it states them.
    fn wires(&self) -> Option<WireModes>;
    /// The subset the asker narrows its rows to, where it states one.
    fn subset(&self) -> Option<u32>;
    /// What the request demands of the worker that takes it.
    fn demands(&self) -> Demands;
    /// Whether the asker sends a pair's decode after its prefill has run:
    /// the pair's first decision then files the decode side chosen.
    fn decode_later(&self) -> bool;
    /// How many block hashes the request carries.
    fn hashes(&self) -> usize;
    /// How many of them the generation of `spec` reads.
    fn hashes_read(&self, spec: &AppliedSpec) -> usize;
    /// The request's block hashes, all of them.
    fn block_hashes(&self) -> &[u64];
    /// The cache namespace seed and what the request's blocks hash.
    fn cache_namespace(&self) -> (u64, HashKind);
    /// The request's seed.
    fn seed(&self) -> u64;
    /// Hands `decide` the request in the engine's form, with what the
    /// deciding host's view adds to it. `name` is asked first, over that
    /// form with no row named, and what it answers stands as the request's
    /// named rows ([`RequestFacts::named`]) for `decide`: the form is built
    /// once. A decision that names no row hands [`unnamed`].
    fn read<'n, R>(
        &self,
        lists: &mut Self::Lists,
        viewed: Viewed<'_>,
        name: impl FnOnce(&RequestFacts<'_>, &Constraints<'_>) -> Option<NamedAffinity<'n>>,
        decide: impl FnOnce(&RequestFacts<'_>, &Constraints<'_>) -> R,
    ) -> R;
}

/// The naming of a decision that names no row ([`Asked::read`]).
pub(crate) fn unnamed(_: &RequestFacts<'_>, _: &Constraints<'_>) -> Option<NamedAffinity<'static>> {
    None
}

/// How many of a generation's bindings count their shadows in a row of
/// their own ([`Part::shadowed`]). The bindings behind them count in the
/// last row.
pub(crate) const SHADOW_ROWS: usize = 16;

/// One decider's part of the counts that every decision can move, on cache
/// lines of its own. Where a fleet's gateways speak fewer wires than its
/// members, each decision counts the candidates it lost; where a host
/// hashes past what its generation reads, each counts its request. One
/// count for every decider would be a line they all write.
#[derive(Debug, Default)]
#[repr(align(128))]
pub(crate) struct Part {
    /// Candidates the wires of their decision's gateway took from its
    /// view, members that share no wire with it.
    pub(crate) undialed: AtomicU64,
    /// Requests that carried more block hashes than their generation
    /// reads of them.
    pub(crate) hashes_cut: AtomicU64,
    /// The shadows of this decider's decisions ([`Part::shadowed`]),
    /// laid at its first: the part of a decider that shadows nothing
    /// stays its one pair of lines.
    shadowed: OnceLock<Box<ShadowCounts>>,
}

/// One decider's counts of its shadows, on lines of their own: a row a
/// binding by its position in its generation, a count a kind.
#[derive(Debug, Default)]
#[repr(align(128))]
struct ShadowCounts([[AtomicU64; ShadowKind::COUNT]; SHADOW_ROWS]);

impl ShadowCounts {
    fn of(&self, binding: u32, kind: ShadowKind) -> &AtomicU64 {
        let row = (binding as usize).min(SHADOW_ROWS - 1);
        &self.0[row][kind.index()]
    }
}

impl Part {
    /// The count of the shadows that said `kind` beside a decision of
    /// the binding at position `binding`, for the decider that counts
    /// one: its counts are laid here at its first.
    pub(crate) fn shadowed(&self, binding: u32, kind: ShadowKind) -> &AtomicU64 {
        self.shadowed.get_or_init(Box::default).of(binding, kind)
    }

    /// That count as it stands, for a reader: none where the decider
    /// has counted no shadow.
    pub(crate) fn shadowed_so_far(&self, binding: u32, kind: ShadowKind) -> u64 {
        let counts = self.shadowed.get();
        counts.map_or(0, |counts| counts.of(binding, kind).load(Ordering::Relaxed))
    }
}

/// One of the parts' counts, summed over `parts`.
pub(crate) fn summed(parts: &[Part], count: impl Fn(&Part) -> &AtomicU64) -> u64 {
    let parts = parts.iter();
    parts.map(|part| count(part).load(Ordering::Relaxed)).sum()
}

/// The stage a decision is shed under when its pool has rows and its
/// gateway shares a wire with none of them, or its request's runtime is
/// none of theirs ([`Unmet::Runtime`]).
pub(crate) const WIRES: &str = "wires";

/// The stage a decision is shed under when its gateway hints a subset of 0
/// of a pool that has rows: a decision over no candidates, as it asked.
pub(crate) const SUBSET_HINT: &str = "subset-hint";

/// What a deciding thread keeps across decisions: the engine's run of a
/// decision allocates only while these grow. The lists are those of the
/// form its requests are asked in ([`Asked::Lists`]).
#[derive(Default)]
pub(crate) struct Buffers<L> {
    pub(crate) view: ViewBuffers,
    /// The rows of a lent view's mask a hold of the asking gateway stands
    /// on ([`World::held`]): asked where a veto masks the view.
    pub(crate) held: Vec<u32>,
    /// What each row of a built view states for a request's hard demands
    /// ([`World::demand_facts`]): read for a request that states one.
    pub(crate) demanded: Vec<DemandFacts>,
    /// Both sides' rows of a pair decision and the table of their classes.
    pub(crate) pair: PairBuffers,
    /// The rows of an encode-pair decision, made at the thread's first
    /// one: a thread that decides none keeps none.
    pub(crate) encode: Option<Box<EncodePairBuffers>>,
    pub(crate) scratch: Scratch,
    pub(crate) lists: L,
    pub(crate) named: Named,
    /// What a keyed decision reads of its key in the store, where its
    /// rule has an in-flight cap.
    pub(crate) keyed: Keyed,
}

/// A request as its decider takes it.
#[derive(Clone, Copy)]
pub(crate) struct Taken<'a, A> {
    pub(crate) request: &'a A,
    pub(crate) gateway: GatewayId,
    /// The decider's first read of its clock: the decision's instant.
    pub(crate) now: Instant,
    /// What is left of the request's budget at that instant.
    pub(crate) left: Duration,
    /// The store's sessions lent to the decision, where its host keeps
    /// them: its engine asks them for a key its own table holds no pin
    /// of, and the door reads there the count of a key whose rule has an
    /// in-flight cap.
    pub(crate) pins: Option<&'a StorePins<'a>>,
    /// The place of the decider that decides, at its file of leases: a
    /// decision whose profile holds a stage that may refuse by a report
    /// of leases applies what waits at the files of leases through it.
    /// None in process, where no file's frames wait for a look.
    pub(crate) place: Option<&'a AtWork>,
}

impl<'a, A> Taken<'a, A> {
    /// When the dispatch span of a lease filed for the request ends: the
    /// span counts from the read the request was taken at.
    pub(crate) fn due(&self) -> Instant {
        self.now + DISPATCH_TTL
    }
}

/// The store's pins of record as a decision's engine reads them
/// ([`PinSource`]): asked for a key the engine's table holds no pin of,
/// under no lock but the pins' own.
pub(crate) struct StorePins<'a> {
    pub(crate) store: &'a MemoryStore,
    /// The decision's instant: a pin's end and its age are read against
    /// it.
    pub(crate) now: Instant,
}

impl std::fmt::Debug for StorePins<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorePins").finish_non_exhaustive()
    }
}

impl StorePins<'_> {
    /// The worker of `pin`, a pin of record as a read of the store found
    /// it, where it stands at the decision's instant for a rule of
    /// `ttl_micros`.
    fn standing(&self, pin: Pin, ttl_micros: Option<u64>) -> Option<WorkerId> {
        // A pin at or past its own end is no pin, whether or not the sweep
        // has run: its end is the instant the engine's table forgot the
        // key, or the latest a lease's own rule counted where no table
        // held it. A TTL counted from the dispatch ends later than that,
        // and so may the TTL of a rule applied since. The end stands on
        // the decision clock: the ask's instant is turned into it here, at
        // a miss that finds a pin with an end and nowhere else.
        let ended = |ends: u64| ends <= self.store.micros(self.now);
        if pin.ends.is_some_and(ended) {
            return None;
        }
        // The asking rule's TTL holds as well, against the pin's last use:
        // a pin as old as that TTL is no pin, to the microsecond.
        let age = self.now.saturating_duration_since(pin.last_used);
        ttl_micros
            .is_none_or(|ttl| age < Duration::from_micros(ttl))
            .then_some(pin.worker)
    }
}

impl PinSource for StorePins<'_> {
    fn pinned(&self, key: Session, ttl_micros: Option<u64>) -> Option<WorkerId> {
        let pin = SessionsStore::lookup(self.store, key.get())?;
        self.standing(pin, ttl_micros)
    }
}

/// What a keyed decision under a rule with an in-flight cap reads of its
/// key in the store: the lists a decider keeps across its decisions.
#[derive(Debug, Default)]
pub(crate) struct Keyed {
    /// The workers that carry the key's requests in flight, each with its
    /// count, as the store holds them.
    carried: Vec<(WorkerId, u32)>,
    /// The same by row of the decision's candidates, ascending: what the
    /// engine reads beside the rows
    /// ([`RequestFacts::lent_session_in_flight`]).
    rows: Vec<KeyedRow>,
}

impl Keyed {
    /// Reads `session` in the store under one hold of the key's shard:
    /// the key's pin of record, kept for the engine's ask, and the key's
    /// requests in flight by row of `candidates`, the rows the decision is
    /// made over, which stand in the order of their ids. A worker with no
    /// row there is no candidate, and its count is dropped. `None` for a
    /// request that carries no key.
    fn read<'a>(
        &'a mut self,
        pins: &'a StorePins<'a>,
        session: Session,
        candidates: &[Candidate],
    ) -> Option<(KeyRead<'a>, &'a [KeyedRow])> {
        if session == Session::NONE {
            return None;
        }
        let pin = pins.store.keyed(session, &mut self.carried);
        debug_assert!(candidates.is_sorted_by_key(|row| row.id));
        let found = |&(worker, count): &(WorkerId, u32)| {
            let at = candidates.binary_search_by_key(&worker, |row| row.id);
            Some(KeyedRow {
                row: candidates[at.ok()?].position,
                in_flight: u16::try_from(count).unwrap_or(u16::MAX),
            })
        };
        self.rows.clear();
        self.rows.extend(self.carried.iter().filter_map(found));
        self.rows.sort_unstable_by_key(|keyed| keyed.row);
        Some((KeyRead { pins, session, pin }, &self.rows))
    }

    /// Hands `decide` the request with its key's count beside the rows,
    /// where the request carries a key. One hold of the key's shard in
    /// `sessions` reads the count and the pin of record: the engine is
    /// handed the count by row, and asks that read for the pin. A request
    /// that carries no key is handed on as it stands.
    // Forced into each of the hand's two calls: out of line a request
    // with no key would pay a call ahead of its engine's.
    #[inline(always)]
    fn lend<'a, R>(
        &'a mut self,
        sessions: &'a StorePins<'a>,
        facts: &RequestFacts<'_>,
        constraints: &Constraints<'_>,
        candidates: &[Candidate],
        decide: impl FnOnce(&RequestFacts<'_>, &Constraints<'_>) -> R,
    ) -> R {
        let read = self.read(sessions, facts.session(), candidates);
        let lent;
        let (facts, constraints) = match &read {
            Some((pin, rows)) => {
                let facts = RequestFacts {
                    lent_session_in_flight: Some(rows),
                    session_counted: true,
                    ..*facts
                };
                let constraints = Constraints {
                    pins: Some(pin),
                    ..*constraints
                };
                lent = (facts, constraints);
                (&lent.0, &lent.1)
            }
            None => (facts, constraints),
        };
        decide(facts, constraints)
    }
}

/// A keyed decision's one read of its key's shard, as the engine asks it
/// for the pin of record ([`PinSource`]): the pin the read found beside
/// the key's count, so the decision holds the shard once. Another key,
/// which a single's decision never asks, is read from the store.
#[derive(Debug)]
struct KeyRead<'a> {
    pins: &'a StorePins<'a>,
    session: Session,
    pin: Option<Pin>,
}

impl PinSource for KeyRead<'_> {
    fn pinned(&self, key: Session, ttl_micros: Option<u64>) -> Option<WorkerId> {
        if key != self.session {
            return self.pins.pinned(key, ttl_micros);
        }
        self.pins.standing(self.pin?, ttl_micros)
    }
}

/// Whether `phase` is one of the sides a pair plan decides at its prefill.
fn pair_side(plan: Plan, phase: Phase) -> bool {
    match plan {
        Plan::Single => false,
        Plan::PdPair => matches!(phase, Phase::Prefill | Phase::Decode),
        Plan::Epd { .. } => !matches!(phase, Phase::Single),
    }
}

/// The path a request takes to its decision under the plan its model is
/// bound to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Entry {
    /// A request that is no side of a pair plan: decided by
    /// [`decide_in`], and its pick issued one lease ([`issue`]).
    Single,
    /// The prefill side of a pair plan, the pair's one request: decided
    /// for both sides by `pair_in`, and its pick issued a lease a side
    /// (`issue_pair`).
    Pair,
    /// Another side of a pair plan. A pair's alternates ride its answer
    /// and a retry is a new pair request, so that side has no decision,
    /// and one side's entry of the engine never makes one.
    Side,
}

/// The path `taken`'s request takes in the world of one load
/// ([`pair_side`]). A single request is no plan's side: its binding is
/// not looked up here, and the engine looks it up once.
pub(crate) fn entry<A: Asked>(world: &World, taken: &Taken<'_, A>) -> Entry {
    entry_and_plan(world, taken).0
}

#[inline]
fn entry_and_plan<A: Asked>(world: &World, taken: &Taken<'_, A>) -> (Entry, Plan) {
    let (model, phase) = (taken.request.model(), taken.request.phase());
    if matches!(phase, Phase::Single) {
        return (Entry::Single, Plan::Single);
    }
    let binding = world.spec().engine().binding(model, phase);
    match binding {
        Some(binding) if pair_side(binding.plan, phase) => match phase {
            Phase::Prefill => (Entry::Pair, binding.plan),
            _ => (Entry::Side, binding.plan),
        },
        _ => (Entry::Single, Plan::Single),
    }
}

/// The engine's decision over one view of [`hand`]'s, with what a pick's
/// lease takes of its request. Over rows the store named it is made at the
/// entry that states what the pick's first target holds of them
/// ([`Engine::decide_stated`]), and over any other view at the plain entry.
/// Under `KEEPS` each is the entry's keeping form, which keeps what the
/// decision read of its session key for the request's shadow
/// ([`Engine::decide_keeping`]).
struct ByEngine<'a, const KEEPS: bool>(&'a Engine);

impl<const KEEPS: bool> ByEngine<'_, KEEPS> {
    // Forced into each form's one call: out of line the plain form would
    // pay a call the closure it stands for never made.
    #[inline(always)]
    fn decided<const STATED: bool>(
        self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        constraints: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> (Decision, Option<Booking>) {
        let engine = self.0;
        let decision = match (STATED, KEEPS) {
            (false, false) => engine.decide(facts, pool, candidates, constraints, scratch),
            (true, false) => engine.decide_stated(facts, pool, candidates, constraints, scratch),
            (false, true) => engine.decide_keeping(facts, pool, candidates, constraints, scratch),
            (true, true) => {
                engine.decide_stated_keeping(facts, pool, candidates, constraints, scratch)
            }
        };
        // What a pick's lease takes of its request, read where the
        // request stands as the engine read it.
        let booking = decision.pick().and_then(|pick| {
            Some(Booking {
                target: pick.targets.first()?.id,
                picked: Picked::of(&pick.targets),
                verdict: pick.session_verdict,
                // The scratch holds this decision's rule's TTL where
                // a rule governed the pick, and an earlier decision's
                // where none did.
                ttl_micros: if pick.session_verdict == SessionVerdict::None {
                    None
                } else {
                    scratch.session_ttl_micros()
                },
                generation: pick.trace.generation,
                session: facts.session(),
                counted: facts.session_counted,
                tokens: facts.prompt_tokens,
                now_micros: facts.now_micros,
                built: None,
            })
        });
        (decision, booking)
    }
}

impl Decides<(Decision, Option<Booking>)> for ByEngine<'_, false> {
    fn over(
        self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        constraints: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> (Decision, Option<Booking>) {
        self.decided::<false>(facts, pool, candidates, constraints, scratch)
    }

    fn over_named(
        self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        constraints: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> (Decision, Option<Booking>) {
        self.decided::<true>(facts, pool, candidates, constraints, scratch)
    }

    /// What the request states of itself left none of a pool that has
    /// rows: said under that stage's name, where the engine would name no
    /// stage.
    fn emptied(
        self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        _: &Constraints<'_>,
        _: &mut Scratch,
        stage: &'static str,
    ) -> (Decision, Option<Booking>) {
        let none = shed(ShedReason::NoEndpoints, stage, facts.seed, pool.generation);
        (none, None)
    }
}

/// The engine's decision over the world of one load, for a request that
/// is no side of a pair plan ([`Entry::Single`]): the rows and the engine
/// are of one publication. What the decision counts, it counts in `own`,
/// its decider's part. No part of a pair is read, built or carried here:
/// a pair's one request takes its own path (`pair_in`).
pub(crate) fn decide_in<A: Asked>(
    world: &World,
    taken: &Taken<'_, A>,
    buffers: &mut Buffers<A::Lists>,
    own: &Part,
) -> (Decision, Option<Booking>) {
    let spec = world.spec();
    let engine = spec.engine();
    if taken.request.hashes() > taken.request.hashes_read(spec) {
        own.hashes_cut.fetch_add(1, Ordering::Relaxed);
    }
    let plain = ByEngine::<false>(engine);
    let ((decision, booking), built) = hand(world, taken, buffers, &own.undialed, plain);
    (
        decision,
        booking.map(|booking| Booking { built, ..booking }),
    )
}

/// What a request's budget must still hold at the end of its live
/// decision for its shadow to be asked. A shadow asked with less could
/// take a served answer past its budget, where the decider's last read
/// refuses it.
pub(crate) const SHADOW_FLOOR: Duration = Duration::from_millis(1);

/// What a decision's shadow said beside it, where one was due.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Shadowed {
    /// The position of the request's binding in its generation.
    pub(crate) binding: u32,
    /// The shadow's decision beside the live one, and
    /// [`ShadowKind::Skipped`] where the shadow was due and not asked.
    pub(crate) kind: ShadowKind,
    /// The first target of the shadow's pick.
    pub(crate) first: Option<WorkerId>,
}

/// What became of a decision's shadow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Shadow {
    /// None was due: the request's binding asks none, or the door made
    /// the decision itself over a view its gateway left no row.
    NotDue,
    /// One was due: what it said, or that it was not asked.
    Said(Shadowed),
    /// The shadow panicked, and the panic was held where it was asked.
    Faulted,
}

/// [`ByEngine`] under a generation that shadows. The live decision is
/// made at the engine's keeping entries and taken with its booking as
/// values. Then the binding's shadow is asked over the same view
/// ([`Engine::shadow`]), ahead of the commit: the rows and the counts are
/// those the live decision read, and no stage has heard of its lease.
/// The shadow writes the scratch and its own stages, and nothing the
/// commit reads.
///
/// A shadow that is due is not asked in two places, and is said as
/// skipped. A request with less than [`SHADOW_FLOOR`] of its budget left
/// keeps that for its answer. And the rows the live decision was handed
/// are not the shadow profile's where that profile would be handed
/// others ([`ShadowRows::Apart`]). A shadow profile that reads no column
/// is asked without the rows named for the live one
/// ([`ShadowRows::Unnamed`]). One that is named rows of its own is asked
/// over rows the door names again for it ([`ShadowRows::Own`],
/// [`OwnRows`]), behind the live decision and the floor.
///
/// A shadow that panics is held here, inside the catch its host keeps
/// round the decision: the live decision stands as decided.
///
/// A decider is built for one way of asking its shadow ([`ShadowAsk`]):
/// one whose shadow reads the live decision's rows carries nothing of a
/// naming.
struct Shadowing<'a, L, R> {
    engine: &'a Engine,
    /// The request's binding and how its shadow reads the live
    /// decision's rows, as the seat read them ahead of the order
    /// ([`Engine::shadow_for`]).
    due: (u32, ShadowRows),
    /// What the request's budget still holds at the end of the live
    /// decision: asked there, and only where a shadow is due.
    left: L,
    /// How the shadow is asked.
    ask: R,
}

/// How a [`Shadowing`] decider asks its shadow, by the rows the shadow
/// profile reads: those of its live decision ([`LiveRows`]), or none of
/// them ([`OwnRows`]).
trait ShadowAsk {
    /// Whether the live decision reads the affinity column its needs
    /// declare ([`Decides::reads_column`]).
    fn reads_column(&self) -> bool;

    /// The shadow's decision over the live decision's view, by `rows`.
    #[expect(
        clippy::too_many_arguments,
        reason = "the engine's shadow entry and its inputs, beside the rows' word"
    )]
    fn shadow(
        self,
        engine: &Engine,
        rows: ShadowRows,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        constraints: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Option<Decision>;
}

/// The asking of a shadow that reads the rows its live decision was
/// handed ([`ShadowRows::Same`]), or the view without the rows named for
/// the live profile ([`ShadowRows::Unnamed`]).
struct LiveRows;

impl ShadowAsk for LiveRows {
    fn reads_column(&self) -> bool {
        true
    }

    // Forced into the decider's one call: the asking stands in line
    // there, as it did where the decider held no other.
    #[inline(always)]
    fn shadow(
        self,
        engine: &Engine,
        rows: ShadowRows,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        constraints: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Option<Decision> {
        // Rows of the shadow's own have an asking of their own
        // ([`OwnRows`]): a caller that hands them here would have the
        // shadow asked over the live profile's rows.
        debug_assert!(
            rows != ShadowRows::Own,
            "a shadow named rows of its own is asked by its own order"
        );
        match rows {
            // A profile that reads no column is named no row, live or
            // as a shadow: the names would turn its pick to a draw.
            ShadowRows::Unnamed => {
                let unnamed = RequestFacts {
                    named: None,
                    ..*facts
                };
                engine.shadow(&unnamed, pool, candidates, constraints, scratch)
            }
            _ => engine.shadow(facts, pool, candidates, constraints, scratch),
        }
    }
}

/// The asking of a shadow that reads none of its live decision's rows:
/// one that is named rows of its own ([`ShadowRows::Own`]), and one that
/// is skipped ([`ShadowRows::Apart`]). It holds what the door names such
/// rows by, beside the request: the world of the decision's one load,
/// the gateway and the wires its view was built and narrowed for,
/// whether a demand may have taken a row of it, and the buffer the rows
/// are written on.
struct OwnRows<'a> {
    world: &'a World,
    gateway: GatewayId,
    wires: Option<WireModes>,
    cut: bool,
    named: &'a mut Named,
    /// Whether the live profile reads the affinity column by its own
    /// stages: no column is filled for such a shadow's sake.
    column: bool,
    /// The binding's position: its shadow's rows are named under that
    /// shadow profile's own cap ([`Engine::shadow_cap_at`]).
    binding: u32,
}

impl ShadowAsk for OwnRows<'_> {
    fn reads_column(&self) -> bool {
        self.column
    }

    fn shadow(
        self,
        engine: &Engine,
        rows: ShadowRows,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        constraints: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> Option<Decision> {
        // A shadow that is skipped is asked nothing: its decider stops
        // ahead of the asking.
        if rows != ShadowRows::Own {
            return None;
        }
        // A profile that is named rows of its own reads them in the
        // column's place, as it does live.
        let cap = engine.shadow_cap_at(self.binding);
        let named = self.name(cap, facts, candidates);
        let own = RequestFacts {
            named: Some(named.affinity()),
            affinity: &[],
            ..*facts
        };
        engine.shadow(&own, pool, candidates, constraints, scratch)
    }
}

impl<'a> OwnRows<'a> {
    /// The rows of `candidates` named for the shadow of `facts`' request
    /// under `cap`, as [`hand`] names a live profile's: by the request's
    /// block hashes as the engine reads them, on the decision's seed, at
    /// the instant its view was built. A cap full of holders the shadow
    /// profile's pass refuses is not read past: that lookup
    /// ([`World::name_kept`]) takes the live profile's test of a row.
    fn name(self, cap: u32, facts: &RequestFacts<'_>, candidates: &[Candidate]) -> &'a Named {
        let Self {
            world,
            gateway,
            wires,
            cut,
            named,
            ..
        } = self;
        let role = role_of(facts.phase);
        let naming = Naming {
            gateway,
            block_hashes: facts.block_hashes,
            namespace_seed: facts.namespace_seed,
            seed: facts.seed,
            cap,
            wires,
            now_micros: facts.now_micros,
        };
        let (model, kind) = (facts.model, facts.hash_kind);
        if cut {
            world.name_in::<true>(model, role, &naming, candidates, kind, named);
        } else {
            world.name_in::<false>(model, role, &naming, candidates, kind, named);
        }
        named
    }
}

impl<L: FnOnce() -> Duration, R: ShadowAsk> Shadowing<'_, L, R> {
    // Forced into each form's one call, as the plain decider's is.
    #[inline(always)]
    fn decided<const STATED: bool>(
        self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        constraints: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> ((Decision, Option<Booking>), Shadow) {
        let engine = self.engine;
        let keeping = ByEngine::<true>(engine);
        let live = keeping.decided::<STATED>(facts, pool, candidates, constraints, scratch);
        let (binding, rows) = self.due;
        let said = |kind, first| {
            Shadow::Said(Shadowed {
                binding,
                kind,
                first,
            })
        };
        // Rows the shadow profile would not be handed, in either form.
        if rows == ShadowRows::Apart || (self.left)() < SHADOW_FLOOR {
            return (live, said(ShadowKind::Skipped, None));
        }
        #[cfg(test)]
        if let Some(asking) = ON_SHADOW.take() {
            asking();
        }
        let ask = self.ask;
        let asked = catch_unwind(AssertUnwindSafe(|| {
            ask.shadow(engine, rows, facts, pool, candidates, constraints, scratch)
        }));
        let shadow = match asked {
            Ok(Some(shadow)) => {
                let first = shadow.pick().and_then(|pick| pick.targets.first());
                said(
                    ShadowKind::of(&live.0, &shadow),
                    first.map(|target| target.id),
                )
            }
            Ok(None) => Shadow::NotDue,
            Err(payload) => {
                // A payload's own drop can panic: the engine's helper
                // frees it under a catch of its own.
                release(payload);
                Shadow::Faulted
            }
        };
        (live, shadow)
    }
}

impl<L: FnOnce() -> Duration, R: ShadowAsk> Decides<((Decision, Option<Booking>), Shadow)>
    for Shadowing<'_, L, R>
{
    fn over(
        self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        constraints: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> ((Decision, Option<Booking>), Shadow) {
        self.decided::<false>(facts, pool, candidates, constraints, scratch)
    }

    fn over_named(
        self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        constraints: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> ((Decision, Option<Booking>), Shadow) {
        self.decided::<true>(facts, pool, candidates, constraints, scratch)
    }

    /// The door's own shed over a view with no row, under the name of
    /// the stage that took the last of them: no profile decided it, so
    /// no shadow is set beside it.
    fn emptied(
        self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        constraints: &Constraints<'_>,
        scratch: &mut Scratch,
        stage: &'static str,
    ) -> ((Decision, Option<Booking>), Shadow) {
        let plain = ByEngine::<false>(self.engine);
        let none = plain.emptied(facts, pool, constraints, scratch, stage);
        (none, Shadow::NotDue)
    }

    fn reads_column(&self) -> bool {
        self.ask.reads_column()
    }
}

/// [`decide_in`] for a request whose binding names a shadow: the live
/// decision with its booking, and what became of that shadow
/// ([`Shadowing`]). `due` is the engine's word of the binding
/// ([`Engine::shadow_for`]), read once ahead of the decision. `left` says
/// what the request's budget still holds at the end of the live decision,
/// by its host's own clock. The shadow is asked over the rows its live
/// decision was handed ([`LiveRows`]): one that reads none of them is
/// decided by [`decide_own_in`].
pub(crate) fn decide_shadowed_in<A: Asked>(
    world: &World,
    taken: &Taken<'_, A>,
    buffers: &mut Buffers<A::Lists>,
    own: &Part,
    due: (u32, ShadowRows),
    left: impl FnOnce() -> Duration,
) -> ((Decision, Option<Booking>), Shadow) {
    shadowed_by(world, taken, buffers, own, due, left, LiveRows)
}

/// [`decide_shadowed_in`] for a shadow that reads none of its live
/// decision's rows: one that is named rows of its own, or one that is
/// skipped ([`OwnRows`]). No column is filled for such a shadow's sake:
/// the live decision's is filled where its own profile reads it.
///
/// `named` is the buffer the shadow's rows are named on. Its host keeps
/// it beside `buffers`, which the hand holds while the shadow is named.
/// It is left with no record to write where no shadow was named: a
/// naming an earlier decision left is no word of this one's issue.
pub(crate) fn decide_own_in<A: Asked>(
    world: &World,
    taken: &Taken<'_, A>,
    buffers: &mut Buffers<A::Lists>,
    own: &Part,
    due: (u32, ShadowRows),
    named: &mut Named,
    left: impl FnOnce() -> Duration,
) -> ((Decision, Option<Booking>), Shadow) {
    let engine = world.spec().engine();
    named.clear_sent();
    let ask = OwnRows {
        world,
        gateway: taken.gateway,
        wires: taken.request.wires(),
        cut: !taken.request.demands().is_none(),
        named,
        column: engine.live_column_at(due.0, taken.request.phase()),
        binding: due.0,
    };
    shadowed_by(world, taken, buffers, own, due, left, ask)
}

/// The hand of a shadowed decision by one way of asking its shadow
/// ([`ShadowAsk`]): the live decision with its booking, and what became
/// of that shadow.
#[inline(always)]
fn shadowed_by<A: Asked, R: ShadowAsk>(
    world: &World,
    taken: &Taken<'_, A>,
    buffers: &mut Buffers<A::Lists>,
    own: &Part,
    due: (u32, ShadowRows),
    left: impl FnOnce() -> Duration,
    ask: R,
) -> ((Decision, Option<Booking>), Shadow) {
    let spec = world.spec();
    let engine = spec.engine();
    if taken.request.hashes() > taken.request.hashes_read(spec) {
        own.hashes_cut.fetch_add(1, Ordering::Relaxed);
    }
    let shadowing = Shadowing {
        engine,
        due,
        left,
        ask,
    };
    let (((decision, booking), shadow), built) =
        hand(world, taken, buffers, &own.undialed, shadowing);
    let booking = booking.map(|booking| Booking { built, ..booking });
    ((decision, booking), shadow)
}

/// The engine's pair decision over the world of one load, for the request
/// that enters a pair plan at its prefill. Both sides' rows are written
/// for it of one publication ([`World::build_pair`]), with what the leases
/// on file book on each row's worker, less what the gateway states of
/// itself takes from each side ([`narrow`]); the engine's pair entry
/// decides over them once, and the request's constraints are read by both
/// sides. A side with rows that the gateway's own word leaves none is shed
/// under that constraint's name, as a single's pool is. The store's
/// narrowing of a pair keeps no hard demand yet: a pair's request that
/// states one is shed under the wires' name, never placed past it. A
/// pick's booking holds both sides, each worker's count beside its own
/// pool's row.
///
/// A pair whose profile on either side holds a stage that may refuse on a
/// count it is shown is decided behind an apply of what waits at the files
/// of leases, as a single is ([`hand`]), where its decider lent its place.
pub(crate) fn pair_in<A: Asked>(
    world: &World,
    taken: &Taken<'_, A>,
    buffers: &mut Buffers<A::Lists>,
    own: &Part,
) -> (Decision, Option<PairBooking>) {
    let Buffers {
        pair,
        scratch,
        lists,
        ..
    } = buffers;
    let later = taken.request.decode_later();
    pair_with::<false, _, _, _>(
        world,
        taken,
        own,
        pair,
        (lists, scratch),
        |pair, _, model, ask, wires, subset| {
            let built = world.build_pair(model, &ask, pair)?;
            world.count_pair_in_flight(&built, &ask, pair);
            Some((built, pair.narrow(wires, subset, &built)))
        },
        (
            |engine, pair, facts, pool, cx, scratch| {
                engine.decide_pair(facts, pool, pair.pools(), cx, cx, scratch)
            },
            |facts, decision, built, scratch| {
                pair_booking(world, facts, decision, built, scratch, later)
            },
        ),
    )
}

/// Decides a request's pair and places its encode items over one world.
pub(crate) fn encode_pair_in<A: Asked>(
    world: &World,
    taken: &Taken<'_, A>,
    items: EncodeItems<'_>,
    placed: &mut EncodePlaced,
    buffers: &mut Buffers<A::Lists>,
    own: &Part,
) -> (Decision, Option<PairBooking>) {
    let Buffers {
        scratch,
        lists,
        encode,
        ..
    } = buffers;
    let rows: &mut EncodePairBuffers = encode.get_or_insert_with(Box::default);
    let later = taken.request.decode_later();
    let answer = pair_with::<false, _, _, _>(
        world,
        taken,
        own,
        rows,
        (lists, scratch),
        |rows, engine, model, mut ask, wires, subset| {
            if !items.keys.is_empty() {
                // The prefill binding names the encode profile for the pair.
                let needs = engine.pair_encode_needs(model);
                ask.affinity |= needs.affinity;
                ask.signal_cells |= needs.signal_cells;
            }
            let built = world.build_encode_pair(model, &ask, rows)?;
            // Counts follow the build's row order until narrowing moves it.
            world.count_pair_in_flight(&built, &ask, rows.pair_mut());
            let (took, emptied) = rows.narrow(wires, subset, &built);
            // A request without keys reads the pair's narrowing alone.
            let narrowed = if items.keys.is_empty() && (took > 0 || emptied) {
                rows.pair_mut().narrowing(&built, took > 0)
            } else {
                (took, emptied)
            };
            Some((built, narrowed))
        },
        (
            |engine, rows, facts, pool, cx, scratch| {
                let (pools, encode) = rows.pools();
                engine.decide_encode_pair(
                    facts, pool, pools, encode, items, cx, cx, cx, scratch, placed,
                )
            },
            |facts, decision, built, scratch| {
                pair_booking(world, facts, decision, built, scratch, later)
            },
        ),
    );
    // A subset or wire cut can shed before the engine's entry clears what
    // an earlier decision placed.
    if answer.0.pick().is_none() {
        placed.clear();
    }
    answer
}

/// Reads a built pair, decides over it, and takes what `book` files of the
/// decision. The decision is shown the facts of the pool it places in
/// first: the prefills', or under `DECODE` the decodes', where a decode
/// call places in that pool alone.
fn pair_with<const DECODE: bool, A: Asked, R, B>(
    world: &World,
    taken: &Taken<'_, A>,
    own: &Part,
    rows: &mut R,
    (lists, scratch): (&mut A::Lists, &mut Scratch),
    build: impl FnOnce(
        &mut R,
        &Engine,
        &str,
        Ask,
        Option<WireModes>,
        Option<u32>,
    ) -> Option<((Built, Built), (u64, bool))>,
    (decide, book): (
        impl FnOnce(
            &Engine,
            &R,
            &RequestFacts<'_>,
            &PoolFacts<'_>,
            &Constraints<'_>,
            &mut Scratch,
        ) -> Decision,
        impl FnOnce(&RequestFacts<'_>, &Decision, Option<(Built, Built)>, &Scratch) -> Option<B>,
    ),
) -> (Decision, Option<B>) {
    let Taken {
        request,
        gateway,
        now,
        left,
        pins,
        place,
    } = *taken;
    let spec = world.spec();
    let engine = spec.engine();
    let model = request.model();
    if request.hashes() > request.hashes_read(spec) {
        own.hashes_cut.fetch_add(1, Ordering::Relaxed);
    }
    // What the pair's profiles read. The prefill lookup answers for the
    // prefill's profile alone and the decode lookup carries the pair's
    // decode profile: both sides are built by one ask, so it takes the
    // column where either reads it.
    let (prefill, _, prefill_by_reports) = engine.reads_for(model, Phase::Prefill);
    let (decode, _, decode_by_reports) = engine.reads_for(model, Phase::Decode);
    // A stage of either side that may refuse by a report of leases reads
    // what a hand would have applied in place, ahead of both sides' rows.
    if let Some(place) = place.filter(|_| prefill_by_reports || decode_by_reports) {
        place.apply_waiting();
    }
    let ask = Ask {
        gateway,
        now,
        pinned: None,
        affinity: prefill.affinity || decode.affinity,
        signal_cells: prefill.signal_cells || decode.signal_cells,
    };
    // One side alone is no pair: the engine answers over no candidate, as
    // it answers any host that hands it none.
    let mut pool = PoolFacts {
        generation: spec.generation(),
        ..PoolFacts::default()
    };
    let (wires, subset) = (request.wires(), request.subset());
    // The store narrows a pair's sides by its gateway's wires and keeps no
    // hard demand there yet: a request that states one is shed as an
    // emptied side is, never placed past its demand. Where both sides were
    // built it reads under the wires' name until the store keeps the
    // demand: a stage of its own would leave the hosts' tables again with
    // that change. Where a side has no pool it reads `none`, as a pair
    // that states no demand does.
    let demanded = !request.demands().is_none();
    let mut emptied = false;
    let built = build(rows, engine, model, ask, wires, subset).map(|(built, (took, none))| {
        let shown = if DECODE { built.1 } else { built.0 };
        pool = shown.pool_facts();
        emptied = none || demanded;
        if took > 0 {
            own.undialed.fetch_add(took, Ordering::Relaxed);
        }
        // The pair's list takes its name from the one routine that names a
        // cut. No demand takes a row of a pair yet.
        let cut = Took {
            undialed: took,
            ..Took::default()
        };
        pool.reach = cut_name(pool.reach, wires, cut);
        built
    });
    let viewed = Viewed {
        now_micros: built.map_or_else(|| world.micros(now), |(prefill, _)| prefill.now_micros),
        budget_micros: u32::try_from(left.as_micros()).unwrap_or(u32::MAX),
        // A pair's columns and cells ride its two pools, and a pair names
        // no rows.
        affinity: &[],
        stalled: &[],
        held: &[],
        in_flight: None,
        max_block_hashes: request.hashes_read(spec),
        pins: pins.map(|pins| pins as &dyn PinSource),
        signal_cells: None,
    };
    request.read(lists, viewed, unnamed, |facts, constraints| {
        if emptied {
            let emptied_by = if constraints.subset_hint == Some(0) {
                SUBSET_HINT
            } else {
                WIRES
            };
            let none = shed(
                ShedReason::NoEndpoints,
                emptied_by,
                facts.seed,
                pool.generation,
            );
            return (none, None);
        }
        let decision = decide(engine, rows, facts, &pool, constraints, scratch);
        let booking = book(facts, &decision, built, scratch);
        (decision, booking)
    })
}

/// What each side's lease takes of the pair's pick. The decode's session
/// is the request's under the decode phase. Where the gateway sends the
/// decode `later`, its side is filed chosen: it holds the pick's decode
/// targets and books nothing.
fn pair_booking(
    world: &World,
    facts: &RequestFacts<'_>,
    decision: &Decision,
    built: Option<(Built, Built)>,
    scratch: &Scratch,
    later: bool,
) -> Option<PairBooking> {
    let (pick, built) = decision.pick().zip(built)?;
    let first = pick.targets.first()?;
    let chosen = first.pair?;
    let ttl_micros = if pick.session_verdict == SessionVerdict::None {
        None
    } else {
        scratch.session_ttl_micros()
    };
    let (state, slot) = if later {
        (LeaseState::Chosen, None)
    } else {
        (
            LeaseState::issued(ttl_micros),
            world.booked(&built.1, chosen.decode_id),
        )
    };
    let decode = DecodeSide {
        target: chosen.decode_id,
        picked: Picked::of_decodes(&pick.targets),
        session: RequestFacts {
            phase: Phase::Decode,
            ..*facts
        }
        .session(),
        state,
        tokens: facts.prompt_tokens,
        target_index: 0,
    };
    Some(PairBooking {
        prefill: Booking {
            target: first.id,
            picked: Picked::of(&pick.targets),
            verdict: pick.session_verdict,
            ttl_micros,
            generation: pick.trace.generation,
            session: facts.session(),
            // A pair's decision reads no key's count: a sessions rule runs
            // under the single plan alone.
            counted: false,
            tokens: facts.prompt_tokens,
            now_micros: facts.now_micros,
            built: Some(built.0),
        },
        decode,
        slot,
    })
}

/// What a pick's lease takes of its decision's hand-over.
#[derive(Clone, Copy)]
pub(crate) struct Booking {
    /// The pick's first target: the worker the lease books.
    pub(crate) target: WorkerId,
    /// The pick's other targets and the ranks, for a `Dispatched` that
    /// names one by its index.
    pub(crate) picked: Picked,
    /// What the decision did with the request's session key.
    pub(crate) verdict: SessionVerdict,
    /// The TTL of the rule behind that verdict; none without one.
    pub(crate) ttl_micros: Option<u64>,
    /// The generation that decided.
    pub(crate) generation: u64,
    /// The request's session; none for a request with no key.
    pub(crate) session: Session,
    /// Whether the decision read its key's count: its lease then books
    /// one request of the key on its worker
    /// ([`MemoryStore::issue_keyed`]).
    pub(crate) counted: bool,
    /// The token-work the lease books: the prompt, as its host counted it.
    pub(crate) tokens: u32,
    /// The decision's instant in the rows' clock.
    pub(crate) now_micros: u64,
    /// The build the decision read: where its target's count stands.
    pub(crate) built: Option<Built>,
}

/// What a pair's two leases take of its decision's hand-over: the prefill
/// side's booking, which is a single's, and the decode side with the slot
/// of its worker's count as the decision's build of the decode pool names
/// it.
#[derive(Clone, Copy)]
pub(crate) struct PairBooking {
    pub(crate) prefill: Booking,
    pub(crate) decode: DecodeSide,
    pub(crate) slot: Option<Slot>,
}

/// Issues the lease of a single's pick, one a decision, ahead of the read
/// that lets its answer leave: a credit on the pick's first target, minted
/// and filed by the store under `term`, the fence's at the decision's
/// first read, at that read's instant, from which [`DISPATCH_TTL`] runs
/// for its gateway to name a dispatch. The lease books the count the
/// decision's own world lays beside its target's row, and the engine that
/// decided is told of it by `told`, the telling its caller hands. A lease
/// on file is a request on its way to that target: the record of sent
/// prefixes takes the prefix `named` holds, where the target's pool keeps
/// one ([`World::sent`]).
///
/// The store builds its issue for each telling, so a caller with a
/// telling of its own leaves every other caller's issue as it was built.
///
/// # Errors
///
/// The store's refusal: it stands under another term, or mints no id
/// under this one.
pub(crate) fn issue<A: Asked>(
    store: &MemoryStore,
    world: &World,
    taken: &Taken<'_, A>,
    term: u64,
    booking: Booking,
    named: &Named,
    told: impl Fn(&Lease),
) -> Result<Lease, LeaseRefused> {
    let record = record_of(taken, term, &booking, Side::Single);
    let booked = booked_in(world, &booking);
    // Under one hold of the turn the store applies what is handed, files
    // the lease, tells the engine that decided of it, and ends and tells
    // the leases it finds past their dispatch deadline. A decision that
    // read its key's count files a lease that books the key, whatever it
    // did with the pin: the next read holds this request.
    let lease = if booking.counted {
        store.issue_keyed(record, booked, told)
    } else {
        store.issue_decided(record, booked, told)
    }?;
    world.sent(named, booking.target);
    Ok(lease)
}

/// Issues the two leases of a pair's pick, [`issue`] for a pair: a lease a
/// side under the one id the store mints, one record and one write of the
/// file, each side booking the count beside its own pool's row, and the
/// engine told once a side it issues, the prefill's first. A decode side
/// filed chosen is told at its own issue and dated then: a decode call's,
/// or else its dispatch ([`issued_later`]). `Ok` is the prefill side's
/// lease, then the decode side's.
///
/// # Errors
///
/// Those of [`issue`].
pub(crate) fn issue_pair<A: Asked>(
    store: &MemoryStore,
    world: &World,
    taken: &Taken<'_, A>,
    term: u64,
    decision: &Decision,
    booking: PairBooking,
) -> Result<(Lease, Lease), LeaseRefused> {
    let PairBooking {
        prefill,
        decode,
        slot,
    } = booking;
    let record = record_of(taken, term, &prefill, Side::Prefill);
    let booked = booked_in(world, &prefill);
    let engine = world.spec().engine();
    let lease = store.issue_decided_pair(record, decode, [booked, slot], |lease| {
        engine.on_decision_issued(lease, decision);
    })?;
    // A pair's two leases share the id the store minted.
    Ok((lease, decode.lease(&lease)))
}

/// The record of the lease `booking` asks on `side`, as its door hands it
/// to the store: the store mints its id, and writes the mark of a pair's
/// lease where it files one.
#[inline(always)]
fn record_of<A: Asked>(
    taken: &Taken<'_, A>,
    term: u64,
    booking: &Booking,
    side: Side,
) -> LeaseRecord {
    let lease = Lease {
        id: LeaseId::default(),
        kind: LeaseKind::Credit,
        side,
        target_index: 0,
        booked: booking.target,
        generation: booking.generation,
        session: booking.session,
        session_verdict: booking.verdict,
        issued_micros: booking.now_micros,
        tokens: booking.tokens,
    };
    LeaseRecord {
        lease,
        gateway: taken.gateway,
        request: fingerprint(taken.request.id()),
        target: booking.target,
        picked: booking.picked,
        decode: None,
        term,
        issued_at: taken.now,
        dispatch_deadline: taken.due(),
        state: LeaseState::issued(booking.ttl_micros),
    }
}

/// The slot of the count `booking`'s lease books: its target's, as the
/// build its decision read names it.
#[inline(always)]
fn booked_in(world: &World, booking: &Booking) -> Option<Slot> {
    let built = booking.built?;
    world.booked(&built, booking.target)
}

/// A shed the deciders say in the engine's place, in the shape of the
/// engine's own.
pub(crate) fn shed(
    reason: ShedReason,
    emptied_by: &'static str,
    seed: u64,
    generation: u64,
) -> Decision {
    Decision::Shed(Shed {
        reason,
        emptied_by,
        retry_after_secs: 1,
        wait_allowed: false,
        trace: Trace {
            seed,
            generation,
            ..Trace::default()
        },
    })
}

/// How a decision is made over the view [`hand`] reads, in three forms:
/// over rows of which no stage of the profile's own reads the holdings,
/// over rows the store named for the profile's own affinity stages, and
/// over a pool that has rows of which the request's own narrowing left
/// none. A function of the view decides alike over all three. A decider
/// that says more of a decision over named rows writes the second form,
/// and [`hand`] asks it at a call of its own, so a decision that names no
/// row meets no test for it. A decider that names the stage an emptied
/// pool is shed under writes the third.
pub(crate) trait Decides<R> {
    fn over(
        self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        constraints: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> R;

    fn over_named(
        self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        constraints: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> R
    where
        Self: Sized,
    {
        self.over(facts, pool, candidates, constraints, scratch)
    }

    /// The answer for a pool that has rows of which the request's own
    /// narrowing left none, and the stage that took the last of them:
    /// asked over no rows unless the decider names the stage itself.
    fn emptied(
        self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        constraints: &Constraints<'_>,
        scratch: &mut Scratch,
        _stage: &'static str,
    ) -> R
    where
        Self: Sized,
    {
        self.over(facts, pool, &[], constraints, scratch)
    }

    /// Whether the affinity column the decision's needs declare has a
    /// reader over the rows this decider is handed. The needs join a
    /// shadow profile's to the live one's
    /// ([`Engine::reads_for`](crate::framework::Engine::reads_for)). A
    /// decider says no where the shadow alone declared the column and
    /// reads none of the live decision's rows: [`hand`] then fills none.
    fn reads_column(&self) -> bool {
        true
    }
}

impl<R, F> Decides<R> for F
where
    F: FnOnce(&RequestFacts<'_>, &PoolFacts<'_>, &[Candidate], &Constraints<'_>, &mut Scratch) -> R,
{
    fn over(
        self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        constraints: &Constraints<'_>,
        scratch: &mut Scratch,
    ) -> R {
        self(facts, pool, candidates, constraints, scratch)
    }
}

/// What [`hand`] reads of a pool for one request: the pool's facts, the
/// decision's instant in the rows' clock, the rows, the mask of the asking
/// gateway's vetoes beside lent rows with the rows of it a hold stands on,
/// the affinity column, the leases' counts beside lent rows, and the build
/// the rows were read by.
type Read<'a> = (
    PoolFacts<'a>,
    u64,
    &'a [Candidate],
    (&'a [u32], &'a [u32]),
    &'a [AffinityFacts],
    Option<&'a crate::SignalCells>,
    Option<crate::LentCounts<'a>>,
    Option<Built>,
);

/// Hands `decide` what the engine reads for one request: the request with
/// what is left of its budget, the pool's facts under the generation its
/// rows were published under, and those rows as the asking gateway reaches
/// them at the decision's instant, with the affinity column where the
/// request's profile declares it. A request that states no wires, no hint
/// of 0 and no hard demand reads its pool by the store's one door
/// ([`World::view`]): the rows where their publication left them, with its
/// gateway's standing vetoes and the leases' counts beside them, or rows
/// the door writes where the request's profile shows them to a plugin. One
/// that states wires, a hint of 0 or a demand is handed rows written for
/// it ([`World::build`]), what the leases on file book on each row's
/// worker written over them ([`World::count_in_flight`]), less what the
/// gateway states of itself and what the request's demands take
/// ([`narrow`], [`narrow_demanded`]). The rows its wires take are counted
/// in `undialed`, the decider's own part, and the list the narrowing
/// leaves takes a name of its own ([`cut_name`]). The request's block
/// hashes are cut at what that generation reads of them. `decide`'s answer
/// comes back with the build its rows were read by, `None` where no member
/// serves the model.
///
/// A request whose profile reads the affinity column by its own stages is
/// handed no column: the rows the store names for it stand in the column's
/// place ([`World::name`]), at most its sample of each list, and every
/// holder and every warming row where it takes none
/// ([`Engine::named_for`](crate::framework::Engine::named_for)). Where a
/// demand took a row, the rows that stay are named within themselves
/// ([`World::name_cut`]). The naming is the store's one reader of its
/// index: the column it builds holds a row's age and no holder. Over the
/// rows it named the decision is asked in the decider's form for them
/// ([`Decides::over_named`]).
///
/// A pool that has rows of which the request's own narrowing left none is
/// answered in the decider's form for it ([`Decides::emptied`]), under
/// the stage that took the last row: the request's hint of 0, the last
/// demand its rows failed first ([`Unmet`]), or its gateway's wires.
///
/// A request whose profile holds a stage that may refuse by a report of
/// leases ([`Filter::refuses_by_reports`](crate::framework::Filter::refuses_by_reports))
/// is handed its rows behind an apply of what waits at the files of leases
/// ([`AtWork::apply_waiting`]), where its decider lent its place.
pub(crate) fn hand<A: Asked, R>(
    world: &World,
    taken: &Taken<'_, A>,
    buffers: &mut Buffers<A::Lists>,
    undialed: &AtomicU64,
    decide: impl Decides<R>,
) -> (R, Option<Built>) {
    let Taken {
        request,
        gateway,
        now,
        left,
        pins,
        place,
    } = *taken;
    let spec = world.spec();
    let (model, phase) = (request.model(), request.phase());
    let engine = spec.engine();
    // What the decision's profile reads, whether it shows its rows to a
    // plugin, and whether a stage of it may refuse by a report of leases.
    let (needs, shown, by_reports) = engine.reads_for(model, phase);
    // Such a stage reads what a hand would have applied in place: what
    // the gateways reported is applied first, at this decider's own file
    // and at each other file that shows a watch. A decision under no such
    // stage reads no place.
    if let Some(place) = place.filter(|_| by_reports) {
        place.apply_waiting();
    }
    // A profile whose own stages read the column is handed the rows the
    // store names in the column's place, and no column: a stage that
    // reads holders is fed them. The sample caps the rows named, and a
    // profile that takes none is named every one. A profile that shows
    // its rows to a plugin, or whose warm-up stages rule by two rulings,
    // is named no row and keeps its column. The deciding profile answers
    // for itself: the needs are merged over a pair's two phases and a
    // sessions rule, so they say only where to ask, and no other decision
    // pays the second walk. They join a shadow profile's too: a column
    // that the shadow alone declared, and reads none of, is not filled
    // ([`Decides::reads_column`]).
    let column = needs.affinity && decide.reads_column();
    let cap = if column {
        engine.named_for(model, phase)
    } else {
        0
    };
    let names = cap > 0;
    let ask = Ask {
        gateway,
        now,
        pinned: None,
        affinity: column && !names,
        signal_cells: needs.signal_cells,
    };
    let role = role_of(phase);
    let Buffers {
        view,
        held: holds,
        demanded,
        scratch,
        lists,
        named,
        keyed,
        ..
    } = buffers;
    let (wires, demands) = (request.wires(), request.demands());
    // No member serves the model in the role: the engine answers over no
    // candidate, as it answers any host that hands it none.
    let none = || PoolFacts {
        generation: spec.generation(),
        ..PoolFacts::default()
    };
    // Nothing of the request's own is cut out of the rows the door hands:
    // the publication's own, the leases' counts beside them, or a copy the
    // door writes with the counts on it for a plugin, which is shown a
    // row's bits and ages as the row's own fields. A request that states a
    // hard demand is handed written rows, as one that states wires is.
    let by_door = wires.is_none() && request.subset() != Some(0) && demands.is_none();
    // The last demand the rows a demand took failed first, where one took
    // a row of the build: the rows left are named within themselves then.
    let mut unmet: Option<Unmet> = None;
    let (pool, now_micros, candidates, masks, affinity, cells, counts, built): Read<'_> = if by_door
    {
        match world.view(model, role, &ask, shown, view) {
            Some(read) => (
                read.pool_facts(),
                read.built.now_micros,
                read.candidates,
                (read.stalled, world.held(&read, &ask, holds)),
                read.affinity,
                read.signal_cells,
                read.in_flight,
                Some(read.built),
            ),
            None => (
                none(),
                world.micros(now),
                &[],
                (&[], &[]),
                &[],
                None,
                None,
                None,
            ),
        }
    } else {
        let built = world.build(model, role, &ask, view);
        let (pool, now_micros) = match built {
            Some(built) => {
                // What the leases on file book on each row's worker,
                // written before the request's own narrowing moves a row:
                // rows that are written carry their counts. A request that
                // keeps no row is counted nothing.
                if keeps_a_row(wires, request.subset()) {
                    world.count_in_flight(&built, &ask, &mut view.candidates);
                }
                // The pool's facts are of the rows before the request's own
                // narrowing, all but the list's name: the rows the wires
                // left are another list of the pool.
                let mut pool: PoolFacts<'_> = built.pool_facts();
                let took = if demands.is_none() {
                    Took {
                        undialed: narrow(view, wires, request.subset()),
                        ..Took::default()
                    }
                } else {
                    world.demand_facts(&built, &ask, demanded);
                    narrow_demanded(view, demanded, wires, request.subset(), demands)
                };
                unmet = took.unmet;
                if took.undialed > 0 {
                    undialed.fetch_add(took.undialed, Ordering::Relaxed);
                }
                // The rows the wires and a demand left are another list of
                // the pool, named by the one routine that names a cut.
                pool.reach = cut_name(pool.reach, wires, took);
                if took == Took::default() {
                    // Every row stands where the build left it, so the
                    // pool's engines name these rows as they name the
                    // publication's: a key is placed alike over both.
                    pool.engines = world.engines(&built);
                }
                (pool, built.now_micros)
            }
            None => (none(), world.micros(now)),
        };
        (
            pool,
            now_micros,
            &view.candidates,
            (&[], &[]),
            &view.affinity,
            // The cells the build wrote, each kept row's at its new place.
            view.cells(),
            None,
            built,
        )
    };
    let (stalled, held) = masks;
    let viewed = Viewed {
        now_micros,
        budget_micros: u32::try_from(left.as_micros()).unwrap_or(u32::MAX),
        affinity,
        stalled,
        held,
        in_flight: counts,
        max_block_hashes: request.hashes_read(spec),
        pins: pins.map(|pins| pins as &dyn PinSource),
        signal_cells: cells,
    };
    if candidates.is_empty() && pool.pool_size > 0 {
        // The request's own narrowing left none of a pool that has rows.
        let answer = emptied(request, lists, viewed, decide, &pool, scratch, unmet);
        return (answer, built);
    }
    // The store's sessions, where the decision's rule has an in-flight
    // cap: a keyed request reads its key's count there. Under any other
    // rule the engine is asked at once, past the key's read.
    let sessions = pins.filter(|_| needs.session_in_flight);
    if names {
        // A row is named by its place among the rows the decision is made
        // over: the lent rows, or the built ones after their narrowing.
        let hashes = request.block_hashes();
        let (namespace_seed, hash_kind) = request.cache_namespace();
        let naming = Naming {
            gateway,
            // The first hashes, as many as the generation reads of them:
            // the engine's request takes the same count.
            block_hashes: hashes.get(..request.hashes_read(spec)).unwrap_or(hashes),
            namespace_seed,
            seed: request.seed(),
            cap,
            wires,
            now_micros,
        };
        // The index reads a view by its gateway's wires alone. Rows a
        // demand cut are named within the rows that stay: a row the
        // demand took is named by nobody and takes no place of the cap.
        // The naming is a step of the read: it runs once the request
        // stands in the engine's form, and the rows it names are written
        // into that form in place.
        let name = move |facts: &RequestFacts<'_>, constraints: &Constraints<'_>| {
            // The names borrow the buffer past this call: the closure
            // gives its hold of it up.
            let named = named;
            if unmet.is_some() {
                world.name_in::<true>(model, role, &naming, candidates, hash_kind, named);
            } else {
                world.name_in::<false>(model, role, &naming, candidates, hash_kind, named);
            }
            // Under a cap the holders named may all be rows the pass
            // refuses, and a warm worker behind them is then never named.
            // A naming that fills its cap is held to the pass's own test
            // of a row, where the engine hands one: a cap full of refused
            // holders is read past. A naming shorter than its cap, or with
            // none, is asked nothing and the test is not made.
            world.name_kept(model, role, &naming, candidates, named, || {
                let test = engine.keeps_for(model, phase, facts, constraints)?;
                Some(move |row: &Candidate| test.keeps(row))
            });
            Some(named.affinity())
        };
        // Over the rows it named the decider is asked in its form for
        // them, at this arm's own call: a decision that names no row
        // meets no test for it.
        let answer = request.read(lists, viewed, name, |facts, constraints| {
            // The engine's ask, written once for the two ways to it: at
            // once, or behind the key's read.
            let ask = |facts: &RequestFacts<'_>, constraints: &Constraints<'_>| {
                decide.over_named(facts, &pool, candidates, constraints, scratch)
            };
            match sessions {
                None => ask(facts, constraints),
                Some(sessions) => keyed.lend(sessions, facts, constraints, candidates, ask),
            }
        });
        return (answer, built);
    }
    named.clear_sent();
    let answer = request.read(lists, viewed, unnamed, |facts, constraints| {
        let ask = |facts: &RequestFacts<'_>, constraints: &Constraints<'_>| {
            decide.over(facts, &pool, candidates, constraints, scratch)
        };
        match sessions {
            None => ask(facts, constraints),
            Some(sessions) => keyed.lend(sessions, facts, constraints, candidates, ask),
        }
    });
    (answer, built)
}

/// The answer for a pool that has rows of which the request's own narrowing
/// left none, in the decider's form for it ([`Decides::emptied`]), under
/// the stage that took the last of them: the request's hint of 0, the last
/// demand its rows failed first, or its gateway's wires. A routine of its
/// own, out of line: a pool that keeps a row pays one test for this answer
/// and none of its code.
#[cold]
#[inline(never)]
fn emptied<A: Asked, R>(
    request: &A,
    lists: &mut A::Lists,
    viewed: Viewed<'_>,
    decide: impl Decides<R>,
    pool: &PoolFacts<'_>,
    scratch: &mut Scratch,
    unmet: Option<Unmet>,
) -> R {
    let stage = if request.subset() == Some(0) {
        SUBSET_HINT
    } else {
        unmet.map_or(WIRES, Unmet::stage)
    };
    request.read(lists, viewed, unnamed, |facts, constraints| {
        decide.emptied(facts, pool, constraints, scratch, stage)
    })
}

/// The role whose pool decides a request of `phase`: the pool [`hand`]
/// reads its rows of, and the one a shadow's own rows are named in.
const fn role_of(phase: Phase) -> Role {
    match phase {
        Phase::Single => Role::Single,
        Phase::Prefill => Role::Prefill,
        Phase::Decode => Role::Decode,
        Phase::Encode => Role::Encode,
    }
}

/// Whether the request's own narrowing can leave a row: a subset of none
/// and a mask of no wire clear every row ([`narrow`]).
pub(crate) fn keeps_a_row(wires: Option<WireModes>, subset: Option<u32>) -> bool {
    subset != Some(0) && wires != Some(WireModes::NONE)
}

/// What a narrowing took from a built view.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Took {
    /// The rows the gateway's wires took: members that share no wire with
    /// it.
    pub(crate) undialed: u64,
    /// The name of the rows a demand took, by their places in the build's
    /// list: `0` where no demand took a row.
    pub(crate) demanded: u64,
    /// The last of the demands those rows failed first, in the order a
    /// host answers for them ([`Unmet`]): the stage a pool the demands
    /// left nobody is shed under. `None` where no demand took a row.
    pub(crate) unmet: Option<Unmet>,
}

/// [`narrow`] for a request that states hard demands ([`Demands`]): one
/// pass keeps the rows that share a wire with the gateway and meet every
/// demand, each by its own vision bit and its entry of `facts`, the
/// build's column ([`World::demand_facts`]). A row with no entry states
/// nothing there. The kept rows keep their order and take their positions,
/// and every column beside them follows in the same step: the affinity
/// column, the signal cells and the facts. A hint of 0 keeps none.
///
/// The rows a demand took are named by their places in the build's list
/// ([`Took::demanded`]), never by what the demand states: two requests
/// that leave one list share its name, so a rotation over that list is one
/// rotation whatever its requests stated.
pub(crate) fn narrow_demanded(
    view: &mut ViewBuffers,
    facts: &mut Vec<DemandFacts>,
    wires: Option<WireModes>,
    subset: Option<u32>,
    demands: Demands,
) -> Took {
    if subset == Some(0) {
        view.candidates.clear();
        view.affinity.clear();
        view.signal_cells.refill(0);
        facts.clear();
        return Took::default();
    }
    // The column is by row, where the profile declared one.
    let column = !view.affinity.is_empty();
    // The signal cells are by row too, where the build wrote any.
    let mut cells =
        crate::store::world::KeptCells::of(&mut view.signal_cells, view.candidates.len());
    let mut took = Took::default();
    let (mut kept, mut position) = (0, 0_u32);
    for at in 0..view.candidates.len() {
        let row = view.candidates[at];
        if wires.is_some_and(|wires| !row.wire_modes.intersects(wires)) {
            took.undialed += 1;
            continue;
        }
        // A row with no entry states nothing.
        let stated = facts.get(at).copied().unwrap_or_default();
        if let Some(unmet) = demands.unmet_by(row.vision, stated) {
            // One more than the place: the first row's taking is no zero.
            // Each place turns the fold by a product of the places' own
            // word: the name is of these places in their order, and of
            // no set of wires ([`cut_name`]).
            took.demanded = (took.demanded ^ (at as u64 + 1))
                .wrapping_mul(PLACES)
                .max(1);
            took.unmet = took.unmet.max(Some(unmet));
            continue;
        }
        view.candidates[kept] = Candidate { position, ..row };
        position += 1;
        if column {
            view.affinity[kept] = view.affinity[at];
        }
        if let Some(cells) = &mut cells {
            cells.keep(at);
        }
        if let Some(entry) = facts.get_mut(kept) {
            *entry = stated;
        }
        kept += 1;
    }
    view.candidates.truncate(kept);
    if column {
        view.affinity.truncate(kept);
    }
    if let Some(cells) = cells {
        cells.end();
    }
    facts.truncate(kept);
    took
}

/// The name of the list a gateway's wires cut out of a build's
/// ([`PoolFacts::reach`]): the build's word turned by the wires. Each of the
/// eight sets of wires turns it by its own multiple of one odd word, eight
/// words that differ and none of them zero: no cut reads as its build's
/// whole list, and no two cuts of one build read alike.
pub(crate) fn cut_reach(built: u64, wires: WireModes) -> u64 {
    built ^ (u64::from(wires.bits()) + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// The word the places a demand took are folded by ([`narrow_demanded`]):
/// another odd word than the wires' ([`cut_reach`]).
const PLACES: u64 = 0xD6E8_FEB8_6659_FD93;

/// The name of the list a request's own narrowing left of a build's
/// ([`PoolFacts::reach`]), the one routine that names a cut list: the
/// build's word turned by the gateway's wires where they took a row
/// ([`cut_reach`]), then by the places a demand took ([`Took::demanded`]).
/// The parts are kept apart. The places are folded by a word of their
/// own, and their turn is a product over the wires' turn, never a second
/// XOR beside it: a demand's cut reads as no wires' cut, and a cut by
/// both never reads as the build's whole list. A narrowing in which no
/// demand took a row names its list as its wires alone do.
pub(crate) fn cut_name(built: u64, wires: Option<WireModes>, took: Took) -> u64 {
    let dialed = match wires {
        Some(wires) if took.undialed > 0 => cut_reach(built, wires),
        _ => built,
    };
    if took.demanded == 0 {
        return dialed;
    }
    (dialed ^ took.demanded)
        .wrapping_mul(0xBF58_476D_1CE4_E5B9)
        .rotate_left(32)
}

/// What a decision's view adds to its request: the facts that are the
/// deciding host's own, never the frame's.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Viewed<'a> {
    /// The decision's instant in the rows' clock.
    pub(crate) now_micros: u64,
    /// What is left of the request's budget at the decision.
    pub(crate) budget_micros: u32,
    /// The candidates' affinity column, as the view's build filled it.
    pub(crate) affinity: &'a [AffinityFacts],
    /// The rows a veto of the asking gateway masks, by position, where the
    /// candidates are a publication's own rows; empty where they were
    /// written for the decision and carry their own stalled bits.
    pub(crate) stalled: &'a [u32],
    /// The rows of `stalled` a hold of the asking gateway stands on.
    pub(crate) held: &'a [u32],
    /// What the leases on file book on each row, by position, where the
    /// candidates are a publication's own rows; `None` where they were
    /// written for the decision and carry their counts.
    pub(crate) in_flight: Option<crate::LentCounts<'a>>,
    /// How many of the request's block hashes the deciding generation
    /// reads.
    pub(crate) max_block_hashes: usize,
    /// The pins of record the host lends the decision ([`PinSource`]);
    /// `None` where it keeps none.
    pub(crate) pins: Option<&'a dyn PinSource>,
    /// The candidates' signal cells, as the view handed them: `None` for
    /// a profile that reads no signal, and for a pair, whose cells ride
    /// its pools.
    pub(crate) signal_cells: Option<&'a crate::SignalCells>,
}

/// Tells the engine what was said of a lease: one function of the record
/// found on file and the report, whoever calls it. `true` where a stage's
/// call panicked and was held, for the first of a streak: its host may
/// write a line for that one.
///
/// The engine told is the one of the lease's own generation, where
/// `world` holds that generation: a later generation's stages never saw
/// the issue and hear nothing of it. The caller loads `world` for this
/// word, or once for the words it tells in a row: behind a later
/// generation's start, a word told under such a load still reaches the
/// lease's own engine. A dispatch also feeds the counts of sends of the
/// generation `world` holds, whichever generation issued the lease: a
/// send is the host's, whatever document decided it
/// ([`Engine::dispatched_sent`]). A count that does not know the rank
/// the report names enters it only where `world` lists the worker at
/// that rank ([`World::lists_rank`]), so a late report enters no rank
/// behind a removal. A rank a count knows takes the send unasked: a
/// removal forgets the worker's ranks, and a stale rank of a listed
/// worker may take a late send that no row reads.
/// A dispatch is told on the target its report names by index,
/// as the record holds the pick, and the lease told names that index:
/// the record was read before the door wrote it. A report that names no
/// target of the pick tells nothing. A pair's decode side that stood
/// chosen is told its issue first ([`chosen_dispatched`]): no call
/// issued it, and it goes out as the first pick's.
///
/// A pair's lease is told by side: a dispatch on the side its frame names,
/// with that side's lease and on that side's worker, and an end once a
/// side the lease held live, the frame's side or with none every side
/// ([`Sides::ending`]).
///
/// Its caller is whoever applies the report, under the turn of the leases'
/// file: what it calls is bounded and takes no door of the leases. It
/// enters as the decider whose file the lease is on, on whichever thread
/// applies.
///
/// A stage's call that panics is held here, whoever applies: a decider
/// inside a decision, as it takes up or leaves its work or behind the
/// give-up of an answer its stream refused, a session, the gateways'
/// sweep, a host's own thread. The lease's event stands on file as
/// applied and the fault is counted on the store
/// ([`MemoryStore::lifecycle_faults`]). It costs no request its decision
/// and ends no thread.
pub(crate) fn told(
    store: &MemoryStore,
    world: &World,
    found: &Sides,
    said: Said,
    by: Option<Decider>,
) -> bool {
    // The lease's own decider acts while its event is told, whichever
    // thread applies the report.
    let _acting = by.map(Decider::enter);
    let Err(payload) = catch_unwind(AssertUnwindSafe(|| tell(store, world, found, said))) else {
        store.call_returned();
        return false;
    };
    // A payload's own drop can panic: the engine's helper frees it under a
    // catch of its own.
    release(payload);
    store.call_faulted()
}

/// The engine that hears a lease's words under `spec`: the one of the
/// generation the lease was issued under, where that generation still
/// stands. A later generation's engine heard no issue of the lease and
/// hears nothing of it: no dispatch, no end, and no issue of a side a call
/// decides later.
#[inline]
fn issuing<'a>(spec: &'a AppliedSpec, lease: &Lease) -> Option<&'a Engine> {
    (spec.generation() == lease.generation).then(|| spec.engine())
}

/// What [`told`] tells, under its catch.
fn tell(store: &MemoryStore, world: &World, found: &Sides, said: Said) {
    let spec = world.spec();
    let issuing = || issuing(spec, &found.record.lease);
    match said {
        Said::Dispatched(dispatch, at) => {
            // The side the frame names: its lease, and its worker at
            // the index the frame names.
            let Some((lease, stood)) = found.side(dispatch.side) else {
                return;
            };
            let named = found.target_of(dispatch.side, dispatch.target_index);
            let Some((id, dp_rank)) = named else {
                return;
            };
            let target = Target {
                id,
                dp_rank,
                ..Target::default()
            };
            // The record was read before the door wrote the dispatched
            // index: the lease told carries it, for a stage that reads
            // the index off the lease.
            let lease = Lease {
                target_index: dispatch.target_index,
                ..lease
            };
            spec.engine().dispatched_sent(
                &lease,
                &target,
                dispatch.side,
                dispatch.since_decision_micros,
                || world.lists_rank(id, dp_rank),
            );
            // The table hears first and the pin of record reads it.
            // The stages hear last, so nothing is held across their
            // loop: a value kept for the pin behind it cost every
            // lease a store and a reload a profile. A stage's call
            // that unwinds there leaves the pin of record written,
            // beside the table's entry.
            let engine = issuing();
            if let Some(engine) = engine {
                engine.dispatched_table(&lease, &target, dispatch.side);
            }
            pinned(store, &lease, stood, engine, id, at);
            if let Some(engine) = engine {
                if stood.is_chosen() {
                    // A decode side no call issued goes out as the first
                    // pick's, and this dispatch books it: the stages hear
                    // its issue first, as they hear every side's.
                    chosen_dispatched(world, engine, found, &lease, &target, dispatch, at);
                } else {
                    engine.dispatched_stages(
                        &lease,
                        &target,
                        dispatch.side,
                        dispatch.since_decision_micros,
                    );
                }
            }
        }
        Said::Completed(outcome) => {
            if let Some(engine) = issuing() {
                let side = Some(outcome.side);
                found.each_ending(side, |lease| engine.on_completed(lease, &outcome));
            }
        }
        Said::Released(side, reason) => {
            if let Some(engine) = issuing() {
                found.each_ending(side, |lease| engine.on_released(lease, reason));
            }
        }
    }
}

/// Tells `engine`'s stages the issue of a pair's decode side the store
/// files behind the pair's first decision: at its decode call, or at the
/// dispatch that sends it with no call ([`chosen_dispatched`]).
///
/// One rule dates an issue: the instant the store files the side, on the
/// store's own clock. A side filed in its decision bears that decision's
/// instant ([`record_of`]). A side filed later is told here under its
/// lease dated `at`, the instant of that filing as its caller states it.
/// In process that is the instant the host hands its call, where the
/// door reads no clock. No served controller files a side later yet: a
/// frame asks for no later decode, and a handle over a link refuses the
/// decode call. On the day one does, it is the decider's read at the
/// call, or the read of the dispatch's frame. The lease on file keeps
/// the instant of the pair's first decision for its other words.
///
/// A stage that dates its bookings by its issues reads a date far behind
/// its newest as a clock that went back, and ends what it passed. So a
/// later issue bears neither the instant of that first decision nor the
/// date a gateway gives a dispatch.
fn issued_later(world: &World, engine: &Engine, lease: &Lease, decision: &Decision, at: Instant) {
    let issued = Lease {
        issued_micros: world.micros(at),
        ..*lease
    };
    engine.on_decision_issued(&issued, decision);
}

/// Tells `engine`'s stages the dispatch of a pair's decode side that no
/// call issued, as the frame read at `at` names it: the side's issue
/// first ([`issued_later`]), then the dispatch's own word. The side goes
/// out as the first pick's and the store books it at this dispatch, so
/// the dispatch is the side's issue too: a stage that books at an issue
/// books the worker that serves, as it does at a decode call's issue.
///
/// The issue names the pair's pick as the lease's record keeps it: every
/// target's two workers and their ranks, with no trace beside the
/// generation. The record keeps no position, and each target bears the
/// default one, 0, as the target of a dispatch's word does: a stage reads
/// the worker by its id. The issue's lease names the target the dispatch
/// took and that target's worker as the one it books. The dispatch's
/// word then keeps the lease's own instant and names the same worker as
/// the one the issue booked: no stage books it twice.
///
/// Out of line: no other dispatch builds a pick, and the teller's frame
/// stays as short as it was.
#[inline(never)]
fn chosen_dispatched(
    world: &World,
    engine: &Engine,
    found: &Sides,
    lease: &Lease,
    target: &Target,
    dispatch: Dispatch,
    at: Instant,
) {
    let named = |index| {
        let prefill = found.target_of(Side::Prefill, index)?;
        Some((prefill, found.target_of(Side::Decode, index)?))
    };
    let mut targets = Targets::new();
    for ((id, dp_rank), (decode_id, decode_dp_rank)) in (0..=u8::MAX).map_while(named) {
        targets.push(Target {
            id,
            dp_rank,
            pair: Some(Pair {
                decode_id,
                decode_dp_rank,
                ..Pair::default()
            }),
            ..Target::default()
        });
    }
    let decision = Decision::Pick(Pick {
        targets,
        trace: Trace {
            generation: lease.generation,
            ..Trace::default()
        },
        ..Pick::default()
    });
    let sent = Lease {
        booked: target.id,
        ..*lease
    };
    issued_later(world, engine, &sent, &decision, at);
    engine.dispatched_stages(&sent, target, dispatch.side, dispatch.since_decision_micros);
}

/// Writes a keyed lease's pin of record at its dispatch (design 5.1, the
/// sessions row: written at the commit, never at selection). The pin names
/// the worker the key's rule prefers once the engine was told, as its
/// table holds it: a dispatch on the failover leaves the preferred worker
/// pinned. Where no engine was told it names the dispatched worker. A
/// lease no sessions rule governed pins nothing.
///
/// The pin ends where the engine's table forgets the key, and never under
/// a rule without a TTL. Where the table holds no such key, or no engine
/// was told, the lease's own rule ends it: the TTL its decision was made
/// under, as its issued state carries it, counted from the lease's
/// issue as the table counted it then. So it does where the dispatch
/// itself made the table's entry anew and no decision has read it: that
/// entry holds no instant yet. An end the lease's rule counts never moves
/// the pin's end earlier: a lease dispatched behind a newer one of its
/// key, or behind a dispatch whose table ended the pin later, leaves the
/// later end standing. Either end stands on the decision clock: a
/// dispatch converts no clock. The table's lock is let go before the
/// store's pin is written: no lock is held across the two. `stood` is
/// the dispatched side's state as it stood before this dispatch.
fn pinned(
    store: &MemoryStore,
    lease: &Lease,
    stood: LeaseState,
    engine: Option<&Engine>,
    dispatched: WorkerId,
    at: Instant,
) {
    if lease.session == Session::NONE || lease.session_verdict == SessionVerdict::None {
        return;
    }
    let held = engine.and_then(|engine| engine.sessions().held(lease.session));
    let preferred = held
        .map(|(pins, _)| pins[0])
        .filter(|pin| *pin != WorkerId(0));
    let (key, worker) = (lease.session.get(), preferred.unwrap_or(dispatched));
    match held {
        // The table's word is the key's rule as it stands: it moves the
        // pin's end either way.
        Some((_, Forgets::At(forgotten))) => {
            SessionsStore::pin(store, key, worker, at, Some(forgotten));
        }
        Some((_, Forgets::Never)) => SessionsStore::pin(store, key, worker, at, None),
        // The lease's own rule is the word of a decision that may be older
        // than the pin's last writer: it never moves the end earlier.
        Some((_, Forgets::Unread)) | None => {
            pinned_by_rule(store, key, worker, at, lease.issued_micros, stood);
        }
    }
}

/// The pin of a lease whose own rule ends it, [`pinned`]'s last arm. Few
/// leases come here, so it stands out of the path every dispatch takes,
/// and the store's door is forced into it: the arm is one call.
/// The rule's word lives in a side's issued state alone, and the door
/// tells a side's dispatch once, with the side as it stood issued. A side
/// past that state would read as a rule with no TTL: it pins nothing by
/// its lease's rule.
#[cold]
#[inline(never)]
fn pinned_by_rule(
    store: &MemoryStore,
    key: u64,
    worker: WorkerId,
    at: Instant,
    issued_micros: u64,
    stood: LeaseState,
) {
    if !stood.is_issued() {
        return;
    }
    let ends = stood
        .session_ttl_micros()
        .map(|ttl| issued_micros.saturating_add(ttl));
    SessionsStore::pin_no_earlier(store, key, worker, at, ends);
}

/// A completion, said whole: the lease's word goes where `hand` takes it, a
/// session's hand at the file or the door's report, and the gateway's
/// completion is written on the served worker for the health precedence,
/// whether or not the lease is on file. A served controller's handler of a
/// `Completed` frame and a handle over an in-process door both call here, so
/// what one means stands once.
pub(crate) fn completed(
    store: &MemoryStore,
    gateway: GatewayId,
    outcome: &Outcome,
    now: Instant,
    hand: impl FnOnce(Said),
) {
    hand(Said::Completed(*outcome));
    // The sink writes for a worker it holds a row for and for no other.
    HealthSink::completed(store, gateway, outcome.served, outcome.class, now);
}

/// The counter at which a term's leader is told to leave: three quarters of
/// the ids one term mints. A leader that leaves there takes its next term
/// with the counter at its start, so no term reaches the counter's end; the
/// quarter left is what the term still mints between the word and its
/// fence's close.
pub(crate) const TERM_MARK: u32 = 3 << 30;

/// Whether `id` is at or past its term's mark ([`TERM_MARK`]): the counter
/// is the id's low half, as the contract lays an id out. At or past, and
/// not the mark's own id alone: the mint steps over an id on file, and a
/// term's ids need not all pass through one decider.
pub(crate) fn past_mark(id: LeaseId) -> bool {
    (id.0 & u64::from(u32::MAX)) >= u64::from(TERM_MARK)
}

/// The decider a model's decisions run on among `deciders` of them: the
/// model is the key, so what a model's decisions write has one writer.
pub(crate) fn decider_of(model: &str, deciders: usize) -> usize {
    let deciders = u64::try_from(deciders.max(1)).unwrap_or(1);
    usize::try_from(fingerprint(model.as_bytes()) % deciders).unwrap_or(0)
}

/// A request and its constraints as the host holds them. A door reads
/// these facts in process. A link carries them to the controller.
#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    /// The request's id as the host states it: its lease keeps the id's
    /// fingerprint.
    pub id: &'a [u8],
    /// The request as the engine reads it. The deciding view supplies the
    /// clock value, affinity, named rows, lease counts and the key's count
    /// under a rule with an in-flight cap. A budget of 0 leaves a decision
    /// in process unbounded. Over a link it says the request's time is spent.
    pub facts: RequestFacts<'a>,
    /// What the host alone knows. The door writes its standing vetoes and
    /// the pins of record over the host's.
    pub constraints: Constraints<'a>,
}

impl Asked for Request<'_> {
    type Lists = ();

    fn model(&self) -> &str {
        self.facts.model
    }

    fn phase(&self) -> Phase {
        self.facts.phase
    }

    fn id(&self) -> &[u8] {
        self.id
    }

    fn wires(&self) -> Option<WireModes> {
        let stated = self.constraints.wire_modes;
        (stated != WireModes::NONE).then_some(stated)
    }

    fn subset(&self) -> Option<u32> {
        self.constraints.subset_hint
    }

    fn demands(&self) -> Demands {
        self.constraints.demands
    }

    fn decode_later(&self) -> bool {
        self.constraints.decode_later
    }

    fn hashes(&self) -> usize {
        self.facts.block_hashes.len()
    }

    fn hashes_read(&self, spec: &AppliedSpec) -> usize {
        spec.block_hashes_read(self.facts.hash_kind, self.facts.block_size)
    }

    fn block_hashes(&self) -> &[u64] {
        self.facts.block_hashes
    }

    fn cache_namespace(&self) -> (u64, HashKind) {
        (self.facts.namespace_seed, self.facts.hash_kind)
    }

    fn seed(&self) -> u64 {
        self.facts.seed
    }

    fn read<'n, R>(
        &self,
        (): &mut (),
        viewed: Viewed<'_>,
        name: impl FnOnce(&RequestFacts<'_>, &Constraints<'_>) -> Option<NamedAffinity<'n>>,
        decide: impl FnOnce(&RequestFacts<'_>, &Constraints<'_>) -> R,
    ) -> R {
        let hashes = self.facts.block_hashes;
        let mut facts = RequestFacts {
            // The first hashes, as deep as the generation reads.
            block_hashes: hashes.get(..viewed.max_block_hashes).unwrap_or(hashes),
            budget_micros: viewed.budget_micros,
            now_micros: viewed.now_micros,
            affinity: viewed.affinity,
            // Written below, by the naming's answer.
            named: None,
            lent_in_flight: viewed.in_flight.as_ref(),
            // The door decides over the store's rows: a list a host lent
            // names rows of its own, and its word that it counted stands
            // on rows the door does not read. The door's read of the key
            // is the one source of its count.
            lent_session_in_flight: None,
            session_counted: false,
            // Cells a caller handed are by its own list's positions too:
            // read by the store's, an in-range read would answer another
            // row's value. The door reads its own view's cells.
            signal_cells: viewed.signal_cells,
            ..self.facts
        };
        let constraints = Constraints {
            stalled: viewed.stalled,
            held: viewed.held,
            pins: viewed.pins,
            ..self.constraints
        };
        facts.named = name(&facts, &constraints);
        decide(&facts, &constraints)
    }
}

/// A request its host makes from the world the door decides under
/// ([`Door::decide_under`]). What the host reads of the applied document to
/// state the request, the sources of its routing key for one
/// ([`AppliedSpec::key_sources_for`]), is then of the same load as the
/// engine and the rows.
pub trait Maker {
    /// The model of the request. The door takes its seat by it, ahead of
    /// the load: the request [`Maker::make`] makes names this model.
    fn model(&self) -> &str;
    /// The request, made from the world its decision is made under. It is
    /// made inside the door's hold of its seat: a maker that waits holds
    /// up the seat's other requests, and one that asks the door again can
    /// wait on the seat it holds.
    fn make<'w>(&'w self, world: &'w World) -> Request<'w>;
}

/// A request as the door's order takes it up under the world it loaded:
/// one its host stated ahead of the load ([`Request`]), or one its host
/// makes from that world ([`Maker`]).
trait Stated {
    /// The model the request is for. The door takes its seat by it, ahead
    /// of the load.
    fn model(&self) -> &str;
    /// The seat the request decides at, where it names one
    /// ([`Called`]). `None` for a request that decides at its caller's
    /// own seat: every request but a decode call.
    #[inline(always)]
    fn own_seat(&self) -> Option<usize> {
        None
    }
    /// Hands `order` the request as it stands under `world`.
    fn under<R>(&self, world: &World, order: impl FnOnce(&Request<'_>) -> R) -> R;
}

impl Stated for Request<'_> {
    fn model(&self) -> &str {
        self.facts.model
    }

    // Forced into the order's one call: a request stated ahead of the load
    // pays nothing for the form a host's maker takes.
    #[inline(always)]
    fn under<R>(&self, _: &World, order: impl FnOnce(&Request<'_>) -> R) -> R {
        order(self)
    }
}

/// A decode call's request with the seat of its lease's file. The call
/// decides at that seat, behind the decision that stands there: a caller
/// that takes up a seat's work applies what was handed to its file, so a
/// prefill's dispatch that waits there is applied ahead of the call's read.
/// From another seat the call would read a file that applies nothing while
/// its own caller decides, and a sent prefill would read as not sent.
struct Called<'r, 'q>(&'r Request<'q>, usize);

impl Stated for Called<'_, '_> {
    fn model(&self) -> &str {
        self.0.facts.model
    }

    fn own_seat(&self) -> Option<usize> {
        Some(self.1)
    }

    #[inline(always)]
    fn under<R>(&self, _: &World, order: impl FnOnce(&Request<'_>) -> R) -> R {
        order(self.0)
    }
}

/// A [`Maker`]'s request, as the door's order takes it up.
struct Made<'m, M>(&'m M);

impl<M: Maker> Stated for Made<'_, M> {
    fn model(&self) -> &str {
        self.0.model()
    }

    fn under<R>(&self, world: &World, order: impl FnOnce(&Request<'_>) -> R) -> R {
        let request = self.0.make(world);
        debug_assert_eq!(request.facts.model, self.0.model(), "the seat's model");
        order(&request)
    }
}

/// A decision as its host takes it.
#[derive(Debug)]
pub struct Decided {
    /// The engine's decision.
    pub decision: Decision,
    /// The lease issued for a pick, with the target index it books. For
    /// a pair this is the prefill side. `None` for a decision that
    /// picks no worker.
    pub lease: Option<Leased>,
}

impl Decided {
    /// Whether the lease is a pair's prefill side. The decode side's
    /// lease shares its id and dispatch span.
    #[must_use]
    pub fn pair(&self) -> bool {
        self.lease
            .is_some_and(|leased| leased.lease.side == Side::Prefill)
    }
}

/// A pick's lease as its host receives it, with its dispatch deadline.
#[derive(Clone, Copy, Debug)]
pub struct Leased {
    /// The lease issued for the pick.
    pub lease: Lease,
    /// When its dispatch span ends on the host's clock. The store ends
    /// leases whose dispatch span has passed and whose dispatch is still
    /// owed.
    pub due: Instant,
}

/// What one pass of a door's clock ended ([`Door::sweep`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Swept {
    /// The leases past their dispatch span, ended as abandoned.
    pub leases: usize,
    /// The pins of record past their end, gone from their table.
    pub pins: usize,
}

/// Why a door has no decision for a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refused {
    /// The request is a decode or encode side sent on its own. A pair is
    /// asked at its prefill, and that one request decides both sides.
    PairSide,
    /// The store filed no lease for the pick, at the second try too: its
    /// term mints no id, or another hand raised it under both tries.
    Unleased,
    /// The decision panicked. It cost this request alone: the next one
    /// decides on new buffers.
    Faulted,
    /// The store takes no decode side under the lease a decode call
    /// named, for the reason it carries.
    Decode(DecodeRefused),
}

/// A host's door to a decision in process: the order of a decision's
/// calls, run on the caller's thread. No frame is built and no thread is
/// hopped.
///
/// **Whose file.** The store holds a file of leases a decider
/// ([`MemoryStore::with_deciders`]), and a host in process has no
/// deciding thread. The door keeps a seat a file, and the seat follows
/// who owns what the decision reads.
///
/// Under a generation that holds no stage keeping state a decider, a
/// request decides at its caller's seat: the one the thread took last,
/// where no other thread holds it. A thread's first seat is its number
/// among the threads that have called a door, so threads start at seats
/// of their own. Where its seat is held the caller takes the next free
/// one, and keeps that. Where every seat is held it waits at its own,
/// behind the one decision that stands there. A host that lays as many
/// files as it has threads that decide parks none of them.
///
/// Under a generation that holds such a stage, a plugin's, every request
/// decides at its model's seat, by the rule the served controller keeps
/// ([`Engine::keeps_deciders`]). Every profile's stages hear every lease,
/// so the stage's state has one writer a model and hears a model's
/// leases whole, and a model's requests decide one at a time. The files
/// a host lays past those its models name gain such a generation
/// nothing.
///
/// At its seat a caller acts as that seat's decider ([`Decider::enter`])
/// and works at its file for the one decision, so the lease is filed
/// there. Over a store of one file callers decide one at a time.
///
/// **What a decision reads.** A lease is counted at its issue, on a
/// count every file shares, so a decision reads every lease filed before
/// it began. Decisions made at once, at different seats, each read the
/// counts without the lease the others are still filing: one a seat at
/// work, at most. No stage takes a hard bound from that read: a bound is
/// taken at a booking, as the store's reservation is. What a host says
/// of a lease is read once it is applied at the lease's seat: at once
/// where no caller decides there, and by that caller otherwise. No
/// thread watches a seat's file between decisions.
///
/// **A report from another thread.** A host says what became of a lease
/// through [`Door::report`], on whichever thread it stands, and that
/// thread is never held for a decision. The word goes into the ring of
/// the file the lease's id names. Where no caller decides at that file
/// it is applied at once. Where one does, that caller applies it: every
/// decision here applies what the ring holds as it takes up its work and
/// as it leaves it. A full ring takes no more until then: the door keeps
/// the word beside it, in the order said, and the caller hands it on as
/// it leaves. Whoever applies a word tells the engine that issued the
/// lease.
///
/// **The term.** A term gives the leases' ids their span, and in process
/// no election hands a term over. The door starts a store that stands
/// under no term at term 1, and raises the term at its mark, three
/// quarters of the ids a term mints: the leases on file keep their files
/// across the rise. A decision that a rise crosses is made once more
/// under the new term, so a rise costs no request its decision.
///
/// **The clock.** In process nothing runs on a clock of its own: the
/// served controller's sweep has no place here. The host's timer calls
/// [`Door::sweep`], which ends the leases past their dispatch span that
/// no issue reached, and lets the pins of record past their end go.
///
/// A store has one host at its files: a door, or the served controller's
/// deciders, never both.
pub struct Door {
    store: Arc<MemoryStore>,
    /// A seat a file of the store: its holder decides as that file's
    /// decider.
    seats: Box<[Seat]>,
    /// The last generation's seat flags, checked again against the world
    /// loaded at the seat.
    seat_flags: AtomicU8,
    /// What the decisions count, a part a decider.
    parts: Box<[Part]>,
    /// The term the door files its leases under: the store's, as the door
    /// last read or raised it.
    term: AtomicU64,
    /// The decisions that panicked.
    faults: AtomicU64,
    /// What the reports left beside a full ring, a list a file.
    kept: Box<[Kept]>,
    /// The rows named again for a decision's shadow, a buffer a seat
    /// ([`OwnNamed`]).
    own_named: Box<[OwnNamed]>,
}

/// The buffer a seat's holder names a shadow's own rows on, where the
/// shadow profile is named rows of its own ([`ShadowRows::Own`]). It
/// stands beside its seat, on lines of its own, and not in it: a seat
/// keeps its size, and no decision reaches the buffer but one whose
/// shadow is named such rows. Its holder is the seat's, so its lock is
/// never waited for. It grows at the seat's first such shadow.
#[derive(Default)]
#[repr(align(128))]
struct OwnNamed {
    named: Mutex<Named>,
}

/// What the door keeps for one file while its ring is full
/// ([`Door::report`]).
#[derive(Default)]
struct Kept {
    /// The words kept, the words a caller hands on, and each report that
    /// looks at the ring under the list's lock. At zero a report goes
    /// straight to the ring, and the door's pass ends the file's due
    /// leases ([`Door::sweep`]).
    waits: AtomicUsize,
    /// The words kept, in the order said.
    words: Mutex<Vec<Word>>,
}

/// What a host said of a lease: the gateway, the lease and the word.
type Word = (GatewayId, LeaseId, Said);

/// Counts words off a file's count as their holder is through with them,
/// an unwind included: the batch a caller hands on, or the one word of a
/// report that looks at the ring.
struct Handed<'a>(&'a AtomicUsize, usize);

impl Drop for Handed<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(self.1, Ordering::SeqCst);
    }
}

/// A seat at a file of the store, on lines of its own: a thread that
/// decides at one seat writes no line the thread at the next one reads.
#[derive(Default)]
#[repr(align(128))]
struct Seat {
    /// The seat's state across decisions, including its sample count.
    /// Only decisions take this lock.
    slot: Mutex<DeciderSlot>,
    /// The seat's recent decisions. A read or a decision that keeps a
    /// record takes this lock.
    samples: Mutex<Samples>,
}

/// What a seat's holder keeps under the decision lock, including the
/// count toward its next sample.
#[derive(Default)]
struct DeciderSlot {
    buffers: Buffers<()>,
    /// The holder's place at the seat's file, taken at the seat's first
    /// decision.
    at_file: Option<AtWork>,
    sample_count: SampleCount,
}

thread_local! {
    /// The seat the calling thread took last at a door. Until it moves it
    /// is the thread's number among the threads that have called one, so
    /// threads start at seats of their own.
    static SEAT: Cell<usize> = Cell::new(CALLERS.fetch_add(1, Ordering::Relaxed));
}

#[cfg(test)]
thread_local! {
    static ON_SAMPLE_READ: Cell<Option<Box<dyn FnOnce()>>> = const { Cell::new(None) };
    /// Called once between a live decision and its shadow.
    static ON_SHADOW: Cell<Option<Box<dyn FnOnce()>>> = const { Cell::new(None) };
}

/// The threads that have called a door so far.
static CALLERS: AtomicUsize = AtomicUsize::new(0);

/// The decider a door decides as at a store that names none: such a store
/// has one file ([`MemoryStore::new`]).
const UNNAMED: Decider = Decider::new(0);

/// The door's word to the engine of a report applied to a lease
/// ([`told`]), under the name the lease's decision ran under: its file's
/// decider, and [`UNNAMED`] at a store that names none. A stage's call
/// that panics there is counted on the store for the host to read
/// ([`MemoryStore::lifecycle_faults`]), and no line is written: a host in
/// process writes its own.
fn quiet(store: &MemoryStore, world: &World, found: &Sides, said: Said, by: Option<Decider>) {
    // A stage that keeps a decider's state hears a lease's events under
    // the name it heard the issue under, whichever thread applies them.
    told(store, world, found, said, Some(by.unwrap_or(UNNAMED)));
}

impl Door {
    /// A door over `store`, with a seat for each of its files. The store
    /// is given the door's word to the engine: whoever applies a report
    /// to a lease tells the engine that issued it, and a dispatch feeds
    /// the standing generation's counts of sends for a row its world
    /// still lists. A store that names no decider has one file: the door
    /// decides there, and tells what is said of its leases, as decider 0.
    #[must_use]
    pub fn new(store: Arc<MemoryStore>) -> Self {
        let deciders = store.deciders();
        store.tell_through(quiet);
        if LeasesStore::term(&*store) == 0 {
            LeasesStore::set_term(&*store, 1);
        }
        let term = AtomicU64::new(LeasesStore::term(&*store));
        Self {
            seats: (0..deciders).map(|_| Seat::default()).collect(),
            seat_flags: AtomicU8::new(0),
            parts: (0..deciders).map(|_| Part::default()).collect(),
            kept: (0..deciders).map(|_| Kept::default()).collect(),
            own_named: (0..deciders).map(|_| OwnNamed::default()).collect(),
            store,
            term,
            faults: AtomicU64::new(0),
        }
    }

    /// The store the door decides over, for what a host reads and
    /// publishes there. What became of a lease it says through
    /// [`Door::report`], and its clock runs through [`Door::sweep`] alone:
    /// the store's own pass ([`LeasesStore::expire`]) asks no host, and
    /// would end a lease whose dispatch the door still keeps.
    #[must_use]
    pub fn store(&self) -> &Arc<MemoryStore> {
        &self.store
    }

    /// How many decisions panicked, and how many shadows did beside a
    /// decision that stood: a shadow's panic refuses no request, and its
    /// live answer is issued. A decision made again behind a lease the
    /// store refused counts each panic of its shadow.
    #[must_use]
    pub fn faults(&self) -> u64 {
        self.faults.load(Ordering::Relaxed)
    }

    /// Candidates the wires of their decisions' gateways took from their
    /// views, over every decision made here.
    #[must_use]
    pub fn undialed(&self) -> u64 {
        summed(&self.parts, |part| &part.undialed)
    }

    /// Requests that carried more block hashes than their generation
    /// reads of them, over every decision made here.
    #[must_use]
    pub fn hashes_cut(&self) -> u64 {
        summed(&self.parts, |part| &part.hashes_cut)
    }

    /// How many decisions made here had a shadow that said `kind`
    /// beside them ([`ShadowKind`]), for the binding at position
    /// `binding` of its generation. A shadow is counted behind its
    /// decision's commit, and a decision the door refused counts none.
    /// The count is a position's: a later generation's binding at the
    /// same position counts on, and a binding past the sixteenth
    /// counts with the sixteenth.
    #[must_use]
    pub fn shadowed(&self, binding: u32, kind: ShadowKind) -> u64 {
        let parts = self.parts.iter();
        parts.map(|part| part.shadowed_so_far(binding, kind)).sum()
    }

    /// One decision for `gateway`'s `request` at `now`, on the calling
    /// thread: one load of the published rows, the engine's decision and,
    /// for a pick, its lease, filed at the issue with the engine told.
    ///
    /// # Errors
    ///
    /// [`Refused`]: no decision is made for the request.
    pub fn decide(
        &self,
        gateway: GatewayId,
        request: &Request<'_>,
        now: Instant,
    ) -> Result<Decided, Refused> {
        self.deciding(gateway, request, now, Whole(self))
    }

    /// [`Door::decide`] for a request its host makes from the world the
    /// door decides under. `maker` is handed that world behind the door's
    /// load ([`Maker::make`]), once for each run of the order: a load an
    /// apply moved to another decider is dropped before it asks. It is
    /// asked again where the order runs a second time behind a lease the
    /// store refused.
    ///
    /// A maker that panics is a decision that faulted.
    ///
    /// # Errors
    ///
    /// [`Refused`]: no decision is made for the request.
    pub fn decide_under(
        &self,
        gateway: GatewayId,
        maker: &impl Maker,
        now: Instant,
    ) -> Result<Decided, Refused> {
        let (decided, _) = self.decide_under_stated(gateway, maker, now)?;
        Ok(decided)
    }

    /// [`Door::decide_under`], with what the made request said of its
    /// decode: whether a pair's decode side was filed chosen
    /// ([`Asked::decode_later`]). The handle holds the side by it.
    pub(crate) fn decide_under_stated(
        &self,
        gateway: GatewayId,
        maker: &impl Maker,
        now: Instant,
    ) -> Result<(Decided, bool), Refused> {
        let made = Made(maker);
        let mut later = false;
        let order = WholeStated {
            door: self,
            later: &mut later,
        };
        let decided = self.deciding(gateway, &made, now, order)?;
        Ok((decided, later))
    }

    /// Decides an `epd` request's pair and places its encode items in
    /// `placed`. A pair pick books the prefill and decode. For other
    /// plans this answers as [`Door::decide`] and clears `placed`.
    /// A shed leaves `placed` empty.
    ///
    /// # Errors
    ///
    /// [`Refused`]: no decision is made for the request. `placed` is cleared.
    pub fn decide_encode_pair(
        &self,
        gateway: GatewayId,
        request: &Request<'_>,
        items: EncodeItems<'_>,
        placed: &mut EncodePlaced,
        now: Instant,
    ) -> Result<Decided, Refused> {
        let order = WithEncode {
            door: self,
            items,
            placed: &mut *placed,
        };
        let decided = self.deciding(gateway, request, now, order);
        if decided.is_err() {
            placed.clear();
        }
        decided
    }

    /// Decides the decode side of the pair's lease `lease`, for the gateway
    /// it was issued to, while its prefill is dispatched and held and its
    /// decode side is chosen. The decode profile runs over the decodes that
    /// pair with that prefill in the rows of this instant. A pick is issued
    /// under the same lease id, and the answer's lease is the decode side's.
    /// A shed files nothing: the side stays chosen, and the host keeps the
    /// decode the first pick named.
    ///
    /// The call decides at the seat of its lease's file, behind the
    /// decision that stands there: a prefill's dispatch handed to that file
    /// is applied ahead of the call's read. Under an engine that keeps its
    /// deciders the call decides at its model's seat, which is the lease's
    /// unless a reload turned the keeping on between the two decisions.
    /// Behind such a reload the call first stands at its lease's seat for
    /// no decision, so what waits at that file is applied all the same.
    ///
    /// # Errors
    ///
    /// [`Refused::Decode`] says why the lease takes no decode side, and
    /// [`Refused::Faulted`] that the decision panicked. Either files nothing.
    pub fn decide_decode(
        &self,
        gateway: GatewayId,
        request: &Request<'_>,
        lease: LeaseId,
        now: Instant,
    ) -> Result<Decided, Refused> {
        // The call mints no id: it spends no term and raises none.
        let term = self.term.load(Ordering::Relaxed);
        let called = Called(request, self.store.file_of(lease));
        self.at_a_seat(term, gateway, &called, now, DecodeCall(self, lease))
    }

    /// Finds a sampled decision by gateway and request id. A read holds
    /// only the ring lock at each seat, for one scan of up to 1,024 records.
    #[must_use]
    pub fn explain(&self, gateway: GatewayId, request_id: &[u8]) -> Option<DecisionRecord> {
        let key = fingerprint(request_id);
        self.seats
            .iter()
            .filter_map(|seat| {
                let samples = seat.samples.lock();
                #[cfg(test)]
                if let Some(reading) = ON_SAMPLE_READ.take() {
                    reading();
                }
                samples.find(0, gateway, key)
            })
            .max_by_key(|record| record.at)
    }

    /// [`Door::explain`] with names from one read of the store's world.
    /// The record is copied and each ring lock released before that read.
    #[must_use]
    pub fn explained(&self, gateway: GatewayId, request_id: &[u8]) -> Option<Explained> {
        let record = self.explain(gateway, request_id)?;
        let names = self
            .store
            .deciding(|world| record.names(world.spec().engine()));
        Some(Explained { record, names })
    }

    /// What `gateway` says of `lease`: a dispatch, a completion or a
    /// release, on whichever thread the host says it. The thread is never
    /// held for a decision: it stands at most behind one hold of the
    /// file's turn, an apply of what the ring holds or a lease's filing.
    /// The word goes into the ring of the file the lease's id names: it
    /// is applied at once where no caller decides there, and by that
    /// caller otherwise. A full ring takes no more while a caller
    /// decides: the door keeps the word, and each later word for that
    /// file behind it, and the caller hands them on as it leaves. So what
    /// is said of one lease reaches its file in the order said.
    ///
    /// What the door keeps grows with what its host says while one
    /// decision stands: the host's requests in flight bound it.
    ///
    /// A completion said here ends the lease and is no health write: the
    /// gateway's completed dispatch on the worker that served is the
    /// host's to write ([`HealthSink::completed`]). A host that holds a
    /// handle over this door says one completion and the handle writes both
    /// ([`crate::handle::Handle::completed`]).
    pub fn report(&self, gateway: GatewayId, lease: LeaseId, said: Said) {
        let at = self.store.file_of(lease);
        let hand = Hand::new(gateway);
        if self.kept[at].waits.load(Ordering::SeqCst) != 0
            || !self.store.try_hand(at, &hand, lease, said)
        {
            self.keep(at, &hand, lease, said);
        }
    }

    /// One pass of the door's clock at `now`, the call of its host's
    /// timer: about once a second is ample. A lease whose host named no
    /// dispatch by the end of its span is ended as abandoned and its
    /// engine is told, at each file where no caller decides: a caller
    /// that decides ends its own file's due leases at its issue, and one
    /// that takes up its work meanwhile waits for the pass at its file.
    /// A file is passed too while words the door keeps for it wait
    /// ([`Door::report`]): what was said of a lease is applied before the
    /// end of its span is read, and the next pass ends what is still due.
    /// The pins of record past their end leave their table once a pass
    /// of the standing rules is due. A host that never calls keeps such a
    /// lease on file, counted on its worker, until the next issue at its
    /// file, and such a pin for the life of the store.
    pub fn sweep(&self, now: Instant) -> Swept {
        let store = &*self.store;
        // A kept word counts until its caller has handed it on.
        let settled = |at: usize| self.kept[at].waits.load(Ordering::SeqCst) == 0;
        Swept {
            pins: store.sweep_pins(now),
            leases: store.expire_where(now, settled).len(),
        }
    }

    /// Runs the order again if the store refused its lease under the
    /// term the door held.
    fn deciding(
        &self,
        gateway: GatewayId,
        request: &impl Stated,
        now: Instant,
        mut order: impl Orders,
    ) -> Result<Decided, Refused> {
        match self.once(gateway, request, now, &mut order) {
            Err(Refused::Unleased) => self.once(gateway, request, now, order),
            decided => decided,
        }
    }

    /// One run of the order for `request`, under the term the door holds.
    /// Behind a lease at the term's mark the term is raised. Behind a
    /// lease the store refused it is read again, and raised where it was
    /// spent.
    fn once(
        &self,
        gateway: GatewayId,
        request: &impl Stated,
        now: Instant,
        order: impl Orders,
    ) -> Result<Decided, Refused> {
        let term = self.term.load(Ordering::Relaxed);
        let outcome = self.at_a_seat(term, gateway, request, now, order);
        match &outcome {
            // An id at or past the term's mark: the door raises the term,
            // as an election's hand-over does for a served controller.
            Ok(Decided {
                lease: Some(leased),
                ..
            }) if past_mark(leased.lease.id) => self.rise(term),
            Err(Refused::Unleased) => self.rise(term),
            _ => {}
        }
        outcome
    }

    /// Takes the seat the last generation called for and checks it
    /// against the world loaded there. If an apply changed which seat
    /// the request belongs at, it takes that seat and loads again. The
    /// same world governs the decision, its shadow and its sampling. A call
    /// that names its lease's seat and decides at its model's applies what
    /// waits at the lease's file first ([`Door::stand_at`]).
    fn at_a_seat(
        &self,
        term: u64,
        gateway: GatewayId,
        request: &impl Stated,
        now: Instant,
        mut order: impl Orders,
    ) -> Result<Decided, Refused> {
        let store = &*self.store;
        let mut seen = self.seat_flags.load(Ordering::Relaxed);
        loop {
            let (at, mut slot) = if seen & KEEPS_DECIDERS != 0 {
                // A decode call's lease stands at another seat's file
                // where a reload turned the keeping on between the pair's
                // two decisions: the call reads that file from its
                // model's seat, behind what was handed there.
                let away = |own: &usize| *own != decider_of(request.model(), self.seats.len());
                if let Some(own) = request.own_seat().filter(away) {
                    self.stand_at(own);
                }
                self.home(request.model())
            } else if let Some(at) = request.own_seat() {
                (at, self.seats[at].slot.lock())
            } else {
                self.seat()
            };
            // The store lays at most one file for each byte-sized name.
            debug_assert!(at < Decider::COUNT);
            let _acting = Decider::new(at as u8).enter();
            let DeciderSlot {
                buffers,
                at_file,
                sample_count,
            } = &mut *slot;
            let at_file = at_file.get_or_insert_with(|| self.store.at_work());
            // The thread works at the seat's file for this decision: what
            // was handed is applied as it takes up the work.
            at_file.work();
            // One load of the world, for the seat's question and the
            // decision.
            let world = store.loaded();
            let engine = world.spec().engine();
            let flags = engine.seat_flags();
            if flags != seen {
                // Writing only on a change keeps this line shared by all
                // callers. Only a change of the kept bit moves the seat.
                self.seat_flags.store(flags, Ordering::Relaxed);
                if (flags ^ seen) & KEEPS_DECIDERS != 0 {
                    drop(world);
                    self.leave(at, at_file);
                    seen = flags;
                    continue;
                }
            }
            let pins = StorePins { store, now };
            // The request is taken up under this load: one its host makes
            // is made here, of the world its decision and its sample read.
            let taken_up = |request: &Request<'_>| {
                let taken = Taken {
                    request,
                    gateway,
                    now,
                    // The host's own budget, as the engine is told it.
                    left: match request.facts.budget_micros {
                        0 => Duration::MAX,
                        micros => Duration::from_micros(micros.into()),
                    },
                    pins: Some(&pins),
                    // No place is lent in process: no file's frames wait
                    // for a look. What is said of a lease is applied at
                    // once where no caller decides at its file, and a file
                    // whose caller decides is that caller's, as a
                    // decider's at work is.
                    place: None,
                };
                // One test for a generation that neither shadows nor
                // samples: its order is the plain one, which stands last.
                if flags & (SAMPLES_DECISIONS | SHADOWS) != 0 {
                    // The request's shadow is asked for first, by one
                    // lookup of its binding. A request whose binding
                    // names none is ordered as a generation that shadows
                    // nowhere orders it.
                    let due = match flags & SHADOWS {
                        0 => None,
                        _ => engine.shadow_for(request.facts.model, request.facts.phase),
                    };
                    // Each order's answer is built in its own place:
                    // two that met in one would cost a copy of it.
                    let Some((binding, rows)) = due else {
                        let decided = order.plain(&world, at, term, &taken, buffers);
                        if flags & SAMPLES_DECISIONS != 0 {
                            self.sampled(at, engine, &taken, &decided, None, sample_count);
                        }
                        return decided;
                    };
                    let mut due = Due {
                        binding,
                        rows,
                        said: None,
                    };
                    let decided = order.shadowed(&world, at, term, &taken, buffers, &mut due);
                    if flags & SAMPLES_DECISIONS != 0 {
                        self.sampled(at, engine, &taken, &decided, due.said, sample_count);
                    }
                    return decided;
                }
                order.plain(&world, at, term, &taken, buffers)
            };
            let decided = catch_unwind(AssertUnwindSafe(|| request.under(&world, taken_up)));
            // The load ends with the decision.
            drop(world);
            self.leave(at, at_file);
            return decided.unwrap_or_else(|payload| {
                // A payload's own drop can panic: the engine's helper
                // frees it under a catch of its own. The next decision
                // reads nothing this one left.
                release(payload);
                *buffers = Buffers::default();
                self.faults.fetch_add(1, Ordering::Relaxed);
                Err(Refused::Faulted)
            });
        }
    }

    /// Leaves the work at seat `at`'s file: what was handed meanwhile is
    /// applied, and what the door kept beside a full ring goes on behind
    /// it. Until it has, the door's pass ends no lease at the file.
    fn leave(&self, at: usize, at_file: &mut AtWork) {
        at_file.rest();
        if self.kept[at].waits.load(Ordering::SeqCst) != 0 {
            self.hand_kept(at);
        }
    }

    /// Stands at seat `at` for no decision, behind the decision that
    /// stands there: the thread takes up the work at the seat's file, which
    /// applies what was handed to it, and leaves. For a call that reads a
    /// lease on file there and decides at another seat.
    #[cold]
    fn stand_at(&self, at: usize) {
        let mut slot = self.seats[at].slot.lock();
        // The store lays at most one file for each byte-sized name.
        debug_assert!(at < Decider::COUNT);
        let _acting = Decider::new(at as u8).enter();
        let at_file = slot.at_file.get_or_insert_with(|| self.store.at_work());
        at_file.work();
        self.leave(at, at_file);
    }

    /// Keeps a made decision in the seat's samples where its turn to be
    /// sampled came, with what its shadow said.
    // Forced into each order's tail: the sampling stands in line there,
    // as it did where one order ran.
    #[inline(always)]
    fn sampled(
        &self,
        at: usize,
        engine: &Engine,
        taken: &Taken<'_, Request<'_>>,
        decided: &Result<Decided, Refused>,
        said: Option<Shadowed>,
        count: &mut SampleCount,
    ) {
        let Ok(answer) = decided else { return };
        if count.keeps(0, engine, &answer.decision) {
            let mut samples = self.seats[at].samples.lock();
            samples.push(0, engine, taken, &answer.decision, said);
        }
    }

    /// Takes the seat of `model`'s decider and waits for it: a model's
    /// requests there decide one at a time.
    fn home(&self, model: &str) -> (usize, MutexGuard<'_, DeciderSlot>) {
        let at = decider_of(model, self.seats.len());
        (at, self.seats[at].slot.lock())
    }

    /// Takes the seat the calling thread decides at: the one it took
    /// last, which until it moves is its number among the threads that
    /// have called a door. A number past this door's seats is counted
    /// round them once, and the thread keeps that seat.
    fn seat(&self) -> (usize, MutexGuard<'_, DeciderSlot>) {
        let seats = self.seats.len();
        // A thread at its end no longer has its own storage: it starts
        // at the first seat.
        let last = SEAT.try_with(Cell::get).unwrap_or(0);
        let own = if last < seats { last } else { self.round(last) };
        match self.seats[own].slot.try_lock() {
            Some(slot) => (own, slot),
            None => self.another(own),
        }
    }

    /// `last` counted round the door's seats and kept as the calling
    /// thread's seat: the division is paid once a thread, not at each
    /// decision.
    #[cold]
    fn round(&self, last: usize) -> usize {
        let own = last % self.seats.len();
        let _ = SEAT.try_with(|seat| seat.set(own));
        own
    }

    /// The next free seat past `own`, which another thread holds: the
    /// calling thread takes it and keeps it. Where no seat is free it
    /// waits at `own`, behind the decision that stands there.
    #[cold]
    fn another(&self, own: usize) -> (usize, MutexGuard<'_, DeciderSlot>) {
        let seats = self.seats.len();
        for step in 1..seats {
            let at = (own + step) % seats;
            if let Some(slot) = self.seats[at].slot.try_lock() {
                let _ = SEAT.try_with(|seat| seat.set(at));
                return (at, slot);
            }
        }
        (own, self.seats[own].slot.lock())
    }

    /// The term past `term`, where the store still stands under `term`;
    /// the store's own where another hand raised it.
    fn rise(&self, term: u64) {
        let store = &*self.store;
        if LeasesStore::term(store) == term {
            LeasesStore::set_term(store, term + 1);
        }
        self.term.store(LeasesStore::term(store), Ordering::Relaxed);
    }

    /// Keeps a word for the file at `at` behind what the door holds for
    /// it, where its ring is full or words wait already: under the list's
    /// lock, so the caller that hands the list on takes this word too.
    #[cold]
    fn keep(&self, at: usize, hand: &Hand, lease: LeaseId, said: Said) {
        let kept = &self.kept[at];
        let mut words = kept.words.lock();
        // The report counts itself and then reads who is at work. A
        // caller that leaves its work says so and then reads the count.
        // One of the two sees the other: a word kept here has a caller
        // that hands it on.
        let alone = kept.waits.fetch_add(1, Ordering::SeqCst) == 0;
        if alone {
            // The report's own count goes as its look ends, an unwind
            // included, unless its word is kept.
            let mut looks = Handed(&kept.waits, 1);
            if self.store.try_hand(at, hand, lease, said) {
                return;
            }
            looks.1 = 0;
        }
        words.push((hand.gateway(), lease, said));
    }

    /// Hands the file at `at` what the door kept for it, in the order
    /// said, until nothing waits. The caller holds the seat at `at` and
    /// has left its work: no thread is at work there, so each word is
    /// applied as it is handed and no hand waits.
    #[cold]
    fn hand_kept(&self, at: usize) {
        let kept = &self.kept[at];
        loop {
            let words = mem::take(&mut *kept.words.lock());
            if words.is_empty() {
                // A report that counted itself and kept nothing hands
                // its own word: no thread is at work at the file.
                return;
            }
            // The words count until they are handed: a word said
            // meanwhile is kept behind them.
            let _handed = Handed(&kept.waits, words.len());
            for (gateway, lease, said) in words {
                self.store.hand(&Hand::new(gateway), lease, said);
            }
        }
    }

    /// The order of one decision over `world`, the caller's one load for
    /// it. No fence is read and no arrival budget is counted: those exist
    /// where a wire and an election do.
    fn order(
        &self,
        world: &World,
        at: usize,
        term: u64,
        taken: &Taken<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
        entry: Entry,
    ) -> Result<Decided, Refused> {
        match entry {
            Entry::Single => {
                let (decision, booking) = decide_in(world, taken, buffers, &self.parts[at]);
                let lease =
                    self.file::<PLAIN>(world, term, taken, &buffers.named, &decision, booking)?;
                Ok(Decided { decision, lease })
            }
            Entry::Pair => self.pair(world, at, term, taken, buffers),
            Entry::Side => Err(Refused::PairSide),
        }
    }

    /// What stands behind a single's decision, in each of its orders
    /// ([`Self::order`], [`Self::order_shadowed`], [`Self::order_own`]):
    /// a pick's lease is issued, the engine that decided is told of it,
    /// and the record of sent prefixes takes the prefix of `named`
    /// ([`issue`]). `named` is the live decision's naming, or the
    /// naming of a shadow its pool keeps the record for
    /// ([`Self::shadowed_order`]). A step that joins a single's commit
    /// is written here, once, and stands in every order.
    ///
    /// Each order builds a copy of its own (`ORDER`), with a telling of
    /// its own: the store builds its issue for each telling, so no
    /// order's issue moves where another's is built.
    #[inline(always)]
    fn file<const ORDER: u8>(
        &self,
        world: &World,
        term: u64,
        taken: &Taken<'_, Request<'_>>,
        named: &Named,
        decision: &Decision,
        booking: Option<Booking>,
    ) -> Result<Option<Leased>, Refused> {
        let engine = world.spec().engine();
        let told = |lease: &Lease| {
            engine.on_decision_issued(lease, decision);
        };
        let lease = booking
            .map(|booking| issue(&self.store, world, taken, term, booking, named, told))
            .transpose()
            .map_err(|_| Refused::Unleased)?;
        Ok(lease.map(|lease| Leased {
            lease,
            due: taken.due(),
        }))
    }

    /// [`Self::order`] for a request whose binding names a shadow
    /// (`due`, [`Engine::shadow_for`]). Such a binding is a single's: a
    /// pair's names none, and takes the plain order. The decision is made
    /// at the engine's keeping entries, and the shadow is asked over the
    /// same view ahead of the commit ([`decide_shadowed_in`]). The commit
    /// is the plain order's ([`Self::file`]). What the shadow said comes
    /// back counted, behind a commit that held: a decision whose lease
    /// the store refused counts none. A shadow that panicked is counted
    /// with the door's faults and the scratch it left is dropped, both
    /// behind the commit: the live decision is issued as decided. The
    /// budget is the one its host stated at the decision's instant: in
    /// process the door reads no clock.
    ///
    /// A shadow that reads none of its live decision's rows takes an
    /// order of its own ([`Self::order_own`]): this one pays one test
    /// for it and runs no part of a naming.
    fn order_shadowed(
        &self,
        world: &World,
        at: usize,
        term: u64,
        taken: &Taken<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
        due: &mut Due,
    ) -> Result<Decided, Refused> {
        if matches!(due.rows, ShadowRows::Own | ShadowRows::Apart) {
            return self.order_own(world, at, term, taken, buffers, due);
        }
        self.shadowed_order::<SHADOWED>(world, at, term, taken, buffers, due)
    }

    /// [`Self::order_shadowed`] for a shadow that reads none of its live
    /// decision's rows: one that is named rows of its own, or one that
    /// is skipped ([`decide_own_in`]). Out of line, with a copy of the
    /// commit of its own: no other order runs a part of it.
    #[inline(never)]
    fn order_own(
        &self,
        world: &World,
        at: usize,
        term: u64,
        taken: &Taken<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
        due: &mut Due,
    ) -> Result<Decided, Refused> {
        self.shadowed_order::<OWN_ROWS>(world, at, term, taken, buffers, due)
    }

    /// The body of the two shadowed orders, built once for each
    /// ([`Self::order_shadowed`], [`Self::order_own`]): the decision
    /// with its shadow, the commit ([`Self::file`]), then what the
    /// shadow said or its fault.
    ///
    /// The record of sent prefixes takes the live decision's naming.
    /// Where the live profile is named no row, its pool keeps the
    /// record for a shadow that is named rows of its own
    /// ([`Engine::pool_record_for`]). The issue then writes the prefixes
    /// that shadow's naming folded, on the live decision's worker: the
    /// shadow reads where the requests went. A shadow that was not
    /// asked named nothing, and its decision writes no record.
    #[inline(always)]
    fn shadowed_order<const ORDER: u8>(
        &self,
        world: &World,
        at: usize,
        term: u64,
        taken: &Taken<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
        due: &mut Due,
    ) -> Result<Decided, Refused> {
        let own = &self.parts[at];
        let word = (due.binding, due.rows);
        // The order of a shadow's own rows holds its seat's buffer for
        // them through the commit: the issue may write from it.
        let mut lent = match ORDER {
            OWN_ROWS => Some(self.own_named[at].named.lock()),
            _ => None,
        };
        let ((decision, booking), shadow) = match lent.as_deref_mut() {
            Some(named) => decide_own_in(world, taken, buffers, own, word, named, || taken.left),
            None => decide_shadowed_in(world, taken, buffers, own, word, || taken.left),
        };
        let named = match lent.as_deref() {
            Some(own_named) if buffers.named.sent_pool().is_none() => own_named,
            _ => &buffers.named,
        };
        let filed = self.file::<ORDER>(world, term, taken, named, &decision, booking);
        if shadow == Shadow::Faulted {
            // The next decision reads nothing the shadow left.
            buffers.scratch = Scratch::default();
            self.faults.fetch_add(1, Ordering::Relaxed);
        }
        let lease = filed?;
        if let Shadow::Said(shadow) = shadow {
            let count = own.shadowed(shadow.binding, shadow.kind);
            count.fetch_add(1, Ordering::Relaxed);
            due.said = Some(shadow);
        }
        Ok(Decided { decision, lease })
    }

    /// A pair's order: decided for both sides and, for a pick, issued a
    /// lease a side under one id. Out of line: a single's order carries no
    /// part of a pair's.
    #[inline(never)]
    fn pair(
        &self,
        world: &World,
        at: usize,
        term: u64,
        taken: &Taken<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
    ) -> Result<Decided, Refused> {
        let (decision, booking) = pair_in(world, taken, buffers, &self.parts[at]);
        self.file_pair(world, term, taken, decision, booking)
    }

    fn file_pair(
        &self,
        world: &World,
        term: u64,
        taken: &Taken<'_, Request<'_>>,
        decision: Decision,
        booking: Option<PairBooking>,
    ) -> Result<Decided, Refused> {
        let lease = booking
            .map(|booking| {
                issue_pair(&self.store, world, taken, term, &decision, booking)
                    .map(|(prefill, _)| prefill)
            })
            .transpose()
            .map_err(|_| Refused::Unleased)?;
        Ok(Decided {
            decision,
            lease: lease.map(|lease| Leased {
                lease,
                due: taken.due(),
            }),
        })
    }

    /// A decode call's order. The store's own test of the lease comes
    /// first, so a call it refuses runs no stage. Then the pair's rows of
    /// this instant, the engine's decode entry over the held prefill's
    /// partners and, for a pick, the side's issue under the lease's id,
    /// dated at this call ([`issued_later`]).
    /// The decode profile is shown the decode pool's facts: its size, its
    /// list and whether every row of it holds a report.
    #[inline(never)]
    fn decode(
        &self,
        world: &World,
        at: usize,
        taken: &Taken<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
        lease: LeaseId,
    ) -> Result<Decided, Refused> {
        let store = &*self.store;
        let prefill = store
            .held_prefill(taken.gateway, lease)
            .map_err(Refused::Decode)?;
        let Buffers {
            pair,
            scratch,
            lists,
            ..
        } = buffers;
        let (decision, side) = pair_with::<true, _, _, _>(
            world,
            taken,
            &self.parts[at],
            pair,
            (lists, scratch),
            |pair, _, model, ask, wires, subset| {
                let built = world.build_pair(model, &ask, pair)?;
                world.count_pair_in_flight(&built, &ask, pair);
                Some((built, pair.narrow(wires, subset, &built)))
            },
            (
                |engine, pair, facts, pool, cx, scratch| {
                    let facts = RequestFacts {
                        phase: Phase::Decode,
                        ..*facts
                    };
                    engine.decide_decode(&facts, pool, pair.pools(), prefill, cx, scratch)
                },
                |facts, decision, built, _| {
                    let (pick, (_, decodes)) = decision.pick().zip(built)?;
                    let first = pick.targets.first()?.pair?;
                    let side = DecodeSide {
                        target: first.decode_id,
                        picked: Picked::of_decodes(&pick.targets),
                        session: RequestFacts {
                            phase: Phase::Decode,
                            ..*facts
                        }
                        .session(),
                        state: LeaseState::ISSUED,
                        tokens: facts.prompt_tokens,
                        target_index: 0,
                    };
                    Some((side, world.booked(&decodes, side.target)))
                },
            ),
        );
        let Some((side, slot)) = side else {
            return Ok(Decided {
                decision,
                lease: None,
            });
        };
        let spec = world.spec();
        let lease = store
            .issue_decided_decode(taken.gateway, lease, side, slot, taken.now, |lease| {
                // The lease keeps the generation of its first decision, and
                // the engine of that generation alone hears its words
                // ([`issuing`]). Behind a reload the engine of this instant
                // decided the side and hears nothing of the lease.
                let Some(engine) = issuing(spec, lease) else {
                    return;
                };
                // The side is on file by now, and its issue is dated at
                // this call. A stage that panics while it hears of the
                // issue is held and counted, as under a report: the host
                // still gets the pick the file holds.
                let heard = || issued_later(world, engine, lease, &decision, taken.now);
                match catch_unwind(AssertUnwindSafe(heard)) {
                    Ok(()) => store.call_returned(),
                    Err(payload) => {
                        release(payload);
                        store.call_faulted();
                    }
                }
            })
            .map_err(Refused::Decode)?;
        Ok(Decided {
            decision,
            lease: Some(Leased {
                lease,
                due: taken.due(),
            }),
        })
    }
}

/// The plain order's copy of a single's commit ([`Door::file`]).
const PLAIN: u8 = 0;
/// The copy of the order whose shadow reads its live decision's rows.
const SHADOWED: u8 = 1;
/// The copy of the order whose shadow reads none of them.
const OWN_ROWS: u8 = 2;

/// A decision's shadow at its seat: what the seat read of the request's
/// binding ahead of the order ([`Engine::shadow_for`]), and what the
/// order leaves of the shadow.
struct Due {
    /// The binding's position in its generation.
    binding: u32,
    /// How its shadow reads the rows of the live decision.
    rows: ShadowRows,
    /// What the shadow said, behind a commit that held.
    said: Option<Shadowed>,
}

/// The order of one decision at its seat, in the two forms a seat runs
/// it in. A generation that neither shadows nor samples runs the plain
/// form and no part of the other. Under one that shadows, a request
/// whose binding names no shadow runs the plain form too.
trait Orders {
    /// Decides the request and, for a pick, issues its lease.
    fn plain(
        &mut self,
        world: &World,
        at: usize,
        term: u64,
        taken: &Taken<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
    ) -> Result<Decided, Refused>;

    /// [`Self::plain`] for a request whose binding names a shadow
    /// ([`Door::order_shadowed`]). What the shadow said is left in
    /// `due`: the answer is built where it is returned, and is not
    /// carried beside a second value.
    fn shadowed(
        &mut self,
        world: &World,
        at: usize,
        term: u64,
        taken: &Taken<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
        due: &mut Due,
    ) -> Result<Decided, Refused>;
}

/// An order lent to a run. A decision whose lease the store refused runs
/// its order twice: the first run borrows it and the second takes it, so
/// each run is an instance of its own, as each closure's was.
impl<O: Orders> Orders for &mut O {
    #[inline(always)]
    fn plain(
        &mut self,
        world: &World,
        at: usize,
        term: u64,
        taken: &Taken<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
    ) -> Result<Decided, Refused> {
        (**self).plain(world, at, term, taken, buffers)
    }

    #[inline(always)]
    fn shadowed(
        &mut self,
        world: &World,
        at: usize,
        term: u64,
        taken: &Taken<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
        due: &mut Due,
    ) -> Result<Decided, Refused> {
        (**self).shadowed(world, at, term, taken, buffers, due)
    }
}

/// The order of [`Door::decide`] and [`Door::decide_under`]: the request
/// takes the path its plan gives it ([`entry`]).
struct Whole<'d>(&'d Door);

impl Orders for Whole<'_> {
    // Forced into the seat's call: out of line the plain order would pay
    // a call the closure it stands for never made.
    #[inline(always)]
    fn plain(
        &mut self,
        world: &World,
        at: usize,
        term: u64,
        taken: &Taken<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
    ) -> Result<Decided, Refused> {
        let entry = entry(world, taken);
        self.0.order(world, at, term, taken, buffers, entry)
    }

    fn shadowed(
        &mut self,
        world: &World,
        at: usize,
        term: u64,
        taken: &Taken<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
        due: &mut Due,
    ) -> Result<Decided, Refused> {
        self.0.order_shadowed(world, at, term, taken, buffers, due)
    }
}

/// [`Whole`] for [`Door::decide_under_stated`]: each run of the order
/// notes what the made request said of its decode
/// ([`Asked::decode_later`]).
struct WholeStated<'d, 'l> {
    door: &'d Door,
    later: &'l mut bool,
}

impl Orders for WholeStated<'_, '_> {
    #[inline(always)]
    fn plain(
        &mut self,
        world: &World,
        at: usize,
        term: u64,
        taken: &Taken<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
    ) -> Result<Decided, Refused> {
        *self.later = taken.request.decode_later();
        Whole(self.door).plain(world, at, term, taken, buffers)
    }

    fn shadowed(
        &mut self,
        world: &World,
        at: usize,
        term: u64,
        taken: &Taken<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
        due: &mut Due,
    ) -> Result<Decided, Refused> {
        *self.later = taken.request.decode_later();
        Whole(self.door).shadowed(world, at, term, taken, buffers, due)
    }
}

/// The order of [`Door::decide_decode`]: the decode side of the pair's
/// lease on file ([`Door::decode`]). A pair's binding names no shadow, so
/// the seat runs the plain form.
struct DecodeCall<'d>(&'d Door, LeaseId);

impl Orders for DecodeCall<'_> {
    #[inline(always)]
    fn plain(
        &mut self,
        world: &World,
        at: usize,
        _: u64,
        taken: &Taken<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
    ) -> Result<Decided, Refused> {
        self.0.decode(world, at, taken, buffers, self.1)
    }

    fn shadowed(
        &mut self,
        world: &World,
        at: usize,
        term: u64,
        taken: &Taken<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
        _: &mut Due,
    ) -> Result<Decided, Refused> {
        self.plain(world, at, term, taken, buffers)
    }
}

/// The order of [`Door::decide_encode_pair`]: the request that enters an
/// `epd` plan is decided as a pair with its encode items placed, and any
/// other as [`Whole`] orders it, with `placed` cleared.
struct WithEncode<'d, 'i, 'p> {
    door: &'d Door,
    items: EncodeItems<'i>,
    placed: &'p mut EncodePlaced,
}

impl WithEncode<'_, '_, '_> {
    /// The pair's order, for the request that enters an `epd` plan.
    /// `Err` is the path of any other request, with `placed` cleared.
    #[inline(always)]
    fn pair(
        &mut self,
        world: &World,
        at: usize,
        term: u64,
        taken: &Taken<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
    ) -> Result<Result<Decided, Refused>, Entry> {
        let (entry, plan) = entry_and_plan(world, taken);
        if !matches!((entry, plan), (Entry::Pair, Plan::Epd { .. })) {
            self.placed.clear();
            return Err(entry);
        }
        let own = &self.door.parts[at];
        let (decision, booking) =
            encode_pair_in(world, taken, self.items, self.placed, buffers, own);
        Ok(self.door.file_pair(world, term, taken, decision, booking))
    }
}

impl Orders for WithEncode<'_, '_, '_> {
    #[inline(always)]
    fn plain(
        &mut self,
        world: &World,
        at: usize,
        term: u64,
        taken: &Taken<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
    ) -> Result<Decided, Refused> {
        match self.pair(world, at, term, taken, buffers) {
            Ok(decided) => decided,
            Err(entry) => self.door.order(world, at, term, taken, buffers, entry),
        }
    }

    fn shadowed(
        &mut self,
        world: &World,
        at: usize,
        term: u64,
        taken: &Taken<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
        due: &mut Due,
    ) -> Result<Decided, Refused> {
        // A binding that names a shadow is a single's: its request
        // enters no pair, and `placed` is cleared as for any other.
        self.placed.clear();
        self.door
            .order_shadowed(world, at, term, taken, buffers, due)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    pub(crate) mod sent_prefixes;

    use std::{
        cell::Cell,
        collections::BTreeSet,
        error::Error,
        sync::{
            atomic::{AtomicBool, Ordering},
            mpsc, Arc, Barrier,
        },
        thread,
        time::{Duration, Instant},
    };

    use parking_lot::Mutex;

    use super::{
        decider_of, pinned, Door, Maker, Refused, Request, Swept, World, DISPATCH_TTL,
        ON_SAMPLE_READ, SEAT, TERM_MARK,
    };
    use crate::{
        builtins::{Best, Random},
        catalog::Catalog,
        decider::Decider,
        decision::{Decision, Lifted, ShedReason, Target, Targets},
        demands::{DemandFacts, Demands, References, Unmet},
        framework::{DecisionScope, EncodeItems, EncodePlaced, Filter, Picker, TieBand},
        lease::{Lease, LeaseId, Lifecycle, Outcome, ReleaseReason, Side, TransportClass},
        sessions::SessionVerdict,
        spec::{AppliedSpec, Needs, PolicySpec},
        store::{
            Ask, DecodeMark, DecodeRefused, Dispatch, EncodeHandoff, EncodeOffer, GatewayId,
            HealthSink, HealthWrite, LeaderBits, LeaseRecord, LeaseState, LeasesStore, LoadSink,
            LoadSource, MemoryStore, Picked, Said, SessionsStore, VetoWrite, ViewBuffers,
            WorkerFacts, WorkerStatus, WorkersStore,
        },
        view::{
            Candidate, Constraints, Identity, KeyedRow, LoadRow, Phase, PoolFacts, RequestFacts,
            Role, Session, WireModes, WorkerId,
        },
        DecisionRecord,
    };

    mod shadow;

    pub(crate) const MODEL: &str = "model-a";
    pub(crate) const GATEWAY: GatewayId = GatewayId(11);

    /// A preset of this build as generation `generation`.
    fn preset(name: &str, generation: u64) -> AppliedSpec {
        let mut document = PolicySpec::preset(name).expect("a preset of this build");
        document.generation = generation;
        AppliedSpec::validate(&document).expect("the document validates")
    }

    /// A store of `deciders` files under the `random` preset, with three
    /// ready members that serve [`MODEL`].
    pub(crate) fn fleet(deciders: usize) -> Arc<MemoryStore> {
        stocked(MemoryStore::new().with_deciders(deciders))
    }

    /// `store` under the `random` preset, with three ready members that
    /// serve [`MODEL`].
    fn stocked(store: MemoryStore) -> Arc<MemoryStore> {
        for name in ["w0", "w1", "w2"] {
            member(&store, name, Role::Single);
        }
        store
            .apply(preset("random", 1))
            .expect("above the unbound start");
        Arc::new(store)
    }

    fn member(store: &MemoryStore, name: &str, role: Role) {
        let key = format!("http://{name}:8000");
        let facts = WorkerFacts {
            key: key.clone(),
            engine: key,
            incarnation: 1,
            role,
            models: vec![MODEL.to_owned()],
            dp_size: 1,
            max_running: Some(4_096),
            ..WorkerFacts::default()
        };
        ready(store, facts);
    }

    fn ready(store: &MemoryStore, facts: WorkerFacts) -> WorkerId {
        let id = store.upsert(facts).id();
        store.publish_now();
        let row = WorkersStore::get(store, id);
        assert!(row.is_some(), "the worker has a row");
        if let Some(row) = row.filter(|row| row.status != WorkerStatus::Ready) {
            assert!(store
                .set_status(id, WorkerStatus::Ready, row.revision)
                .is_some());
        }
        id
    }

    /// A model under a pair plan, with ready workers in each pool.
    pub(crate) fn pair_fleet(
        prefills: usize,
        decodes: usize,
    ) -> Result<Arc<MemoryStore>, Box<dyn Error>> {
        pair_fleet_over(MemoryStore::new(), prefills, decodes)
    }

    /// [`pair_fleet`] over `store`.
    fn pair_fleet_over(
        store: MemoryStore,
        prefills: usize,
        decodes: usize,
    ) -> Result<Arc<MemoryStore>, Box<dyn Error>> {
        for n in 0..prefills {
            member(&store, &format!("p{n}"), Role::Prefill);
        }
        for n in 0..decodes {
            member(&store, &format!("d{n}"), Role::Decode);
        }
        let document = PolicySpec::pair_presets("least_load", "least_load")?;
        apply_pair_document(&store, document)?;
        Ok(Arc::new(store))
    }

    /// The stock pair document filters its prefills by a bit the host
    /// supplies. A row the store builds reads that bit open, so the document
    /// as written places a pair over the store's rows. With the bit unset it
    /// kept no prefill and shed every pair.
    #[test]
    fn a_pair_is_placed_over_the_stores_rows_under_the_stock_pair_document() {
        let store = MemoryStore::new();
        member(&store, "p0", Role::Prefill);
        member(&store, "d0", Role::Decode);
        let document = PolicySpec::pair_presets("least_load", "least_load");
        let applied = AppliedSpec::validate(&document.expect("the stock presets"));
        let applied = store.apply(applied.expect("the stock document"));
        applied.expect("a first generation");
        let door = Door::new(Arc::new(store));
        let answer = door.decide(GATEWAY, &pair_request(b"pair"), Instant::now());
        let decided = answer.expect("no refusal");
        assert!(decided.lease.is_some(), "the stock document shed the pair");
    }

    fn apply_epd(store: &MemoryStore) -> Result<(), Box<dyn Error>> {
        member(store, "e0", Role::Encode);
        let mut document = PolicySpec::epd_presets("least_load", "least_load", "least_load")?;
        document.generation = 2;
        apply_pair_document(store, document)
    }

    fn encode_facts(key: &str, role: Role, connector: &str) -> WorkerFacts {
        WorkerFacts {
            key: key.into(),
            engine: key.into(),
            engine_kind: "runtime".into(),
            incarnation: 1,
            role,
            models: vec![MODEL.into()],
            wire_modes: WireModes::GRPC,
            dp_size: 1,
            max_running: Some(4_096),
            encode: EncodeOffer {
                handoff: EncodeHandoff::Push,
                producer: role == Role::Encode,
                consumer: role == Role::Prefill,
                connector: Some(connector.into()),
                ..EncodeOffer::default()
            },
            ..WorkerFacts::default()
        }
    }

    fn encode_member(store: &MemoryStore, key: &str, role: Role, connector: &str) -> WorkerId {
        ready(store, encode_facts(key, role, connector))
    }

    fn encode_document() -> Result<PolicySpec, Box<dyn Error>> {
        let mut document =
            PolicySpec::epd_presets("consistent_hashing", "least_load", "least_load")?;
        for side in ["prefill", "decode"] {
            let profile = document
                .profiles
                .get_mut(side)
                .ok_or("missing pair profile")?;
            profile
                .filter
                .retain(|stage| !matches!(stage.name(), "has-capacity" | "has-room"));
            profile.pick = "count-first".into();
        }
        let encode = document
            .profiles
            .get_mut("encode")
            .ok_or("missing encode profile")?;
        encode.score.push("prefix-affinity".into());
        Ok(document)
    }

    fn encode_door() -> Result<Door, Box<dyn Error>> {
        let store = Arc::new(MemoryStore::new());
        encode_member(&store, "grpc://d0:8000", Role::Decode, "c0");
        store.apply(AppliedSpec::validate(&encode_document()?)?)?;
        Ok(Door::new(store))
    }

    pub(crate) fn push_fleet() -> Result<(Door, WorkerId), Box<dyn Error>> {
        let door = encode_door()?;
        let store = door.store();
        encode_member(store, "grpc://p0:8000", Role::Prefill, "c0");
        let encoder = encode_member(store, "grpc://e0:8000", Role::Encode, "c0");
        Ok((door, encoder))
    }

    pub(crate) fn pair_request(id: &[u8]) -> Request<'_> {
        let mut asked = request(id, 7);
        asked.facts.phase = Phase::Prefill;
        asked
    }

    const KEYS: [Identity<'static>; 2] = [Identity::hashed(11), Identity::hashed(12)];

    fn items() -> EncodeItems<'static> {
        EncodeItems {
            keys: &KEYS,
            ..EncodeItems::default()
        }
    }

    fn release(door: &Door, decided: &super::Decided) {
        if let Some(leased) = decided.lease {
            door.report(
                GATEWAY,
                leased.lease.id,
                Said::Released(None, ReleaseReason::ClientGone),
            );
        }
    }

    fn place_one(door: &Door, now: Instant) -> Result<EncodePlaced, Refused> {
        let mut placed = EncodePlaced::new();
        let items = EncodeItems {
            one: Some(Identity::hashed(99)),
            ..items()
        };
        let decided =
            door.decide_encode_pair(GATEWAY, &pair_request(b"first"), items, &mut placed, now)?;
        release(door, &decided);
        Ok(placed)
    }

    fn apply_pair_document(
        store: &MemoryStore,
        mut document: PolicySpec,
    ) -> Result<(), Box<dyn Error>> {
        for profile in document.profiles.values_mut() {
            profile.filter.clear();
        }
        store.apply(AppliedSpec::validate(&document)?)?;
        Ok(())
    }

    /// Applies `generation` of the `random` preset with the picker `name`
    /// in its pick's place.
    pub(crate) fn picking<F>(store: &MemoryStore, generation: u64, name: &'static str, picker: F)
    where
        F: Fn() -> Box<dyn Picker> + Send + Sync + 'static,
    {
        let mut catalog = Catalog::builtin();
        let added = catalog.picker(name, picker);
        added.expect("a free name");
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        document.generation = generation;
        let profile = document.profiles.get_mut("random");
        profile.expect("the preset's profile").pick = name.into();
        let digest = document.digest().expect("the document digests");
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[]);
        let applied = store.apply(spec.expect("the catalog holds the stage"));
        applied.expect("a later generation");
    }

    /// A request for [`MODEL`] under `id` and `seed`.
    pub(crate) fn request(id: &[u8], seed: u64) -> Request<'_> {
        Request {
            id,
            facts: RequestFacts {
                model: MODEL,
                seed,
                ..RequestFacts::default()
            },
            constraints: Constraints::default(),
        }
    }

    pub(crate) fn later_request(id: &[u8]) -> Request<'_> {
        let mut asked = request(id, 7);
        asked.facts.phase = Phase::Prefill;
        asked.constraints.decode_later = true;
        asked
    }

    pub(crate) fn load_decodes(
        store: &MemoryStore,
        [busy, idle]: [WorkerId; 2],
        at: Instant,
    ) -> Instant {
        for (worker, queued) in [(busy, 80), (idle, 0)] {
            store.record(
                worker,
                0,
                LoadRow {
                    queued,
                    live_rate: 2_000,
                    ..LoadRow::default()
                },
                at,
                LoadSource::Stream,
            );
        }
        let now = store.publish_due().map_or(at, |due| due.max(at));
        store.publish(now);
        now
    }

    /// What the leases on file book on each member, as a decision reads it.
    fn booked(store: &MemoryStore, role: Role) -> Vec<(WorkerId, u32)> {
        let ask = Ask {
            gateway: GATEWAY,
            now: Instant::now(),
            pinned: None,
            affinity: false,
            signal_cells: false,
        };
        let mut out = ViewBuffers::default();
        store.deciding(|world| {
            if let Some(built) = world.build(MODEL, role, &ask, &mut out) {
                world.count_in_flight(&built, &ask, &mut out.candidates);
            }
        });
        let rows = out.candidates.iter();
        rows.map(|row| (row.id, row.in_flight)).collect()
    }

    /// The members the leases on file book, each with its count.
    pub(crate) fn counted(store: &MemoryStore) -> Vec<(WorkerId, u32)> {
        let mut rows = booked(store, Role::Single);
        rows.retain(|&(_, count)| count > 0);
        rows
    }

    /// How many leases each of `workers` in the pair pools holds.
    pub(crate) fn pair_counts<const N: usize>(
        store: &MemoryStore,
        workers: [WorkerId; N],
    ) -> [u32; N] {
        let prefill = booked(store, Role::Prefill);
        let decode = booked(store, Role::Decode);
        workers.map(|id| {
            prefill
                .iter()
                .chain(&decode)
                .find_map(|&(worker, count)| (id == worker).then_some(count))
                .unwrap_or(0)
        })
    }

    /// The lease of one decision of `door` and the pick's first target.
    fn leased(door: &Door, id: &[u8], seed: u64) -> (LeaseId, WorkerId) {
        let decided = door.decide(GATEWAY, &request(id, seed), Instant::now());
        let decided = decided.expect("a decision");
        let first = decided
            .decision
            .pick()
            .and_then(|pick| pick.targets.first());
        let leased = decided.lease.expect("a pick's lease");
        (leased.lease.id, first.expect("a pick over the fleet").id)
    }

    /// Says what `GATEWAY` reports of `lease` through the door from
    /// another thread, as a host's runtime does: how many of its leases
    /// that thread then finds on file. A report is applied at once only
    /// where no other thread is at work at the file.
    fn report(door: &Door, lease: LeaseId, said: Said) -> usize {
        thread::scope(|scope| {
            let reported = scope.spawn(move || {
                door.report(GATEWAY, lease, said);
                LeasesStore::live(&**door.store(), GATEWAY)
            });
            reported.join().expect("the report's thread")
        })
    }

    fn gone() -> Said {
        Said::Released(None, ReleaseReason::ClientGone)
    }

    fn sampling(store: &MemoryStore, generation: u64, sample: u32) {
        let mut document = PolicySpec::preset("random").expect("a preset");
        document.generation = generation;
        document.explain.sample = sample;
        store
            .apply(AppliedSpec::validate(&document).expect("a valid document"))
            .expect("a later generation");
    }

    /// Holds a read at the first seat until the returned sender drops.
    fn hold_read<'scope>(
        scope: &'scope thread::Scope<'scope, '_>,
        door: &'scope Door,
    ) -> Result<mpsc::Sender<()>, mpsc::RecvTimeoutError> {
        let (held, entered) = mpsc::channel();
        let (release, resume) = mpsc::channel();
        scope.spawn(move || {
            ON_SAMPLE_READ.set(Some(Box::new(move || {
                let _ = held.send(());
                let _ = resume.recv();
            })));
            let _ = door.explain(GATEWAY, b"read");
        });
        entered.recv_timeout(PATIENCE)?;
        Ok(release)
    }

    #[test]
    fn a_decision_that_keeps_no_record_waits_on_no_read() {
        for sample in [0, 2] {
            let store = fleet(2);
            sampling(&store, 2, sample);
            let door = &Door::new(store);
            thread::scope(|scope| {
                let _read = hold_read(scope, door).expect("the read holds the first ring");
                let (answer, answered) = mpsc::channel();
                scope.spawn(move || {
                    seated(0);
                    let (lease, _) = leased(door, b"caller", 7);
                    let _ = answer.send((door.store().file_of(lease), SEAT.get()));
                });
                assert_eq!(
                    answered.recv_timeout(PATIENCE),
                    Ok((0, 0)),
                    "sample {sample}"
                );
            });
            assert!(door.explain(GATEWAY, b"caller").is_none());
        }
    }

    #[test]
    fn a_read_of_a_ring_holds_no_decision_of_its_model() {
        let store = fleet(1);
        picking(&store, 2, "scoped", || Box::new(Scoped));
        let door = &Door::new(store);
        let _ = leased(door, b"warm", 7);
        thread::scope(|scope| {
            let _read = hold_read(scope, door).expect("the read holds the model's ring");
            let (answer, answered) = mpsc::channel();
            scope.spawn(move || {
                let (lease, _) = leased(door, b"home", 7);
                let _ = answer.send(door.store().file_of(lease));
            });
            assert_eq!(answered.recv_timeout(PATIENCE), Ok(0));
        });
    }

    #[test]
    fn a_kept_record_waits_for_a_read_and_is_found() {
        let store = fleet(1);
        sampling(&store, 2, 1);
        let door = &Door::new(store);
        let id = b"kept";
        let decision = thread::scope(|scope| {
            let read = hold_read(scope, door).expect("the read holds the ring");
            let (answer, answered) = mpsc::channel();
            scope.spawn(move || {
                let decided = door
                    .decide(GATEWAY, &request(id, 7), Instant::now())
                    .expect("a decision");
                let _ = answer.send(decided.decision);
            });
            let due = Instant::now() + PATIENCE;
            while LeasesStore::live(&**door.store(), GATEWAY) == 0 {
                assert!(Instant::now() < due, "the decision files its lease");
                thread::yield_now();
            }
            assert_eq!(
                answered.recv_timeout(Duration::from_millis(100)),
                Err(mpsc::RecvTimeoutError::Timeout),
                "a decision that keeps a record waits for the read"
            );
            assert!(door.seats[0].slot.is_locked());
            drop(read);
            answered
                .recv_timeout(PATIENCE)
                .expect("the decision returns after the read")
        });
        let record = door.explain(GATEWAY, id).expect("the kept record");
        assert_eq!(record.decision, decision);
    }

    #[test]
    fn sampling_off_allocates_no_slots_and_explains_nothing() {
        let door = Door::new(fleet(1));
        let _ = leased(&door, b"off", 1);
        assert!(door.explain(GATEWAY, b"off").is_none());
        assert_eq!(door.seats[0].samples.lock().slots(), 0);
        assert_eq!(door.seats[0].slot.lock().sample_count.counted(), 0);
    }

    #[test]
    fn sample_one_keeps_the_answer_and_instant_of_each_pick_and_shed() {
        let store = fleet(1);
        sampling(&store, 2, 1);
        let door = Door::new(store);
        let now = Instant::now();
        for (id, model) in [(b"pick", MODEL), (b"shed", "empty")] {
            let mut ask = request(id, 7);
            ask.facts.model = model;
            let answer = door.decide(GATEWAY, &ask, now).expect("a decision");
            assert_eq!(answer.decision.pick().is_some(), model == MODEL);
            let record = door.explain(GATEWAY, id).expect("the sampled decision");
            assert_eq!(record.decision, answer.decision);
            assert_eq!(record.binding, Some(0));
            assert_eq!(record.at, now);
            assert!(door.explain(GatewayId::LOCAL, id).is_none());
        }
    }

    #[test]
    fn the_door_explains_a_sampled_decision_with_its_names() {
        use crate::{Explained, Names};

        let store = fleet(1);
        sampling(&store, 2, 1);
        let door = Door::new(Arc::clone(&store));
        let id = b"named";
        assert!(door.explained(GATEWAY, id).is_none());

        let now = Instant::now();
        let answer = door
            .decide(GATEWAY, &request(id, 7), now)
            .expect("a decision");
        let record = door.explain(GATEWAY, id).expect("a sample");
        assert_eq!(record.decision, answer.decision);
        assert_eq!(record.at, now);
        assert!(door.explained(GatewayId::LOCAL, id).is_none());
        assert_eq!(
            door.explained(GATEWAY, id),
            Some(Explained {
                record,
                names: Some(Names {
                    plan: "single",
                    profile: "random".into(),
                    picker: "random".into(),
                    filters: vec![
                        "healthy".into(),
                        "not-overloaded".into(),
                        "not-tried".into()
                    ],
                    scorers: vec![],
                }),
            })
        );

        sampling(&store, 3, 0);
        assert_eq!(
            door.explained(GATEWAY, id),
            Some(Explained {
                record,
                names: None
            })
        );
    }

    #[test]
    fn the_door_sheds_before_the_first_profile_and_names_the_plan_alone() {
        let store = fleet(1);
        sampling(&store, 2, 1);
        let door = Door::new(store);

        let id = b"narrowed";
        let mut ask = request(id, 7);
        ask.constraints.subset_hint = Some(0);
        let answer = door.decide(GATEWAY, &ask, Instant::now()).expect("a shed");
        let Decision::Shed(shed) = answer.decision else {
            panic!("the subset left no candidate");
        };
        assert_eq!(shed.emptied_by, super::SUBSET_HINT);
        let explained = door.explained(GATEWAY, id).expect("the sampled shed");
        assert_eq!(explained.record.binding, Some(0));
        assert_eq!(
            explained.names,
            Some(crate::Names {
                plan: "single",
                ..crate::Names::default()
            }),
        );
        assert_eq!(shed.trace.made_under(), None);
    }

    #[test]
    fn each_seat_keeps_every_nth_answer_by_its_own_count() {
        let store = fleet(2);
        sampling(&store, 2, 3);
        let door = Door::new(store);
        for n in 1..=6 {
            for seat in [0_u8, 1] {
                let at = usize::from(seat);
                seated(at);
                let id = [seat, n];
                let (lease, _) = leased(&door, &id, u64::from(n));
                assert_eq!(door.store().file_of(lease), at);
                door.report(GATEWAY, lease, gone());
                if n < 3 {
                    assert_eq!(door.seats[at].samples.lock().slots(), 0);
                }
            }
        }
        for seat in [0, 1] {
            for n in 1..=6 {
                assert_eq!(door.explain(GATEWAY, &[seat, n]).is_some(), n % 3 == 0);
            }
        }
    }

    #[test]
    fn the_ring_wraps_at_1024_records_and_loses_the_oldest() {
        use std::io::{self, Write as _};

        let store = fleet(1);
        sampling(&store, 2, 1);
        let door = Door::new(store);
        for n in 0_u32..1_025 {
            let id = n.to_le_bytes();
            let mut ask = request(&id, u64::from(n));
            ask.facts.model = "empty";
            let answer = door.decide(GATEWAY, &ask, Instant::now()).expect("a shed");
            assert!(matches!(answer.decision, Decision::Shed(_)));
        }
        assert!(door.explain(GATEWAY, &0_u32.to_le_bytes()).is_none());
        assert!(door.explain(GATEWAY, &1_u32.to_le_bytes()).is_some());
        assert!(door.explain(GATEWAY, &1_024_u32.to_le_bytes()).is_some());
        let slots = door.seats[0].samples.lock().slots();
        assert_eq!(slots, 1_024);
        let bytes = size_of::<DecisionRecord>();
        writeln!(
            io::stdout(),
            "record_bytes={bytes} ring_bytes={}",
            slots * bytes
        )
        .expect("stdout");
    }

    /// A maker that says the generation it was handed, and applies another
    /// before its request is back at the door where it is told one.
    struct Reads<'t> {
        store: &'t MemoryStore,
        handed: Cell<u64>,
        applies: Option<u64>,
        id: &'t [u8],
    }

    impl Maker for Reads<'_> {
        fn model(&self) -> &str {
            MODEL
        }

        fn make<'w>(&'w self, world: &'w World) -> Request<'w> {
            self.handed.set(world.spec().generation());
            if let Some(generation) = self.applies {
                let later = self.store.apply(preset("random", generation));
                later.expect("a later generation");
            }
            request(self.id, 7)
        }
    }

    /// A request its host makes is made from the world its decision is made
    /// under. An apply that lands while the host makes it is the next
    /// decision's: the one in hand keeps the world it handed.
    #[test]
    fn a_request_is_made_from_the_world_its_decision_is_made_under() {
        let store = fleet(1);
        let door = Door::new(Arc::clone(&store));
        let standing = || store.with_world(|world| world.spec().generation());
        let reads = |applies, id| Reads {
            store: &store,
            handed: Cell::new(0),
            applies,
            id,
        };
        let raced = reads(Some(2), b"made");
        let decided = door.decide_under(GATEWAY, &raced, Instant::now());
        let decided = decided.expect("a decision");
        assert_eq!(decided.decision.trace().generation, raced.handed.get());
        assert_eq!((raced.handed.get(), standing()), (1, 2));

        let next = reads(None, b"next");
        let decided = door.decide_under(GATEWAY, &next, Instant::now());
        let decided = decided.expect("a decision");
        assert_eq!(
            (next.handed.get(), decided.decision.trace().generation),
            (2, 2)
        );
    }

    #[test]
    fn an_apply_turns_sampling_on_and_off_and_starts_the_count_over() {
        let store = fleet(1);
        let door = Door::new(Arc::clone(&store));
        let _ = leased(&door, b"off-1", 1);
        sampling(&store, 2, 2);
        let _ = leased(&door, b"first", 2);
        sampling(&store, 3, 2);
        let _ = leased(&door, b"first-again", 3);
        let _ = leased(&door, b"kept", 4);
        sampling(&store, 4, 0);
        let _ = leased(&door, b"off-2", 5);
        for id in ["off-1", "first", "first-again", "off-2"] {
            assert!(door.explain(GATEWAY, id.as_bytes()).is_none(), "{id}");
        }
        let kept = door.explain(GATEWAY, b"kept").expect("the old sample");
        assert_eq!(kept.decision.trace().generation, 3);
    }

    /// A dispatch to a pick's first target.
    fn dispatched() -> Said {
        let dispatch = Dispatch {
            target_index: 0,
            side: Side::Single,
            since_decision_micros: 250,
        };
        Said::Dispatched(dispatch, Instant::now())
    }

    /// Places the calling thread at seat `at` of the doors it calls next.
    fn seated(at: usize) {
        SEAT.set(at);
    }

    /// A completion that `served` answered.
    fn completed(served: WorkerId) -> Said {
        Said::Completed(Outcome {
            side: Side::Single,
            served,
            status: 200,
            class: TransportClass::Ok,
            ttft_micros: 1_000,
            tokens_in: 12,
            tokens_out: 34,
            duration_micros: 5_000,
        })
    }

    /// After a decision its lease is on file and is counted on the pick's
    /// first target. A report handed between decisions is applied at once:
    /// no caller decides at the file.
    #[test]
    fn a_decision_files_its_lease_and_books_its_first_target() {
        let store = fleet(1);
        let door = Door::new(Arc::clone(&store));
        let (lease, first) = leased(&door, b"r-1", 7);
        let on_file = LeasesStore::get(&*store, GATEWAY, lease).expect("the lease is on file");
        assert_eq!(on_file.target, first);
        // The lease names the worker its issue books: the engine hears the
        // issue with it, and a decision given up is told of it.
        assert_eq!(on_file.lease.booked, first);
        assert_eq!(counted(&store), [(first, 1)]);
        assert_eq!(report(&door, lease, gone()), 0, "applied at once");
        assert_eq!(counted(&store), []);
    }

    /// Sixty-four callers at once on one model: every decision has its
    /// lease, no id is minted twice, and each member's count is the leases
    /// that book it.
    #[test]
    fn sixty_four_callers_at_once_on_one_model_each_take_a_lease() {
        let store = fleet(4);
        let door = Door::new(Arc::clone(&store));
        let (callers, each) = (64_u8, 8_u8);
        let start = Barrier::new(usize::from(callers));
        let leases: Vec<(LeaseId, WorkerId)> = thread::scope(|scope| {
            let asked: Vec<_> = (0..callers)
                .map(|caller| {
                    let (door, start) = (&door, &start);
                    scope.spawn(move || {
                        start.wait();
                        let turns = (0..each).map(|turn| {
                            let seed = u64::from(caller) * 256 + u64::from(turn);
                            leased(door, &[caller, turn], seed)
                        });
                        turns.collect::<Vec<_>>()
                    })
                })
                .collect();
            let joined = asked.into_iter().map(|caller| caller.join());
            joined
                .flat_map(|leases| leases.expect("a caller's thread"))
                .collect()
        });
        let all = usize::from(callers) * usize::from(each);
        let ids: BTreeSet<u64> = leases.iter().map(|&(id, _)| id.0).collect();
        assert_eq!(
            (leases.len(), ids.len()),
            (all, all),
            "a lease a decision, no id twice"
        );
        assert_eq!(LeasesStore::live(&*store, GATEWAY), all);
        for (member, count) in booked(&store, Role::Single) {
            let books = leases.iter().filter(|&&(_, first)| first == member).count();
            assert_eq!(usize::try_from(count).ok(), Some(books), "{member:?}");
        }
        assert_eq!(door.faults(), 0);
    }

    /// A model no member serves has no pool: the door's decision is the
    /// engine's shed, and no lease is filed.
    #[test]
    fn a_model_no_member_serves_is_shed_and_files_no_lease() {
        let store = fleet(1);
        let door = Door::new(Arc::clone(&store));
        let mut asked = request(b"r-1", 7);
        asked.facts.model = "model-none";
        let decided = door.decide(GATEWAY, &asked, Instant::now());
        let decided = decided.expect("a decision");
        assert!(decided.decision.pick().is_none(), "no pool, no pick");
        assert!(decided.lease.is_none());
        assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
    }

    #[test]
    fn a_pair_plans_prefill_is_decided_with_a_lease_a_side() {
        for encode in [false, true] {
            let store = pair_fleet(1, 1).expect("a pair fleet");
            if encode {
                apply_epd(&store).expect("an encode pair plan");
            }
            let door = Door::new(Arc::clone(&store));
            if !encode {
                store.spend_ids(0);
            }
            let mut asked = request(b"pair", 7);
            asked.facts.phase = Phase::Prefill;
            let now = Instant::now();
            let answer = door
                .decide(GATEWAY, &asked, now)
                .expect("a pair's decision");
            assert!(matches!(answer.decision, Decision::Pick(_)), "{answer:?}");
            let pick = answer.decision.pick().expect("a pair pick");
            let target = pick.targets.first().expect("the first pair");
            let pair = target.pair.expect("the target names a decode");
            let leased = answer.lease.expect("the pair's lease");
            assert_eq!(LeasesStore::live(&*store, GATEWAY), 1);
            let filed = store
                .sides(GATEWAY, leased.lease.id)
                .expect("the lease is on file");
            assert_eq!(filed.record.term, if encode { 1 } else { 2 });
            let (decode, state) = filed.side(Side::Decode).expect("the decode side");
            assert_eq!(filed.record.decode, Some(DecodeMark));
            assert!(answer.pair());
            assert_eq!(
                (leased.lease.side, leased.lease.booked),
                (Side::Prefill, target.id)
            );
            assert_eq!(
                (decode.id, decode.side, decode.booked),
                (leased.lease.id, Side::Decode, pair.decode_id)
            );
            assert_eq!(
                (filed.record.state, state),
                (LeaseState::ISSUED, LeaseState::ISSUED)
            );
            assert_eq!(leased.due, now + DISPATCH_TTL);
            assert_eq!(filed.record.dispatch_deadline, leased.due);
            assert_eq!(pair_counts(&store, [target.id, pair.decode_id]), [1, 1]);
        }
    }

    #[test]
    fn a_pair_asked_with_its_decode_later_books_the_prefill_alone() {
        const PICK_FAULT: u32 = 1;
        const ISSUE_FAULT: u32 = 2;

        #[derive(Debug)]
        struct Hears(Arc<Mutex<Vec<Side>>>);

        impl Lifecycle for Hears {
            fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
                self.0.lock().push(lease.side);
                assert_ne!(lease.tokens, ISSUE_FAULT, "issue hook fault");
            }
        }

        impl Picker for Hears {
            fn name(&self) -> &'static str {
                "hears-pair"
            }

            fn pick(
                &self,
                facts: &RequestFacts<'_>,
                rows: &[Candidate],
                kept: &[u32],
                scores: &[f32],
                band: TieBand,
                out: &mut Targets,
            ) {
                assert_ne!(facts.prompt_tokens, PICK_FAULT, "picker fault");
                Best.pick(facts, rows, kept, scores, band, out);
            }
        }

        let store = pair_fleet(1, 1).expect("a pair fleet");
        let heard = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&heard);
        let mut catalog = Catalog::builtin();
        catalog
            .picker("hears-pair", move || Box::new(Hears(Arc::clone(&kept))))
            .expect("a free picker name");
        let mut document = PolicySpec::pair_presets("least_load", "least_load").expect("a spec");
        document.generation = 2;
        for profile in document.profiles.values_mut() {
            profile.filter.clear();
        }
        document
            .profiles
            .get_mut("decode")
            .expect("decode profile")
            .pick = "hears-pair".into();
        let digest = document.digest().expect("a digest");
        let spec =
            AppliedSpec::validate_digested(&document, digest, &catalog, &[]).expect("a valid spec");
        store.apply(spec).expect("a new generation");
        let door = Door::new(Arc::clone(&store));
        let now = Instant::now();
        let mut asked = later_request(b"pair");
        for (later, state, count) in [
            (true, LeaseState::Chosen, 0),
            (false, LeaseState::ISSUED, 1),
        ] {
            asked.constraints.decode_later = later;
            let answer = door.decide(GATEWAY, &asked, now).expect("a pair");
            let lease = answer.lease.expect("a lease").lease.id;
            let target = answer
                .decision
                .pick()
                .expect("the first pick")
                .targets
                .first()
                .expect("a target");
            let decode = target.pair.expect("a decode in the pick").decode_id;
            let filed = store.sides(GATEWAY, lease).expect("both sides on file");
            assert_eq!(filed.record.state, LeaseState::ISSUED);
            assert_eq!(
                filed.decode.map(|side| (side.state, side.target)),
                Some((state, decode))
            );
            assert_eq!(pair_counts(&store, [target.id, decode]), [1, count]);
            let issued = if later {
                &[Side::Prefill][..]
            } else {
                &[Side::Prefill, Side::Decode][..]
            };
            assert_eq!(heard.lock().as_slice(), issued);
            heard.lock().clear();
            if later {
                door.report(GATEWAY, lease, pair_dispatch(Side::Prefill, now));
                let before = store.sides(GATEWAY, lease);
                let mut call = asked;
                call.facts.prompt_tokens = PICK_FAULT;
                assert_eq!(
                    door.decide_decode(GATEWAY, &call, lease, now).err(),
                    Some(Refused::Faulted)
                );
                assert_eq!(store.sides(GATEWAY, lease), before);
                assert_eq!(door.faults(), 1);
                assert!(heard.lock().is_empty());

                call.facts.prompt_tokens = ISSUE_FAULT;
                let answer = door
                    .decide_decode(GATEWAY, &call, lease, now)
                    .expect("the issued decode still has an answer");
                assert_eq!(
                    answer.lease.map(|leased| leased.lease.side),
                    Some(Side::Decode)
                );
                assert_eq!(pair_counts(&store, [target.id, decode]), [1, 1]);
                assert_eq!(heard.lock().as_slice(), [Side::Decode]);
                assert_eq!(store.lifecycle_faults(), 1);
                heard.lock().clear();
            }
            door.report(
                GATEWAY,
                lease,
                Said::Released(None, ReleaseReason::ClientGone),
            );
        }
        assert!(!Constraints::default().decode_later);
    }

    #[test]
    fn the_call_issues_the_decode_on_the_loads_of_its_instant() {
        let store = pair_fleet(1, 2).expect("a pair fleet");
        let door = Door::new(Arc::clone(&store));
        let now = Instant::now();
        let mut asked = later_request(b"pair");
        let answer = door.decide(GATEWAY, &asked, now).expect("the pair");
        let lease = answer.lease.expect("a lease").lease.id;
        let pick = answer.decision.pick().expect("the initial pick");
        let [first, other] = pick.targets.as_slice() else {
            panic!("the pair has two decode targets");
        };
        let first_decode = first.pair.expect("the first decode").decode_id;
        let other_decode = other.pair.expect("the alternate decode").decode_id;
        let workers = [first.id, first_decode, other_decode];
        let sent = now + Duration::from_millis(1);
        door.report(GATEWAY, lease, pair_dispatch(Side::Prefill, sent));
        let call = load_decodes(
            &store,
            [first_decode, other_decode],
            now + Duration::from_secs(8),
        );
        asked.constraints.decode_later = false;
        asked.facts.session_key = Identity::plain("decode-key");
        asked.facts.prompt_tokens = 321;
        let before = store.sides(GATEWAY, lease).expect("the held prefill");
        let answer = door
            .decide_decode(GATEWAY, &asked, lease, call)
            .expect("a decode decision");
        let leased = answer.lease.expect("the decode lease");
        assert_eq!(
            (leased.lease.id, leased.lease.side, leased.lease.booked),
            (lease, Side::Decode, other_decode)
        );
        assert_eq!(leased.due, call + Duration::from_secs(5));
        let after = store.sides(GATEWAY, lease).expect("both sides on file");
        assert_eq!(after.record.state, before.record.state);
        assert_eq!(after.record.target, first.id);
        assert_eq!(after.record.dispatch_deadline, leased.due);
        let decode = after.decode.expect("the issued decode");
        let expected_session = RequestFacts {
            phase: Phase::Decode,
            ..asked.facts
        }
        .session();
        assert_eq!(
            (decode.state, decode.session, decode.tokens),
            (LeaseState::ISSUED, expected_session, 321)
        );
        assert_eq!(
            (leased.lease.session, leased.lease.tokens),
            (expected_session, 321)
        );
        assert_eq!(decode.target_at(0).map(|(id, _)| id), Some(other_decode));
        assert_eq!(decode.target_at(1).map(|(id, _)| id), Some(first_decode));
        assert_eq!(pair_counts(&store, workers), [1, 0, 1]);
        assert_eq!(
            answer
                .decision
                .pick()
                .and_then(|pick| pick.targets.first())
                .and_then(|target| target.pair)
                .map(|pair| pair.decode_id),
            Some(other_decode)
        );
    }

    fn pair_dispatch(side: Side, at: Instant) -> Said {
        Said::Dispatched(
            Dispatch {
                side,
                target_index: 0,
                since_decision_micros: 0,
            },
            at,
        )
    }

    /// One prefill and two decodes. A decode call shows its profile the
    /// decode pool's facts beside the decode rows: the pool the profile
    /// places in, and never the prefills'.
    #[test]
    fn a_decode_call_shows_its_profile_the_decode_pools_facts() {
        #[derive(Debug)]
        struct Shown(Arc<Mutex<Vec<(u32, usize)>>>);

        impl Lifecycle for Shown {}

        impl Filter for Shown {
            fn name(&self) -> &'static str {
                "shown-pool"
            }

            fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
                true
            }

            fn scope(&self) -> Option<&dyn DecisionScope> {
                Some(self)
            }
        }

        impl DecisionScope for Shown {
            fn open(&self, _: &RequestFacts<'_>, rows: &[Candidate], pool: &PoolFacts<'_>) {
                self.0.lock().push((pool.pool_size, rows.len()));
            }

            fn close(&self) {}
        }

        let store = pair_fleet(1, 2).expect("one prefill and two decodes");
        let shown = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&shown);
        let mut catalog = Catalog::builtin();
        catalog
            .filter("shown-pool", move || Box::new(Shown(Arc::clone(&kept))))
            .expect("a free filter name");
        let mut document = PolicySpec::pair_presets("least_load", "least_load").expect("a spec");
        document.generation = 2;
        for profile in document.profiles.values_mut() {
            profile.filter.clear();
        }
        let decode = document.profiles.get_mut("decode");
        let filters = &mut decode.expect("the decode profile").filter;
        filters.push("shown-pool".into());
        let digest = document.digest().expect("a digest");
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[]);
        let applied = store.apply(spec.expect("a valid spec"));
        applied.expect("a later generation");
        let door = Door::new(store);
        let now = Instant::now();
        let asked = later_request(b"pool");
        let first = door.decide(GATEWAY, &asked, now).expect("a pair");
        let lease = first.lease.expect("a lease").lease.id;
        door.report(GATEWAY, lease, pair_dispatch(Side::Prefill, now));
        shown.lock().clear();
        let called = door.decide_decode(GATEWAY, &asked, lease, now);
        assert!(called.expect("a decode").decision.pick().is_some());
        // The pool's size as its facts say it, and the rows read.
        assert_eq!(shown.lock().as_slice(), [(2, 2)]);
    }

    /// A pair's decode asked later and sent with no decode call goes out
    /// as the first pick's, here on its second target. The stages hear the
    /// side's issue ahead of its dispatch, as they hear every side's: the
    /// pick as the lease's record keeps it, under a lease that names the
    /// target taken. The issue is dated when the store is told, here
    /// three seconds behind the dispatch the report names, and never by
    /// the gateway's date of that dispatch. The dispatch's own word then
    /// names that worker as the one the issue booked, under the lease's
    /// own instant.
    #[test]
    fn a_decode_sent_with_no_call_is_told_issued_ahead_of_its_dispatch() {
        let store = pair_fleet(1, 2).expect("one prefill and two decodes");
        let heard = hearing_decode(&store);
        let door = Door::new(store);
        let now = Instant::now();
        let asked = later_request(b"pair");
        let first = door.decide(GATEWAY, &asked, now).expect("a pair");
        let decodes = decodes_of(&first.decision);
        assert_eq!(decodes.len(), 2, "a decode a target");
        let leased = first.lease.expect("a lease").lease;
        door.report(GATEWAY, leased.id, pair_dispatch(Side::Prefill, now));
        // The side is chosen, not issued: no word of it yet.
        assert_eq!(heard.lock().as_slice(), []);
        let sent = Dispatch {
            side: Side::Decode,
            target_index: 1,
            since_decision_micros: 250,
        };
        let told = now + Duration::from_secs(3);
        door.report(GATEWAY, leased.id, Said::Dispatched(sent, told));
        let issued = leased.issued_micros;
        assert_eq!(
            heard.lock().as_slice(),
            [
                ("issued", 1, decodes[1], issued + 3_000_000, decodes.clone()),
                ("dispatched", 1, decodes[1], issued, vec![decodes[1]]),
            ]
        );
    }

    /// A decode call made three seconds behind its pair's first decision.
    /// The stages hear the side's issue dated at the call, when the store
    /// files the side. The lease the call hands back keeps the first
    /// decision's instant, as the lease on file does for its other words.
    #[test]
    fn a_decode_call_dates_its_issue_at_the_call() {
        let store = pair_fleet(1, 2).expect("one prefill and two decodes");
        let heard = hearing_decode(&store);
        let door = Door::new(store);
        let now = Instant::now();
        let asked = later_request(b"pair");
        let first = door.decide(GATEWAY, &asked, now).expect("a pair");
        let leased = first.lease.expect("a lease").lease;
        door.report(GATEWAY, leased.id, pair_dispatch(Side::Prefill, now));
        let called = now + Duration::from_secs(3);
        let decided = door
            .decide_decode(GATEWAY, &asked, leased.id, called)
            .expect("a decode");
        let decodes = decodes_of(&decided.decision);
        let side = decided.lease.expect("the decode's lease").lease;
        let issued = leased.issued_micros;
        assert_eq!((side.side, side.issued_micros), (Side::Decode, issued));
        assert_eq!(
            heard.lock().as_slice(),
            [("issued", 0, decodes[0], issued + 3_000_000, decodes.clone())]
        );
    }

    /// The two ranks of one engine are two rows, each under its own key
    /// and id. The second rank is the lighter: the pair's first pick sends
    /// the prefill there, and the decode call names the held prefill at
    /// that same rank.
    #[test]
    fn a_decode_call_names_the_held_prefill_at_its_own_rank() {
        let store = pair_fleet(0, 1).expect("a decode worker under the pair plan");
        let rank = |dp_rank: u16| {
            let facts = WorkerFacts {
                key: format!("http://p-ranks:8000/{dp_rank}"),
                engine: "http://p-ranks:8000".into(),
                incarnation: 1,
                role: Role::Prefill,
                models: vec![MODEL.to_owned()],
                dp_rank,
                dp_size: 2,
                max_running: Some(4_096),
                ..WorkerFacts::default()
            };
            ready(&store, facts)
        };
        let ranks = [rank(0), rank(1)];
        let at = Instant::now();
        for (worker, dp_rank, queued) in [(ranks[0], 0, 80), (ranks[1], 1, 0)] {
            let row = LoadRow {
                queued,
                live_rate: 2_000,
                ..LoadRow::default()
            };
            store.record(worker, dp_rank, row, at, LoadSource::Stream);
        }
        let now = store.publish_due().map_or(at, |due| due.max(at));
        store.publish(now);
        let door = Door::new(store);
        let asked = later_request(b"ranks");
        // Every target of a pick names its prefill, by worker and rank.
        let prefills = |decision: &Decision| -> Vec<(WorkerId, u16)> {
            let pick = decision.pick();
            pick.into_iter()
                .flat_map(|pick| pick.targets.iter())
                .map(|target| (target.id, target.dp_rank))
                .collect()
        };
        let first = door.decide(GATEWAY, &asked, now).expect("a pair");
        assert_eq!(prefills(&first.decision), [(ranks[1], 1)]);
        let lease = first.lease.expect("a lease").lease.id;
        door.report(GATEWAY, lease, pair_dispatch(Side::Prefill, now));
        let called = door
            .decide_decode(GATEWAY, &asked, lease, now)
            .expect("a decode");
        assert_eq!(prefills(&called.decision), [(ranks[1], 1)]);
    }

    #[test]
    fn the_call_is_refused_by_name_and_moves_nothing() {
        let store = pair_fleet(1, 1).expect("a pair fleet");
        let door = Door::new(Arc::clone(&store));
        let now = Instant::now();
        let asked = later_request(b"pair");
        let answer = door.decide(GATEWAY, &asked, now).expect("a pair");
        let lease = answer.lease.expect("a lease").lease.id;
        let filed = store.sides(GATEWAY, lease).expect("both sides");
        let decode = filed.decode.expect("the chosen side");
        let workers = [filed.record.target, decode.target];
        let refused = |gateway, id, expected| {
            let (sides, counts) = (store.sides(GATEWAY, id), pair_counts(&store, workers));
            for subset_hint in [None, Some(0)] {
                let mut request = asked;
                request.constraints.subset_hint = subset_hint;
                assert_eq!(
                    door.decide_decode(gateway, &request, id, now).err(),
                    Some(Refused::Decode(expected))
                );
            }
            assert_eq!(store.sides(GATEWAY, id), sides);
            assert_eq!(pair_counts(&store, workers), counts);
        };
        refused(GATEWAY, LeaseId(0), DecodeRefused::NoLease);
        refused(GatewayId(GATEWAY.0 + 1), lease, DecodeRefused::NoLease);
        refused(GATEWAY, lease, DecodeRefused::PrefillNotHeld);

        door.report(GATEWAY, lease, pair_dispatch(Side::Prefill, now));
        let _ = door
            .decide_decode(GATEWAY, &asked, lease, now)
            .expect("a decode");
        refused(GATEWAY, lease, DecodeRefused::DecodeStands);
        door.report(GATEWAY, lease, pair_dispatch(Side::Decode, now));
        refused(GATEWAY, lease, DecodeRefused::DecodeStands);

        let fallback = door
            .decide(GATEWAY, &asked, now)
            .expect("another pair")
            .lease
            .expect("a lease")
            .lease
            .id;
        door.report(GATEWAY, fallback, pair_dispatch(Side::Prefill, now));
        door.report(GATEWAY, fallback, pair_dispatch(Side::Decode, now));
        refused(GATEWAY, fallback, DecodeRefused::DecodeStands);

        let mut ordinary = asked;
        ordinary.constraints.decode_later = false;
        let ordinary = door
            .decide(GATEWAY, &ordinary, now)
            .expect("an ordinary pair")
            .lease
            .expect("a lease")
            .lease
            .id;
        door.report(GATEWAY, ordinary, pair_dispatch(Side::Prefill, now));
        refused(GATEWAY, ordinary, DecodeRefused::NotChosen);

        let ending = door
            .decide(GATEWAY, &asked, now)
            .expect("a pair")
            .lease
            .expect("a lease")
            .lease
            .id;
        let ending_record = LeasesStore::get(&*store, GATEWAY, ending).expect("a prefill on file");
        door.report(GATEWAY, ending, pair_dispatch(Side::Prefill, now));
        door.report(
            GATEWAY,
            ending,
            Said::Completed(Outcome {
                side: Side::Prefill,
                served: filed.record.target,
                status: 200,
                class: TransportClass::Ok,
                ttft_micros: 0,
                tokens_in: 0,
                tokens_out: 0,
                duration_micros: 0,
            }),
        );
        refused(GATEWAY, ending, DecodeRefused::NoLease);
        let ended = LeaseRecord {
            state: LeaseState::Completed,
            ..ending_record
        };
        LeasesStore::issue_pair(&*store, ended, decode).expect("an ended prefill still on file");
        refused(GATEWAY, ending, DecodeRefused::PrefillNotHeld);
    }

    #[test]
    fn a_shed_at_the_call_leaves_the_side_chosen() {
        let store = pair_fleet(1, 1).expect("a pair fleet");
        let door = Door::new(Arc::clone(&store));
        let now = Instant::now();
        let mut asked = later_request(b"pair");
        let first = door.decide(GATEWAY, &asked, now).expect("the pair");
        let lease = first.lease.expect("a lease").lease.id;
        door.report(GATEWAY, lease, pair_dispatch(Side::Prefill, now));
        let before = store.sides(GATEWAY, lease).expect("both sides");
        let decode = before.decode.expect("the chosen decode");
        let workers = [before.record.target, decode.target];
        asked.constraints.subset_hint = Some(0);
        let shed = door
            .decide_decode(GATEWAY, &asked, lease, now)
            .expect("a shed decision");
        let Decision::Shed(reason) = shed.decision else {
            panic!("the subset left no rows: {shed:?}");
        };
        assert_eq!(
            (reason.reason, reason.emptied_by),
            (ShedReason::NoEndpoints, "subset-hint")
        );
        assert!(shed.lease.is_none());
        assert_eq!(store.sides(GATEWAY, lease), Some(before));
        assert_eq!(decode.state, LeaseState::Chosen);
        assert_eq!(pair_counts(&store, workers), [1, 0]);
        door.report(GATEWAY, lease, pair_dispatch(Side::Decode, now));
        let decode = store
            .sides(GATEWAY, lease)
            .and_then(|sides| sides.decode)
            .expect("the dispatched decode");
        assert_eq!(
            (decode.state, decode.target),
            (LeaseState::Dispatched { at: now }, workers[1])
        );
        assert_eq!(pair_counts(&store, workers), [1, 1]);
    }

    /// A word of a pair's decode side as a stage hears it: its name, the
    /// lease's target index, the worker the lease books, the lease's
    /// instant and the decodes the word names.
    type DecodeWord = (&'static str, u8, WorkerId, u64, Vec<WorkerId>);

    /// The decode each target of a pick names, in the pick's order.
    fn decodes_of(decision: &Decision) -> Vec<WorkerId> {
        let targets = decision
            .pick()
            .into_iter()
            .flat_map(|pick| pick.targets.iter());
        targets
            .filter_map(|target| target.pair)
            .map(|pair| pair.decode_id)
            .collect()
    }

    /// Applies the pair presets to `store` as generation 2, the decode side
    /// picking through a stage that keeps every issue and every dispatch
    /// it hears of a decode side: the words are returned.
    fn hearing_decode(store: &MemoryStore) -> Arc<Mutex<Vec<DecodeWord>>> {
        #[derive(Debug)]
        struct Hears(Arc<Mutex<Vec<DecodeWord>>>);

        impl Lifecycle for Hears {
            fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
                if lease.side == Side::Decode {
                    let named = decodes_of(decision);
                    let word = ("issued", lease.target_index, lease.booked);
                    let mut heard = self.0.lock();
                    heard.push((word.0, word.1, word.2, lease.issued_micros, named));
                }
            }

            fn on_dispatched(&self, lease: &Lease, target: &Target, side: Side, _: u32) {
                if side == Side::Decode {
                    let word = ("dispatched", lease.target_index, lease.booked);
                    let mut heard = self.0.lock();
                    heard.push((word.0, word.1, word.2, lease.issued_micros, vec![target.id]));
                }
            }
        }

        impl Picker for Hears {
            fn name(&self) -> &'static str {
                "hears-decode"
            }

            fn pick(
                &self,
                facts: &RequestFacts<'_>,
                rows: &[Candidate],
                kept: &[u32],
                scores: &[f32],
                band: TieBand,
                out: &mut Targets,
            ) {
                Best.pick(facts, rows, kept, scores, band, out);
            }
        }

        let heard = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&heard);
        let mut catalog = Catalog::builtin();
        let added = catalog.picker("hears-decode", move || Box::new(Hears(Arc::clone(&kept))));
        added.expect("a free picker name");
        let mut document = PolicySpec::pair_presets("least_load", "least_load").expect("a spec");
        document.generation = 2;
        for profile in document.profiles.values_mut() {
            profile.filter.clear();
        }
        let decode = document.profiles.get_mut("decode");
        decode.expect("the decode profile").pick = "hears-decode".into();
        let digest = document.digest().expect("a digest");
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[]);
        let applied = store.apply(spec.expect("a valid spec"));
        applied.expect("a later generation");
        heard
    }

    /// Applies the pair presets to `store` as `generation`, the decode side
    /// picking through a stage that keeps every word it hears of leases in
    /// `heard`: a stage hears the words of both sides.
    fn hearing_pair(store: &MemoryStore, generation: u64, heard: &Arc<Mutex<Vec<&'static str>>>) {
        #[derive(Debug)]
        struct Hears(Arc<Mutex<Vec<&'static str>>>);

        impl Lifecycle for Hears {
            fn on_decision_issued(&self, _: &Lease, _: &Decision) {
                self.0.lock().push("issued");
            }

            fn on_dispatched(&self, _: &Lease, _: &Target, _: Side, _: u32) {
                self.0.lock().push("dispatched");
            }

            fn on_completed(&self, _: &Lease, _: &Outcome) {
                self.0.lock().push("completed");
            }

            fn on_released(&self, _: &Lease, _: ReleaseReason) {
                self.0.lock().push("released");
            }
        }

        impl Picker for Hears {
            fn name(&self) -> &'static str {
                "hears-all"
            }

            fn pick(
                &self,
                facts: &RequestFacts<'_>,
                rows: &[Candidate],
                kept: &[u32],
                scores: &[f32],
                band: TieBand,
                out: &mut Targets,
            ) {
                Best.pick(facts, rows, kept, scores, band, out);
            }
        }

        let kept = Arc::clone(heard);
        let mut catalog = Catalog::builtin();
        let added = catalog.picker("hears-all", move || Box::new(Hears(Arc::clone(&kept))));
        added.expect("a free picker name");
        let mut document = PolicySpec::pair_presets("least_load", "least_load").expect("a spec");
        document.generation = generation;
        for profile in document.profiles.values_mut() {
            profile.filter.clear();
        }
        let decode = document.profiles.get_mut("decode");
        decode.expect("the decode profile").pick = "hears-all".into();
        let digest = document.digest().expect("a digest");
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[]);
        let applied = store.apply(spec.expect("a valid spec"));
        applied.expect("a later generation");
    }

    /// A reload between a pair's two decisions. The generation of the
    /// call's instant decides the side, and its engine hears nothing of
    /// the lease: an older engine heard the lease's issue. One rule says
    /// which engine hears a lease's word, at the call as in the telling.
    #[test]
    fn a_decode_called_behind_a_reload_is_told_to_no_engine() {
        let store = pair_fleet(1, 1).expect("a pair fleet");
        let first = Arc::new(Mutex::new(Vec::new()));
        hearing_pair(&store, 2, &first);
        let door = Door::new(Arc::clone(&store));
        let now = Instant::now();
        let asked = later_request(b"pair");
        let answer = door.decide(GATEWAY, &asked, now).expect("the pair");
        let lease = answer.lease.expect("a lease").lease.id;
        door.report(GATEWAY, lease, pair_dispatch(Side::Prefill, now));
        assert_eq!(*first.lock(), ["issued", "dispatched"]);
        let reloaded = Arc::new(Mutex::new(Vec::new()));
        hearing_pair(&store, 3, &reloaded);
        let answer = door.decide_decode(GATEWAY, &asked, lease, now);
        let leased = answer.expect("a decode decision").lease;
        assert_eq!(leased.map(|leased| leased.lease.side), Some(Side::Decode));
        door.report(GATEWAY, lease, pair_dispatch(Side::Decode, now));
        let gone = Said::Released(None, ReleaseReason::ClientGone);
        door.report(GATEWAY, lease, gone);
        assert!(store.sides(GATEWAY, lease).is_none());
        assert_eq!(*reloaded.lock(), [""; 0]);
        assert_eq!(*first.lock(), ["issued", "dispatched"]);
    }

    /// A caller that stands at its seat until the case sends it on: it
    /// says when it stands there, with its seat's file at work.
    struct Stands {
        stood: mpsc::Sender<()>,
        go: Mutex<mpsc::Receiver<()>>,
    }

    impl Maker for Stands {
        fn model(&self) -> &str {
            MODEL
        }

        fn make<'w>(&'w self, _: &'w World) -> Request<'w> {
            self.stood.send(()).expect("the case waits for the seat");
            let word = self.go.lock().recv_timeout(PATIENCE);
            word.expect("the case sends the caller on");
            request(b"held", 7)
        }
    }

    /// A decode call decides at the seat of its lease's file, behind the
    /// decision that stands there. A prefill's dispatch handed while
    /// another caller decided at that file waits for that caller, and the
    /// call reads it as sent: from a free seat of its own it would read
    /// the file as it stood and be refused.
    #[test]
    fn a_decode_call_decides_behind_what_waits_at_its_leases_file() {
        let files = MemoryStore::new().with_deciders(2);
        let store = pair_fleet_over(files, 1, 1).expect("a pair fleet of two files");
        let door = Arc::new(Door::new(Arc::clone(&store)));
        let now = Instant::now();
        seated(0);
        let answer = door.decide(GATEWAY, &later_request(b"pair"), now);
        let lease = answer.expect("the pair").lease.expect("a lease").lease.id;
        assert_eq!(store.file_of(lease), 0);
        // Another caller decides at the lease's file and stands there.
        let (stood, at_seat) = mpsc::channel();
        let (word, go) = mpsc::channel();
        let other = Arc::clone(&door);
        let other = thread::spawn(move || {
            seated(0);
            let go = Mutex::new(go);
            let held = other.decide_under(GATEWAY, &Stands { stood, go }, Instant::now());
            held.is_ok()
        });
        at_seat.recv_timeout(PATIENCE).expect("the other caller");
        // The prefill's dispatch is handed meanwhile and waits at the file.
        door.report(GATEWAY, lease, pair_dispatch(Side::Prefill, now));
        let prefill =
            |store: &MemoryStore| store.sides(GATEWAY, lease).map(|sides| sides.record.state);
        assert_eq!(prefill(&store), Some(LeaseState::ISSUED));
        // The call comes from a thread whose own seat is free.
        let (answered, answer) = mpsc::channel();
        let caller = Arc::clone(&door);
        let caller = thread::spawn(move || {
            seated(1);
            let decided = caller.decide_decode(GATEWAY, &later_request(b"pair"), lease, now);
            let side = decided.map(|decided| decided.lease.map(|leased| leased.lease.side));
            answered.send(side).ok();
        });
        let early = answer.recv_timeout(Duration::from_millis(100));
        assert_eq!(early, Err(mpsc::RecvTimeoutError::Timeout));
        word.send(()).expect("the other caller waits");
        assert_eq!(answer.recv_timeout(PATIENCE), Ok(Ok(Some(Side::Decode))));
        assert_eq!(other.join().ok(), Some(true));
        assert!(caller.join().is_ok());
    }

    /// A reload turns the keeping of deciders on between a pair's two
    /// decisions, so the call decides at its model's seat while its lease
    /// stands at the file of the seat its first decision took. A prefill's
    /// dispatch handed while another caller decided at that file waits for
    /// that caller. The call stands behind that caller first and reads the
    /// dispatch as sent: straight from its model's seat it would read the
    /// file as it stood and be refused.
    #[test]
    fn a_decode_call_under_kept_deciders_first_applies_what_waits_at_its_leases_file() {
        // A count of files under which the model's seat is not the first,
        // where the pair's first decision is made.
        let files = (2..=8).find(|&files| decider_of(MODEL, files) != 0);
        let files = files.expect("the model names a later file under some count");
        let home = decider_of(MODEL, files);
        let store = MemoryStore::new().with_deciders(files);
        let store = pair_fleet_over(store, 1, 1).expect("a pair fleet");
        let door = Arc::new(Door::new(Arc::clone(&store)));
        let now = Instant::now();
        seated(0);
        let answer = door.decide(GATEWAY, &later_request(b"pair"), now);
        let lease = answer.expect("the pair").lease.expect("a lease").lease.id;
        assert_eq!(store.file_of(lease), 0, "no stage keeps state a decider");
        // Another caller decides at the lease's file and stands there.
        let (stood, at_seat) = mpsc::channel();
        let (word, go) = mpsc::channel();
        let other = Arc::clone(&door);
        let other = thread::spawn(move || {
            seated(0);
            let go = Mutex::new(go);
            let held = other.decide_under(GATEWAY, &Stands { stood, go }, Instant::now());
            held.is_ok()
        });
        at_seat.recv_timeout(PATIENCE).expect("the other caller");
        // The reload: the decode profile picks through a stage that keeps
        // state a decider. The next decision finds the stage and leaves
        // the door's word that requests decide at their model's seat.
        let mut catalog = Catalog::builtin();
        let added = catalog.picker("scoped", || Box::new(Scoped));
        added.expect("a free picker name");
        let mut document = PolicySpec::pair_presets("least_load", "least_load").expect("a spec");
        document.generation = 3;
        for profile in document.profiles.values_mut() {
            profile.filter.clear();
        }
        let decode = document.profiles.get_mut("decode");
        decode.expect("the decode profile").pick = "scoped".into();
        let digest = document.digest().expect("a digest");
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[]);
        let applied = store.apply(spec.expect("a valid spec"));
        applied.expect("a later generation");
        let kept = door.decide(GATEWAY, &later_request(b"kept"), now);
        let kept = kept
            .expect("a pair under the reload")
            .lease
            .expect("a lease");
        assert_eq!(store.file_of(kept.lease.id), home, "the model's seat");
        // The prefill's dispatch is handed meanwhile and waits at the file.
        door.report(GATEWAY, lease, pair_dispatch(Side::Prefill, now));
        let prefill =
            |store: &MemoryStore| store.sides(GATEWAY, lease).map(|sides| sides.record.state);
        assert_eq!(prefill(&store), Some(LeaseState::ISSUED));
        // The call's own seat is its model's, and that seat is free.
        let (answered, answer) = mpsc::channel();
        let caller = Arc::clone(&door);
        let caller = thread::spawn(move || {
            seated(1);
            let decided = caller.decide_decode(GATEWAY, &later_request(b"pair"), lease, now);
            let side = decided.map(|decided| decided.lease.map(|leased| leased.lease.side));
            answered.send(side).ok();
        });
        let early = answer.recv_timeout(Duration::from_millis(100));
        assert_eq!(early, Err(mpsc::RecvTimeoutError::Timeout));
        word.send(()).expect("the other caller waits");
        assert_eq!(answer.recv_timeout(PATIENCE), Ok(Ok(Some(Side::Decode))));
        assert_eq!(other.join().ok(), Some(true));
        assert!(caller.join().is_ok());
    }

    #[test]
    fn a_decode_sent_on_its_own_is_refused_by_name() {
        let store = pair_fleet(1, 1).expect("a pair fleet");
        let door = Door::new(Arc::clone(&store));
        let mut asked = request(b"side", 7);
        asked.facts.phase = Phase::Decode;
        assert_eq!(
            door.decide(GATEWAY, &asked, Instant::now()).err(),
            Some(Refused::PairSide)
        );
        assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);

        apply_epd(&store).expect("an encode pair plan");
        for phase in [Phase::Decode, Phase::Encode] {
            asked.facts.phase = phase;
            assert_eq!(
                door.decide(GATEWAY, &asked, Instant::now()).err(),
                Some(Refused::PairSide)
            );
            assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
        }
    }

    #[test]
    fn an_encode_pair_places_each_item_on_an_encoder_of_the_chosen_prefills_class(
    ) -> Result<(), Box<dyn Error>> {
        let door = encode_door()?;
        let store = door.store();
        let classes = [("p0", "e0", "c0"), ("p1", "e1", "c1")].map(|(p, e, c)| {
            let prefill = encode_member(store, &format!("grpc://{p}:8000"), Role::Prefill, c);
            let encoder = encode_member(store, &format!("grpc://{e}:8000"), Role::Encode, c);
            (prefill, encoder)
        });
        assert!(door.seats[0].slot.lock().buffers.encode.is_none());
        store.spend_ids(0);
        let now = Instant::now();
        let mut placed = EncodePlaced::new();
        let decided = door
            .decide_encode_pair(GATEWAY, &pair_request(b"pair"), items(), &mut placed, now)
            .expect("a decision");
        let pick = decided.decision.pick().expect("a pair pick");
        let target = pick.targets.first().expect("the first pair");
        let pair = target.pair.expect("the decode");
        let (_, encoder) = classes
            .into_iter()
            .find(|&(prefill, _)| prefill == target.id)
            .expect("the prefill's class");
        let encoders: Vec<_> = placed.encoders().iter().map(|placed| placed.id).collect();
        assert_eq!(encoders, [encoder, encoder]);
        assert!(!placed.one());
        let leased = decided.lease.expect("the prefill's lease");
        let filed = store
            .sides(GATEWAY, leased.lease.id)
            .expect("both sides on file");
        let (decode, state) = filed.side(Side::Decode).expect("the decode side");
        assert_eq!(
            (leased.lease.side, leased.lease.booked),
            (Side::Prefill, target.id)
        );
        assert_eq!(
            (decode.id, decode.booked),
            (leased.lease.id, pair.decode_id)
        );
        assert_eq!(
            (filed.record.state, state),
            (LeaseState::ISSUED, LeaseState::ISSUED)
        );
        assert_eq!(filed.record.term, 2);
        assert_eq!(pair_counts(store, [target.id, pair.decode_id]), [1, 1]);
        assert_eq!(LeasesStore::live(&**store, GATEWAY), 1);
        assert!(booked(store, Role::Encode)
            .iter()
            .all(|&(_, count)| count == 0));
        let rows = door.seats[0]
            .slot
            .lock()
            .buffers
            .encode
            .as_deref()
            .map(std::ptr::from_ref);
        assert!(rows.is_some(), "the seat keeps its encode buffers");
        let next = door
            .decide_encode_pair(GATEWAY, &pair_request(b"next"), items(), &mut placed, now)
            .expect("another decision");
        let picked = next
            .decision
            .pick()
            .and_then(|pick| pick.targets.first())
            .expect("a pair");
        assert_ne!(picked.id, target.id, "the first prefill is already booked");
        assert_eq!(
            door.seats[0]
                .slot
                .lock()
                .buffers
                .encode
                .as_deref()
                .map(std::ptr::from_ref),
            rows,
            "the same seat reuses its buffers"
        );
        Ok(())
    }

    #[test]
    fn equal_keys_are_placed_on_one_encoder() -> Result<(), Box<dyn Error>> {
        let door = encode_door()?;
        let store = door.store();
        encode_member(store, "grpc://p0:8000", Role::Prefill, "c0");
        let encoders = ["grpc://e0:8000", "grpc://e1:8000"]
            .map(|key| encode_member(store, key, Role::Encode, "c0"));
        let now = Instant::now();
        let mut placed = EncodePlaced::new();
        let mut seen = BTreeSet::new();
        for n in 0..32_u8 {
            let keys = [Identity::hashed(u64::from(n) << 59); 2];
            let items = EncodeItems {
                keys: &keys,
                ..items()
            };
            let id = [n];
            let mut asked = pair_request(&id);
            let decided = door
                .decide_encode_pair(GATEWAY, &asked, items, &mut placed, now)
                .expect("a decision");
            let [first, second] = placed.encoders() else {
                panic!("two placements from {:?}", decided.decision);
            };
            assert_eq!(first.id, second.id);
            seen.insert(first.id);
            release(&door, &decided);
            if n == 0 {
                let excluded = [first.id];
                asked.constraints.excluded = &excluded;
                let retried = door
                    .decide_encode_pair(GATEWAY, &asked, items, &mut placed, now)
                    .expect("a decision");
                assert_eq!(placed.encoders().len(), 2);
                assert!(placed
                    .encoders()
                    .iter()
                    .all(|target| target.id != excluded[0]));
                release(&door, &retried);
            }
        }
        assert_eq!(
            seen,
            BTreeSet::from(encoders),
            "different keys can take either encoder"
        );
        Ok(())
    }

    #[test]
    fn the_one_key_is_placed_in_the_items_stead_under_a_class_that_pushes(
    ) -> Result<(), Box<dyn Error>> {
        let (door, encoder) = push_fleet()?;
        let placed = place_one(&door, Instant::now()).expect("a decision");
        assert!(placed.one());
        assert_eq!(placed.encoders().len(), 1);
        assert_eq!(
            placed.encoders().first().map(|target| target.id),
            Some(encoder)
        );
        Ok(())
    }

    #[test]
    fn a_request_without_items_is_decided_as_its_pair_and_places_nothing(
    ) -> Result<(), Box<dyn Error>> {
        let door = encode_door()?;
        encode_member(door.store(), "grpc://p0:8000", Role::Prefill, "c0");
        ready(
            door.store(),
            WorkerFacts {
                wire_modes: WireModes::ZMQ,
                ..encode_facts("zmq://e1:8000", Role::Encode, "c0")
            },
        );
        let now = Instant::now();
        let mut placed = place_one(&door, now).expect("a decision");
        let had_one = placed.one();
        let mut asked = pair_request(b"no items");
        asked.constraints.wire_modes = WireModes::GRPC;
        let ordinary = door.decide(GATEWAY, &asked, now).expect("a decision");
        release(&door, &ordinary);
        let bare = door
            .decide_encode_pair(GATEWAY, &asked, EncodeItems::default(), &mut placed, now)
            .expect("a decision");
        assert_eq!(bare.decision, ordinary.decision);
        assert!(had_one && !placed.one() && placed.encoders().is_empty());
        release(&door, &bare);

        let with_decode_images = EncodeItems {
            decode_images: true,
            ..EncodeItems::default()
        };
        let images = door
            .decide_encode_pair(GATEWAY, &asked, with_decode_images, &mut placed, now)
            .expect("a decision");
        assert!(
            matches!(images.decision, Decision::Shed(shed) if shed.emptied_by == "decode-vision")
        );
        assert_eq!(door.undialed(), 0, "the pair's wires took no row");
        let keyed = door
            .decide_encode_pair(GATEWAY, &asked, items(), &mut placed, now)
            .expect("a decision");
        assert!(matches!(keyed.decision, Decision::Shed(shed) if shed.emptied_by == "wires"));
        assert_eq!(door.undialed(), 1, "the wire took the encoder");
        Ok(())
    }

    #[test]
    fn a_shed_names_its_stage_and_places_nothing() -> Result<(), Box<dyn Error>> {
        for stage in ["encoders", "encode-vision", "subset-hint", "wires"] {
            let (door, encoder) = push_fleet()?;
            let store = door.store();
            let now = Instant::now();
            let mut placed = place_one(&door, now).expect("a decision");
            let had_one = placed.one();
            let mut asked = pair_request(b"shed");
            match stage {
                "encoders" => {
                    let row = WorkersStore::get(&**store, encoder).expect("an encoder");
                    assert!(store
                        .set_status(encoder, WorkerStatus::NotReady, row.revision)
                        .is_some());
                }
                "encode-vision" => asked.facts.media.images = 1,
                "subset-hint" => {
                    asked.constraints.subset_hint = Some(0);
                    asked.constraints.wire_modes = WireModes::ZMQ;
                }
                _ => asked.constraints.wire_modes = WireModes::ZMQ,
            }
            let answer = door
                .decide_encode_pair(GATEWAY, &asked, items(), &mut placed, now)
                .expect("a decision");
            let Decision::Shed(shed) = answer.decision else {
                panic!("a shed at {stage}, got {:?}", answer.decision);
            };
            assert_eq!(
                (shed.reason, shed.emptied_by),
                (ShedReason::NoEndpoints, stage)
            );
            assert!(had_one, "the previous decision placed the one key");
            assert!(!placed.one() && placed.encoders().is_empty(), "{stage}");
            assert!(answer.lease.is_none());
            assert_eq!(LeasesStore::live(&**store, GATEWAY), 0);
            assert_eq!(door.undialed(), if stage == "wires" { 3 } else { 0 });
            if stage == "subset-hint" {
                let slot = door.seats[0].slot.lock();
                let (pair, encode) = slot
                    .buffers
                    .encode
                    .as_deref()
                    .expect("the built views")
                    .pools();
                assert!(pair.prefills.is_empty() && pair.decodes.is_empty());
                assert!(encode.encoders.is_empty() && encode.affinity.is_empty());
                assert!(encode.facts.is_empty() && encode.prefill_classes.is_empty());
            }
        }
        Ok(())
    }

    #[test]
    fn a_gateways_wires_narrow_the_encoders_with_their_facts() -> Result<(), Box<dyn Error>> {
        let door = encode_door()?;
        let store = door.store();
        encode_member(store, "grpc://p0:8000", Role::Prefill, "c0");
        let other = ready(
            store,
            WorkerFacts {
                wire_modes: WireModes::ZMQ,
                vision: true,
                ..encode_facts("zmq://e1:8000", Role::Encode, "c0")
            },
        );
        let mut own = encode_facts("grpc://e1:8000", Role::Encode, "c0");
        own.encode.references = true;
        let own = ready(store, own);
        assert!(other < own, "the cut moves the kept row to the front");
        let now = Instant::now();
        let mut asked = pair_request(b"wires");
        asked.constraints.wire_modes = WireModes::GRPC;
        let items = EncodeItems {
            references: true,
            ..items()
        };
        let mut placed = EncodePlaced::new();
        let answer = door
            .decide_encode_pair(GATEWAY, &asked, items, &mut placed, now)
            .expect("a decision");
        assert!(answer.pair(), "{:?}", answer.decision);
        assert_eq!(placed.encoders().len(), 2);
        assert!(placed
            .encoders()
            .iter()
            .all(|target| (target.id, target.position) == (own, 0)));
        assert_eq!(door.undialed(), 1);
        let slot = door.seats[0].slot.lock();
        let (_, encode) = slot
            .buffers
            .encode
            .as_deref()
            .expect("the built views")
            .pools();
        assert_eq!(encode.affinity.len(), 1, "the kept row's affinity");
        drop(slot);
        release(&door, &answer);

        asked.facts.media.images = 1;
        let answer = door
            .decide_encode_pair(GATEWAY, &asked, items, &mut placed, now)
            .expect("a decision");
        assert!(
            matches!(answer.decision, Decision::Shed(shed) if shed.emptied_by == "encode-vision")
        );
        assert!(!placed.one() && placed.encoders().is_empty());
        assert_eq!(door.undialed(), 2);
        Ok(())
    }

    #[test]
    fn a_prefill_cut_by_the_wires_takes_its_class_with_it() -> Result<(), Box<dyn Error>> {
        let door = encode_door()?;
        let store = door.store();
        let other = ready(
            store,
            WorkerFacts {
                wire_modes: WireModes::ZMQ,
                ..encode_facts("zmq://p0:8000", Role::Prefill, "c0")
            },
        );
        let own = encode_member(store, "grpc://p1:8000", Role::Prefill, "c1");
        assert!(other < own, "the cut moves the kept row to the front");
        encode_member(store, "grpc://e0:8000", Role::Encode, "c0");
        let encoder = encode_member(store, "grpc://e1:8000", Role::Encode, "c1");
        let mut asked = pair_request(b"wires");
        asked.constraints.wire_modes = WireModes::GRPC;
        let mut placed = EncodePlaced::new();
        let answer = door
            .decide_encode_pair(GATEWAY, &asked, items(), &mut placed, Instant::now())
            .expect("a decision");
        let pick = answer.decision.pick().expect("a pair pick");
        assert_eq!(
            pick.targets
                .first()
                .map(|target| (target.id, target.position)),
            Some((own, 0))
        );
        assert_eq!(placed.encoders().len(), 2);
        assert!(placed.encoders().iter().all(|target| target.id == encoder));
        Ok(())
    }

    #[test]
    fn a_single_and_a_pd_pair_request_decide_as_before_and_clear_the_placed(
    ) -> Result<(), Box<dyn Error>> {
        let (donor, _) = push_fleet()?;
        let now = Instant::now();
        for (store, asked, pair) in [
            (fleet(1), request(b"single", 7), false),
            (pair_fleet(1, 1)?, pair_request(b"pair"), true),
        ] {
            let door = Door::new(store);
            let ordinary = door.decide(GATEWAY, &asked, now).expect("a decision");
            release(&door, &ordinary);
            assert!(door.seats[0].slot.lock().buffers.encode.is_none());
            let mut placed = place_one(&donor, now).expect("a decision");
            let had_one = placed.one();
            let answer = door
                .decide_encode_pair(GATEWAY, &asked, items(), &mut placed, now)
                .expect("a decision");
            assert_eq!(answer.decision, ordinary.decision);
            assert_eq!(answer.pair(), pair);
            assert!(door.seats[0].slot.lock().buffers.encode.is_none());
            assert!(
                had_one && !placed.one() && placed.encoders().is_empty(),
                "{:?}",
                asked.facts.phase
            );
        }
        for phase in [Phase::Decode, Phase::Encode] {
            let mut asked = pair_request(b"side");
            asked.facts.phase = phase;
            let mut placed = place_one(&donor, now).expect("a decision");
            assert_eq!(
                donor
                    .decide_encode_pair(GATEWAY, &asked, items(), &mut placed, now)
                    .err(),
                Some(Refused::PairSide)
            );
            assert!(!placed.one() && placed.encoders().is_empty());
        }
        Ok(())
    }

    #[test]
    fn a_single_under_no_pair_plan_is_decided_as_before() {
        let store = fleet(1);
        let door = Door::new(Arc::clone(&store));
        let asked = request(b"single", 7);
        let answer = door
            .decide(GATEWAY, &asked, Instant::now())
            .expect("a single's decision");
        assert!(!answer.pair());
        let leased = answer.lease.expect("the single's lease");
        let filed = store
            .sides(GATEWAY, leased.lease.id)
            .expect("the lease is on file");
        assert_eq!(leased.lease.side, Side::Single);
        assert!(filed.decode.is_none());
        assert_eq!(LeasesStore::live(&*store, GATEWAY), 1);
        assert_eq!(counted(&store), [(filed.record.target, 1)]);
    }

    #[test]
    fn a_pair_whose_decode_pool_holds_no_worker_is_shed_and_files_nothing() {
        let store = pair_fleet(1, 0).expect("a prefill fleet");
        let door = Door::new(Arc::clone(&store));
        let mut asked = request(b"pair", 7);
        asked.facts.phase = Phase::Prefill;
        let answer = door
            .decide(GATEWAY, &asked, Instant::now())
            .expect("a decision");
        assert!(matches!(answer.decision, Decision::Shed(_)));
        assert!(answer.lease.is_none());
        assert!(!answer.pair());
        assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
        assert_eq!(
            booked(&store, Role::Prefill).first().map(|row| row.1),
            Some(0)
        );
    }

    /// A gateway that shares no wire with its pool's members is shed, and
    /// the door counts the candidates its wires took.
    #[test]
    fn a_gateway_whose_wires_reach_no_member_is_shed_and_counted() {
        let store = fleet(1);
        let door = Door::new(Arc::clone(&store));
        let mut asked = request(b"r-1", 7);
        asked.constraints.wire_modes = WireModes::ZMQ;
        let decided = door.decide(GATEWAY, &asked, Instant::now());
        let decided = decided.expect("a decision");
        let Decision::Shed(shed) = decided.decision else {
            panic!("the gateway can dial no member");
        };
        assert_eq!(shed.emptied_by, super::WIRES);
        assert_eq!(shed.trace.made_under(), None);
        assert!(decided.lease.is_none());
        assert_eq!((door.undialed(), door.hashes_cut()), (3, 0));
    }

    /// The door hands a decision the asking gateway's vetoes as the store
    /// keeps them. A pool it doubts on every member is steered: a doubt is
    /// not an outage. A pool it holds on every member is shed at the health
    /// stage, and one doubted member among held ones takes the request.
    #[test]
    fn a_gateways_holds_reach_the_doors_decision() {
        let store = fleet(1);
        let door = Door::new(Arc::clone(&store));
        let now = Instant::now();
        let until = now + Duration::from_secs(60);
        let members: Vec<WorkerId> = booked(&store, Role::Single)
            .iter()
            .map(|&(id, _)| id)
            .collect();
        let veto = |member: WorkerId, hold: Option<Instant>, stands: bool| {
            let written = store.gateway_veto_noted(GATEWAY, member, Some(until), hold, stands, now);
            assert_eq!(written, VetoWrite::Written);
        };
        let decide = |id: &[u8]| {
            let decided = door.decide(GATEWAY, &request(id, 7), now);
            decided.expect("a decision").decision
        };
        for &member in &members {
            veto(member, None, false);
        }
        assert!(decide(b"r-1").pick().is_some(), "a doubt is not an outage");
        for &member in &members {
            veto(member, Some(until), true);
        }
        let shed = decide(b"r-2");
        assert!(
            matches!(&shed, Decision::Shed(shed) if shed.emptied_by == "healthy"),
            "{shed:?}"
        );
        veto(members[1], None, true);
        let pick = decide(b"r-3");
        let first = pick.pick().and_then(|pick| pick.targets.first());
        assert_eq!(first.map(|target| target.id), Some(members[1]));
    }

    /// A ready member of [`MODEL`] under `role` in `store` that takes images
    /// where `vision`, and states `stated` beside it for a request's hard
    /// demands.
    fn states_facts(
        store: &MemoryStore,
        name: &str,
        role: Role,
        (vision, stated): (bool, DemandFacts),
    ) -> WorkerId {
        let key = format!("http://{name}:8000");
        let facts = WorkerFacts {
            key: key.clone(),
            engine: key,
            incarnation: 1,
            role,
            models: vec![MODEL.to_owned()],
            dp_size: 1,
            max_running: Some(4_096),
            vision,
            demand_facts: stated,
            ..WorkerFacts::default()
        };
        let id = store.upsert(facts).id();
        let row = WorkersStore::get(store, id).expect("the row");
        if row.status != WorkerStatus::Ready {
            let ready = store.set_status(id, WorkerStatus::Ready, row.revision);
            ready.expect("the row stands");
        }
        id
    }

    /// [`states_facts`] for a member that states a runtime and no more: `0`
    /// states none.
    fn states(store: &MemoryStore, name: &str, role: Role, runtime: u32) -> WorkerId {
        let stated = DemandFacts {
            runtime,
            ..DemandFacts::default()
        };
        states_facts(store, name, role, (false, stated))
    }

    /// [`stating_facts`] for members that state a runtime and no more.
    fn stating(stated: &[u32]) -> (Arc<MemoryStore>, Vec<WorkerId>) {
        let facts = stated.iter().map(|&runtime| {
            let stated = DemandFacts {
                runtime,
                ..DemandFacts::default()
            };
            (false, stated)
        });
        stating_facts(&facts.collect::<Vec<_>>())
    }

    /// A store under the `random` preset whose ready members of [`MODEL`]
    /// state the facts of `stated`, one a member, in that order: its vision
    /// bit, then what it states beside it.
    fn stating_facts(stated: &[(bool, DemandFacts)]) -> (Arc<MemoryStore>, Vec<WorkerId>) {
        let store = MemoryStore::new();
        let ids = stated.iter().enumerate().map(|(place, &stated)| {
            states_facts(&store, &format!("r{place}"), Role::Single, stated)
        });
        let ids: Vec<WorkerId> = ids.collect();
        store.publish_now();
        let applied = store.apply(preset("random", 1));
        applied.expect("above the unbound start");
        (Arc::new(store), ids)
    }

    /// A request that demands a runtime is decided over the members that
    /// state it, whatever the draw, and its lease books one of them. A
    /// request that demands none is decided over the pool.
    #[test]
    fn a_demanded_runtime_keeps_the_members_that_state_it() {
        let (store, ids) = stating(&[7, 9, 7, 0]);
        let door = Door::new(Arc::clone(&store));
        let placed = |runtime: u32| {
            let mut placed = BTreeSet::new();
            for seed in 0..64_u64 {
                let id = [seed.to_le_bytes(), u64::from(runtime).to_le_bytes()].concat();
                let mut asked = request(&id, seed);
                asked.constraints.demands = Demands {
                    runtime,
                    ..Demands::default()
                };
                let decided = door.decide(GATEWAY, &asked, Instant::now());
                let decided = decided.expect("a decision");
                let pick = decided.decision.pick().expect("a member states it");
                placed.insert(pick.targets.first().expect("a target").id);
                assert!(decided.lease.is_some(), "a pick's lease");
            }
            placed
        };
        let (first, second, third, fourth) = (ids[0], ids[1], ids[2], ids[3]);
        assert_eq!(placed(7), BTreeSet::from([first, third]));
        let books = booked(&store, Role::Single);
        let on = |id: WorkerId| {
            let book = books.iter().find(|(member, _)| *member == id);
            book.map_or(0, |(_, count)| *count)
        };
        assert_eq!((on(first) + on(third), on(second) + on(fourth)), (64, 0));
        assert_eq!(placed(9), BTreeSet::from([second]));
        assert_eq!(placed(0), ids.iter().copied().collect());
        assert_eq!(door.undialed(), 0, "a demand takes no row as a wire does");
    }

    /// A demand no member of a pool meets leaves the request nobody: it is
    /// shed under the stage a gateway's wires are, and no lease is filed. A
    /// member that states no runtime meets no demand for one.
    #[test]
    fn a_demand_no_member_meets_is_shed_and_files_no_lease() {
        let (store, _) = stating(&[7, 0]);
        let door = Door::new(Arc::clone(&store));
        let mut asked = request(b"r-1", 7);
        asked.constraints.demands = Demands {
            runtime: 9,
            ..Demands::default()
        };
        let decided = door.decide(GATEWAY, &asked, Instant::now());
        let decided = decided.expect("a decision");
        match decided.decision {
            Decision::Shed(shed) => assert_eq!(
                (shed.reason, shed.emptied_by),
                (ShedReason::NoEndpoints, super::WIRES)
            ),
            other => panic!("a member took it: {other:?}"),
        }
        assert!(decided.lease.is_none());
        assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
        assert_eq!(door.undialed(), 0);
    }

    /// The store narrows a pair's sides by its gateway's wires and keeps no
    /// hard demand there yet. So a pair's request that states one is shed
    /// under the wires' name, whatever its sides state, and files
    /// nothing: it is never placed past its demand. One that states none
    /// is decided.
    #[test]
    fn a_pair_whose_request_states_a_demand_is_shed_and_files_nothing() {
        let store = MemoryStore::new();
        states(&store, "p0", Role::Prefill, 7);
        states(&store, "d0", Role::Decode, 7);
        store.publish_now();
        let document = PolicySpec::pair_presets("least_load", "least_load");
        apply_pair_document(&store, document.expect("the pair presets")).expect("a pair plan");
        let store = Arc::new(store);
        let door = Door::new(Arc::clone(&store));
        let decide = |id: &[u8], runtime: u32| {
            let mut asked = request(id, 7);
            asked.facts.phase = Phase::Prefill;
            asked.constraints.demands = Demands {
                runtime,
                ..Demands::default()
            };
            let decided = door.decide(GATEWAY, &asked, Instant::now());
            decided.expect("a decision")
        };
        let stated = decide(b"stated", 7);
        match &stated.decision {
            Decision::Shed(shed) => assert_eq!(
                (shed.reason, shed.emptied_by),
                (ShedReason::NoEndpoints, super::WIRES)
            ),
            other => panic!("a pair was placed past a demand: {other:?}"),
        }
        assert!(stated.lease.is_none());
        assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
        let plain = decide(b"plain", 0);
        assert!(plain.decision.pick().is_some() && plain.pair(), "{plain:?}");
        assert_eq!(door.undialed(), 0);
    }

    /// A list a demand cut and a list the wires cut of one build take two
    /// names, and a cut by both never reads as the build's whole list. Each
    /// place below is the one whose taking read as a set of wires' cut
    /// while the two were folded by one word.
    #[test]
    fn a_cut_by_a_demand_is_named_apart_from_a_cut_by_the_wires() {
        use super::{cut_name, cut_reach, narrow_demanded, Took};

        for build in [0_u64, 0x5EED_0001] {
            for wires in [WireModes::HTTP, WireModes::GRPC, WireModes::ZMQ] {
                // Rows that all state one runtime but the one at the place
                // that equals the set's bits.
                let place = wires.bits();
                let mut view = ViewBuffers::default();
                let mut facts = Vec::new();
                for position in 0..place + 2 {
                    view.candidates.push(Candidate {
                        id: WorkerId(u64::from(position) + 10),
                        position: u32::from(position),
                        wire_modes: wires,
                        ..Candidate::default()
                    });
                    facts.push(DemandFacts {
                        runtime: if position == place { 9 } else { 7 },
                        ..DemandFacts::default()
                    });
                }
                let demands = Demands {
                    runtime: 7,
                    ..Demands::default()
                };
                let took = narrow_demanded(&mut view, &mut facts, None, None, demands);
                assert_eq!(
                    (took.undialed, view.candidates.len()),
                    (0, usize::from(place) + 1)
                );
                let by_demand = cut_name(build, None, took);
                let by_wires = cut_reach(build, wires);
                assert_ne!(by_demand, by_wires, "place {place}");
                assert_ne!(by_demand, build, "place {place}");
                // The same place taken beside those wires' own cut.
                let beside = Took {
                    undialed: 1,
                    ..took
                };
                let both = cut_name(build, Some(wires), beside);
                assert!(
                    both != build && both != by_wires && both != by_demand,
                    "place {place}"
                );
                // No demand took a row: the wires' name alone, or the build's.
                let dialed = Took {
                    undialed: 1,
                    ..Took::default()
                };
                assert_eq!(cut_name(build, Some(wires), dialed), by_wires);
                assert_eq!(cut_name(build, Some(wires), Took::default()), build);
            }
        }
    }

    /// Two lists a demand cuts out of one build turn two rotations. A pool
    /// of sixteen members a wire that state the runtime, with one more a
    /// wire that states another at the place equal to its wire's bits:
    /// requests of the two wires that alternate each draw every member of
    /// their own list. Under one name for both lists each drew every
    /// other one.
    #[test]
    fn two_lists_a_demand_cut_out_of_one_build_turn_two_rotations() {
        let store = MemoryStore::new();
        let grpc = (WireModes::GRPC, 7);
        let zmq = (WireModes::ZMQ, 7);
        let member = |name: usize, (wire_modes, runtime): (WireModes, u32)| {
            let key = format!("http://w{name}:8000");
            WorkerFacts {
                key: key.clone(),
                engine: key,
                incarnation: 1,
                role: Role::Single,
                models: vec![MODEL.to_owned()],
                dp_size: 1,
                wire_modes,
                demand_facts: DemandFacts {
                    runtime,
                    ..DemandFacts::default()
                },
                ..WorkerFacts::default()
            }
        };
        // A build lists its members in the order of their ids, and an id
        // is of its member's key: every member is written once for its
        // id, then again as its place in that order states.
        let mut written: Vec<(WorkerId, usize)> = (0..34)
            .map(|name| (store.upsert(member(name, grpc)).id(), name))
            .collect();
        written.sort_unstable();
        // The build's places 2 and 4 are the gRPC and the ZMQ member of
        // another runtime.
        let mut places = vec![grpc, zmq, (WireModes::GRPC, 9), grpc, (WireModes::ZMQ, 9)];
        places.extend([grpc; 14]);
        places.extend([zmq; 15]);
        for ((id, name), stated) in written.into_iter().zip(places) {
            let again = store.upsert(member(name, stated)).id();
            assert_eq!(again, id, "a member written again keeps its id");
            let row = WorkersStore::get(&store, id).expect("the row");
            if row.status != WorkerStatus::Ready {
                let ready = store.set_status(id, WorkerStatus::Ready, row.revision);
                ready.expect("the row stands");
            }
        }
        store.publish_now();
        let applied = store.apply(preset("round_robin", 1));
        applied.expect("above the unbound start");
        let store = Arc::new(store);
        let door = Door::new(Arc::clone(&store));
        let mut drawn = [BTreeSet::new(), BTreeSet::new()];
        for turn in 0..32_u64 {
            let list = usize::from(turn % 2 == 1);
            let id = turn.to_le_bytes();
            let mut asked = request(&id, turn);
            asked.constraints.wire_modes = [WireModes::GRPC, WireModes::ZMQ][list];
            asked.constraints.demands = Demands {
                runtime: 7,
                ..Demands::default()
            };
            let decided = door.decide(GATEWAY, &asked, Instant::now());
            let decided = decided.expect("a decision");
            let pick = decided.decision.pick().expect("sixteen members state it");
            drawn[list].insert(pick.targets.first().expect("a target").id);
        }
        let [by_grpc, by_zmq] = drawn;
        assert_eq!((by_grpc.len(), by_zmq.len()), (16, 16));
        assert!(by_grpc.is_disjoint(&by_zmq));
    }

    /// A request whose demand took a row is named over the rows that stay:
    /// the member that holds its first block is named a holder at its place
    /// among them and the member short of the level a warming row at its
    /// own. The member the demand took, a holder short of the level too,
    /// is named by nobody.
    #[test]
    fn a_demand_that_took_a_row_names_the_rows_that_stay() {
        use std::{sync::atomic::AtomicU64, time::Duration};

        use kv_index::chain_prefix_hash;

        use super::{hand, Buffers, Constraints, PoolFacts, Scratch, Taken};
        use crate::{
            store::{
                BatchFacts, BatchOutcome, ContentHash, KvEvent, KvEventSink, LoadSink, LoadSource,
                SequenceHash, Signals, StoredBlock,
            },
            view::LoadRow,
        };

        let store = MemoryStore::new();
        let applied = store.apply(preset("cache_aware", 1));
        applied.expect("above the unbound start");
        // A member that publishes its blocks and holds the chain of
        // `contents`.
        let member = |name: &str, runtime: u32, contents: &[u64]| {
            let facts = WorkerFacts {
                key: name.to_owned(),
                engine: name.to_owned(),
                incarnation: 1,
                role: Role::Single,
                models: vec![MODEL.to_owned()],
                dp_size: 1,
                signals: Signals {
                    events: true,
                    ..Signals::default()
                },
                demand_facts: DemandFacts {
                    runtime,
                    ..DemandFacts::default()
                },
                ..WorkerFacts::default()
            };
            let id = store.upsert(facts).id();
            let row = WorkersStore::get(&store, id).expect("the row");
            if row.status != WorkerStatus::Ready {
                let ready = store.set_status(id, WorkerStatus::Ready, row.revision);
                ready.expect("the row stands");
            }
            store.publish_now();
            let mut blocks: Vec<StoredBlock> = Vec::with_capacity(contents.len());
            for &content in contents {
                let seq_hash = match blocks.last() {
                    Some(parent) => chain_prefix_hash(parent.seq_hash, ContentHash(content)),
                    None => SequenceHash(content),
                };
                blocks.push(StoredBlock {
                    seq_hash,
                    content_hash: ContentHash(content),
                });
            }
            if !blocks.is_empty() {
                let events = [KvEvent::Stored {
                    parent: None,
                    blocks,
                }];
                let batch = BatchFacts {
                    sequence: 1,
                    ..BatchFacts::default()
                };
                assert_eq!(store.admit(id, 0, batch, &events), BatchOutcome::Applied);
            }
            id
        };
        // One member fills past the level's floor, so the two short of it
        // are the warming rows at the next window. The demand takes one of
        // those two, which holds the request's first block as the full
        // member does.
        let filled: Vec<u64> = (1..=1_100).collect();
        let full = member("full", 7, &filled);
        let short = member("short", 7, &[]);
        member("gone", 9, &[1]);
        let window = Instant::now() + Duration::from_secs(1);
        store.record(full, 0, LoadRow::default(), window, LoadSource::Stream);
        let hashes = [1_u64];
        let mut asked = request(b"named", 7);
        asked.facts.block_hashes = &hashes;
        asked.constraints.demands = Demands {
            runtime: 7,
            ..Demands::default()
        };
        let taken = Taken {
            request: &asked,
            gateway: GATEWAY,
            now: Instant::now(),
            left: Duration::MAX,
            pins: None,
            place: None,
        };
        let mut buffers = Buffers::default();
        let undialed = AtomicU64::new(0);
        // What the decider is handed: its rows, and the rows named, each by
        // its member.
        let read = |facts: &RequestFacts<'_>,
                    _: &PoolFacts<'_>,
                    rows: &[Candidate],
                    _: &Constraints<'_>,
                    _: &mut Scratch| {
            let named = facts.named.expect("the profile names its rows");
            let id_of = |row: u32| rows[row as usize].id;
            let holders: Vec<WorkerId> = named.holders.iter().map(|held| id_of(held.row)).collect();
            let warming: Vec<WorkerId> = named.warming.iter().map(|&row| id_of(row)).collect();
            let rows: BTreeSet<WorkerId> = rows.iter().map(|row| row.id).collect();
            (rows, holders, warming)
        };
        let handed = store.deciding(|world| hand(world, &taken, &mut buffers, &undialed, read));
        let ((rows, holders, warming), _) = handed;
        assert_eq!(rows, [full, short].into_iter().collect());
        assert_eq!(holders, [full]);
        assert_eq!(warming, [short]);
    }

    /// Media is a demand as a runtime is: a request that needs vision or
    /// carries media by reference is decided over the members that take it.
    /// Vision is read of a member's own bit, the one a pairing plan reads,
    /// and the rest of what it states beside it. A pool that holds none is
    /// shed under the last demand its members failed first, in a host's
    /// order of answers: under the plan's own word where that is vision, and
    /// under the wires where its runtime emptied it.
    #[test]
    fn a_media_demand_keeps_the_members_that_take_it_and_names_its_shed() {
        let member = |vision: bool, references: bool, host_pipeline: bool| {
            let stated = DemandFacts {
                runtime: 7,
                references,
                host_pipeline,
            };
            (vision, stated)
        };
        let (store, ids) = stating_facts(&[
            member(true, false, false),
            member(true, true, true),
            member(false, true, false),
        ]);
        let (plain, host, own) = (ids[0], ids[1], ids[2]);
        let door = Door::new(Arc::clone(&store));
        let decide = |tag: u8, demands: Demands| {
            let mut placed = BTreeSet::new();
            for seed in 0..48_u64 {
                let id = [&[tag][..], &seed.to_le_bytes()].concat();
                let mut asked = request(&id, seed);
                asked.constraints.demands = demands;
                let decided = door.decide(GATEWAY, &asked, Instant::now());
                match decided.expect("a decision").decision {
                    Decision::Shed(shed) => return Err((shed.reason, shed.emptied_by)),
                    decision => {
                        let first = decision.pick().and_then(|pick| pick.targets.first());
                        placed.insert(first.expect("a target").id);
                    }
                }
            }
            Ok(placed)
        };
        let asks = |vision: bool, references: References| Demands {
            vision,
            references,
            ..Demands::default()
        };
        let kept = |members: &[WorkerId]| Ok(members.iter().copied().collect::<BTreeSet<_>>());
        assert_eq!(
            decide(1, asks(true, References::None)),
            kept(&[plain, host])
        );
        assert_eq!(decide(2, asks(false, References::Any)), kept(&[host, own]));
        assert_eq!(decide(3, asks(false, References::Host)), kept(&[host]));
        assert_eq!(decide(4, asks(false, References::Worker)), kept(&[own]));
        // One member takes no images, one processes no reference and one
        // runs the other pipeline: the last of the three is the pipeline.
        let nobody = |stage: &'static str| Err((ShedReason::NoEndpoints, stage));
        assert_eq!(
            decide(5, asks(true, References::Worker)),
            nobody("pipeline")
        );
        // Every member fails at its runtime, the first demand of the order.
        let elsewhere = Demands {
            runtime: 9,
            ..asks(true, References::Host)
        };
        assert_eq!(decide(6, elsewhere), nobody(super::WIRES));
        assert_eq!(door.undialed(), 0);
        // A pool none of whose members takes images is shed under the word
        // the pairing plan sheds under.
        let (blind, _) = stating_facts(&[member(false, true, true), member(false, false, false)]);
        let blind = Door::new(blind);
        let mut asked = request(b"blind", 1);
        asked.constraints.demands = asks(true, References::Any);
        let decided = blind.decide(GATEWAY, &asked, Instant::now());
        let shed = match decided.expect("a decision").decision {
            Decision::Shed(shed) => Ok((shed.reason, shed.emptied_by)),
            decision => Err(decision.pick().is_some()),
        };
        assert_eq!(
            shed,
            Ok((ShedReason::NoEndpoints, crate::framework::VISION))
        );
    }

    /// A hint of 0 keeps no row whatever the request demands: its pool is
    /// shed under the hint's stage, and under no demand's.
    #[test]
    fn a_hint_of_none_beside_a_demand_is_shed_under_the_hint() {
        let (store, _) = stating_facts(&[(false, DemandFacts::default())]);
        let door = Door::new(store);
        let mut asked = request(b"both", 1);
        asked.constraints.subset_hint = Some(0);
        asked.constraints.demands = Demands {
            vision: true,
            ..Demands::default()
        };
        let decided = door.decide(GATEWAY, &asked, Instant::now());
        let shed = match decided.expect("a decision").decision {
            Decision::Shed(shed) => Ok((shed.reason, shed.emptied_by)),
            decision => Err(decision.pick().is_some()),
        };
        assert_eq!(shed, Ok((ShedReason::NoEndpoints, super::SUBSET_HINT)));
    }

    /// A model no member serves is no pool the request's own narrowing
    /// emptied: a request for it that states a demand is shed under no
    /// stage, as one that states none is.
    #[test]
    fn a_demand_for_a_model_no_member_serves_is_shed_under_no_stage() {
        let (store, _) = stating_facts(&[(true, DemandFacts::default())]);
        let door = Door::new(store);
        let mut asked = request(b"absent", 1);
        asked.facts.model = "a-model-nobody-serves";
        asked.constraints.demands = Demands {
            vision: true,
            ..Demands::default()
        };
        let decided = door.decide(GATEWAY, &asked, Instant::now());
        let shed = match decided.expect("a decision").decision {
            Decision::Shed(shed) => Ok((shed.reason, shed.emptied_by)),
            decision => Err(decision.pick().is_some()),
        };
        assert_eq!(shed, Ok((ShedReason::NoEndpoints, "none")));
    }

    /// A row a vision demand took is named by nobody, as one a runtime took
    /// is: the member that holds the request's first block takes no image,
    /// and the request is decided over the member that does.
    #[test]
    fn a_vision_demand_that_took_the_holder_names_the_rows_that_stay() {
        use crate::store::{
            BatchFacts, BatchOutcome, ContentHash, KvEvent, KvEventSink, SequenceHash, Signals,
            StoredBlock,
        };

        let store = MemoryStore::new();
        let applied = store.apply(preset("cache_aware", 1));
        applied.expect("above the unbound start");
        let member = |name: &str, vision: bool| {
            let facts = WorkerFacts {
                key: name.to_owned(),
                engine: name.to_owned(),
                incarnation: 1,
                role: Role::Single,
                models: vec![MODEL.to_owned()],
                dp_size: 1,
                vision,
                signals: Signals {
                    events: true,
                    ..Signals::default()
                },
                ..WorkerFacts::default()
            };
            let id = store.upsert(facts).id();
            let row = WorkersStore::get(&store, id).expect("the row");
            if row.status != WorkerStatus::Ready {
                let ready = store.set_status(id, WorkerStatus::Ready, row.revision);
                ready.expect("the row stands");
            }
            id
        };
        let (blind, sighted) = (member("blind", false), member("sighted", true));
        store.publish_now();
        let hashes = [17_u64];
        let events = [KvEvent::Stored {
            parent: None,
            blocks: vec![StoredBlock {
                seq_hash: SequenceHash(hashes[0]),
                content_hash: ContentHash(hashes[0]),
            }],
        }];
        let batch = BatchFacts {
            sequence: 1,
            ..BatchFacts::default()
        };
        assert_eq!(store.admit(blind, 0, batch, &events), BatchOutcome::Applied);
        let door = Door::new(Arc::new(store));
        let mut asked = request(b"seen", 1);
        asked.facts.block_hashes = &hashes;
        asked.constraints.demands = Demands {
            vision: true,
            ..Demands::default()
        };
        let decided = door.decide(GATEWAY, &asked, Instant::now());
        let decision = decided.expect("a decision").decision;
        let first = decision.pick().and_then(|pick| pick.targets.first());
        assert_eq!(first.map(|target| target.id), Some(sighted));
    }

    /// One pass narrows a built view by its gateway's wires and its
    /// request's demands: the rows kept take their places in order, and the
    /// affinity column and the facts follow them. The rows a demand took
    /// are named by their places: two requests that leave one list give it
    /// one name whatever they state, and two lists take two names.
    #[test]
    fn a_narrowing_by_demands_keeps_every_column_in_step_and_names_the_rows_taken() {
        use super::{narrow_demanded, Took};
        use crate::view::AffinityFacts;

        let wires = [
            WireModes::GRPC,
            WireModes::GRPC,
            WireModes::ZMQ,
            WireModes::GRPC,
            WireModes::GRPC,
        ];
        let built = |column: bool, stated: [u32; 5]| {
            let mut view = ViewBuffers::default();
            let mut facts = Vec::new();
            for ((position, wire_modes), runtime) in (0_u32..).zip(wires).zip(stated) {
                view.candidates.push(Candidate {
                    id: WorkerId(u64::from(position) + 10),
                    position,
                    wire_modes,
                    ..Candidate::default()
                });
                if column {
                    view.affinity.push(AffinityFacts {
                        admitted_age_ms: position + 100,
                        ..AffinityFacts::default()
                    });
                }
                facts.push(DemandFacts {
                    runtime,
                    ..DemandFacts::default()
                });
            }
            (view, facts)
        };
        let read = |view: &ViewBuffers, facts: &[DemandFacts]| {
            let rows = view.candidates.iter().map(|row| (row.id.0, row.position));
            let ages = view.affinity.iter().map(|facts| facts.admitted_age_ms);
            let stated = facts.iter().map(|facts| facts.runtime);
            (
                rows.collect::<Vec<_>>(),
                ages.collect::<Vec<_>>(),
                stated.collect::<Vec<_>>(),
            )
        };
        let grpc = Some(WireModes::GRPC);
        let sevens = [7, 9, 7, 0, 7];
        let kept = vec![(10, 0), (14, 1)];
        // The wire takes the third row, the demand the second and the
        // fourth, which states no runtime.
        let (mut view, mut facts) = built(true, sevens);
        let took = narrow_demanded(
            &mut view,
            &mut facts,
            grpc,
            None,
            Demands {
                runtime: 7,
                ..Demands::default()
            },
        );
        assert_eq!(took.undialed, 1);
        assert_ne!(took.demanded, 0);
        assert_eq!(took.unmet, Some(Unmet::Runtime));
        assert_eq!(
            read(&view, &facts),
            (kept.clone(), vec![100, 104], vec![7, 7])
        );
        // The same rows taken of a view with no column, under other
        // numbers and a positive hint: one list, one name.
        let (mut view, mut facts) = built(false, [3, 5, 3, 0, 3]);
        let again = narrow_demanded(
            &mut view,
            &mut facts,
            grpc,
            Some(2),
            Demands {
                runtime: 3,
                ..Demands::default()
            },
        );
        assert_eq!(again, took);
        assert_eq!(read(&view, &facts), (kept, vec![], vec![3, 3]));
        // Another demand leaves another list, under another name.
        let (mut view, mut facts) = built(true, sevens);
        let other = narrow_demanded(
            &mut view,
            &mut facts,
            None,
            None,
            Demands {
                runtime: 9,
                ..Demands::default()
            },
        );
        assert_eq!(other.undialed, 0);
        assert!(other.demanded != 0 && other.demanded != took.demanded);
        assert_eq!(read(&view, &facts), (vec![(11, 0)], vec![101], vec![9]));
        // No demand: the wires alone take a row, and the list has no name
        // of a demand's.
        let (mut view, mut facts) = built(true, sevens);
        let none = narrow_demanded(&mut view, &mut facts, grpc, None, Demands::default());
        assert_eq!(
            none,
            Took {
                undialed: 1,
                demanded: 0,
                unmet: None,
            }
        );
        assert_eq!(read(&view, &facts).0, [(10, 0), (11, 1), (13, 2), (14, 3)]);
        // A hint of 0 keeps none.
        let (mut view, mut facts) = built(true, sevens);
        let hinted = narrow_demanded(
            &mut view,
            &mut facts,
            None,
            Some(0),
            Demands {
                runtime: 7,
                ..Demands::default()
            },
        );
        assert_eq!(hinted, Took::default());
        assert_eq!(read(&view, &facts), (vec![], vec![], vec![]));
        // A row with no entry of the column meets no demand.
        let (mut view, mut facts) = built(true, sevens);
        facts.truncate(1);
        narrow_demanded(
            &mut view,
            &mut facts,
            None,
            None,
            Demands {
                runtime: 7,
                ..Demands::default()
            },
        );
        assert_eq!(read(&view, &facts), (vec![(10, 0)], vec![100], vec![7]));
    }

    /// The leader's doubt of a worker reaches the door's decision as a
    /// gateway's own doubt does, for every gateway of the store. A doubted
    /// member beside sound ones is passed over. A pool doubted on every
    /// member is steered with the doubt lifted, and the trace says so: a
    /// doubt is not an outage.
    #[test]
    fn the_leaders_doubt_of_a_worker_reaches_the_doors_decision() {
        let store = fleet(1);
        let door = Door::new(Arc::clone(&store));
        let now = Instant::now();
        let members: Vec<WorkerId> = booked(&store, Role::Single)
            .iter()
            .map(|&(id, _)| id)
            .collect();
        let doubt = |member: WorkerId, stalled: bool| {
            let row = WorkersStore::get(&*store, member).expect("the member's row");
            let bits = LeaderBits { stalled };
            let written = store.leader_bits(member, row.revision, bits, &|| true);
            assert!(matches!(written, HealthWrite::Stood(_)), "{written:?}");
        };
        let decide = |id: &[u8], seed: u64| {
            let decided = door.decide(GATEWAY, &request(id, seed), now);
            decided.expect("a decision").decision
        };

        // One doubted member beside two sound ones: no seed picks it, and
        // no veto is lifted.
        doubt(members[0], true);
        for seed in 0..32_u64 {
            let decision = decide(&seed.to_be_bytes(), seed);
            let pick = decision.pick().expect("two sound members stand");
            let first = pick.targets.first().map(|target| target.id);
            assert_ne!(first, Some(members[0]));
            assert_eq!(pick.trace.lifted, Lifted::None);
        }

        // The pool doubted whole: steered, with the doubt lifted.
        doubt(members[1], true);
        doubt(members[2], true);
        let decision = decide(b"whole", 7);
        let pick = decision.pick().expect("a doubt is not an outage");
        assert_eq!(pick.trace.lifted, Lifted::Liveness);

        // The doubt cleared on one member: it takes the request, and no
        // veto is lifted.
        doubt(members[1], false);
        let decision = decide(b"cleared", 7);
        let pick = decision.pick().expect("a sound member stands");
        let first = pick.targets.first().map(|target| target.id);
        assert_eq!(first, Some(members[1]));
        assert_eq!(pick.trace.lifted, Lifted::None);
    }

    /// In process no election hands a term over: the door starts a store
    /// under no term at term 1 and raises the term behind the lease at its
    /// mark. The leases of the term before stay on their file.
    #[test]
    fn a_door_raises_the_term_at_its_mark_and_the_leases_stay_on_file() {
        let store = fleet(1);
        let door = Door::new(Arc::clone(&store));
        assert_eq!(LeasesStore::term(&*store), 1);
        store.spend_ids(u32::MAX - TERM_MARK + 2);
        let ids = [b"a", b"b", b"c"].map(|id| leased(&door, id, 7).0);
        let [under, at, next] = ids.map(|id| id.0);
        assert_eq!(under, 1 << 32 | u64::from(TERM_MARK - 1));
        assert_eq!(at, 1 << 32 | u64::from(TERM_MARK));
        assert_eq!(next >> 32, 2, "the term rose behind the mark's lease");
        assert_eq!(LeasesStore::term(&*store), 2);
        for id in ids {
            let on_file = LeasesStore::get(&*store, GATEWAY, id);
            assert!(on_file.is_some(), "{id:?} is on its file");
        }
    }

    /// A term the door did not read costs no request its decision: one
    /// whose ids are spent is raised, one another hand raised is read
    /// again, and the request is decided once more under it.
    #[test]
    fn a_term_the_door_did_not_read_costs_no_request_its_decision() {
        let store = fleet(1);
        let door = Door::new(Arc::clone(&store));
        store.spend_ids(0);
        let (spent, _) = leased(&door, b"a", 7);
        assert_eq!(spent.0 >> 32, 2, "the door raised a term with no id left");
        LeasesStore::set_term(&*store, 7);
        let (raised, _) = leased(&door, b"b", 7);
        assert_eq!(raised.0 >> 32, 7, "the term another hand raised");
        assert_eq!(LeasesStore::live(&*store, GATEWAY), 2);
        assert_eq!(door.faults(), 0);
    }

    /// What a stage heard of leases: each call, with the decider its
    /// thread acted as.
    type Heard = Arc<Mutex<Vec<(&'static str, Option<Decider>)>>>;

    /// The `random` picker, keeping what the engine tells it of leases.
    #[derive(Debug)]
    struct Hears(Heard);

    impl Hears {
        fn heard(&self, call: &'static str) {
            self.0.lock().push((call, Decider::acting()));
        }
    }

    impl Lifecycle for Hears {
        fn on_decision_issued(&self, _: &Lease, _: &Decision) {
            self.heard("issued");
        }

        fn on_dispatched(&self, _: &Lease, _: &Target, _: Side, _: u32) {
            self.heard("dispatched");
        }

        fn on_completed(&self, _: &Lease, _: &Outcome) {
            self.heard("completed");
        }

        fn on_released(&self, _: &Lease, _: ReleaseReason) {
            self.heard("released");
        }
    }

    impl Picker for Hears {
        fn name(&self) -> &'static str {
            "hears"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            Random.pick(facts, candidates, kept, scores, band, out);
        }
    }

    /// The engine that decided hears of its lease what the served path
    /// tells it: the issue as the door files it, then each report as the
    /// store applies it. Every call runs under the name of the seat its
    /// caller decided at: the lease is on that seat's file.
    #[test]
    fn the_engine_hears_a_leases_issue_and_each_report_of_it() {
        // A later seat: a thread that names no decider files in the first.
        let (files, at) = (4, 2);
        let store = fleet(files);
        let heard = Heard::default();
        let kept = Arc::clone(&heard);
        picking(&store, 2, "hears", move || {
            Box::new(Hears(Arc::clone(&kept)))
        });
        let door = Door::new(Arc::clone(&store));
        seated(at);
        let (first, served) = leased(&door, b"r-1", 7);
        assert_eq!(store.file_of(first), at, "filed at the caller's seat");
        assert_eq!(report(&door, first, dispatched()), 1);
        assert_eq!(report(&door, first, completed(served)), 0);
        let (second, _) = leased(&door, b"r-2", 8);
        assert_eq!(report(&door, second, gone()), 0);
        let by = u8::try_from(at).ok().map(Decider::new);
        let calls = ["issued", "dispatched", "completed", "issued", "released"];
        assert_eq!(*heard.lock(), calls.map(|call| (call, by)));
        assert_eq!(store.lifecycle_faults(), 0);
    }

    /// A picker that panics, for a decision that does.
    #[derive(Debug)]
    struct Panics;

    impl Lifecycle for Panics {}

    impl Picker for Panics {
        fn name(&self) -> &'static str {
            "panics"
        }

        fn pick(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            _: &[f32],
            _: TieBand,
            _: &mut Targets,
        ) {
            panic!("the case's own panic, inside a decision");
        }
    }

    /// A decision that panics costs its own request alone: the caller is
    /// refused and no lease is filed. The thread left its work, so a
    /// report handed behind the fault is applied at once, and the next
    /// request decides.
    #[test]
    fn a_decision_that_panics_costs_its_request_alone() {
        let store = fleet(1);
        let door = Door::new(Arc::clone(&store));
        let (before, _) = leased(&door, b"r-0", 7);
        picking(&store, 2, "panics", || Box::new(Panics));
        let refused = door.decide(GATEWAY, &request(b"r-1", 7), Instant::now());
        assert_eq!(refused.err(), Some(Refused::Faulted));
        assert_eq!((door.faults(), LeasesStore::live(&*store, GATEWAY)), (1, 1));
        assert_eq!(report(&door, before, gone()), 0, "no thread is at work");
        store
            .apply(preset("random", 3))
            .expect("a later generation");
        let (lease, _) = leased(&door, b"r-2", 7);
        assert!(LeasesStore::get(&*store, GATEWAY, lease).is_some());
    }

    /// How long a case waits for a thread it holds, or for one it expects
    /// back.
    pub(crate) const PATIENCE: Duration = Duration::from_secs(20);

    /// The seed of the request a [`Stops`] picker holds.
    const HELD: u64 = 99;

    /// What a case shares with its [`Stops`] picker.
    #[derive(Debug)]
    struct Scene {
        /// Where the held request stands, said as it stops there.
        stands: Mutex<mpsc::Sender<&'static str>>,
        /// The case's word to go on, one a stop.
        word: Mutex<mpsc::Receiver<()>>,
        /// The held request stands between its pick and its lease's issue.
        held: AtomicBool,
        /// The next dispatch the stage hears stops.
        hears: AtomicBool,
        /// What the stage heard of leases: each call, with its lease.
        heard: Mutex<Vec<(&'static str, LeaseId)>>,
    }

    impl Scene {
        /// The held request stops at `place` until the case's word.
        fn stop(&self, place: &'static str) {
            let said = self.stands.lock().send(place);
            said.expect("the case holds the other end");
            let word = self.word.lock().recv();
            word.expect("the case holds the other end");
        }

        fn hear(&self, call: &'static str, lease: &Lease) {
            self.heard.lock().push((call, lease.id));
        }
    }

    /// The `random` picker with two stops for the request under [`HELD`]:
    /// in its pick, and at the issue of its lease. A case may ask for a
    /// third, at the next dispatch the stage hears. It keeps what the
    /// engine tells it of leases.
    #[derive(Debug)]
    struct Stops(Arc<Scene>);

    impl Lifecycle for Stops {
        fn on_decision_issued(&self, lease: &Lease, _: &Decision) {
            self.0.hear("issued", lease);
            if self.0.held.swap(false, Ordering::SeqCst) {
                self.0.stop("issue");
            }
        }

        fn on_dispatched(&self, lease: &Lease, _: &Target, _: Side, _: u32) {
            self.0.hear("dispatched", lease);
            if self.0.hears.swap(false, Ordering::SeqCst) {
                self.0.stop("dispatched");
            }
        }

        fn on_completed(&self, lease: &Lease, _: &Outcome) {
            self.0.hear("completed", lease);
        }

        fn on_released(&self, lease: &Lease, _: ReleaseReason) {
            self.0.hear("released", lease);
        }
    }

    impl Picker for Stops {
        fn name(&self) -> &'static str {
            "stops"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            if facts.seed == HELD {
                self.0.held.store(true, Ordering::SeqCst);
                self.0.stop("pick");
            }
            Random.pick(facts, candidates, kept, scores, band, out);
        }
    }

    /// A case's ends of a scene: where the held request says it stands,
    /// and the word that sends it on.
    pub(crate) struct Ends {
        scene: Arc<Scene>,
        stood: mpsc::Receiver<&'static str>,
        word: mpsc::Sender<()>,
    }

    impl Ends {
        /// Applies the [`Stops`] picker to `store` as generation 2.
        pub(crate) fn over(store: &MemoryStore) -> Self {
            let (stands, stood) = mpsc::channel();
            let (word, heard) = mpsc::channel();
            let scene = Arc::new(Scene {
                stands: Mutex::new(stands),
                word: Mutex::new(heard),
                held: AtomicBool::new(false),
                hears: AtomicBool::new(false),
                heard: Mutex::default(),
            });
            let shared = Arc::clone(&scene);
            picking(store, 2, "stops", move || {
                Box::new(Stops(Arc::clone(&shared)))
            });
            Self { scene, stood, word }
        }

        /// Waits until the held request stands at `place`.
        pub(crate) fn stands(&self, place: &str) {
            assert_eq!(self.stood.recv_timeout(PATIENCE), Ok(place));
        }

        /// Sends the held request on from its stop.
        pub(crate) fn go(&self) {
            self.word.send(()).expect("the held request waits");
        }
    }

    /// One decision of `door` for the request under [`HELD`], on a thread
    /// of its own placed at seat `at`: its lease.
    fn held_at(door: &Arc<Door>, at: usize) -> thread::JoinHandle<Option<LeaseId>> {
        let door = Arc::clone(door);
        thread::spawn(move || {
            seated(at);
            let decided = door.decide(GATEWAY, &request(b"held", HELD), Instant::now());
            let leased = decided.ok().and_then(|decided| decided.lease);
            leased.map(|leased| leased.lease.id)
        })
    }

    /// [`held_at`] the one seat of a store of one file.
    pub(crate) fn held(door: &Arc<Door>) -> thread::JoinHandle<Option<LeaseId>> {
        held_at(door, 0)
    }

    /// Says `words` of leases through `door` in their order, on a thread
    /// of its own as a host's runtime does: whether every report returned.
    fn said(door: &Arc<Door>, words: Vec<(LeaseId, Said)>) -> bool {
        let (done, returned) = mpsc::channel();
        let door = Arc::clone(door);
        thread::spawn(move || {
            for (lease, said) in words {
                door.report(GATEWAY, lease, said);
            }
            done.send(()).ok();
        });
        returned.recv_timeout(PATIENCE).is_ok()
    }

    /// A report at a full ring holds no thread for a decision. While one
    /// decision stands, more reports than the ring takes all return, and
    /// none is applied: they are the deciding thread's to apply. As that
    /// caller leaves, what the ring held and what the door kept beside it
    /// are both applied.
    #[test]
    fn a_report_at_a_full_ring_returns_while_the_decision_stands() {
        let store = fleet(1);
        let door = Arc::new(Door::new(Arc::clone(&store)));
        let more = MemoryStore::HANDOVER + 8;
        let leases = (0..more).map(|n| leased(&door, &n.to_le_bytes(), 7).0);
        let words: Vec<_> = leases.map(|lease| (lease, gone())).collect();
        let ends = Ends::over(&store);
        let held = held(&door);
        ends.stands("pick");
        let returned = said(&door, words);
        let on_file = LeasesStore::live(&*store, GATEWAY);
        ends.go();
        ends.stands("issue");
        ends.go();
        let lease = held.join().expect("the held decision's thread");
        assert!(returned, "every report returned while the decision stood");
        assert_eq!(on_file, more, "a report is the deciding thread's to apply");
        assert!(lease.is_some(), "the held request has its lease");
        assert_eq!(LeasesStore::live(&*store, GATEWAY), 1, "its lease alone");
    }

    /// A door's caller keeps no watch over its file. No thread of a door
    /// looks at a file again, so no report is left for a look: with a
    /// lease on file the file shows no watch while a decision stands and
    /// none behind it, and what is said of a lease then is applied before
    /// its report returns.
    #[test]
    fn a_doors_file_shows_no_watch_while_a_decision_stands_or_behind_it() {
        let store = fleet(1);
        let door = Arc::new(Door::new(Arc::clone(&store)));
        let (first, _) = leased(&door, b"r-1", 7);
        let ends = Ends::over(&store);
        let held = held(&door);
        ends.stands("pick");
        let deciding = store.watches();
        ends.go();
        ends.stands("issue");
        ends.go();
        let second = held.join().expect("the held decision's thread");
        assert!(second.is_some(), "the held request has its lease");
        assert_eq!(deciding, 0, "a lease is on file and a caller decides");
        assert_eq!(store.watches(), 0, "two leases are on file, no caller");
        door.report(GATEWAY, first, gone());
        assert_eq!(LeasesStore::live(&*store, GATEWAY), 1, "applied as said");
    }

    /// What is said of one lease reaches its file in the order said, past
    /// a full ring. A first lease's dispatch finds the ring full and is
    /// kept beside it. Its completion and a second lease's dispatch are
    /// said once the ring has room again, and are kept behind the first
    /// word all the same. The second lease's completion is said while
    /// the caller hands those words on, and is kept behind them. The
    /// engine hears each lease's dispatch ahead of its completion.
    #[test]
    fn what_is_said_of_a_lease_reaches_its_file_in_the_order_said() {
        let store = fleet(1);
        let door = Arc::new(Door::new(Arc::clone(&store)));
        let fill = (0..MemoryStore::HANDOVER).map(|n| leased(&door, &n.to_le_bytes(), 7).0);
        let mut words: Vec<_> = fill.map(|lease| (lease, gone())).collect();
        let ends = Ends::over(&store);
        let (first, served) = leased(&door, b"r-1", 7);
        let (second, other) = leased(&door, b"r-2", 8);
        words.push((first, dispatched()));
        let held = held(&door);
        ends.stands("pick");
        let kept = said(&door, words);
        ends.go();
        // The held decision's issue applied what the ring held.
        ends.stands("issue");
        let room = vec![(first, completed(served)), (second, dispatched())];
        let behind = said(&door, room);
        ends.scene.hears.store(true, Ordering::SeqCst);
        ends.go();
        // The caller hands the kept words on: the first one is applied.
        ends.stands("dispatched");
        let meanwhile = said(&door, vec![(second, completed(other))]);
        ends.go();
        held.join().expect("the held decision's thread");
        assert!(kept && behind && meanwhile, "no report waits");
        let heard = ends.scene.heard.lock();
        for lease in [first, second] {
            let of_lease = heard.iter().filter(|&&(_, of)| of == lease);
            let calls: Vec<_> = of_lease.map(|&(call, _)| call).collect();
            assert_eq!(calls, ["issued", "dispatched", "completed"], "{lease:?}");
        }
    }

    /// A lease of the term before is reported after the rise as before
    /// it. Its dispatch and its completion are applied to it on its file,
    /// its count is freed, and the engine that issued it hears each.
    #[test]
    fn a_lease_of_the_term_before_is_reported_after_the_rise() {
        let store = fleet(1);
        let heard = Heard::default();
        let kept = Arc::clone(&heard);
        picking(&store, 2, "hears", move || {
            Box::new(Hears(Arc::clone(&kept)))
        });
        let door = Door::new(Arc::clone(&store));
        store.spend_ids(u32::MAX - TERM_MARK + 2);
        let (before, served) = leased(&door, b"a", 7);
        let (at, other) = leased(&door, b"b", 7);
        let terms = (before.0 >> 32, at.0 >> 32, LeasesStore::term(&*store));
        assert_eq!(terms, (1, 1, 2), "the term rose behind the mark's lease");
        assert_eq!(report(&door, before, dispatched()), 2);
        let on_file = LeasesStore::get(&*store, GATEWAY, before).expect("the lease is on file");
        assert!(matches!(on_file.state, LeaseState::Dispatched { .. }));
        assert_eq!((on_file.term, on_file.target), (1, served));
        assert_eq!(report(&door, before, completed(served)), 1);
        assert_eq!(counted(&store), [(other, 1)], "the mark's lease alone");
        let by = Some(Decider::new(0));
        let calls = ["issued", "issued", "dispatched", "completed"];
        assert_eq!(*heard.lock(), calls.map(|call| (call, by)));
        assert_eq!(store.lifecycle_faults(), 0);
    }

    /// The `random` picker, slow: its decision stands long enough for the
    /// reports of other threads to fill its file's ring.
    #[derive(Debug)]
    struct Slow;

    impl Lifecycle for Slow {}

    impl Picker for Slow {
        fn name(&self) -> &'static str {
            "slow"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            thread::sleep(Duration::from_millis(2));
            Random.pick(facts, candidates, kept, scores, band, out);
        }
    }

    /// Reports from four threads at once, beside slow decisions at their
    /// file: eight rings' worth of releases, so the ring fills while a
    /// decision stands and the door keeps what it does not take. No
    /// decision follows the last one, and no word waits for one: every
    /// lease reported is gone.
    #[test]
    fn reports_at_once_beside_slow_decisions_leave_no_word_behind() {
        let store = fleet(1);
        let door = Door::new(Arc::clone(&store));
        let (threads, each, slow) = (4, 2 * MemoryStore::HANDOVER, 6_u8);
        let leases = (0..threads * each).map(|n| leased(&door, &n.to_le_bytes(), 7).0);
        let leases: Vec<_> = leases.collect();
        picking(&store, 2, "slow", || Box::new(Slow));
        let start = Barrier::new(threads + 1);
        thread::scope(|scope| {
            for reported in leases.chunks(each) {
                let (door, start) = (&door, &start);
                scope.spawn(move || {
                    start.wait();
                    for &lease in reported {
                        door.report(GATEWAY, lease, gone());
                    }
                });
            }
            start.wait();
            for request in 0..slow {
                leased(&door, &[request], 7);
            }
        });
        assert_eq!(LeasesStore::live(&*store, GATEWAY), usize::from(slow));
        assert_eq!((door.faults(), store.lifecycle_faults()), (0, 0));
    }

    /// The door's clock is its host's call. A lease whose host names no
    /// dispatch by the end of its span is ended as abandoned by the pass
    /// at that instant, with no issue behind it: its count is freed, and
    /// the engine that issued it hears the release.
    #[test]
    fn a_sweep_ends_a_lease_no_dispatch_was_named_for() {
        let store = fleet(1);
        let heard = Heard::default();
        let kept = Arc::clone(&heard);
        picking(&store, 2, "hears", move || {
            Box::new(Hears(Arc::clone(&kept)))
        });
        let door = Door::new(Arc::clone(&store));
        let decided = door.decide(GATEWAY, &request(b"r-1", 7), Instant::now());
        let leased = decided.expect("a decision").lease;
        let due = leased.expect("a pick's lease").due;
        let inside = due - Duration::from_millis(1);
        assert_eq!(door.sweep(inside), Swept::default(), "inside its span");
        assert_eq!(LeasesStore::live(&*store, GATEWAY), 1);
        let ended = Swept { leases: 1, pins: 0 };
        assert_eq!(door.sweep(due), ended, "at its span's end");
        assert_eq!(LeasesStore::live(&*store, GATEWAY), 0);
        assert_eq!(counted(&store), []);
        let by = Some(Decider::new(0));
        assert_eq!(*heard.lock(), [("issued", by), ("released", by)]);
    }

    /// A pin of record past its end leaves its table at the door's pass,
    /// under a rule that sweeps.
    #[test]
    fn a_sweep_drops_a_pin_of_record_past_its_end() {
        let store = fleet(1);
        let door = Door::new(Arc::clone(&store));
        let mut document = PolicySpec::preset("manual").expect("a preset of this build");
        document.generation = 2;
        let rule = document.bindings[0].sessions.as_mut();
        let rule = rule.expect("the preset's rule");
        (rule.ttl_s, rule.sweep_s) = (60, 30);
        let ruled = AppliedSpec::validate(&document).expect("the document validates");
        store.apply(ruled).expect("a later generation");
        let now = Instant::now();
        let at = |seconds: u64| now + Duration::from_secs(seconds);
        let sessions: &dyn SessionsStore = &*store;
        sessions.pin(1, WorkerId(1), now, Some(store.micros(at(60))));
        let dropped = Swept { leases: 0, pins: 1 };
        assert_eq!(door.sweep(at(61)), dropped, "a second past the pin's end");
        assert_eq!(sessions.len(), 0);
    }

    /// A request for [`MODEL`] under the session key `key`.
    fn keyed(id: &[u8], key: u64, seed: u64) -> Request<'_> {
        let mut request = request(id, seed);
        request.facts.session_key = Identity::hashed(key);
        request
    }

    /// One decision of `door` for `asked`: its lease, the pick's first
    /// target and what the rule did with the key.
    fn placed(door: &Door, asked: &Request<'_>) -> (LeaseId, WorkerId, SessionVerdict) {
        let decided = door.decide(GATEWAY, asked, Instant::now());
        let decided = decided.expect("a decision");
        let pick = decided.decision.pick().expect("a pick");
        let first = pick.targets.first().expect("a target").id;
        let lease = decided.lease.expect("a pick's lease").lease.id;
        (lease, first, pick.session_verdict)
    }

    /// The `manual` preset as `generation`, its rule capped at `cap`
    /// requests of a key on a worker. The rule places a key on the worker
    /// with the fewest requests, so a request placed again at the cap
    /// leaves the key's worker, which carries the most.
    fn capped(generation: u64, cap: u16) -> PolicySpec {
        let mut document = PolicySpec::preset("manual").expect("a preset of this build");
        document.generation = generation;
        let rule = document.bindings[0].sessions.as_mut();
        let rule = rule.expect("the preset's rule");
        rule.inflight_cap = cap;
        rule.assignment = crate::Assignment::MinLoad;
        document
    }

    /// Under a rule with an in-flight cap the door counts a key's requests
    /// by the leases on file. The key keeps its worker under the cap, the
    /// request that finds the cap met is placed again and lands on a
    /// worker with fewer requests, and an end on the key's worker takes
    /// the next request back.
    #[test]
    fn a_key_at_its_cap_is_handed_on_and_returns_at_an_end() {
        use SessionVerdict::{Claimed, Pinned, Respill};
        let store = fleet(1);
        let ruled = AppliedSpec::validate(&capped(2, 2)).expect("the document validates");
        store.apply(ruled).expect("a later generation");
        let door = Door::new(Arc::clone(&store));
        // The key's first request claims a worker, and its dispatch pins
        // the key there.
        let (first, owner, verdict) = placed(&door, &keyed(b"r-1", 7, 1));
        assert_eq!(verdict, Claimed(owner));
        assert_eq!(report(&door, first, dispatched()), 1);
        // One request of the key in flight: the second stays.
        let (second, worker, verdict) = placed(&door, &keyed(b"r-2", 7, 2));
        assert_eq!((worker, verdict), (owner, Pinned));
        // Two in flight on the key's worker: the third is placed again,
        // on a worker with fewer requests.
        let (_, other, verdict) = placed(&door, &keyed(b"r-3", 7, 3));
        assert_eq!(verdict, Respill);
        assert_ne!(other, owner);
        // Another key reads its own count, and none here.
        let (_, _, verdict) = placed(&door, &keyed(b"r-4", 8, 4));
        assert!(matches!(verdict, Claimed(_)), "{verdict:?}");
        // An end on the key's worker: the next request is under the cap.
        assert_eq!(report(&door, second, gone()), 3);
        let (_, worker, verdict) = placed(&door, &keyed(b"r-5", 7, 5));
        assert_eq!((worker, verdict), (owner, Pinned));
    }

    /// A request placed again at the cap counts on the worker it lands on,
    /// whatever its pick did with the pin: on the key's worker again, on a
    /// worker under the cap, which takes the pin, or on another worker at
    /// the cap, which carries no verdict. So the key's counts hold every
    /// lease of the key on file, and an end takes one off its own worker.
    #[test]
    fn a_request_placed_at_the_cap_counts_wherever_it_lands() {
        let store = fleet(1);
        let ruled = AppliedSpec::validate(&capped(2, 1)).expect("the document validates");
        store.apply(ruled).expect("a later generation");
        let door = Door::new(Arc::clone(&store));
        let session = keyed(b"r-0", 7, 0).facts.session();
        // The key's leases on file by worker, against the store's counts.
        let holds = |placements: &[(LeaseId, WorkerId, SessionVerdict)]| {
            let mut carried = Vec::new();
            store.keyed(session, &mut carried);
            let total: u32 = carried.iter().map(|&(_, count)| count).sum();
            assert_eq!(total as usize, placements.len(), "{placements:?}");
            for (worker, count) in carried {
                let held = placements.iter().filter(|placed| placed.1 == worker);
                assert_eq!(held.count(), count as usize, "{placements:?}");
            }
        };
        let mut placements = Vec::new();
        for n in 1..=9_u8 {
            let id = [b'r', b'-', b'0' + n];
            let placement = placed(&door, &keyed(&id, 7, u64::from(n)));
            report(&door, placement.0, dispatched());
            placements.push(placement);
            holds(&placements);
        }
        // With every worker at the cap of one, a pick off the key's worker
        // carries no verdict.
        let unmoved =
            |placed: &&(LeaseId, WorkerId, SessionVerdict)| placed.2 == SessionVerdict::None;
        assert_ne!(
            placements.iter().filter(unmoved).count(),
            0,
            "{placements:?}"
        );
        // An end takes one off the worker its lease booked.
        let (first, _, _) = placements.remove(0);
        report(&door, first, gone());
        holds(&placements);
    }

    /// What a decision handed its picker of its key's count: each row's
    /// worker with the row's own count, the rows lent beside them, and the
    /// host's word that it holds the count.
    #[derive(Clone, Debug, Default)]
    struct Handed {
        rows: Vec<(WorkerId, u16)>,
        lent: Option<Vec<KeyedRow>>,
        counted: bool,
    }

    impl Handed {
        /// The workers the lent rows name, each with its count; `None`
        /// where no list was lent. No row carries a count of its own.
        fn lent(&self) -> Option<Vec<(WorkerId, u16)>> {
            assert!(self.rows.iter().all(|&(_, own)| own == 0), "{self:?}");
            let named = |keyed: &KeyedRow| (self.rows[keyed.row as usize].0, keyed.in_flight);
            let lent = self.lent.as_ref()?;
            assert!(lent.is_sorted_by(|a, b| a.row < b.row), "{self:?}");
            Some(lent.iter().map(named).collect())
        }
    }

    /// The `random` picker, keeping what each decision hands it of its
    /// key's count.
    #[derive(Debug)]
    struct Counted(Arc<Mutex<Vec<Handed>>>);

    impl Lifecycle for Counted {}

    impl Picker for Counted {
        fn name(&self) -> &'static str {
            "counted"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            let rows = candidates.iter().map(|row| (row.id, row.session_in_flight));
            self.0.lock().push(Handed {
                rows: rows.collect(),
                lent: facts.lent_session_in_flight.map(<[KeyedRow]>::to_vec),
                counted: facts.session_counted,
            });
            Random.pick(facts, candidates, kept, scores, band, out);
        }
    }

    /// A ready member that serves [`MODEL`] and states `wires`.
    fn dialed(store: &MemoryStore, name: &str, wires: WireModes) -> WorkerId {
        let key = format!("http://{name}:8000");
        let facts = WorkerFacts {
            key: key.clone(),
            engine: key,
            incarnation: 1,
            role: Role::Single,
            wire_modes: wires,
            models: vec![MODEL.to_owned()],
            dp_size: 1,
            max_running: Some(4_096),
            ..WorkerFacts::default()
        };
        let id = store.upsert(facts).id();
        store.publish_now();
        let row = WorkersStore::get(store, id).expect("the row");
        let ready = store.set_status(id, WorkerStatus::Ready, row.revision);
        ready.expect("the row stands");
        store.publish_now();
        id
    }

    /// A request under the session key `key` whose gateway states `wires`.
    fn stated(id: &[u8], key: u64, seed: u64, wires: WireModes) -> Request<'_> {
        let mut asked = keyed(id, key, seed);
        asked.constraints.wire_modes = wires;
        asked
    }

    /// Applies `document` to `store` with the counting picker as the pick
    /// of its `manual` profile, under a rule whose keys the profile's own
    /// stages place: the picker is handed each key no pin takes and each
    /// key placed again at its cap. Returns the reader of what the last
    /// decision handed the picker.
    fn counting(store: &MemoryStore, mut document: PolicySpec) -> impl Fn() -> Handed {
        let handed = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&handed);
        let mut catalog = Catalog::builtin();
        let added = catalog.picker("counted", move || Box::new(Counted(Arc::clone(&kept))));
        added.expect("a free name");
        let profile = document.profiles.get_mut("manual");
        profile.expect("the preset's profile").pick = "counted".into();
        let rule = document.bindings[0].sessions.as_mut();
        rule.expect("the preset's rule").assignment = crate::Assignment::Delegate;
        let digest = document.digest().expect("the document digests");
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[]);
        let applied = store.apply(spec.expect("the catalog holds the stage"));
        applied.expect("a later generation");
        move || {
            let handed = handed.lock();
            let last: Option<&Handed> = handed.last();
            last.cloned().expect("a decision reached the picker")
        }
    }

    /// The door hands a keyed decision its key's count beside the rows, by
    /// row, and writes none into a row: over the publication's own rows,
    /// and over the rows written for a gateway that states its wires. A
    /// worker that is no candidate of the decision is dropped from the
    /// count, and a request with no key is handed none.
    #[test]
    fn the_door_hands_a_keys_count_beside_the_rows_and_writes_none_into_a_row() {
        let store = fleet(1);
        // A cap of one. The picker's draw may land on the key's worker
        // again, which is the pin taken.
        let last = counting(&store, capped(2, 1));
        let door = Door::new(Arc::clone(&store));
        // A key's first request: none of its requests is in flight.
        let (first, owner, _) = placed(&door, &keyed(b"r-1", 7, 1));
        assert_eq!((last().lent(), last().counted), (Some(vec![]), true));
        report(&door, first, dispatched());
        // The second finds the cap met and is placed again: the picker
        // reads the first on the key's worker, beside rows that carry no
        // count.
        let (_, other, verdict) = placed(&door, &keyed(b"r-2", 7, 2));
        assert_eq!(verdict == SessionVerdict::Pinned, other == owner);
        assert_eq!(verdict == SessionVerdict::Respill, other != owner);
        assert_eq!(last().lent(), Some(vec![(owner, 1)]));
        assert_eq!(last().rows.len(), 3);
        // A request with no key is handed no count, a list its host lent
        // included: the door decides over the store's rows, and a host's
        // list names rows of its own.
        let hosts = [KeyedRow {
            row: 0,
            in_flight: 3,
        }];
        let mut hosted = request(b"r-3", 3);
        hosted.facts.lent_session_in_flight = Some(&hosts);
        hosted.facts.session_counted = true;
        placed(&door, &hosted);
        assert_eq!((last().lent(), last().counted), (None, false));
        // Rows written for a gateway that states its wires. Two members
        // that speak them join. The key's workers are none of that
        // gateway's candidates, so their counts are dropped.
        let wires = WireModes::GRPC;
        let joined = [dialed(&store, "w3", wires), dialed(&store, "w4", wires)];
        placed(&door, &stated(b"r-4", 7, 4, wires));
        assert_eq!(last().lent(), Some(vec![]));
        assert_eq!(last().rows.len(), 2);
        // Another key's request stands on one of the two: its count names
        // that member's row of the written list.
        let (lease, held, _) = placed(&door, &stated(b"r-5", 9, 5, wires));
        assert!(joined.contains(&held));
        report(&door, lease, dispatched());
        let (_, next, verdict) = placed(&door, &stated(b"r-6", 9, 6, wires));
        assert_eq!(verdict == SessionVerdict::Pinned, next == held);
        assert_eq!(verdict == SessionVerdict::Respill, next != held);
        assert_eq!(last().lent(), Some(vec![(held, 1)]));
        assert_eq!(last().rows.len(), 2);
    }

    /// A profile whose own stages read the affinity column is asked over
    /// the rows the store names for it ([`Decides::over_named`]). A keyed
    /// decision reads its key's count at that call as at the other: the
    /// picker is handed the count beside the rows, and each lease books the
    /// key.
    #[test]
    fn a_decision_over_named_rows_reads_its_keys_count() {
        let store = fleet(1);
        let mut document = capped(2, 1);
        let profile = document.profiles.get_mut("manual");
        let profile = profile.expect("the preset's profile");
        profile.score.push("prefix-affinity".into());
        let last = counting(&store, document);
        let named = store.with_world(|world| {
            let engine = world.spec().engine();
            engine.named_for(MODEL, Phase::Single)
        });
        assert_ne!(named, 0, "the profile is named its rows");
        let door = Door::new(Arc::clone(&store));
        let (first, owner, _) = placed(&door, &keyed(b"r-1", 7, 1));
        assert_eq!((last().lent(), last().counted), (Some(vec![]), true));
        report(&door, first, dispatched());
        // The second finds the cap met: the picker reads the first on the
        // key's worker.
        placed(&door, &keyed(b"r-2", 7, 2));
        assert_eq!(last().lent(), Some(vec![(owner, 1)]));
        // Each of the two leases books the key.
        let mut carried = Vec::new();
        store.keyed(keyed(b"r-0", 7, 0).facts.session(), &mut carried);
        let booked: u32 = carried.iter().map(|&(_, count)| count).sum();
        assert_eq!(booked, 2, "{carried:?}");
    }

    /// Reads one signal of every row it scores and keeps what it read.
    #[derive(Debug)]
    struct ReadsHeld {
        held: crate::SignalColumn,
        read: Arc<Mutex<Vec<Option<f64>>>>,
    }

    impl Lifecycle for ReadsHeld {}

    impl crate::Scorer for ReadsHeld {
        fn name(&self) -> &'static str {
            "reads-held"
        }

        fn score(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            let rows = kept.iter().map(|&row| &candidates[row as usize]);
            let read = rows.map(|row| self.held.read(facts, row));
            self.read.lock().extend(read);
            out.fill(0.0);
            TieBand::NORMALIZED
        }

        fn needs(&self) -> Needs {
            Needs {
                signal_cells: true,
                ..Needs::default()
            }
        }
    }

    /// The door reads its own view's cells, never the request's: cells a
    /// caller hands on the request's facts are by its own list's positions.
    /// Here the caller's cells hold a value at every position the store's
    /// pool has and the store holds none, and every read through the door
    /// is no value.
    #[test]
    fn the_door_reads_none_of_the_cells_a_caller_hands_on_the_request() {
        let store = fleet(1);
        let read = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&read);
        let mut catalog = Catalog::builtin();
        let added = catalog.scorer_with("reads-held", move |build| {
            Ok(Box::new(ReadsHeld {
                held: build.signal_column("held")?,
                read: Arc::clone(&kept),
            }))
        });
        added.expect("a free name");
        let document: PolicySpec = serde_json::from_str(
            r#"{"apiVersion":"smg.policy/v1","generation":2,
                "signals":{"held":{"unit":"tokens"}},
                "profiles":{"p":{"score":["reads-held"],"pick":"best"}}}"#,
        )
        .expect("a well-formed document");
        let digest = document.digest().expect("the document digests");
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[]);
        let applied = store.apply(spec.expect("the catalog holds the stage"));
        applied.expect("a later generation");
        let door = Door::new(Arc::clone(&store));
        let cells = [crate::SignalCell {
            value: 7.0,
            sampled_at_micros: 0,
        }; 8];
        let cells = crate::SignalCells::new(cells.to_vec(), 1);
        let mut hosted = request(b"r-1", 1);
        hosted.facts.signal_cells = Some(&cells);
        placed(&door, &hosted);
        let read = read.lock();
        assert!(!read.is_empty(), "the stage was asked");
        assert!(read.iter().all(Option::is_none), "{read:?}");
    }

    /// A cap full of holders the pass refuses is read past: the cases, over
    /// the `cache_aware` preset.
    mod refused_holders {
        use std::{
            cmp::Reverse,
            sync::atomic::AtomicU64,
            time::{Duration, Instant},
        };

        use kv_index::chain_prefix_hash;

        use super::{request, GATEWAY, MODEL};
        use crate::{
            demands::{DemandFacts, Demands},
            door::{hand, Buffers, Request, Scratch, Taken},
            spec::{AppliedSpec, BestParams, OverloadSpec, PickStage, PolicySpec},
            store::{
                Ask, BatchFacts, BatchOutcome, ContentHash, HealthSink, KvEvent, KvEventSink,
                LoadSink, LoadSource, MemoryStore, Named, Naming, SequenceHash, Signals,
                StoredBlock, VetoWrite, ViewBuffers, WorkerFacts, WorkerStatus, WorkersStore,
            },
            view::{Candidate, Constraints, LoadRow, PoolFacts, RequestFacts, Role, WorkerId},
        };

        /// The blocks of the cases' request.
        const BLOCKS: [u64; 3] = [1, 2, 3];

        /// A store under the `cache_aware` preset whose picker takes a
        /// sample of `cap` rows, which caps a naming at `cap`. A worker
        /// whose engine reports a queued request reads over its thresholds,
        /// at once: the store publishes with no window.
        fn sampled(cap: u32) -> MemoryStore {
            let mut document = PolicySpec::preset("cache_aware").expect("a preset of this build");
            let profile = document.profiles.get_mut("cache_aware");
            let profile = profile.expect("the preset's profile");
            profile.pick = PickStage::Best(BestParams { sample: cap }).into();
            document.bindings[0].overload = Some(OverloadSpec {
                waiting_requests: Some(1),
                ..OverloadSpec::default()
            });
            let store = MemoryStore::with_publish_window(Duration::ZERO);
            let spec = AppliedSpec::validate(&document).expect("the document validates");
            store.apply(spec).expect("above the unbound start");
            store
        }

        /// A ready member that states `runtime`, reports its blocks and
        /// holds the first `depth` blocks of the request.
        fn holder_at(store: &MemoryStore, name: &str, depth: usize, runtime: u32) -> WorkerId {
            let facts = WorkerFacts {
                key: name.to_owned(),
                engine: name.to_owned(),
                incarnation: 1,
                role: Role::Single,
                models: vec![MODEL.to_owned()],
                dp_size: 1,
                signals: Signals {
                    events: true,
                    ..Signals::default()
                },
                demand_facts: DemandFacts { runtime },
                ..WorkerFacts::default()
            };
            let id = store.upsert(facts).id();
            let row = WorkersStore::get(store, id).expect("the row");
            if row.status != WorkerStatus::Ready {
                let ready = store.set_status(id, WorkerStatus::Ready, row.revision);
                ready.expect("the row stands");
            }
            store.publish_now();
            let mut blocks: Vec<StoredBlock> = Vec::with_capacity(depth);
            for &content in &BLOCKS[..depth] {
                let seq_hash = match blocks.last() {
                    Some(parent) => chain_prefix_hash(parent.seq_hash, ContentHash(content)),
                    None => SequenceHash(content),
                };
                let content_hash = ContentHash(content);
                blocks.push(StoredBlock {
                    seq_hash,
                    content_hash,
                });
            }
            if !blocks.is_empty() {
                let parent = None;
                let events = [KvEvent::Stored { parent, blocks }];
                let batch = BatchFacts {
                    sequence: 1,
                    ..BatchFacts::default()
                };
                assert_eq!(store.admit(id, 0, batch, &events), BatchOutcome::Applied);
            }
            id
        }

        /// [`holder_at`] for a member that states no runtime.
        fn holder(store: &MemoryStore, name: &str, depth: usize) -> WorkerId {
            holder_at(store, name, depth, 0)
        }

        /// What a decision is handed and decides: the holders named, by
        /// member with their depths, the deepest first, the members that
        /// read over their thresholds, and the engine's pick over the rows
        /// with the overlap it states.
        struct Decided {
            holders: Vec<(WorkerId, u32)>,
            over: Vec<WorkerId>,
            pick: Option<(WorkerId, u16)>,
        }

        /// The holders of `named` by their members in `rows`, the deepest
        /// first.
        fn by_depth(named: &[crate::NamedHolder], rows: &[Candidate]) -> Vec<(WorkerId, u32)> {
            let held =
                |holder: &crate::NamedHolder| (rows[holder.row as usize].id, holder.overlap_blocks);
            let mut holders: Vec<(WorkerId, u32)> = named.iter().map(held).collect();
            holders.sort_by_key(|&(_, blocks)| Reverse(blocks));
            holders
        }

        /// The decision of `asked` through the door's hand, by the engine.
        fn decided(store: &MemoryStore, asked: &Request<'_>) -> Decided {
            let taken = Taken {
                request: asked,
                gateway: GATEWAY,
                now: Instant::now(),
                left: Duration::MAX,
                pins: None,
                place: None,
            };
            let mut buffers = Buffers::default();
            let undialed = AtomicU64::new(0);
            store.deciding(|world| {
                let engine = world.spec().engine();
                let read = |facts: &RequestFacts<'_>,
                            pool: &PoolFacts<'_>,
                            rows: &[Candidate],
                            cx: &Constraints<'_>,
                            scratch: &mut Scratch| {
                    let named = facts.named.expect("the profile names its rows");
                    let holders = by_depth(named.holders, rows);
                    let over = rows.iter().filter(|row| row.health.overloaded);
                    let mut over: Vec<WorkerId> = over.map(|row| row.id).collect();
                    over.sort_unstable();
                    // The engine's form over named rows: it states the
                    // pick's overlap.
                    let decision = engine.decide_stated(facts, pool, rows, cx, scratch);
                    let pick = decision.pick().and_then(|pick| {
                        let first = pick.targets.first()?;
                        Some((first.id, pick.overlap().blocks))
                    });
                    Decided {
                        holders,
                        over,
                        pick,
                    }
                };
                hand(world, &taken, &mut buffers, &undialed, read).0
            })
        }

        /// The holders a naming under no test names for `facts` under a cap
        /// of `cap`, as the door names them: by member, the deepest first.
        fn plainly_named(
            store: &MemoryStore,
            facts: &RequestFacts<'_>,
            cap: u32,
        ) -> Vec<(WorkerId, u32)> {
            let mut view = ViewBuffers::default();
            let mut named = Named::default();
            store.deciding(|world| {
                let ask = Ask {
                    gateway: GATEWAY,
                    now: Instant::now(),
                    pinned: None,
                    affinity: false,
                    signal_cells: false,
                };
                let read = world.view(MODEL, Role::Single, &ask, false, &mut view);
                let read = read.expect("the pool");
                let naming = Naming {
                    gateway: GATEWAY,
                    block_hashes: facts.block_hashes,
                    namespace_seed: facts.namespace_seed,
                    seed: facts.seed,
                    cap,
                    wires: None,
                    now_micros: read.built.now_micros,
                };
                world.name(MODEL, Role::Single, &naming, read.candidates, &mut named);
                by_depth(&named.holders, read.candidates)
            })
        }

        /// A host's filter that reads the column the naming writes is asked
        /// by its pass alone. Ahead of the naming no row holds the column:
        /// a test made there would refuse the deep holder, which the filter
        /// keeps once it is named, and the lookup made again would name the
        /// shallow one in its place.
        #[test]
        fn a_filter_that_reads_the_naming_is_asked_by_its_pass_alone() {
            use crate::{Catalog, Filter, Needs};

            /// Keeps one worker only where the naming says it holds a block.
            #[derive(Debug)]
            struct Warm(WorkerId);

            impl Filter for Warm {
                fn name(&self) -> &'static str {
                    "warm"
                }

                fn row_wise(&self) -> bool {
                    true
                }

                fn needs(&self) -> Needs {
                    Needs {
                        affinity: true,
                        ..Needs::default()
                    }
                }

                fn keep(
                    &self,
                    facts: &RequestFacts<'_>,
                    row: &Candidate,
                    _: &Constraints<'_>,
                ) -> bool {
                    row.id != self.0 || facts.affinity_at(row.position).overlap_blocks != 0
                }
            }

            let store = sampled(1);
            let deep = holder(&store, "deep", 3);
            holder(&store, "shallow", 1);
            holder(&store, "idle", 0);
            let mut catalog = Catalog::builtin();
            let made = catalog.filter("warm", move || Box::new(Warm(deep)));
            made.expect("a name the catalog does not hold");
            let mut document = PolicySpec::preset("cache_aware").expect("a preset of this build");
            document.generation += 1;
            let profile = document.profiles.get_mut("cache_aware");
            let profile = profile.expect("the preset's profile");
            profile.pick = PickStage::Best(BestParams { sample: 1 }).into();
            profile.filter.push("warm".into());
            let spec = AppliedSpec::validate_in(&document, &catalog, &[]);
            let applied = store.apply(spec.expect("the document validates"));
            applied.expect("the next generation");
            store.publish_now();
            let mut asked = request(b"warm", 7);
            asked.facts.block_hashes = &BLOCKS;
            let decided = decided(&store, &asked);
            assert_eq!(decided.holders, [(deep, 3)]);
            assert_eq!(decided.pick, Some((deep, 3)));
        }

        /// A host's filter that reads the key's counts is asked by its pass
        /// alone. The counts are lent behind the naming: a test made ahead
        /// of them would refuse the deep holder, where the key has a
        /// request in flight, and the shallow one would be picked.
        #[test]
        fn a_filter_that_reads_the_keys_counts_is_asked_by_its_pass_alone() {
            use std::sync::Arc;

            use crate::{
                door::Door,
                spec::{PrefixAffinityParams, ScoreStage},
                view::Identity,
                Catalog, Filter, Needs,
            };

            /// Keeps one worker for a streaming request only where its key
            /// has a request in flight there.
            #[derive(Debug)]
            struct KeyLocal(WorkerId);

            impl Filter for KeyLocal {
                fn name(&self) -> &'static str {
                    "key-local"
                }

                fn row_wise(&self) -> bool {
                    true
                }

                fn needs(&self) -> Needs {
                    Needs {
                        session_in_flight: true,
                        ..Needs::default()
                    }
                }

                fn keep(
                    &self,
                    facts: &RequestFacts<'_>,
                    row: &Candidate,
                    _: &Constraints<'_>,
                ) -> bool {
                    row.id != self.0 || !facts.streaming || facts.session_in_flight(row) > 0
                }
            }

            let store = sampled(1);
            let deep = holder(&store, "deep", 3);
            holder(&store, "shallow", 2);
            holder(&store, "idle", 0);
            let mut catalog = Catalog::builtin();
            let made = catalog.filter("key-local", move || Box::new(KeyLocal(deep)));
            made.expect("a name the catalog does not hold");
            let mut document = PolicySpec::preset("cache_aware").expect("a preset of this build");
            document.generation += 1;
            let profile = document.profiles.get_mut("cache_aware");
            let profile = profile.expect("the preset's profile");
            profile.pick = PickStage::Best(BestParams { sample: 1 }).into();
            profile.filter.push("key-local".into());
            profile.score =
                vec![ScoreStage::PrefixAffinity(PrefixAffinityParams::default()).into()];
            let spec = AppliedSpec::validate_in(&document, &catalog, &[]);
            let applied = store.apply(spec.expect("the document validates"));
            applied.expect("the next generation");
            let door = Door::new(Arc::new(store));
            let now = Instant::now();
            // The key's first request goes to the deep holder and stays in
            // flight there.
            let mut first = request(b"first", 7);
            first.facts.block_hashes = &BLOCKS;
            first.facts.session_key = Identity::hashed(123);
            let primed = door.decide(GATEWAY, &first, now).expect("the key's first");
            let target = primed.decision.pick().and_then(|pick| pick.targets.first());
            assert_eq!(target.map(|target| target.id), Some(deep));
            assert!(primed.lease.is_some(), "the key's lease is live");
            // Its second, streaming: the filter keeps the deep holder by
            // the key's count there.
            let mut second = request(b"second", 7);
            second.facts.block_hashes = &BLOCKS;
            second.facts.session_key = first.facts.session_key;
            second.facts.streaming = true;
            let second = door
                .decide(GATEWAY, &second, now)
                .expect("the key's second");
            let pick = second.decision.pick().expect("a pick");
            let picked = pick.targets.first().map(|target| target.id);
            assert_eq!((picked, pick.overlap().blocks), (Some(deep), 3));
        }

        /// Rows a demand took take no round of the lookup made again: it
        /// is kept within the rows that stay. Five deeper holders of
        /// another runtime stand ahead of the warm one, more than the
        /// rounds: the retry is named the warm one all the same.
        #[test]
        fn rows_a_demand_took_take_no_round_of_the_lookup_made_again() {
            let store = sampled(1);
            let tried = [holder_at(&store, "tried", 3, 7)];
            for other in 0..5 {
                holder_at(&store, &format!("other-runtime-{other}"), 2, 9);
            }
            let warm = holder_at(&store, "warm", 1, 7);
            holder_at(&store, "idle", 0, 7);
            let mut asked = request(b"cut", 7);
            asked.facts.block_hashes = &BLOCKS;
            asked.facts.tried = &tried;
            asked.constraints.demands = Demands { runtime: 7 };
            let decided = decided(&store, &asked);
            assert_eq!(decided.holders, [(warm, 1)]);
            assert_eq!(decided.pick, Some((warm, 1)));
        }

        /// A retry is named the holder behind the one it tried. Under a
        /// cap of one the deeper holder would fill the cap, and it is the
        /// one row the retry may not take: the shallower one is named, and
        /// the pick states that worker's own overlap.
        #[test]
        fn a_retry_is_named_the_holder_behind_the_one_it_tried() {
            let store = sampled(1);
            let deep = holder(&store, "deep", 3);
            let shallow = holder(&store, "shallow", 2);
            holder(&store, "idle", 0);
            let tried = [deep];
            let mut asked = request(b"retry", 7);
            asked.facts.block_hashes = &BLOCKS;
            asked.facts.tried = &tried;
            let decided = decided(&store, &asked);
            assert_eq!(decided.holders, [(shallow, 2)]);
            assert_eq!(decided.pick, Some((shallow, 2)));
        }

        /// A hot prefix: the two workers that hold it whole read over
        /// their thresholds and would fill a cap of two. Other rows stand
        /// under theirs, so no veto is lifted, and the holder under its
        /// thresholds is named.
        #[test]
        fn a_cap_full_of_holders_over_their_thresholds_names_the_one_under_them() {
            let store = sampled(2);
            let mut hot = [holder(&store, "hot-a", 3), holder(&store, "hot-b", 3)];
            hot.sort_unstable();
            let cool = holder(&store, "cool", 1);
            holder(&store, "idle", 0);
            let now = Instant::now();
            for id in hot {
                let queued = LoadRow {
                    queued: 1,
                    ..LoadRow::default()
                };
                store.record(id, 0, queued, now, LoadSource::Stream);
            }
            store.publish_now();
            let mut asked = request(b"hot", 7);
            asked.facts.block_hashes = &BLOCKS;
            let decided = decided(&store, &asked);
            assert_eq!(decided.over, hot, "the two read over their thresholds");
            assert_eq!(decided.holders, [(cool, 1)]);
            assert_eq!(decided.pick, Some((cool, 1)));
        }

        /// A cap full of holders behind the asking gateway's veto is read
        /// past, under a doubt as under a hold: the holder the gateway
        /// vetoes nothing of is named.
        #[test]
        fn a_cap_full_of_holders_behind_the_gateways_veto_is_read_past() {
            let store = sampled(2);
            let doubted = holder(&store, "doubted", 3);
            let held = holder(&store, "held", 3);
            let free = holder(&store, "free", 1);
            holder(&store, "idle", 0);
            let now = Instant::now();
            let until = now + Duration::from_secs(60);
            let veto = |member: WorkerId, hold: Option<Instant>, stands: bool| {
                let written =
                    store.gateway_veto_noted(GATEWAY, member, Some(until), hold, stands, now);
                assert_eq!(written, VetoWrite::Written);
            };
            veto(doubted, None, false);
            veto(held, Some(until), false);
            let mut asked = request(b"veto", 7);
            asked.facts.block_hashes = &BLOCKS;
            let decided = decided(&store, &asked);
            assert_eq!(decided.holders, [(free, 1)]);
            assert_eq!(decided.pick, Some((free, 1)));
        }

        /// A holder that is not ready is read past where it fills the cap.
        #[test]
        fn a_holder_that_is_not_ready_is_read_past_where_it_fills_the_cap() {
            let store = sampled(1);
            let down = holder(&store, "down", 3);
            let up = holder(&store, "up", 2);
            holder(&store, "idle", 0);
            let row = WorkersStore::get(&store, down).expect("the row");
            let unready = store.set_status(down, WorkerStatus::NotReady, row.revision);
            unready.expect("the row stands");
            store.publish_now();
            let mut asked = request(b"down", 7);
            asked.facts.block_hashes = &BLOCKS;
            let decided = decided(&store, &asked);
            assert_eq!(decided.holders, [(up, 2)]);
            assert_eq!(decided.pick, Some((up, 2)));
        }

        /// A naming no holder of which is refused names what a naming
        /// under no test names: the same rows by the same draw. Three
        /// holders stand at one depth under a cap of two, so the decision's
        /// seed draws the two.
        #[test]
        fn a_naming_no_holder_of_which_is_refused_names_what_main_names() {
            let store = sampled(2);
            for name in ["a", "b", "c"] {
                holder(&store, name, 2);
            }
            holder(&store, "idle", 0);
            let mut drawn = std::collections::BTreeSet::new();
            for seed in 0..32 {
                let mut asked = request(b"plain", seed);
                asked.facts.block_hashes = &BLOCKS;
                let plain = plainly_named(&store, &asked.facts, 2);
                assert_eq!(plain.len(), 2, "seed {seed}");
                assert_eq!(decided(&store, &asked).holders, plain, "seed {seed}");
                drawn.insert(plain);
            }
            assert!(drawn.len() > 1, "the seed draws the two: {drawn:?}");
        }

        /// A naming one of whose holders passes stands as it was made. The
        /// retry tried the deeper of the two a cap of two names and the
        /// other passes: no lookup is made again, the tried row stays named
        /// for the pass to drop, and the holder behind the two is not read.
        #[test]
        fn a_naming_one_holder_of_which_passes_stands_as_made() {
            let store = sampled(2);
            let deep = holder(&store, "deep", 3);
            let mid = holder(&store, "mid", 2);
            holder(&store, "shallow", 1);
            holder(&store, "idle", 0);
            let tried = [deep];
            let mut asked = request(b"stands", 7);
            asked.facts.block_hashes = &BLOCKS;
            asked.facts.tried = &tried;
            let decided = decided(&store, &asked);
            assert_eq!(decided.holders, plainly_named(&store, &asked.facts, 2));
            assert_eq!(decided.holders, [(deep, 3), (mid, 2)]);
            assert_eq!(decided.pick, Some((mid, 2)));
        }

        /// Where no holder passes the first read stands. Every row was
        /// tried, so the pass's soft filter steps aside and reads the names
        /// a naming under no test hands it.
        #[test]
        fn where_no_holder_passes_the_first_read_stands() {
            let store = sampled(1);
            let deep = holder(&store, "deep", 3);
            let shallow = holder(&store, "shallow", 2);
            let idle = holder(&store, "idle", 0);
            let tried = [deep, shallow, idle];
            let mut asked = request(b"all", 7);
            asked.facts.block_hashes = &BLOCKS;
            asked.facts.tried = &tried;
            let decided = decided(&store, &asked);
            assert_eq!(decided.holders, plainly_named(&store, &asked.facts, 1));
            assert_eq!(decided.holders, [(deep, 3)]);
            assert_eq!(decided.pick, Some((deep, 3)));
        }

        /// A view a demand cut is named within the rows that stay, and
        /// its holders are held to the test alike. The demand takes the
        /// deepest holder and the retry tried the next: the third is named.
        #[test]
        fn a_cut_view_is_named_within_the_rows_that_stay_and_asks_the_test() {
            let store = sampled(1);
            holder_at(&store, "gone", 3, 9);
            let deep = holder_at(&store, "deep", 2, 7);
            let shallow = holder_at(&store, "shallow", 1, 7);
            holder_at(&store, "idle", 0, 7);
            let tried = [deep];
            let mut asked = request(b"cut", 7);
            asked.facts.block_hashes = &BLOCKS;
            asked.facts.tried = &tried;
            asked.constraints.demands = Demands { runtime: 7 };
            let decided = decided(&store, &asked);
            assert_eq!(decided.holders, [(shallow, 1)]);
            assert_eq!(decided.pick, Some((shallow, 1)));
        }
    }

    /// One read of a key's shard names the rows that carry the key's
    /// requests, ascending, with no row for a worker that is no candidate,
    /// and keeps the key's pin of record for the engine's ask as the read
    /// found it. Another key is asked of the store.
    #[test]
    fn a_keys_read_names_its_rows_in_order_and_keeps_the_pin_it_found() {
        use super::{Keyed, StorePins};
        use crate::view::PinSource;
        const KEY: Session = Session::from_raw(41);
        let store = MemoryStore::new();
        let now = Instant::now();
        LeasesStore::set_term(&store, 1);
        // Requests of the key in flight on three workers, the last of
        // which has no row among the candidates.
        for (id, worker) in [(1, 30), (2, 10), (3, 30), (4, 99)] {
            let record = LeaseRecord {
                lease: Lease {
                    id: LeaseId(id),
                    session: KEY,
                    session_verdict: SessionVerdict::Pinned,
                    ..Lease::default()
                },
                gateway: GATEWAY,
                request: id,
                target: WorkerId(worker),
                picked: Picked::default(),
                decode: None,
                term: 1,
                issued_at: now,
                dispatch_deadline: now + Duration::from_secs(5),
                state: LeaseState::ISSUED,
            };
            LeasesStore::issue(&store, record).expect("filed");
        }
        SessionsStore::pin(&store, KEY.get(), WorkerId(30), now, None);
        let candidates: Vec<Candidate> = (0..)
            .zip([10, 20, 30])
            .map(|(position, id)| Candidate {
                position,
                id: WorkerId(id),
                ..Candidate::default()
            })
            .collect();
        let pins = StorePins { store: &store, now };
        let mut keyed = Keyed::default();
        let read = keyed.read(&pins, KEY, &candidates);
        let (read, rows) = read.expect("a keyed request");
        let row = |row, in_flight| KeyedRow { row, in_flight };
        assert_eq!(rows, [row(0, 1), row(2, 2)]);
        assert_eq!(read.pinned(KEY, None), Some(WorkerId(30)));
        // A pin written behind the read is the next decision's.
        SessionsStore::pin(&store, KEY.get(), WorkerId(10), now, None);
        assert_eq!(read.pinned(KEY, None), Some(WorkerId(30)));
        let other = Session::from_raw(42);
        SessionsStore::pin(&store, other.get(), WorkerId(20), now, None);
        assert_eq!(read.pinned(other, None), Some(WorkerId(20)));
        // A request with no key reads nothing.
        assert!(keyed.read(&pins, Session::NONE, &candidates).is_none());
    }

    /// The `random` picker, keeping the budget each request tells it.
    #[derive(Debug)]
    struct Budgets(Arc<Mutex<Vec<u32>>>);

    impl Lifecycle for Budgets {}

    impl Picker for Budgets {
        fn name(&self) -> &'static str {
            "budgets"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            self.0.lock().push(facts.budget_micros);
            Random.pick(facts, candidates, kept, scores, band, out);
        }
    }

    /// The engine is told the budget a request's host states. A request
    /// that states none tells no bound: the most the field holds.
    #[test]
    fn the_engine_is_told_the_budget_a_request_states() {
        let store = fleet(1);
        let told = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&told);
        picking(&store, 2, "budgets", move || {
            Box::new(Budgets(Arc::clone(&kept)))
        });
        let door = Door::new(Arc::clone(&store));
        leased(&door, b"r-1", 7);
        let mut bounded = request(b"r-2", 8);
        bounded.facts.budget_micros = 5_000;
        let decided = door.decide(GATEWAY, &bounded, Instant::now());
        assert!(decided.is_ok_and(|decided| decided.lease.is_some()));
        assert_eq!(*told.lock(), [u32::MAX, 5_000]);
    }

    /// A store that names no decider has one file, and the door decides
    /// for it as decider 0. The engine hears each report of a lease under
    /// that name too, whichever thread applies it: a report's own thread,
    /// or the pass of the door's clock.
    #[test]
    fn a_store_that_names_no_decider_hears_its_leases_as_decider_zero() {
        let store = stocked(MemoryStore::new());
        let heard = Heard::default();
        let kept = Arc::clone(&heard);
        picking(&store, 2, "hears", move || {
            Box::new(Hears(Arc::clone(&kept)))
        });
        let door = Door::new(Arc::clone(&store));
        let (first, served) = leased(&door, b"r-1", 7);
        assert_eq!(report(&door, first, dispatched()), 1);
        assert_eq!(report(&door, first, completed(served)), 0);
        let decided = door.decide(GATEWAY, &request(b"r-2", 8), Instant::now());
        let second = decided.expect("a decision").lease;
        let due = second.expect("a pick's lease").due;
        assert_eq!(door.sweep(due), Swept { leases: 1, pins: 0 });
        let by = Some(Decider::new(0));
        let calls = ["issued", "dispatched", "completed", "issued", "released"];
        assert_eq!(*heard.lock(), calls.map(|call| (call, by)));
        assert_eq!(store.lifecycle_faults(), 0);
    }

    /// A word the door keeps beside a full ring is said: until its caller
    /// has handed it on, the pass of the door's clock ends no lease at
    /// that file. Here a lease's dispatch stands kept at the end of the
    /// lease's span, as it does between a caller's rest and its hand: the
    /// pass leaves the lease on file, and the word handed on finds it.
    #[test]
    fn a_sweep_ends_no_lease_at_a_file_whose_kept_words_wait() {
        let store = fleet(1);
        let door = Door::new(Arc::clone(&store));
        let decided = door.decide(GATEWAY, &request(b"r-1", 7), Instant::now());
        let leased = decided.expect("a decision").lease;
        let leased = leased.expect("a pick's lease");
        let id = leased.lease.id;
        // What a caller leaves between its rest and its hand: the word
        // kept and counted, and no thread at work at the file.
        let kept = &door.kept[0];
        kept.words.lock().push((GATEWAY, id, dispatched()));
        kept.waits.fetch_add(1, Ordering::SeqCst);
        assert_eq!(door.sweep(leased.due), Swept::default(), "its word waits");
        door.hand_kept(0);
        assert_eq!(kept.waits.load(Ordering::SeqCst), 0);
        let on_file = LeasesStore::get(&*store, GATEWAY, id).expect("the lease is on file");
        assert!(matches!(on_file.state, LeaseState::Dispatched { .. }));
        assert_eq!(door.sweep(leased.due), Swept::default(), "dispatched");
        assert_eq!(counted(&store).len(), 1);
    }

    /// The pin's writer reads its lease's rule off the issued state, and
    /// the door tells a dispatch once, with the record as it stood
    /// issued. A record past that state, were one told, would read as a
    /// rule with no TTL: it pins nothing, and a pin that stands keeps
    /// its end.
    #[test]
    fn a_record_past_its_issue_pins_nothing_by_its_leases_rule() {
        const KEY: Session = Session::from_raw(41);
        let store = MemoryStore::new();
        let t0 = Instant::now();
        let mut record = LeaseRecord {
            lease: Lease {
                id: LeaseId(1),
                session: KEY,
                session_verdict: SessionVerdict::Claimed(WorkerId(7)),
                issued_micros: store.micros(t0),
                ..Lease::default()
            },
            gateway: GATEWAY,
            request: 1,
            target: WorkerId(7),
            picked: Picked::default(),
            decode: None,
            term: 0,
            issued_at: t0,
            dispatch_deadline: t0 + Duration::from_secs(5),
            state: LeaseState::Dispatched { at: t0 },
        };
        pinned(&store, &record.lease, record.state, None, WorkerId(7), t0);
        assert!(SessionsStore::lookup(&store, KEY.get()).is_none());
        // As issued under a rule of a minute it pins to the minute's
        // end; told once more as the door wrote it, it leaves that end.
        record.state = LeaseState::issued(Some(60_000_000));
        pinned(&store, &record.lease, record.state, None, WorkerId(7), t0);
        record.state = LeaseState::Dispatched { at: t0 };
        pinned(&store, &record.lease, record.state, None, WorkerId(7), t0);
        let pin = SessionsStore::lookup(&store, KEY.get()).expect("a pin");
        let ends = store.micros(t0) + 60_000_000;
        assert_eq!((pin.worker, pin.ends), (WorkerId(7), Some(ends)));
    }

    /// A caller is not held behind another's decision: where its seat is
    /// held it decides at the next free one while that decision stands,
    /// files there, and keeps that seat for its next decision.
    #[test]
    fn a_caller_whose_seat_is_held_decides_at_the_next_and_keeps_it() {
        let store = fleet(3);
        let door = Arc::new(Door::new(Arc::clone(&store)));
        let ends = Ends::over(&store);
        // The held request stands at its lease's issue, at seat 0: the
        // scene's stop at an issue is the next issue's, so no other
        // decision is made between its pick and its issue.
        let held = held_at(&door, 0);
        ends.stands("pick");
        ends.go();
        ends.stands("issue");
        let (asked, decided) = mpsc::channel();
        let caller = Arc::clone(&door);
        thread::spawn(move || {
            seated(0);
            let leases = [b"r-1", b"r-2"].map(|id| leased(&caller, id, 7).0);
            asked.send((leases, SEAT.get())).ok();
        });
        let beside = decided.recv_timeout(PATIENCE);
        let (leases, kept) = beside.expect("it decides while the held decision stands");
        assert_eq!(leases.map(|lease| store.file_of(lease)), [1, 1]);
        assert_eq!(kept, 1, "the seat it moved to is the thread's");
        ends.go();
        let stood = held.join().expect("the held thread").expect("its lease");
        assert_eq!(store.file_of(stood), 0);
        assert_eq!(LeasesStore::live(&*store, GATEWAY), 3);
        assert_eq!(door.faults(), 0);
    }

    /// Threads start at seats of their own: a thread's first seat is its
    /// number among the threads that have called a door, and it keeps the
    /// seat while no other thread holds it.
    #[test]
    fn two_threads_that_never_called_start_at_two_seats() {
        // As many seats as a store lays: two threads' numbers name two of
        // them unless 255 other threads first call a door between them.
        let store = fleet(Decider::COUNT);
        let door = Door::new(Arc::clone(&store));
        let seats = [0_u8, 1].map(|caller| {
            thread::scope(|scope| {
                let asked = scope.spawn(|| {
                    let (first, _) = leased(&door, &[caller, 0], 7);
                    let (second, _) = leased(&door, &[caller, 1], 8);
                    (store.file_of(first), store.file_of(second))
                });
                let (first, second) = asked.join().expect("the caller's thread");
                assert_eq!(first, second, "a thread keeps its seat");
                first
            })
        });
        assert_ne!(seats[0], seats[1]);
    }

    /// A thread's seat at a door of fewer seats is its last seat counted
    /// round them, once: the thread keeps the seat it then takes.
    #[test]
    fn a_seat_past_a_doors_seats_is_counted_round_them() {
        let store = fleet(2);
        let door = Door::new(Arc::clone(&store));
        seated(5);
        let (lease, _) = leased(&door, b"r-1", 7);
        assert_eq!(store.file_of(lease), 1);
        assert_eq!(SEAT.get(), 1, "counted once and kept");
    }

    #[test]
    fn a_seat_at_the_doors_count_wraps_to_zero_and_keeps_it() {
        let store = fleet(2);
        let door = Door::new(Arc::clone(&store));
        seated(2);
        let (lease, _) = leased(&door, b"at-the-count", 7);
        assert_eq!(store.file_of(lease), 0);
        assert_eq!(SEAT.get(), 0);
    }

    /// A picker that holds a decision's scope, as a plugin's stage does:
    /// its profile keeps state a decider.
    #[derive(Debug)]
    struct Scoped;

    impl Lifecycle for Scoped {}

    impl DecisionScope for Scoped {
        fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}

        fn close(&self) {}
    }

    impl Picker for Scoped {
        fn name(&self) -> &'static str {
            "scoped"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            Random.pick(facts, candidates, kept, scores, band, out);
        }

        fn scope(&self) -> Option<&dyn DecisionScope> {
            Some(self)
        }
    }

    #[test]
    fn sampling_keeps_a_plain_generations_callers_at_their_seats() {
        let files = (2..=8)
            .find(|&files| decider_of(MODEL, files) != 0)
            .expect("the model names a later file");
        let store = fleet(files);
        let door = Door::new(Arc::clone(&store));
        seated(0);
        picking(&store, 2, "scoped", || Box::new(Scoped));
        let (scoped, _) = leased(&door, b"scoped", 7);
        assert_eq!(store.file_of(scoped), decider_of(MODEL, files));
        sampling(&store, 3, 1);
        for id in [b"one", b"two"] {
            let (lease, _) = leased(&door, id, 8);
            assert_eq!(store.file_of(lease), 0);
            assert!(door.explain(GATEWAY, id).is_some());
        }
        assert_eq!(SEAT.get(), 0);
    }

    /// The seat follows who owns what a decision reads and what hears its
    /// lease. Under a generation that holds a stage keeping state a
    /// decider a request decides at its model's seat, wherever its
    /// caller's thread sits, from the first decision under that
    /// generation on. Under a generation with no such stage the thread
    /// decides at its own seat again.
    #[test]
    fn a_generation_that_keeps_state_a_decider_seats_each_request_at_its_model() {
        // A count of files under which the model's seat is not the first,
        // where the calling thread sits.
        let files = (2..=8).find(|&files| decider_of(MODEL, files) != 0);
        let files = files.expect("the model names a later file under some count");
        let home = decider_of(MODEL, files);
        let store = fleet(files);
        let door = Door::new(Arc::clone(&store));
        seated(0);
        let (own, _) = leased(&door, b"r-1", 7);
        assert_eq!(store.file_of(own), 0, "no stage keeps state a decider");
        picking(&store, 2, "scoped", || Box::new(Scoped));
        // The first decision under the generation finds the stage where
        // it stands: the caller leaves its own seat with nothing decided
        // and sits at the model's. The next reads the door's word.
        let kept = [b"r-2", b"r-3"].map(|id| leased(&door, id, 8).0);
        assert_eq!(kept.map(|lease| store.file_of(lease)), [home, home]);
        assert_eq!(SEAT.get(), 0, "the thread's seat is still its own");
        // The seat it left is at rest: what is said of the lease on file
        // there is applied at once.
        assert_eq!(report(&door, own, gone()), 2, "applied at once");
        let plain = store.apply(preset("random", 3));
        plain.expect("a later generation");
        for id in [b"r-4", b"r-5"] {
            let (again, _) = leased(&door, id, 9);
            assert_eq!(store.file_of(again), 0, "the thread's seat again");
        }
        assert_eq!(door.faults(), 0);
    }

    /// A picker that holds a decision's scope and panics once armed: where
    /// it picks, or else where it is asked what it needs.
    #[derive(Debug)]
    struct Armed {
        armed: Arc<AtomicBool>,
        at_pick: bool,
    }

    impl Armed {
        fn called(&self, at_pick: bool) {
            let fires = self.at_pick == at_pick && self.armed.load(Ordering::Relaxed);
            assert!(!fires, "the case's own panic");
        }
    }

    impl Lifecycle for Armed {}

    impl DecisionScope for Armed {
        fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}

        fn close(&self) {}
    }

    impl Picker for Armed {
        fn name(&self) -> &'static str {
            "armed"
        }

        fn needs(&self) -> Needs {
            self.called(false);
            Needs::default()
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            self.called(true);
            Random.pick(facts, candidates, kept, scores, band, out);
        }

        fn scope(&self) -> Option<&dyn DecisionScope> {
            Some(self)
        }
    }

    /// A stage is asked what it needs once, where its generation is built:
    /// the engine keeps the answer and a decision calls no stage for it.
    /// So a stage whose answer would panic later costs no request.
    #[test]
    fn a_stage_is_asked_what_it_needs_at_the_build_alone() {
        let store = fleet(2);
        let door = Door::new(Arc::clone(&store));
        let armed = Arc::new(AtomicBool::new(false));
        let stage = Arc::clone(&armed);
        picking(&store, 2, "armed", move || {
            Box::new(Armed {
                armed: Arc::clone(&stage),
                at_pick: false,
            })
        });
        armed.store(true, Ordering::Relaxed);
        for id in [b"r-1", b"r-2"] {
            let decided = door.decide(GATEWAY, &request(id, 7), Instant::now());
            assert!(decided.is_ok(), "{:?}", decided.err());
        }
        assert_eq!(door.faults(), 0);
    }

    /// So with the encode profile of a pair: a request with encode items
    /// reads the answer kept at the build.
    #[test]
    fn an_encode_profiles_stage_is_asked_what_it_needs_at_the_build_alone(
    ) -> Result<(), Box<dyn Error>> {
        let (door, _) = push_fleet()?;
        let armed = Arc::new(AtomicBool::new(false));
        let stage = Arc::clone(&armed);
        let mut catalog = Catalog::builtin();
        catalog.picker("armed", move || {
            Box::new(Armed {
                armed: Arc::clone(&stage),
                at_pick: false,
            })
        })?;
        let mut document = encode_document()?;
        document.generation = 2;
        let encode = document.profiles.get_mut("encode");
        encode.ok_or("missing encode profile")?.pick = "armed".into();
        let digest = document.digest()?;
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])?;
        door.store().apply(spec)?;
        armed.store(true, Ordering::Relaxed);
        let mut placed = EncodePlaced::new();
        for id in [b"r-1", b"r-2"] {
            let asked = pair_request(id);
            let decided =
                door.decide_encode_pair(GATEWAY, &asked, items(), &mut placed, Instant::now());
            assert!(decided.is_ok(), "{:?}", decided.err());
        }
        assert_eq!(door.faults(), 0);
        Ok(())
    }

    /// A request that states a hard demand is decided over the rows the
    /// demand leaves, and each of them reads its own signal value at its
    /// new place: the cells follow the rows through that narrowing as the
    /// affinity column does.
    #[test]
    fn rows_a_demand_leaves_read_their_own_signal_values() {
        use crate::store::{SignalSink, Taken};

        /// What the stage read: each row's worker with its value.
        type Read = Vec<(WorkerId, Option<f64>)>;

        /// Reads one signal of every row it scores and keeps each row's
        /// worker with what it read.
        #[derive(Debug)]
        struct ReadsEach {
            held: crate::SignalColumn,
            read: Arc<Mutex<Read>>,
        }

        impl Lifecycle for ReadsEach {}

        impl crate::Scorer for ReadsEach {
            fn name(&self) -> &'static str {
                "reads-each"
            }

            fn score(
                &self,
                facts: &RequestFacts<'_>,
                candidates: &[Candidate],
                kept: &[u32],
                out: &mut [f32],
            ) -> TieBand {
                let rows = kept.iter().map(|&row| &candidates[row as usize]);
                let read = rows.map(|row| (row.id, self.held.read(facts, row)));
                self.read.lock().extend(read);
                out.fill(0.0);
                TieBand::NORMALIZED
            }

            fn needs(&self) -> Needs {
                Needs {
                    signal_cells: true,
                    ..Needs::default()
                }
            }
        }

        let (store, ids) = stating(&[9, 7, 9, 7]);
        let read = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&read);
        let mut catalog = Catalog::builtin();
        let added = catalog.scorer_with("reads-each", move |build| {
            Ok(Box::new(ReadsEach {
                held: build.signal_column("held")?,
                read: Arc::clone(&kept),
            }))
        });
        added.expect("a free name");
        let document: PolicySpec = serde_json::from_str(
            r#"{"apiVersion":"smg.policy/v1","generation":2,
                "signals":{"held":{"unit":"tokens"}},
                "profiles":{"p":{"score":["reads-each"],"pick":"best"}}}"#,
        )
        .expect("a well-formed document");
        let digest = document.digest().expect("the document digests");
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[]);
        let applied = store.apply(spec.expect("the catalog holds the stage"));
        applied.expect("a later generation");
        let now = Instant::now();
        for (id, value) in ids.iter().zip([10.0, 11.0, 12.0, 13.0]) {
            let held = store.signal_column("held");
            let taken = store.signal_value(*id, 0, &held, "tokens", value, now);
            assert_eq!(taken, Taken::Written);
        }
        store.publish_now();
        let door = Door::new(Arc::clone(&store));
        let mut asked = request(b"stays", 7);
        asked.constraints.demands = Demands { runtime: 7 };
        let decided = door.decide(GATEWAY, &asked, now).expect("a decision");
        assert!(decided.decision.pick().is_some(), "{decided:?}");
        // The demand left the second and the fourth member: each read the
        // value written for it, and no other row was read.
        let mut read = read.lock().clone();
        read.sort_by_key(|&(id, _)| id);
        let mut left = [(ids[1], Some(11.0)), (ids[3], Some(13.0))];
        left.sort_by_key(|&(id, _)| id);
        assert_eq!(read, left);
    }

    /// A stage of the encode profile alone reads a signal: the door asks
    /// the view for the cells on that profile's need, and the stage reads
    /// its encoder's value.
    #[test]
    fn an_encode_profiles_stage_reads_its_encoders_signal_value() -> Result<(), Box<dyn Error>> {
        use crate::{
            spec::{AcrossRanks, SignalSpec},
            store::{SignalSink, Taken},
        };

        let (door, encoder) = push_fleet()?;
        let read = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&read);
        let mut catalog = Catalog::builtin();
        catalog.scorer_with("reads-held", move |build| {
            Ok(Box::new(ReadsHeld {
                held: build.signal_column("held")?,
                read: Arc::clone(&kept),
            }))
        })?;
        let mut document = encode_document()?;
        document.generation = 2;
        let held = SignalSpec {
            unit: "tokens".into(),
            max_age_ms: None,
            across_ranks: AcrossRanks::default(),
        };
        document.signals.insert("held".into(), held);
        let encode = document.profiles.get_mut("encode");
        let encode = encode.ok_or("missing encode profile")?;
        encode.score.push("reads-held".into());
        let digest = document.digest()?;
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])?;
        door.store().apply(spec)?;
        let now = Instant::now();
        let held = door.store().signal_column("held");
        let taken = door
            .store()
            .signal_value(encoder, 0, &held, "tokens", 7.0, now);
        assert_eq!(taken, Taken::Written);
        door.store().publish_now();
        let mut placed = EncodePlaced::new();
        let asked = pair_request(b"r-1");
        let decided = door.decide_encode_pair(GATEWAY, &asked, items(), &mut placed, now);
        assert!(decided.is_ok(), "{:?}", decided.err());
        let read = read.lock();
        assert!(!read.is_empty(), "the encode profile's stage was asked");
        assert!(read.iter().all(|value| *value == Some(7.0)), "{read:?}");
        Ok(())
    }

    /// A decision that panics after its caller left its own seat for the
    /// model's costs its request alone, and the seat it left is at rest.
    #[test]
    fn a_stage_that_panics_at_the_models_seat_costs_its_request_alone() {
        // The model's seat is not the first, where the calling thread
        // sits.
        let files = (2..=8).find(|&files| decider_of(MODEL, files) != 0);
        let files = files.expect("the model names a later file under some count");
        let store = fleet(files);
        let door = Door::new(Arc::clone(&store));
        seated(0);
        let (own, _) = leased(&door, b"r-0", 7);
        let armed = Arc::new(AtomicBool::new(false));
        let stage = Arc::clone(&armed);
        picking(&store, 2, "armed", move || {
            Box::new(Armed {
                armed: Arc::clone(&stage),
                at_pick: true,
            })
        });
        armed.store(true, Ordering::Relaxed);
        // The first request under the generation leaves the caller's own
        // seat for the model's. The next reads the door's word.
        for id in [b"r-1", b"r-2"] {
            let refused = door.decide(GATEWAY, &request(id, 7), Instant::now());
            assert_eq!(refused.err(), Some(Refused::Faulted));
        }
        assert_eq!(door.faults(), 2);
        // What is said of the lease on file at the caller's own seat is
        // applied at once: no thread is left at work there.
        assert_eq!(report(&door, own, gone()), 0, "applied at once");
    }
}
