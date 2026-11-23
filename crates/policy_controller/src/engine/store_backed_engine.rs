//! The engine over the policy store: the order of one decision's calls,
//! written once.
//!
//! A decision is one load of the published rows
//! ([`MemoryStore::deciding`]), the engine's decision over them, the
//! lease filed at the issue and the engine's hearing of it, in that
//! order. Every host runs this one order, behind a transport of its own.
//! The served controller runs it on its deciders' threads, behind the
//! fence and the budget that a wire and an election bring. A host that
//! embeds the engine runs it on the caller's thread ([`StoreBackedEngine`]): no
//! frame is built and no thread is hopped. A single's issued lease records the
//! prefix for its first target when the naming read a record of sent prefixes
//! and the store has heard no cache event of the target.
//!
//! A request reaches the order in the form it was asked in (the trait
//! `RequestForm`): a frame as the served controller checked it, or the facts a
//! host in process holds ([`Request`]).
//!
//! A host with encode items can place them through
//! [`StoreBackedEngine::decide_encode_pair`]. The pair's prefill and decode are
//! the sides that hold leases.

use std::{
    cell::Cell,
    mem,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use parking_lot::{Mutex, MutexGuard};

pub(crate) use crate::store::snapshot::narrow;
use crate::{
    document::spec::AppliedSpec,
    engine::{
        decision::{
            shed, Decision, Pair, Pick, ShadowKind, ShedReason, Target, Targets, Trace,
            SUBSET_HINT, WIRES,
        },
        decision_engine::{
            DecideOutcome, Decided, DecisionEngine, DecisionRefused, DecodeRefused, HeldDecision,
            Leased, Request, RequestMaker, SweepOutcome,
        },
        decision_slot::DecisionSlot,
        demands::{self, DemandFacts, Demands},
        framework::{release, EncodeItems, EncodePlaced, Engine, Plan, Scratch, KEEPS_SLOT_STATE},
        prompt_hashes::{HashSeeds, PromptHashes},
        sticky_sessions::{SessionExpiry, SessionVerdict},
        view::{
            fingerprint, AffinityFacts, AffinityRows, Candidate, Constraints, HashKind, KeyedRow,
            Phase, PinSource, PoolFacts, RequestFacts, Role, Session, WireModes, WorkerId,
        },
    },
    stages::lifecycle::Lifecycle,
    store::{
        AffinityQuery, AffinityRowsBuffers, AtWork, DecodeIssue, DecodeSide, EncodePairBuffers,
        GatewayId, Hand, HealthSink, LeaseRecord, LeaseRefused, LeaseState, LeasesStore,
        MemoryStore, PairBuffers, PickRecord, Pin, SessionsStore, Sides, StoreSnapshot,
        ViewBuffers, ViewFacts, ViewQuery,
    },
    tracking::{
        in_flight::WorkerSlot,
        lease::{Dispatch, Lease, LeaseId, LeaseKind, Outcome, Said, Side, DISPATCH_TTL},
    },
};

/// A request as the store-backed engine reads it: what the rows' narrowing,
/// the engine's request and the lease take of it, in whatever form it was
/// asked.
pub(crate) trait RequestForm {
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
    /// Whether the request asks the decode side of a pair's lease: a
    /// request of the served wire alone, read at the decode phase. It
    /// takes a side's path whatever its model is bound to ([`entry`]).
    fn asks_decode(&self) -> bool {
        false
    }
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
        name: impl FnOnce(&RequestFacts<'_>, &Constraints<'_>) -> Option<AffinityRows<'n>>,
        decide: impl FnOnce(&RequestFacts<'_>, &Constraints<'_>) -> R,
    ) -> R;
}

/// The naming of a decision that names no row ([`RequestForm::read`]).
pub(crate) fn unnamed(_: &RequestFacts<'_>, _: &Constraints<'_>) -> Option<AffinityRows<'static>> {
    None
}

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
}

/// One of the parts' counts, summed over `parts`.
pub(crate) fn summed(parts: &[Part], count: impl Fn(&Part) -> &AtomicU64) -> u64 {
    let parts = parts.iter();
    parts.map(|part| count(part).load(Ordering::Relaxed)).sum()
}

/// What a deciding thread keeps across decisions: the engine's run of a
/// decision allocates only while these grow. The lists are those of the
/// form its requests are asked in ([`RequestForm::Lists`]).
#[derive(Default)]
pub(crate) struct Buffers<L> {
    pub(crate) view: ViewBuffers,
    /// The rows of a lent view's mask a hold of the asking gateway stands
    /// on ([`StoreSnapshot::held`]): asked where a veto masks the view.
    pub(crate) held: Vec<u32>,
    /// What each row of a built view states for a request's hard demands
    /// ([`StoreSnapshot::demand_facts`]): read for a request that states one.
    pub(crate) demanded: Vec<DemandFacts>,
    /// Both sides' rows of a pair decision and the table of their classes.
    pub(crate) pair: PairBuffers,
    /// The rows of an encode-pair decision, made at the thread's first
    /// one: a thread that decides none keeps none.
    pub(crate) encode: Option<Box<EncodePairBuffers>>,
    pub(crate) scratch: Scratch,
    pub(crate) lists: L,
    pub(crate) named: AffinityRowsBuffers,
    /// What a keyed decision reads of its key in the store, where its
    /// rule has an in-flight cap.
    pub(crate) keyed: Keyed,
}

/// A request as its decision slot takes it.
#[derive(Clone, Copy)]
pub(crate) struct DecisionCall<'a, A> {
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

impl<'a, A> DecisionCall<'a, A> {
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
    /// and one side's entry of the engine never makes one. The one
    /// decision a side has is the decode of a lease its request names
    /// ([`RequestForm::asks_decode`]).
    Side,
}

/// The path `taken`'s request takes in the world of one load
/// ([`pair_side`]). A single request is no plan's side: its binding is
/// not looked up here, and the engine looks it up once.
pub(crate) fn entry<A: RequestForm>(world: &StoreSnapshot, taken: &DecisionCall<'_, A>) -> Entry {
    entry_and_plan(world, taken).0
}

#[inline]
fn entry_and_plan<A: RequestForm>(
    world: &StoreSnapshot,
    taken: &DecisionCall<'_, A>,
) -> (Entry, Plan) {
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
        // A decode call whose model a reload bound to no pair plan is a
        // side's request still: its lease's own test answers it.
        _ if taken.request.asks_decode() => (Entry::Side, Plan::Single),
        _ => (Entry::Single, Plan::Single),
    }
}

/// The engine's decision over one view of [`hand`]'s, with what a pick's
/// lease takes of its request. Over rows the store named it is made at the
/// entry that states what the pick's first target holds of them
/// ([`Engine::decide_stated`]), and over any other view at the plain entry.
struct ByEngine<'a>(&'a Engine);

impl ByEngine<'_> {
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
        let decision = if STATED {
            engine.decide_stated(facts, pool, candidates, constraints, scratch)
        } else {
            engine.decide(facts, pool, candidates, constraints, scratch)
        };
        // What a pick's lease takes of its request, read where the
        // request stands as the engine read it.
        let booking = decision.pick().and_then(|pick| {
            Some(Booking {
                target: pick.targets.first()?.id,
                picked: PickRecord::of(&pick.targets),
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

impl Decides<(Decision, Option<Booking>)> for ByEngine<'_> {
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
pub(crate) fn decide_in<A: RequestForm>(
    world: &StoreSnapshot,
    taken: &DecisionCall<'_, A>,
    buffers: &mut Buffers<A::Lists>,
    own: &Part,
) -> (Decision, Option<Booking>) {
    let spec = world.spec();
    let engine = spec.engine();
    if taken.request.hashes() > taken.request.hashes_read(spec) {
        own.hashes_cut.fetch_add(1, Ordering::Relaxed);
    }
    let plain = ByEngine(engine);
    let ((decision, booking), built) = hand(world, taken, buffers, &own.undialed, plain);
    (
        decision,
        booking.map(|booking| Booking { built, ..booking }),
    )
}

/// The engine's pair decision over the world of one load, for the request
/// that enters a pair plan at its prefill. Both sides' rows are written
/// for it of one publication ([`StoreSnapshot::build_pair`]), with what the
/// leases on file book on each row's worker, less what the gateway states of
/// itself takes from each side ([`narrow`]); the engine's pair entry decides
/// over them once, and the request's constraints are read by both sides. A side
/// with rows that the gateway's own word leaves none is shed under that
/// constraint's name, as a single's pool is. The store's narrowing of a pair
/// keeps no hard demand yet: a pair's request that states one is shed under the
/// wires' name, never placed past it. A pick's booking holds both sides, each
/// worker's count beside its own pool's row.
///
/// A pair whose profile on either side holds a stage that may refuse on a
/// count it is shown is decided behind an apply of what waits at the files
/// of leases, as a single is ([`hand`]), where its decider lent its place.
pub(crate) fn pair_in<A: RequestForm>(
    world: &StoreSnapshot,
    taken: &DecisionCall<'_, A>,
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
pub(crate) fn encode_pair_in<A: RequestForm>(
    world: &StoreSnapshot,
    taken: &DecisionCall<'_, A>,
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
fn pair_with<const DECODE: bool, A: RequestForm, R, B>(
    world: &StoreSnapshot,
    taken: &DecisionCall<'_, A>,
    own: &Part,
    rows: &mut R,
    (lists, scratch): (&mut A::Lists, &mut Scratch),
    build: impl FnOnce(
        &mut R,
        &Engine,
        &str,
        ViewQuery,
        Option<WireModes>,
        Option<u32>,
    ) -> Option<((ViewFacts, ViewFacts), (u64, bool))>,
    (decide, book): (
        impl FnOnce(
            &Engine,
            &R,
            &RequestFacts<'_>,
            &PoolFacts<'_>,
            &Constraints<'_>,
            &mut Scratch,
        ) -> Decision,
        impl FnOnce(&RequestFacts<'_>, &Decision, Option<(ViewFacts, ViewFacts)>, &Scratch) -> Option<B>,
    ),
) -> (Decision, Option<B>) {
    let DecisionCall {
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
    let ask = ViewQuery {
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
    world: &StoreSnapshot,
    facts: &RequestFacts<'_>,
    decision: &Decision,
    built: Option<(ViewFacts, ViewFacts)>,
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
        picked: PickRecord::of_decodes(&pick.targets),
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
            picked: PickRecord::of(&pick.targets),
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
    pub(crate) picked: PickRecord,
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
    pub(crate) built: Option<ViewFacts>,
}

/// What a pair's two leases take of its decision's hand-over: the prefill
/// side's booking, which is a single's, and the decode side with the slot
/// of its worker's count as the decision's build of the decode pool names
/// it.
#[derive(Clone, Copy)]
pub(crate) struct PairBooking {
    pub(crate) prefill: Booking,
    pub(crate) decode: DecodeSide,
    pub(crate) slot: Option<WorkerSlot>,
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
/// one ([`StoreSnapshot::sent`]).
///
/// The store builds its issue for each telling, so a caller with a
/// telling of its own leaves every other caller's issue as it was built.
///
/// # Errors
///
/// The store's refusal: it stands under another term, or mints no id
/// under this one.
pub(crate) fn issue<A: RequestForm>(
    store: &MemoryStore,
    world: &StoreSnapshot,
    taken: &DecisionCall<'_, A>,
    term: u64,
    booking: Booking,
    named: &AffinityRowsBuffers,
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
pub(crate) fn issue_pair<A: RequestForm>(
    store: &MemoryStore,
    world: &StoreSnapshot,
    taken: &DecisionCall<'_, A>,
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

/// What a decode call's pick files: the decode side, beside the slot of
/// the count it books.
pub(crate) type DecodeBooking = (DecodeSide, Option<WorkerSlot>);

/// The decode side of the pair's lease `lease` decided over one world, for
/// both doors. The store's own test of the lease comes first, so a call
/// it refuses runs no stage. Then the pair's rows are built and narrowed
/// as [`pair_in`] builds them, and the engine's decode entry runs over the
/// partners of the held prefill, handed the decode the lease's first
/// decision chose: a decode profile that keeps its pick answers that decode
/// again while its filters keep it
/// ([`Engine::decide_decode`](crate::engine::framework::Engine::decide_decode)).
/// The decode profile is shown the decode pool's facts: its size, its list
/// and whether every row of it holds a report. A pick comes with what its
/// issue files.
///
/// # Errors
///
/// [`DecodeRefused`]: why the lease takes no decode side. Nothing is built.
pub(crate) fn decode_in<A: RequestForm>(
    world: &StoreSnapshot,
    store: &MemoryStore,
    taken: &DecisionCall<'_, A>,
    lease: LeaseId,
    buffers: &mut Buffers<A::Lists>,
    own: &Part,
) -> Result<(Decision, Option<DecodeBooking>), DecodeRefused> {
    let (prefill, first_decode) = store.held_pair(taken.gateway, lease)?;
    let Buffers {
        pair,
        scratch,
        lists,
        ..
    } = buffers;
    Ok(pair_with::<true, _, _, _>(
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
                let facts = RequestFacts {
                    phase: Phase::Decode,
                    ..*facts
                };
                let pools = pair.pools();
                engine.decide_decode(&facts, pool, pools, prefill, first_decode, cx, scratch)
            },
            |facts, decision, built, _| {
                let (pick, (_, decodes)) = decision.pick().zip(built)?;
                let first = pick.targets.first()?.pair?;
                let side = DecodeSide {
                    target: first.decode_id,
                    picked: PickRecord::of_decodes(&pick.targets),
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
    ))
}

/// Files a decode call's pick under the lease's id and tells the engine,
/// under one hold of the file. The side is on file when the engine hears:
/// a stage that panics there is held and counted, as under a report, and
/// the caller still gets the pick the file holds. The side's lease comes
/// with its issue's count, for a door that may undo it.
///
/// # Errors
///
/// [`DecodeRefused`]: the lease moved since the call's first test.
pub(crate) fn issue_decode<A: RequestForm>(
    store: &MemoryStore,
    world: &StoreSnapshot,
    taken: &DecisionCall<'_, A>,
    lease: LeaseId,
    decision: &Decision,
    booking: DecodeBooking,
) -> Result<(Lease, DecodeIssue), DecodeRefused> {
    let spec = world.spec();
    store.issue_decided_decode(taken.gateway, lease, booking, taken.now, |lease| {
        // The lease keeps the generation of its first decision, and the
        // engine of that generation alone hears its words ([`issuing`]).
        // Behind a reload the engine of this instant decided the side and
        // hears nothing of the lease.
        let Some(engine) = issuing(spec, lease) else {
            return;
        };
        // The issue is dated at this call ([`issued_later`]).
        let heard = || issued_later(world, engine, lease, decision, taken.now);
        match catch_unwind(AssertUnwindSafe(heard)) {
            Ok(()) => store.call_returned(),
            Err(payload) => {
                release(payload);
                store.call_faulted();
            }
        }
    })
}

/// The record of the lease `booking` asks on `side`, as its door hands it
/// to the store: the store mints its id, and writes the mark of a pair's
/// lease where it files one.
#[inline(always)]
fn record_of<A: RequestForm>(
    taken: &DecisionCall<'_, A>,
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
fn booked_in(world: &StoreSnapshot, booking: &Booking) -> Option<WorkerSlot> {
    let built = booking.built?;
    world.booked(&built, booking.target)
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
    Option<&'a crate::load::SignalCells>,
    Option<crate::tracking::PoolInFlightCounts<'a>>,
    Option<ViewFacts>,
);

/// Hands `decide` what the engine reads for one request: the request with
/// what is left of its budget, the pool's facts under the generation its
/// rows were published under, and those rows as the asking gateway reaches
/// them at the decision's instant, with the affinity column where the
/// request's profile declares it. A request that states no wires, no hint
/// of 0 and no hard demand reads its pool by the store's one door
/// ([`StoreSnapshot::view`]): the rows where their publication left them, with
/// its gateway's standing vetoes and the leases' counts beside them, or rows
/// the door writes where the request's profile shows them to a plugin. One that
/// states wires, a hint of 0 or a demand is handed rows written for it
/// ([`StoreSnapshot::build`]), what the leases on file book on each row's
/// worker written over them ([`StoreSnapshot::count_in_flight`]), less what the
/// gateway states of itself and what the request's demands take ([`narrow`],
/// [`narrow_demanded`]). The rows its wires take are counted in `undialed`, the
/// decider's own part, and the list the narrowing leaves takes a name of its
/// own ([`cut_name`]). The request's block hashes are cut at what that
/// generation reads of them. `decide`'s answer comes back with the build its
/// rows were read by, `None` where no member serves the model.
///
/// A request whose profile reads the affinity column by its own stages is
/// handed no column: the rows the store names for it stand in the column's
/// place ([`StoreSnapshot::fill_affinity_rows`]), at most its sample of each
/// list, and every holder and every warming row where it takes none
/// ([`Engine::affinity_rows_cap`](crate::engine::framework::Engine::affinity_rows_cap)).
/// Where a demand took a row, the rows that stay are named within themselves
/// ([`StoreSnapshot::fill_affinity_rows_narrowed`]). The naming is the store's
/// one reader of its index: the column it builds holds a row's age and no
/// holder. Over the rows it named the decision is asked in the decider's form
/// for them ([`Decides::over_named`]).
///
/// A pool that has rows of which the request's own narrowing left none is
/// answered in the decider's form for it ([`Decides::emptied`]), under
/// the stage that took the last row: the request's hint of 0, the last
/// demand its rows failed first
/// ([`UnmetDemand`](crate::engine::demands::UnmetDemand)), or its gateway's
/// wires.
///
/// A request whose profile holds a stage that may refuse by a report of
/// leases ([`Filter::refuses_by_reports`](crate::engine::framework::Filter::refuses_by_reports))
/// is handed its rows behind an apply of what waits at the files of leases
/// ([`AtWork::apply_waiting`]), where its decider lent its place.
pub(crate) fn hand<A: RequestForm, R>(
    world: &StoreSnapshot,
    taken: &DecisionCall<'_, A>,
    buffers: &mut Buffers<A::Lists>,
    undialed: &AtomicU64,
    decide: impl Decides<R>,
) -> (R, Option<ViewFacts>) {
    let DecisionCall {
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
    // pays the second walk. They join a shadow profile's too
    // ([`Engine::reads_for`](crate::engine::framework::Engine::reads_for)).
    let column = needs.affinity;
    let cap = if column {
        engine.affinity_rows_cap(model, phase)
    } else {
        0
    };
    let names = cap > 0;
    let ask = ViewQuery {
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
    let wires = request.wires();
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
    let by_door = wires.is_none() && request.subset() != Some(0) && request.demands().is_none();
    // The last demand the rows a demand took failed first, where one took
    // a row of the build: the rows left are named within themselves then.
    let mut unmet: Option<demands::UnmetDemand> = None;
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
                // The request's demands, read where a row is written: the
                // path that writes none carries none of them past its test.
                // A prefill's rows are held to the input alone.
                let demands = request.demands().held_by(role);
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
    // The store's sessions, where the decision's rule has an in-flight
    // cap: a keyed request reads its key's count there. Under any other
    // rule the engine is asked at once, past the key's read.
    let sessions = pins.filter(|_| needs.session_in_flight);
    if names {
        // A row is named by its place among the rows the decision is made
        // over: the lent rows, or the built ones after their narrowing.
        let hashes = request.block_hashes();
        let (namespace_seed, hash_kind) = request.cache_namespace();
        let naming = AffinityQuery {
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
            if candidates.is_empty() && pool.pool_size > 0 {
                let subset = request.subset();
                return emptied(decide, facts, &pool, constraints, scratch, subset, unmet);
            }
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
        if candidates.is_empty() && pool.pool_size > 0 {
            let subset = request.subset();
            return emptied(decide, facts, &pool, constraints, scratch, subset, unmet);
        }
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
/// own, out of line, asked inside the request's read where the decider is
/// asked: a pool that keeps a row pays one test for this answer, the test
/// its decider had, and the door's routine keeps the frame it had.
#[cold]
#[inline(never)]
fn emptied<R>(
    decide: impl Decides<R>,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    constraints: &Constraints<'_>,
    scratch: &mut Scratch,
    subset: Option<u32>,
    unmet: Option<demands::UnmetDemand>,
) -> R {
    let stage = if subset == Some(0) {
        SUBSET_HINT
    } else {
        unmet.map_or(WIRES, demands::UnmetDemand::stage)
    };
    decide.emptied(facts, pool, constraints, scratch, stage)
}

/// The role whose pool decides a request of `phase`: the pool [`hand`]
/// reads its rows of.
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
    /// host answers for them
    /// ([`UnmetDemand`](crate::engine::demands::UnmetDemand)): the stage a pool
    /// the demands left nobody is shed under. `None` where no demand took a
    /// row.
    pub(crate) unmet: Option<demands::UnmetDemand>,
}

/// [`narrow`] for a request that states hard demands ([`Demands`]): one
/// pass keeps the rows that share a wire with the gateway and meet every
/// demand, each by its own vision bit and its entry of `facts`, the
/// build's column ([`StoreSnapshot::demand_facts`]). A row with no entry states
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
        crate::store::snapshot::KeptCells::of(&mut view.signal_cells, view.candidates.len());
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
    pub(crate) in_flight: Option<crate::tracking::PoolInFlightCounts<'a>>,
    /// How many of the request's block hashes the deciding generation
    /// reads.
    pub(crate) max_block_hashes: usize,
    /// The pins of record the host lends the decision ([`PinSource`]);
    /// `None` where it keeps none.
    pub(crate) pins: Option<&'a dyn PinSource>,
    /// The candidates' signal cells, as the view handed them: `None` for
    /// a profile that reads no signal, and for a pair, whose cells ride
    /// its pools.
    pub(crate) signal_cells: Option<&'a crate::load::SignalCells>,
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
/// that rank ([`StoreSnapshot::lists_rank`]), so a late report enters no rank
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
    world: &StoreSnapshot,
    found: &Sides,
    said: Said,
    by: Option<DecisionSlot>,
) -> bool {
    // The lease's own decider acts while its event is told, whichever
    // thread applies the report.
    let _acting = by.map(DecisionSlot::enter);
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
pub(crate) fn issuing<'a>(spec: &'a AppliedSpec, lease: &Lease) -> Option<&'a Engine> {
    (spec.generation() == lease.generation).then(|| spec.engine())
}

/// What [`told`] tells, under its catch.
fn tell(store: &MemoryStore, world: &StoreSnapshot, found: &Sides, said: Said) {
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
fn issued_later(
    world: &StoreSnapshot,
    engine: &Engine,
    lease: &Lease,
    decision: &Decision,
    at: Instant,
) {
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
    world: &StoreSnapshot,
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
        Some((_, SessionExpiry::At(forgotten))) => {
            SessionsStore::pin(store, key, worker, at, Some(forgotten));
        }
        Some((_, SessionExpiry::Never)) => SessionsStore::pin(store, key, worker, at, None),
        // The lease's own rule is the word of a decision that may be older
        // than the pin's last writer: it never moves the end earlier.
        Some((_, SessionExpiry::Unread)) | None => {
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

impl RequestForm for Request<'_> {
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
        name: impl FnOnce(&RequestFacts<'_>, &Constraints<'_>) -> Option<AffinityRows<'n>>,
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

/// A request as the door's order takes it up under the world it loaded:
/// one its host stated ahead of the load ([`Request`]), or one its host
/// makes under that world's document ([`RequestMaker`]).
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
    fn under<R>(&self, world: &StoreSnapshot, order: impl FnOnce(&Request<'_>) -> R) -> R;
}

impl Stated for Request<'_> {
    fn model(&self) -> &str {
        self.facts.model
    }

    // Forced into the order's one call: a request stated ahead of the load
    // pays nothing for the form a host's maker takes.
    #[inline(always)]
    fn under<R>(&self, _: &StoreSnapshot, order: impl FnOnce(&Request<'_>) -> R) -> R {
        order(self)
    }
}

/// A decode call's request with the seat of its lease's file: one its host
/// stated, or one its host makes. The call decides at that seat, behind the
/// decision that stands there: a caller that takes up a seat's work applies
/// what was handed to its file, so a prefill's dispatch that waits there is
/// applied ahead of the call's read. From another seat the call would read
/// a file that applies nothing while its own caller decides, and a sent
/// prefill would read as not sent.
struct Called<'s, S>(&'s S, usize);

impl<S: Stated> Stated for Called<'_, S> {
    fn model(&self) -> &str {
        self.0.model()
    }

    fn own_seat(&self) -> Option<usize> {
        Some(self.1)
    }

    #[inline(always)]
    fn under<R>(&self, world: &StoreSnapshot, order: impl FnOnce(&Request<'_>) -> R) -> R {
        self.0.under(world, order)
    }
}

/// A [`RequestMaker`]'s request, as the door's order takes it up.
struct Made<'m, M>(&'m M);

impl<M: RequestMaker> Stated for Made<'_, M> {
    fn model(&self) -> &str {
        self.0.model()
    }

    fn under<R>(&self, world: &StoreSnapshot, order: impl FnOnce(&Request<'_>) -> R) -> R {
        let request = self.0.make(world.spec());
        debug_assert_eq!(request.facts.model, self.0.model(), "the seat's model");
        order(&request)
    }
}

/// A host's engine for a decision in process: the order of a decision's
/// calls, run on the caller's thread. No frame is built and no thread is
/// hopped.
///
/// **Whose file.** The store holds a file of leases a decider
/// ([`MemoryStore::with_decision_slots`]), and a host in process has no
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
/// ([`Engine::keeps_slot_state`]). Every profile's stages hear every lease,
/// so the stage's state has one writer a model and hears a model's
/// leases whole, and a model's requests decide one at a time. The files
/// a host lays past those its models name gain such a generation
/// nothing.
///
/// At its seat a caller acts as that seat's decider ([`DecisionSlot::enter`])
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
/// through [`StoreBackedEngine::report`], on whichever thread it stands, and
/// that thread is never held for a decision. The word goes into the ring of the
/// file the lease's id names. Where no caller decides at that file it is
/// applied at once. Where one does, that caller applies it: every decision here
/// applies what the ring holds as it takes up its work and as it leaves it. A
/// full ring takes no more until then: the door keeps the word beside it, in
/// the order said, and the caller hands it on as it leaves. Whoever applies a
/// word tells the engine that issued the lease.
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
/// [`StoreBackedEngine::sweep`], which ends the leases past their dispatch span
/// that no issue reached, and lets the pins of record past their end go.
///
/// A store has one host at its files: a door, or the served controller's
/// deciders, never both.
pub struct StoreBackedEngine {
    store: Arc<MemoryStore>,
    /// A seat a file of the store: its holder decides as that file's
    /// decider.
    seats: Box<[SlotState]>,
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
}

/// What the door keeps for one file while its ring is full
/// ([`StoreBackedEngine::report`]).
#[derive(Default)]
struct Kept {
    /// The words kept, the words a caller hands on, and each report that
    /// looks at the ring under the list's lock. At zero a report goes
    /// straight to the ring, and the door's pass ends the file's due
    /// leases ([`StoreBackedEngine::sweep`]).
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

/// A decision slot's state at a file of the store, on lines of its own: a
/// thread that decides at one slot writes no line the thread at the next one
/// reads.
#[derive(Default)]
#[repr(align(128))]
struct SlotState {
    /// The seat's state across decisions. Only decisions take this lock.
    slot: Mutex<SlotBuffers>,
}

/// What a decision slot's holder keeps under the decision lock.
#[derive(Default)]
struct SlotBuffers {
    buffers: Buffers<()>,
    /// The holder's place at the seat's file, taken at the seat's first
    /// decision.
    at_file: Option<AtWork>,
}

thread_local! {
    /// The slot the calling thread took last at an engine. Until it moves it
    /// is the thread's number among the threads that have called one, so
    /// threads start at slots of their own.
    static LAST_SLOT: Cell<usize> = Cell::new(CALLERS.fetch_add(1, Ordering::Relaxed));
}

/// The threads that have called a door so far.
static CALLERS: AtomicUsize = AtomicUsize::new(0);

/// The decider a door decides as at a store that names none: such a store
/// has one file ([`MemoryStore::new`]).
const UNNAMED: DecisionSlot = DecisionSlot::new(0);

/// The door's word to the engine of a report applied to a lease
/// ([`told`]), under the name the lease's decision ran under: its file's
/// decider, and [`UNNAMED`] at a store that names none. A stage's call
/// that panics there is counted on the store for the host to read
/// ([`MemoryStore::lifecycle_faults`]), and no line is written: a host in
/// process writes its own.
fn quiet(
    store: &MemoryStore,
    world: &StoreSnapshot,
    found: &Sides,
    said: Said,
    by: Option<DecisionSlot>,
) {
    // A stage that keeps a decider's state hears a lease's events under
    // the name it heard the issue under, whichever thread applies them.
    told(store, world, found, said, Some(by.unwrap_or(UNNAMED)));
}

impl StoreBackedEngine {
    /// A door over `store`, with a seat for each of its files. The store
    /// is given the door's word to the engine: whoever applies a report
    /// to a lease tells the engine that issued it, and a dispatch feeds
    /// the standing generation's counts of sends for a row its world
    /// still lists. A store that names no decider has one file: the door
    /// decides there, and tells what is said of its leases, as decider 0.
    #[must_use]
    pub fn new(store: Arc<MemoryStore>) -> Self {
        let deciders = store.decision_slots();
        store.tell_through(quiet);
        if LeasesStore::term(&*store) == 0 {
            LeasesStore::set_term(&*store, 1);
        }
        let term = AtomicU64::new(LeasesStore::term(&*store));
        Self {
            seats: (0..deciders).map(|_| SlotState::default()).collect(),
            seat_flags: AtomicU8::new(0),
            parts: (0..deciders).map(|_| Part::default()).collect(),
            kept: (0..deciders).map(|_| Kept::default()).collect(),
            store,
            term,
            faults: AtomicU64::new(0),
        }
    }

    /// The store the door decides over, for what a host reads and
    /// publishes there. What became of a lease it says through
    /// [`StoreBackedEngine::report`], and its clock runs through
    /// [`StoreBackedEngine::sweep`] alone: the store's own pass
    /// ([`LeasesStore::expire`]) asks no host, and would end a lease whose
    /// dispatch the door still keeps.
    #[must_use]
    pub fn store(&self) -> &Arc<MemoryStore> {
        &self.store
    }

    /// How many decisions panicked.
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

    /// One decision for `gateway`'s `request` at `now`, on the calling
    /// thread: one load of the published rows, the engine's decision and,
    /// for a pick, its lease, filed at the issue with the engine told.
    ///
    /// # Errors
    ///
    /// [`DecisionRefused`]: no decision is made for the request.
    pub fn decide(
        &self,
        gateway: GatewayId,
        request: &Request<'_>,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        self.deciding(gateway, request, now, Whole(self))
    }

    /// [`StoreBackedEngine::decide`] for a request its host makes from the
    /// world the door decides under. `maker` is handed that world behind the
    /// door's load ([`RequestMaker::make`]), once for each run of the
    /// order: a load an apply moved to another decider is dropped before it
    /// asks. It is asked again where the order runs a second time behind a
    /// lease the store refused.
    ///
    /// A maker that panics is a decision that faulted.
    ///
    /// # Errors
    ///
    /// [`DecisionRefused`]: no decision is made for the request.
    pub fn decide_under(
        &self,
        gateway: GatewayId,
        maker: &impl RequestMaker,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        let (decided, _) = self.decide_under_stated(gateway, maker, now)?;
        Ok(decided)
    }

    /// [`StoreBackedEngine::decide_under`], with what the made request said of
    /// its decode: whether a pair's decode side was filed chosen
    /// ([`RequestForm::decode_later`]). The handle holds the side by it.
    pub(crate) fn decide_under_stated(
        &self,
        gateway: GatewayId,
        maker: &impl RequestMaker,
        now: Instant,
    ) -> Result<(DecideOutcome, bool), DecisionRefused> {
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
    /// plans this answers as [`StoreBackedEngine::decide`] and clears `placed`.
    /// A shed leaves `placed` empty.
    ///
    /// # Errors
    ///
    /// [`DecisionRefused`]: no decision is made for the request. `placed` is cleared.
    pub fn decide_encode_pair(
        &self,
        gateway: GatewayId,
        request: &Request<'_>,
        items: EncodeItems<'_>,
        placed: &mut EncodePlaced,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
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
    /// [`DecisionRefused::Decode`] says why the lease takes no decode side, and
    /// [`DecisionRefused::Faulted`] that the decision panicked. Either files
    /// nothing.
    pub fn decide_decode(
        &self,
        gateway: GatewayId,
        request: &Request<'_>,
        lease: LeaseId,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        self.decode_called(gateway, request, lease, now)
    }

    /// [`StoreBackedEngine::decide_decode`] for a request its host makes from
    /// the world the call decides under ([`RequestMaker::make`]). The
    /// sources of the request's routing key and the decode profile that reads
    /// the key are then of one load, across a reload that lands beside the
    /// call.
    ///
    /// A maker that panics is a decision that faulted.
    ///
    /// # Errors
    ///
    /// As [`StoreBackedEngine::decide_decode`].
    pub fn decide_decode_under(
        &self,
        gateway: GatewayId,
        maker: &impl RequestMaker,
        lease: LeaseId,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        self.decode_called(gateway, &Made(maker), lease, now)
    }

    /// The decode call for `request` at the seat of its lease's file.
    fn decode_called(
        &self,
        gateway: GatewayId,
        request: &impl Stated,
        lease: LeaseId,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        // The call mints no id: it spends no term and raises none.
        let term = self.term.load(Ordering::Relaxed);
        let called = Called(request, self.store.file_of(lease));
        self.at_a_seat(term, gateway, &called, now, DecodeCall(self, lease))
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
    /// ([`crate::tracking::DecisionGuard::completed`]).
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
    /// ([`StoreBackedEngine::report`]): what was said of a lease is applied
    /// before the end of its span is read, and the next pass ends what is still
    /// due. The pins of record past their end leave their table once a pass of
    /// the standing rules is due. A host that never calls keeps such a lease on
    /// file, counted on its worker, until the next issue at its file, and such
    /// a pin for the life of the store.
    pub fn sweep(&self, now: Instant) -> SweepOutcome {
        let store = &*self.store;
        // A kept word counts until its caller has handed it on.
        let settled = |at: usize| self.kept[at].waits.load(Ordering::SeqCst) == 0;
        SweepOutcome {
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
    ) -> Result<DecideOutcome, DecisionRefused> {
        match self.once(gateway, request, now, &mut order) {
            Err(DecisionRefused::Unleased) => self.once(gateway, request, now, order),
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
    ) -> Result<DecideOutcome, DecisionRefused> {
        let term = self.term.load(Ordering::Relaxed);
        let outcome = self.at_a_seat(term, gateway, request, now, order);
        match &outcome {
            // An id at or past the term's mark: the door raises the term,
            // as an election's hand-over does for a served controller.
            Ok(DecideOutcome {
                lease: Some(leased),
                ..
            }) if past_mark(leased.lease.id) => self.rise(term),
            Err(DecisionRefused::Unleased) => self.rise(term),
            _ => {}
        }
        outcome
    }

    /// Takes the seat the last generation called for and checks it
    /// against the world loaded there. If an apply changed which seat
    /// the request belongs at, it takes that seat and loads again. A call
    /// that names its lease's seat and decides at its model's applies what
    /// waits at the lease's file first ([`StoreBackedEngine::stand_at`]).
    fn at_a_seat(
        &self,
        term: u64,
        gateway: GatewayId,
        request: &impl Stated,
        now: Instant,
        mut order: impl Orders,
    ) -> Result<DecideOutcome, DecisionRefused> {
        let store = &*self.store;
        let mut seen = self.seat_flags.load(Ordering::Relaxed);
        loop {
            let (at, mut slot) = if seen & KEEPS_SLOT_STATE != 0 {
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
            debug_assert!(at < DecisionSlot::COUNT);
            let _acting = DecisionSlot::new(at as u8).enter();
            let SlotBuffers { buffers, at_file } = &mut *slot;
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
                if (flags ^ seen) & KEEPS_SLOT_STATE != 0 {
                    drop(world);
                    self.leave(at, at_file);
                    seen = flags;
                    continue;
                }
            }
            let pins = StorePins { store, now };
            // The request is taken up under this load: one its host makes
            // is made here, of the world its decision reads.
            let taken_up = |request: &Request<'_>| {
                let taken = DecisionCall {
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
                Err(DecisionRefused::Faulted)
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
        debug_assert!(at < DecisionSlot::COUNT);
        let _acting = DecisionSlot::new(at as u8).enter();
        let at_file = slot.at_file.get_or_insert_with(|| self.store.at_work());
        at_file.work();
        self.leave(at, at_file);
    }

    /// Takes the seat of `model`'s decider and waits for it: a model's
    /// requests there decide one at a time.
    fn home(&self, model: &str) -> (usize, MutexGuard<'_, SlotBuffers>) {
        let at = decider_of(model, self.seats.len());
        (at, self.seats[at].slot.lock())
    }

    /// Takes the seat the calling thread decides at: the one it took
    /// last, which until it moves is its number among the threads that
    /// have called a door. A number past this door's seats is counted
    /// round them once, and the thread keeps that seat.
    fn seat(&self) -> (usize, MutexGuard<'_, SlotBuffers>) {
        let seats = self.seats.len();
        // A thread at its end no longer has its own storage: it starts
        // at the first seat.
        let last = LAST_SLOT.try_with(Cell::get).unwrap_or(0);
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
        let _ = LAST_SLOT.try_with(|seat| seat.set(own));
        own
    }

    /// The next free seat past `own`, which another thread holds: the
    /// calling thread takes it and keeps it. Where no seat is free it
    /// waits at `own`, behind the decision that stands there.
    #[cold]
    fn another(&self, own: usize) -> (usize, MutexGuard<'_, SlotBuffers>) {
        let seats = self.seats.len();
        for step in 1..seats {
            let at = (own + step) % seats;
            if let Some(slot) = self.seats[at].slot.try_lock() {
                let _ = LAST_SLOT.try_with(|seat| seat.set(at));
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
        world: &StoreSnapshot,
        at: usize,
        term: u64,
        taken: &DecisionCall<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
        entry: Entry,
    ) -> Result<DecideOutcome, DecisionRefused> {
        match entry {
            Entry::Single => {
                let (decision, booking) = decide_in(world, taken, buffers, &self.parts[at]);
                let lease = self.file(world, term, taken, &buffers.named, &decision, booking)?;
                Ok(DecideOutcome {
                    decision,
                    lease,
                    decode_session: Session::NONE,
                })
            }
            Entry::Pair => self.pair(world, at, term, taken, buffers),
            Entry::Side => Err(DecisionRefused::PairSide),
        }
    }

    /// What stands behind a single's decision ([`Self::order`]): a pick's
    /// lease is issued, the engine that decided is told of it, and the
    /// record of sent prefixes takes the prefix of `named`, the decision's
    /// naming ([`issue`]).
    #[inline(always)]
    fn file(
        &self,
        world: &StoreSnapshot,
        term: u64,
        taken: &DecisionCall<'_, Request<'_>>,
        named: &AffinityRowsBuffers,
        decision: &Decision,
        booking: Option<Booking>,
    ) -> Result<Option<Leased>, DecisionRefused> {
        let engine = world.spec().engine();
        let told = |lease: &Lease| {
            engine.on_decision_issued(lease, decision);
        };
        let lease = booking
            .map(|booking| issue(&self.store, world, taken, term, booking, named, told))
            .transpose()
            .map_err(|_| DecisionRefused::Unleased)?;
        Ok(lease.map(|lease| Leased {
            lease,
            due: taken.due(),
        }))
    }

    /// A pair's order: decided for both sides and, for a pick, issued a
    /// lease a side under one id. Out of line: a single's order carries no
    /// part of a pair's.
    #[inline(never)]
    fn pair(
        &self,
        world: &StoreSnapshot,
        at: usize,
        term: u64,
        taken: &DecisionCall<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
    ) -> Result<DecideOutcome, DecisionRefused> {
        let (decision, booking) = pair_in(world, taken, buffers, &self.parts[at]);
        self.file_pair(world, term, taken, decision, booking)
    }

    fn file_pair(
        &self,
        world: &StoreSnapshot,
        term: u64,
        taken: &DecisionCall<'_, Request<'_>>,
        decision: Decision,
        booking: Option<PairBooking>,
    ) -> Result<DecideOutcome, DecisionRefused> {
        let lease = booking
            .map(|booking| {
                issue_pair(&self.store, world, taken, term, &decision, booking)
                    .map(|(prefill, _)| prefill)
            })
            .transpose()
            .map_err(|_| DecisionRefused::Unleased)?;
        Ok(DecideOutcome {
            decision,
            lease: lease.map(|lease| Leased {
                lease,
                due: taken.due(),
            }),
            decode_session: Session::NONE,
        })
    }

    /// A decode call's order at this door: the decision ([`decode_in`]) and,
    /// for a pick, the side's issue under the lease's id
    /// ([`Self::file_decode`]).
    #[inline(never)]
    fn decode(
        &self,
        world: &StoreSnapshot,
        at: usize,
        taken: &DecisionCall<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
        lease: LeaseId,
    ) -> Result<DecideOutcome, DecisionRefused> {
        let (store, own) = (&*self.store, &self.parts[at]);
        let decided = decode_in(world, store, taken, lease, buffers, own);
        self.file_decode(world, taken, lease, decided)
    }

    /// Files what a decode call decided ([`Self::decode`]): for a pick,
    /// the side's issue under the lease's id ([`issue_decode`]).
    #[inline(always)]
    fn file_decode(
        &self,
        world: &StoreSnapshot,
        taken: &DecisionCall<'_, Request<'_>>,
        lease: LeaseId,
        decided: Result<(Decision, Option<DecodeBooking>), DecodeRefused>,
    ) -> Result<DecideOutcome, DecisionRefused> {
        let store = &*self.store;
        let (decision, booking) = decided.map_err(DecisionRefused::Decode)?;
        let lease = booking
            .map(|booking| issue_decode(store, world, taken, lease, &decision, booking))
            .transpose()
            .map_err(DecisionRefused::Decode)?
            .map(|(lease, _)| lease);
        Ok(DecideOutcome {
            decision,
            lease: lease.map(|lease| Leased {
                lease,
                due: taken.due(),
            }),
            decode_session: Session::NONE,
        })
    }
}

impl DecisionEngine for StoreBackedEngine {
    #[inline]
    fn decide_under(
        &self,
        gateway: GatewayId,
        maker: &impl RequestMaker,
        now: Instant,
    ) -> Result<Decided, DecisionRefused> {
        let (decided, later) = self.decide_under_stated(gateway, maker, now)?;
        Ok(Decided::Whole(decided, later))
    }

    // A request stated ahead of the load pays nothing for a maker's form.
    #[inline]
    fn decide(
        &self,
        gateway: GatewayId,
        request: &Request<'_>,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        Self::decide(self, gateway, request, now)
    }

    // The file of the store's leases holds the prefill a lease booked.
    #[inline]
    fn decide_decode_under(
        &self,
        gateway: GatewayId,
        maker: &impl RequestMaker,
        held: &HeldDecision<'_>,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        Self::decide_decode_under(self, gateway, maker, held.lease.id, now)
    }

    #[inline]
    fn decide_decode(
        &self,
        gateway: GatewayId,
        request: &Request<'_>,
        held: &HeldDecision<'_>,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        Self::decide_decode(self, gateway, request, held.lease.id, now)
    }

    #[inline]
    fn decide_encode_pair(
        &self,
        gateway: GatewayId,
        request: &Request<'_>,
        items: EncodeItems<'_>,
        placed: &mut EncodePlaced,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        Self::decide_encode_pair(self, gateway, request, items, placed, now)
    }

    // The file finds the lease's record by its id, and holds whatever
    // else the view lends. A completion is also written on the served
    // worker's row, for the health precedence.
    #[inline]
    fn say(&self, gateway: GatewayId, held: &HeldDecision<'_>, said: Said, now: Instant) {
        let lease = held.lease.id;
        match said {
            Said::Completed(outcome) => {
                completed(self.store(), gateway, &outcome, now, |said| {
                    self.report(gateway, lease, said);
                });
            }
            Said::Dispatched(..) | Said::Released(..) => self.report(gateway, lease, said),
        }
    }

    fn sweep(&self, now: Instant) -> SweepOutcome {
        Self::sweep(self, now)
    }

    fn with_spec<R>(&self, read: impl FnOnce(&AppliedSpec) -> R) -> R {
        self.store().with_snapshot(|world| read(world.spec()))
    }

    fn prompt_hashes(
        &self,
        model: &str,
        tokens: Option<&[u32]>,
        text: Option<&str>,
        seeds: HashSeeds,
    ) -> PromptHashes {
        self.store().prompt_hashes(model, tokens, text, seeds)
    }

    fn faults(&self) -> u64 {
        Self::faults(self)
    }

    // This engine asks no shadow: none is counted.
    fn shadowed(&self, _: u32, _: ShadowKind) -> u64 {
        0
    }
}

/// The order of one decision at its seat. A binding's shadow is asked by
/// no order: a request whose binding names one is decided as any other.
trait Orders {
    /// Decides the request and, for a pick, issues its lease.
    fn plain(
        &mut self,
        world: &StoreSnapshot,
        at: usize,
        term: u64,
        taken: &DecisionCall<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
    ) -> Result<DecideOutcome, DecisionRefused>;
}

/// An order lent to a run. A decision whose lease the store refused runs
/// its order twice: the first run borrows it and the second takes it, so
/// each run is an instance of its own, as each closure's was.
impl<O: Orders> Orders for &mut O {
    #[inline(always)]
    fn plain(
        &mut self,
        world: &StoreSnapshot,
        at: usize,
        term: u64,
        taken: &DecisionCall<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
    ) -> Result<DecideOutcome, DecisionRefused> {
        (**self).plain(world, at, term, taken, buffers)
    }
}

/// The order of [`StoreBackedEngine::decide`] and
/// [`StoreBackedEngine::decide_under`]: the request takes the path its plan
/// gives it ([`entry`]).
struct Whole<'d>(&'d StoreBackedEngine);

impl Orders for Whole<'_> {
    // Forced into the seat's call: out of line the plain order would pay
    // a call the closure it stands for never made.
    #[inline(always)]
    fn plain(
        &mut self,
        world: &StoreSnapshot,
        at: usize,
        term: u64,
        taken: &DecisionCall<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
    ) -> Result<DecideOutcome, DecisionRefused> {
        let entry = entry(world, taken);
        self.0.order(world, at, term, taken, buffers, entry)
    }
}

/// [`Whole`] for [`StoreBackedEngine::decide_under_stated`]: each run of the
/// order notes what the made request said of its decode
/// ([`RequestForm::decode_later`]).
struct WholeStated<'d, 'l> {
    door: &'d StoreBackedEngine,
    later: &'l mut bool,
}

impl Orders for WholeStated<'_, '_> {
    #[inline(always)]
    fn plain(
        &mut self,
        world: &StoreSnapshot,
        at: usize,
        term: u64,
        taken: &DecisionCall<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
    ) -> Result<DecideOutcome, DecisionRefused> {
        *self.later = taken.request.decode_later();
        Whole(self.door).plain(world, at, term, taken, buffers)
    }
}

/// The order of [`StoreBackedEngine::decide_decode`]: the decode side of the
/// pair's lease on file ([`StoreBackedEngine::decode`]).
struct DecodeCall<'d>(&'d StoreBackedEngine, LeaseId);

impl Orders for DecodeCall<'_> {
    #[inline(always)]
    fn plain(
        &mut self,
        world: &StoreSnapshot,
        at: usize,
        _: u64,
        taken: &DecisionCall<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
    ) -> Result<DecideOutcome, DecisionRefused> {
        self.0.decode(world, at, taken, buffers, self.1)
    }
}

/// The order of [`StoreBackedEngine::decide_encode_pair`]: the request that
/// enters an `epd` plan is decided as a pair with its encode items placed, and
/// any other as [`Whole`] orders it, with `placed` cleared.
struct WithEncode<'d, 'i, 'p> {
    door: &'d StoreBackedEngine,
    items: EncodeItems<'i>,
    placed: &'p mut EncodePlaced,
}

impl WithEncode<'_, '_, '_> {
    /// The pair's order, for the request that enters an `epd` plan.
    /// `Err` is the path of any other request, with `placed` cleared.
    #[inline(always)]
    fn pair(
        &mut self,
        world: &StoreSnapshot,
        at: usize,
        term: u64,
        taken: &DecisionCall<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
    ) -> Result<Result<DecideOutcome, DecisionRefused>, Entry> {
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
        world: &StoreSnapshot,
        at: usize,
        term: u64,
        taken: &DecisionCall<'_, Request<'_>>,
        buffers: &mut Buffers<()>,
    ) -> Result<DecideOutcome, DecisionRefused> {
        match self.pair(world, at, term, taken, buffers) {
            Ok(decided) => decided,
            Err(entry) => self.door.order(world, at, term, taken, buffers, entry),
        }
    }
}

// The fixtures left here are read by the cases of a pair and of a handle
// alone, and those stand under `in-process`. A build without that feature
// reads none of these.
#[cfg(test)]
#[cfg_attr(
    not(feature = "in-process"),
    expect(
        dead_code,
        reason = "the cases that read these fixtures are built under `in-process` alone"
    )
)]
pub(crate) mod tests {
    use std::{error::Error, sync::Arc, time::Instant};

    use super::{Request, StoreBackedEngine};
    use crate::{
        document::spec::{AppliedSpec, PolicySpec},
        engine::view::{Constraints, Phase, RequestFacts, Role, WireModes, WorkerId},
        store::{
            EncodeHandoff, EncodeOffer, GatewayId, MemoryStore, ViewBuffers, ViewQuery,
            WorkerFacts, WorkerStatus, WorkersStore,
        },
    };

    pub(crate) const MODEL: &str = "model-a";
    pub(crate) const GATEWAY: GatewayId = GatewayId(11);

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

    fn encode_door() -> Result<StoreBackedEngine, Box<dyn Error>> {
        let store = Arc::new(MemoryStore::new());
        encode_member(&store, "grpc://d0:8000", Role::Decode, "c0");
        store.apply(AppliedSpec::validate(&encode_document()?)?)?;
        Ok(StoreBackedEngine::new(store))
    }

    pub(crate) fn push_fleet() -> Result<(StoreBackedEngine, WorkerId), Box<dyn Error>> {
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

    /// What the leases on file book on each member, as a decision reads it.
    fn booked(store: &MemoryStore, role: Role) -> Vec<(WorkerId, u32)> {
        let ask = ViewQuery {
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
}
