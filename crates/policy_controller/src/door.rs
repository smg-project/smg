//! The decision door: the order of one decision's calls, written once.
//!
//! A decision is one load of the published rows
//! ([`MemoryStore::deciding`]), the engine's decision over them, the
//! lease filed at the issue and the engine's hearing of it, in that
//! order. Every host runs this one order, behind a transport of its own.
//! The served controller runs it on its deciders' threads, behind the
//! fence and the budget that a wire and an election bring. A host that
//! embeds the engine runs it on the caller's thread ([`Door`]): no frame
//! is built and no thread is hopped.
//!
//! A request reaches the order in the form it was asked in (the trait
//! `Asked`): a frame as the served controller checked it, or the facts a
//! host in process holds ([`Request`]).
//!
//! A generation may sample picks and sheds at the seat that decided.
//! [`Door::explain`] reads them by gateway and request id.

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

#[cfg(feature = "controller")]
use crate::store::{DecodeSide, PairBuffers};
use crate::{
    counts::Slot,
    decider::Decider,
    decision::{Decision, Shed, ShedReason, Target, Trace},
    framework::{release, Engine, Plan, Scratch, KEEPS_DECIDERS, SAMPLES_DECISIONS},
    lease::{Lease, LeaseId, LeaseKind, Lifecycle, Outcome, Side},
    sessions::{Forgets, SessionVerdict},
    spec::AppliedSpec,
    store::{
        leases::DISPATCH_TTL, Ask, AtWork, Built, GatewayId, Hand, HealthSink, LeaseRecord,
        LeaseRefused, LeaseState, LeasesStore, MemoryStore, Named, Naming, Picked, Said,
        SessionsStore, Sides, ViewBuffers, World,
    },
    view::{
        fingerprint, AffinityFacts, Candidate, Constraints, NamedAffinity, Phase, PinSource,
        PoolFacts, RequestFacts, Role, Session, WireModes, WorkerId,
    },
};

mod explain;

pub use explain::DecisionRecord;
pub(crate) use explain::Samples;

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
    /// How many block hashes the request carries.
    fn hashes(&self) -> usize;
    /// How many of them the generation of `spec` reads.
    fn hashes_read(&self, spec: &AppliedSpec) -> usize;
    /// The request's block hashes, all of them.
    fn block_hashes(&self) -> &[u64];
    /// The request's seed.
    fn seed(&self) -> u64;
    /// Hands `decide` the request in the engine's form, with what the
    /// deciding host's view adds to it.
    fn read<R>(
        &self,
        lists: &mut Self::Lists,
        viewed: Viewed<'_>,
        decide: impl FnOnce(&RequestFacts<'_>, &Constraints<'_>) -> R,
    ) -> R;
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

/// The stage a decision is shed under when its pool has rows and its
/// gateway shares a wire with none of them.
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
    /// Both sides' rows of a pair decision and the table of their classes:
    /// a served controller's, the one host that decides a pair.
    #[cfg(feature = "controller")]
    pub(crate) pair: PairBuffers,
    pub(crate) scratch: Scratch,
    pub(crate) lists: L,
    pub(crate) named: Named,
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
    /// The pins of record lent to the decision, where its host keeps them:
    /// its engine asks them for a key its own table holds no pin of.
    pub(crate) pins: Option<&'a dyn PinSource>,
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

impl PinSource for StorePins<'_> {
    fn pinned(&self, key: Session, ttl_micros: Option<u64>) -> Option<WorkerId> {
        let pin = SessionsStore::lookup(self.store, key.get())?;
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

/// Whether `phase` is a side of `plan`, which no side's entry of the
/// engine decides: prefill and decode under either pair plan, and encode
/// under the plan that names an encode side. A pair is decided at its
/// prefill. The encode side rides the pair: the engine's entry keeps the
/// encoders the pair is compatible with, a bit the store sets on no row
/// yet, so it would shed each encode request as if no encoder stood.
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
    /// (`issue_pair`). A served controller decides it, and no other host.
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
    let (model, phase) = (taken.request.model(), taken.request.phase());
    if matches!(phase, Phase::Single) {
        return Entry::Single;
    }
    let binding = world.spec().engine().binding(model, phase);
    match binding {
        Some(binding) if pair_side(binding.plan, phase) => match phase {
            Phase::Prefill => Entry::Pair,
            _ => Entry::Side,
        },
        _ => Entry::Single,
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
    let ((decision, booking), built) = hand(
        world,
        taken,
        buffers,
        &own.undialed,
        |facts, pool, candidates, constraints, scratch| {
            if candidates.is_empty() && pool.pool_size > 0 {
                // What the gateway states of itself left none of a pool
                // that has rows: said under that constraint's name, where
                // the engine would name no stage.
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
            let decision = engine.decide(facts, pool, candidates, constraints, scratch);
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
                    tokens: facts.prompt_tokens,
                    now_micros: facts.now_micros,
                    built: None,
                })
            });
            (decision, booking)
        },
    );
    (
        decision,
        booking.map(|booking| Booking { built, ..booking }),
    )
}

/// The engine's pair decision over the world of one load, for the request
/// that enters a pair plan at its prefill. Both sides' rows are written
/// for it of one publication ([`World::build_pair`]), with what the leases
/// on file book on each row's worker, less what the gateway states of
/// itself takes from each side ([`narrow`]); the engine's pair entry
/// decides over them once, and the request's constraints are read by both
/// sides. A side with rows that the gateway's own word leaves none is shed
/// under that constraint's name, as a single's pool is. A pick's booking
/// holds both sides, each worker's count beside its own pool's row.
#[cfg(feature = "controller")]
pub(crate) fn pair_in<A: Asked>(
    world: &World,
    taken: &Taken<'_, A>,
    buffers: &mut Buffers<A::Lists>,
    own: &Part,
) -> (Decision, Option<PairBooking>) {
    let Taken {
        request,
        gateway,
        now,
        left,
        pins,
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
    let (prefill, _) = engine.reads_for(model, Phase::Prefill);
    let (decode, _) = engine.reads_for(model, Phase::Decode);
    let ask = Ask {
        gateway,
        now,
        pinned: None,
        affinity: prefill.affinity || decode.affinity,
    };
    let Buffers {
        pair,
        scratch,
        lists,
        ..
    } = buffers;
    let built = world.build_pair(model, &ask, pair);
    // One side alone is no pair: the engine answers over no candidate, as
    // it answers any host that hands it none.
    let mut pool = PoolFacts {
        generation: spec.generation(),
        ..PoolFacts::default()
    };
    let (wires, subset) = (request.wires(), request.subset());
    let mut emptied = false;
    if let Some(built) = &built {
        // Rows that are written carry their counts, laid before the
        // request's own narrowing moves a row.
        world.count_pair_in_flight(built, &ask, pair);
        pool = built.0.pool_facts();
        let mut took = 0;
        for (view, built) in pair.sides_mut().into_iter().zip([built.0, built.1]) {
            took += narrow(view, wires, subset);
            emptied |= view.candidates.is_empty() && built.pool_size > 0;
        }
        if took > 0 {
            own.undialed.fetch_add(took, Ordering::Relaxed);
            if let Some(wires) = wires {
                pool.reach = cut_reach(pool.reach, wires);
            }
        }
    }
    let viewed = Viewed {
        now_micros: built.map_or_else(|| world.micros(now), |(prefill, _)| prefill.now_micros),
        budget_micros: u32::try_from(left.as_micros()).unwrap_or(u32::MAX),
        // A pair's columns ride its two pools, and a pair names no rows.
        affinity: &[],
        stalled: &[],
        in_flight: None,
        max_block_hashes: request.hashes_read(spec),
        named: None,
        pins,
    };
    request.read(lists, viewed, |facts, constraints| {
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
        let decision = engine.decide_pair(
            facts,
            &pool,
            pair.pools(),
            constraints,
            constraints,
            scratch,
        );
        // What each side's lease takes of the request, read where the
        // request stands as the engine read it: the decode side's session
        // is the request's under the decode phase.
        let booking = decision.pick().zip(built).and_then(|(pick, built)| {
            let first = pick.targets.first()?;
            let chosen = first.pair?;
            // The pair's one decision: its rule's TTL as a single's
            // booking carries it, in each side's issued state. No pair
            // carries one yet: a sessions rule runs under the single
            // plan alone.
            let ttl_micros = if pick.session_verdict == SessionVerdict::None {
                None
            } else {
                scratch.session_ttl_micros()
            };
            let decode = DecodeSide {
                target: chosen.decode_id,
                picked: Picked::of_decodes(&pick.targets),
                session: RequestFacts {
                    phase: Phase::Decode,
                    ..*facts
                }
                .session(),
                state: LeaseState::issued(ttl_micros),
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
                    tokens: facts.prompt_tokens,
                    now_micros: facts.now_micros,
                    built: Some(built.0),
                },
                decode,
                slot: world.booked(&built.1, chosen.decode_id),
            })
        });
        (decision, booking)
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
#[cfg(feature = "controller")]
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
/// decided is told of it.
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
    decision: &Decision,
    booking: Booking,
) -> Result<Lease, LeaseRefused> {
    let record = record_of(taken, term, &booking, Side::Single);
    let booked = booked_in(world, &booking);
    // Under one hold of the turn the store applies what is handed, files
    // the lease, tells the engine that decided of it, and ends and tells
    // the leases it finds past their dispatch deadline.
    let engine = world.spec().engine();
    store.issue_decided(record, booked, |lease| {
        engine.on_decision_issued(lease, decision);
    })
}

/// Issues the two leases of a pair's pick, [`issue`] for a pair: a lease a
/// side under the one id the store mints, one record and one write of the
/// file, each side booking the count beside its own pool's row, and the
/// engine told once a side, the prefill's first. `Ok` is the prefill
/// side's lease, then the decode side's.
///
/// # Errors
///
/// Those of [`issue`].
#[cfg(feature = "controller")]
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

/// What [`hand`] reads of a pool for one request: the pool's facts, the
/// decision's instant in the rows' clock, the rows, the mask of the asking
/// gateway's vetoes beside lent rows, the affinity column, the leases'
/// counts beside lent rows, and the build the rows were read by.
type Read<'a> = (
    PoolFacts<'a>,
    u64,
    &'a [Candidate],
    &'a [u32],
    &'a [AffinityFacts],
    Option<crate::LentCounts<'a>>,
    Option<Built>,
);

/// Hands `decide` what the engine reads for one request: the request with
/// what is left of its budget, the pool's facts under the generation its
/// rows were published under, and those rows as the asking gateway reaches
/// them at the decision's instant, with the affinity column where the
/// request's profile declares it. A request that states no wires and no
/// hint of 0 reads its pool by the store's one door ([`World::view`]): the
/// rows where their publication left them, with its gateway's standing
/// vetoes and the leases' counts beside them, or rows the door writes
/// where the request's profile shows them to a plugin. One that states
/// wires or a hint of 0 is handed rows written for it ([`World::build`]),
/// what the leases on file book on each row's worker written over them
/// ([`World::count_in_flight`]), less what the gateway states of itself
/// takes ([`narrow`]). The rows its wires take are counted in `undialed`,
/// the decider's own part, and the list they leave takes a name of its
/// own ([`cut_reach`]). The request's block hashes are cut at what that
/// generation reads of them. `decide`'s answer comes back with the build
/// its rows were read by, `None` where no member serves the model.
///
/// A request whose profile reads the affinity column by its own stages is
/// handed no column: the rows the store names for it stand in the column's
/// place ([`World::name`]), at most its sample of each list, and every
/// holder and every warming row where it takes none
/// ([`Engine::named_for`](crate::framework::Engine::named_for)). The
/// naming is the store's one reader of its index: the column it builds
/// holds a row's age and no holder.
pub(crate) fn hand<A: Asked, R>(
    world: &World,
    taken: &Taken<'_, A>,
    buffers: &mut Buffers<A::Lists>,
    undialed: &AtomicU64,
    decide: impl FnOnce(
        &RequestFacts<'_>,
        &PoolFacts<'_>,
        &[Candidate],
        &Constraints<'_>,
        &mut Scratch,
    ) -> R,
) -> (R, Option<Built>) {
    let Taken {
        request,
        gateway,
        now,
        left,
        pins,
    } = *taken;
    let spec = world.spec();
    let (model, phase) = (request.model(), request.phase());
    let engine = spec.engine();
    // What the decision's profile reads, and whether it shows its rows to
    // a plugin.
    let (needs, shown) = engine.reads_for(model, phase);
    // A profile whose own stages read the column is handed the rows the
    // store names in the column's place, and no column: a stage that
    // reads holders is fed them. The sample caps the rows named, and a
    // profile that takes none is named every one. A profile that shows
    // its rows to a plugin, or whose warm-up stages rule by two rulings,
    // is named no row and keeps its column. The deciding profile answers
    // for itself: the needs are merged over a pair's two phases and a
    // sessions rule, so they say only where to ask, and no other decision
    // pays the second walk.
    let cap = if needs.affinity {
        engine.named_for(model, phase)
    } else {
        0
    };
    let names = cap > 0;
    let ask = Ask {
        gateway,
        now,
        pinned: None,
        affinity: needs.affinity && !names,
    };
    let role = match phase {
        Phase::Single => Role::Single,
        Phase::Prefill => Role::Prefill,
        Phase::Decode => Role::Decode,
        Phase::Encode => Role::Encode,
    };
    let Buffers {
        view,
        scratch,
        lists,
        named,
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
    // row's bits and ages as the row's own fields.
    let by_door = wires.is_none() && request.subset() != Some(0);
    let (pool, now_micros, candidates, stalled, affinity, counts, built): Read<'_> = if by_door {
        match world.view(model, role, &ask, shown, view) {
            Some(read) => (
                read.pool_facts(),
                read.built.now_micros,
                read.candidates,
                read.stalled,
                read.affinity,
                read.in_flight,
                Some(read.built),
            ),
            None => (none(), world.micros(now), &[], &[], &[], None, None),
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
                let took = narrow(view, wires, request.subset());
                if took > 0 {
                    undialed.fetch_add(took, Ordering::Relaxed);
                    if let Some(wires) = wires {
                        pool.reach = cut_reach(pool.reach, wires);
                    }
                } else {
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
            &[],
            &view.affinity,
            None,
            built,
        )
    };
    if names {
        // A row is named by its place among the rows the decision is made
        // over: the lent rows, or the built ones after their narrowing.
        let hashes = request.block_hashes();
        let naming = Naming {
            gateway,
            // The first hashes, as many as the generation reads of them:
            // the engine's request takes the same count.
            block_hashes: hashes.get(..request.hashes_read(spec)).unwrap_or(hashes),
            seed: request.seed(),
            cap,
            wires,
            now_micros,
        };
        world.name(model, role, &naming, candidates, named);
    }
    let viewed = Viewed {
        now_micros,
        budget_micros: u32::try_from(left.as_micros()).unwrap_or(u32::MAX),
        affinity,
        stalled,
        in_flight: counts,
        max_block_hashes: request.hashes_read(spec),
        named: names.then(|| named.affinity()),
        pins,
    };
    let answer = request.read(lists, viewed, |facts, constraints| {
        decide(facts, &pool, candidates, constraints, scratch)
    });
    (answer, built)
}

/// Whether the request's own narrowing can leave a row: a subset of none
/// and a mask of no wire clear every row ([`narrow`]).
pub(crate) fn keeps_a_row(wires: Option<WireModes>, subset: Option<u32>) -> bool {
    subset != Some(0) && wires != Some(WireModes::NONE)
}

/// Narrows a built view by what the request's gateway states of itself,
/// and returns how many rows its wires took. A hint of 0 keeps none, and
/// leaves the wires nothing to take. A gateway that states wires keeps the
/// members that share one with it: a member that states none is taken as
/// one that states another is, and a gateway that states none keeps every
/// row. A positive hint cuts nothing. The kept rows keep their order and
/// take their positions, and the affinity column follows them. A request
/// that states neither pays the two compares.
pub(crate) fn narrow(view: &mut ViewBuffers, wires: Option<WireModes>, subset: Option<u32>) -> u64 {
    if subset == Some(0) {
        view.candidates.clear();
        view.affinity.clear();
        return 0;
    }
    let Some(wires) = wires else {
        return 0;
    };
    let rows = view.candidates.len();
    let dialed = |row: &Candidate| row.wire_modes.intersects(wires);
    // The column is by row, where the profile declared one: it is cut by
    // the rows' wires before the rows are.
    let mut beside = view.candidates.iter();
    view.affinity.retain(|_| beside.next().is_some_and(dialed));
    view.candidates.retain(dialed);
    let took = rows - view.candidates.len();
    if took > 0 {
        for (row, position) in view.candidates.iter_mut().zip(0..) {
            row.position = position;
        }
    }
    took as u64
}

/// The name of the list a gateway's wires cut out of a build's
/// ([`PoolFacts::reach`]): the build's word turned by the wires. Each of the
/// eight sets of wires turns it by its own multiple of one odd word, eight
/// words that differ and none of them zero: no cut reads as its build's
/// whole list, and no two cuts of one build read alike.
pub(crate) fn cut_reach(built: u64, wires: WireModes) -> u64 {
    built ^ (u64::from(wires.bits()) + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
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
    /// What the leases on file book on each row, by position, where the
    /// candidates are a publication's own rows; `None` where they were
    /// written for the decision and carry their counts.
    pub(crate) in_flight: Option<crate::LentCounts<'a>>,
    /// How many of the request's block hashes the deciding generation
    /// reads.
    pub(crate) max_block_hashes: usize,
    /// The rows the deciding host named in the column's place, where the
    /// request's profile takes them.
    pub(crate) named: Option<NamedAffinity<'a>>,
    /// The pins of record the host lends the decision ([`PinSource`]);
    /// `None` where it keeps none.
    pub(crate) pins: Option<&'a dyn PinSource>,
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
/// lease's own engine. A dispatch is told on the target its report names
/// by index, as the record holds the pick, and the lease told names that
/// index: the record was read before the door wrote it. A report that
/// names no target of the pick tells nothing.
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

/// What [`told`] tells, under its catch.
fn tell(store: &MemoryStore, world: &World, found: &Sides, said: Said) {
    let spec = world.spec();
    let engine = (spec.generation() == found.record.lease.generation).then(|| spec.engine());
    match (said, engine) {
        (Said::Dispatched(dispatch, at), engine) => {
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
            // The table hears first and the pin of record reads it.
            // The stages hear last, so nothing is held across their
            // loop: a value kept for the pin behind it cost every
            // lease a store and a reload a profile. A stage's call
            // that unwinds there leaves the pin of record written,
            // beside the table's entry.
            if let Some(engine) = engine {
                engine.dispatched_table(&lease, &target, dispatch.side);
            }
            pinned(store, &lease, stood, engine, id, at);
            if let Some(engine) = engine {
                engine.dispatched_stages(
                    &lease,
                    &target,
                    dispatch.side,
                    dispatch.since_decision_micros,
                );
            }
        }
        (Said::Completed(outcome), Some(engine)) => {
            let side = Some(outcome.side);
            found.each_ending(side, |lease| engine.on_completed(lease, &outcome));
        }
        (Said::Released(side, reason), Some(engine)) => {
            found.each_ending(side, |lease| engine.on_released(lease, reason));
        }
        (Said::Completed(_) | Said::Released(..), None) => {}
    }
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
/// `Completed` frame and a host's handle both say a completion here, so
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

/// A request as a host in process asks it: the engine's request and the
/// host's constraints as the host holds them. No frame is built.
#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    /// The request's id as the host states it: its lease keeps the id's
    /// fingerprint.
    pub id: &'a [u8],
    /// The request as the engine reads it. The door writes over the
    /// host's what the deciding view adds: the instant in the rows'
    /// clock, the affinity column, the rows the store names and the
    /// leases' counts. A budget of 0 states none, and the engine is told
    /// no bound. A frame's 0 says none is left: a host itself refuses a
    /// request whose time is spent.
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

    fn hashes(&self) -> usize {
        self.facts.block_hashes.len()
    }

    fn hashes_read(&self, spec: &AppliedSpec) -> usize {
        spec.block_hashes_read(self.facts.hash_kind, self.facts.block_size)
    }

    fn block_hashes(&self) -> &[u64] {
        self.facts.block_hashes
    }

    fn seed(&self) -> u64 {
        self.facts.seed
    }

    fn read<R>(
        &self,
        (): &mut (),
        viewed: Viewed<'_>,
        decide: impl FnOnce(&RequestFacts<'_>, &Constraints<'_>) -> R,
    ) -> R {
        let hashes = self.facts.block_hashes;
        let facts = RequestFacts {
            // The first hashes, as deep as the generation reads.
            block_hashes: hashes.get(..viewed.max_block_hashes).unwrap_or(hashes),
            budget_micros: viewed.budget_micros,
            now_micros: viewed.now_micros,
            affinity: viewed.affinity,
            named: viewed.named,
            lent_in_flight: viewed.in_flight.as_ref(),
            ..self.facts
        };
        let constraints = Constraints {
            stalled: viewed.stalled,
            pins: viewed.pins,
            ..self.constraints
        };
        decide(&facts, &constraints)
    }
}

/// A decision as its host takes it.
#[derive(Debug)]
pub struct Decided {
    /// The engine's decision.
    pub decision: Decision,
    /// The lease filed for a pick: one a decision, a credit on the pick's
    /// first target. `None` for a decision that picks no worker.
    pub lease: Option<Leased>,
}

/// A pick's lease as its door hands it over: the lease the store filed,
/// and when its dispatch span ends.
#[derive(Clone, Copy, Debug)]
pub struct Leased {
    /// The lease as the store filed it.
    pub lease: Lease,
    /// When its dispatch span ends: the host names a dispatch before
    /// then, or the lease is ended as abandoned, by the next issue at its
    /// file or by the door's clock ([`Door::sweep`]).
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
    /// The request is for a side of a pair plan. The served path answers
    /// a pair's one request with two leases; this door's answer carries
    /// one, so it decides no side of a pair plan yet.
    PairSide,
    /// The store filed no lease for the pick, at the second try too: its
    /// term mints no id, or another hand raised it under both tries.
    Unleased,
    /// The decision panicked. It cost this request alone: the next one
    /// decides on new buffers.
    Faulted,
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

/// A seat on lines of its own: a thread that decides at one seat writes
/// no line the thread at the next one reads.
#[derive(Default)]
#[repr(align(128))]
struct Seat(Mutex<DeciderSlot>);

/// What a seat's holder keeps across decisions.
#[derive(Default)]
struct DeciderSlot {
    buffers: Buffers<()>,
    /// The holder's place at the seat's file, taken at the seat's first
    /// decision.
    at_file: Option<AtWork>,
    samples: Samples,
}

thread_local! {
    /// The seat the calling thread took last at a door. Until it moves it
    /// is the thread's number among the threads that have called one, so
    /// threads start at seats of their own.
    static SEAT: Cell<usize> = Cell::new(CALLERS.fetch_add(1, Ordering::Relaxed));
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
    /// to a lease tells the engine that issued it. A store that names no
    /// decider has one file: the door decides there, and tells what is
    /// said of its leases, as decider 0.
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
    /// [`Refused`]: no decision is made for the request.
    pub fn decide(
        &self,
        gateway: GatewayId,
        request: &Request<'_>,
        now: Instant,
    ) -> Result<Decided, Refused> {
        match self.once(gateway, request, now) {
            // The store filed no lease under the term the door held. The
            // term was read again, and raised where it was spent: the
            // request is decided once more under it.
            Err(Refused::Unleased) => self.once(gateway, request, now),
            decided => decided,
        }
    }

    /// Finds a sampled decision by gateway and request id. Each seat is
    /// held for at most one scan of its 1,024 slots.
    #[must_use]
    pub fn explain(&self, gateway: GatewayId, request_id: &[u8]) -> Option<DecisionRecord> {
        let key = fingerprint(request_id);
        self.seats
            .iter()
            .filter_map(|seat| seat.0.lock().samples.find(0, gateway, key))
            .max_by_key(|record| record.at)
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
    /// handle says one completion and the handle writes both
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

    /// One run of the order for `request`, under the term the door holds.
    /// Behind a lease at the term's mark the term is raised. Behind a
    /// lease the store refused it is read again, and raised where it was
    /// spent.
    fn once(
        &self,
        gateway: GatewayId,
        request: &Request<'_>,
        now: Instant,
    ) -> Result<Decided, Refused> {
        let term = self.term.load(Ordering::Relaxed);
        let outcome = self.at_a_seat(term, gateway, request, now);
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
    /// same world governs the decision and its sampling.
    fn at_a_seat(
        &self,
        term: u64,
        gateway: GatewayId,
        request: &Request<'_>,
        now: Instant,
    ) -> Result<Decided, Refused> {
        let store = &*self.store;
        let mut seen = self.seat_flags.load(Ordering::Relaxed);
        loop {
            let (at, mut slot) = if seen & KEEPS_DECIDERS != 0 {
                self.home(request.facts.model)
            } else {
                self.seat()
            };
            // The store lays at most one file for each byte-sized name.
            debug_assert!(at < Decider::COUNT);
            let _acting = Decider::new(at as u8).enter();
            let DeciderSlot {
                buffers,
                at_file,
                samples,
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
            };
            let order = || self.order(&world, at, term, &taken, buffers);
            let decided = catch_unwind(AssertUnwindSafe(order));
            if flags & SAMPLES_DECISIONS != 0 {
                if let Ok(Ok(answer)) = &decided {
                    samples.keep(0, engine, &taken, &answer.decision);
                }
            }
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

    /// Takes the seat of `model`'s decider and waits for it: a model's
    /// requests there decide one at a time.
    fn home(&self, model: &str) -> (usize, MutexGuard<'_, DeciderSlot>) {
        let at = decider_of(model, self.seats.len());
        (at, self.seats[at].0.lock())
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
        match self.seats[own].0.try_lock() {
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
            if let Some(slot) = self.seats[at].0.try_lock() {
                let _ = SEAT.try_with(|seat| seat.set(at));
                return (at, slot);
            }
        }
        (own, self.seats[own].0.lock())
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
    ) -> Result<Decided, Refused> {
        // A pair plan's one request is answered with two leases, and
        // this door's answer carries one: each side of a pair plan is
        // refused ahead of its decision. A single request is no
        // plan's side, and its binding is not looked up here.
        if entry(world, taken) != Entry::Single {
            return Err(Refused::PairSide);
        }
        let (decision, booking) = decide_in(world, taken, buffers, &self.parts[at]);
        let lease = booking
            .map(|booking| issue(&self.store, world, taken, term, &decision, booking))
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
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{
        collections::BTreeSet,
        sync::{
            atomic::{AtomicBool, Ordering},
            mpsc, Arc, Barrier,
        },
        thread,
        time::{Duration, Instant},
    };

    use parking_lot::Mutex;

    use super::{
        decider_of, pinned, DecisionRecord, Door, Refused, Request, Swept, SEAT, TERM_MARK,
    };
    use crate::{
        builtins::Random,
        catalog::Catalog,
        decider::Decider,
        decision::{Decision, Target, Targets},
        framework::{DecisionScope, Picker, TieBand},
        lease::{Lease, LeaseId, Lifecycle, Outcome, ReleaseReason, Side, TransportClass},
        sessions::SessionVerdict,
        spec::{AppliedSpec, Needs, PolicySpec},
        store::{
            Ask, Dispatch, GatewayId, LeaseRecord, LeaseState, LeasesStore, MemoryStore, Picked,
            Said, SessionsStore, ViewBuffers, WorkerFacts, WorkerStatus, WorkersStore,
        },
        view::{
            Candidate, Constraints, Phase, PoolFacts, RequestFacts, Role, Session, WireModes,
            WorkerId,
        },
    };

    const MODEL: &str = "model-a";
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
            let key = format!("http://{name}:8000");
            let facts = WorkerFacts {
                key: key.clone(),
                engine: key,
                incarnation: 1,
                role: Role::Single,
                models: vec![MODEL.to_owned()],
                dp_size: 1,
                max_running: Some(4_096),
                ..WorkerFacts::default()
            };
            let id = store.upsert(facts).id();
            store.publish_now();
            let row = WorkersStore::get(&store, id).expect("the row");
            if row.status != WorkerStatus::Ready {
                let ready = store.set_status(id, WorkerStatus::Ready, row.revision);
                ready.expect("the row stands");
            }
        }
        store
            .apply(preset("random", 1))
            .expect("above the unbound start");
        Arc::new(store)
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
        let spec = AppliedSpec::validate_digested(&document, digest, &catalog);
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

    /// What the leases on file book on each member, as a decision reads it.
    fn booked(store: &MemoryStore) -> Vec<(WorkerId, u32)> {
        let ask = Ask {
            gateway: GATEWAY,
            now: Instant::now(),
            pinned: None,
            affinity: false,
        };
        let mut out = ViewBuffers::default();
        store.deciding(|world| {
            if let Some(built) = world.build(MODEL, Role::Single, &ask, &mut out) {
                world.count_in_flight(&built, &ask, &mut out.candidates);
            }
        });
        let rows = out.candidates.iter();
        rows.map(|row| (row.id, row.in_flight)).collect()
    }

    /// The members the leases on file book, each with its count.
    pub(crate) fn counted(store: &MemoryStore) -> Vec<(WorkerId, u32)> {
        let mut rows = booked(store);
        rows.retain(|&(_, count)| count > 0);
        rows
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

    #[test]
    fn sampling_off_allocates_no_slots_and_explains_nothing() {
        let door = Door::new(fleet(1));
        let _ = leased(&door, b"off", 1);
        assert!(door.explain(GATEWAY, b"off").is_none());
        let seat = door.seats[0].0.lock();
        assert_eq!(seat.samples.slots(), 0);
        assert_eq!(seat.samples.counted(), 0);
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
    fn each_seat_keeps_every_nth_answer_by_its_own_count() {
        let store = fleet(2);
        sampling(&store, 2, 3);
        let door = Door::new(store);
        thread::scope(|scope| {
            let door = &door;
            for seat in [0_u8, 1] {
                scope.spawn(move || {
                    let at = usize::from(seat);
                    seated(at);
                    for n in 1..=6 {
                        let id = [seat, n];
                        let (lease, _) = leased(door, &id, u64::from(n));
                        assert_eq!(door.store().file_of(lease), at);
                        door.report(GATEWAY, lease, gone());
                        if n < 3 {
                            assert_eq!(door.seats[at].0.lock().samples.slots(), 0);
                        }
                    }
                });
            }
        });
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
        let slots = door.seats[0].0.lock().samples.slots();
        assert_eq!(slots, 1_024);
        let bytes = size_of::<DecisionRecord>();
        writeln!(
            io::stdout(),
            "record_bytes={bytes} ring_bytes={}",
            slots * bytes
        )
        .expect("stdout");
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
        for (member, count) in booked(&store) {
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

    /// The served path answers a pair plan's one request with two leases.
    /// The door's answer carries one, so the door refuses each side of a
    /// pair plan ahead of its decision, and files nothing.
    #[test]
    fn a_request_under_a_pair_plan_is_refused_and_files_no_lease() {
        let store = MemoryStore::new().with_deciders(1);
        for (name, role) in [("p0", Role::Prefill), ("d0", Role::Decode)] {
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
            let id = store.upsert(facts).id();
            store.publish_now();
            let row = WorkersStore::get(&store, id).expect("the row");
            if row.status != WorkerStatus::Ready {
                let ready = store.set_status(id, WorkerStatus::Ready, row.revision);
                ready.expect("the row stands");
            }
        }
        let pair = PolicySpec::pair_presets("least_load", "least_load");
        let mut document = pair.expect("a pair of presets of this build");
        document.generation = 1;
        let paired = AppliedSpec::validate(&document).expect("the document validates");
        store.apply(paired).expect("above the unbound start");
        let door = Door::new(Arc::new(store));
        for phase in [Phase::Prefill, Phase::Decode] {
            let mut asked = request(b"r-1", 7);
            asked.facts.phase = phase;
            let refused = door.decide(GATEWAY, &asked, Instant::now()).err();
            assert_eq!(refused, Some(Refused::PairSide), "{phase:?}");
        }
        assert_eq!(LeasesStore::live(&**door.store(), GATEWAY), 0);
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
        assert!(decided.decision.pick().is_none(), "no member it can dial");
        assert!(decided.lease.is_none());
        assert_eq!((door.undialed(), door.hashes_cut()), (3, 0));
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

    /// A picker that holds a decision's scope and panics where it is
    /// asked what it needs, once armed.
    #[derive(Debug)]
    struct Needy(Arc<AtomicBool>);

    impl Lifecycle for Needy {}

    impl DecisionScope for Needy {
        fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}

        fn close(&self) {}
    }

    impl Picker for Needy {
        fn name(&self) -> &'static str {
            "needy"
        }

        fn needs(&self) -> Needs {
            assert!(!self.0.load(Ordering::Relaxed), "the case's own panic");
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
            Random.pick(facts, candidates, kept, scores, band, out);
        }

        fn scope(&self) -> Option<&dyn DecisionScope> {
            Some(self)
        }
    }

    /// A stage is asked what it needs at each decision, and that call can
    /// panic as any other. The seat's question calls no stage, so the
    /// panic is the decision's own, at the first request under the
    /// stage's generation and at a later one: each is refused as faulted,
    /// nothing unwinds into the host, and the seat the caller left is at
    /// rest.
    #[test]
    fn a_stage_that_panics_at_its_needs_costs_its_request_alone() {
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
        picking(&store, 2, "needy", move || {
            Box::new(Needy(Arc::clone(&stage)))
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
