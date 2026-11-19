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

use std::{
    mem,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use parking_lot::Mutex;

use crate::{
    decider::Decider,
    decision::{Decision, Shed, ShedReason, Target, Trace},
    framework::{release, Engine, Plan, Scratch},
    lease::{Lease, LeaseId, LeaseKind, Lifecycle, Side},
    sessions::{Forgets, SessionVerdict},
    spec::AppliedSpec,
    store::{
        leases::DISPATCH_TTL, Ask, AtWork, Built, GatewayId, Hand, LeaseRecord, LeaseRefused,
        LeaseState, LeasesStore, MemoryStore, Named, Naming, Picked, Said, SessionsStore,
        ViewBuffers, World,
    },
    view::{
        fingerprint, AffinityFacts, Candidate, Constraints, NamedAffinity, Phase, PinSource,
        PoolFacts, RequestFacts, Role, Session, WireModes, WorkerId,
    },
};

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

/// Whether `phase` is a side of `plan` that the pair door decides: prefill
/// and decode under either pair plan, and encode under the plan that names
/// an encode side. That side rides the pair: the engine's entry keeps the
/// encoders the pair is compatible with, a bit the store sets on no row
/// yet, so it would shed each encode request as if no encoder stood.
fn pair_side(plan: Plan, phase: Phase) -> bool {
    match plan {
        Plan::Single => false,
        Plan::PdPair => matches!(phase, Phase::Prefill | Phase::Decode),
        Plan::Epd { .. } => !matches!(phase, Phase::Single),
    }
}

/// The engine's decision over the world of one load: the rows and the
/// engine are of one publication. `None` for a request for a side of a
/// pair plan ([`pair_side`]): the pair door decides those, and one side's
/// entry of the engine never does. What a decision that is made counts, it
/// counts in `own`, its decider's part.
pub(crate) fn decide_in<A: Asked>(
    world: &World,
    taken: &Taken<'_, A>,
    buffers: &mut Buffers<A::Lists>,
    own: &Part,
) -> Option<(Decision, Option<Booking>)> {
    let spec = world.spec();
    let engine = spec.engine();
    let (model, phase) = (taken.request.model(), taken.request.phase());
    // A single request is no plan's side: its binding is looked up once, by
    // the engine.
    if !matches!(phase, Phase::Single)
        && engine
            .binding(model, phase)
            .is_some_and(|binding| pair_side(binding.plan, phase))
    {
        return None;
    }
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
    Some((
        decision,
        booking.map(|booking| Booking { built, ..booking }),
    ))
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

/// Issues the lease of a pick, one a decision, ahead of the read that lets
/// its answer leave: a credit on the pick's first target, minted and filed
/// by the store under `term`, the fence's at the decision's first read, at
/// that read's instant, from which [`DISPATCH_TTL`] runs for its gateway
/// to name a dispatch. The lease books the count the decision's own world
/// lays beside its target's row, and the engine that decided is told of
/// it.
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
    let lease = Lease {
        // The store mints the id.
        id: LeaseId::default(),
        kind: LeaseKind::Credit,
        side: Side::Single,
        target_index: 0,
        booked: booking.target,
        generation: booking.generation,
        session: booking.session,
        session_verdict: booking.verdict,
        issued_micros: booking.now_micros,
        tokens: booking.tokens,
    };
    let record = LeaseRecord {
        lease,
        gateway: taken.gateway,
        request: fingerprint(taken.request.id()),
        target: booking.target,
        picked: booking.picked,
        decode: None,
        term,
        issued_at: taken.now,
        dispatch_deadline: taken.due(),
        state: LeaseState::Issued,
        session_ttl_micros: booking.ttl_micros,
    };
    let built = booking.built;
    let booked = built.and_then(|built| world.booked(&built, booking.target));
    // Under one hold of the turn the store applies what is handed, files
    // the lease, tells the engine that decided of it, and ends and tells
    // the leases it finds past their dispatch deadline.
    let engine = world.spec().engine();
    let lease = store.issue_decided(record, booked, |lease| {
        engine.on_decision_issued(lease, decision);
    })?;
    Ok(lease)
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
/// The engine told is the one of the lease's own generation, where that
/// generation still stands: a later generation's stages never saw the
/// issue and hear nothing of it. A dispatch is told on the target its
/// report names by index, as the record holds the pick, and the lease told
/// names that index: the record was read before the door wrote it. A
/// report that names no target of the pick tells nothing.
///
/// Its caller is whoever applies the report, under the turn of the leases'
/// file: what it calls is bounded and takes no door of the leases. It
/// enters as the decider whose file the lease is on, on whichever thread
/// applies.
///
/// A stage's call that panics is held here, whoever applies: a decider
/// inside a decision or as it takes up or leaves its work, a session, the
/// gateways' sweep, a host's own thread. The lease's event stands on file
/// as applied and the fault is counted on the store
/// ([`MemoryStore::lifecycle_faults`]). It costs no request its decision
/// and ends no thread.
pub(crate) fn told(
    store: &MemoryStore,
    world: &World,
    record: &LeaseRecord,
    said: Said,
    by: Option<Decider>,
) -> bool {
    // The lease's own decider acts while its event is told, whichever
    // thread applies the report.
    let _acting = by.map(Decider::enter);
    let Err(payload) = catch_unwind(AssertUnwindSafe(|| tell(store, world, record, said))) else {
        store.call_returned();
        return false;
    };
    // A payload's own drop can panic: the engine's helper frees it under a
    // catch of its own.
    release(payload);
    store.call_faulted()
}

/// What [`told`] tells, under its catch.
fn tell(store: &MemoryStore, world: &World, record: &LeaseRecord, said: Said) {
    let spec = world.spec();
    let engine = (spec.generation() == record.lease.generation).then(|| spec.engine());
    match (said, engine) {
        (Said::Dispatched(dispatch, at), engine) => {
            let Some((id, dp_rank)) = record.target_at(dispatch.target_index) else {
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
                ..record.lease
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
            pinned(store, record, engine, id, at);
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
            engine.on_completed(&record.lease, &outcome);
        }
        (Said::Released(_, reason), Some(engine)) => engine.on_released(&record.lease, reason),
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
/// under, as its record on file carries it, counted from the lease's
/// issue as the table counted it then. So it does where the dispatch
/// itself made the table's entry anew and no decision has read it: that
/// entry holds no instant yet. An end the lease's rule counts never moves
/// the pin's end earlier: a lease dispatched behind a newer one of its
/// key, or behind a dispatch whose table ended the pin later, leaves the
/// later end standing. Either end stands on the decision clock: a
/// dispatch converts no clock. The table's lock is let go before the
/// store's pin is written: no lock is held across the two.
fn pinned(
    store: &MemoryStore,
    record: &LeaseRecord,
    engine: Option<&Engine>,
    dispatched: WorkerId,
    at: Instant,
) {
    let lease = &record.lease;
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
            let ends = record
                .session_ttl_micros
                .map(|ttl| lease.issued_micros.saturating_add(ttl));
            SessionsStore::pin_no_earlier(store, key, worker, at, ends);
        }
    }
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
    /// leases' counts. A budget of 0 states none.
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
    /// The request is for a side of a pair plan: no door decides one yet.
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
/// deciding thread. A caller's thread decides for the decider its
/// request's model names, by the rule the served controller keeps. It
/// takes that decider's slot, acts as that decider ([`Decider::enter`])
/// and works at its file for the one decision, so the lease is filed
/// there. Callers that ask for one model decide one at a time, in the
/// order they took the slot. Callers whose models name different
/// deciders decide beside each other.
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
    /// A slot a decider: its holder decides for that decider.
    slots: Box<[Mutex<Slot>]>,
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
    /// straight to the ring.
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

/// What a decider's holder keeps across decisions.
#[derive(Default)]
struct Slot {
    buffers: Buffers<()>,
    /// The seat at the decider's file, taken at the slot's first decision.
    seat: Option<AtWork>,
}

/// The decider a door decides as at a store that names none: such a store
/// has one file ([`MemoryStore::new`]).
const UNNAMED: Decider = Decider::new(0);

/// The door's word to the engine of a report applied to a lease
/// ([`told`]), under the name the lease's decision ran under: its file's
/// decider, and [`UNNAMED`] at a store that names none. A stage's call
/// that panics there is counted on the store for the host to read
/// ([`MemoryStore::lifecycle_faults`]), and no line is written: a host in
/// process writes its own.
fn quiet(
    store: &MemoryStore,
    world: &World,
    record: &LeaseRecord,
    said: Said,
    by: Option<Decider>,
) {
    // A stage that keeps a decider's state hears a lease's events under
    // the name it heard the issue under, whichever thread applies them.
    told(store, world, record, said, Some(by.unwrap_or(UNNAMED)));
}

impl Door {
    /// A door over `store`, with a slot for each of its files. The store
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
            slots: (0..deciders).map(|_| Mutex::default()).collect(),
            parts: (0..deciders).map(|_| Part::default()).collect(),
            kept: (0..deciders).map(|_| Kept::default()).collect(),
            store,
            term,
            faults: AtomicU64::new(0),
        }
    }

    /// The store the door decides over, for what a host reads and
    /// publishes there. What became of a lease it says through
    /// [`Door::report`].
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
    /// The pins of record past their end leave their table once a pass
    /// of the standing rules is due. A host that never calls keeps such a
    /// lease on file, counted on its worker, until the next issue at its
    /// file, and such a pin for the life of the store.
    pub fn sweep(&self, now: Instant) -> Swept {
        let store = &*self.store;
        Swept {
            pins: store.sweep_pins(now),
            leases: LeasesStore::expire(store, now).len(),
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
        let at = decider_of(request.facts.model, self.slots.len());
        let term = self.term.load(Ordering::Relaxed);
        let outcome = {
            let mut slot = self.slots[at].lock();
            let _acting = Decider::new(u8::try_from(at).unwrap_or(u8::MAX)).enter();
            let Slot { buffers, seat } = &mut *slot;
            let seat = seat.get_or_insert_with(|| self.store.at_work());
            // The thread works at the decider's file for this decision:
            // what was handed is applied as it takes up the work.
            seat.work();
            let order = || self.order(at, term, gateway, request, now, buffers);
            let decided = catch_unwind(AssertUnwindSafe(order));
            // It leaves the work: what was handed meanwhile is applied,
            // and what the door kept beside a full ring goes on behind it.
            seat.rest();
            if self.kept[at].waits.load(Ordering::SeqCst) != 0 {
                self.hand_kept(at);
            }
            decided.unwrap_or_else(|payload| {
                // A payload's own drop can panic: the engine's helper
                // frees it under a catch of its own. The next decision
                // reads nothing this one left.
                release(payload);
                *buffers = Buffers::default();
                self.faults.fetch_add(1, Ordering::Relaxed);
                Err(Refused::Faulted)
            })
        };
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
    /// said, until nothing waits. The caller holds the file's slot and has
    /// left its work: no thread is at work there, so each word is applied
    /// as it is handed and no hand waits.
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

    /// The order of one decision. No fence is read and no arrival budget
    /// is counted: those exist where a wire and an election do.
    fn order(
        &self,
        at: usize,
        term: u64,
        gateway: GatewayId,
        request: &Request<'_>,
        now: Instant,
        buffers: &mut Buffers<()>,
    ) -> Result<Decided, Refused> {
        let store = &*self.store;
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
        store.deciding(|world| {
            let decided = decide_in(world, &taken, buffers, &self.parts[at]);
            let (decision, booking) = decided.ok_or(Refused::PairSide)?;
            let lease = booking
                .map(|booking| issue(store, world, &taken, term, &decision, booking))
                .transpose()
                .map_err(|_| Refused::Unleased)?;
            Ok(Decided {
                decision,
                lease: lease.map(|lease| Leased {
                    lease,
                    due: taken.due(),
                }),
            })
        })
    }
}

#[cfg(test)]
mod tests {
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

    use super::{decider_of, Door, Refused, Request, Swept, TERM_MARK};
    use crate::{
        builtins::Random,
        catalog::Catalog,
        decider::Decider,
        decision::{Decision, Target, Targets},
        framework::{Picker, TieBand},
        lease::{Lease, LeaseId, Lifecycle, Outcome, ReleaseReason, Side, TransportClass},
        spec::{AppliedSpec, PolicySpec},
        store::{
            Ask, Dispatch, GatewayId, LeaseState, LeasesStore, MemoryStore, Said, SessionsStore,
            ViewBuffers, WorkerFacts, WorkerStatus, WorkersStore,
        },
        view::{Candidate, Constraints, RequestFacts, Role, WireModes, WorkerId},
    };

    const MODEL: &str = "model-a";
    const GATEWAY: GatewayId = GatewayId(11);

    /// A preset of this build as generation `generation`.
    fn preset(name: &str, generation: u64) -> AppliedSpec {
        let mut document = PolicySpec::preset(name).expect("a preset of this build");
        document.generation = generation;
        AppliedSpec::validate(&document).expect("the document validates")
    }

    /// A store of `deciders` files under the `random` preset, with three
    /// ready members that serve [`MODEL`].
    fn fleet(deciders: usize) -> Arc<MemoryStore> {
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
    fn picking<F>(store: &MemoryStore, generation: u64, name: &'static str, picker: F)
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
    fn request(id: &[u8], seed: u64) -> Request<'_> {
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
    fn counted(store: &MemoryStore) -> Vec<(WorkerId, u32)> {
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

    /// A dispatch to a pick's first target.
    fn dispatched() -> Said {
        let dispatch = Dispatch {
            target_index: 0,
            side: Side::Single,
            since_decision_micros: 250,
        };
        Said::Dispatched(dispatch, Instant::now())
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
    /// store applies it. Every call runs under the name of the decider the
    /// request's model names: the lease is on that decider's file.
    #[test]
    fn the_engine_hears_a_leases_issue_and_each_report_of_it() {
        // A count of files under which the model names a later file: a
        // thread that names no decider files in the first.
        let files = (2..=8).find(|&files| decider_of(MODEL, files) != 0);
        let files = files.expect("the model names a later file under some count");
        let store = fleet(files);
        let heard = Heard::default();
        let kept = Arc::clone(&heard);
        picking(&store, 2, "hears", move || {
            Box::new(Hears(Arc::clone(&kept)))
        });
        let door = Door::new(Arc::clone(&store));
        let (first, served) = leased(&door, b"r-1", 7);
        assert_eq!(report(&door, first, dispatched()), 1);
        assert_eq!(report(&door, first, completed(served)), 0);
        let (second, _) = leased(&door, b"r-2", 8);
        assert_eq!(report(&door, second, gone()), 0);
        let by = u8::try_from(decider_of(MODEL, files))
            .ok()
            .map(Decider::new);
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
    const PATIENCE: Duration = Duration::from_secs(20);

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
    struct Ends {
        scene: Arc<Scene>,
        stood: mpsc::Receiver<&'static str>,
        word: mpsc::Sender<()>,
    }

    impl Ends {
        /// Applies the [`Stops`] picker to `store` as generation 2.
        fn over(store: &MemoryStore) -> Self {
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
        fn stands(&self, place: &str) {
            assert_eq!(self.stood.recv_timeout(PATIENCE), Ok(place));
        }

        /// Sends the held request on from its stop.
        fn go(&self) {
            self.word.send(()).expect("the held request waits");
        }
    }

    /// One decision of `door` for the request under [`HELD`], on a thread
    /// of its own: its lease.
    fn held(door: &Arc<Door>) -> thread::JoinHandle<Option<LeaseId>> {
        let door = Arc::clone(door);
        thread::spawn(move || {
            let decided = door.decide(GATEWAY, &request(b"held", HELD), Instant::now());
            let leased = decided.ok().and_then(|decided| decided.lease);
            leased.map(|leased| leased.lease.id)
        })
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
}
