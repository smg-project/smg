//! The decision's order: the calls of one decision, written once.
//!
//! A decision is one load of the published rows
//! ([`MemoryStore::deciding`]), the engine's decision over them, the
//! lease filed at the issue and the engine's hearing of it, in that
//! order. The served controller runs it on its deciders' threads, behind
//! the fence and the budget that a wire and an election bring.
//!
//! A request reaches the order in the form it was asked in ([`Asked`]):
//! a frame as the served controller checked it.

use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use crate::{
    decision::{Decision, Shed, ShedReason, Trace},
    framework::{Plan, Scratch},
    lease::{Lease, LeaseId, LeaseKind, Lifecycle, Side},
    sessions::SessionVerdict,
    spec::AppliedSpec,
    store::{
        leases::DISPATCH_TTL, Ask, Built, GatewayId, LeaseRecord, LeaseRefused, LeaseState,
        MemoryStore, Named, Naming, Picked, SessionsStore, ViewBuffers, World,
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
