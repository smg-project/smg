//! The world a decision reads (design 4.1 and 5.5): every pool derived
//! from one membership, each a slice of whole candidate rows, and the build
//! of one decision's view from it.
//!
//! One invariant: **a decision reads one publication.** A world is
//! everything the store published together: a pool per model and role, its
//! rows sorted by worker id, the gateways' standing vetoes and the
//! generation they were published under ([`World::spec`]). One load hands a
//! decision all of it, so both sides of a pair decision, their load
//! records, the vetoes and the engine with its number are of one
//! publication, never a prefill pool of one and a decode pool of the next.
//!
//! **A membership write owes its build.** A registration, its refresh and
//! a patch write their row and leave one membership build owed. The store's
//! host pays it at the window's end for every write since the last, so a
//! storm of registrations builds once a window, and an asker pays it at
//! once ([`MemoryStore::publish_now`](crate::store::MemoryStore::publish_now)).
//! A removal and a replaced incarnation pay the build before they answer:
//! the worker that left is in no pool from there on, and a storm of
//! removals builds once a removal. So does a registration that takes its
//! worker off a model, for the pools it leaves.
//!
//! What a membership build derives (where each pool stands, each pool's
//! shape) is built once and shared by every world until the next membership
//! build. A world's epoch is the membership's at its build and is the one
//! thing to compare between two worlds' pools: a world's identity moves
//! with every publication. The pair classes are derived with the shapes
//! where a write moved a member or an offer, and once more where a
//! generation moves the pairing mode: a new world of the same epoch.
//! Between two membership builds a publication derives nothing: it is one
//! small object, a pointer per pool, and new rows for each pool a write was
//! folded into.
//!
//! **A pair class is a fact of the rows.** A model's pair classes come from
//! what its prefills and decodes report ([`PairClasses`]), and a membership
//! build holds them before any row is built: a worker's row carries its one
//! class, and the model's table says which classes pair. The table names
//! classes, never rows, so a pool with another gateway's host-local rows
//! left out reads the same table. A build that pays one worker's writes
//! derives the classes for that worker alone, in the models it left and
//! joined, from what the writer keeps of each model's derivation. A build
//! of writes that moved no member and no member's offer carries the last
//! build's classes and reads no member's facts for them. Two workers'
//! writes are one whole derivation.
//!
//! **The encode classes stand beside them.** Where a model's encoders state
//! an end of the encode handoff, the same write derives which encoders each
//! prefill's handoff reaches ([`EncodeClasses`]), by the rows of the two
//! pools. A write carries them in every model whose encoders and prefills
//! it leaves as they stood, a decode's write and another model's among
//! them, and derives them whole in the models it moves one of.
//!
//! A row in a pool is the candidate row itself. Its fixed fields are
//! written when the membership's first world is built. Its moving facts
//! (the load record with its sample instant, the two status bits) are
//! folded into the writer's own copy of the pool, and a publication copies
//! each pool that moved, whole. A published world never changes: a decision
//! copies one run of memory nobody is writing and sees each row as one
//! publication left it, through no lock.
//!
//! **Published rows are written again only when nothing can read them.**
//! A pool's copy goes over the rows the pool published before its last,
//! once the writer holds the last reference to them ([`Arc::get_mut`]): a
//! world lists each pool's rows by a counted pointer and every reader holds
//! a world, so that count is the proof, never a clock or a number of
//! windows. Rows a world still lists are left as they are: the pool
//! takes other rows the writer holds alone, or the publication allocates,
//! which the store counts
//! ([`MemoryStore::publication_allocations`](crate::store::MemoryStore::publication_allocations)).
//!
//! **The store publishes at its own rate, never at its writers'.** Every
//! deciding thread fetches every publication, so there is one world a
//! window for the whole store: the load records folded into any pool are
//! published together once per [`PUBLISH_WINDOW`], with the build the
//! membership writes left owed, and a status change, a veto or a leave at
//! once, through the same door, with whatever else waits. A load write
//! waits for an owed build, and else publishes by itself when the last
//! window is a window old; one that lands inside the window waits, and the
//! sink names the instant it is due
//! ([`LoadSink::publish_due`](crate::store::LoadSink::publish_due)).
//!
//! A decision's build is a copy and one pass ([`World::build`]): the rows,
//! less another gateway's host-local ones, then what is the decision's own
//! (the position, the report's age), then the asking gateway's standing
//! vetoes in the pool, reached by a walk among the few gateways that hold
//! one there and by a search among many, each published with its row's
//! place (masked by position in a pool every gateway copies whole, by id
//! in a pool that holds host-local rows), then the pinned worker, found by
//! its id in the slice the decision is made over: the sorted slice is its
//! own index.
//!
//! **A veto is published in the form the build reads.** The writer keeps
//! each gateway's list and publishes each pool's standing vetoes as one run
//! a gateway, by row. A veto's write builds its own gateway's run again in
//! the pools its worker has a row in and shares every other run with the
//! publication before. A membership build, which moves the rows, and a
//! window's publication that drops a veto past its end build every run
//! again. A veto leaves with its worker's row in the membership build's own
//! step, so no publication shows the row without it.

use std::{
    mem,
    sync::Arc,
    time::{Duration, Instant},
};

use rustc_hash::FxHashMap;

use crate::{
    affinity::WarmRule,
    counts::{Counts, KeptCounts, LentCounts, Slot},
    framework::{Engine, OverloadThresholds, PairPools, PairTable},
    lease::{Lifecycle, Signal},
    spec::{AppliedSpec, Listed, Pairing},
    store::{
        encoding::{EncodeClasses, EncodeHandoff, EncodeMember},
        index::{KvTables, Named, PoolIndex, Reading},
        loads::LoadSample,
        pairing::{known, Derivation, Member, PairClasses, Pool},
        workers::{Overload, Reachability, WorkerFacts, WorkerRow, WorkerStatus},
        GatewayId,
    },
    view::{
        fingerprint, AffinityFacts, Candidate, Engines, Health, OverloadRegime, Phase, PoolFacts,
        Role, WireModes, WorkerId,
    },
    EncodePool, EncoderFacts,
};

#[cfg(test)]
mod tests;

/// How long a folded load record may wait for its publication: the window
/// the gateway's load monitor already publishes pushed records by. A record
/// steers within one window of its receipt. The value is inherited, so a
/// comparison against that monitor differs by one thing fewer; a test hands
/// its own to [`MemoryStore::with_publish_window`](crate::store::MemoryStore::with_publish_window).
pub const PUBLISH_WINDOW: Duration = Duration::from_millis(20);

/// Microseconds from `origin` to `at`, saturating at both ends.
pub(super) fn micros(origin: Instant, at: Instant) -> u64 {
    u64::try_from(at.saturating_duration_since(origin).as_micros()).unwrap_or(u64::MAX)
}

/// Milliseconds from `at_micros` to `now_micros`, saturating at `u32::MAX`
/// and at zero; never wrapping.
fn age_ms(now_micros: u64, at_micros: u64) -> u32 {
    u32::try_from(now_micros.saturating_sub(at_micros) / 1_000).unwrap_or(u32::MAX)
}

/// What a decision brings to the build of its view.
#[derive(Clone, Copy, Debug)]
pub struct Ask {
    /// The gateway the decision is for: its vetoes mask a worker for it
    /// alone, and a host-local worker of another gateway is not its worker.
    pub gateway: GatewayId,
    /// The decision's instant on the host's clock.
    pub now: Instant,
    /// The worker the request's session key is pinned to, when the host
    /// found one.
    pub pinned: Option<WorkerId>,
    /// Whether the profile's stages declared the affinity column
    /// ([`Needs::affinity`](crate::Needs::affinity)): the build fills it
    /// then, and leaves it empty otherwise.
    pub affinity: bool,
}

/// What a decision brings to the naming of its rows ([`World::name`]).
#[derive(Clone, Copy, Debug)]
pub struct Naming<'a> {
    /// The gateway the decision is for: the view was built for it.
    pub gateway: GatewayId,
    /// The request's block hashes, in order from its first block
    /// ([`RequestFacts::block_hashes`](crate::RequestFacts::block_hashes)).
    pub block_hashes: &'a [u64],
    /// The decision's seed: where a list is cut, the rows kept are drawn
    /// from it, on streams of the naming's own.
    pub seed: u64,
    /// The most rows either list may hold: what the engine names for the
    /// decision ([`Engine::named_for`](crate::Engine::named_for)), which is
    /// 0 for a profile that keeps its column. A host names no rows there:
    /// a naming under 0 holds every row.
    pub cap: u32,
    /// The wires the view was narrowed by since its build, where the
    /// decision's gateway states them.
    pub wires: Option<WireModes>,
    /// The decision's instant on the store's clock, in microseconds, as
    /// its build read it ([`Built::now_micros`]): a warming row is named
    /// until its own end.
    pub now_micros: u64,
}

/// The buffers a host keeps across decisions: a build clears and fills
/// them, and allocates only while they grow.
#[derive(Debug, Default)]
pub struct ViewBuffers {
    /// The decision's candidate slice.
    pub candidates: Vec<Candidate>,
    /// The affinity column: one entry per candidate under a profile that
    /// declared it, empty otherwise.
    pub affinity: Vec<AffinityFacts>,
    /// The rows a veto of the asking gateway masks, by position, where the
    /// view is a publication's own rows ([`World::view`]); empty where the
    /// rows were written for the decision.
    pub stalled: Vec<u32>,
    /// What the decision read of its rows' requests in flight, where the
    /// view lent its rows with the leases' counts beside them
    /// ([`World::view`]): a row's first read is kept here, so a decision
    /// reads one count a row.
    pub kept: KeptCounts,
}

/// The buffers a host keeps across pair decisions: both sides' views and
/// the table that says which of their classes pair, all of one publication.
/// A pair build clears and fills them ([`World::build_pair`]).
///
/// What the store numbers for a table leaves it only with that table: the
/// rows' classes are numbered for the table built beside them, and by the
/// numbers alone they would pair wrongly. So nothing here is read apart:
/// [`PairBuffers::pools`] hands the engine's pair entry all of it.
#[derive(Debug, Default)]
pub struct PairBuffers {
    /// The prefill pool's view.
    prefill: ViewBuffers,
    /// The decode pool's view.
    decode: ViewBuffers,
    /// Where each prefill class's list ends in `listed`.
    ends: Vec<u32>,
    /// The decode classes each prefill class pairs with, list after list.
    listed: Vec<u32>,
}

impl PairBuffers {
    /// What the engine's pair entry reads: the two pools, their columns and
    /// the table of the publication the rows were built from.
    pub fn pools(&self) -> PairPools<'_> {
        PairPools {
            prefills: &self.prefill.candidates,
            decodes: &self.decode.candidates,
            prefill_affinity: &self.prefill.affinity,
            decode_affinity: &self.decode.affinity,
            table: PairTable::new(&self.ends, &self.listed),
        }
    }

    /// Both sides' views, the prefill's first: what a request's own
    /// gateway states of itself is taken from each ahead of the engine's
    /// pair entry.
    #[cfg(feature = "controller")]
    pub(crate) fn sides_mut(&mut self) -> [&mut ViewBuffers; 2] {
        [&mut self.prefill, &mut self.decode]
    }

    fn clear(&mut self) {
        for side in [&mut self.prefill, &mut self.decode] {
            side.candidates.clear();
            side.affinity.clear();
        }
        self.ends.clear();
        self.listed.clear();
    }
}

/// The buffers a host keeps across the decisions of requests that carry
/// encode items: a pair's buffers, the encode pool's view and the two
/// columns the encode side reads, all of one publication
/// ([`World::build_encode_pair`]).
///
/// An encode class is numbered for the publication that holds it, as a pair
/// class is for its table. So nothing here is read apart:
/// [`EncodePairBuffers::pools`] hands the engine's entry all of it.
#[derive(Debug, Default)]
pub struct EncodePairBuffers {
    pair: PairBuffers,
    /// The encode pool's view.
    encode: ViewBuffers,
    /// Each encoder's facts: one entry a row of the encode view.
    facts: Vec<EncoderFacts>,
    /// The encode class each row of the prefill view names.
    prefill_classes: Vec<u32>,
}

impl EncodePairBuffers {
    /// What the engine's encode-pair entry reads: the pair's pools with
    /// their table, and the encode pool with its columns, of the
    /// publication the rows were built from.
    pub fn pools(&self) -> (PairPools<'_>, EncodePool<'_>) {
        let encode = EncodePool {
            encoders: &self.encode.candidates,
            facts: &self.facts,
            prefill_classes: &self.prefill_classes,
            affinity: &self.encode.affinity,
        };
        (self.pair.pools(), encode)
    }
}

/// What a build read beside the rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Built {
    /// The rows written: the pool as the asking gateway reaches it.
    pub pool_size: u32,
    /// The store's epoch at the membership build the world's pools are of.
    pub membership_epoch: u64,
    /// The decision's instant in the rows' own clock: what the request's
    /// `now_micros` takes.
    pub now_micros: u64,
    /// The number of the generation the world was published under
    /// ([`World::spec`]): the rows and it are of one load.
    pub generation: u64,
    /// Which list of the pool was written, as [`PoolFacts::reach`] takes
    /// it: the asking gateway's id where the pool holds a host-local row,
    /// each gateway being handed a list of its own there, or where a veto
    /// of that gateway masks a row of its copy, and `0` for a pool every
    /// gateway dials whole and reads alike.
    pub reach: u64,
    /// The pool the rows were read from, by its place among its world's
    /// pools: what `World::booked` takes, and good for that world alone.
    pub(crate) pool: u32,
    /// How many rows of the pool hold no report: where it is 0 the pool
    /// facts say that every row holds one ([`PoolFacts::reported`]).
    /// Counted over the pool as it was published: the list of a gateway
    /// that reaches fewer rows may hold fewer, so the facts never say it of
    /// a list that holds such a row.
    pub unreported: u32,
}

impl Built {
    /// The pool facts of the decision, under the generation the build read
    /// with its rows. No ring, no overload regime and no zone capacity are
    /// derived yet.
    pub fn pool_facts(&self) -> PoolFacts<'static> {
        PoolFacts {
            pool_size: self.pool_size,
            membership_epoch: self.membership_epoch,
            ring_epoch: 0,
            regime: OverloadRegime::None,
            generation: self.generation,
            zones: &[],
            reach: self.reach,
            reported: self.unreported == 0,
            engines: Engines::default(),
        }
    }
}

/// One decision's view of a pool ([`World::view`]): its rows, and beside
/// them what is the decision's own where no row carries it.
#[derive(Clone, Copy, Debug)]
pub struct PoolView<'a> {
    /// The decision's candidate slice, in the pool's order: a row's index
    /// is its position. A publication's own rows where the pool was lent,
    /// the decision's written copy otherwise.
    pub candidates: &'a [Candidate],
    /// The rows a veto of the asking gateway masks at the decision's
    /// instant, by position, in order: what
    /// [`Constraints::stalled`](crate::Constraints::stalled) takes. Empty
    /// where the rows carry their stalled bits. The rows of it a hold of
    /// that gateway stands on are asked apart ([`World::held`]): a view
    /// under no veto has none, and carries nothing for them.
    pub stalled: &'a [u32],
    /// One more than the position of the pinned worker's row: what
    /// [`RequestFacts::pinned_at`](crate::RequestFacts::pinned_at) takes.
    /// Zero where no pinned worker is in the pool and where the rows carry
    /// their session bits.
    pub pinned_at: u32,
    /// The affinity column: one entry a row under a profile that declared
    /// it, empty otherwise.
    pub affinity: &'a [AffinityFacts],
    /// What the leases on file book on each row, by position: what
    /// [`RequestFacts::lent_in_flight`](crate::RequestFacts::lent_in_flight)
    /// takes. `None` where the rows carry their counts.
    pub in_flight: Option<LentCounts<'a>>,
    /// The pool's engines ([`Engines`]), where it holds one of several
    /// rows and the candidates stand at the pool's own positions: a pool
    /// that holds a host-local row hands none.
    pub engines: Engines<'a>,
    /// What the view read beside the rows.
    pub built: Built,
}

impl<'a> PoolView<'a> {
    /// The pool facts of the decision ([`Built::pool_facts`]) with the
    /// pool's engines beside them.
    pub fn pool_facts(&self) -> PoolFacts<'a> {
        let mut facts: PoolFacts<'a> = self.built.pool_facts();
        facts.engines = self.engines;
        facts
    }
}

/// One pool's shape: what a membership build derives of it and no
/// publication changes.
struct PoolShape {
    /// Each row's admission instant, in the rows' clock.
    admitted_micros: Box<[u64]>,
    /// Each row's worker's slot in the store's counts of the leases on
    /// file ([`Counts`]): the slot its member keeps across membership
    /// builds.
    booked: Box<[Slot]>,
    /// The host-local rows, by index, each with the gateway that holds its
    /// socket; empty for a pool every gateway dials whole.
    locals: Box<[(u32, GatewayId)]>,
    /// The pair classes of the pool's model, on its prefill pool where the
    /// model has a decode too: a pair build reads them by the one lookup
    /// that finds its two pools. Counted: a membership build hands them to
    /// the shape it derives where they stand, one of writes that moved no offer
    /// ([`Classes::Carry`]) and a written worker's derive that leaves them
    /// as they were ([`Classes::Written`]).
    pairs: Option<Arc<PairClasses>>,
    /// The encode classes of the pool's model, on its prefill pool where an
    /// encoder of the model states an end: a decision that carries encode
    /// items reads its chosen prefill's class beside the pool it chose in.
    /// Counted and carried as the pair classes are.
    encodes: Option<Arc<EncodeClasses>>,
    /// On an encode pool: what a decision reads of each encoder beside its
    /// row, in the rows' order, counted and carried as the classes are.
    /// `None` on every other pool.
    encoders: Option<Arc<[EncoderFacts]>>,
    /// The pool's engines, where one of them holds several rows.
    engines: PoolEngines,
}

/// One pool's engines: each engine's rows by position, the ranks of a
/// data-parallel engine together and in their ranks' order, a single-rank
/// worker alone. Empty, and unallocated, for a pool no engine of which
/// holds two rows: a fleet of single-rank workers lists none.
#[derive(Default)]
struct PoolEngines {
    /// The rows by position, engine after engine.
    rows: Box<[u32]>,
    /// Each engine's end in `rows`.
    ends: Box<[u32]>,
}

impl PoolShape {
    /// The pool's engines as a decision reads them, for rows that stand at
    /// the pool's own positions: a pool that holds a host-local row hands
    /// each gateway a list of its own, whose positions are not these, and
    /// no engines with it. `pool` is the pool's index, its number while the
    /// membership epoch stands.
    fn engines(&self, pool: u32) -> Engines<'_> {
        if self.locals.is_empty() {
            Engines::new(pool, &self.engines.rows, &self.engines.ends)
        } else {
            Engines::default()
        }
    }
}

/// The roles a model's pools are derived for: one slot each.
const ROLES: usize = 4;

/// A role's slot among a model's pools. A role added to the enum stops the
/// build at this match until it has an arm. The compiler checks the arm
/// alone: [`ROLES`] is raised by hand with the new slot, and a slot past it
/// panics at the first membership build that derives that role's pool.
const fn role_slot(role: Role) -> usize {
    match role {
        Role::Single => 0,
        Role::Prefill => 1,
        Role::Decode => 2,
        Role::Encode => 3,
    }
}

/// The phase a role's pool decides: what a binding is matched on.
const fn phase_of(role: Role) -> Phase {
    match role {
        Role::Single => Phase::Single,
        Role::Prefill => Phase::Prefill,
        Role::Decode => Phase::Decode,
        Role::Encode => Phase::Encode,
    }
}

/// What one membership build derives: where each pool stands in a world's
/// rows, and each pool's shape. Built once per membership build, and once
/// more where a generation moves the pairing mode, and shared by every
/// world published until the next.
#[derive(Default)]
struct Derived {
    /// The pools by model, then by role: each an index into a world's rows.
    /// A pool is every member serving one model in one role, whatever its
    /// status (a status change is a health signal and moves no position).
    by_model: FxHashMap<Box<str>, [Option<u32>; ROLES]>,
    pools: Box<[PoolShape]>,
    /// The members in no pool, by id, each with its slot in the counts:
    /// a member that offers no model. With the pools' rows they are every
    /// member the build read ([`World::unpooled_of`]).
    unpooled: Box<[(WorkerId, Slot)]>,
    /// The store's counts of the leases on file, which the pools' slots
    /// name: the one table of the store's life, lent with its shapes.
    counts: Arc<Counts>,
}

impl Derived {
    /// The shape of `model`'s pool of `role`.
    fn shape(&self, model: &str, role: Role) -> Option<&PoolShape> {
        let pool = self.by_model.get(model)?[role_slot(role)]?;
        self.pools.get(pool as usize)
    }

    /// The shape of `model`'s prefill pool: the model's classes stand on it.
    fn prefill(&self, model: &str) -> Option<&PoolShape> {
        self.shape(model, Role::Prefill)
    }

    /// The pair classes of `model`, where it has a prefill and a decode.
    fn pairs(&self, model: &str) -> Option<&Arc<PairClasses>> {
        self.prefill(model)?.pairs.as_ref()
    }

    /// The encode classes of `model`, where it has a prefill and an encoder
    /// that states an end.
    fn encodes(&self, model: &str) -> Option<&Arc<EncodeClasses>> {
        self.prefill(model)?.encodes.as_ref()
    }

    /// The shapes of `model`'s encode pool and of its prefill pool, each
    /// where the model has one, by one read of the model's pools.
    fn ends(&self, model: &str) -> Option<Ends<'_>> {
        let pools = self.by_model.get(model)?;
        let shape =
            |role: Role| pools[role_slot(role)].and_then(|pool| self.pools.get(pool as usize));
        Some(Ends {
            encode: shape(Role::Encode),
            prefill: shape(Role::Prefill),
        })
    }

    /// How many members `pool` holds: none where there is no pool.
    fn members(&self, pool: Option<u32>) -> usize {
        pool.and_then(|pool| self.pools.get(pool as usize))
            .map_or(0, |shape| shape.admitted_micros.len())
    }
}

/// The two pools of one model that the encode handoff reads, as one write
/// left them: the shape of its encode pool and of its prefill pool, each
/// where the model has one.
#[derive(Clone, Copy)]
struct Ends<'a> {
    encode: Option<&'a PoolShape>,
    prefill: Option<&'a PoolShape>,
}

impl Ends<'_> {
    /// Whether a write that asks `classes` left both pools as they stand
    /// here, `sides` being the model's members after it: no member of
    /// either joined or left, and the written worker is in neither. One
    /// worker's write moves that worker's rows alone, so two sides of these
    /// pools' lengths that do not hold the worker are these pools' sides,
    /// row for row.
    fn stand(&self, classes: Classes, sides: &[Vec<Entry<'_>>; ROLES]) -> bool {
        let joined = match classes {
            Classes::Carry => return true,
            Classes::Derive => return false,
            Classes::Written { joined, .. } => joined,
        };
        [(Role::Encode, self.encode), (Role::Prefill, self.prefill)]
            .into_iter()
            .all(|(role, shape)| {
                let entries = &sides[role_slot(role)];
                let holds = |id: WorkerId| {
                    entries
                        .binary_search_by_key(&id, |entry| entry.row.id)
                        .is_ok()
                };
                entries.len() == shape.map_or(0, |shape| shape.admitted_micros.len())
                    && !joined.is_some_and(holds)
            })
    }
}

/// One standing veto as its gateway's run in a pool carries it: the row it
/// masks, by index and by id, and its ends. The index finds the row in a
/// pool every gateway copies whole, the id in a pool that holds host-local
/// rows, whichever gateway asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PoolVeto {
    row: u32,
    id: WorkerId,
    ends: VetoEnds,
}

/// A veto's two ends as the writer keeps them, with no kind. The veto masks
/// its row until `until`. Its hold stands until `hold`, never past the
/// veto's own end: while it does, a decision of the gateway reads the row
/// as behind an open breaker, a veto no rule lifts, and from then to the
/// veto's end as doubted
/// ([`HealthSink::gateway_veto_noted`](crate::store::HealthSink::gateway_veto_noted)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VetoEnds {
    until: Instant,
    hold: Option<Instant>,
}

impl VetoEnds {
    /// The ends as a door hands them: a hold past the veto's end is cut to
    /// it.
    fn new(until: Instant, hold: Option<Instant>) -> Self {
        Self {
            until,
            hold: hold.map(|hold| hold.min(until)),
        }
    }

    /// Whether the hold stands at `now`: the veto then stands too.
    fn held_at(self, now: Instant) -> bool {
        self.hold.is_some_and(|hold| hold > now)
    }
}

/// One gateway's standing vetoes in one pool, sorted by row: never empty.
#[derive(Clone, Debug)]
struct GatewayRun {
    gateway: GatewayId,
    vetoes: Arc<[PoolVeto]>,
    /// Whether a veto of the run has a hold's end, passed or not: a view
    /// that lends its rows reads the holds of such a run alone.
    holds: bool,
}

impl GatewayRun {
    fn new(gateway: GatewayId, vetoes: Arc<[PoolVeto]>) -> Self {
        let holds = vetoes.iter().any(|veto| veto.ends.hold.is_some());
        Self {
            gateway,
            vetoes,
            holds,
        }
    }
}

/// One pool's standing vetoes: a run for each gateway that holds one in
/// the pool, sorted by gateway. A build finds the asking gateway's run
/// among the runs ([`run_of`]) and reads no other gateway's veto; a veto's
/// write builds its own gateway's run again and shares every other.
type PoolVetoes = Arc<[GatewayRun]>;

/// The most runs a pool holds where a build still walks to its own, past
/// half of them at the most; above it the run is found by search. A few
/// gateways that each hold a veto in a pool is the shape a fleet has, and
/// there a walk reads the runs in order while a search waits for each of
/// its steps. The count is where the walk's farthest place, the middle,
/// met the search when the two were timed alone.
const WALKED_RUNS: usize = 160;

/// The runs a walk passes in one step while every one of them is short of
/// the gateway it seeks. A walk run by run is a loop of a few instructions
/// whose time follows where the build's text puts it; by strides the walk
/// is the shorter wherever it sits.
const STRIDE: usize = 4;

/// `gateway`'s run among a pool's runs, sorted by gateway. While the runs
/// are at most [`WALKED_RUNS`] the two ends are asked first, which is the
/// whole answer where one gateway alone holds a veto, and a gateway
/// between them is walked to from the nearer end, over the runs between
/// the two ends alone: from the front to the first gateway not under it,
/// or from the back to the first not above it, [`STRIDE`] runs a step
/// while a whole stride is short of it. Above that count the run is found
/// by one search.
fn run_of(runs: &[GatewayRun], gateway: GatewayId) -> Option<&GatewayRun> {
    if runs.len() > WALKED_RUNS {
        let at = runs
            .binary_search_by_key(&gateway, |run| run.gateway)
            .ok()?;
        return runs.get(at);
    }
    let (first, rest) = runs.split_first()?;
    if gateway <= first.gateway {
        return (gateway == first.gateway).then_some(first);
    }
    let (last, between) = rest.split_last()?;
    if gateway >= last.gateway {
        return (gateway == last.gateway).then_some(last);
    }
    // The middle run says which half holds the gateway's. From either end
    // the walk passes a stride while the stride's far run is short of the
    // gateway, then goes run by run.
    let middle = between.get(between.len() / 2)?;
    let run = if gateway < middle.gateway {
        let mut from = 0;
        while between
            .get(from + STRIDE - 1)
            .is_some_and(|run| run.gateway < gateway)
        {
            from += STRIDE;
        }
        between
            .get(from..)?
            .iter()
            .find(|run| run.gateway >= gateway)
    } else {
        let mut to = between.len();
        while to >= STRIDE
            && between
                .get(to - STRIDE)
                .is_some_and(|run| run.gateway > gateway)
        {
            to -= STRIDE;
        }
        between
            .get(..to)?
            .iter()
            .rev()
            .find(|run| run.gateway <= gateway)
    };
    run.filter(|run| run.gateway == gateway)
}

/// A pool's runs with `gateway`'s veto on the row `row` of worker `id` set
/// to `ends`, or cleared: that gateway's run built again, each other run
/// shared.
fn with_veto(
    runs: &PoolVetoes,
    gateway: GatewayId,
    row: u32,
    id: WorkerId,
    ends: Option<VetoEnds>,
) -> PoolVetoes {
    let found = runs.binary_search_by_key(&gateway, |run| run.gateway);
    let own: &[PoolVeto] = match found {
        Ok(at) => runs.get(at).map(|run| &*run.vetoes).unwrap_or_default(),
        Err(_) => &[],
    };
    // Each arm is one pass of a known length: one allocation, the run's
    // own, and no entry of another gateway is read.
    let own: Arc<[PoolVeto]> = match (own.binary_search_by_key(&row, |veto| veto.row), ends) {
        (Ok(at), Some(ends)) => own
            .iter()
            .enumerate()
            .map(|(n, standing)| PoolVeto {
                ends: if n == at { ends } else { standing.ends },
                ..*standing
            })
            .collect(),
        (Ok(at), None) => {
            let (before, after) = own.split_at(at);
            let after = after.get(1..).unwrap_or_default();
            before.iter().chain(after).copied().collect()
        }
        (Err(at), Some(ends)) => {
            let (before, after) = own.split_at(at);
            before
                .iter()
                .copied()
                .chain(std::iter::once(PoolVeto { row, id, ends }))
                .chain(after.iter().copied())
                .collect()
        }
        // No veto of this gateway on the row: nothing to clear.
        (Err(_), None) => return Arc::clone(runs),
    };
    let (at, held) = match found {
        Ok(at) => (at, 1),
        Err(at) => (at, 0),
    };
    let (before, after) = runs.split_at(at);
    let after = after.get(held..).unwrap_or_default();
    let own = (!own.is_empty()).then(|| GatewayRun::new(gateway, own));
    before
        .iter()
        .cloned()
        .chain(own)
        .chain(after.iter().cloned())
        .collect()
}

/// The gateways' vetoes as the writer keeps them: per gateway its vetoed
/// workers by id, each with its ends. Replaced whole at each of the
/// four writes that change them, all rare: a veto's write, a gateway's
/// leaving that takes its whole list, a membership write that takes a
/// departed worker's vetoes with its row, and a window's publication that
/// drops a veto past its end. A world carries what is
/// derived from them: each pool's runs by gateway ([`PoolVetoes`]).
#[derive(Debug, Default)]
struct Vetoes {
    by_gateway: FxHashMap<GatewayId, Arc<[(WorkerId, VetoEnds)]>>,
}

impl Vetoes {
    /// The vetoes with `gateway`'s veto on `id` set to `ends`, or cleared;
    /// `None` when that changes nothing.
    fn with(&self, gateway: GatewayId, id: WorkerId, ends: Option<VetoEnds>) -> Option<Self> {
        let mut list: Vec<(WorkerId, VetoEnds)> = self
            .by_gateway
            .get(&gateway)
            .map(|list| list.to_vec())
            .unwrap_or_default();
        match (list.binary_search_by_key(&id, |&(worker, _)| worker), ends) {
            (Ok(at), Some(ends)) if list[at].1 == ends => return None,
            (Ok(at), Some(ends)) => list[at].1 = ends,
            (Ok(at), None) => {
                list.remove(at);
            }
            (Err(at), Some(ends)) => list.insert(at, (id, ends)),
            (Err(_), None) => return None,
        }
        let mut by_gateway = self.by_gateway.clone();
        if list.is_empty() {
            by_gateway.remove(&gateway);
        } else {
            by_gateway.insert(gateway, list.into());
        }
        Some(Self { by_gateway })
    }

    /// The vetoes still standing at `at`, when some veto ended at or before
    /// it: what the clock has ended leaves by the store's own clock.
    fn standing_at(&self, at: Instant) -> Option<Self> {
        let stands = |&(_, ends): &(WorkerId, VetoEnds)| ends.until > at;
        if self.by_gateway.values().all(|list| list.iter().all(stands)) {
            return None;
        }
        let mut by_gateway = FxHashMap::default();
        for (gateway, list) in &self.by_gateway {
            let kept: Vec<(WorkerId, VetoEnds)> = list.iter().copied().filter(stands).collect();
            if !kept.is_empty() {
                by_gateway.insert(*gateway, kept.into());
            }
        }
        Some(Self { by_gateway })
    }

    /// The vetoes on the workers in `members`, a list sorted by id, when
    /// some veto names a worker outside it: a worker's vetoes leave with
    /// its row.
    fn of_members(&self, members: &[WorkerId]) -> Option<Self> {
        /// The entries of `list` on a worker in `members`. Both rise by
        /// worker id: one walk over each joins them.
        fn kept<'a>(
            list: &'a [(WorkerId, VetoEnds)],
            members: &'a [WorkerId],
        ) -> impl Iterator<Item = (WorkerId, VetoEnds)> + 'a {
            let mut at = 0;
            list.iter().copied().filter(move |&(id, _)| {
                at = seek(members, at, id, |&member| member);
                members.get(at) == Some(&id)
            })
        }
        let whole = |list: &Arc<[(WorkerId, VetoEnds)]>| kept(list, members).count() == list.len();
        if self.by_gateway.values().all(whole) {
            return None;
        }
        let mut by_gateway = FxHashMap::default();
        for (gateway, list) in &self.by_gateway {
            if whole(list) {
                by_gateway.insert(*gateway, Arc::clone(list));
                continue;
            }
            let kept: Vec<(WorkerId, VetoEnds)> = kept(list, members).collect();
            if !kept.is_empty() {
                by_gateway.insert(*gateway, kept.into());
            }
        }
        Some(Self { by_gateway })
    }
}

/// Why a generation was not applied: its number is not above the standing
/// one. The store's number only rises.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("generation {offered} is not above the standing generation {standing}")]
pub struct StaleGeneration {
    /// The number of the generation the store holds.
    pub standing: u64,
    /// The number of the generation that was offered.
    pub offered: u64,
}

/// How many members of the standing membership name an engine family or a
/// zone that the standing generation's list, which holds entries, does not
/// hold: each reads position 0 in that field. Counted wherever the members
/// are read under the lists: at a membership build and at an apply that
/// changes them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OutsideLists {
    /// Members outside `view.engines`.
    pub engines: u32,
    /// Members outside `view.zones`.
    pub zones: u32,
}

/// Everything a decision reads, as one publication left it. A decision
/// borrows it for its build
/// ([`MemoryStore::build_view`](crate::store::MemoryStore::build_view)), and
/// both sides of a pair decision are built from the same one
/// ([`MemoryStore::build_pair`](crate::store::MemoryStore::build_pair)). A
/// world never changes once published.
pub struct World {
    epoch: u64,
    origin: Instant,
    derived: Arc<Derived>,
    /// Each pool's rows as this publication left them, sorted by worker id,
    /// at the pool's index.
    rows: Box<[Arc<[Candidate]>]>,
    /// The standing vetoes by pool, each pool's by gateway: empty while no
    /// veto stands in the store.
    vetoes: Arc<[PoolVetoes]>,
    /// How many rows of each pool hold no report, at the pool's index
    /// ([`Working::unreported`]): kept by the writes and shown with the rows
    /// a publication shows, so a decision reads one number. One list for
    /// every world until a count moves.
    unreported: Arc<[u32]>,
    /// The generation this publication was made under: its engine and its
    /// number, read with the rows in one load.
    spec: Arc<AppliedSpec>,
    /// The served decider reads this with the world, before following
    /// the pointer to the generation.
    #[cfg(feature = "controller")]
    samples_decisions: bool,
    /// Each pool's standing in its model's index, at the pool's index.
    indexes: Arc<[Option<Arc<PoolIndex>>]>,
}

impl std::fmt::Debug for World {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("World")
            .field("epoch", &self.epoch)
            .field("pools", &self.rows.len())
            .finish_non_exhaustive()
    }
}

impl World {
    /// The store's epoch at the membership write this world's pools were
    /// derived by: the one thing to compare between two worlds' pools. A
    /// status change moves the store's epoch and no world's, and a
    /// publication between two membership writes is a new world of the same
    /// epoch. The pair classes move once more, at a generation of another
    /// pairing mode ([`World::pairs`]).
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The generation this world was published under: the engine a decision
    /// over these rows runs, and its number. The store starts under the
    /// unbound generation ([`AppliedSpec::unbound`]), and a reader that saw
    /// a generation never sees a lower one: the store's one entry
    /// ([`MemoryStore::apply`](crate::store::MemoryStore::apply)) admits a
    /// number above the standing one alone.
    ///
    /// The store's deciding door
    /// ([`MemoryStore::deciding`](crate::store::MemoryStore::deciding))
    /// hands the engine under the borrow its rows are built from, never an
    /// owned world, whose shared count a deciding thread would write.
    pub fn spec(&self) -> &AppliedSpec {
        &self.spec
    }

    #[cfg(feature = "controller")]
    pub(crate) fn samples_decisions(&self) -> bool {
        self.samples_decisions
    }

    /// Tells this world's engine each rank's report, where a stage of it
    /// keeps a number of them ([`Lifecycle::keeps_loads`]): what the store
    /// does before it shows a world, so a decision over the world finds
    /// the numbers of its own rows. An engine with no such stage says so
    /// by one word made at its build, and the tell is not entered: by
    /// count the call alone read 29 instructions a publication.
    pub(crate) fn tell_kept_loads(&self) {
        let engine = self.spec.engine();
        if engine.keeps_loads() {
            self.tell_loads(engine);
        }
    }

    /// Tells each stage of `engine` that keeps a number of the reports
    /// each rank's report as this world shows it to a decision, with the
    /// rank's slot and the report's sample instant ([`Signal::Load`]). A
    /// stage that keeps none is sent nothing. A member of several pools is
    /// told once a pool, the same word.
    pub fn tell_loads(&self, engine: &Engine) {
        engine.tell_loads(|tell| {
            for (rows, shape) in self.rows.iter().zip(self.derived.pools.iter()) {
                for (row, &slot) in rows.iter().zip(shape.booked.iter()) {
                    if let Some(load) = row.load {
                        tell(&Signal::Load {
                            worker: row.id,
                            dp_rank: row.dp_rank,
                            row: load,
                            slot,
                            sampled_at_micros: row.load_sampled_at_micros,
                        });
                    }
                }
            }
        });
    }

    /// `at` in the rows' clock: microseconds since the store's origin.
    pub fn micros(&self, at: Instant) -> u64 {
        micros(self.origin, at)
    }

    /// How many members serve `model` in `role`, every status and every
    /// gateway's host-local rows counted; `None` when none does.
    pub fn members(&self, model: &str, role: Role) -> Option<usize> {
        self.pool(model, role).map(|(rows, _)| rows.len())
    }

    /// The pair classes of `model` as this world's pools were derived, with
    /// the table a pair decision reads beside the two pools
    /// ([`PairClasses::table`]); `None` when the model has no prefill or no
    /// decode. A membership write derives them, and so does a generation of
    /// another pairing mode at the same epoch: the epoch alone does not key
    /// the classes a reader keeps across worlds.
    pub fn pairs(&self, model: &str) -> Option<&PairClasses> {
        self.derived.pairs(model).map(|classes| &**classes)
    }

    /// The encode classes of `model` as this world's pools were derived: the
    /// class each row of its prefill pool names and each
    /// class's encoders by their rows in its encode pool, another gateway's
    /// host-local rows counted ([`EncodeClasses`]). A class's number is read
    /// under this world's borrow and never kept. `None` when the model has
    /// no prefill or no encoder that states an end. A membership write that
    /// moves one of the model's encoders or prefills derives them, and so
    /// does a generation of another pairing mode at the same epoch, as it
    /// does the pair classes ([`World::pairs`]).
    pub fn encodes(&self, model: &str) -> Option<&EncodeClasses> {
        self.derived.encodes(model).map(|classes| &**classes)
    }

    fn pool(&self, model: &str, role: Role) -> Option<(&[Candidate], &PoolShape)> {
        let index = self.index(model, role)?;
        Some((self.rows.get(index)?, self.derived.pools.get(index)?))
    }

    /// How many rows of the pool at `index` hold no report, as this
    /// publication showed them. A pool the counts do not reach reads as
    /// holding one: the facts then say nothing of its reports.
    fn unreported_at(&self, index: usize) -> u32 {
        self.unreported.get(index).copied().unwrap_or(u32::MAX)
    }

    fn index(&self, model: &str, role: Role) -> Option<usize> {
        Some(self.derived.by_model.get(model)?[role_slot(role)]? as usize)
    }

    /// The slot of worker `id`'s count of the leases on file, as the pool
    /// `built` read names it beside the worker's row: what the lease of a
    /// decision over that build books. `None` for a worker the pool does
    /// not list. `built` is a build of this world.
    pub(crate) fn booked(&self, built: &Built, id: WorkerId) -> Option<Slot> {
        let pool = built.pool as usize;
        let (rows, shape) = (self.rows.get(pool)?, self.derived.pools.get(pool)?);
        let row = rows.binary_search_by_key(&id, |row| row.id).ok()?;
        shape.booked.get(row).copied()
    }

    /// The slot of worker `id`'s count wherever this world lists it: a
    /// member's rows name one. `None` for a worker in no pool: a member
    /// there is read apart ([`World::unpooled_of`]). It searches every
    /// pool, for a lease written outside a decision.
    pub(crate) fn booked_of(&self, id: WorkerId) -> Option<Slot> {
        let mut pools = self.rows.iter().zip(self.derived.pools.iter());
        pools.find_map(|(rows, shape)| {
            let row = rows.binary_search_by_key(&id, |row| row.id).ok()?;
            shape.booked.get(row).copied()
        })
    }

    /// The slot of worker `id`'s count where the worker is a member in no
    /// pool of this world: one that offers no model. `None` for a member
    /// a pool lists, and for a worker that was no member when this
    /// world's membership was built. A lease's door reads it behind the
    /// pools and the table, off the path of a listed target.
    pub(crate) fn unpooled_of(&self, id: WorkerId) -> Option<Slot> {
        let unpooled = &self.derived.unpooled;
        let at = unpooled.binary_search_by_key(&id, |&(of, _)| of).ok()?;
        unpooled.get(at).map(|&(_, slot)| slot)
    }

    /// Writes over `rows`, the candidates `built` left for `ask` before
    /// anything narrowed them, what the leases on file book on each row's
    /// worker at this call: one load a row, in the build's own walk of the
    /// rows the asking gateway reaches. For rows that are written: a
    /// decision over lent rows reads the table beside them
    /// ([`World::view`]). `built` is a build of this world.
    pub fn count_in_flight(&self, built: &Built, ask: &Ask, rows: &mut [Candidate]) {
        let Some(shape) = self.derived.pools.get(built.pool as usize) else {
            return;
        };
        // The rows as the build left them: a narrowing before this call
        // would lay a count on another worker's row.
        debug_assert_eq!(rows.len(), built.pool_size as usize);
        let counts = &self.derived.counts;
        let mut at = 0;
        for (from, to) in Segments::of(&shape.locals, ask.gateway, shape.booked.len()) {
            let booked = shape.booked.get(from..to).unwrap_or_default();
            let reached = rows.get_mut(at..).unwrap_or_default();
            for (row, in_flight) in reached.iter_mut().zip(counts.of(booked)) {
                row.in_flight = in_flight;
            }
            at += booked.len();
        }
    }

    /// Writes over both sides' rows of `out`, as [`World::build_pair`] left
    /// them for `ask`, what the leases on file book on each row's worker:
    /// [`World::count_in_flight`], a side at a time. `built` is that
    /// build's pair.
    pub fn count_pair_in_flight(&self, built: &(Built, Built), ask: &Ask, out: &mut PairBuffers) {
        self.count_in_flight(&built.0, ask, &mut out.prefill.candidates);
        self.count_in_flight(&built.1, ask, &mut out.decode.candidates);
    }

    /// The engines of the pool `built` read ([`Engines`]), for a decision
    /// over the rows a build wrote ([`World::build`]) as the build left
    /// them: a narrowing moves a row from the position the list names.
    /// None for a pool that holds a host-local row, and for a pool no
    /// engine of which holds two rows. `built` is a build of this world.
    pub fn engines(&self, built: &Built) -> Engines<'_> {
        self.derived
            .pools
            .get(built.pool as usize)
            .map_or_else(Engines::default, |shape| shape.engines(built.pool))
    }

    /// Build one decision's view of the pool serving `model` in `role` into
    /// `out`; `None`, with the buffers cleared, when no member serves it.
    ///
    /// The candidates are the pool's rows in the pool's order, by worker
    /// id, less the host-local rows of another gateway: the asking gateway
    /// holds no socket to those, so its pool is the pool without them.
    /// Each row is as this publication left it, its engine's overload
    /// verdict with the rest ([`LoadSink::report`](crate::store::LoadSink::report)),
    /// with what is the decision's own written over it: its position in the slice, its
    /// report's age at `ask.now`, the bit of a veto of the asking gateway
    /// that is in this publication and stands at that instant, and the
    /// session bit on the pinned worker. A veto writes the stalled bit,
    /// and the breaker bit beside it while its hold stands: the store
    /// keeps no kind, and the hold says which veto no rule lifts. A veto
    /// past its end masks nothing and costs the build one
    /// compare until a window's publication drops it; a store that
    /// publishes no window keeps it. A window's publication drops every
    /// veto ended by its own instant, so a decision dated before that
    /// instant finds a veto that still stood at its date already gone: a
    /// decision reads one publication, and its date restores nothing the
    /// publication dropped. The health sink's own answer keeps its compare.
    /// Under `ask.affinity` the column holds one entry per candidate with
    /// the worker's age since admission.
    ///
    /// A row's in-flight count is not written here: every decision moves
    /// the count of the leases on file, so no publication carries it, and
    /// a decision that reads it has it written over these rows
    /// ([`World::count_in_flight`]).
    ///
    /// What no fact of the store feeds yet reads as the row's default: the
    /// two session counts, the two admission bits, the four positions into
    /// declared lists, the encode bit, and the column's four index facts.
    /// The breaker bit reads a hold of the asking gateway and nothing else
    /// yet. The
    /// healthy bit is the status alone: the health sink's precedence rule
    /// (a gateway's completed dispatches against the prober) is not
    /// applied here yet.
    pub fn build(
        &self,
        model: &str,
        role: Role,
        ask: &Ask,
        out: &mut ViewBuffers,
    ) -> Option<Built> {
        let Some(index) = self.index(model, role) else {
            out.candidates.clear();
            out.affinity.clear();
            out.stalled.clear();
            return None;
        };
        self.build_at(index, ask, out)
    }

    /// [`World::build`] past the model's lookup: the view of the pool at
    /// `index` of this world's rows, into buffers cleared first. Inlined
    /// into both doors: a single build stays one call, and a pair build
    /// runs its two sides under its one lookup with no call between them.
    #[inline(always)]
    fn build_at(&self, index: usize, ask: &Ask, out: &mut ViewBuffers) -> Option<Built> {
        out.candidates.clear();
        out.affinity.clear();
        out.stalled.clear();
        let (rows, shape) = (self.rows.get(index)?, self.derived.pools.get(index)?);
        Some(self.written(index, rows, shape, ask, out))
    }

    /// The pool at `index` copied into `out` and written for the decision:
    /// [`World::build`] past its lookup, into buffers its caller cleared.
    #[inline(always)]
    fn written(
        &self,
        index: usize,
        rows: &[Candidate],
        shape: &PoolShape,
        ask: &Ask,
        out: &mut ViewBuffers,
    ) -> Built {
        let now_micros = micros(self.origin, ask.now);
        for (from, to) in Segments::of(&shape.locals, ask.gateway, rows.len()) {
            out.candidates.extend_from_slice(&rows[from..to]);
        }
        for (position, row) in out.candidates.iter_mut().enumerate() {
            row.position = position as u32;
            let sampled_at = row.load_sampled_at_micros;
            if let Some(load) = &mut row.load {
                load.report_age_ms = age_ms(now_micros, sampled_at);
            }
        }
        if let Some(row) = ask.pinned.and_then(|id| row_of_id(&mut out.candidates, id)) {
            row.session_hit = true;
        }
        if ask.affinity {
            let admitted = &shape.admitted_micros;
            for (from, to) in Segments::of(&shape.locals, ask.gateway, admitted.len()) {
                out.affinity
                    .extend(admitted[from..to].iter().map(|&admitted| AffinityFacts {
                        admitted_age_ms: age_ms(now_micros, admitted),
                        ..AffinityFacts::default()
                    }));
            }
        }
        // The asking gateway's run among the gateways that hold a veto in
        // this pool: no other gateway's veto is read.
        let own = self
            .vetoes
            .get(index)
            .and_then(|runs| run_of(runs, ask.gateway));
        // Whether a veto of the asking gateway masks a row of its list:
        // asked last, so no loop of the build holds the answer.
        let masked =
            own.is_some_and(|run| mask(run, ask.now, shape.locals.is_empty(), &mut out.candidates));
        Built {
            pool: index as u32,
            pool_size: out.candidates.len() as u32,
            membership_epoch: self.epoch,
            now_micros,
            generation: self.spec.generation(),
            // A copy a veto masks reads apart from the other gateways': it
            // is the asking gateway's own list, as one short of another's
            // host-local rows is.
            reach: if shape.locals.is_empty() && !masked {
                0
            } else {
                ask.gateway.0
            },
            unreported: self.unreported_at(index),
        }
    }

    /// One decision's view of the pool serving `model` in `role`; `None`,
    /// with the buffers cleared, when no member serves it.
    ///
    /// Where every gateway reaches the pool whole, the view is the
    /// publication's own rows, lent: nothing is written into a row, and
    /// what [`World::build`] writes there stands beside the rows. A row's
    /// position was published with it. The vetoes of the asking gateway
    /// that stand at `ask.now` are handed as positions, in order, read from
    /// its own run alone and by the build's rule: a veto past its end masks
    /// nothing. The rows of them a hold stands on are handed once more,
    /// apart: where the build writes a row both bits, the view names the
    /// row twice. The pinned worker's row is handed as one more than its
    /// position. Under `ask.affinity` the column is filled as the build
    /// fills it. A row's report age stays what the report carried when it
    /// was recorded. The report age limit uses the decision's instant less
    /// the row's sample instant on either form of the row.
    ///
    /// The rows are copied and written ([`World::build`]) where they cannot
    /// be lent: in a pool that holds host-local rows, where the asking
    /// gateway's positions are its own, and for a decision that asks for
    /// `written` rows. A profile that shows its rows to a plugin
    /// ([`Profile::scoped`](crate::Profile::scoped)) asks for them: a
    /// plugin's view carries each row's bits and ages as the row's fields.
    ///
    /// What the leases on file book on each row goes into no publication:
    /// the table is lent beside lent rows ([`PoolView::in_flight`]) and
    /// its counts are written over rows that are written
    /// ([`World::count_in_flight`]). Both forms come back as one shape, so
    /// a caller hands the mask, the pin and the counts on whichever rows
    /// it was given.
    pub fn view<'a>(
        &'a self,
        model: &str,
        role: Role,
        ask: &Ask,
        written: bool,
        out: &'a mut ViewBuffers,
    ) -> Option<PoolView<'a>> {
        out.candidates.clear();
        out.affinity.clear();
        out.stalled.clear();
        let index = self.index(model, role)?;
        let (rows, shape) = (self.rows.get(index)?, self.derived.pools.get(index)?);
        if written || !shape.locals.is_empty() {
            let built = self.written(index, rows, shape, ask, out);
            self.count_in_flight(&built, ask, &mut out.candidates);
            return Some(PoolView {
                candidates: &out.candidates,
                stalled: &[],
                pinned_at: 0,
                affinity: &out.affinity,
                in_flight: None,
                engines: shape.engines(index as u32),
                built,
            });
        }
        let now_micros = micros(self.origin, ask.now);
        // The asking gateway's run alone, by the build's rule.
        let own = self
            .vetoes
            .get(index)
            .and_then(|runs| run_of(runs, ask.gateway));
        if let Some(run) = own {
            let standing = run.vetoes.iter().filter(|veto| veto.ends.until > ask.now);
            out.stalled.extend(standing.map(|veto| veto.row));
        }
        // The search of the build's own pin, over the same rows.
        let pinned_at = ask
            .pinned
            .and_then(|id| rows.binary_search_by_key(&id, |row| row.id).ok())
            .map_or(0, |at| at as u32 + 1);
        if ask.affinity {
            let admitted = shape.admitted_micros.iter();
            out.affinity.extend(admitted.map(|&admitted| AffinityFacts {
                admitted_age_ms: age_ms(now_micros, admitted),
                ..AffinityFacts::default()
            }));
        }
        // A view a veto masks is the asking gateway's own list, as a copy
        // a veto masks is.
        let reach = if out.stalled.is_empty() {
            0
        } else {
            ask.gateway.0
        };
        Some(PoolView {
            candidates: rows,
            stalled: &out.stalled,
            pinned_at,
            affinity: &out.affinity,
            in_flight: Some(self.derived.counts.lend(&shape.booked, &mut out.kept)),
            engines: shape.engines(index as u32),
            built: Built {
                pool: index as u32,
                pool_size: rows.len() as u32,
                membership_epoch: self.epoch,
                now_micros,
                generation: self.spec.generation(),
                reach,
                unreported: self.unreported_at(index),
            },
        })
    }

    /// The rows of a lent view's mask ([`PoolView::stalled`]) a hold of the
    /// asking gateway stands on at the decision's instant, by position, in
    /// order, into `rows`: what
    /// [`Constraints::held`](crate::Constraints::held) takes beside the
    /// mask. Asked of the publication the view was read from, with the
    /// view's own `ask`. A hold stands no longer than its veto, so a held
    /// row is a row of the mask.
    ///
    /// A view under no veto has no held row and is answered here at one
    /// test: its pool's vetoes are not read again. Rows written for a
    /// decision carry the breaker bit beside the stalled one where a hold
    /// stands, and their view's mask is empty.
    #[inline]
    pub fn held<'a>(&self, view: &PoolView<'_>, ask: &Ask, rows: &'a mut Vec<u32>) -> &'a [u32] {
        if view.stalled.is_empty() {
            return &[];
        }
        self.held_rows(view.built.pool, ask, rows)
    }

    /// The walk behind [`Self::held`], out of line: the asking gateway's
    /// run in pool `pool`, and the row of each of its vetoes whose hold
    /// stands at the ask's instant. A run with no hold is one test.
    #[inline(never)]
    fn held_rows<'a>(&self, pool: u32, ask: &Ask, rows: &'a mut Vec<u32>) -> &'a [u32] {
        rows.clear();
        let own = self
            .vetoes
            .get(pool as usize)
            .and_then(|runs| run_of(runs, ask.gateway));
        if let Some(run) = own.filter(|run| run.holds) {
            let held = run.vetoes.iter().filter(|veto| veto.ends.held_at(ask.now));
            rows.extend(held.map(|veto| veto.row));
        }
        rows
    }

    /// Build both sides of one pair decision into `out`: the model's
    /// prefill pool, its decode pool ([`World::build`] for each) and the
    /// table of their classes, all as this publication holds them. `None`,
    /// with every buffer cleared, when either pool has no member: one side
    /// alone is no pair, and nothing of it is built. The model's name is
    /// looked up once for both sides and the table.
    pub fn build_pair(
        &self,
        model: &str,
        ask: &Ask,
        out: &mut PairBuffers,
    ) -> Option<(Built, Built)> {
        // One lookup of the model finds both pools. The classes stand on
        // the prefill pool's shape where both pools hold a member, so
        // neither side is built for a model that has one side alone.
        let built = self.derived.by_model.get(model).and_then(|pools| {
            let prefill = pools[role_slot(Role::Prefill)]? as usize;
            let decode = pools[role_slot(Role::Decode)]? as usize;
            let classes = self.derived.pools.get(prefill)?.pairs.as_deref()?;
            let prefill = self.build_at(prefill, ask, &mut out.prefill)?;
            let decode = self.build_at(decode, ask, &mut out.decode)?;
            classes.table_into(&mut out.ends, &mut out.listed);
            Some((prefill, decode))
        });
        if built.is_none() {
            out.clear();
        }
        built
    }

    /// Build one decision's pair and its encode pool into `out`: the two
    /// pools and the table of [`World::build_pair`], the model's encode
    /// pool, the encode class each prefill names and each encoder's facts,
    /// all as this publication holds them. A class's number is read here
    /// beside the rows it names and leaves with them through
    /// [`EncodePairBuffers::pools`], never apart.
    ///
    /// The columns follow the rows: a row left out of the asking gateway's
    /// view, another gateway's host-local worker, takes its entry with it.
    /// A model with no encoder builds an empty encode pool, and a model
    /// whose encoders state no end names class `0` on every row. `None`,
    /// with every buffer cleared, when either side of the pair has no
    /// member. The pair is built by the pair's door as it stands; the
    /// encode side looks the model up once more.
    pub fn build_encode_pair(
        &self,
        model: &str,
        ask: &Ask,
        out: &mut EncodePairBuffers,
    ) -> Option<(Built, Built)> {
        out.encode.candidates.clear();
        out.encode.affinity.clear();
        out.facts.clear();
        out.prefill_classes.clear();
        let pair = self.build_pair(model, ask, &mut out.pair);
        // `None`: the pair's door cleared its buffers, and the encode side's
        // are clear already.
        let built = pair?;
        // The pair was built, so the index names the model and its pools.
        let pools = self.derived.by_model.get(model);
        let shape_of = |role: Role| {
            let at = pools?[role_slot(role)]? as usize;
            Some((at, self.derived.pools.get(at)?))
        };
        let classes =
            shape_of(Role::Prefill).and_then(|(_, shape)| Some((shape, shape.encodes.as_deref()?)));
        match classes {
            Some((shape, classes)) => {
                let named = classes.prefills();
                for (from, to) in Segments::of(&shape.locals, ask.gateway, named.len()) {
                    out.prefill_classes.extend_from_slice(&named[from..to]);
                }
            }
            None => out.prefill_classes.resize(built.0.pool_size as usize, 0),
        }
        if let Some((encode, shape)) = shape_of(Role::Encode) {
            if self.build_at(encode, ask, &mut out.encode).is_some() {
                let stated = shape.encoders.as_deref().unwrap_or(&[]);
                for (from, to) in Segments::of(&shape.locals, ask.gateway, stated.len()) {
                    out.facts.extend_from_slice(&stated[from..to]);
                }
            }
        }
        Some(built)
    }

    /// Name the rows the affinity stages read in the column's place
    /// ([`NamedAffinity`](crate::NamedAffinity)) for `view`, the rows a
    /// decision of `model` in `role` is made over: the pool's as
    /// [`World::view`] lends them, or as [`World::build`] wrote them, which
    /// its caller may have narrowed by its gateway's wires since. `named`
    /// takes the request's deepest holders, by one lookup of the pool's
    /// index under the profile's cap (its sample, or no cap where it takes
    /// none), and the pool's warming rows as this publication read them,
    /// less those whose naming ended by the decision's instant. Under a cap
    /// the holders' lookup lists nothing of the pool's length, and the
    /// warming rows are read where the publication holds them: they are
    /// copied only for a view that is not the pool whole, and past the
    /// first row's end. No row is named, and no index is read, where the
    /// view holds no row or no member serves `model` in `role`.
    ///
    /// The lookup is of this instant: an index is written between two
    /// publications, so a holder may be named a window before its row's
    /// load shows the request that made it one.
    pub fn name(
        &self,
        model: &str,
        role: Role,
        naming: &Naming<'_>,
        view: &[Candidate],
        named: &mut Named,
    ) {
        named.clear();
        // A view with no row (a subset hint of 0, wires no member shares)
        // names nobody and reads no index.
        if view.is_empty() {
            return;
        }
        let Some(index) = self.index(model, role) else {
            return;
        };
        let (Some(rows), Some(shape), Some(Some(held))) = (
            self.rows.get(index),
            self.derived.pools.get(index),
            self.indexes.get(index),
        ) else {
            return;
        };
        held.name(naming, rows, &shape.locals, view, named);
    }

    /// The pool's standing in its model's index, for the cases: a
    /// membership write that kept it publishes the same one.
    #[cfg(test)]
    pub(super) fn standing(&self, model: &str, role: Role) -> Option<Arc<PoolIndex>> {
        self.indexes.get(self.index(model, role)?)?.clone()
    }

    /// Every pool's standing, for the cases: a membership write that moved
    /// none publishes the same list.
    #[cfg(test)]
    pub(super) fn standings(&self) -> Arc<[Option<Arc<PoolIndex>>]> {
        Arc::clone(&self.indexes)
    }

    /// The workers of the pool a naming's read is kept within, by id: for
    /// the cases.
    #[cfg(test)]
    pub(crate) fn reached(&self, model: &str, role: Role, naming: &Naming<'_>) -> Vec<WorkerId> {
        let Some(index) = self.index(model, role) else {
            return Vec::new();
        };
        let (Some(rows), Some(shape), Some(Some(held))) = (
            self.rows.get(index),
            self.derived.pools.get(index),
            self.indexes.get(index),
        ) else {
            return Vec::new();
        };
        let reached = held.reached(naming, &shape.locals);
        reached
            .iter()
            .filter_map(|&row| rows.get(row as usize).map(|row| row.id))
            .collect()
    }
}

/// Write a veto's bits on each row of `rows` a veto of `run` masks at
/// `now`, and say whether one did: the stalled bit, and the breaker bit
/// beside it while the veto's hold stands.
///
/// A pool every gateway dials whole is copied whole (`whole`): a veto's row
/// is its position. Another gateway's host-local rows are left out, so
/// there the row is found by its id.
///
/// Out of line: a build that holds no veto of the asking gateway pays one
/// test for this walk, and the build's row loop pays nothing for it.
#[inline(never)]
fn mask(run: &GatewayRun, now: Instant, whole: bool, rows: &mut [Candidate]) -> bool {
    let mut masked = false;
    for veto in run.vetoes.iter() {
        if veto.ends.until <= now {
            continue;
        }
        let row = if whole {
            rows.get_mut(veto.row as usize)
        } else {
            row_of_id(rows, veto.id)
        };
        if let Some(row) = row {
            // Every standing veto masks its row, as the mask beside lent
            // rows names it: a stage that reads the stalled bit refuses
            // the row whichever way its host handed it. A hold says more,
            // by the breaker bit: it refuses execution as an open breaker
            // does, and no rule lifts it. A veto with no hold, or past its
            // hold's end, is a doubt alone.
            row.health.stalled = true;
            if veto.ends.held_at(now) {
                row.health.breaker_open = true;
            }
            masked = true;
        }
    }
    masked
}

/// The row of worker `id` in a slice sorted by worker id.
fn row_of_id(rows: &mut [Candidate], id: WorkerId) -> Option<&mut Candidate> {
    let at = rows.binary_search_by_key(&id, |row| row.id).ok()?;
    rows.get_mut(at)
}

/// The runs of a pool's rows a gateway's view takes: every row but the
/// host-local ones of another gateway, as index ranges in order.
struct Segments<'a> {
    locals: std::slice::Iter<'a, (u32, GatewayId)>,
    gateway: GatewayId,
    from: usize,
    len: usize,
    done: bool,
}

impl<'a> Segments<'a> {
    fn of(locals: &'a [(u32, GatewayId)], gateway: GatewayId, len: usize) -> Self {
        Self {
            locals: locals.iter(),
            gateway,
            from: 0,
            len,
            done: false,
        }
    }
}

impl Iterator for Segments<'_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        if self.done {
            return None;
        }
        for &(index, owner) in self.locals.by_ref() {
            let index = (index as usize).min(self.len);
            if owner == self.gateway {
                continue;
            }
            let run = (self.from, index);
            self.from = (index + 1).min(self.len);
            if run.0 < run.1 {
                return Some(run);
            }
        }
        self.done = true;
        (self.from < self.len).then_some((self.from, self.len))
    }
}

/// One member's fields under the standing generation's lists as last read
/// ([`Writer::rebuild`], [`Writer::list`]).
#[derive(Clone, Copy)]
struct Held {
    id: WorkerId,
    /// The revision of the member's row at the read: a write of the row's
    /// facts moves it.
    revision: u64,
    listed: Listed,
}

/// The candidate row of one member: its fixed fields from its registration,
/// its status bits, its load record when one is on file, and its engine
/// family, zone and labels as `listed` reads them under the standing
/// generation's lists. Its pair class is the membership's to derive
/// ([`pair_up`]).
fn row_of(
    member: &WorkerRow,
    sample: Option<&LoadSample>,
    origin: Instant,
    listed: Listed,
) -> Candidate {
    let facts = &member.facts;
    Candidate {
        id: member.id,
        role: facts.role,
        dp_rank: facts.dp_rank,
        dp_size: facts.dp_size,
        health: health_of(member.status),
        wire_modes: facts.wire_modes,
        load: sample.map(|sample| sample.row),
        load_sampled_at_micros: sample.map_or(0, |sample| micros(origin, sample.sampled_at)),
        host_local: facts.reachable_from != Reachability::Any,
        vision: facts.vision,
        // The low 32 bits of the registration key's fingerprint.
        key: fingerprint(facts.key.as_bytes()) as u32,
        engine_kind: listed.engine_kind,
        zone: listed.zone,
        labels: listed.labels,
        ..Candidate::default()
    }
}

/// The two bits a status sets: only a ready worker is healthy, and a
/// draining one says so.
fn health_of(status: WorkerStatus) -> Health {
    Health {
        healthy: status == WorkerStatus::Ready,
        draining: status == WorkerStatus::Draining,
        ..Health::default()
    }
}

/// One member on its way into a pool: its row, what it registered with, and
/// the two facts of its place.
#[derive(Clone, Copy)]
struct Entry<'a> {
    row: Candidate,
    facts: &'a WorkerFacts,
    admitted_micros: u64,
    local: Option<GatewayId>,
    /// The thresholds its registration names.
    own: Overload,
    /// Its member's slot in the store's counts.
    booked: Slot,
}

/// The engines of the pool `entries` list, which stand by id: the ranks of
/// one data-parallel engine share its group in `ranked`, and every other
/// member is an engine alone. `placed` holds a slot a group, each free on
/// entry and on return. Nothing is listed, and nothing allocated, where no
/// engine of the pool holds two rows.
fn engines_of(
    entries: &[Entry<'_>],
    ranked: &[(WorkerId, u32, u64)],
    placed: &mut [u32],
) -> PoolEngines {
    if ranked.is_empty() {
        return PoolEngines::default();
    }
    // Each row's engine, the engines numbered as they first appear, and
    // each engine's count of rows.
    let mut of = Vec::with_capacity(entries.len());
    let mut sizes: Vec<u32> = Vec::new();
    let mut next = 0;
    for entry in entries {
        let group = ranked_at(ranked, &mut next, entry.row.id)
            .and_then(|(group, _)| placed.get_mut(group as usize));
        let engine = match group {
            Some(place) if *place != u32::MAX => *place,
            free => {
                let engine = sizes.len() as u32;
                sizes.push(0);
                if let Some(place) = free {
                    *place = engine;
                }
                engine
            }
        };
        if let Some(size) = sizes.get_mut(engine as usize) {
            *size += 1;
        }
        of.push(engine);
    }
    // The groups' slots are freed for the next pool.
    next = 0;
    for entry in entries {
        let group = ranked_at(ranked, &mut next, entry.row.id);
        if let Some(place) = group.and_then(|(group, _)| placed.get_mut(group as usize)) {
            *place = u32::MAX;
        }
    }
    if sizes.len() == entries.len() {
        return PoolEngines::default();
    }
    // Each engine's end, then its rows written from its start.
    let mut ends = sizes;
    let mut end = 0;
    for size in &mut ends {
        end += *size;
        *size = end;
    }
    let mut at: Vec<u32> = std::iter::once(0)
        .chain(ends.iter().copied())
        .take(ends.len())
        .collect();
    let mut rows = vec![0_u32; entries.len()];
    for (position, &engine) in (0_u32..).zip(&of) {
        let Some(slot) = at.get_mut(engine as usize) else {
            continue;
        };
        if let Some(row) = rows.get_mut(*slot as usize) {
            *row = position;
        }
        *slot += 1;
    }
    // An engine's rows stand in its ranks' order: by id they need not.
    let rank_of = |row: &u32| entries.get(*row as usize).map(|entry| entry.facts.dp_rank);
    let mut from = 0;
    for &end in &ends {
        if let Some(engine) = rows.get_mut(from as usize..end as usize) {
            engine.sort_unstable_by_key(rank_of);
        }
        from = end;
    }
    PoolEngines {
        rows: rows.into_boxed_slice(),
        ends: ends.into_boxed_slice(),
    }
}

/// The list a membership build fills for a pool that stood at `standing`
/// members: room for them and one more, the most one write adds to a pool, so
/// the build of one write allocates the list once. A build that pays a
/// storm of joins grows the list as it fills, and so does a pool that did
/// not stand, which takes no room.
fn list_of<T>(standing: usize) -> Vec<T> {
    if standing == 0 {
        Vec::new()
    } else {
        Vec::with_capacity(standing + 1)
    }
}

/// One side's whole entries for a model's pools. `places` lists the side's
/// members by their place in `entries`, in order, and the list is made
/// once at its length. `last` says no model behind this one reads
/// `entries`: a side that lists every member then takes them as they
/// stand, so a fleet of one model on one side writes each entry once.
fn side_of<T: Copy>(entries: &mut Vec<T>, places: &[u32], last: bool) -> Vec<T> {
    if last && places.len() == entries.len() {
        mem::take(entries)
    } else {
        places.iter().map(|&at| entries[at as usize]).collect()
    }
}

/// What the pairing rule reads of one side's entries, in place: each
/// worker's id, wires, runtime and descriptor.
impl<'a> Pool<'a> for [Entry<'a>] {
    fn count(&self) -> usize {
        self.len()
    }

    fn id(&self, at: usize) -> WorkerId {
        self[at].row.id
    }

    fn member(&self, at: usize) -> Member<'a> {
        let entry = &self[at];
        let facts = entry.facts;
        Member::new(
            entry.row.id,
            facts.wire_modes,
            &facts.engine_kind,
            facts.pairing.as_deref(),
        )
    }
}

/// What a membership write asks of a model's classes. The pair classes
/// follow each variant's words. The encode classes are carried under
/// [`Classes::Carry`], and under [`Classes::Written`] in every model whose
/// encoders and prefills the write left as they stood ([`Ends::stand`]).
/// Elsewhere they are derived whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Classes {
    /// Every model's classes are derived whole: a generation of another
    /// pairing mode, which moves every verdict.
    Derive,
    /// One worker's write moved a member or a member's offer: `left` is
    /// the worker the last write held and `joined` the worker as it stands
    /// now, each where there is one, and one id where a registration moved
    /// its own offer. The classes are derived for that worker alone in the
    /// models it left and joined ([`Derivation::written`]) and carried in
    /// every other.
    Written {
        left: Option<WorkerId>,
        joined: Option<WorkerId>,
    },
    /// No member joined or left and no member's offer moved
    /// ([`same_offer`]): the classes are the last write's, carried, and no
    /// member's facts are read for them. Every pool holds the rows it
    /// held, each admitted when it was and stating the wires it stated: a
    /// pool's standing in its model's index is carried on the same word
    /// ([`Writer::index_pools`]).
    Carry,
}

impl Classes {
    /// What one write asks that stands for a write asking `self` and a
    /// later one asking `next`: the build that pays both derives once.
    fn then(self, next: Self) -> Self {
        match (self, next) {
            (Self::Carry, asked) | (asked, Self::Carry) => asked,
            // One worker written twice: the worker the last build held and
            // the worker as it stands now. A worker that came and went
            // between two builds moved nothing.
            (
                Self::Written {
                    left,
                    joined: Some(between),
                },
                Self::Written {
                    left: Some(again),
                    joined,
                },
            ) if between == again => {
                if left.is_none() && joined.is_none() {
                    Self::Carry
                } else {
                    Self::Written { left, joined }
                }
            }
            // Two workers' writes, or a whole derivation owed already.
            _ => Self::Derive,
        }
    }
}

/// Whether two versions of one worker's facts stand in the same pools with
/// the same offer: the role and the models place a worker in its pools, and
/// the wires, the runtime and the descriptor are what the pairing rule reads of
/// it, the runtime as an offer holds it ([`known`]); the runtime and the
/// stated end are what [`encode_member`] reads. A write from one such
/// version to the other moves no pair class and no encode class. The
/// models are read as they are listed: a refresh that only reorders them
/// is answered as a moved offer, for that worker alone, where
/// [`WorkerFacts::same_models`] reads the reorder as the same worker for
/// the status a new incarnation takes.
pub(super) fn same_offer(old: &WorkerFacts, new: &WorkerFacts) -> bool {
    old.role == new.role
        && old.models == new.models
        && old.wire_modes == new.wire_modes
        && known(&old.engine_kind) == known(&new.engine_kind)
        && old.pairing == new.pairing
        && old.encode == new.encode
}

/// The models `old` stands in a pool of and `new` does not, where both are
/// one worker's facts under one registration: every model of the old list
/// where the role moved, and the old list less the new where it did not.
/// None where the worker left no pool: a refresh, an offer that moved
/// inside its pools, a model added. The names move out of the facts the
/// write replaced, which end here: a write that takes no model off frees
/// them whole ahead of its world and allocates nothing for the list.
pub(super) fn left_models(old: WorkerFacts, new: &WorkerFacts) -> Vec<String> {
    let moved = old.role != new.role;
    let mut models = old.models;
    models
        .extract_if(.., |model| moved || !new.models.contains(model))
        .collect()
}

/// One model's pair classes under `mode`, each prefill's and each decode's
/// class written on its row. `carried` is the last write's and `kept` holds
/// the model's derivation as that write left it: the classes are carried
/// where this write moved no member and no offer, derived for the written
/// worker alone where it moved one, and derived whole for a model's first
/// pair and for a generation of another mode (`classes`). `None` when a
/// side has no member: one side alone is no pair, and its rows keep class
/// 0.
fn pair_up(
    sides: &mut [Vec<Entry<'_>>; ROLES],
    mode: Pairing,
    classes: Classes,
    carried: Option<&Arc<PairClasses>>,
    kept: &mut FxHashMap<Box<str>, Derivation>,
    model: &str,
) -> Option<Arc<PairClasses>> {
    let (prefill, decode) = (role_slot(Role::Prefill), role_slot(Role::Decode));
    if sides[prefill].is_empty() || sides[decode].is_empty() {
        return None;
    }
    let (prefills, decodes) = (sides[prefill].as_slice(), sides[decode].as_slice());
    let classes = match (classes, carried) {
        // No member and no offer moved: the last write's classes stand, and
        // what is kept of their derivation is not looked up.
        (Classes::Carry, Some(carried)) => Arc::clone(carried),
        (classes, carried) => match kept.get_mut(model) {
            Some(derivation) => match (classes, carried) {
                (Classes::Written { left, joined }, Some(carried)) => derivation
                    .written(carried, prefills, decodes, mode, (left, joined))
                    .map_or_else(|| Arc::clone(carried), Arc::new),
                _ => Arc::new(derivation.whole(prefills, decodes, mode)),
            },
            // The model's first pair.
            None => {
                let mut derivation = Derivation::default();
                let classes = Arc::new(derivation.whole(prefills, decodes, mode));
                kept.insert(model.into(), derivation);
                classes
            }
        },
    };
    for (side, classed) in [(prefill, classes.prefills()), (decode, classes.decodes())] {
        let ids = sides[side].iter().map(|entry| entry.row.id);
        debug_assert!(
            classed.iter().map(|&(id, _)| id).eq(ids),
            "the classes are these members'"
        );
        for (entry, &(_, class)) in sides[side].iter_mut().zip(classed) {
            entry.row.pair_class = class;
        }
    }
    Some(classes)
}

/// What the encode handoff's rule reads of one entry: its worker's runtime
/// and stated end.
fn encode_member<'a>(entry: &Entry<'a>) -> EncodeMember<'a> {
    EncodeMember {
        runtime: &entry.facts.engine_kind,
        offer: &entry.facts.encode,
    }
}

/// One model's encode classes, each side in its rows' order: what the last
/// write left on the model's prefill pool (`last_prefill`), classes or
/// none, where this write left the model's encoders and prefills as they
/// stood ([`Ends::stand`]), else derived. `None` when the model has none:
/// no prefill, or no encoder that states an end. A model with no encoder
/// pays no derivation, and a write that carries pays none whatever the
/// model holds.
fn encode_up(
    sides: &[Vec<Entry<'_>>; ROLES],
    last_prefill: Option<&PoolShape>,
) -> Option<Arc<EncodeClasses>> {
    let encoders = &sides[role_slot(Role::Encode)];
    let prefills = &sides[role_slot(Role::Prefill)];
    if let Some(last) = last_prefill {
        debug_assert!(
            last.encodes.iter().all(|classes| {
                classes.encoders().len() == encoders.len()
                    && classes.prefills().len() == prefills.len()
            }),
            "the classes are these members'"
        );
        return last.encodes.clone();
    }
    if encoders.is_empty() || prefills.is_empty() {
        return None;
    }
    let classes = EncodeClasses::derive(
        encoders.iter().map(encode_member),
        prefills.iter().map(encode_member),
    );
    (classes.classes() != 0).then(|| Arc::new(classes))
}

/// What a decision reads of each encoder beside its row, in the rows'
/// order: its class among `classes`, `0` where it states no end or the
/// model has no class; whether its registration states a push, with its
/// role or without; and whether it takes an item's references. They are
/// what the last write left on the model's encode pool (`last_encode`)
/// where this write left the model's encoders and prefills as they stood
/// ([`Ends::stand`]), and `None` where the model has no encoder.
fn encoder_facts(
    encoders: &[Entry<'_>],
    classes: Option<&EncodeClasses>,
    last_encode: Option<&PoolShape>,
) -> Option<Arc<[EncoderFacts]>> {
    if let Some(last) = last_encode {
        debug_assert!(
            last.encoders
                .iter()
                .all(|stated| stated.len() == encoders.len()),
            "the facts are these members'"
        );
        return last.encoders.clone();
    }
    if encoders.is_empty() {
        return None;
    }
    let classed = classes.map_or(&[][..], EncodeClasses::encoders);
    let stated = encoders.iter().enumerate().map(|(at, entry)| {
        let offer = &entry.facts.encode;
        EncoderFacts {
            class: classed.get(at).copied().unwrap_or(0),
            pushes: offer.handoff == EncodeHandoff::Push,
            references: offer.references,
        }
    });
    Some(stated.collect())
}

/// One member's row in one pool of the writer's working copy.
#[derive(Clone, Copy)]
struct Place {
    id: WorkerId,
    pool: u32,
    row: u32,
}

/// The writer's own copy of one pool: what records and status changes are
/// folded into between two publications.
struct Working {
    rows: Vec<Candidate>,
    /// Folded since the last publication.
    dirty: bool,
    /// How many of the rows hold no report ([`unreported`]): taken where a
    /// membership write builds the rows, then moved by each write that
    /// gives a row its first report or takes its last ([`rows_at`]). A
    /// publication reads it and passes over no row.
    unreported: u32,
    /// Each row's own overload thresholds, from its registration.
    own: Vec<Overload>,
    /// The thresholds the standing generation's binding hands this pool;
    /// `None` when no binding matches it or its block is switched off.
    binding: Option<OverloadThresholds>,
}

/// One engine as the overload verdict reads it: the ranks of one
/// data-parallel engine, which share its key, a member a rank; or a
/// single-rank worker alone, whose engine is its own key.
#[derive(Clone, Copy, Default)]
struct EngineRows {
    /// Where the engine's rows stand in the writer's one list of them: the
    /// first, and how many.
    at: u32,
    len: u32,
    /// A row of the engine has a threshold, or an answer left one of its
    /// ranks unnamed. An engine with neither is not judged at a write: its
    /// rows' bits are clear and stay so.
    judged: bool,
    /// The last word of the engine's poll was that no report came: its
    /// verdict reads clear, whatever is on file, until its next record or
    /// report. It outlives a membership build, which judges again, and it
    /// is kept and ended while no threshold stands.
    silent: bool,
    /// When the poll of that word began, in microseconds from the origin;
    /// read while `silent` alone. A membership build carries the word over
    /// no record that came at or after it and that no row took at its write.
    said: u64,
}

/// One row of an engine: a rank's row in one pool, where a verdict is
/// written.
#[derive(Clone, Copy, Default)]
struct EngineRow {
    pool: u32,
    row: u32,
    /// The first of its rank's rows: where the judge reads the rank's
    /// record on file. A rank's rows carry the same record in every pool.
    rank: bool,
}

/// How many report periods a rank's sample may lie behind its engine's
/// newest receipt and still be judged. The period told is no shorter than
/// the time between two of the caller's polls and than a poll's deadline,
/// so a rank that loses a report is heard again inside three periods. A
/// rank that stays silent is counted through three periods, so through
/// one lost report at the least, and ends at the first judged write
/// after them.
const SAMPLE_PERIODS: u64 = 3;

/// The engine's rows, in the writer's list of every engine's.
fn rows_of<'a>(rows: &'a [EngineRow], engine: &EngineRows) -> &'a [EngineRow] {
    let from = engine.at as usize;
    rows.get(from..from + engine.len as usize)
        .unwrap_or_default()
}

/// The group and the receipt of the rank `id` in `ranked`, which stands by
/// id, or `None` for a member that is no rank of a data-parallel engine.
/// The callers ask in rising id order: `next` is their cursor.
fn ranked_at(
    ranked: &[(WorkerId, u32, u64)],
    next: &mut usize,
    id: WorkerId,
) -> Option<(u32, u64)> {
    while ranked.get(*next).is_some_and(|&(at, ..)| at < id) {
        *next += 1;
    }
    ranked
        .get(*next)
        .filter(|&&(at, ..)| at == id)
        .map(|&(_, group, heard)| (group, heard))
}

/// The build the membership writes since the last one left owed.
#[derive(Clone, Copy)]
struct Owed {
    /// What those writes ask of the pair classes, as one write would.
    classes: Classes,
    /// When the first of them was written: the build is due from then, or
    /// from the window's end where that is later.
    since: Instant,
}

/// The write side of the world, behind the store's load lock: the working
/// copy of every pool, where each member's rows stand in them, what the
/// last publication held, and the publication window. Every write that
/// publishes hands back the world it made; the store shows it under the
/// same lock, so the worlds reach the deciders in the order they were made.
pub(super) struct Writer {
    window: Duration,
    origin: Instant,
    /// The store's epoch at the last membership build.
    epoch: u64,
    derived: Arc<Derived>,
    /// Each model's derivation of its pair classes as the last membership
    /// build left it, for the models that hold a prefill and a decode: the
    /// next build of one worker's writes derives for that worker alone.
    derivations: FxHashMap<Box<str>, Derivation>,
    /// Each pool's rows as the last publication left them.
    published: Vec<Arc<[Candidate]>>,
    /// How many rows of each pool hold no report, as the last publication
    /// showed them ([`Working::unreported`]): one list for every world until
    /// a count moves.
    unreported: Arc<[u32]>,
    /// Each pool's spare: the rows it published before its last, kept for
    /// its next publication to write over; none before the pool's first
    /// publication after a membership build. Beside the working copies,
    /// not in them: a load write's step from pool to pool is as it was.
    spares: Vec<Option<Arc<[Candidate]>>>,
    working: Vec<Working>,
    /// Rows that left their pool and that a world may still list: a spare a
    /// publication found listed by a world some reader holds, and the
    /// spares and the published rows of the pools a membership build
    /// retired. The writer never lets go of rows a world lists, so a
    /// decision that drops the last world listing them frees no rows.
    held: Vec<Arc<[Candidate]>>,
    /// The held rows a publication found no world listing any more, for
    /// that publication alone: a pool of their length with no spare to
    /// write over takes them, and the publication frees the rest.
    unlisted: Vec<Arc<[Candidate]>>,
    /// How many times a publication allocated a pool's rows.
    allocations: u64,
    /// How many times the pools were built from the membership.
    builds: u64,
    /// Every replaced world a reader still held when the cell let it go
    /// ([`Self::replaced`]), until a publication finds no reader holding it
    /// and lets it go. The cell holds the standing world and the writer
    /// each of these, so a decision that drops the last reader's hold
    /// frees nothing: not the world, its list of pools, its vetoes, its
    /// derived tables nor its generation.
    worlds: Vec<Arc<World>>,
    /// How many worlds the writer published.
    publications: u64,
    /// Every member's rows, sorted by worker id: a worker serving several
    /// models has a row in each model's pool.
    places: Vec<Place>,
    /// The store's counts of the leases on file: the one table of the
    /// store's life, which the leases' file writes and a world lends.
    counts: Arc<Counts>,
    vetoes: Arc<Vetoes>,
    /// The vetoes as the world carries them, by pool and by gateway. A
    /// veto's write builds its own gateway's runs again; a membership build
    /// and a window's publication that drops a veto past its end build
    /// every run again.
    pool_vetoes: Arc<[PoolVetoes]>,
    /// The generation every world is published under until the next is
    /// applied: the unbound one at the start.
    spec: Arc<AppliedSpec>,
    /// Each pool's standing in its model's index: made by a membership
    /// write, its warming rows read again by a window's publication.
    indexes: Arc<[Option<Arc<PoolIndex>>]>,
    /// The buffers a pool's reading is made in.
    reading: Reading,
    /// The lane table's count of moves as the last membership write read
    /// it ([`KvTables::moves`]): while it stands, no pool's place in an
    /// index moved.
    lanes: u64,
    /// The store's index tables, where a membership write reads each
    /// pool's standing ([`Self::over`]). A writer made alone holds tables
    /// of its own with no lane.
    kv: Arc<KvTables>,
    /// The generation before the standing one, kept one apply longer: the
    /// publication that lets go of the last world published under it
    /// frees no engine, the next apply does. Its whole engine stays
    /// resident that long, its session table included, as it stood when
    /// it was replaced.
    retired: Option<Arc<AppliedSpec>>,
    /// The engines of the membership, as the overload verdict reads them.
    engines: Vec<EngineRows>,
    /// Every engine's rows in one list, an engine's together. A membership
    /// write fills both lists again in place: it allocates for no engine.
    engine_rows: Vec<EngineRow>,
    /// Each place's engine, in the places' order: beside the places, not
    /// in them, so a write that judges no engine steps over them as it
    /// did.
    place_engines: Vec<u32>,
    /// Each place's row among the engines' rows, in the places' order: a
    /// rank's receipt stands at its first place's, found by no search.
    /// Written with the receipts, and empty where none is kept.
    place_slots: Vec<u32>,
    /// When each rank of a data-parallel engine was last heard: the
    /// receipt of its record on file, in microseconds from the origin,
    /// in the engine rows' order. Empty for a membership of single-rank
    /// workers, whose one sample is its engine's newest. A record's
    /// write keeps it while an engine has a threshold, and the writes
    /// that start a judgment, a membership write and an applied
    /// generation, read every receipt again from the samples.
    heard: Vec<u64>,
    /// How far behind its engine's newest receipt a rank's sample may
    /// lie and still be judged, in microseconds; none until a report
    /// period is told.
    sample_bound: Option<u64>,
    /// An engine has a threshold.
    judging: bool,
    /// How many engines' last word was that no report came.
    silenced: u32,
    /// The ranks an engine's last answer left unnamed beside a rank it
    /// named, each by its member and its rank's number, sorted. A rank
    /// stands here until an answer or a record names it again, its
    /// member's rows go, or the build that follows its member's
    /// registration under another rank number. While a rank the answer
    /// named has a record on file, an unnamed rank's rows carry the
    /// overload bit ([`Self::judge`]): a decision that drops overloaded
    /// rows takes no rank its engine stopped naming while another
    /// answers. Empty until an answer names part of an engine.
    unnamed: Vec<(WorkerId, u16)>,
    /// How many writes to the load table moved an overload verdict.
    flips: u64,
    /// The standing members outside the standing generation's lists.
    outside: OutsideLists,
    /// Each member's fields under the standing generation's lists, in id
    /// order, beside the revision of its row they were read at. A
    /// membership build reads a member again only where that revision
    /// moved, so it reads the members written since the last build and
    /// carries every other's. Empty while the generation declares no list.
    listed: Vec<Held>,
    /// The build before's `listed`, kept for its room.
    listed_spare: Vec<Held>,
    /// The last publication the window counts from: a window's end, or a
    /// load write that published by itself.
    last_window: Option<Instant>,
    /// A folded load record waits for the window's end.
    waiting: bool,
    /// The build the membership writes since the last one left owed; `None`
    /// while the working rows are the membership's.
    owed: Option<Owed>,
}

impl Writer {
    /// The write side of a store with no member, whose rows' clock starts
    /// at `origin`.
    pub(super) fn new(window: Duration, origin: Instant) -> Self {
        Self {
            window,
            origin,
            epoch: 0,
            derived: Arc::default(),
            derivations: FxHashMap::default(),
            published: Vec::new(),
            unreported: Arc::from(Vec::new()),
            spares: Vec::new(),
            working: Vec::new(),
            held: Vec::new(),
            unlisted: Vec::new(),
            allocations: 0,
            builds: 0,
            worlds: Vec::new(),
            publications: 0,
            places: Vec::new(),
            counts: Arc::default(),
            vetoes: Arc::default(),
            pool_vetoes: Arc::default(),
            spec: Arc::new(AppliedSpec::unbound()),
            indexes: Arc::default(),
            reading: Reading::default(),
            lanes: 0,
            kv: Arc::default(),
            retired: None,
            engines: Vec::new(),
            engine_rows: Vec::new(),
            place_engines: Vec::new(),
            place_slots: Vec::new(),
            heard: Vec::new(),
            sample_bound: None,
            judging: false,
            silenced: 0,
            unnamed: Vec::new(),
            flips: 0,
            outside: OutsideLists::default(),
            listed: Vec::new(),
            listed_spare: Vec::new(),
            last_window: None,
            waiting: false,
            owed: None,
        }
    }

    /// The store's counts of the leases on file, for the leases' file to
    /// write.
    pub(super) fn counts(&self) -> Arc<Counts> {
        Arc::clone(&self.counts)
    }

    /// The writer over the store's index tables: every membership write
    /// reads its pools' standings in `kv`. The store hands them once, at
    /// its making, and writes its lanes into the same tables.
    pub(super) fn over(mut self, kv: Arc<KvTables>) -> Self {
        self.kv = kv;
        self
    }

    /// A world of everything published so far: one small object, a pointer
    /// per pool. Nothing derived is built here.
    pub(super) fn world(&self) -> Arc<World> {
        Arc::new(World {
            epoch: self.epoch,
            origin: self.origin,
            derived: Arc::clone(&self.derived),
            rows: self.published.clone().into_boxed_slice(),
            unreported: Arc::clone(&self.unreported),
            vetoes: Arc::clone(&self.pool_vetoes),
            spec: Arc::clone(&self.spec),
            #[cfg(feature = "controller")]
            samples_decisions: self.spec.engine().explain_sample() != 0,
            indexes: Arc::clone(&self.indexes),
        })
    }

    /// Take back the world the cell let go as its successor was shown. One
    /// no reader holds is freed here, by the write, as a world always was;
    /// one a reader holds is kept among `worlds`, so that reader's hold is
    /// not the last. The cell's swap pays each reader's borrow of the
    /// outgoing world into its count before it returns, so the count read
    /// here is every reader's.
    pub(super) fn replaced(&mut self, world: Arc<World>) {
        if let Err(held) = Arc::try_unwrap(world) {
            self.worlds.push(held);
        }
    }

    /// Publish every world from here on under `spec`, the first at once:
    /// the standing membership under the new generation. A number at or
    /// under the standing one is refused by naming the standing one, and
    /// nothing is published: the number only rises, so no reader sees a
    /// generation and then a lower one. Each pool takes the thresholds of
    /// the new generation's binding, and every engine is judged again
    /// under them from what is on file. A generation whose lists differ
    /// from the standing one's has each of `members`, the standing
    /// membership, read again under them, so the first world of a
    /// generation holds rows of its own lists.
    ///
    /// The pair classes are derived under the generation's pairing mode.
    /// A generation of the standing mode keeps the rows' classes; one
    /// of another mode derives `members`, the membership as it stands at
    /// the store's `epoch`, again by the build's own door
    /// ([`Writer::rebuild`]) and publishes once, under the new generation:
    /// the epoch is the membership's and the generation the document's,
    /// and what is derived from both is published under both. That build
    /// takes in every membership write since the last, so none stays owed.
    /// It reads every member where the generation's lists differ from the
    /// standing one's and carries each member's fields where they are the
    /// same. Either way each pool's warming rows are ruled by the new
    /// generation: the build's door reads the index tables under it, and a
    /// generation of the standing mode rules each pool's standing again.
    ///
    /// The generation this one replaces is kept until the next apply, and
    /// the one kept until now is handed back for the caller to let go
    /// outside the store's lock: an engine is freed by the thread that
    /// applies, never by a decider. A world a decider holds across two
    /// applies keeps its generation past the hand-back: the engine then
    /// goes with that world, at the publication that lets it go.
    pub(super) fn apply<'a>(
        &mut self,
        spec: Arc<AppliedSpec>,
        epoch: u64,
        members: impl Iterator<Item = (&'a WorkerRow, Slot)>,
        sample_of: impl Fn(WorkerId, u16) -> Option<LoadSample>,
    ) -> Result<(Arc<World>, Option<Arc<AppliedSpec>>), StaleGeneration> {
        let standing = self.spec.generation();
        if spec.generation() <= standing {
            return Err(StaleGeneration {
                standing,
                offered: spec.generation(),
            });
        }
        let relisted = spec.lists() != self.spec.lists();
        let derive = spec.pairing() != self.spec.pairing();
        let before = self.retired.replace(mem::replace(&mut self.spec, spec));
        let world = if derive {
            if relisted {
                // No member's fields are carried into other lists.
                self.listed.clear();
            }
            self.rebuild(epoch, members, sample_of, Classes::Derive)
        } else {
            if relisted {
                self.list(members);
            }
            self.bind();
            // A generation of the standing mode comes with no membership
            // write: the receipts are read from the samples as the rows
            // stand.
            self.hear_again(sample_of);
            self.judge_every();
            // A generation may rule a pool's warming rows by other numbers,
            // or rule none.
            self.rule_indexes();
            self.publish()
        };
        Ok((world, before))
    }

    /// Read every member under the standing generation's lists: its engine
    /// family and its zone as positions, its labels as bits, in each of
    /// its rows. For an apply whose lists differ: a membership build reads
    /// the members written since the last and carries every other's
    /// ([`Writer::rebuild`]). A pool with a row that moved is published
    /// whole.
    fn list<'a>(&mut self, members: impl Iterator<Item = (&'a WorkerRow, Slot)>) {
        let Self {
            places,
            working,
            spec,
            outside,
            listed: held,
            ..
        } = self;
        let lists = spec.lists();
        let listing = !lists.is_empty();
        *outside = OutsideLists::default();
        held.clear();
        for (member, _) in members {
            let facts = &member.facts;
            let listed = lists.listed(&facts.engine_kind, facts.zone.as_deref(), &facts.labels);
            outside.engines += u32::from(listed.engine_outside);
            outside.zones += u32::from(listed.zone_outside);
            if listing {
                held.push(Held {
                    id: member.id,
                    revision: member.revision,
                    listed,
                });
            }
            let now = (listed.engine_kind, listed.zone, listed.labels);
            for place in places_of(places, member.id) {
                let Some(pool) = working.get_mut(place.pool as usize) else {
                    continue;
                };
                let Some(row) = pool.rows.get_mut(place.row as usize) else {
                    continue;
                };
                if (row.engine_kind, row.zone, row.labels) == now {
                    continue;
                }
                (row.engine_kind, row.zone, row.labels) = now;
                pool.dirty = true;
            }
        }
        held.sort_unstable_by_key(|held| held.id);
    }

    /// Each pool's thresholds under the standing generation: those of the
    /// binding the pool is decided under, which is the first to match its
    /// model and its role's phase, or the model's pair binding for a
    /// decode pool
    /// ([`Engine::pool_binding`](crate::framework::Engine::pool_binding)).
    /// Whether any pool has a pair.
    fn bind(&mut self) -> bool {
        let Self {
            derived,
            working,
            spec,
            ..
        } = self;
        let mut bound = false;
        for (model, pools) in &derived.by_model {
            for &index in pools.iter().flatten() {
                let Some(pool) = working.get_mut(index as usize) else {
                    continue;
                };
                let Some(role) = pool.rows.first().map(|row| row.role) else {
                    continue;
                };
                pool.binding = spec
                    .engine()
                    .pool_binding(model, phase_of(role))
                    .and_then(|binding| binding.overload);
                bound |= pool.binding.is_some();
            }
        }
        bound
    }

    /// Whether a row of the engine has a threshold: its own, or its
    /// pool's.
    fn has_threshold(&self, engine: usize) -> bool {
        self.engines.get(engine).is_some_and(|engine| {
            rows_of(&self.engine_rows, engine).iter().any(|at| {
                self.working.get(at.pool as usize).is_some_and(|pool| {
                    pool.binding.is_some()
                        || pool
                            .own
                            .get(at.row as usize)
                            .is_some_and(|own| *own != Overload::default())
                })
            })
        })
    }

    /// Judge every engine again from what is on file. A membership build's
    /// rows start with no bit, and an applied generation may hand any pool
    /// another pair or none: an engine that had a threshold is judged once
    /// more under what it has now, so a bit no threshold holds is cleared.
    ///
    /// A door comes here too where its write ended the last mark, behind
    /// its own judgment of the write ([`Self::judged`]). That judgment
    /// read the written engine as the whole write left it, the mark gone,
    /// and counted the write, so the write is counted once. This pass
    /// finds every verdict as its last write left it and moves none: from
    /// it an engine with no threshold is judged at no write. Whether any
    /// row's bit moved, which a door holds to none
    /// ([`Self::judge_every_behind_a_door`]).
    pub(super) fn judge_every(&mut self) -> bool {
        self.judging = false;
        let mut moved = false;
        for index in 0..self.engines.len() {
            // A rank stands unnamed only while the writer judges: a record
            // reads a mark behind that one test.
            let has = self.has_threshold(index) || self.unnamed_in(index);
            self.judging |= has;
            let Some(engine) = self.engines.get_mut(index) else {
                continue;
            };
            let had = mem::replace(&mut engine.judged, true);
            if had || has {
                moved |= self.judge(index as u32);
            }
            if let Some(engine) = self.engines.get_mut(index) {
                engine.judged = has;
            }
        }
        moved
    }

    /// The pass over every engine as a door runs it, behind its own
    /// judgment of a write that ended the last mark
    /// ([`Self::judge_every`]). That judgment counted the write and this
    /// pass counts nothing, so the pass must move no verdict: one it moved
    /// would go uncounted, and unpublished where the door's own judgment
    /// moved none. A debug build stops there.
    pub(super) fn judge_every_behind_a_door(&mut self) {
        let moved = self.judge_every();
        debug_assert!(!moved, "the pass behind a door's judgment moved a verdict");
    }

    /// Judge one engine over what is on file for its ranks and write the
    /// verdict on each row where it changes; whether any row's bit moved.
    ///
    /// The verdict is the engine's: the queued requests summed over its
    /// ranks on file against a row's count, and their active KV usage
    /// summed against the row's permille times the ranks on file, which is
    /// the mean compared with no division. Each row is judged under its
    /// own pair: the value its registration names, else its pool's
    /// binding's. An engine with nothing on file reads clear, and so does
    /// one whose last poll brought no report. An engine no row of which
    /// has a threshold is not judged, unless an answer left a rank of it
    /// unnamed ([`Self::name`]): that rank's rows carry the bit while a
    /// rank the answer named has a record on file, whatever the
    /// thresholds say, and with no such rank it reads as its engine does.
    fn judge(&mut self, engine: u32) -> bool {
        let Self {
            engines,
            engine_rows,
            working,
            heard,
            sample_bound,
            unnamed,
            ..
        } = self;
        let Some(engine) = engines.get(engine as usize).filter(|engine| engine.judged) else {
            return false;
        };
        let rows = rows_of(engine_rows, engine);
        let load_of = |at: &EngineRow| {
            working
                .get(at.pool as usize)
                .and_then(|pool| pool.rows.get(at.row as usize))
                .and_then(|row| row.load)
        };
        // A rank the engine's last answer left unnamed is avoided while a
        // rank that answer named has a record on file: alone it answers.
        let is_unnamed = |id: WorkerId, rank: u16| unnamed.binary_search(&(id, rank)).is_ok();
        let named = !unnamed.is_empty()
            && rows.iter().filter(|at| at.rank).any(|at| {
                working
                    .get(at.pool as usize)
                    .and_then(|pool| pool.rows.get(at.row as usize))
                    .is_some_and(|row| row.load.is_some() && !is_unnamed(row.id, row.dp_rank))
            });
        // A sample is a fact while its engine still says it: once a report
        // period is told, a rank whose receipt lies further behind the
        // engine's newest than the bound is not counted. One rank's sample
        // is its engine's newest. An engine no rank of which is heard is
        // judged by no write: its verdict stands as it was last judged.
        let from = engine.at as usize;
        let receipts = match *sample_bound {
            Some(bound) if rows.len() > 1 => heard
                .get(from..from + rows.len())
                .map(|heard| (heard, bound)),
            _ => None,
        };
        // The sums over the ranks heard no earlier than `oldest`, then the
        // newest and the eldest receipt among them.
        let sum = |oldest: u64| {
            let (mut queued, mut usage, mut ranks) = (0_u64, 0_u64, 0_u64);
            let (mut newest, mut eldest) = (0_u64, u64::MAX);
            for (index, at) in rows.iter().enumerate().filter(|(_, at)| at.rank) {
                let Some(load) = load_of(at) else {
                    continue;
                };
                let heard = receipts
                    .and_then(|(heard, _)| heard.get(index))
                    .copied()
                    .unwrap_or(0);
                if heard < oldest {
                    continue;
                }
                queued += u64::from(load.queued);
                usage += u64::from(load.active_kv_permille);
                ranks += 1;
                newest = newest.max(heard);
                eldest = eldest.min(heard);
            }
            ((queued, usage, ranks), (newest, eldest))
        };
        // One walk sums an engine no sample of which has ended; a second
        // leaves the ended ones out.
        let (queued, usage, ranks) = if engine.silent {
            (0, 0, 0)
        } else {
            let (whole, (newest, eldest)) = sum(0);
            let oldest = receipts.map_or(0, |(_, bound)| newest.saturating_sub(bound));
            if eldest < oldest {
                sum(oldest).0
            } else {
                whole
            }
        };
        let mut moved = false;
        for &EngineRow { pool, row, .. } in rows {
            let Some(pool) = working.get_mut(pool as usize) else {
                continue;
            };
            let own = pool.own.get(row as usize).copied().unwrap_or_default();
            let waiting = own
                .waiting_requests
                .or_else(|| pool.binding.map(|pair| pair.waiting_requests));
            let permille = own
                .kv_usage_permille
                .or_else(|| pool.binding.map(|pair| pair.kv_usage_permille));
            let over = ranks > 0
                && (waiting.is_some_and(|at| queued >= u64::from(at))
                    || permille.is_some_and(|at| usage >= u64::from(at) * ranks));
            let Some(row) = pool.rows.get_mut(row as usize) else {
                continue;
            };
            let over = over || (named && is_unnamed(row.id, row.dp_rank));
            if row.health.overloaded != over {
                row.health.overloaded = over;
                pool.dirty = true;
                moved = true;
            }
        }
        moved
    }

    /// A write changed what is on file for the engine: judge it once, and
    /// count the write when a verdict moved. `heard` is a record or a
    /// report's row, which ends the word that none came, whether or not a
    /// threshold stands: a generation that brings one judges by a word
    /// that is current. An engine no row of which has a threshold is not
    /// judged.
    pub(super) fn judged(&mut self, engine: u32, heard: bool) -> bool {
        let Some(rows) = self.engines.get_mut(engine as usize) else {
            return false;
        };
        let ended = heard && mem::take(&mut rows.silent);
        let judged = rows.judged;
        self.silenced -= u32::from(ended);
        if !judged {
            return false;
        }
        let moved = self.judge(engine);
        self.flips += u64::from(moved);
        moved
    }

    /// The engine the member is a rank of; `None` for a member with no
    /// row.
    pub(super) fn engine_of(&self, id: WorkerId) -> Option<u32> {
        let first = self.places.partition_point(|place| place.id < id);
        self.places
            .get(first)
            .filter(|place| place.id == id)
            .and_then(|_| self.engine_at(first as u32))
    }

    /// The engine of the member whose place stands at `place`.
    pub(super) fn engine_at(&self, place: u32) -> Option<u32> {
        self.place_engines.get(place as usize).copied()
    }

    /// A record received at `at` changed what is on file for the member
    /// whose first place stands at `place`: its engine is heard, and
    /// judged once ([`Self::judge_heard`]). The record names its rank, so
    /// a mark an answer left on it ends ([`Self::recorded_marked`]). While
    /// no engine has a threshold, none is silent and no rank stands
    /// unnamed there is no word to end and no verdict to move, and the
    /// write reads no engine and no mark.
    pub(super) fn recorded(&mut self, place: u32, id: WorkerId, rank: u16, at: Instant) -> bool {
        if !self.judging && self.silenced == 0 {
            return false;
        }
        if !self.unnamed.is_empty() {
            return self.recorded_marked(place, id, rank, at);
        }
        self.judge_heard(place, at)
    }

    /// The member whose first place stands at `place` was heard at `at`:
    /// its rank takes the receipt, and its engine is judged once with its
    /// word that no report came ended ([`Self::judged`]).
    fn judge_heard(&mut self, place: u32, at: Instant) -> bool {
        let Some(engine) = self.engine_at(place) else {
            return false;
        };
        self.hear(place, at);
        self.judged(engine, true)
    }

    /// The record door in a store that holds a mark. The record names its
    /// rank again, which ends the mark an answer left on the rank
    /// ([`Self::name`]), and its engine is heard and judged as at any
    /// record. Where the last mark went with it, every engine is judged
    /// once more behind that judgment
    /// ([`Self::judge_every_behind_a_door`]). Cold and
    /// out of line: the record door runs at every pushed record and comes
    /// here only while a mark stands, and with this step inlined a record
    /// paid for it in a store that holds no mark.
    #[cold]
    #[inline(never)]
    fn recorded_marked(&mut self, place: u32, id: WorkerId, rank: u16, at: Instant) -> bool {
        self.name(id, rank, false);
        let moved = self.judge_heard(place, at);
        if self.unnamed.is_empty() {
            self.judge_every_behind_a_door();
        }
        moved
    }

    /// The member whose first place stands at `place` was heard at `at`:
    /// the receipt its sample is aged from, at the place's own row among
    /// the engines' rows. Kept for the ranks of a data-parallel engine,
    /// and only while an engine has a threshold: a write that starts a
    /// judgment reads every receipt again from the samples, a membership
    /// write in its walk and an applied generation by
    /// [`Self::hear_again`].
    pub(super) fn hear(&mut self, place: u32, at: Instant) {
        if self.heard.is_empty() || !self.judging {
            return;
        }
        let Some(&slot) = self.place_slots.get(place as usize) else {
            return;
        };
        // A member's later place holds no receipt: its row is no rank's.
        let row = self.engine_rows.get(slot as usize);
        if !row.is_some_and(|row| row.rank) {
            return;
        }
        if let Some(heard) = self.heard.get_mut(slot as usize) {
            *heard = micros(self.origin, at);
        }
    }

    /// Every rank's receipt again, from its sample on file: what an
    /// applied generation judges by, since a record's write keeps the
    /// receipts only while an engine has a threshold. A membership write
    /// reads them in its own walk.
    fn hear_again(&mut self, sample_of: impl Fn(WorkerId, u16) -> Option<LoadSample>) {
        for (heard, at) in self.heard.iter_mut().zip(&self.engine_rows) {
            if !at.rank {
                continue;
            }
            let sample = self
                .working
                .get(at.pool as usize)
                .and_then(|pool| pool.rows.get(at.row as usize))
                .and_then(|row| sample_of(row.id, row.dp_rank));
            *heard = sample.map_or(0, |sample| micros(self.origin, sample.received_at));
        }
    }

    /// Every ready rank is heard once a `period` from here on: a judgment
    /// counts a rank only while its receipt lies within three periods of
    /// its engine's newest, and every engine with a threshold is judged
    /// again under it. A period is told once: another changes nothing.
    /// Whether `period` is the one that stands, and whether a verdict
    /// moved.
    pub(super) fn heard_every(&mut self, period: Duration) -> (bool, bool) {
        let bound = u64::try_from(period.as_micros())
            .unwrap_or(u64::MAX)
            .saturating_mul(SAMPLE_PERIODS);
        if let Some(told) = self.sample_bound {
            return (told == bound, false);
        }
        self.sample_bound = Some(bound);
        let mut moved = false;
        for index in 0..self.engines.len() {
            moved |= self.judge(index as u32);
        }
        (true, moved)
    }

    /// The engine's ranks, each by its member and its rank's number.
    pub(super) fn ranks_of(&self, engine: u32) -> impl Iterator<Item = (WorkerId, u16)> + '_ {
        let rows = self
            .engines
            .get(engine as usize)
            .map(|engine| rows_of(&self.engine_rows, engine))
            .unwrap_or_default();
        rows.iter().filter(|at| at.rank).filter_map(|at| {
            let pool = self.working.get(at.pool as usize)?;
            let row = pool.rows.get(at.row as usize)?;
            Some((row.id, row.dp_rank))
        })
    }

    /// The word that no report came for the engine, by a poll that began at
    /// `polled_at`: from here its verdict reads clear until its next record
    /// or report, and the engine is judged once. The word keeps that
    /// instant for the next membership build ([`Self::rebuild`]). It is kept
    /// for an engine with no threshold too, which is not judged. Whether a
    /// verdict moved.
    pub(super) fn silence(&mut self, engine: u32, polled_at: Instant) -> bool {
        let said = micros(self.origin, polled_at);
        let Some(rows) = self.engines.get_mut(engine as usize) else {
            return false;
        };
        let was = mem::replace(&mut rows.silent, true);
        rows.said = said;
        self.silenced += u32::from(!was);
        self.judged(engine, false)
    }

    /// The engine's last answer named another of its ranks and not `rank`
    /// of the member `id` (`unnamed`), or an answer or a record names the
    /// rank again. The caller judges the engine once its write is whole
    /// ([`Self::judged`]), and every engine once more behind that where
    /// the write ended the last mark
    /// ([`Self::judge_every_behind_a_door`]). From the
    /// mark on the engine is judged at each write, threshold or none,
    /// until the last mark's end, a membership write or a generation
    /// finds it with no mark and no threshold. Whether the mark moved.
    pub(super) fn name(&mut self, id: WorkerId, rank: u16, unnamed: bool) -> bool {
        match (self.unnamed.binary_search(&(id, rank)), unnamed) {
            (Err(at), true) => {
                self.unnamed.insert(at, (id, rank));
                let engine = self
                    .engine_of(id)
                    .and_then(|engine| self.engines.get_mut(engine as usize));
                if let Some(engine) = engine {
                    engine.judged = true;
                }
                self.judging = true;
                true
            }
            (Ok(at), false) => {
                self.unnamed.remove(at);
                true
            }
            _ => false,
        }
    }

    /// Whether any rank stands unnamed: a door's one test before it reads
    /// an answer or a record for a mark.
    pub(super) fn any_unnamed(&self) -> bool {
        !self.unnamed.is_empty()
    }

    /// The member's rows went with an eviction: its ranks' marks go with
    /// them, and a rank with nothing on file is a rank again. The caller
    /// judges the engine, and every engine once more behind it where the
    /// last mark went ([`Self::judge_every_behind_a_door`]). Whether a
    /// mark went.
    pub(super) fn forget(&mut self, id: WorkerId) -> bool {
        let before = self.unnamed.len();
        self.unnamed.retain(|&(member, _)| member != id);
        self.unnamed.len() != before
    }

    /// The marks of the members a membership write removed go with them,
    /// and so does a mark on a rank number its member no longer holds: a
    /// refresh may move the number under one id, and no answer or record
    /// names the old one again.
    fn forget_gone(&mut self) {
        let Self {
            unnamed,
            places,
            working,
            ..
        } = self;
        unnamed.retain(|&(id, rank)| {
            places_of(places, id).any(|place| {
                working
                    .get(place.pool as usize)
                    .and_then(|pool| pool.rows.get(place.row as usize))
                    .is_some_and(|row| row.dp_rank == rank)
            })
        });
    }

    /// Whether an answer left a rank of the engine unnamed.
    fn unnamed_in(&self, engine: usize) -> bool {
        !self.unnamed.is_empty()
            && self.engines.get(engine).is_some_and(|engine| {
                rows_of(&self.engine_rows, engine).iter().any(|at| {
                    self.working
                        .get(at.pool as usize)
                        .and_then(|pool| pool.rows.get(at.row as usize))
                        .is_some_and(|row| {
                            self.unnamed.binary_search(&(row.id, row.dp_rank)).is_ok()
                        })
                })
            })
    }

    /// How many writes to the load table moved an overload verdict.
    pub(super) fn flips(&self) -> u64 {
        self.flips
    }

    /// The standing members outside the standing generation's lists.
    pub(super) fn outside(&self) -> OutsideLists {
        self.outside
    }

    /// Derive the pools of the membership `members` at the store's `epoch`,
    /// each row with its load record on file, make them the working copy
    /// and publish them through the one door: every pool is new, so the
    /// door copies none, every earlier fold is in the world this returns
    /// and nothing waits. The pair classes are derived for the written
    /// worker alone, derived whole or carried from the last write, as the
    /// caller says (`classes`). A member's engine family, zone and labels
    /// under the standing generation's lists are read where its row's
    /// revision moved since they were last read, and carried where it
    /// stands. Each pool takes its standing in its model's index as the
    /// write leaves the store's tables ([`Self::over`]).
    pub(super) fn rebuild<'a>(
        &mut self,
        epoch: u64,
        members: impl Iterator<Item = (&'a WorkerRow, Slot)>,
        sample_of: impl Fn(WorkerId, u16) -> Option<LoadSample>,
        classes: Classes,
    ) -> Arc<World> {
        // A build takes in every membership write: none stays owed.
        self.owed = None;
        self.builds += 1;
        let origin = self.origin;
        // The last write's classes: carried, or what the written worker's
        // derive starts from.
        let last = (classes != Classes::Derive).then(|| Arc::clone(&self.derived));
        // An engine whose last poll brought no report stays silent across
        // the write: its verdict is not raised again from what is still on
        // file. The word is carried by its members' ids, which the outgoing
        // places hold in order. Each id stands with the instant its engine's
        // word was said for. The list is empty while no engine is silent.
        let mut silent: Vec<(WorkerId, u64)> = Vec::new();
        if self.silenced > 0 {
            let engines = &self.engines;
            let word = |(place, &engine): (&Place, &u32)| {
                let rows = engines.get(engine as usize)?;
                rows.silent.then_some((place.id, rows.said))
            };
            let outgoing = self.places.iter().zip(&self.place_engines);
            silent.extend(outgoing.filter_map(word));
            silent.dedup();
        }
        // The ranks of the data-parallel engines, each with its engine's
        // group by key and the receipt of its sample on file. A fleet of
        // single-rank workers lists none.
        let mut by_key: FxHashMap<&'a str, u32> = FxHashMap::default();
        // The records on file that no row took at their write, each by its
        // receipt: the newest of an engine's ranks at its group's number, a
        // single-rank worker's under its own id. Read while a word may be
        // carried.
        let mut fresh: Vec<Option<u64>> = Vec::new();
        let mut alone: Vec<(WorkerId, u64)> = Vec::new();
        let mut ranked: Vec<(WorkerId, u32, u64)> = Vec::new();
        // A member names a threshold of its own.
        let mut owned = false;
        // Each model's members a side, by their place in `entries`: a
        // member's entry is written once, whatever it offers.
        let mut groups: FxHashMap<Box<str>, [Vec<u32>; ROLES]> = FxHashMap::default();
        // The members that offer no model, by their place in `members`: in
        // no pool, and named by the world all the same. None in most
        // fleets.
        let mut bare: Vec<u32> = Vec::new();
        // The members' ids are read by the vetoes alone: a membership build
        // with no veto standing does not collect them.
        let standing = !self.vetoes.by_gateway.is_empty();
        let mut ids = Vec::new();
        // The members in id order, by this write's one sort, each id beside
        // its row and the slot its write gave it: an id is one member, so
        // every pool's entries, the ranks and the ids are pushed in order
        // and nothing below sorts them. The sort moves an id, a reference
        // and a slot, never a whole entry.
        let mut members: Vec<(WorkerId, &'a WorkerRow, Slot)> = members
            .map(|(member, slot)| (member.id, member, slot))
            .collect();
        members.sort_unstable_by_key(|&(id, ..)| id);
        // Every member's entry, in that order. A pool's list of whole
        // entries is made from it below for one model at a time, at its
        // length: a build holds one model's lists and never every
        // model's, and moves none to grow.
        let mut entries: Vec<Entry<'a>> = Vec::with_capacity(members.len());
        // Each member's three fields under the standing generation's lists:
        // carried from the last read where the member's row stands at the
        // revision it was read at, read again where it moved or is new. A
        // build so reads the members written since the last, and a
        // generation with no list costs it this one test.
        let lists = self.spec.lists();
        let listing = !lists.is_empty();
        let outgoing = mem::take(&mut self.listed);
        let mut incoming = mem::take(&mut self.listed_spare);
        incoming.clear();
        let mut from = 0;
        let mut outside = OutsideLists::default();
        for &(_, member, slot) in &members {
            if standing {
                ids.push(member.id);
            }
            let facts = &member.facts;
            let sample = sample_of(member.id, facts.dp_rank);
            // The ranks of a data-parallel engine share its key. A
            // single-rank worker's engine is its own key, which no other
            // row has: it is an engine alone, and no key is looked up.
            // A record on file that no row took: the build asks what the
            // record's load write asked, whether a row of the member stands
            // at its rank. None does for a member new to the rows and for
            // one a refresh moved to another rank.
            let rank = facts.dp_rank;
            let unheard = !silent.is_empty()
                && sample.is_some()
                && rows_at(&self.places, &mut self.working, member.id, rank, |_| false).is_none();
            if facts.dp_size > 1 {
                let next = by_key.len() as u32;
                let group = *by_key.entry(facts.engine.as_str()).or_insert(next);
                let heard = sample.map_or(0, |sample| micros(origin, sample.received_at));
                ranked.push((member.id, group, heard));
                if unheard {
                    // The groups are dense numbers: an engine's newest
                    // receipt is one slot, and one lookup a silent member.
                    let at = group as usize;
                    if fresh.len() <= at {
                        fresh.resize(at + 1, None);
                    }
                    if let Some(newest) = fresh.get_mut(at) {
                        *newest = Some(newest.map_or(heard, |was| was.max(heard)));
                    }
                }
            } else if unheard {
                let heard = sample.map_or(0, |sample| micros(origin, sample.received_at));
                alone.push((member.id, heard));
            }
            owned |= facts.overload != Overload::default();
            let listed = if listing {
                while outgoing.get(from).is_some_and(|held| held.id < member.id) {
                    from += 1;
                }
                let listed = match outgoing.get(from) {
                    Some(held) if held.id == member.id && held.revision == member.revision => {
                        held.listed
                    }
                    _ => lists.listed(&facts.engine_kind, facts.zone.as_deref(), &facts.labels),
                };
                outside.engines += u32::from(listed.engine_outside);
                outside.zones += u32::from(listed.zone_outside);
                incoming.push(Held {
                    id: member.id,
                    revision: member.revision,
                    listed,
                });
                listed
            } else {
                Listed::default()
            };
            let entry = Entry {
                row: row_of(member, sample.as_ref(), origin, listed),
                facts,
                admitted_micros: micros(origin, member.admitted_at),
                local: match facts.reachable_from {
                    Reachability::Any => None,
                    Reachability::Gateway(gateway) => Some(gateway),
                },
                own: facts.overload,
                // The slot the member's first write gave it: a build gives
                // none and moves none.
                booked: slot,
            };
            let at = entries.len() as u32;
            entries.push(entry);
            if facts.models.is_empty() {
                // In no pool, and done: the loop below starts on a model.
                bare.push(at);
                continue;
            }
            let side = role_slot(facts.role);
            for model in &facts.models {
                match groups.get_mut(model.as_str()) {
                    // A model listed twice is one pool's one row. A member's
                    // place is the last of its side until the next member is
                    // read, so an earlier listing is found there: no model
                    // before this one is read again.
                    Some(sides) => {
                        let list = &mut sides[side];
                        if list.last() != Some(&at) {
                            list.push(at);
                        }
                    }
                    None => {
                        // Each side takes room for its pool's last members
                        // and the one this write may add.
                        let before = self.derived.by_model.get(model.as_str());
                        let room = before.map_or([0; ROLES], |pools| {
                            pools.map(|pool| self.derived.members(pool))
                        });
                        let mut sides: [Vec<u32>; ROLES] = room.map(list_of);
                        sides[side].push(at);
                        groups.insert(model.as_str().into(), sides);
                    }
                }
            }
        }
        // The outgoing pools' rows wait with the held ones, the spares and
        // the published: the publication that finds no world listing them
        // gives them to a pool of their length or frees them, this write's
        // own at the earliest for the spares, the next for the published.
        self.held.extend(self.spares.drain(..).flatten());
        self.held.append(&mut self.published);
        self.working.clear();
        self.places.clear();
        let mut by_model = FxHashMap::default();
        let mut pools = Vec::new();
        // Each data-parallel engine's place among the engines of the pool
        // being derived: a slot a group, free between two pools.
        let mut placed = vec![u32::MAX; by_key.len()];
        let models = groups.len();
        for (nth, (model, sides_at)) in groups.into_iter().enumerate() {
            // The model's whole entries a side, freed with the model: the
            // build holds one model's lists at a time. No model behind the
            // last reads `entries`.
            let last_model = nth + 1 == models;
            let mut sides: [Vec<Entry<'a>>; ROLES] =
                sides_at.map(|side| side_of(&mut entries, &side, last_model));
            // Every side stands in id order, as the walk above pushed it,
            // before a class is derived: the classes are numbered in it, and
            // each row is built with its class.
            debug_assert!(sides
                .iter()
                .all(|entries| entries.is_sorted_by(|a, b| a.row.id < b.row.id)));
            // The model's classes go on its prefill pool's shape: a model
            // that has classes has that pool.
            // One read of the last write's pools of the model serves its
            // pair classes, its encode classes and its encoders' facts.
            let ends = last.as_deref().and_then(|last| last.ends(&model));
            let carried = ends.and_then(|ends| ends.prefill?.pairs.as_ref());
            let (mode, kept) = (self.spec.pairing(), &mut self.derivations);
            let mut pairs = pair_up(&mut sides, mode, classes, carried, kept, &model);
            // The encode classes and the encoders' facts are the last
            // write's in every model whose encoders and prefills this write
            // left as they stood, and derived whole in the others.
            let stood = ends.filter(|ends| ends.stand(classes, &sides));
            let mut encodes = encode_up(&sides, stood.and_then(|ends| ends.prefill));
            let last_encode = stood.and_then(|ends| ends.encode);
            let mut stated = encoder_facts(
                &sides[role_slot(Role::Encode)],
                encodes.as_deref(),
                last_encode,
            );
            let mut indexes = [None; ROLES];
            for (side, entries) in sides.into_iter().enumerate() {
                if entries.is_empty() {
                    continue;
                }
                // A row's position is published with it: a pool every
                // gateway reaches whole is read where it stands.
                let rows: Vec<Candidate> = (0u32..)
                    .zip(&entries)
                    .map(|(position, entry)| Candidate {
                        position,
                        ..entry.row
                    })
                    .collect();
                let index = self.working.len() as u32;
                self.places
                    .extend(entries.iter().enumerate().map(|(row, entry)| Place {
                        id: entry.row.id,
                        pool: index,
                        row: row as u32,
                    }));
                pools.push(PoolShape {
                    admitted_micros: entries.iter().map(|entry| entry.admitted_micros).collect(),
                    booked: entries.iter().map(|entry| entry.booked).collect(),
                    locals: entries
                        .iter()
                        .enumerate()
                        .filter_map(|(row, entry)| entry.local.map(|gateway| (row as u32, gateway)))
                        .collect(),
                    pairs: if side == role_slot(Role::Prefill) {
                        pairs.take()
                    } else {
                        None
                    },
                    encodes: if side == role_slot(Role::Prefill) {
                        encodes.take()
                    } else {
                        None
                    },
                    encoders: if side == role_slot(Role::Encode) {
                        stated.take()
                    } else {
                        None
                    },
                    engines: engines_of(&entries, &ranked, &mut placed),
                });
                self.spares.push(None);
                self.working.push(Working {
                    rows,
                    dirty: false,
                    // Taken at this write's end, once its rows stand.
                    unreported: 0,
                    // No list where no member names a threshold of its own:
                    // a row with none reads its pool's.
                    own: if owned {
                        entries.iter().map(|entry| entry.own).collect()
                    } else {
                        Vec::new()
                    },
                    binding: None,
                });
                indexes[side] = Some(index);
            }
            by_model.insert(model, indexes);
        }
        if classes != Classes::Carry {
            // A model that lost a side keeps no derivation.
            let (prefill, decode) = (role_slot(Role::Prefill), role_slot(Role::Decode));
            let paired =
                |pools: &[Option<u32>; ROLES]| pools[prefill].is_some() && pools[decode].is_some();
            self.derivations
                .retain(|model, _| by_model.get(model).is_some_and(paired));
        }
        self.places
            .sort_unstable_by_key(|place| (place.id, place.pool));
        // The members' fields as this build read or carried them, and the
        // count outside the lists: none where the generation declares no
        // list, whatever the last one counted.
        self.listed = incoming;
        self.listed_spare = outgoing;
        self.outside = outside;
        // The engines, from the places, which stand by member: a member's
        // first place is where its record is read, each is where a verdict
        // is written. The writer's lists are written over in place, so the
        // write allocates for no engine.
        let mut engines = mem::take(&mut self.engines);
        engines.clear();
        self.engine_rows.clear();
        self.place_engines.clear();
        self.place_slots.clear();
        // A record no row took was heard for no engine, and no write before
        // this build knew the engine it is for: the word is carried to that
        // engine over no such record that came at or after the word's poll
        // began. A record's write leaves one, for a rank that waits for
        // this build; a poll's answer asks for the build first.
        if !fresh.is_empty() {
            silent.retain(|&(id, said)| {
                let at = ranked.binary_search_by_key(&id, |&(of, ..)| of).ok();
                let group = at.and_then(|at| ranked.get(at)).map(|&(_, group, _)| group);
                let newest = group.and_then(|group| fresh.get(group as usize).copied().flatten());
                !newest.is_some_and(|received| received >= said)
            });
        }
        // A single-rank worker is an engine alone: its own record ends the
        // word its own id carries.
        if !alone.is_empty() {
            silent.retain(|&(id, said)| {
                let at = alone.binary_search_by_key(&id, |&(of, _)| of).ok();
                let received = at.and_then(|at| alone.get(at));
                !received.is_some_and(|&(_, received)| received >= said)
            });
        }
        // The word a member's engine takes from it: the instant it was said
        // for.
        let carried = |id: WorkerId| {
            let at = silent.binary_search_by_key(&id, |&(of, _)| of).ok()?;
            silent.get(at).map(|&(_, said)| said)
        };
        let mut silenced = 0;
        let mut last = None;
        if ranked.is_empty() {
            // Single-rank workers alone: a member is an engine, and its
            // rows are the engine's as they come, in one walk. One rank's
            // sample is its engine's newest, so no receipt is kept.
            self.heard.clear();
            for place in &self.places {
                let rank = last != Some(place.id);
                if rank {
                    last = Some(place.id);
                    let said = carried(place.id);
                    silenced += u32::from(said.is_some());
                    engines.push(EngineRows {
                        at: self.engine_rows.len() as u32,
                        len: 0,
                        judged: false,
                        silent: said.is_some(),
                        said: said.unwrap_or_default(),
                    });
                }
                if let Some(engine) = engines.last_mut() {
                    engine.len += 1;
                }
                self.place_engines
                    .push(engines.len().saturating_sub(1) as u32);
                self.engine_rows.push(EngineRow {
                    pool: place.pool,
                    row: place.row,
                    rank,
                });
            }
        } else {
            // Ranks that share an engine do not stand together: each place
            // takes its engine, the engines' rows are counted and given
            // their room, then written in the places' order. The places
            // and the ranks both stand by id, the ranks as the walk above
            // pushed them, so a cursor finds a place's rank as the places
            // pass.
            debug_assert!(ranked.is_sorted_by(|a, b| a.0 < b.0));
            let mut of_group = vec![u32::MAX; by_key.len()];
            let mut engine = 0_u32;
            let mut next = 0;
            for place in &self.places {
                if last != Some(place.id) {
                    last = Some(place.id);
                    let group = ranked_at(&ranked, &mut next, place.id)
                        .and_then(|(group, _)| of_group.get_mut(group as usize));
                    let known = group.as_deref().copied().filter(|&known| known != u32::MAX);
                    engine = match known {
                        Some(known) => known,
                        None => {
                            engines.push(EngineRows::default());
                            let new = (engines.len() - 1) as u32;
                            if let Some(group) = group {
                                *group = new;
                            }
                            new
                        }
                    };
                    if let Some(rows) = engines.get_mut(engine as usize) {
                        if let Some(said) = carried(place.id) {
                            silenced += u32::from(!rows.silent);
                            rows.silent = true;
                            rows.said = rows.said.max(said);
                        }
                    }
                }
                if let Some(rows) = engines.get_mut(engine as usize) {
                    rows.len += 1;
                }
                self.place_engines.push(engine);
            }
            let mut rows = 0;
            for engine in &mut engines {
                engine.at = rows;
                rows += mem::take(&mut engine.len);
            }
            self.engine_rows.resize(rows as usize, EngineRow::default());
            // Each rank's receipt stands beside its row, from its sample on
            // file as the walk above read it: a judgment ages a sample from
            // where the writes before left it.
            self.heard.clear();
            self.heard.resize(rows as usize, 0);
            last = None;
            next = 0;
            self.place_slots.resize(self.places.len(), u32::MAX);
            let placed = self.places.iter().zip(&self.place_engines);
            for (index, (place, &engine)) in placed.enumerate() {
                let Some(engine) = engines.get_mut(engine as usize) else {
                    continue;
                };
                let slot = engine.at + engine.len;
                if let Some(of_place) = self.place_slots.get_mut(index) {
                    *of_place = slot;
                }
                let at = slot as usize;
                let rank = last != Some(place.id);
                if let Some(row) = self.engine_rows.get_mut(at) {
                    *row = EngineRow {
                        pool: place.pool,
                        row: place.row,
                        rank,
                    };
                }
                if rank {
                    let receipt =
                        ranked_at(&ranked, &mut next, place.id).map_or(0, |(_, heard)| heard);
                    if let Some(heard) = self.heard.get_mut(at) {
                        *heard = receipt;
                    }
                }
                engine.len += 1;
                last = Some(place.id);
            }
        }
        self.engines = engines;
        self.silenced = silenced;
        self.epoch = epoch;
        // In id order, as the walk above pushed them.
        // A place names its member's id and slot: the list stands by id, as
        // the walk read the members.
        let unpooled: Box<[(WorkerId, Slot)]> = bare
            .iter()
            .filter_map(|&at| members.get(at as usize))
            .map(|&(id, _, slot)| (id, slot))
            .collect();
        debug_assert!(unpooled.is_sorted_by(|a, b| a.0 < b.0));
        let before = mem::replace(
            &mut self.derived,
            Arc::new(Derived {
                by_model,
                pools: pools.into_boxed_slice(),
                unpooled,
                counts: Arc::clone(&self.counts),
            }),
        );
        // A worker's vetoes leave with its row, in this write's own hold:
        // no publication shows the row without them. The ids stand in
        // order as the walk above pushed them.
        if standing {
            debug_assert!(ids.is_sorted_by(|a, b| a < b));
            if let Some(vetoes) = self.vetoes.of_members(&ids) {
                self.vetoes = Arc::new(vetoes);
            }
        }
        self.derive_vetoes();
        // Every row is new and carries no bit: each engine is judged again
        // from what is on file, under the standing generation's pairs. A
        // write with no pair in any pool and none of a member's own has no
        // engine to judge. A pool's rows are cut for the publication after
        // that judgment, so the write copies each pool once, with a
        // verdict on it or none. A rank an answer left unnamed keeps its
        // mark across the write while its member stands, and its engine
        // is judged for it.
        self.forget_gone();
        if self.bind() || owned || !self.unnamed.is_empty() {
            self.judge_every();
        } else {
            self.judging = false;
        }
        for pool in &mut self.working {
            pool.dirty = false;
            self.published.push(Arc::from(pool.rows.as_slice()));
        }
        self.index_pools(&before, classes);
        // Every pool's rows are new: each count is taken again, by the one
        // pass over a pool's rows that a membership write pays for it.
        self.unreported = self
            .working
            .iter_mut()
            .map(|pool| {
                pool.unreported = unreported(&pool.rows);
                pool.unreported
            })
            .collect();
        self.publish()
    }

    /// Every pool's runs built again from the gateways' lists and the
    /// members' places: for a membership build, which moves the rows, and
    /// for a window's publication that dropped a veto past its end. A
    /// veto's own write builds its gateway's runs alone
    /// ([`Self::write_veto`]). No list at all while no veto stands.
    fn derive_vetoes(&mut self) {
        if self.vetoes.by_gateway.is_empty() {
            self.pool_vetoes = Arc::default();
            return;
        }
        // The gateways are taken in order, and a gateway's list is sorted
        // by worker id as a pool's rows are: each pool's runs come out
        // sorted by gateway and each run by row, with no sort of entries.
        let mut gateways: Vec<_> = self.vetoes.by_gateway.iter().collect();
        gateways.sort_unstable_by_key(|&(&gateway, _)| gateway);
        let mut pools: Vec<Vec<(GatewayId, Vec<PoolVeto>)>> = vec![Vec::new(); self.working.len()];
        for (&gateway, list) in gateways {
            // The list and the places both rise by worker id: one walk
            // over each joins them.
            let mut at = 0;
            for &(id, ends) in list.iter() {
                at = seek(&self.places, at, id, |place| place.id);
                while let Some(place) = self.places.get(at).filter(|place| place.id == id) {
                    at += 1;
                    let Some(pool) = pools.get_mut(place.pool as usize) else {
                        continue;
                    };
                    let veto = PoolVeto {
                        row: place.row,
                        id,
                        ends,
                    };
                    match pool.last_mut() {
                        Some((owner, run)) if *owner == gateway => run.push(veto),
                        // A run grows as its vetoes come: the gateway's
                        // list spans every pool, and a run holds one
                        // pool's share of it.
                        _ => pool.push((gateway, vec![veto])),
                    }
                }
            }
        }
        self.pool_vetoes = pools
            .into_iter()
            .map(|runs| {
                runs.into_iter()
                    .map(|(gateway, vetoes)| GatewayRun::new(gateway, vetoes.into()))
                    .collect()
            })
            .collect();
    }

    /// One veto's write as the pools carry it: in each pool the worker has
    /// a row in, the writing gateway's run takes the veto, moves its end
    /// or loses it. No other gateway's run and no other pool's is built
    /// again: a publication shares them with the one before.
    fn write_veto(&mut self, gateway: GatewayId, id: WorkerId, ends: Option<VetoEnds>) {
        if self.vetoes.by_gateway.is_empty() {
            // The last veto left: no list at all, as before the first.
            self.pool_vetoes = Arc::default();
            return;
        }
        // A worker's places stand by pool: one walk over the pools meets
        // each. Before the first veto there is no list, so every pool
        // starts from none.
        let none = PoolVetoes::default();
        let mut places = places_of(&self.places, id).peekable();
        self.pool_vetoes = (0..self.working.len())
            .map(|pool| {
                let runs = self.pool_vetoes.get(pool).unwrap_or(&none);
                match places.next_if(|place| place.pool as usize == pool) {
                    Some(place) => with_veto(runs, gateway, place.row, id, ends),
                    None => Arc::clone(runs),
                }
            })
            .collect();
    }

    /// Fold one rank's load record, or its absence, into the member's rows
    /// at that rank ([`rows_at`]); where the member's first place stands
    /// when a row took it. The caller judges the member's engine once its
    /// write is whole ([`Writer::recorded`], [`Writer::judged`]) and says
    /// when that is ([`Writer::folded_at`]).
    pub(super) fn fold_load(
        &mut self,
        id: WorkerId,
        rank: u16,
        sample: Option<&LoadSample>,
    ) -> Option<u32> {
        let Self {
            places,
            working,
            origin,
            ..
        } = self;
        rows_at(places, working, id, rank, |row| {
            row.load = sample.map(|sample| sample.row);
            row.load_sampled_at_micros =
                sample.map_or(0, |sample| micros(*origin, sample.sampled_at));
            true
        })
    }

    /// A load write folded its rows at `at`: it publishes by itself when
    /// the last window is a window old, else it waits for the window's end.
    /// While a build is owed it waits for the call that pays the build: a
    /// load write that opened the next window by itself would put the build
    /// off by a window each time. The world to show, when it published.
    #[must_use]
    pub(super) fn folded_at(&mut self, at: Instant) -> Option<Arc<World>> {
        let inside = |last| self.window_end(last).is_none_or(|end| at < end);
        if self.owed.is_some() || self.last_window.is_some_and(inside) {
            self.waiting = true;
            return None;
        }
        Some(self.publish_window(at))
    }

    /// A load write with no instant of its own folded its rows: it waits
    /// for the window's end, or publishes when no window ever opened.
    #[must_use]
    pub(super) fn folded(&mut self) -> Option<Arc<World>> {
        if self.last_window.is_some() {
            self.waiting = true;
            None
        } else {
            Some(self.publish())
        }
    }

    /// Fold a status change into the member's rows and publish at once: a
    /// decision does not wait a window for a worker that left rotation.
    /// `None` for a worker with no row, and for a change between two
    /// statuses a row reads alike: nothing a decision reads moved.
    #[must_use]
    pub(super) fn fold_status(&mut self, id: WorkerId, status: WorkerStatus) -> Option<Arc<World>> {
        let Self {
            places, working, ..
        } = self;
        let health = health_of(status);
        let mut folded = false;
        for place in places_of(places, id) {
            let Some(pool) = working.get_mut(place.pool as usize) else {
                continue;
            };
            let Some(row) = pool.rows.get_mut(place.row as usize) else {
                continue;
            };
            if (row.health.healthy, row.health.draining) == (health.healthy, health.draining) {
                continue;
            }
            row.health.healthy = health.healthy;
            row.health.draining = health.draining;
            pool.dirty = true;
            folded = true;
        }
        folded.then(|| self.publish())
    }

    /// Set `gateway`'s veto on `id` until `until` and its hold until `hold`,
    /// never past the veto's own end, or clear both, and publish at once;
    /// `None` when the write changes neither end.
    #[must_use]
    pub(super) fn veto(
        &mut self,
        gateway: GatewayId,
        id: WorkerId,
        until: Option<Instant>,
        hold: Option<Instant>,
    ) -> Option<Arc<World>> {
        let ends = until.map(|until| VetoEnds::new(until, hold));
        let vetoes = self.vetoes.with(gateway, id, ends)?;
        self.vetoes = Arc::new(vetoes);
        self.write_veto(gateway, id, ends);
        Some(self.publish())
    }

    /// Lift every veto of the gateways in `gone` and publish once: they
    /// are gone past their grace period. Every pool's runs are built again,
    /// once for them all, as for a membership write. `None` when none of
    /// them held a veto.
    #[must_use]
    pub(super) fn drop_gateways(&mut self, gone: &[GatewayId]) -> Option<Arc<World>> {
        let held = |gateway: &GatewayId| self.vetoes.by_gateway.contains_key(gateway);
        if !gone.iter().any(held) {
            return None;
        }
        let mut by_gateway = self.vetoes.by_gateway.clone();
        for gateway in gone {
            by_gateway.remove(gateway);
        }
        self.vetoes = Arc::new(Vetoes { by_gateway });
        self.derive_vetoes();
        Some(self.publish())
    }

    /// When what waits is due: the end of the window the last publication
    /// opened, for the folds and for a build the membership writes left
    /// owed. A build owed past that end, or with no window opened yet, is
    /// due from its first write. `None` when nothing waits, and for a
    /// window the clock cannot add.
    pub(super) fn due(&self) -> Option<Instant> {
        let end = self.last_window.map(|last| self.window_end(last));
        match (self.owed, end) {
            (Some(owed), None) => Some(owed.since),
            (Some(owed), Some(end)) => end.map(|end| end.max(owed.since)),
            (None, Some(end)) if self.waiting => end,
            (None, _) => None,
        }
    }

    /// A membership write moved the rows of record: the pools are built
    /// again once, when the window ends ([`Writer::due`]), for every write
    /// since the last build. `classes` is what this write asks of the pair
    /// classes and `at` when it was written. `true` where no build was
    /// owed before: the write named an instant the host has not read.
    pub(super) fn owe(&mut self, classes: Classes, at: Instant) -> bool {
        let first = self.owed.is_none();
        self.owed = Some(match self.owed {
            Some(owed) => Owed {
                classes: owed.classes.then(classes),
                since: owed.since,
            },
            None => Owed { classes, since: at },
        });
        first
    }

    /// Whether a build is owed and due at `now`: the store pays it with the
    /// membership in hand ([`Writer::build_owed`]).
    pub(super) fn build_due(&self, now: Instant) -> bool {
        self.owed.is_some() && self.due().is_some_and(|due| now >= due)
    }

    /// Whether a membership write left a build owed that none has paid:
    /// the standing pools then list no member written since their build.
    pub(super) fn owes(&self) -> bool {
        self.owed.is_some()
    }

    /// Whether the member `id` waits at `rank` for a build: a membership
    /// write left one owed and no row of the member stands at the rank, by
    /// the walk a load write folds a rank's record with ([`rows_at`]). A
    /// rank inside the window of its registration reads so, and one a
    /// refresh moved under its id. So does a worker that left while
    /// another's build is owed: the caller knows its members.
    pub(super) fn waits(&mut self, id: WorkerId, rank: u16) -> bool {
        self.owed.is_some()
            && rows_at(&self.places, &mut self.working, id, rank, |_| false).is_none()
    }

    /// Pay the build the membership writes left owed: the pools of
    /// `members`, the membership at the store's `epoch`, with every record
    /// on file. A host pays it at the window's end, `at`, which opens the
    /// next window and lets the vetoes past their end go. An asked build
    /// names no instant and leaves the window as it stands: the asker's own
    /// time holds no later record back. `None` when no build is owed:
    /// another call paid it.
    #[must_use]
    pub(super) fn build_owed<'a>(
        &mut self,
        at: Option<Instant>,
        epoch: u64,
        members: impl Iterator<Item = (&'a WorkerRow, Slot)>,
        sample_of: impl Fn(WorkerId, u16) -> Option<LoadSample>,
    ) -> Option<Arc<World>> {
        let owed = self.owed.take()?;
        if let Some(at) = at {
            self.last_window = Some(at);
            // A veto past its end leaves here, as at a window's publication:
            // the build derives every run from the vetoes that stand.
            if let Some(vetoes) = self.vetoes.standing_at(at) {
                self.vetoes = Arc::new(vetoes);
            }
        }
        Some(self.rebuild(epoch, members, sample_of, owed.classes))
    }

    /// The end of the window `last` opened; `None` for a window so long
    /// that the clock cannot add it, which no instant ends.
    fn window_end(&self, last: Instant) -> Option<Instant> {
        last.checked_add(self.window)
    }

    /// The window's end: publish what waits when `now` is at or past the
    /// due instant; a call before it publishes nothing, so a caller cannot
    /// publish faster than the window.
    #[must_use]
    pub(super) fn publish_if_due(&mut self, now: Instant) -> Option<Arc<World>> {
        self.due()
            .is_some_and(|due| now >= due)
            .then(|| self.publish_window(now))
    }

    /// The one door every publication goes through: each pool folded since
    /// the last publication takes new rows, whole, and one world holds
    /// every pool. What waited for the window rides along.
    ///
    /// A pool's new rows are its working copy written over rows the writer
    /// holds the last reference to ([`written_over`]): the pool's spare,
    /// else held rows of its length that no world lists any more. That is
    /// exact: every world listing such rows left the store's cell by the
    /// publication that replaced them, the cell's swap pays each reader's
    /// borrow of the outgoing world into that world's count before it
    /// returns, and a world counts on the rows it lists for as long as it
    /// lives. Rows some world still lists are left whole to that world's
    /// reader, and a pool with none to write over takes a new copy, which
    /// is counted: a publication never waits for a reader. Two buffers a
    /// pool carry every window, and one more for each publication a reader
    /// still holds. A publication asks each held row's count once, and each
    /// pool with no spare to write over reads each unlisted row's length.
    fn publish(&mut self) -> Arc<World> {
        // A world no reader holds any more is let go here, by a write, and
        // before the rows are asked: what it alone listed is free for this
        // publication.
        self.worlds
            .retain_mut(|world| Arc::get_mut(world).is_none());
        let unlisted = |rows: &mut Arc<[Candidate]>| Arc::get_mut(rows).is_some();
        self.unlisted.extend(self.held.extract_if(.., unlisted));
        // One count a pool: every membership write takes the list again.
        debug_assert_eq!(self.unreported.len(), self.working.len());
        let pools = self.working.iter_mut().zip(&mut self.published);
        // Whether this publication moves a count. The list is then taken
        // once more from the pools: the worlds that list the old counts
        // keep them.
        let mut moved = false;
        for (at, ((pool, published), spare)) in pools.zip(&mut self.spares).enumerate() {
            if !pool.dirty {
                // No write touched this pool: it holds the count listed.
                debug_assert_eq!(self.unreported.get(at), Some(&pool.unreported));
                continue;
            }
            let mut next = spare.take();
            // A spare some world still lists stays the writer's to free.
            let listed = next.take_if(|over| !written_over(over, &pool.rows));
            self.held.extend(listed);
            if next.is_none() {
                let fits = |over: &mut Arc<[Candidate]>| written_over(over, &pool.rows);
                let found = self.unlisted.iter_mut().position(fits);
                next = found.map(|at| self.unlisted.swap_remove(at));
            }
            let next = next.unwrap_or_else(|| {
                self.allocations += 1;
                Arc::from(pool.rows.as_slice())
            });
            *spare = Some(mem::replace(published, next));
            pool.dirty = false;
            // The count the writes kept: no row is passed over for it.
            let count = pool.unreported;
            debug_assert_eq!(count, unreported(&pool.rows));
            moved |= self.unreported.get(at) != Some(&count);
        }
        if moved {
            // One allocation: the counts go straight into their shared list.
            self.unreported = self.working.iter().map(|pool| pool.unreported).collect();
        }
        // Rows no world lists and no pool took are freed here, by a write.
        self.unlisted.clear();
        self.waiting = false;
        self.publications += 1;
        self.world()
    }

    /// How many worlds the writer published: the first world, made with
    /// the store, is not one.
    pub(super) fn publications(&self) -> u64 {
        self.publications
    }

    /// How many times a publication allocated a pool's rows: it found
    /// neither the pool's spare nor held rows of the pool's length free of
    /// every world. A membership build's own rows are not counted.
    pub(super) fn allocations(&self) -> u64 {
        self.allocations
    }

    /// How many ranks stand unnamed by their engine's last answer.
    #[cfg(test)]
    pub(super) fn unnamed_ranks(&self) -> usize {
        self.unnamed.len()
    }

    /// Whether a load write is judged ([`Self::recorded`]).
    #[cfg(test)]
    pub(super) fn judges_writes(&self) -> bool {
        self.judging
    }

    /// How many times the pools were built from the membership: a build an
    /// ask or a window's end paid, or a generation of another pairing mode.
    pub(super) fn builds(&self) -> u64 {
        self.builds
    }

    /// How many rows that left their pool the writer still keeps.
    #[cfg(test)]
    pub(super) fn held_rows(&self) -> usize {
        self.held.len() + self.unlisted.len()
    }

    /// How many whole derivations made what the writer keeps of `model`'s
    /// pair classes; `None` where it keeps nothing of them.
    #[cfg(test)]
    pub(super) fn whole_derivations(&self, model: &str) -> Option<usize> {
        self.derivations.get(model).map(Derivation::wholes)
    }

    /// Each pool's standing in its model's index as this membership write
    /// leaves it. A pool the write did not move keeps its standing: its
    /// tables, and its warming rows as a write or a window last read them.
    /// A write that moved no pool looks none up and allocates nothing
    /// here. A pool moved where its rows are not as many, a row was
    /// admitted at another instant, a lane's row states other wires
    /// ([`PoolIndex::holds`]), or a lane of the pool was made or retired
    /// ([`KvTables::stands`]); its standing is made ([`KvTables::pool`])
    /// with its warming rows under the standing generation's ruling for
    /// the pool's model and phase, read at the last window's instant or an
    /// admission after it. A pool is found at its place, or among
    /// `before`, the pools as they stood, where the write moved its place.
    /// A write that moved no member and no offer ([`Classes::Carry`]) moved
    /// no row, no admission and no wire a row states: where no lane moved
    /// either, every pool stands as it did and none is walked. Under
    /// [`Classes::Derive`] the pools may stand under a generation of another
    /// pairing mode, one the standings were not ruled by: every kept pool's
    /// warming rows are ruled and read again.
    fn index_pools(&mut self, before: &Derived, classes: Classes) {
        let Self {
            derived,
            working,
            spec,
            reading,
            indexes: held,
            origin,
            last_window,
            lanes,
            kv,
            ..
        } = self;
        let window = last_window.map_or(0, |at| micros(*origin, at));
        // No lane was made or retired since the last write stood its
        // pools: no pool's place in an index moved, and none is looked up.
        let moves = kv.moves();
        let still = mem::replace(lanes, moves) == moves;
        if still && classes == Classes::Carry {
            return;
        }
        let ruled = classes != Classes::Derive;
        // The list is copied at the first pool whose standing is new:
        // where the pools are as many, every other stands in it already.
        let same = held.len() == working.len();
        let mut next = (!same).then(|| vec![None; working.len()]);
        for (model, sides) in &derived.by_model {
            // The model's pools as they stood, looked up at the first of
            // its pools that does not stand at its place.
            let mut stood = None;
            for (side, pool) in sides.iter().enumerate() {
                let Some(at) = pool.map(|pool| pool as usize) else {
                    continue;
                };
                let (Some(rows), Some(shape)) = (working.get(at), derived.pools.get(at)) else {
                    continue;
                };
                let (rows, admitted) = (&*rows.rows, &*shape.admitted_micros);
                let here = held.get(at).and_then(Option::as_ref);
                let here = here.filter(|standing| standing.is_of(model, rows));
                let was = here.map(|standing| (at, standing)).or_else(|| {
                    let sides = *stood.get_or_insert_with(|| before.by_model.get(model));
                    let old = sides?.get(side).copied().flatten()? as usize;
                    Some((old, held.get(old)?.as_ref()?))
                });
                let kept = was.filter(|(_, standing)| {
                    standing.holds(rows, admitted) && (still || kv.stands(model, rows, standing))
                });
                let now = || admitted.iter().copied().fold(window, u64::max);
                let read = match kept {
                    Some((old, standing)) => {
                        let read = if ruled {
                            None
                        } else {
                            standing.again(ruling_of(spec, model, rows), now(), reading)
                        };
                        match read {
                            Some(read) => Arc::new(read),
                            None if same && old == at => continue,
                            None => Arc::clone(standing),
                        }
                    }
                    None => {
                        let rule = ruling_of(spec, model, rows);
                        Arc::new(kv.pool(model, rows, admitted, rule, now(), reading))
                    }
                };
                let next = next.get_or_insert_with(|| held.to_vec());
                if let Some(slot) = next.get_mut(at) {
                    *slot = Some(read);
                }
            }
        }
        if let Some(next) = next {
            *held = next.into();
        }
    }

    /// Rule every pool's warming rows again under the standing generation,
    /// at the last window's instant: it may rule a pool by other numbers,
    /// or rule none.
    fn rule_indexes(&mut self) {
        let Self {
            derived,
            working,
            spec,
            reading,
            indexes,
            origin,
            last_window,
            ..
        } = self;
        let now = last_window.map_or(0, |at| micros(*origin, at));
        let mut next = None;
        for (model, sides) in &derived.by_model {
            for &pool in sides.iter().flatten() {
                let at = pool as usize;
                let (Some(Some(held)), Some(rows)) = (indexes.get(at), working.get(at)) else {
                    continue;
                };
                let rule = ruling_of(spec, model, &rows.rows);
                if let Some(read) = held.again(rule, now, reading) {
                    put_read(&mut next, indexes, at, read);
                }
            }
        }
        if let Some(next) = next {
            *indexes = next.into();
        }
    }

    /// Read each ruled pool's warming rows again at the last window's
    /// instant, under the ruling it was read by. A pool no ruling names
    /// warms nobody, so a window reads nothing of it. The list is new only
    /// when a pool's rows moved: a publication that moved none allocates
    /// nothing.
    fn read_indexes(&mut self) {
        let Self {
            reading,
            indexes,
            origin,
            last_window,
            ..
        } = self;
        let mut next = None;
        for (at, held) in indexes.iter().enumerate() {
            let Some(held) = held else {
                continue;
            };
            let Some(rule) = held.rule() else {
                continue;
            };
            let now = last_window.map_or(0, |at| micros(*origin, at));
            if let Some(read) = held.again(Some(rule), now, reading) {
                put_read(&mut next, indexes, at, read);
            }
        }
        if let Some(next) = next {
            *indexes = next.into();
        }
    }

    fn publish_window(&mut self, at: Instant) -> Arc<World> {
        self.last_window = Some(at);
        // A veto that ended at or before this window's instant leaves the
        // table here and waits for no gateway's clear: a decision walks
        // only what stands.
        if let Some(vetoes) = self.vetoes.standing_at(at) {
            self.vetoes = Arc::new(vetoes);
            self.derive_vetoes();
        }
        self.read_indexes();
        self.publish()
    }
}

/// Sets pool `at`'s standing to `read` in `next`, the list a writer
/// publishes in `standing`'s place: a copy of it, made at the first pool
/// whose rows moved.
fn put_read(
    next: &mut Option<Vec<Option<Arc<PoolIndex>>>>,
    standing: &[Option<Arc<PoolIndex>>],
    at: usize,
    read: PoolIndex,
) {
    let next = next.get_or_insert_with(|| standing.to_vec());
    if let Some(slot) = next.get_mut(at) {
        *slot = Some(Arc::new(read));
    }
}

/// The warm-up ruling of the profile a pool's decisions resolve to under
/// `spec`, by the pool's model and its role's phase: as a request of that
/// phase asked alone resolves its profile, not by the pair binding a decode
/// pool's thresholds take
/// ([`Engine::pool_binding`](crate::framework::Engine::pool_binding)). The
/// hand names rows for such a request alone; a pair's sides read their
/// columns. `None` where the profile's warm-up stages rule by different
/// numbers: it is named no rows.
fn ruling_of(spec: &AppliedSpec, model: &str, rows: &[Candidate]) -> Option<WarmRule> {
    let role = rows.first()?.role;
    spec.engine().warm_rule_for(model, phase_of(role))
}

/// How many of `rows` hold no report: the pass a membership write takes over
/// each pool it builds. Every later write moves the pool's count by the row
/// it folds ([`rows_at`]), so no publication and no decision takes the pass:
/// a decision reads whether the count is 0 ([`PoolFacts::reported`]).
fn unreported(rows: &[Candidate]) -> u32 {
    rows.iter().filter(|row| row.load.is_none()).count() as u32
}

/// Writes `rows` over `over` when `over` is of `rows`' length and the
/// writer holds the last reference to it: no world lists it, so nothing
/// can read it. Whether it did.
fn written_over(over: &mut Arc<[Candidate]>, rows: &[Candidate]) -> bool {
    if over.len() != rows.len() {
        return false;
    }
    let Some(over) = Arc::get_mut(over) else {
        return false;
    };
    over.copy_from_slice(rows);
    true
}

/// The first index at or after `from` whose worker id is not under `id`,
/// in a list sorted by worker id: a few steps on foot, then a search of
/// what is left. A walk over ids in rising order pays for the distance it
/// moves: two sorted lists are joined in one pass, and a short list still
/// skips a long one by search.
fn seek<T>(sorted: &[T], from: usize, id: WorkerId, id_of: impl Fn(&T) -> WorkerId) -> usize {
    const ON_FOOT: usize = 8;
    let rest = sorted.get(from..).unwrap_or_default();
    let (near, far) = rest.split_at(rest.len().min(ON_FOOT));
    match near.iter().position(|entry| id_of(entry) >= id) {
        Some(step) => from + step,
        None => from + near.len() + far.partition_point(|entry| id_of(entry) < id),
    }
}

/// Hand `fold` each row of member `id` that stands at `rank` in the working
/// copy; a pool is marked written where `fold` says it wrote, and its count
/// of rows without a report ([`Working::unreported`]) moves where `fold`
/// gave a row its first report or took its last. Where the
/// member's first place stands when a row was handed, and `None` where no
/// row of the member stands at the rank: the member has no place yet, or
/// its rows stand at another rank. A load write folds a rank's record by
/// this walk and a membership build asks it which records no row took, so
/// the two read one rule.
fn rows_at(
    places: &[Place],
    working: &mut [Working],
    id: WorkerId,
    rank: u16,
    mut fold: impl FnMut(&mut Candidate) -> bool,
) -> Option<u32> {
    let first = places.partition_point(|place| place.id < id);
    let of_member = places.get(first..).unwrap_or_default();
    let mut at = None;
    for place in of_member.iter().take_while(|place| place.id == id) {
        let Some(pool) = working.get_mut(place.pool as usize) else {
            continue;
        };
        let Some(row) = pool.rows.get_mut(place.row as usize) else {
            continue;
        };
        if row.dp_rank != rank {
            continue;
        }
        let held = row.load.is_some();
        pool.dirty |= fold(row);
        match (held, row.load.is_some()) {
            (false, true) => pool.unreported = pool.unreported.saturating_sub(1),
            (true, false) => pool.unreported = pool.unreported.saturating_add(1),
            _ => {}
        }
        at = Some(first as u32);
    }
    at
}

/// The places of worker `id` in a list sorted by worker id.
fn places_of(places: &[Place], id: WorkerId) -> impl Iterator<Item = Place> + '_ {
    let from = places.partition_point(|place| place.id < id);
    places
        .get(from..)
        .unwrap_or_default()
        .iter()
        .take_while(move |place| place.id == id)
        .copied()
}
