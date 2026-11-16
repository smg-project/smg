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
//! What a membership write derives (where each pool stands, each pool's
//! shape) is built once and shared by every world until the next membership
//! write. A world's epoch is the membership's and is the one thing to
//! compare between two worlds' pools: a world's identity moves with every
//! publication. The pair classes are derived with the shapes where a write
//! moves a member or an offer, and once more where a generation moves the
//! pairing mode: a new world of the same epoch. Between two membership
//! writes a publication derives nothing: it is one small object, a pointer
//! per pool, and new rows for each pool a write was folded into.
//!
//! **A pair class is a fact of the rows.** A model's pair classes come from
//! what its prefills and decodes report ([`PairClasses`]), and a membership
//! write holds them before any row is built: a worker's row carries its one
//! class, and the model's table says which classes pair. The table names
//! classes, never rows, so a pool with another gateway's host-local rows
//! left out reads the same table. A write that moves one worker or its
//! offer derives the classes for that worker alone, in the models it left
//! and joined, from what the writer keeps of each model's derivation. A
//! write that moves no member and no member's offer carries the last
//! write's classes and reads no member's facts for them.
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
//! published together once per [`PUBLISH_WINDOW`], and a status change, a
//! veto or a membership change at once, through the same door, with
//! whatever else waits. A load write publishes by itself when the last
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
//! publication before. A membership write, which moves the rows, and a
//! window's publication that drops a veto past its end build every run
//! again. A veto leaves with its worker's row in the membership write's own
//! step, so no publication shows the row without it.

use std::{
    mem,
    sync::Arc,
    time::{Duration, Instant},
};

use rustc_hash::FxHashMap;

use crate::{
    framework::{OverloadThresholds, PairPools, PairTable},
    spec::{AppliedSpec, Pairing},
    store::{
        loads::LoadSample,
        pairing::{known, Derivation, Member, PairClasses, Pool},
        workers::{Overload, Reachability, WorkerFacts, WorkerRow, WorkerStatus},
        GatewayId,
    },
    view::{
        fingerprint, AffinityFacts, Candidate, Health, OverloadRegime, Phase, PoolFacts, Role,
        WorkerId,
    },
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

    fn clear(&mut self) {
        for side in [&mut self.prefill, &mut self.decode] {
            side.candidates.clear();
            side.affinity.clear();
        }
        self.ends.clear();
        self.listed.clear();
    }
}

/// What a build read beside the rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Built {
    /// The rows written: the pool as the asking gateway reaches it.
    pub pool_size: u32,
    /// The store's epoch at the membership write the world was built by.
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
    /// where the rows carry their stalled bits.
    pub stalled: &'a [u32],
    /// One more than the position of the pinned worker's row: what
    /// [`RequestFacts::pinned_at`](crate::RequestFacts::pinned_at) takes.
    /// Zero where no pinned worker is in the pool and where the rows carry
    /// their session bits.
    pub pinned_at: u32,
    /// The affinity column: one entry a row under a profile that declared
    /// it, empty otherwise.
    pub affinity: &'a [AffinityFacts],
    /// What the view read beside the rows.
    pub built: Built,
}

/// One pool's shape: what a membership write derives of it and no
/// publication changes.
struct PoolShape {
    /// Each row's admission instant, in the rows' clock.
    admitted_micros: Box<[u64]>,
    /// The host-local rows, by index, each with the gateway that holds its
    /// socket; empty for a pool every gateway dials whole.
    locals: Box<[(u32, GatewayId)]>,
    /// The pair classes of the pool's model, on its prefill pool where the
    /// model has a decode too: a pair build reads them by the one lookup
    /// that finds its two pools. Counted: a membership write hands them to
    /// the shape it derives where they stand, a write that moves no offer
    /// ([`Classes::Carry`]) and a written worker's derive that leaves them
    /// as they were ([`Classes::Written`]).
    pairs: Option<Arc<PairClasses>>,
}

/// The roles a model's pools are derived for: one slot each.
const ROLES: usize = 4;

/// A role's slot among a model's pools. A role added to the enum stops the
/// build at this match until it has an arm. The compiler checks the arm
/// alone: [`ROLES`] is raised by hand with the new slot, and a slot past it
/// panics at the first membership write that derives that role's pool.
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

/// What one membership write derives: where each pool stands in a world's
/// rows, and each pool's shape. Built once per membership write, and once
/// more where a generation moves the pairing mode, and shared by every
/// world published until the next.
#[derive(Default)]
struct Derived {
    /// The pools by model, then by role: each an index into a world's rows.
    /// A pool is every member serving one model in one role, whatever its
    /// status (a status change is a health signal and moves no position).
    by_model: FxHashMap<Box<str>, [Option<u32>; ROLES]>,
    pools: Box<[PoolShape]>,
}

impl Derived {
    /// The pair classes of `model`, where it has a prefill and a decode.
    fn pairs(&self, model: &str) -> Option<&Arc<PairClasses>> {
        let prefill = self.by_model.get(model)?[role_slot(Role::Prefill)]?;
        self.pools.get(prefill as usize)?.pairs.as_ref()
    }

    /// How many members `pool` holds: none where there is no pool.
    fn members(&self, pool: Option<u32>) -> usize {
        pool.and_then(|pool| self.pools.get(pool as usize))
            .map_or(0, |shape| shape.admitted_micros.len())
    }
}

/// One standing veto as its gateway's run in a pool carries it: the row it
/// masks, by index and by id, and its end. The index finds the row in a
/// pool every gateway copies whole, the id in a pool that holds host-local
/// rows, whichever gateway asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PoolVeto {
    row: u32,
    id: WorkerId,
    until: Instant,
}

/// One gateway's standing vetoes in one pool, sorted by row: never empty.
#[derive(Clone, Debug)]
struct GatewayRun {
    gateway: GatewayId,
    vetoes: Arc<[PoolVeto]>,
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
/// until `until`, or cleared: that gateway's run built again, each other
/// run shared.
fn with_veto(
    runs: &PoolVetoes,
    gateway: GatewayId,
    row: u32,
    id: WorkerId,
    until: Option<Instant>,
) -> PoolVetoes {
    let found = runs.binary_search_by_key(&gateway, |run| run.gateway);
    let own: &[PoolVeto] = match found {
        Ok(at) => runs.get(at).map(|run| &*run.vetoes).unwrap_or_default(),
        Err(_) => &[],
    };
    // Each arm is one pass of a known length: one allocation, the run's
    // own, and no entry of another gateway is read.
    let own: Arc<[PoolVeto]> = match (own.binary_search_by_key(&row, |veto| veto.row), until) {
        (Ok(at), Some(until)) => own
            .iter()
            .enumerate()
            .map(|(n, standing)| PoolVeto {
                until: if n == at { until } else { standing.until },
                ..*standing
            })
            .collect(),
        (Ok(at), None) => {
            let (before, after) = own.split_at(at);
            let after = after.get(1..).unwrap_or_default();
            before.iter().chain(after).copied().collect()
        }
        (Err(at), Some(until)) => {
            let (before, after) = own.split_at(at);
            before
                .iter()
                .copied()
                .chain(std::iter::once(PoolVeto { row, id, until }))
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
    let own = (!own.is_empty()).then_some(GatewayRun {
        gateway,
        vetoes: own,
    });
    before
        .iter()
        .cloned()
        .chain(own)
        .chain(after.iter().cloned())
        .collect()
}

/// The gateways' vetoes as the writer keeps them: per gateway its vetoed
/// workers by id, each until an instant. Replaced whole at each of the
/// four writes that change them, all rare: a veto's write, a gateway's
/// leaving that takes its whole list, a membership write that takes a
/// departed worker's vetoes with its row, and a window's publication that
/// drops a veto past its end. A world carries what is
/// derived from them: each pool's runs by gateway ([`PoolVetoes`]).
#[derive(Debug, Default)]
struct Vetoes {
    by_gateway: FxHashMap<GatewayId, Arc<[(WorkerId, Instant)]>>,
}

impl Vetoes {
    /// The vetoes with `gateway`'s veto on `id` set until `until`, or
    /// cleared; `None` when that changes nothing.
    fn with(&self, gateway: GatewayId, id: WorkerId, until: Option<Instant>) -> Option<Self> {
        let mut list: Vec<(WorkerId, Instant)> = self
            .by_gateway
            .get(&gateway)
            .map(|list| list.to_vec())
            .unwrap_or_default();
        match (list.binary_search_by_key(&id, |&(worker, _)| worker), until) {
            (Ok(at), Some(until)) if list[at].1 == until => return None,
            (Ok(at), Some(until)) => list[at].1 = until,
            (Ok(at), None) => {
                list.remove(at);
            }
            (Err(at), Some(until)) => list.insert(at, (id, until)),
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
        let stands = |&(_, until): &(WorkerId, Instant)| until > at;
        if self.by_gateway.values().all(|list| list.iter().all(stands)) {
            return None;
        }
        let mut by_gateway = FxHashMap::default();
        for (gateway, list) in &self.by_gateway {
            let kept: Vec<(WorkerId, Instant)> = list.iter().copied().filter(stands).collect();
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
            list: &'a [(WorkerId, Instant)],
            members: &'a [WorkerId],
        ) -> impl Iterator<Item = (WorkerId, Instant)> + 'a {
            let mut at = 0;
            list.iter().copied().filter(move |&(id, _)| {
                at = seek(members, at, id, |&member| member);
                members.get(at) == Some(&id)
            })
        }
        let whole = |list: &Arc<[(WorkerId, Instant)]>| kept(list, members).count() == list.len();
        if self.by_gateway.values().all(whole) {
            return None;
        }
        let mut by_gateway = FxHashMap::default();
        for (gateway, list) in &self.by_gateway {
            if whole(list) {
                by_gateway.insert(*gateway, Arc::clone(list));
                continue;
            }
            let kept: Vec<(WorkerId, Instant)> = kept(list, members).collect();
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
/// hold: each reads position 0 in that field. Counted at every write that
/// reads the members under the lists, a membership write and an apply that
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
    /// The generation this publication was made under: its engine and its
    /// number, read with the rows in one load.
    spec: Arc<AppliedSpec>,
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

    fn pool(&self, model: &str, role: Role) -> Option<(&[Candidate], &PoolShape)> {
        let index = self.index(model, role)?;
        Some((self.rows.get(index)?, self.derived.pools.get(index)?))
    }

    fn index(&self, model: &str, role: Role) -> Option<usize> {
        Some(self.derived.by_model.get(model)?[role_slot(role)]? as usize)
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
    /// report's age at `ask.now`, the stalled bit where a veto of the
    /// asking gateway is in this publication and stands at that instant,
    /// whichever kind of veto it is, and the session bit on the pinned
    /// worker. A veto past its end masks nothing and costs the build one
    /// compare until a window's publication drops it; a store that
    /// publishes no window keeps it. A window's publication drops every
    /// veto ended by its own instant, so a decision dated before that
    /// instant finds a veto that still stood at its date already gone: a
    /// decision reads one publication, and its date restores nothing the
    /// publication dropped. The health sink's own answer keeps its compare.
    /// Under `ask.affinity` the column holds one entry per candidate with
    /// the worker's age since admission.
    ///
    /// What no fact of the store feeds yet reads as the row's default: the
    /// in-flight and session counts, the two admission bits, the breaker
    /// bit, the four positions into declared lists, the
    /// vision and encode bits, and the column's four index facts. The
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
    /// nothing. The pinned worker's row is handed as one more than its
    /// position. Under `ask.affinity` the column is filled as the build
    /// fills it. A row's report age stays what the report carried when it
    /// was recorded: the age at the decision is the decision's instant less
    /// the row's sample instant, and no stage of this crate reads it.
    ///
    /// The rows are copied and written ([`World::build`]) where they cannot
    /// be lent: in a pool that holds host-local rows, where the asking
    /// gateway's positions are its own, and for a decision that asks for
    /// `written` rows. A profile that shows its rows to a plugin
    /// ([`Profile::scoped`](crate::Profile::scoped)) asks for them: a
    /// plugin's view carries each row's bits and ages as the row's fields.
    /// Both forms come back as one shape, so a caller hands the mask and
    /// the pin on whichever rows it was given.
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
            return Some(PoolView {
                candidates: &out.candidates,
                stalled: &[],
                pinned_at: 0,
                affinity: &out.affinity,
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
            let standing = run.vetoes.iter().filter(|veto| veto.until > ask.now);
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
            built: Built {
                pool_size: rows.len() as u32,
                membership_epoch: self.epoch,
                now_micros,
                generation: self.spec.generation(),
                reach,
            },
        })
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
}

/// Write the stalled bit on each row of `rows` a veto of `run` masks at
/// `now`, and say whether one did.
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
        if veto.until <= now {
            continue;
        }
        let row = if whole {
            rows.get_mut(veto.row as usize)
        } else {
            row_of_id(rows, veto.id)
        };
        if let Some(row) = row {
            row.health.stalled = true;
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

/// The candidate row of one member: its fixed fields from its registration,
/// its status bits, and its load record when one is on file. Its pair class
/// is the membership's to derive ([`pair_up`]), and its engine family, zone
/// and labels are the standing generation's lists' to say
/// ([`Writer::list`]).
fn row_of(member: &WorkerRow, sample: Option<&LoadSample>, origin: Instant) -> Candidate {
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
        // The low 32 bits of the registration key's fingerprint.
        key: fingerprint(facts.key.as_bytes()) as u32,
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
}

/// The list a membership write fills for a pool that stood at `standing`
/// members: room for them and one more, the most a membership write adds a
/// pool, so the list is allocated once. A pool that did not stand takes no
/// room, and its list grows as it fills.
fn list_of<T>(standing: usize) -> Vec<T> {
    if standing == 0 {
        Vec::new()
    } else {
        Vec::with_capacity(standing + 1)
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

/// What a membership write asks of the pair classes.
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
    /// member's facts are read for them.
    Carry,
}

/// Whether two versions of one worker's facts stand in the same pools with
/// the same offer: the role and the models place a worker in its pools, and
/// the wires, the runtime and the descriptor are what the pairing rule reads of
/// it, the runtime as an offer holds it ([`known`]). A write from one such
/// version to the other moves no pair class.
pub(super) fn same_offer(old: &WorkerFacts, new: &WorkerFacts) -> bool {
    old.role == new.role
        && old.models == new.models
        && old.wire_modes == new.wire_modes
        && known(&old.engine_kind) == known(&new.engine_kind)
        && old.pairing == new.pairing
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
    /// A row of the engine has a threshold. An engine with none is not
    /// judged at a write: its rows' bits are clear and stay so.
    judged: bool,
    /// The last word of the engine's poll was that no report came: its
    /// verdict reads clear, whatever is on file, until its next record or
    /// report. It outlives a membership write, which judges again, and it
    /// is kept and ended while no threshold stands.
    silent: bool,
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

/// The write side of the world, behind the store's load lock: the working
/// copy of every pool, where each member's rows stand in them, what the
/// last publication held, and the publication window. Every write that
/// publishes hands back the world it made; the store shows it under the
/// same lock, so the worlds reach the deciders in the order they were made.
pub(super) struct Writer {
    window: Duration,
    origin: Instant,
    /// The store's epoch at the last membership write.
    epoch: u64,
    derived: Arc<Derived>,
    /// Each model's derivation of its pair classes as the last membership
    /// write left it, for the models that hold a prefill and a decode: the
    /// next write of one worker derives for that worker alone.
    derivations: FxHashMap<Box<str>, Derivation>,
    /// Each pool's rows as the last publication left them.
    published: Vec<Arc<[Candidate]>>,
    /// Each pool's spare: the rows it published before its last, kept for
    /// its next publication to write over; none before the pool's first
    /// publication after a membership write. Beside the working copies,
    /// not in them: a load write's step from pool to pool is as it was.
    spares: Vec<Option<Arc<[Candidate]>>>,
    working: Vec<Working>,
    /// Rows that left their pool and that a world may still list: a spare a
    /// publication found listed by a world some reader holds, and the
    /// spares and the published rows of the pools a membership write
    /// retired. The writer never lets go of rows a world lists, so a
    /// decision that drops the last world listing them frees no rows.
    held: Vec<Arc<[Candidate]>>,
    /// The held rows a publication found no world listing any more, for
    /// that publication alone: a pool of their length with no spare to
    /// write over takes them, and the publication frees the rest.
    unlisted: Vec<Arc<[Candidate]>>,
    /// How many times a publication allocated a pool's rows.
    allocations: u64,
    /// Every replaced world a reader still held when the cell let it go
    /// ([`Self::replaced`]), until a publication finds no reader holding it
    /// and lets it go. The cell holds the standing world and the writer
    /// each of these, so a decision that drops the last reader's hold
    /// frees nothing: not the world, its list of pools, its vetoes, its
    /// derived tables nor its generation.
    worlds: Vec<Arc<World>>,
    /// Every member's rows, sorted by worker id: a worker serving several
    /// models has a row in each model's pool.
    places: Vec<Place>,
    vetoes: Arc<Vetoes>,
    /// The vetoes as the world carries them, by pool and by gateway. A
    /// veto's write builds its own gateway's runs again; a membership write
    /// and a window's publication that drops a veto past its end build
    /// every run again.
    pool_vetoes: Arc<[PoolVetoes]>,
    /// The generation every world is published under until the next is
    /// applied: the unbound one at the start.
    spec: Arc<AppliedSpec>,
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
    /// How many writes to the load table moved an overload verdict.
    flips: u64,
    /// The standing members outside the standing generation's lists.
    outside: OutsideLists,
    /// The last publication the window counts from: a window's end, or a
    /// load write that published by itself.
    last_window: Option<Instant>,
    /// A folded load record waits for the window's end.
    waiting: bool,
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
            spares: Vec::new(),
            working: Vec::new(),
            held: Vec::new(),
            unlisted: Vec::new(),
            allocations: 0,
            worlds: Vec::new(),
            places: Vec::new(),
            vetoes: Arc::default(),
            pool_vetoes: Arc::default(),
            spec: Arc::new(AppliedSpec::unbound()),
            retired: None,
            engines: Vec::new(),
            engine_rows: Vec::new(),
            place_engines: Vec::new(),
            place_slots: Vec::new(),
            heard: Vec::new(),
            sample_bound: None,
            judging: false,
            silenced: 0,
            flips: 0,
            outside: OutsideLists::default(),
            last_window: None,
            waiting: false,
        }
    }

    /// A world of everything published so far: one small object, a pointer
    /// per pool. Nothing derived is built here.
    pub(super) fn world(&self) -> Arc<World> {
        Arc::new(World {
            epoch: self.epoch,
            origin: self.origin,
            derived: Arc::clone(&self.derived),
            rows: self.published.clone().into_boxed_slice(),
            vetoes: Arc::clone(&self.pool_vetoes),
            spec: Arc::clone(&self.spec),
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
    /// of another mode derives `members`, the standing membership, again by
    /// the membership write's own door ([`Writer::rebuild`]), which reads
    /// each member under the lists that stand, and publishes once, under
    /// the new generation at the standing epoch: the epoch is the
    /// membership's and the generation the document's, and what is derived
    /// from both is published under both.
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
        members: impl Iterator<Item = &'a WorkerRow>,
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
            self.rebuild(self.epoch, members, sample_of, Classes::Derive)
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
            self.publish()
        };
        Ok((world, before))
    }

    /// Read every member under the standing generation's lists: its engine
    /// family and its zone as positions, its labels as bits, in each of
    /// its rows. For a membership write, whose rows are new, and for an
    /// apply whose lists differ. A pool with a row that moved is published
    /// whole.
    fn list<'a>(&mut self, members: impl Iterator<Item = &'a WorkerRow>) {
        let Self {
            places,
            working,
            spec,
            outside,
            ..
        } = self;
        let lists = spec.lists();
        *outside = OutsideLists::default();
        for member in members {
            let facts = &member.facts;
            let listed = lists.listed(&facts.engine_kind, facts.zone.as_deref(), &facts.labels);
            outside.engines += u32::from(listed.engine_outside);
            outside.zones += u32::from(listed.zone_outside);
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

    /// Judge every engine again from what is on file. A membership write's
    /// rows start with no bit, and an applied generation may hand any pool
    /// another pair or none: an engine that had a threshold is judged once
    /// more under what it has now, so a bit no threshold holds is cleared.
    fn judge_every(&mut self) {
        self.judging = false;
        for index in 0..self.engines.len() {
            let has = self.has_threshold(index);
            self.judging |= has;
            let Some(engine) = self.engines.get_mut(index) else {
                continue;
            };
            let had = mem::replace(&mut engine.judged, true);
            if had || has {
                self.judge(index as u32);
            }
            if let Some(engine) = self.engines.get_mut(index) {
                engine.judged = has;
            }
        }
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
    /// has a threshold is not judged.
    fn judge(&mut self, engine: u32) -> bool {
        let Self {
            engines,
            engine_rows,
            working,
            heard,
            sample_bound,
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
    /// judged once ([`Self::judged`]). While no engine has a threshold
    /// and none is silent there is no word to end and no verdict to
    /// move, and the write reads no engine.
    pub(super) fn recorded(&mut self, place: u32, at: Instant) -> bool {
        if !self.judging && self.silenced == 0 {
            return false;
        }
        let Some(engine) = self.engine_at(place) else {
            return false;
        };
        self.hear(place, at);
        self.judged(engine, true)
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

    /// The word that no report came for the engine: from here its verdict
    /// reads clear until its next record or report, and the engine is
    /// judged once. The word is kept for an engine with no threshold too,
    /// which is not judged. Whether a verdict moved.
    pub(super) fn silence(&mut self, engine: u32) -> bool {
        let Some(rows) = self.engines.get_mut(engine as usize) else {
            return false;
        };
        let was = mem::replace(&mut rows.silent, true);
        self.silenced += u32::from(!was);
        self.judged(engine, false)
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
    /// caller says (`classes`).
    pub(super) fn rebuild<'a>(
        &mut self,
        epoch: u64,
        members: impl Iterator<Item = &'a WorkerRow>,
        sample_of: impl Fn(WorkerId, u16) -> Option<LoadSample>,
        classes: Classes,
    ) -> Arc<World> {
        let origin = self.origin;
        // The last write's classes: carried, or what the written worker's
        // derive starts from.
        let last = (classes != Classes::Derive).then(|| Arc::clone(&self.derived));
        // An engine whose last poll brought no report stays silent across
        // the write: its verdict is not raised again from what is still on
        // file. The word is carried by its members' ids, which the outgoing
        // places hold in order. The list is empty while no engine is
        // silent.
        let mut silent: Vec<WorkerId> = Vec::new();
        if self.silenced > 0 {
            let engines = &self.engines;
            let of_silent = |&(_, &engine): &(&Place, &u32)| {
                engines
                    .get(engine as usize)
                    .is_some_and(|engine| engine.silent)
            };
            let outgoing = self.places.iter().zip(&self.place_engines);
            silent.extend(outgoing.filter(of_silent).map(|(place, _)| place.id));
            silent.dedup();
        }
        // The ranks of the data-parallel engines, each with its engine's
        // group by key and the receipt of its sample on file. A fleet of
        // single-rank workers lists none.
        let mut by_key: FxHashMap<&'a str, u32> = FxHashMap::default();
        let mut ranked: Vec<(WorkerId, u32, u64)> = Vec::new();
        // A member names a threshold of its own.
        let mut owned = false;
        let mut groups: FxHashMap<Box<str>, [Vec<Entry<'a>>; ROLES]> = FxHashMap::default();
        // The members' ids are read by the vetoes alone: a membership write
        // with no veto standing does not collect them.
        let standing = !self.vetoes.by_gateway.is_empty();
        let mut ids = Vec::new();
        // The members in id order, by this write's one sort, each id beside
        // its row: an id is one member, so every pool's entries, the ranks
        // and the ids are pushed in order and nothing below sorts them. The
        // sort moves an id and a reference, never a whole entry.
        let mut members: Vec<(WorkerId, &'a WorkerRow)> =
            members.map(|member| (member.id, member)).collect();
        members.sort_unstable_by_key(|&(id, _)| id);
        for &(_, member) in &members {
            if standing {
                ids.push(member.id);
            }
            let facts = &member.facts;
            let sample = sample_of(member.id, facts.dp_rank);
            // The ranks of a data-parallel engine share its key. A
            // single-rank worker's engine is its own key, which no other
            // row has: it is an engine alone, and no key is looked up.
            if facts.dp_size > 1 {
                let next = by_key.len() as u32;
                let group = *by_key.entry(facts.engine.as_str()).or_insert(next);
                let heard = sample.map_or(0, |sample| micros(origin, sample.received_at));
                ranked.push((member.id, group, heard));
            }
            owned |= facts.overload != Overload::default();
            let entry = Entry {
                row: row_of(member, sample.as_ref(), origin),
                facts,
                admitted_micros: micros(origin, member.admitted_at),
                local: match facts.reachable_from {
                    Reachability::Any => None,
                    Reachability::Gateway(gateway) => Some(gateway),
                },
                own: facts.overload,
            };
            let side = role_slot(facts.role);
            for model in &facts.models {
                match groups.get_mut(model.as_str()) {
                    // A model listed twice is one pool's one row. A member's
                    // entry is the last of its side until the next member is
                    // read, so an earlier listing is found there: no model
                    // before this one is read again.
                    Some(sides) => {
                        let list = &mut sides[side];
                        if list.last().is_none_or(|last| last.row.id != member.id) {
                            list.push(entry);
                        }
                    }
                    None => {
                        // Each side takes room for its pool's last members
                        // and the one this write may add.
                        let before = self.derived.by_model.get(model.as_str());
                        let room = before.map_or([0; ROLES], |pools| {
                            pools.map(|pool| self.derived.members(pool))
                        });
                        let mut sides: [Vec<Entry<'a>>; ROLES] = room.map(list_of);
                        sides[side].push(entry);
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
        for (model, mut sides) in groups {
            // Every side stands in id order, as the walk above pushed it,
            // before a class is derived: the classes are numbered in it, and
            // each row is built with its class.
            debug_assert!(sides
                .iter()
                .all(|entries| entries.is_sorted_by(|a, b| a.row.id < b.row.id)));
            // The model's classes go on its prefill pool's shape: a model
            // that has classes has that pool.
            let carried = last.as_deref().and_then(|last| last.pairs(&model));
            let (mode, kept) = (self.spec.pairing(), &mut self.derivations);
            let mut pairs = pair_up(&mut sides, mode, classes, carried, kept, &model);
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
                });
                self.spares.push(None);
                self.working.push(Working {
                    rows,
                    dirty: false,
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
        // Each row's engine family, zone and labels under the standing
        // generation's lists, before the rows are cut for the publication.
        // The members are walked once more where the generation declares a
        // list. One with none counts no member outside a list: an apply
        // that derives the classes again may have dropped the last one's.
        if self.spec.lists().is_empty() {
            self.outside = OutsideLists::default();
        } else {
            self.list(members.iter().map(|&(_, member)| member));
        }
        // The engines, from the places, which stand by member: a member's
        // first place is where its record is read, each is where a verdict
        // is written. The writer's lists are written over in place, so the
        // write allocates for no engine.
        let mut engines = mem::take(&mut self.engines);
        engines.clear();
        self.engine_rows.clear();
        self.place_engines.clear();
        self.place_slots.clear();
        let carries = !silent.is_empty();
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
                    let carried = carries && silent.binary_search(&place.id).is_ok();
                    silenced += u32::from(carried);
                    engines.push(EngineRows {
                        at: self.engine_rows.len() as u32,
                        len: 0,
                        judged: false,
                        silent: carried,
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
                        let carried = carries && silent.binary_search(&place.id).is_ok();
                        silenced += u32::from(carried && !rows.silent);
                        rows.silent |= carried;
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
        self.derived = Arc::new(Derived {
            by_model,
            pools: pools.into_boxed_slice(),
        });
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
        // verdict on it or none.
        if self.bind() || owned {
            self.judge_every();
        } else {
            self.judging = false;
        }
        for pool in &mut self.working {
            pool.dirty = false;
            self.published.push(Arc::from(pool.rows.as_slice()));
        }
        self.publish()
    }

    /// Every pool's runs built again from the gateways' lists and the
    /// members' places: for a membership write, which moves the rows, and
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
            for &(id, until) in list.iter() {
                at = seek(&self.places, at, id, |place| place.id);
                while let Some(place) = self.places.get(at).filter(|place| place.id == id) {
                    at += 1;
                    let Some(pool) = pools.get_mut(place.pool as usize) else {
                        continue;
                    };
                    let veto = PoolVeto {
                        row: place.row,
                        id,
                        until,
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
                    .map(|(gateway, vetoes)| GatewayRun {
                        gateway,
                        vetoes: vetoes.into(),
                    })
                    .collect()
            })
            .collect();
    }

    /// One veto's write as the pools carry it: in each pool the worker has
    /// a row in, the writing gateway's run takes the veto, moves its end
    /// or loses it. No other gateway's run and no other pool's is built
    /// again: a publication shares them with the one before.
    fn write_veto(&mut self, gateway: GatewayId, id: WorkerId, until: Option<Instant>) {
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
                    Some(place) => with_veto(runs, gateway, place.row, id, until),
                    None => Arc::clone(runs),
                }
            })
            .collect();
    }

    /// Fold one rank's load record, or its absence, into the member's rows;
    /// where the member's first place stands when a row took it. The
    /// caller judges the member's engine once its write is whole
    /// ([`Writer::recorded`], [`Writer::judged`]) and says when that is
    /// ([`Writer::folded_at`]).
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
        let first = places.partition_point(|place| place.id < id);
        let of_member = places.get(first..).unwrap_or_default();
        let mut folded = None;
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
            row.load = sample.map(|sample| sample.row);
            row.load_sampled_at_micros =
                sample.map_or(0, |sample| micros(*origin, sample.sampled_at));
            pool.dirty = true;
            folded = Some(first as u32);
        }
        folded
    }

    /// A load write folded its rows at `at`: it publishes by itself when
    /// the last window is a window old, else it waits for the window's end.
    /// The world to show, when it published.
    #[must_use]
    pub(super) fn folded_at(&mut self, at: Instant) -> Option<Arc<World>> {
        match self.last_window {
            Some(last) if self.window_end(last).is_none_or(|end| at < end) => {
                self.waiting = true;
                None
            }
            _ => Some(self.publish_window(at)),
        }
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

    /// Set `gateway`'s veto on `id` until `until`, or clear it, and publish
    /// at once; `None` when the write changes no veto.
    #[must_use]
    pub(super) fn veto(
        &mut self,
        gateway: GatewayId,
        id: WorkerId,
        until: Option<Instant>,
    ) -> Option<Arc<World>> {
        let vetoes = self.vetoes.with(gateway, id, until)?;
        self.vetoes = Arc::new(vetoes);
        self.write_veto(gateway, id, until);
        Some(self.publish())
    }

    /// Lift every veto of `gateway` and publish at once: the gateway is
    /// gone past its grace period. Every pool's runs are built again, as
    /// for a membership write. `None` when the gateway held no veto.
    #[must_use]
    pub(super) fn drop_gateway(&mut self, gateway: GatewayId) -> Option<Arc<World>> {
        if !self.vetoes.by_gateway.contains_key(&gateway) {
            return None;
        }
        let mut by_gateway = self.vetoes.by_gateway.clone();
        by_gateway.remove(&gateway);
        self.vetoes = Arc::new(Vetoes { by_gateway });
        self.derive_vetoes();
        Some(self.publish())
    }

    /// When the waiting folds are due: the end of the window the last
    /// publication opened; `None` when nothing waits, and for a window the
    /// clock cannot add.
    pub(super) fn due(&self) -> Option<Instant> {
        if !self.waiting {
            return None;
        }
        self.last_window.and_then(|last| self.window_end(last))
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
        let pools = self.working.iter_mut().zip(&mut self.published);
        for ((pool, published), spare) in pools.zip(&mut self.spares) {
            if !pool.dirty {
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
        }
        // Rows no world lists and no pool took are freed here, by a write.
        self.unlisted.clear();
        self.waiting = false;
        self.world()
    }

    /// How many times a publication allocated a pool's rows: it found
    /// neither the pool's spare nor held rows of the pool's length free of
    /// every world. A membership write's own rows are not counted.
    pub(super) fn allocations(&self) -> u64 {
        self.allocations
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

    fn publish_window(&mut self, at: Instant) -> Arc<World> {
        self.last_window = Some(at);
        // A veto that ended at or before this window's instant leaves the
        // table here and waits for no gateway's clear: a decision walks
        // only what stands.
        if let Some(vetoes) = self.vetoes.standing_at(at) {
            self.vetoes = Arc::new(vetoes);
            self.derive_vetoes();
        }
        self.publish()
    }
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
