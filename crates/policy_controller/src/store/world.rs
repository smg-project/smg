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
//! compare: a world's identity moves with every publication. Between two
//! membership writes a publication derives nothing: it is one small object,
//! a pointer per pool, and new rows for each pool a write was folded into.
//!
//! A row in a pool is the candidate row itself. Its fixed fields are
//! written when the membership's first world is built. Its moving facts
//! (the load record with its sample instant, the two status bits) are
//! folded into the writer's own copy of the pool, and a publication copies
//! each pool that moved, whole. A published world never changes: a decision
//! copies one run of memory nobody is writing and sees each row as one
//! publication left it, through no lock.
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
    sync::Arc,
    time::{Duration, Instant},
};

use rustc_hash::FxHashMap;

use crate::{
    spec::AppliedSpec,
    store::{
        loads::LoadSample,
        workers::{Reachability, WorkerRow, WorkerStatus},
        GatewayId,
    },
    view::{
        fingerprint, AffinityFacts, Candidate, Health, OverloadRegime, PoolFacts, Role, WorkerId,
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
        }
    }
}

/// One pool's shape: what a membership write derives of it and no
/// publication changes.
struct PoolShape {
    /// Each row's admission instant, in the rows' clock.
    admitted_micros: Box<[u64]>,
    /// The host-local rows, by index, each with the gateway that holds its
    /// socket; empty for a pool every gateway dials whole.
    locals: Box<[(u32, GatewayId)]>,
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

/// What one membership write derives: where each pool stands in a world's
/// rows, and each pool's shape. Built once per membership write and shared
/// by every world published until the next.
#[derive(Default)]
struct Derived {
    /// The pools by model, then by role: each an index into a world's rows.
    /// A pool is every member serving one model in one role, whatever its
    /// status (a status change is a health signal and moves no position).
    by_model: FxHashMap<Box<str>, [Option<u32>; ROLES]>,
    pools: Box<[PoolShape]>,
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

/// `gateway`'s run among a pool's runs, sorted by gateway. While the runs
/// are at most [`WALKED_RUNS`] the two ends are asked first, which is the
/// whole answer where one gateway alone holds a veto, and a gateway
/// between them is walked to from the nearer end: from the front to the
/// first gateway not under it, or from the back to the first not above
/// it. Above that count the run is found by one search.
fn run_of(runs: &[GatewayRun], gateway: GatewayId) -> Option<&GatewayRun> {
    if runs.len() > WALKED_RUNS {
        let at = runs
            .binary_search_by_key(&gateway, |run| run.gateway)
            .ok()?;
        return runs.get(at);
    }
    let (first, last) = (runs.first()?, runs.last()?);
    if gateway <= first.gateway {
        return (gateway == first.gateway).then_some(first);
    }
    if gateway >= last.gateway {
        return (gateway == last.gateway).then_some(last);
    }
    // The middle run says which half holds the gateway's.
    let middle = runs.get(runs.len() / 2)?;
    let run = if gateway < middle.gateway {
        runs.iter().find(|run| run.gateway >= gateway)
    } else {
        runs.iter().rev().find(|run| run.gateway <= gateway)
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
/// three writes that change them, all rare: a veto's write, a membership
/// write that takes a departed worker's vetoes with its row, and a window's
/// publication that drops a veto past its end. A world carries what is
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
    /// derived by: the one thing to compare between two worlds. A status
    /// change moves the store's epoch and no world's, and a publication
    /// between two membership writes is a new world of the same epoch.
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
    /// A door that decides is to hand the engine under the borrow its rows
    /// are built from, never an owned world, whose shared count a deciding
    /// thread would write. No door of this store decides yet.
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
    /// Each row is as this publication left it, with what is the
    /// decision's own written over it: its position in the slice, its
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
    /// in-flight and session counts, the two admission bits, the overload
    /// bit, the breaker bit, the four positions into declared lists, the
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
        out.candidates.clear();
        out.affinity.clear();
        let index = self.index(model, role)?;
        let (rows, shape) = (self.rows.get(index)?, self.derived.pools.get(index)?);
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
        // The asking gateway's run among the gateways that hold a veto in
        // this pool: no other gateway's veto is read.
        let own = self
            .vetoes
            .get(index)
            .and_then(|runs| run_of(runs, ask.gateway));
        if let Some(run) = own {
            // A pool every gateway dials whole is copied whole: a veto's
            // row is its position. Another gateway's host-local rows are
            // left out, so there the row is found by its id.
            let whole = shape.locals.is_empty();
            for veto in run.vetoes.iter() {
                if veto.until <= ask.now {
                    continue;
                }
                let row = if whole {
                    out.candidates.get_mut(veto.row as usize)
                } else {
                    row_of_id(&mut out.candidates, veto.id)
                };
                if let Some(row) = row {
                    row.health.stalled = true;
                }
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
        Some(Built {
            pool_size: out.candidates.len() as u32,
            membership_epoch: self.epoch,
            now_micros,
            generation: self.spec.generation(),
        })
    }
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
/// its status bits, and its load record when one is on file.
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
        pair_class: facts.pair_class,
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
    /// Each pool's rows as the last publication left them.
    published: Vec<Arc<[Candidate]>>,
    working: Vec<Working>,
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
            published: Vec::new(),
            working: Vec::new(),
            places: Vec::new(),
            vetoes: Arc::default(),
            pool_vetoes: Arc::default(),
            spec: Arc::new(AppliedSpec::unbound()),
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

    /// Publish every world from here on under `spec`, the first at once:
    /// the standing membership, its rows as they are, under the new
    /// generation. A number at or under the standing one is refused by
    /// naming the standing one, and nothing is published: the number only
    /// rises, so no reader sees a generation and then a lower one.
    pub(super) fn apply(&mut self, spec: Arc<AppliedSpec>) -> Result<Arc<World>, StaleGeneration> {
        let standing = self.spec.generation();
        if spec.generation() <= standing {
            return Err(StaleGeneration {
                standing,
                offered: spec.generation(),
            });
        }
        self.spec = spec;
        Ok(self.publish())
    }

    /// Derive the pools of the membership `members` at the store's `epoch`,
    /// each row with its load record on file, make them the working copy
    /// and publish them through the one door: every pool is new, so the
    /// door copies none, every earlier fold is in the world this returns
    /// and nothing waits.
    pub(super) fn rebuild<'a>(
        &mut self,
        epoch: u64,
        members: impl Iterator<Item = &'a WorkerRow>,
        sample_of: impl Fn(WorkerId, u16) -> Option<LoadSample>,
    ) -> Arc<World> {
        #[derive(Clone, Copy)]
        struct Entry {
            row: Candidate,
            admitted_micros: u64,
            local: Option<GatewayId>,
        }
        let origin = self.origin;
        let mut groups: FxHashMap<Box<str>, [Vec<Entry>; ROLES]> = FxHashMap::default();
        // The members' ids are read by the vetoes alone: a membership write
        // with no veto standing neither collects nor sorts them.
        let standing = !self.vetoes.by_gateway.is_empty();
        let mut ids = Vec::new();
        for member in members {
            if standing {
                ids.push(member.id);
            }
            let facts = &member.facts;
            let sample = sample_of(member.id, facts.dp_rank);
            let entry = Entry {
                row: row_of(member, sample.as_ref(), origin),
                admitted_micros: micros(origin, member.admitted_at),
                local: match facts.reachable_from {
                    Reachability::Any => None,
                    Reachability::Gateway(gateway) => Some(gateway),
                },
            };
            for (n, model) in facts.models.iter().enumerate() {
                // A model listed twice is one pool's one row.
                if facts.models[..n].contains(model) {
                    continue;
                }
                let side = role_slot(facts.role);
                match groups.get_mut(model.as_str()) {
                    Some(sides) => sides[side].push(entry),
                    None => {
                        let mut sides: [Vec<Entry>; ROLES] = Default::default();
                        sides[side].push(entry);
                        groups.insert(model.as_str().into(), sides);
                    }
                }
            }
        }
        self.working.clear();
        self.published.clear();
        self.places.clear();
        let mut by_model = FxHashMap::default();
        let mut pools = Vec::new();
        for (model, sides) in groups {
            let mut indexes = [None; ROLES];
            for (side, mut entries) in sides.into_iter().enumerate() {
                if entries.is_empty() {
                    continue;
                }
                entries.sort_unstable_by_key(|entry| (entry.row.id, entry.row.dp_rank));
                let rows: Vec<Candidate> = entries.iter().map(|entry| entry.row).collect();
                let index = self.working.len() as u32;
                self.places
                    .extend(rows.iter().enumerate().map(|(row, candidate)| Place {
                        id: candidate.id,
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
                });
                self.published.push(Arc::from(rows.as_slice()));
                self.working.push(Working { rows, dirty: false });
                indexes[side] = Some(index);
            }
            by_model.insert(model, indexes);
        }
        self.places
            .sort_unstable_by_key(|place| (place.id, place.pool));
        self.epoch = epoch;
        self.derived = Arc::new(Derived {
            by_model,
            pools: pools.into_boxed_slice(),
        });
        // A worker's vetoes leave with its row, in this write's own hold:
        // no publication shows the row without them.
        if standing {
            ids.sort_unstable();
            if let Some(vetoes) = self.vetoes.of_members(&ids) {
                self.vetoes = Arc::new(vetoes);
            }
        }
        self.derive_vetoes();
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
    /// whether a row took it. The caller says when the write is whole
    /// ([`Writer::folded_at`]).
    pub(super) fn fold_load(
        &mut self,
        id: WorkerId,
        rank: u16,
        sample: Option<&LoadSample>,
    ) -> bool {
        let Self {
            places,
            working,
            origin,
            ..
        } = self;
        let mut folded = false;
        for place in places_of(places, id) {
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
            folded = true;
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
    fn publish(&mut self) -> Arc<World> {
        for (pool, published) in self.working.iter_mut().zip(&mut self.published) {
            if pool.dirty {
                *published = Arc::from(pool.rows.as_slice());
                pool.dirty = false;
            }
        }
        self.waiting = false;
        self.world()
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
