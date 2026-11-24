//! The store's published world (design 4.1 and 5.5): every pool derived
//! from one membership, each a slice of whole candidate rows.
//!
//! One invariant: **a decision reads one publication.** A world is
//! everything the store published together: a pool per model and role, its
//! rows sorted by worker id, the gateways' standing vetoes and the
//! generation they were published under ([`StoreSnapshot::spec`]). One load
//! hands a decision all of it, so the vetoes it reads and the engine with
//! its number are of one publication, never the vetoes of one and the
//! engine of the next.
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
//! each pool that moved, whole. A published world never changes: a reader
//! reads memory nobody is writing and sees each row as one publication left
//! it, through no lock.
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
//! **A veto is published in its gateway's run.** The writer keeps
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

#[cfg(feature = "in-process")]
use crate::engine::policy_engine::PolicyEngine;
use crate::{
    document::spec::{AppliedSpec, IndexKind, Pairing, ViewListPositions},
    engine::{
        framework::{Engine, OverloadThresholds},
        view::{fingerprint, Candidate, Health, Phase, Role, WorkerId},
        warm_up::WarmUpRule,
    },
    load::signals::{ReportSignals, SignalCell, SignalCells},
    stages::lifecycle::Lifecycle,
    store::{
        applied::Applied,
        encoding::{EncodeClasses, EncodeMember},
        index::{KvTables, PoolIndex, Predict, Reading},
        loads::LoadSample,
        pairing::{known, Derivation, Member, PairClasses, Pool},
        workers::{Overload, Reachability, WorkerFacts, WorkerRow, WorkerStatus},
        GatewayId,
    },
    tracking::{
        in_flight::{InFlightCounts, WorkerSlot},
        lease::Signal,
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

/// One pool's shape: what a membership build derives of it and no
/// publication changes.
struct PoolShape {
    /// Each row's admission instant, in the rows' clock.
    admitted_micros: Box<[u64]>,
    /// Each row's worker's slot in the store's counts of the leases on
    /// file ([`InFlightCounts`]): the slot its member keeps across membership
    /// builds.
    booked: Box<[WorkerSlot]>,
    /// The pair classes of the pool's model, on its prefill pool where the
    /// model has a decode too. Counted: a membership build hands them to
    /// the shape it derives where they stand, one of writes that moved no offer
    /// ([`Classes::Carry`]) and a written worker's derive that leaves them
    /// as they were ([`Classes::Written`]).
    pairs: Option<Arc<PairClasses>>,
    /// The encode classes of the pool's model, on its prefill pool where an
    /// encoder of the model states an end: a decision that carries encode
    /// items reads its chosen prefill's class beside the pool it chose in.
    /// Counted and carried as the pair classes are.
    encodes: Option<Arc<EncodeClasses>>,
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
pub(crate) const fn phase_of(role: Role) -> Phase {
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
    /// member the build read ([`StoreSnapshot::unpooled_of`]).
    unpooled: Box<[(WorkerId, WorkerSlot)]>,
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
/// masks, by index and by id, and its ends. A veto's write finds it in its
/// run by the index, and a reader is handed the id.
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
}

impl GatewayRun {
    fn new(gateway: GatewayId, vetoes: Arc<[PoolVeto]>) -> Self {
        Self { gateway, vetoes }
    }
}

/// One pool's standing vetoes: a run for each gateway that holds one in
/// the pool, sorted by gateway. A reader finds the asking gateway's run
/// among the runs ([`run_of`]) and reads no other gateway's veto; a veto's
/// write builds its own gateway's run again and shares every other.
type PoolVetoes = Arc<[GatewayRun]>;

/// One gateway's vetoes on the workers of one pool, lent from a
/// publication for one decision ([`StoreSnapshot::gateway_vetoes`]): the
/// publication's own run, read by worker id.
///
/// A decision over worker lists names a worker by its id and finds the
/// worker's place in its own list. It reads a veto by that id and by
/// nothing of a pool's rows, so this reader can stand on the writer's own
/// lists, which are by worker id ([`Vetoes`]), and no caller changes.
#[cfg_attr(
    not(any(test, feature = "controller")),
    expect(
        dead_code,
        reason = "a controller's decision over worker lists reads its gateway's vetoes here"
    )
)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct GatewayVetoes<'a> {
    vetoes: &'a [PoolVeto],
}

/// One standing veto as a decision over worker lists reads it: the vetoed
/// worker, and whether the veto's hold stands. While the hold stands no
/// rule lifts the veto, as none lifts an open breaker. From the hold's end
/// to the veto's own the worker is doubted alone.
#[cfg_attr(
    not(any(test, feature = "controller")),
    expect(
        dead_code,
        reason = "a controller's decision over worker lists reads its gateway's vetoes here"
    )
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StandingVeto {
    pub(crate) id: WorkerId,
    pub(crate) held: bool,
}

#[cfg_attr(
    not(any(test, feature = "controller")),
    expect(
        dead_code,
        reason = "a controller's decision over worker lists reads its gateway's vetoes here"
    )
)]
impl<'a> GatewayVetoes<'a> {
    /// The vetoes that stand at `now`, ascending by worker id, each once:
    /// a veto past its end is not handed, though its publication still
    /// holds it.
    pub(crate) fn standing_at(self, now: Instant) -> impl Iterator<Item = StandingVeto> + 'a {
        self.vetoes
            .iter()
            .filter(move |veto| veto.ends.until > now)
            .map(move |veto| StandingVeto {
                id: veto.id,
                held: veto.ends.held_at(now),
            })
    }
}

/// The most runs a pool holds where a reader still walks to its own, past
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

/// Everything the store publishes together, as one publication left it. A
/// decision borrows it
/// ([`MemoryStore::deciding`](crate::store::MemoryStore::deciding)) for the
/// engine of its generation and its gateway's standing vetoes. A world never
/// changes once published.
pub struct StoreSnapshot {
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
    /// a publication shows. One list for every world until a count moves.
    #[expect(
        dead_code,
        reason = "the decision views were this count's one reader: it leaves in a following change"
    )]
    unreported: Arc<[u32]>,
    /// The epoch of the overload verdicts these rows carry
    /// ([`Self::overload_epoch`]).
    overload_epoch: u64,
    /// The generation this publication was made under: its engine and its
    /// number, read with the rows in one load.
    spec: Arc<Applied>,
    /// The served decider reads this with the world, before following
    /// the pointer to the generation.
    #[cfg(feature = "controller")]
    samples_decisions: bool,
    /// Each pool's standing in its model's index, at the pool's index.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the decision views were the standings' one reader: they leave in a following change"
        )
    )]
    indexes: Arc<[Option<Arc<PoolIndex>>]>,
    /// Each pool's signal cells as this publication left them, at the
    /// pool's index: one cell a row a signal the generation declares, in
    /// the rows' order. No list under a generation that declares none. One
    /// list for every world until a pool's cells move.
    #[expect(
        dead_code,
        reason = "the decision views were these cells' one reader: they leave in a following change"
    )]
    cells: Arc<[Arc<SignalCells>]>,
}

impl std::fmt::Debug for StoreSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoreSnapshot")
            .field("epoch", &self.epoch)
            .field("pools", &self.rows.len())
            .finish_non_exhaustive()
    }
}

impl StoreSnapshot {
    /// The store's epoch at the membership write this world's pools were
    /// derived by: the one thing to compare between two worlds' pools. A
    /// status change moves the store's epoch and no world's, and a
    /// publication between two membership writes is a new world of the same
    /// epoch. The pair classes move once more, at a generation of another
    /// pairing mode ([`StoreSnapshot::pairs`]).
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
    /// hands the engine under a borrow of the world, never an owned world,
    /// whose shared count a deciding thread would write.
    pub fn spec(&self) -> &AppliedSpec {
        &self.spec
    }

    /// The front that decides under this publication's document: the
    /// engine object whose stages the store tells the words of a lease
    /// ([`Applied`]). Handed under the borrow of the world, as the document
    /// is.
    #[cfg(feature = "in-process")]
    #[cfg_attr(
        not(any(feature = "controller", test)),
        expect(dead_code, reason = "a controller's deciding thread asks it")
    )]
    pub(crate) fn front(&self) -> &PolicyEngine {
        self.spec.front()
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
    /// rank's slot, the report's sample instant and no named value
    /// ([`Signal::Load`]). A stage that keeps none is sent nothing. A
    /// member of several pools is told once a pool, the same word.
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
                            signals: ReportSignals::NONE,
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

    /// The epoch of this world's overload verdicts: the one thing to
    /// compare between two worlds' verdicts. It moves with a judgment that
    /// moved a row's bit, whatever started it, and with every membership
    /// build, whose rows are judged from no bit. Two worlds of one epoch
    /// carry the same verdict on every row, so a host that mirrors the
    /// verdicts ([`Self::overloaded`]) reads them again only where the
    /// epoch moved.
    pub fn overload_epoch(&self) -> u64 {
        self.overload_epoch
    }

    /// Every row this world publishes under an overload verdict, pool by
    /// pool. A worker has a row in each pool it serves, and each row is
    /// judged under its own pool's thresholds. A host that mirrors the
    /// verdict onto its own worker object reads the worker overloaded
    /// where any of its rows is handed, and clear where none is.
    pub fn overloaded(&self) -> impl Iterator<Item = &Candidate> + '_ {
        self.rows
            .iter()
            .flat_map(|pool| pool.iter())
            .filter(|row| row.health.overloaded)
    }

    /// The row of worker `id` in the first pool that lists it, by one
    /// binary search a pool; `None` for a worker no pool lists. A member
    /// holds one rank in every pool that lists it, and each of its rows
    /// carries that rank's load record: the first pool found answers for
    /// its figures. Its overload verdict is each pool's own
    /// ([`Self::overloaded`]).
    pub fn first_row_of(&self, id: WorkerId) -> Option<&Candidate> {
        let (pool, row) = self.pooled_row(id)?;
        self.rows.get(pool)?.get(row)
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
    /// does the pair classes ([`StoreSnapshot::pairs`]).
    pub fn encodes(&self, model: &str) -> Option<&EncodeClasses> {
        self.derived.encodes(model).map(|classes| &**classes)
    }

    fn pool(&self, model: &str, role: Role) -> Option<(&[Candidate], &PoolShape)> {
        let index = self.index(model, role)?;
        Some((self.rows.get(index)?, self.derived.pools.get(index)?))
    }

    fn index(&self, model: &str, role: Role) -> Option<usize> {
        Some(self.derived.by_model.get(model)?[role_slot(role)]? as usize)
    }

    /// The slot of worker `id`'s count wherever this world lists it: a
    /// member's rows name one. `None` for a worker in no pool: a member
    /// there is read apart ([`StoreSnapshot::unpooled_of`]). It searches every
    /// pool, for a lease written outside a decision.
    pub(crate) fn booked_of(&self, id: WorkerId) -> Option<WorkerSlot> {
        let (pool, row) = self.pooled_row(id)?;
        self.derived.pools.get(pool)?.booked.get(row).copied()
    }

    /// The first pool that lists `id`, with the member's row there: one
    /// binary search a pool, as a pool's rows stand in id order.
    fn pooled_row(&self, id: WorkerId) -> Option<(usize, usize)> {
        self.rows.iter().enumerate().find_map(|(pool, rows)| {
            let row = rows.binary_search_by_key(&id, |row| row.id).ok()?;
            Some((pool, row))
        })
    }

    /// The slot of worker `id`'s count where the worker is a member in no
    /// pool of this world: one that offers no model. `None` for a member
    /// a pool lists, and for a worker that was no member when this
    /// world's membership was built. A lease's door reads it behind the
    /// pools and the table, off the path of a listed target.
    pub(crate) fn unpooled_of(&self, id: WorkerId) -> Option<WorkerSlot> {
        let unpooled = &self.derived.unpooled;
        let at = unpooled.binary_search_by_key(&id, |&(of, _)| of).ok()?;
        unpooled.get(at).map(|&(_, slot)| slot)
    }

    /// `gateway`'s vetoes on the workers of `model`'s `role` pool, lent
    /// for one decision and read by worker id
    /// ([`GatewayVetoes::standing_at`]). Nothing is copied. Where no veto
    /// stands in the store the answer is one test, and the model is not
    /// looked up.
    ///
    /// It hands every veto of the gateway on a worker of the pool. A
    /// caller finds each worker's place in its own list and passes over a
    /// worker that list does not hold: a pool names a worker another
    /// gateway alone reaches, and a list can be of another membership.
    #[cfg_attr(
        not(any(test, feature = "controller")),
        expect(
            dead_code,
            reason = "a controller's decision over worker lists reads its gateway's vetoes here"
        )
    )]
    pub(crate) fn gateway_vetoes(
        &self,
        model: &str,
        role: Role,
        gateway: GatewayId,
    ) -> GatewayVetoes<'_> {
        if self.vetoes.is_empty() {
            return GatewayVetoes { vetoes: &[] };
        }
        let run = self
            .index(model, role)
            .and_then(|pool| self.vetoes.get(pool))
            .and_then(|runs| run_of(runs, gateway));
        GatewayVetoes {
            vetoes: run.map_or(&[], |run| &run.vetoes),
        }
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
}

/// One member's fields under the standing generation's lists as last read
/// ([`Writer::rebuild`], [`Writer::list`]).
#[derive(Clone, Copy)]
struct Held {
    id: WorkerId,
    /// The revision of the member's row at the read: a write of the row's
    /// facts moves it.
    revision: u64,
    listed: ViewListPositions,
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
    listed: ViewListPositions,
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
        // No admission window is the store's to know: both bits read open,
        // and the host that asks gates capacity at its own commit.
        has_capacity: true,
        has_room: true,
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
    /// The thresholds its registration names.
    own: Overload,
    /// Its member's slot in the store's counts.
    booked: WorkerSlot,
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

/// Whether the prefixes sent to a worker under `old` still say what it
/// holds under `new`, where both are one worker's facts under one
/// registration: the same offer at the same rank. A rank is a cache of
/// its own. Every write that keeps a worker under its id asks here, and
/// the worker leaves the records of sent prefixes where it says no.
pub(super) fn same_cache(old: &WorkerFacts, new: &WorkerFacts) -> bool {
    old.dp_rank == new.dp_rank && same_offer(old, new)
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
    /// The rows' signal cells, row after row: one cell a signal the
    /// standing generation declares, and none under a generation that
    /// declares none.
    cells: Vec<SignalCell>,
    /// A cell was written since the last publication.
    cells_dirty: bool,
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

/// What the load table holds of one rank, as a membership build and an
/// applied generation read it: its load sample, and its signal cells under
/// the standing generation's columns. A closure reads the samples of a
/// table that keeps no cell.
pub(super) trait OnFile {
    /// The rank's load sample.
    fn sample(&self, id: WorkerId, rank: u16) -> Option<LoadSample>;

    /// The rank's signal cells, one a signal the standing generation
    /// declares; `None` where no value was written for the rank.
    fn cells(&self, id: WorkerId, rank: u16) -> Option<&[SignalCell]> {
        let _ = (id, rank);
        None
    }
}

impl<F: Fn(WorkerId, u16) -> Option<LoadSample>> OnFile for F {
    fn sample(&self, id: WorkerId, rank: u16) -> Option<LoadSample> {
        self(id, rank)
    }
}

/// The signal cells of `rows`, row after row, as the table holds them:
/// `signals` cells a row, and cells of no value for a rank with none on
/// file.
fn cells_beside(rows: &[Candidate], signals: usize, on_file: &impl OnFile) -> Vec<SignalCell> {
    let mut cells = Vec::with_capacity(rows.len() * signals);
    for row in rows {
        let held = on_file.cells(row.id, row.dp_rank);
        match held.filter(|held| held.len() == signals) {
            Some(held) => cells.extend_from_slice(held),
            None => cells.resize(cells.len() + signals, SignalCell::NONE),
        }
    }
    cells
}

/// Writes `cells` over `over` when the writer holds the last reference to
/// it, by the rule a pool's rows are written over ([`written_over`]).
/// Whether it did.
fn cells_over(over: &mut Arc<SignalCells>, cells: &[SignalCell], signals: usize) -> bool {
    let Some(over) = Arc::get_mut(over) else {
        return false;
    };
    over.refill(signals).extend_from_slice(cells);
    true
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
    /// Each pool's signal cells as the last publication left them
    /// ([`StoreSnapshot::cells`]): one list for every world until a pool's
    /// cells move, and no list under a generation that declares no signal.
    cells: Arc<[Arc<SignalCells>]>,
    /// Each pool's spare cells: the block it published before its last,
    /// kept for its next publication to write over, as its rows' spare is.
    /// A block some world still lists is let go and stays that world's: a
    /// world is freed by a write ([`Self::replaced`]), so no decision
    /// frees a block.
    cell_spares: Vec<Option<Arc<SignalCells>>>,
    /// A write moved a pool's cells since the last publication: the one
    /// word a publication tests for them.
    cells_moved: bool,
    /// How many times a publication allocated a pool's rows or its cells.
    allocations: u64,
    /// How many times the pools were built from the membership.
    builds: u64,
    /// Every replaced world a reader still held when the cell let it go
    /// ([`Self::replaced`]), until a publication finds no reader holding it
    /// and lets it go. The cell holds the standing world and the writer
    /// each of these, so a decision that drops the last reader's hold
    /// frees nothing: not the world, its list of pools, its vetoes, its
    /// derived tables nor its generation.
    worlds: Vec<Arc<StoreSnapshot>>,
    /// How many worlds the writer published.
    publications: u64,
    /// Every member's rows, sorted by worker id: a worker serving several
    /// models has a row in each model's pool.
    places: Vec<Place>,
    /// The store's counts of the leases on file: the one table of the
    /// store's life, which the leases' file writes and a world lends.
    counts: Arc<InFlightCounts>,
    vetoes: Arc<Vetoes>,
    /// The vetoes as the world carries them, by pool and by gateway. A
    /// veto's write builds its own gateway's runs again; a membership build
    /// and a window's publication that drops a veto past its end build
    /// every run again.
    pool_vetoes: Arc<[PoolVetoes]>,
    /// The generation every world is published under until the next is
    /// applied: the unbound one at the start.
    spec: Arc<Applied>,
    /// Each pool's standing in its model's index: made by a membership
    /// write, its warming rows read again by a window's publication.
    indexes: Arc<[Option<Arc<PoolIndex>>]>,
    /// The buffers a pool's reading is made in.
    reading: Reading,
    /// The lane table's count of moves as the last membership write read
    /// it ([`KvTables::moves`]): while it stands, no pool's place in an
    /// index moved.
    lanes: u64,
    /// How many membership writes walked the pools
    /// ([`Self::index_pools`]), for the cases.
    #[cfg(test)]
    walks: u64,
    /// The store's index tables, where a membership write reads each
    /// pool's standing ([`Self::over`]). A writer made alone holds tables
    /// of its own with no lane.
    kv: Arc<KvTables>,
    /// The generation before the standing one, kept one apply longer: the
    /// publication that lets go of the last world published under it
    /// frees no engine, the next apply does. Its whole engine stays
    /// resident that long, its session table included, as it stood when
    /// it was replaced.
    retired: Option<Arc<Applied>>,
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
    /// The epoch of the working rows' overload verdicts. It moves with a
    /// judgment that moved a row's bit ([`Self::judge`]) and with every
    /// membership build, whose rows start with no bit. A world carries the
    /// epoch of its publication ([`StoreSnapshot::overload_epoch`]).
    overload_epoch: u64,
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
            cells: Arc::default(),
            cell_spares: Vec::new(),
            cells_moved: false,
            allocations: 0,
            builds: 0,
            worlds: Vec::new(),
            publications: 0,
            places: Vec::new(),
            counts: Arc::default(),
            vetoes: Arc::default(),
            pool_vetoes: Arc::default(),
            spec: Arc::new(Applied::new(AppliedSpec::unbound(), origin)),
            indexes: Arc::default(),
            reading: Reading::default(),
            lanes: 0,
            #[cfg(test)]
            walks: 0,
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
            overload_epoch: 0,
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
    pub(super) fn counts(&self) -> Arc<InFlightCounts> {
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
    pub(super) fn world(&self) -> Arc<StoreSnapshot> {
        Arc::new(StoreSnapshot {
            epoch: self.epoch,
            origin: self.origin,
            derived: Arc::clone(&self.derived),
            rows: self.published.clone().into_boxed_slice(),
            unreported: Arc::clone(&self.unreported),
            overload_epoch: self.overload_epoch,
            vetoes: Arc::clone(&self.pool_vetoes),
            spec: Arc::clone(&self.spec),
            #[cfg(feature = "controller")]
            samples_decisions: self.spec.engine().explain_sample() != 0,
            indexes: Arc::clone(&self.indexes),
            cells: Arc::clone(&self.cells),
        })
    }

    /// Take back the world the cell let go as its successor was shown. One
    /// no reader holds is freed here, by the write, as a world always was;
    /// one a reader holds is kept among `worlds`, so that reader's hold is
    /// not the last. The cell's swap pays each reader's borrow of the
    /// outgoing world into its count before it returns, so the count read
    /// here is every reader's.
    pub(super) fn replaced(&mut self, world: Arc<StoreSnapshot>) {
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
        spec: Arc<Applied>,
        epoch: u64,
        members: impl Iterator<Item = (&'a WorkerRow, WorkerSlot)>,
        on_file: impl OnFile,
    ) -> Result<(Arc<StoreSnapshot>, Option<Arc<Applied>>), StaleGeneration> {
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
            self.rebuild(epoch, members, on_file, Classes::Derive)
        } else {
            if relisted {
                self.list(members);
            }
            self.bind();
            // A generation of the standing mode comes with no membership
            // write: the receipts are read from the samples as the rows
            // stand, and the cells under the generation's own columns.
            self.hear_again(&on_file);
            self.read_cells(&on_file);
            self.judge_every();
            // A generation may rule a pool's warming rows by other numbers,
            // or rule none, and it may turn a pool's record of sent prefixes.
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
    fn list<'a>(&mut self, members: impl Iterator<Item = (&'a WorkerRow, WorkerSlot)>) {
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
        for (member, ..) in members {
            let facts = &member.facts;
            let listed = lists.positions(&facts.engine_kind, facts.zone.as_deref(), &facts.labels);
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
    /// decode pool, and for an encode pool where that binding's plan is
    /// `epd`
    /// ([`Engine::pool_binding`](crate::engine::framework::Engine::pool_binding)).
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
            overload_epoch,
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
        *overload_epoch += u64::from(moved);
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
    fn hear_again(&mut self, on_file: &impl OnFile) {
        for (heard, at) in self.heard.iter_mut().zip(&self.engine_rows) {
            if !at.rank {
                continue;
            }
            let sample = self
                .working
                .get(at.pool as usize)
                .and_then(|pool| pool.rows.get(at.row as usize))
                .and_then(|row| on_file.sample(row.id, row.dp_rank));
            *heard = sample.map_or(0, |sample| micros(self.origin, sample.received_at));
        }
    }

    /// Every pool's signal cells read from the table under the standing
    /// generation's columns, and published whole with the next world: for
    /// a membership build, whose rows are new, and for an applied
    /// generation, whose columns may be other ones. One test where neither
    /// the standing generation nor the one before declares a signal.
    fn read_cells(&mut self, on_file: &impl OnFile) {
        let signals = self.spec.signal_columns().len();
        if signals == 0 && self.cells.is_empty() {
            return;
        }
        self.cells_moved = false;
        self.cell_spares.clear();
        self.cell_spares.resize(self.working.len(), None);
        for pool in &mut self.working {
            pool.cells_dirty = false;
            pool.cells = if signals == 0 {
                Vec::new()
            } else {
                cells_beside(&pool.rows, signals, on_file)
            };
        }
        self.cells = if signals == 0 {
            Arc::default()
        } else {
            let block = |pool: &Working| Arc::new(SignalCells::new(pool.cells.clone(), signals));
            self.working.iter().map(block).collect()
        };
    }

    /// Write one value into the cells of member `id`'s rows at `rank`,
    /// under `column`. Whether a row took it: none does for a member with
    /// no place yet and for one whose rows stand at another rank, and the
    /// build that gives the rank its row reads the value from the table.
    pub(super) fn fold_signal(
        &mut self,
        id: WorkerId,
        rank: u16,
        column: usize,
        cell: SignalCell,
    ) -> bool {
        let signals = self.spec.signal_columns().len();
        let Self {
            places, working, ..
        } = self;
        let mut folded = false;
        for place in places_of(places, id) {
            let Some(pool) = working.get_mut(place.pool as usize) else {
                continue;
            };
            let row = place.row as usize;
            if pool.rows.get(row).is_none_or(|row| row.dp_rank != rank) {
                continue;
            }
            if let Some(held) = pool.cells.get_mut(row * signals + column) {
                *held = cell;
                pool.cells_dirty = true;
                folded = true;
            }
        }
        self.cells_moved |= folded;
        folded
    }

    /// Member `id`'s rows hold no value any more: its cells left the table
    /// with its load samples. Whether a row held one.
    pub(super) fn forget_signals(&mut self, id: WorkerId) -> bool {
        let signals = self.spec.signal_columns().len();
        let Self {
            places, working, ..
        } = self;
        let mut forgot = false;
        for place in places_of(places, id) {
            let Some(pool) = working.get_mut(place.pool as usize) else {
                continue;
            };
            let from = place.row as usize * signals;
            let Some(held) = pool.cells.get_mut(from..from + signals) else {
                continue;
            };
            if held.iter().any(|cell| !cell.is_none()) {
                held.fill(SignalCell::NONE);
                pool.cells_dirty = true;
                forgot = true;
            }
        }
        self.cells_moved |= forgot;
        forgot
    }

    /// The standing generation: the one every world is published under
    /// until the next is applied.
    pub(super) fn spec(&self) -> &AppliedSpec {
        &self.spec
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

    /// Forget the fields read under the generation's lists. A snapshot
    /// can give the same id and revision different facts.
    pub(super) fn clear_listed(&mut self) {
        self.listed.clear();
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
        members: impl Iterator<Item = (&'a WorkerRow, WorkerSlot)>,
        on_file: impl OnFile,
        classes: Classes,
    ) -> Arc<StoreSnapshot> {
        // A build takes in every membership write: none stays owed.
        self.owed = None;
        self.builds += 1;
        // A build's rows start with no bit: a verdict the last world showed
        // can go with no judgment to move it.
        self.overload_epoch += 1;
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
        let mut members: Vec<(WorkerId, &'a WorkerRow, WorkerSlot)> = members
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
            let sample = on_file.sample(member.id, facts.dp_rank);
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
                    _ => lists.positions(&facts.engine_kind, facts.zone.as_deref(), &facts.labels),
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
                ViewListPositions::default()
            };
            let entry = Entry {
                row: row_of(member, sample.as_ref(), origin, listed),
                facts,
                admitted_micros: micros(origin, member.admitted_at),
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
            // pair classes and its encode classes.
            let ends = last.as_deref().and_then(|last| last.ends(&model));
            let carried = ends.and_then(|ends| ends.prefill?.pairs.as_ref());
            let (mode, kept) = (self.spec.pairing(), &mut self.derivations);
            let mut pairs = pair_up(&mut sides, mode, classes, carried, kept, &model);
            // The encode classes are the last write's in every model whose
            // encoders and prefills this write left as they stood, and
            // derived whole in the others.
            let stood = ends.filter(|ends| ends.stand(classes, &sides));
            let mut encodes = encode_up(&sides, stood.and_then(|ends| ends.prefill));
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
                    // Read at this write's end, once every pool stands.
                    cells: Vec::new(),
                    cells_dirty: false,
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
        let unpooled: Box<[(WorkerId, WorkerSlot)]> = bare
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
        // Every pool's rows are new, and so are their cells.
        self.read_cells(&on_file);
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
    pub(super) fn folded_at(&mut self, at: Instant) -> Option<Arc<StoreSnapshot>> {
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
    pub(super) fn folded(&mut self) -> Option<Arc<StoreSnapshot>> {
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
    pub(super) fn fold_status(
        &mut self,
        id: WorkerId,
        status: WorkerStatus,
    ) -> Option<Arc<StoreSnapshot>> {
        let health = health_of(status);
        self.fold_health(id, |row| {
            let moved = (row.healthy, row.draining) != (health.healthy, health.draining);
            row.healthy = health.healthy;
            row.draining = health.draining;
            moved
        })
    }

    /// Write `fold` over the health bits of each of the member's rows, and
    /// publish at once where it says a bit of one moved: a pool with a row
    /// that moved is published whole. `None` where none did.
    fn fold_health(
        &mut self,
        id: WorkerId,
        fold: impl Fn(&mut Health) -> bool,
    ) -> Option<Arc<StoreSnapshot>> {
        let Self {
            places, working, ..
        } = self;
        let mut folded = false;
        for place in places_of(places, id) {
            let Some(pool) = working.get_mut(place.pool as usize) else {
                continue;
            };
            let Some(row) = pool.rows.get_mut(place.row as usize) else {
                continue;
            };
            if fold(&mut row.health) {
                pool.dirty = true;
                folded = true;
            }
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
    ) -> Option<Arc<StoreSnapshot>> {
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
    pub(super) fn drop_gateways(&mut self, gone: &[GatewayId]) -> Option<Arc<StoreSnapshot>> {
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
        members: impl Iterator<Item = (&'a WorkerRow, WorkerSlot)>,
        on_file: impl OnFile,
    ) -> Option<Arc<StoreSnapshot>> {
        let owed = self.owed.take()?;
        if let Some(at) = at {
            self.last_window = Some(at);
            // A veto past its end leaves here, as at a window's publication:
            // the build derives every run from the vetoes that stand.
            if let Some(vetoes) = self.vetoes.standing_at(at) {
                self.vetoes = Arc::new(vetoes);
            }
        }
        Some(self.rebuild(epoch, members, on_file, owed.classes))
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
    pub(super) fn publish_if_due(&mut self, now: Instant) -> Option<Arc<StoreSnapshot>> {
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
    fn publish(&mut self) -> Arc<StoreSnapshot> {
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
        if mem::take(&mut self.cells_moved) {
            self.publish_cells();
        }
        // Rows no world lists and no pool took are freed here, by a write.
        self.unlisted.clear();
        self.waiting = false;
        self.publications += 1;
        self.world()
    }

    /// Each pool whose cells a write moved since the last publication takes
    /// a new block, whole, as its rows do: the pool's working cells written
    /// over its spare where no world lists the spare any more, and over a
    /// new block, which is counted, where one does. The worlds take one
    /// new list of the blocks, in one allocation.
    fn publish_cells(&mut self) {
        let signals = self.spec.signal_columns().len();
        let Self {
            working,
            cells,
            cell_spares,
            allocations,
            ..
        } = self;
        debug_assert_eq!(cells.len(), working.len());
        let pools = working.iter_mut().zip(cell_spares.iter_mut());
        let blocks = pools.zip(cells.iter()).map(|((pool, spare), standing)| {
            if !pool.cells_dirty {
                return Arc::clone(standing);
            }
            pool.cells_dirty = false;
            let over = spare
                .take()
                .and_then(|mut over| cells_over(&mut over, &pool.cells, signals).then_some(over));
            let next = over.unwrap_or_else(|| {
                *allocations += 1;
                Arc::new(SignalCells::new(pool.cells.clone(), signals))
            });
            *spare = Some(Arc::clone(standing));
            next
        });
        *cells = blocks.collect();
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

    /// How many times the pools were built from the membership: a build an
    /// ask or a window's end paid, or a generation of another pairing mode.
    pub(super) fn builds(&self) -> u64 {
        self.builds
    }

    /// How many membership writes walked the pools to stand each in its
    /// model's index ([`Self::index_pools`]).
    #[cfg(test)]
    pub(super) fn walks(&self) -> u64 {
        self.walks
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
    /// no row, no admission and no wire a row states: where no lane, no
    /// block size of record and no row's word to record moved either
    /// ([`KvTables::moves`]), every pool stands as it did and none is
    /// walked. A pool is also made anew when its sent-prefix settings
    /// change. Under [`Classes::Derive`] the pools may stand under a
    /// generation of another pairing mode, one the standings were not ruled
    /// by: every kept pool's warming rows are ruled and read again.
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
        // No lane was made or retired, no block size of record moved and
        // no row's word to record was cleared since the last write stood
        // its pools: no pool's place in an index moved, no record's bound
        // and no row's word, and none is looked up.
        let moves = kv.moves();
        let still = mem::replace(lanes, moves) == moves;
        if still && classes == Classes::Carry {
            return;
        }
        #[cfg(test)]
        {
            self.walks += 1;
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
                let predict = predict_of(spec, kv, model, rows);
                let here = held.get(at).and_then(Option::as_ref);
                let here = here.filter(|standing| standing.is_of(model, rows));
                let was = here.map(|standing| (at, standing)).or_else(|| {
                    let sides = *stood.get_or_insert_with(|| before.by_model.get(model));
                    let old = sides?.get(side).copied().flatten()? as usize;
                    Some((old, held.get(old)?.as_ref()?))
                });
                let kept = was.filter(|(_, standing)| {
                    standing.predict() == predict
                        && standing.holds(rows, admitted)
                        && (still || kv.stands(model, rows, standing))
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
                        Arc::new(kv.pool(model, rows, admitted, rule, predict, now(), reading))
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
    /// or rule none. A pool whose settings for the record of sent prefixes
    /// moved is made anew.
    fn rule_indexes(&mut self) {
        let Self {
            derived,
            working,
            spec,
            reading,
            indexes,
            kv,
            origin,
            last_window,
            ..
        } = self;
        let now = last_window.map_or(0, |at| micros(*origin, at));
        let mut next = None;
        for (model, sides) in &derived.by_model {
            for &pool in sides.iter().flatten() {
                let at = pool as usize;
                let (Some(Some(held)), Some(rows), Some(shape)) =
                    (indexes.get(at), working.get(at), derived.pools.get(at))
                else {
                    continue;
                };
                let rule = ruling_of(spec, model, &rows.rows);
                let predict = predict_of(spec, kv, model, &rows.rows);
                let read = if held.predict() == predict {
                    held.again(rule, now, reading)
                } else {
                    Some(kv.pool(
                        model,
                        &rows.rows,
                        &shape.admitted_micros,
                        rule,
                        predict,
                        now,
                        reading,
                    ))
                };
                if let Some(read) = read {
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

    fn publish_window(&mut self, at: Instant) -> Arc<StoreSnapshot> {
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
/// ([`Engine::pool_binding`](crate::engine::framework::Engine::pool_binding)). The
/// hand names rows for such a request alone; a pair's sides read their
/// columns. `None` where the profile's warm-up stages rule by different
/// numbers: it is named no rows.
fn ruling_of(spec: &AppliedSpec, model: &str, rows: &[Candidate]) -> Option<WarmUpRule> {
    let role = rows.first()?.role;
    spec.engine().pool_rule_for(model, phase_of(role))
}

/// The tree's placement bound for a pool that keeps a record of sent
/// prefixes: its profile reads them, its model's pair does at the decode
/// side, or its binding's shadow does where that profile is named no row
/// ([`Engine::pool_record_for`](crate::engine::framework::Engine::pool_record_for)).
fn predict_of(
    spec: &AppliedSpec,
    kv: &KvTables,
    model: &str,
    rows: &[Candidate],
) -> Option<Predict> {
    let phase = phase_of(rows.first()?.role);
    let ahead = spec.engine().pool_record_for(model, phase)?;
    let index = spec.index_for(model);
    if index.kind != IndexKind::Tree {
        return None;
    }
    let size = kv
        .block_size(model)
        .unwrap_or_else(|| spec.block_size())
        .max(1);
    let bound = (index.max_size / u64::from(size)).max(1);
    Some(Predict {
        bound: usize::try_from(bound).unwrap_or(usize::MAX),
        ahead,
    })
}

/// How many of `rows` hold no report: the pass a membership write takes over
/// each pool it builds. Every later write moves the pool's count by the row
/// it folds ([`rows_at`]), so no publication takes the pass.
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
