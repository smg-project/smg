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
//! build. A world's epoch is the membership's at its build: a world's
//! identity moves with every publication.
//! Between two membership builds a publication derives nothing: it is one
//! small object, a pointer per pool, and new rows for each pool a write was
//! folded into.
//!
//! **The store derives no pair class.** A model's pair classes are derived
//! by whoever holds its worker objects, from what its prefills and decodes
//! report ([`PairClasses::derive`](crate::store::pairing::PairClasses::derive)).
//! A row of a world reads class `0`.
//!
//! A row in a pool is the candidate row itself. Its fixed fields are
//! written when the membership's first world is built. Its two status bits
//! are folded into the writer's own copy of the pool, and a publication copies
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
//! takes other rows the writer holds alone, or the publication allocates.
//!
//! **The store publishes at its own rate, never at its writers'.** Every
//! deciding thread fetches every publication, so the build the membership
//! writes left owed is published once per [`PUBLISH_WINDOW`], and a status
//! change, a veto or a leave at once, through the same door. The store
//! names the instant the build is due
//! ([`MemoryStore::publish_due`](crate::store::MemoryStore::publish_due)).
//!
//! **A veto is published in its gateway's run.** The writer keeps
//! each gateway's list and publishes each pool's standing vetoes as one run
//! a gateway, by row. A veto's write builds its own gateway's run again in
//! the pools its worker has a row in and shares every other run with the
//! publication before. A membership build, which moves the rows, builds
//! every run again, and one paid at a window's end drops the vetoes past
//! their end first. A veto leaves with its worker's row in the membership
//! build's own step, so no publication shows the row without it.

#![expect(
    clippy::exhaustive_structs,
    reason = "the store's views for a decision leave the crate, and their types with them"
)]

use std::{
    mem,
    sync::Arc,
    time::{Duration, Instant},
};

use rustc_hash::FxHashMap;

#[cfg(feature = "in-process")]
use crate::engine::in_process::InProcess;
use crate::{
    document::spec::{AppliedSpec, ViewListPositions},
    engine::view::{fingerprint, Candidate, Health, Role, WorkerId},
    store::{
        applied::{Applied, EngineCounts},
        workers::{Reachability, WorkerFacts, WorkerRow, WorkerStatus},
        GatewayId,
    },
    tracking::in_flight::{InFlightCounts, WorkerSlot},
};

#[cfg(test)]
mod tests;

/// How long a build the membership writes left owed may wait for its
/// publication: the window the gateway's load monitor publishes pushed
/// records by. A test hands its own to
/// [`MemoryStore::with_publish_window`](crate::store::MemoryStore::with_publish_window).
pub const PUBLISH_WINDOW: Duration = Duration::from_millis(20);

/// Microseconds from `origin` to `at`, saturating at both ends.
pub(super) fn micros(origin: Instant, at: Instant) -> u64 {
    u64::try_from(at.saturating_duration_since(origin).as_micros()).unwrap_or(u64::MAX)
}

/// One pool's shape: what a membership build derives of it and no
/// publication changes.
struct PoolShape {
    /// Each row's worker's slot in the store's counts of the leases on
    /// file ([`InFlightCounts`]): the slot its member keeps across membership
    /// builds.
    booked: Box<[WorkerSlot]>,
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

/// What one membership build derives: where each pool stands in a world's
/// rows, and each pool's shape. Built once per membership build and shared
/// by every world published until the next.
#[derive(Default)]
struct Derived {
    /// The pools by model, then by role: each an index into a world's rows.
    /// A pool is every member serving one model in one role, whatever its
    /// status (a status change is a health signal and moves no position).
    by_model: FxHashMap<Box<str>, [Option<u32>; ROLES]>,
    pools: Box<[PoolShape]>,
    /// Every member the build read, by id, each with its slot in the
    /// counts: a member a pool lists, and one that offers no model
    /// ([`StoreSnapshot::booked_of`]).
    slots: Box<[(WorkerId, WorkerSlot)]>,
}

impl Derived {
    /// How many members `pool` holds: none where there is no pool.
    fn members(&self, pool: Option<u32>) -> usize {
        pool.and_then(|pool| self.pools.get(pool as usize))
            .map_or(0, |shape| shape.booked.len())
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
/// departed worker's vetoes with its row, and a build paid at a window's
/// end, which drops a veto past its end. A world carries what is
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
    /// The store's epoch at the membership write this world's pools were
    /// derived by. A status change moves the store's epoch and no world's.
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
    spec: Arc<Applied>,
    /// The served decider reads this with the world, before following
    /// the pointer to the generation.
    #[cfg(feature = "controller")]
    samples_decisions: bool,
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

    /// This publication's document as its store holds it ([`Applied`]):
    /// what a word of the store's membership and its host's tick reach.
    pub(crate) fn applied(&self) -> &Applied {
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
    pub(crate) fn front(&self) -> &InProcess {
        self.spec.front()
    }

    #[cfg(feature = "controller")]
    pub(crate) fn samples_decisions(&self) -> bool {
        self.samples_decisions
    }

    /// `at` in the rows' clock: microseconds since the store's origin.
    pub fn micros(&self, at: Instant) -> u64 {
        micros(self.origin, at)
    }

    /// The row of worker `id` in the first pool that lists it, by one
    /// binary search a pool; `None` for a worker no pool lists.
    pub fn first_row_of(&self, id: WorkerId) -> Option<&Candidate> {
        let (pool, row) = self.pooled_row(id)?;
        self.rows.get(pool)?.get(row)
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

    /// The slot of worker `id`'s count, where the worker was a member
    /// when this world's membership was built: in a pool, or in none
    /// where it offers no model. `None` for a worker that was no member
    /// then. One binary search of the members by id, whatever the pools:
    /// a lease written outside a decision reads it too.
    pub(crate) fn booked_of(&self, id: WorkerId) -> Option<WorkerSlot> {
        let slots = &self.derived.slots;
        let at = slots.binary_search_by_key(&id, |&(of, _)| of).ok()?;
        slots.get(at).map(|&(_, slot)| slot)
    }

    /// The first pool that lists `id`, with the member's row there: one
    /// binary search a pool, as a pool's rows stand in id order.
    fn pooled_row(&self, id: WorkerId) -> Option<(usize, usize)> {
        self.rows.iter().enumerate().find_map(|(pool, rows)| {
            let row = rows.binary_search_by_key(&id, |row| row.id).ok()?;
            Some((pool, row))
        })
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
/// its status bits, and its engine family, zone and labels as `listed`
/// reads them under the standing generation's lists. Its pair class reads
/// `0`: the store derives none.
fn row_of(member: &WorkerRow, listed: ViewListPositions) -> Candidate {
    let facts = &member.facts;
    Candidate {
        id: member.id,
        role: facts.role,
        dp_rank: facts.dp_rank,
        dp_size: facts.dp_size,
        health: health_of(member.status),
        wire_modes: facts.wire_modes,
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

/// One member on its way into a pool: its row and its slot in the store's
/// counts.
#[derive(Clone, Copy)]
struct Entry {
    row: Candidate,
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

/// One member's row in one pool of the writer's working copy.
#[derive(Clone, Copy)]
struct Place {
    id: WorkerId,
    pool: u32,
    row: u32,
}

/// The writer's own copy of one pool: what status changes are folded into
/// between two publications.
struct Working {
    rows: Vec<Candidate>,
    /// Folded since the last publication.
    dirty: bool,
}

/// The write side of the world, behind one lock of the store: the working
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
    /// Each pool's rows as the last publication left them.
    published: Vec<Arc<[Candidate]>>,
    /// Each pool's spare: the rows it published before its last, kept for
    /// its next publication to write over; none before the pool's first
    /// publication after a membership build. Beside the working copies,
    /// not in them.
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
    /// builds every run again.
    pool_vetoes: Arc<[PoolVetoes]>,
    /// The generation every world is published under until the next is
    /// applied: the unbound one at the start.
    spec: Arc<Applied>,
    /// The generation before the standing one, kept one apply longer: the
    /// publication that lets go of the last world published under it
    /// frees no engine, the next apply does. Its whole engine stays
    /// resident that long, its session table included, as it stood when
    /// it was replaced.
    retired: Option<Arc<Applied>>,
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
    /// The last publication the window counts from: a window's end.
    last_window: Option<Instant>,
    /// The build the membership writes since the last one left owed, by the
    /// instant the first of them was written: the build is due from then,
    /// or from the window's end where that is later. `None` while the
    /// working rows are the membership's.
    owed: Option<Instant>,
}

impl Writer {
    /// The write side of a store with no member, whose rows' clock starts
    /// at `origin`. The engine of its unbound start counts the panics it
    /// holds into `counts`, the store's, as every engine behind it does.
    pub(super) fn new(window: Duration, origin: Instant, counts: &EngineCounts) -> Self {
        Self {
            window,
            origin,
            epoch: 0,
            derived: Arc::default(),
            published: Vec::new(),
            spares: Vec::new(),
            working: Vec::new(),
            held: Vec::new(),
            unlisted: Vec::new(),
            builds: 0,
            worlds: Vec::new(),
            publications: 0,
            places: Vec::new(),
            counts: Arc::default(),
            vetoes: Arc::default(),
            pool_vetoes: Arc::default(),
            spec: Arc::new(Applied::new(AppliedSpec::unbound(), origin, counts)),
            retired: None,
            outside: OutsideLists::default(),
            listed: Vec::new(),
            listed_spare: Vec::new(),
            last_window: None,
            owed: None,
        }
    }

    /// The store's counts of the leases on file, for the leases' file to
    /// write.
    pub(super) fn counts(&self) -> Arc<InFlightCounts> {
        Arc::clone(&self.counts)
    }

    /// A world of everything published so far: one small object, a pointer
    /// per pool. Nothing derived is built here.
    pub(super) fn world(&self) -> Arc<StoreSnapshot> {
        Arc::new(StoreSnapshot {
            epoch: self.epoch,
            origin: self.origin,
            derived: Arc::clone(&self.derived),
            rows: self.published.clone().into_boxed_slice(),
            vetoes: Arc::clone(&self.pool_vetoes),
            spec: Arc::clone(&self.spec),
            #[cfg(feature = "controller")]
            samples_decisions: self.spec.engine().explain_sample() != 0,
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
    /// generation and then a lower one. A generation whose lists differ
    /// from the standing one's has each of `members`, the standing
    /// membership, read again under them, so the first world of a
    /// generation holds rows of its own lists.
    ///
    /// No pool is built again, whatever the generation's pairing mode: the
    /// store derives no pair class, and a build the membership writes left
    /// owed stays owed.
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
        members: impl Iterator<Item = (&'a WorkerRow, WorkerSlot)>,
    ) -> Result<(Arc<StoreSnapshot>, Option<Arc<Applied>>), StaleGeneration> {
        let standing = self.spec.generation();
        if spec.generation() <= standing {
            return Err(StaleGeneration {
                standing,
                offered: spec.generation(),
            });
        }
        let relisted = spec.lists() != self.spec.lists();
        let before = self.retired.replace(mem::replace(&mut self.spec, spec));
        if relisted {
            self.list(members);
        }
        Ok((self.publish(), before))
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
    /// make them the working copy and publish them through the one door:
    /// every pool is new, so the door copies none, every earlier fold is in
    /// the world this returns and nothing waits. A member's engine family,
    /// zone and labels under the standing generation's lists are read where
    /// its row's revision moved since they were last read, and carried
    /// where it stands.
    pub(super) fn rebuild<'a>(
        &mut self,
        epoch: u64,
        members: impl Iterator<Item = (&'a WorkerRow, WorkerSlot)>,
    ) -> Arc<StoreSnapshot> {
        // A build takes in every membership write: none stays owed.
        self.owed = None;
        self.builds += 1;
        // Each model's members a side, by their place in `entries`: a
        // member's entry is written once, whatever it offers.
        let mut groups: FxHashMap<Box<str>, [Vec<u32>; ROLES]> = FxHashMap::default();
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
        let mut entries: Vec<Entry> = Vec::with_capacity(members.len());
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
                row: row_of(member, listed),
                // The slot the member's first write gave it: a build gives
                // none and moves none.
                booked: slot,
            };
            let at = entries.len() as u32;
            entries.push(entry);
            if facts.models.is_empty() {
                // In no pool, and done: the loop below starts on a model.
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
            let sides: [Vec<Entry>; ROLES] =
                sides_at.map(|side| side_of(&mut entries, &side, last_model));
            // Every side stands in id order, as the walk above pushed it:
            // a pool's rows are searched by id.
            debug_assert!(sides
                .iter()
                .all(|entries| entries.is_sorted_by(|a, b| a.row.id < b.row.id)));
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
                    booked: entries.iter().map(|entry| entry.booked).collect(),
                });
                self.spares.push(None);
                self.working.push(Working { rows, dirty: false });
                indexes[side] = Some(index);
            }
            by_model.insert(model, indexes);
        }
        self.places
            .sort_unstable_by_key(|place| (place.id, place.pool));
        // The members' fields as this build read or carried them, and the
        // count outside the lists: none where the generation declares no
        // list, whatever the last one counted.
        self.listed = incoming;
        self.listed_spare = outgoing;
        self.outside = outside;
        self.epoch = epoch;
        // Every member's id and slot, in the id order of this write's one
        // sort: a member in a pool, and one that offers no model.
        let slots: Box<[(WorkerId, WorkerSlot)]> =
            members.iter().map(|&(id, _, slot)| (id, slot)).collect();
        debug_assert!(slots.is_sorted_by(|a, b| a.0 < b.0));
        self.derived = Arc::new(Derived {
            by_model,
            pools: pools.into_boxed_slice(),
            slots,
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
        for pool in &mut self.working {
            pool.dirty = false;
            self.published.push(Arc::from(pool.rows.as_slice()));
        }
        self.publish()
    }

    /// Every pool's runs built again from the gateways' lists and the
    /// members' places: for a membership build, which moves the rows. A
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

    /// When the build the membership writes left owed is due: the end of
    /// the window the last publication opened. A build owed past that end,
    /// or with no window opened yet, is due from its first write. `None`
    /// when none is owed, and for a window the clock cannot add.
    pub(super) fn due(&self) -> Option<Instant> {
        let end = self.last_window.map(|last| self.window_end(last));
        match (self.owed, end) {
            (Some(since), None) => Some(since),
            (Some(since), Some(end)) => end.map(|end| end.max(since)),
            (None, _) => None,
        }
    }

    /// A membership write moved the rows of record: the pools are built
    /// again once, when the window ends ([`Writer::due`]), for every write
    /// since the last build. `at` is when this write was written. `true`
    /// where no build was owed before: the write named an instant the host
    /// has not read.
    pub(super) fn owe(&mut self, at: Instant) -> bool {
        let first = self.owed.is_none();
        if first {
            self.owed = Some(at);
        }
        first
    }

    /// Whether a build is owed and due at `now`: the store pays it with the
    /// membership in hand ([`Writer::build_owed`]).
    pub(super) fn build_due(&self, now: Instant) -> bool {
        self.owed.is_some() && self.due().is_some_and(|due| now >= due)
    }

    /// Pay the build the membership writes left owed: the pools of
    /// `members`, the membership at the store's `epoch`. A host pays it at
    /// the window's end, `at`, which opens the next window and lets the
    /// vetoes past their end go. An asked build names no instant and leaves
    /// the window as it stands. `None` when no build is owed: another call
    /// paid it.
    #[must_use]
    pub(super) fn build_owed<'a>(
        &mut self,
        at: Option<Instant>,
        epoch: u64,
        members: impl Iterator<Item = (&'a WorkerRow, WorkerSlot)>,
    ) -> Option<Arc<StoreSnapshot>> {
        self.owed.take()?;
        if let Some(at) = at {
            self.last_window = Some(at);
            // A veto that ended at or before the window's instant leaves the
            // table here and waits for no gateway's clear: the build derives
            // every run from the vetoes that stand.
            if let Some(vetoes) = self.vetoes.standing_at(at) {
                self.vetoes = Arc::new(vetoes);
            }
        }
        Some(self.rebuild(epoch, members))
    }

    /// The end of the window `last` opened; `None` for a window so long
    /// that the clock cannot add it, which no instant ends.
    fn window_end(&self, last: Instant) -> Option<Instant> {
        last.checked_add(self.window)
    }

    /// The one door every publication goes through: each pool folded since
    /// the last publication takes new rows, whole, and one world holds
    /// every pool.
    ///
    /// A pool's new rows are its working copy written over rows the writer
    /// holds the last reference to ([`written_over`]): the pool's spare,
    /// else held rows of its length that no world lists any more. That is
    /// exact: every world listing such rows left the store's cell by the
    /// publication that replaced them, the cell's swap pays each reader's
    /// borrow of the outgoing world into that world's count before it
    /// returns, and a world counts on the rows it lists for as long as it
    /// lives. Rows some world still lists are left whole to that world's
    /// reader, and a pool with none to write over takes a new copy: a
    /// publication never waits for a reader. Two buffers a pool carry every
    /// window, and one more for each publication a reader still holds. A
    /// publication asks each held row's count once, and each pool with no
    /// spare to write over reads each unlisted row's length.
    fn publish(&mut self) -> Arc<StoreSnapshot> {
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
            let next = next.unwrap_or_else(|| Arc::from(pool.rows.as_slice()));
            *spare = Some(mem::replace(published, next));
            pool.dirty = false;
        }
        // Rows no world lists and no pool took are freed here, by a write.
        self.unlisted.clear();
        self.publications += 1;
        self.world()
    }

    /// How many worlds the writer published: the first world, made with
    /// the store, is not one.
    pub(super) fn publications(&self) -> u64 {
        self.publications
    }

    /// How many times the pools were built from the membership: a build an
    /// ask or a window's end paid, or a generation of another pairing mode.
    pub(super) fn builds(&self) -> u64 {
        self.builds
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
