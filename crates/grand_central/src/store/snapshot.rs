//! The store's published world (design 4.1 and 5.5): what the store
//! publishes together of one membership.
//!
//! One invariant: **a decision reads one publication.** A world is
//! everything the store published together: the gateways' standing vetoes
//! by pool, each member's slot in the store's counts and the generation
//! they were published under ([`StoreSnapshot::spec`]). One load hands a
//! decision all of it, so the vetoes it reads and the engine with its
//! number are of one publication, never the vetoes of one and the engine of
//! the next. A published world never changes: a reader reads memory nobody
//! is writing, through no lock.
//!
//! **A world holds no row of a worker.** A decision reads a worker's facts
//! and its status on its worker object. A world names a worker by its id
//! alone: in a veto, and beside its slot.
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
//! What a membership build derives (the pool of each model and role, each
//! member's slot) is built once and shared by every world until the next
//! membership build. Between two membership builds a publication derives
//! nothing: it is one small object.
//!
//! **The store derives no pair class.** A model's pair classes are derived
//! by whoever holds its worker objects, from what its prefills and decodes
//! report ([`PairClasses::derive`](crate::store::pairing::PairClasses::derive)).
//!
//! **The store publishes at its own rate, never at its writers'.** Every
//! deciding thread fetches every publication, so the build the membership
//! writes left owed is published once per [`PUBLISH_WINDOW`], and a veto or
//! a leave at once, through the same door. A status change publishes no
//! world: none holds a status. The store names the instant the build is due
//! ([`MemoryStore::publish_due`](crate::store::MemoryStore::publish_due)).
//!
//! **A veto is published in its gateway's run.** The writer keeps
//! each gateway's list and publishes each pool's standing vetoes as one run
//! a gateway, by worker id. A veto's write builds its own gateway's run
//! again in the pools its worker is a member of and shares every other run
//! with the publication before. A membership build, which moves the pools,
//! builds every run again, and one paid at a window's end drops the vetoes
//! past their end first. A veto leaves with its worker in the membership
//! build's own step, so no publication shows a veto on a worker that left.

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
    document::spec::{AppliedSpec, ViewLists},
    engine::view::{Role, WorkerId},
    store::{
        applied::{Applied, EngineCounts},
        workers::{WorkerFacts, WorkerRow},
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

/// What one membership build derives: each pool's index among a world's
/// vetoes, and each member's slot. Built once per membership build and
/// shared by every world published until the next.
#[derive(Default)]
struct Derived {
    /// The pools by model, then by role: each an index into a world's
    /// vetoes. A pool is every member serving one model in one role,
    /// whatever its status.
    by_model: FxHashMap<Box<str>, [Option<u32>; ROLES]>,
    /// How many pools the build derived.
    pools: usize,
    /// Every member the build read, by id, each with its slot in the
    /// counts: a member of a pool, and one that offers no model
    /// ([`StoreSnapshot::booked_of`]).
    slots: Box<[(WorkerId, WorkerSlot)]>,
}

/// One standing veto as its gateway's run in a pool carries it: the worker
/// it masks and its ends. A veto's write finds it in its run by the id,
/// and a reader is handed the id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PoolVeto {
    id: WorkerId,
    ends: VetoEnds,
}

/// A veto's two ends as the writer keeps them, with no kind. The veto masks
/// its worker until `until`. Its hold stands until `hold`, never past the
/// veto's own end: while it does, a decision of the gateway reads the worker
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

/// One gateway's standing vetoes in one pool, sorted by worker id: never
/// empty.
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
/// worker's place in its own list. It reads a veto by that id, so this
/// reader can stand on the writer's own lists, which are by worker id
/// ([`Vetoes`]), and no caller changes.
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

/// A pool's runs with `gateway`'s veto on worker `id` set to `ends`, or
/// cleared: that gateway's run built again, each other run shared.
fn with_veto(
    runs: &PoolVetoes,
    gateway: GatewayId,
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
    let own: Arc<[PoolVeto]> = match (own.binary_search_by_key(&id, |veto| veto.id), ends) {
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
                .chain(std::iter::once(PoolVeto { id, ends }))
                .chain(after.iter().copied())
                .collect()
        }
        // No veto of this gateway on the worker: nothing to clear.
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
    origin: Instant,
    derived: Arc<Derived>,
    /// The standing vetoes by pool, each pool's by gateway: empty while no
    /// veto stands in the store.
    vetoes: Arc<[PoolVetoes]>,
    /// The generation this publication was made under: its engine and its
    /// number, read with the vetoes in one load.
    spec: Arc<Applied>,
    /// The served decider reads this with the world, before following
    /// the pointer to the generation.
    #[cfg(feature = "controller")]
    samples_decisions: bool,
}

impl std::fmt::Debug for StoreSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoreSnapshot")
            .field("pools", &self.derived.pools)
            .finish_non_exhaustive()
    }
}

impl StoreSnapshot {
    /// The generation this world was published under: the engine a decision
    /// beside these vetoes runs, and its number. The store starts under the
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

    /// `at` in the store's clock: microseconds since the store's origin.
    pub fn micros(&self, at: Instant) -> u64 {
        micros(self.origin, at)
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

/// Whether one member names an engine family or a zone outside the standing
/// generation's lists, as last read ([`Writer::relist`], [`Writer::list`]).
#[derive(Clone, Copy)]
struct Held {
    id: WorkerId,
    /// The revision of the member's row at the read: a write of the row's
    /// facts moves it.
    revision: u64,
    engine_outside: bool,
    zone_outside: bool,
}

impl Held {
    /// What `member` reads under `lists`: the two bits alone, so its labels
    /// are not read ([`ViewLists::outside`]).
    fn read(lists: &ViewLists, member: &WorkerRow) -> Self {
        let facts = &member.facts;
        let (engine_outside, zone_outside) =
            lists.outside(&facts.engine_kind, facts.zone.as_deref());
        Self {
            id: member.id,
            revision: member.revision,
            engine_outside,
            zone_outside,
        }
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

/// One member's place in one pool.
#[derive(Clone, Copy)]
struct Place {
    id: WorkerId,
    pool: u32,
}

/// The write side of the world, behind one lock of the store: which pools
/// each member stands in, the gateways' vetoes, the worlds a reader still
/// holds, and the publication window. Every write that publishes hands
/// back the world it made; the store shows it under the same lock, so the
/// worlds reach the deciders in the order they were made.
pub(super) struct Writer {
    window: Duration,
    origin: Instant,
    derived: Arc<Derived>,
    /// How many times the pools were built from the membership.
    builds: u64,
    /// Every replaced world a reader still held when the cell let it go
    /// ([`Self::replaced`]), until a publication finds no reader holding it
    /// and lets it go. The cell holds the standing world and the writer
    /// each of these, so a decision that drops the last reader's hold
    /// frees nothing: not the world, its vetoes, its derived tables nor its
    /// generation.
    worlds: Vec<Arc<StoreSnapshot>>,
    /// How many worlds the writer published.
    publications: u64,
    /// Every member's places, sorted by worker id and then by pool: a
    /// worker serving several models has a place in each model's pool.
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
    /// What each member reads under the standing generation's lists, in id
    /// order, beside the revision of its row it was read at. A
    /// membership build reads a member again only where that revision
    /// moved, so it reads the members written since the last build and
    /// carries every other's. Empty while the generation's lists name no
    /// engine family and no zone: no member stands outside such lists.
    listed: Vec<Held>,
    /// The build before's `listed`, kept for its room.
    listed_spare: Vec<Held>,
    /// The last publication the window counts from: a window's end.
    last_window: Option<Instant>,
    /// The build the membership writes since the last one left owed, by the
    /// instant the first of them was written: the build is due from then,
    /// or from the window's end where that is later. `None` while the
    /// pools are the membership's.
    owed: Option<Instant>,
}

impl Writer {
    /// The write side of a store with no member, whose clock starts at
    /// `origin`. The engine of its unbound start counts the panics it
    /// holds into `counts`, the store's, as every engine behind it does.
    pub(super) fn new(window: Duration, origin: Instant, counts: &EngineCounts) -> Self {
        Self {
            window,
            origin,
            derived: Arc::default(),
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

    /// A world of everything published so far: one small object. Nothing
    /// derived is built here.
    pub(super) fn world(&self) -> Arc<StoreSnapshot> {
        Arc::new(StoreSnapshot {
            origin: self.origin,
            derived: Arc::clone(&self.derived),
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
    /// membership, read again under them, so the members outside the lists
    /// are counted under the generation's own lists from its first world.
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

    /// Read every member under the standing generation's lists and count
    /// those outside them. For an apply whose lists differ: a membership
    /// build reads the members written since the last and carries every
    /// other's ([`Writer::relist`]). Lists that name no engine family and
    /// no zone leave no member outside: no member is read under them.
    fn list<'a>(&mut self, members: impl Iterator<Item = (&'a WorkerRow, WorkerSlot)>) {
        let Self {
            spec,
            outside,
            listed: held,
            ..
        } = self;
        let lists = spec.lists();
        *outside = OutsideLists::default();
        held.clear();
        if !lists.names_a_family_or_zone() {
            return;
        }
        for (member, ..) in members {
            let read = Held::read(lists, member);
            outside.engines += u32::from(read.engine_outside);
            outside.zones += u32::from(read.zone_outside);
            held.push(read);
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

    /// How many members' reads under the standing generation's lists are
    /// kept ([`Self::list`], [`Self::relist`]).
    #[cfg(test)]
    pub(super) fn listed(&self) -> usize {
        self.listed.len()
    }

    /// What the members of a membership build read under the standing
    /// generation's lists, and the count of those outside them: carried
    /// from the last read where a member's row stands at the revision it
    /// was read at, read again where it moved or is new. A build so reads
    /// the members written since the last. `members` stand in id order.
    ///
    /// Lists that name no engine family and no zone leave no member outside
    /// them. Under such lists the pass ends at its first test: nothing was
    /// read under them, and the apply that brought them cleared what stood
    /// ([`Self::list`]).
    fn relist(&mut self, members: &[(WorkerId, &WorkerRow, WorkerSlot)]) {
        let lists = self.spec.lists();
        if !lists.names_a_family_or_zone() {
            return;
        }
        let outgoing = mem::take(&mut self.listed);
        let mut incoming = mem::take(&mut self.listed_spare);
        incoming.clear();
        let mut from = 0;
        let mut outside = OutsideLists::default();
        for &(_, member, _) in members {
            while outgoing.get(from).is_some_and(|held| held.id < member.id) {
                from += 1;
            }
            let read = match outgoing.get(from) {
                Some(held) if held.id == member.id && held.revision == member.revision => *held,
                _ => Held::read(lists, member),
            };
            outside.engines += u32::from(read.engine_outside);
            outside.zones += u32::from(read.zone_outside);
            incoming.push(read);
        }
        self.listed = incoming;
        self.listed_spare = outgoing;
        self.outside = outside;
    }

    /// Derive the pools of the membership `members` and publish them
    /// through the one door: each pool's index, each member's places and
    /// its slot, and the vetoes' runs by pool. Whether a member stands
    /// outside the standing generation's lists is read where its row's
    /// revision moved since it was last read, and carried where it stands
    /// ([`Self::relist`]).
    pub(super) fn rebuild<'a>(
        &mut self,
        members: impl Iterator<Item = (&'a WorkerRow, WorkerSlot)>,
    ) -> Arc<StoreSnapshot> {
        // A build takes in every membership write: none stays owed.
        self.owed = None;
        self.builds += 1;
        // The members' ids are read by the vetoes alone: a membership build
        // with no veto standing does not collect them.
        let standing = !self.vetoes.by_gateway.is_empty();
        let mut ids = Vec::new();
        // The members in id order, by this write's one sort, each id beside
        // its row and the slot its write gave it: an id is one member, so
        // the places, the slots and the ids are pushed in order.
        let mut members: Vec<(WorkerId, &'a WorkerRow, WorkerSlot)> = members
            .map(|(member, slot)| (member.id, member, slot))
            .collect();
        members.sort_unstable_by_key(|&(id, ..)| id);
        // What the members read under the standing generation's lists, in a
        // walk of its own: the walk below holds no line of it. A generation
        // with no list costs the build this one test. Nothing was read
        // under it, and the apply that took the lists away cleared what
        // stood ([`Self::list`]).
        if !self.spec.lists().is_empty() {
            self.relist(&members);
        }
        // A pool takes its index where the walk first meets a member of it.
        let mut by_model: FxHashMap<Box<str>, [Option<u32>; ROLES]> = FxHashMap::default();
        let mut pools = 0u32;
        self.places.clear();
        for &(_, member, _) in &members {
            if standing {
                ids.push(member.id);
            }
            let facts = &member.facts;
            let side = role_slot(facts.role);
            for model in &facts.models {
                let pool = match by_model.get_mut(model.as_str()) {
                    Some(sides) => *sides[side].get_or_insert_with(|| {
                        pools += 1;
                        pools - 1
                    }),
                    None => {
                        let mut sides = [None; ROLES];
                        sides[side] = Some(pools);
                        by_model.insert(model.as_str().into(), sides);
                        pools += 1;
                        pools - 1
                    }
                };
                // No model read before this one is looked at again: a model
                // listed twice leaves its place twice, for the sort below.
                self.places.push(Place {
                    id: member.id,
                    pool,
                });
            }
        }
        // A member's places stand by pool: a veto's write walks them so. A
        // model listed twice is one place in its pool: the two it left
        // stand side by side here, and one stays.
        self.places
            .sort_unstable_by_key(|place| (place.id, place.pool));
        self.places.dedup_by_key(|place| (place.id, place.pool));
        // Every member's id and slot, in the id order of this write's one
        // sort: a member in a pool, and one that offers no model.
        let slots: Box<[(WorkerId, WorkerSlot)]> =
            members.iter().map(|&(id, _, slot)| (id, slot)).collect();
        debug_assert!(slots.is_sorted_by(|a, b| a.0 < b.0));
        self.derived = Arc::new(Derived {
            by_model,
            pools: pools as usize,
            slots,
        });
        // A worker's vetoes leave with it, in this write's own hold: no
        // publication shows a veto on a worker that left. The ids stand in
        // order as the walk above pushed them.
        if standing {
            debug_assert!(ids.is_sorted_by(|a, b| a < b));
            if let Some(vetoes) = self.vetoes.of_members(&ids) {
                self.vetoes = Arc::new(vetoes);
            }
        }
        self.derive_vetoes();
        self.publish()
    }

    /// Every pool's runs built again from the gateways' lists and the
    /// members' places: for a membership build, which moves the pools. A
    /// veto's own write builds its gateway's runs alone
    /// ([`Self::write_veto`]). No list at all while no veto stands.
    fn derive_vetoes(&mut self) {
        if self.vetoes.by_gateway.is_empty() {
            self.pool_vetoes = Arc::default();
            return;
        }
        // The gateways are taken in order, and a gateway's list is sorted
        // by worker id: each pool's runs come out sorted by gateway and
        // each run by worker id, with no sort of entries.
        let mut gateways: Vec<_> = self.vetoes.by_gateway.iter().collect();
        gateways.sort_unstable_by_key(|&(&gateway, _)| gateway);
        let mut pools: Vec<Vec<(GatewayId, Vec<PoolVeto>)>> = vec![Vec::new(); self.derived.pools];
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
                    let veto = PoolVeto { id, ends };
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

    /// One veto's write as the pools carry it: in each pool the worker is
    /// a member of, the writing gateway's run takes the veto, moves its end
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
        self.pool_vetoes = (0..self.derived.pools)
            .map(|pool| {
                let runs = self.pool_vetoes.get(pool).unwrap_or(&none);
                let member = places.next_if(|place| place.pool as usize == pool);
                if member.is_some() {
                    with_veto(runs, gateway, id, ends)
                } else {
                    Arc::clone(runs)
                }
            })
            .collect();
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
    /// `members`, the standing membership. A host pays it at
    /// the window's end, `at`, which opens the next window and lets the
    /// vetoes past their end go. An asked build names no instant and leaves
    /// the window as it stands. `None` when no build is owed: another call
    /// paid it.
    #[must_use]
    pub(super) fn build_owed<'a>(
        &mut self,
        at: Option<Instant>,
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
        Some(self.rebuild(members))
    }

    /// The end of the window `last` opened; `None` for a window so long
    /// that the clock cannot add it, which no instant ends.
    fn window_end(&self, last: Instant) -> Option<Instant> {
        last.checked_add(self.window)
    }

    /// The one door every publication goes through: one world of what the
    /// writer holds. A publication never waits for a reader.
    fn publish(&mut self) -> Arc<StoreSnapshot> {
        // A world no reader holds any more is let go here, by a write.
        self.worlds
            .retain_mut(|world| Arc::get_mut(world).is_none());
        self.publications += 1;
        self.world()
    }

    /// How many worlds the writer published: the first world, made with
    /// the store, is not one.
    pub(super) fn publications(&self) -> u64 {
        self.publications
    }

    /// How many times the pools were built from the membership: a build an
    /// ask or a window's end paid, or one a laid snapshot made.
    pub(super) fn builds(&self) -> u64 {
        self.builds
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
