//! The policy engine in a gateway's own process.
//!
//! The gateway keeps its workers where it keeps them with the classic
//! policies: its registry publishes one shared list of worker objects a
//! model at each membership change, and a worker's health and its vetoes
//! live on the worker object. The gateway's classic placement pass holds
//! the rules of a placement: which workers a request may take, what a retry
//! skips and what becomes of a pool no worker of which passes. This front
//! gives that pass its pick:
//!
//! - **Once a membership.** [`PolicyEngine::pool_members`] makes the
//!   engine's value of a pool's shared list: the list's handle and the
//!   pool's own word. The gateway keeps the value beside the list it was
//!   made from, for as long as that list stands. No worker is read for it
//!   and nothing of a worker is copied.
//! - **Once a place.** The gateway builds one policy for each place it
//!   places for: a model and a phase. [`PolicyEngine::place`] finds the
//!   place's profile by the document's own bindings, there and at no
//!   request.
//! - **Once a request.** [`PolicyEngine::pick`] asks for one pick over the
//!   pool, under a profile the engine picks by its picker alone
//!   ([`Place::Alone`]). The pick turns the pool's own word and reads the
//!   worker the turn lands on, by the pass's own test of a worker: no
//!   row, no table, no request fact, no lock, and nothing allocated. A
//!   list nothing is kept of (a part of a pool, as the pass narrowed it)
//!   is passed over once ([`PolicyEngine::pick_among`]).
//!
//! **A whole decision.** A gateway whose routers ask the engine for the
//! whole decision asks through [`DecisionEngine`], as it asks a controller's
//! engine: the binding, its rules and the profile, over the list its
//! request's maker hands ([`RequestMaker::workers`]), with a lease the
//! request's guard holds to its end. The engine files no lease and keeps no
//! note of a request: the guard lends back what a later word needs
//! ([`HeldDecision`]). A single request is decided in [`single`], and a pair
//! in [`pair`], with its decode asked again behind its prefill.
//!
//! The engine keeps no worker list, no health and no count of requests in
//! flight. What it keeps is its stages' own: a rotation, and for the stages
//! that read who holds a prompt, the record of the prefixes it sent
//! ([`overlap`]).

mod keys;
pub mod overlap;
pub(super) mod pair;
mod single;

use std::{
    cell::RefCell,
    hash::Hasher,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, OnceLock,
    },
    time::Instant,
};

use rustc_hash::FxHasher;
use smg_worker::{identity::basic::Worker, runtime::RoutingState};

use self::overlap::{Affinity, PoolAffinity};
use super::{
    decision::ShadowKind,
    decision_engine::{
        AlonePick, DecideOutcome, Decided, DecisionEngine, DecisionRefused, HeldDecision, Request,
        RequestMaker, SweepOutcome, WorkerLists,
    },
    framework::{
        release, AloneProfile, EncodeItems, EncodePlaced, NotAlone, Picker, Place, PoolWord,
    },
    lease_events,
    prompt_hashes::{HashSeeds, PromptHashes},
    records::{DecisionClock, PoolRecords},
    view::{fingerprint, Phase, WireModes, WorkerId},
};
use crate::{
    document::spec::AppliedSpec,
    names::intern,
    policies::consistent_hashing::PoolRing,
    stages::Lifecycle,
    store::{pairing::Refusal, GatewayId, Said},
    tracking::lease::{EventKind, EventSet, Side},
};

/// The engine of one policy document, for a gateway that decides in its own
/// process.
///
/// Built once, at the gateway's start, from the document in force. It holds
/// the document's stages and their state, and no worker.
#[derive(Debug)]
pub struct PolicyEngine {
    spec: AppliedSpec,
    /// The pools made so far: each draws the place its turns start at from
    /// its number.
    epochs: AtomicU64,
    /// The one clock of the engine's decisions: a decision's instant and a
    /// load report's sample instant are counted from its origin.
    clock: DecisionClock,
    /// How many whole decisions panicked ([`DecisionEngine::faults`]).
    faults: AtomicU64,
    /// How many words of a lease panicked in a stage that heard them.
    unheard: AtomicU64,
    /// The id of the last lease issued. A pick that is issued one writes
    /// it: one atomic step, on cache lines of its own.
    leases: OwnWord,
    /// The kinds of event the document takes ([`Lifecycle::hears`]), read
    /// once, here. A document that takes no word on a lease
    /// ([`EventSet::LEASE`]) costs a request no lease, no id and no word,
    /// and its plain requests are answered a place ([`Decided::Alone`]).
    hears: EventSet,
    /// What the affinity stages read across decisions: the host's indexes
    /// of cache events, once it handed them, and the engine's record of the
    /// prefixes it sent, one a model ([`Self::affinity`]).
    affinity: Affinity,
}

/// One pool of one model as the engine keeps it: the gateway's shared list
/// of worker objects and the pool's own word.
///
/// Made by [`PolicyEngine::pool_members`] from a list the gateway's
/// registry published, and kept by the gateway beside that list. It holds
/// the list itself, not a copy: a worker's live facts are read on the
/// list's objects.
#[derive(Debug)]
pub struct PoolMembers {
    workers: Arc<[Arc<dyn Worker>]>,
    /// The pool's own word, for the picker that places over the pool by
    /// itself ([`PolicyEngine::pick`]): a rotation counts its turns here.
    rotation: OwnWord,
    /// What a whole decision reads of the list beside its workers' live
    /// state ([`Self::records`]): made at the list's first whole decision,
    /// and by no pick a placement pass asks ([`PolicyEngine::pick`]).
    records: OnceLock<PoolRecords>,
    /// The wires the list's workers are dialed on ([`Self::wires`]): read
    /// of their registrations at the first whole decision over the list.
    wires: OnceLock<WireModes>,
    /// The model of the list's first single request and the profile the
    /// engine picks alone for it, where the document holds one
    /// ([`Self::alone`]).
    alone: OnceLock<(Box<str>, Option<AloneProfile>)>,
    /// The list's hash ring ([`Self::ring`]): built off the path of a
    /// request, and by no decision ([`PolicyEngine::fill_ring`]).
    ring: OnceLock<PoolRing>,
    /// Whether the list is a part of a pool, one ask's value
    /// ([`Self::part`]). Its turns are then its workers' set's own, kept
    /// by the picker ([`PolicyEngine::pick_among`]): `rotation` counts
    /// none, and nothing is kept in `alone`.
    part: bool,
    /// What a decision whose stages read affinity keeps of the list
    /// ([`Self::affinity`]): read at the list's first such decision, and by
    /// no other.
    affinity: PoolAffinity,
}

/// A word every pick over a pool writes, on cache lines of its own: what
/// stands beside it, in this value and in the gateway's snapshot that keeps
/// the value, is read by every request and written by none.
#[derive(Debug)]
#[repr(align(128))]
struct OwnWord(AtomicU64);

impl PoolMembers {
    /// The value of a part of a pool, for one ask: `part` is the workers a
    /// host narrowed a request to, in their pool's order, as a list of
    /// their own. The host keeps nothing of the value.
    ///
    /// A pick over it takes the part's own turn
    /// ([`PolicyEngine::pick_among`]): the picker keeps a rotation for the
    /// set of the part's workers that pass, so the requests narrowed to one
    /// part take its workers in turn, whatever turns their pool takes
    /// between them. Making the value reads no worker, draws nothing and
    /// writes no word two requests share.
    #[must_use]
    pub fn part(part: &Arc<[Arc<dyn Worker>]>) -> Self {
        Self {
            workers: Arc::clone(part),
            rotation: OwnWord(AtomicU64::new(0)),
            records: OnceLock::new(),
            wires: OnceLock::new(),
            alone: OnceLock::new(),
            // A part takes no ring: nothing fills one ask's value, and a
            // request that reads a ring over it is placed as one that
            // carries no key ([`RequestFacts::ring_awaited`]).
            ring: OnceLock::new(),
            part: true,
            // So is what its affinity stages keep of it.
            affinity: PoolAffinity::default(),
        }
    }

    /// The workers of the pool: the list the value was made from.
    #[must_use]
    pub fn workers(&self) -> &Arc<[Arc<dyn Worker>]> {
        &self.workers
    }

    /// The list's records under the document `spec`, on `clock`: the half
    /// of each worker's record that no request changes, the pool's engines
    /// and its number ([`PoolRecords`]). Made at the first call and kept
    /// for as long as the list stands; every later call is one read. No
    /// live value of a worker is kept in them.
    #[must_use]
    pub fn records(&self, spec: &AppliedSpec, clock: &DecisionClock) -> &PoolRecords {
        self.records
            .get_or_init(|| PoolRecords::new(&self.workers, spec, clock))
    }

    /// How many times the list's reports were converted so far
    /// ([`PoolRecords::conversions`]): zero for a list no decision made
    /// records of. It makes none: what a bench counts against the snapshots
    /// published, over a list its engine decides.
    #[must_use]
    pub fn conversions(&self) -> u64 {
        self.records.get().map_or(0, PoolRecords::conversions)
    }

    /// Every wire a worker of the list is dialed on, as the workers'
    /// registrations state it: read once a list, at the first call. A
    /// request that dials each of them may be sent to any worker of the
    /// list. A host hands a whole decision such a list
    /// ([`RequestMaker::workers`]): the pool of its request's route, or a
    /// list of its own for a request held to one wire.
    #[must_use]
    pub fn wires(&self) -> WireModes {
        *self.wires.get_or_init(|| {
            self.workers.iter().fold(WireModes::NONE, |wires, worker| {
                wires.union(WireModes::from(*worker.connection_mode()))
            })
        })
    }

    /// The cell of the list's hash ring: each worker of the list at its
    /// positions, by its id and its stable key. Empty until the gateway
    /// fills it, off the path of a request ([`PolicyEngine::fill_ring`]):
    /// a decision reads the cell, lends the ring where it stands, and
    /// neither builds one nor waits for one. A list no ring policy decides
    /// builds none. The ring is never changed: a membership change
    /// publishes another list, which takes a ring of its own.
    #[must_use]
    pub fn ring(&self) -> &OnceLock<PoolRing> {
        &self.ring
    }

    /// What a decision whose stages read affinity keeps of the list: each
    /// worker's number in its model's index of cache events and the list's
    /// warming rows, as a ruling read again at most once a second
    /// ([`PoolAffinity`]). A caller that asks a whole decision of its own
    /// over the list's records hands it to [`overlap::decide`], as a single
    /// request's decision here does.
    #[must_use]
    pub fn affinity(&self) -> &PoolAffinity {
        &self.affinity
    }

    /// The profile the engine picks alone for a single request of `model`
    /// over this list ([`Place::Alone`]), under a binding that states no
    /// sessions rule: `None` where the place's decision asks more than one
    /// pick. Found once a list, for the model of its first request: a
    /// gateway's list is one model's. Another model's request is answered
    /// by the document at its call.
    fn alone(&self, spec: &AppliedSpec, model: &str) -> Option<AloneProfile> {
        let find = || {
            let engine = spec.engine();
            let Place::Alone(profile) = engine.place_for(model, Phase::Single) else {
                return None;
            };
            // A sessions rule pins a keyed request ahead of the profile.
            let ruled = engine.binding(model, Phase::Single)?.sessions.is_some();
            (!ruled).then_some(profile)
        };
        // A part is one ask's: its answer is kept nowhere, so nothing is
        // allocated for it.
        if self.part {
            return find();
        }
        let (named, alone) = self.alone.get_or_init(|| (model.into(), find()));
        if **named == *model {
            *alone
        } else {
            find()
        }
    }
}

impl PolicyEngine {
    /// The engine of `spec`, the document in force.
    #[must_use]
    pub fn new(spec: AppliedSpec) -> Self {
        let hears = spec.engine().hears();
        Self {
            hears,
            spec,
            epochs: AtomicU64::new(0),
            clock: DecisionClock::new(),
            faults: AtomicU64::new(0),
            unheard: AtomicU64::new(0),
            leases: OwnWord(AtomicU64::new(0)),
            affinity: Affinity::default(),
        }
    }

    /// What the engine keeps for its affinity stages: the gateway hands its
    /// indexes of cache events here, once, at its start
    /// ([`Affinity::set_indexes`]), and reads the record's own count
    /// ([`Affinity::left_out`]).
    #[must_use]
    pub fn affinity(&self) -> &Affinity {
        &self.affinity
    }

    /// The worker at `address` left its pool or was replaced: the records
    /// of sent prefixes forget it, so no later request is named a worker
    /// that is gone. The gateway says it where it tells its classic
    /// policies. A worker no record holds costs one lookup a model.
    pub fn forget(&self, address: &str) {
        // The id every decision knew the worker by ([`records::worker_id`]).
        self.affinity
            .forget(WorkerId(fingerprint(address.as_bytes())));
    }

    /// The clock of the engine's decisions. A caller that makes a list's
    /// records ([`PoolMembers::records`]) makes them on it.
    #[must_use]
    pub fn clock(&self) -> &DecisionClock {
        &self.clock
    }

    /// How many words of a lease (its dispatch, its end, its release)
    /// panicked in a stage that heard them. The word stands as said: the
    /// request it is of goes on, and the stages after the one that
    /// panicked did not hear it.
    #[must_use]
    pub fn unheard(&self) -> u64 {
        self.unheard.load(Ordering::Relaxed)
    }

    /// The document the engine decides under.
    #[must_use]
    pub fn spec(&self) -> &AppliedSpec {
        &self.spec
    }

    /// The engine's value of one pool: `pool` is the shared list the
    /// gateway's registry published for it.
    ///
    /// Called once a list, by the gateway, which keeps the value for as
    /// long as the list stands. It reads no worker, so its cost does not
    /// grow with the list.
    #[must_use]
    pub fn pool_members(&self, pool: &Arc<[Arc<dyn Worker>]>) -> PoolMembers {
        let epoch = self.epochs.fetch_add(1, Ordering::Relaxed) + 1;
        PoolMembers {
            workers: Arc::clone(pool),
            // A pool starts its turns at a drawn worker, so a fleet whose
            // membership keeps changing does not lean on its first one.
            rotation: OwnWord(AtomicU64::new(mixed(epoch) >> 32)),
            records: OnceLock::new(),
            wires: OnceLock::new(),
            alone: OnceLock::new(),
            ring: OnceLock::new(),
            part: false,
            affinity: PoolAffinity::default(),
        }
    }

    /// Whether the document places a single request of `model` by a hash
    /// ring: a stage that may decide such a request reads one. The gateway
    /// keeps a ring for each list of such a model's workers
    /// ([`Self::fill_ring`]), and none for another model's.
    #[must_use]
    pub fn reads_ring(&self, model: &str) -> bool {
        let engine = self.spec.engine();
        engine.needs_for(model, Phase::Single).ring
    }

    /// Builds the hash ring of `members`, a list of `model`'s workers,
    /// where the document places a single request of the model by a ring
    /// ([`Self::reads_ring`]), and says whether the list's ring stands.
    /// `false` where it places such a request by none: the list takes none.
    ///
    /// For the gateway, off the path of a request, where it hears of a list:
    /// the build hashes every worker's positions and sorts them, and no
    /// decision does it ([`PoolMembers::ring`]). It makes the list's
    /// records where no decision has yet. A second call is one read.
    pub fn fill_ring(&self, model: &str, members: &PoolMembers) -> bool {
        if !self.reads_ring(model) {
            return false;
        }
        members
            .ring
            .get_or_init(|| keys::ring_of(members.records(&self.spec, &self.clock)));
        true
    }

    /// What the document holds for one place the gateway builds a policy
    /// for, `model` at `phase` ([`Engine::place_for`]). The first binding that
    /// matches names the place's profile. The answer is that profile where
    /// one pick of its picker is its whole decision ([`Place::Alone`]),
    /// the stage of it that asks more where it is not
    /// ([`Place::NotAlone`]), [`Place::Unbound`] where no binding names
    /// the model for the phase, and [`Place::NoSide`] where the binding
    /// that names it plans no request of the phase. A policy that serves
    /// every model is asked for with the empty model id, which the
    /// bindings of every model alone match.
    ///
    /// Asked where the gateway builds the place's policy, and at no
    /// request: a policy's name finds nothing here.
    ///
    /// [`Engine::place_for`]: super::framework::Engine::place_for
    #[must_use]
    pub fn place(&self, model: &str, phase: Phase) -> Place<'_> {
        self.spec.engine().place_for(model, phase)
    }

    /// The name of `profile` as the document writes it, as a label that
    /// lives as long as the process: a caller that counts its picks under
    /// the profile's name hands it to a metric. One allocation a distinct
    /// name.
    #[must_use]
    pub fn profile_label(&self, profile: AloneProfile) -> &'static str {
        let profiles = self.spec.engine().profiles();
        profiles
            .get(usize::from(profile.0))
            .map_or("", |profile| intern(&profile.name))
    }

    /// The first place a binding of the document names whose profile the
    /// engine does not decide by one pick ([`Engine::not_alone`]). `None`
    /// where it decides every place of the document so. A gateway that
    /// starts under a document its operator supplied refuses the start by
    /// what this names: nothing else would decide that place.
    ///
    /// [`Engine::not_alone`]: super::framework::Engine::not_alone
    #[must_use]
    pub fn not_alone(&self) -> Option<NotAlone<'_>> {
        self.spec.engine().not_alone()
    }

    /// One pick under `profile` over `members`, the value this engine made
    /// of a pool ([`Self::pool_members`]): the place in the pool's list of
    /// the worker picked. `None` where no worker of the pool passes.
    ///
    /// The entry for a caller that holds every other rule of a placement
    /// itself, as a gateway's classic placement pass does: which workers a
    /// request may take, what a retry skips, and what becomes of a pool no
    /// worker of which passes. The picker turns the pool's own word, kept
    /// in `members`, and the worker a turn lands on is read once and tested
    /// by `admits`, the caller's own test of a worker's read: the one its
    /// pass asks of every candidate, so the rule of a lifted veto has one
    /// home. No row is read, no table looked up, no lock taken and nothing
    /// allocated. Where the picker's turns all land on refused workers the
    /// pool is passed over once, as [`Self::pick_among`] passes over a list.
    /// A part of a pool ([`PoolMembers::part`]) has no word of its own: the
    /// pick over it is [`Self::pick_among`]'s.
    #[must_use]
    pub fn pick(
        &self,
        profile: AloneProfile,
        members: &PoolMembers,
        admits: impl Fn(RoutingState) -> bool + Copy,
    ) -> Option<usize> {
        let picker = self.picker_of(profile)?;
        if members.part {
            return among(picker, &members.workers, admits);
        }
        pick_by(picker, &members.workers, &members.rotation.0, admits)
    }

    /// One pick under `profile` among `workers`, a list this engine keeps
    /// nothing of (a part of a pool, as a caller narrowed it): the place in
    /// `workers` of the worker picked. `None` where no worker passes.
    ///
    /// The list is passed over once, a read and the caller's test
    /// (`admits`) each, and the picker picks among the workers that pass: a
    /// rotation turns
    /// the one their set keeps, named by the workers' handles. The list of
    /// them is the calling thread's own: nothing is allocated after a
    /// thread's first picks.
    #[must_use]
    pub fn pick_among(
        &self,
        profile: AloneProfile,
        workers: &[Arc<dyn Worker>],
        admits: impl Fn(RoutingState) -> bool + Copy,
    ) -> Option<usize> {
        among(self.picker_of(profile)?, workers, admits)
    }

    /// The picker of `profile`.
    #[inline]
    fn picker_of(&self, profile: AloneProfile) -> Option<&dyn Picker> {
        let profiles = self.spec.engine().profiles();
        Some(profiles.get(usize::from(profile.0))?.picker.as_ref())
    }

    /// A whole decision under the engine's catch: a panic in it costs its
    /// request alone and is counted ([`DecisionEngine::faults`]).
    #[inline]
    fn caught<R>(
        &self,
        decide: impl FnOnce() -> Result<R, DecisionRefused>,
    ) -> Result<R, DecisionRefused> {
        match catch_unwind(AssertUnwindSafe(decide)) {
            Ok(decided) => decided,
            Err(payload) => {
                release(payload);
                self.faults.fetch_add(1, Ordering::Relaxed);
                Err(DecisionRefused::Faulted)
            }
        }
    }

    /// The one entry ([`DecisionEngine::decide_under`]) for a host that
    /// hands a clock and no instant: `clock` is read once, where the
    /// decision is dated. A place ([`Decided::Alone`]) dates nothing, so it
    /// is answered with no read of a clock, and with no copy of a decision:
    /// the catch hands the place back, and a whole decision is written
    /// beside it.
    ///
    /// # Errors
    ///
    /// [`DecisionRefused`]: the engine made no decision for the request.
    pub fn decide_at(
        &self,
        maker: &impl RequestMaker,
        clock: impl FnOnce() -> Instant,
    ) -> Result<Decided, DecisionRefused> {
        let mut whole = None;
        let alone = self.caught(|| self.placed_or(maker, clock, &mut whole))?;
        match (alone, whole) {
            (Some(pick), _) => Ok(Decided::Alone(pick)),
            (None, Some((decided, later))) => Ok(Decided::Whole(decided, later)),
            // The pass writes its decision ahead of every answer but a
            // place.
            (None, None) => Err(DecisionRefused::Faulted),
        }
    }

    /// The decision's pass: the place it answers, or its whole decision
    /// written into `whole` with whether the request made stated the later
    /// decode.
    fn placed_or(
        &self,
        maker: &impl RequestMaker,
        clock: impl FnOnce() -> Instant,
        whole: &mut Option<(DecideOutcome, bool)>,
    ) -> Result<Option<AlonePick>, DecisionRefused> {
        // A plain request of a place one pick decides is decided ahead
        // of its making: the pick reads no fact of it.
        let mut picked = None;
        if let (Some(plain), WorkerLists::Single(members)) = (maker.plain(), maker.workers()) {
            if let Some(pick) = single::alone(self, maker.model(), plain, members) {
                // No stage takes a word of the request: the place is
                // the whole answer.
                if !self.hears.intersects(EventSet::LEASE) {
                    return Ok(Some(pick));
                }
                picked = Some((pick, members, plain));
            }
        }
        let now = clock();
        if let Some((pick, members, plain)) = picked {
            if let Some(decided) = single::noted(self, pick, members, plain, now) {
                *whole = Some((decided, false));
                return Ok(None);
            }
        }
        let mut request = maker.make(&self.spec);
        let later = request.constraints.decode_later;
        let decided = match (request.facts.phase, maker.workers()) {
            (Phase::Single, WorkerLists::Single(members)) => {
                single::decide(self, &mut request, members, maker.loads(), now)
            }
            // A pair is decided at its prefill, over its host's two
            // lists.
            (Phase::Prefill, WorkerLists::Pair(lists)) => {
                let decided = pair::decided(self, &lists, &mut request, None, now);
                *whole = Some((decided, later));
                return Ok(None);
            }
            // A request that hands no list of its own is decided over
            // no worker.
            (Phase::Single | Phase::Prefill, _) => single::over_none(self, &request),
            (Phase::Decode | Phase::Encode, _) => return Err(DecisionRefused::PairSide),
        };
        let decided = single::issued(self, decided, &request, now);
        *whole = Some((decided, later));
        Ok(None)
    }
}

/// The engine of a gateway's own process, asked for whole decisions.
///
/// It decides over the lists the request's maker hands and reads a worker
/// where the gateway keeps it. It files no lease: an id is one step of a
/// counter, the stages hear the issue inside the decision, on the calling
/// thread, and each later word is told from what the request's guard lends
/// back ([`HeldDecision`]). A lease whose dispatch is never named ends at
/// its guard's drop: no table holds it, and no pass looks for it.
///
/// What every pick shares is the id counter alone, one atomic step on cache
/// lines of its own. No lock of the engine is taken: a stage's own lock is
/// its stage's.
///
/// A panic in a decision costs that request its decision and is counted
/// ([`DecisionEngine::faults`]). A panic in a stage that hears a word is
/// held here too and counted ([`PolicyEngine::unheard`]): the request goes
/// on.
impl DecisionEngine for PolicyEngine {
    fn decide_under(
        &self,
        _: GatewayId,
        maker: &impl RequestMaker,
        now: Instant,
    ) -> Result<Decided, DecisionRefused> {
        self.decide_at(maker, move || now)
    }

    // The guard asks only where the pair's prefill is held and its decode
    // stands chosen. A call that hands no pair's lists is decided over no
    // worker, and its guard keeps the decode the first pick named.
    fn decide_decode_under(
        &self,
        _: GatewayId,
        maker: &impl RequestMaker,
        held: &HeldDecision<'_>,
        now: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        self.caught(|| {
            let mut request = maker.make(&self.spec);
            Ok(match maker.workers() {
                WorkerLists::Pair(lists) => {
                    pair::decided(self, &lists, &mut request, Some(held), now)
                }
                _ => single::issued(self, single::over_none(self, &request), &request, now),
            })
        })
    }

    // A side sent on its own: the encode family's pair has no entry here.
    fn decide_encode_pair(
        &self,
        _: GatewayId,
        _: &Request<'_>,
        _: EncodeItems<'_>,
        placed: &mut EncodePlaced,
        _: Instant,
    ) -> Result<DecideOutcome, DecisionRefused> {
        placed.clear();
        Err(DecisionRefused::PairSide)
    }

    fn say(&self, _: GatewayId, held: &HeldDecision<'_>, said: Said, _: Instant) {
        // A single's word no stage takes is told to nobody. A pair's
        // dispatch may carry its decode's issue: every word of one is told.
        let kind = match said {
            Said::Dispatched(..) => EventKind::Dispatched,
            Said::Completed(_) => EventKind::Completed,
            Said::Released(..) => EventKind::Released,
        };
        if held.lease.side == Side::Single && !self.hears.contains(kind) {
            return;
        }
        let engine = self.spec.engine();
        let tell = || match held.lease.side {
            Side::Single => lease_events::single(engine, held, said),
            // A pair's guard holds its prefill side's lease.
            Side::Prefill => lease_events::pair(engine, held, said, |at| self.clock.micros(at)),
            // No lease is issued for a side on its own.
            Side::Decode | Side::Encode => {}
        };
        if let Err(payload) = catch_unwind(AssertUnwindSafe(tell)) {
            release(payload);
            self.unheard.fetch_add(1, Ordering::Relaxed);
        }
    }

    // No lease is filed, so no pass ends one: a lease ends at its guard's
    // word or drop. The session keys past their end leave the engine's
    // table here, off the path of a request: from this pass's first run no
    // read of a key sweeps the table. The records of sent prefixes take a
    // bounded step of the work they left waiting.
    fn sweep(&self, now: Instant) -> SweepOutcome {
        let sessions = self.spec.engine().sessions();
        sessions.sweep_due(self.clock.micros(now));
        self.affinity.sweep();
        SweepOutcome::default()
    }

    fn with_spec<R>(&self, read: impl FnOnce(&AppliedSpec) -> R) -> R {
        read(&self.spec)
    }

    // By the block size the model's cache events taught the host's index,
    // so the index finds the blocks, and by the document's where the index
    // knows none.
    fn prompt_hashes(
        &self,
        model: &str,
        tokens: Option<&[u32]>,
        text: Option<&str>,
        seeds: HashSeeds,
    ) -> PromptHashes {
        let size = self.affinity.block_size(model);
        PromptHashes::under_document(&self.spec, size, tokens, text, seeds)
    }

    // The host's own pair index says why a fleet pairs nothing: the engine
    // keeps none.
    fn pair_refusal(&self, _: &str, _: WireModes) -> Option<Refusal> {
        None
    }

    fn faults(&self) -> u64 {
        self.faults.load(Ordering::Relaxed)
    }

    // No shadow is asked here: a document that states one does not start.
    fn shadowed(&self, _: u32, _: ShadowKind) -> u64 {
        0
    }
}

/// [`PolicyEngine::pick`] by one test of a worker's read, compiled once a
/// test: the picker's own pick over the pool's word, then the pass.
#[inline(always)]
fn pick_by(
    picker: &dyn Picker,
    workers: &[Arc<dyn Worker>],
    word: &AtomicU64,
    admits: impl Fn(RoutingState) -> bool + Copy,
) -> Option<usize> {
    let pool = PoolWord {
        rows: workers.len(),
        word,
    };
    let picked = picker.pick_alone(&pool, &mut |at| {
        workers
            .get(at)
            .is_some_and(|worker| admits(worker.routing_state()))
    });
    if picked.is_some() {
        return picked;
    }
    among(picker, workers, admits)
}

/// [`PolicyEngine::pick_among`] by one test of a worker's read: the workers
/// of `workers` that pass, listed in the thread's list, their set named by
/// their handles' addresses in the list's order (one word hashed a worker,
/// and nothing of a worker read for it), and the picker's pick among them.
#[inline(always)]
fn among(
    picker: &dyn Picker,
    workers: &[Arc<dyn Worker>],
    admits: impl Fn(RoutingState) -> bool,
) -> Option<usize> {
    let pass = |passed: &mut Vec<u32>| {
        passed.clear();
        let mut set = FxHasher::default();
        for (worker, at) in workers.iter().zip(0u32..) {
            if admits(worker.routing_state()) {
                passed.push(at);
                set.write_usize(Arc::as_ptr(worker).addr());
            }
        }
        if passed.is_empty() {
            return None;
        }
        let at = picker.pick_among(set.finish(), passed.len());
        passed.get(at).map(|&position| position as usize)
    };
    PASSED.with(|passed| match passed.try_borrow_mut() {
        Ok(mut passed) => pass(&mut passed),
        // A picker asked for a pick from inside its own, on this thread:
        // that pick takes a list of its own.
        Err(_) => pass(&mut Vec::new()),
    })
}

thread_local! {
    /// The workers of a list that passed a pass, by their place in it: the
    /// calling thread's own list, so a pass allocates nothing after a
    /// thread's first picks.
    static PASSED: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// The SplitMix64 finalizer: every bit of `z` moves every bit of the word.
const fn finalized(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A word drawn from a count: counts one apart give words far apart.
const fn mixed(count: u64) -> u64 {
    finalized(count.wrapping_mul(GOLDEN))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use openai_protocol::worker::WorkerStatus;
    use smg_worker::{runtime::StallReason, test_support::ready_worker};

    use super::*;
    use crate::{
        document::spec::{Needs, PolicySpec},
        engine::framework::InTheWay,
    };

    const MODEL: &str = "model-a";

    fn engine_of(document: &PolicySpec) -> PolicyEngine {
        PolicyEngine::new(AppliedSpec::validate(document).expect("the document validates"))
    }

    fn preset(name: &str) -> PolicyEngine {
        let mut document = PolicySpec::preset(name).expect("a built-in preset");
        document.generation = 1;
        engine_of(&document)
    }

    fn pool(workers: usize) -> Arc<[Arc<dyn Worker>]> {
        (0..workers)
            .map(|at| ready_worker(&format!("http://worker-{at}:8000"), MODEL))
            .collect()
    }

    /// The profile of the model's single placements, which the engine picks
    /// alone.
    fn alone(engine: &PolicyEngine) -> AloneProfile {
        match engine.place(MODEL, Phase::Single) {
            Place::Alone(profile) => profile,
            other => panic!("not picked alone: {other:?}"),
        }
    }

    /// Round robin's preset with `filter` added to its profile's filters.
    fn round_robin_with(filter: &str) -> PolicyEngine {
        let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
        let profile = document
            .profiles
            .get_mut("round_robin")
            .expect("the preset's profile");
        profile.filter.push(filter.into());
        engine_of(&document)
    }

    /// The value made of a pool holds the shared list itself, and making it
    /// reads no worker: a pool of workers that are not ready is made as any
    /// other, and each pool made draws a start of its own.
    #[test]
    fn the_members_hold_the_shared_list() {
        let engine = preset("round_robin");
        let list = pool(5);
        for worker in list.iter() {
            worker.set_status(WorkerStatus::NotReady);
        }
        let members = engine.pool_members(&list);
        assert!(Arc::ptr_eq(members.workers(), &list));
        let again = engine.pool_members(&list);
        assert!(Arc::ptr_eq(again.workers(), &list));
        assert_ne!(
            members.rotation.0.load(Ordering::Relaxed),
            again.rotation.0.load(Ordering::Relaxed),
            "each pool made draws the place its turns start at"
        );
    }

    /// A part of a pool takes turns of its own: the requests narrowed to
    /// one part take its workers in turn, each over a value of the part
    /// made for it, whatever turns the pool takes between them.
    #[test]
    fn a_part_of_a_pool_takes_its_workers_in_turn() {
        let engine = preset("round_robin");
        let profile = alone(&engine);
        let list = pool(8);
        let members = engine.pool_members(&list);
        let part: Arc<[Arc<dyn Worker>]> = [1, 4, 6].map(|at| Arc::clone(&list[at])).into();
        let mut seen = [0_u32; 3];
        for _ in 0..9 {
            let turn = engine.pick(profile, &members, RoutingState::eligible);
            assert!(turn.is_some());
            let narrowed = PoolMembers::part(&part);
            assert!(Arc::ptr_eq(narrowed.workers(), &part));
            let at = engine.pick(profile, &narrowed, RoutingState::eligible);
            seen[at.expect("a worker of the part")] += 1;
        }
        assert_eq!(seen, [3; 3], "each worker of the part takes its turn");
    }

    /// A place is found by the document's bindings, a model and a phase. A
    /// profile's name finds nothing: a document that names its profile
    /// anything is answered as the preset is. A model no binding names is
    /// unbound, and so is a policy that serves every model (the empty id)
    /// under a document that binds some models only.
    #[test]
    fn a_place_is_found_by_its_model_and_phase() {
        let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
        let profile = document
            .profiles
            .remove("round_robin")
            .expect("the preset's profile");
        document.profiles.insert("anything".to_string(), profile);
        document.bindings[0].profile = Some("anything".to_string());
        document.bindings[0].matcher.model = "served-*".to_string();
        let engine = engine_of(&document);
        assert!(matches!(
            engine.place("served-a", Phase::Single),
            Place::Alone(_)
        ));
        assert_eq!(engine.place("other", Phase::Single), Place::Unbound);
        assert_eq!(engine.place("", Phase::Single), Place::Unbound);
        // The preset binds every model, the empty id among them.
        let every = preset("round_robin");
        assert_eq!(
            every.place("", Phase::Single),
            every.place(MODEL, Phase::Single)
        );
        assert!(matches!(every.place("", Phase::Single), Place::Alone(_)));
    }

    /// The places one pick decides: the picker places by itself, no stage
    /// scores, and the filters are the two tests of a worker's health.
    /// Round robin's preset is one. No other preset is one yet, and each
    /// names its own profile and the stage of it that asks more.
    #[test]
    fn a_place_is_picked_alone_where_its_picker_decides_by_itself() {
        assert!(matches!(
            preset("round_robin").place(MODEL, Phase::Single),
            Place::Alone(_)
        ));
        for name in [
            "random",
            "passthrough",
            "power_of_two",
            "least_load",
            "consistent_hashing",
            "prefix_hash",
            "manual",
            "cache_aware",
            "bucket",
        ] {
            let engine = preset(name);
            let Place::NotAlone { profile, stage } = engine.place(MODEL, Phase::Single) else {
                panic!("{name} is picked alone");
            };
            assert_eq!(profile, name);
            assert!(!stage.to_string().is_empty(), "{name}: {stage}");
        }
        assert_eq!(
            preset("random").place(MODEL, Phase::Single),
            Place::NotAlone {
                profile: "random",
                stage: InTheWay::Picker("random"),
            }
        );
        assert!(matches!(
            preset("least_load").place(MODEL, Phase::Single),
            Place::NotAlone {
                stage: InTheWay::Scorer(_),
                ..
            }
        ));
    }

    /// A side of a pair holds one test beyond a worker's health: the prefill
    /// side its window's slot, the decode side its room. A gateway's pass
    /// holds each at its own side's place and at no other. So each side's
    /// profile is picked alone at its own place, and the same filter stands
    /// in the way at every other place.
    #[test]
    fn a_pairs_side_is_picked_alone_at_its_own_place_only() {
        let mut document =
            PolicySpec::pair_presets("round_robin", "round_robin").expect("built-in presets");
        document.generation = 1;
        let engine = engine_of(&document);
        assert!(matches!(
            engine.place(MODEL, Phase::Prefill),
            Place::Alone(_)
        ));
        assert!(matches!(
            engine.place(MODEL, Phase::Decode),
            Place::Alone(_)
        ));
        // The pair plan has no side for a single request, and none for an
        // encode one: a binding names the model, so the place is not
        // unbound.
        assert_eq!(engine.place(MODEL, Phase::Single), Place::NoSide);
        assert_eq!(engine.place(MODEL, Phase::Encode), Place::NoSide);

        for (filter, own) in [
            ("has-capacity", Phase::Prefill),
            ("has-room", Phase::Decode),
        ] {
            let engine = round_robin_with(filter);
            for phase in [Phase::Single, Phase::Prefill, Phase::Decode, Phase::Encode] {
                let place = engine.place(MODEL, phase);
                if phase == own {
                    assert!(matches!(place, Place::Alone(_)), "{filter} at {phase:?}");
                } else {
                    let stage = InTheWay::Filter(filter);
                    let profile = "round_robin";
                    assert_eq!(place, Place::NotAlone { profile, stage }, "{phase:?}");
                }
            }
        }
    }

    /// Every request of a document is decided whole where each of its
    /// places is picked alone or its stages read nothing the host does not
    /// feed. Round robin asks nothing of a host. Random and passthrough
    /// read a worker's record and no more. Least load reads the workers'
    /// reports, and the default policy who holds the prompt beside them.
    /// A host that feeds less than a profile declares, a profile that
    /// reads a hash ring, one under a sessions rule and a pair's plan are
    /// not decided so.
    #[test]
    fn a_document_is_decided_whole_where_each_place_is_picked_alone_or_fed() {
        let whole = |engine: &PolicyEngine, fed: Needs| engine.spec().engine().single_and_fed(fed);
        let nothing = Needs::default();
        for name in ["round_robin", "random", "passthrough"] {
            assert!(whole(&preset(name), nothing), "{name}");
        }
        for name in ["power_of_two", "least_load", "bucket"] {
            let engine = preset(name);
            let declared = engine.spec().engine().needs_for(MODEL, Phase::Single);
            assert!(!declared.affinity && !declared.ring, "{name}");
            assert!(whole(&engine, declared), "{name}");
        }
        assert!(
            !whole(&preset("least_load"), nothing),
            "its scorer reads reports"
        );
        let default = preset("cache_aware");
        let declared = default.spec().engine().needs_for(MODEL, Phase::Single);
        assert!(declared.affinity && declared.loads && declared.sent_prefixes);
        assert!(whole(&default, declared));
        assert!(!whole(&default, nothing));
        let unfed = Needs {
            loads: false,
            ..declared
        };
        assert!(!whole(&default, unfed), "its load stages read reports");
        // A ring is no part of what the default policy's host feeds, and a
        // sessions rule is decided by no whole pass here yet.
        for name in ["consistent_hashing", "prefix_hash", "manual"] {
            assert!(!whole(&preset(name), declared), "{name}");
        }
        let mut pair =
            PolicySpec::pair_presets("round_robin", "round_robin").expect("built-in presets");
        pair.generation = 1;
        assert!(!whole(&engine_of(&pair), declared));
    }

    /// A pick tests a worker's health whole, as the pass does. A profile
    /// that leaves one of the two health tests out would gain it by the
    /// pick, so it is not picked alone, and the answer names the test.
    #[test]
    fn a_profile_without_a_health_test_is_not_picked_alone() {
        for (at, left_out) in [(0, "healthy"), (1, "not-overloaded")] {
            let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
            let profile = document
                .profiles
                .get_mut("round_robin")
                .expect("the preset's profile");
            assert_eq!(profile.filter[at].name(), left_out);
            profile.filter.remove(at);
            let stage = InTheWay::NoFilter(left_out);
            let profile = "round_robin";
            assert_eq!(
                engine_of(&document).place(MODEL, Phase::Single),
                Place::NotAlone { profile, stage }
            );
        }
    }

    /// With every worker ready a pick is the pool's own turn: one strict
    /// rotation at every pool size, from the position the pool drew.
    #[test]
    fn a_pick_turns_the_pools_own_rotation() {
        for workers in [1, 2, 3, 8, 16, 64] {
            let engine = preset("round_robin");
            let profile = alone(&engine);
            let members = engine.pool_members(&pool(workers));
            let first = engine
                .pick(profile, &members, RoutingState::eligible)
                .expect("a worker");
            for step in 1..2 * workers {
                let at = engine.pick(profile, &members, RoutingState::eligible);
                assert_eq!(at, Some((first + step) % workers), "{workers} workers");
            }
        }
    }

    /// The rotation is the pool's own word: two pools of one engine turn a
    /// rotation each, and a turn of one moves no turn of the other.
    #[test]
    fn each_pool_turns_its_own_word() {
        let engine = preset("round_robin");
        let profile = alone(&engine);
        let (one, other) = (pool(5), pool(5));
        let (one, other) = (engine.pool_members(&one), engine.pool_members(&other));
        let first = engine
            .pick(profile, &one, RoutingState::eligible)
            .expect("a worker");
        let apart = engine
            .pick(profile, &other, RoutingState::eligible)
            .expect("a worker");
        for step in 1..=10 {
            assert_eq!(
                engine.pick(profile, &one, RoutingState::eligible),
                Some((first + step) % 5)
            );
            assert_eq!(
                engine.pick(profile, &other, RoutingState::eligible),
                Some((apart + step) % 5)
            );
        }
    }

    /// Workers that are not ready take no request, and the ready ones take
    /// exact shares. In a small pool a turn on such a worker goes to the
    /// pass at once, and the ready workers' own rotation places the
    /// request. In a pool that earns turns the pool's next turn does. Past
    /// a run of such workers longer than those turns both rotations place
    /// requests, and both are exact.
    #[test]
    fn a_pick_keeps_exact_shares_past_workers_not_ready() {
        let engine = preset("round_robin");
        let profile = alone(&engine);
        let shares = |members: &PoolMembers, picks: usize| {
            let mut times = vec![0usize; members.workers().len()];
            for _ in 0..picks {
                times[engine
                    .pick(profile, members, RoutingState::eligible)
                    .expect("a worker")] += 1;
            }
            times
        };
        let leave = |list: &[Arc<dyn Worker>], ready: &dyn Fn(usize) -> bool| {
            for (at, worker) in list.iter().enumerate() {
                if !ready(at) {
                    worker.set_status(WorkerStatus::NotReady);
                }
            }
        };

        // Eight workers, one not ready: seven turns of eight place by the
        // pool's turn and one by the seven ready workers' own rotation.
        let list = pool(8);
        leave(&list, &|at| at != 5);
        let members = engine.pool_members(&list);
        let times = shares(&members, 8 * 7 * 3);
        for (at, &times) in times.iter().enumerate() {
            assert_eq!(times, if at == 5 { 0 } else { 24 }, "worker {at} of 8");
        }

        // Sixty-four workers, one in four not ready, none beside another:
        // the pool's next turn places each request, and no pass runs.
        let list = pool(64);
        leave(&list, &|at| at % 4 != 1);
        let members = engine.pool_members(&list);
        let times = shares(&members, 48 * 5);
        for (at, &times) in times.iter().enumerate() {
            assert_eq!(times, if at % 4 == 1 { 0 } else { 5 }, "worker {at} of 64");
        }

        // Sixty-four workers, sixteen in a row not ready, three turns a
        // pick. From the pool's turn on the first of the run, each turn of
        // the pool places 53 requests: 48 by the pool's turn and five by
        // the ready workers' own rotation, which comes round once in 48
        // such picks. Over 48 turns of the pool each ready worker takes 53.
        let list = pool(64);
        leave(&list, &|at| !(20..36).contains(&at));
        let members = engine.pool_members(&list);
        members.rotation.0.store(20, Ordering::Relaxed);
        let times = shares(&members, 48 * 53);
        for (at, &times) in times.iter().enumerate() {
            let expected = if (20..36).contains(&at) { 0 } else { 53 };
            assert_eq!(times, expected, "worker {at} of 64, a run not ready");
        }
    }

    /// A worker over its overload thresholds takes a request only where the
    /// caller lifted that veto, and a worker that is not ready, doubted or
    /// behind an open breaker under neither word. A pool no worker of which
    /// passes names nobody: its caller decides such a pool.
    #[test]
    fn a_pick_tests_a_worker_as_the_worker_tests_itself() {
        let engine = preset("round_robin");
        let profile = alone(&engine);
        let list = pool(4);
        list[0].set_status(WorkerStatus::NotReady);
        list[1].set_overloaded(true);
        list[2].set_stall(Some(StallReason::Wedged));
        let members = engine.pool_members(&list);
        for _ in 0..8 {
            assert_eq!(
                engine.pick(profile, &members, RoutingState::eligible),
                Some(3)
            );
        }
        let lifted: BTreeSet<usize> = (0..8)
            .filter_map(|_| {
                engine.pick(profile, &members, RoutingState::eligible_ignoring_overload)
            })
            .collect();
        assert_eq!(lifted, BTreeSet::from([1, 3]));

        list[3].set_overloaded(true);
        assert_eq!(engine.pick(profile, &members, RoutingState::eligible), None);
        assert_eq!(
            engine.pick_among(profile, &list, RoutingState::eligible),
            None
        );
        assert!(engine
            .pick(profile, &members, RoutingState::eligible_ignoring_overload)
            .is_some());
        // A pool that holds no worker.
        let empty = engine.pool_members(&pool(0));
        assert_eq!(engine.pick(profile, &empty, RoutingState::eligible), None);
        assert_eq!(
            engine.pick_among(profile, &[], RoutingState::eligible),
            None
        );
    }

    /// A list the engine keeps nothing of is passed over once, and the
    /// workers that pass take turns by their set's own rotation: exact
    /// shares, named by their place in the list handed.
    #[test]
    fn a_pick_among_a_list_turns_the_rotation_of_the_workers_that_pass() {
        let engine = preset("round_robin");
        let profile = alone(&engine);
        let list = pool(9);
        // A part of the pool, as a caller narrows it: every third worker
        // left out, and one of the others not ready.
        let part: Vec<Arc<dyn Worker>> = (0..9)
            .filter(|at| at % 3 != 0)
            .map(|at| Arc::clone(&list[at]))
            .collect();
        part[2].set_status(WorkerStatus::NotReady);
        let mut times = [0usize; 6];
        for _ in 0..5 * 7 {
            times[engine
                .pick_among(profile, &part, RoutingState::eligible)
                .expect("a worker")] += 1;
        }
        assert_eq!(times, [7, 7, 0, 7, 7, 7]);
    }
}
