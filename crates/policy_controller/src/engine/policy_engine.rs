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
//! - **Once a membership.** The gateway publishes each list with an empty
//!   cell beside it ([`ListCell`](super::list_cell::ListCell)) and lends
//!   the two together ([`LentList`]). The engine makes its value of the
//!   list into the cell at the list's first ask: the list's handle, the
//!   pool's own word and the list's records. The value stands for as long
//!   as that list does. Nothing of a worker's live state is copied.
//! - **Once a place.** The gateway builds one policy for each place it
//!   places for: a model and a phase. [`PolicyEngine::place_for`] finds
//!   the place's profile by the document's own bindings, there and at no
//!   request.
//! - **Once a request.** [`PolicyEngine::pick`] asks for one pick over the
//!   pool, under a profile the engine picks by its picker alone
//!   ([`Place::Alone`]). The pick is the picker's own rule: a rotation
//!   turns the pool's own word and reads the worker the turn lands on, a
//!   scan reads the pool's workers in order up to the first that passes.
//!   Each read is the pass's own test of a worker: no row, no table, no
//!   request fact, no lock, and nothing allocated. A list nothing is kept
//!   of (a part of a pool, as the pass narrowed it) is passed over once
//!   ([`PolicyEngine::pick_among`]).
//!
//! **A whole decision.** A gateway whose routers ask the engine for the
//! whole decision asks through [`DecisionEngine`], as it asks a controller's
//! engine: the binding, its rules and the profile, over the list its
//! request's maker hands ([`RequestMaker::workers`]), with a lease the
//! request's guard holds to its end. The engine files no lease and keeps no
//! note of a request: the guard lends back what a later word needs
//! ([`HeldDecision`]). A single request is decided in [`single`], and a pair
//! in [`pair`], with its decode asked again behind its prefill. Where a
//! single's binding names a shadow, that profile decides after the whole
//! decision and its decision is counted ([`with_shadow`]). An encode
//! request is decided with its pair in [`encode`], over three lists
//! ([`PolicyEngine::decide_encode_at`]).
//!
//! The engine keeps no worker list, no health and no count of requests in
//! flight. What it keeps is its stages' own: a rotation, and for the stages
//! that read who holds a prompt, the record of the prefixes it sent
//! ([`overlap`]).

mod budget;
mod document;
pub(super) mod encode;
mod keys;
mod labels;
pub(super) mod membership;
pub mod overlap;
pub(super) mod pair;
pub(super) mod place;
mod signals;
mod single;
mod stands_alone;
pub(super) mod start;
mod task;
mod told_loads;
mod with_shadow;

use std::{
    cell::RefCell,
    hash::Hasher,
    panic::{catch_unwind, AssertUnwindSafe},
    ptr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, OnceLock,
    },
    time::Instant,
};

use rustc_hash::FxHasher;
use smg_worker::{identity::basic::Worker, load::state::LoadSnapshot, runtime::RoutingState};

pub(crate) use self::document::Loaded;
use self::{
    document::InForce,
    encode::EncodeLists,
    overlap::{Affinity, PoolAffinity},
};
use super::{
    decision::ShadowKind,
    decision_engine::{
        AlonePick, DecideOutcome, Decided, DecisionEngine, DecisionRefused, HeldDecision, Request,
        RequestMaker, WorkerLists,
    },
    framework::{release, AloneProfile, NotAlone, Picker, Place, PoolWord, ShadowRows},
    lease_events,
    list_cell::{HandedLists, LentList},
    prompt_hashes::{HashSeeds, PromptHashes},
    records::{DecisionClock, PoolRecords},
    shadow::{Counted, Shadowed},
    sticky_sessions::{shards_for_this_host, SessionTable},
    view::{FixedFacts, Phase, WireModes, WorkerId},
};
use crate::{
    document::spec::AppliedSpec,
    store::GatewayId,
    tracking::lease::{EventKind, EventSet, Membership, Said, Side},
};

/// How many documents the engines of this process took so far.
static APPLIED: AtomicU64 = AtomicU64::new(0);

/// The number of the next document an engine of this process takes: each
/// is drawn here, so no two applied documents share one.
fn next_applied() -> u64 {
    APPLIED.fetch_add(1, Ordering::Relaxed) + 1
}

/// The engine of one policy document, for a gateway that decides in its own
/// process.
///
/// Built once, at the gateway's start, from the document in force. It holds
/// the document's stages and their state, and no worker.
#[derive(Debug)]
pub struct PolicyEngine {
    /// What the engine holds of its document ([`InForce`]): the document,
    /// its number, what the engine reads of the document once, and the
    /// shadow decisions counted under it.
    document: InForce,
    /// The pools made so far: each draws the place its turns start at from
    /// its number.
    epochs: AtomicU64,
    /// The one clock of the engine's decisions: a decision's instant and a
    /// load report's sample instant are counted from its origin.
    clock: DecisionClock,
    /// How many whole decisions panicked, and how many shadow profiles did
    /// after a live decision that stood ([`DecisionEngine::faults`]).
    faults: AtomicU64,
    /// How many words of a lease and changes of membership panicked in a
    /// stage that heard them.
    unheard: AtomicU64,
    /// How many steps of the engine's own task panicked: a run of the
    /// upkeep, a step of the registry's follow, or a telling of load
    /// reports ([`Self::upkeep_faults`]).
    upkeep_faults: AtomicU64,
    /// The id of the last lease issued. A pick that is issued one writes
    /// it: one atomic step, on cache lines of its own.
    leases: OwnWord,
    /// The session table, one for the engine's life: the pins of every key
    /// a sessions rule placed. The document's engine reads and writes it
    /// through a pointer ([`AppliedSpec::share_sessions`]), so a pin
    /// outlives the document it was made under. It has the host's shards
    /// whatever the first document rules: no table ever takes its place.
    sessions: Arc<SessionTable>,
    /// What the affinity stages read across decisions: the host's indexes
    /// of cache events, once it handed them, and the engine's record of the
    /// prefixes it sent, one a model ([`Self::affinity`]).
    affinity: Affinity,
    /// The host's load channel, once it handed it ([`Self::set_loads`]),
    /// and the newest snapshot the engine's own task keeps at hand
    /// ([`Self::run`]).
    loads: task::Loads,
    /// The host's worker registry, once it handed it
    /// ([`Self::set_registry`]): the engine's own task follows its events.
    members: membership::Handed,
    /// The host's worker lists, once it handed them ([`Self::set_lists`]):
    /// the engine's own task fills their cells ahead of their requests.
    pub(super) lists: HandedLists,
}

/// The document's word of a binding's shadow: the binding's position, and
/// how its shadow profile reads the live decision's affinity rows.
type ShadowWord = (u32, ShadowRows);

/// One pool of one model as the engine keeps it: the gateway's shared list
/// of worker objects and the pool's own word.
///
/// Made where a list's cell is filled ([`PolicyEngine::members_of`]), from
/// a list the gateway's registry published, and kept in that cell beside
/// the list. It holds the list itself, not a copy: a worker's live facts
/// are read on the list's objects.
#[derive(Debug)]
pub(crate) struct PoolMembers {
    workers: Arc<[Arc<dyn Worker>]>,
    /// The pool's own word, for the picker that places over the pool by
    /// itself ([`PolicyEngine::pick`]): a rotation counts its turns here.
    /// A single request's whole decision over the list lends it to its
    /// draw ([`PoolFacts::word`](super::view::PoolFacts::word)), so the
    /// pick and that draw turn the one rotation.
    rotation: OwnWord,
    /// What a whole decision reads of the list beside its workers' live
    /// state ([`Self::records`]): made with the value where a list's cell
    /// is filled ([`Loaded::pool_members_with_fixed_facts`]). A value
    /// a case makes without them ([`PolicyEngine::pool_members`]) makes
    /// them at its first whole decision.
    records: OnceLock<PoolRecords>,
    /// The model and the phase the list was first asked for under a
    /// document a stage of which reads a named signal, and whether a
    /// decision of that model at that phase reads one
    /// ([`Self::reads_signal`]): such a decision reads the list's signal
    /// cells beside its records. A single request's list is asked for the
    /// single phase, a pair's prefills for the prefill phase and its
    /// decodes for the decode phase.
    signal: OnceLock<(Box<str>, Phase, bool)>,
    /// The wires the list's workers are dialed on ([`Self::wires`]): read
    /// of their registrations at the first whole decision over the list.
    wires: OnceLock<WireModes>,
    /// The model of the list's first single request and the profile the
    /// engine picks alone for it, where the document holds one
    /// ([`Self::alone`]).
    alone: OnceLock<(Box<str>, Option<AloneProfile>)>,
    /// The model of the list's first whole decision under a document that
    /// names a shadow, and the document's word of that model's shadow
    /// ([`Self::shadow`]).
    shadow: OnceLock<(Box<str>, Option<ShadowWord>)>,
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
    /// The workers of the pool: the list the value was made from.
    #[must_use]
    pub fn workers(&self) -> &Arc<[Arc<dyn Worker>]> {
        &self.workers
    }

    /// The list's records under the document `spec`, on `clock`: the half
    /// of each worker's record that no request changes, the pool's engines
    /// and its number ([`PoolRecords`]). Made with the value where a cell's
    /// fill makes it
    /// ([`Loaded::pool_members_with_fixed_facts`]), and else at the
    /// first call; kept for as long as the list stands, and every later
    /// call is one read. No live value of a worker is kept in them.
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
    /// sessions rule and names no shadow: `None` where the place's decision
    /// asks more than one pick. Found once a list, for the model of its
    /// first request: a gateway's list is one model's. Another model's
    /// request is answered by the document at its call.
    fn alone(&self, spec: &AppliedSpec, model: &str) -> Option<AloneProfile> {
        let find = || match spec.engine().place_for(model, Phase::Single) {
            Place::Alone(profile) => Some(profile),
            _ => None,
        };
        let (named, alone) = self.alone.get_or_init(|| (model.into(), find()));
        if **named == *model {
            *alone
        } else {
            find()
        }
    }

    /// What the document says of the shadow of a request of `model` that
    /// stands alone at `phase` ([`ShadowWord`]): `None` under a binding
    /// that names no shadow. A single request's word is found once a list,
    /// as [`Self::alone`] is: a request of the list's model pays no second
    /// read of its binding. A binding matches by phase, so a request of
    /// another phase asks the document each time.
    fn shadow(&self, spec: &AppliedSpec, model: &str, phase: Phase) -> Option<ShadowWord> {
        let find = || spec.engine().shadow_for(model, phase);
        if phase != Phase::Single {
            return find();
        }
        let (named, shadow) = self.shadow.get_or_init(|| (model.into(), find()));
        if **named == *model {
            *shadow
        } else {
            find()
        }
    }
}

impl PolicyEngine {
    /// The engine of `spec`, the document in force, handed nothing of its
    /// caller's data. A program has its engine by [`Self::start`], which
    /// builds it here and makes every hand-over.
    #[must_use]
    pub fn new(mut spec: AppliedSpec) -> Self {
        // The table takes the document's two times where the document's
        // engine takes the pointer.
        let sessions = Arc::new(SessionTable::new(0).with_shards(shards_for_this_host()));
        spec.share_sessions(Arc::clone(&sessions));
        Self {
            document: InForce::of(spec),
            epochs: AtomicU64::new(0),
            clock: DecisionClock::new(),
            faults: AtomicU64::new(0),
            unheard: AtomicU64::new(0),
            upkeep_faults: AtomicU64::new(0),
            leases: OwnWord(AtomicU64::new(0)),
            sessions,
            affinity: Affinity::default(),
            loads: task::Loads::default(),
            members: membership::Handed::default(),
            lists: HandedLists::default(),
        }
    }

    /// [`Self::new`] on the clock of a host that reads one of its own and
    /// builds an engine for each of its documents: the engine's clock
    /// starts at `origin`, the host's. An instant then reads alike under
    /// every document and on the host's clock, so what one document's
    /// stages stamped and the next takes over, a profile's count of sends
    /// ([`Engine::succeed`](crate::engine::framework::Engine::succeed)),
    /// is read on the clock it was written on.
    #[must_use]
    pub(crate) fn with_origin(spec: AppliedSpec, origin: Instant) -> Self {
        Self {
            clock: DecisionClock::from_origin(origin),
            ..Self::new(spec)
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

    /// The worker at `address` was removed from its host's registry, and so
    /// from every pool: the records of sent prefixes forget it, so no later
    /// request is named a worker that is gone, and the document's stages
    /// are told ([`Membership::Removed`]). Both name the worker by the
    /// fingerprint of its address: the id every decision knew it by, where
    /// its program states none ([`FixedFacts::id`]). A stage that keeps a
    /// range, a count or a credit of the worker drops it. A dispatch said
    /// behind the leave enters its rank in a count of sends again. The
    /// session table is not told: a leave takes no pin away, so a
    /// session's pin stands on a worker that is gone until the session's
    /// idle time or two other workers end it, and is a routable pin again
    /// once the worker is back.
    ///
    /// A host says it at a removal, where it tells its classic policies,
    /// and not where a worker leaves one model's pool or its object is
    /// replaced at its address: both leave the worker standing. A host
    /// that hands its registry says it nowhere ([`Self::set_registry`]),
    /// and a program that states ids of its own hands it: this call names
    /// a worker by its address alone.
    ///
    /// In the records a worker none of them holds costs one lookup a model.
    /// Every stage is told whatever the records hold, and one that keeps a
    /// state of workers takes its lock or copies its table for it: the call
    /// belongs at a change of membership, off a request's path.
    ///
    /// A worker and one registered at its address behind its removal share
    /// one id. A host says this first and [`Self::joined`] behind it: said
    /// the other way round, the worker that serves is left told gone.
    pub fn forget(&self, address: &str) {
        self.left(WorkerId::from_address(address));
    }

    /// The worker at `address` joined: the document's stages are told
    /// ([`Membership::Added`]) under the fingerprint of its address, the id
    /// every decision names the worker by where its program states none
    /// ([`FixedFacts::id`]). A stage that keeps a pruned worker out for a
    /// while lists a worker back at the address it left at once. A host
    /// says it at a registration, where it tells its classic policies. A
    /// host that hands its registry says it nowhere
    /// ([`Self::set_registry`]).
    pub fn joined(&self, address: &str) {
        self.came(WorkerId::from_address(address));
    }

    /// A removal by the worker's id: what [`Self::forget`] says.
    fn left(&self, id: WorkerId) {
        self.affinity.forget(id);
        self.told(&Membership::Removed(id));
    }

    /// A join by the worker's id: what [`Self::joined`] says.
    fn came(&self, id: WorkerId) {
        self.told(&Membership::Added(id));
    }

    /// Tells the document's stages one change of membership, on the calling
    /// thread, and not the session table. A stage that panics at it is held
    /// and counted ([`Self::word_caught`]): the registration or the removal
    /// goes on.
    fn told(&self, change: &Membership<'_>) {
        let engine = self.document.spec.engine();
        self.word_caught(|| engine.membership_stages(change));
    }

    /// One word to the document's stages, `tell`, under a catch of its own:
    /// what the word answers, and `None` where a stage panicked at it. The
    /// panic is held here and counted ([`Self::unheard`]). The stages behind
    /// the one that panicked do not hear the word, and what the word is of
    /// goes on: a request, a registration or a removal.
    #[inline]
    fn word_caught<R>(&self, tell: impl FnOnce() -> R) -> Option<R> {
        match catch_unwind(AssertUnwindSafe(tell)) {
            Ok(answered) => Some(answered),
            Err(payload) => {
                release(payload);
                self.unheard.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// The clock of the engine's decisions: a list's records are made on
    /// it.
    #[must_use]
    pub fn clock(&self) -> &DecisionClock {
        &self.clock
    }

    /// How many words of a lease (its issue, its dispatch, its end, its
    /// release) and changes of membership panicked in a stage that heard
    /// them. The word stands as said: the request, the registration or the
    /// removal it is of goes on, and the stages after the one that panicked
    /// did not hear it.
    ///
    /// A lease's issue is such a word at a first decision and at a decode's
    /// own call alike: the call answers its pick and the lease all the same
    /// ([`DecisionEngine::decide_under`],
    /// [`DecisionEngine::decide_decode_under`]).
    #[must_use]
    pub fn unheard(&self) -> u64 {
        self.unheard.load(Ordering::Relaxed)
    }

    /// The document the engine decides under.
    #[must_use]
    pub fn spec(&self) -> &AppliedSpec {
        &self.document.spec
    }

    /// The number of the document the engine decides under, which no other
    /// applied document of the process has. A list's value is made under it
    /// and read under it alone ([`Self::members_of`]).
    pub(super) fn applied(&self) -> u64 {
        self.document.applied
    }

    /// How many values of a list the engine made so far, for a case that
    /// counts the fills of a cell.
    #[cfg(test)]
    pub(super) fn values_made(&self) -> u64 {
        self.epochs.load(Ordering::Relaxed)
    }

    /// The engine's value of one pool with no records made: `pool` is the
    /// shared list the gateway's registry published for it.
    ///
    /// What [`Loaded::pool_members_with_fixed_facts`] builds on, and what a
    /// case holds for a decision's inner functions. It reads no worker, so
    /// its cost does not grow with the list. Under this value every worker
    /// states nothing: none takes images, none meets a demand for a runtime
    /// or for media by reference, and each holds a request of any size.
    #[must_use]
    pub(crate) fn pool_members(&self, pool: &Arc<[Arc<dyn Worker>]>) -> PoolMembers {
        let epoch = self.epochs.fetch_add(1, Ordering::Relaxed) + 1;
        PoolMembers {
            workers: Arc::clone(pool),
            // A pool starts its turns at a drawn worker, so a fleet whose
            // membership keeps changing does not lean on its first one.
            rotation: OwnWord(AtomicU64::new(mixed(epoch) >> 32)),
            records: OnceLock::new(),
            signal: OnceLock::new(),
            wires: OnceLock::new(),
            alone: OnceLock::new(),
            shadow: OnceLock::new(),
            affinity: PoolAffinity::default(),
        }
    }
}

impl Loaded<'_, '_> {
    /// [`PolicyEngine::pool_members`] with the list's records made of what its
    /// program states of its workers beside their objects ([`FixedFacts`]):
    /// the id it names a worker by where it keeps ids of its own, whether a
    /// worker takes images, or what it states for a request's hard demands.
    /// The fill of a list's cell calls it ([`Self::members_of`]): inside
    /// the first decision that asks the cell, or by the engine's own task
    /// ahead of it ([`PolicyEngine::set_lists`]). `facts_of` is asked once a
    /// worker, in the list's order, here: every fact a program says of a
    /// worker is a field of that one answer. A request that states a demand
    /// is then held to the workers of the list that meet it
    /// ([`PoolRecords::meeting`]).
    ///
    /// The list's records are made of the answers at this call, and not at
    /// the list's first whole decision: a list's records are also made
    /// with no request at hand, by a pick that is noted. So the cost grows
    /// with the list, once for each value made of it.
    #[must_use]
    pub(crate) fn pool_members_with_fixed_facts(
        self,
        pool: &Arc<[Arc<dyn Worker>]>,
        facts_of: impl FnMut(&Arc<dyn Worker>) -> FixedFacts,
    ) -> PoolMembers {
        let records =
            PoolRecords::with_fixed_facts(pool, &self.document.spec, &self.clock, facts_of);
        PoolMembers {
            records: OnceLock::from(records),
            ..self.pool_members(pool)
        }
    }
}

impl PolicyEngine {
    /// What the document holds for one place the gateway builds a policy
    /// for, `model` at `phase` ([`Engine::place_for`]). The first binding that
    /// matches names the place's profile. The answer is that profile where
    /// one pick of its picker is its whole decision ([`Place::Alone`]),
    /// what asks more where it is not, a stage, a sessions rule or a shadow
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
    pub fn place_for(&self, model: &str, phase: Phase) -> Place<'_> {
        self.document.spec.engine().place_for(model, phase)
    }

    /// The first place a binding of the document names that the engine
    /// does not decide by one pick ([`Engine::not_alone`]). `None`
    /// where it decides every place of the document so. A gateway that
    /// starts under a document its operator supplied refuses the start by
    /// what this names: nothing else would decide that place.
    ///
    /// [`Engine::not_alone`]: super::framework::Engine::not_alone
    #[must_use]
    pub fn not_alone(&self) -> Option<NotAlone<'_>> {
        self.document.spec.engine().not_alone()
    }

    /// One pick under `profile` over `list`, a pool its caller lends with
    /// its cell: the place in the pool's list of the worker picked. `None`
    /// where no worker of the pool passes. The engine's value of the pool
    /// is asked of the cell: a read once the cell holds it, and the value's
    /// making at the list's first ask under the engine's document, whose
    /// cost grows with the list.
    ///
    /// The entry for a caller that holds every other rule of a placement
    /// itself, as a gateway's classic placement pass does: which workers a
    /// request may take, what a retry skips, and what becomes of a pool no
    /// worker of which passes. The picker's rule runs over the pool, whose
    /// own word is kept in that value, and each worker the rule asks is read
    /// once and tested by `admits`, the caller's own test of a worker's
    /// read: the one its pass asks of every candidate, so the rule of a
    /// lifted veto has one home. Behind the list's first ask no row is
    /// read, no table looked up, no lock taken and nothing allocated. Where
    /// a rotation's turn lands on a refused worker the pool is passed over
    /// once, as [`Self::pick_among`] passes over a list. A rule that asked
    /// every worker and kept none is not followed by that pass.
    #[must_use]
    pub fn pick(
        &self,
        profile: AloneProfile,
        list: LentList<'_>,
        admits: impl Fn(RoutingState) -> bool + Copy,
    ) -> Option<usize> {
        let own = self.loaded();
        own.pick_over(profile, own.members_of(list), admits)
    }
}

impl Loaded<'_, '_> {
    /// [`PolicyEngine::pick`] over `members`, the engine's value of the pool, which
    /// its caller asked of the pool's cell.
    #[inline]
    pub(super) fn pick_over(
        self,
        profile: AloneProfile,
        members: &PoolMembers,
        admits: impl Fn(RoutingState) -> bool + Copy,
    ) -> Option<usize> {
        let picker = self.picker_of(profile)?;
        pick_by(picker, &members.workers, &members.rotation.0, admits)
    }
}

impl PolicyEngine {
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
        among(self.loaded().picker_of(profile)?, workers, admits)
    }
}

impl<'d> Loaded<'_, 'd> {
    /// The picker of `profile`.
    #[inline]
    fn picker_of(self, profile: AloneProfile) -> Option<&'d dyn Picker> {
        let profiles = self.document.spec.engine().profiles();
        Some(profiles.get(usize::from(profile.0))?.picker.as_ref())
    }

    /// The request `maker` makes, as this engine decides it. A request that
    /// states no budget is made with the widest one: a stage reads the
    /// number as the time left ([`budget::shown`]).
    #[inline(always)]
    fn made<'w>(&'w self, maker: &'w impl RequestMaker) -> Request<'w> {
        let mut request = maker.make(&self.document.spec);
        request.facts.budget_micros = budget::shown(request.facts.budget_micros);
        request
    }
}

impl PolicyEngine {
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
        self.loaded().decide_at(maker, clock)
    }
}

impl Loaded<'_, '_> {
    /// [`PolicyEngine::decide_at`] under the document this value holds:
    /// for a caller that reads that document further behind the decision,
    /// as the call that places a request does.
    pub(super) fn decide_at(
        self,
        maker: &impl RequestMaker,
        clock: impl FnOnce() -> Instant,
    ) -> Result<Decided, DecisionRefused> {
        let mut whole = None;
        let alone = self.decide_into(maker, clock, &mut whole)?;
        match (alone, whole) {
            (Some(pick), _) => Ok(Decided::Alone(pick)),
            (None, Some((decided, later))) => Ok(Decided::Whole(decided, later)),
            // The pass writes its decision ahead of every answer but a
            // place.
            (None, None) => Err(DecisionRefused::Faulted),
        }
    }

    /// The decision of [`Self::decide_at`] for a caller that keeps the
    /// room of a whole decision itself, as the call that places a request
    /// does: the place it answers, or `None` with the decision written
    /// into `whole` beside whether the request stated the later decode.
    /// Its caller hands `whole` empty, and only a whole decision fills it.
    pub(super) fn decide_into(
        self,
        maker: &impl RequestMaker,
        clock: impl FnOnce() -> Instant,
        whole: &mut Option<(DecideOutcome, bool)>,
    ) -> Result<Option<AlonePick>, DecisionRefused> {
        self.caught(|| self.placed_or(maker, clock, whole))
    }
}

impl PolicyEngine {
    /// One request's decision and nothing its host hears of later: no lease
    /// is issued, no id is taken and no stage is told. For a host that
    /// files a pick's lease itself and tells the stages of the issue under
    /// that lease
    /// ([`Lifecycle::on_decision_issued`](crate::stages::Lifecycle::on_decision_issued)
    /// of the document's engine), as a controller's service does in its
    /// store.
    ///
    /// The request is made once and decided whole, at `now`, as
    /// [`Self::decide_at`] decides a request that is not plain
    /// ([`RequestMaker::plain`]): a request that stands alone
    /// ([`Loaded::stands_alone`]) over its host's one list, a pair at
    /// its prefill over its host's two lists ([`pair::decide`], the test
    /// of the request's wires in it), and a request that hands no such
    /// list over no worker. This entry never asks whether a request is
    /// plain, so it answers no place: a pick names the alternates its
    /// whole decision drew, for a host that hands a decision's targets to
    /// another process and cannot ask again cheaply.
    ///
    /// A request at its prefill that hands an encode request's three lists
    /// ([`WorkerLists::Encode`]) is decided as its pair, over the pair's
    /// two of the three.
    ///
    /// A binding's shadow decides behind the live decision, as under the
    /// leasing entry. What it said is answered beside the decision and
    /// counted nowhere here: the caller counts it once its own filing of
    /// the lease stands, and a shed's, which files none, all the same
    /// ([`Self::count_shadow`]). So a decision whose issue fails counts
    /// none under either entry. A pair's prefill answers no word: no
    /// shadow runs on a pair's side here yet.
    ///
    /// # Errors
    ///
    /// [`DecisionRefused::PairSide`] for a decode or an encode request
    /// that is a side of its plan's pair, or that hands more than one
    /// list. A decode behind its pair's prefill has an entry of its own
    /// ([`Self::decide_decode_without_lease`]).
    /// [`DecisionRefused::Faulted`] where the decision panicked.
    #[cfg_attr(
        not(any(test, feature = "controller")),
        expect(dead_code, reason = "a controller's deciding thread calls it")
    )]
    pub(crate) fn decide_without_lease(
        &self,
        maker: &impl RequestMaker,
        now: Instant,
    ) -> Result<(super::decision::Decision, Option<Shadowed>), DecisionRefused> {
        let own = self.loaded();
        self.caught(|| {
            let mut request = own.made(maker);
            let mut said = None;
            let decision = match (request.facts.phase, maker.workers()) {
                // A single, and a request of a phase its plan pairs no side
                // of: one profile over its host's one list.
                (phase, WorkerLists::Single(lent))
                    if own.stands_alone(request.facts.model, phase) =>
                {
                    // The engine's own value of the list, asked of its
                    // cell.
                    let members = own.members_of(lent);
                    Ok(own.decide_made(maker, &mut request, members, now, &mut said))
                }
                // A pair is decided at its prefill, over its host's two
                // lists, and so is a request that hands an encode
                // request's three.
                (
                    Phase::Prefill,
                    WorkerLists::Pair(lists) | WorkerLists::Encode(EncodeLists { pair: lists, .. }),
                ) => {
                    let read = self.loads_read(maker);
                    let loads = read.snapshot();
                    let sends_to = |target: &_| maker.sends_to(target);
                    Ok(pair::decide(
                        own,
                        &lists,
                        &mut request,
                        None,
                        loads,
                        now,
                        sends_to,
                    ))
                }
                (Phase::Single | Phase::Prefill, _) => Ok(single::over_none(own, &request)),
                // A request that stands alone and hands no list.
                (phase, WorkerLists::None) if own.stands_alone(request.facts.model, phase) => {
                    Ok(single::over_none(own, &request))
                }
                (Phase::Decode | Phase::Encode, _) => Err(DecisionRefused::PairSide),
            };
            decision.map(|decision| (decision, said))
        })
    }

    /// A pair's decode decided again behind the prefill of `held`, the
    /// target its host holds of the pair's first decision, and nothing its
    /// host hears of later ([`Self::decide_without_lease`]). The decision
    /// is the one the leasing entry makes for a guard's decode call
    /// ([`DecisionEngine::decide_decode_under`]). The host files the decode
    /// side's issue and tells the stages of it.
    ///
    /// A call that hands no pair's lists is decided over no worker.
    ///
    /// # Errors
    ///
    /// [`DecisionRefused::Faulted`] where the decision panicked.
    #[cfg_attr(
        not(any(test, feature = "controller")),
        expect(dead_code, reason = "a controller's deciding thread calls it")
    )]
    pub(crate) fn decide_decode_without_lease(
        &self,
        maker: &impl RequestMaker,
        held: &super::decision::Target,
        now: Instant,
    ) -> Result<super::decision::Decision, DecisionRefused> {
        let own = self.loaded();
        self.caught(|| {
            let mut request = own.made(maker);
            Ok(match maker.workers() {
                WorkerLists::Pair(lists) | WorkerLists::Encode(EncodeLists { pair: lists, .. }) => {
                    let read = self.loads_read(maker);
                    let loads = read.snapshot();
                    // The decode asked again is asked which decode it is sent to.
                    let sends_to = |target: &_| maker.sends_to(target);
                    pair::decide(own, &lists, &mut request, Some(held), loads, now, sends_to)
                }
                _ => single::over_none(own, &request),
            })
        })
    }
}

impl Loaded<'_, '_> {
    /// The whole decision of a made request that stands alone
    /// ([`Self::stands_alone`]) over its host's list, for every entry that
    /// decides one. It reads the snapshot the request
    /// gives where it gives one, and the newest the engine's own task
    /// keeps at hand otherwise ([`PolicyEngine::loads_read`]).
    ///
    /// Under a binding that names a shadow, the shadow profile decides
    /// behind the live one ([`with_shadow::decide`]). What it said is
    /// written into `said`, for the caller to count behind its issue. A
    /// document that names no shadow pays one test: its path holds nothing
    /// else of a shadow.
    ///
    /// A request whose profile reads a named signal is decided with its
    /// list's signal cells in its facts, made of that one snapshot
    /// ([`signals::decide`]). A document none of whose stages reads one
    /// pays one test.
    #[inline]
    fn decide_made<'a>(
        self,
        maker: &impl RequestMaker,
        request: &mut Request<'a>,
        members: &'a PoolMembers,
        now: Instant,
        said: &mut Option<Shadowed>,
    ) -> super::decision::Decision {
        let read = self.loads_read(maker);
        let loads = read.snapshot();
        // Under a document a stage of which reads a named signal the
        // list's word says whether this request's profile reads one, at
        // the request's own phase.
        let spec = &self.document.spec;
        let (model, phase) = (request.facts.model, request.facts.phase);
        if self.document.signals && members.reads_signal(spec, model, phase) {
            return signals::decide(self, request, members, maker, loads, now, said);
        }
        self.decide_with_loads(request, members, maker, loads, now, said)
    }

    /// [`Self::decide_made`] behind its choice of the snapshot: the
    /// decision of a made request that stands alone over `loads`, by the
    /// live profile of its phase's binding and, where the binding names
    /// one, its shadow behind
    /// it. A request that is lent its list's signal cells is decided here
    /// too, as the copy that carries them ([`signals::decide`]).
    ///
    /// `maker` is the request's own, handed on for its entry in a token
    /// tree its host lends. A copy that carries signal cells is decided
    /// inside that tree like the request it was made of, and hands its
    /// maker as a trait object ([`signals::decide`]).
    #[inline]
    pub(super) fn decide_with_loads(
        self,
        request: &mut Request<'_>,
        members: &PoolMembers,
        maker: &(impl RequestMaker + ?Sized),
        loads: Option<&Arc<LoadSnapshot>>,
        now: Instant,
        said: &mut Option<Shadowed>,
    ) -> super::decision::Decision {
        // Under a document that names a shadow the list's word says
        // whether this binding does.
        if self.document.any_shadow {
            let (model, phase) = (request.facts.model, request.facts.phase);
            if let Some(binding) = members.shadow(&self.document.spec, model, phase) {
                let (decision, shadow) =
                    with_shadow::decide(self, request, members, maker, loads, now, binding);
                *said = shadow;
                return decision;
            }
        }
        single::decide(
            self,
            request,
            members,
            maker,
            loads,
            now,
            single::WithoutShadow,
        )
    }

    /// The decision's pass: the place it answers, or its whole decision
    /// written into `whole` with whether the request made stated the later
    /// decode.
    fn placed_or(
        self,
        maker: &impl RequestMaker,
        clock: impl FnOnce() -> Instant,
        whole: &mut Option<(DecideOutcome, bool)>,
    ) -> Result<Option<AlonePick>, DecisionRefused> {
        // A plain single request of a place one pick decides is decided
        // ahead of its making: the pick reads no fact of it. A request of
        // another phase asks no cell here.
        let mut picked = None;
        // The cell the pick asked, with the value it gave.
        let mut asked = None;
        match (maker.plain(), maker.workers()) {
            (Some(plain), WorkerLists::Single(lent)) if plain.phase == Phase::Single => {
                // The engine's own value of the list, asked of its cell.
                let members = self.members_of(lent);
                asked = Some((lent.cell, members));
                if let Some(pick) = single::alone(self, maker.model(), plain, members) {
                    // No stage takes a word of the request: the place
                    // is the whole answer.
                    if !self.document.hears.intersects(EventSet::LEASE) {
                        return Ok(Some(pick));
                    }
                    picked = Some((pick, members, plain));
                }
            }
            _ => {}
        }
        let now = clock();
        if let Some((pick, members, plain)) = picked {
            // The lease states the text's two lengths: the one place a
            // plain request's text is passed over, for its characters.
            let text = (maker.prompt_chars(), maker.prompt_bytes());
            if let Some(decided) = single::noted(self, pick, members, plain, text, now) {
                *whole = Some((decided, false));
                return Ok(None);
            }
        }
        let mut request = self.made(maker);
        let later = request.constraints.decode_later;
        let decided = match (request.facts.phase, maker.workers()) {
            // A single, and a request of a phase its plan pairs no side of:
            // one profile over its host's one list.
            (phase, WorkerLists::Single(lent)) if self.stands_alone(request.facts.model, phase) => {
                // The engine's own value of the list the request was made
                // for: a maker may name its list while it makes the
                // request. The cell the pick asked is not asked again.
                let members = match asked {
                    Some((cell, members)) if ptr::eq(cell, lent.cell) => members,
                    _ => self.members_of(lent),
                };
                let mut said = None;
                let decision = self.decide_made(maker, &mut request, members, now, &mut said);
                // What a shadow said is counted behind the lease's issue.
                if let Some(said) = said {
                    let decided = with_shadow::issued(self, decision, &request, now, said);
                    *whole = Some((decided, later));
                    return Ok(None);
                }
                decision
            }
            // A pair is decided at its prefill, over its host's two
            // lists. An encode request's three hand its pair's two: a
            // request that states no item here is its pair's alone.
            (
                Phase::Prefill,
                WorkerLists::Pair(lists) | WorkerLists::Encode(EncodeLists { pair: lists, .. }),
            ) => {
                let read = self.loads_read(maker);
                let loads = read.snapshot();
                let sends_to = |target: &_| maker.sends_to(target);
                let decided = pair::decided(self, &lists, &mut request, None, loads, now, sends_to);
                *whole = Some((decided, later));
                return Ok(None);
            }
            // A request that hands no list of its own is decided over
            // no worker.
            (Phase::Single | Phase::Prefill, _) => single::over_none(self, &request),
            (phase, WorkerLists::None) if self.stands_alone(request.facts.model, phase) => {
                single::over_none(self, &request)
            }
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
/// lines of its own. No lock of the engine is waited for, only the one build
/// of a list's records, wires or one-pick profile: a stage's lock is its own.
///
/// A panic in a decision costs that request its decision and is counted
/// ([`DecisionEngine::faults`]). A shadow's panic costs its request nothing
/// and is counted with them. A panic in a stage that hears a word is held
/// here too and counted ([`PolicyEngine::unheard`]): the request goes on.
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
        let own = self.loaded();
        self.caught(|| {
            let mut request = own.made(maker);
            Ok(match maker.workers() {
                // An encode request's decode is asked again over its
                // pair's two lists.
                WorkerLists::Pair(lists) | WorkerLists::Encode(EncodeLists { pair: lists, .. }) => {
                    let read = self.loads_read(maker);
                    let loads = read.snapshot();
                    // The decode asked again is asked which decode it is sent to.
                    let sends_to = |target: &_| maker.sends_to(target);
                    pair::decided(own, &lists, &mut request, Some(held), loads, now, sends_to)
                }
                _ => single::issued(own, single::over_none(own, &request), &request, now),
            })
        })
    }

    fn say(&self, _: GatewayId, held: &HeldDecision<'_>, said: Said, _: Instant) {
        // A single's word no stage takes is told to nobody. A pair's
        // dispatch may carry its decode's issue: every word of one is told.
        let kind = match said {
            Said::Dispatched(..) => EventKind::Dispatched,
            Said::Completed(_) => EventKind::Completed,
            Said::Released(..) => EventKind::Released,
        };
        if held.lease.side == Side::Single && !self.document.hears.contains(kind) {
            return;
        }
        let engine = self.document.spec.engine();
        let tell = || match held.lease.side {
            Side::Single => lease_events::single(engine, held, said),
            // A pair's guard holds its prefill side's lease.
            Side::Prefill => lease_events::pair(engine, held, said, |at| self.clock.micros(at)),
            // No lease is issued for a side on its own.
            Side::Decode | Side::Encode => {}
        };
        self.word_caught(tell);
    }

    // No lease is filed, so no pass ends one: a lease ends at its guard's
    // word or drop. The session keys past their end leave the engine's
    // table here, off the path of a request: from this pass's first run no
    // read of a key sweeps the table. The records of sent prefixes take a
    // bounded step of the work they left waiting.
    fn sweep(&self, now: Instant) {
        self.sessions.sweep_due(self.clock.micros(now));
        self.affinity.sweep();
    }

    fn with_spec<R>(&self, read: impl FnOnce(&AppliedSpec) -> R) -> R {
        read(&self.document.spec)
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
        PromptHashes::under_document(&self.document.spec, size, tokens, text, seeds)
    }

    fn faults(&self) -> u64 {
        self.faults.load(Ordering::Relaxed)
    }

    // The shadow decisions counted for single requests. No shadow profile
    // runs on a pair's side here.
    fn shadowed(&self, binding: u32, kind: ShadowKind) -> u64 {
        let counts = self.document.shadow_counts.get();
        counts.map_or(0, |counts| {
            counts
                .of(binding, Counted::Single, kind)
                .load(Ordering::Relaxed)
        })
    }
}

/// [`PolicyEngine::pick`] by one test of a worker's read, compiled once a
/// test: the picker's own pick over the pool's word, then the pass where
/// that pick left the pool open.
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
    // A rule that asked every worker and kept none has the pool's answer:
    // the pass would read them all again to name nobody.
    if picked.is_some() || picker.asks_every_worker() {
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
    use std::{cell::Cell, collections::BTreeSet};

    use openai_protocol::worker::WorkerStatus;
    use smg_worker::{runtime::StallReason, test_support::ready_worker};

    use super::*;
    use crate::{
        document::spec::{Needs, PolicySpec},
        engine::{framework::InTheWay, list_cell::tests::PublishedList},
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
        match engine.place_for(MODEL, Phase::Single) {
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

    /// What a host says of its workers when it hands their list is in the
    /// list's records, each worker's at its place: the worker it named
    /// takes images and no other does. The host is asked once a worker, in
    /// the list's order, at the call. A list handed with nothing said says
    /// of no worker that it takes images.
    #[test]
    fn a_lists_records_take_the_image_bit_its_host_says() {
        let engine = preset("round_robin");
        let (list, asked) = (pool(4), Cell::new(0));
        let members = engine
            .loaded()
            .pool_members_with_fixed_facts(&list, |worker| {
                asked.set(asked.get() + 1);
                FixedFacts {
                    vision: Arc::ptr_eq(worker, &list[1]),
                    ..FixedFacts::default()
                }
            });
        assert_eq!(asked.get(), 4, "asked at the call, once a worker");
        let sees = |members: &PoolMembers| {
            let fixed = members.records(engine.spec(), &engine.clock).fixed();
            fixed.iter().map(|row| row.vision).collect::<Vec<_>>()
        };
        assert_eq!(sees(&members), [false, true, false, false]);
        assert_eq!(asked.get(), 4, "and not again where the records are read");
        assert_eq!(sees(&engine.pool_members(&list)), [false; 4]);
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
            engine.place_for("served-a", Phase::Single),
            Place::Alone(_)
        ));
        assert_eq!(engine.place_for("other", Phase::Single), Place::Unbound);
        assert_eq!(engine.place_for("", Phase::Single), Place::Unbound);
        // The preset binds every model, the empty id among them.
        let every = preset("round_robin");
        assert_eq!(
            every.place_for("", Phase::Single),
            every.place_for(MODEL, Phase::Single)
        );
        assert!(matches!(
            every.place_for("", Phase::Single),
            Place::Alone(_)
        ));
    }

    /// The places one pick decides: the picker places by itself, no stage
    /// scores, and the filters are the two tests of a worker's health.
    /// Round robin's preset is one, random's and passthrough's. No other preset
    /// is one, and each names its own profile and the stage of it that asks more.
    #[test]
    fn a_place_is_picked_alone_where_its_picker_decides_by_itself() {
        for name in ["round_robin", "random", "passthrough"] {
            let engine = preset(name);
            let place = engine.place_for(MODEL, Phase::Single);
            assert!(matches!(place, Place::Alone(_)), "{place:?}");
        }
        for name in [
            "power_of_two",
            "least_load",
            "consistent_hashing",
            "prefix_hash",
            "manual",
            "cache_aware",
            "bucket",
        ] {
            let engine = preset(name);
            let Place::NotAlone {
                profile,
                in_the_way: stage,
            } = engine.place_for(MODEL, Phase::Single)
            else {
                panic!("{name} is picked alone");
            };
            assert_eq!(profile, name);
            assert!(!stage.to_string().is_empty(), "{name}: {stage}");
        }
        assert_eq!(
            preset("manual").place_for(MODEL, Phase::Single),
            Place::NotAlone {
                profile: "manual",
                in_the_way: InTheWay::Picker("manual"),
            }
        );
        assert!(matches!(
            preset("least_load").place_for(MODEL, Phase::Single),
            Place::NotAlone {
                in_the_way: InTheWay::Scorer(_),
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
            engine.place_for(MODEL, Phase::Prefill),
            Place::Alone(_)
        ));
        assert!(matches!(
            engine.place_for(MODEL, Phase::Decode),
            Place::Alone(_)
        ));
        // The pair plan has no side for a single request, and none for an
        // encode one: a binding names the model, so the place is not
        // unbound.
        assert_eq!(engine.place_for(MODEL, Phase::Single), Place::NoSide);
        assert_eq!(engine.place_for(MODEL, Phase::Encode), Place::NoSide);

        for (filter, own) in [
            ("has-capacity", Phase::Prefill),
            ("has-room", Phase::Decode),
        ] {
            let engine = round_robin_with(filter);
            for phase in [Phase::Single, Phase::Prefill, Phase::Decode, Phase::Encode] {
                let place = engine.place_for(MODEL, phase);
                if phase == own {
                    assert!(matches!(place, Place::Alone(_)), "{filter} at {phase:?}");
                } else {
                    let stage = InTheWay::Filter(filter);
                    let profile = "round_robin";
                    let named = Place::NotAlone {
                        profile,
                        in_the_way: stage,
                    };
                    assert_eq!(place, named, "{phase:?}");
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
                engine_of(&document).place_for(MODEL, Phase::Single),
                Place::NotAlone {
                    profile,
                    in_the_way: stage
                }
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
            let members = PublishedList::new(pool(workers));
            let first = engine
                .pick(profile, members.lent(), RoutingState::eligible)
                .expect("a worker");
            for step in 1..2 * workers {
                let at = engine.pick(profile, members.lent(), RoutingState::eligible);
                assert_eq!(at, Some((first + step) % workers), "{workers} workers");
            }
        }
    }

    /// The `random` preset's pick is one draw of its picker over the pool:
    /// every ready worker takes a near even share and a worker that is not
    /// ready takes none, by the pool's own draws and by a pass over a list
    /// alike. A pool whose draws miss is passed over, and the pass names
    /// its ready worker.
    #[test]
    fn a_random_pick_is_one_draw_over_the_ready_workers() {
        let engine = preset("random");
        let profile = alone(&engine);
        let list = pool(8);
        list[3].set_status(WorkerStatus::NotReady);
        let members = PublishedList::new(Arc::clone(&list));
        let (mut picks, mut passes) = ([0u32; 8], [0u32; 8]);
        for _ in 0..4_000 {
            let at = engine.pick(profile, members.lent(), RoutingState::eligible);
            picks[at.expect("a ready worker")] += 1;
            let at = engine.pick_among(profile, &list, RoutingState::eligible);
            passes[at.expect("a ready worker")] += 1;
        }
        for shares in [picks, passes] {
            assert_eq!(shares[3], 0, "{shares:?}");
            let even = |&share: &u32| (350..800).contains(&share);
            assert_eq!(
                shares.iter().filter(|share| even(share)).count(),
                7,
                "{shares:?}"
            );
        }

        // One ready worker of 64: eight draws miss it nearly nine times in
        // ten, and the pass over the pool then names it.
        let wide = pool(64);
        for worker in wide.iter().skip(1) {
            worker.set_status(WorkerStatus::NotReady);
        }
        let members = PublishedList::new(wide);
        for _ in 0..200 {
            let at = engine.pick(profile, members.lent(), RoutingState::eligible);
            assert_eq!(at, Some(0));
        }
    }

    /// The rotation is the pool's own word: two pools of one engine turn a
    /// rotation each, and a turn of one moves no turn of the other.
    #[test]
    fn each_pool_turns_its_own_word() {
        let engine = preset("round_robin");
        let profile = alone(&engine);
        let (one, other) = (pool(5), pool(5));
        let (one, other) = (
            PublishedList::new(Arc::clone(&one)),
            PublishedList::new(Arc::clone(&other)),
        );
        let first = engine
            .pick(profile, one.lent(), RoutingState::eligible)
            .expect("a worker");
        let apart = engine
            .pick(profile, other.lent(), RoutingState::eligible)
            .expect("a worker");
        for step in 1..=10 {
            assert_eq!(
                engine.pick(profile, one.lent(), RoutingState::eligible),
                Some((first + step) % 5)
            );
            assert_eq!(
                engine.pick(profile, other.lent(), RoutingState::eligible),
                Some((apart + step) % 5)
            );
        }
    }

    /// Workers that are not ready take no request, and the ready ones take
    /// exact shares. At every size a turn on such a worker goes to the
    /// pass, and the ready workers' own rotation places the request, as
    /// the classic router's turn does
    /// (`model_gateway/src/policies/round_robin.rs:151-153`, `:164-167`):
    /// the turn is not handed to the list's next worker. Its own case of
    /// four workers holds the shares
    /// (`model_gateway/src/policies/round_robin.rs:430-441`).
    #[test]
    fn a_pick_keeps_exact_shares_past_workers_not_ready() {
        let engine = preset("round_robin");
        let profile = alone(&engine);
        let shares = |members: &PublishedList, picks: usize| {
            let mut times = vec![0usize; members.workers().len()];
            for _ in 0..picks {
                times[engine
                    .pick(profile, members.lent(), RoutingState::eligible)
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
        let members = PublishedList::new(Arc::clone(&list));
        let times = shares(&members, 8 * 7 * 3);
        for (at, &times) in times.iter().enumerate() {
            assert_eq!(times, if at == 5 { 0 } else { 24 }, "worker {at} of 8");
        }

        // Sixty-four workers, one in four not ready, none beside another.
        // Over three turns of the pool each ready worker takes three by
        // the pool's turn, and the 48 turns on a worker not ready go once
        // round the ready workers' own rotation.
        let list = pool(64);
        leave(&list, &|at| at % 4 != 1);
        let members = PublishedList::new(Arc::clone(&list));
        let times = shares(&members, 64 * 3);
        for (at, &times) in times.iter().enumerate() {
            assert_eq!(times, if at % 4 == 1 { 0 } else { 4 }, "worker {at} of 64");
        }

        // Sixty-four workers, sixteen in a row not ready, from the pool's
        // turn on the first of the run: the same shares. The worker behind
        // the run takes no more than its peers.
        let list = pool(64);
        leave(&list, &|at| !(20..36).contains(&at));
        let members = PublishedList::new(Arc::clone(&list));
        let made = engine.members_of(members.lent());
        made.rotation.0.store(20, Ordering::Relaxed);
        let times = shares(&members, 64 * 3);
        for (at, &times) in times.iter().enumerate() {
            let expected = if (20..36).contains(&at) { 0 } else { 4 };
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
        let members = PublishedList::new(Arc::clone(&list));
        for _ in 0..8 {
            assert_eq!(
                engine.pick(profile, members.lent(), RoutingState::eligible),
                Some(3)
            );
        }
        let lifted: BTreeSet<usize> = (0..8)
            .filter_map(|_| {
                engine.pick(
                    profile,
                    members.lent(),
                    RoutingState::eligible_ignoring_overload,
                )
            })
            .collect();
        assert_eq!(lifted, BTreeSet::from([1, 3]));

        list[3].set_overloaded(true);
        assert_eq!(
            engine.pick(profile, members.lent(), RoutingState::eligible),
            None
        );
        assert_eq!(
            engine.pick_among(profile, &list, RoutingState::eligible),
            None
        );
        assert!(engine
            .pick(
                profile,
                members.lent(),
                RoutingState::eligible_ignoring_overload
            )
            .is_some());
        // A pool that holds no worker.
        let empty = PublishedList::new(pool(0));
        assert_eq!(
            engine.pick(profile, empty.lent(), RoutingState::eligible),
            None
        );
        assert_eq!(
            engine.pick_among(profile, &[], RoutingState::eligible),
            None
        );
    }

    /// A pick that asked every worker of a pool and kept none has the pool's
    /// answer: a scan reads each worker once and no pass reads them again.
    /// A rotation's one turn stops short of the pool, which is then passed
    /// over.
    #[test]
    fn a_pick_that_asked_every_worker_is_not_followed_by_a_pass() {
        for workers in [1, 2, 8, 64] {
            let list = pool(workers);
            for worker in list.iter() {
                worker.set_overloaded(true);
            }
            let reads_under = |name: &str| {
                let engine = preset(name);
                let members = PublishedList::new(Arc::clone(&list));
                let reads = Cell::new(0);
                let counted = |state: RoutingState| {
                    reads.set(reads.get() + 1);
                    state.eligible()
                };
                assert_eq!(engine.pick(alone(&engine), members.lent(), counted), None);
                reads.get()
            };
            assert_eq!(reads_under("passthrough"), workers, "each worker once");
            assert!(reads_under("round_robin") > workers, "turns, then a pass");
        }
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

    /// The maker receives the document that governs the single request.
    /// The pick records the generation the maker reads.
    #[test]
    fn a_request_is_made_from_the_document_its_decision_is_made_under() {
        use super::single::tests::{request, Asks};

        struct ReadsDocument<'a> {
            asks: Asks<'a>,
            expected: &'a AppliedSpec,
            generation: Cell<Option<u64>>,
        }

        impl RequestMaker for ReadsDocument<'_> {
            fn model(&self) -> &str {
                self.asks.model()
            }

            fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
                assert!(
                    ptr::eq(spec, self.expected),
                    "the maker receives the engine's own document"
                );
                self.generation.set(Some(spec.generation()));
                self.asks.make(spec)
            }

            fn workers(&self) -> WorkerLists<'_> {
                self.asks.workers()
            }
        }

        let mut document = PolicySpec::preset("random").expect("a built-in preset");
        document.generation = 17;
        let engine = engine_of(&document);
        let members = PublishedList::new(pool(3));
        let maker = ReadsDocument {
            asks: Asks::of(request(), Some(&members)),
            expected: engine.spec(),
            generation: Cell::new(None),
        };

        let (outcome, _) = engine
            .decide_at(&maker, Instant::now)
            .expect("a decision")
            .whole()
            .expect("a whole decision");
        let pick = outcome.decision.pick().expect("a pick");
        let generation = maker.generation.get().expect("the maker ran");
        assert_eq!(generation, document.generation);
        assert_eq!(pick.trace.generation, generation);
    }
}
