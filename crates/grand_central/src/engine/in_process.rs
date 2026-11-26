//! The engine in its host's own process.
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
//!   places for: a model and a phase. [`SpecHandle::place_for`] finds the
//!   place's profile by the bindings of the document the handle shows,
//!   there and at no request.
//! - **Once a request.** [`InProcess::pick`] asks for one pick over the
//!   pool, under that handle and a profile the engine picks by its picker
//!   alone ([`Place::Alone`]). The pick is the picker's own rule: a rotation
//!   turns the pool's own word and reads the worker the turn lands on, a
//!   scan reads the pool's workers in order up to the first that passes.
//!   Each read is the pass's own test of a worker: no row, no table, no
//!   request fact, no lock, and nothing allocated. A list nothing is kept
//!   of (a part of a pool, as the pass narrowed it) is passed over once
//!   ([`InProcess::pick_among`]).
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
//! ([`InProcess::decide_encode_at`]).
//!
//! **A new document.** The engine takes a new document while it runs
//! ([`InProcess::apply`]). It holds the document in force as one value
//! and replaces it whole ([`document`]). A decision loads that value once,
//! at its entry, and ends under the document it began under.
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
mod sample;
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

use arc_swap::ArcSwap;
use parking_lot::Mutex;
use rustc_hash::FxHasher;
use smg_worker::{identity::basic::Worker, load::state::LoadSnapshot, runtime::RoutingState};

pub(crate) use self::document::Loaded;
pub use self::document::SpecHandle;
// The count of a request that names its worker, for a controller's served
// decision: the one the one call counts by.
#[cfg(feature = "controller")]
pub(crate) use self::place::target as named_worker;
use self::{
    document::InForce,
    encode::EncodeLists,
    overlap::{Affinity, PoolAffinity},
    place::Made,
};
use super::{
    decision_engine::{
        AlonePick, DecideOutcome, Decided, DecisionEngine, DecisionRefused, HeldDecision, Request,
        RequestMaker, WorkerLists,
    },
    framework::{release, AloneProfile, Engine, NotAlone, Picker, Place, PoolWord, ShadowRows},
    lease_events,
    list_cell::{HandedLists, LentList},
    prompt_hashes::{HashSeeds, PromptHashes},
    records::{DecisionClock, PoolRecords},
    shadow::Shadowed,
    sticky_sessions::{shards_for_this_host, SessionTable},
    view::{FixedFacts, Phase, WireModes, WorkerId},
};
use crate::{
    document::spec::AppliedSpec,
    stages::Lifecycle,
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

/// The panics an engine's own catches held, one count a catch: a whole
/// decision's ([`DecisionEngine::faults`]), a word's to the stages
/// ([`InProcess::unheard`]) and a step's of the engine's own task
/// ([`InProcess::upkeep_faults`]).
///
/// An engine counts into counts of its own, for its life: a count
/// outlasts the document it rose under, and a decision still running
/// under a document that was replaced counts where the engine's host
/// reads.
#[derive(Debug, Default)]
pub(crate) struct HeldFaults {
    decisions: AtomicU64,
    words: AtomicU64,
    upkeep: AtomicU64,
}

impl HeldFaults {
    /// The three counts as one read answers them: the decisions', the
    /// words' and the task's steps'.
    pub(crate) fn read(&self) -> [u64; 3] {
        [&self.decisions, &self.words, &self.upkeep].map(|count| count.load(Ordering::Relaxed))
    }
}

/// The counter family a host publishes an engine's held panics under, one
/// series a kind ([`POLICY_FAULT_KINDS`]). A gateway's metrics endpoint and
/// a controller's publish this one family, so its name, its kinds and its
/// description ([`POLICY_FAULTS_HELP`]) are said here once.
pub const POLICY_FAULTS_TOTAL: &str = "smg_policy_faults_total";

/// The description of [`POLICY_FAULTS_TOTAL`], the same on every page that
/// publishes the family. It names the engine's own catches and no host's
/// detail: which words and which steps pass those catches is each host's
/// to say.
pub const POLICY_FAULTS_HELP: &str = "Panics Grand Central held under its own catches, by kind: \
    decision (a decision that panicked, a shadow's beside a live decision among them), event (a \
    stage at a word Grand Central told it), upkeep (a step of Grand Central's upkeep)";

/// The kinds of [`POLICY_FAULTS_TOTAL`], in the order of an engine's three
/// counts: the decisions that panicked ([`DecisionEngine::faults`]), the
/// words a stage did not hear ([`InProcess::unheard`]) and the steps of
/// the upkeep that panicked ([`InProcess::upkeep_faults`]).
pub const POLICY_FAULT_KINDS: [&str; 3] = ["decision", "event", "upkeep"];

/// The engine that decides in its host's own process.
///
/// Built once, at the gateway's start, from the document in force then. It
/// holds that document's stages and their state, and no worker. A later
/// document is put in force on the same engine ([`Self::apply`]).
#[derive(Debug)]
pub struct InProcess {
    /// What the engine holds of the document in force ([`InForce`]): the
    /// document, its number, what the engine reads of the document once,
    /// and the shadow decisions counted under it. Replaced whole where a
    /// new document is put in force ([`Self::apply`]). A decision reads it
    /// once, at its entry, through the reference its thread keeps
    /// ([`Self::in_force`]): the cell itself is read where a thread kept
    /// another document or none.
    document: ArcSwap<InForce>,
    /// The number of the document in force ([`InForce::applied`]): what an
    /// entry compares with the number of the document its thread read
    /// last ([`Self::in_force`]). Written behind the document where a new
    /// one is put in force ([`Self::apply`]).
    applied: AtomicU64,
    /// Held while a change of membership is told and while a new document
    /// is put in force ([`Self::apply`]), so no word is told to a document
    /// that a new one is replacing. No decision takes it.
    telling: Mutex<()>,
    /// The pools made so far: each draws the place its turns start at from
    /// its number.
    epochs: AtomicU64,
    /// The one clock of the engine's decisions: a decision's instant and a
    /// load report's sample instant are counted from its origin.
    clock: DecisionClock,
    /// The panics the engine's catches held ([`HeldFaults`]). Its
    /// decisions: how many whole decisions panicked, and how many shadow
    /// profiles did after a live decision that stood
    /// ([`DecisionEngine::faults`]). Its words: how many words of a lease
    /// and changes of membership panicked in a stage that heard them
    /// ([`Self::unheard`]). Its task's steps: a run of the upkeep, a step
    /// of the registry's follow, or a telling of load reports
    /// ([`Self::upkeep_faults`]). The engine's own, for its life.
    held: HeldFaults,
    /// The last lease id handed out. A pair's lease writes it, and a thread
    /// once for a run of its single requests' ids (`single::lease_id`):
    /// one atomic step, on cache lines of its own.
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
/// Made where a list's cell is filled ([`InProcess::members_of`]), from
/// a list the gateway's registry published, and kept in that cell beside
/// the list. It holds the list itself, not a copy: a worker's live facts
/// are read on the list's objects.
#[derive(Debug)]
pub(crate) struct PoolMembers {
    workers: Arc<[Arc<dyn Worker>]>,
    /// The pool's own word, for the picker that places over the pool by
    /// itself ([`InProcess::pick`]): a rotation counts its turns here.
    /// A single request's whole decision over the list lends it to its
    /// draw ([`PoolFacts::word`](super::view::PoolFacts::word)), so the
    /// pick and that draw turn the one rotation.
    rotation: OwnWord,
    /// What a whole decision reads of the list beside its workers' live
    /// state ([`Self::records`]): made with the value where a list's cell
    /// is filled ([`Loaded::pool_members_with_fixed_facts`]). A value
    /// a case makes without them ([`InProcess::pool_members`]) makes
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
    /// warming rows, as a ruling read again once a second, and once the
    /// index handed a number on to another worker ([`PoolAffinity`]). A
    /// caller that asks a whole decision of its own
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

impl InProcess {
    /// The engine of `spec`, the document in force, handed nothing of its
    /// caller's data. A program has its engine by [`Self::start`], which
    /// builds it here and makes every hand-over.
    #[must_use]
    pub fn new(mut spec: AppliedSpec) -> Self {
        // The table takes the document's two times where the document's
        // engine takes the pointer.
        let sessions = Arc::new(SessionTable::new(0).with_shards(shards_for_this_host()));
        spec.share_sessions(Arc::clone(&sessions));
        let document = InForce::of(spec);
        Self {
            applied: AtomicU64::new(document.applied),
            document: ArcSwap::from_pointee(document),
            telling: Mutex::new(()),
            epochs: AtomicU64::new(0),
            clock: DecisionClock::new(),
            held: HeldFaults::default(),
            leases: OwnWord(AtomicU64::new(0)),
            sessions,
            affinity: Affinity::default(),
            loads: task::Loads::default(),
            members: membership::Handed::default(),
            lists: HandedLists::default(),
        }
    }

    /// [`Self::new`] on the clock of a host that reads one of its own, as
    /// a store does: the engine's clock starts at `origin`, the host's. An
    /// instant then reads alike on the engine's clock and on the host's,
    /// so a lease the host dates is read by a stage on the clock it was
    /// written on.
    #[must_use]
    pub(crate) fn with_origin(spec: AppliedSpec, origin: Instant) -> Self {
        Self {
            clock: DecisionClock::from_origin(origin),
            ..Self::new(spec)
        }
    }

    /// The three counts of the panics the engine held, as one read answers
    /// them ([`HeldFaults::read`]): the decisions', the words' and the
    /// task's steps'. A store reads them of its one engine.
    #[must_use]
    pub(crate) fn held_faults(&self) -> [u64; 3] {
        self.held.read()
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
    /// behind the leave enters its rank in a count of sends again, until a
    /// join of the id drops it ([`Self::joined`]). The
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
    ///
    /// A count of sends drops what it holds under the id: a dispatch said
    /// behind the id's removal, and with it a send to this worker said
    /// ahead of this word.
    pub fn joined(&self, address: &str) {
        self.came(WorkerId::from_address(address));
    }

    /// A removal by the worker's id: what [`Self::forget`] says.
    fn left(&self, id: WorkerId) {
        self.tell(&Membership::Removed(id));
    }

    /// A join by the worker's id: what [`Self::joined`] says.
    fn came(&self, id: WorkerId) {
        self.tell(&Membership::Added(id));
    }

    /// Tells the stages of the document in force one change of membership,
    /// on the calling thread ([`Self::told`]). It holds the telling's lock
    /// for the word: a new document is put in force between two words and
    /// never during one ([`Self::apply`]).
    fn tell(&self, change: &Membership<'_>) {
        let _held = self.telling.lock();
        self.told(self.document.load().spec.engine(), change);
    }

    /// Tells the stages of `engine` one change of membership, and not
    /// the session table. The records of sent prefixes hear it first
    /// ([`Self::hears`]). A stage that panics at it is held and counted
    /// ([`Self::word_caught`]): the registration or the removal goes on.
    /// Its caller holds the telling's lock. A decision takes that lock
    /// nowhere.
    fn told(&self, engine: &Engine, change: &Membership<'_>) {
        self.hears(engine, change, |engine| {
            self.word_caught(|| engine.membership_stages(change));
        });
    }

    /// One word to the document's stages, `tell`, under a catch of its own:
    /// what the word answers, and `None` where a stage panicked at it. The
    /// panic is held here and counted ([`Self::unheard`]). The stages behind
    /// the one that panicked do not hear the word, and what the word is of
    /// goes on: a request, a registration, a removal or an apply.
    #[inline]
    fn word_caught<R>(&self, tell: impl FnOnce() -> R) -> Option<R> {
        match catch_unwind(AssertUnwindSafe(tell)) {
            Ok(answered) => Some(answered),
            Err(payload) => {
                release(payload);
                self.held.words.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// One word of a store's membership, said under the store's lock of its
    /// workers ([`Self::hears`]): the document's engine hears it whole, its
    /// session table and then its stages ([`Lifecycle::on_membership`]). A
    /// stage that panics at it is held and counted, as at a gateway's word
    /// ([`Self::word_caught`]): the session table, told ahead of the
    /// stages, stays told, the stages behind the one that panicked do not
    /// hear the word, and the store's write goes on with what it still
    /// owes.
    ///
    /// It holds the telling's lock for the word, as a gateway's word does
    /// ([`Self::tell`]). The locks are taken in one order: the store's, the
    /// telling's, then the records' own ([`Affinity::forget`]). A decision
    /// takes a record's lock and neither of the other two.
    pub(crate) fn on_membership(&self, word: &Membership<'_>) {
        let _held = self.telling.lock();
        self.hears(self.document.load().spec.engine(), word, |engine| {
            self.word_caught(|| engine.on_membership(word));
        });
    }

    /// [`Self::on_membership`] for what the document in force built new:
    /// the stages and the counts of sends made under its own generation,
    /// and its session table ([`Engine::hearing`]). A store says the list
    /// of its members through this at the document's apply: what the
    /// document kept of the standing one heard each member as it came, and
    /// is told nothing again. The word passes the same locks and the same
    /// catch: a stage that panics at the list is held and counted.
    pub(crate) fn on_membership_built_new(&self, word: &Membership<'_>) {
        let _held = self.telling.lock();
        let document = self.document.load();
        let own = document.spec.generation();
        let new = document.spec.engine().hearing(|made| made == own);
        self.hears(&new, word, |new| {
            self.word_caught(|| new.on_membership(word));
        });
    }

    /// One change of membership as the engine hears it under either host,
    /// the one place a word of membership passes: the records of sent
    /// prefixes first, then `engine` by `core`, the host's own way of
    /// telling it. The engine is the document's, or the part of it a word
    /// is told to ([`Engine::hearing`]). Its caller holds the telling's
    /// lock.
    ///
    /// | The word | The records of sent prefixes |
    /// |---|---|
    /// | `Added` | Nothing. |
    /// | `Removed` | Each forgets the worker. |
    /// | `Replaced` | Each forgets the old id. |
    /// | `Relabelled` | Nothing. |
    /// | `Resized` | Nothing. |
    /// | `Snapshot` | Each forgets every worker that is not among the ids. |
    /// | `Left` | Each forgets the worker. |
    ///
    /// A worker that stands under its id and serves a model no more is
    /// forgotten whole, in every model's record: an entry is a prediction,
    /// and the next request of a prefix writes it again.
    fn hears(&self, engine: &Engine, change: &Membership<'_>, core: impl FnOnce(&Engine)) {
        match *change {
            Membership::Removed(id)
            | Membership::Replaced { old: id, .. }
            | Membership::Left { id, .. } => self.affinity.forget(id),
            Membership::Snapshot(standing) => self.affinity.keep(standing),
            Membership::Added(_) | Membership::Relabelled(_) | Membership::Resized(_) => {}
        }
        core(engine);
    }

    /// The clock of the engine's decisions: a list's records are made on
    /// it.
    #[must_use]
    pub fn clock(&self) -> &DecisionClock {
        &self.clock
    }

    /// How many words of a lease (its issue, its dispatch, its end, its
    /// release) and changes of membership panicked in a stage that heard
    /// them. The word stands as said: the request, the registration, the
    /// removal or the apply it is of goes on, and the stages after the one
    /// that panicked did not hear it.
    ///
    /// A lease's issue is such a word at a first decision and at a decode's
    /// own call alike: the call answers its pick and the lease all the same
    /// ([`DecisionEngine::decide_under`],
    /// [`DecisionEngine::decide_decode_under`]).
    #[must_use]
    pub fn unheard(&self) -> u64 {
        self.held.words.load(Ordering::Relaxed)
    }

    /// The document in force, in a handle that keeps showing it whatever
    /// document is put in force behind the call ([`SpecHandle`]): one
    /// counted reference of the document.
    #[must_use]
    pub fn spec(&self) -> SpecHandle {
        SpecHandle(self.document.load_full())
    }

    /// The number of the document in force, which no other applied
    /// document of the process has. A list's value is made under it
    /// and read under it alone ([`Self::members_of`]). Read of the
    /// engine's cell, as [`Self::spec`] is: no request's entry asks it, so
    /// the calling thread keeps no document for it.
    pub(super) fn applied(&self) -> u64 {
        self.document.load().applied
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
    /// [`InProcess::pool_members`] with the list's records made of what its
    /// program states of its workers beside their objects ([`FixedFacts`]):
    /// the id it names a worker by where it keeps ids of its own, whether a
    /// worker takes images, or what it states for a request's hard demands.
    /// The fill of a list's cell calls it ([`Self::members_of`]): inside
    /// the first decision that asks the cell, or by the engine's own task
    /// ahead of it ([`InProcess::set_lists`]). `facts_of` is asked once a
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

impl SpecHandle {
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
    /// request: a policy's name finds nothing here. A profile it answers
    /// is picked under this handle alone ([`InProcess::pick`]).
    ///
    /// [`Engine::place_for`]: super::framework::Engine::place_for
    #[must_use]
    pub fn place_for(&self, model: &str, phase: Phase) -> Place<'_> {
        self.0.spec.engine().place_for(model, phase)
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
        self.0.spec.engine().not_alone()
    }
}

impl InProcess {
    /// One pick under `profile` over `list`, a pool its caller lends with
    /// its cell: the place in the pool's list of the worker picked. `None`
    /// where no worker of the pool passes. The engine's value of the pool
    /// is asked of the cell: a read once the cell holds it, and the value's
    /// making at the list's first ask under `document`, whose cost grows
    /// with the list.
    ///
    /// `document` is the handle `profile` was asked of
    /// ([`SpecHandle::place_for`]): the pick runs under that document,
    /// whatever document the engine holds in force by then. A caller that
    /// keeps a place's profile keeps its handle beside it, and asks both
    /// again behind a new document it puts in force ([`Self::apply`]).
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
        document: &SpecHandle,
        profile: AloneProfile,
        list: LentList<'_>,
        admits: impl Fn(RoutingState) -> bool + Copy,
    ) -> Option<usize> {
        let own = self.under(&document.0);
        own.pick_over(profile, own.members_of(list), admits)
    }
}

impl Loaded<'_, '_> {
    /// [`InProcess::pick`] over `members`, the engine's value of the pool, which
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

impl InProcess {
    /// One pick under `profile` among `workers`, a list this engine keeps
    /// nothing of (a part of a pool, as a caller narrowed it): the place in
    /// `workers` of the worker picked. `None` where no worker passes.
    /// `document` is the handle `profile` was asked of, as at
    /// [`Self::pick`].
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
        document: &SpecHandle,
        profile: AloneProfile,
        workers: &[Arc<dyn Worker>],
        admits: impl Fn(RoutingState) -> bool + Copy,
    ) -> Option<usize> {
        among(self.under(&document.0).picker_of(profile)?, workers, admits)
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

impl InProcess {
    /// A whole decision under the engine's catch: a panic in it costs its
    /// request alone and is counted ([`DecisionEngine::faults`]). A side's
    /// shadow of a pair runs under a catch of its own here too
    /// (`pair::side_shadows`): its panic is counted the same and costs
    /// its request nothing.
    #[inline]
    fn caught<R>(
        &self,
        decide: impl FnOnce() -> Result<R, DecisionRefused>,
    ) -> Result<R, DecisionRefused> {
        match catch_unwind(AssertUnwindSafe(decide)) {
            Ok(decided) => decided,
            Err(payload) => {
                release(payload);
                self.held.decisions.fetch_add(1, Ordering::Relaxed);
                Err(DecisionRefused::Faulted)
            }
        }
    }

    /// The one entry ([`DecisionEngine::decide_under`]) for a host that
    /// hands a clock and no instant: `clock` is read once, where the
    /// decision is dated. A place ([`Decided::Alone`]) dates nothing, so it
    /// is answered with no read of a clock, and with no copy of a decision:
    /// the catch hands the place back, and a whole decision is written
    /// beside it. A place that is kept as a sample ([`Self::explained`])
    /// reads the clock for its record.
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
        let document = self.in_force();
        let alone = self
            .under(&document)
            .decide_into(maker, None, clock, &mut whole);
        // The document is let go ahead of the answer's making: an answer
        // made while it is held is moved once more behind its release.
        drop(document);
        match (alone?, whole) {
            (Some(pick), _) => Ok(Decided::Alone(pick)),
            (None, Some((decided, later))) => Ok(Decided::Whole(decided, later)),
            // The pass writes its decision ahead of every answer but a
            // place.
            (None, None) => Err(DecisionRefused::Faulted),
        }
    }
}

impl Loaded<'_, '_> {
    /// The decision of [`InProcess::decide_at`] under the document this
    /// value holds, for a caller that keeps the room of a whole decision
    /// itself, as the call that places a request does, and that reads the
    /// document further behind it: the place it answers, or `None` with the
    /// decision written into `whole` beside whether the request stated the
    /// later decode. Its caller hands `whole` empty, and only a whole
    /// decision fills it.
    ///
    /// `late` is the request as the engine's own call made it, where that
    /// call asks ([`place::decide`]): the lookup that names a request's
    /// holders asks it for the block hashes its making left to the road
    /// that reads them ([`Made::hashed_whole`]). A host's own maker hashes
    /// its request itself and hands none.
    fn decide_into(
        self,
        maker: &impl RequestMaker,
        late: Option<&Made<'_, '_>>,
        clock: impl FnOnce() -> Instant,
        whole: &mut Option<(DecideOutcome, bool)>,
    ) -> Result<Option<AlonePick>, DecisionRefused> {
        self.caught(|| self.placed_or(maker, late, clock, whole))
    }
}

impl Loaded<'_, '_> {
    /// One request's decision and nothing its host hears of later: no lease
    /// is issued, no id is taken and no stage is told. For a host that
    /// files a pick's lease itself and tells the stages of the issue under
    /// that lease
    /// ([`Lifecycle::on_decision_issued`]
    /// of the document's engine), as a controller's service does in its
    /// store.
    ///
    /// The decision runs under the document this value holds, at each of
    /// its steps. Such a host hands the document of the state its request
    /// read ([`SpecHandle::under`]), so the decision, the lease it files
    /// and the stages it tells are one document's, whatever document is
    /// put in force meanwhile.
    ///
    /// The request is made once and decided whole, at `now`, as
    /// [`InProcess::decide_at`] decides a request that is not plain
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
    /// none under either entry. A pair's prefill answers no word: the
    /// pair's decision alone asks no side's shadow ([`pair::decide`]), so
    /// none runs here yet.
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
        self,
        maker: &impl RequestMaker,
        now: Instant,
    ) -> Result<(super::decision::Decision, Option<Shadowed>), DecisionRefused> {
        let own = self;
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
                    Ok(own.decide_made(maker, None, &mut request, members, now, &mut said))
                }
                // A pair is decided at its prefill, over its host's two
                // lists, and so is a request that hands an encode
                // request's three.
                (
                    Phase::Prefill,
                    WorkerLists::Pair(lists) | WorkerLists::Encode(EncodeLists { pair: lists, .. }),
                ) => {
                    let read = own.loads_read(maker);
                    let loads = read.snapshot();
                    let sends_to = |target: &_| maker.sends_to(target);
                    let counted = || maker.session_counted_under();
                    Ok(pair::decide(
                        own,
                        &lists,
                        &mut request,
                        None,
                        loads,
                        now,
                        sends_to,
                        counted,
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
    /// `issued_micros` is the instant the pair's lease was issued at, as
    /// the host's record of that lease states it, `0` where it states none.
    /// A claim the call makes for its key takes that date
    /// ([`pair::date_decode_claim`]): the host tells each word of the side
    /// under that lease's date.
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
        self,
        maker: &impl RequestMaker,
        held: &super::decision::Target,
        issued_micros: u64,
        now: Instant,
    ) -> Result<super::decision::Decision, DecisionRefused> {
        let own = self;
        self.caught(|| {
            let mut request = own.made(maker);
            let decided = Ok(match maker.workers() {
                WorkerLists::Pair(lists) | WorkerLists::Encode(EncodeLists { pair: lists, .. }) => {
                    let read = own.loads_read(maker);
                    let loads = read.snapshot();
                    // The decode asked again is asked which decode it is sent to.
                    let sends_to = |target: &_| maker.sends_to(target);
                    let counted = || maker.session_counted_under();
                    pair::decide(
                        own,
                        &lists,
                        &mut request,
                        Some(held),
                        loads,
                        now,
                        sends_to,
                        counted,
                    )
                }
                _ => single::over_none(own, &request),
            });
            // A claim the call made takes the date of the pair's lease. The
            // decision is read where the answer stands, as a guard's call
            // reads it ([`pair::issued`]).
            if let Ok(decision) = &decided {
                pair::date_decode_claim(own, &request.facts, decision, issued_micros);
            }
            decided
        })
    }
}

impl Loaded<'_, '_> {
    /// The whole decision of a made request that stands alone
    /// ([`Self::stands_alone`]) over its host's list, for every entry that
    /// decides one. It reads the snapshot the request
    /// gives where it gives one, and the newest the engine's own task
    /// keeps at hand otherwise ([`Self::loads_read`]).
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
        late: Option<&Made<'_, '_>>,
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
            return signals::decide(self, request, members, maker, late, loads, now, said);
        }
        self.decide_with_loads(request, members, maker, late, loads, now, said)
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
    /// maker as a trait object ([`signals::decide`]). `late` is handed on
    /// with it ([`Self::decide_into`]).
    #[inline]
    #[expect(
        clippy::too_many_arguments,
        reason = "a made request, its list, its maker and the request its call made, the loads, the instant and the shadow's word"
    )]
    fn decide_with_loads(
        self,
        request: &mut Request<'_>,
        members: &PoolMembers,
        maker: &(impl RequestMaker + ?Sized),
        late: Option<&Made<'_, '_>>,
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
                    with_shadow::decide(self, request, members, maker, late, loads, now, binding);
                *said = shadow;
                return decision;
            }
        }
        single::decide(
            self,
            request,
            members,
            maker,
            late,
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
        late: Option<&Made<'_, '_>>,
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
                    // No stage takes a word of the request and the
                    // document keeps no sample: the place is the whole
                    // answer.
                    let document = self.document;
                    if !document.leases_or_samples {
                        return Ok(Some(pick));
                    }
                    // A document that keeps a sample and takes no word: the
                    // place is the whole answer all the same, counted
                    // toward the thread's next sample behind the pick.
                    if !document.hears.intersects(EventSet::LEASE) {
                        sample::place(self, maker, pick, members, clock);
                        return Ok(Some(pick));
                    }
                    // The session table alone takes a word, and it has none
                    // for the pick, made under a binding that states no
                    // rule: the place is the whole answer where its worker
                    // reads ready a second time, as a noted pick's target
                    // is read at its walk. A worker that turned not ready
                    // behind the pick is noted.
                    if !document.hears_beside_table.intersects(EventSet::LEASE)
                        && single::still_ready(pick, members)
                    {
                        if document.samples {
                            sample::place(self, maker, pick, members, clock);
                        }
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
                sample::noted(self, maker, now, whole.as_ref());
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
                let decision = self.decide_made(maker, late, &mut request, members, now, &mut said);
                // What a shadow said is counted behind the lease's issue.
                if let Some(said) = said {
                    let decided = with_shadow::issued(self, decision, &request, now, said);
                    *whole = Some((decided, later));
                    sample::whole(self, maker, &request, now, whole.as_ref(), Some(said));
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
                let counted = || maker.session_counted_under();
                let decided = pair::decided(
                    self,
                    &lists,
                    &mut request,
                    None,
                    loads,
                    now,
                    sends_to,
                    counted,
                );
                *whole = Some((decided, later));
                sample::whole(self, maker, &request, now, whole.as_ref(), None);
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
        sample::whole(self, maker, &request, now, whole.as_ref(), None);
        Ok(None)
    }
}

/// The engine of a gateway's own process, asked for whole decisions.
///
/// It decides over the lists the request's maker hands and reads a worker
/// where the gateway keeps it. It files no lease: an id is taken of a
/// counter, the stages hear the issue inside the decision, on the calling
/// thread, and each later word is told from what the request's guard lends
/// back ([`HeldDecision`]). A lease whose dispatch is never named ends at
/// its guard's drop: no table holds it, and no pass looks for it.
///
/// What every pick shares is the id counter alone, one atomic step on cache
/// lines of its own, which a thread takes once for a run of its single
/// requests' ids. No lock of the engine is waited for, only the one build
/// of a list's records, wires or one-pick profile: a stage's lock is its own.
///
/// A panic in a decision costs that request its decision and is counted
/// ([`DecisionEngine::faults`]). A shadow's panic costs its request nothing
/// and is counted with them. A panic in a stage that hears a word is held
/// here too and counted ([`InProcess::unheard`]): the request goes on.
///
/// A word of a lease is told to the stages of the document in force, where
/// that document issued the lease. A lease issued under a document that a
/// new one replaced since is told to nobody ([`InProcess::apply`]). The
/// shadow counts are those of the document in force.
impl DecisionEngine for InProcess {
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
        let document = self.in_force();
        let own = self.under(&document);
        self.caught(|| {
            let mut request = own.made(maker);
            Ok(match maker.workers() {
                // An encode request's decode is asked again over its
                // pair's two lists.
                WorkerLists::Pair(lists) | WorkerLists::Encode(EncodeLists { pair: lists, .. }) => {
                    let read = own.loads_read(maker);
                    let loads = read.snapshot();
                    // The decode asked again is asked which decode it is sent to.
                    let sends_to = |target: &_| maker.sends_to(target);
                    let counted = || maker.session_counted_under();
                    pair::decided(
                        own,
                        &lists,
                        &mut request,
                        Some(held),
                        loads,
                        now,
                        sends_to,
                        counted,
                    )
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
        // A word of a decision made under an earlier document goes to the
        // stages that stood when its lease was issued, and to no stage
        // built since. Its dispatch is counted by every count of sends in
        // force: a send is a fact of its worker, whatever document decided
        // it.
        let document = self.in_force();
        let sends = document.spec.engine();
        let mut engine = sends;
        if held.lease.generation != document.spec.generation() {
            engine = document.stood_at(held.lease.generation);
        } else if held.lease.side == Side::Single && !document.hears.contains(kind) {
            return;
        }
        let micros = |at| self.clock.micros(at);
        let tell = || match held.lease.side {
            Side::Single => lease_events::single(&lease_events::of_host(engine, sends), held, said),
            // A pair's guard holds its prefill side's lease.
            Side::Prefill => {
                lease_events::pair(&lease_events::of_host(engine, sends), held, said, micros);
            }
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
        read(&self.in_force().spec)
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
        PromptHashes::under_document(&self.in_force().spec, size, tokens, text, seeds)
    }

    fn faults(&self) -> u64 {
        self.held.decisions.load(Ordering::Relaxed)
    }
}

/// [`InProcess::pick`] by one test of a worker's read, compiled once a
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

/// [`InProcess::pick_among`] by one test of a worker's read: the workers
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
#[path = "../../tests/unit/engine/in_process/tests.rs"]
mod tests;
