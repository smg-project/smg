//! The records a decision reads, written from a gateway's own worker
//! objects.
//!
//! A gateway that decides in its own process keeps its workers where it
//! keeps them with no policy engine: its registry's list of worker objects
//! a pool, a worker's health and its count of requests in flight on the
//! worker object, and the load monitor's snapshot of the workers' reports.
//! None of it is copied into a store. Every stage call takes records
//! ([`Candidate`]), so a decision of a profile that scores is handed
//! records written for that one decision, from those three:
//!
//! - **Once a membership.** [`PoolRecords::new`] reads what a worker's
//!   registration states and no request changes: its id, its key, its role,
//!   its rank, its wire, and its engine kind and labels as the document
//!   lists them. It lists the pool's engines and numbers the pool. The
//!   gateway keeps the value beside the list it was made from. No live
//!   value of a worker is in it. What no worker object holds, the program
//!   that holds it says here too, once a worker
//!   ([`PoolRecords::with_fixed_facts`]): whether the worker takes images,
//!   the id it knows the worker by where it keeps ids of its own, and
//!   what the worker states for a request's hard demands.
//! - **Once a load snapshot.** A snapshot is published whole and never
//!   changed, so a worker's report reads the same at every decision that
//!   holds the snapshot. One decision that writes every record of a list
//!   under a snapshot converts the list's reports, and the list keeps
//!   those rows for the decisions behind it that hold the same snapshot
//!   ([`Reported`]). Draws, which write a few records each, convert the
//!   list's reports once they looked up as many under the snapshot as the
//!   list holds ([`LookUps`]). A decision that holds another snapshot
//!   never reads the rows: it reads its own snapshot's reports. Under a
//!   document one of whose stages reads a named signal, the same
//!   conversion makes the list's signal cells from the same reports, and
//!   a decision whose profile reads one is lent them
//!   ([`PoolRecords::cells`]).
//! - **Once a decision.** What a worker object says is read where the
//!   gateway keeps it, at the decision's instant ([`Live`]): ready,
//!   breaker, overload, stall, requests in flight. So is the age of the
//!   worker's report. [`PoolRecords::write`] writes every worker's record
//!   into a buffer of its caller, and [`PoolRecords::read`] one worker's.
//!   Neither waits for a lock. A write allocates nothing once the buffer has
//!   grown to the pool, but for the one decision a snapshot that converts
//!   the list's reports.
//!
//! [`decide`] is one single decision over such records, in the calling
//! thread's own buffer. It writes the records the decision reads: every
//! worker's where the profile's stages read the pool, and where the
//! profile's picker draws first, the drawn workers' alone. Under a binding
//! that names a shadow profile, [`with_shadow::decide`] is that decision
//! followed by the shadow profile's, over every worker's record.
//!
//! A host that decides for many gateways states the asking gateway's
//! standing vetoes beside the records. Where the decision's profile shows
//! its rows to a plugin, they are written into the records ([`vetoes`]).
//!
//! One clock dates both sides ([`DecisionClock`]): a decision's instant and
//! a report's sample instant are microseconds since one origin, so a load
//! stage's credit of dispatches is released by a report sampled after them
//! and by no other.

pub(crate) mod vetoes;
pub(crate) mod with_shadow;

use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    mem, ptr,
    rc::Rc,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Weak,
    },
    time::{Duration, Instant},
};

use arc_swap::{ArcSwapOption, Guard};
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use smg_worker::{
    identity::basic::Worker,
    load::state::LoadSnapshot,
    worker::{WorkerLoadResponse, WorkerSpec, WorkerStatus},
};

use super::{
    decision::Decision,
    demands::{DemandFacts, Demands, UnmetDemand},
    framework::{Engine, Scratch},
    view::{
        Candidate, Constraints, Engines, FixedFacts, Health, LoadRow, OverloadRegime, PoolFacts,
        RequestFacts, Role, SignalSource, WireModes, WorkerId, LOAD_REPORT_CLOCK_HISTORY_MS,
    },
};
use crate::{
    document::spec::AppliedSpec,
    load::{load_row, signal_values, ReportSignals, SignalCells, SignalColumn},
};

/// A worker's id in the engine's own state (a load stage's credit of
/// dispatches, a hash ring, session pins, a stage's bookings): the
/// fingerprint of its registration key, its address.
///
/// The same for every worker object registered at one address, so it is
/// the same at every attempt of a request and across a membership change,
/// and a removal names the id every decision knew the worker by.
///
/// A program that keeps ids of its own states each worker's with its
/// list's fixed facts ([`FixedFacts::id`]): the list's records then carry
/// that id in this one's place, and a removal said by address still names
/// this one.
#[must_use]
pub fn worker_id(worker: &dyn Worker) -> WorkerId {
    WorkerId::from_address(worker.url())
}

/// The data-parallel rank a pick that reads no record names for `worker`:
/// the rank its registration states, and the first where it states none.
#[must_use]
pub fn worker_rank(worker: &dyn Worker) -> u16 {
    u16::try_from(worker.dp_rank().unwrap_or(0)).unwrap_or(u16::MAX)
}

/// The clock of one engine's decisions over a gateway's own data, and its
/// count of the pools made.
///
/// A decision's instant ([`RequestFacts::now_micros`]) and a record's sample
/// instant ([`Candidate::load_sampled_at_micros`]) are microseconds since
/// this clock's origin. The origin stands an hour before the clock was
/// made, where the host's clock reaches that far, so a report sampled
/// before the engine started keeps an instant of its own.
///
/// The count of pools is the engine's membership epoch
/// ([`PoolFacts::membership_epoch`]): it moves each time a pool's records
/// are made ([`PoolRecords::new`]), which a gateway does once for each list
/// its registry publishes. One epoch stands for every pool of the engine,
/// as a stage that keeps what it learned of the pools reads it.
///
/// One a process: an engine that decides in a gateway's own process holds
/// it beside its document.
#[derive(Debug)]
pub struct DecisionClock {
    origin: Instant,
    pools: AtomicU64,
}

impl DecisionClock {
    /// The clock, started now.
    #[must_use]
    pub fn new() -> Self {
        let now = Instant::now();
        let history = Duration::from_millis(u64::from(LOAD_REPORT_CLOCK_HISTORY_MS));
        Self {
            origin: now.checked_sub(history).unwrap_or(now),
            pools: AtomicU64::new(0),
        }
    }

    /// The clock whose origin is `origin`: for a host that reads a clock of
    /// its own and builds an engine for each of its documents, so one
    /// instant reads alike on the host's clock and on every engine's.
    #[must_use]
    #[cfg_attr(
        not(feature = "in-process"),
        expect(dead_code, reason = "the engine over worker lists takes it")
    )]
    pub(crate) fn from_origin(origin: Instant) -> Self {
        Self {
            origin,
            pools: AtomicU64::new(0),
        }
    }

    /// `at` on this clock: microseconds since its origin, saturating at
    /// both ends.
    #[must_use]
    pub fn micros(&self, at: Instant) -> u64 {
        u64::try_from(at.saturating_duration_since(self.origin).as_micros()).unwrap_or(u64::MAX)
    }

    /// The membership epoch as it stands: the count of the pools made.
    #[inline]
    fn membership_epoch(&self) -> u64 {
        self.pools.load(Ordering::Relaxed)
    }
}

impl Default for DecisionClock {
    fn default() -> Self {
        Self::new()
    }
}

/// Where the live half of a record is read for one decision, and when: the
/// decision's instant on the engine's clock, the load monitor's snapshot,
/// and the request's session key where a stage counts its requests.
///
/// Made once a decision ([`Live::at`]) and handed to each write of it
/// ([`PoolRecords::write`], [`PoolRecords::read`], [`decide`]), so every
/// record of one decision is of one instant and one snapshot.
#[derive(Clone, Copy, Debug)]
pub struct Live<'a> {
    clock: &'a DecisionClock,
    now_micros: u64,
    loads: Option<&'a Arc<LoadSnapshot>>,
    /// The name the request's session key is counted under on a worker's
    /// object, where a stage of the document reads a worker's count of the
    /// key's requests in flight.
    counted_key: Option<&'a str>,
    /// Whether a stage of the document reads a worker's count of session
    /// keys with a request in flight.
    sessions: bool,
    /// The list's own word, where the decision is over one list
    /// ([`Self::turning`]).
    word: Option<&'a AtomicU64>,
}

impl<'a> Live<'a> {
    /// The reads of one decision at `now` on `clock`, under `engine`'s
    /// document: the workers' reports from `loads`, the load monitor's
    /// snapshot as the caller holds it for the decision, and `session_key`,
    /// the name the request's session key is counted under on a worker's
    /// object where it has one
    /// ([`SessionCountedUnder::name`](super::view::SessionCountedUnder::name)).
    ///
    /// The two counts of sessions a worker object keeps are read only for a
    /// document one of whose stages reads them ([`Engine::needs`]): the
    /// key's requests in flight on a worker
    /// ([`Candidate::session_in_flight`]) where the request carries a key,
    /// and the keys with a request in flight on it
    /// ([`Candidate::active_sessions`]). Under any other document a record
    /// carries zero in both and no count is read.
    #[must_use]
    pub fn at(
        engine: &Engine,
        clock: &'a DecisionClock,
        now: Instant,
        loads: Option<&'a Arc<LoadSnapshot>>,
        session_key: Option<&'a str>,
    ) -> Self {
        let needs = engine.needs();
        Self {
            clock,
            now_micros: clock.micros(now),
            loads,
            counted_key: session_key.filter(|_| needs.session_in_flight),
            sessions: needs.active_sessions,
            word: None,
        }
    }

    /// These reads for a decision over the one list whose own word is
    /// `word`, the word a pick over the list alone counts its turns in: the
    /// pool's facts made by them lend it ([`PoolFacts::word`]), so a picker
    /// that takes turns counts a draw's turn in it too. A pair's decision
    /// reads two lists by one value of these and lends none.
    #[must_use]
    pub(crate) fn turning(self, word: &'a AtomicU64) -> Self {
        Self {
            word: Some(word),
            ..self
        }
    }

    /// The decision's instant on the engine's clock: what the request's
    /// facts carry ([`RequestFacts::now_micros`]).
    #[must_use]
    pub fn now_micros(&self) -> u64 {
        self.now_micros
    }

    /// Writes the live half of `worker`'s record into `row`, whose fixed
    /// half stands, and says whether the worker carries a report: what the
    /// worker object says ([`Self::object`]), the worker's report from the
    /// load snapshot by one look-up of its address ([`report_into`]), and
    /// the report's age at the decision ([`Self::age`]).
    ///
    /// For a record written alone, where its list holds no rows of the
    /// decision's snapshot ([`Reported`]).
    #[inline]
    fn write(&self, worker: &dyn Worker, row: &mut Candidate) -> bool {
        self.object(worker, row);
        let reported = self
            .loads
            .is_some_and(|loads| report_into(worker, loads, self.clock, row));
        self.age(row);
        reported
    }

    /// Writes what `worker`'s object says into `row`: one read of the
    /// object for its status, its circuit breaker, its overload and
    /// liveness vetoes and its count of requests in flight, and the two
    /// counts of sessions where a stage of the document reads them. A
    /// worker whose breaker refuses execution carries the stalled bit
    /// beside the breaker's, as a veto that holds is written: no rule
    /// lifts it.
    ///
    /// Read at every decision and kept nowhere.
    #[inline]
    fn object(&self, worker: &dyn Worker, row: &mut Candidate) {
        let state = worker.routing_state();
        let held = !state.can_execute;
        row.health = Health {
            healthy: state.healthy,
            breaker_open: held,
            stalled: state.stalled || held,
            // Asked of a worker that is not ready alone: a ready one is
            // not being drained.
            draining: !state.healthy && worker.status() == WorkerStatus::Draining,
            overloaded: state.overloaded,
        };
        row.in_flight = u32::try_from(state.load).unwrap_or(u32::MAX);
        if let Some(key) = self.counted_key {
            row.session_in_flight =
                u16::try_from(worker.routing_key_inflight(key)).unwrap_or(u16::MAX);
        }
        if self.sessions {
            row.active_sessions = u16::try_from(worker.routing_key_load()).unwrap_or(u16::MAX);
        }
    }

    /// Writes the age of `row`'s report at the decision: the decision's
    /// instant less the sample's, both on the engine's clock, the two
    /// values a stage is handed ([`RequestFacts::now_micros`],
    /// [`Candidate::load_sampled_at_micros`]). Milliseconds, saturating at
    /// `u32::MAX` and at zero; never wrapping.
    #[inline]
    fn age(&self, row: &mut Candidate) {
        if let Some(load) = &mut row.load {
            let micros = self.now_micros.saturating_sub(row.load_sampled_at_micros);
            load.report_age_ms = u32::try_from(micros / 1_000).unwrap_or(u32::MAX);
        }
    }
}

/// Writes `worker`'s report of `loads` into `row`, with the sample's
/// instant on `clock`, and says whether the snapshot holds a report of the
/// worker: one look-up of the worker's address, one conversion
/// ([`load_row`]). The report's age is left at rest: it is a decision's to
/// state ([`Live::age`]).
#[inline]
fn report_into(
    worker: &dyn Worker,
    loads: &LoadSnapshot,
    clock: &DecisionClock,
    row: &mut Candidate,
) -> bool {
    let Some((report, sampled_at)) = loads.report(worker.url()) else {
        return false;
    };
    let (load, sampled_at_micros) = stated_report(report, sampled_at, row.role, clock);
    row.load = Some(load);
    row.load_sampled_at_micros = sampled_at_micros;
    true
}

/// A worker's role as its record states it, by its registration: one
/// reading for a list's record and for the report a stage is told.
pub(super) fn stated_role(spec: &WorkerSpec) -> Role {
    Role::from(spec.worker_type)
}

/// A worker's rank as its record states it, by its registration, and the
/// first where it states none: one reading for a list's record and for the
/// report a stage is told.
pub(super) fn stated_rank(spec: &WorkerSpec) -> u16 {
    u16::try_from(spec.dp_rank.unwrap_or(0)).unwrap_or(u16::MAX)
}

/// How many ranks a worker object stands for, by its registration: the
/// width it states where it registers no rank, and one where it registers
/// a rank or states no width. One reading for a list's cells and for the
/// report a stage is told ([`whole`]).
pub(super) fn stated_ranks(spec: &WorkerSpec) -> usize {
    match (spec.dp_rank, spec.dp_size) {
        (None, Some(width)) => width.max(1),
        _ => 1,
    }
}

/// A worker's report as a record states it: the row of a worker of `role`,
/// converted at the report's own sample ([`load_row`]), and the sample's
/// instant on `clock`. One conversion for a list's record and for the
/// report a stage is told ([`Signal::Load`](crate::tracking::lease::Signal)).
#[inline]
pub(super) fn stated_report(
    report: &WorkerLoadResponse,
    sampled_at: Instant,
    role: Role,
    clock: &DecisionClock,
) -> (LoadRow, u64) {
    // The conversion is asked the age of a report at its own sample.
    let row = load_row(report, role, sampled_at, sampled_at, SignalSource::Poll);
    (row, clock.micros(sampled_at))
}

/// A report's named values as a record states them, written into the row
/// of `cells` at `position`, and whether a named value is read of the
/// report at all. They are made by the crate's one conversion
/// ([`signal_values`]) under `columns`, the document's, and written where
/// a decision may read them ([`SignalCells::write`]): under the signal's
/// own unit and in range, each sample instant on `clock`. A report that
/// is not the whole of its worker object states none ([`whole`]), and
/// nothing is written of it: `ranks` says how many ranks the object stands
/// for ([`stated_ranks`]), and is asked of a report of several rows alone.
/// One reading for a list's cells and for the report a stage is told.
fn write_signals(
    report: &WorkerLoadResponse,
    sampled_at: Instant,
    ranks: impl FnOnce() -> usize,
    columns: &[SignalColumn],
    clock: &DecisionClock,
    position: u32,
    cells: &mut SignalCells,
) -> bool {
    if !whole(report, ranks) {
        return false;
    }
    signal_values(report, columns, sampled_at, |column, unit, value, at| {
        cells.write(position, column, unit, value, clock.micros(at));
    });
    true
}

/// A report's named values as a stage is told them
/// ([`Signal::Load`](crate::tracking::lease::Signal)): one cell a column
/// of `columns`, as a list's cells state the report ([`write_signals`]),
/// and none for a report that is not the whole of its worker object,
/// which stands for `ranks` ranks ([`stated_ranks`]). They stand in
/// `cells`, which the caller keeps from one report to the next: behind the
/// call it holds this report's values and no other's, and its room is
/// kept.
pub(super) fn stated_signals<'c>(
    report: &WorkerLoadResponse,
    sampled_at: Instant,
    ranks: usize,
    columns: &[SignalColumn],
    clock: &DecisionClock,
    cells: &'c mut SignalCells,
) -> ReportSignals<'c> {
    cells.blank(1, columns.len());
    if !write_signals(report, sampled_at, || ranks, columns, clock, 0, cells) {
        return ReportSignals::NONE;
    }
    ReportSignals::new(cells.row(0).unwrap_or_default())
}

/// The records of one list with each worker's report written, as one load
/// snapshot holds the reports: the half of a record that stands for as
/// long as the snapshot does.
///
/// A load snapshot is published whole and never changed, so a report's
/// conversion is the same at every decision that holds the snapshot. A
/// list keeps the rows of one snapshot (`PoolRecords::reported`): one
/// decision that writes every record of the list and finds the list holds
/// none of its snapshot converts the list's reports, and leaves the rows
/// for the decisions behind it ([`PoolRecords::convert`]). So does the
/// draw that pays the last of the list's length ([`LookUps`]).
///
/// The rows are of one snapshot and one list, and are read by a decision
/// that holds that snapshot and that list and by no other
/// ([`Self::is_of`]). They are replaced whole, never changed, and they die
/// with their list. Nothing a worker object says is in them: that is read
/// at every decision ([`Live::object`]), and so is the report's age
/// ([`Live::age`]).
#[derive(Debug)]
struct Reported {
    /// The snapshot the reports were read from, named and not held. The
    /// name keeps the snapshot's place in memory from any later snapshot,
    /// so the rows are never taken for another snapshot's.
    snapshot: Weak<LoadSnapshot>,
    /// The instant of the decision that converted the rows, on the
    /// engine's clock. A decision that began no later holds a snapshot no
    /// newer than theirs, and leaves them in place.
    since_micros: u64,
    /// Each worker's record by position: its fixed half, and where the
    /// snapshot holds a report of the worker, the report with its sample's
    /// instant on the engine's clock. The report's age is at rest, and so
    /// is every value a worker object says.
    rows: Box<[Candidate]>,
    /// Whether every worker carries a report ([`PoolFacts::reported`]).
    all: bool,
    /// The list's signal cells as the snapshot's reports state them
    /// ([`PoolRecords::write_cells`]): one cell a worker a signal the
    /// document declares, by position. Empty under a document none of
    /// whose stages reads a signal.
    cells: SignalCells,
}

impl Reported {
    /// Whether the rows are those of `loads`.
    #[inline]
    fn is_of(&self, loads: &Arc<LoadSnapshot>) -> bool {
        ptr::eq(self.snapshot.as_ptr(), Arc::as_ptr(loads))
    }
}

/// The rows a list holds, as one decision read the slot: held for the
/// decision through the calling thread's own protection, with no lock and
/// no write another thread reads.
type Held = Guard<Option<Arc<Reported>>>;

/// One list of a gateway's worker objects as the engine reads it: what is
/// built once a membership, and the list's reports as one load snapshot
/// holds them.
///
/// It holds the half of each worker's record that no request changes, the
/// pool's engines and the pool's number. It holds no worker object and no
/// value a worker object says: a worker's health and its count of requests
/// in flight are read on the object at each decision ([`Live`]). A
/// worker's load report is read from the load snapshot the decision holds:
/// converted once for the list, by the first decision that writes every
/// record under that snapshot or by the draw that pays the last of the
/// list's length in look-ups, and read by the decisions behind it that
/// hold the same snapshot.
///
/// Made from one list and good for that list alone: the list a gateway's
/// registry publishes at a membership change is a new list, and takes new
/// records.
#[derive(Debug)]
pub struct PoolRecords {
    /// Each worker's record with its fixed fields written and every live
    /// field at rest, by position.
    fixed: Box<[Candidate]>,
    /// Whether every worker of the list is a regular one.
    regular: bool,
    /// Each worker's id, by position.
    ids: Box<[WorkerId]>,
    /// What each worker states for a request's hard demands, by position
    /// ([`FixedFacts::demands`]), and none where no worker of the list
    /// states one: such a list allocates no column.
    demand_facts: Option<Box<[DemandFacts]>>,
    /// The ids in their own order, each with its worker's position
    /// ([`Self::place_of`]). Empty where two rows of the list carry one
    /// id.
    places: Box<[(WorkerId, u32)]>,
    /// The positions of the pool's rows engine after engine, and each
    /// engine's end among them ([`Engines::new`]). Both empty where no
    /// engine of the pool holds several rows.
    engine_rows: Box<[u32]>,
    engine_ends: Box<[u32]>,
    /// The pool's number among the engine's pools ([`Engines::pool`]).
    number: u32,
    /// The generation of the document the records were read under: its
    /// lists name the engine kinds and the labels.
    generation: u64,
    /// The signals that document declares, in column order, where a stage
    /// of it reads one: what the list's signal cells are made under
    /// ([`Self::write_cells`]). None under a document none of whose stages
    /// reads a signal, and a conversion then makes no cells.
    signals: Arc<[SignalColumn]>,
    /// The list's rows with the reports of one load snapshot written
    /// ([`Reported`]), and none before a decision converted the list's
    /// reports ([`Self::convert`]).
    ///
    /// Read by one load of the slot through the calling thread's own
    /// protection: no lock, and no write another thread reads. Written by
    /// one atomic store of the whole value, by the one decision that holds
    /// the claim ([`Self::convert`]).
    reported: ArcSwapOption<Reported>,
    /// The claim to convert the list's reports, with what the list's draws
    /// paid toward it ([`LookUps`]): taken by one compare and exchange and
    /// never waited for, by the one decision that converts or counts, and
    /// given back when it is done. A decision that finds it taken converts
    /// nothing, counts nothing and waits for nothing: it writes its own
    /// records from its own snapshot.
    converting: Mutex<LookUps>,
    /// How many times the list's reports were converted: once a snapshot
    /// a decision wrote every record under, or the list's draws paid for.
    conversions: AtomicU64,
}

impl PoolRecords {
    /// The records of `workers`, under the document `spec` and on `clock`,
    /// whose count of pools this moves. Nothing is said of a worker beside
    /// its object ([`Self::with_fixed_facts`]).
    ///
    /// Called once a list. It reads each worker's registration and none of
    /// its live state.
    #[must_use]
    pub fn new(workers: &[Arc<dyn Worker>], spec: &AppliedSpec, clock: &DecisionClock) -> Self {
        Self::with_fixed_facts(workers, spec, clock, |_| FixedFacts::default())
    }

    /// [`Self::new`], with what the program says of each worker beside its
    /// object ([`FixedFacts`]): `facts_of` is asked once a worker, here, and
    /// nowhere else. Whether a worker takes images goes into its
    /// record, and so does the id its program states, in place of the
    /// fingerprint of its address ([`worker_id`]). What it states for a
    /// request's hard demands goes into a column beside the records
    /// ([`Self::meeting`]), which the first worker that states one makes:
    /// a list where none does allocates nothing for it.
    #[must_use]
    pub fn with_fixed_facts(
        workers: &[Arc<dyn Worker>],
        spec: &AppliedSpec,
        clock: &DecisionClock,
        mut facts_of: impl FnMut(&Arc<dyn Worker>) -> FixedFacts,
    ) -> Self {
        let lists = spec.lists();
        let mut demand_facts: Option<Box<[DemandFacts]>> = None;
        let mut fixed: Box<[Candidate]> = workers
            .iter()
            .zip(0u32..)
            .map(|(worker, position)| {
                let spec = &worker.metadata().spec;
                // The fingerprint of the registration key: the worker's
                // key, and its id where its program states none.
                let address = worker_id(worker.as_ref());
                let listed = if lists.is_empty() {
                    None
                } else {
                    let kind = if spec.runtime_type.is_specified() {
                        spec.runtime_type.as_str()
                    } else {
                        ""
                    };
                    let labels: BTreeMap<String, String> = spec
                        .labels
                        .iter()
                        .map(|(name, value)| (name.clone(), value.clone()))
                        .collect();
                    // A worker object states no zone.
                    Some(lists.positions(kind, None, &labels))
                };
                let facts = facts_of(worker);
                let demands = facts.demands;
                if demands != DemandFacts::default() {
                    let column = demand_facts
                        .get_or_insert_with(|| vec![DemandFacts::default(); workers.len()].into());
                    if let Some(entry) = column.get_mut(position as usize) {
                        *entry = demands;
                    }
                }
                Candidate {
                    position,
                    id: facts.id_or(address),
                    role: stated_role(spec),
                    // Values wider than a record's fields take the largest
                    // value they hold.
                    dp_rank: stated_rank(spec),
                    dp_size: u16::try_from(spec.dp_size.unwrap_or(1)).unwrap_or(u16::MAX),
                    wire_modes: WireModes::from(spec.connection_mode),
                    engine_kind: listed.map_or(0, |listed| listed.engine_kind),
                    labels: listed.map_or(0, |listed| listed.labels),
                    vision: facts.vision,
                    // No admission window is the record's to know: both
                    // bits read open, and a caller that holds a window
                    // writes its own.
                    has_capacity: true,
                    has_room: true,
                    // The low 32 bits of the registration key's
                    // fingerprint.
                    key: address.0 as u32,
                    ..Candidate::default()
                }
            })
            .collect();
        let ids: Box<[WorkerId]> = fixed.iter().map(|row| row.id).collect();
        let mut places: Vec<(WorkerId, u32)> = ids.iter().copied().zip(0u32..).collect();
        places.sort_unstable();
        // An id two rows carry names neither of them.
        if places
            .windows(2)
            .any(|pair| matches!(pair, [a, b] if a.0 == b.0))
        {
            places.clear();
        }
        let (engine_rows, engine_ends) = engines_of(workers, &mut fixed);
        // The first pool is number one: an epoch of zero says no
        // membership.
        let number = clock.pools.fetch_add(1, Ordering::Relaxed) + 1;
        Self {
            regular: fixed.iter().all(|row| row.role == Role::Single),
            fixed,
            ids,
            demand_facts,
            places: places.into(),
            engine_rows,
            engine_ends,
            number: number as u32,
            generation: spec.generation(),
            signals: if spec.needs().signal_cells {
                spec.shared_signal_columns()
            } else {
                Arc::default()
            },
            reported: ArcSwapOption::empty(),
            converting: Mutex::new(LookUps::default()),
            conversions: AtomicU64::new(0),
        }
    }

    /// How many workers the list holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fixed.len()
    }

    /// Whether the list holds no worker.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fixed.is_empty()
    }

    /// Whether every worker of the list is a regular one, read once when
    /// the records were made: a keyed decision is told it with no row read.
    pub(crate) fn regular(&self) -> bool {
        self.regular
    }

    /// Each worker's id, by its place in the list: the one its program
    /// stated ([`FixedFacts::id`]), or the fingerprint of its address
    /// ([`worker_id`]).
    #[must_use]
    pub fn ids(&self) -> &[WorkerId] {
        &self.ids
    }

    /// The place in the list of the worker whose id is `id`
    /// ([`Self::ids`]), and `None` where the list holds no such worker:
    /// what finds the one record an id names ([`Self::read`]) with no walk
    /// of the list.
    ///
    /// Where two rows of the list carry one id (one address listed twice,
    /// or two addresses of one fingerprint), no id of the list has a
    /// place: such an id names neither row.
    ///
    /// The ids are sorted once a list, when its records are made. A
    /// question is one binary search and allocates nothing.
    #[must_use]
    pub fn place_of(&self, id: WorkerId) -> Option<u32> {
        let at = self
            .places
            .binary_search_by_key(&id, |&(listed, _)| listed)
            .ok()?;
        self.places.get(at).map(|&(_, place)| place)
    }

    /// Whether an engine of the list holds several rows, as a data-parallel
    /// engine holds one for each of its ranks ([`Engines`]).
    #[must_use]
    pub(crate) fn has_ranks(&self) -> bool {
        !self.engine_ends.is_empty()
    }

    /// The workers of the engines a retry tried, written into `whole`:
    /// `tried` as its host named them, and behind them the id of every
    /// other row of the engine that holds a tried worker's row
    /// ([`Engines`]), each once. A retry that names one rank of a
    /// data-parallel engine so names them all: its ranks share the engine a
    /// failed attempt reached.
    ///
    /// An id the list does not hold stays as it was named and adds nothing:
    /// the list says nothing of its engine. Where two rows of the list
    /// carry one id, no id has a place ([`Self::place_of`]) and nothing is
    /// added.
    ///
    /// One binary search for each tried id, and nothing allocated once
    /// `whole` has grown: what a retry pays, and only over a list with
    /// such an engine ([`Self::has_ranks`]).
    pub(crate) fn tried_engines(&self, tried: &[WorkerId], whole: &mut Vec<WorkerId>) {
        whole.clear();
        whole.extend_from_slice(tried);
        self.more_tried_engines(tried, whole);
    }

    /// [`Self::tried_engines`] for a further list of the same decision, as
    /// a pair's second list is: adds to `whole` the id of every row of an
    /// engine of this list that holds a row `tried` names, where `whole`
    /// does not hold it yet, and takes nothing out of `whole`.
    pub(crate) fn more_tried_engines(&self, tried: &[WorkerId], whole: &mut Vec<WorkerId>) {
        for &id in tried {
            let Some(place) = self.place_of(id) else {
                continue;
            };
            let ranks = self.engine_of(place).iter();
            for &rank in ranks.filter_map(|&row| self.ids.get(row as usize)) {
                if !whole.contains(&rank) {
                    whole.push(rank);
                }
            }
        }
    }

    /// The rows of the engine that holds the row at `place`, that row
    /// among them, by the engine's place the row states
    /// ([`Candidate::engine`]), and none where no engine of the list holds
    /// several rows.
    fn engine_of(&self, place: u32) -> &[u32] {
        let Some(row) = self.fixed.get(place as usize) else {
            return &[];
        };
        let engines = Engines::new(self.number, &self.engine_rows, &self.engine_ends);
        engines.rows(row.engine as usize)
    }

    /// The fixed half of each worker's record, by its place in the list:
    /// its position, id, key, role, rank, wire, engine kind and labels,
    /// its engine's place among the list's engines, whether it takes
    /// images where the program said so, and every live field at rest.
    #[must_use]
    pub fn fixed(&self) -> &[Candidate] {
        &self.fixed
    }

    /// The places of the list whose worker meets `demands`, written into
    /// `kept`: of every place, in the list's order, or of `among` alone, in
    /// its order, the places the request is already held to. A worker is
    /// judged by its record's vision bit and by what the program said it
    /// states ([`Self::with_fixed_facts`]). A list no worker of which states
    /// anything keeps no column, and each of its workers states nothing.
    /// The demands are judged as they are handed: a caller that judges a
    /// prefill side's list hands them through [`Demands::held_by`].
    ///
    /// The answer is the last of the demands the workers left out failed
    /// first, in the order a host answers for them ([`UnmetDemand`]): the
    /// stage a list the demands leave nobody is shed under. `None` where
    /// every worker judged meets them.
    ///
    /// One pass over the half of the records no request changes: no worker
    /// object is read, and nothing is allocated once `kept` has grown to
    /// the list.
    pub fn meeting(
        &self,
        demands: Demands,
        among: Option<&[u32]>,
        kept: &mut Vec<u32>,
    ) -> Option<UnmetDemand> {
        kept.clear();
        let column = self.demand_facts.as_deref().unwrap_or_default();
        let mut unmet = None;
        let mut judge = |place: u32| {
            let Some(row) = self.fixed.get(place as usize) else {
                return;
            };
            let stated = column.get(place as usize).copied().unwrap_or_default();
            match demands.unmet_by(row.vision, stated) {
                None => kept.push(place),
                failed => unmet = unmet.max(failed),
            }
        };
        match among {
            None => (0..self.fixed.len() as u32).for_each(&mut judge),
            Some(places) => places.iter().copied().for_each(&mut judge),
        }
        unmet
    }

    /// The generation of the document the records were read under. Records
    /// read under another generation's lists are made again.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// How many times the list's reports were converted so far: once for
    /// each load snapshot a decision wrote every record under
    /// ([`Self::write`]) or the list's draws paid for. What a bench and a
    /// case count against the snapshots published.
    #[must_use]
    pub fn conversions(&self) -> u64 {
        self.conversions.load(Ordering::Relaxed)
    }

    /// The pool's facts beside its records: `reported` is whether every
    /// record written carries a report. They lend the list's own word where
    /// `live` holds it ([`Live::turning`]), written here, where the facts
    /// are made: a decision makes no second copy of them to add it.
    fn facts<'a>(&'a self, live: &Live<'a>, reported: bool) -> PoolFacts<'a> {
        PoolFacts {
            pool_size: self.fixed.len() as u32,
            membership_epoch: live.clock.membership_epoch(),
            ring_epoch: 0,
            regime: OverloadRegime::None,
            generation: self.generation,
            zones: &[],
            reach: 0,
            reported,
            engines: Engines::new(self.number, &self.engine_rows, &self.engine_ends),
            word: live.word,
        }
    }

    /// Every worker's record for one decision, written into `rows` in the
    /// list's order, and the pool's facts beside them. A record's position
    /// is the worker's place in `workers`, which is the list these records
    /// were made from.
    ///
    /// Read on the worker object at this call and kept nowhere ([`Live`]):
    /// a worker's status, its circuit breaker, its overload and liveness
    /// vetoes and its count of requests in flight.
    ///
    /// A worker's load report with its sample's instant is the one the
    /// decision's snapshot holds, and never another snapshot's. Where the
    /// list holds the rows of that snapshot, the report is copied from
    /// them. Where it does not, one decision converts the list's reports
    /// and leaves the rows for the decisions behind it, and every other
    /// decision looks up each worker's address in its own snapshot and
    /// converts the report into `rows`, waiting for nothing: the list's
    /// claim is taken by one compare and exchange, never waited for. A
    /// decision that began no later than the one that converted the list's
    /// rows leaves them in place. A worker the snapshot holds no report
    /// of carries none, and the pool facts say whether every worker
    /// carries one ([`PoolFacts::reported`]). The report's age is this
    /// decision's.
    ///
    /// Left for the caller to write where it holds them: the pair class
    /// (zero), the two admission bits (open) and the session key's pin.
    ///
    /// No lock is waited for here beyond what a read of a worker object takes.
    /// Nothing is allocated once `rows` has grown to the list. The one
    /// exception is the conversion: one allocation, by one decision, at
    /// most once a published snapshot a list.
    ///
    /// The facts borrow the records and what `live` borrows, the list's own
    /// word among it. The rows borrow their buffer alone.
    pub fn write<'a, 'f>(
        &'f self,
        workers: &[Arc<dyn Worker>],
        live: &Live<'f>,
        rows: &'a mut Vec<Candidate>,
    ) -> (&'a mut [Candidate], PoolFacts<'f>) {
        debug_assert_eq!(
            workers.len(),
            self.fixed.len(),
            "the records are those of the list they are written from"
        );
        rows.clear();
        let reported = match live.loads {
            Some(loads) => {
                let held = self.reported.load();
                let converted;
                let reported = match held.as_deref().filter(|reported| reported.is_of(loads)) {
                    Some(reported) => Some(reported),
                    None => {
                        // It looks every report up: it pays the list's length.
                        converted =
                            self.convert(workers, loads, live, &held, self.fixed.len() as u32);
                        converted.as_deref()
                    }
                };
                match reported {
                    Some(reported) => {
                        rows.extend(reported.rows.iter().zip(workers).map(|(reported, worker)| {
                            let mut row = *reported;
                            live.object(worker.as_ref(), &mut row);
                            live.age(&mut row);
                            row
                        }));
                        reported.all
                    }
                    // Another decision is converting, or this one holds a
                    // snapshot no newer than the list's rows: its own
                    // snapshot's reports, read worker by worker.
                    None => {
                        let mut all = true;
                        rows.extend(self.fixed.iter().zip(workers).map(|(fixed, worker)| {
                            let mut row = *fixed;
                            all &= live.write(worker.as_ref(), &mut row);
                            row
                        }));
                        all
                    }
                }
            }
            // No snapshot: no worker carries a report.
            None => {
                rows.extend(self.fixed.iter().zip(workers).map(|(fixed, worker)| {
                    let mut row = *fixed;
                    live.object(worker.as_ref(), &mut row);
                    row
                }));
                self.fixed.is_empty()
            }
        };
        (rows.as_mut_slice(), self.facts(live, reported))
    }

    /// The list's rows with the reports of `loads` written, where the
    /// calling decision is the one that converts them: it found that the
    /// list holds none of its snapshot (`held` is what it read of the
    /// slot). `None` where it is not the one, and writes its own records
    /// from its own snapshot.
    ///
    /// **A conversion is paid for.** `looked_up` is how many reports the
    /// calling decision looks up where the list holds no rows of its
    /// snapshot: all of them for a whole write, the few it asked about for
    /// a draw. They are counted under the claim, and the decision converts
    /// where the count has reached the list's length ([`LookUps`]).
    ///
    /// **One decision converts a snapshot.** The decision takes the
    /// list's claim first, by one compare and exchange. Another decision
    /// that meets the same snapshot meanwhile finds the claim taken,
    /// converts nothing, allocates nothing and waits for nothing. The
    /// claim is given back when the conversion ends, also where it
    /// unwinds. Under the claim the slot is read again: where the rows of
    /// this snapshot were left since `held` was read, they are taken as
    /// they are.
    ///
    /// A decision that began no later than the decision that converted
    /// the list's rows holds a snapshot no newer than theirs, and does not
    /// replace them.
    ///
    /// The conversion is one look-up of each worker's address in the
    /// snapshot, one conversion of each report, and one allocation of the
    /// rows, left in the slot by one atomic store of the whole value. It
    /// is the one allocation a decision over records makes once its
    /// thread's buffer has grown: at most once a published snapshot a
    /// list, by one decision.
    #[inline(never)]
    fn convert(
        &self,
        workers: &[Arc<dyn Worker>],
        loads: &Arc<LoadSnapshot>,
        live: &Live<'_>,
        held: &Held,
        looked_up: u32,
    ) -> Option<Arc<Reported>> {
        if held
            .as_deref()
            .is_some_and(|reported| live.now_micros <= reported.since_micros)
        {
            return None;
        }
        let mut claim = self.converting.try_lock()?;
        let again = self.reported.load();
        if let Some(reported) = again.as_ref().filter(|reported| reported.is_of(loads)) {
            return Some(Arc::clone(reported));
        }
        if !claim.paid(loads, looked_up, self.fixed.len()) {
            return None;
        }
        let mut all = true;
        let rows = self
            .fixed
            .iter()
            .zip(workers)
            .map(|(fixed, worker)| {
                let mut row = *fixed;
                all &= report_into(worker.as_ref(), loads, live.clock, &mut row);
                row
            })
            .collect();
        // The list's signal cells stand with its rows: one conversion a
        // snapshot makes both, under a document a stage of which reads one.
        let mut cells = SignalCells::default();
        if !self.signals.is_empty() {
            self.write_cells(workers, loads, live.clock, &mut cells);
        }
        let converted = Arc::new(Reported {
            snapshot: Arc::downgrade(loads),
            since_micros: live.now_micros,
            rows,
            all,
            cells,
        });
        self.reported.store(Some(Arc::clone(&converted)));
        self.conversions.fetch_add(1, Ordering::Relaxed);
        drop(claim);
        Some(converted)
    }

    /// One worker's record for one decision: the record [`Self::write`]
    /// writes at `position`, for a caller that reads a few workers of a
    /// pool and not the pool. `None` for a position past the list.
    ///
    /// The worker's report is read from the list's rows where the list
    /// holds those of the decision's snapshot, and by one look-up of the
    /// worker's address in the snapshot where it does not. A read of one
    /// record converts no other worker's report.
    #[must_use]
    pub fn read(
        &self,
        workers: &[Arc<dyn Worker>],
        live: &Live<'_>,
        position: u32,
    ) -> Option<Candidate> {
        let fixed = self.fixed.get(position as usize)?;
        workers.get(position as usize)?;
        let mut row = *fixed;
        Singly::new(self, workers, live).write(fixed, &mut row);
        Some(row)
    }

    /// Writes into `cells` the list's signal cells as the reports of
    /// `loads` state them: one cell a worker a signal the document
    /// declares, by the worker's place in the list, each sample instant on
    /// `clock`. A report's values are written as a record states them
    /// ([`write_signals`]): made by the crate's one conversion, under the
    /// signal's own unit, and in range. A worker the snapshot holds no
    /// report of, and a name its report does not state, keep a cell with
    /// no value. So does a worker whose report is not its whole
    /// ([`whole`]): the worker object says how many ranks it stands for
    /// ([`stated_ranks`]).
    ///
    /// The cells are what this snapshot's reports state and nothing more:
    /// no value of an earlier snapshot is kept.
    fn write_cells(
        &self,
        workers: &[Arc<dyn Worker>],
        loads: &LoadSnapshot,
        clock: &DecisionClock,
        cells: &mut SignalCells,
    ) {
        cells.blank(self.fixed.len(), self.signals.len());
        for (fixed, worker) in self.fixed.iter().zip(workers) {
            let Some((report, sampled_at)) = loads.report(worker.url()) else {
                continue;
            };
            let ranks = || stated_ranks(&worker.metadata().spec);
            let (columns, position) = (&self.signals, fixed.position);
            write_signals(report, sampled_at, ranks, columns, clock, position, cells);
        }
    }

    /// The list's signal cells for one decision, held for it: the cells of
    /// the decision's snapshot ([`Self::write_cells`]), and `None` under no
    /// snapshot, where every worker reads no value.
    ///
    /// For a decision whose profile reads a signal, under a document a
    /// stage of which does. The cells are the list's own where the list
    /// holds the rows of the decision's snapshot. Where it does not, this
    /// decision converts the list's reports, as one that writes every
    /// record does ([`Self::write`]), and the decisions behind it read
    /// what it left. A decision that is not the one to convert (another
    /// holds the list's claim, or this one began no later than the
    /// decision that converted the list's rows) writes the cells of its
    /// own snapshot into the calling thread's buffer and waits for
    /// nothing.
    // In line at its two callers, a single request's lend and a pair's: as
    // a call it costs a single request's decision 32 instructions, counted
    // with Callgrind.
    #[inline(always)]
    pub(crate) fn cells(&self, workers: &[Arc<dyn Worker>], live: &Live<'_>) -> Option<HeldCells> {
        let loads = live.loads?;
        let held = self.reported.load();
        if held
            .as_deref()
            .is_some_and(|reported| reported.is_of(loads))
        {
            return Some(HeldCells(Cells::Listed(held)));
        }
        // It reads every report: it pays the list's length.
        let paid = self.fixed.len() as u32;
        if let Some(converted) = self.convert(workers, loads, live, &held, paid) {
            return Some(HeldCells(Cells::Converted(converted)));
        }
        let mut own = CELLS.with(Cell::take);
        self.write_cells(workers, loads, live.clock, &mut own);
        Some(HeldCells(Cells::Own(own)))
    }
}

/// Whether `report` states the whole of its worker object, which stands
/// for `ranks()` ranks ([`stated_ranks`]): what a named value of the
/// worker is made of. A list's cells take no named value of any other
/// report ([`PoolRecords::write_cells`]), and a stage is told none
/// ([`stated_signals`]).
///
/// A report that counts more ranks than it lists is not the whole, as a
/// partial answer: the ranks it leaves out state nothing. A report of
/// several rows is the whole where it lists as many ranks as the object
/// stands for, and under no other count. A worker's pushed report counts
/// the rows it holds, so its own count does not say that a rank is still
/// to report: the object's registration does. An object that registers a
/// rank or states no width stands for one rank, so no report of several
/// rows is read of it.
///
/// A report of one row is read as the worker's whole, whatever the object
/// stands for, and `ranks` is not asked. So the first record of an object
/// of several ranks is read as the worker's own until a second rank's
/// record stands beside it.
///
/// Each rank's entry states its own sample instant where the load monitor
/// wrote it, and the conversion dates each rank's values by it
/// ([`signal_values`]). A pushed record replaces its own rank's entry and
/// dates that entry alone, so a rank that stopped reporting keeps its last
/// values under its own last sample, and a maximum age ends the worker's
/// value there. The worker's value is made of its ranks' as the signal
/// declares, under the oldest sample among them.
fn whole(report: &WorkerLoadResponse, ranks: impl FnOnce() -> usize) -> bool {
    let listed = report.loads.len();
    let counted = usize::try_from(report.dp_rank_count).unwrap_or(0);
    listed >= counted && (listed <= 1 || listed == ranks())
}

/// A list's signal cells as one decision holds them
/// ([`PoolRecords::cells`]): from where it is lent them to the end of what
/// reads its request's facts. The cells stand as long as this value does,
/// whatever becomes of the list's rows meanwhile.
#[derive(Debug)]
pub(crate) struct HeldCells(Cells);

/// Where the cells a decision holds stand.
#[derive(Debug)]
enum Cells {
    /// In the list's rows of the decision's snapshot, as the decision read
    /// the slot: held through the calling thread's own protection.
    Listed(Held),
    /// In the rows this decision converted.
    Converted(Arc<Reported>),
    /// The decision's own, in the calling thread's buffer, which takes it
    /// back at this value's end.
    Own(SignalCells),
}

impl HeldCells {
    /// The cells held: one cell a worker a signal the document declares,
    /// by the worker's place in the list.
    pub(crate) fn cells(&self) -> Option<&SignalCells> {
        match &self.0 {
            Cells::Listed(held) => held.as_deref().map(|reported| &reported.cells),
            Cells::Converted(converted) => Some(&converted.cells),
            Cells::Own(own) => Some(own),
        }
    }
}

impl Drop for HeldCells {
    fn drop(&mut self) {
        if let Cells::Own(own) = &mut self.0 {
            // The thread's buffer keeps its room for the next decision
            // that writes its own cells. A thread that is ending has none
            // to give it back to.
            let _ = CELLS.try_with(|cells| cells.set(mem::take(own)));
        }
    }
}

thread_local! {
    /// The calling thread's own signal cells, for a decision that reads a
    /// signal and is not the one to convert its list's reports
    /// ([`PoolRecords::cells`]). The decision takes them for its time and
    /// gives them back ([`HeldCells`]), so they keep their room after a
    /// thread's first such decision of a list's size. A decision a stage
    /// asked for from inside its own, on this thread, finds none and makes
    /// its own.
    static CELLS: Cell<SignalCells> = Cell::new(SignalCells::default());
}

/// What a list's draws paid toward the conversion of its reports under one
/// load snapshot: how many records they wrote by looking a worker's report
/// up, because the list held no rows of that snapshot. The list's claim
/// guards it (`PoolRecords::converting`).
///
/// A draw writes the few records it asks about, so the first draw under a
/// new snapshot must not convert the whole list: under a policy that reads
/// two records, one request would pay for every worker. But a look-up
/// costs several times what a converted row costs to copy. So the draws
/// over a list count their look-ups, and the draw that brings the count to
/// the list's length converts the list's reports
/// ([`PoolRecords::convert`]). The count held less than the list's length
/// before that draw, and the conversion then looks up every report.
/// Look-ups it never held (below) or lost to another snapshot come on top.
///
/// The count is written under the claim alone, which is never waited for:
/// a draw that finds the claim taken leaves its look-ups uncounted. Only a
/// decision that looked a report up writes it, so none writes it once the
/// list holds the snapshot's rows.
#[derive(Debug, Default)]
struct LookUps {
    /// The snapshot the look-ups were made in, named and not held, as a
    /// list's rows name theirs ([`Reported`]): no later snapshot stands in
    /// its place in memory, so none inherits the count.
    snapshot: Weak<LoadSnapshot>,
    /// The records written by a look-up under that snapshot.
    records: u32,
}

impl LookUps {
    /// Counts `records` more look-ups under `loads`, where a count of
    /// another snapshot starts again, and says whether the count has
    /// reached `list`, the list's length.
    fn paid(&mut self, loads: &Arc<LoadSnapshot>, records: u32, list: usize) -> bool {
        if !ptr::eq(self.snapshot.as_ptr(), Arc::as_ptr(loads)) {
            *self = Self {
                snapshot: Arc::downgrade(loads),
                records: 0,
            };
        }
        self.records = self.records.saturating_add(records);
        self.records as usize >= list
    }
}

/// One decision's writes of single records of a list: a draw's, and a
/// read of one worker.
///
/// It holds the list's rows for the decision where they are of the
/// decision's snapshot ([`Reported`]), read once. Where the list holds
/// none, each record is written by a look-up of its worker's report, and a
/// draw pays those toward the list's conversion at its end
/// ([`Self::paid`]).
struct Singly<'a> {
    pool: &'a PoolRecords,
    workers: &'a [Arc<dyn Worker>],
    live: &'a Live<'a>,
    /// The list's rows, where they are of the decision's snapshot.
    held: Option<Held>,
    /// The records this decision wrote without the list's rows.
    looked_up: Cell<u32>,
}

impl<'a> Singly<'a> {
    fn new(pool: &'a PoolRecords, workers: &'a [Arc<dyn Worker>], live: &'a Live<'a>) -> Self {
        let held = live.loads.and_then(|loads| {
            let held = pool.reported.load();
            held.as_deref()
                .is_some_and(|reported| reported.is_of(loads))
                .then_some(held)
        });
        Self {
            pool,
            workers,
            live,
            held,
            looked_up: Cell::new(0),
        }
    }

    /// Pays the decision's look-ups toward the conversion of its list's
    /// reports ([`LookUps`]): the draw that pays the last of the list's
    /// length converts them, and the draws behind it copy the rows.
    fn paid(&self) {
        let looked_up = self.looked_up.get();
        if let (Some(loads), true) = (self.live.loads, looked_up > 0) {
            let held = self.pool.reported.load();
            let _ = self
                .pool
                .convert(self.workers, loads, self.live, &held, looked_up);
        }
    }

    /// Writes into `row` the record of the worker `fixed` is the fixed
    /// half of: from the list's rows of the decision's snapshot where the
    /// list holds them, and by a look-up of the worker's own report where
    /// it does not. What the worker object says is read here either way.
    #[inline]
    fn write(&self, fixed: &Candidate, row: &mut Candidate) {
        let at = fixed.position as usize;
        let Some(worker) = self.workers.get(at) else {
            *row = *fixed;
            return;
        };
        let reported = self
            .held
            .as_deref()
            .and_then(|held| held.as_deref())
            .and_then(|reported| reported.rows.get(at));
        match reported {
            Some(reported) => {
                *row = *reported;
                self.live.object(worker.as_ref(), row);
                self.live.age(row);
            }
            None => {
                *row = *fixed;
                self.looked_up.set(self.looked_up.get() + 1);
                self.live.write(worker.as_ref(), row);
            }
        }
    }
}

/// The engines of the pool `workers` list, as [`Engines`] takes them: the
/// positions engine after engine, the ranks of one data-parallel engine
/// together in their ranks' order, and each engine's end among them. An
/// engine is the address its ranks share, or a worker's own. Both lists are
/// empty where no engine of the pool holds two rows: nothing is listed, and
/// a decision reads rows alone.
///
/// Each record of `fixed` is written its engine's place among the pool's
/// engines here ([`Candidate::engine`]), in the order each engine's first
/// row stands in the list, which is the order the engines are listed in:
/// on every pool, also on one that lists none, where each row states a
/// place of its own.
fn engines_of(workers: &[Arc<dyn Worker>], fixed: &mut [Candidate]) -> (Box<[u32]>, Box<[u32]>) {
    let mut engine_of: FxHashMap<&str, usize> = FxHashMap::default();
    let mut engines: Vec<Vec<u32>> = Vec::new();
    for ((worker, row), position) in workers.iter().zip(fixed.iter_mut()).zip(0u32..) {
        let spec = &worker.metadata().spec;
        let address = spec.dp_base_url.as_deref().unwrap_or(&spec.url);
        let engine = *engine_of.entry(address).or_insert(engines.len());
        row.engine = engine as u32;
        match engines.get_mut(engine) {
            Some(rows) => rows.push(position),
            None => engines.push(vec![position]),
        }
    }
    if engines.iter().all(|rows| rows.len() < 2) {
        return (Box::default(), Box::default());
    }
    let mut rows = Vec::with_capacity(workers.len());
    let mut ends = Vec::with_capacity(engines.len());
    for mut engine in engines {
        engine.sort_by_key(|&position| fixed.get(position as usize).map(|row| row.dp_rank));
        rows.extend(engine);
        ends.push(rows.len() as u32);
    }
    (rows.into(), ends.into())
}

/// What one thread keeps for its decisions over records: the records and
/// the engine's scratch.
#[derive(Debug, Default)]
struct Buffers {
    rows: Vec<Candidate>,
    scratch: Scratch,
}

thread_local! {
    /// The calling thread's own buffers, so a decision allocates nothing
    /// after a thread's first decisions of a pool's size.
    static BUFFERS: Rc<RefCell<Buffers>> = Rc::new(RefCell::new(Buffers::default()));
}

/// One decision of a single request over a gateway's own data: the records
/// of `workers`, the list `pool` was made from, read as `live` says and put
/// to the engine's whole decision ([`Engine::decide`]), which runs the
/// binding's rules and its profile's stages as it does over any records.
///
/// The records written are the ones the decision reads. Where the
/// request's profile draws first and its picker is one of the crate's own,
/// the picker draws over the list and a worker's record is written when
/// the draw asks about the worker: two workers for a pick of two, whatever
/// the pool's size. The rows the request names join them. The draws over a
/// list convert its reports once they looked up as many as the list holds,
/// and copy the rows from there. Every other
/// decision is handed every worker's record ([`PoolRecords::write`]), and
/// so is a draw that kept no worker. Both give the pick the whole write
/// gives: a stage that answers row by row reads no other row.
///
/// `facts` are the request's, as its host made them. The decision's instant
/// is written into them here ([`RequestFacts::now_micros`]): the one `live`
/// was made at ([`Live::now_micros`]).
///
/// A target of the decision names its worker by position, the worker's
/// place in `workers`, and by id ([`PoolRecords::ids`]).
///
/// Where `live` carries the list's own word, as the engine's decision of a
/// single request over one list hands it, the pool's facts lend it to the
/// decision ([`PoolFacts::word`]): a picker that takes turns counts a
/// draw's turn in it.
///
/// A standing veto `cx` states beside the records
/// ([`Constraints::stalled`]) is written into them where the request's
/// profile shows its rows to a plugin: the stalled bit, and the breaker
/// bit where the veto holds. The engine declines such a profile over a
/// mask beside its rows.
///
/// The records and the engine's scratch are the calling thread's own.
pub fn decide(
    engine: &Engine,
    pool: &PoolRecords,
    workers: &[Arc<dyn Worker>],
    live: &Live<'_>,
    facts: &mut RequestFacts<'_>,
    cx: &Constraints<'_>,
) -> Decision {
    facts.now_micros = live.now_micros;
    let facts = &*facts;
    let decide = |buffers: &mut Buffers| {
        let Buffers { rows, scratch } = buffers;
        // No record says yet whether it carries a report: a draw reads that
        // fact of no pool.
        let unsaid = pool.facts(live, false);
        if let Some(drawn) = engine.drawn_for(facts, &unsaid, pool.len()) {
            if rows.len() < pool.len() {
                rows.resize(pool.len(), Candidate::default());
            }
            let rows = &mut rows[..pool.len()];
            let singly = Singly::new(pool, workers, live);
            let mut write = |fixed: &Candidate, row: &mut Candidate| singly.write(fixed, row);
            let decision = engine.decide_drawn(
                drawn,
                facts,
                &unsaid,
                &pool.fixed,
                rows,
                &mut write,
                cx,
                scratch,
            );
            singly.paid();
            return decision;
        }
        let (rows, pool_facts) = pool.write(workers, live, rows);
        let written;
        let cx = if cx.stalled.is_empty() {
            cx
        } else {
            written = vetoes::written(engine, facts.model, facts.phase, rows, cx);
            &written
        };
        engine.decide(facts, &pool_facts, rows, cx, scratch)
    };
    with_buffers(decide)
}

/// Runs one decision over the calling thread's own buffers: the one borrow
/// of them, for a decision without a shadow ([`decide`]) and for one
/// followed by its binding's shadow profile ([`with_shadow::decide`]).
#[inline]
fn with_buffers<T>(decide: impl FnOnce(&mut Buffers) -> T) -> T {
    // The key's own call hands out the pointer alone: what a decision
    // answers through that call is copied on its way out of it.
    let buffers = BUFFERS.with(Rc::clone);
    let Ok(mut buffers) = buffers.try_borrow_mut() else {
        // A stage asked for a decision from inside its own, on this
        // thread: that decision takes buffers of its own.
        return decide(&mut Buffers::default());
    };
    decide(&mut buffers)
}

#[cfg(test)]
mod tests;
