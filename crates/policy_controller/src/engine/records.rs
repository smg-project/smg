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
//!   value of a worker is in it.
//! - **Once a load snapshot.** A snapshot is published whole and never
//!   changed, so a worker's report reads the same at every decision that
//!   holds the snapshot. One decision that writes every record of a list
//!   under a snapshot converts the list's reports, and the list keeps
//!   those rows for the decisions behind it that hold the same snapshot
//!   ([`Reported`]). A decision that holds another snapshot never reads
//!   them: it reads its own snapshot's reports.
//! - **Once a decision.** What a worker object says is read where the
//!   gateway keeps it, at the decision's instant ([`Live`]): ready,
//!   breaker, overload, stall, requests in flight. So is the age of the
//!   worker's report. [`PoolRecords::write`] writes every worker's record
//!   into a buffer of its caller, and [`PoolRecords::read`] one worker's.
//!   Neither takes a lock. A write allocates nothing once the buffer has
//!   grown to the pool, but for the one decision a snapshot that converts
//!   the list's reports.
//!
//! [`decide`] is one single decision over such records, in the calling
//! thread's own buffer. It writes the records the decision reads: every
//! worker's where the profile's stages read the pool, and where the
//! profile's picker draws first, the drawn workers' alone.
//!
//! One clock dates both sides ([`DecisionClock`]): a decision's instant and
//! a report's sample instant are microseconds since one origin, so a load
//! stage's credit of dispatches is released by a report sampled after them
//! and by no other.

use std::{
    cell::RefCell,
    collections::BTreeMap,
    ptr,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Weak,
    },
    time::{Duration, Instant},
};

use arc_swap::{ArcSwapOption, Guard};
use rustc_hash::FxHashMap;
use smg_worker::{identity::basic::Worker, load::state::LoadSnapshot, worker::WorkerStatus};

use super::{
    decision::Decision,
    framework::{Engine, Scratch},
    view::{
        Candidate, Constraints, Engines, Health, OverloadRegime, PoolFacts, RequestFacts, Role,
        SignalSource, WireModes, WorkerId, LOAD_REPORT_CLOCK_HISTORY_MS,
    },
};
use crate::{document::spec::AppliedSpec, load::load_row};

/// A worker's id in the engine's own state (a load stage's credit of
/// dispatches, a hash ring, session pins, a stage's bookings): the
/// fingerprint of its registration key, its address.
///
/// The same for every worker object registered at one address, so it is
/// the same at every attempt of a request and across a membership change,
/// and a removal names the id every decision knew the worker by.
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
    /// The request's session key, where a stage of the document reads a
    /// worker's count of the key's requests in flight.
    counted_key: Option<&'a str>,
    /// Whether a stage of the document reads a worker's count of session
    /// keys with a request in flight.
    sessions: bool,
    /// The places of the decision's list the request may take, where its
    /// host narrowed it ([`Self::keeping`]), ascending. `None` for a
    /// decision over the whole list.
    kept: Option<&'a [usize]>,
}

impl<'a> Live<'a> {
    /// The reads of one decision at `now` on `clock`, under `engine`'s
    /// document: the workers' reports from `loads`, the load monitor's
    /// snapshot as the caller holds it for the decision, and `session_key`,
    /// the text of the request's session key where it carries one.
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
            kept: None,
        }
    }

    /// The same reads for a decision its host narrowed to a part of the
    /// list: `places` are the places of the list the request may take,
    /// ascending, as a placement pass names a narrowed list, and `None`
    /// for the whole list. The decision's record of a worker at any other
    /// place is written as that of a worker whose breaker refuses
    /// execution: the stalled bit beside the breaker's, a veto that holds,
    /// so no stage keeps the row and no rule lifts it.
    ///
    /// The records, the ring and the number a decision reads stay the
    /// list's own: nothing is made for a part. The hold is written on the
    /// decision's own rows and never on the list's converted rows, which
    /// are every request's.
    #[must_use]
    pub fn keeping(self, places: Option<&'a [usize]>) -> Self {
        Self {
            kept: places,
            ..self
        }
    }

    /// Writes the host's hold into `row`, the decision's own record of the
    /// worker at `position`, where the host narrowed the request to places
    /// that place is not among ([`Self::keeping`]). Nothing where nothing
    /// narrows the request: one test.
    #[inline]
    fn hold(&self, position: u32, row: &mut Candidate) {
        let out = |kept: &[usize]| kept.binary_search(&(position as usize)).is_err();
        if self.kept.is_some_and(out) {
            Self::held(row);
        }
    }

    /// Marks `row` as one a hold of the host stands on.
    #[inline]
    fn held(row: &mut Candidate) {
        row.health.breaker_open = true;
        row.health.stalled = true;
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
    // The conversion is asked the age of a report at its own sample.
    row.load = Some(load_row(
        report,
        row.role,
        sampled_at,
        sampled_at,
        SignalSource::Poll,
    ));
    row.load_sampled_at_micros = clock.micros(sampled_at);
    true
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
/// for the decisions behind it ([`PoolRecords::convert`]).
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
/// record under that snapshot, and read by the decisions behind it that
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
    /// Each worker's id, by position.
    ids: Box<[WorkerId]>,
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
    /// The list's rows with the reports of one load snapshot written
    /// ([`Reported`]), and none before a decision wrote every record under
    /// a snapshot.
    ///
    /// Read by one load of the slot through the calling thread's own
    /// protection: no lock, and no write another thread reads. Written by
    /// one atomic store of the whole value, by the one decision that holds
    /// the claim ([`Self::convert`]).
    reported: ArcSwapOption<Reported>,
    /// The claim to convert the list's reports: set by one compare and
    /// exchange, by the one decision that converts, and cleared when it is
    /// done. A decision that finds it set converts nothing and waits for
    /// nothing: it writes its own records from its own snapshot.
    converting: AtomicBool,
    /// How many times the list's reports were converted: once a snapshot
    /// a decision wrote every record under.
    conversions: AtomicU64,
}

impl PoolRecords {
    /// The records of `workers`, under the document `spec` and on `clock`,
    /// whose count of pools this moves.
    ///
    /// Called once a list. It reads each worker's registration and none of
    /// its live state.
    #[must_use]
    pub fn new(workers: &[Arc<dyn Worker>], spec: &AppliedSpec, clock: &DecisionClock) -> Self {
        let lists = spec.lists();
        let fixed: Box<[Candidate]> = workers
            .iter()
            .zip(0u32..)
            .map(|(worker, position)| {
                let spec = &worker.metadata().spec;
                let id = worker_id(worker.as_ref());
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
                Candidate {
                    position,
                    id,
                    role: Role::from(spec.worker_type),
                    // Values wider than a record's fields take the largest
                    // value they hold.
                    dp_rank: u16::try_from(spec.dp_rank.unwrap_or(0)).unwrap_or(u16::MAX),
                    dp_size: u16::try_from(spec.dp_size.unwrap_or(1)).unwrap_or(u16::MAX),
                    wire_modes: WireModes::from(spec.connection_mode),
                    engine_kind: listed.map_or(0, |listed| listed.engine_kind),
                    labels: listed.map_or(0, |listed| listed.labels),
                    // No admission window is the record's to know: both
                    // bits read open, and a caller that holds a window
                    // writes its own.
                    has_capacity: true,
                    has_room: true,
                    // The low 32 bits of the registration key's
                    // fingerprint.
                    key: id.0 as u32,
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
        let (engine_rows, engine_ends) = engines_of(workers, &fixed);
        // The first pool is number one: an epoch of zero says no
        // membership.
        let number = clock.pools.fetch_add(1, Ordering::Relaxed) + 1;
        Self {
            fixed,
            ids,
            places: places.into(),
            engine_rows,
            engine_ends,
            number: number as u32,
            generation: spec.generation(),
            reported: ArcSwapOption::empty(),
            converting: AtomicBool::new(false),
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

    /// Each worker's id ([`worker_id`]), by its place in the list.
    #[must_use]
    pub fn ids(&self) -> &[WorkerId] {
        &self.ids
    }

    /// The place in the list of the worker whose id is `id`
    /// ([`worker_id`]), and `None` where the list holds no such worker:
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

    /// The list's rows by id: each worker's id with its place in the list,
    /// ids ascending and each once. Empty where two rows of the list carry
    /// one id ([`Self::has_distinct_ids`]): no id of such a list names a row.
    #[must_use]
    pub fn by_id(&self) -> &[(WorkerId, u32)] {
        &self.places
    }

    /// Whether every row of the list has an id of its own: an id then names
    /// one row ([`Self::place_of`]).
    #[must_use]
    pub fn has_distinct_ids(&self) -> bool {
        self.places.len() == self.fixed.len()
    }

    /// The fixed half of each worker's record, by its place in the list:
    /// its position, id, key, role, rank, wire, engine kind and labels, and
    /// every live field at rest.
    #[must_use]
    pub fn fixed(&self) -> &[Candidate] {
        &self.fixed
    }

    /// The generation of the document the records were read under. Records
    /// read under another generation's lists are made again.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// How many times the list's reports were converted so far: once for
    /// each load snapshot a decision wrote every record under
    /// ([`Self::write`]). What a bench and a case count against the
    /// snapshots published.
    #[must_use]
    pub fn conversions(&self) -> u64 {
        self.conversions.load(Ordering::Relaxed)
    }

    /// The pool's facts beside its records: `reported` is whether every
    /// record written carries a report.
    fn facts(&self, clock: &DecisionClock, reported: bool) -> PoolFacts<'_> {
        PoolFacts {
            pool_size: self.fixed.len() as u32,
            membership_epoch: clock.membership_epoch(),
            ring_epoch: 0,
            regime: OverloadRegime::None,
            generation: self.generation,
            zones: &[],
            reach: 0,
            reported,
            engines: Engines::new(self.number, &self.engine_rows, &self.engine_ends),
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
    /// No lock is taken here beyond what a read of a worker object takes.
    /// Nothing is allocated once `rows` has grown to the list. The one
    /// exception is the conversion: one allocation, by one decision, at
    /// most once a published snapshot a list.
    pub fn write<'a>(
        &'a self,
        workers: &[Arc<dyn Worker>],
        live: &Live<'_>,
        rows: &'a mut Vec<Candidate>,
    ) -> (&'a mut [Candidate], PoolFacts<'a>) {
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
                        converted = self.convert(workers, loads, live, &held);
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
        // Where the decision's host narrowed the request, every place it
        // may not take stands held on the decision's own rows: one walk of
        // the rows beside the places kept, and none where nothing narrows.
        if let Some(kept) = live.kept {
            let mut kept = kept.iter().peekable();
            for (place, row) in rows.iter_mut().enumerate() {
                if kept.next_if(|&&at| at == place).is_none() {
                    Live::held(row);
                }
            }
        }
        (rows.as_mut_slice(), self.facts(live.clock, reported))
    }

    /// The list's rows with the reports of `loads` written, where the
    /// calling decision is the one that converts them: it found that the
    /// list holds none of its snapshot (`held` is what it read of the
    /// slot). `None` where it is not the one, and writes its own records
    /// from its own snapshot.
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
    ) -> Option<Arc<Reported>> {
        if held
            .as_deref()
            .is_some_and(|reported| live.now_micros <= reported.since_micros)
        {
            return None;
        }
        let claim = Claim::take(&self.converting)?;
        let again = self.reported.load();
        if let Some(reported) = again.as_ref().filter(|reported| reported.is_of(loads)) {
            return Some(Arc::clone(reported));
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
        let converted = Arc::new(Reported {
            snapshot: Arc::downgrade(loads),
            since_micros: live.now_micros,
            rows,
            all,
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
}

/// A list's claim to convert its reports, held by the one decision that
/// converts and given back when it is dropped: at the conversion's end,
/// and where the conversion unwinds.
struct Claim<'a>(&'a AtomicBool);

impl<'a> Claim<'a> {
    /// The claim, where no other decision holds it: one compare and
    /// exchange, never a wait.
    #[inline]
    fn take(word: &'a AtomicBool) -> Option<Self> {
        word.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .ok()
            .map(|_| Self(word))
    }
}

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// One decision's writes of single records of a list: a draw's, and a
/// read of one worker.
///
/// It holds the list's rows for the decision where they are of the
/// decision's snapshot ([`Reported`]), read once.
struct Singly<'a> {
    workers: &'a [Arc<dyn Worker>],
    live: &'a Live<'a>,
    /// The list's rows, where they are of the decision's snapshot.
    held: Option<Held>,
}

impl<'a> Singly<'a> {
    fn new(pool: &PoolRecords, workers: &'a [Arc<dyn Worker>], live: &'a Live<'a>) -> Self {
        let held = live.loads.and_then(|loads| {
            let held = pool.reported.load();
            held.as_deref()
                .is_some_and(|reported| reported.is_of(loads))
                .then_some(held)
        });
        Self {
            workers,
            live,
            held,
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
                self.live.write(worker.as_ref(), row);
            }
        }
        self.live.hold(fixed.position, row);
    }
}

/// The engines of the pool `workers` list, as [`Engines`] takes them: the
/// positions engine after engine, the ranks of one data-parallel engine
/// together in their ranks' order, and each engine's end among them. An
/// engine is the address its ranks share, or a worker's own. Both lists are
/// empty where no engine of the pool holds two rows: nothing is listed, and
/// a decision reads rows alone.
fn engines_of(workers: &[Arc<dyn Worker>], fixed: &[Candidate]) -> (Box<[u32]>, Box<[u32]>) {
    let mut engine_of: FxHashMap<&str, usize> = FxHashMap::default();
    let mut engines: Vec<Vec<u32>> = Vec::new();
    for (worker, position) in workers.iter().zip(0u32..) {
        let spec = &worker.metadata().spec;
        let address = spec.dp_base_url.as_deref().unwrap_or(&spec.url);
        let engine = *engine_of.entry(address).or_insert(engines.len());
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
    static BUFFERS: RefCell<Buffers> = RefCell::new(Buffers::default());
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
/// the pool's size. The rows the request names join them. Every other
/// decision is handed every worker's record ([`PoolRecords::write`]), and
/// so is a draw that kept no worker. Both give the pick the whole write
/// gives: a stage that answers row by row reads no other row.
///
/// `facts` are the request's, as its host made them. The decision's instant
/// is written into them here ([`RequestFacts::now_micros`]): the one `live`
/// was made at ([`Live::now_micros`]).
///
/// A target of the decision names its worker by position, the worker's
/// place in `workers`, and by id ([`worker_id`]).
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
        // No record says yet whether it carries a report: a draw reads the
        // fact of no pool.
        let unsaid = pool.facts(live.clock, false);
        if let Some(drawn) = engine.drawn_for(facts, &unsaid, pool.len()) {
            if rows.len() < pool.len() {
                rows.resize(pool.len(), Candidate::default());
            }
            let rows = &mut rows[..pool.len()];
            let singly = Singly::new(pool, workers, live);
            let mut write = |fixed: &Candidate, row: &mut Candidate| singly.write(fixed, row);
            return engine.decide_drawn(
                drawn,
                facts,
                &unsaid,
                &pool.fixed,
                rows,
                &mut write,
                cx,
                scratch,
            );
        }
        let (rows, pool_facts) = pool.write(workers, live, rows);
        engine.decide(facts, &pool_facts, rows, cx, scratch)
    };
    BUFFERS.with(|buffers| match buffers.try_borrow_mut() {
        Ok(mut buffers) => decide(&mut buffers),
        // A stage asked for a decision from inside its own, on this
        // thread: that decision takes buffers of its own.
        Err(_) => decide(&mut Buffers::default()),
    })
}

#[cfg(test)]
mod tests;
