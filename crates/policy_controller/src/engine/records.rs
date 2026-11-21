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
//! - **Once a decision.** A record's live half is read where the gateway
//!   keeps it, at the decision's instant ([`Live`]): what the worker object
//!   says (ready, breaker, overload, stall, requests in flight) and the
//!   worker's report in the load snapshot with its sample instant.
//!   [`PoolRecords::write`] writes every worker's record into a buffer of
//!   its caller, and [`PoolRecords::read`] one worker's. Neither takes a
//!   lock of its own, and a write allocates nothing once the buffer has
//!   grown to the pool.
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
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use rustc_hash::FxHashMap;
use smg_worker::{identity::basic::Worker, load::state::LoadSnapshot, worker::WorkerStatus};

use super::{
    decision::Decision,
    framework::{Engine, Scratch},
    view::{
        fingerprint, Candidate, Constraints, Engines, Health, OverloadRegime, PoolFacts,
        RequestFacts, Role, SignalSource, WireModes, WorkerId, LOAD_REPORT_CLOCK_HISTORY_MS,
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
    WorkerId(fingerprint(worker.url().as_bytes()))
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
    /// worker object says ([`Self::object`]), then the worker's report from
    /// the load snapshot, by one look-up of its address.
    ///
    /// The report's age is the decision's instant less the sample's, both
    /// on the engine's clock: the two values a stage is handed
    /// ([`RequestFacts::now_micros`], [`Candidate::load_sampled_at_micros`]).
    #[inline]
    fn write(&self, worker: &dyn Worker, row: &mut Candidate) -> bool {
        self.object(worker, row);
        let Some((report, sampled_at)) = self.loads.and_then(|loads| loads.report(worker.url()))
        else {
            return false;
        };
        let sampled_micros = self.clock.micros(sampled_at);
        // The conversion is asked the age of a report at its own sample:
        // the decision's age is written beside it, from the clock's values.
        let mut load = load_row(report, row.role, sampled_at, sampled_at, SignalSource::Poll);
        load.report_age_ms = age_ms(self.now_micros, sampled_micros);
        row.load = Some(load);
        row.load_sampled_at_micros = sampled_micros;
        true
    }

    /// Writes what `worker`'s object says into `row`: one read of the
    /// object for its status, its circuit breaker, its overload and
    /// liveness vetoes and its count of requests in flight, and the two
    /// counts of sessions where a stage of the document reads them. A
    /// worker whose breaker refuses execution carries the stalled bit
    /// beside the breaker's, as a veto that holds is written: no rule
    /// lifts it.
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
}

/// Milliseconds from `at_micros` to `now_micros` on one clock, saturating at
/// `u32::MAX` and at zero; never wrapping.
#[inline]
fn age_ms(now_micros: u64, at_micros: u64) -> u32 {
    u32::try_from(now_micros.saturating_sub(at_micros) / 1_000).unwrap_or(u32::MAX)
}

/// One list of a gateway's worker objects as the engine reads it: what is
/// built once a membership.
///
/// It holds the half of each worker's record that no request changes, the
/// pool's engines and the pool's number. It holds no worker object and no
/// live value: a worker's health, its count of requests in flight and its
/// load report are read where the gateway keeps them, at each decision
/// ([`Live`]).
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
        let ids = fixed.iter().map(|row| row.id).collect();
        let (engine_rows, engine_ends) = engines_of(workers, &fixed);
        // The first pool is number one: an epoch of zero says no
        // membership.
        let number = clock.pools.fetch_add(1, Ordering::Relaxed) + 1;
        Self {
            fixed,
            ids,
            engine_rows,
            engine_ends,
            number: number as u32,
            generation: spec.generation(),
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
    /// Read at this call and kept nowhere ([`Live`]): a worker's status,
    /// its circuit breaker, its overload and liveness vetoes and its count
    /// of requests in flight, from the worker object, and its load report
    /// with the report's sample instant, from the load snapshot. A worker
    /// the snapshot holds no report of carries none, and the pool facts say
    /// whether every worker carries one ([`PoolFacts::reported`]).
    ///
    /// Left for the caller to write where it holds them: the pair class
    /// (zero), the two admission bits (open) and the session key's pin.
    ///
    /// No lock is taken here beyond what a read of a worker object takes,
    /// and nothing is allocated once `rows` has grown to the list.
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
        rows.extend_from_slice(&self.fixed);
        let mut unreported = 0u32;
        for (row, worker) in rows.iter_mut().zip(workers) {
            unreported += u32::from(!live.write(worker.as_ref(), row));
        }
        (rows.as_mut_slice(), self.facts(live.clock, unreported == 0))
    }

    /// One worker's record for one decision: the record [`Self::write`]
    /// writes at `position`, read the same way, for a caller that reads a
    /// few workers of a pool and not the pool. `None` for a position past
    /// the list.
    #[must_use]
    pub fn read(
        &self,
        workers: &[Arc<dyn Worker>],
        live: &Live<'_>,
        position: u32,
    ) -> Option<Candidate> {
        let mut row = *self.fixed.get(position as usize)?;
        live.write(workers.get(position as usize)?.as_ref(), &mut row);
        Some(row)
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
            let mut write = |fixed: &Candidate, row: &mut Candidate| {
                *row = *fixed;
                if let Some(worker) = workers.get(fixed.position as usize) {
                    live.write(worker.as_ref(), row);
                }
            };
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
