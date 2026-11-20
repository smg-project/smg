//! The policy engine in a gateway's own process.
//!
//! The gateway keeps its workers where it keeps them with the classic
//! policies: its registry publishes one shared list of worker objects a
//! model at each membership change, and a worker's health and its requests
//! in flight live on the worker object. This front decides over that data
//! in place:
//!
//! - **Once a membership.** [`PolicyEngine::pool_members`] takes a pool's
//!   shared list and writes one row a worker with the worker's fixed facts
//!   (its id, role, wire, rank and what the document's lists read of it).
//!   The gateway keeps the value beside the list it was made from, for as
//!   long as that list stands. Nothing of it is built at a request.
//! - **Once a request.** [`PolicyEngine::place`] borrows those rows and the
//!   list. No row is copied or written. A stage that asks for a row's
//!   health bits or its count is answered by the worker object at the row's
//!   position ([`LiveRows`]): the worker is read once, by the one call that
//!   answers all of them, and the decision keeps that read, so no worker is
//!   read twice and no decision sees two values of one worker. Only the
//!   workers a decision asks about are read. The decision's scratch is the
//!   calling thread's own: no lock is taken and nothing is allocated after
//!   a thread's first decisions.
//!
//! The engine keeps no worker list, no health and no count of requests in
//! flight. What it keeps is its stages' own: a rotation, a session's pins.
//!
//! A decision names one worker. A host that retries asks again and names
//! the workers it tried ([`Request::tried`]).

use std::{
    cell::RefCell,
    collections::BTreeMap,
    hash::Hasher,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use rustc_hash::FxHasher;
use smg_worker::identity::basic::{ConnectionMode, Worker, WorkerType};

use super::{
    decision::{Decision, ShedReason},
    framework::{Engine, Scratch},
    view::{
        fingerprint, Candidate, Constraints, Health, LiveRow, LiveRows, ModelFingerprint, Phase,
        PoolFacts, RequestFacts, Role, WireModes, WorkerId,
    },
};
use crate::document::spec::{AppliedSpec, Listed};

/// The phases a pool's rows may be asked in.
const PHASES: [Phase; 4] = [Phase::Single, Phase::Prefill, Phase::Decode, Phase::Encode];

const fn phase_at(phase: Phase) -> usize {
    match phase {
        Phase::Single => 0,
        Phase::Prefill => 1,
        Phase::Decode => 2,
        Phase::Encode => 3,
    }
}

/// The engine of one policy document, for the gateway that hosts it.
///
/// Built once, at the gateway's start, from the document in force. It holds
/// the document's stages and their state, and no worker.
#[derive(Debug)]
pub struct PolicyEngine {
    spec: AppliedSpec,
    /// The pools read so far: each takes the next number as its membership
    /// epoch, so a stage that keeps state by the pool keeps one a list.
    epochs: AtomicU64,
    /// The engine's clock starts here: a decision that reads one is told
    /// the microseconds since.
    origin: Instant,
}

/// One pool of one model as the engine reads it: the gateway's shared list
/// of worker objects and one row a worker, in the list's order.
///
/// Made by [`PolicyEngine::pool_members`] from a list the gateway's
/// registry published, and kept by the gateway beside that list. It holds
/// the list itself, not a copy: a worker's live facts are read on the
/// list's objects.
///
/// The rows hold nothing of the model's id. What the engine reads of the id
/// (its fingerprint, and what its profiles ask of a host) is kept for the
/// spelling the pool was first asked under, with that spelling: a registry
/// answers an alias and its canonical id with one list, a binding matches
/// the id as spelled, and a request under another spelling is read by its
/// own.
#[derive(Debug)]
pub struct PoolMembers {
    workers: Arc<[Arc<dyn Worker>]>,
    rows: Box<[Candidate]>,
    /// The pool's membership epoch: this list's own, and no other list's
    /// under this engine. A rotation is kept by it, so two pools of one
    /// model never turn one rotation between them.
    epoch: u64,
    /// The model id the readings below were taken for, as spelled.
    spelling: Box<str>,
    model: ModelFingerprint,
    /// By phase: a decision reads no clock ([`Engine::clock_free`]).
    clock_free: [bool; PHASES.len()],
    /// By phase: the profile shows its rows to a plugin, which reads a
    /// row's facts as the row's own fields.
    shown: [bool; PHASES.len()],
}

/// What the engine reads of a model's id for one phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Spelled {
    model: ModelFingerprint,
    clock_free: bool,
    shown: bool,
}

impl Spelled {
    fn of(engine: &Engine, model: &str, phase: Phase) -> Self {
        Self {
            model: ModelFingerprint::of(model),
            clock_free: engine.clock_free(model, phase),
            shown: engine.reads_for(model, phase).1,
        }
    }
}

impl PoolMembers {
    /// The workers of the pool, in the rows' order: the list the value was
    /// made from.
    #[must_use]
    pub fn workers(&self) -> &Arc<[Arc<dyn Worker>]> {
        &self.workers
    }

    /// How many workers the pool holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the pool holds no worker.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The id of the worker at `position` of the list: what a retry names
    /// as tried ([`Request::tried`]). `None` past the list.
    #[must_use]
    pub fn id_at(&self, position: usize) -> Option<WorkerId> {
        self.rows.get(position).map(|row| row.id)
    }

    /// What the engine reads of `request`'s model id: the kept reading for
    /// the kept spelling, and the id's own reading for any other. The
    /// compare is of the two spellings' bytes: no id is hashed and no
    /// binding walked for the spelling the pool is asked under.
    fn spelled(&self, engine: &Engine, request: &Request<'_>) -> Spelled {
        if *self.spelling == *request.model {
            let at = phase_at(request.phase);
            Spelled {
                model: self.model,
                clock_free: self.clock_free[at],
                shown: self.shown[at],
            }
        } else {
            Spelled::of(engine, request.model, request.phase)
        }
    }
}

/// What a router states of one attempt of a request.
///
/// Built over [`Request::of`]: `Request { tried, ..Request::of(model) }`.
#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    /// The model the request is for, as the router resolved it.
    pub model: &'a str,
    /// A regular attempt is single. A pair is asked at its prefill.
    pub phase: Phase,
    /// The workers the request's failed attempts were sent to: the
    /// decision names none of them while another stands.
    pub tried: &'a [WorkerId],
    /// The workers this request may not be sent to, by position in the
    /// pool's list, in rising order, each once: a demand only the router
    /// can test (a wire a retry is held to, media a worker must take). The
    /// decision names none of them, under any rule. Empty for a request
    /// every worker of the pool may take, which is decided over the pool's
    /// own rows; any other is decided over rows written for it, a read of
    /// each worker left.
    pub refused: &'a [u32],
}

impl<'a> Request<'a> {
    /// A single request for `model` that tried no worker yet and may be
    /// sent to every worker of its pool.
    #[must_use]
    pub const fn of(model: &'a str) -> Self {
        Self {
            model,
            phase: Phase::Single,
            tried: &[],
            refused: &[],
        }
    }
}

/// The worker a decision placed a request on.
#[derive(Debug)]
pub struct Placement {
    worker: Arc<dyn Worker>,
    id: WorkerId,
    profile: Option<u16>,
}

impl Placement {
    /// The worker object the request goes to.
    #[must_use]
    pub fn worker(&self) -> &Arc<dyn Worker> {
        &self.worker
    }

    /// The worker's id in the engine's rows: what a retry names as tried
    /// ([`Request::tried`]).
    #[must_use]
    pub fn id(&self) -> WorkerId {
        self.id
    }

    /// The profile that decided, by its position in the document.
    #[must_use]
    pub fn profile(&self) -> Option<u16> {
        self.profile
    }

    /// The worker object, for the router that takes its load guard and
    /// sends the request.
    #[must_use]
    pub fn into_worker(self) -> Arc<dyn Worker> {
        self.worker
    }
}

/// Why a decision placed the request on no worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unplaced {
    /// The engine's reason.
    pub reason: ShedReason,
    /// The stage that left no worker.
    pub emptied_by: &'static str,
    /// The pacing hint, in seconds; `0` for none.
    pub retry_after_secs: u16,
    /// The binding lets the host hold the request for a worker to come.
    pub wait_allowed: bool,
    /// The profile that decided, by its position in the document; `None`
    /// where the request was declined ahead of every profile.
    pub profile: Option<u16>,
    /// The decision was a ticket for a queue. This host holds none: the
    /// request is not placed.
    pub queued: bool,
}

impl PolicyEngine {
    /// The engine of `spec`, the document in force.
    #[must_use]
    pub fn new(spec: AppliedSpec) -> Self {
        Self {
            spec,
            epochs: AtomicU64::new(0),
            origin: Instant::now(),
        }
    }

    /// The document the engine decides under.
    #[must_use]
    pub fn spec(&self) -> &AppliedSpec {
        &self.spec
    }

    /// The engine's reading of one pool of `model`: `pool` is the shared
    /// list the gateway's registry published for it.
    ///
    /// Called once a list, by the gateway, which keeps the value for as
    /// long as the list stands and hands it to no other engine. It costs
    /// one pass over the list.
    #[must_use]
    pub fn pool_members(&self, model: &str, pool: &Arc<[Arc<dyn Worker>]>) -> PoolMembers {
        let engine = self.spec.engine();
        let rows = pool
            .iter()
            .zip(0u32..)
            .map(|(worker, position)| self.row_of(position, worker.as_ref()))
            .collect();
        let readings = PHASES.map(|phase| Spelled::of(engine, model, phase));
        PoolMembers {
            workers: Arc::clone(pool),
            rows,
            epoch: self.epochs.fetch_add(1, Ordering::Relaxed) + 1,
            spelling: model.into(),
            model: ModelFingerprint::of(model),
            clock_free: readings.map(|spelled| spelled.clock_free),
            shown: readings.map(|spelled| spelled.shown),
        }
    }

    /// One decision for one attempt of `request` over `members`, the value
    /// this engine made of the pool ([`Self::pool_members`]).
    ///
    /// # Errors
    ///
    /// The engine's refusal where no worker of the pool takes the request.
    pub fn place(
        &self,
        request: &Request<'_>,
        members: &PoolMembers,
    ) -> Result<Placement, Unplaced> {
        THREAD.with(|thread| match thread.try_borrow_mut() {
            Ok(mut thread) => self.place_on(request, members, &mut thread),
            // A stage of the decision asked the engine again on this thread:
            // the inner decision takes a scratch of its own.
            Err(_) => self.place_on(request, members, &mut Thread::new()),
        })
    }

    fn place_on(
        &self,
        request: &Request<'_>,
        members: &PoolMembers,
        thread: &mut Thread,
    ) -> Result<Placement, Unplaced> {
        let engine = self.spec.engine();
        let spelled = members.spelled(engine, request);
        let seed = thread.seed();
        // A plugin reads a row's facts as the row's own fields, and a
        // request some workers may not take is decided over the others
        // alone: both are handed rows written for the decision.
        let writes = spelled.shown || !request.refused.is_empty();
        let stamp = if writes {
            0
        } else {
            thread.stamp(members.rows.len())
        };
        let Thread {
            scratch,
            kept,
            written,
            positions,
            ..
        } = thread;
        let hosted = Hosted {
            workers: &members.workers,
            stamp,
            kept,
        };
        let (candidates, live, reach): (&[Candidate], Option<&dyn LiveRows>, u64) = if writes {
            write(members, request.refused, written, positions);
            (written, None, reach_of(request.refused))
        } else {
            (&members.rows, Some(&hosted), 0)
        };
        let pool = PoolFacts {
            pool_size: u32::try_from(members.rows.len()).unwrap_or(u32::MAX),
            membership_epoch: members.epoch,
            generation: engine.generation(),
            reach,
            ..PoolFacts::default()
        };
        let facts = RequestFacts {
            model: request.model,
            model_fingerprint: spelled.model,
            phase: request.phase,
            tried: request.tried,
            seed,
            now_micros: if spelled.clock_free {
                0
            } else {
                u64::try_from(self.origin.elapsed().as_micros()).unwrap_or(u64::MAX)
            },
            live,
            ..RequestFacts::default()
        };
        let decision = engine.decide(&facts, &pool, candidates, &Constraints::default(), scratch);
        let profile = decision.trace().made_under();
        match decision {
            Decision::Pick(pick) => pick
                .targets
                .as_slice()
                .first()
                .and_then(|target| {
                    // A written row stands at its place among the rows
                    // written, and names its place in the list beside it.
                    let position = if writes {
                        *positions.get(target.position as usize)?
                    } else {
                        target.position
                    };
                    let worker = members.workers.get(position as usize)?;
                    Some(Placement {
                        worker: Arc::clone(worker),
                        id: target.id,
                        profile,
                    })
                })
                .ok_or(Unplaced {
                    reason: ShedReason::PolicyDeclined,
                    emptied_by: "targets",
                    retry_after_secs: 0,
                    wait_allowed: false,
                    profile,
                    queued: false,
                }),
            Decision::Shed(shed) => Err(Unplaced {
                reason: shed.reason,
                emptied_by: shed.emptied_by,
                retry_after_secs: shed.retry_after_secs,
                wait_allowed: shed.wait_allowed,
                profile,
                queued: false,
            }),
            // No built-in issues a ticket, and this host holds no queue.
            Decision::Queue(_) => Err(Unplaced {
                reason: ShedReason::FlowLimit,
                emptied_by: "queue",
                retry_after_secs: 0,
                wait_allowed: false,
                profile,
                queued: true,
            }),
        }
    }

    /// The row of the worker at `position` of its pool: the facts of its
    /// registration, which stand for as long as the worker is in the list.
    /// The live fields are left as a row holds none: a decision reads them
    /// on the worker.
    fn row_of(&self, position: u32, worker: &dyn Worker) -> Candidate {
        let registered = &worker.metadata().spec;
        // The worker's address names it, as the gateway's own policies name
        // a worker by it: a worker replaced under its address keeps its id.
        let key = fingerprint(worker.url().as_bytes());
        let lists = self.spec.lists();
        let listed = if lists.is_empty() {
            Listed::default()
        } else {
            let engine = if registered.runtime_type.is_specified() {
                registered.runtime_type.as_str()
            } else {
                ""
            };
            let labels: BTreeMap<String, String> = registered
                .labels
                .iter()
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect();
            lists.listed(engine, None, &labels)
        };
        Candidate {
            position,
            id: WorkerId(key),
            role: match worker.worker_type() {
                WorkerType::Regular => Role::Single,
                WorkerType::Prefill => Role::Prefill,
                WorkerType::Decode => Role::Decode,
                WorkerType::Encode => Role::Encode,
            },
            // Values wider than a row's fields take the largest value they hold.
            dp_rank: u16::try_from(registered.dp_rank.unwrap_or(0)).unwrap_or(u16::MAX),
            dp_size: u16::try_from(registered.dp_size.unwrap_or(1)).unwrap_or(u16::MAX),
            wire_modes: match worker.connection_mode() {
                ConnectionMode::Http => WireModes::HTTP,
                ConnectionMode::Grpc => WireModes::GRPC,
                ConnectionMode::Zmq => WireModes::ZMQ,
            },
            // The gateway gates a worker's capacity where it takes the
            // worker's load guard: both bits read open here.
            has_capacity: true,
            has_room: true,
            // The low 32 bits of the address's fingerprint.
            key: key as u32,
            engine_kind: listed.engine_kind,
            zone: listed.zone,
            labels: listed.labels,
            ..Candidate::default()
        }
    }
}

/// A worker's live facts, read on the worker object: one read answers its
/// status, its breaker, its liveness veto, its overload verdict and its
/// requests in flight.
fn read(worker: &dyn Worker) -> LiveRow {
    let state = worker.routing_state();
    LiveRow {
        health: Health {
            healthy: state.healthy,
            // An open breaker refuses the worker, and no rule lifts it.
            breaker_open: !state.can_execute,
            stalled: state.stalled,
            // A worker being drained is not ready: the status says it.
            draining: false,
            overloaded: state.overloaded,
        },
        in_flight: u32::try_from(state.load).unwrap_or(u32::MAX),
    }
}

/// Writes into `written` the rows of the workers of `members` that
/// `refused` does not name, each with its worker's live facts and its place
/// among the rows written, and into `positions` each written row's place in
/// the list.
fn write(
    members: &PoolMembers,
    refused: &[u32],
    written: &mut Vec<Candidate>,
    positions: &mut Vec<u32>,
) {
    debug_assert!(
        refused.is_sorted_by(|a, b| a < b),
        "the rows a request may not take are named in rising order, each once"
    );
    written.clear();
    positions.clear();
    let mut refused = refused.iter().copied().peekable();
    let rows = members.workers.iter().zip(&*members.rows);
    for ((worker, row), position) in rows.zip(0u32..) {
        if refused.next_if_eq(&position).is_some() {
            continue;
        }
        let live = read(worker.as_ref());
        written.push(Candidate {
            position: u32::try_from(written.len()).unwrap_or(u32::MAX),
            health: live.health,
            in_flight: live.in_flight,
            ..*row
        });
        positions.push(position);
    }
}

/// Which of a pool's lists a request's written rows are
/// ([`PoolFacts::reach`]): the rows it may not take, hashed. Two requests
/// that may take the same workers turn one rotation, and the pool's whole
/// list, which is `0`, another.
fn reach_of(refused: &[u32]) -> u64 {
    if refused.is_empty() {
        return 0;
    }
    let mut reach = FxHasher::default();
    for &position in refused {
        reach.write_u32(position);
    }
    reach.finish() | 1
}

/// The reader of one decision's live rows: the pool's worker objects, and
/// the calling thread's cells, where a position's first read is kept for
/// the decision.
#[derive(Debug)]
struct Hosted<'a> {
    workers: &'a [Arc<dyn Worker>],
    /// The decision's mark, in a cell's high half: a cell that holds
    /// another decision's mark holds nothing for this one.
    stamp: u64,
    kept: &'a [AtomicU64],
}

/// The bits of a kept cell below the decision's mark: five of the row's
/// health, the rest its requests in flight.
const HEALTH_BITS: u32 = 5;
const COUNT_MAX: u64 = (1 << (32 - HEALTH_BITS)) - 1;
const LOW_HALF: u64 = u32::MAX as u64;

const fn packed(row: LiveRow) -> u64 {
    let health = row.health;
    let bits = health.healthy as u64
        | (health.breaker_open as u64) << 1
        | (health.stalled as u64) << 2
        | (health.draining as u64) << 3
        | (health.overloaded as u64) << 4;
    let count = if row.in_flight as u64 > COUNT_MAX {
        COUNT_MAX
    } else {
        row.in_flight as u64
    };
    bits | (count << HEALTH_BITS)
}

const fn unpacked(cell: u64) -> LiveRow {
    LiveRow {
        health: Health {
            healthy: cell & 1 != 0,
            breaker_open: cell & (1 << 1) != 0,
            stalled: cell & (1 << 2) != 0,
            draining: cell & (1 << 3) != 0,
            overloaded: cell & (1 << 4) != 0,
        },
        in_flight: ((cell & LOW_HALF) >> HEALTH_BITS) as u32,
    }
}

impl LiveRows for Hosted<'_> {
    #[inline]
    fn read(&self, position: u32) -> LiveRow {
        let at = position as usize;
        let (Some(worker), Some(cell)) = (self.workers.get(at), self.kept.get(at)) else {
            // No row of the pool stands there: nobody asks, and a row that
            // is not there is not routable.
            return LiveRow::default();
        };
        let held = cell.load(Ordering::Relaxed);
        if held & !LOW_HALF == self.stamp {
            return unpacked(held);
        }
        let cell_of = packed(read(worker.as_ref()));
        cell.store(self.stamp | cell_of, Ordering::Relaxed);
        // What the cell keeps is what the decision reads: a count past the
        // cell's width reads as the width's most at every read alike.
        unpacked(cell_of)
    }
}

/// What a deciding thread keeps between its decisions.
#[derive(Debug)]
struct Thread {
    scratch: Scratch,
    /// One cell a row of the largest pool this thread decided over: the
    /// live facts a decision read, under its mark.
    kept: Vec<AtomicU64>,
    /// The last decision's mark, from 1.
    mark: u32,
    /// The rows written for a decision that is handed written rows.
    written: Vec<Candidate>,
    /// Each written row's place in its pool's list.
    positions: Vec<u32>,
    /// The state of the thread's seeds.
    seeds: u64,
}

thread_local! {
    static THREAD: RefCell<Thread> = RefCell::new(Thread::new());
}

/// The threads that decided so far: each starts its seeds from its number.
static THREADS: AtomicU64 = AtomicU64::new(0);

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

impl Thread {
    fn new() -> Self {
        let number = THREADS.fetch_add(1, Ordering::Relaxed);
        // The wall clock at the thread's first decision tells two runs of
        // the process apart.
        let started = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos() as u64);
        Self {
            scratch: Scratch::default(),
            kept: Vec::new(),
            mark: 0,
            written: Vec::new(),
            positions: Vec::new(),
            seeds: started ^ number.wrapping_mul(GOLDEN),
        }
    }

    /// The next decision's seed: the one randomness its stages get.
    fn seed(&mut self) -> u64 {
        self.seeds = self.seeds.wrapping_add(GOLDEN);
        let mut z = self.seeds;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// The next decision's mark over a pool of `rows`, in a cell's high
    /// half, with a cell for each row. The cells are sized at a thread's
    /// first decision over a pool that large, and cleared when the marks
    /// come round.
    fn stamp(&mut self, rows: usize) -> u64 {
        if self.kept.len() < rows {
            self.kept.resize_with(rows, AtomicU64::default);
        }
        self.mark = self.mark.wrapping_add(1);
        if self.mark == 0 {
            for cell in &self.kept {
                cell.store(0, Ordering::Relaxed);
            }
            self.mark = 1;
        }
        u64::from(self.mark) << 32
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use openai_protocol::worker::WorkerStatus;
    use smg_worker::test_support::ready_worker;

    use super::*;
    use crate::document::spec::PolicySpec;

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

    fn place(engine: &PolicyEngine, members: &PoolMembers, request: &Request<'_>) -> usize {
        let placed = engine.place(request, members).expect("a worker is placed");
        members
            .workers()
            .iter()
            .position(|worker| Arc::ptr_eq(worker, placed.worker()))
            .expect("the placed worker is one of the pool")
    }

    /// The members hold the registry's list itself, and a row a worker in
    /// the list's order, each under an id made of the worker's address.
    #[test]
    fn the_members_hold_the_shared_list_and_a_row_a_worker() {
        let engine = preset("round_robin");
        let list = pool(5);
        let members = engine.pool_members(MODEL, &list);
        assert!(Arc::ptr_eq(members.workers(), &list));
        assert_eq!(members.len(), 5);
        for (at, worker) in list.iter().enumerate() {
            let id = WorkerId(fingerprint(worker.url().as_bytes()));
            assert_eq!(members.id_at(at), Some(id));
        }
        assert_eq!(members.id_at(5), None);
    }

    /// Round robin over the shared list turns one strict rotation at every
    /// pool size, the pools under the engine's own floor too: each worker
    /// is named once a turn of the pool.
    #[test]
    fn round_robin_names_each_worker_once_a_turn() {
        for workers in [1, 2, 3, 8, 16, 64] {
            let engine = preset("round_robin");
            let members = engine.pool_members(MODEL, &pool(workers));
            let request = Request::of(MODEL);
            let first = place(&engine, &members, &request);
            for step in 1..2 * workers {
                let at = place(&engine, &members, &request);
                assert_eq!(at, (first + step) % workers, "{workers} workers");
            }
        }
    }

    /// A worker that is not ready and one over its overload thresholds are
    /// read on the worker object at the decision, and named for no request.
    #[test]
    fn a_decision_reads_a_workers_vetoes_on_the_worker() {
        let engine = preset("round_robin");
        let list = pool(4);
        let members = engine.pool_members(MODEL, &list);
        let request = Request::of(MODEL);
        list[1].set_status(WorkerStatus::NotReady);
        list[2].set_overloaded(true);
        for _ in 0..16 {
            let at = place(&engine, &members, &request);
            assert!(at == 0 || at == 3, "worker {at} is vetoed");
        }
        // The rows were made before the vetoes and hold none of them: the
        // veto lifted is read at the next decision, as it was set.
        list[1].set_status(WorkerStatus::Ready);
        list[2].set_overloaded(false);
        let named: BTreeSet<usize> = (0..8).map(|_| place(&engine, &members, &request)).collect();
        assert_eq!(named.len(), 4);
    }

    /// Two pools of one model turn a rotation each: asks that alternate
    /// between them name every worker of both.
    #[test]
    fn two_pools_of_one_model_turn_a_rotation_each() {
        let engine = preset("round_robin");
        let (first, second) = (pool(4), pool(4));
        let pools = [
            engine.pool_members(MODEL, &first),
            engine.pool_members(MODEL, &second),
        ];
        let request = Request::of(MODEL);
        let mut named = [[0u32; 4]; 2];
        for _ in 0..8 {
            for (pool, named) in pools.iter().zip(&mut named) {
                named[place(&engine, pool, &request)] += 1;
            }
        }
        assert_eq!(named, [[2; 4]; 2]);
    }

    /// The document binds one spelling of a model. A pool first asked under
    /// either spelling answers a request under each by that spelling's own
    /// binding: the first spelling to arrive fixes nothing.
    #[test]
    fn a_pool_first_asked_under_one_spelling_reads_another_by_its_own() {
        let mut document = PolicySpec::preset("round_robin").expect("a built-in preset");
        document.generation = 1;
        for binding in &mut document.bindings {
            binding.matcher.model = "canonical".to_owned();
        }
        let engine = engine_of(&document);
        let list = pool(4);
        for first in ["canonical", "alias"] {
            let members = engine.pool_members(first, &list);
            assert!(engine.place(&Request::of("canonical"), &members).is_ok());
            let unbound = engine
                .place(&Request::of("alias"), &members)
                .expect_err("no binding names the alias");
            assert_eq!(
                (unbound.reason, unbound.emptied_by, unbound.profile),
                (ShedReason::PolicyDeclined, "binding", None),
                "members first asked as {first}"
            );
        }
    }

    /// What the engine reads of a model's id is kept for the spelling the
    /// pool was first asked under, and read anew for another spelling.
    #[test]
    fn the_kept_reading_is_the_kept_spellings_alone() {
        let engine = preset("round_robin");
        let members = engine.pool_members("kept", &pool(2));
        let inner = engine.spec().engine();
        for spelling in ["kept", "other"] {
            let request = Request::of(spelling);
            assert_eq!(
                members.spelled(inner, &request),
                Spelled::of(inner, spelling, Phase::Single),
                "{spelling}"
            );
        }
        assert_ne!(
            members.spelled(inner, &Request::of("kept")).model,
            members.spelled(inner, &Request::of("other")).model
        );
    }

    /// A decision reads a worker once and keeps the read: a second ask of
    /// the same row within the decision answers the first read, whatever
    /// the worker says by then, and the next decision reads the worker
    /// again.
    #[test]
    fn a_decision_keeps_its_first_read_of_a_worker() {
        let list = pool(3);
        let mut thread = Thread::new();
        let stamp = thread.stamp(list.len());
        let reader = Hosted {
            workers: &list,
            stamp,
            kept: &thread.kept,
        };
        let first = reader.read(1);
        assert!(first.health.healthy && first.in_flight == 0);
        list[1].increment_load();
        list[1].set_overloaded(true);
        assert_eq!(reader.read(1), first);
        // Another row of the same decision is its own read.
        list[2].increment_load();
        assert_eq!(reader.read(2).in_flight, 1);
        let stamp = thread.stamp(list.len());
        let next = Hosted {
            workers: &list,
            stamp,
            kept: &thread.kept,
        };
        let again = next.read(1);
        assert!(again.health.overloaded && again.in_flight == 1);
        // A place past the pool is no worker's.
        assert_eq!(next.read(9), LiveRow::default());
    }

    /// A cell keeps every health bit and the count beside the mark, and a
    /// count past the cell's width reads as the width's most.
    #[test]
    fn a_kept_cell_holds_the_bits_and_the_count() {
        for bits in 0u32..32 {
            let row = LiveRow {
                health: Health {
                    healthy: bits & 1 != 0,
                    breaker_open: bits & 2 != 0,
                    stalled: bits & 4 != 0,
                    draining: bits & 8 != 0,
                    overloaded: bits & 16 != 0,
                },
                in_flight: 7 + bits,
            };
            assert_eq!(unpacked(packed(row)), row);
            assert_eq!(packed(row) & !LOW_HALF, 0);
        }
        let most = LiveRow {
            in_flight: u32::MAX,
            ..LiveRow::default()
        };
        assert_eq!(u64::from(unpacked(packed(most)).in_flight), COUNT_MAX);
    }

    /// The marks come round after their last value: the cells are cleared,
    /// so a read kept under a mark of the round before answers no decision.
    #[test]
    fn the_marks_come_round_over_cleared_cells() {
        let list = pool(1);
        let mut thread = Thread::new();
        thread.mark = u32::MAX - 1;
        let last = thread.stamp(1);
        assert_eq!(last, u64::from(u32::MAX) << 32);
        let reader = Hosted {
            workers: &list,
            stamp: last,
            kept: &thread.kept,
        };
        let _ = reader.read(0);
        assert_ne!(thread.kept[0].load(Ordering::Relaxed), 0);
        let first = thread.stamp(1);
        assert_eq!(first, 1 << 32);
        assert_eq!(thread.kept[0].load(Ordering::Relaxed), 0);
    }

    /// A request that may not be sent to some workers is decided over the
    /// others alone: none of the refused is named, and the rotation names
    /// each of the others in turn.
    #[test]
    fn a_request_is_never_placed_on_a_worker_it_may_not_take() {
        let engine = preset("round_robin");
        let list = pool(20);
        let members = engine.pool_members(MODEL, &list);
        let refused: Vec<u32> = (0..20).filter(|at| at % 4 != 1).collect();
        let request = Request {
            refused: &refused,
            ..Request::of(MODEL)
        };
        let mut named = [0u32; 20];
        for _ in 0..10 {
            named[place(&engine, &members, &request)] += 1;
        }
        for (at, count) in named.iter().enumerate() {
            let expected = if at % 4 == 1 { 2 } else { 0 };
            assert_eq!(*count, expected, "worker {at}");
        }
        // Every worker refused: nobody is placed.
        let all: Vec<u32> = (0..20).collect();
        let nobody = Request {
            refused: &all,
            ..Request::of(MODEL)
        };
        assert!(engine.place(&nobody, &members).is_err());
    }

    /// A retry names the workers it tried, and is placed on another while
    /// one stands.
    #[test]
    fn a_retry_is_placed_on_a_worker_it_did_not_try() {
        let engine = preset("round_robin");
        let list = pool(3);
        let members = engine.pool_members(MODEL, &list);
        let tried = [
            members.id_at(0).expect("a row"),
            members.id_at(2).expect("a row"),
        ];
        let request = Request {
            tried: &tried,
            ..Request::of(MODEL)
        };
        for _ in 0..6 {
            assert_eq!(place(&engine, &members, &request), 1);
        }
    }

    /// The live reader and written rows decide alike: over a fleet with
    /// vetoed workers, two engines of one document, one asked through the
    /// reader and one over rows written with the same facts, name the same
    /// worker first for the same seeds.
    #[test]
    fn the_live_reader_and_written_rows_name_the_same_worker() {
        for name in ["round_robin", "random", "power_of_two"] {
            let (live, written) = (preset(name), preset(name));
            let list = pool(48);
            for (at, worker) in list.iter().enumerate() {
                match at % 7 {
                    2 => worker.set_status(WorkerStatus::NotReady),
                    5 => {
                        worker.set_overloaded(true);
                    }
                    _ => {}
                }
                for _ in 0..at % 5 {
                    worker.increment_load();
                }
            }
            let members = live.pool_members(MODEL, &list);
            let other = written.pool_members(MODEL, &list);
            let rows: Vec<Candidate> = list
                .iter()
                .zip(&*other.rows)
                .map(|(worker, row)| {
                    let live = read(worker.as_ref());
                    Candidate {
                        health: live.health,
                        in_flight: live.in_flight,
                        ..*row
                    }
                })
                .collect();
            let mut thread = Thread::new();
            let mut scratch = Scratch::default();
            for seed in 1..=200u64 {
                let facts = RequestFacts {
                    model: MODEL,
                    model_fingerprint: ModelFingerprint::of(MODEL),
                    seed,
                    ..RequestFacts::default()
                };
                let pool_of = |members: &PoolMembers| PoolFacts {
                    pool_size: 48,
                    membership_epoch: members.epoch,
                    generation: 1,
                    ..PoolFacts::default()
                };
                let stamp = thread.stamp(list.len());
                let reader = Hosted {
                    workers: &list,
                    stamp,
                    kept: &thread.kept,
                };
                let asked = RequestFacts {
                    live: Some(&reader),
                    ..facts
                };
                let cx = Constraints::default();
                let by_reader = live.spec().engine().decide(
                    &asked,
                    &pool_of(&members),
                    &members.rows,
                    &cx,
                    &mut thread.scratch,
                );
                let by_rows = written.spec().engine().decide(
                    &facts,
                    &pool_of(&other),
                    &rows,
                    &cx,
                    &mut scratch,
                );
                let first = |decision: &Decision| match decision {
                    Decision::Pick(pick) => pick.targets.as_slice().first().map(|t| t.id),
                    _ => None,
                };
                assert!(first(&by_reader).is_some(), "{name} seed {seed}");
                assert_eq!(first(&by_reader), first(&by_rows), "{name} seed {seed}");
            }
        }
    }
}
