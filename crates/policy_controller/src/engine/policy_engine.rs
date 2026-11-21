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
//!   list. No row is copied or written. A row the decision asks about is
//!   answered by the worker object at the row's position ([`LiveRows`]):
//!   one call on the worker answers its status, its breaker, its vetoes
//!   and its count together, and the decision keeps that read, so what it
//!   decides it decides from one value of each worker. Only the workers a
//!   decision asks about are read: one where the rotation's worker takes
//!   the request, and the whole pool only where no worker near the turn
//!   does. A pass over the whole pool is the classic policy's: one read
//!   and one test a worker. The decision's scratch is the calling thread's
//!   own: no lock is taken and nothing is allocated after a thread's first
//!   decisions.
//! - **Rows written for a decision.** Three kinds of decision are made
//!   over rows written for them, one read of each worker left: a request
//!   its router limits to some of the pool's workers
//!   ([`PlacementRequest::refused`]), a profile whose stages read a row's
//!   facts off the row ([`Engine::decides_live`]), and a pool the pass
//!   over its own rows left nobody, which is read once more for the rule
//!   that decides such a pool, as the classic placement reads it. The rows
//!   go into a list the thread keeps.
//!
//! The engine keeps no worker list, no health and no count of requests in
//! flight. What it keeps is its stages' own: a rotation, a session's pins.
//!
//! A decision names one worker. A router that retries asks again and names
//! the workers it tried ([`PlacementRequest::tried`]).

use std::{
    cell::RefCell,
    collections::BTreeMap,
    hash::Hasher,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use rustc_hash::FxHasher;
use smg_worker::{
    identity::basic::{ConnectionMode, Worker, WorkerType},
    runtime::RoutingState,
};

use super::{
    decision::{Decision, ShedReason},
    framework::{Engine, Scratch},
    view::{
        fingerprint, Candidate, Constraints, Health, LiveCell, LiveRow, LiveRows, LiveTest,
        ModelFingerprint, Phase, PoolFacts, RequestFacts, Role, WireModes, WorkerId,
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

/// The engine of one policy document, for a gateway that decides in its own
/// process.
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
/// (its fingerprint, and what its profiles ask of their caller) is kept for the
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
    /// By phase: the decision is made over these rows in place
    /// ([`Engine::decides_live`]); any other is made over rows written
    /// for it.
    live: [bool; PHASES.len()],
}

/// What the engine reads of a model's id for one phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Spelled {
    model: ModelFingerprint,
    clock_free: bool,
    live: bool,
}

impl Spelled {
    fn of(engine: &Engine, model: &str, phase: Phase) -> Self {
        Self {
            model: ModelFingerprint::of(model),
            clock_free: engine.clock_free(model, phase),
            live: engine.decides_live(model, phase),
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
    /// as tried ([`PlacementRequest::tried`]). `None` past the list.
    #[must_use]
    pub fn id_at(&self, position: usize) -> Option<WorkerId> {
        self.rows.get(position).map(|row| row.id)
    }

    /// What the engine reads of `request`'s model id: the kept reading for
    /// the kept spelling, and the id's own reading for any other. The
    /// compare is of the two spellings' bytes: no id is hashed and no
    /// binding walked for the spelling the pool is asked under.
    fn spelled(&self, engine: &Engine, request: &PlacementRequest<'_>) -> Spelled {
        if *self.spelling == *request.model {
            let at = phase_at(request.phase);
            Spelled {
                model: self.model,
                clock_free: self.clock_free[at],
                live: self.live[at],
            }
        } else {
            Spelled::of(engine, request.model, request.phase)
        }
    }
}

/// What a router states of one attempt of a request.
///
/// Built over [`PlacementRequest::of`]: `PlacementRequest { tried, ..PlacementRequest::of(model) }`.
#[derive(Clone, Copy, Debug)]
pub struct PlacementRequest<'a> {
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

impl<'a> PlacementRequest<'a> {
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
    /// ([`PlacementRequest::tried`]).
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
    /// The binding lets the gateway hold the request for a worker to come.
    pub wait_allowed: bool,
    /// The profile that decided, by its position in the document; `None`
    /// where the request was declined ahead of every profile.
    pub profile: Option<u16>,
    /// The decision was a ticket for a queue. This engine holds none: the
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
            live: readings.map(|spelled| spelled.live),
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
        request: &PlacementRequest<'_>,
        members: &PoolMembers,
    ) -> Result<Placement, Unplaced> {
        WORKSPACE.with(|workspace| match workspace.try_borrow_mut() {
            Ok(mut workspace) => self.place_on(request, members, &mut workspace),
            // A stage of the decision asked the engine again on this thread:
            // the inner decision takes a workspace of its own.
            Err(_) => self.place_on(request, members, &mut Workspace::new()),
        })
    }

    fn place_on(
        &self,
        request: &PlacementRequest<'_>,
        members: &PoolMembers,
        workspace: &mut Workspace,
    ) -> Result<Placement, Unplaced> {
        let engine = self.spec.engine();
        let spelled = members.spelled(engine, request);
        let seed = workspace.seed();
        // The pool's own rows, read in place: a request every worker of the
        // pool may take, under a profile that reads no live fact off a row.
        let in_place = spelled.live && request.refused.is_empty();
        let mark = if in_place {
            workspace.mark(members.rows.len())
        } else {
            0
        };
        let Workspace {
            scratch,
            cells,
            written,
            positions,
            ..
        } = workspace;
        let pool = PoolFacts {
            pool_size: u32::try_from(members.rows.len()).unwrap_or(u32::MAX),
            membership_epoch: members.epoch,
            generation: engine.generation(),
            ..PoolFacts::default()
        };
        let now_micros = if spelled.clock_free {
            0
        } else {
            u64::try_from(self.origin.elapsed().as_micros()).unwrap_or(u64::MAX)
        };
        let facts = RequestFacts {
            model: request.model,
            model_fingerprint: spelled.model,
            phase: request.phase,
            tried: request.tried,
            seed,
            now_micros,
            ..RequestFacts::default()
        };
        if in_place {
            let rows = WorkerRows {
                workers: &members.workers,
                cells: cells.get(..members.workers.len()).unwrap_or(&[]),
                mark,
                whole: AtomicBool::new(false),
            };
            let decided = engine.decide_live(&facts, &pool, &members.rows, &rows, scratch);
            if let Some(decision) = decided {
                return placed(decision, members, None);
            }
        }
        // Rows written for this decision, each with one read of its worker:
        // a request held to some of the pool's workers, a profile whose
        // stages read a row's facts off the row, or a pool the pass over
        // its own rows left nobody.
        write(members, request.refused, written, positions);
        let pool = PoolFacts {
            reach: reach_of(request.refused),
            ..pool
        };
        let decision = engine.decide(&facts, &pool, written, &Constraints::default(), scratch);
        placed(decision, members, Some(positions))
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

/// What a router is told of `decision`: the worker object its first target
/// names, or why no worker was placed. `positions` is each written row's
/// place in the pool's list, where the decision was made over rows written
/// for it: a written row stands at its place among the rows written.
fn placed(
    decision: Decision,
    members: &PoolMembers,
    positions: Option<&[u32]>,
) -> Result<Placement, Unplaced> {
    let profile = decision.trace().made_under();
    match decision {
        Decision::Pick(pick) => pick
            .targets
            .as_slice()
            .first()
            .and_then(|target| {
                let position = match positions {
                    Some(positions) => *positions.get(target.position as usize)?,
                    None => target.position,
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
        // No built-in issues a ticket, and this engine holds no queue.
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

/// A worker's live facts, read on the worker object: one read answers its
/// status, its breaker, its liveness veto, its overload verdict and its
/// requests in flight.
fn read(worker: &dyn Worker) -> LiveRow {
    row_of_state(&worker.routing_state())
}

/// The row one read of a worker makes.
#[inline]
fn row_of_state(state: &RoutingState) -> LiveRow {
    LiveRow::new(
        health_of(state),
        u32::try_from(state.load).unwrap_or(u32::MAX),
    )
}

/// A worker's health bits as its one read says them.
#[inline]
fn health_of(state: &RoutingState) -> Health {
    Health {
        healthy: state.healthy,
        // An open breaker refuses the worker, and no rule lifts it.
        breaker_open: !state.can_execute,
        stalled: state.stalled,
        // A worker being drained is not ready: the status says it.
        draining: false,
        overloaded: state.overloaded,
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
            health: live.health(),
            in_flight: live.in_flight(),
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

/// The live rows of one decision over a pool: the pool's worker objects,
/// and the calling thread's cells, where a worker's first read is kept for
/// the decision.
#[derive(Debug)]
struct WorkerRows<'a> {
    workers: &'a [Arc<dyn Worker>],
    /// One cell a worker, in the list's order.
    cells: &'a [LiveCell],
    /// The decision's mark: a cell that holds another holds nothing for
    /// this decision.
    mark: u32,
    /// Every cell holds this decision's read: a pass over the whole pool
    /// read them all.
    whole: AtomicBool,
}

impl LiveRows for WorkerRows<'_> {
    #[inline]
    fn read(&self, position: u32) -> LiveRow {
        let at = position as usize;
        let (Some(worker), Some(cell)) = (self.workers.get(at), self.cells.get(at)) else {
            // No row of the pool stands there: nobody asks, and a row that
            // is not there is not routable.
            return LiveRow::default();
        };
        if let Some(kept) = cell.kept(self.mark) {
            return kept;
        }
        let row = read(worker.as_ref());
        cell.keep(self.mark, row);
        row
    }

    fn whole(&self) -> &[LiveCell] {
        if !self.whole.load(Ordering::Relaxed) {
            for (worker, cell) in self.workers.iter().zip(self.cells) {
                if cell.kept(self.mark).is_none() {
                    cell.keep(self.mark, read(worker.as_ref()));
                }
            }
            self.whole.store(true, Ordering::Relaxed);
        }
        self.cells
    }

    /// The pass a classic policy makes over its pool: one read of a worker
    /// and one test of what it says, the worker's own
    /// ([`RoutingState::eligible`]). A worker that passes has its read
    /// kept, so a stage that reads its count reads the same value, and a
    /// worker that is refused costs nothing more. The rows the decision's
    /// draws refused are left out, and no cell is looked at for the others:
    /// the decision read none of them.
    fn pass(&self, test: LiveTest, refused: Option<&[u32]>, kept: &mut Vec<u32>) {
        let Some(refused) = refused else {
            // The decision holds reads nobody named: each row is tested by
            // its one read, the rows not read yet by a read made now.
            return test.pass_whole(self.whole(), kept);
        };
        match (test.routable, test.not_overloaded) {
            (true, true) => self.pass_by(RoutingState::eligible, refused, kept),
            (true, false) => {
                self.pass_by(RoutingState::eligible_ignoring_overload, refused, kept);
            }
            _ => self.pass_by(|state| test.passes(health_of(&state)), refused, kept),
        }
    }
}

impl WorkerRows<'_> {
    /// [`LiveRows::pass`] by one test of a worker's read, compiled once a
    /// test: the rows between the refused ones, then the rows past the
    /// last, each read once.
    #[inline(always)]
    fn pass_by(&self, admits: impl Fn(RoutingState) -> bool, refused: &[u32], kept: &mut Vec<u32>) {
        let rows = self.workers.len().min(self.cells.len());
        let rows = u32::try_from(rows).unwrap_or(u32::MAX);
        let mut from = 0u32;
        for until in refused.iter().copied().chain([rows]) {
            let until = until.min(rows);
            // A row named twice, or out of its order, opens no span.
            let span = from as usize..until as usize;
            let workers = self.workers.get(span.clone()).unwrap_or_default();
            let cells = self.cells.get(span).unwrap_or_default();
            for ((worker, cell), position) in workers.iter().zip(cells).zip(from..until) {
                let state = worker.routing_state();
                if admits(state) {
                    cell.keep(self.mark, row_of_state(&state));
                    kept.push(position);
                }
            }
            from = from.max(until.saturating_add(1));
        }
    }
}

/// What a deciding thread keeps between its decisions.
#[derive(Debug)]
struct Workspace {
    scratch: Scratch,
    /// One cell a row of the largest pool this thread decided over in
    /// place: what a decision read of its workers, under its mark.
    cells: Vec<LiveCell>,
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
    static WORKSPACE: RefCell<Workspace> = RefCell::new(Workspace::new());
}

/// The threads that decided so far: each starts its seeds from its number.
static THREADS: AtomicU64 = AtomicU64::new(0);

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

impl Workspace {
    fn new() -> Self {
        let number = THREADS.fetch_add(1, Ordering::Relaxed);
        // The wall clock at the thread's first decision tells two runs of
        // the process apart.
        let started = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos() as u64);
        Self {
            scratch: Scratch::default(),
            cells: Vec::new(),
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

    /// The next decision's mark over a pool of `rows`, with a cell for each
    /// row. The cells are sized at a thread's first decision over a pool
    /// that large, and emptied when the marks come round: a read kept under
    /// a mark of the round before would read as this decision's.
    fn mark(&mut self, rows: usize) -> u32 {
        if self.cells.len() < rows {
            self.cells.resize_with(rows, LiveCell::default);
        }
        self.mark = self.mark.wrapping_add(1);
        if self.mark == 0 {
            for cell in &self.cells {
                cell.clear();
            }
            self.mark = 1;
        }
        self.mark
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use openai_protocol::worker::WorkerStatus;
    use smg_worker::{runtime::StallReason, test_support::ready_worker};

    use super::*;
    use crate::{document::spec::PolicySpec, policies::power_of_two::PowerOfTwoMetric};

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

    /// The reader of the next decision over `list`.
    fn reader_of<'a>(list: &'a [Arc<dyn Worker>], workspace: &'a mut Workspace) -> WorkerRows<'a> {
        let mark = workspace.mark(list.len());
        WorkerRows {
            workers: list,
            cells: &workspace.cells[..list.len()],
            mark,
            whole: AtomicBool::new(false),
        }
    }

    fn place(
        engine: &PolicyEngine,
        members: &PoolMembers,
        request: &PlacementRequest<'_>,
    ) -> usize {
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
            let request = PlacementRequest::of(MODEL);
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
        let request = PlacementRequest::of(MODEL);
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
        let request = PlacementRequest::of(MODEL);
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
            assert!(engine
                .place(&PlacementRequest::of("canonical"), &members)
                .is_ok());
            let unbound = engine
                .place(&PlacementRequest::of("alias"), &members)
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
            let request = PlacementRequest::of(spelling);
            assert_eq!(
                members.spelled(inner, &request),
                Spelled::of(inner, spelling, Phase::Single),
                "{spelling}"
            );
        }
        assert_ne!(
            members.spelled(inner, &PlacementRequest::of("kept")).model,
            members.spelled(inner, &PlacementRequest::of("other")).model
        );
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
        let request = PlacementRequest {
            refused: &refused,
            ..PlacementRequest::of(MODEL)
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
        let nobody = PlacementRequest {
            refused: &all,
            ..PlacementRequest::of(MODEL)
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
        let request = PlacementRequest {
            tried: &tried,
            ..PlacementRequest::of(MODEL)
        };
        for _ in 0..6 {
            assert_eq!(place(&engine, &members, &request), 1);
        }
    }

    /// The pool's own rows and rows written with the same facts decide
    /// alike: two engines of one document, one asked over the reader and
    /// one over written rows, give the same answer for the same seeds.
    /// Over a pool all ready each names the same worker. A pool the pass
    /// leaves nobody is handed back by the reader's engine, and the front
    /// places it as the engine asked over written rows decides it: a pool
    /// every worker of which is over its thresholds is steered to the one
    /// worker with the fewest requests in flight, and a pool nobody serves
    /// is shed at the health filter.
    #[test]
    fn the_pools_own_rows_and_written_rows_decide_alike() {
        // What sets a fleet's worker at a place of its list.
        type Prepare = fn(usize, &Arc<dyn Worker>);
        let fleets: [(&str, Prepare); 3] = [
            ("all ready", |_, _| {}),
            ("all over their thresholds", |at, worker| {
                worker.set_overloaded(true);
                // One worker holds no request, and no two the same count.
                for _ in 0..(at + 41) % 48 {
                    worker.increment_load();
                }
            }),
            ("nobody ready", |_, worker| {
                worker.set_status(WorkerStatus::NotReady);
            }),
        ];
        for name in ["round_robin", "random"] {
            for (fleet, prepare) in fleets {
                let (live, written) = (preset(name), preset(name));
                let list = pool(48);
                for (at, worker) in list.iter().enumerate() {
                    prepare(at, worker);
                }
                let members = live.pool_members(MODEL, &list);
                let other = written.pool_members(MODEL, &list);
                let rows: Vec<Candidate> = list
                    .iter()
                    .zip(&*other.rows)
                    .map(|(worker, row)| {
                        let live = read(worker.as_ref());
                        Candidate {
                            health: live.health(),
                            in_flight: live.in_flight(),
                            ..*row
                        }
                    })
                    .collect();
                let mut workspace = Workspace::new();
                let mut by_rows = Scratch::default();
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
                    let mark = workspace.mark(list.len());
                    let reader = WorkerRows {
                        workers: &list,
                        cells: &workspace.cells[..list.len()],
                        mark,
                        whole: AtomicBool::new(false),
                    };
                    let by_reader = live.spec().engine().decide_live(
                        &facts,
                        &pool_of(&members),
                        &members.rows,
                        &reader,
                        &mut workspace.scratch,
                    );
                    let by_rows = written.spec().engine().decide(
                        &facts,
                        &pool_of(&other),
                        &rows,
                        &Constraints::default(),
                        &mut by_rows,
                    );
                    let said = |decision: &Decision| match decision {
                        Decision::Pick(pick) => {
                            let first = pick.targets.as_slice().first().map(|t| t.id);
                            (first, "", pick.trace.lifted)
                        }
                        Decision::Shed(shed) => (None, shed.emptied_by, shed.trace.lifted),
                        Decision::Queue(queue) => (None, "queue", queue.trace.lifted),
                    };
                    let (first, emptied_by, _) = said(&by_rows);
                    match by_reader {
                        Some(by_reader) => {
                            assert_eq!(fleet, "all ready", "{name}: a worker stands");
                            assert_eq!(said(&by_reader), said(&by_rows), "{name}, seed {seed}");
                        }
                        None => {
                            assert_ne!(fleet, "all ready", "{name}: handed back");
                            match live.place(&PlacementRequest::of(MODEL), &members) {
                                Ok(placed) => assert_eq!(Some(placed.id()), first, "{name}"),
                                Err(unplaced) => {
                                    assert_eq!((None, unplaced.emptied_by), (first, emptied_by));
                                }
                            }
                        }
                    }
                    if fleet == "all over their thresholds" {
                        let idle = members.id_at(7).expect("a row");
                        assert_eq!(first, Some(idle), "{name}, seed {seed}");
                    }
                }
            }
        }
    }

    /// A decision reads a worker once and keeps the read: a second ask of
    /// the same row within the decision answers the first read, whatever
    /// the worker says by then, a pass over the whole pool reads only the
    /// workers not read yet, and the next decision reads each again.
    #[test]
    fn a_decision_keeps_its_first_read_of_a_worker() {
        let list = pool(3);
        let mut workspace = Workspace::new();
        let mark = workspace.mark(list.len());
        let reader = WorkerRows {
            workers: &list,
            cells: &workspace.cells[..list.len()],
            mark,
            whole: AtomicBool::new(false),
        };
        let first = reader.read(1);
        assert!(first.health().healthy && first.in_flight() == 0);
        list[1].increment_load();
        list[1].set_overloaded(true);
        assert_eq!(reader.read(1), first);
        // The whole pool: the row read already stands as it was read, and
        // the others are read now.
        list[2].increment_load();
        let whole: Vec<LiveRow> = reader.whole().iter().map(LiveCell::row).collect();
        assert_eq!(whole[1], first);
        assert_eq!(whole[2].in_flight(), 1);
        list[0].increment_load();
        assert_eq!(reader.whole()[0].row().in_flight(), 0, "read once");
        // The next decision reads each worker again.
        let mark = workspace.mark(list.len());
        let next = WorkerRows {
            workers: &list,
            cells: &workspace.cells[..list.len()],
            mark,
            whole: AtomicBool::new(false),
        };
        let again = next.read(1);
        assert!(again.health().overloaded && again.in_flight() == 1);
        assert_eq!(next.read(0).in_flight(), 1);
        // A place past the pool is no worker's.
        assert_eq!(next.read(9), LiveRow::default());
    }

    /// The pass over a pool's workers leaves the rows whose one read passes
    /// the decision's test, under each test a decision asks: every veto is
    /// read on the worker object, as the worker's own test of its routing
    /// state reads it, and the read of a row that passes is kept. A row the
    /// decision's draw refused is named to the pass, which leaves it out
    /// and does not read it again. A decision that names no row it read is
    /// answered from each row's one read.
    #[test]
    fn the_pass_leaves_the_rows_whose_one_read_passes_the_test() {
        let list = pool(16);
        for (at, worker) in list.iter().enumerate() {
            if at & 1 != 0 {
                worker.set_status(WorkerStatus::NotReady);
            }
            if at & 2 != 0 {
                worker.set_overloaded(true);
            }
            if at & 4 != 0 {
                worker.set_stall(Some(StallReason::Wedged));
            }
            if at & 8 != 0 {
                for _ in 0..16 {
                    worker.record_outcome(500);
                }
            }
        }
        let states: Vec<RoutingState> = list.iter().map(|worker| worker.routing_state()).collect();
        for (at, state) in states.iter().enumerate() {
            assert_eq!(state.healthy, at & 1 == 0, "worker {at}");
            assert_eq!(state.overloaded, at & 2 != 0, "worker {at}");
            assert_eq!(state.stalled, at & 4 != 0, "worker {at}");
            assert_eq!(state.can_execute, at & 8 == 0, "worker {at}");
        }
        let mut workspace = Workspace::new();
        let mut kept = Vec::new();
        for (routable, not_overloaded) in
            [(true, true), (true, false), (false, true), (false, false)]
        {
            let test = LiveTest {
                routable,
                not_overloaded,
            };
            let expected: Vec<u32> = (0u32..)
                .zip(&states)
                .filter(|(_, state)| test.passes(health_of(state)))
                .map(|(at, _)| at)
                .collect();
            let reader = reader_of(&list, &mut workspace);
            kept.clear();
            reader.pass(test, Some(&[]), &mut kept);
            assert_eq!(kept, expected, "{test:?}");
            for (at, cell) in (0u32..).zip(reader.cells) {
                let passed = expected.contains(&at);
                assert_eq!(cell.kept(reader.mark).is_some(), passed, "worker {at}");
            }
        }
        let test = LiveTest {
            routable: true,
            not_overloaded: true,
        };
        let reader = reader_of(&list, &mut workspace);
        assert!(!test.passes_row(reader.read(1)));
        // The worker is ready by the pass, and its refusal stands: the row
        // is named, and the pass reads no named row.
        list[1].set_status(WorkerStatus::Ready);
        kept.clear();
        reader.pass(test, Some(&[1]), &mut kept);
        assert_eq!(kept, [0]);
        // A decision that names no row it read keeps the row's one read.
        list[1].set_status(WorkerStatus::NotReady);
        let reader = reader_of(&list, &mut workspace);
        assert!(!test.passes_row(reader.read(1)));
        list[1].set_status(WorkerStatus::Ready);
        kept.clear();
        reader.pass(test, None, &mut kept);
        assert_eq!(kept, [0]);
        // The next decision reads the worker as it stands.
        let reader = reader_of(&list, &mut workspace);
        kept.clear();
        reader.pass(test, Some(&[]), &mut kept);
        assert_eq!(kept, [0, 1]);
    }

    /// The marks come round after their last value: the cells are emptied,
    /// so a read kept under a mark of the round before answers no decision.
    #[test]
    fn the_marks_come_round_over_emptied_cells() {
        let list = pool(1);
        let mut workspace = Workspace::new();
        workspace.mark = u32::MAX - 1;
        let last = workspace.mark(1);
        assert_eq!(last, u32::MAX);
        let reader = WorkerRows {
            workers: &list,
            cells: &workspace.cells[..1],
            mark: last,
            whole: AtomicBool::new(false),
        };
        let _ = reader.read(0);
        assert!(workspace.cells[0].kept(last).is_some());
        let first = workspace.mark(1);
        assert_eq!(first, 1);
        assert!(workspace.cells[0].kept(last).is_none());
        assert!(workspace.cells[0].kept(first).is_none());
    }

    /// One worker in eight is not ready, and in another fleet three in
    /// four. None of them is named, and the ready workers are named in
    /// turn: each exactly once a turn of the ready ones, as the classic
    /// rotation among the ready workers names them. With the workers ready
    /// again each is named once a turn of the pool.
    #[test]
    fn a_turn_on_a_worker_not_ready_goes_to_the_next_ready_one() {
        for ready in [
            (|at: usize| at % 8 != 5) as fn(usize) -> bool,
            (|at: usize| at % 4 == 2) as fn(usize) -> bool,
        ] {
            let engine = preset("round_robin");
            let list = pool(64);
            let members = engine.pool_members(MODEL, &list);
            let request = PlacementRequest::of(MODEL);
            for (at, worker) in list.iter().enumerate() {
                if !ready(at) {
                    worker.set_status(WorkerStatus::NotReady);
                }
            }
            let serving = (0..64).filter(|&at| ready(at)).count();
            let mut named = [0u32; 64];
            for _ in 0..serving * 50 {
                named[place(&engine, &members, &request)] += 1;
            }
            for (at, &times) in named.iter().enumerate() {
                let expected = if ready(at) { 50 } else { 0 };
                assert_eq!(times, expected, "worker {at}");
            }
            for worker in list.iter() {
                worker.set_status(WorkerStatus::Ready);
            }
            let first = place(&engine, &members, &request);
            for step in 1..128 {
                assert_eq!(place(&engine, &members, &request), (first + step) % 64);
            }
        }
    }

    /// A pool none of whose workers passes every filter is decided over
    /// rows written for it, by the engine's stages one by one and its rule
    /// for such a pool: a pool nobody serves places the request on no
    /// worker and names the filter that left none, a pool over its
    /// overload thresholds is steered to the worker with the fewest
    /// requests in flight, and a retry that tried every worker is placed
    /// on one of them again.
    #[test]
    fn a_pool_no_worker_of_which_passes_is_decided_over_written_rows() {
        let engine = preset("round_robin");
        let request = PlacementRequest::of(MODEL);

        let list = pool(6);
        let members = engine.pool_members(MODEL, &list);
        for worker in list.iter() {
            worker.set_status(WorkerStatus::NotReady);
        }
        let unplaced = engine
            .place(&request, &members)
            .expect_err("nobody serves the model");
        assert_eq!(unplaced.emptied_by, "healthy");
        assert!(unplaced.profile.is_some());

        let list = pool(6);
        let members = engine.pool_members(MODEL, &list);
        for (at, worker) in list.iter().enumerate() {
            worker.set_overloaded(true);
            if at != 4 {
                worker.increment_load();
            }
        }
        for _ in 0..8 {
            assert_eq!(place(&engine, &members, &request), 4);
        }

        let list = pool(3);
        let members = engine.pool_members(MODEL, &list);
        let tried: Vec<WorkerId> = (0..3).filter_map(|at| members.id_at(at)).collect();
        let retry = PlacementRequest {
            tried: &tried,
            ..PlacementRequest::of(MODEL)
        };
        assert!(place(&engine, &members, &retry) < 3);
    }

    /// A profile whose stages read a row's facts off the row is decided
    /// over rows written for the decision: a picker that takes the fewest
    /// requests in flight places every request on the idle worker of a
    /// pool whose others hold one.
    #[test]
    fn a_profile_that_reads_rows_is_decided_over_written_rows() {
        let mut document = PolicySpec::power_of_two(PowerOfTwoMetric::LeastRequests)
            .expect("a form of the built-in preset");
        document.generation = 1;
        let engine = engine_of(&document);
        assert!(!engine.spec().engine().decides_live(MODEL, Phase::Single));
        let list = pool(5);
        let members = engine.pool_members(MODEL, &list);
        for (at, worker) in list.iter().enumerate() {
            if at != 3 {
                worker.increment_load();
            }
        }
        for _ in 0..8 {
            assert_eq!(place(&engine, &members, &PlacementRequest::of(MODEL)), 3);
        }
    }
}
