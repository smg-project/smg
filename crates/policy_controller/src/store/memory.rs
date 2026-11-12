//! The one-process store: every ingress trait over plain tables behind
//! mutexes. The controller's own store once its drivers move in (soft state
//! its sources republish, no log), and the double the contract tests and the
//! controller's tests run over. Nothing here is on a decision path: a view is
//! built from snapshots the store hands out, not through these locks.

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::{
    lease::{LeaseId, Outcome, ReleaseReason, Side, TransportClass},
    store::{
        health::HealthSink,
        kv_events::{
            Admission, BatchFacts, BatchOutcome, KvCursor, KvEventSink, RankCursor, ResyncReason,
        },
        leases::{LeaseRecord, LeaseRefused, LeaseState, LeasesStore},
        loads::{LoadSample, LoadSink, LoadSource, PollMode},
        reservations::{ReservationKind, ReservationStore},
        sessions::{Pin, SessionsStore},
        workers::{
            identity, MembershipChange, MembershipTail, WorkerFacts, WorkerPatch, WorkerRow,
            WorkerStatus, WorkersStore,
        },
        GatewayId,
    },
    view::{LoadRow, WorkerId},
};

/// Membership changes kept for watchers; a watcher further behind snapshots
/// again (the registry's broadcast is sized the same today).
const CHANGE_LOG: usize = 1_024;

/// The store. `Default` is empty with term `0`.
#[derive(Default)]
pub struct MemoryStore {
    workers: Mutex<Workers>,
    health: Mutex<HealthTables>,
    loads: Mutex<FxHashMap<WorkerId, Ranks>>,
    kv: Mutex<Kv>,
    reservations: Mutex<Reservations>,
    sessions: Mutex<FxHashMap<u64, Pin>>,
    leases: Mutex<Leases>,
}

impl std::fmt::Debug for MemoryStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // One guard per table, each dropped before the next is taken: the
        // mutex is not reentrant and a chained statement keeps every guard
        // to its end.
        let (workers, epoch) = {
            let workers = self.workers.lock();
            (workers.rows.len(), workers.epoch)
        };
        let leases = self.leases.lock().records.len();
        f.debug_struct("MemoryStore")
            .field("workers", &workers)
            .field("epoch", &epoch)
            .field("leases", &leases)
            .finish_non_exhaustive()
    }
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// How many resyncs were recorded for `reason`.
    pub fn resyncs(&self, reason: ResyncReason) -> u64 {
        self.kv
            .lock()
            .resyncs
            .get(reason.as_str())
            .copied()
            .unwrap_or(0)
    }

    /// Batches applied for one rank since its last clear: the stand-in for
    /// the index's blocks of the rank.
    pub fn applied_batches(&self, id: WorkerId, rank: u16) -> u64 {
        self.kv
            .lock()
            .applied
            .get(&(id, rank))
            .copied()
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------- workers

#[derive(Default)]
struct Workers {
    rows: FxHashMap<WorkerId, Row>,
    by_key: HashMap<String, WorkerId>,
    epoch: u64,
    log: VecDeque<(u64, MembershipChange)>,
}

struct Row {
    row: WorkerRow,
    probes: Probes,
}

/// The prober's counters for one row.
#[derive(Default)]
struct Probes {
    failures: u32,
    successes: u32,
    pending_total: u32,
}

impl Workers {
    fn record(&mut self, change: MembershipChange) -> MembershipChange {
        self.epoch += 1;
        if self.log.len() == CHANGE_LOG {
            self.log.pop_front();
        }
        self.log.push_back((self.epoch, change));
        change
    }

    /// The row under `id` when `revision` is still its revision.
    fn guarded(&mut self, id: WorkerId, revision: u64) -> Option<&mut Row> {
        self.rows
            .get_mut(&id)
            .filter(|row| row.row.revision == revision)
    }

    fn set_status(&mut self, id: WorkerId, status: WorkerStatus) -> Option<MembershipChange> {
        let row = self.rows.get_mut(&id)?;
        let from = row.row.status;
        if from == status {
            return None;
        }
        row.row.status = status;
        row.row.revision += 1;
        Some(self.record(MembershipChange::StatusChanged {
            id,
            from,
            to: status,
        }))
    }
}

impl WorkersStore for MemoryStore {
    fn upsert(&self, facts: WorkerFacts) -> MembershipChange {
        let mut workers = self.workers.lock();
        let now = Instant::now();
        if let Some(old_id) = workers.by_key.get(&facts.key).copied() {
            let same_incarnation = workers
                .rows
                .get(&old_id)
                .is_some_and(|row| row.row.facts.incarnation == facts.incarnation);
            if same_incarnation {
                let Some(row) = workers.rows.get_mut(&old_id) else {
                    return MembershipChange::Removed { id: old_id };
                };
                let resized = row.row.facts.dp_size != facts.dp_size;
                row.row.facts = facts;
                row.row.revision += 1;
                return workers.record(if resized {
                    MembershipChange::Resized { id: old_id }
                } else {
                    MembershipChange::Relabelled { id: old_id }
                });
            }
            // A new incarnation under the key: the old id leaves with its
            // counters; its status carries over when the registration
            // matches (role, models and wires), as the registry inherits a
            // replaced worker's runtime state today.
            let old = workers.rows.remove(&old_id);
            let new_id = identity(&facts.key, facts.incarnation);
            let status = old
                .filter(|old| {
                    old.row.facts.role == facts.role
                        && old.row.facts.models == facts.models
                        && old.row.facts.wire_modes == facts.wire_modes
                })
                .map_or(WorkerStatus::Pending, |old| old.row.status);
            workers.by_key.insert(facts.key.clone(), new_id);
            workers.rows.insert(
                new_id,
                Row {
                    row: WorkerRow {
                        id: new_id,
                        facts,
                        status,
                        revision: 1,
                        admitted_at: now,
                    },
                    probes: Probes::default(),
                },
            );
            return workers.record(MembershipChange::Replaced {
                old: old_id,
                new: new_id,
            });
        }
        let id = identity(&facts.key, facts.incarnation);
        workers.by_key.insert(facts.key.clone(), id);
        workers.rows.insert(
            id,
            Row {
                row: WorkerRow {
                    id,
                    facts,
                    status: WorkerStatus::Pending,
                    revision: 1,
                    admitted_at: now,
                },
                probes: Probes::default(),
            },
        );
        workers.record(MembershipChange::Registered { id })
    }

    fn patch(&self, id: WorkerId, patch: WorkerPatch, revision: u64) -> Option<MembershipChange> {
        let mut workers = self.workers.lock();
        let row = workers.guarded(id, revision)?;
        if let Some(labels) = patch.labels {
            row.row.facts.labels = labels;
        }
        if let Some(priority) = patch.priority {
            row.row.facts.priority = priority;
        }
        if let Some(overload) = patch.overload {
            row.row.facts.overload = overload;
        }
        row.row.revision += 1;
        Some(workers.record(MembershipChange::Relabelled { id }))
    }

    fn remove(&self, id: WorkerId, revision: u64) -> Option<MembershipChange> {
        let mut workers = self.workers.lock();
        workers.guarded(id, revision)?;
        let row = workers.rows.remove(&id)?;
        workers.by_key.remove(&row.row.facts.key);
        Some(workers.record(MembershipChange::Removed { id }))
    }

    fn set_status(
        &self,
        id: WorkerId,
        status: WorkerStatus,
        revision: u64,
    ) -> Option<MembershipChange> {
        let mut workers = self.workers.lock();
        workers.guarded(id, revision)?;
        workers.set_status(id, status)
    }

    fn get(&self, id: WorkerId) -> Option<WorkerRow> {
        self.workers.lock().rows.get(&id).map(|row| row.row.clone())
    }

    fn by_key(&self, key: &str) -> Option<WorkerRow> {
        let workers = self.workers.lock();
        let id = workers.by_key.get(key)?;
        workers.rows.get(id).map(|row| row.row.clone())
    }

    fn epoch(&self) -> u64 {
        self.workers.lock().epoch
    }

    fn snapshot(&self) -> (u64, Vec<WorkerRow>) {
        let workers = self.workers.lock();
        let mut rows: Vec<WorkerRow> = workers.rows.values().map(|row| row.row.clone()).collect();
        rows.sort_by(|a, b| a.facts.key.cmp(&b.facts.key));
        (workers.epoch, rows)
    }

    fn changes_since(&self, epoch: u64) -> MembershipTail {
        let workers = self.workers.lock();
        if epoch > workers.epoch {
            return MembershipTail::SnapshotRequired;
        }
        if epoch == workers.epoch {
            return MembershipTail::Changes(Vec::new());
        }
        match workers.log.front() {
            // The change after `epoch` is still in the log.
            Some(&(first, _)) if first <= epoch + 1 => MembershipTail::Changes(
                workers
                    .log
                    .iter()
                    .filter(|(at, _)| *at > epoch)
                    .copied()
                    .collect(),
            ),
            _ => MembershipTail::SnapshotRequired,
        }
    }
}

// ----------------------------------------------------------------- health

#[derive(Default)]
struct HealthTables {
    /// A gateway's veto on a worker, until the instant.
    vetoes: FxHashMap<(GatewayId, WorkerId), Instant>,
    /// The latest completed dispatch of class `Ok` on a worker by each
    /// gateway: the precedence rule is per gateway.
    ok_at: FxHashMap<(GatewayId, WorkerId), Instant>,
}

/// The status machine of one probe over the row's counters: the registry's
/// `compute_next_status`, with the counters here instead of on the handle.
fn next_status(row: &mut Row, ok: bool) -> Option<WorkerStatus> {
    let policy = row.row.facts.health;
    let status = row.row.status;
    let failure = policy.failure_threshold.max(1);
    let success = policy.success_threshold.max(1);
    let liveness = failure * 3;
    let pending_cap = failure * 10;
    let probes = &mut row.probes;
    if ok {
        probes.failures = 0;
        probes.successes += 1;
        probes.pending_total += 1;
        if matches!(
            status,
            WorkerStatus::Pending | WorkerStatus::NotReady | WorkerStatus::Failed
        ) && probes.successes >= success
        {
            probes.successes = 0;
            probes.pending_total = 0;
            return Some(WorkerStatus::Ready);
        }
        if status == WorkerStatus::Pending && probes.pending_total >= pending_cap {
            probes.successes = 0;
            return Some(WorkerStatus::Failed);
        }
        return None;
    }
    probes.successes = 0;
    probes.failures += 1;
    probes.pending_total += 1;
    match status {
        WorkerStatus::Ready if probes.failures >= failure => {
            probes.failures = 0;
            Some(WorkerStatus::NotReady)
        }
        WorkerStatus::NotReady if probes.failures >= liveness => {
            probes.failures = 0;
            Some(WorkerStatus::Failed)
        }
        WorkerStatus::Pending if probes.pending_total >= pending_cap => {
            probes.failures = 0;
            Some(WorkerStatus::Failed)
        }
        _ => None,
    }
}

impl HealthSink for MemoryStore {
    fn probe_outcome(
        &self,
        id: WorkerId,
        ok: bool,
        revision: u64,
    ) -> Option<(WorkerStatus, WorkerStatus)> {
        let mut workers = self.workers.lock();
        let row = workers.guarded(id, revision)?;
        let from = row.row.status;
        let to = next_status(row, ok)?;
        workers.set_status(id, to).map(|_| (from, to))
    }

    fn connected(&self, id: WorkerId, revision: u64) -> Option<(WorkerStatus, WorkerStatus)> {
        let mut workers = self.workers.lock();
        let row = workers.guarded(id, revision)?;
        let from = row.row.status;
        if matches!(from, WorkerStatus::Draining | WorkerStatus::Ready) {
            return None;
        }
        row.probes = Probes::default();
        workers
            .set_status(id, WorkerStatus::Ready)
            .map(|_| (from, WorkerStatus::Ready))
    }

    fn gateway_veto(&self, gateway: GatewayId, id: WorkerId, until: Option<Instant>) {
        let mut health = self.health.lock();
        match until {
            Some(until) => {
                health.vetoes.insert((gateway, id), until);
            }
            None => {
                health.vetoes.remove(&(gateway, id));
            }
        }
    }

    fn vetoed(&self, gateway: GatewayId, id: WorkerId, now: Instant) -> bool {
        self.health
            .lock()
            .vetoes
            .get(&(gateway, id))
            .is_some_and(|until| *until > now)
    }

    fn completed(&self, gateway: GatewayId, id: WorkerId, class: TransportClass, at: Instant) {
        if class != TransportClass::Ok {
            return;
        }
        let mut health = self.health.lock();
        let latest = health.ok_at.entry((gateway, id)).or_insert(at);
        if at > *latest {
            *latest = at;
        }
    }

    fn completed_ok_since(&self, gateway: GatewayId, id: WorkerId, since: Instant) -> bool {
        self.health
            .lock()
            .ok_at
            .get(&(gateway, id))
            .is_some_and(|at| *at >= since)
    }
}

// ------------------------------------------------------------------ loads

impl LoadSource {
    /// A source whose records arrive on their own (a stream or its relay):
    /// what the poll decision counts as "the stream working".
    const fn pushes(self) -> bool {
        matches!(self, Self::Stream | Self::Relay(_))
    }
}

/// One worker's ranks: the latest sample per rank and, apart from it, the
/// instant each rank's own record last arrived (a poll never moves it).
#[derive(Default)]
struct Ranks {
    samples: BTreeMap<u16, LoadSample>,
    pushed_at: BTreeMap<u16, Instant>,
}

impl LoadSink for MemoryStore {
    fn record(
        &self,
        id: WorkerId,
        rank: u16,
        row: LoadRow,
        received_at: Instant,
        source: LoadSource,
    ) {
        let age = Duration::from_millis(u64::from(row.report_age_ms));
        let sample = LoadSample {
            row,
            received_at,
            sampled_at: received_at.checked_sub(age).unwrap_or(received_at),
            source,
        };
        let mut loads = self.loads.lock();
        let ranks = loads.entry(id).or_default();
        if source.pushes() {
            ranks.pushed_at.insert(rank, received_at);
        }
        ranks.samples.insert(rank, sample);
    }

    fn report(
        &self,
        id: WorkerId,
        rows: &[(u16, LoadRow)],
        polled_at: Instant,
        source: LoadSource,
    ) {
        let mut loads = self.loads.lock();
        let ranks = loads.entry(id).or_default();
        for &(rank, row) in rows {
            let pushed_since = ranks
                .pushed_at
                .get(&rank)
                .is_some_and(|&at| at >= polled_at);
            if pushed_since {
                continue;
            }
            let age = Duration::from_millis(u64::from(row.report_age_ms));
            ranks.samples.insert(
                rank,
                LoadSample {
                    row,
                    received_at: polled_at,
                    sampled_at: polled_at.checked_sub(age).unwrap_or(polled_at),
                    source,
                },
            );
        }
    }

    fn poll_mode(&self, id: WorkerId, now: Instant, interval: Duration) -> PollMode {
        let loads = self.loads.lock();
        let Some(ranks) = loads.get(&id) else {
            return PollMode::Poll;
        };
        // The receipts, not the samples: a poll's row replaces a silent
        // rank's sample but never its receipt, so one pushing rank cannot
        // hide another that fell silent.
        let mut pushing = ranks.pushed_at.values();
        let Some(first) = pushing.next() else {
            return PollMode::Poll;
        };
        let fresh = |at: &Instant| now.saturating_duration_since(*at) < interval;
        if fresh(first) && pushing.all(fresh) {
            PollMode::Suppressed
        } else {
            PollMode::Fallback
        }
    }

    fn load(&self, id: WorkerId, rank: u16) -> Option<LoadSample> {
        self.loads.lock().get(&id)?.samples.get(&rank).copied()
    }

    fn ranks(&self, id: WorkerId) -> usize {
        self.loads
            .lock()
            .get(&id)
            .map_or(0, |ranks| ranks.samples.len())
    }

    fn evict(&self, id: WorkerId) {
        self.loads.lock().remove(&id);
    }
}

// -------------------------------------------------------------- KV events

#[derive(Default)]
struct Kv {
    cursors: FxHashMap<(WorkerId, u16), RankCursor>,
    /// Batches applied per rank since its last clear.
    applied: FxHashMap<(WorkerId, u16), u64>,
    block_sizes: HashMap<String, u32>,
    resyncs: HashMap<&'static str, u64>,
}

impl Kv {
    fn note_resync(&mut self, reason: ResyncReason) {
        *self.resyncs.entry(reason.as_str()).or_insert(0) += 1;
    }

    fn clear_rank(&mut self, id: WorkerId, rank: u16) {
        self.applied.remove(&(id, rank));
    }
}

impl KvEventSink for MemoryStore {
    /// The store applies nothing but the count of a batch; the index is the
    /// host's.
    type Batch = ();

    fn admit(&self, id: WorkerId, rank: u16, facts: BatchFacts, (): &Self::Batch) -> BatchOutcome {
        let mut kv = self.kv.lock();
        if let Some(chunk) = facts.snapshot {
            if chunk.index == 0 {
                kv.clear_rank(id, rank);
                kv.note_resync(ResyncReason::Snapshot);
            }
            kv.cursors
                .entry((id, rank))
                .or_default()
                .resync_to(facts.sequence, chunk.unknown_before);
            *kv.applied.entry((id, rank)).or_insert(0) += 1;
            return BatchOutcome::Applied;
        }
        let admission = kv
            .cursors
            .entry((id, rank))
            .or_default()
            .admit(facts.sequence, facts.clears);
        match admission {
            Admission::Stale => return BatchOutcome::Skipped,
            Admission::Replay { expected } => {
                return BatchOutcome::Gap {
                    expected,
                    received: facts.sequence,
                };
            }
            Admission::Restart => {
                kv.clear_rank(id, rank);
                kv.note_resync(ResyncReason::PublisherRestart);
            }
            Admission::Unrecovered { cleared: true, .. } => {
                kv.clear_rank(id, rank);
                kv.note_resync(ResyncReason::GapCleared);
            }
            Admission::Apply | Admission::Unrecovered { cleared: false, .. } => {}
        }
        *kv.applied.entry((id, rank)).or_insert(0) += 1;
        BatchOutcome::Applied
    }

    fn reconnected(&self, id: WorkerId) {
        let mut kv = self.kv.lock();
        for ((worker, _), cursor) in &mut kv.cursors {
            if *worker == id {
                cursor.reconnected();
            }
        }
    }

    fn resync(&self, id: WorkerId, rank: Option<u16>, reason: ResyncReason) {
        let mut kv = self.kv.lock();
        let ranks: Vec<u16> = kv
            .cursors
            .keys()
            .filter(|(worker, at)| *worker == id && rank.is_none_or(|rank| rank == *at))
            .map(|(_, at)| *at)
            .collect();
        for at in ranks {
            if let Some(cursor) = kv.cursors.get_mut(&(id, at)) {
                cursor.reset();
            }
            kv.clear_rank(id, at);
        }
        kv.note_resync(reason);
    }

    fn cursor(&self, id: WorkerId, rank: u16) -> Option<KvCursor> {
        self.kv
            .lock()
            .cursors
            .get(&(id, rank))
            .map(RankCursor::view)
    }

    fn removed(&self, id: WorkerId) {
        let mut kv = self.kv.lock();
        kv.cursors.retain(|(worker, _), _| *worker != id);
        kv.applied.retain(|(worker, _), _| *worker != id);
    }

    fn block_size(&self, model: &str) -> Option<u32> {
        self.kv.lock().block_sizes.get(model).copied()
    }

    fn set_block_size(&self, model: &str, size: u32) {
        let mut kv = self.kv.lock();
        let of_record = kv.block_sizes.entry(model.to_string()).or_insert(size);
        if size > *of_record {
            *of_record = size;
        }
    }
}

// ----------------------------------------------------------- reservations

#[derive(Default)]
struct Reservations {
    held: FxHashMap<(WorkerId, ReservationKind), u32>,
    by_lease: FxHashMap<LeaseId, Vec<(WorkerId, ReservationKind, u32)>>,
}

impl ReservationStore for MemoryStore {
    fn window(&self, id: WorkerId, _kind: ReservationKind) -> Option<u32> {
        // Both kinds read the engine's admission window today; a configured
        // per-side cap is the host's until the window moves into the store.
        self.workers.lock().rows.get(&id)?.row.facts.max_running
    }

    fn try_reserve(&self, id: WorkerId, kind: ReservationKind, count: u32, lease: LeaseId) -> bool {
        let window = self.window(id, kind);
        let mut reservations = self.reservations.lock();
        let holds_kind = reservations
            .by_lease
            .get(&lease)
            .is_some_and(|held| held.iter().any(|(_, k, _)| *k == kind));
        if holds_kind {
            return false;
        }
        let held = reservations.held.entry((id, kind)).or_insert(0);
        let total = held.saturating_add(count);
        if window.is_some_and(|window| total > window) {
            return false;
        }
        *held = total;
        reservations
            .by_lease
            .entry(lease)
            .or_default()
            .push((id, kind, count));
        true
    }

    fn release(&self, lease: LeaseId) {
        let mut reservations = self.reservations.lock();
        let Some(held) = reservations.by_lease.remove(&lease) else {
            return;
        };
        for (id, kind, count) in held {
            if let Some(total) = reservations.held.get_mut(&(id, kind)) {
                *total = total.saturating_sub(count);
            }
        }
    }

    fn held(&self, id: WorkerId, kind: ReservationKind) -> u32 {
        self.reservations
            .lock()
            .held
            .get(&(id, kind))
            .copied()
            .unwrap_or(0)
    }
}

// --------------------------------------------------------------- sessions

impl SessionsStore for MemoryStore {
    fn pin(&self, key: u64, worker: WorkerId, at: Instant) {
        if key == 0 {
            return;
        }
        self.sessions.lock().insert(
            key,
            Pin {
                worker,
                last_used: at,
            },
        );
    }

    fn lookup(&self, key: u64) -> Option<Pin> {
        self.sessions.lock().get(&key).copied()
    }

    fn unpin(&self, key: u64) {
        self.sessions.lock().remove(&key);
    }

    fn drop_worker(&self, worker: WorkerId) -> usize {
        let mut sessions = self.sessions.lock();
        let before = sessions.len();
        sessions.retain(|_, pin| pin.worker != worker);
        before - sessions.len()
    }

    fn expire(&self, before: Instant) -> usize {
        let mut sessions = self.sessions.lock();
        let count = sessions.len();
        sessions.retain(|_, pin| pin.last_used >= before);
        count - sessions.len()
    }

    fn merge(&self, key: u64, pin: Pin) {
        if key == 0 {
            return;
        }
        let mut sessions = self.sessions.lock();
        let entry = sessions.entry(key).or_insert(pin);
        if pin.last_used > entry.last_used {
            *entry = pin;
        }
    }

    fn len(&self) -> usize {
        self.sessions.lock().len()
    }
}

// ----------------------------------------------------------------- leases

#[derive(Default)]
struct Leases {
    term: u64,
    records: FxHashMap<(GatewayId, LeaseId), LeaseRecord>,
}

impl LeasesStore for MemoryStore {
    fn term(&self) -> u64 {
        self.leases.lock().term
    }

    fn set_term(&self, term: u64) {
        let mut leases = self.leases.lock();
        leases.term = leases.term.max(term);
    }

    fn issue(&self, record: LeaseRecord) -> Result<(), LeaseRefused> {
        let mut leases = self.leases.lock();
        if record.term > leases.term {
            return Err(LeaseRefused::TermAhead {
                lease: record.term,
                store: leases.term,
            });
        }
        let key = (record.gateway, record.lease.id);
        if leases.records.contains_key(&key) {
            return Err(LeaseRefused::Duplicate);
        }
        leases.records.insert(key, record);
        Ok(())
    }

    fn dispatched(&self, gateway: GatewayId, lease: LeaseId, _side: Side, at: Instant) -> bool {
        let mut leases = self.leases.lock();
        let Some(record) = leases.records.get_mut(&(gateway, lease)) else {
            return false;
        };
        if record.state == LeaseState::Issued {
            record.state = LeaseState::Dispatched { at };
        }
        true
    }

    fn completed(&self, gateway: GatewayId, lease: LeaseId, outcome: &Outcome) -> bool {
        let mut leases = self.leases.lock();
        let Some(record) = leases.records.get(&(gateway, lease)) else {
            return false;
        };
        // A pair's room lives until its decode side ends: the prefill side's
        // completion leaves the record standing.
        if record.decode.is_some() && outcome.side == Side::Prefill {
            return true;
        }
        leases.records.remove(&(gateway, lease));
        true
    }

    fn released(&self, gateway: GatewayId, lease: LeaseId, _reason: ReleaseReason) -> bool {
        self.leases
            .lock()
            .records
            .remove(&(gateway, lease))
            .is_some()
    }

    fn get(&self, gateway: GatewayId, lease: LeaseId) -> Option<LeaseRecord> {
        self.leases.lock().records.get(&(gateway, lease)).copied()
    }

    fn expire(&self, now: Instant) -> Vec<LeaseRecord> {
        let mut leases = self.leases.lock();
        let gone: Vec<(GatewayId, LeaseId)> = leases
            .records
            .iter()
            .filter(|(_, record)| {
                record.state == LeaseState::Issued && record.dispatch_deadline <= now
            })
            .map(|(key, _)| *key)
            .collect();
        gone.into_iter()
            .filter_map(|key| leases.records.remove(&key))
            .map(|mut record| {
                record.state = LeaseState::Released(ReleaseReason::Abandoned);
                record
            })
            .collect()
    }

    fn restore(&self, gateway: GatewayId, records: &[LeaseRecord]) -> usize {
        let mut leases = self.leases.lock();
        let term = leases.term;
        leases.records.retain(|(owner, _), _| *owner != gateway);
        for record in records {
            if record.gateway != gateway || record.term > term {
                continue;
            }
            leases.records.insert((gateway, record.lease.id), *record);
        }
        leases
            .records
            .keys()
            .filter(|(owner, _)| *owner == gateway)
            .count()
    }

    fn drop_gateway(&self, gateway: GatewayId) -> usize {
        let mut leases = self.leases.lock();
        let before = leases.records.len();
        leases.records.retain(|(owner, _), _| *owner != gateway);
        before - leases.records.len()
    }

    fn live(&self, gateway: GatewayId) -> usize {
        self.leases
            .lock()
            .records
            .iter()
            .filter(|((owner, _), record)| {
                *owner == gateway
                    && matches!(
                        record.state,
                        LeaseState::Issued | LeaseState::Dispatched { .. }
                    )
            })
            .count()
    }
}
