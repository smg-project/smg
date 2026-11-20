//! The worker load sink over the store.
//!
//! The collector holds its reports lock at a record, a poll's report or a
//! forget, so a later object's first sample never lands between an
//! object's forget and its eviction. Its poll question and a failed fetch
//! run outside that lock, and neither lays a sample an eviction could wipe.
//! Where the collector holds the lock, the adapter takes its member map
//! lock inside it, then reads the registry. On a miss it also reads the
//! holder, under the map's write lock. It releases the map before calling
//! the store, which takes its own locks and may pay for a membership build
//! on a poll answer. What a worker over several ranks stated has a lock of
//! its own, in the worker's entry. It is taken behind the map's release
//! and held across the store's calls for that worker's values: the read
//! of another rank's row, and the writes. The store calls no adapter, so
//! nothing takes the two the other way. The entry's note of its last
//! report of one rank has a lock taken inside the map's, which holds no
//! other. The adapter calls no collector.

use std::{
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use parking_lot::{Mutex, RwLock};
use rustc_hash::FxHashMap;
use smg_worker::{
    identity::basic::Worker,
    load::port::{LoadSink, PollMode, PolledMember, PolledRanks},
    registry::{Generation, WorkerId as ObjectId},
    worker::{ConnectionMode, WorkerLoadResponse, WorkerStatus as ObjectStatus},
};

use super::{object::role, Lead, Objects};
use crate::{
    load_report::{load_row, signal_values, RankValues, ReportOf},
    signals::SignalColumn,
    store::{LoadSink as StoreLoadSink, LoadSource, PollMode as StorePollMode, SignalSink},
    view::{Role, SignalSource, WorkerId},
};

#[derive(Clone, Copy)]
struct Member {
    row: WorkerId,
    rank: u16,
    role: Role,
}

struct KeptMember {
    worker: Arc<dyn Worker>,
    id: ObjectId,
    member: Member,
    /// The rank and the sample instant of the last report of one rank the
    /// worker's report took, while the entry has no keeper: what dates
    /// that rank's entry at the first report of several ranks.
    alone: Mutex<Option<(i32, Instant)>>,
    /// What each rank of the worker last stated of the declared signals,
    /// made at the first sign of several ranks. It goes with the entry.
    keeper: OnceLock<Keeper>,
}

/// What each rank of one worker last stated, behind the worker's own lock.
type Keeper = Arc<Mutex<RankValues>>;

/// A report's part in what its worker's entry keeps of the ranks.
struct Heard<'a> {
    report: &'a WorkerLoadResponse,
    columns: &'a [SignalColumn],
    /// The report's sample instant, where the worker's report took it: a
    /// pushed record always, a polled answer once the store took its row.
    taken_at: Option<Instant>,
}

impl KeptMember {
    /// The entry's keeper for `heard`'s report: the one it has, or one
    /// made at the first sign that the worker stands over several ranks.
    /// The signs are a report of several ranks, and a report of one rank
    /// behind a report of another rank. Until then a report of one rank
    /// leaves its rank and its instant in the entry, and nothing is
    /// allocated. Under a document that declares no signal nothing is
    /// read.
    fn keeper_for(&self, heard: &Heard<'_>) -> Option<Keeper> {
        if heard.columns.is_empty() {
            return None;
        }
        if let Some(keeper) = self.keeper.get() {
            return Some(Arc::clone(keeper));
        }
        let mut alone = self.alone.lock();
        let only = match heard.report.loads.as_slice() {
            [only] if only.worker.is_none() => Some(only.dp_rank),
            _ => None,
        };
        let another = only
            .zip(*alone)
            .is_some_and(|(rank, (last, _))| rank != last);
        if another || RankValues::keeps(heard.report, heard.columns) {
            let behind = || Arc::new(Mutex::new(RankValues::behind(*alone)));
            return Some(Arc::clone(self.keeper.get_or_init(behind)));
        }
        if let Some(taken) = only.zip(heard.taken_at) {
            *alone = Some(taken);
        }
        None
    }
}

/// The worker crate's load port over a store's load sink.
///
/// The overload verdict lives on the store's rows. The adapter leaves the
/// object's overload flag alone. A rank worker's row uses its own rank.
/// For any other worker the row covers the whole report, and a record is
/// written at the rank it came from. The store reads a worker that is no
/// rank at its own rank alone: where such a worker stands over an engine
/// of several ranks, the records of its other ranks are written and not
/// read, so its load under-reads until its registration states the width.
/// An unready rank counts as served when checking an empty
/// answer, so the engine's verdict stands until its next record or report.
/// An object that is not `Ready` lays no sample, by a record or by a poll's
/// answer: its forget took its row's loads, and they stay off the file.
/// Loads are a row's, not an object's. A forgotten object's row is evicted
/// and its engine judged again, unless a later object was heard or asked
/// under the row first: the loads are that object's then. Behind a reset
/// no entry says whose they are, and a replaced object's forget keeps them.
/// A reset drops the loads of every row that is not `Ready`, by the row's
/// own status.
/// The host publishes the store's pending writes.
///
/// Behind each row go the values its report states of the signals the
/// document in force declares ([`signal_values`]), written to the store's
/// signal sink at the row's rank. A polled row the store passed over
/// takes its values with it. Under a document that declares no signal a
/// record reads one empty list more.
///
/// A worker over several ranks reports them at as many instants: a record
/// is one rank's, and its report lists every rank at its latest record. So
/// that worker's values are made over what each rank last stated, each
/// statement under its own sample instant ([`RankValues`]). The statements
/// are kept in the worker's entry from its first sign of several ranks: a
/// report of several ranks, or a report of one rank behind a report of
/// another. From then on the keeper is told every report of the worker.
/// Until then a report of one rank leaves its rank and its instant in the
/// entry, which dates that rank's entry at the first report of several
/// ranks: a worker's first records list one rank, and a worker of one
/// rank allocates nothing. The statements go with the entry at a forget,
/// at a reset and when another object takes its address, and end at a new
/// document. A rank's entry the keeper was told nothing of is read under
/// that rank's own record as the store holds its row, and a poll's row in
/// a rank's place dates none. A polled answer is one sample of every rank
/// it lists, and states them all, though its row is passed over.
pub struct LoadAdapter<G, S: ?Sized> {
    objects: Arc<Objects<G>>,
    store: Arc<S>,
    lead: Arc<dyn Lead>,
    term: u64,
    /// One entry per URL heard or polled. Forget drops the object's entry
    /// and reset clears the map when registry events were lost.
    members: RwLock<FxHashMap<String, KeptMember>>,
    period_taken: bool,
}

impl<G: Generation, S: StoreLoadSink + SignalSink + ?Sized> LoadAdapter<G, S> {
    /// For the lead under `term`. Tells the store once how often the
    /// collector hears each ready rank, by its stream or a poll. The store
    /// keeps the first period it is told: [`Self::period_taken`] says
    /// whether this one stands.
    pub fn new(
        objects: Arc<Objects<G>>,
        store: Arc<S>,
        lead: Arc<dyn Lead>,
        term: u64,
        heard_every: Duration,
    ) -> Self {
        let period_taken = store.heard_every(heard_every);
        Self {
            objects,
            store,
            lead,
            term,
            members: RwLock::default(),
            period_taken,
        }
    }

    /// Whether the store took this adapter's period. `false` where another
    /// period stood first: the store keeps that one and ages a rank's
    /// sample by it.
    pub fn period_taken(&self) -> bool {
        self.period_taken
    }

    fn leads(&self) -> bool {
        self.lead.term() == Some(self.term)
    }

    fn is_current(&self, id: &ObjectId, worker: &Arc<dyn Worker>) -> bool {
        self.objects
            .registry()
            .get(id)
            .is_some_and(|current| Arc::ptr_eq(&current, worker))
    }

    fn read_member(&self, worker: &Arc<dyn Worker>) -> Option<KeptMember> {
        let id = self.objects.registry().get_id_by_url(worker.url())?;
        let standing = self.objects.standing_of(&id, worker)?;
        if !self.is_current(&id, worker) {
            return None;
        }
        Some(KeptMember {
            worker: Arc::clone(worker),
            id,
            member: Member {
                row: standing.row,
                rank: u16::try_from(worker.dp_rank().unwrap_or(0)).ok()?,
                role: role(*worker.worker_type()),
            },
            alone: Mutex::new(None),
            keeper: OnceLock::new(),
        })
    }

    fn member(&self, worker: &Arc<dyn Worker>) -> Option<Member> {
        self.member_heard(worker, None).map(|(member, _)| member)
    }

    /// The member of `worker`, and for a report its entry's keeper, where
    /// the entry has one or the report makes one ([`KeptMember::keeper_for`]):
    /// one lookup says both.
    fn member_heard(
        &self,
        worker: &Arc<dyn Worker>,
        heard: Option<&Heard<'_>>,
    ) -> Option<(Member, Option<Keeper>)> {
        let url = worker.url();
        let found = |kept: &KeptMember| {
            let keeper = heard.and_then(|heard| kept.keeper_for(heard));
            (kept.member, keeper)
        };
        if let Some(kept) = self.members.read().get(url) {
            // The registry can retire the cached object before the collector
            // sees the event. The kept id lets this check borrow the key.
            if Arc::ptr_eq(&kept.worker, worker) && self.is_current(&kept.id, &kept.worker) {
                return Some(found(kept));
            }
        }
        let mut members = self.members.write();
        let kept = self.read_member(worker)?;
        let member = found(&kept);
        members.insert(url.to_string(), kept);
        Some(member)
    }

    /// A polled answer of one rank whose row the store took: its rank and
    /// instant go to the worker's entry, where the entry has no keeper.
    fn heard_alone(&self, worker: &Arc<dyn Worker>, heard: &Heard<'_>) {
        let members = self.members.read();
        if let Some(kept) = members.get(worker.url()) {
            if Arc::ptr_eq(&kept.worker, worker) {
                kept.keeper_for(heard);
            }
        }
    }

    fn write_report<'a>(
        &self,
        answer: impl Iterator<Item = (&'a Arc<dyn Worker>, Option<&'a WorkerLoadResponse>)>,
        polled_at: Instant,
        received_at: Instant,
    ) -> Vec<i32> {
        if !self.leads() {
            return Vec::new();
        }
        let mut answer = answer.peekable();
        let Some((first, _)) = answer.peek() else {
            return Vec::new();
        };
        let width = first.dp_size().unwrap_or(1);
        let source = match first.connection_mode() {
            ConnectionMode::Grpc => LoadSource::Poll,
            ConnectionMode::Http | ConnectionMode::Zmq => LoadSource::Scrape,
        };
        let columns = self.store.signal_columns();
        // Each row's worker and report, kept past its row where the
        // document declares a signal.
        let mut stated = Vec::new();
        let rows: Vec<_> = answer
            .filter_map(|(worker, report)| {
                // A member whose status left `Ready` while its poll was out
                // lays no sample either: it stands as a rank not asked.
                if worker.status() != ObjectStatus::Ready {
                    return None;
                }
                let report = report.filter(|report| !report.loads.is_empty());
                // The answer's part in what the worker's entry keeps of
                // its ranks. Whether the store takes its row is known
                // behind the rows.
                let heard = report.map(|report| Heard {
                    report,
                    columns: &columns,
                    taken_at: None,
                });
                let (member, keeper) = self.member_heard(worker, heard.as_ref())?;
                let row = report.map(|report| {
                    load_row(report, member.role, received_at, received_at, source.into())
                });
                if !columns.is_empty() {
                    stated.push((worker, report, keeper));
                }
                Some((member.row, member.rank, row))
            })
            .collect();
        if rows.is_empty() || (rows.len() != width && rows.iter().all(|(_, _, row)| row.is_none()))
        {
            return Vec::new();
        }
        let passed = self.store.report(&rows, polled_at, source);
        // The rows first, then their values, each dated as its row is: at
        // the poll's start. The positions the store passed over stand in
        // the answer's order.
        let mut skipped = passed.iter().copied().peekable();
        for (position, (&(row, rank, _), (worker, report, keeper))) in
            rows.iter().zip(stated).enumerate()
        {
            let passed_over = skipped.next_if_eq(&position).is_some();
            let Some(report) = report else {
                continue;
            };
            let write = |column: &SignalColumn, unit: &str, value: f64, at: Instant| {
                self.store.signal_value(row, rank, column, unit, value, at);
            };
            let Some(keeper) = keeper else {
                // A worker that showed one rank: its answer is one sample.
                // Where the store took its row, its rank and its instant
                // stay in the worker's entry.
                if !passed_over {
                    signal_values(report, &columns, polled_at, write);
                    let taken = Heard {
                        report,
                        columns: &columns,
                        taken_at: Some(polled_at),
                    };
                    self.heard_alone(worker, &taken);
                }
                continue;
            };
            // An answer is one sample of each rank it lists: the keeper is
            // told them all, though the store passed its row over.
            let mut keeper = keeper.lock();
            if passed_over {
                keeper.keep(report, ReportOf::EveryRank, &columns, polled_at);
            } else {
                keeper.values(report, ReportOf::EveryRank, &columns, polled_at, write);
            }
        }
        passed
            .into_iter()
            .filter_map(|position| rows.get(position))
            .map(|&(_, rank, _)| i32::from(rank))
            .collect()
    }
}

impl<G, S> LoadSink for LoadAdapter<G, S>
where
    G: Generation,
    S: StoreLoadSink + SignalSink + ?Sized + 'static,
{
    fn record(
        &self,
        worker: &Arc<dyn Worker>,
        rank: i32,
        report: &WorkerLoadResponse,
        received_at: Instant,
        sampled_at: Instant,
    ) {
        // A worker that is not `Ready` still feeds the collector's sink.
        // The store judges every rank on file, so its record would lay
        // again the sample its forget took off, and hold the engine's
        // verdict over the ranks that stay. The collector reads the status
        // and says the forget under one hold: a record that reads `Ready`
        // lands ahead of the eviction, and a later one is dropped.
        if !self.leads() || worker.status() != ObjectStatus::Ready {
            return;
        }
        let Ok(rank) = u16::try_from(rank) else {
            return;
        };
        // What the document in force declares, read ahead of the lookup:
        // the record's part in what the worker's entry keeps of its ranks.
        let columns = self.store.signal_columns();
        let heard = Heard {
            report,
            columns: &columns,
            taken_at: Some(sampled_at),
        };
        let Some((member, keeper)) = self.member_heard(worker, Some(&heard)) else {
            return;
        };
        let row = load_row(
            report,
            member.role,
            received_at,
            sampled_at,
            SignalSource::Events,
        );
        self.store
            .record(member.row, rank, row, received_at, LoadSource::Stream);
        // The row first, then its values.
        let write = |column: &SignalColumn, unit: &str, value: f64, at: Instant| {
            self.store
                .signal_value(member.row, rank, column, unit, value, at);
        };
        let Some(keeper) = keeper else {
            signal_values(report, &columns, sampled_at, write);
            return;
        };
        // The worker stands over several ranks, and its report lists each
        // at its latest record: this record states its own rank. Another
        // rank's entry is read where nothing of it is kept yet, under the
        // instant the keeper was told, else under that rank's own last
        // record as the store holds its row.
        let heard = |other: i32| {
            let other = u16::try_from(other).ok()?;
            let sample = self.store.load(member.row, other)?;
            // A poll's row stands at the member's own rank, whatever
            // ranks its answer listed: it is no record of that rank.
            matches!(sample.source, LoadSource::Stream).then_some(sample.sampled_at)
        };
        let of = ReportOf::OneRank {
            rank: i32::from(rank),
            heard: &heard,
        };
        keeper
            .lock()
            .values(report, of, &columns, sampled_at, write);
    }

    fn poll_mode(&self, worker: &Arc<dyn Worker>, now: Instant, interval: Duration) -> PollMode {
        if !self.leads() {
            return PollMode::Suppressed;
        }
        let Some(member) = self.member(worker) else {
            return PollMode::Suppressed;
        };
        match self.store.poll_mode(member.row, now, interval) {
            StorePollMode::Poll => PollMode::Poll,
            StorePollMode::Fallback => PollMode::Fallback,
            StorePollMode::Suppressed => PollMode::Suppressed,
        }
    }

    fn report(&self, answer: &[PolledMember<'_>], polled_at: Instant) -> PolledRanks {
        let newer = self.write_report(
            answer.iter().map(|member| (member.worker, member.answer)),
            polled_at,
            Instant::now(),
        );
        PolledRanks {
            newer,
            leaving: Vec::new(),
        }
    }

    fn report_empty(&self, answer: &[PolledMember<'_>], polled_at: Instant) -> Vec<i32> {
        self.write_report(
            answer.iter().map(|member| (member.worker, None)),
            polled_at,
            polled_at,
        );
        Vec::new()
    }

    fn fetch_failed(&self, members: &[Arc<dyn Worker>], polled_at: Instant) {
        self.write_report(
            members.iter().map(|worker| (worker, None)),
            polled_at,
            polled_at,
        );
    }

    fn forget(&self, worker: &Arc<dyn Worker>) {
        // The entry goes whatever the term: it holds the worker object, and
        // with it the object's engine connection.
        let kept = {
            let mut members = self.members.write();
            if members
                .get(worker.url())
                .is_some_and(|kept| Arc::ptr_eq(&kept.worker, worker))
            {
                members.remove(worker.url())
            } else {
                None
            }
        };
        if !self.leads() {
            return;
        }
        // Loads are a row's, and the entry says whose stand on it: a later
        // object's first call puts its own entry at the address. Under this
        // object's own entry none has laid a sample, so the loads go
        // whatever stands under the row, and the engine is judged again: a
        // replacement that is never asked would hold them.
        // With no entry of its own, behind a reset or a later object's
        // call, they go only where this object still stands under the row.
        let Some(kept) = kept.or_else(|| self.read_member(worker)) else {
            return;
        };
        self.store.evict(kept.member.row);
    }

    fn reset(&self) {
        // The loads of a row that is `Ready` stay. They end with the write
        // that removes their worker, and a reset at a term's start would
        // wipe the loads a standby followed.
        self.members.write().clear();
        // The collector resets behind events it lost. A worker that left
        // `Ready` among them was never forgotten, so the store drops the
        // loads of every row that is not `Ready`. The status read is the
        // row's: a standby followed it, and its objects may not have.
        if self.leads() {
            self.store.evict_unready();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::slice;

    use parking_lot::Mutex;
    use smg_worker::{
        load::port::GroupTick,
        registry::WorkerRegistry,
        worker::{LoadTelemetry, NamedValue, SchedulerLoadSnapshot, WorkerStatus as ObjectStatus},
    };

    use super::*;
    use crate::{
        spec::{AcrossRanks, AppliedSpec, PolicySpec, SignalSpec},
        store::{
            Ask, GatewayId, LoadSample, LoadSource, MemoryStore, Overload,
            PollMode as StorePollMode, Taken, ViewBuffers, WorkerFacts, WorkerPatch, WorkerStatus,
            WorkersStore,
        },
        view::{LoadRow, Role, SignalSource, WireModes, WorkerId},
        worker_host::{object, Members, SoleLead},
    };

    const URL: &str = "http://loads:8000";
    const MODEL: &str = "m";
    const PERIOD: Duration = Duration::from_secs(2);

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct Calls {
        records: usize,
        reports: usize,
        polls: usize,
        evicts: usize,
        periods: usize,
        period: Option<Duration>,
    }

    /// A real store that also counts calls which might change no row.
    struct Store {
        memory: MemoryStore,
        calls: Mutex<Calls>,
    }

    impl StoreLoadSink for Store {
        fn record(
            &self,
            id: WorkerId,
            rank: u16,
            row: LoadRow,
            received_at: Instant,
            source: LoadSource,
        ) {
            self.calls.lock().records += 1;
            self.memory.record(id, rank, row, received_at, source);
        }

        fn report(
            &self,
            answer: &[(WorkerId, u16, Option<LoadRow>)],
            polled_at: Instant,
            source: LoadSource,
        ) -> Vec<usize> {
            self.calls.lock().reports += 1;
            self.memory.report(answer, polled_at, source)
        }

        fn poll_mode(&self, id: WorkerId, now: Instant, interval: Duration) -> StorePollMode {
            self.calls.lock().polls += 1;
            self.memory.poll_mode(id, now, interval)
        }

        fn heard_every(&self, period: Duration) -> bool {
            let mut calls = self.calls.lock();
            calls.periods += 1;
            calls.period = Some(period);
            self.memory.heard_every(period)
        }

        fn load(&self, id: WorkerId, rank: u16) -> Option<LoadSample> {
            self.memory.load(id, rank)
        }

        fn ranks(&self, id: WorkerId) -> usize {
            self.memory.ranks(id)
        }

        fn evict(&self, id: WorkerId) {
            self.calls.lock().evicts += 1;
            self.memory.evict(id);
        }

        fn evict_unready(&self) {
            self.memory.evict_unready();
        }
    }

    impl SignalSink for Store {
        fn signal_columns(&self) -> Arc<[SignalColumn]> {
            self.memory.signal_columns()
        }

        fn signal_value(
            &self,
            id: WorkerId,
            rank: u16,
            column: &SignalColumn,
            unit: &str,
            value: f64,
            sampled_at: Instant,
        ) -> Taken {
            self.memory
                .signal_value(id, rank, column, unit, value, sampled_at)
        }
    }

    struct Replica {
        objects: Arc<Objects<Members>>,
        store: Arc<Store>,
    }

    impl Replica {
        fn new() -> Self {
            Self {
                objects: Arc::new(Objects::new(Arc::new(WorkerRegistry::new()))),
                store: Arc::new(Store {
                    memory: MemoryStore::with_publish_window(Duration::ZERO),
                    calls: Mutex::default(),
                }),
            }
        }

        fn adapter(&self, lead: Arc<dyn Lead>, term: u64) -> LoadAdapter<Members, Store> {
            LoadAdapter::new(
                Arc::clone(&self.objects),
                self.store.clone(),
                lead,
                term,
                PERIOD,
            )
        }

        fn lead(&self) -> LoadAdapter<Members, Store> {
            self.adapter(Arc::new(SoleLead), SoleLead::TERM)
        }

        fn add(&self, facts: WorkerFacts) -> (WorkerId, Arc<dyn Worker>) {
            let id = self.store.memory.upsert(facts).id();
            let row = WorkersStore::get(&self.store.memory, id).expect("registered row");
            self.store
                .memory
                .set_status(id, WorkerStatus::Ready, row.revision);
            self.store.memory.publish_now();
            self.objects.follow(&self.store.memory);
            (id, self.at(&row.facts.key))
        }

        fn at(&self, url: &str) -> Arc<dyn Worker> {
            self.objects.registry().get_by_url(url).expect("an object")
        }

        fn replace(&self, id: WorkerId) -> Arc<dyn Worker> {
            let row = WorkersStore::get(&self.store.memory, id).expect("a row");
            let patch = WorkerPatch {
                priority: Some(20),
                ..Default::default()
            };
            assert!(self.store.memory.patch(id, patch, row.revision).is_some());
            self.objects.follow(&self.store.memory);
            self.at(&row.facts.key)
        }

        fn two_ranks(&self, waiting_requests: Option<u32>) -> [(WorkerId, Arc<dyn Worker>); 2] {
            [0, 1].map(|rank| {
                self.add(WorkerFacts {
                    key: format!("{URL}@{rank}"),
                    dp_size: 2,
                    dp_rank: rank,
                    overload: Overload {
                        waiting_requests,
                        ..Default::default()
                    },
                    ..facts(URL)
                })
            })
        }

        fn overloaded(&self, id: WorkerId, now: Instant) -> bool {
            let mut out = ViewBuffers::default();
            let ask = Ask {
                gateway: GatewayId(1),
                now,
                pinned: None,
                affinity: false,
                signal_cells: false,
            };
            self.store
                .memory
                .world()
                .build(MODEL, Role::Single, &ask, &mut out)
                .expect("the pool");
            out.candidates
                .iter()
                .find(|row| row.id == id)
                .expect("a candidate")
                .health
                .overloaded
        }

        /// Show what the store's load writes left waiting, as its host
        /// does.
        fn publish(&self) {
            if let Some(due) = self.store.memory.publish_due() {
                self.store.memory.publish(due);
            }
        }
    }

    fn facts(url: &str) -> WorkerFacts {
        WorkerFacts {
            key: url.to_string(),
            engine: url.to_string(),
            incarnation: 1,
            models: vec![MODEL.to_string()],
            dp_size: 1,
            wire_modes: WireModes::GRPC,
            ..Default::default()
        }
    }

    fn report(rank: i32, running: i32) -> WorkerLoadResponse {
        WorkerLoadResponse {
            dp_rank_count: 1,
            loads: vec![SchedulerLoadSnapshot {
                dp_rank: rank,
                num_running_reqs: running,
                num_waiting_reqs: running,
                gen_throughput: 30.0,
                prefill_throughput: Some(900.0),
                // A rank's core: it states neither fixed token count.
                kept: LoadTelemetry::USED_TOKENS | LoadTelemetry::TOTAL_TOKENS,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn polled<'a>(
        worker: &'a Arc<dyn Worker>,
        answer: Option<&'a WorkerLoadResponse>,
    ) -> PolledMember<'a> {
        PolledMember {
            worker,
            answer,
            on_file: None,
        }
    }

    fn busy_pair(
        replica: &Replica,
        adapter: &impl LoadSink,
        now: Instant,
    ) -> [(WorkerId, Arc<dyn Worker>); 2] {
        let pair = replica.two_ranks(Some(5));
        adapter.record(&pair[0].1, 0, &report(0, 6), now, now);
        adapter.record(&pair[1].1, 1, &report(1, 1), now, now);
        for (id, worker) in &pair {
            assert!(replica.overloaded(*id, now));
            assert!(!worker.is_overloaded());
        }
        pair
    }

    #[test]
    fn a_record_writes_the_ranks_row_with_the_collectors_sample_instant() {
        for (role, rate) in [
            (Role::Single, 30),
            (Role::Prefill, 900),
            (Role::Decode, 30),
            (Role::Encode, 30),
        ] {
            let replica = Replica::new();
            let (id, worker) = replica.add(WorkerFacts {
                dp_rank: 1,
                dp_size: 2,
                role,
                ..facts(URL)
            });
            let adapter = replica.lead();
            let sampled = Instant::now();
            let received = sampled + Duration::from_millis(37);
            adapter.record(&worker, 1, &report(1, 5), received, sampled);
            let row = replica.store.load(id, 1).expect("the pushed row");
            assert_eq!((row.received_at, row.sampled_at), (received, sampled));
            assert_eq!(row.row.report_age_ms, 37);
            assert_eq!(
                (row.source, row.row.source),
                (LoadSource::Stream, SignalSource::Events)
            );
            assert_eq!((row.row.running, row.row.live_rate), (5, rate));
            assert_eq!(
                adapter.member(&worker).map(|member| member.role),
                Some(role)
            );
            assert_eq!(replica.store.ranks(id), 1);
        }
    }

    #[test]
    fn a_record_uses_its_given_rank_if_it_fits() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        let adapter = replica.lead();
        let now = Instant::now();
        let last = i32::from(u16::MAX);
        adapter.record(&worker, last, &report(last, 2), now, now);
        for rank in [-1, last + 1] {
            adapter.record(&worker, rank, &report(rank, 9), now, now);
        }
        assert_eq!(replica.store.calls.lock().records, 1);
        assert_eq!(
            replica
                .store
                .load(id, u16::MAX)
                .map(|sample| sample.row.running),
            Some(2)
        );
        assert_eq!(replica.store.ranks(id), 1);
    }

    #[test]
    fn a_record_of_a_replaced_object_is_passed_over() {
        let replica = Replica::new();
        let (id, old) = replica.add(facts(URL));
        let adapter = replica.lead();
        let now = Instant::now();
        adapter.record(&old, 0, &report(0, 1), now, now);
        let before = replica.store.load(id, 0).expect("the old row");

        let replacement = replica.replace(id);
        adapter.record(&old, 0, &report(0, 9), now, now);
        assert_eq!(replica.store.load(id, 0), Some(before));

        adapter.record(&replacement, 0, &report(0, 2), now, now);
        adapter.record(&old, 0, &report(0, 9), now, now);
        assert_eq!(
            replica.store.load(id, 0).map(|sample| sample.row.running),
            Some(2)
        );
        assert_eq!(replica.store.calls.lock().records, 2);

        let (incarnation, _) = replica.add(WorkerFacts {
            incarnation: 2,
            ..facts(URL)
        });
        assert_ne!(incarnation, id);
        adapter.record(&replacement, 0, &report(0, 9), now, now);
        assert_eq!(
            adapter.poll_mode(&replacement, now, PERIOD),
            PollMode::Suppressed
        );
        assert_eq!(replica.store.calls.lock().records, 2);
        assert_eq!(replica.store.ranks(id), 0);
        assert_eq!(replica.store.ranks(incarnation), 0);
    }

    #[test]
    fn a_record_of_a_worker_with_no_row_is_passed_over() {
        let replica = Replica::new();
        let (id, paired) = replica.add(facts(URL));
        let row = WorkersStore::get(&replica.store.memory, id).expect("a row");
        let unpaired = object(&row);
        let adapter = replica.lead();
        let now = Instant::now();
        let load = report(0, 9);
        let before = *replica.store.calls.lock();
        adapter.record(&unpaired, 0, &load, now, now);
        assert_eq!(
            adapter.poll_mode(&unpaired, now, PERIOD),
            PollMode::Suppressed
        );
        assert_eq!(
            adapter.report(&[polled(&unpaired, Some(&load))], now),
            PolledRanks::default()
        );
        assert!(adapter
            .report_empty(&[polled(&unpaired, None)], now)
            .is_empty());
        adapter.fetch_failed(slice::from_ref(&unpaired), now);
        adapter.forget(&unpaired);
        assert_eq!(*replica.store.calls.lock(), before);
        assert_eq!(replica.store.ranks(id), 0);
        assert!(adapter.members.read().is_empty());

        let registry = replica.objects.registry();
        let object_id = registry.get_id_by_url(URL).expect("a registry id");
        assert!(registry.replace(&object_id, unpaired));
        adapter.record(&paired, 0, &load, now, now);
        assert_eq!(*replica.store.calls.lock(), before);
    }

    #[test]
    fn the_poll_question_is_the_stores() {
        let replica = Replica::new();
        let (_, worker) = replica.add(facts(URL));
        let adapter = replica.lead();
        let now = Instant::now();
        assert_eq!(adapter.poll_mode(&worker, now, PERIOD), PollMode::Poll);
        adapter.record(&worker, 0, &report(0, 2), now, now);
        assert_eq!(
            adapter.poll_mode(&worker, now, PERIOD),
            PollMode::Suppressed
        );
        let later = now + PERIOD + Duration::from_millis(1);
        assert_eq!(
            adapter.poll_mode(&worker, later, PERIOD),
            PollMode::Fallback
        );
    }

    #[test]
    fn a_polls_answer_is_one_write_and_names_the_ranks_whose_stream_is_newer() {
        let replica = Replica::new();
        let [(zero, first), (one, second)] = replica.two_ranks(None);
        let adapter = replica.lead();
        let asked = Instant::now();
        let pushed = asked + Duration::from_millis(1);
        adapter.record(&second, 1, &report(1, 7), pushed, pushed);
        let stream = replica.store.load(one, 1).expect("stream row");
        let mut first_answer = report(0, 3);
        first_answer.sampled_at = Some(asked - PERIOD);
        let second_answer = report(1, 4);
        let unpaired = object(&WorkersStore::get(&replica.store.memory, zero).expect("a row"));
        let ranks = adapter.report(
            &[
                polled(&unpaired, None),
                polled(&second, Some(&second_answer)),
                polled(&first, Some(&first_answer)),
            ],
            asked,
        );
        assert_eq!(ranks.newer, [1]);
        assert!(ranks.leaving.is_empty());
        assert_eq!(replica.store.calls.lock().reports, 1);
        assert_eq!(replica.store.load(one, 1), Some(stream));
        let sample = replica.store.load(zero, 0).expect("the poll's row");
        assert_eq!((sample.row.running, sample.row.report_age_ms), (3, 0));
        assert_eq!((sample.sampled_at, sample.received_at), (asked, asked));
        assert_eq!(adapter.poll_mode(&first, asked, PERIOD), PollMode::Poll);
    }

    #[test]
    fn a_rank_the_answer_leaves_out_keeps_its_row_and_no_rank_leaves_the_file() {
        let replica = Replica::new();
        let [(zero, first), (one, second)] = replica.two_ranks(None);
        let adapter = replica.lead();
        let now = Instant::now();
        let old = report(1, 7);
        adapter.record(&second, 1, &old, now, now);
        let before = replica.store.load(one, 1).expect("row on file");
        let asked = now + PERIOD;
        let empty = WorkerLoadResponse::default();
        let absent = PolledMember {
            on_file: Some(&old),
            ..polled(&second, Some(&empty))
        };
        let ranks = adapter.report(&[polled(&first, Some(&report(0, 3))), absent], asked);
        assert_eq!(ranks, PolledRanks::default());
        assert_eq!(replica.store.load(one, 1), Some(before));
        assert!(!replica.overloaded(zero, asked));
        assert!(replica.overloaded(one, asked));
    }

    #[test]
    fn an_empty_answer_changes_nothing_on_file_and_answers_no_rank() {
        let replica = Replica::new();
        let adapter = replica.lead();
        let now = Instant::now();
        let [(zero, first), (one, second)] = busy_pair(&replica, &adapter, now);
        let rows = [replica.store.load(zero, 0), replica.store.load(one, 1)];
        let empty = WorkerLoadResponse::default();
        let asked = now + PERIOD;
        assert!(adapter
            .report_empty(
                &[polled(&first, Some(&empty)), polled(&second, None)],
                asked
            )
            .is_empty());
        assert_eq!(
            [replica.store.load(zero, 0), replica.store.load(one, 1)],
            rows
        );
        assert!(!replica.overloaded(zero, asked));
        assert!(!replica.overloaded(one, asked));
        assert_eq!(replica.store.calls.lock().reports, 1);
    }

    #[test]
    fn an_empty_answer_of_a_poll_that_left_a_rank_out_says_nothing() {
        let replica = Replica::new();
        let adapter = replica.lead();
        let now = Instant::now();
        let [(zero, first), (one, second)] = busy_pair(&replica, &adapter, now);
        let asked = now + PERIOD;
        assert!(adapter
            .report_empty(&[polled(&second, None)], asked)
            .is_empty());
        assert_eq!(replica.store.calls.lock().reports, 0);
        assert!(replica.overloaded(zero, asked));
        assert!(replica.overloaded(one, asked));
        assert_eq!(adapter.poll_mode(&first, asked, PERIOD), PollMode::Fallback);
    }

    #[test]
    fn a_failed_fetch_is_the_word_that_no_report_came() {
        let replica = Replica::new();
        let adapter = replica.lead();
        let now = Instant::now() - PERIOD;
        let [(zero, first), (one, second)] = busy_pair(&replica, &adapter, now);
        let rows = [replica.store.load(zero, 0), replica.store.load(one, 1)];
        let asked = now + Duration::from_millis(1);
        adapter.fetch_failed(slice::from_ref(&second), asked);
        assert_eq!(replica.store.calls.lock().reports, 0);
        assert!(replica.overloaded(one, Instant::now()));
        adapter.fetch_failed(&[first, second], asked);
        assert_eq!(replica.store.calls.lock().reports, 1);
        assert_eq!(
            [replica.store.load(zero, 0), replica.store.load(one, 1)],
            rows
        );
        assert!(!replica.overloaded(one, Instant::now()));
    }

    /// A rank that pushed while the poll was out is the newer: the failed
    /// fetch's word that no report came leaves its verdict.
    #[test]
    fn a_failed_fetch_leaves_the_verdict_of_a_push_during_its_poll() {
        let replica = Replica::new();
        let adapter = replica.lead();
        let asked = Instant::now() - PERIOD;
        let pushed = asked + Duration::from_millis(1);
        let [(zero, first), (one, second)] = busy_pair(&replica, &adapter, pushed);
        adapter.fetch_failed(&[first, second], asked);
        assert_eq!(replica.store.calls.lock().reports, 1);
        assert!(replica.overloaded(zero, pushed));
        assert!(replica.overloaded(one, pushed));
    }

    /// A worker whose status left `Ready` is forgotten, and its object
    /// still stands under its row: the row's loads go with the entry.
    #[test]
    fn a_forgotten_worker_whose_status_left_ready_has_nothing_on_file() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        let adapter = replica.lead();
        let now = Instant::now();
        adapter.record(&worker, 0, &report(0, 2), now, now);
        worker.set_status(ObjectStatus::NotReady);
        adapter.forget(&worker);
        assert_eq!(replica.store.calls.lock().evicts, 1);
        assert_eq!(replica.store.ranks(id), 0);
        assert!(adapter.members.read().is_empty());
    }

    /// A busy rank that leaves `Ready` does not hold its engine's verdict
    /// over the rank that stays: its forget takes its sample off the file,
    /// and the engine is judged again. An eviction names no instant, so the
    /// judgment waits for the host's publication.
    #[test]
    fn a_rank_that_left_ready_does_not_hold_its_engines_verdict() {
        let replica = Replica::new();
        let adapter = replica.lead();
        let now = Instant::now();
        let [(zero, first), (one, _second)] = busy_pair(&replica, &adapter, now);
        first.set_status(ObjectStatus::NotReady);
        adapter.forget(&first);
        assert_eq!(replica.store.ranks(zero), 0);
        replica.publish();
        assert!(!replica.overloaded(one, now));
        // The rank drains its queue and keeps pushing: nothing is laid.
        adapter.record(&first, 0, &report(0, 6), now, now);
        assert_eq!(replica.store.ranks(zero), 0);
        replica.publish();
        assert!(!replica.overloaded(one, now));
    }

    /// A rank's status left `Ready` while its engine's poll was out. The
    /// answer lays the row of the rank that stays and none for the rank
    /// that left.
    #[test]
    fn a_poll_answer_lays_no_sample_for_a_rank_that_left_ready() {
        let replica = Replica::new();
        let adapter = replica.lead();
        let now = Instant::now();
        let [(zero, first), (one, second)] = replica.two_ranks(Some(5));
        first.set_status(ObjectStatus::NotReady);
        adapter.forget(&first);
        let (busy, idle) = (report(0, 6), report(1, 1));
        adapter.report(
            &[polled(&first, Some(&busy)), polled(&second, Some(&idle))],
            now,
        );
        assert_eq!(replica.store.ranks(zero), 0);
        assert_eq!(replica.store.ranks(one), 1);
        replica.publish();
        assert!(!replica.overloaded(one, now));
    }

    /// A reset drops the adapter's entries and keeps the store's rows. A
    /// worker forgotten behind it is read again, and its row is evicted.
    #[test]
    fn a_forget_behind_a_reset_still_evicts_its_row() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        let adapter = replica.lead();
        let now = Instant::now();
        adapter.record(&worker, 0, &report(0, 2), now, now);
        adapter.reset();
        worker.set_status(ObjectStatus::NotReady);
        adapter.forget(&worker);
        assert_eq!(replica.store.calls.lock().evicts, 1);
        assert_eq!(replica.store.ranks(id), 0);
    }

    /// A worker whose row went leaves the registry. Its forget evicts the
    /// row: nothing of it is on file.
    #[test]
    fn a_forgotten_worker_whose_row_went_has_nothing_on_file() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        let adapter = replica.lead();
        let now = Instant::now();
        adapter.record(&worker, 0, &report(0, 2), now, now);
        let row = WorkersStore::get(&replica.store.memory, id).expect("a row");
        assert!(replica.store.memory.remove(id, row.revision).is_some());
        replica.objects.follow(&replica.store.memory);
        adapter.forget(&worker);
        assert_eq!(replica.store.calls.lock().evicts, 1);
        assert_eq!(replica.store.ranks(id), 0);
        assert!(adapter.members.read().is_empty());
    }

    /// The holder replaced the object under a row that stands, and no call
    /// for the replacement has come: the entry is still the old object's,
    /// and so is the sample. Its forget evicts the row. A replacement that
    /// is never asked would hold the old sample otherwise.
    #[test]
    fn a_replaced_object_forgotten_before_its_replacement_is_heard_has_nothing_on_file() {
        let replica = Replica::new();
        let (id, old) = replica.add(facts(URL));
        let adapter = replica.lead();
        let now = Instant::now();
        adapter.record(&old, 0, &report(0, 3), now, now);
        let _replacement = replica.replace(id);
        adapter.forget(&old);
        assert_eq!(replica.store.calls.lock().evicts, 1);
        assert_eq!(replica.store.ranks(id), 0);
        assert!(adapter.members.read().is_empty());
    }

    /// The replacement's record came first: the entry is its own, and so
    /// are the row's loads. The old object's late forget evicts nothing.
    #[test]
    fn a_late_forget_of_an_old_object_keeps_the_replacements_load() {
        let replica = Replica::new();
        let (id, old) = replica.add(facts(URL));
        let adapter = replica.lead();
        let now = Instant::now();
        adapter.record(&old, 0, &report(0, 1), now, now);
        let replacement = replica.replace(id);
        adapter.record(&replacement, 0, &report(0, 2), now, now);
        adapter.forget(&old);
        assert_eq!(replica.store.calls.lock().evicts, 0);
        assert_eq!(
            replica.store.load(id, 0).map(|sample| sample.row.running),
            Some(2)
        );
        assert_eq!(adapter.members.read().len(), 1);
    }

    /// Behind a reset no entry says whose loads stand on a replaced row:
    /// the old object's forget keeps them for the object that stands.
    #[test]
    fn a_forget_of_a_replaced_object_behind_a_reset_keeps_the_rows_loads() {
        let replica = Replica::new();
        let (id, old) = replica.add(facts(URL));
        let adapter = replica.lead();
        let now = Instant::now();
        adapter.record(&old, 0, &report(0, 3), now, now);
        let _replacement = replica.replace(id);
        adapter.reset();
        adapter.forget(&old);
        assert_eq!(replica.store.calls.lock().evicts, 0);
        assert_eq!(
            replica.store.load(id, 0).map(|sample| sample.row.running),
            Some(3)
        );
    }

    #[test]
    fn a_reset_drops_the_cache_and_leaves_the_stores_rows() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        let adapter = replica.lead();
        let now = Instant::now();
        adapter.record(&worker, 0, &report(0, 5), now, now);
        assert_eq!(adapter.members.read().len(), 1);
        let before = replica.store.load(id, 0).expect("the row");
        adapter.reset();
        assert!(adapter.members.read().is_empty());
        assert_eq!(replica.store.load(id, 0), Some(before));
        assert_eq!(
            adapter.poll_mode(&worker, now, PERIOD),
            PollMode::Suppressed
        );
    }

    /// The collector lost the event of a rank that left `Ready`, so it said
    /// no forget. Its reset drops the loads of the row that is not `Ready`,
    /// and the rank that stays is judged alone.
    #[test]
    fn a_reset_drops_the_loads_of_a_row_that_is_not_ready() {
        let replica = Replica::new();
        let adapter = replica.lead();
        let now = Instant::now();
        let [(zero, _first), (one, _second)] = busy_pair(&replica, &adapter, now);
        let row = WorkersStore::get(&replica.store.memory, zero).expect("a row");
        replica
            .store
            .memory
            .set_status(zero, WorkerStatus::NotReady, row.revision);
        replica.publish();
        assert!(replica.overloaded(one, now));
        adapter.reset();
        assert_eq!(replica.store.ranks(zero), 0);
        assert_eq!(replica.store.ranks(one), 1);
        replica.publish();
        assert!(!replica.overloaded(one, now));
    }

    #[test]
    fn a_members_wire_picks_the_source() {
        for (wire_modes, source, signal) in [
            (WireModes::GRPC, LoadSource::Poll, SignalSource::Poll),
            (WireModes::HTTP, LoadSource::Scrape, SignalSource::Scrape),
            (WireModes::ZMQ, LoadSource::Scrape, SignalSource::Scrape),
        ] {
            let replica = Replica::new();
            let (id, worker) = replica.add(WorkerFacts {
                wire_modes,
                ..facts(URL)
            });
            let adapter = replica.lead();
            adapter.report(&[polled(&worker, Some(&report(0, 3)))], Instant::now());
            let load = replica.store.load(id, 0).expect("a sampled row");
            assert_eq!((load.source, load.row.source), (source, signal));
        }
    }

    struct TestLead(Mutex<Option<u64>>);

    impl Lead for TestLead {
        fn term(&self) -> Option<u64> {
            *self.0.lock()
        }
    }

    #[test]
    fn a_replica_that_follows_writes_nothing_and_asks_no_engine() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        let lead = Arc::new(TestLead(Mutex::new(Some(1))));
        let adapter = replica.adapter(lead.clone(), 1);
        let now = Instant::now();
        adapter.record(&worker, 0, &report(0, 1), now, now);
        let saved = replica.store.load(id, 0);
        let before = *replica.store.calls.lock();
        *lead.0.lock() = None;

        let load = report(0, 9);
        let later = now + PERIOD;
        assert_eq!(
            adapter.poll_mode(&worker, later, PERIOD),
            PollMode::Suppressed
        );
        adapter.record(&worker, 0, &load, later, later);
        assert_eq!(
            adapter.report(&[polled(&worker, Some(&load))], later),
            PolledRanks::default()
        );
        assert!(adapter
            .report_empty(&[polled(&worker, None)], later)
            .is_empty());
        adapter.fetch_failed(slice::from_ref(&worker), later);
        adapter.forget(&worker);
        assert!(adapter.members.read().is_empty());
        adapter.reset();
        assert!(adapter.members.read().is_empty());
        assert!(adapter.needs_loads());
        adapter.tick(&GroupTick {
            workers: slice::from_ref(&worker),
            reports: &[],
        });
        assert_eq!(*replica.store.calls.lock(), before);
        assert_eq!(replica.store.load(id, 0), saved);
    }

    #[test]
    fn an_adapter_of_an_earlier_term_writes_nothing() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        let lead = Arc::new(TestLead(Mutex::new(Some(1))));
        let adapter = replica.adapter(lead.clone(), 1);
        let now = Instant::now();
        adapter.record(&worker, 0, &report(0, 1), now, now);
        let before = replica
            .store
            .load(id, 0)
            .expect("the adapter wrote under term 1");
        *lead.0.lock() = Some(2);

        let later = now + PERIOD;
        adapter.record(&worker, 0, &report(0, 9), later, later);
        adapter.report(&[polled(&worker, Some(&report(0, 9)))], later);
        adapter.forget(&worker);
        assert_eq!(
            adapter.poll_mode(&worker, later, PERIOD),
            PollMode::Suppressed
        );
        assert_eq!(replica.store.load(id, 0), Some(before));
        assert_eq!(replica.store.calls.lock().records, 1);
    }

    /// An adapter whose term ended still drops a forgotten worker's entry:
    /// the entry holds the worker object, and with it the object's engine
    /// connection. It evicts nothing, since it writes nothing.
    #[test]
    fn a_forget_drops_the_entry_whatever_the_term() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        let lead = Arc::new(TestLead(Mutex::new(Some(1))));
        let adapter = replica.adapter(lead.clone(), 1);
        let now = Instant::now();
        adapter.record(&worker, 0, &report(0, 1), now, now);
        let held = Arc::strong_count(&worker);
        *lead.0.lock() = Some(2);

        adapter.forget(&worker);
        assert!(adapter.members.read().is_empty());
        assert_eq!(Arc::strong_count(&worker), held - 1);
        assert_eq!(replica.store.calls.lock().evicts, 0);
        assert_eq!(replica.store.ranks(id), 1);
    }

    /// The store keeps the first period it is told. An adapter built with
    /// another one reads that its period did not stand.
    #[test]
    fn an_adapter_says_whether_the_store_took_its_period() {
        let replica = Replica::new();
        assert!(replica.lead().period_taken());
        let other = LoadAdapter::new(
            Arc::clone(&replica.objects),
            replica.store.clone(),
            Arc::new(SoleLead),
            SoleLead::TERM,
            PERIOD * 2,
        );
        assert!(!other.period_taken());
        assert!(replica.lead().period_taken(), "the first period again");
    }

    #[test]
    fn the_store_is_told_once_how_often_a_rank_is_heard() {
        let replica = Replica::new();
        let (_, worker) = replica.add(facts(URL));
        let adapter = replica.lead();
        assert_eq!(replica.store.calls.lock().period, Some(PERIOD));
        let now = Instant::now();
        adapter.record(&worker, 0, &report(0, 4), now, now);
        adapter.poll_mode(&worker, now, PERIOD);
        adapter.report(&[polled(&worker, Some(&report(0, 3)))], now + PERIOD);
        adapter.report_empty(&[polled(&worker, None)], now + PERIOD);
        adapter.fetch_failed(slice::from_ref(&worker), now + PERIOD);
        adapter.forget(&worker);
        adapter.reset();
        assert_eq!(replica.store.calls.lock().periods, 1);
    }

    /// One row's cells as a publication holds them: each cell's value and
    /// its sample instant, or none.
    type Held = Vec<Option<(f64, u64)>>;

    /// What [`declare`] declares: the depth in items and the fixed count
    /// of used tokens.
    const DECLARED: [(&str, &str); 2] = [("depth", "items"), ("kv_tokens_used", "tokens")];

    /// The `random` preset as the next generation, declaring the depth in
    /// items and the fixed count of used tokens, applied to the store.
    fn declare(replica: &Replica) {
        declare_as(replica, 2, &DECLARED);
    }

    /// As [`declare`], as the generation `generation` and declaring
    /// `signals`, each by its name and unit.
    fn declare_as(replica: &Replica, generation: u64, signals: &[(&str, &str)]) {
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        document.generation = generation;
        for &(name, unit) in signals {
            let spec = SignalSpec {
                unit: unit.to_owned(),
                max_age_ms: None,
                across_ranks: AcrossRanks::default(),
            };
            document.signals.insert(name.to_owned(), spec);
        }
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        let applied = replica.store.memory.apply(spec);
        applied.expect("a newer generation");
    }

    /// Each row's signal cells once the store's writes are shown, by
    /// worker, in the declared names' order: the depth, then the tokens.
    fn cells(replica: &Replica) -> FxHashMap<WorkerId, Held> {
        replica.publish();
        replica.store.memory.deciding(|world| {
            let ask = Ask {
                gateway: GatewayId(1),
                now: Instant::now(),
                pinned: None,
                affinity: false,
                signal_cells: true,
            };
            let mut out = ViewBuffers::default();
            let view = world
                .view(MODEL, Role::Single, &ask, false, &mut out)
                .expect("the pool");
            let block = view.signal_cells.expect("the publication's block");
            let held = |cell: &crate::SignalCell| {
                (!cell.is_none()).then_some((cell.value, cell.sampled_at_micros))
            };
            let cells = block.cells().chunks(block.signals());
            let ids = view.candidates.iter().map(|row| row.id);
            ids.zip(cells.map(|cells| cells.iter().map(held).collect()))
                .collect()
        })
    }

    /// One rank's report: its depth by name, a name no document declares,
    /// and the fixed count of used tokens.
    fn stating(rank: i32, depth: f64, age_ms: Option<u32>) -> WorkerLoadResponse {
        let stated = |name: &str, value| NamedValue {
            name: name.into(),
            unit: "items".into(),
            value,
            age_ms,
        };
        WorkerLoadResponse {
            dp_rank_count: 1,
            loads: vec![SchedulerLoadSnapshot {
                dp_rank: rank,
                num_running_reqs: 1,
                num_used_tokens: 1_024 + rank,
                max_total_num_tokens: 8_192,
                values: vec![stated("depth", depth), stated("other", 1.0)],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn a_records_declared_values_follow_its_row_into_the_store() {
        let replica = Replica::new();
        // The second rank of two: a value written at another rank reaches
        // no row.
        let (id, worker) = replica.add(WorkerFacts {
            dp_rank: 1,
            dp_size: 2,
            ..facts(URL)
        });
        declare(&replica);
        let adapter = replica.lead();
        let sampled = Instant::now() + Duration::from_secs(1);
        let received = sampled + Duration::from_millis(37);
        adapter.record(&worker, 1, &stating(1, 3.0, Some(20)), received, sampled);
        // A value is as old as it says before its record's sample, and one
        // that states no age is as old as its record.
        let store = &replica.store.memory;
        let aged = store.micros(sampled - Duration::from_millis(20));
        let held = [Some((3.0, aged)), Some((1_025.0, store.micros(sampled)))];
        assert_eq!(cells(&replica)[&id], held);
        // The name no document declares was never offered: no write
        // counts it.
        assert_eq!(store.counts().signal_writes, [2, 0, 0, 0, 0, 0]);
        // A record that leaves the names out leaves their cells.
        let later = received + PERIOD;
        adapter.record(&worker, 1, &report(1, 4), later, later);
        let row = replica.store.load(id, 1).expect("the later row");
        assert_eq!(row.row.running, 4);
        assert_eq!(cells(&replica)[&id], held);
        assert_eq!(store.counts().signal_writes, [2, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn a_polled_answers_values_follow_the_rows_the_store_took() {
        let replica = Replica::new();
        let [(zero, first), (one, second)] = replica.two_ranks(None);
        declare(&replica);
        let adapter = replica.lead();
        let asked = Instant::now();
        // Rank 0's own record lands while the poll is out: its polled row
        // is passed over, and takes its values with it.
        let pushed = asked + Duration::from_millis(1);
        adapter.record(&first, 0, &report(0, 7), pushed, pushed);
        let answers = [stating(0, 3.0, None), stating(1, 5.0, None)];
        let ranks = adapter.report(
            &[
                polled(&first, Some(&answers[0])),
                polled(&second, Some(&answers[1])),
            ],
            asked,
        );
        assert_eq!(ranks.newer, [0]);
        // A polled value is dated as its row is: at the poll's start.
        let store = &replica.store.memory;
        let at = store.micros(asked);
        let held = cells(&replica);
        assert_eq!(held[&zero], [None, None]);
        assert_eq!(held[&one], [Some((5.0, at)), Some((1_025.0, at))]);
        assert_eq!(store.counts().signal_writes, [2, 0, 0, 0, 0, 0]);
        // A failed fetch and an answer with no row write no value.
        let empty = WorkerLoadResponse::default();
        adapter.report(&[polled(&first, Some(&empty))], asked + PERIOD);
        adapter.fetch_failed(slice::from_ref(&first), asked + PERIOD);
        assert_eq!(store.counts().signal_writes, [2, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn a_record_that_lays_no_row_writes_no_value() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        declare(&replica);
        let now = Instant::now();
        // An adapter whose term ended, then a worker that is not ready.
        let lead = Arc::new(TestLead(Mutex::new(Some(2))));
        let ended = replica.adapter(lead, 1);
        ended.record(&worker, 0, &stating(0, 3.0, None), now, now);
        worker.set_status(ObjectStatus::NotReady);
        let adapter = replica.lead();
        adapter.record(&worker, 0, &stating(0, 3.0, None), now, now);
        assert_eq!(replica.store.ranks(id), 0);
        let counts = replica.store.memory.counts();
        assert_eq!(counts.signal_writes, [0; 6]);
    }

    /// A report of a worker over several ranks: one entry a rank, with the
    /// depth the rank states by name where it states one. No entry states
    /// a fixed count.
    fn file(ranks: &[(i32, Option<f64>)]) -> WorkerLoadResponse {
        let entry = |&(rank, depth): &(i32, Option<f64>)| SchedulerLoadSnapshot {
            dp_rank: rank,
            num_running_reqs: 1,
            kept: LoadTelemetry::USED_TOKENS | LoadTelemetry::TOTAL_TOKENS,
            values: depth
                .map(|value| NamedValue {
                    name: "depth".into(),
                    unit: "items".into(),
                    value,
                    age_ms: None,
                })
                .into_iter()
                .collect(),
            ..Default::default()
        };
        WorkerLoadResponse {
            dp_rank_count: i32::try_from(ranks.len()).unwrap_or(i32::MAX),
            loads: ranks.iter().map(entry).collect(),
            ..Default::default()
        }
    }

    /// The depth the row of `id` holds once the store's writes are shown:
    /// its value and its sample instant.
    fn depth(replica: &Replica, id: WorkerId) -> Option<(f64, u64)> {
        cells(replica)[&id][0]
    }

    /// One row over an engine of two ranks. Rank 1 states its depth once
    /// and falls silent, and rank 0 states its own each second. The row's
    /// sum stays as old as rank 1's statement, so the signal's maximum age
    /// ends it.
    #[test]
    fn a_silent_ranks_value_keeps_its_own_instant_behind_a_row_of_several_ranks() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        declare(&replica);
        let adapter = replica.lead();
        let start = Instant::now() + Duration::from_secs(1);
        let at = |ms: u64| start + Duration::from_millis(ms);
        let both = file(&[(0, Some(1.0)), (1, Some(100.0))]);
        adapter.record(&worker, 1, &both, at(0), at(0));
        for second in 1..=9 {
            let now = at(second * 1_000);
            adapter.record(&worker, 0, &both, now, now);
        }
        let silent = replica.store.memory.micros(at(0));
        assert_eq!(depth(&replica, id), Some((101.0, silent)));
    }

    /// A record may leave a name out. One such record of rank 1 leaves
    /// rank 1's last statement standing, and the row's value follows rank
    /// 0's next one.
    #[test]
    fn a_record_with_no_value_from_one_rank_leaves_the_others_standing() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        declare(&replica);
        let adapter = replica.lead();
        let start = Instant::now() + Duration::from_secs(1);
        let at = |ms: u64| start + Duration::from_millis(ms);
        let both = file(&[(0, Some(1.0)), (1, Some(2.0))]);
        adapter.record(&worker, 1, &both, at(0), at(0));
        adapter.record(&worker, 0, &both, at(10), at(10));
        assert_eq!(depth(&replica, id).map(|(value, _)| value), Some(3.0));
        let bare = file(&[(0, Some(1.0)), (1, None)]);
        adapter.record(&worker, 1, &bare, at(20), at(20));
        let moved = file(&[(0, Some(3.0)), (1, None)]);
        adapter.record(&worker, 0, &moved, at(30), at(30));
        assert_eq!(depth(&replica, id).map(|(value, _)| value), Some(5.0));
    }

    #[test]
    fn a_polled_answer_of_several_ranks_states_each_rank_though_its_row_is_passed_over() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        declare(&replica);
        let adapter = replica.lead();
        let start = Instant::now() + Duration::from_secs(1);
        let at = |ms: u64| start + Duration::from_millis(ms);
        let micros = |at: Instant| replica.store.memory.micros(at);
        let both = file(&[(0, Some(1.0)), (1, Some(2.0))]);
        adapter.record(&worker, 1, &both, at(0), at(0));
        adapter.record(&worker, 0, &both, at(10), at(10));
        // Rank 0 pushes while a poll is out: the store passes the answer's
        // row over, and the row's value stays the records' own.
        let asked = at(1_000);
        let pushed = file(&[(0, Some(5.0)), (1, Some(2.0))]);
        adapter.record(&worker, 0, &pushed, at(1_001), at(1_001));
        let answer = file(&[(0, Some(4.0)), (1, Some(6.0))]);
        let ranks = adapter.report(&[polled(&worker, Some(&answer))], asked);
        assert_eq!(ranks.newer, [0]);
        assert_eq!(depth(&replica, id), Some((7.0, micros(at(0)))));
        // Rank 0's next record reads rank 1 as the answer stated it, at
        // the poll's start. Rank 0's own record is newer than the answer.
        adapter.record(&worker, 0, &pushed, at(1_500), at(1_500));
        assert_eq!(depth(&replica, id), Some((11.0, micros(asked))));
        // An answer whose row the store takes writes the row's value.
        let later = at(3_000);
        let answer = file(&[(0, Some(1.0)), (1, Some(1.0))]);
        let ranks = adapter.report(&[polled(&worker, Some(&answer))], later);
        assert!(ranks.newer.is_empty());
        assert_eq!(depth(&replica, id), Some((2.0, micros(later))));
    }

    /// A worker's statements go with its entry. Behind a reset the store
    /// still holds each rank's row: a rank's last statement is read from
    /// its entry again, under its own record. Behind a forget the rows
    /// went too, and the rank states again.
    #[test]
    fn a_workers_statements_go_with_its_entry() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        declare(&replica);
        let adapter = replica.lead();
        let start = Instant::now() + Duration::from_secs(1);
        let at = |ms: u64| start + Duration::from_millis(ms);
        let micros = |at: Instant| replica.store.memory.micros(at);
        let both = file(&[(0, Some(1.0)), (1, Some(2.0))]);
        adapter.record(&worker, 1, &both, at(0), at(0));
        // A reset drops the entry, and rank 1's statement with it. The
        // store holds rank 1's row still: rank 0's record reads rank 1's
        // entry, under rank 1's own record.
        adapter.reset();
        adapter.record(&worker, 0, &both, at(10), at(10));
        assert_eq!(depth(&replica, id), Some((3.0, micros(at(0)))));
        // A forget takes the rows too: rank 1's entry has no instant, and
        // counts when rank 1 states again.
        adapter.forget(&worker);
        adapter.record(&worker, 0, &both, at(40), at(40));
        assert_eq!(depth(&replica, id), None);
        adapter.record(&worker, 1, &both, at(50), at(50));
        adapter.record(&worker, 0, &both, at(60), at(60));
        assert_eq!(depth(&replica, id), Some((3.0, micros(at(50)))));
    }

    /// A worker's first records list one rank, and are read as a worker
    /// of one rank. The next rank's first record reads the first rank's
    /// entry, under that rank's own record: a rank that stated once and
    /// fell silent counts all the same, as old as it is.
    #[test]
    fn a_ranks_first_statement_counts_at_the_next_ranks_first_record() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        declare(&replica);
        let adapter = replica.lead();
        let start = Instant::now() + Duration::from_secs(1);
        let at = |ms: u64| start + Duration::from_millis(ms);
        let alone = file(&[(1, Some(100.0))]);
        adapter.record(&worker, 1, &alone, at(0), at(0));
        assert_eq!(depth(&replica, id), None);
        let both = file(&[(0, Some(1.0)), (1, Some(100.0))]);
        adapter.record(&worker, 0, &both, at(9_000), at(9_000));
        let first = replica.store.memory.micros(at(0));
        assert_eq!(depth(&replica, id), Some((101.0, first)));
    }

    /// A new document ends the statements, though it declares the same
    /// signals. The next record reads each rank's latest record again,
    /// under that rank's own instant, so the row's value moves at once.
    #[test]
    fn a_new_document_reads_each_ranks_latest_record_again() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        declare(&replica);
        let adapter = replica.lead();
        let start = Instant::now() + Duration::from_secs(1);
        let at = |ms: u64| start + Duration::from_millis(ms);
        let both = file(&[(0, Some(1.0)), (1, Some(2.0))]);
        adapter.record(&worker, 1, &both, at(0), at(0));
        adapter.record(&worker, 0, &both, at(10), at(10));
        assert_eq!(depth(&replica, id).map(|(value, _)| value), Some(3.0));
        declare_as(&replica, 3, &DECLARED);
        let moved = file(&[(0, Some(4.0)), (1, Some(2.0))]);
        adapter.record(&worker, 0, &moved, at(30), at(30));
        let first = replica.store.memory.micros(at(0));
        assert_eq!(depth(&replica, id), Some((6.0, first)));
    }

    /// A signal removed and declared again starts with no value, as the
    /// store's cells do: nothing a rank stated passes the document that
    /// stood between. A rank whose latest record leaves the name out
    /// counts when it states the name again.
    #[test]
    fn a_signal_declared_again_starts_from_the_latest_records() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        declare(&replica);
        let adapter = replica.lead();
        let start = Instant::now() + Duration::from_secs(1);
        let at = |ms: u64| start + Duration::from_millis(ms);
        let both = file(&[(0, Some(1.0)), (1, Some(2.0))]);
        adapter.record(&worker, 1, &both, at(0), at(0));
        adapter.record(&worker, 0, &both, at(10), at(10));
        let bare = file(&[(0, Some(1.0)), (1, None)]);
        adapter.record(&worker, 1, &bare, at(20), at(20));
        assert_eq!(depth(&replica, id).map(|(value, _)| value), Some(3.0));
        // Two documents between two reports: the depth goes, and comes
        // back as it was declared.
        declare_as(&replica, 3, &DECLARED[1..]);
        declare_as(&replica, 4, &DECLARED);
        let moved = file(&[(0, Some(4.0)), (1, None)]);
        adapter.record(&worker, 0, &moved, at(30), at(30));
        assert_eq!(depth(&replica, id), None);
        let both = file(&[(0, Some(4.0)), (1, Some(2.0))]);
        adapter.record(&worker, 1, &both, at(40), at(40));
        adapter.record(&worker, 0, &both, at(50), at(50));
        let restated = replica.store.memory.micros(at(40));
        assert_eq!(depth(&replica, id), Some((6.0, restated)));
    }

    /// A poll's row stands in the place of a silent rank's sample at the
    /// row's own rank, though its answer leaves that rank out and the
    /// rank's pushed entry stays on file. The entry is as old as the
    /// rank's own record, which the worker's entry noted: it is not dated
    /// by the poll, and no old value reads fresh.
    #[test]
    fn a_polls_row_in_a_ranks_place_dates_no_entry() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        declare(&replica);
        let adapter = replica.lead();
        let start = Instant::now() + Duration::from_secs(1);
        let at = |ms: u64| start + Duration::from_millis(ms);
        let alone = file(&[(0, Some(100.0))]);
        adapter.record(&worker, 0, &alone, at(0), at(0));
        // An answer that lists rank 1 alone, with no depth: its row takes
        // the place of rank 0's sample.
        let answer = file(&[(1, None)]);
        adapter.report(&[polled(&worker, Some(&answer))], at(9_000));
        let both = file(&[(0, Some(100.0)), (1, Some(1.0))]);
        adapter.record(&worker, 1, &both, at(10_000), at(10_000));
        let bare = file(&[(0, None), (1, Some(1.0))]);
        adapter.record(&worker, 0, &bare, at(11_000), at(11_000));
        // Rank 0's depth counts as old as rank 0 stated it.
        let first = replica.store.memory.micros(at(0));
        assert_eq!(depth(&replica, id), Some((101.0, first)));
    }

    /// An answer of one rank is one sample, and its entry is laid on the
    /// worker's pushed report. Where the store took its row, the entry is
    /// as old as the answer: the next rank's first record counts it so.
    #[test]
    fn an_answer_of_one_rank_counts_under_its_own_instant() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        declare(&replica);
        let adapter = replica.lead();
        let start = Instant::now() + Duration::from_secs(1);
        let at = |ms: u64| start + Duration::from_millis(ms);
        let alone = file(&[(0, Some(1.0))]);
        adapter.record(&worker, 0, &alone, at(0), at(0));
        let answer = file(&[(0, Some(2.0))]);
        let asked = at(5_000);
        let ranks = adapter.report(&[polled(&worker, Some(&answer))], asked);
        assert!(ranks.newer.is_empty());
        // The report after rank 1's first record: rank 0 as the answer
        // laid it, and rank 1's own.
        let both = file(&[(0, Some(2.0)), (1, Some(100.0))]);
        adapter.record(&worker, 1, &both, at(6_000), at(6_000));
        let bare = file(&[(0, None), (1, Some(100.0))]);
        adapter.record(&worker, 0, &bare, at(7_000), at(7_000));
        let answered = replica.store.memory.micros(asked);
        assert_eq!(depth(&replica, id), Some((102.0, answered)));
    }

    /// An answer can leave out a rank the pushed report still lists. The
    /// rank's statements go with the answer, and the next record reads the
    /// rank's entry again, under its own record: the row's value keeps
    /// moving with the other ranks.
    #[test]
    fn a_rank_an_answer_leaves_out_counts_again_from_its_entry() {
        let replica = Replica::new();
        let (id, worker) = replica.add(facts(URL));
        declare(&replica);
        let adapter = replica.lead();
        let start = Instant::now() + Duration::from_secs(1);
        let at = |ms: u64| start + Duration::from_millis(ms);
        let micros = |at: Instant| replica.store.memory.micros(at);
        let all = file(&[(0, Some(1.0)), (1, Some(2.0)), (2, Some(4.0))]);
        adapter.record(&worker, 2, &all, at(0), at(0));
        adapter.record(&worker, 1, &all, at(10), at(10));
        adapter.record(&worker, 0, &all, at(20), at(20));
        assert_eq!(depth(&replica, id), Some((7.0, micros(at(0)))));
        // Rank 0 pushes while a poll is out, and the answer lists ranks 0
        // and 1 alone: the store passes its row over.
        let asked = at(1_000);
        let pushed = file(&[(0, Some(3.0)), (1, Some(2.0)), (2, Some(4.0))]);
        adapter.record(&worker, 0, &pushed, at(1_001), at(1_001));
        assert_eq!(depth(&replica, id), Some((9.0, micros(at(0)))));
        let answer = file(&[(0, Some(3.0)), (1, Some(2.0))]);
        let ranks = adapter.report(&[polled(&worker, Some(&answer))], asked);
        assert_eq!(ranks.newer, [0]);
        // Rank 0's next record counts rank 2 from its entry, as old as
        // rank 2's own record.
        let next = file(&[(0, Some(5.0)), (1, Some(2.0)), (2, Some(4.0))]);
        adapter.record(&worker, 0, &next, at(1_500), at(1_500));
        assert_eq!(depth(&replica, id), Some((11.0, micros(at(0)))));
    }

    /// A rank worker's report is one rank's, by a record and by a poll:
    /// its last value is the store's cell, and its entry makes no keeper.
    #[test]
    fn a_worker_of_one_rank_keeps_no_statement() {
        let replica = Replica::new();
        let [(zero, first), (one, second)] = replica.two_ranks(None);
        declare(&replica);
        let adapter = replica.lead();
        let now = Instant::now() + Duration::from_secs(1);
        adapter.record(&first, 0, &stating(0, 3.0, None), now, now);
        adapter.record(&second, 1, &stating(1, 5.0, None), now, now);
        let asked = now + PERIOD;
        let answers = [stating(0, 4.0, None), stating(1, 6.0, None)];
        adapter.report(
            &[
                polled(&first, Some(&answers[0])),
                polled(&second, Some(&answers[1])),
            ],
            asked,
        );
        let at = replica.store.memory.micros(asked);
        assert_eq!(depth(&replica, zero), Some((4.0, at)));
        assert_eq!(depth(&replica, one), Some((6.0, at)));
        let members = adapter.members.read();
        assert_eq!(members.len(), 2);
        assert!(members.values().all(|kept| kept.keeper.get().is_none()));
    }
}
