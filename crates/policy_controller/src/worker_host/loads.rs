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
//! on a poll answer. The adapter calls no collector.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use smg_worker::{
    identity::basic::Worker,
    load::port::{LoadSink, PollMode, PolledMember, PolledRanks},
    registry::{Generation, WorkerId as ObjectId},
    worker::{ConnectionMode, WorkerLoadResponse, WorkerStatus as ObjectStatus},
};

use super::{object::role, Lead, Objects};
use crate::{
    load_report::{load_row, signal_values},
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
        })
    }

    fn member(&self, worker: &Arc<dyn Worker>) -> Option<Member> {
        let url = worker.url();
        if let Some(kept) = self.members.read().get(url) {
            // The registry can retire the cached object before the collector
            // sees the event. The kept id lets this check borrow the key.
            if Arc::ptr_eq(&kept.worker, worker) && self.is_current(&kept.id, &kept.worker) {
                return Some(kept.member);
            }
        }
        let mut members = self.members.write();
        let kept = self.read_member(worker)?;
        let member = kept.member;
        members.insert(url.to_string(), kept);
        Some(member)
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
        // Each row's report, kept past its row where the document declares
        // a signal.
        let mut stated = Vec::new();
        let rows: Vec<_> = answer
            .filter_map(|(worker, report)| {
                // A member whose status left `Ready` while its poll was out
                // lays no sample either: it stands as a rank not asked.
                if worker.status() != ObjectStatus::Ready {
                    return None;
                }
                let member = self.member(worker)?;
                let report = report.filter(|report| !report.loads.is_empty());
                let row = report.map(|report| {
                    load_row(report, member.role, received_at, received_at, source.into())
                });
                if !columns.is_empty() {
                    stated.push(report);
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
        for (position, (&(row, rank, _), report)) in rows.iter().zip(stated).enumerate() {
            if skipped.next_if_eq(&position).is_some() {
                continue;
            }
            let Some(report) = report else {
                continue;
            };
            signal_values(report, &columns, polled_at, |name, unit, value, at| {
                self.store.signal_value(row, rank, name, unit, value, at);
            });
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
        let Some(member) = self.member(worker) else {
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
        let columns = self.store.signal_columns();
        signal_values(report, &columns, sampled_at, |name, unit, value, at| {
            self.store
                .signal_value(member.row, rank, name, unit, value, at);
        });
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
        worker::{NamedValue, SchedulerLoadSnapshot, WorkerStatus as ObjectStatus},
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
        fn signal_columns(&self) -> Arc<[crate::SignalColumn]> {
            self.memory.signal_columns()
        }

        fn signal_value(
            &self,
            id: WorkerId,
            rank: u16,
            name: &str,
            unit: &str,
            value: f64,
            sampled_at: Instant,
        ) -> Taken {
            self.memory
                .signal_value(id, rank, name, unit, value, sampled_at)
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

    /// The `random` preset as the next generation, declaring the depth in
    /// items and the fixed count of used tokens, applied to the store.
    fn declare(replica: &Replica) {
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        document.generation = 2;
        for (name, unit) in [("depth", "items"), ("kv_tokens_used", "tokens")] {
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
}
